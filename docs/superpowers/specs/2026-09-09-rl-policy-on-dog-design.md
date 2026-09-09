# CPG-RL policy 上機（D1 Max、v2.3 權重）—— 設計（2026-09-09）

## 目的

把 `weights/cpg_rl_max_v2_3_params.pkl`（G4 ✅ G7 ✅ G5 差 0.01 G6 ❌ −0.5°/s）跑在狗上。
狗上只有 Python 3.10 ＋ numpy 1.21.5，沒有 jax/brax/mujoco。
今天（2026-09-09）晚上上機；上機前所有驗證都在本機做完。

原則：**RL 只在開迴路 `A_kp250_walk` 周圍調變，退路永遠是 A**（A 已在 trip17 兩趟零中止走完）。
M9 的起身／坐下／護欄／ChatterWatch／`--confirm`／說兩次全部不變。

## 範圍

做：numpy 匯出與前向、狗端 obs 組裝、M9 `--policy` 模式、離線回放驗證、部署清單、操作卡。
不做：v2.4 的 `head_err`（等權重）、IMU 偏置修正（訓練已隨機化 ±3°）、任何增益變更、地形。

## 1. 元件

| 檔 | 在哪跑 | 做什麼 | 依賴 |
|---|---|---|---|
| `task7/inference/export_policy_np.py` | 本機 | pkl → `weights/cpg_rl_max_v2_3_np.npz` | jax/brax |
| `task7/realbot/policy_np.py` | 狗 | 載 npz、MLP 前向、`act_to_cmd`、`baseline_action`、`slew_sway` | numpy |
| `task7/realbot/rl_obs.py` | 狗 | shm 關節＋`imu_central`＋CPG 狀態 → 66 維 obs | numpy、coord、cpg |
| `task7/realbot/M9_gait.py` | 狗 | `--policy` 模式：`PolicyGaitStream` | 上面三個 |
| `task7/inference/replay_policy.py` | 本機 | trip17 M9 log 回放，狗端堆疊 vs jax 堆疊 | 全部 |
| `task7/realbot/push_to_dog.sh` | 本機 | 加 npz 與兩個新檔 | — |
| `task7/docs/現場操作卡_RL_v2.3_2026-09-09.md` | — | 上機順序 | — |

### 1.1 `export_policy_np.py`

brax PPO policy 網路 = `running_statistics.normalize(obs)` → MLP `(256,256,128)`、swish → 線性到 `2·act_dim`
→ 取前 `act_dim` 為 loc → deterministic 動作 = `tanh(loc)`（`NormalTanhDistribution.mode`）。
npz 內容：`mean(66)`, `std(66)`, `W0..W3`, `b0..b3`, `obs_dim=66`, `act_dim=10`, `preset="v2.3"`, `src_sha256`。
正規化的確切式子（std 的 epsilon、有無 clip）**不從記憶寫**，以「numpy 前向 vs `local_infer_max.load_policy`
在 1000 筆隨機 obs 上 max|Δ| < 1e-5」的測試釘住；不過就改匯出直到過。

### 1.2 `policy_np.py`

- `load(path) -> Policy`；`Policy.infer(obs) -> a(10)`；`Policy.obs_dim/act_dim/preset`。
- `act_to_cmd(a, layout="nomux")`、`baseline_action("nomux")`、`slew_sway(prev, tgt)`：與 `local_infer_max`
  逐行同義，常數 `MU_MIN/MAX 1/2`、`OMEGA_MIN/MAX 0/2`、`SWAY_MAX 0.060`、`SWAY_SLEW 0.004`、`A["mu_x"]`。
  測試釘住與 `local_infer_max` 同值。
- 回傳的 mux/muy/om 是 **MJCF 腿序陣列（FR, FL, RR, RL）**；由 `rl_obs.to_shm_legs()` 轉成 dict `{fl, fr, bl, br}` 給 `cpg.step`。

### 1.3 `rl_obs.py`

`build(frame, c, cmd, last_a) -> obs(66) float32`，`OBS_LAYOUT` 與 `obs_max` 相同：
`gravity 3, gyro 3, joint_pos 12, joint_vel 12, cmd 2, last_action 10, cpg 24`。

| 欄 | 來源 | 換算 |
|---|---|---|
| gravity | `imu_central` quat（xyzw）| → wxyz → 機身系 (0,0,−1)（自寫 `w2b`，測試對 `cpg_max.w2b`）|
| gyro | `imu_central` gyro | rad/s，scale (1,1,1)（H 文件定案）|
| joint_pos | shm position（馬達座標）| `coord.to_ctrl` − `HOME12`，**MJCF 腿序**（同 `real_obs.LEG_NAMES`）|
| joint_vel | shm velocity | ÷ `coord.SIGN` |
| cmd | (vx, wz) | 原樣 |
| last_action | 上一步 policy 輸出 | 原樣 |
| cpg | `GaitStream.c`（dict fl/fr/bl/br）| 轉 MJCF 腿序：rx, rx_d, ry, ry_d, sinθ, cosθ |

`HOME12` 與腿序表在檔內以常數寫死，測試釘住等於 `max_model.HOME12` / `real_obs.LEG_NAMES`。
`frame` 是一個小類別：`pos12, vel12（shm 序）, quat_xyzw, gyro`；由 M9 在每個 50 Hz 推進點讀 shm 填。

### 1.4 `M9_gait.py --policy`

新參數：`--policy PATH.npz`、`--vx 0.30`、`--wz 0.0`、`--policy-gain 1.0`（0 = 乾跑：跑 obs 與推論但動作固定基準）。
`--policy` 必須搭 `--interactive`（走到喊停）。

`PolicyGaitStream(GaitStream)`：覆寫 `sample()` 內「推進一步」的位置——

```
frame = read_shm()                       # 關節 12＋imu_central
obs   = rl_obs.build(frame, self.c, cmd, last_a)
a     = policy.infer(obs)                # 出錯/NaN/超時 → a = base_act, 記警告
act   = base_act + gain·u(i)·(a − base_act)      # u(i) = min(1, i/50) 起步淡入
mux, muy, om, sw_t = act_to_cmd(act)
sway  = slew_sway(sway, sw_t)
self.c = step(self.c, mux, muy, om, 0.02)
q = joint_targets(..., sway=(sx, sy))    # 直接給 foot_targets，取代相位式 body_sway
```

- 退回條件（當步用 `base_act`，印一行警告，連 25 步觸發 → 永久退回開迴路）：
  obs 含 NaN／inf；shm tick 與上一步相同（狀態沒更新）；推論耗時 > 10 ms。
- 鍵 `o`：永久切回開迴路 A（`act ≡ base_act`，sway 斜率退到 0），走路繼續。
- log：每步多記 `obs(66)`、`a(10)`、`act(10)`、`sway(2)`、原始 `imu(10)`、推論 ms、退回原因。
  補上「走路時原始 gyro_z」的洞。
- 軌跡檔的 13 欄「說兩次」改為對 `BASELINE_A` 檢查（kp 250／abad 60／kd 2／wheel_kd 0.5／x_off −0.030／g_c 0.048…），
  不一致拒跑，訊息同現有格式。
- 停走淡出、坐下、護欄 70 N·m / 0.6 rad、ChatterWatch、`--hold-max/--walk-max` 全部沿用。

### 1.5 `replay_policy.py`

輸入 trip17 的 M9 log（`m9_rec`）。對每一幀：
狗端 `rl_obs.build` + `policy_np.infer` vs 本機 `real_obs`-風格 `obs_max.build_obs` + `load_policy`。
印 obs 各欄 max|Δ|、動作 max|Δ|、通過與否。CPG 狀態用固定假值（非零、四腿不同）以驗腿序。
限制（寫進輸出）：M9 舊 log 只有 roll/pitch，gyro_z ≡ 0，這裡驗的是正負號、順序、單位。

## 2. 測試（本機 pytest，`task7/tests/`）

1. `test_policy_np.py`：npz vs jax 前向 1000 筆 < 1e-5；常數釘住；`act_to_cmd`/`baseline_action`/`slew_sway` 與 `local_infer_max` 逐點相同。
2. `test_rl_obs.py`：同一幀（含隨機 quat/gyro/關節/CPG）`rl_obs.build` 與 `obs_max.build_obs`（經 `RealFrame` 路徑）逐位元相同；腿序打亂會被抓到。
3. **閉迴路說兩次**：MuJoCo 內跑 `local_infer_max` 的 rollout 200 步，每步同時用狗端堆疊（`rl_obs` + `policy_np` + `realbot/cpg`）算動作與 12 目標角，
   逐步 max|Δ| < 1e-6（動作）與 < 1e-9 rad（目標角）。
4. `test_m9_gait.py` 增：`PolicyGaitStream` 用假 shm 與假 policy 跑 100 步不炸；gain 0 時 12 目標角與 `GaitStream` 逐幀相同；
   NaN obs 觸發退回；`o` 鍵後動作 ≡ 基準；AST 測試涵蓋新呼叫點。
5. `test_push_to_dog`（若既有）加新檔名。

## 3. 上機順序（操作卡）

1. `push_to_dog.sh` → 狗上 `python3 -c "import policy_np, rl_obs"`；`M_env_probe` 確認 numpy。
2. **乾跑**：`--policy … --policy-gain 0 --walk-max 5`：走的是純 A，看 log 裡 obs 範圍（gravity ≈ (0,0,−1)、gyro 量級、joint 與 CPG 非零）、推論 ms、無退回。
3. `--policy-gain 1 --walk-max 5`：原地／短走，看護欄數字與 roll。
4. `--walk-max 10`：與 trip17 對照 roll 峰值、膝／髖峰值、偏航。
任何一步中止就停在該步，拉 log 回本機分析。

## 4. 風險

- 正規化式子若對錯，動作會靜默偏掉 → 測試 1 是硬門。
- 實機 obs 分布與模擬不同（joint_vel 濾波、IMU 偏置）→ 訓練已隨機化；乾跑先看 obs 範圍。
- policy 迴圈多讀一次 shm 與 imu（唯讀，<1 ms）；推論 0.14 ms；20 ms 預算內。
- G6 未解：60 s 偏 −30°；10 s 走約 5°，可接受，不是安全問題。
