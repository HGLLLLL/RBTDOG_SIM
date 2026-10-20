# 純 RL 自訓 v1c 設計：左右對稱＋落地輕＋延遲／IMU 偏差隨機化（2026-10-20）

狀態：設計已與使用者逐段定案（起點、範圍、reward 改動、DR、及格線、測試）。下一步是寫實作計畫。
前一版 spec：`2026-10-13-rl-wheel-v1-design.md`（環境、地形、訓練線、§11 坑清單都沿用，本文只寫差異）。
reward 現況盤點：`docs/results/Y_v1b_reward總表_2026-10-20.md`。

## 0. 動機（使用者 10-20 定）

v1b 上機 24 趟 0 摔，步態整體可用，剩兩個問題：
1. **平移落地重**（踏地聲大）。現有 reward 沒有任何接觸力／落地項，`acc_leg`／`rate` 在平移時只有 −0.01～−0.04，壓不住。
2. **左平移持續側傾 7～10°、左右不對稱**（右平移 1～4°）。`angxy` 罰的是角速度，固定側傾時 ≈ 0；三版 policy（10k／v1／v1b）都不對稱、歪的方向隨機 → 是訓練時對稱性自發破缺，不是哪個權重不夠。

查證過、**不採用**的假設：
- 「碰撞懲罰讓步態變用力」：`wgap` 在 v1 的 5 億步全程為 0（gap 349 → 135 mm 從沒碰到 80 mm 線）、`ww` 平地恆 0，沒輸出過梯度；降低它拿不回 10k 的步法。
- 「σ 0.25 → 0.5 分段讓步態從容」：10k 的從容是「不追滿比較划算」的副產品（側移 0.5 只做 0.36），放寬 σ 買到的是欠追蹤不是輕踏。使用者要的是步態不是欠追蹤 → **σ 全域維持 0.25**。
- 「similar to default 解對稱」：就是現有 `dev` 項（legged_gym 命名），罰離站姿的距離、不罰左右差；加重會先壓掉抬腳。
- 「從容」（步頻、關節速度）這版**不列目標**：`acc_leg`／`rate` 權重不動、`air_time` 不加；之後需要再開一版。

## 1. 範圍與起點

- **起點：從頭訓、一開始就開鏡像**（使用者定；不從 v1b 續訓）。累計 10 億步，Colab G4 兩個 session。
- **範圍**：鏡像增強、`tilt`、`impact`、動作延遲 DR、IMU 常數偏差 DR，一次全加（使用者定）。歸因靠：開訓前攤帳、每項單獨的 reward 單元測試、訓練曲線分項列印、5 億步中場檢查。
- **不包含**：上機（trip29 另排）、`rl16_trip_report.py` 抬腳欄修正、M9 硬上限修正、真樓梯。
- **狗端零改動**：policy 仍是同一種 MLP／`rl16` 版面，只多一個 `rl_wheel_v1c_np.npz`。`--grav-offset 0.7 -1.5` 照舊。

## 2. 環境改動（`inference/rl_env_wheel.py`）

**原則**：建構子預設值＝v1b 行為；新東西集中在一個預設字典 `V1C = dict(mirror=True, w_tilt=10.0, w_impact=3.0, delay_max=2, grav_bias_deg=2.0)`，notebook 用 `WheelEnv(**rw.V1C)`，工具用 `--preset v1c`。golden 測試釘住 `WheelEnv()`。

### 2.1 鏡像增強（`mirror=True`；照 v3.7b `MIRROR_AUG` 的做法，那版設計過沒訓過）

- 出生／重生時抽 `info["mir"] ~ Bernoulli(0.5)`（用 spawn key 的 fold_in，不動既有 split 次序）。
- `mir` 為真：policy 看到的 obs 翻（`state` 與 `privileged_state` 都翻，後者的機身線速度 y 變號）；policy 輸出的動作翻回來才進 `_physics`。`info["last_a"]` 存**物理側**的動作，obs 的 `last_action` 切片由鏡像表一起翻，所以 policy 看到的是自己上一步的輸出；`rate` 用物理側 a − last_a，鏡像是帶符號的置換，平方和不變。
- reward 全部用真實物理算，不碰。
- 鏡像表（obs 53／act 16，腿序 FR FL RR RL、PhysX 分組序 ABAD×4 HIP×4 KNEE×4 [WHEEL×4]）：腿 FR↔FL、RR↔RL；ABAD 位置／速度／動作變號；角速度 x、z 變號；重力 y 變號；指令 vy、wz 變號；髖、膝、輪速、輪動作只換腿不變號（四腿關節軸同向：ABAD x 軸、髖膝輪 y 軸；y → −y 的鏡射下繞 y 的轉動不變號）。
- `fixed_cmd`（攤帳、驗收、golden）固定 `mir=False`。
- metrics 加 `mir`（回合比例，曲線上應在 0.5 附近）。

### 2.2 `tilt`：姿態角懲罰（`w_tilt`，起 10）

- `tilt = −w_tilt · g_y²`，g ＝ **真實**重力在機身系的投影（`_obs` 的 `grav`，不是帶偏差的 obs）。
- 只罰 roll：上樓梯／斜坡俯仰是正當的。`angxy` 原樣留著。
- 量級：roll 9° → sin²(9°)·10 ≈ −0.25／步（與平移多出來的 `tau_leg` 同級）；3° → −0.03。

### 2.3 `impact`：落地衝擊（`w_impact`，起 3）

- `impact = −w_impact · Σ_腿 [touch_k ∧ ¬touch_prev_k] · max(0, −vz_prev_k)²`，`vz_prev_k` ＝ 觸地**前一步**的足端垂直速度（`info` 多存 `touch_prev`（4 bool）與 `vfoot_prev`（4×3））。
- 接觸法向力峰只記 metrics（`impact_fn`），不進 reward（16 邊形輪的力會跳）。
- 估計：v1b 平移抬 35 mm、3 步/秒 → 落地 vz 0.3～0.6 m/s，每步平均落地事件 ≈ 0.24 → 每步 ≈ −0.05×w ≈ −0.15。
- metrics 加 `impact`、`impact_vz`（落地 vz 平均，診斷）。

### 2.4 動作延遲 DR（`delay_max=2`）

- 出生／重生時抽 `d ∈ {0, 1, 2}`，**整回合固定**（實機延遲是系統性的 ≈ 30 ms，不是每步抖動）。`info["a_hist"]` 長度 3 的環形緩衝（物理側動作）。
- `_physics` 吃 `a_hist[d]`；`info["last_a"]`／obs 的 `last_action` 仍是 policy 這步的輸出（狗上 `rl_obs_rl` 餵的就是 policy 自己的上一次輸出）。`qvel_prev`／`foot_prev` 不受影響。
- 重生時緩衝清零、重抽 d。`fixed_cmd` 下 d＝0。metrics 加 `delay`。

### 2.5 IMU 常數偏差 DR（`grav_bias_deg=2.0`）

- 出生／重生時抽 roll、pitch 各 U(−2°, 2°)，整回合固定；只旋轉 `state` 的重力三維（`grav_obs = R_x(roll)·R_y(pitch)·grav`），角速度、critic 的 `privileged_state` 不動。
- 既有 obs 雜訊照舊疊在後面。`fixed_cmd` 下偏差 0。

### 2.6 不動的

`wgap`（80 mm、2.0）、`ww`、`lim`（2.0）、`w_ang`（2.5）、σ 0.25、`acc_leg`／`rate`／`dev`、課程（2 m）、推力、既有雜訊與 DR、網路、匯出格式。`TERM_KEYS` 加 `tilt`、`impact`（16 項）；`METRIC_KEYS` 加 `mir`、`delay`、`impact_vz`、`impact_fn`。

## 3. 開訓前關卡（本機 CPU MJX）

1. **golden 原樣通過**（§5.1）。
2. **攤帳**：`diag/wheel_env_align.py --preset v1c` 拿 10k 與 v1b 跑（`fixed_cmd`，鏡像／延遲／偏差關），`outputs/reward_audit_wheel_{10k,v1b}_v1c.md`。要看到：
   - `impact`：10k 的值 < v1b 的一半（量的才是使用者聽到的那個聲音）。不過 → 權重不定、不開訓。
   - `tilt`：v1b 左平移 0.5 明顯大於右平移 0.5（約 −0.25 對 −0.03）。
   - 新增兩項在 v1b 平移時合計每步 −0.1～−0.5，沒有單項 > 1（不淹沒追蹤項）。
3. **鏡像對齊**（§5.2 的測試即是）。

## 4. 訓練（`notebooks/build_nb_wheel.py` → `rl_wheel_v1c_colab.ipynb`）

- 超參數、網路、throughput 格、RESUME 機制沿用 v1 spec §5；`env = rw.WheelEnv(**rw.V1C)`、`eval_env = rw.WheelEnv(noise=False, push=False, **rw.V1C)`；存檔名 `rl_wheel_v1c_step*.pkl`、`rl_wheel_v1c_params.pkl`；`RESUME="auto"` 只認 v1c 的檔。
- seed 0、`TIMESTEPS` 5 億／session、兩個 session。
- 進度列多印 `tilt`、`impact`、`impact_vz`、`mir`、`delay`。
- **停損線（累計 2 億步）**：沿用 `len < 600`／`lin < 0.5`／gap 中位 < 40 mm；新增 `impact`、`tilt` 沒比 step 0 往 0 走。
- **5 億步中場檢查**：session 1 結束把 `step000500000000.pkl` 拉回本機匯出、跑 §6 全表。roll 左右差沒縮、或 `impact_vz` 沒比 v1b 低 → 先調權重再跑 session 2。

## 5. 測試（pytest；先寫測試再改程式；全套每批跑一次，`test_rl_env_v3_golden.py` 不能動）

1. **`tests/test_rl_env_wheel_golden.py`（改任何程式之前先寫、先產生、先 commit）**：`WheelEnv()`（預設＝v1b）固定 seed、固定動作序列、三條指令 12 步的 obs／reward／qpos／ctrl 逐位元；寫法照 `test_rl_env_v3_golden.py`。改完所有東西它必須原樣通過。
2. **`tests/test_rl_env_wheel_mirror.py`**：
   - 鏡像表兩次 ＝ 恆等（obs 與 act）。
   - 把 MuJoCo 狀態真的左右翻（`qpos` y 取負、四元數鏡射、腿互換、ABAD 變號；`qvel` 同理）→ `_obs` 的結果等於原 obs 套鏡像表（1e-6）。
   - 翻過的狀態用翻過的動作跑 5 步，軌跡等於原軌跡翻過去（1e-4；同時驗證模型左右對稱）。
   - `mir=True` 回合：`state.obs` ＝ 鏡像表(真實 obs)；物理 ctrl ＝ 鏡像表(policy 動作)。
3. `test_rl_env_wheel_reward.py` 加：`tilt` roll 9° ＝ −w·sin²(9°)、水平 0、只動 pitch 0；`impact` 觸地事件（上步沒觸地、vz_prev −0.5）＝ −w·0.25，持續觸地／抬腿時 0；`TERM_KEYS`／`METRIC_KEYS` 更新。
4. `test_rl_env_wheel_train.py` 加：d＝2 時第 t 步 ctrl ＝ 第 t−2 步動作、d＝0 同現況、obs `last_action` 不延遲、重生清緩衝；IMU 偏差 (2°, 0) 時 `state` 重力偏 2° 而 `privileged_state` 不偏；`mir`／`delay` 分布。
5. `test_notebook_wheel.py`：產生的 ipynb 含 `V1C`、`rl_wheel_v1c` 檔名、新進度欄。
6. `test_export_policy_wheel.py` 不動（匯出格式不變）。

## 6. 訓完的及格線（同一套工具、同一套旗標；v1b 的表當門檻，任一格不過不上機、v1b 權重不動）

| 類別 | 門檻 | v1b 現值 |
|---|---|---|
| 摔 | 平地 real／ideal／擾動 13 條、斜走 9 條（`eval_rl_only_real_diag_*`）、倒退 5 條、崎嶇地 114 趟 | 0／0／0／0／0／1 |
| 追蹤 | vx 0.3／0.5／1.0 誤差 ≤ 0.05；vy 0.3 ≥ 0.25；vy ±0.5 ≥ 0.42；wz 0.5 ≥ 24°/s；wz 1.0 ≥ 54°/s | 0.33／0.44～0.46／26～27／57 |
| **對稱（目的）** | vy ±0.5 的 roll 峰左右差 ≤ 3°，兩邊都 ≤ 6°；斜走 ±0.3 同樣 | 9.3 對 3.6 |
| **落地輕（目的）** | vy ±0.5 的 `impact_vz`（驗收表新欄）低於 v1b；聽覺上機再驗 | 待攤帳定 |
| 安全 | ABAD 誤差峰 ≤ 0.70、髖膝 ≤ 0.45；關節速峰 ≤ 9.4；後輪 gap ≥ 200 mm；飽和 ≤ 6% | 0.56／0.28／9.4／210～325 |
| 延遲／偏差強健 | 延遲 1 步＋雜訊全表 0 摔；延遲 2 步左平移 0.5 不摔、誤差峰在護欄 0.80／0.55 內；`--grav-bias 2 2` 不歸零直走／平移 0 摔 | 延遲 2 步 roll 峰 27°、誤差破護欄 |

驗收工具改動：`rl_only_infer.py`、`diag/rl_only_rough_terrain.py` 加「落地 vz」欄（同 §2.3 口徑，CPU 端算），其餘欄位不變。

## 7. 匯出與上機

`export_policy_wheel.py --params weights/rl_wheel_v1c_params.pkl --out weights/rl_wheel_v1c_np.npz`（工具不動）；影子跟跑 `RL_ONLY_NPZ=… pytest tests/test_policy_rl_{np,closed_loop}.py`；trip29 操作卡照 trip26 卡 §8 的格式，順序站立 → 直走 → 轉 → 平移 → 斜走，重點盯左右 roll 與踏地聲。

## 8. 檔案清單

- 改：`inference/rl_env_wheel.py`、`inference/diag/wheel_env_align.py`、`inference/rl_only_infer.py`、`inference/diag/rl_only_rough_terrain.py`、`notebooks/build_nb_wheel.py`、`tests/test_rl_env_wheel_reward.py`、`tests/test_rl_env_wheel_train.py`、`tests/test_notebook_wheel.py`。
- 新：`tests/test_rl_env_wheel_golden.py`（＋`tests/data/rl_env_wheel_golden.npz`）、`tests/test_rl_env_wheel_mirror.py`、`outputs/reward_audit_wheel_{10k,v1b}_v1c.md`。
- 不動：`export_policy_wheel.py`、`realbot/policy_np.py`、`realbot/rl_obs_rl.py`、`realbot/M9_gait.py`、`rl_env_v3.py` 與其 golden。

## 9. 順序

golden 測試（釘 v1b）→ 環境改動（TDD，§5）→ 攤帳關卡（§3）→ notebook 產生與測試 → Colab session 1（5 億）→ 中場檢查 → session 2 → 全表及格線（§6）→ 匯出 → trip29 卡。
