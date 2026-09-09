# RL v2.3 policy 上機 —— 實作計畫（2026-09-09）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `cpg_rl_max_v2_3_params.pkl` 以純 numpy 跑在 D1 Max 狗上（M9 `--policy`），上機前本機驗證兩套堆疊逐位元相同。

**Architecture:** 本機用 jax 匯出 npz；狗上 `policy_np`（MLP）＋ `rl_obs`（shm → 66 維 obs）＋ `PolicyGaitStream`（在 `GaitStream` 的 50 Hz 推進點插入 obs→policy→CPG 調變）。退路永遠是開迴路 A：任何異常當步用基準動作，`o` 鍵永久退回。

**Tech Stack:** Python 3.10 / numpy 1.21（狗）；jax / brax 0.14.2 / mujoco 3.10（本機驗證）。測試 `conda run -n rbtdog pytest task7/tests`。

## Global Constraints

- 狗上只能 import：標準庫、numpy、`realbot/` 內檔案。不得 import `inference/` 或 mujoco。
- 常數（MU 1–2、OMEGA 0–2、SWAY_MAX 0.060、SWAY_SLEW 0.004、HOME12、腿序 FR/FL/RR/RL、`A["mu_x"]` 1.8）在狗端檔內寫死，測試釘住等於 `inference/` 的值。
- v2.3 = layout `nomux`、act 10、obs 66、無 head_err、淡入 50 步。
- 護欄與階段流程不動：只在 `GaitStream` 推進點與 `--policy` 參數上動 M9。
- 每個 task 結束 pytest 全綠再 commit。

---

### Task 1: `export_policy_np.py` ＋ `policy_np.py`

**Files:** Create `task7/inference/export_policy_np.py`、`task7/realbot/policy_np.py`、`task7/tests/test_policy_np.py`；產出 `task7/weights/cpg_rl_max_v2_3_np.npz`。

**Interfaces（Produces）:**
- `policy_np.load(path) -> Policy`；`Policy.infer(obs: np.ndarray(66)) -> np.ndarray(10)`；屬性 `obs_dim, act_dim, preset, layout, baseline: dict`。
- `policy_np.act_to_cmd(a, layout="nomux") -> (mux4, muy4, om4, sway_t2)`（MJCF 腿序）。
- `policy_np.baseline_action(layout="nomux") -> np.ndarray(10)`；`policy_np.slew_sway(prev2, tgt2)`。
- npz 鍵：`mean, std, W0..W3, b0..b3, obs_dim, act_dim, preset, layout, baseline_json, src_sha256`。

brax 結構（已查）：params = (normalizer, policy, value)；normalizer.mean/std (66,)；policy `hidden_0..3`，`hidden_3` 輸出 20 = 2·act；`normalize = (x−mean)/std`，無 clip；隱藏層 swish；deterministic = `tanh(loc)`，loc = 前 10。

- [ ] 測試：`test_forward_matches_brax`（1000 筆 obs ~ N(0,1)·3，max|Δ| < 1e-5）、`test_constants_pinned`、`test_act_to_cmd_same_as_local_infer`（100 筆隨機 a，逐點相同）、`test_baseline_action_same`、`test_slew_same`、`test_npz_meta`（obs 66/act 10/preset v2.3/baseline == BASELINE_A 子集）。
- [ ] 跑測試失敗 → 寫匯出與前向 → 測試過 → commit。

### Task 2: `rl_obs.py`

**Files:** Create `task7/realbot/rl_obs.py`、`task7/tests/test_rl_obs.py`。

**Interfaces（Produces）:**
- `rl_obs.Frame(pos12_shm: dict name->rad(motor), vel12_shm: dict name->rad/s(motor), quat_xyzw(4), gyro(3))`。
- `rl_obs.build(frame, c: dict(cpg dict fl/fr/bl/br), cmd(2), last_a(10)) -> np.float32(66)`。
- `rl_obs.LEG_MJCF = ("FR","FL","RR","RL")`、`rl_obs.LEG_SHM = {"FR":"fr","FL":"fl","RR":"br","RL":"bl"}`、`rl_obs.LEG_NAMES`（12 個 shm 關節名，MJCF 序）。
- `rl_obs.to_shm_legs(arr4) -> dict{fl,fr,bl,br}`、`rl_obs.from_shm_legs(d) -> arr4(MJCF 序)`。
- `rl_obs.w2b(quat_wxyz, v)`。
- `rl_obs.read_frame(state_ro: Shm, imu: Shm, idx: dict) -> Frame`（狗上用，測試不跑）。

- [ ] 測試：`test_matches_obs_max`（隨機 quat 正規化、gyro、關節、CPG 四腿不同值：`rl_obs.build` vs `real_obs.RealFrame` → `obs_max.build_obs`，`np.array_equal`）、`test_leg_order_shuffle_detected`、`test_w2b_matches_cpg_max`、`test_constants_pinned`（HOME12、LEG_NAMES == real_obs.LEG_NAMES）。
- [ ] 實作 → 測試過 → commit。

### Task 3: 閉迴路「說兩次」測試

**Files:** Create `task7/tests/test_policy_closed_loop.py`。

在 MuJoCo（`cpg_walk_max.Robot`，原始網格模型）跑 200 步：jax 堆疊（`obs_max` + `load_policy` + `cpg_max`）與狗端堆疊（`rl_obs` + `policy_np` + `realbot/cpg` + `realbot/kin`）**共用同一個機器人狀態**，每步各自算 obs、動作、12 目標角；斷言 obs `array_equal`、動作 max|Δ|<1e-6、目標角 max|Δ|<1e-9（`realbot/cpg` 與 `cpg_max` 本來就有逐幀對照測試，若 IK 有 1e-9 以上差異，放寬到 1e-6 並記下）。狗端 CPG 狀態以 dict 維護，`to_shm_legs`/`from_shm_legs` 轉換。含淡入 50 步與 sway 斜率。

- [ ] 寫測試 → 跑 → 修差異 → commit。

### Task 4: M9 `--policy`

**Files:** Modify `task7/realbot/M9_gait.py`（args、`PolicyGaitStream`、說兩次、log、鍵 `o`）；Modify `task7/tests/test_m9_gait.py`。

**Interfaces:**
- args：`--policy PATH`（需 `--traj` ＋ `--interactive`）、`--vx 0.30`、`--wz 0.0`、`--policy-gain 1.0`。
- `PolicyGaitStream(GaitStream)`：`__init__(p, f0, ks, policy, cmd, gain, reader: callable()->Frame, log: list)`；覆寫 `sample()` 的推進；屬性 `fallback_forever: bool`、`n_fallback`、`set_open_loop()`。
- 說兩次：traj `params`/`baseline_ref` 與 `policy.baseline` 比對（omega, duty, d_step, d_step_y, x_off, g_c, z_sag, mu_x, mu_y, seq, sway 全 0），不一致印表並 return 1。
- 退回條件：obs NaN/inf、`frame` 的 state tick 未前進、推論 > 10 ms → 當步 `base_act`；累計 25 步 → `fallback_forever`。
- 鍵盤：`KeyWatch.pressed()` 改回傳按到的行內容；GAIT 中輸入 `o`+Enter → `set_open_loop()`，空行 Enter 維持原語意（推進階段）。
- log：`out["policy"] = {"path", "vx", "wz", "gain", "steps": [{"t", "obs", "a", "act", "sway", "imu", "ms", "fb"}]}`。

- [ ] 測試：假 reader（回傳固定 Frame，tick 遞增）＋ 假 policy（回傳基準動作）→ `PolicyGaitStream` 100 步 12 目標角與 `GaitStream` 逐幀 `array_equal`（gain 1 且 a=base 也應相同）；gain 0 時任意 policy 亦相同；NaN obs → 退回計數 +1；`set_open_loop()` 後 sway 回 0 且動作＝基準；AST 呼叫點測試仍過。
- [ ] 實作 → 全部測試 → commit。

### Task 5: `replay_policy.py` ＋ 跑 trip17 回放

**Files:** Create `task7/inference/replay_policy.py`；輸出 `task7/outputs/replay_policy_trip17.md`。

輸入 `logs/m_logs_trip17/M9_*.json`（找兩趟 17:00/17:01），經 `m9_rec.load` → 每幀 `rl_obs.Frame` 與 `real_obs.RealFrame`；CPG 狀態用固定假值（四腿 rx 1.2/1.4/1.6/1.8 等）；last_a 用基準；比 obs 各欄 max|Δ| 與動作 max|Δ|；印 obs 各欄範圍（gravity、gyro、joint_pos、joint_vel）供乾跑對照。標注 gyro_z ≡ 0 的限制。

- [ ] 寫 → 跑 → 結果進 md → commit。

### Task 6: 部署與操作卡

**Files:** Modify `task7/realbot/push_to_dog.sh`（加 `policy_np.py`、`rl_obs.py`、`weights/cpg_rl_max_v2_3_np.npz`）；Create `task7/docs/現場操作卡_RL_v2.3_2026-09-09.md`；Modify `task7/HANDOFF.md`（頂部加節）、`task7/docs/README.md` 索引。

操作卡內容：push、狗上 import 檢查、三段上機（gain 0 乾跑 5 s → gain 1 短走 5 s → 10 s）指令逐字、每段看什麼數字、中止就拉 log。

- [ ] 寫 → 若有 push_to_dog 測試更新 → commit。
