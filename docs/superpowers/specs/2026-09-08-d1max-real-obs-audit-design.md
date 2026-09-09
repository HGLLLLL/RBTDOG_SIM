# D1 Max 實機 observation 盤點 —— 設計（2026-09-08）

## 目的

CPG-RL 以 `A_kp250_walk`（LS/kp250/abad60/kd2/wheel_kd0.5）為基準重訓之前，
把 68 維 observation（`task7/inference/obs_max.py`）裡**每一個來自感測器的欄位**
在實機上的真實樣貌量清楚，產出訓練端直接可用的**雜訊模型 + 延遲 + 座標換算**，
避免「訓練完才發現上機讀到的東西跟模擬不一樣」。

使用者決策：**IMU（重力向量 + 角速度）可以進 obs，先驗證再訓練。**
使用者約束：**實機測試成本高，今天一輪就要量齊，之後不能再回頭補量。**

## 現況（盤點前）

| obs 欄位 | 維 | 實機驗證 |
|---|---|---|
| `joint_pos` | 12 | ★ 1 kHz 活串流、座標鏈 V1 已閉合、每趟 M9 都在記 |
| `joint_vel` | 12 | 有記；**已知雜訊 47%**（角度差分 9%）；模擬裡是乾淨的 |
| `gravity`（由 quat） | 3 | roll/pitch 有記；xyzw 只有旁證；8/26 吊掛時 IMU roll +7.1° vs 力矩反推 +2.5° **未解** |
| `gyro` | 3 | **從沒任何一趟記過**：單位、軸向、正負全未知 |
| `cmd`/`last_action`/`cpg` | 38 | 自產，與感測器無關 |

另外三個「不是欄位但會讓上機失敗」的未知：`imu_central` 的**更新率**；
狗上**有沒有 numpy、MLP 前向幾 ms**（兩份自家文件互相矛盾，從沒查過）；
**指令→動作的閉迴路延遲**（只知道迴圈 200 Hz、最壞 gap 12 ms）。

## 上機流程（全部唯讀；不改 M9、不改 M6、不碰 `joint_cmd`）

錄製一律用既有的 `task7/realbot/M6_load_probe.py <label> --record`
（同輪讀 `joint_state`/`joint_cmd`/`imu_central`，每筆記 16 關節 `{q,v,tau,des,ff,kp,kd}` + IMU 原始 10 值）。

| 階 | 狗上指令 | 狗的狀態 | 操作者要記的 |
|---|---|---|---|
| I-a 平放 | `M6 flat --record --secs 30 --hz 200` | 趴平不動，地面用手機水平儀確認 | — |
| I-b 跳舞 | `M6 dance --record --secs 90 --hz 500` | 遙控器跳舞（左右扭動） | **一開始先往狗的哪一側倒** |
| I-c 原地轉 | `M6 turn --record --secs 30 --hz 200` | 遙控器原地轉：左轉、停、右轉（沒有就跳過） | 先左還是先右 |
| I-d 電腦探測 | `python3 M_env_probe.py` | 趴著（測的是 RK3588 這顆電腦，與狗的動作無關） | — |
| II 步態側錄 | ssh#2：`M6 walk --record --secs 120 --hz 200`；ssh#1 照 `現場操作卡_互動式前進_trip17_2026-09-03.md` 跑 `A_kp250_walk` | 站→走 10 s→停→趴 | 同 9/3 |

每階回答什麼：

- **I-a**：quat 順序（xyzw/wxyz 兩種解，平放時哪個 roll/pitch≈0）、IMU 安裝偏置（vs 腿 FK）、
  acc 單位（模長 9.81 或 1.0）、gyro 靜態偏置與雜訊、`joint_pos`/`joint_vel` 靜態量化步階與雜訊。
- **I-b**：`gyro = k · ω(quat)` 逐軸擬合 → k≈1 rad/s、k≈57.3 deg/s、k<0 軸反；
  IMU roll/pitch vs 腿 FK 傾角（四輪共面）的偏置與相關；**第 1 個 roll 極值在第幾秒、往哪側**（對眼睛）。
- **I-b（500 Hz）**：順帶得到 `imu_central` 與 `joint_state` 的更新率（相鄰筆相同值計數；每筆都變 → ≥500 Hz）。
- **I-c**：gyro z 正負（左轉 = 由上往下看逆時針 = MJCF +z）。
- **I-d**：numpy / onnxruntime 有無；pure-Python 與 numpy 的 (68→256→256→128→12) 前向時間；
  CPU 核數與親和性限制。
- **II**：走路時 30 個感測欄位的真實分布；`joint_vel` 雜訊頻譜（200 Hz 看 50 Hz 以上混疊）；
  IMU 動態雜訊；**des→q 閉迴路延遲**（M6 同輪記 des 與 q）；IMU vs 關節相對延遲；
  mc_ctrl 凍結期間 IMU 仍活著（再次證實）。

對齊：M6 記到的 `des` 就是 M9 寫進 `joint_cmd` 的值，逐位元相同 → 兩份 log 用 `des` 對齊，不靠波形猜。

## 離線工具（全部新增於 `task7/inference/`，`task7/realbot/` 只加 `M_env_probe.py`）

### ① `real_obs.py` —— M6 錄檔 → 68 維 obs 序列

- 輸入：M6 `--record` 的 JSON。輸出：`(N, 68)` float32 + 時間軸 + 附帶的 raw dict。
- **不另寫一份數學**：造一個帶 `qpos`/`qvel` 的假 `MjData`（`RealFrame`），餵**同一支 `obs_max.build_obs`**。
- 換算（都是既有式子）：
  - quat `xyzw` → `wxyz`（`w2b` 吃 MuJoCo 順序）；順序由參數 `--quat-order` 指定，預設 `xyzw`，由 ② 定案
  - 關節 馬達座標 → 控制器座標：`coord.to_ctrl`（V1：`ctrl = (motor − offset) / side_sign`），速度只乘 `side_sign`
  - SHM 腿序 `fl,fr,bl,br` → MJCF `FR,FL,RR,RL`，按名稱對應（`play_gait_traj.MM2SHM` 的反向）
  - gyro 乘 ② 定案的 `k`（單位/軸向），置入 `qvel[3:6]`
- `cmd`/`last_action`/`cpg` 38 維填零（自產欄位，對照不看它們）。
- CLI：`real_obs.py rec.json --out obs.npz`。

### ② `imu_check.py` —— 階 I 判讀

- 輸入：flat / dance / turn 三個錄檔（缺 turn 可省）。
- 輸出一張表（stdout + `outputs/imu_check.json`）：
  - 平放：兩種順序解出的 roll/pitch；acc 模長；gyro 偏置 (3)；gyro 雜訊 std (3)；joint q/v 量化步階與 std
  - 跳舞：逐軸最小平方擬合 `gyro_i = k_i · ω_i(quat)`，附相關係數；ω(quat) 由 `q_{t+1} ⊗ q_t^{-1}` 轉機身系角速度
  - 跳舞：IMU roll/pitch vs 腿 FK 傾角（`leg_kin` FK，四輪共面平面擬合）的偏置、斜率、相關；FK 不共面的樣本標記並排除
  - 跳舞：第 1 個 |roll|>2° 極值的時間與符號，翻成「往狗的左/右」
  - 跳舞（500 Hz）：`imu_central` 與 `joint_state` 的更新率（相鄰筆值不變的比例 → Hz）
  - 轉向：gyro z 在「先轉那段」的平均符號
- 判定寫死在程式裡並印出結論句：`quat_order`、`gyro_scale`（3 軸 ±1 或 ±57.3）、`imu_mount_rp_deg`、
  `imu_rate_hz`、`gyro_bias`，並存進 `outputs/imu_check.json` 供 ① 與訓練讀。

### ③ `obs_compare.py` —— 階 II 逐欄對照

- 模擬側：沿用 `play_gait_traj.py` 的播放迴圈（抽出成可呼叫函式，不改其 CLI 行為），每 50 Hz 步收 `build_obs`。
- 實機側：walk 錄檔經 ① → obs；用 `des` 找 M9 的 GAIT 段（`kp` 從 0 升起後、`des` 開始週期變化）；
  對齊到模擬同一相位（膝 `des` 互相關）；重取樣到 50 Hz。
- 每欄輸出：sim/real 的 mean、std、min、max；雜訊 std（對 50 Hz 序列做 5 點移動中位數的殘差）；
  real 對 sim 的相關與延遲 ms；符號一致性（相關 <0 標紅）。
- 額外三項：`joint_vel` 的 real vs 角度差分 之比較（雜訊倍率、相關）；200 Hz `joint_vel` PSD 在 25 Hz 以上的能量占比；
  **des→q 延遲**：每個腿關節 `q` 對 `des` 的互相關峰值延遲（ms）。
- 產出：`outputs/obs_noise_model.json`
  ```
  {"schema":"obs_noise/1","source":"...","quat_order":"xyzw","gyro_scale":[...],
   "latency_ms":{"des_to_q":..,"imu_vs_joint":..},
   "noise_std":{"gravity":[..3],"gyro":[..3],"joint_pos":[..12],"joint_vel":[..12]},
   "range_real":{...},"range_sim":{...}}
  ```
  與 `docs/results/H_實機obs盤點_2026-09-08.md`（表格由程式印、文字我寫）。

### ④ `realbot/M_env_probe.py` —— 狗上環境探測（零風險）

- 純標準函式庫；不開任何 shm。
- 印：python 版本、numpy/onnxruntime 有無與版本、`os.cpu_count()`、`os.sched_getaffinity(0)`、
  pure-Python (68,256,256,128,12) MLP 前向 100 次的平均/最大 ms；若有 numpy 再量 numpy 版；
  結果存 `~/m_logs/ENV_*.json`。

## 測試（`task7/tests/`）

- `test_real_obs.py`：用 MuJoCo rollout 合成一份假 M6 錄檔（已知 xyzw、gyro rad/s、馬達座標）→ ① 還原的 obs
  與 rollout 當下 `build_obs` 的 30 個感測欄位逐點相符（atol 1e-6）；腿序錯置的錄檔要被抓到（名稱對應）。
- `test_imu_check.py`：合成資料（已知順序 wxyz、gyro deg/s、軸 y 反）→ ② 判出 `quat_order=wxyz`、
  `gyro_scale=[57.3,−57.3,57.3]`；更新率合成 250 Hz 要判成 250±5%。
- `test_obs_compare.py`：模擬 obs 加已知雜訊與 3 步延遲當假實機 → ③ 量回的雜訊 std 與延遲在 ±20% 內。
- `test_m_env_probe.py`：MLP 前向對 numpy 參考結果一致（在本機跑）。
- 既有 719 項不動、全綠。

## 明確不做

不改 `M9_gait.py`、不改 `M6_load_probe.py`、不寫 `joint_cmd`、不訓練、不改 `obs_max.OBS_LAYOUT`。
今天的終點：`outputs/imu_check.json`、`outputs/obs_noise_model.json`、`docs/results/H_實機obs盤點_2026-09-08.md`。

## 風險

- 跳舞可能抬腳 → 腿 FK 共面參考在那些樣本失效；程式標記排除，靠 acc 與 quat↔gyro 撐。
- 遙控器沒有原地轉 → gyro z 正負只剩 quat 微分的一致性（假設 quat 的 yaw 軸向正確）；文件註明。
- 500 Hz 錄 90 s Python 可能偶爾掉拍 → 更新率用實際達到的 dt 算，估的是下界。
