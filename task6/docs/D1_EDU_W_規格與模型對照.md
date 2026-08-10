# 智元 D1 EDU 輪足版（ZSL-1w）規格、模型與 SDK 對照

本文是 `task6/` 的規格總表：官方標稱值、模擬模型實際採用值、實機 SDK 能力邊界，
以及三個尚未定案的風險項（腿序、kp/kd 來源、部署路徑）。

**所有數值以程式碼為準**，主要出處：

- `task6/inference/d1_model.py` — 常數的單一事實來源
- `task6/model/d1_edu_w/d1_edu_w.xml` — MJCF 本體
- `task6/model/d1_edu_w/SOURCE.md` — 資產來源與授權
- `task6/docs/D1EDU_輪足_lowlevel_調查與實測指南.md` — SDK / 板載介面的完整調查與實機實測

---

## 1. 官方規格（來源：https://www.agibot.com/products/D1_Pro）

| 項目 | 數值 |
|---|---|
| 重量 | 15.5 kg（含電池）← **點足版數字**，輪足版 URDF 為 20.56 kg |
| 站立尺寸 | 635 × 360 × 420 mm |
| 自由度 | 12（每腿 abad / hip / knee）＋ 輪足版另有 4 顆輪（URDF 共 16 個 revolute 關節）|
| 關節峰值扭矩 | 48 N·m（URDF `effort` 為 28，本專案取 28）|
| 關節行程 | 側擺 ±28°、大腿 −170°~66°、小腿 35°~156° |
| 最大速度 | 3.5 m/s ｜ 最大爬坡 40° ｜ 連續上階 16 cm |
| 額定負載 | 5 kg |

質量差異的解釋：20.56 − 15.5 ≈ 5.06 kg ≈ 4 ×[(0.901 − 0.063) + (0.765 − 0.207)]，
正好是輪子與加粗小腿相對點足版的增重，兩份資料不衝突（詳見 `model/d1_edu_w/SOURCE.md`）。

---

## 2. 模擬模型採用值

| 項目 | 值 | 出處 |
|---|---|---|
| 總質量 | 20.56 kg | `ZSL-1W.urdf` 17 個 link 之和（`test_total_mass_matches_urdf`）|
| 機身 link 質量 | 6.7155497 kg | URDF `BASE_LINK` |
| 輪 | 半徑 0.0710 m、半寬 0.0240 m、每顆 0.90130429 kg | STL 實測 + URDF |
| 輪碰撞 geom | `cylinder`、`condim=6`、`friction 0.8 0.02 0.01`、`priority=1` | 鎖死的輪貼地是一條線，不是點 |
| 力矩上限 `TAU_MAX` | 28 N·m（MJCF `forcerange="-28 28"`）| URDF `effort`（官網 48 為峰值，取保守值）|
| kp / kd | 80 / 1（MJCF `gainprm="80 0 0"` / `biasprm="0 -80 -1"`）| 點足版官方 demo `lowlevel_demo.py:68-73` ⚠ 見下 |
| home 關節角 `HOME3` | `[0, +1.05, −2.00]` | 解輪心在髖正下方；**膝軸 `+y`**（點足版為 `-y`）|
| keyframe 機身高 | 0.2948 m | **純運動學值**（四輪剛好觸地）|
| `NOMINAL_HEIGHT` | **0.2695 m** | 實際站定高度；實測 1s/2s/3s/5s = 0.26981 / 0.27002 / 0.26932 / 0.26907 |
| `CTRL_DT` / `SIM_DT` | 0.02（50 Hz）/ 0.004 | 落在 SDK 建議的 20~50 Hz 內 |
| `D_STEP` / `D_STEP_Y` | **0.12 / 0.09** | 前後 / 側向兩個獨立尺度 ⚠ 見 §2.2 |
| `OBS_DIM` / `ACT_DIM` | **69** / 12 | 見 §4 |
| armature / damping | 0.01 / 0.1 | **URDF 未提供，為假設值** |
| 求解器迭代 | CPU 場景用 MuJoCo 預設 100/50；`scene_mjx.xml` 才降為 1/5 | 實測 1/5 會讓同段開迴路只走 3.82 m（預設 4.57 m）|

### 2.1 ⚠️ kp/kd 的來源限制

輪足版 SDK 沒有 LowLevel demo，這組值取自**點足版** demo。
已由關卡 2、3 驗證可用：站定沉降 2.47 cm 後停住（之後 1 秒 z 峰對峰 1.36 mm、末速 −0.0004 m/s），
膝力矩遠低於 28 N·m 上限。但若日後取得輪足版官方值應以官方為準。
訓練時已對 kp ∈ [60, 100]、kd ∈ [0.5, 2.0] 做 domain randomization，容忍此不確定性。

### 2.2 ⚠️ 為什麼側向步幅是另一個常數

側向足端偏移只由 abad 關節吸收（靈敏度約 4.47 rad/m），而本機 abad 行程僅 **±28°**，
遠小於 task4 目標機 Go2 的 ±60°。若沿用 `D_STEP = 0.12`，abad 目標角會達 ±0.536 rad，
**超出致動器 ctrlrange ±0.4687 約 14%**，指令會被 clip、步態失真。
因此拆成兩個常數：前後維持 0.12（不犧牲走速），側向改用 `D_STEP_Y = 0.09`。
測試 `test_d_step_y_is_smaller_than_d_step` 釘住這個關係。

---

## 3. ⚠️ sim2real：官方 SDK 確定不提供，板載介面另有一條獨立路徑

這一節的結論**分兩件事**，兩件事互相獨立，不要混為一談。

### 3.1 ✅ 已確認：官方 SDK 不提供輪足版的關節控制

官方文件 `docs/deploy.md:168` 原文：

> **「ZSL-1w 不提供 LowLevel 接口，因此仅有 `highlevel_demo`。」**

三路交叉驗證：

| 檢查點 | 點足版 zsl-1 | 輪足版 zsl-1w |
|---|---|---|
| `include/<機型>/lowlevel.h` | ✅ 有 | ❌ 只有 `highlevel.h` |
| `docs/api_<機型>.md` | 有「2. LowLevel函数介绍」| ❌ 全篇只有 HighLevel |
| `demo/<機型>/` | lowlevel + highlevel | ❌ 只有 highlevel |
| `nm -D` 導出符號（最硬的證據）| `sendMotorCmd` / `getMotorState` / … | **0 個** LowLevel 符號 |

輪足庫只導出 47 個符號，全部是 `mc_sdk::zsl_1w::HighLevel::` 開頭。
新 SDK（`zsibot/genisom_L1_sdk` v1.0.0）的 `libzsibot.a` 同樣零命中。
**不是漏放標頭檔，是庫裡真的沒有。**

實機透過 HighLevel **能讀** 16 個關節的角度/角速度/力矩與 IMU，
但**寫**只到 `move(vx, vy, yaw_rate)` / `crawl` / `climb` 這一層，下不了關節指令。

### 3.2 ❓ 未定案：板載 `/spline_shm` 共享記憶體介面

對三個 `.so` 做 `strings` 搜 `spline` / `shm` / `/dev/shm`，**零命中**。
代表這條路**根本不在 SDK 庫裡**：`create_spline_shm()` 是標頭檔裡的 `static inline`，
使用者程式直接對機上 daemon 通訊、**完全繞過 SDK**。

→ 所以「輪足庫沒有 LowLevel」這個事實，**管不到共享記憶體那條路**。

目前狀態（2026-08-10，實機序號 SN01PW8626B0012）：

| 面向 | 狀態 |
|---|---|
| SHM **讀取** | ✅ 已在實機驗證：`/spline_shm` 存在、firefly 非 root 可讀、~880 Hz、IMU 四元數與時戳正常 |
| SHM **寫入（下發關節指令）** | ❓ **未驗證**，風險最高的一步 |
| 關節讀值與 URDF 限位 | ⚠️ **對不上**（abad 讀 ±0.98 vs URDF ±0.49；knee 讀 ≈0 vs URDF −2.72~−0.60）→ sim2real 必須補一層「每關節 offset + 正負校正」|

**正確的表述**：
✅ 官方 SDK 不提供輪足版關節控制（已確認）；
❓ 板載 `/spline_shm` 的**寫入**路徑對輪足版是否可用，**尚未經硬體驗證**。
不要說成「完全沒有部署路徑」，也不要說成「已經有部署路徑」。

完整的調查證據、SHM 資料結構、握手協定與三階段實測步驟見
**`task6/docs/D1EDU_輪足_lowlevel_調查與實測指南.md`**（1022 行）。

### 3.3 SDK 已搬家，兩邊都要看

| Repo | 狀態 |
|---|---|
| `zsibot/genisom_L1_sdk` | **現行**，v1.0.0 / MIT / 2026-06-22 發布、2026-08 仍在維護。UDP+JSON、role-based 仲裁、16 路電機溫度 |
| `AgibotTech/agibot_D1_Edu-Ultra` | 官方標示 **停止維護**（commit `db8accd`, v0.2.7），但**文件較全**（新 repo 連 `deploy.md` 都沒有）|

⚠️ 新 repo 裡沒有「zsl-1w 不提供 LowLevel」那句話，但那是**弱證據**——因為新 repo 整份
`deploy.md` 都不存在。**缺席不等於解禁。**

版本需求（輪足版）：SDK ≥ 0.2.7、運控 ≥ 0.3.1、本體 ≥ 0.3.3。

### 3.4 附帶影響

「把輪子鎖死當圓腳」這個組態本身也需要對輪馬達下指令，
因此在取得可用的寫入路徑之前，**該組態只存在於模擬中**。

---

## 4. observation 為什麼是 69 維

從 task4 Go2 版的 76 維移除兩樣，得 **69**。欄位順序的單一事實來源是
`task6/inference/obs_d1.py` 的 `OBS_LAYOUT`（Colab notebook Cell 6 的 `_obs` 必須逐項同序）。

| obs 欄位 | 維度 | 實機來源 |
|---|---|---|
| gravity | 3 | `getQuaternion()` → 轉機身系 |
| gyro | 3 | IMU（`getBodyGyro()`）|
| joint_pos | 12 | 關節角減 `HOME12` |
| joint_vel | 12 | 關節角速度 |
| cmd | 3 | 自產 |
| last_action | 12 | 自存 |
| cpg | 24 | 自算（`rx, rx_d, ry, ry_d, sin θ, cos θ`）|
| **合計** | **69** | |

### 4.1 移除機身線速度（−3）

`getBodyVelocity()` **只存在於 HighLevel**，LowLevel 沒有這個量；
官方 FAQ 又明示 High/Low **不可並用**。因此 obs 不含機身線速度。
reward 仍使用模擬真值速度——reward 只在模擬計算，不受實機限制。

### 4.2 移除腳觸地布林（−4）：實測證明這台沒有可用訊號

關卡 3 以 MuJoCo 接觸為真值、摩擦 0.8、1600 樣本，測四個候選訊號，**全部失敗**：

| 候選訊號 | 站立相 | 擺動相 | 結論 |
|---|---|---|---|
| 膝關節力矩 | p05 = 1.55 N·m | p95 = 8.76 N·m | **擺動相反而更大** |
| 膝位置誤差 | p05 = 0.017 rad | p95 = 0.144 rad | **擺動相反而更大** |
| 膝關節角速度 | p95 = 4.40 rad/s | p05 = 0.26 rad/s | 重疊 |
| Jacobian 反推足端 Fz | 中位 −55 N | 中位 +15 N | 尾部重疊，僅約 85% 準確 |

**根因是輪足構型**：足端 0.901 kg（點足版僅 0.063 kg），擺動腿要在 0.25 秒內把它甩起 8 cm，
所需力矩與造成的追蹤延遲**蓋過站立相的承重訊號**，訊號方向直接反過來。
四個候選都不可用，故整段移除。

測試 `test_no_foot_contact_block_and_no_actuator_force_leak`、
`test_layout_sums_to_69_and_has_no_base_linear_velocity` 釘住這兩個決策。

輪子在模擬中被熔接、無自由度，故不進 obs（`test_wheels_have_no_joint`）。

---

## 5. ⚠️ 腿序：官方文件自相矛盾，且實機與模擬不同

| 來源 | 腿順序 |
|---|---|
| `include/zsl-1/lowlevel.h:30`（點足版 `motorState` 註解）| FL, FR, RL, RR |
| 新 SDK `protocol.md` / README | Left-Front, Right-Front, Left-Rear, Right-Rear（FL 開頭）|
| `docs/architecture.md`「关节控制命令说明」| FR, FL, RR, RL |
| `include/lowlevel/lowlevel.h` SHM enum | FR, FL, RR, RL |
| `ZSL-1W.urdf` joint 名稱前綴 | FBL, FAR, RAR, RBL（child link 才是 FL/FR/RR/RL）|

看起來是**高層 API 用 FL 開頭、SHM 底層用 FR 開頭**，兩套慣例並存。

**實機實測（2026-08-10，SN01PW8626B0012）**：用 `shm_probe` 手扳關節確認
`leg0 = 右前 FR`，即這台走 **SHM enum 那套（FR 開頭）**，不是高層 API 那套。
（leg1~leg3 依 enum 推論為 FL / RR / RL，尚未逐一實測。）

模擬端採 `LEGS = ("FL", "FR", "RL", "RR")`（`d1_model.py:19`，寫成 **tuple** 防止被就地改寫，
否則 `PHASE_OFFSET` 的 trot 對角關係會靜默失效），內部自洽、不影響訓練。
URDF 的 joint 前綴與其 child link 名稱不一致，**以 child link 名稱為準**。

→ **sim2real 前必須逐腿實測確認映射，不得憑文件推測。**
除了腿序，還要處理 §3.2 提到的每關節零點/正負校正。

---

## 6. 部署注意事項（來源：官方 FAQ / deploy.md）

- 控制下發建議 20~50 Hz；本專案 `CTRL_DT = 0.02` = 50 Hz。
- **3 秒未收到 SDK 資料，機器會自動切阻尼模式趴下** → 控制迴圈不可中斷。
- IMU 為原始資料、精度一般 → 訓練已加入雜訊與偏差 DR
  （重力 σ=0.02、角速度 σ=0.10 rad/s、每 episode 取樣一次的角速度偏差 ±0.05 rad/s）。
- 四元數轉尤拉角順序為 ZYX。
- 手柄遙控與 SDK 不可同時使用，SDK 優先權較高；新 SDK 改為 role-based 仲裁。
- 運控程式升級後需重新設定 `SDK_CONFIG` 與 `SDK_CLIENT_IP`。
- 輪足機型禁用：跳躍 / 前跳 / 打招呼 / 後空翻 / 雙腿站立。

---

## 7. 常數雙寫警告

`task6/inference/d1_model.py` 與 `task6/notebooks/cpg_rl_d1w_colab.ipynb`（Cell 4）
各有一份常數。Colab 不 import 本地模組，屬**刻意重複**，使用者已確認保留。

**改一邊就必須改另一邊**，否則訓練與推論會靜默對不起來
（維度對得上、行為對不上，不會有任何錯誤訊息）。涉及：

`MU_MIN` `MU_MAX` `OMEGA_MIN` `OMEGA_MAX` `A_CONV` `D_STEP` `D_STEP_Y` `G_C` `G_P`
`NOMINAL_HEIGHT` `W_COUP` `N_CPG_SUB` `CTRL_DT` `SIM_DT` `KP` `KD` `TAU_MAX`
`OBS_DIM` `ACT_DIM` `HOME3` `LEGS` `PHASE_OFFSET`

（notebook 的 `KP_NOM` / `KD_NOM` / `HOME3_np` 對應 `d1_model` 的 `KP` / `KD` / `HOME3`。）

最容易出錯的三個：**`HOME3`、`D_STEP_Y`、`OBS_DIM`**。
另外 `_obs` 的**欄位順序**也是重複，必須與 `obs_d1.OBS_LAYOUT` 逐項一致。

---

## 8. 已知限制

| 項目 | 內容 |
|---|---|
| 摩擦 > 1.0 步態崩潰 | 開迴路實測 1.5 → −0.85 m、3.0 → +0.19 m。訓練 DR 範圍是 [0.3, 1.0]，範圍內全部正常（0.3/0.5/0.8/1.0 → 5.28/5.20/4.57/4.12 m）。根因是站立相足端速度為正弦（0 → 0.905 m/s）而機身近似等速，相位內必然有滑動 |
| armature / damping 為假設值 | URDF 未提供，取 0.01 / 0.1 |
| kp/kd 取自點足版 | 見 §2.1 |
| 腿序與關節零點未定案 | 見 §5 |
| 無實機部署路徑（已驗證的） | 見 §3 |
