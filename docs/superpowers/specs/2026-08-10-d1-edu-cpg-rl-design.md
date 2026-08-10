# 智元 D1 EDU（ZSL-1）CPG-RL 平地走路 — 設計規格

> 建立：2026-08-10 · 對應任務：task6
> 一句話：把 task4 在 Go2 上驗證過的 CPG-RL 論文標準版，移植到智元 D1 EDU（點足版 ZSL-1），
> 訓出平地走路權重，並在 obs 設計上預先對齊實機 SDK 拿得到的感測量，讓之後 sim2real 不用回頭重訓。

---

## 0. 需求來源（使用者原話）

- 「我 train 一個新的狗的走路模型，新的狗是『智元D1 EDU』，請你先找到他的規格檔案，然後做跟之前一樣的事情，去 train 一個 cpg+rl 的走路模型 for 這隻狗。」
- 「有問題請提出，不要臆測。」
- 「如果他原本就是位置伺服，沒必要特別調整 P,D 值。」
- 「請你確定之後 sim2real 的時候，你模型中需要的 input，這隻狗實機都能提供，我能接受少一點 sensor 來換取 sim2real 的順利（若沒問題的話就不需要犧牲）。」

已確認的決策：
| 決策點 | 結果 |
|---|---|
| 機型 | ZSL-1 **點足版**（非輪足 ZSL-1w） |
| 訓練範圍 | **平地論文標準版**（地形版留下一輪） |
| 實機 | **有實機，之後要 sim2real** → MJCF 需對齊 SDK 慣例 |
| 建模方式 | **menagerie 風格手寫 MJCF** |
| 力矩上限 | **28 N·m** |

---

## 1. 資料來源與授權

| 用途 | 來源 | 版本/授權 |
|---|---|---|
| **URDF + STL 網格** | [zsibot/genisom_model](https://github.com/zsibot/genisom_model) `zsl-1/` | BSD-3-Clause，可納入本專案 |
| **SDK 事實**（介面、腿序、kp/kd、頻率限制）| [AgibotTech/agibot_D1_Edu-Ultra](https://github.com/AgibotTech/agibot_D1_Edu-Ultra) | commit `db8accd` |
| **產品規格** | [agibot.com/products/D1_Pro](https://www.agibot.com/products/D1_Pro) | 官網 |

**官方 SDK repo 不含任何 URDF / mesh / 模擬模型**（已實地 clone 驗證：只有 `demo/`、`docs/`、`include/`、`lib/`、`site/`）。

**來源一致性佐證**（URDF 非官方帳號，故逐項對帳）：

| 項目 | zsibot URDF 實測 | 官網 D1 Edu 規格 | 判定 |
|---|---|---|---|
| 總質量 | 15.19 kg | 15.5 kg（含電池）| ✅ |
| abad 行程 | ±0.4887 rad = ±28.0° | ±28° | ✅ |
| hip 行程 | −1.1519~2.967 rad = −66.0°~170.0° | −170°~66°（符號約定相反）| ✅ |
| knee 行程 | −2.723~−0.602 rad = −156.0°~−34.5° | 35°~156° | ✅ |
| 機身長 | BASE STL 0.5555 m | 站立 0.635 m（含腿）| ✅ 合理 |
| 機型代號 | `ZSL-1` | SDK 稱點足版為 `zsl-1` | ✅ |

六項全對，判定為同一台機器。

---

## 2. 交付結構

鏡像 task4，**不修改 task4 任何既有檔案**：

```
task6/
├── model/d1_edu/
│   ├── d1_edu.xml          機器人本體 MJCF
│   ├── scene.xml           本機 CPU 渲染/推論場景
│   ├── scene_mjx.xml       Colab MJX 訓練場景
│   ├── meshes/*.STL        17 個網格（12 MB）
│   └── LICENSE             zsibot BSD-3
├── notebooks/cpg_rl_d1_colab.ipynb
├── inference/
│   ├── d1_model.py         建模型 + 常數（單一事實來源）
│   ├── cpg_openloop_d1.py  關卡 3：開迴路 CPG 驗證
│   └── local_infer_d1.py   關卡 6：本機推論 + 影片
├── weights/cpg_rl_d1_params.pkl
├── outputs/
├── docs/D1_EDU_規格與模型對照.md
└── README.md
```

---

## 3. MJCF 建模規格

### 3.1 基本原則

質量、慣量張量、關節原點、關節軸、關節限位 **一律照 URDF 原值**，不做任何「參考 Go2」的代換。

### 3.2 逐項規格

| 項目 | 規格 | 依據 |
|---|---|---|
| 質量 | BASE 6.2684、ABAD 0.4541、HIP 1.5016/1.5080、KNEE 0.2073、FOOT 0.0632 kg；總 15.19 kg | URDF |
| 慣量 | 照 URDF `<inertia>` 六個分量 | URDF |
| 幾何 | 髖 x=±0.17449、y=±0.062；大腿 0.200 m；小腿 0.21366 m | URDF |
| 關節軸 | abad `+x`、hip `+y`、**knee `−y`**（與 Go2 的 `+y` 相反）| URDF |
| 關節限位 | abad ±0.4887、hip [−1.1519, 2.967]、knee [−2.723, −0.602] rad | URDF |
| 視覺 | 原 STL | — |
| 碰撞 | 機身 `box` 0.556×0.223×0.137；大腿 `capsule` r≈0.035；小腿 `capsule` r≈0.021；**腳 `sphere` r=0.028** | STL 邊界盒實測 |
| 根關節 | **必須加 `freejoint`** | 見 3.3 |
| home keyframe | 關節 `[0, −0.94, −1.80] × 4`；機身 z ≈ 0.29 m（關卡 1 用實際碰撞幾何校正）| 見 3.4 |
| 致動器 | 12 個 `position`，**kp=80、kd=1** | 見 3.5 |
| 力矩上限 | **28 N·m 全關節** | URDF `effort`；官網 48 N·m 為峰值，取保守值 |
| 速度上限 | 28 rad/s | URDF `velocity` |
| 感測 | 機身 `imu` site + acc / gyro / framepos / framequat 四個 sensor | 沿用 task3 `go2_model.py` |
| 腿序 | 常數 `LEGS = ["FL","FR","RL","RR"]` | 見 3.6 |

### 3.3 freejoint 是必要的（實測到的坑）

直接 `mujoco.MjModel.from_xml_path(ZSL-1.urdf)` 可編譯，但結果是
`nq=12, nbody=13, 總質量 8.918 kg` —— **BASE_LINK 被熔進 worldbody，6.268 kg 憑空消失**
（15.186 − 6.268 = 8.918，數字完全對得上）。加 `freejoint` 後應為 `nq=19, nv=18, 總質量 15.19 kg`。
這是關卡 1 的檢查項。

### 3.4 home 姿態的推導

D1 的 knee 軸是 `−y`，與 Go2 的 `+y` 相反，因此 Go2 的 `[0, +0.9, −1.8]` 不能照抄。
令腳端相對髖的位置為 `Ry(q_hip) · [(0,0,−L1) + Ry(−q_knee) · (0,0,−L2)]`，L1=0.200、L2=0.21366，
解 `foot_x = 0`（腳正下方）得：

| q_hip | q_knee | 髖到腳 |
|---|---|---|
| −0.942 | −1.800 | 0.2574 m |
| −0.781 | −1.500 | 0.3028 m |

取 **`[0, −0.94, −1.80]`**（髖到腳 0.2574 m，最接近 Go2 home 的 0.2648 m，落在限位內）。
**注意 hip 角正負號與 Go2 相反**。

CPG→關節角的 IK 沿用 task4 的「home 姿態數值 Jacobian 求逆」，對關節正負號自動免疫，邏輯不需改。

### 3.5 PD 增益：用原廠值，不自行調校，且刪除 `apply_pd()`

URDF 完全沒有 actuator（編譯後 `nu=0`），所以 D1 **沒有「原本的位置伺服」需要保留**；
MJCF 的致動器由本設計從零撰寫。task4 的 `apply_pd()` 是為了覆寫 menagerie Go2 XML 自帶的增益，
D1 沒有這個包袱 → **`apply_pd()` 整個移除**，增益直接寫進 XML。

增益值取自官方 demo（`demo/zsl-1/python/examples/lowlevel_demo.py:68-73`）：

```python
cmd.kp_abad[i] = 80 ; cmd.kp_hip[i] = 80 ; cmd.kp_knee[i] = 80
cmd.kd_abad[i] = 1  ; cmd.kd_hip[i] = 1  ; cmd.kd_knee[i] = 1
```

→ **kp = 80、kd = 1**。這正是 sim2real 時會塞進 `motorCmd` 的值，模擬與實機一致。
DR 範圍 kp ∈ [60, 100]、kd ∈ [0.5, 2]。

若關卡 2/3 顯示 kp=80/kd=1 在模擬中不穩，**視為發現而非調參藉口**：先回報數據，再與使用者決定。

### 3.6 腿序：官方文件自相矛盾，不猜

| 來源 | 腿順序 |
|---|---|
| `include/zsl-1/lowlevel.h:30`（`motorState` 註解）| FL, FR, RL, RR |
| `docs/architecture.md`「关节控制命令说明」| FR, FL, RR, RL |
| `include/lowlevel/lowlevel.h` enum（底層 SHM）| FR, FL, RR, RL |
| zsibot URDF link 排列 | FL, FR, RR, RL |

本輪採 **`LEGS = ["FL","FR","RL","RR"]`**（與 task4 pipeline 一致），定義為單一具名常數。
模擬內部自洽即可，不影響訓練。**sim2real 前必須用實機做「動一條腿看哪條動」驗證後定案**，
結論寫回 `task6/docs/D1_EDU_規格與模型對照.md`。此項列為已知未決事項，不得臆測。

`motorCmd` 是 SoA 佈局（`q_des_abad[4]` / `q_des_hip[4]` / `q_des_knee[4]`），
不是 per-leg AoS，權重輸出的 12 維要按此重排。

---

## 4. 訓練配方

### 4.1 沿用 task4 論文標準版（`cpg_rl_paper_colab.ipynb`）

動作 12 維（每腿 μx, μy, ω）、`MU∈[1,2]`、`OMEGA∈[0,4.5]`、`A_CONV=50`、`D_STEP=0.12`、
`G_C=0.08`、`G_P=0.01`、`W_COUP=8`、`N_CPG_SUB=4`、`CTRL_DT=0.02`、`SIM_DT=0.004`、
Kuramoto trot 耦合、Brax PPO（policy 256/256/128、value 256³、120M steps、2048 envs、
`normalize_observations=True`）。**reward 逐項複製 task4 論文版 env，不重新設計。**

### 4.2 必須修改的項目

| # | 項目 | Go2（task4） | D1（task6） | 理由 |
|---|---|---|---|---|
| 1 | `SCENE` | `unitree_go2/scene_mjx.xml` | `task6/model/d1_edu/scene_mjx.xml` | 換機型 |
| 2 | `HOME12` | `[0, 0.9, −1.8]×4` | `[0, −0.94, −1.80]×4` | §3.4 |
| 3 | PD | `apply_pd()` 覆寫 90/3 | XML 內 kp=80/kd=1，**刪除 `apply_pd()`** | §3.5 |
| 4 | 力矩上限 | 23.7 / knee 45.43 | **28 全關節** | URDF |
| 5 | 負重 DR | 0~8 kg | **0~5 kg** | 官網額定 payload 5 kg |
| 6 | **obs 維度** | 76 | **73** | §5 |
| 7 | 觸地判定 | 腳掌世界高度 < 3 cm | **膝關節力矩 > 門檻** | §5 |
| 8 | IMU DR | 無 | **重力向量與角速度加雜訊+偏差** | 官方 FAQ：IMU 為原始資料、精度一般 |
| 9 | `D_STEP`/`G_C` | 0.12 / 0.08 | 先維持原值 | 腿長比 0.257/0.265 ≈ 0.97，關卡 3 實測校正 |

---

## 5. 觀測設計：以實機可得性為約束

### 5.1 實機能拿到什麼（硬約束）

官方 FAQ：**「Highlevel 与 Lowlevel 不可以同时使用」**。
自訂 policy 必須走 `LowLevel::sendMotorCmd`，因此只能用 LowLevel 的 getter。
`docs/api_zsl-1.md` §2.4–2.9 列出 LowLevel 全部能力：

`getQuaternion` / `getRPY` / `getBodyAcc` / `getBodyGyro` / `getMotorState`（q, qd, tau 各 12）/ `haveMotorData`

**沒有線速度、沒有位置、沒有足端接觸感測。**
（`getBodyVelocity` / `getWorldVelocity` / `getPosition` 只存在於 `HighLevel`，不可並用。）

### 5.2 逐項對帳與處置

| obs 區塊 | 維度 | 實機可得 | 處置 |
|---|---|---|---|
| 重力向量（機身座標）| 3 | ✅ `getQuaternion` | 保留，加雜訊 DR |
| **機身線速度** | 3 | ❌ | **移除** |
| 機身角速度 | 3 | ✅ `getBodyGyro` | 保留，加雜訊 DR |
| 關節角 | 12 | ✅ `motorState.q_*` | 保留 |
| 關節速度 | 12 | ✅ `motorState.qd_*` | 保留 |
| 指令 | 3 | ✅ 自產 | 保留 |
| 上一動作 | 12 | ✅ 自存 | 保留 |
| **腳觸地布林** | 4 | ❌（無接觸感測）| **改判定**：膝關節力矩門檻 |
| CPG 狀態 | 24 | ✅ 自算 | 保留 |

**新 obs = 3 + 3 + 12 + 12 + 3 + 12 + 4 + 24 = 73 維。**

- **移除機身線速度**：無替代方案（IMU 積分會漂），為必要犧牲。
  **reward 照舊使用模擬真值速度** —— reward 只在模擬計算，不受實機感測限制。
- **觸地改用膝關節力矩門檻**：實機有 `motorState.tau_knee_fb`，模擬有 actuator force，
  **兩邊計算同一個物理量**，因此不必犧牲這 4 維。

  明確定義（避免歧義）：第 `k` 腿的觸地布林 = `|τ_knee[k]| > TAU_CONTACT`，
  其中模擬側 `τ_knee[k]` 取 `data.actuator_force[3k+2]`（該腿 knee 致動器的輸出力矩，
  單位 N·m），實機側取 `motorState.tau_knee_fb[k]`。取絕對值是因為 D1 的 knee 軸為 `−y`，
  站立相的力矩符號與 Go2 相反，用絕對值可免除符號約定爭議。
  `TAU_CONTACT` 在關卡 3 由開迴路 rollout 的力矩分佈訂定（取站立相與擺動相的分離點），
  寫入 `d1_model.py` 作為單一常數，訓練與推論共用。

### 5.3 其他部署約束（寫入文件，本輪不實作）

- SDK 建議下發頻率 20~50 Hz；本設計 `CTRL_DT=0.02` = **50 Hz**，合規。
- **3 秒未收到 SDK 資料，機器自動切阻尼模式趴下** → 部署端需保證迴圈不斷。
- 四元數轉尤拉角順序為 **ZYX**。
- 手柄遙控與 SDK 不可同時；SDK 優先權較高。

---

## 6. 驗證關卡

每一關通過才進下一關；未過先修，不硬 train。

| # | 關卡 | 通過標準 | 執行者 |
|---|---|---|---|
| 1 | MJCF 編譯與對帳 | `nq=19, nv=18`；總質量 15.19 kg ±1%；12 關節限位與 URDF 逐一相符 | 我（本機）|
| 2 | 站姿穩定 | home 姿態靜置 2 s，機身 z 漂移 < 1 cm、無明顯抖動 | 我（本機）|
| 3 | **開迴路 CPG**（無 RL）| 前進 > 1 m；抬腳 3~6 cm；不跌倒 → 證明 IK / PD / CPG 常數正確 | 我（本機）|
| 4 | Smoke test | `obs.shape == (73,)`；reward 有限、非 NaN | 使用者（Colab）|
| 5 | PPO 訓練 | 產出收斂曲線；存 `cpg_rl_d1_params.pkl` | 使用者（Colab）|
| 6 | 本機推論 | 影片 + 前進距離 / 抬腳量 / 末端高度；含抗推測試 | 我（本機）|

關卡 3 是本設計最重要的早期檢查：它在**不花任何 GPU 時數**的前提下，
一次驗證 MJCF、home 姿態、IK Jacobian、PD 增益、CPG 常數五件事是否正確。

---

## 7. 分工

- **我**：全部程式碼與文件；本機執行關卡 1、2、3、6。
- **使用者**：上 Colab 執行關卡 4、5，下載權重回本機。

本機無 GPU（`jax.devices()` 只有 `CpuDevice`，已驗證），訓練必須在 Colab。

---

## 8. 本輪不做（YAGNI）

- 地形訓練（斜坡/凹凸）—— 下一輪，比照 task4 的 terrain2/terrain3b 路線。
- 輪足 ZSL-1w。
- 實機部署與 sim2real 驗證 —— 本輪只做到「權重 + SDK 對照表就緒」。
- 高階 API（`standUp` / `jump` 等）整合。

---

## 9. 已知未決事項

| # | 事項 | 影響 | 何時解 |
|---|---|---|---|
| 1 | **SDK 腿序官方文件自相矛盾**（§3.6）| 擋 sim2real，不擋本輪訓練 | 實機第一次連線時驗證 |
| 2 | 觸地力矩門檻值 | obs 第 8 區塊品質 | 關卡 3 由實測力矩分佈訂定 |
| 3 | kp=80/kd=1 在模擬中的穩定性 | 若不穩需回報並重新決策 | 關卡 2/3 |
| 4 | 官網 48 N·m 與 URDF 28 N·m 的差異 | 已決定取 28（保守）；若訓練顯示力矩飽和嚴重需回報 | 關卡 5 |
