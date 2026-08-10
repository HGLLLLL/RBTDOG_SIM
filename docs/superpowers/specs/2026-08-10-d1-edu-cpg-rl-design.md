# 智元 D1 EDU **輪足版（ZSL-1w）** CPG-RL 平地走路 — 設計規格

> 建立：2026-08-10 · 對應任務：task6
> 一句話：把 task4 在 Go2 上驗證過的 CPG-RL 論文標準版，移植到智元 D1 EDU **輪足版**，
> **四顆輪子鎖死當成 71 mm 圓腳**來走路，並在 obs 設計上預先對齊實機拿得到的感測量。

---

## 修訂紀錄

### 修訂 2（2026-08-10）— 機型由點足版改為輪足版

使用者於實作進行中告知：**實機是輪足版（ZSL-1w），不是點足版（ZSL-1）**。
本規格已整份改寫。修訂 1（點足版）的內容見 git 歷史 commit `d359ab0`。

改變的關鍵事實（全部查證自 `zsl-1w/urdf/ZSL-1W.urdf` 與官方 SDK repo）：

| 項目 | ZSL-1 點足（修訂 1）| **ZSL-1w 輪足（本版）** |
|---|---|---|
| URDF 自由度 | 12 | **16**（每腿多一顆輪子）|
| 總質量 | 15.19 kg | **20.56 kg** |
| 足端 | 0.063 kg 圓球 | **0.901 kg 輪子**（半徑 71 mm、寬 48 mm）|
| 小腿質量 | 0.207 kg | 0.765 kg |
| 膝關節軸 | `0 -1 0` | **`0 1 0`** |
| home 關節角 | `[0, −0.94, −1.80]` | **`[0, +1.05, −2.00]`** |
| 機身站立高 | 0.2914 m | **0.2948 m** |
| SDK 關節控制 | LowLevel 可下發 | **無 LowLevel，只有 HighLevel** |

使用者的三項決策：
1. **鎖輪當圓腳走路** —— 輪子不驅動，動作空間維持 12 維，CPG-RL 架構沿用。
2. **保留 sim2real 約束** —— obs 仍只用實機讀得到的量。
3. **完全換掉點足版** —— task6 只做輪足版。

---

## 0. 需求來源（使用者原話）

- 「我 train 一個新的狗的走路模型，新的狗是『智元D1 EDU』，請你先找到他的規格檔案，然後做跟之前一樣的事情，去 train 一個 cpg+rl 的走路模型 for 這隻狗。」
- 「有問題請提出，不要臆測。」
- 「如果他原本就是位置伺服，沒必要特別調整 P,D 值。」
- 「請你確定之後 sim2real 的時候，你模型中需要的 input，這隻狗實機都能提供，我能接受少一點 sensor 來換取 sim2real 的順利。」
- 「現在有一個重大變更，我的機器狗是『輪足』版本的。」

已確認的決策：

| 決策點 | 結果 |
|---|---|
| 機型 | **ZSL-1w 輪足版** |
| 移動型態 | **鎖輪當圓腳走路**（輪不驅動）|
| 訓練範圍 | 平地論文標準版（地形版留下一輪）|
| 實機 | 有實機，但 **SDK 無 LowLevel，目前無法部署**；仍保留 sim2real 約束 |
| 建模方式 | menagerie 風格手寫 MJCF |
| 力矩上限 | 28 N·m |

---

## 1. 資料來源與授權

| 用途 | 來源 | 版本/授權 |
|---|---|---|
| **URDF + STL 網格** | [zsibot/genisom_model](https://github.com/zsibot/genisom_model) `zsl-1w/` | BSD-3-Clause，commit `e6aa98e` |
| **SDK 事實（文件）** | [AgibotTech/agibot_D1_Edu-Ultra](https://github.com/AgibotTech/agibot_D1_Edu-Ultra) | commit `db8accd`（v0.2.7，**已停維護**，但文件最全）|
| **SDK 事實（現行）** | [zsibot/genisom_L1_sdk](https://github.com/zsibot/genisom_L1_sdk) | v1.0.0 / MIT / 維護中 |
| **產品規格** | [agibot.com/products/D1_Pro](https://www.agibot.com/products/D1_Pro) | 官網 |

**官方 SDK repo 不含任何 URDF / mesh / 模擬模型**（已實地 clone 驗證）。

**質量差異的解釋（佐證兩份資料一致）**：官網標示 15.5 kg，URDF 輪足版為 20.56 kg。
差額 5.06 kg ≈ 4 ×[(0.901 − 0.063) + (0.765 − 0.207)] = 5.58 kg，正好是輪子與加粗小腿相對點足版的增重。
判定：**官網 15.5 kg 是點足版數字，輪足版約 20.5 kg**，兩份資料不衝突。

---

## 2. 交付結構

```
task6/
├── model/d1_edu_w/
│   ├── d1_edu_w.xml        機器人本體 MJCF（12 驅動關節 + 4 顆熔接鎖死的輪）
│   ├── scene.xml           本機 CPU 渲染/推論場景
│   ├── scene_mjx.xml       Colab MJX 訓練場景
│   ├── meshes/*.STL        17 個網格（zsl-1w）
│   ├── LICENSE             zsibot BSD-3
│   └── SOURCE.md           資產來源與 commit 記錄
├── notebooks/cpg_rl_d1w_colab.ipynb
├── inference/
│   ├── d1_model.py         常數單一事實來源 + make_model()
│   ├── cpg_d1.py           CPG + 動作解碼 + IK + 關節目標角
│   ├── obs_d1.py           73 維 observation + 力矩觸地判定
│   ├── cpg_openloop_d1.py  關卡 3：開迴路 CPG 驗證
│   └── local_infer_d1.py   關卡 6：本機推論 + 影片
├── tests/
├── weights/  outputs/
├── docs/D1_EDU_W_規格與模型對照.md
└── README.md
```

---

## 3. MJCF 建模規格

### 3.1 基本原則

質量、慣量張量、關節原點、關節軸、關節限位 **一律照 `ZSL-1W.urdf` 原值**。

### 3.2 逐項規格

| 項目 | 規格 | 依據 |
|---|---|---|
| 質量 | BASE 6.7155497、ABAD 0.45409769、HIP 1.34038、KNEE 0.7651085、**WHEEL 0.90130429** kg；總 20.56 kg | URDF |
| 慣量 | 照 URDF `<inertia>` 六個分量 | URDF |
| 幾何 | 髖 x=±0.17449/±0.1745、y=±0.097412/∓0.097399/∓0.09735；大腿 0.200 m；小腿 0.21366 m | URDF |
| 關節軸 | abad `+x`、hip `+y`、**knee `+y`**（與點足版的 `−y` 相反）| URDF |
| 關節限位 | abad ±0.4887、hip [−1.152, 2.967]、knee [−2.723, −0.602] rad | URDF |
| **輪子** | URDF 的 `*_FOOT_JOINT` 為連續旋轉（限位 ±999999）。**本設計不建立此關節，直接把輪熔接到小腿**（見 3.3）| 使用者決策：鎖輪 |
| 輪幾何 | 半徑 **0.0710 m**、寬 0.0480 m。**碰撞圓柱中心 `(0, ±0.0475, 0)`（STL 幾何中心）；慣量原點 `(0, ±0.04488028, 0)`（真實質心，兩者差 2.6 mm 是輪轂造成，非錯誤）** | STL 實測 + URDF |
| 視覺 | 原 STL | — |
| 碰撞 | 機身 `box` 0.556×0.223×0.137；大腿 `capsule` r≈0.030；小腿 `capsule` r≈0.022；**輪 `cylinder` r=0.071 半高 0.024，軸沿 y** | STL 邊界盒實測 |
| 根關節 | **必須加 `freejoint`** | 見 3.4 |
| home keyframe | 關節 `[0, +1.05, −2.00] × 4`；機身 z = 0.2948 m（**純運動學值**，實際站定沉降至 `NOMINAL_HEIGHT` = 0.2695 m）| 見 3.5 |
| 致動器 | **12 個** `position`，kp=80、kd=1 | 見 3.6 |
| 力矩上限 | 28 N·m 全關節 | URDF `effort` |
| 感測 | 機身 `imu` site + acc / gyro / framequat / framepos | 沿用 task3 慣例 |
| 腿序 | `LEGS = ["FL","FR","RL","RR"]` | 見 3.7 |

### 3.3 輪子鎖死的建模方式

使用者決定「鎖輪當圓腳走路」。本設計**不建立輪子關節**，把 `FOOT_LINK`
當成一個沒有關節的子 body 熔接在小腿末端。理由：

- 動作空間維持 12 維，CPG-RL 架構、obs 維度、訓練配方全部沿用，改動面最小。
- 輪子的 0.901 kg 質量與慣量**完整保留**（不是簡化掉），擺動腿的動力學是真實的。
- 碰撞用 `cylinder` 而非 `sphere`：鎖死的輪子接觸地面是一條沿 y 的線，不是點。

**必須在文件中記錄的限制**：真實機器上「鎖住輪子」需要對輪子馬達下位置/力矩指令，
而輪足版 SDK **沒有 LowLevel**（見 §5.1），因此**這個組態目前只存在於模擬中**。

### 3.4 freejoint 是必要的

URDF 直接編譯會把 `BASE_LINK` 熔進 worldbody，導致 6.716 kg 消失。
加 `freejoint` 後應為 `nq=19, nv=18, nu=12, 總質量 20.56 kg`。這是關卡 1 的檢查項。

### 3.5 home 姿態的推導

輪足版膝軸為 `+y`（與點足版相反），因此點足版的 `[0, −0.94, −1.80]` 不適用。
令輪心相對髖的位置為 `Ry(q_hip) · [(0,0,−L1) + Ry(q_knee) · (0,0,−L2)]`，L1=0.200、L2=0.21366，
解 `x = 0`（輪心在髖正下方），機身高 = |z| + 輪半徑 0.0710：

| q_hip | q_knee | 輪心 z | 機身高 |
|---|---|---|---|
| +1.051 | −2.000 | −0.2238 | **0.2948** |
| +0.942 | −1.800 | −0.2574 | 0.3284 |

取 **`[0, +1.05, −2.00]`**（機身高 0.2948 m，與點足版的 0.2914 m 相當，落在限位內）。
**注意 hip 角為正**，與點足版相反、與 Go2 同向。

CPG→關節角的 IK 沿用「home 姿態數值 Jacobian 求逆」，對關節正負號自動免疫。
IK 的腳端參考點取**輪子碰撞 geom 的中心**，因此輪心相對髖橫向偏移 4.5 cm 會被自動吸收。

**幾何高度 ≠ 站立高度**：0.2948 m 是「輪心在髖正下方」的純運動學解。實際加上地面模擬後，
kp=80 的位置伺服在 20.56 kg 下有靜態撓度（膝角誤差約 0.07 rad），機身會沉降到約 **0.2695 m** 才停住
（實測 t=1s 0.27012 / 3s 0.26970 / 5s 0.26946，末速 −0.00007 m/s）。
這個值定義為 `NOMINAL_HEIGHT`，**訓練的高度獎勵必須用它**，用 keyframe 的 0.2948 會與物理對打（差 8.6%）。

### 3.6 PD 增益：用原廠值，不自行調校

URDF 沒有 actuator，MJCF 的致動器由本設計從零撰寫，**不需要也不得使用 `apply_pd()`**。
增益取自官方 demo `demo/zsl-1/python/examples/lowlevel_demo.py:68-73`：**kp = 80、kd = 1**。

⚠️ 該 demo 是**點足版（zsl-1）**的，輪足版沒有 LowLevel demo 可參考。這是目前能取得的
最接近原廠的數值，列為假設，由關卡 2、3 驗證；若不穩需回報，不得自行調參。
DR 範圍 kp ∈ [60, 100]、kd ∈ [0.5, 2]。

### 3.7 腿序與命名：官方文件自相矛盾，不猜

| 來源 | 腿順序 |
|---|---|
| `include/zsl-1/lowlevel.h:30`（點足版 `motorState` 註解）| FL, FR, RL, RR |
| `docs/architecture.md`「关节控制命令说明」| FR, FL, RR, RL |
| `include/lowlevel/lowlevel.h` enum | FR, FL, RR, RL |
| `ZSL-1W.urdf` 的 joint 名稱前綴 | FBL(→FL), FAR(→FR), RAR(→RR), RBL(→RL) |

本設計採 **`LEGS = ["FL","FR","RL","RR"]`**，模擬內部自洽。
URDF 的 joint 名稱前綴（`FBL_/FAR_/RAR_/RBL_`）與其 child link 名稱（`FL_/FR_/RR_/RL_`）不一致，
**以 child link 名稱為準**。sim2real 前必須用實機驗證後定案。

---

## 4. 訓練配方

### 4.1 沿用 task4 論文標準版

動作 12 維（每腿 μx, μy, ω）、`MU∈[1,2]`、`OMEGA∈[0,4.5]`、`A_CONV=50`、`D_STEP=0.12`、
`D_STEP_Y=0.09`、`G_C=0.08`、`G_P=0.01`、`W_COUP=8`、`N_CPG_SUB=4`、`CTRL_DT=0.02`、`SIM_DT=0.004`、
Kuramoto trot 耦合、Brax PPO（policy 256/256/128、value 256³、120M steps、2048 envs）。
**reward 逐項複製 task4 論文版 env。**

### 4.2 必須修改的項目（相對 task4 的 Go2 版）

| # | 項目 | Go2（task4）| D1 輪足（task6）| 理由 |
|---|---|---|---|---|
| 1 | `SCENE` | `unitree_go2/scene_mjx.xml` | `task6/model/d1_edu_w/scene_mjx.xml` | 換機型 |
| 2 | `HOME12` | `[0, 0.9, −1.8]×4` | `[0, 1.05, −2.00]×4` | §3.5 |
| 3 | PD | `apply_pd()` 覆寫 90/3 | XML 內建 kp=80/kd=1，**刪除 `apply_pd()`** | §3.6 |
| 4 | 力矩上限 | 23.7 / knee 45.43 | 28 全關節 | URDF |
| 5 | 負重 DR | 0~8 kg | **0~5 kg** | 官網額定 payload |
| 6 | obs 維度 | 76 | **73** | §5 |
| 7 | 觸地判定 | 腳掌世界高度 < 3 cm | **膝關節力矩 > 門檻** | §5 |
| 8 | IMU DR | 無 | 重力與角速度加雜訊+偏差 | 官方 FAQ：IMU 為原始資料、精度一般 |
| 9 | `D_STEP`/`G_C` | 0.12 / 0.08 | `D_STEP`=0.12 但**側向另立 `D_STEP_Y`=0.09** | abad 行程僅 ±28°（Go2 ±60°），沿用 0.12 超限 14% |
| 10 | **摩擦 DR 下限** | 0.3 | 0.3（不變）| 鎖死的輪子接觸較滑，低摩擦樣本更重要 |
| 11 | 高度獎勵基準 | `key_qpos[2]` | **`NOMINAL_HEIGHT`=0.2695** | keyframe 的 0.2948 是純運動學值，實際站定會沉降 2.5 cm |

---

## 5. 觀測設計：以實機可得性為約束

### 5.1 ⚠️ 輪足版沒有 LowLevel —— 目前無部署路徑

官方文件 `docs/deploy.md:168` 原文：

> **「ZSL-1w 不提供 LowLevel 接口，因此仅有 `highlevel_demo`。」**

三路交叉驗證：`include/zsl-1w/` 只有 `highlevel.h`（點足版另有 `lowlevel.h`）；
`docs/api_zsl-1w.md` 全篇只有「1. HighLevel函数介绍」；`demo/zsl-1w/` 只有 highlevel demo。
SDK v0.2.7（2025-11-04）為 repo 內最新版。

**結論（SDK 庫層面）**：輪足版透過官方 SDK **能讀**十六個關節的角度/速度/力矩與 IMU，
但**不能下發關節指令**，只能用 `move(vx, vy, yaw_rate)` / `crawl` / `climb` 等原廠內建動作。

#### 補充（2026-08-10 追加調查，修正上述結論的範圍）

SDK 已搬家：新家 [`zsibot/genisom_L1_sdk`](https://github.com/zsibot/genisom_L1_sdk)（v1.0.0、MIT、維護中），
舊的 `AgibotTech/agibot_D1_Edu-Ultra` 官方已標示停止維護。**GENISOM-AI L1 系列 == D1 EDU 系列**，
機型代號相同（`zsl-1`/`xgb` = 點足、`zsl-1w`/`xgw` = 輪足）。新舊兩邊都要看：舊 repo 的
`deploy.md` / `architecture.md` / `faq.md` / 關節方向圖新 repo 沒有。

輪足無 LowLevel 一事有更硬的證據：`nm -D` 驗證 `zsl-1w` 的庫**導出 0 個 LowLevel 符號**，
新 SDK 的 `libzsibot.a` 亦然；點足 `zsl-1` 才有 `sendMotorCmd`。

**但「無部署路徑」這個說法要收窄**：對三個 `.so` 做 `strings` 搜 `spline` / `shm` / `/dev/shm`
**零命中**，代表共享記憶體介面**根本不在 SDK 庫裡**——`create_spline_shm()` 是標頭檔中的
`static inline`，程式直接對機器上的 daemon 通訊、**完全繞過 SDK**。
因此「輪足庫沒有 LowLevel」**管不到 `/spline_shm` 那條路，兩者互相獨立**。

現況正確的說法是：
- ✅ 已確認：官方 SDK **不提供**輪足版關節控制。
- ❓ 未確認：板載 `/spline_shm` 共享記憶體介面對輪足版是否可用。**未經硬體驗證，不得假設可行。**
- 可用來降低風險：官方模擬器 [`zsibot/matrix`](https://github.com/zsibot/matrix) 有
  「Linux 模擬硬體」模式（UeSim ↔ 共享記憶體 ↔ 外部 `mc_ctrl`），理論上可在**不碰實機**的前提下
  先驗證共享記憶體 ABI。

完整調查與逐步實測指南見 `task6/docs/D1EDU_輪足_lowlevel_調查與實測指南.md`。

這些都不影響本專案的模擬訓練，但決定了訓練成果最終能不能上機。

### 5.2 obs 的保守設計原則

輪足版 `HighLevel` **讀得到** `getBodyVelocity()`（機身線速度）。
但它讀得到不代表能拿來部署 —— 部署必須靠尚未存在的 LowLevel，而依點足版 LowLevel 的先例，
**LowLevel 沒有線速度**（`getBodyVelocity` 只存在於 HighLevel，且官方 FAQ 明示兩者不可並用）。

因此採**保守解讀**：obs **不含機身線速度**，維持與修訂 1 相同的 73 維。
如此一旦輪足版 LowLevel 釋出，policy 可直接使用，不需重訓。

### 5.3 逐項對帳

| obs 區塊 | 維度 | 未來 LowLevel 可得？ | 處置 |
|---|---|---|---|
| 重力向量（機身座標）| 3 | ✅ 四元數 | 保留，加雜訊 DR |
| 機身線速度 | 3 | ❌（先例：點足版 LowLevel 無）| **移除** |
| 機身角速度 | 3 | ✅ gyro | 保留，加雜訊 DR |
| 關節角 | 12 | ✅ | 保留 |
| 關節速度 | 12 | ✅ | 保留 |
| 指令 | 3 | ✅ 自產 | 保留 |
| 上一動作 | 12 | ✅ 自存 | 保留 |
| 腳觸地布林 | 4 | ❌ 無接觸感測 | **改判定**：膝關節力矩門檻 |
| CPG 狀態 | 24 | ✅ 自算 | 保留 |

**obs = 3 + 3 + 12 + 12 + 3 + 12 + 4 + 24 = 73 維。**

- **reward 照舊使用模擬真值速度**（reward 只在模擬計算，不受感測限制）。
- **輪子關節狀態不進 obs**：輪子在模擬中被熔接、無自由度，沒有狀態可讀。
- **觸地改用膝關節力矩門檻**：`|τ_knee[k]| > TAU_CONTACT`，
  模擬側取 `data.actuator_force[3k+2]`，實機側取膝關節力矩回授。
  取絕對值以免除符號約定爭議。`TAU_CONTACT` 於關卡 3 由開迴路力矩分佈訂定。

### 5.4 其他部署約束

- 控制頻率：SDK 建議 20~50 Hz；本設計 50 Hz，合規。
- 3 秒未收到 SDK 資料，機器自動切阻尼模式趴下。
- 四元數轉尤拉角順序 ZYX。
- 手柄遙控與 SDK 不可同時；SDK 優先權較高。

---

## 6. 驗證關卡

| # | 關卡 | 通過標準 | 執行者 |
|---|---|---|---|
| 1 | MJCF 編譯與對帳 | `nq=19, nv=18, nu=12`；總質量 20.56 kg ±1%；12 關節限位與 URDF 逐一相符 | 我（本機）|
| 2 | 站姿穩定 | home 姿態靜置 2 s，機身 z 漂移 < 1 cm、無明顯抖動 | 我（本機）|
| 3 | **開迴路 CPG**（無 RL）| 前進 > 1 m；抬腳 3~6 cm；不跌倒 | 我（本機）|
| 4 | Smoke test | `obs.shape == (73,)`；reward 有限、非 NaN | 使用者（Colab）|
| 5 | PPO 訓練 | 收斂曲線；存 `cpg_rl_d1w_params.pkl` | 使用者（Colab）|
| 6 | 本機推論 | 影片 + 前進距離 / 抬腳量 / 末端高度；含抗推測試 | 我（本機）|

關卡 3 特別重要：鎖死的輪子接觸摩擦特性與點足不同（線接觸、易打滑），
**若開迴路走不動，第一個要懷疑的是輪子摩擦與 `D_STEP`，不是 CPG 邏輯。**

---

## 7. 分工

- **我**：全部程式碼與文件；本機執行關卡 1、2、3、6。
- **使用者**：Colab 執行關卡 4、5，下載權重回本機。

---

## 8. 本輪不做（YAGNI）

- 地形訓練、點足版（ZSL-1）、輪腿混合或純輪式移動、實機部署、高階 API 整合。

---

## 9. 已知未決事項

| # | 事項 | 影響 | 何時解 |
|---|---|---|---|
| 1 | **官方 SDK 不提供輪足版關節控制**（`nm -D` 導出 0 個 LowLevel 符號）| 擋住經由 SDK 的 sim2real | 需原廠釋出 |
| 1b | 板載 `/spline_shm` 共享記憶體介面對輪足版是否可用 —— **獨立於 SDK，未經硬體驗證** | 若可行則 sim2real 有路 | 依 `task6/docs/D1EDU_輪足_lowlevel_調查與實測指南.md` 的 Phase 0 唯讀偵察 |
| 1c | 官方 MATRiX 模擬器有 xgw 的 MJCF（在百度網盤 runtime 包，不在 git repo）| 可用來交叉驗證本專案手寫的 MJCF，特別是 armature/damping 與 kp/kd 假設 | 使用者下載後比對 |
| 2 | kp=80/kd=1 取自**點足版** demo，輪足版無對應資料 | 站姿與步態穩定性 | 關卡 2/3；不穩則回報 |
| 3 | SDK 腿序官方文件自相矛盾 | 擋 sim2real，不擋訓練 | 實機驗證 |
| 4 | 觸地力矩門檻值 | obs 品質 | 關卡 3 實測 |
| 5 | armature / damping（URDF 未提供，假設 0.01 / 0.1）| 動力學真實度 | 關卡 2/3 |
| 6 | 鎖死輪子的接觸摩擦係數 | 走路可行性 | 關卡 3；必要時調 DR 下限 |
