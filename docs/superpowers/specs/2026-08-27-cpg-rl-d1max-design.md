# D1 Max（中狗）CPG 基準步態凍結 ＋ CPG-RL 訓練 —— 設計

- 日期：2026-08-27
- 分支：`feat/d1-edu-cpg-rl`
- 目標機：**D1 Max（中狗）**，16 軸輪足，MJCF `task7/model/zgws/zgws.xml`
- 前置文件：`task7/docs/CPG多擾動重掃結果_2026-08-26.md`、
  `task7/docs/D1Max_CPG步態_模擬結果_2026-08-25.md`、
  `task7/docs/實機偵察結果_第二趟_2026-08-25.md`
- 移植基底：`task6/notebooks/cpg_rl_d1w_colab.ipynb`（D1 EDU，reward 已迭代到 v4）

---

## 0. 這份 spec 要解決什麼

兩件事：

1. **凍結一組穩定不跌倒的 CPG 步態**，當作 RL 的基準與對照組。
2. **以該步態為基準做 CPG-RL 訓練**，產出一本可以直接丟 Colab 跑的 notebook，
   而且觀測層的每一欄都必須是**實機底層真的拿得到的量**，避免 sim2real 時 input 對不上。

### 任務一其實已經完成大半 —— 先講清楚現況

`walk` 這組開迴路步態在多擾動掃描（`task7/outputs/sweep_*.json`）中：

| 條件 | 結果 |
|---|---|
| 20 / 60 / 120 / **180 秒**，每格 12 個擾動 | **跌倒 0/12** |
| 行進速度 `speed_travel` | 0.148 – 0.150 m/s（與時長無關） |
| 關節超限 / 力矩飽和 / IK 縮限 | 全部 0.00% |
| 地面摩擦 | ≥0.5 不跌，0.3 會跌 |

所以「平地不跌倒」**已經達成且複驗過**，任務一的工作是**把它凍結成唯一真實來源**，
不是重新去找一組步態。剩下三個缺口，本 spec 只處理第一個：

1. **走不直**（偏航 −0.5 ~ −0.9 °/s，180 s 累積 −51°、側偏 −10.7 m）→ **交給 RL 學**
2. **只在平地驗過**（斜坡／台階／不平地面沒測）→ 本次不做，明確列為未解決
3. **實機一步都沒走過**（連承重站立 M7 都還沒做）→ 本次不做

---

## 1. ★ 本次最大的技術發現：官方 MJCF 不能直接餵給 MJX

實測 `task7/model/zgws/scene_flat.xml` 的碰撞幾何：

| 碰撞 geom | 型別 | 頂點數 | 面數 |
|---|---|---|---|
| `BASE_LINK` | mesh | **98,569** | 199,999 |
| `FAR_FOOT_LINK`（右前輪） | mesh | 31,730 | 63,612 |
| `FBL_FOOT_LINK` | mesh | 31,729 | 63,612 |
| `RAR_FOOT_LINK` | mesh | 31,728 | 63,612 |
| `RBL_FOOT_LINK` | mesh | 31,731 | 63,612 |
| 8 × hip/knee | box | — | — |
| `floor` | plane | — | — |

`mjx.put_model` 本身會過（實測 2.6 s），但 MuJoCo 自己就發警告：
`Mesh "..." has a coplanar face with more than 20 vertices. This may lead to
performance issues and inaccuracies in collision detection.`

MJX 的 plane–convex 碰撞是**逐頂點**計算的。2048 個平行環境 × 98,569 頂點，
單一中間張量就是數百 MB，Colab GPU 上不是慢到不能用就是 OOM。

**現有 CPG 模擬跑得動不能當作 MJX 可行的證據** —— 那是 CPU MuJoCo、單一環境。

### 解法

已確認 MJX 的碰撞函式表支援 `(PLANE, CYLINDER)`：

- 四輪碰撞網格 → `<geom type="cylinder">`，半徑 **0.0961 m**（碰撞網格實算值），
  半寬由該網格頂點在輪軸方向的 bounding box 實算，不用手填
- `BASE_LINK` 碰撞網格 → 方塊（走路時機身本來就不該碰地）
- 8 個 hip/knee 方塊維持原樣
- 視覺網格全部保留，但 `contype=0 conaffinity=0`

### ⚠️ 這會製造「模擬對模擬」的落差，必須量，不可以假設等價

見 §7 的驗收關卡 G1。

---

## 2. 任務一：基準步態凍結

### 2.1 凍結的參數

來源：`task7/inference/cpg_walk_max.py` 的 `GAITS["walk"]` ＋ `max_model.py` 的增益常數。

| 參數 | 值 | 判準來源 |
|---|---|---|
| 相位 | `PHASE_WALK`（側序走 RL→FL→RR→FR） | — |
| `duty` | **0.80** | ≤0.70 是 12/12 全跌；0.80 彈跳最小（17.2 mm） |
| `omega` | 1.4 | 提高會帶來 0.4–0.9 °/s 偏航 |
| `mu_x` / `mu_y` | 1.80 / **1.50** | `mu_y=1.5` → `fy=0`，橫向偏移恰為 0；1.75 是 12/12 全跌 |
| `d_step` / `d_step_y` | 0.10 / 0.12 | — |
| `x_off` | **−40 mm** | 平均俯仰過零在 −40.8 mm（12 擾動） |
| `g_c` | 0.08 | 實際離地約 100 mm，對上原廠 `leg_height=0.10` |
| `z_sag` | 32.5 mm（只加擺動相） | 位置伺服靜態撓度補償 |
| `KP` | ABAD 60 / HIP 120 / KNEE 120 | 原廠 **RL 模式**設定檔值 |
| `KD` | 1.0 | 同上 |
| 輪 | **damp**，`Kd=0.5`，不給位置增益 | hold 模式實測造成 +39° 偏航失控 |
| `CTRL_DT` / `SIM_DT` | 0.02 / 0.002 | 50 Hz 指令、500 Hz PD＝原廠 `controller_dt` |

⚠️ 引用增益時必須標明是**哪一種模式**：原廠「站立」實測是 kp 250 / kd 5.0，
與上表這組 RL 設定檔值不同。本 spec 全程用 RL 那組。

### 2.2 交付

- `task7/inference/gait_baseline.py`：唯一真實來源，`cpg_walk_max.GAITS["walk"]`
  改為引用它，Colab notebook 也引用它。
- `task7/tests/test_gait_baseline.py`：把上表每一個數字釘死。
  理由：掃描期間曾發生「跑到一半共用 MJCF 被別條線改掉」
  （`533e91a` 改了膝關節 range），有測試才擋得住。
- `task7/docs/基準步態凍結_D1Max_walk_2026-08-27.md`：凍結說明 ＋ 複驗數據。

### 2.3 基準複驗

用 `cpg_sweep_max.py` 對凍結後的參數重跑一次確認：
20 / 60 / 180 s × 12 擾動、地面摩擦 0.4 / 0.7 / 1.0 / 1.4。
**驗收：跌倒數全部為 0**（摩擦 0.4 那格若跌，記錄下來並在文件標明門檻，不改參數）。

---

## 3. 訓練模型 `zgws_mjx.xml`

### 3.1 生成方式

`task7/model/zgws/make_mjx_model.py` **程式化**從官方 `zgws.xml` 生成。
不手改官方檔案 —— 手改會與官方版本分岔，之後官方更新無法比對。

三處改動：

1. **碰撞幾何**（見 §1）
2. **致動器**：16 個 `<motor>` → 12 個 `<position kp kv forcerange>`（腿）
   ＋ 4 個 `<velocity kv="0.5" forcerange="-33 33">`（輪阻尼）
   - 已驗證 MuJoCo 3.10 的 `<position>` 支援 `kv`，得到 `biasprm = [0, −kp, −kv]`、
     `biastype = affine`
   - `<position>` 每個**物理步**（500 Hz）算一次 PD → **精確等於原廠內迴圈頻率**，
     不是每個控制步（50 Hz）算一次
   - domain randomization 可直接改 `actuator_gainprm` / `actuator_biasprm`
3. **`frictionloss` 全部保留**：腿 ABAD 1.85 / HIP 1.5 / KNEE 1.5、輪 0.15（都是實機量到的）

### 3.2 ⚠️ 必須在 env 裡斷言的事

`assert sys.actuator_biastype[0] == mjBIAS_AFFINE`。
task4 地形版踩過：biastype 不是 affine 時 `ctrl` 會被當力矩直接施加，機器人直接塌掉。

---

## 4. 動作空間與 CPG

12 維 → `tanh` → 每腿 `(mux, muy, omega)`，振盪器數學與 task4/task6 逐行同構。

### 4.1 ★ 與 task6 最關鍵的差異：`duty` 必須進 env

task6 的 env `_joint_targets` **沒有 `duty_remap`**（等效 duty=0.5），
而 D1 Max **實測 duty ≤ 0.70 是 12/12 全跌**。
照抄 task6 的 env 會得到一台永遠站不住的狗。

env 內完整沿用 task7 的軌跡鏈：

```
cpg_step → duty_remap(0.80) → foot_targets(x_off=−40mm, z_sag=32.5mm 只加擺動相)
         → 每腿各自的 home_foot + knee_sign → 解析式 IK → 12 個關節目標角
```

- **解析式 IK**，不是 task6 的 home 附近線性化。這台抬腿 100 mm、連桿 0.26+0.28 m，
  線性化誤差會直接汙染「抬腿量」這個要評估的指標。
- **每腿各自的 `home_foot`**：站姿是前後鏡像的 X 型，四腿共用一組 HOME 會做出
  前後腿方向相反的怪東西。

### 4.2 參數範圍

| | 範圍 | 依據 |
|---|---|---|
| `mu_x`, `mu_y` | 1.0 – 2.0 | 沿用 task4/task6 |
| `omega` | 0.0 – 2.0 | 掃描：ω=1.8 → 0.316 m/s 但彈跳 +48%、支撐腳 −0.3；ω=2.0 速度反而掉回 0.281。給到 2.0 有餘裕又不過頭 |

混沌區（trot、duty=0.5、速度在 0.11–0.46 m/s 之間跳）因 `duty` 鎖死 0.80 而**不可達**。

---

## 5. 觀測：68 維

每一欄都必須是**實機凍結 `mc_ctrl` 後、直接讀 `/dev/shm` 拿得到的量**。

| 欄位 | 維度 | 實機來源 |
|---|---|---|
| `gravity` | 3 | `/dev/shm/imu_central` quat（**xyzw**）→ 轉機身系 |
| `gyro` | 3 | `/dev/shm/imu_central` gyro |
| `joint_pos` | 12 | `/dev/shm/joint_state` pos − `HOME12` |
| `joint_vel` | 12 | `/dev/shm/joint_state` vel |
| `cmd` | 2 | `(vx, wz)`，自產 |
| `last_action` | 12 | 自存 |
| `cpg` | 24 | `rx, rx_d, ry, ry_d, sin θ, cos θ`，自算 |

合計 **68**。

⚠️ `joint_state` 的 16 軸中**輪關節夾在腿關節之間**（`fl1,fl2,fl3,fl4,fr1,...`），
位址不連續，切片必須用 `LEG_QPOS_IDX` / `LEG_QVEL_IDX`，不可以用連續切片。
⚠️ SHM 腿序是 `FL,FR,BL,BR`，動作設定檔是 `FR,FL,RR,RL` —— **一律按名稱對應**。

### 5.1 刻意排除的三項（要寫進文件）

| 排除 | 理由 |
|---|---|
| **機身線速度** | 底層沒有這個量。高層 SDK 有，但凍結 `mc_ctrl` 後高層就死了。reward 仍可用模擬真值，因為 reward 訓練完就丟棄 |
| **足端觸地** | 沒有感測器。task6 在同型輪足上實測四個候選訊號全部不可用——0.9 kg 的輪子讓擺動相的膝力矩與位置誤差**大於**站立相，訊號方向是反的 |
| **輪子角速度** | 實機讀得到，但實機從未走過路，輪速在真實地面（地毯 vs 磁磚）的行為零資料可對照，而 policy 又控制不了它。放進去等於押一個沒驗證過的量 |

### 5.2 ⚠️ 上實機前的硬性前提：IMU 尚未驗證

`/dev/shm/imu_central` 目前**只有離線快照解碼過，沒有驗證過是不是活的串流**
（`joint_state` 的 1 kHz 活性是驗過的，IMU 沒有）。
而且 `xyzw` 順序是「取樣當下機身剛好水平」推出來的，**不是刻意做的平放實驗**。

四元數順序若錯 → 重力向量翻掉 → policy 直接廢掉。

本 spec 交付一張 `task7/docs/現場操作卡_IMU平放複核.md`：
讀 30 秒串流看數值有沒有在動 ＋ 平放／前傾／側傾三姿態與加速度計交叉驗證。
**這張卡不執行不影響訓練，但不執行就不可以上實機。**

### 5.3 唯一真實來源

`task7/inference/obs_max.py` 定義 `OBS_LAYOUT` 與 `build_obs`。
Colab notebook 的 `_obs` 與本機 `local_infer_max.py` **共用同一份欄位順序定義**，
並由 `task7/tests/test_obs_max.py` 釘住維度與順序。

理由：`np.concatenate` 對長度錯誤的輸入不會報錯，會靜默產生錯誤維度的 obs，
而錯誤維度的 obs 會讓訓練好的權重直接失效。

---

## 6. Reward、終止與 domain randomization

### 6.1 Reward

移植 task6 reward v4 的全部懲罰項（權重都經實測校準），加上偏航率追蹤。

正項：

| 項 | 形式 |
|---|---|
| 前進速度追蹤 | `exp(−(vx − cmd_vx)² / σ)` |
| **偏航率追蹤** | `exp(−(wz − cmd_wz)² / σ)` ← 本次做 RL 的主要理由 |
| 抬腳 | `r_clr`（四輪離地淨空） |
| 存活 | 常數 |

罰項：

| 項 | 針對 |
|---|---|
| `W_PITCH · grav_x²` | 機身俯仰姿態 |
| `W_PITCHRATE · qvel[4]²` | 俯仰角速度 |
| `W_VZ · vz²` | 機身垂直彈跳（v3 打地鼠打出來的洞） |
| `W_OMEGA_VAR · var(各腿 ω)` | 封住「把各腿頻率拆開」這個規避管道 |
| 側向速度 `vy²`、`roll²` | 走直線 |
| 動作變化率 | 抖動 |

⚠️ **權重是照 D1 EDU（20.6 kg）的量值訂的，這台 41 kg、尺度不同，
第一輪訓練後必須重新校準。** notebook 的 `progress` 會逐項印出實測值供校準
（task6 就是這樣從 v1 迭代到 v4 的）。

**reward 可以用上帝視角，obs 不行**：`r_clr` 用 `geom_xpos` 真值，實機沒有這個訊號，
但 reward 只在訓練時計算，訓練產物只有 policy 權重，推論時一行都不會執行到。

### 6.2 終止

- 翻倒：`grav_z > −0.5`
- 低姿：機身高 < **0.29 m**
  （由 task4 Go2 的 0.18 / 0.30 等比例換到 D1 Max 站立高 0.48 m）

低姿護欄是必要的：「塌腰趴著慢慢挪」是很好爬的 local optimum。

### 6.3 指令範圍

| | 範圍 |
|---|---|
| `cmd_vx` | 0.05 – 0.40 m/s（基準開迴路 0.148；開迴路 ω=1.8 曾達 0.316） |
| `cmd_wz` | **60% 機率抽 0**（＝「給我走直」），其餘 40% 在 −0.4 – +0.4 rad/s 均勻抽 |

### 6.4 Domain randomization

| 項目 | 範圍 | 依據 |
|---|---|---|
| 地面摩擦 | 0.4 – 1.4 | 0.3 會跌、≥0.5 能走 |
| 機身 payload | 0 – 5 kg | 官方額定 5 kg，**同時吸收 MJCF 38.8 vs 規格書 41 kg 的 2.2 kg 缺口** |
| 連桿質量 | ±10% | |
| PD 增益 | kp ±20%（腿 48–72 / 96–144）、kd 0.5 – 2.0 | 名目 60/120/120、1.0 |
| 腿關節 `frictionloss` | ×0.5 – ×1.5 | ★ ABAD 的 1.85 掃描已證明是**下界**不是量測值 |
| 輪 `frictionloss` | 0.10 – 0.25 | 兩天兩種條件量到 0.153 / 0.170 |
| IMU | gyro 雜訊 σ=0.10、每 episode 偏差 ±0.05、重力向量雜訊 σ=0.02 | |
| **動作延遲** | 0 – 1 個控制步隨機 | ★ task6 沒做。實機鏈路是「我們寫 shm → 1 kHz daemon 讀 → 馬達」，一定有延遲 |
| 推撞 | 每 2 s 注入 0.6 m/s 速度擾動 | |

---

## 7. 訓練規模與驗收關卡

PPO（brax `0.14.2` / MJX），**版本鎖死不放寬** ——
brax 載到 activation 不匹配的權重**不會報錯**，只會讓 policy 靜默錯亂。

`num_envs=2048`、`episode_length=1000`（20 s）、`batch_size=256`、
`num_minibatches=32`、`unroll_length=20`、`lr=3e-4`、`entropy_cost=1e-2`、
`discounting=0.97`、`normalize_observations=True`、
network `(256,256,128)` / value `(256,256,256)`，先設 **60M steps**。

⚠️ **成本預告**：D1 Max 的 `SIM_DT=0.002`，每控制步 **10 個物理步**，
task6 是 5 個 → 每步計算量約 **2 倍**。notebook 在第一個 eval 就印出實測速率，
由使用者當場決定要不要砍到 40M。

### 驗收關卡

| 關卡 | 內容 | 通過條件 |
|---|---|---|
| **G1** | 同一組 walk 參數，在「原始網格＋迴圈內 PD」與「圓柱＋position 致動器」兩模型各跑 20 s × 12 擾動 | 跌倒數兩邊都 0；行進速度／彈跳／平均俯仰／支撐腳數差異 **≤ ±15%**。超出就先查原因，不進訓練 |
| **G2** | `obs_max.py` 維度與順序測試、`gait_baseline` 常數測試、MJX 模型生成測試 | 全過 |
| **G3** | 訓練後用 `local_infer_max.py` 在**原始網格模型**上回放：20 / 60 / 180 s × 12 擾動 | 跌倒數 0 |
| **G4** | 直線指令（`cmd_wz=0`）下的偏航率 | **顯著優於開迴路的 −0.5 ~ −0.9 °/s** |
| **G5** | 行進速度用 `speed_travel` 量 | **不可以用 `speed_path`，那個高估 68%** |
| **G6** | 錄影 | `task7/outputs/cpg_rl_max.mp4` |

G3–G6 需要訓練完成的權重，屬於使用者跑完 Colab 之後的驗收。

---

## 8. 交付物清單

| 檔案 | 用途 |
|---|---|
| `task7/inference/gait_baseline.py` | 基準步態唯一真實來源 |
| `task7/docs/基準步態凍結_D1Max_walk_2026-08-27.md` | 凍結說明與複驗數據 |
| `task7/model/zgws/make_mjx_model.py` → `zgws_mjx.xml` + `scene_flat_mjx.xml` | MJX 訓練模型生成器 |
| `task7/docs/MJX模型對照_2026-08-27.md` | G1 的 A/B 數據 |
| `task7/inference/obs_max.py` | 觀測層唯一定義 |
| `task7/notebooks/cpg_rl_max_colab.ipynb` | **使用者拿去 Colab 跑的東西** |
| `task7/inference/local_infer_max.py` | 載權重、本機原始網格模型回放／錄影／量測 |
| `task7/tests/test_gait_baseline.py`、`test_obs_max.py`、`test_mjx_model.py` | 測試 |
| `task7/docs/現場操作卡_IMU平放複核.md` | 上實機前的 IMU 驗證流程 |
| `task7/docs/CPG-RL_D1Max_設計_2026-08-27.md` | 設計與結果總結，掛進 `task7/HANDOFF.md` |

---

## 9. 明確不做的事（範圍外）

1. **地形／斜坡／台階訓練** —— 只跑平地，DR 拉滿。
2. **輪子納入動作空間** —— 維持 12 維純腿，輪子只做阻尼。
   輪子位置增益實測會造成 +39° 偏航失控。
3. **全向移動（`vy`）** —— ABAD 行程只有 −0.697 ~ +0.523 rad 且左右鏡像，
   側移學習成本高，本次不做。
4. **實機部署** —— 實機連承重站立（M7）都還沒做。本 spec 全部結論只在模擬內成立。
5. **偏航開迴路 P 控制** —— 由 RL 的偏航率追蹤取代，不另外寫一層。
