# CPG-RL v3：雙模式（輪行／踏步）模仿原廠步態 —— 設計（2026-09-09）

## 目的

使用者總目標：訓出與原廠運控相似的步態。原廠模式表（`task7/docs/results/L_*.md`、資料集 `outputs/ref_gait_dataset.json`）：
- **前進／後退／邊走邊轉＝純差速輪行**，腿站姿只有 ≤ 5° 順從，0.17–0.92 m/s，有效輪距 L_eff 0.375 m。
- **原地旋轉＝對角一對腿同相踏步（2.5 Hz、抬 22 mm、duty 0.8）＋另一對輪反向滾**（左轉 fr+/bl−），±70–77°/s。
- **平移＝移動側前後腿交替踏步（2.1 Hz）＋同側兩輪反向滾**（左移 fl+/bl−0.67）。
- 踏步→輪行切換 0.9–1.5 s。

輪子模型（`results/K_*.md`，M11）：τ = kd·(v_des − v) + τ_ff；τ_f 0.13、b 0.015、延遲 5–7 ms、頻寬 > 4 Hz；實機 kd 1.0 ＋ τ_ff 0.13·sign。

## 範圍

做：新 env `inference/rl_env_v3.py`（不動 v2）、新 MJX 模型變體（輪 kv 1.0、damping 0.015）、模式產生器、動作／觀測／reward、DR、G0–G2 型檢查、Colab notebook、測試。
不做（另開 spec）：狗上 M9 的 v3 推論路徑、身體速度估測器實機驗證、Wasserstein 模仿項進 reward（只在離線評估）。

## 1. 模式產生器（由指令決定，不由 RL 決定）

指令 cmd = (vx, vy, wz)。
```
STEP  if |vy| ≥ 0.03 m/s  or (|wz| ≥ 0.3 rad/s and |vx| < 0.08 m/s)
WHEEL otherwise
```
切換用 1 s 線性混合 `u_mode`（0 = WHEEL, 1 = STEP），腿目標＝(1−u)·站姿 + u·CPG 目標；輪速同樣混合。

### 1.1 WHEEL 模式
- 腿目標＝原廠輪行站姿 `wheel_stance_q12`（資料集：hip 0.52／knee −1.21 前腿、後腿鏡像），加 RL 的足端小偏移（x、z 各 ±20 mm，斜率限制），實現「≤ 5° 順從」。
- 輪速：差速 `v_L = vx − wz·L/2`、`v_R = vx + wz·L/2`，`ω = v / r`（r 0.096、L 0.375），加 RL 殘差 ±20%。
- CPG 相位凍結（θ 不推進），obs 裡的 cpg 段固定。

### 1.2 STEP 模式
- CPG：ω 名目 2.5 Hz（RL 調 1.8–3.2）、duty 0.8、抬高 g_c 22 mm（z_sag 另計）、步長由指令決定：
  - 旋轉（|wz| 主導）：踏步腿＝對角對（wz>0：fl+br；wz<0：fr+bl），**同相**（相位差 0）；另一對腿站姿不踏。
    每步足端位移＝資料集的 (dx, dy)（fl (−21, +39)、br (+16, −25) mm）乘 |wz|/1.3 的比例，方向隨 wz 符號鏡像。
    輪：踏步腿的輪 0（純阻尼）；站姿腿的輪 ±Ω，Ω = 4.1 rad/s × |wz|/1.3（左轉 fr+/bl−）。
  - 平移（|vy| 主導）：踏步腿＝移動側前後腿，**交替**（相位差 0.5）；另一側站姿。
    每步位移 fl (−23, +38)、bl (+20, +29) mm × |vy|/0.06；輪：移動側 前 +Ω、後 −0.67Ω，Ω = 3.3 × |vy|/0.06；另一側 0。
- RL 調變：ω 尺度、逐腿步長尺度 (0.5–1.5)、抬高尺度 (0.6–1.4)、sway (±40 mm)、輪速殘差 ±20%。

### 1.3 動作空間（12 維，tanh 後線性映射）
`[ω_scale, amp×4, lift, sway_x, sway_y, wheel_res×4]`。WHEEL 模式下 amp×4 解成足端 x 偏移（±20 mm）、lift 解成足端 z 偏移（±20 mm）；ω_scale 無效。
基準動作（全 0）＝純開迴路的原廠模式表 → G0 有標準答案。

## 2. 觀測（狗上全部拿得到）

`gravity 3 | gyro 3 | joint_pos 12 | joint_vel 12 | wheel_vel 4 | cmd 3 | mode 2 (u_mode, 1−u_mode) | head_err 1 | last_a 12 | cpg 24` = 76 維。
不放：身體線速度、絕對 yaw、接觸。雜訊：同 v2（gravity 0.005、gyro 0.002、qpos 0.001、qvel 0.2/0.4）＋ wheel_vel N(0, 0.1)。
延遲：動作 1–2 tick（腿與輪同）、joint_vel 延 1 步。IMU 偏轉 ±3°、gyro 偏置。

## 3. Reward

正項：vx 追蹤（真值速度，σ² 0.02）、vy 追蹤（σ² 0.005）、偏航率追蹤（雙尺度核，同 v2.3）、航向誤差（v2.4）、高度、
STEP 模式下踏步腿的抬高接近 22 mm（clearance 核）、WHEEL 模式下腿保持站姿（足端偏移 < 20 mm 的核）。
負項：roll²、pitch²、roll 率、pitch 率、**靜態偏置罰**（roll 與 sway_y 的 EMA 平方，J 報告 F）、Δaction²、Δω²、力矩²、
護欄（τ_bar 58、err_bar 0.45、膝速度 > 14 rad/s）、WHEEL 模式下任何腿抬高 > 10 mm（不該踏步）、STEP 模式下非踏步腿抬高。
終止：跌倒、太低、|τ| > 90 連 3 步。

## 4. 致動器與 DR

- 新 MJX 模型 `zgws_mjx_v3.xml`：腿同 kp250；輪 `<velocity kv=1.0>`、關節 damping 0.015、frictionloss 0.13、armature 0.004。
- 前饋：ctrl_wheel = v_des + sign(v_des)·τ_ff/kv（等價於加 τ_ff），τ_ff 0.13；|v_des| < 0.3 rad/s → 0。
- DR：v2 那組 ＋ 輪 frictionloss 0.10–0.18、damping 0.005–0.03、輪半徑 ±3%、**ABAD 零點偏置 ±3°（靜態外張）**、ABAD kp ×0.4–1.0（E）。
- 指令抽樣：每 episode 先抽模式（WHEEL 55%：vx ∈ [−0.4, 0.9]、wz ∈ ±0.4；TURN 25%：wz ∈ ±[0.5, 1.3]、vx 0；LAT 20%：vy ∈ ±[0.03, 0.08]），
  40% 的 episode 在 3–8 s 時切一次指令（含模式切換）。

## 5. 驗收

- **G0**：零動作（純模式表）—— WHEEL vx 0.5：速度誤差 < 10%、腿抬高 < 10 mm；TURN wz 1.3：偏航率 ≥ 50°/s 不摔；LAT vy 0.06：側移 ≥ 0.03 m/s。
- **G1**：模擬用 M11 的 step 序列跑一次 → τ_f、b 在實測 ±20%。
- **G2**：測試全綠；obs 76 維與推論端一致。
- **G3–G7**：同 v2（跌倒、姿態、力矩護欄）加 vy 與 wz 追蹤誤差。
- **G8（新）**：policy 的 obs／動作分布對原廠錄檔的 Wasserstein（`diag/sim2real_gap_dist.py` 加第三個分布）。

## 6. 風險

- 對角同相踏步（兩腿同時離地）靜態不穩，靠另一對輪的滾動撐；模擬接觸若太軟會摔 → G0 先看。
- 2.5 Hz、0.08 s 擺動對 kp250 伺服是新工況；命令速度用 G 的檢查估。
- 側移速度沒有實機量測，LAT 的 vy 尺度是估的（0.06 m/s）。

## 7. 實作結果（2026-09-09 晚，`inference/rl_env_v3.py`，G0 = `diag/g0_v3.py`）

| 指令 | 零動作行為（CPU MJX，200 步） | 判定 |
|---|---|---|
| WHEEL vx 0.5 | vx 0.49 m/s、roll 0.1°、腿抬 0、膝 20 | ✅ |
| WHEEL vx 0.5 + wz 0.3（弧線） | vx 0.48、偏航 11°/s（原廠 7.7）、膝 42 | ✅（**需輪子位置環 kp 60**；純速度伺服 kv 1.0 偏航 0） |
| TURN wz 1.3 | 偏航 27°/s（原廠 77）、膝 70、踏步腿離地 10 mm | ⚠️ 可用，幅度只有原廠 1/3 |
| LAT vy 0.06 | duty 0.5：摔；duty 0.85／1.5 Hz：站住但膝 103 | ❌ v3.0 不訓（P_LAT 0） |
| STAND | 穩 | ✅ |

發現與決定：
1. **輪行的差速轉向靠速度伺服 kv 1.0 做不到**（命令 5.8/4.6 rad/s 實際 5.2/5.15，打滑阻力撐住差速）。原廠輪子是 kp 60 的位置環
   （目標角逐步累加），改成這樣後弧線 11°/s。模型變體 `scene_flat_mjx_v3p.xml`（`make_mjx_model.WHEEL_MODEL_V3P`），
   env 預設 `wheel_pos=True`，前饋等價為角度偏移 τ_ff/kp。**實機未驗** —— 輪行模式上機那趟要試 kp 20–60 的位置環有沒有抖振。
2. **原廠 2.5 Hz／擺動 0.08 s／抬 22 mm 的踏步在 kp250 位置伺服下做不到**：踏步腿只離地 4 mm（36 mm 撓度吃掉），原地轉 5°/s。
   命令抬高加上 Z_SAG 36 mm 後掃描：2.0 Hz／duty 0.5／抬 40 mm → 27°/s、膝 70；1.5 Hz／50 mm → 49°/s 但膝 88（貼終止線）。
   名目取前者，RL 在 ω ±28%、抬高 ±40%、步長 ±50% 內調。原廠值留在 `REF["factory_*"]`。
3. **平移（原廠式單側踏步）不可行**：原廠靠另一側輪胎側滑；模擬 μ 1.0 不側滑（降到 0.4 也沒救，且弧線變差）。
   只有 duty ≥ 0.85 站得住而膝 > 100。v3.0 不抽平移指令，路徑保留；之後另做四腿蟹行（不像原廠但穩）。
4. 指令抽樣：WHEEL 65%（vx −0.4～0.9、wz ±0.4 半數為 0）、TURN 35%（|wz| 0.5–1.3）；40% episode 中途切指令。

檔案：`rl_env_v3.py`（env、DR）、`model/zgws/{zgws,scene_flat}_mjx_v3{,p}.xml`、`notebooks/cpg_rl_v3_colab.ipynb`（`build_nb_v3.py`）、
`tests/test_rl_env_v3.py`（模式、圖案鏡像、控制律、常數對資料集、短 rollout G0）。
未做：`local_infer_v3.py`（驗收工具，obs/act 與 v2 不同）、狗上推論路徑（M9 v3：輪速指令、位置環、76 維 obs）。
