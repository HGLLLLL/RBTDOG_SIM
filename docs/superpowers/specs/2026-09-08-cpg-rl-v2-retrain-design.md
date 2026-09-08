# CPG-RL v2 重訓（D1 Max）—— 設計（2026-09-08）

## 目的

以實機驗證過的 `A_kp250_walk`（LS / kp250 / abad60 / kd2 / wheel_kd0.5）為基準與對照組，
重新訓練 CPG-RL policy。上一版權重（2026-08-27，kp120 時代）報廢。

使用者訂的目標優先序：
1. **姿態穩定**：把走路時的 roll 搖擺壓下來（實機 ±7~11°，開迴路做不到）
2. **前腳執行率與前後對稱**：上一版前腳幾乎不往前踏
3. **航向**：偏航漂移歸零（上一輪只改善 40%）
4. **速度**：附帶，不追

使用者決策：IMU（重力＋角速度）進 obs（已於 `docs/H_實機obs盤點_2026-09-08.md` 驗證）；
IMU 姿態偏置用隨機化吃掉、不修正；動作空間走**方案 2**（加 body sway）；
**必須確保不會重演馬達力矩過高**（9/3 sway 趟膝 71.3 N·m / 誤差 0.62 中止）。

**訓練在 Colab GPU 跑，不在本機。** 本機只做訓練前檢查（G0–G2）與訓練後驗收（G3–G8）。

## 範圍

本 spec 只涵蓋**訓練 ＋ 模擬驗收**。狗上推論（M9 加 policy 路徑、numpy 推論、護欄整合）另開 spec。
不做：地形、輪子進動作空間、`vy`。

## 1. 動作空間：14 維

`a = [mux×4, muy×4, ω×4, sway_x, sway_y]`，`tanh` 後線性映射：

| | 範圍 | 備註 |
|---|---|---|
| `mu_x`, `mu_y` | 1.0 – 2.0 | 同前 |
| `ω` | 0.0 – 2.0 | 同前 |
| `sway_x`, `sway_y` | **±0.060 m** | 四腳同時相對機身平移＝質心移到支撐多邊形內（E 文件：前腳擺動時 S = −35 mm 是搖擺根因）。M8 實測橫移 140 mm 達成率 94–101%，60 留餘裕 |

sway 的施加做**斜率限制**：目標值每控制步最多變 **0.004 m**（= 200 mm/s），
承重腿不會被瞬間拉走（9/3 那 +0.2 rad 膝誤差就是承重腿吃到指令運動）。

CPG 鏈（與 `gen_gait_traj` / M9 `--live` 同一套數學）：
`cpg_step → duty_remap(0.80) → foot_targets(LS 相位, x_off −0.030, g_c 0.048, z_sag 0.036 只加擺動相) + sway(x,y) → 解析 IK → 12 目標角 → clip 限位`。
基準步態＝固定動作（`atanh` 反推 mu/ω，sway=0），仍有標準答案可對（G0）。

## 2. 觀測：70 維 ＋ 隨機化

`OBS_LAYOUT = [gravity 3, gyro 3, joint_pos 12, joint_vel 12, cmd 2, last_action 14, cpg 24]`。
`obs_max.ACT_DIM` 12 → 14、`OBS_DIM` 68 → 70；`test_obs_max` 的凍結斷言同步改（這是刻意的、會失敗的決定）。

每 episode 抽一次（依 H 文件 §5）：

| 項 | 範圍 | 依據 |
|---|---|---|
| IMU 安裝偏轉 | roll、pitch 各 ±3°（套在 gravity 向量與 gyro 軸向） | 實測 pitch −1.24° / roll +0.63°，蓋過並留餘裕 |
| gyro 偏置 | ±0.02 rad/s | 實測 y 軸 −0.013 |
| 動作延遲 | **固定 1 控制步（20 ms）＋ 再抽 0/1 步** | 實測伺服比模擬多落後 15 ms（兩種 kp 都是） |
| `joint_vel` 延遲 | 固定多 1 步 | driver 濾波實測落後 12–20 ms |

每步雜訊（高斯 std）：joint_pos 0.001 rad、joint_vel ABAD/HIP 0.2、KNEE 0.4 rad/s、gyro 0.002 rad/s、gravity 0.005。

Domain randomization 沿用上一版（地面摩擦 0.4–1.4、payload 0–5 kg、連桿質量 ±10%、
kp ×0.8–1.2 / kv 0.5–2.0、腿摩擦 ×0.5–1.5、輪摩擦 0.10–0.25、推撞 0–0.6 m/s 每 2 s），
**ABAD 的 kp 與摩擦範圍加大到 ×0.6–1.4**（ABAD 側向力矩模擬不準 10×、站立相角度實/模差 0.05 rad）。

## 3. Reward

正項上限總和約 6；每一項在「基準固定動作」上的實測值由 notebook 基準校準格印出，
**沒有一項罰項在基準步態上超過正項總和的 15%**（否則權重下修）。

| 目標 | 項 | 權重（起始值，校準後可改） |
|---|---|---|
| ① 姿態 | `−W_ROLL·grav_y²`、`−W_ROLLRATE·ω_x²`（**新增**）；`−W_PITCH·grav_x²`、`−W_PITCHRATE·ω_y²`（沿用） | W_ROLL 20、W_ROLLRATE 0.05、W_PITCH 20、W_PITCHRATE 0.05；校準準則：基準 roll（模擬 ±3.5°）時 roll 兩項合計 ≈ 正項 10% |
| ② 執行率／對稱 | `+W_EXEC·mean_legs(swing_k · exp(−(x_foot,k − x_cmd,k)²/σ²))`，x 為機身系足端 x（FK 自 qpos）vs CPG 目標 x，σ = 0.03 m；`−W_SYM·(ē_front − ē_rear)²`，ē 為擺動相 x 追蹤誤差的前/後平均 | W_EXEC 1.0、W_SYM 5.0 |
| ③ 航向 | `+W_YAW·exp(−(ω_z − cmd_wz)²/0.05)` | 1.0；60% episode `cmd_wz = 0`，其餘 ±0.4 rad/s |
| ④ 速度 | `+W_VX·exp(−(v_x − cmd_vx)²/0.02)` | 1.5；`cmd_vx` 0.05–0.35 m/s |
| **護欄** | `−W_TAUBAR·Σ_j max(\|τ_j\| − 58, 0)²`、`−W_ERRBAR·Σ_j max(\|e_j\| − 0.45, 0)²` | W_TAUBAR 0.01（超 10 N·m ⇒ −1）、W_ERRBAR 20（超 0.2 rad ⇒ −0.8）。58 = 實機門檻 70 ÷ 1.2（kp250 實測比值 ×1.14 兩次）；0.45 = M9 中止 0.6 減 kp250 設計落後 0.3 的餘裕 |
| 沿用 | `+W_VY·exp(−v_y²/0.02)` 0.5、`+W_H·exp(−400(h−0.4825)²)` 0.5、`+W_CLR·r_clr` 1.5、`−W_VZ·v_z²` **2.0**（由 0.5 上調：41 kg 校準，上輪彈跳退步 2.5×）、`−W_OMEGA_VAR·var(ω)` 0.5、`−W_ACT·Σ(Δa)²` 0.01、`−W_TAU·Στ²` 3e-5 | |

reward 可用上帝視角（真值速度、`geom_xpos`、`actuator_force`），obs 不行——沿用原則。

## 4. 終止

翻倒 `grav_z > −0.4`；機身高 < 0.29 m；**新增：任一腿關節 |τ| > 90 N·m 連續 3 控制步**（模擬版急停：懸崖不是斜坡）。

## 5. 模型與增益

- `task7/model/zgws/make_mjx_model.py` 改成可帶參數：`--kp 60,250,250 --kd 2,2,2 --wheel-kd 0.5 --out zgws_mjx_kp250.xml`。
  預設行為（不帶參數）維持產生原本的 `zgws_mjx.xml`，kp120 線不受影響。
- `task7/inference/gait_baseline.py` 新增 **`BASELINE_A`**（凍結 `outputs/A_kp250_walk.json` 的 `params` ＋ 增益：
  seq ls、duty 0.80、omega 1.4、mu_x 1.80、mu_y 1.50、d_step 0.10、d_step_y 0.12、x_off −0.030、g_c 0.048、
  z_sag 0.036、kp [60,250,250]、kd 2.0、wheel_kd 0.5、wheel_mode damp）。測試斷言與軌跡檔逐欄相等。
- `max_model` 新增 `KP3_A = [60,250,250]`、`KD3_A = [2,2,2]`，不改 `KP3/KD3`。

## 6. 訓練（Colab）

- notebook `task7/notebooks/cpg_rl_max_colab.ipynb` 更新：`BRANCH = "main"`；SCENE 改 `zgws_mjx_kp250.xml`；
  常數自 `gait_baseline.BASELINE_A`／`max_model.KP3_A` import；env 依 §1–§4；DR 依 §2。
- PPO / brax 0.14.2 / mujoco 3.10.0 / jax<0.10 版本鎖死不放寬（activation 不匹配載權重不報錯）。
  2048 env、episode 1000 步（20 s）、batch 256、minibatch 32、unroll 20、lr 3e-4、entropy 1e-2、γ 0.97、
  normalize_observations、policy (256,256,128)、value (256,256,256)、60M steps；第一個 eval 印實測步率決定是否砍到 40M。
- metrics 新增：`roll`、`roll_pk`、`exec_f`、`exec_r`、`tau_pk`、`err_pk`、`sway_x`、`sway_y`；每個 eval 印。
- **Colab 用 `git clone --depth 1` 從 GitHub 拿程式** → 所有改動 commit + `git push origin main` 後才開訓。
- 輸出 `cpg_rl_max_v2_params.pkl` → 放 `task7/weights/`。

## 7. 驗收關卡（原始網格模型 `scene_flat.xml`，不用訓練模型驗訓練模型）

| 關卡 | 內容 | 過關 | 在哪跑 |
|---|---|---|---|
| G0 | env 餵基準固定動作（sway=0）重現 A 步態 | `speed_travel / bounce / roll / support / min_lift` 與 `cpg_walk_max` rollout 逐位相同（`local_infer_max --dummy`） | 本機 CPU |
| G1 | `zgws_mjx_kp250.xml` vs 網格模型，A 步態 20 s × 12 擾動 | 速度／彈跳／支撐／離地／roll 峰值差 ±5% | 本機 |
| G2 | 測試全綠：obs 70 維釘住、`BASELINE_A` 釘住、IK JAX vs numpy 1e-4、sway 斜率限制、護欄項數值 | | 本機 |
| G3 | 20 / 60 / 180 s × 12 擾動 | 0 跌倒 | 本機 |
| **G4 姿態** | roll 峰值與 std vs 開迴路 A（同模擬、同擾動） | **峰值降 ≥ 60%、std 降 ≥ 60%**（兩者都要） | 本機 |
| **G5 執行率** | 前腳執行率、前後差 | ≥ 0.9、\|前−後\| < 0.15 | 本機 |
| G6 航向 | `cmd_wz = 0`，60 s 總偏航（用 `yaw_total`） | < 5° | 本機 |
| **G7 力矩護欄** | 12 擾動逐關節峰值 ×1.2、追蹤誤差 ×1.14 | < 70 N·m、< 0.6 rad；**任一不過不上機** | 本機 |
| G8 | 影片 `outputs/cpg_rl_max_v2.mp4` ＋ `docs/CPG-RL_v2_結果_<日期>.md` | | 本機 |

G3–G7 由同一支 `local_infer_max.py --params … --perturb 12` 一次印完（把 policy 接上 `cpg_sweep_max` 的擾動機制，上輪 §6.2 承認沒做的那項）。

## 8. 交付物

`gait_baseline.BASELINE_A`、`max_model.KP3_A/KD3_A`、`make_mjx_model.py` 參數化 ＋ `zgws_mjx_kp250.xml`、
`obs_max`（ACT 14 / OBS 70）、notebook v2、`local_infer_max.py`（14 維、sway、`--perturb`、G4–G7 指標）、
`weights/cpg_rl_max_v2_params.pkl`、結果文件、HANDOFF 更新。

## 9. 風險與已知限制

- sway 的實機行為只驗過開迴路正弦（9/3，被 ABAD bug 干擾），RL 版 sway 沒上過機 → G7 是上機的唯一守門員。
- 模擬對垂直順從性高估 1.8×（z 方向），roll 搖擺在同組態下模擬忠實（G 文件）；G4 的 60% 是模擬內的相對值，實機要再驗。
- `joint_vel` 雜訊在 KNEE 0.4 rad/s 是步態工況實測；若 policy 對 joint_vel 過度依賴，可加 dropout 式隨機化（不在本輪）。
- 上一輪教訓：policy 把 mu_x 縮到 1.47 省力矩——本輪 W_EXEC 直接獎勵擺動相 x 追蹤，理論上封住；校準格會印基準的 exec 值。

---

## 附錄 A：v2.2（2026-09-08 晚，使用者選方案 A）

v2 權重驗收 G4/G5/G6 全 ❌ 且「比開迴路差」，根因四個（詳 `task7/docs/CPG-RL_v2_設計_2026-09-08.md` §7）：
速度指令逼縮步（基準 0.34 但指令 0.05–0.35、核 σ 0.14）、執行率獎勵獎勵「追上自己縮小的目標」、
對稱項量錯量（追蹤誤差差恆 0）、roll 角度罰項文獻裡基本不存在。

決策（使用者）：**mu_x 固定在基準 1.8，從動作空間拿掉**。

| 項 | v2.1 | **v2.2** |
|---|---|---|
| 動作 | 14（mux,muy,ω ×4 + sway） | **10**（muy,ω ×4 + sway）；obs 66 |
| 指令 vx | 0.05–0.35 | **0.15–0.40**；驗收 0.30 |
| 速度核 | σ²=0.02 | **0.1**（CPG-RL 原文 0.25 的量級） |
| 執行率 | 擺動相 x 追蹤 exp(−dx²/σ²) | **真執行率**：擺動結束時 (足端相對機身實際前跨)/(指令前跨)，clip 0–1，保持到下次擺動 |
| 對稱 | 追蹤誤差差 | **前後執行率差**² |
| 姿態 | roll 角 14%、pitch 5% | roll 角 ~5%、roll 率 ~5%、pitch 角/率各 ~3%（校準量） |
| 偏航 | 4 s EMA 核 | 同 |
| 護欄／終止／延遲／雜訊／sway／DR | 同 | 同 |
| G4 | 峰值與 std 都降 ≥60% | **峰值降 ≥30% 且 std 不升** |

實作：`obs_max` 支援 act_dim ∈ {10, 14}（`obs_dim(act_dim)`）；`rl_env_max.PRESETS["v2.2"]` 含 `ACT_LAYOUT="nomux"`、`EXEC_MODE="rate"`、`CMD_VX=(0.15,0.40)`、`VX_SIG2=0.1`；
`local_infer_max --preset v2.2`；新 notebook `cpg_rl_max_v2_2_colab.ipynb`、權重 `cpg_rl_max_v2_2_params.pkl`。舊 preset 行為不變。

---

## 附錄 B：v2.4（2026-09-08 夜，使用者選 A：給 policy 航向誤差）

v2.3 驗收：G3/G4/G7 ✅（roll −36%、力矩回到基準水準、起步扭動消失）、G5 差 0.01、
**G6 −29.8°/60 s、轉彎只轉出指令 13–21%**。根因：policy 只觀測角速度，看不到累積航向；
4 s EMA 在 ±0.6 rad/s 甩頭振盪下殘留 ±0.02 rad/s，就是 0.5°/s 的底噪，reward 再調也在底噪裡打轉。

決策（使用者）：**obs 加 1 維航向誤差** `head_err = ∫(gyro_z_obs − cmd_wz)·dt`（rad，clip ±1.0），
實機由 policy 迴圈用 IMU gyro_z 自己積分（起走時歸零），不需任何實機沒有的量。

| 項 | v2.3 | **v2.4** |
|---|---|---|
| obs | 66 | **67**（`cmd` 之後插入 `head_err` 1 維；其餘順序不變） |
| reward 新增 | — | `+W_HEAD·exp(−(head_err/0.15)²)`，W_HEAD 1.0（8.6° 時 0.37） |
| gyro 偏置隨機化 | 三軸 ±0.02 | x/y ±0.02、**z ±0.005**（實測 z 偏置 0.001；積分器用的是含偏置的觀測值，policy 要學會容忍） |
| 其餘 | — | 同 v2.3（動作 10 維、雙尺度核、淡入、護欄） |

`obs_max.layout(act_dim, head=True)`；`local_infer_max --preset v2.4` 同步積分；notebook `cpg_rl_max_v2_4_colab.ipynb`、權重 `cpg_rl_max_v2_4_params.pkl`。
上機語意：`head_err` 是「相對起走時航向的偏差」；轉彎指令時是「相對指令航向的偏差」。
