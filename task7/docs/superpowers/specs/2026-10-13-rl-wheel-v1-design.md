# 純 RL 自訓 v1 設計：在我們這邊（MJX + Brax PPO，Colab）重訓同事的崎嶇地形 policy（2026-10-13）

狀態：設計已與使用者逐段定案（訓練目標、硬體、初始權重、驗收門檻、做法、四段設計、樓梯）。下一步是寫實作計畫。

## 0. 動機與背景

- 同事的純 RL 第二版（崎嶇地形訓練，10k）trip25 上機 11 趟 0 摔，但留三個 policy 性質的問題（V 文件 §6）：
  1. 平移時後輪互碰（實體間隙 25 mm；訓練 `enabled_self_collisions: false`）。
  2. 小指令追蹤差：|vy| ≤ 0.3 死區、|wz| 0.5 只 68–70% 且旋轉中心偏。
  3. 輪摩擦：實機直走 0.3 → 0.29，仍差；他的訓練是零關節摩擦（IdealPD）。
- 「接續」他的訓練做不到：他的堆疊是 Isaac Lab（PhysX，4096 envs，GPU），Colab 裝不了 Isaac Sim、本機無 NVIDIA GPU；
  repo 的 `RL_only/` 只有 env.yaml／policy_io.json／observation.py／.pt／.onnx，任務程式碼 `quadruped.tasks.manager_based.wheel`
  與 USD 資產在他電腦（`/home/b50560/isaac_ws/...`）；.pt 0.8 MB 只有 actor（無 critic、無 optimizer），連在他的環境也只能熱啟動。
- 使用者決定：**在我們這邊用現成的 MJX + Brax PPO Colab 線重新訓練**，第一輪就加上述三項改動，從零訓（不用他的 actor 熱啟動），
  Colab Pro（A100／L4），並加入低階樓梯地形。

## 1. 範圍與驗收（使用者定）

- **範圍**：新環境模組照他的 env.yaml 重做（obs 53／動作 16／reward 十項），加三項改動與低階樓梯；訓練、MuJoCo 驗收、匯出狗端 npz。
- **不包含**：上機（trip26 另排，走 `docs/現場SOP_通用_2026-08-27.md`）、M9 硬上限「兩筆相隔 ≥ 4 ms」修正（HANDOFF §4.1，獨立小修）、
  真樓梯（階高 ≥ 15 cm，見 §8）。
- **驗收門檻**（全部在 MuJoCo 驗收表上檢查，過了才考慮上機；對照組＝他的 10k 在同一套工具上的數字）：

| 類別 | 門檻 |
|---|---|
| 不退 | 平地／擾動／崎嶇地（含樓梯）0 摔；直走 0.3 追蹤 ≥ 97%；wz ±1.0 ≥ 95% |
| 改動一 自碰撞 | vy ±0.5、±0.6 平移時後輪**實體間隙** > 40 mm（口徑＝link 中心距 − 34 mm，同 V 文件網格口徑、傾斜時高估 ~7 mm；實機碰撞時 25 mm；他的 10k 在本環境 ~33 mm） |
| 改動二 小指令 | vy 0.3 要動（他是死區）；wz ±0.5 ≥ 90%（他 70%） |
| 改動三 輪摩擦 | real 模式直走 0.3 → ≥ 0.29；real 與 ideal 差 ≤ 5% |

## 2. 做法（三選一選 A）

- **A（採用）**：獨立 MJX 環境模組 `task7/inference/rl_env_wheel.py` ＋ 現成 Brax PPO Colab 線。零件都有先例：Env 結構與 DR 的 vmap 寫法仿 `rl_env_v3.py`（不 import 其 CPG 邏輯，golden 不碰）、obs 函數搬 `rl_only_infer.py`、地形產生函數用 `diag/rl_only_rough_terrain.py`、狗端 npz 走 `rl16` 版面。
- B（不採）：在 `rl_env_v3.py` 加 wheel 模式。929 行且 golden 釘著，兩線會纏。
- C（不採）：移植他的 rsl_rl（PyTorch）接 MJX。最忠實但 JAX↔PyTorch 橋、存檔、正規化全是新管線，風險從環境搬到框架。

## 3. 環境與 reward（`rl_env_wheel.py`）

- Brax `Env`，50 Hz（MJX dt 0.005 × 4 子步），回合 20 s ＝ 1000 步。模型 `scene_wheel_mjx.xml`（§4）。
- **obs 53** 照 `RL_only/policy_io.json`：`ang_vel(3) | projected_gravity(3) | cmd(3) | joint_pos − default(12) | joint_vel(16) | last_action(16)`。
  關節按名稱映射到他的 PhysX 分組序（ABAD×4, HIP×4, KNEE×4, FOOT×4；腿序 FAR, FBL, RAR, RBL）。雜訊照他的均勻範圍：ang ±0.2、grav ±0.05、jpos ±0.01、jvel ±1.5。
  Brax 的 obs 滾動正規化保留（`normalize_observations=True`）。
- **動作 16**：Brax 輸出 tanh 有界 `a_env ∈ [−1, 1]`；環境內 `a = ACT_GAIN · a_env`，`ACT_GAIN = 4.0`，再照他的公式
  `q_des = default + 0.25·a[:12]`（腿 PD 目標，rad）、`v_des = 5.0·a[12:16]`（輪速，rad/s）。物理範圍與他的 |a| ≤ 4 一致（實測他最大 3.5）。
  `last_action` 在 obs 裡放的是 `a`（增益後、他的尺度），對齊驗證時他的輸出直接可比。
- **default pose**：ABAD 0；前 HIP +0.8／後 −0.8；前 KNEE −1.5／後 +1.5；輪 0。站立高 0.485 m。
- **PD**：ABAD kp 60、HIP/KNEE 120、kd 1.0；輪 kp 0、kd 1.0。沿用 `scene_flat_mjx_v3f.xml` 的致動器（S 文件 §2 已確認與他相同）。
- **指令**：vx, vy, wz ∈ [−1, 1]；10 s 重抽；10% 回合站立（三者 0）；heading 模式照他的：抽航向目標，每步 `wz = clip(1.0 · wrap(heading_target − yaw), −1, 1)`。
  **改動二（a）**：30% 的回合從小範圍抽（|vx|, |vy| ≤ 0.4、|wz| ≤ 0.5），其餘 60% 全範圍。
- **reward**（每步，權重＝他的；新項標 ★）：

| 項 | 式 | 權重 |
|---|---|---|
| track_lin_vel_xy_exp | `exp(−‖v_xy_cmd − v_xy‖² / std²)` | 2.5，**std 0.25**（他 0.5，改動二 b） |
| track_ang_vel_z_exp | `exp(−(wz_cmd − wz)² / std²)` | 1.25，**std 0.25**（改動二 b） |
| lin_vel_z_l2 | `vz²` | −2.0 |
| ang_vel_xy_l2 | `‖ω_xy‖²` | −0.05 |
| leg_torques_l2／wheel_torques_l2 | `Σ τ²`（腿 12／輪 4 分開） | −2e-4 各 |
| leg_acc_l2／wheel_acc_l2 | `Σ q̈²` | −2.5e-7 各 |
| action_rate_l2 | `‖a_t − a_{t−1}‖²`（他的尺度） | −0.01 |
| wheel_lateral_slip | 輪接觸時，輪心速度沿輪軸（自旋軸 y）分量的平方和，只算有接觸的輪（他的自訂項，依 yaml 參數重寫） | −0.075 |
| sim2default | `Σ|q − default|`（腿 12），指令範數 < 0.1 時乘 10（他的自訂項 `standing_scale 10、command_threshold 0.1`） | −0.1 |
| dof_pos_limits | 超出 0.9 軟限位的量 | 0.0（保留，與他同） |
| ★ wheel_wheel_contact | 四顆輪子彼此接觸力的 L2（取自 `<pair>` 接觸） | **−1.0**（改動一） |
| ★ wheel_gap（2026-10-13 加） | 前後兩對側向輪實體間隙 gap ＝ 中心距 − 0.034 的 hinge：Σ max(0, 1 − gap/0.08)² | **−2.0**（改動一；40 mm 驗收邊界每對罰 0.5、接觸 2.0；pair 圓盤碰不到實機那種內面接觸，見 §11.9） |

  三個判斷值（std 0.25、−1.0、30%）在 §6 攤帳後若量級不對會調，調了寫進結果文件。
- **終止**：base_link 接觸力 > 1 N，或 20 s（time-out 不算失敗）。
- **domain randomization**（reset 時 vmap，寫法同 `rl_env_v3.make_domain_randomize`）：
  他的：地面摩擦 靜 0.8／動 0.6（我們用 geom_friction 0.6–1.0 均勻）、基座質量 −1～+3 kg、質心 x ±0.05 m、每 10–15 s 推 ±0.5 m/s（x, y）。
  **改動三**：腿關節 frictionloss 用實機辨識值（ABAD／髖膝 1.85／1.5）× 0.5–1.5；輪 frictionloss 0.10–0.18、damping 0.005–0.03（v3 線已有範圍）。
  即「直接在 real 條件下訓」，他的訓練是零關節摩擦。

## 4. 地形與課程

- MJX 一個模型一張 hfield → 所有地形拼成一張大圖，出生位置決定踩哪塊。
- **大圖**：三列 × 十欄的 8 m × 8 m 方塊，解析度 0.1 m，外圍 2 m 平地邊框。欄＝難度等級 0–9，列＝種類：

| 列 | 種類 | 產生函數 | 幅度隨等級線性 |
|---|---|---|---|
| 0 | random_rough | `terrain_grid("rough", amp)` | amp 0.01 → 0.06 m（他的 noise_range） |
| 1 | boxes | `terrain_grid("boxes", amp)`（0.45 m 方格、中央 2 m 平台） | amp 0.025 → 0.10 m |
| 2 | ★ stairs | 新 `terrain_grid("stairs", h)`：金字塔樓梯，踏面 0.3 m，中央 2 m 平台，上樓／下樓各半（反向金字塔） | 階高 h 0.03 → 0.08 m |

  `terrain_grid` 從 `diag/rl_only_rough_terrain.py` 搬到環境模組可 import 的位置（diag 腳本改 import），加 "stairs"。
  方格在 hfield 裡是階梯面而非垂直面，與 U 文件的近似相同。
- **出生**：state.info 帶 `level`（0–9）與 `kind`（0–2 各 1/3）。reset 出生在對應方塊中央 2 m 內隨機點、隨機 yaw，
  z ＝ hfield 查高（JAX gather）＋ 0.485。
- **課程**（他的 `terrain_levels_vel` 規則）：回合結束時，水平位移 > 4 m（方塊一半）→ level +1；位移 < 0.5 · ‖v_xy_cmd‖ · 20 s → level −1；
  夾 0–9；到 9 再升時隨機重抽 0–9。站立回合不升降。等級不進 obs（盲走，同他）。
- **場景檔** `task7/model/zgws/scene_wheel_mjx.xml`：由 `rl_env_wheel.py` 的 `make_scene_xml()` 產生（不手改）＝ `scene_flat_mjx_v3f.xml` 去掉平面、
  加 hfield（nrow／ncol 依大圖、zmax 依最大高度）、加四個輪輪 `<pair>`（FAR–FBL、RAR–RBL、FAR–RAR、FBL–RBL）。
  其他幾何 contype／conaffinity 不動（腿、機身仍不互碰，只開輪輪）。MJX 3.10 支援 cylinder–cylinder 與 hfield 碰撞（已查）。
- **風險**：MJX hfield 碰撞吞吐量沒量過（U 文件在 CPU MuJoCo），§5 throughput 測試一起量。

## 5. 訓練與存檔（`notebooks/build_nb_wheel.py` → `rl_wheel_v1_colab.ipynb`）

- 骨架照 `build_nb_v3.py`：釘 brax 0.14.2／mujoco 3.10.0／`jax[cuda12]<0.10`，版本不對當場停；RESUME 續訓機制沿用。
- **Colab 第一格 throughput 測試**：num_envs 1024／2048／4096 各跑 200 步（JIT 後計時），印每秒步數與 GPU 記憶體。
  選法：取最大且比前一檔至少快 1.6 倍者；4096 記憶體爆就退一檔。
- **PPO**（從 v3 線起步）：`unroll_length 20、num_minibatches 32、num_updates_per_batch 4、learning_rate 3e-4（ADAPTIVE_KL，desired_kl 0.01，1e-5–1e-3）、
  entropy_cost 1e-2、discounting 0.97、normalize_observations True、episode_length 1000`。
  改：`policy_hidden_layer_sizes=(512,256,128)`、`value_hidden_layer_sizes=(512,256,128)`；`batch_size` ＝ 256 × (num_envs / 2048)。
- **總步數**：目標 10 億步（他的 10k ≈ 10k × 4096 × 24 ≈ 9.8 億）。一個 session 跑 3.3 億，`save_checkpoint_path` 指到 Google Drive，
  下一 session `restore_checkpoint_path` 接。Brax 每次 eval（num_evals 20 → 每 5%）存 checkpoint，斷線最多丟 5%。
- **進度欄**（state.metrics 平均，progress 印）：`len`（回合長度）、`lin`（track_lin 項）、`ang`、`ww`（輪輪接觸項）、`slip`、`level`（平均等級）、`small_lin`（小指令回合的 track_lin）。
- **停損線**：2 億步時 `len < 600`、或 `lin < 0.5`、或 `ww` 沒比 step 0 低 → 停，不硬跑完。
- **產出**：`task7/weights/rl_wheel_v1_params.pkl`（Brax 格式，含正規化統計）＋ Drive checkpoint 目錄。

## 6. 開訓前關卡（本機，CPU MJX，不用 Colab）

1. **零訓練對齊**：他的 10k ONNX 放進新環境（`a_env = a_his / 4`，雜訊關、DR 關、flat 等級 0）跑 `rl_only_infer.py` 同樣八個指令，
   追蹤率、摔倒、後輪間距要與 U 文件／`outputs/rl_only/eval_rl_only_real_10k.md` 對得上（追蹤率差 ≤ 3 個百分點、0 摔）。對不上＝環境做歪，不開訓。
2. **reward 攤帳**：同一批 rollout 把十三項各自的每步平均量級列成 `outputs/reward_audit_wheel_v1.md`。
   要看到：輪輪接觸項在 vy ±0.5 非零、小指令回合的追蹤項在 std 0.25 下比 0.5 下明顯低（有東西可咬）、其餘項與他的量級相稱（不被單一項淹沒）。

## 7. 驗證與匯出

- **工具改動**：`rl_only_infer.py`、`diag/rl_only_rough_terrain.py` 各加 `--npz`（用 `realbot/policy_np.py` 推論，與 `--onnx` 並存）
  與 `--self-collision`（載入含輪輪 `<pair>` 的場景；不開輪子互穿，後輪互碰看不出來）。崎嶇地工具加 `stairs:h` 地形，與訓練同一函數。
- **驗收表**：`outputs/rl_wheel/eval_rl_wheel_v1_{real,ideal}.md`、`eval_rl_wheel_rough_real_v1.md`（三種地形 × 3 種子），末節對照 §1 門檻的勾選表；
  他的 10k 用同樣旗標重跑一次當對照列。
- **匯出** `inference/export_policy_wheel.py`：pkl → `weights/rl_wheel_v1_np.npz`，版面 `rl16`（M9 自動選 `gen_direct`／`rl_obs_rl`，狗端堆疊不改）。
  鍵：`mean/std`（正規化）、`W0..3/b0..3`、`activation=swish`、`output=tanh_loc`、**新鍵 `act_gain=4.0`**、`gen_json`（與 `export_policy_rl.py` 同欄位：default pose、0.25／5.0、限位、增益、輪力矩上限）。
  `policy_np.py` 讀 `act_gain`，缺＝1.0 → 舊 npz 逐位元不變。
- **結果文件** `docs/results/W_純RL自訓v1_MuJoCo_<日期>.md`；HANDOFF 留一行。

## 8. 測試（pytest，先寫測試再實作；全套每批跑一次，`test_rl_env_v3_golden.py` 不能動）

- `tests/test_rl_env_wheel.py`：obs 與 `rl_only_infer.Sim.obs` 在同一 qpos/qvel 下逐維相等（1e-5）；動作縮放（a_env=±1 → q_des 偏 ±1.0 rad、輪 ±20 rad/s）；
  每項 reward 在構造狀態下的值與正負（追蹤完美＝權重、站立時 sim2default ×10、輪子分開時 ww＝0）；出生 z ＝ 地形高 + 0.485（三種類各一）；
  兩顆後輪擺到重疊時 pair 接觸力 > 0；課程升／降／到頂重抽；base 觸地終止、time-out 不終止 done 旗標。
- `tests/test_terrain_grid.py`：stairs 的階高、踏面、中央平台、邊緣壓平。
- `tests/test_export_policy_wheel.py`：npz 推論 == Brax 推論（隨機 obs，1e-5）；`tests/test_policy_np_act_gain.py`：缺鍵＝1.0、舊 `rl_only_*_np.npz` 輸出逐位元不變。
- 本機 CPU 煙霧：4 envs × 50 步 jit 跑通（在 test 裡）。

## 9. 真樓梯（這輪不做，前置條件）

- 原廠規格「連續 25 cm 台階」，階高 > 輪半徑 9.6 cm，輪子碾不上去，要抬腿跨階。需要：(1) 抬腳／騰空／落點 reward（他的 reward 沒抬腳項，U 文件證明不抬腳）；
  (2) obs 加本體歷史（盲爬）或高度圖（狗上有前後兩顆 RoboSense RSAIRY 光達 10 Hz，但高度圖管線要在 RK3588 上另做，N 文件相機節點已吃 70% CPU）；
  (3) 狗端 obs 組裝與 M9 版面跟著改；(4) 上機安全另訂。另開 spec。

## 10. 檔案清單

| 新／改 | 路徑 | 用途 |
|---|---|---|
| 新 | `task7/inference/rl_env_wheel.py` | 環境、reward、DR、課程、`make_scene_xml()`、`terrain_grid`（含 stairs） |
| 新（產生） | `task7/model/zgws/scene_wheel_mjx.xml` | hfield 大圖＋輪輪 pair |
| 新 | `task7/notebooks/build_nb_wheel.py` → `rl_wheel_v1_colab.ipynb` | Colab 訓練 |
| 新 | `task7/inference/export_policy_wheel.py` | pkl → rl16 npz |
| 改 | `task7/realbot/policy_np.py` | `act_gain` 鍵 |
| 改 | `task7/inference/rl_only_infer.py`、`diag/rl_only_rough_terrain.py` | `--npz`、`--self-collision`、stairs、改 import terrain_grid |
| 新 | `task7/tests/test_rl_env_wheel.py`、`test_terrain_grid.py`、`test_export_policy_wheel.py`、`test_policy_np_act_gain.py` | §8 |
| 新 | `task7/outputs/reward_audit_wheel_v1.md`、`outputs/rl_wheel/eval_*.md`、`docs/results/W_*.md` | §6、§7 |

## 11. 實作註記（2026-10-13 執行時發現，plan 已照此做）

1. **MJX 3.10 無 hfield–cylinder 碰撞**（實測 `NotImplementedError`）：訓練模型 `zgws_mjx_wheel.xml` 的輪子改凸網格圓盤（半徑 0.096、半寬 0.005、16 邊），hfield–mesh、mesh–mesh 都支援；CPU 驗收工具仍用圓柱模型。零訓練對齊（§6）就是在驗這個差異。
2. **Brax `AutoResetWrapper` 只還原第一次 reset 的狀態、不重呼叫 reset**：回合結束（摔或 t ≥ episode_len）的重生、課程升降、指令重抽全在 `WheelEnv.step` 內用 `jnp.where` 做；訓練用 `rl_env_wheel.wrap_env`（Vmap／DR → Episode → `RespawnWrapper`）傳 `ppo.train(wrap_env_fn=…)`。`WheelEnv` 只回 `done=fell`，time-out 交給 EpisodeWrapper 設 truncation。
3. **地形基準高 `BASE_H = 1.0`、`ZMAX = 2.0`**：反向樓梯（中央是坑）要往下挖，hfield 資料必須 ≥ 0。CPU 崎嶇地工具的 ZMAX 也由 0.30 改 1.0（stairs 0.08 的平台 0.8 m 超出舊量程會穿地；rough／boxes 幾何不變）。
4. **輪輪接觸懲罰的法向力**：場景是金字塔摩擦錐（`opt.cone=0`）、condim 3，每個接觸 4 列 efc，法向力＝4 列和（同 `mj_contactForce`），不是第 0 列；`ww = Σ(fn/100 N)²`。MJX 與 CPU MuJoCo 同一姿勢同一步的輪輪法向力差約 15%（2755 vs 3245 N），`ww_force` 不要當絕對牛頓數讀。
5. 場景檔補 `<option timestep="0.005"/>`（原本沒有，載入後是預設 0.002）。
6. 狗端 `gen_direct` 只收 `ref.family == "rl_only"`（那是 rl16 介面族的名字）：`rl_wheel` npz 的 `ref.family` 維持 `"rl_only"`，用 `preset="rl_wheel"` 與 `gen_json.source.family` 區分；狗端不改。
7. 續訓：brax `restore_params` 不還原 `env_steps`，notebook 用 `STEP_OFFSET`（從 RESUME 檔名 `step<N>` 解析）讓 checkpoint 檔名與進度列是累計步數；optimizer 與 ADAPTIVE_KL 學習率在續訓時重置。
8. **對齊關卡改絕對差**：16 邊網格輪子對 CPU 圓柱平面的追蹤差 vx 列 ≤ 0.022 m/s、wz 列 ≤ 2.0 °/s、0 摔；vy 0.3 在他的死區（0.04–0.14 m/s）百分比無意義。同幾何（cpu′）≤ 1.6 pp。32／48 邊實驗：vx 微好、偏航反而差（MJX 對細網格偏低），維持 16 邊。
9. **pair 圓盤碰不到實機的那種碰撞**：v3f 碰撞圓盤在 link y=−0.047（輪胎外側平面、半寬 5 mm），官方網格 y∈[−0.070,+0.017]，實機後輪互碰是內面（+0.017）相碰；兩圓盤要腿交叉才會碰。所以改動一的主力是 `wgap`（實體間隙 hinge，閾值 80 mm、權重 2.0；他的 10k 在 vy 0.5 時 gap 最小 33 mm），`ww` 力項只擋交叉。驗收與工具的 gap 口徑統一為「中心距 − 34 mm」。
