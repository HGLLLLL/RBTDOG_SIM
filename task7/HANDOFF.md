# task7 交接：D1 Max 現況卡

- 最後更新：**2026-10-20 深夜（v1c 從頭訓 5 億步失敗＝側向軸死掉；改 v1d＝v1b 熱啟動＋V1C（impact 門檻 0.3），notebook 已 push，等使用者 Colab）**（分支 `main`）
- **這份只放「現在」，上限 150 行。** 收工時覆寫本檔，不是往上疊。
  被取代的段落搬到 `docs/handoff_history/`（2026-08-25～09-22 全文在 `HANDOFF_2026-08-25_2026-09-22_全文.md`；
  09-22 §1 在 `HANDOFF_2026-09-22_§1_v3.7b_Colab訓練.md`；10-06 §1 trip25 在 `HANDOFF_2026-10-06_§1_trip25.md`；10-14 §1 trip26／27 在 `HANDOFF_2026-10-14_§1_trip26-27收工.md`。查歷史用 grep）。
- 規格／SHM／座標／IMU 慣例／原廠參數 → `docs/D1Max_機器狗資訊.md`（95 KB，查章節）。目錄地圖 → `README.md`；文件索引 → `docs/README.md`；每次上機 → `docs/現場SOP_通用_2026-08-27.md`。

---

## 1. 下次第一件事：**v1d 開訓（Colab，`notebooks/rl_wheel_v1d_colab.ipynb`）**，訓完跑及格線

- **v1c 從頭訓 5 億步（10-20 22:15）的結果：不能用。** 所有帶 vy 的指令實際 vy 0.00（完全不抬腳，步頻 0.14）、直走超速 20%、原地轉蹲到 0.35 m、斜走加轉摔；崎嶇地 114 趟 0 摔但全靠滾。
  攤帳（`outputs/reward_audit_wheel_v1c_s1.md`）：平移 0.5 的 `lin` 0.04 vs v1b 2.40、`impact` 0、`roll` −0.02 → 每步少拿 2 分只為省 0.3 分 ＝ **局部解，不是理性取捨**；判斷是 `impact`（無門檻、w 3）把早期笨拙落地（1～2 m/s、每次 −3～−12）的探索殺掉，policy 學成「永遠不抬腳」。v1 同樣 5 億步平移做到 0.48。
  表：`outputs/rl_wheel/eval_rl_only_real_{v1c_s1,diag_v1c_s1}.md`、`eval_rl_wheel_rough_real_v1c_s1.md`；權重 `weights/rl_wheel_v1c_{params.pkl,np.npz}` 留作對照。使用者的 Colab 進度列還沒拿到（要：vyerr／impact 前 1 億步走勢）。
- **v1d（使用者選 C，10-20 深夜）**：**從 v1b 熱啟動**（`INIT_FROM` 還原 normalizer／policy／value，步數從 0 算）續訓，環境同 `V1C` 但 `impact` 改門檻形式 `max(0, vz − 0.3)²`（`impact_v0`；10k 等級 0.10–0.19 m/s 免費、v1b 0.43–0.71 仍罰）。每 session 5 億，**使用者：有必要可以多跑幾個 session**。檔名／雲端硬碟資料夾全用 v1d，`RESUME="auto"` 不會撿到 v1c 的 step 檔。代價：不保證「天生對稱」，靠鏡像把左右拉齊；拉不齊再走「從頭訓＋impact 暖身」。
  門檻後攤帳（同檔名覆寫）：`impact` 10k −0.001／−0.001／−0.005、v1b −0.040／−0.240／−0.069（vy +0.5／−0.5／+0.6）；roll 不變。

- **v1c 是什麼**（spec `docs/superpowers/specs/2026-10-20-rl-wheel-v1c-design.md`、plan `docs/superpowers/plans/2026-10-20-rl-wheel-v1c.md`、reward 盤點 `docs/results/Y_v1b_reward總表_2026-10-20.md`）：
  從頭訓、`WheelEnv(**rw.V1C)`＝鏡像增強（每回合 50% obs 翻給 policy、動作翻回物理）＋ `roll`（只罰 roll，w 10）＋ `impact`（觸地前一步足端 vz²，w 3，出生後 25 步不計）＋ 動作延遲 DR 0–2 步 ＋ IMU 常數偏差 DR ±2°。
  目標只有兩個：**左右對稱**（v1b 左平移 roll 7–10°、右 1–4°）與**落地輕**。σ 0.25、`acc_leg`／`rate`、`wgap`、`lim`、`w_ang` 全不動；「從容」（步頻）這版不列目標。
  **不採用**（查證過，spec §0）：降碰撞懲罰（`wgap` 訓練全程為 0 沒輸出過梯度）、σ 放寬（買到的是欠追蹤）、加重 `dev`（罰離站姿不罰左右差）。
- **攤帳關卡（10-20 晚，`outputs/reward_audit_wheel_{10k,v1b}_v1c.md`）**：(a) `impact` 10k／v1b ＝ 0.11／0.03／0.20（vy +0.5／−0.5／+0.6）→ 過；每次落地 vz（Σimpact_vz／Σtd_n）10k 0.12／0.10／0.19、v1b 0.43／0.71／0.44 m/s。
  (b) v1b `roll` 左平移 −0.179 vs 右 −0.024 → 過。(c) v1b `roll+impact` −0.33／−0.53／−0.28，右平移那列超 −0.5 下限 7%，單項最大 0.51 < 1 → **接受，w_impact 維持 3.0**。
  順帶：v1b **右**平移落地反而比左重（0.71 vs 0.43 m/s、94 vs 77 次），roll 是左大；兩個問題在不同邊。
- **Colab 步驟**：先 `git push`（notebook 是 clone GitHub main）→ 開 `notebooks/rl_wheel_v1d_colab.ipynb`（v1c 那本留著不用），第 4 格掛載雲端硬碟（授權一次就能離開）、G4、throughput 格自動選 4096、`TIMESTEPS` 5 億／session；**每次 eval 的 step 檔與最後的 `rl_wheel_v1c_params.pkl` 自動複製到 `MyDrive/rl_wheel_v1d/`**（10-20 使用者要求），session 2 掛載後 `RESUME="auto"` 自動從雲端硬碟接；第一個 session 訓練格要印「熱啟動自 rbtdog_sim/task7/weights/rl_wheel_v1b_params.pkl」。
  進度列讀法：`roll`、`impact` 往 0；`land_vz` ＝ Σvz／Σ落地次數（MJX 接觸口徑；v1b 左／右平移 0.43／0.70），往下；**和本機驗收表「落地 vz」（CPU 淨空口徑，v1b 0.36／0.37）不同尺度、不能互比**；`mir` ≈ 0.5、`d` ≈ 1.0；eval 也開鏡像／延遲／偏差，eval reward 不是乾淨名目數字。
  停損（累計 2 億）：`len < 600`／`lin < 0.5`／gap 中位 < 40 mm／`impact`、`roll` 沒往 0 走／`level` 明顯低於 v1b 同步數（崎嶇地 impact 大：stairs 8 側移 −0.75／步、boxes 8 −0.48，平地 −0.17；最終審查補量）。**5 億步中場檢查**：step 檔拉回本機 `export_policy_wheel.py` → 跑 §6 表，roll 左右差沒縮或「落地 vz」沒比 v1b 低 → 先調權重再跑 session 2。
- **訓完及格線**（spec §6；工具同 v1b、每條 `--self-collision`；完整指令在 plan Task 11）：摔 0（平地 real／ideal／push、斜走 9 條、倒退 5 條）、崎嶇地 114 趟 ≤ 1；vy ±0.5 ≥ 0.42、wz 0.5 ≥ 24°/s；**對稱** vy ±0.5 roll 峰左右差 ≤ 3° 且都 ≤ 6°；**落地輕** 驗收表「落地 vz」欄低於 v1b（v1b 0.36／0.37）；ABAD 誤差峰 ≤ 0.70、髖膝 ≤ 0.45、gap ≥ 200 mm；延遲 1 步＋雜訊 0 摔、延遲 2 步左平移不摔。任一格不過不上機，v1b 權重不動。**跑及格線前先修 `landing_vz` 的遲滯**（5 mm 門檻無遲滯／最小騰空時間，抖動會灌水事件數、拉低平均；修完 v1b 的 `eval_rl_only_real_landcol_v1b.md` 要重跑當基準）。
  結果寫 `docs/results/Z_純RL自訓v1c_MuJoCo_<日期>.md`；過了寫 trip29 操作卡（照 trip26 卡 §8 格式）。
- **全套 pytest @98a57c3（環境改動全部完成後）：1118 passed, 1 warning（jaxopt），37 分**（10-20 晚）。
- **工具坑**：`conda run -n rbtdog python - <<EOF` 不會把 heredoc 餵進 stdin，python 片段先寫成檔案；`--cmds` 以負號開頭要寫 `--cmds=-0.3,...`。
- **給經理的訊息（10-20 下午）**寫「已開始訓練」；實際開訓要等 push 後在 Colab 按下去。

## 2. 權重現況

| 權重 | 狀態 | 驗收表（`outputs/`） |
|---|---|---|
| **rl_wheel_v1d**（待訓）`notebooks/rl_wheel_v1d_colab.ipynb` | v1b 熱啟動＋V1C（impact 門檻 0.3）；及格線見 §1 | `reward_audit_wheel_{10k,v1b}_v1c.md`（門檻版）、`rl_wheel/align_*_v1c.md` |
| rl_wheel_v1c（從頭訓 5 億，**失敗**）`weights/rl_wheel_v1c_{params.pkl,np.npz}`（sha `06d649a136c3`） | 側向軸死掉、不抬腳（§1）；留作對照 | `rl_wheel/eval_rl_only_real_{v1c_s1,diag_v1c_s1}.md`、`eval_rl_wheel_rough_real_v1c_s1.md`、`reward_audit_wheel_v1c_s1.md` |
| **rl_wheel_v1b**（自訓，MJX，10 億步）`weights/rl_wheel_v1b_params.pkl`／`rl_wheel_v1b_np.npz`（sha `a02a4631d3f3`） | ★ **已上機（trip26／27／28）**：trip26＋27 24 趟 0 摔，trip28（10-20）主管實體看步態；後輪不碰、慢轉不飄、vy 0.3 會動；`--grav-offset 0.7 -1.5` 標準旗標；弱點＝左平移 roll 7–10°、平移落地重。模擬：斜走 9 條、倒退 5 條 0 摔（`rl_wheel/eval_rl_only_real_{diag,back}_*_v1b.md`） | `rl_wheel/eval_rl_only_{real,ideal}_v1b.md`、`eval_rl_wheel_rough_real_v1b.md`、`eval_rl_only_real_landcol_v1b.md`（落地 vz 欄）；X 文件 |
| rl_wheel_v1（5 億步）`weights/rl_wheel_v1_params.pkl`／`_np.npz`（sha `83220469ef90`） | 被 v1b 取代。 | `rl_wheel/eval_rl_only_*_v1.md`；W 文件 |
| **RL_only 10k** `weights/rl_only_2026-10-01_10k_np.npz`（sha `5ec719a97175`） | ★ 已上機（trip25）11 趟 0 摔；平移後輪互碰、慢轉飄 → 被 v1b 取代；落地最輕（每次 0.10–0.19 m/s）當 `impact` 項的對照 | `rl_only/eval_rl_only_{real,ideal}_10k.md`、`eval_rl_only_rough_real_10k.md` |
| RL_only 5k／0924 | 對照組／trip24 已上機 12 趟 0 摔 | `rl_only/eval_rl_only_*_5k.md`／T 文件 |
| **v3.6f** `weights/cpg_rl_v3_6f_params.pkl`／`cpg_rl_v3_6f_np.npz`（sha `cac8679831c3`） | CPG-RL 線可部署最佳（`--mirror --mirror-lat`），trip22／23 驗過；線暫停 | `eval_cpg_rl_v3_6f_params_mirrorall.md`、Q／R 文件 |
| v3.7f／v3.5f／v3.7bf | 不用／上一代／沒訓 | — |

- 報告影片（本機，mp4 被 .gitignore）：`outputs/rl_wheel/report_v1b_mesh.mp4`、`rough_compare_10k_vs_v1b_mesh_v2.mp4`（報告用）、`diag_v1b_mesh.mp4`（斜走六段，10-20）；各有 `_small`。
- ⚠️ 2026-09-22 之前的 `outputs/eval_*.md` 都含每 2 s 的訓練推力。

## 3. 工具一行版

- **自訓環境** `inference/rl_env_wheel.py`（`WheelEnv`、`V1C` 預設字典、`make_domain_randomize`、`wrap_env`；鏡像表 `mirror_obs/act/priv/data`；`IMPACT_SETTLE` 25）；地形 `wheel_terrain.py`；對齊／攤帳 `diag/wheel_env_align.py [--preset v1c] [--npz …] --tag …` → `outputs/reward_audit_wheel_<tag>.md`（每次落地 vz ＝ 欄 `impact_vz`／`td_n`，不能只讀 `impact_vz`）；匯出 `export_policy_wheel.py --params …`；notebook 產生器 `notebooks/build_nb_wheel.py`（寫 v1 與 v1c 兩本）。
- **golden** `tests/test_rl_env_wheel_golden.py` 釘住 `WheelEnv()`（＝v1b）長回合＋短回合（重生／推力／指令切換）逐位元；失敗＝預設路徑被動到，查清楚再決定，不放寬。`tests/test_rl_env_v3_golden.py` 釘 v3.3。
- **驗收工具** `rl_only_infer.py --npz … --self-collision [--delay N] [--noise] [--push 0.5] [--grav-bias R P] [--grav-offset R P] [--video … --mesh]` → `outputs/rl_wheel/eval_rl_only_{real,ideal}_<tag>.md`（**新欄「落地 vz m/s（次）」**＝觸地前一步下落速度，`landing_vz()`）；崎嶇地 `diag/rl_only_rough_terrain.py --npz … --self-collision --tag …`（同樣多一欄）；並排影片 `diag/rl_only_rough_compare_video.py`。
- **純 RL 匯出** `inference/export_policy_rl.py`；影子跟跑 `RL_ONLY_NPZ=… pytest tests/test_policy_rl_{np,closed_loop}.py`。
- **上機** `realbot/M9_gait.py --v3 PATH.npz --interactive --cmd vx vy wz …`（rl16 強制不鏡像；`--wheel-mode kd1`、`--vmax 18`、`--grav-offset 0.7 -1.5` 必帶；READY 站好讀值差 > 1° 就改當天值）；操作卡 `docs/現場操作卡_純RL自訓v1b_trip26_2026-10-14.md`（§7 trip27、§8 trip28 展示含斜走）。
- **實機 log 分析** `inference/diag/rl16_trip_report.py`（⚠️ 抬腳欄偏置待修，X §5）；里程法只能同腿前後比（X §3／§6）。
- **CPG-RL v3 線**（暫停）`inference/local_infer_v3.py --preset … --mirror --mirror-lat`；`export_policy_v3.py`。
- **給同事的獨立 repo `~/d1max-rl-deploy`**：commit 不加 AI 署名；改過的驗收工具要同步搬。
- 全套測試 `conda run -n rbtdog python -m pytest task7/tests -q`（約 30 分）。

## 4. 未解／待決（依優先序）

0. **v1c**：Colab 開訓 → 中場檢查 → 及格線 → Z 文件 → trip29（§1）。plan 的 Minor 審查項（測試覆蓋：鏡像＋延遲同開、重生後 delay 重抽、對稱化測試的覆寫範圍、IMU 偏差 .T 方向）記在 `.superpowers/sdd/progress.md`，最終審查修正波處理；最終審查 Minor 清單在 `.superpowers/sdd/progress.md`。
0′. **模型左右不對稱（10-20 量到）**：`base_link` 質心 y +3.45 mm、RAR 膝 0.8598 vs 其他 0.8631 kg、**機身慣量主軸 iquat 有小傾斜（影響比質心偏移大）**；質心在 DR ±50 mm 內、慣量傾斜 DR 沒隨機化。訓練模型沒改。要不要把慣量姿態加進 DR → 使用者定。測試 `tests/test_rl_env_wheel_mirror.py::test_model_lr_asymmetry_is_within_dr`。
0″. `rl16_trip_report.py` 抬腳欄要修（X §5）；里程法併入。
1. **M9 硬上限誤中止**（trip25 ④）：`tau_hot` 連續 2 筆要加「相隔 ≥ 4 ms」（`M9_gait.py` 約 2075 行，`test_m9_gait.py` 要一起改）。
1′. 影子跟跑 Δact 門檻放寬到 1e-3（U §2）。
2. **CPG-RL v3 線（暫停）**：要重啟看 T／R 文件與 v3.8 草案。
3. **狗端 `~/m_logs` 沒清**（trip26／27／28 混舊檔）→ 下次上機前 `bash task7/realbot/clean_dog_logs.sh` 看一次再 `--delete`；本機 `m_logs_trip22/`、`m_logs_trip24/` 舊檔待清（09-30 指令已給）。
4. repo 根目錄 `RL_only/`（學長 onnx／pt，7.5 MB）、`1014_ref.txt`、`1014_reply.txt` 未入版控、位置不對；`RL_only/` 被工具當預設路徑引用，搬之前要改 `rl_only_infer.py`／`wheel_env_align.py`／`rl_only_rough_terrain.py` 的預設。

## 5. 硬限制與防呆（不隨版本變）

- 輪速伺服 **kd ≤ 1.0**；步態段一律 `--wheel-mode kd1`；RL 趟必帶 `--vmax 18`；ABAD 承重外張 ±4.5°。
- 誤差護欄：純 RL 10-01 版起 `--emax-abad 0.80 --emax 0.55`；v3 `0.91／0.55`。
- MJX 3.10 **無 hfield–cylinder 碰撞**（訓練輪子是膠囊；pair 圓盤在輪胎外側、碰不到實機內面互碰，靠 `wgap`）；Brax AutoReset 不重呼叫 reset（重生在 `WheelEnv.step` 內）。
- **模型左右不對稱**（§4 0′）：鏡像等變性只在對稱化副本上成立（4 步 qpos 3e-5）；hfield 接觸點數鏡像兩邊可差 1。鏡像增強把這些當雜訊。
- **診斷量口徑**：每次落地 vz ＝ Σ`impact_vz`／Σ`td_n`，除步數會被無落地的步稀釋（10-20 審查抓到）；出生後 25 步的落地不算事件。
- 「輪子的圖案」和「腿的剛度」耦合：換增益不能只換增益。停損線要信。
- 上機每次：`push_to_dog.sh` 看到「✅ 全部一致」才往下；兩個終端機，一個備 `sudo ~/estop_max.sh`；收工 `pull_from_dog.sh tripNN` 一定帶編號；吊帶 `H` < 292 mm（實務 260）；每趟跑完 `sudo kill -CONT <PID>`。
- `outputs/stale/` 不推上狗；狗上走 `/dev/shm/{joint_cmd,joint_state,imu_central}`；推論排 cpu 0–5。
- 實機 log 讀法：200 Hz `samples[].j[name] = (q, q_des, τ, v)`、`w[name] = (pos, v, effort)`；50 Hz `policy.log[].imu = [quat xyzw, gyro]`；IMU **+roll ＝ 右側低**。
- 操作卡的預期值只能來自驗過的模擬表；實機異常先問「模擬同指令＋1° 側傾會不會也這樣」。

## 6. 報告偵察線（2026-09-22 完成，不阻塞）

`docs/results/N_感測器與資源實測_2026-09-22.md`、`O_導航與AI架構_2026-09-22.md`。

## 7. 里程碑（一行版）

08-27 承重站立 → 09-02 第一次走路 → 09-09 RL v2.3 首次上機 → 09-15 v3.1 統一產生器 → 09-22 v3.6f → 09-23／29 v3.6f 上機（trip22／23）→ 09-29 主線換純 RL → 09-30 純 RL 首次上機（trip24）12 趟 0 摔 →
10-06 10k 上機（trip25）11 趟 0 摔 → 10-13 自訓 v1／v1b → 10-14 v1b 上機（trip26＋27）24 趟 0 摔、`--grav-offset` 定案 → **10-20 trip28 主管實體看步態；v1c 環境（鏡像＋roll＋impact＋延遲／IMU 偏差 DR）完成、攤帳關卡過；從頭訓 5 億步側向軸死掉 → v1d＝v1b 熱啟動＋impact 門檻，等 Colab**。
