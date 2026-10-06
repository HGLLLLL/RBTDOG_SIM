# task7 交接：D1 Max 現況卡

- 最後更新：**2026-10-06 收工（trip25：純 RL 第二版 10k 上機 11 趟 0 摔）**（分支 `main`）
- **這份只放「現在」，上限 150 行。** 收工時覆寫本檔，不是往上疊。
  被取代的段落搬到 `docs/handoff_history/`（2026-08-25～09-22 全文在 `HANDOFF_2026-08-25_2026-09-22_全文.md`；
  09-22 §1「Colab 跑 v3.7b」在 `HANDOFF_2026-09-22_§1_v3.7b_Colab訓練.md`。查歷史用 grep，不要整份讀）。
- 規格／SHM／座標／IMU 慣例／原廠參數 → `docs/D1Max_機器狗資訊.md`（95 KB，查章節）。目錄地圖 → `README.md`；文件索引 → `docs/README.md`；每次上機 → `docs/現場SOP_通用_2026-08-27.md`。

---

## 1. 下次第一件事：**把 V 文件 §6 的重訓清單交給同事**；下次上機再補 ⑥⑦、5k 對照、崎嶇地

- **trip25 一句話**（2026-10-06，10k）：11 趟 10 完成、0 摔、0 退回、1 趟步態前誤中止。直走 0.3 → **0.29（97%；0924 只 0.20）**；平移 ±0.5 真踏步抬 16–76 mm、自轉偏移 ≤ 1.4°/s；
  原地轉 ±1.0 → **98／99%**；**右轉 −0.5 不塌（−19.6，trip24 塌到 −7.8）**。護欄用量 ABAD 誤差 0.50／0.80、τ 峰 32、最慢 17.9 ms、溫度 ≤ 27 °C。
  → `docs/results/V_純RL第二版_首次上機_trip25_2026-10-06.md`；逐趟分析 `inference/diag/rl16_trip_report.py`。
- **三個問題**（都是 policy 性質或 M9 小修，部署堆疊沒事）：
  1. 平移後輪互碰：實體間隙 25 mm（網格算），訓練 `enabled_self_collisions: false`；模擬試過各種 ABAD 夾限都保不住步態 → **重訓開自碰撞**。vy ±0.6 因此沒跑。
  2. 慢速旋轉 ±0.5 只 68–70%、四輪不對稱→旋轉中心偏、機身飄；模擬同樣（73–78%）→ 重訓小指令追蹤（與 vy ≤ 0.3 死區同類）。
  3. ④ 第一次右轉在 `KP_DOWN` 被 br3 膝 +144／+132 兩筆（相隔 1 ms 的追趕 tick）誤中止；歷史同類單筆 4 次都沒中止 → **M9 硬上限加「兩筆相隔 ≥ 4 ms」**（安全路徑，回來改＋測試，§4.1）。
- 可展示範圍（10k）：vx 0.3、vy ±0.5（會輕碰後輪）、wz ±1.0；wz ±0.5 能轉但飄。沒跑：vx 0.5、vy ±0.6、組合、5k、崎嶇地。旗標：
  ```
  --v3 rl_only_2026-10-01_10k_np.npz --interactive --no-mirror --kp 120 --kp-abad 60 --kd 1.0 --wheel-mode kd1 --wheel-kd 1.0 \
  --vmax 18 --vcmd-max 16 --emax-abad 0.80 --emax 0.55 --tau-total 5 --wheel-vmax 30 --hold-max 15
  ```
- 模擬結論（U 文件）：平地／擾動／崎嶇地 294 趟 0 摔（0924 崎嶇地 10 摔 4 卡）；**崎嶇地直走是貼地輪行＋腿順應，不抬腳**（接觸判定；「淨空」口徑在方格邊緣會誤判，已更正）；建議 10k 為主、5k 對照。
  影片：崎嶇地並排 `outputs/rl_only/rough_compare_0924_vs_10k_mesh.mp4`（10-06 晚重做：只留差異大的 5 段 × 7 s，舊版摔 4 段；字幕改接觸判定）、`rough_compare_5k_vs_10k_mesh.mp4`（`_small` 各 ~4 MB）、平地 `flat_{10k,5k}_mesh_5s.mp4`、`rough_{0924,5k,10k}.mp4`。
- 前一趟 trip24（0924）→ T 文件；給主管的表 `outputs/report_tables/表1_*`、`表2_*`。v3.6f 線：上機驗收全部做完（R 文件）；v3.8 草案存檔。

## 2. 權重現況

| 權重 | 狀態 | 驗收表（`outputs/`） |
|---|---|---|
| **RL_only 10k** `weights/rl_only_2026-10-01_10k_np.npz`（sha `5ec719a97175`） | ★ **已上機（trip25）11 趟 0 摔**：直走 0.29、平移真踏步、轉 ±1.0 98–99%、右轉 −0.5 不塌；平移後輪互碰、慢轉飄 → 重訓。 | `rl_only/eval_rl_only_{real,ideal}_10k.md`、`eval_rl_only_rough_real_10k.md`、`logs_1006/` |
| RL_only 5k `weights/rl_only_2026-10-01_5k_np.npz`（sha `eb4cff3750ec`） | 對照組，**還沒上機**；模擬同 10k 但低速超速、平移後輪間距較寬（124–156 mm）。 | 同上 `_5k` |
| RL_only 0924 `weights/rl_only_2026-09-24_np.npz`（sha `45327a7285e4`） | **已上機（trip24）12 趟 0 摔**；崎嶇地模擬會摔，右轉 −0.5 實機塌。 | `rl_only/eval_rl_only_{real,real_vy_sweep,real_wz_sweep}.md`、T 文件 |
| **v3.6f** `weights/cpg_rl_v3_6f_params.pkl`／狗端 `cpg_rl_v3_6f_np.npz`（sha `cac8679831c3`） | CPG-RL 線可部署最佳，配 `--mirror --mirror-lat`；trip22／23 全段驗過。弱點：平移抬腳 5–10 mm。 | `eval_cpg_rl_v3_6f_params_mirrorall.md`、Q／R 文件 |
| v3.7f `weights/cpg_rl_v3_7f_params.pkl` | 平移真踏步（`--preset v37c`）但原地轉只 ±58 → 使用者決定不用。 | `eval_cpg_rl_v3_7f_params_*.md` |
| v3.5f／v3.7bf | 上一代基準／還沒訓。 | `eval_cpg_rl_v3_5f_params.md`／— |

- 報告影片：純 RL `outputs/report_rl_only_5s.mp4`（0924）；v3.6f `outputs/report_v36f_mirror_5s.mp4`。
- ⚠️ 2026-09-22 之前的 `outputs/eval_*.md` 都含每 2 s 的訓練推力；v3.5f 起已重跑無推力版。

## 3. 工具一行版

- **純 RL 匯出** `inference/export_policy_rl.py --onnx … --out task7/weights/…_np.npz`（`--out` 無聲覆寫）。
- **純 RL 平地重現** `inference/rl_only_infer.py --onnx … --cmds "vx,vy,wz;…" --tag X [--ideal] [--delay N] [--noise] [--push 0.5] [--video …]` → `outputs/rl_only/eval_rl_only_{real,ideal}_X.md`。
- **純 RL 崎嶇地重現** `inference/diag/rl_only_rough_terrain.py --onnx … --tag X [--terrains "rough:0.04,boxes:0.05"] [--seeds 3] [--video …]`
  → `eval_rl_only_rough_real_X.md`；場景 `model/zgws/scene_rough_mjx_v3f.xml`（腳本產生）；抬腳口徑＝對地淨空。
- **純 RL 範圍／側傾掃描** `inference/diag/rl_only_range_sweep.py <onnx>`、`rl_only_wz_tilt_sweep.py <onnx>`（印到 stdout）。
- **純 RL 影子跟跑** `RL_ONLY_ONNX=… RL_ONLY_NPZ=… pytest task7/tests/test_policy_rl_{np,closed_loop}.py`（不帶環境變數＝0924）。
- **上機** `realbot/M9_gait.py --v3 PATH.npz --interactive …`（rl16 版面自動選 `gen_direct`／`rl_obs_rl`、強制不鏡像）；`--wheel-mode kd1` 一定要帶。
- **實機 log 分析** `inference/diag/m9_v3_trip_report.py [目錄stamp:名稱 …]`（FK 足端離地、偏航率、ABAD 用量、時序；rl16 log 的 FK／偏航部分通用）。
- **CPG-RL v3 線** `inference/local_infer_v3.py --preset … --mirror --mirror-lat`；`export_policy_v3.py`；影子 `tests/test_policy_v3_closed_loop.py`。
- **給同事的獨立 repo `~/d1max-rl-deploy`**（2026-09-30）：純 RL 上機線；**commit 不加 AI 署名**；這次改的 `tests/test_policy_rl_*.py`（環境變數）、`diag/rl_only_rough_terrain.py`、兩個掃描腳本、`push_to_dog.sh` 要同步搬過去（那邊沒有 `task7/` 這層）。
- 全套測試 `conda run -n rbtdog python -m pytest task7/tests -q`；`test_rl_env_v3_golden.py` 釘住 v3.3 逐位元不變。

## 4. 未解／待決（依優先序）

0. **純 RL 線重訓清單（V §6）**：自碰撞（平移後輪互碰）、小指令追蹤（vy ≤ 0.3 死區、wz 0.5 只 70% 且飄）、輪摩擦補償。右轉止點餘裕已解（實機驗過）。
   站立／輪行時腿小幅調整的代價實測：溫度最高 27 °C、τ 峰 28，沒事。
1. **M9 硬上限誤中止**（trip25 ④）：`tau_hot` 連續 2 筆要加「相隔 ≥ 4 ms」（追趕 tick 不計；程式在 `M9_gait.py` 約 2075 行，`test_m9_gait.py` 有釘住快速路徑字面的測試要一起改）。
1′. **影子跟跑 Δact 門檻**放寬到 1e-3（新 policy 對 `last_action` 敏感、兩端各餵自己的動作沿軌跡放大；感測段 Δobs 仍 0）—— 理由在測試註解與 U §2。
2. **CPG-RL v3 線（暫停）**：低速原地轉拖地＝抬高隨 |wz|/1.3 縮（R §2）、平移側滑、平移 0.20 護欄不夠、ABAD 貼止點、τ 逼近 TMAX（不動 TMAX）、
   「夾抬高只在平移族」沒寫入（使用者中止）、v3.8 草案 `docs/superpowers/specs/2026-09-29-cpg-rl-v3.8-turn-decouple-design.md`。要重啟看 T／R 文件。
3. **log 目錄待清**（2026-09-30 指令已給使用者；10-06 收工已把 trip25 裡 9 個與 trip24 相同的 0930 檔刪掉、**8 趟 09-30 16:27–16:45 的 M9（trip24 當天下午，沒分析過）搬進 trip24**）：`m_logs_trip22/` 67 個 09-09 舊檔；`m_logs_trip24/` 混進 09-09／09-29 舊檔，其中 8 個是 trip23 沒拉到的
   （`M9_20260929_12{0837,0936,1135,1238}` 等，**沒分析過**）→ 搬 trip23 再刪其餘；狗端 `~/m_logs` 用 `clean_dog_logs.sh` 清。
4. M9 兩個小修（整步超時 20 ms、同號孤立尖峰）已上機（trip24）；`Gen.step` 回傳的 `c` 是活物件、`rl_obs_v3.read_frame` 本機沒測試（實機已驗）。

## 5. 硬限制與防呆（不隨版本變）

- 輪速伺服 **kd ≤ 1.0**（kd ≥ 2 driver 迴路自激）；步態段一律 `--wheel-mode kd1`（tau 模式腳離地時 50 Hz 外環發散，trip22）。
- RL 趟必帶 `--vmax 18`；ABAD 承重外張 ±4.5°；kp250 撓度 15 mm、kp120 相對命令再下沉 ≈ 30 mm。
- 誤差護欄：舊路徑 ABAD 0.6／髖膝 0.45；v3 `--emax-abad 0.91 --emax 0.55`；純 RL 0924 `0.77／0.46`、**10-01 版 `0.80／0.55`**（模擬擾動峰 × 1.3）。
- 「輪子的圖案」和「腿的剛度」耦合：**換增益不能只換增益**。停損線要信（v3.6f 沒停、事後證明該停）。
- 上機每次：`bash task7/realbot/push_to_dog.sh` 看到「✅ 全部一致」才往下；兩個終端機，一個備 `sudo ~/estop_max.sh`；收工 `pull_from_dog.sh tripNN` **一定帶編號**。
- 吊帶高度 `H` < **292 mm**（實務 260）；每趟跑完立刻 `sudo kill -CONT <PID>` 解凍；程式整個死掉那條路軟體救不了，吊帶是唯一安全網。
- `outputs/stale/` 是作廢軌跡，不要推上狗；`push_to_dog.sh` 只推 `outputs/[AB]_*.json`。狗上走 `/dev/shm/{joint_cmd,joint_state,imu_central}`，不碰 eCAL；推論排 cpu 0–5。
- 實機 log 讀法：200 Hz `samples[].j[name] = (q, q_des, τ, v)`、`w[name] = (pos, v, effort)`；50 Hz `policy.log[].imu = [quat xyzw, gyro]`；IMU **+roll ＝ 右側低**。
- 操作卡的預期值只能來自驗過的模擬表；實機異常先問「模擬同指令＋1° 側傾會不會也這樣」再懷疑堆疊（trip24 ④′ 就是這樣找到根因）。

## 6. 報告偵察線（2026-09-22 完成，不阻塞）

`docs/results/N_感測器與資源實測_2026-09-22.md`、`O_導航與AI架構_2026-09-22.md`；規格已整進 `D1Max_機器狗資訊.md` §1.6、§3.2b、§4.1b。

## 7. 里程碑（一行版）

08-27 承重站立 → 09-02 第一次走路 → 09-09 RL v2.3 首次上機（trip19）、原廠參考錄製 → 09-15 v3.1 統一產生器 → 09-22 v3.6f（可部署最佳）→
09-23／09-29 v3.6f 上機驗收完（trip22／23）→ 09-29 主線換同事純 RL（RL_only）→ **09-30 純 RL 首次上機（trip24）12 趟 0 摔** →
10-06 純 RL 第二版（崎嶇地形 10k／5k）MuJoCo 重現 0 摔 → **同日 trip25 上機 11 趟 0 摔：直走 0.29、平移真踏步、轉 ±1.0 98–99%、右轉 −0.5 不塌；平移後輪互碰、慢轉飄待重訓**。
各版為什麼改：`docs/superpowers/specs/2026-09-*-cpg-rl-v3.*.md`；純 RL 線：S／T／U 文件。
