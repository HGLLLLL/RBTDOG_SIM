# task7 交接：D1 Max 現況卡

- 最後更新：**2026-09-23 上機後收工**（分支 `main`，全部已 commit）
- **這份只放「現在」，上限 150 行。** 收工時覆寫本檔，不是往上疊。
  被取代的段落搬到 `docs/handoff_history/`（2026-08-25～09-22 全文 91 KB 在
  `HANDOFF_2026-08-25_2026-09-22_全文.md`；09-22 的 §1「Colab 跑 v3.7b」原文在
  `HANDOFF_2026-09-22_§1_v3.7b_Colab訓練.md`。要查歷史脈絡用 grep，不要整份讀）。
- 規格／SHM／座標／IMU 慣例／原廠參數 → `docs/D1Max_機器狗資訊.md`（95 KB，查章節，不整份讀）。
- 目錄地圖 → `README.md`；文件索引 → `docs/README.md`；每次上機 → `docs/現場SOP_通用_2026-08-27.md`。

---

## 1. 下次第一件事（使用者定：下週）：v3.6f 上機第二天 —— 段 ⑦ 放大（原地轉 1.3 驗抬腳）＋ 決定平移要不要留

- 2026-09-23 trip22 **⓪①②④⑤⑥ 全過、③ 跳過、⑦ 沒跑**：12 趟 0 摔 0 急停；原地轉 ±42–46°/s（指令 45.8）；直走 0.29 m/s；
  平移有側移但自轉 ±3–4.5°/s、ABAD 貼止點。結果表 `docs/results/Q_v3.6f首次上機_2026-09-23.md`，操作卡 `docs/現場操作卡_RL_v3.6f_2026-09-23.md`。
- 上機前一樣 `bash task7/realbot/push_to_dog.sh` 看到「✅ 全部一致」；狗上 T1 先定義共同旗標 `C`（下面），每段乾跑再 `--confirm`。
- 段 ⑦：`--cmd 0 0 1.3`／`0 0 -1.3`（模擬膝峰 43、側傾峰 3.9°）與 `0 ±0.15 0`，`--walk-max 10 --hold-max 12`，吊帶必掛。
- 錄影用 ④⑤⑥ 那組指令（`--walk-max 10 --hold-max 12`，實際照指令動 9 s）。

共同旗標（增益 npz 會「說兩次」，不要自己改）：

```
--v3 cpg_rl_v3_6f_np.npz --interactive --kp 120 --kp-abad 60 --kd 1.0 --wheel-mode kd1 --wheel-kd 1.0 \
--vmax 18 --vcmd-max 16 --emax-abad 0.91 --emax 0.55 --tau-total 5 --wheel-vmax 30 --hold-max 15
```

`--wheel-vmax 30`（v3 專用）＝四輪實測 |velocity| 超過就中止：輪子走 effort 力矩欄沒有速度回授，
單向失控 ChatterWatch 抓不到（那條只認正負翻轉）。effort 本身另外夾在 ±33 N·m（npz 從場景 forcerange 帶上來）。

執行紀錄與護欄怎麼定的：spec `docs/superpowers/specs/2026-09-22-cpg-rl-v3.6f-on-dog-design.md` §6。

## 2. 權重現況

| 權重 | 狀態 | 驗收表（`outputs/`） |
|---|---|---|
| **v3.6f** `weights/cpg_rl_v3_6f_params.pkl` | ★ **目前可部署最佳**，配 `--mirror --mirror-lat`：左右轉 +83.4／−84.3（原廠 108%）、左右平移 0.252／−0.250、0 摔。弱點：平移抬腳只 5–10 mm、主動側 0.3–1.0 步/秒（原廠 2.1）。右平移鏡像代價：側傾 0.37→0.63°。 | `eval_cpg_rl_v3_6f_params_mirrorall.md` |
| **v3.6f 狗端** `weights/cpg_rl_v3_6f_np.npz` | ★ 上機用這個。`export_policy_v3.py` 產：MLP（88→256→256→128→48）＋產生器參數＋週期表＋鏡像表，**狗端零常數**。sha `cac8679831c3`，layout `v3`，obs 88 act 24。`push_to_dog.sh` 已含。 | — |
| v3.7f `weights/cpg_rl_v3_7f_params.pkl` | 平移用 `--preset v37c`（只夾抬高）：2.3–2.4 步/秒＝原廠頻率、抬 36–40 mm、vy 109%、0 摔，**第一次真踏步**。但原地轉只 ±58（設定沒改＝趟間變異）→ **使用者決定不用**。訓練時 REF 無輪前饋，驗它要 `--preset v37`。 | `eval_cpg_rl_v3_7f_params_{v37,v37c,v37d}_lat.md`、`..._v37c_mirror.md` |
| v3.5f `weights/cpg_rl_v3_5f_params.pkl` | 上一代基準（左轉 +80.8、右轉 −64、ABAD 漂 6–8°）。 | `eval_cpg_rl_v3_5f_params.md` |
| v3.7bf | 還沒訓（見 §4 後續）。 | — |

- 報告影片（給主管）：`outputs/report_v36f_mirror_5s.mp4`（1.1 MB 版 `_small`）＝ v3.6f 全鏡像、每段 5 s。
- ⚠️ 2026-09-22 之前的所有 `outputs/eval_*.md` 都含每 2 s 的訓練推力（直走 roll std 0.36 其實是 0.06）；v3.5f 起已重跑無推力版。

## 3. 工具一行版

- **驗收（模擬）** `inference/local_infer_v3.py`：`--preset {v33,v35,v36,v37,v37b,v37c,v37d}`、`--gains factory`、
  `--mirror`（原地轉鏡像）、`--mirror-lat`（平移也鏡像）、`--turn-weights`（|wz|>0.1 換權重）、`--tag`、`--push`（預設關）、`--video-*`。
  表的欄位：步/秒、每週期抬腳頂點、vx 漂、航向、ABAD 漂、摔。對標原廠：`outputs/ref_gait_dataset.md` 末段。
- **匯出狗端 npz** `inference/export_policy_v3.py`：`--params …_params.pkl --out task7/weights/…_np.npz`（旗標是 `--params`，不是 `--weights`）；
  權重＋產生器快照＋週期表＋鏡像表打包成一個 npz（狗端不留任何常數；`--out` 會無聲覆寫，注意）。
- **影子跟跑** `tests/test_policy_v3_closed_loop.py`：MuJoCo 內 jax 堆疊驅動、狗端 numpy 堆疊跟跑，
  五指令各 300 步比 obs(88)／動作(24)／q_des(12)／輪 τ(4)，並印護欄峰值。5 passed、約 40 s。
- **上機** `realbot/M9_gait.py --v3 PATH.npz`：必須搭 `--interactive`；與 `--traj`／`--policy`／`--teleop` 互斥。
  `--cmd VX VY WZ`（整趟固定）、`--policy-gain 0|…|1`（0 ＝零動作名目＝退路）、`--mirror`／`--no-mirror`（預設開）、
  `--emax-abad`（ABAD）／`--emax`（髖膝）分關節護欄（只在 `--v3` 下分家）。
  輪子在步態三段：`--wheel-mode kd1`（velocity=名目、kd 1.0 讓驅動器 1 kHz 做外環、effort 只剩前饋；tau 模式的 50 Hz 外環在腳離地時發散，trip22 實測）；其餘階段純阻尼 kd 0.5。
  log 每步：`obs`／`a`／`a_raw`／`q_des`／`tau`／`v_wheel`／`right`／`cmd`／**`sat`（這一步被限位夾掉幾個關節）**／原始 imu／ms／退回原因，
  整趟另記 `n_clamp`（IK 縮限）／**`n_sat`（至少一顆頂住限位的步數，平移趟本來就大）**／`fallback_total`／`open_loop`。

- **給外部環境重訓的模型包**：2026-09-23 打包交付後已刪（本地不留）。訓練用的是 `scene_flat_mjx_v3f.xml`，不是 `max.urdf`；要再打一包照 `docs/results/P_模型打包給外部重訓_2026-09-23.md` §3。

## 4. 未解／待決（依優先序）

0. **0.8 原地轉與 0.12 平移幾乎拖地**：命令抬高 40–60 mm 只追到 25 mm、機身又下沉 25 mm → 離地 0–8 mm（結果文件 Q §4.0）。根因是旋轉族抬高隨 |wz|/1.3 縮；段 ⑦ 1.3 先驗，低速要抬得起來得解耦重訓。
1. **平移自轉偏移**（實機 ±3–4.5°/s，左右反向同大 → 產生器側向輪子圖案的偏航力矩）與側滑（抬 4–10 mm）：v3.7 線的事（真踏步、輪前饋）。
   `fl1_hip_roll` effort 讀值毛刺（kp 250 站立段每趟 4–5 筆孤立 −50…−121）：v3 慢速累計已改只算步態段；根治要讓 `commit_peak` 剔同號孤立尖峰。
   停走時輪子一步煞停有背隙聲：淡出段把 `v_nom` 斜降。ABAD 貼止點的發熱沒量（每趟 ≤ 10 s）。
2. **平移 0.20 在現行護欄下不可部署**：名目最大命令速度 15.84 rad/s，M9 要求 `--vmax ≥ 1.2×` → 要開到 19，
   已是腿關節規格上限 19.9 rad/s 的 95%。所以段 ⑦ 的平移放大只到 **0.15**（名目 14.67）。
   要跑 0.20 得先另外決定護欄，不是現場臨時放寬。
3. **平移名目騎在 ABAD 止點上**：平移指令下 `fl1/bl1_hip_roll`（鏡像＝`fr1/br1`）的命令有 35%／27% 的週期（除以**指令有效步**；
   M9 乾跑印的分母含前 1 s 站姿全步數 → 31%／25%，整趟約 56%）
   釘在 −0.523 rad 機構止點，實際關節只到 ∓0.15–0.19 → 命令-實際誤差 0.70 rad（`--emax-abad 0.91` 就是這樣來的）。
   訓練時就這樣（ABAD 幅度＝原廠 ×1.6 補 kp120 撓度），v3 乾跑限位檢查已改成比產生器自己的 LO12/HI12。
   **實機沒驗過持續貼止點會不會發熱／飽和** → 段 ③ 盯這兩顆的力矩。
4. **腿力矩逼近 `TMAX`**：模擬腿 τ 峰 42–48 N·m，M9 `TMAX` ABAD/HIP 50、KNEE 70，
   快速中止（連續 `--tau-hits` 3 步超過）在實機有機會跳。**`TMAX` 不動**；誤中止時調 `--tau-hits`／`--tau-total`。
5. **平移品質未解**：右平移航向 +42°、vx 漂 0.03–0.06、低速（0.12）仍偏滑。
6. **「夾抬高只在平移族生效」沒寫入**（使用者中止）。要做：`rl_env_v3.step` 的 `LIFT_NONNEG` 段乘 `_lat_weight(P["A"]) > 0.5`。
   背景：`--turn-weights` 用 v3.7f（平移）＋v3.6f（旋轉）跑過一次，旋轉只 +75／−77，因為全域夾抬高把 v3.6f 旋轉族的 lift 也夾了。
7. **上機前要知道的四個小坑**（v3 線的 Minor，其餘 triage 清單在 `.superpowers/sdd/progress.md`）：
   - `python3 M9_gait.py --help` **已修好**（`--vmax` 的 help 裸 `%` → `%%`，離開碼 0）；之前會 `ValueError: incomplete format`。
   - `realbot/rl_obs_v3.read_frame`（真 shm 讀取）**本機沒有測試**，段 ①（`--confirm` 的第一趟）是它第一次被真的跑到 ——
     所以段 ① 的過關條件多一條：收工 JSON `policy.fallback_total == 0`、`open_loop == false`。
   - `v3_wheel_rows` 的 `velocity` 欄在真路徑上幾乎是死的（`velocity=v_meas` 讓 kd 項 ≈ 0，力矩全靠 `effort`）——這是設計，但別靠它調速。
     **寫側（`effort` 欄）的符號實機沒驗過**（讀側由影子跟跑釘住）：段 ② 要看四輪 `velocity` 與 `effort` 的符號，接反就是往後走。
   - `Gen.step` 回傳的 `c` 是活物件（沒有 copy），外部要留用得自己 `np.array(...)`；`gen.cmd_prev` 已改成複本，不用擔心。
8. 後續（RL 訓練線，不阻塞上機）：Colab 跑 `notebooks/cpg_rl_v3_7bf_colab.ipynb`（已 push，v3.7f ＋輪前饋＋鏡像增強）
   —— 原文與停損條件見 `docs/handoff_history/HANDOFF_2026-09-22_§1_v3.7b_Colab訓練.md`。
9. 續訓：notebook 有 `RESUME`（`restore_params`），只做過原始碼比對，本機 jax 0.10 跑不動 `ppo.train`。

## 5. 硬限制與防呆（不隨版本變）

- 輪速伺服 **kd ≤ 1.0**（kd ≥ 2 driver 迴路自激，F 文件）；實機控制律 kd 1.0 ＋ 前饋 0.13（K 文件）。
- RL 趟必帶 `--vmax 18`；ABAD 承重外張 ±4.5°；kp250 撓度 36 mm（抬腿要加 `Z_SAG`），**換到 kp120 機身會再下沉 ≈ 36 mm**。
- M9 誤差護欄：舊路徑（`--traj`／`--policy`）ABAD 0.6／髖膝 0.45；**v3 走 `--emax-abad` 0.91／`--emax` 0.55**（見 §1）。
- 「輪子的圖案」和「腿的剛度」耦合：**換增益不能只換增益**（`lat_omega` 3.3 是 kp250 下擬的，軟增益要 2.2）。
- 停損線要信：v3.6f 1 億步 `step` 0.25 沒停跑完，事後證明抬腳確實沒學起來。
- 上機每次：`bash task7/realbot/push_to_dog.sh` 看到「✅ 全部一致」才往下；兩個終端機，一個備 `sudo ~/estop_max.sh`。
- 吊帶高度 `H` 必須 < **292 mm**（實務 260），掛太高整趟變成「空中抬腿」而畫面每欄都漂亮。
- 每趟跑完立刻 `sudo kill -CONT <PID>` 解凍；程式整個死掉那條路軟體救不了，吊帶是唯一安全網。
- `outputs/stale/` 是作廢軌跡，不要推上狗；`push_to_dog.sh` 只推 `outputs/[AB]_*.json`。
- 狗上：我們走 `/dev/shm/{joint_cmd,joint_state,imu_central}`，不碰 eCAL；推論排 cpu 0–5（cpu7 是 `robot_hal_node/HwLoop`、cpu6 UART 中斷）；RK 閒、NX 滿，東西塞 RK；NPU 三核全閒。
- 改 `rl_env_v3` 後跑 `conda run -n rbtdog python -m pytest task7/tests -q`；`test_rl_env_v3_golden.py` 釘住 v3.3 逐位元不變。

## 6. 報告偵察線（2026-09-22 完成，不阻塞 RL 線）

- 結果：`docs/results/N_感測器與資源實測_2026-09-22.md`（1.3／7.1／7.2／7.4）、`docs/results/O_導航與AI架構_2026-09-22.md`（5.x／6.x）。
  規格已整進 `D1Max_機器狗資訊.md` §1.6、§3.2b、§4.1b（eCAL 三層）。原始資料 `outputs/recon{3,3b,4,5}_20260922/`。
- 沒解、不阻塞：超音波量程（驅動沒填）、UWB 要基站、RTK 要室外、四個 0 發布者 topic、光達自己的 IP。
- 報告項目清單截圖：`docs/figs/報告項目清單_截圖_2026-09-22.jpg`。

## 7. 里程碑（一行版）

08-27 承重站立 → 09-02 第一次走路（kp120）→ 09-03 LS+kp250 零中止 → 09-09 RL v2.3 首次上機（trip19）、原廠參考錄製（trip21）、M11 輪辨識 →
09-15 v3.1 統一運動學產生器 → 09-16 v3.3／v3.4f／v3.5f → 09-22 v3.6f（可部署最佳）、v3.7f、報告偵察完成 →
09-23 v3.6f 上機路徑做完（匯出／產生器／obs／M9 `--v3`／影子跟跑，等上機）。
各版為什麼改：`docs/superpowers/specs/2026-09-*-cpg-rl-v3.*.md`（每份 §6 或 §8/§9 是執行紀錄）。
