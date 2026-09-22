# task7 交接：D1 Max 現況卡

- 最後更新：**2026-09-22 收工**（分支 `main`，全部已 commit）
- **這份只放「現在」，上限 150 行。** 收工時覆寫本檔，不是往上疊。
  被取代的段落搬到 `docs/handoff_history/`（2026-08-25～09-22 的舊 HANDOFF 全文 91 KB 在
  `docs/handoff_history/HANDOFF_2026-08-25_2026-09-22_全文.md`，要查歷史脈絡用 grep，不要整份讀）。
- 規格／SHM／座標／IMU 慣例／原廠參數 → `docs/D1Max_機器狗資訊.md`（95 KB，查章節，不整份讀）。
- 目錄地圖 → `README.md`；文件索引 → `docs/README.md`；每次上機 → `docs/現場SOP_通用_2026-08-27.md`。

---

## 1. 明天第一件事：Colab 跑 `notebooks/cpg_rl_v3_7bf_colab.ipynb`（已 push）

- v3.7b ＝ v3.7f ＋ 輪前饋 `CYC_LAT_FF` ＋ `lift` 只能加 `LIFT_NONNEG` ＋ `W_STEP` 2.5 ＋ **鏡像增強 `MIRROR_AUG`**（env 層 50% 回合 obs／動作鏡像，policy 本身左右對稱）。
- 停損：1 億步時 進度 yaw < 3.5、len < 600、`step` < 0.45 任一 → 停。
- 訓完驗收：
  ```bash
  conda run -n rbtdog python task7/inference/local_infer_v3.py --preset v37b --gains factory --mirror \
    --weights task7/weights/cpg_rl_v3_7bf_params.pkl --seeds 3 --mesh --video task7/outputs/eval_v37bf_mesh.mp4
  ```
  `--preset v37b` 驗收時自動關 MIRROR_AUG。比較表：`inference/diag/compare_evals.py`。
- 若旋轉又退步：先查 `CMD_VY` 上限 0.22／解耦讓平移族樣本好學、旋轉族相對吃虧（對比 v3.6f 進度 yaw 曲線 2.5）。
- spec：`docs/superpowers/specs/2026-09-22-cpg-rl-v3.7-decouple-mirror-design.md`（§6 v3.7f 訓練曲線分析）；證據 `outputs/v37_pretest.md`（E1–E16）、攤帳 `outputs/reward_audit_v37b.md`。

## 2. 權重現況

| 權重 | 狀態 | 驗收表（`outputs/`） |
|---|---|---|
| **v3.6f** `weights/cpg_rl_v3_6f_params.pkl` | ★ **目前可部署最佳**，配 `--mirror --mirror-lat`：左右轉 +83.4／−84.3（原廠 108%）、左右平移 0.252／−0.250、0 摔。弱點：平移抬腳只 5–10 mm、主動側 0.3–1.0 步/秒（原廠 2.1）。右平移鏡像代價：側傾 0.37→0.63°。 | `eval_cpg_rl_v3_6f_params_mirrorall.md` |
| v3.7f `weights/cpg_rl_v3_7f_params.pkl` | 平移用 `--preset v37c`（只夾抬高）：2.3–2.4 步/秒＝原廠頻率、抬 36–40 mm、vy 109%、0 摔，**第一次真踏步**。但原地轉只 ±58（設定沒改＝趟間變異）→ **使用者決定不用**。訓練時 REF 無輪前饋，驗它要 `--preset v37`。 | `eval_cpg_rl_v3_7f_params_{v37,v37c,v37d}_lat.md`、`..._v37c_mirror.md` |
| v3.5f `weights/cpg_rl_v3_5f_params.pkl` | 上一代基準（左轉 +80.8、右轉 −64、ABAD 漂 6–8°）。 | `eval_cpg_rl_v3_5f_params.md` |
| v3.7bf | 明天訓。 | — |

- 報告影片（給主管）：`outputs/report_v36f_mirror_5s.mp4`（1.1 MB 版 `_small`）＝ v3.6f 全鏡像、每段 5 s。
- ⚠️ 2026-09-22 之前的所有 `outputs/eval_*.md` 都含每 2 s 的訓練推力（直走 roll std 0.36 其實是 0.06）；v3.5f 起已重跑無推力版。

## 3. 驗收工具 `inference/local_infer_v3.py`

`--preset {v33,v35,v36,v37,v37b,v37c,v37d}`、`--gains factory`、`--mirror`（原地轉鏡像推論）、`--mirror-lat`（平移也鏡像）、
`--turn-weights`（|wz|>0.1 換另一顆權重，`rl_env_v3.route_policy`）、`--tag`、`--push`（預設關）、`--video-secs/--video-start/--video-skip`。
表的欄位：步/秒、每週期抬腳頂點、vx 漂、航向、ABAD 漂、摔（兩組）。對標原廠：`outputs/ref_gait_dataset.md` 末段。

## 4. 未解／待決（依優先序）

1. **平移未解**：右平移航向 +42°、vx 漂 0.03–0.06、低速（0.12）仍偏滑。v3.7b 的前饋與鏡像增強是對這三項下的。
2. **「夾抬高只在平移族生效」沒寫入**（使用者中止）。要做：`rl_env_v3.step` 的 `LIFT_NONNEG` 段乘 `_lat_weight(P["A"]) > 0.5`。
   背景：`--turn-weights` 用 v3.7f（平移）＋v3.6f（旋轉）跑過一次，旋轉只 +75／−77，因為全域夾抬高把 v3.6f 旋轉族的 lift 也夾了。
3. **上機 M9 v3 路徑還沒做**（另開 spec；v3.4f 起增益與 v2.3 的 SOP 不同，不能沿用）：
   `rl_obs` 改 88 維（四輪速、cmd 3 維、head_err 積分）、v3 產生器搬 numpy、政策匯出 88→24、影子跟跑釘住、
   `mirror_obs` 的 numpy 版、實機腿增益 250→120、M9 護欄 ABAD 放寬到 0.6、輪 kd 0.5→0.1、`--tau-total` 慢速路徑開。
4. 續訓：notebook 有 `RESUME`（`restore_params`），只做過原始碼比對，本機 jax 0.10 跑不動 `ppo.train`。

## 5. 硬限制與防呆（不隨版本變）

- 輪速伺服 **kd ≤ 1.0**（kd ≥ 2 driver 迴路自激，F 文件）；實機控制律 kd 1.0 ＋ 前饋 0.13（K 文件）。
- RL 趟必帶 `--vmax 18`；ABAD 承重外張 ±4.5°；M9 誤差護欄 ABAD 0.6／髖膝 0.45 rad；kp250 撓度 36 mm（抬腿要加 `Z_SAG`）。
- 「輪子的圖案」和「腿的剛度」耦合：**換增益不能只換增益**（`lat_omega` 3.3 是 kp250 下擬的，軟增益要 2.2）。
- 停損線要信：v3.6f 1 億步 `step` 0.25 沒停跑完，事後證明抬腳確實沒學起來。
- 上機每次：`bash task7/realbot/push_to_dog.sh` 看到「✅ 全部一致」才往下；兩個終端機，一個備 `sudo ~/estop_max.sh`。
- 吊帶高度 `H` 必須 < **292 mm**（實務 260），掛太高整趟變成「空中抬腿」而畫面每欄都漂亮。
- 每趟跑完立刻 `sudo kill -CONT <PID>` 解凍；程式整個死掉那條路軟體救不了，吊帶是唯一安全網。
- `outputs/stale/` 是作廢軌跡，不要推上狗；`push_to_dog.sh` 只推 `outputs/[AB]_*.json`。
- 狗上：我們走 `/dev/shm/{joint_cmd,joint_state,imu_central}`，不碰 eCAL；推論排 cpu 0–5（cpu7 是 `robot_hal_node/HwLoop`、cpu6 UART 中斷）；RK 閒、NX 滿，東西塞 RK；NPU 三核全閒。

## 6. 報告偵察線（2026-09-22 完成，不阻塞 RL 線）

- 結果：`docs/results/N_感測器與資源實測_2026-09-22.md`（1.3／7.1／7.2／7.4）、`docs/results/O_導航與AI架構_2026-09-22.md`（5.x／6.x）。
  規格已整進 `D1Max_機器狗資訊.md` §1.6、§3.2b、§4.1b（eCAL 三層）。原始資料 `outputs/recon{3,3b,4,5}_20260922/`。
- 沒解、不阻塞：超音波量程（驅動沒填）、UWB 要基站、RTK 要室外、四個 0 發布者 topic、光達自己的 IP。
- 報告項目清單截圖：`docs/figs/報告項目清單_截圖_2026-09-22.jpg`。

## 7. 里程碑（一行版）

08-27 承重站立 → 09-02 第一次走路（kp120）→ 09-03 LS+kp250 零中止 → 09-09 RL v2.3 首次上機（trip19）、原廠參考錄製（trip21）、M11 輪辨識 →
09-15 v3.1 統一運動學產生器 → 09-16 v3.3／v3.4f／v3.5f → 09-22 v3.6f（可部署最佳）、v3.7f、報告偵察完成。
各版為什麼改：`docs/superpowers/specs/2026-09-*-cpg-rl-v3.*.md`（每份 §8 或 §9 是執行紀錄）。
