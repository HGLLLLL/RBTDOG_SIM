# task7 — 智元 D1 Max（zsm-1w）輪足機器狗：底層控制 → CPG 步態 → CPG-RL 上機

**這是一台跟 task6 的 D1 EDU 不同的狗，SDK 完全不共用。** 開這個 task 就是為了不要把兩台的程式碼與參數混在一起。

## 三個入口

| 要做什麼 | 讀哪份 |
|---|---|
| 接手／看現況／下一步 | **[`HANDOFF.md`](HANDOFF.md)**（現況卡，≤150 行；歷史在 `docs/handoff_history/`） |
| 找某份文件 | [`docs/README.md`](docs/README.md)（索引） |
| 查狗的規格、SHM 佈局、座標與 IMU 慣例、原廠參數、實測物理量 | **[`docs/D1Max_機器狗資訊.md`](docs/D1Max_機器狗資訊.md)**（查章節） |
| 上機 | `docs/現場SOP_通用_2026-08-27.md` → 對應的 `docs/現場操作卡_*.md` |

## 現況（2026-09-22）

- 底層控制鏈：`/dev/shm/joint_cmd` 寫入（★ 實機驗證）；起身／坐下／步態全在 `realbot/M9_gait.py`。
- 實機已走過：開迴路 CPG `A_kp250_walk`（trip17）、CPG-RL v2.3（trip19，roll std −23%、pitch std −47%）。
- **現行主線是 CPG-RL v3**（統一運動學產生器仿原廠步態，`inference/rl_env_v3.py`）：
  模擬最佳權重 v3.6f（原地轉 ±84 達原廠 108%、平移 0.25、0 摔），v3.7 系列在補平移踏步。**v3 還沒上機**，狗端 M9 v3 路徑待做。
- 另一條線：報告用的實機偵察（感測器／資源／導航／AI 架構）2026-09-22 完成，`docs/results/N_*.md`、`O_*.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `realbot/` | 狗上跑的（純標準庫＋numpy）：`shm_io`（SHM 底層）、`coord`（★ 座標換算／限位／姿勢的唯一來源）、`kin`／`cpg`（IK／CPG 的 stdlib 移植）、`M0`–`M11`（風險遞增的實機模組；**`M9_gait.py` 是步態主程式**，含 `--policy`／`--teleop`）、`policy_np`／`rl_obs`（RL 推論與 obs，目前是 v2.3 的 76 維）、`teleop_cmd`、`estop_max.sh`（★ 急停）、`push_to_dog.sh`／`pull_from_dog.sh`／`clean_dog_logs.sh`、`recon*`（偵察腳本，全程唯讀） |
| `inference/` | 本機模擬與分析。**v3 線**：`rl_env_v3`（產生器＋MJX env，`W35/W36/W37/W37B` 權重集、`route_policy`、`mirror_obs`）、`local_infer_v3`（驗收＋錄影）、`diag/v37_pretest.py`、`diag/compare_evals.py`、`diag/replay_factory.py`。**v2 線（v3 仍匯入它的模組，不能刪）**：`rl_env_max`、`local_infer_max`、`obs_max`（★ obs 唯一定義）。共用：`max_model`／`cpg_max`／`leg_kin`／`cpg_walk_max`、`gait_baseline`（★ 基準步態唯一來源）、`export_policy_np`、`replay_policy`／`policy_log_summary`（實機 log 分析）、`ref_extract`／`ref_gait_analysis`（原廠錄檔→對標表）、`wheel_sysid`、`diag/`（其餘診斷腳本，見 `diag/README.md`） |
| `notebooks/` | Colab 訓練 notebook；`build_nb_v3.py` 產生 v3 系列（`cpg_rl_v3_*_colab.ipynb`），`build_nb_v2.py` 產生 v2 系列 |
| `weights/` | RL 權重 `.pkl`（brax）；`cpg_rl_max_v2_3_np.npz` 是狗上用的 numpy 版；哪顆最好看 `HANDOFF.md` §2 |
| `model/zgws/` | 官方 MJCF（已填實測輪摩擦）＋ MJX 訓練模型產生器（`scene_flat_mjx_v3f.xml` 是原廠增益版）；`SOURCE.md` 有質量分佈與致動器的坑 |
| `outputs/` | 模擬產物：驗收表 `eval_*.md`、影片 `*.mp4`（未進版控）、G0 回歸 `g0_*.txt`、攤帳 `reward_audit_*.md`、原廠對標 `ref_gait_dataset.md`／`ref_cmd_cycles.json`、偵察原始資料 `recon*_20260922/`、軌跡檔（`A_*.json` 才推上狗；`stale/` 作廢） |
| `logs/m_logs_trip*/` | 每趟實機原始 log（trip3–21；M9 json/log） |
| `reference/matrix_zgws/` | 原廠 MATRiX 設定檔原件 |
| `tests/` | `conda run -n rbtdog python -m pytest task7/tests -q`；`test_rl_env_v3_golden.py` 釘住 v3.3 逐位元不變 |
| `docs/` | `D1Max_機器狗資訊.md`、SOP／操作卡、`results/`（定案證據）、`design/`、`superpowers/specs/`（每版 spec）、`handoff_history/`、`figs/`、`archive/` |

## 從 task6 帶過來會出事的四件事

1. **增益與力矩門檻**：這台 41 kg、腿關節 150 N·m，是 D1 EDU 的五倍量級。task6 的 `kp=20/kd=0.7`、「力矩 >5 N·m 保護」全部不適用。
2. **IP `192.168.168.100`**：在 D1 EDU 是我們電腦的靜態 IP，在 D1 Max 是狗的 Orin NX；電腦端要改別的。
3. **SDK 程式碼**：`mc_sdk::` 那套跟這台的 `robot_sdk::SDKClient` 毫無關係，一行都不能重用。
4. **站姿是前後鏡像的 X 型**（`hip_stand_pos = [0.6, 0.6, −0.6, −0.6]`），D1 EDU 是四腿同號；「四條腿共用一個 `HOME3`」照抄會錯。

## 常用指令

```bash
bash task7/model/zgws/fetch_assets.sh                     # 取回 54 MB 網格（未進版控）
conda run -n rbtdog python -m pytest task7/tests -q       # 全部測試
bash task7/realbot/push_to_dog.sh                         # 推檔＋校驗
conda run -n rbtdog python task7/inference/local_infer_v3.py --help   # RL 權重驗收
conda run -n rbtdog python task7/inference/policy_log_summary.py task7/logs/m_logs_tripNN/M9_*.json
```
