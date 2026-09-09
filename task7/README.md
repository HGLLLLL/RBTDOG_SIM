# task7 — 智元 D1 Max（zsm-1w）輪足機器狗：底層控制 → CPG 步態 → CPG-RL 上機

**這是一台跟 task6 的 D1 EDU 不同的狗，SDK 完全不共用。** 開這個 task 就是為了不要把兩台的程式碼與參數混在一起。

## 三個入口

| 要做什麼 | 讀哪份 |
|---|---|
| 接手／看現況／下一步 | **[`HANDOFF.md`](HANDOFF.md)**（最上面那節最新） |
| 找某份文件 | [`docs/README.md`](docs/README.md)（索引） |
| 查狗的規格、SHM 佈局、座標與 IMU 慣例、原廠參數、實測物理量 | **[`docs/D1Max_機器狗資訊.md`](docs/D1Max_機器狗資訊.md)** |
| 上機 | `docs/現場SOP_通用_2026-08-27.md` → 對應的 `docs/現場操作卡_*.md` |

## 現況（2026-09-09）

- 底層控制鏈：`/dev/shm/joint_cmd` 寫入（★ 實機驗證）；起身／坐下／步態全在 `realbot/M9_gait.py`。
- 開迴路 CPG 步態 `A_kp250_walk`（LS 序列、kp 250／ABAD 60／kd 2／輪 kd 0.5）：trip17 實機零中止走通。
- **CPG-RL v2.3 已上實機**（trip19）：四趟乾淨、鍵盤遙控 7.6 s；roll std −23%、pitch std −47%。
  結果 `docs/I_*.md`，sim2real 總結 `docs/J_*.md`。
- 下一步：v2.4（航向誤差進 obs）＋ 靜態側傾隨機化／偏置罰 → 重訓 → 上機。

## 目錄

| 路徑 | 內容 |
|---|---|
| `realbot/` | 狗上跑的（純標準庫＋numpy）：`shm_io`（SHM 底層）、`coord`（★ 座標換算／限位／姿勢的唯一來源）、`kin`／`cpg`（IK／CPG 的 stdlib 移植）、`M0`–`M10`（風險遞增的實機模組；**`M9_gait.py` 是步態主程式**，含 `--policy`／`--teleop`）、`policy_np`／`rl_obs`（RL 推論與 obs）、`teleop_cmd`（四向遙控映射，純 CPG 版）、`estop_max.sh`（★ 急停）、`push_to_dog.sh`／`pull_from_dog.sh`／`clean_dog_logs.sh` |
| `inference/` | 本機模擬與分析：`max_model`／`cpg_max`／`leg_kin`／`cpg_walk_max`（模型、CPG、IK、rollout）、`gait_baseline`（★ 基準步態唯一來源）、`obs_max`（★ obs 唯一定義）、`rl_env_max`（MJX 訓練 env）、`local_infer_max`（RL 權重驗收 G3–G7、錄影）、`export_policy_np`（pkl→npz）、`replay_policy`／`policy_log_summary`（實機 log 分析）、`cpg_teleop_sim`（遙控模擬）、`m6_rec`／`m9_rec`／`real_obs`／`imu_check`／`obs_compare`（實機 obs 工具）、`diag/`（sim2real 量化等診斷腳本） |
| `notebooks/` | Colab 訓練 notebook（`cpg_rl_max_v2_*_colab.ipynb`，由 `build_nb_v2.py` 產生） |
| `weights/` | RL 權重：`cpg_rl_max_v2_3_params.pkl`（brax）＋ `cpg_rl_max_v2_3_np.npz`（狗上用） |
| `model/zgws/` | 官方 MJCF（已填實測輪摩擦）＋ MJX 訓練模型產生器；`SOURCE.md` 有質量分佈與致動器的坑 |
| `outputs/` | 軌跡檔（`A_*.json` 才推上狗；`stale/` 作廢）、模擬影片、回放與摘要 md |
| `logs/m_logs_trip*/` | 每趟實機原始 log（M9 json/log） |
| `reference/matrix_zgws/` | 原廠 MATRiX 設定檔原件 |
| `tests/` | 830+ 項；`conda run -n rbtdog python -m pytest task7/tests -q` |
| `docs/` | 結果文件、操作卡、設計；`docs/archive/` 是已完成／被取代的舊文件（沒刪，只是收起來） |

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
conda run -n rbtdog python task7/inference/policy_log_summary.py task7/logs/m_logs_tripNN/M9_*.json
```
