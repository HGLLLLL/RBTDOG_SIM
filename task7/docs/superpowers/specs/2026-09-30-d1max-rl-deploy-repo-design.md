# 給同事的獨立 repo：`d1max-rl-deploy` 設計（2026-09-30）

目的：同事要在另一台電腦、使用者不在場時操作 D1 Max，並在重訓純 RL policy 後自己完成「匯出 → 驗證 → 上機」。
把 `task7/` 裡純 RL 線需要的檔案整理成一個獨立、可上傳 GitHub（私有）的 repo，附一份完整 README。

## 1. 已定案的選擇

| 項目 | 決定 |
|---|---|
| 範圍 | 純 RL 上機線 ＋ 重訓後重新匯出／驗證工具鏈（`export_policy_rl`、`test_policy_rl_*`、`rl_only_infer`）。不含 CPG-RL v3.x 線 |
| 公開性 | 私有 repo。狗的 IP、ssh 帳號、WiFi／ssh 密碼全部保留在文件裡（現場離線可用） |
| README 語言 | 繁體中文，結構仿熱門 repo 的 README 慣例，無表情符號 |
| 資料 | 純 RL 的 MuJoCo 驗收表（`outputs/rl_only/*.md`）、報告表（`outputs/report_tables/`）、trip24 一趟範例 log（`M9_20260930_105129`，左平移 0.5）。其餘 log 與影片不放 |
| 目錄命名 | 扁平沿用原名（去掉 `task7/` 這層），腳本相對路徑與文件引用改動最小 |
| 測試 | 只留純 RL 線 5 支；`test_m9_gait` 不進（依賴 v2.3／v3.6f 權重、開迴路軌跡、M8／M10） |
| 位置與名稱 | `~/d1max-rl-deploy`，與 `rbtdog_sim` 平行 |
| git | 全新 `git init`；首次 commit 用使用者的 git 身分（HGL），不加任何 co-author trailer；不建 CONTRIBUTORS 檔；README 不提 AI |

## 2. 目錄結構

```
d1max-rl-deploy/
  README.md
  environment.yml                 本機 conda 環境（python 3.11、numpy、mujoco 3.10、onnx、onnxruntime、pytest、imageio、imageio-ffmpeg、pillow）
  .gitignore                      logs/ 與 outputs/ 的內容（範例除外）、*.mp4、__pycache__、model/zgws/assets/
  RL_only/                        訓練端輸出：2026-09-24_13-23-11.{onnx,pt,_env.yaml}、policy_io.json、observation.py
  weights/rl_only_2026-09-24_np.npz
  realbot/                        推上狗的檔案、三支部署腳本、急停
  inference/                      export_policy_rl.py、rl_only_infer.py、max_model.py、leg_kin.py、diag/{m9_v3_trip_report,report_tables_rl_only,rl_only_range_sweep,rl_only_wz_tilt_sweep}.py、diag/README.md
  model/zgws/                     scene_flat_mjx_v3f.xml、zgws_mjx_v3f.xml、scene_flat.xml、zgws.xml、make_mjx_model.py、fetch_assets.sh、SOURCE.md（不含 assets/ 網格）
  reference/hang_torque_ref.json
  tests/                          test_gen_direct、test_policy_rl_np、test_policy_rl_closed_loop、test_kin、test_shm_imu
  docs/                           現場SOP_通用、現場操作卡_純RL_rl16、純RL步態部署交接、D1Max_機器狗資訊、results/{S,T,K,P,H}
  logs/m_logs_trip24/             M9_20260930_105129.{json,log}
  outputs/rl_only/                13 個 eval_rl_only_*.md
  outputs/report_tables/          表1、表2 的 svg 與 png
```

### realbot/ 取捨

留：`shm_io coord kin cpg M0_probe M5_leg_pose M6_load_probe M7_standup M9_gait M_faultwatch M_env_probe M_freezetest policy_np rl_obs rl_obs_v3 gen_v3 gen_direct rl_obs_rl estop_max.sh push_to_dog.sh pull_from_dog.sh clean_dog_logs.sh`。
`kin cpg gen_v3 rl_obs_v3 M5 M7` 是 `M9_gait.py` 直接 import 的，不能少。
丟：`M1–M4、M8、M10、M11、teleop_cmd、shm_decode、recon*、ecal_shm_topology.json`。

## 3. 要改的程式碼（只改路徑與清單，不改行為）

| 檔案 | 改什麼 |
|---|---|
| `tests/test_policy_rl_np.py`、`tests/test_policy_rl_closed_loop.py` | `ROOT.parent / "RL_only"` → `ROOT / "RL_only"` |
| `inference/rl_only_infer.py` | `RL = HERE.parent.parent / "RL_only"` → `HERE.parent / "RL_only"`；docstring 去 `task7/` |
| `inference/export_policy_rl.py` | `ROOT = HERE.parent.parent` → `HERE.parent`；docstring 去 `task7/` |
| `inference/diag/rl_only_range_sweep.py`、`rl_only_wz_tilt_sweep.py` | `sys.path` 與 `RL_only/` 路徑改成以檔案位置為準，不依賴 cwd |
| `inference/diag/m9_v3_trip_report.py` | 去掉寫死的 `/home/huang/...`：`sys.path` 以檔案位置為準；加 `--dir` 參數，預設 `logs/m_logs_trip24`；預設 RUNS 改成範例那一趟 |
| `inference/diag/report_tables_rl_only.py` | `OUT` 改成以檔案位置為準 |
| `realbot/push_to_dog.sh` | FILES 清單修剪成純 RL 那一套（拿掉 v2.3／v3.6f 權重、M2／M8／M10／M11、teleop_cmd、軌跡檔迴圈）；`NOT_PUSHED` 對應更新；加 `DRY=1` 模式（只列檔案、檢查存在與漏檔，不連狗）；訊息裡的 `task7/` 去掉、`hang_rehearsal` 提示改成「檔案缺失即停」 |
| `realbot/pull_from_dog.sh`、`clean_dog_logs.sh` | 註解裡的 `task7/` 去掉 |
| `realbot/estop_max.sh` | 一行不動 |
| `docs/*.md`、`model/zgws/SOURCE.md`、`inference/diag/README.md` | `task7/` 前綴去掉；純 RL 操作卡末段「與 v3.6f 操作卡 §7–§8 相同」改指 README 對應章節 |

## 4. README 大綱

1. 簡介與範圍（這個 repo 做什麼、不做什麼；已驗證狀態一句話：trip24 12 趟 0 摔）
2. 系統概觀（資料流圖：joint_state/imu → obs → policy → gen_direct → joint_cmd；本機與狗端分工）
3. 目錄說明（每個目錄一句話）
4. 環境安裝（本機 conda；狗端零安裝；ssh 金鑰；連線資訊）
5. 首次上機（安全前提清單 → push → 狗上檢查 → 乾跑 → 段 ①②③④ → 放大 → 鍵盤流程 → 急停 → 收工拉 log）
6. 重訓後重新部署（把 ONNX 與 policy_io.json 放進 `RL_only/` → 匯出 npz → 跑兩支測試 → MuJoCo 重現 → 更新 push 清單與操作卡旗標）
7. 資料放哪（log、模擬表、權重、訓練端輸出）
8. 硬限制與防呆（自 HANDOFF §5 精簡，不隨版本變的）
9. 已知限制（vy ≤ 0.3 死區、wz 0.5 不可用、輪速 0.7 倍）
10. 文件索引
11. 故障排除（push 不一致、scp 權限、sudo kill -CONT 忘打、被 estop 殺掉沒 JSON）

## 5. 驗證

- 新 repo 內 `conda run -n rbtdog python -m pytest tests -q` 全過。
- `DRY=1 bash realbot/push_to_dog.sh`：列出的檔案全部存在，`realbot/` 沒有既不在 FILES 也不在 NOT_PUSHED 的檔。
- `python inference/export_policy_rl.py --out /tmp/x.npz` 產出的權重與 `weights/` 內的一致（`test_committed_npz_matches_source` 就是這個）。
- `python inference/rl_only_infer.py --cmds "0.5,0,0" --secs 2` 跑完不報錯。
- `python inference/diag/m9_v3_trip_report.py` 不帶參數對範例 log 跑完。
- `grep -rn "task7/\|/home/huang"` 在新 repo 內為零（保留歷史文件裡描述性提及者除外，逐條確認）。
- `git log` 只有一筆、作者 HGL、無 trailer。
