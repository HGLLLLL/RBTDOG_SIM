# d1max-rl-deploy 獨立 repo 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 從 `~/rbtdog_sim/task7` 抽出純 RL 上機線與重訓後匯出／驗證工具鏈，組成 `~/d1max-rl-deploy`，附繁中 README，可直接上傳私有 GitHub。

**Architecture:** 扁平沿用原目錄名（去掉 `task7/`），只搬需要的檔案；程式碼只改路徑與清單，不改行為；文件去 `task7/` 前綴；README 是主要入口，把純 RL 操作卡引用的 v3.6f 卡 §7–§8 內容併進 README。

**Tech Stack:** bash、Python 3.11（numpy、mujoco 3.10、onnx、onnxruntime、pytest）、git。

## Global Constraints

- 位置 `~/d1max-rl-deploy`；全新 `git init`；commit 作者為使用者 git 身分，**不加任何 co-author trailer**，不建 CONTRIBUTORS。
- README 繁體中文、無表情符號、不提 AI。
- `realbot/estop_max.sh` 一行不動。
- 新 repo 內不得殘留 `/home/huang`；`task7/` 只允許出現在說明「來源 repo」的句子裡。
- 密碼、IP、帳號保留（私有 repo）。
- 驗證指令一律 `conda run -n rbtdog python ...`。

---

### Task 1: 建骨架、搬檔案、環境檔、git init

**Files:** 建立 `~/d1max-rl-deploy/` 全部目錄；`.gitignore`；`environment.yml`。

- [ ] **Step 1: 搬檔案**

```bash
SRC=~/rbtdog_sim/task7; DST=~/d1max-rl-deploy
mkdir -p $DST/{RL_only,weights,realbot,inference/diag,model/zgws,reference,tests,docs/results,logs/m_logs_trip24,outputs/rl_only,outputs/report_tables}
cp ~/rbtdog_sim/RL_only/* $DST/RL_only/
cp $SRC/weights/rl_only_2026-09-24_np.npz $DST/weights/
for f in shm_io coord kin cpg M0_probe M5_leg_pose M6_load_probe M7_standup M9_gait M_faultwatch M_env_probe M_freezetest policy_np rl_obs rl_obs_v3 gen_v3 gen_direct rl_obs_rl; do cp $SRC/realbot/$f.py $DST/realbot/; done
cp $SRC/realbot/{estop_max,push_to_dog,pull_from_dog,clean_dog_logs}.sh $DST/realbot/
cp $SRC/inference/{export_policy_rl,rl_only_infer,max_model,leg_kin}.py $DST/inference/
cp $SRC/inference/diag/{m9_v3_trip_report,report_tables_rl_only,rl_only_range_sweep,rl_only_wz_tilt_sweep}.py $DST/inference/diag/
cp $SRC/model/zgws/{scene_flat_mjx_v3f,zgws_mjx_v3f,scene_flat,zgws}.xml $SRC/model/zgws/{make_mjx_model.py,fetch_assets.sh,SOURCE.md} $DST/model/zgws/
cp $SRC/reference/hang_torque_ref.json $DST/reference/
cp $SRC/tests/{test_gen_direct,test_policy_rl_np,test_policy_rl_closed_loop,test_kin,test_shm_imu}.py $DST/tests/
cp $SRC/docs/{現場SOP_通用_2026-08-27,現場操作卡_純RL_rl16_2026-09-29,純RL步態部署交接_2026-09-29,D1Max_機器狗資訊}.md $DST/docs/
cp $SRC/docs/results/{S_*,T_*,K_*,P_*,H_*}.md $DST/docs/results/
cp $SRC/logs/m_logs_trip24/M9_20260930_105129.{json,log} $DST/logs/m_logs_trip24/
cp $SRC/outputs/rl_only/*.md $DST/outputs/rl_only/
cp $SRC/outputs/report_tables/* $DST/outputs/report_tables/
```

- [ ] **Step 2: `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
*.mp4
model/zgws/assets/
logs/*
!logs/m_logs_trip24/
outputs/*
!outputs/rl_only/
!outputs/report_tables/
outputs/rl_only/*.png
.DS_Store
```

- [ ] **Step 3: `environment.yml`**（版本取自現行 rbtdog 環境）

```yaml
name: d1max
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
      - numpy==2.4.6
      - mujoco==3.10.0
      - onnx==1.23.0
      - onnxruntime==1.30.0
      - pytest==9.1.1
      - imageio==2.37.3
      - imageio-ffmpeg==0.6.0
      - pillow==12.2.0
```

- [ ] **Step 4: `git init`** `cd ~/d1max-rl-deploy && git init -q -b main`

### Task 2: 程式碼路徑修改

**Files:** `tests/test_policy_rl_np.py`、`tests/test_policy_rl_closed_loop.py`、`inference/rl_only_infer.py`、`inference/export_policy_rl.py`、`inference/diag/rl_only_range_sweep.py`、`inference/diag/rl_only_wz_tilt_sweep.py`、`inference/diag/report_tables_rl_only.py`、`inference/diag/m9_v3_trip_report.py`、`inference/max_model.py`（docstring）。

- [ ] **Step 1:** 兩支測試 `ROOT.parent / "RL_only"` → `ROOT / "RL_only"`；docstring「repo 根 `RL_only/`」不變。
- [ ] **Step 2:** `rl_only_infer.py` 第 20 行 `RL = HERE.parent / "RL_only"`；docstring 的 `task7/inference/`→`inference/`、`task7/outputs/`→`outputs/`。
- [ ] **Step 3:** `export_policy_rl.py` 第 28 行 `ROOT = HERE.parent`；docstring 同上。
- [ ] **Step 4:** 兩支 sweep 前三行改成：

```python
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
import rl_only_infer as ri, onnxruntime as ort
io = json.load(open(ROOT / "RL_only" / "policy_io.json"))
sess = ort.InferenceSession(str(ROOT / "RL_only" / "2026-09-24_13-23-11.onnx"), providers=["CPUExecutionProvider"])
```

- [ ] **Step 5:** `report_tables_rl_only.py` `OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "report_tables")`。
- [ ] **Step 6:** `m9_v3_trip_report.py` 開頭改成 argparse：`--dir`（預設 `ROOT/logs/m_logs_trip24`）、位置參數 `stamp:名稱`（預設 `20260930_105129:左平移 +0.5`）；`sys.path` 用 `ROOT/realbot`。
- [ ] **Step 7:** `max_model.py` docstring 的 `task7/` 去掉。
- [ ] **Step 8:** 驗證 `conda run -n rbtdog python -m pytest tests -q` 全過；`python inference/diag/m9_v3_trip_report.py` 跑完。

### Task 3: 部署腳本

**Files:** `realbot/push_to_dog.sh`、`realbot/pull_from_dog.sh`、`realbot/clean_dog_logs.sh`。

- [ ] **Step 1:** `push_to_dog.sh` FILES 改為：shm_io coord kin cpg M0_probe M5_leg_pose M6_load_probe M7_standup M9_gait M_faultwatch M_freezetest M_env_probe estop_max.sh policy_np rl_obs rl_obs_v3 gen_v3 gen_direct rl_obs_rl ＋ `../weights/rl_only_2026-09-24_np.npz`（改成變數 `NPZ`，可用環境變數覆寫）；刪掉軌跡檔區塊；`NOT_PUSHED=(push_to_dog.sh pull_from_dog.sh clean_dog_logs.sh)`；REF 缺失直接 exit 1；加 `DRY=1`：在 scp 之前列出 FILES 並逐檔 `[[ -f ]]`，然後 exit 0；訊息去 `task7/`。
- [ ] **Step 2:** `pull_from_dog.sh`、`clean_dog_logs.sh` 註解去 `task7/`。
- [ ] **Step 3:** `bash -n` 三支；`DRY=1 bash realbot/push_to_dog.sh` 無漏檔警告、全部存在。

### Task 4: 文件

**Files:** `docs/*.md`、`docs/results/*.md`、`model/zgws/SOURCE.md`、`inference/diag/README.md`（新寫，四支一句話）。

- [ ] **Step 1:** `sed -i 's#task7/##g'` 於 docs、SOURCE.md；逐條檢查 P 文件 14 處是否都是路徑。
- [ ] **Step 2:** 純 RL 操作卡第 121 行改為「鍵盤流程、急停三步、資料回收見 README 第 5 節」；§0 的 `cd ~/rbtdog_sim && bash task7/realbot/push_to_dog.sh` 改成 `cd ~/d1max-rl-deploy && bash realbot/push_to_dog.sh`；§4 的分析指令改 `logs/m_logs_trip24`。
- [ ] **Step 3:** SOP §3、§6 指令去 `task7/`。
- [ ] **Step 4:** `grep -rn "task7/\|/home/huang" .` 逐條確認。

### Task 5: README

**Files:** `README.md`（依 spec §4 十一節）。

- [ ] **Step 1:** 寫 README，操作段落逐字引用操作卡的旗標與指令，鍵盤流程與急停 A/B/C 三段自 v3.6f 卡 §7–§8 改寫成純 RL 版（權重檔名、站姿 0.8/−1.5）。
- [ ] **Step 2:** `grep -c` 無表情符號（掃 emoji 區段）；連結逐一 `test -e`。

### Task 6: 最終驗證與首次 commit

- [ ] `conda run -n rbtdog python -m pytest tests -q` → 全過。
- [ ] `conda run -n rbtdog python inference/export_policy_rl.py --out /tmp/x.npz` sha 與 weights 一致。
- [ ] `conda run -n rbtdog python inference/rl_only_infer.py --cmds "0.5,0,0" --secs 2 --tag smoke` 不報錯，之後刪掉產生的 `eval_rl_only_real_smoke.md`。
- [ ] `du -sh .git` 前總體積 < 20 MB。
- [ ] `git add -A && git commit -m "初始版本：D1 Max 純 RL 步態上機與重訓後部署工具鏈"`（作者 HGL，無 trailer）；`git log --format='%an%n%b'` 確認。
