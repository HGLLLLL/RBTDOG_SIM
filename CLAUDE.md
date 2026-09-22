# rbtdog_sim

MuJoCo 機器狗模擬與實機部署。**現行工作只有 `task7/`（智元 D1 Max，CPG-RL 步態訓練 → 上機）。**
`task3`–`task6`（Go1／D1 EDU）已結案，除非被明確指到，不要讀。

## 接手（每個 session 開頭）

1. 只讀 `task7/HANDOFF.md`（現況卡，≤150 行）。它說「明天第一件事」是什麼就做什麼。
2. 需要細節才開對應文件：目錄地圖 `task7/README.md`、文件索引 `task7/docs/README.md`。
3. **大檔不要整份讀進主對話**：`task7/docs/D1Max_機器狗資訊.md`（95 KB）查章節；
   `task7/docs/handoff_history/`（舊 HANDOFF 全文）用 grep；要跨多檔彙整就派 subagent 只回結論。

## 收工（每次結束前）

- **覆寫** `task7/HANDOFF.md`，不是往上疊：更新權重表、明天第一件事、未解清單。超過 150 行就把被取代的段落搬到
  `task7/docs/handoff_history/`（原文照搬，加日期）。
- 定案的結果寫成 `task7/docs/results/<字母>_<題目>_<日期>.md`，HANDOFF 只留一行加連結。
- 新檔案放對位置：實機原始 log → `task7/logs/`；模擬產物 → `task7/outputs/`；不要留在 repo 根目錄。

## 記憶（`~/.claude/.../memory/`）

- 每份 ≤ 3 KB。只寫跨 session 才需要的判斷、教訓、使用者要求；事實與數字指向 repo 文件，不重抄。
- 方法論類（`diagnostic-tools-lie`、`v31-generator-lessons`）留在記憶，是使用者的決定。
- task7 的進度只寫在 `d1-max-task7`，內容是「起點在 HANDOFF」加硬要求，不重複 HANDOFF。

## 工作方式

- 繁體中文溝通；先研究再動手；報告數字具體、不迴避限制。
- 改 env／reward 前先看該版 spec 的執行紀錄與「沒採用」清單，不重掃已掃過的東西。
- 改 `rl_env_v3` 後跑 `conda run -n rbtdog python -m pytest task7/tests -q`；`test_rl_env_v3_golden.py` 釘住 v3.3 逐位元不變。
- 派 subagent 用 opus，不用 haiku。
- 上機相關的任何改動都對照 `task7/docs/現場SOP_通用_2026-08-27.md` 的安全前提。
