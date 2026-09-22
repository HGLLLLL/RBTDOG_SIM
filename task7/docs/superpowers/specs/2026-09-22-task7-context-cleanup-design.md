# task7 整理：讓 session 變輕、東西找得到（2026-09-22）

## 問題

- `HANDOFF.md` 91 KB／1245 行只增不減，接手必讀；同一天疊 9 節「★★★★★」，「目前最佳」出現在兩個版本。
- 記憶 27 份共 145 KB，一半是 task3–6 已結案；`d1-max-task7`（14 KB）與 `cpg-rl-d1max-line`（17 KB）和 HANDOFF 重複。
- `README.md`、`docs/README.md` 停在 2026-09-09。
- `outputs/` 140 檔平鋪、`logs/` 610 MB、`notebooks/`／`weights/` 各 13 版；根目錄躺著 5 個 recon 資料夾。

## 決定（使用者 2026-09-22 拍板：方案 B，先做 1→3→6；方法論留在記憶）

1. **HANDOFF 改現況卡**：上限 150 行，只放權重表、明天第一件事、驗收指令、未解、硬限制。
   舊全文 `git mv` 到 `docs/handoff_history/HANDOFF_2026-08-25_2026-09-22_全文.md`，一字不改。
2. **記憶瘦身**：task3–6 與已被 repo 文件取代的記憶搬到 `memory/archive/`（不刪）；`d1-max-task7` 重寫成 ≤3 KB。
   `diagnostic-tools-lie`、`v31-generator-lessons` 等方法論留在記憶。
3. **兩份 README 更新到現況**；根目錄 recon 資料夾與 `outputs/recon*_20260922/` 比對後，補齊缺的 log 再刪根目錄副本。
6. **根目錄 `CLAUDE.md`**（每個 session 自動載入）：接手只讀 HANDOFF、大檔不整份讀、收工覆寫不疊、記憶 ≤3 KB。

## 之後再做（4→5，排在 Colab 訓練空檔）

4. `task7/archive/`：`git mv` v2 系列 notebook、v2 與 v3.0–v3.5 權重、`outputs/` 的 g0／sweep／舊 eval、`logs/` trip3–18。
   **程式不搬**：`rl_env_v3` 匯入 `rl_env_max`，`M9_gait`／`policy_np` 用 v2 模組。
5. `outputs/` 改一版一個子目錄（`outputs/v36f/`），驗收表固定檔名；要改 `local_infer_v3.py` 的輸出路徑與 `compare_evals.py`。

## 不做

- 不開 task8：import 路徑與 40 份測試全要改，且 v3 依賴 v2 模組搬不乾淨；task6→task7 的經驗是換資料夾不改流程會再長回來。
