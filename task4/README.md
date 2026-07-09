# task4 — CPG-RL：Model-based + Learning-based 混合走路

用「CPG（中樞模式產生器）+ RL」的 hybrid 方法讓 Go2 走路，取代 task3 的純 CPG。
訓練在 Colab GPU（MJX），推論在本機 CPU（接回 task3 的 Go2 + 軟體 PD + 羅盤）。

## 資料夾結構

```
task4/
├── docs/                    教學與總結文件
│   ├── hybrid_locomotion_tutorial.md      hybrid 方法觀念白話版（四大類）
│   ├── cpg_rl_implementation_guide.md     CPG-RL 完整 step-by-step 教學
│   ├── cpg_rl_5dim_summary.md             5 維小規模測試結論總整
│   └── cpg_rl_paper_summary.md            論文標準版結論總整
├── notebooks/               Colab 訓練 notebook（皆含強化 DR：抗推/負重0~8kg/摩擦[0.3,1]）
│   ├── cpg_rl_colab.ipynb                 5 維起步版
│   ├── cpg_rl_paper_colab.ipynb           論文標準版（硬編碼 trot 耦合 W_COUP=8）
│   └── cpg_rl_paper_nocoup_colab.ipynb    論文忠實版（無耦合 W_COUP=0，RL 自己協調四腿）
├── inference/               本機 CPU 推論腳本
│   ├── local_infer.py                     5 維版
│   ├── local_infer_paper.py               論文版（--w_coup 8 耦合 / 0 無耦合）
│   └── compare.py                         純CPG vs RL 四大對比實驗(--exp 1..4)
├── weights/                 訓練好的權重
│   ├── cpg_rl_params.pkl                  5 維版
│   └── cpg_rl_paper_params.pkl            論文標準版
└── outputs/                 推論輸出（影片/軌跡圖）
```

## 兩個版本

| | 5 維起步版 | 論文標準版 |
|---|---|---|
| 動作 | 5 維（4 μ + 1 共用 ω）| 12 維（每腿 μx, μy, ω）|
| 腳掌 | 純矢狀面 x-z | 2D（含側向 y）|
| 相位 | 固定 trot 偏移 | 腿間耦合振盪器 |
| 抬腳 | `LIFT·amp·sin`（會拖地）| 固定 g_c（不拖地）|
| 觀測 | 51 維 | 76 維（加腳觸地布林等）|

## 使用流程

1. **訓練**：把 `notebooks/` 的 notebook 上傳 Colab（設 GPU），逐格跑，先過 Smoke test 再訓練，訓完存權重下載。
2. **推論**（本機，需 `pip install jax brax`）：
   ```bash
   # 論文版
   MUJOCO_GL=egl conda run -n rbtdog python inference/local_infer_paper.py \
       --params weights/cpg_rl_paper_params.pkl --secs 20 --video --push
   # 5 維版
   MUJOCO_GL=egl conda run -n rbtdog python inference/local_infer.py \
       --params weights/cpg_rl_params.pkl --secs 20 --video --push
   ```
   影片與圖輸出到 `outputs/`。`--dummy` 可在沒權重時先測管線。

3. **論文版兩個耦合變體**：推論時 `--w_coup` 必須與訓練一致——
   - 耦合版權重 `cpg_rl_paper_params.pkl` → `--w_coup 8`（預設）
   - 無耦合版權重 `cpg_rl_paper_nocoup_params.pkl` → `--w_coup 0`
4. **對比實驗**（純CPG vs RL，四支影片+圖表到 `outputs/`）：
   ```bash
   MUJOCO_GL=egl conda run -n rbtdog python inference/compare.py \
       --exp 1 --params weights/cpg_rl_paper_params.pkl --w_coup 8
   # --exp 1 推撞 / 2 摩擦 / 3 負載 / 4 直線50m
   ```

依賴 task3 的 `go2_gait.py`、`walk_line.py`（推論腳本用絕對路徑 import）。
