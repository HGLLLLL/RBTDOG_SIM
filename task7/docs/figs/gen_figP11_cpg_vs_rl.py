#!/usr/bin/env python3
"""圖 P11：純 CPG（A_kp250）vs RL v2.3 —— 只列有差別的指標，論文風格表格。

資料來源（2026-09-08，原始網格模型、60 s、12 皮米擾動、實機延遲 1 步、淡入 1 s）：
  純 CPG：local_infer_max --dummy --preset v2.3 --secs 60（開迴路無指令，走 0.35 m/s）
  RL v2.3：local_infer_max --params cpg_rl_max_v2_3_params.pkl --preset v2.3 --secs 60 --perturb 12 --compare
用法：conda run -n rbtdog python task7/docs/figs/gen_figP11_cpg_vs_rl.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "圖P11_純CPG_vs_RL_v2.3.png"

# (指標, 純 CPG, RL v2.3, 變化, 好/壞)
ROWS = [
    ("週期俯仰 (°)",        "4.18", "2.34", "−44 %",  True),
    ("Roll 峰值 (°)",        "3.88", "2.50", "−36 %",  True),
    ("Roll 標準差 (°)",      "2.57", "1.21", "−53 %",  True),
    ("機身彈跳 (mm)",        "16.9", "13.8", "−18 %",  True),
    ("支撐腳數（平均）",     "2.70", "3.22", "+0.5 腳", True),
    ("前腳執行率",           "0.88", "1.05", "+19 %",  True),
    ("後腳執行率（過衝）",   "1.47", "1.21", "過衝 −55 %", True),
    ("60 s 偏航 (°)",        "+4.8", "−29.8", "退步 6×（0.08→0.50 °/s）", False),
    ("60 s 側偏 (m)",        "+1.9", "−5.3",  "退步 2.8×", False),
]

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(ROWS) + 1.9))
ax.axis("off")
cols = ["指標", "純 CPG", "RL v2.3", "變化"]
cells = [[r[0], r[1], r[2], r[3]] for r in ROWS]
tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center",
               colWidths=[0.34, 0.16, 0.16, 0.34])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11.5)
tbl.scale(1.0, 1.75)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_linewidth(0)
    if r == 0:
        cell.set_facecolor("#2b2b2b")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("#f4f4f4" if r % 2 else "white")
        if c == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.03
        if c == 3:
            good = ROWS[r - 1][4]
            cell.get_text().set_color("#1a7f37" if good else "#c0392b")
            cell.get_text().set_fontweight("bold")
# 表頭上下粗線（論文三線表）
n = len(ROWS)
for c in range(4):
    tbl[(0, c)].visible_edges = "TB"
    tbl[(0, c)].set_linewidth(1.4)
    tbl[(0, c)].set_edgecolor("black")
    tbl[(n, c)].visible_edges = "B"
    tbl[(n, c)].set_linewidth(1.4)
    tbl[(n, c)].set_edgecolor("black")
fig.suptitle("純 CPG 步態 vs RL v2.3（僅列有差異的指標）", fontsize=14, fontweight="bold", y=0.97)
fig.text(0.5, 0.035,
         "同條件：原始網格模型、60 s × 12 擾動、實機延遲 1 步、起步淡入 1 s；RL 指令 vx 0.30 m/s（純 CPG 開迴路 0.35）。"
         "跌倒 0/12、峰值力矩 ×1.2 = 66.8 vs 67.6 N·m（門檻 70）兩者相同，未列。",
         ha="center", fontsize=8.5, color="#555555", wrap=True)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(OUT, dpi=200)
print("→", OUT)
