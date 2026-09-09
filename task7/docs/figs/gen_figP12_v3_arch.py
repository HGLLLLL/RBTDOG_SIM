#!/usr/bin/env python3
"""圖P12：CPG-RL v3 雙模式架構（2026-09-09 最終版），對照使用者當天最初的手繪架構圖（docs/image.png）。
與原圖不同的地方用紅框＋「新／改／刪」標籤標出。

    conda run -n rbtdog python task7/docs/figs/gen_figP12_v3_arch.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "圖P12_v3雙模式架構.png"
plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RED = "#c0392b"
fig, ax = plt.subplots(figsize=(17.5, 12.2))
ax.set_xlim(0, 172)
ax.set_ylim(-16, 104)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec="#555", changed=None, title_size=12.5, size=10.2, lw=1.4):
    """changed: None／'新'／'改'  → 紅框＋標籤。"""
    ec2, lw2 = (RED, 2.6) if changed else (ec, lw)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5", fc=fc, ec=ec2, lw=lw2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=title_size, fontweight="bold")
    for i, ln in enumerate(lines):
        col = RED if ln.startswith(("★", "＋", "－")) else "#222"
        ax.text(x + 1.6, y + h - 6.2 - i * 3.25, ln, ha="left", va="top", fontsize=size, color=col)
    if changed:
        ax.text(x + w - 1.0, y + h + 0.9, f" {changed} ", ha="right", va="bottom", fontsize=10.5, color="white",
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec=RED))


def arrow(p, q, text="", color="#333", lw=1.8, rad=0.0, tpos=0.5, toff=(0, 1.2), size=10):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        mx, my = p[0] + (q[0] - p[0]) * tpos + toff[0], p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=size, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# ---------------- 左：指令與模式選擇
box(2, 62, 24, 30, "使用者控制輸入", [
    "(vx, vy, ωz)",
    "遙控器／鍵盤",
    "",
    "vx 前後、ωz 旋轉",
    "邊走邊轉＝差速",
], fc="#fde2e4")

box(2, 28, 24, 28, "模式選擇器", [
    "★ 由指令決定，不是 RL 學",
    "|ωz| 大且 |vx| 小 → 踏步",
    "其他 → 輪行",
    "切換 1 s 線性混合 u",
], fc="#fff3cd", changed="新")

# ---------------- 中：步態產生器（雙模式）＋ RL
box(31, 47, 46, 46, "步態產生器（雙模式）", [
    "【輪行模式】",
    "  腿：原廠站姿（hip 0.52 / knee −1.21）",
    "  ＋ RL 小順從（足端 ±20 mm）",
    "  輪：差速 IK  ωL=(vx−ωz·L/2)/r  ωR=(vx+ωz·L/2)/r",
    "【踏步模式（原廠圖案）】",
    "  對角一對腿同相踏步（左轉 FL+RR）",
    "  ★ 另一對腿站姿、輪反向滾（FR +Ω / RL −Ω）",
    "  2.0 Hz・duty 0.5・抬 40 mm（＋撓度 36 mm）",
    "  步長 (dx, dy) 來自原廠錄檔 × |ωz|",
], fc="#d4edda", changed="改")

box(31, 2, 46, 40, "RL 策略網路（調變，不接管）", [
    "輸出 12 維（tanh，零動作＝純原廠模式表）：",
    "  ω 尺度 ±28%",
    "  逐腿步幅尺度 ×4（0.5–1.5）",
    "  抬腿尺度 0.6–1.4",
    "  body sway x, y（±40 mm）",
    "  ＋ 輪速殘差 ×4（±20%）",
    "－ 不再輸出相位差／步態序列（由圖案固定）",
], fc="#ffe5cc", changed="改")

# ---------------- 中右：關節軌跡 + 輪速
box(82, 66, 24, 20, "腿部關節軌跡", [
    "足端目標 → 解析 IK",
    "q_leg(t)，12 關節",
    "kp 250 / ABAD 60",
], fc="#eeeeee")

box(82, 38, 24, 24, "輪子指令", [
    "ω_wheel ×4",
    "★ 位置環 kp 60",
    "  （目標角逐步累加）",
    "＋ 前饋 τ_ff 0.13",
    "|ω| < 0.3 → 0（起動門檻）",
], fc="#e5d8f5", changed="改")

# ---------------- 右：機器狗 + 觀測
box(112, 47, 26, 46, "輪足機器狗", [
    "（MJX 模擬 / 實機）",
    "",
    "模擬器新增（M11 實測）：",
    "  輪摩擦 0.13、黏滯 0.015",
    "  延遲 1–2 tick、輪速雜訊 0.1",
    "  ABAD 剛度 ×0.4–1.0（靜態側傾）",
], fc="#dbe9f6", changed="改")

box(143, 47, 27, 46, "狀態觀測 s_t（76 維）", [
    "重力向量（機體姿態）",
    "角速度",
    "關節角 / 角速度",
    "輪子轉速",
    "指令 (vx, vy, ωz)",
    "＋ 模式 u、航向誤差 ∫",
    "＋ 上一步動作、CPG 相位",
    "－ 機體速度、絕對 yaw、接觸",
    "   （實機拿不到）",
], fc="#e8e8e8", changed="改", size=9.6)

# ---------------- 下右：reward
box(112, 2, 58, 40, "Reward 設計", [
    "追蹤：vx、ωz（雙尺度核）、航向誤差",
    "姿態：roll / pitch 與其角速度",
    "＋ 靜態偏置罰：roll、sway_y 的長期平均",
    "＋ 模式紀律：輪行不抬腿、踏步只抬該抬的腿",
    "＋ 原廠模仿：踏步腿抬高 ≈ 參考值",
    "平滑：Δ動作、Δω、力矩",
    "護欄：|τ| 58、誤差 0.45、膝速度 14 rad/s",
], fc="#d1f0e8", changed="改")

# ---------------- 箭頭
arrow((26, 77), (31, 77), "(vx, vy, ωz)")
arrow((14, 62), (14, 56))                                   # 指令 → 模式選擇器
arrow((26, 42), (31, 60), "模式 u", rad=-0.2, tpos=0.55, toff=(-1, 1))
arrow((54, 47), (54, 42), "調整參數", color="#b25a00", tpos=0.5, toff=(9, -0.5))
arrow((77, 78), (82, 78), "足端目標")
arrow((77, 60), (82, 52), "ω_wheel", tpos=0.5, toff=(-1, 1.5))
arrow((106, 76), (112, 76), "q_leg(t)")
arrow((106, 50), (112, 60), "τ_wheel", tpos=0.45, toff=(-1, 1.5))
arrow((138, 60), (143, 60), "感測", toff=(0, 1.0))
arrow((155, 47), (155, 42.5), "", color="#333")
arrow((125, 47), (125, 42.5), "", color="#333")
# 觀測回到 RL、reward 回到 RL
arrow((156, 93.5), (156, 97), "", color="#1a5fb4")
arrow((156, 97), (54, 97), "s_t（觀測）→ RL", color="#1a5fb4", rad=0.0, tpos=0.5, toff=(0, 0.8))
arrow((54, 97), (54, 93.5), "", color="#1a5fb4")
arrow((112, 22), (77, 22), "r_t（reward）→ 更新 RL", color="#1a5fb4", tpos=0.5, toff=(0, 0.8), size=10)

# ---------------- 標題與圖例
ax.text(2, 103.5, "CPG-RL v3 雙模式架構（2026-09-09 定案）", fontsize=16, fontweight="bold", va="top")
ax.text(2, 99.3, "紅框＝與當天最初手繪架構不同之處；行首 ＋ 新增、－ 移除、★ 關鍵決定", fontsize=10.5, va="top", color=RED)
ax.text(2, -2.5, "為什麼這樣改（原廠錄製 L 文件＋M11 輪子辨識）：", fontsize=10.5, va="top", fontweight="bold")
for i, ln in enumerate([
    "1. 原廠前進只用輪子、腿站姿；原地轉是「對角腿同相踏步＋另一對輪反向」→ 模式由指令決定，圖案寫死，RL 只調變。",
    "2. 差速轉向靠速度伺服撐不住打滑阻力（模擬偏航 0）→ 輪子改成原廠式位置環 kp 60。",
    "3. 觀測只放實機拿得到的量；reward 加靜態偏置罰（J 報告的 sim2real 第一根源）與模式紀律。",
]):
    ax.text(2, -6 - i * 3.4, ln, fontsize=9.8, va="top")

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("→", OUT)
