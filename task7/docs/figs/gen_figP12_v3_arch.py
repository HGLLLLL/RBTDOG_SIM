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
    "前後、左右、旋轉",
    "三個速度指令",
], fc="#fde2e4")

box(2, 28, 24, 28, "模式選擇器", [
    "★ 看指令決定用哪種走法",
    "  前後／邊走邊轉 → 輪行",
    "  原地轉、左右平移 → 踏步",
    "切換時 1 秒漸變",
], fc="#fff3cd", changed="新")

# ---------------- 中：步態產生器（雙模式）＋ RL
box(31, 47, 46, 46, "步態產生器（兩種走法，照原廠）", [
    "【輪行】腿站著不動，四個輪子滾",
    "  左右輪速差 → 轉彎",
    "【踏步】只抬該抬的腿，輪子配合滾",
    "  原地轉：對角兩腿一起抬，",
    "          另外兩腿的輪子一正一反滾",
    "  平移：往哪邊走，那一側前後腿輪流抬，",
    "          同側輪子一正一反滾",
    "  步頻、抬高、每步距離都來自原廠錄檔",
], fc="#d4edda", changed="改")

box(31, 2, 46, 40, "RL 策略網路（只微調，不接管）", [
    "輸出 12 個「微調量」（全 0 ＝ 純原廠走法）：",
    "  踏步快慢 ±28%",
    "  每條腿的步距 ×4",
    "  抬腿高度",
    "  機身重心左右前後移（sway）",
    "  ＋ 四個輪子的轉速修正（±20%）",
    "－ 不再決定步態順序（由原廠圖案固定）",
], fc="#ffe5cc", changed="改")

# ---------------- 中右：關節軌跡 + 輪速
box(82, 66, 24, 20, "腿部關節軌跡", [
    "腳要到的位置 →",
    "算出 12 個關節角度",
    "位置伺服（kp 250）",
], fc="#eeeeee")

box(82, 38, 24, 24, "輪子指令", [
    "四個輪子的轉速",
    "★ 改用位置環（目標角",
    "  一直累加，像原廠）",
    "＋ 補一點摩擦前饋",
    "太慢的指令直接當 0",
], fc="#e5d8f5", changed="改")

# ---------------- 右：機器狗 + 觀測
box(112, 47, 26, 46, "輪足機器狗", [
    "（模擬器 / 實機）",
    "",
    "模擬器新加的（實機量的）：",
    "  輪子摩擦、延遲、轉速雜訊",
    "  側向關節較軟（承重會外張）",
    "  機身有點歪著站",
], fc="#dbe9f6", changed="改")

box(143, 47, 27, 46, "狀態觀測 s_t", [
    "機身傾斜（重力方向）",
    "機身角速度",
    "關節角度／角速度",
    "輪子轉速",
    "指令 (vx, vy, ωz)",
    "＋ 現在是哪種走法、",
    "   走偏了多少（航向累計）",
    "－ 拿掉機身速度、絕對朝向、",
    "   腳有沒有著地（實機量不到）",
], fc="#e8e8e8", changed="改", size=9.6)

# ---------------- 下右：reward
box(112, 2, 58, 40, "Reward 設計", [
    "跟上指令：前後速度、轉速、不走偏",
    "姿態穩：少搖晃（roll / pitch）",
    "＋ 不要固定歪一邊（長期平均的側傾、重心偏移）",
    "＋ 守規矩：輪行時腿不抬；踏步只抬該抬的腿",
    "＋ 像原廠：抬腿高度接近原廠錄到的值",
    "動作平順、少耗力",
    "安全線：力矩、追蹤誤差、膝關節速度",
], fc="#d1f0e8", changed="改")

# ---------------- 箭頭
arrow((26, 77), (31, 77), "指令")
arrow((14, 62), (14, 56))                                   # 指令 → 模式選擇器
arrow((26, 42), (31, 60), "哪種走法", rad=-0.2, tpos=0.55, toff=(-1, 1))
arrow((54, 47), (54, 42), "微調量", color="#b25a00", tpos=0.5, toff=(8, -0.5))
arrow((77, 78), (82, 78), "腳的位置")
arrow((77, 60), (82, 52), "輪速", tpos=0.5, toff=(-1, 1.5))
arrow((106, 76), (112, 76), "關節角")
arrow((106, 50), (112, 60), "輪力矩", tpos=0.45, toff=(-1, 1.5))
arrow((138, 60), (143, 60), "感測", toff=(0, 1.0))
arrow((155, 47), (155, 42.5), "", color="#333")
arrow((125, 47), (125, 42.5), "", color="#333")
# 觀測回到 RL、reward 回到 RL
arrow((156, 93.5), (156, 97), "", color="#1a5fb4")
arrow((156, 97), (54, 97), "觀測 s_t → 給 RL 看", color="#1a5fb4", rad=0.0, tpos=0.5, toff=(0, 0.8))
arrow((54, 97), (54, 93.5), "", color="#1a5fb4")
arrow((112, 22), (77, 22), "獎勵 r_t → 更新 RL", color="#1a5fb4", tpos=0.5, toff=(0, 0.8), size=10)

# ---------------- 標題與圖例
ax.text(2, 103.5, "CPG-RL v3 雙模式架構（2026-09-09 定案）", fontsize=16, fontweight="bold", va="top")
ax.text(2, 99.3, "紅框＝與今天最初手繪架構不同的地方；行首 ＋ 新增、－ 拿掉、★ 關鍵決定", fontsize=10.5, va="top", color=RED)
ax.text(2, -2.5, "為什麼這樣改（依據：原廠遙控錄製的 17 段動作、輪子系統辨識）：", fontsize=10.5, va="top", fontweight="bold")
for i, ln in enumerate([
    "1. 原廠前後移動只用輪子、腿站著；原地轉和平移才踏步，而且踏步的圖案固定 → 走法由指令決定、圖案照抄原廠，RL 只負責微調。",
    "2. 輪子用速度控制在模擬裡轉不了彎（地面阻力撐住兩邊輪速差）→ 改成原廠用的位置環。",
    "3. 觀測只放實機量得到的東西；獎勵加「不要固定歪一邊」（今天實機最大的 sim2real 差距）與「守規矩」。",
]):
    ax.text(2, -6 - i * 3.4, ln, fontsize=9.8, va="top")

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("→", OUT)
