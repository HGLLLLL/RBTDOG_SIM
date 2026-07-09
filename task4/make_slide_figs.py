"""報告投影片資料流圖 —— 黑白學術流程圖風格（參考 ref/image.png，改成橫向蛇形、約 1:1）。
輸出到 task4/outputs/：fig1..fig4。文字精簡。
用法: conda run -n rbtdog python make_slide_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm

OUT = "/home/huang/rbtdog_sim/task4/outputs"
FP = fm.FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSerifCJK-Medium.ttc")
FPB = fm.FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSerifCJK-Bold.ttc")

BLACK = "#111111"
GRAY_FILL = "#ececec"          # 共用骨架的淡灰底（其餘純白）
W, HH = 3.3, 1.55              # 方塊寬高
DX, DY = 4.5, 3.0             # 欄距、列距
LW = 1.3


def new_ax():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def rect(ax, cx, cy, text, fs=13, shared=False, w=W, h=HH):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=True,
                           facecolor=(GRAY_FILL if shared else "white"),
                           edgecolor=BLACK, linewidth=LW, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontproperties=FP, fontsize=fs, color=BLACK, zorder=3)


def hexa(ax, cx, cy, text, fs=13, w=W, h=HH):
    ind = 0.5
    pts = [(cx - w / 2, cy), (cx - w / 2 + ind, cy + h / 2), (cx + w / 2 - ind, cy + h / 2),
           (cx + w / 2, cy), (cx + w / 2 - ind, cy - h / 2), (cx - w / 2 + ind, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=BLACK, linewidth=LW, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontproperties=FP, fontsize=fs, color=BLACK, zorder=3)


def stad(ax, cx, cy, text, fs=13, w=W, h=HH):
    ax.add_patch(FancyBboxPatch((cx - w / 2 + h / 2, cy - h / 2), w - h, h,
                                boxstyle=f"round,pad=0,rounding_size={h/2}",
                                facecolor="white", edgecolor=BLACK, linewidth=LW, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontproperties=FP, fontsize=fs, color=BLACK, zorder=3)


def arr(ax, x1, y1, x2, y2, lw=LW, color=BLACK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))


def connect(ax, p1, p2, w=W, h=HH):
    """相鄰方塊：同列→水平箭頭；同欄→垂直箭頭。"""
    (x1, y1), (x2, y2) = p1, p2
    if abs(y1 - y2) < 1e-6:                      # 同列
        if x2 > x1: arr(ax, x1 + w / 2, y1, x2 - w / 2, y2)
        else:       arr(ax, x1 - w / 2, y1, x2 + w / 2, y2)
    else:                                         # 換列（垂直往下）
        arr(ax, x1, y1 - h / 2, x2, y2 + h / 2)


def snake(n, ncols, x0=0.0, y0=0.0):
    pos = []
    for i in range(n):
        r, k = divmod(i, ncols)
        c = k if r % 2 == 0 else ncols - 1 - k
        pos.append((x0 + c * DX, y0 - r * DY))
    return pos


def title(ax, t):
    ax.text(ax.get_xlim()[0] + 0.2, ax.get_ylim()[1] - 0.15, t,
            fontproperties=FPB, fontsize=17, color=BLACK, va="top", ha="left")


def finish(ax, xs, ys, pad=1.2, top=2.6, left=0.0, bottom=0.0):
    xmin, xmax = min(xs) - W / 2 - pad - left, max(xs) + W / 2 + pad
    ymin, ymax = min(ys) - HH / 2 - pad - bottom, max(ys) + HH / 2 + pad + top
    span = max(xmax - xmin, ymax - ymin)           # 撐成正方形（≤1:1）
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig); print("[fig]", name)


# ─── fig1：純 CPG（開迴路）───
def fig1():
    fig, ax = new_ax()
    nodes = ["時鐘 t", "相位\nfreq·t", "腳掌軌跡", "IK", "關節角", "PD → 力矩", "物理"]
    shared = [False, False, True, True, True, True, True]
    P = snake(len(nodes), 3)
    finish(ax, [p[0] for p in P], [p[1] for p in P], left=2.6)
    title(ax, "純 CPG：開迴路（不看回饋）")
    for i, ((x, y), t) in enumerate(zip(P, nodes)):
        if i == 0: hexa(ax, x, y, t)
        else: rect(ax, x, y, t, shared=shared[i])
    for i in range(len(P) - 1):
        connect(ax, P[i], P[i + 1])
    # 回授：物理(左下) → 繞左側 → 時鐘(左上)
    railx = P[0][0] - W / 2 - 1.5
    (x6, y6), (x0, y0) = P[6], P[0]
    seg = [(x6 - W / 2, y6), (railx, y6), (railx, y0)]
    for a, b in zip(seg[:-1], seg[1:]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=BLACK, lw=LW, zorder=1))
    arr(ax, railx, y0, x0 - W / 2, y0)
    ax.text(railx - 0.25, (y0 + y6) / 2, "不看\n回饋", fontproperties=FP, fontsize=11,
            color=BLACK, ha="right", va="center")
    save(fig, "fig1_cpg_dataflow.png")


# ─── fig2：CPG-RL（閉迴路）───
def fig2():
    fig, ax = new_ax()
    nodes = ["76 維觀測", "策略網路", "12 維動作\nμx, μy, ω", "CPG 振盪器",
             "腳掌位置", "IK", "關節角", "PD → 力矩", "物理"]
    shared = [False, False, False, False, True, True, True, True, True]
    P = snake(len(nodes), 3)
    finish(ax, [p[0] for p in P], [p[1] for p in P], pad=1.4, left=1.8, bottom=1.4)
    title(ax, "CPG-RL：閉迴路（每 20ms 看回饋）")
    for i, ((x, y), t) in enumerate(zip(P, nodes)):
        if i == 0: hexa(ax, x, y, t)
        else: rect(ax, x, y, t, shared=shared[i])
    for i in range(len(P) - 1):
        connect(ax, P[i], P[i + 1])
    # 回授：物理(右下) → 繞底部左側 → 76維觀測(左上)
    (xL, yL), (x0, y0) = P[8], P[0]
    raily = yL - HH / 2 - 1.2
    railx = x0 - W / 2 - 1.5
    seg = [(xL, yL - HH / 2), (xL, raily), (railx, raily), (railx, y0)]
    for a, b in zip(seg[:-1], seg[1:]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=BLACK, lw=LW, zorder=1))
    arr(ax, railx, y0, x0 - W / 2, y0)
    ax.text((xL + railx) / 2, raily - 0.35, "閉迴路回授：被推 / 變滑即時應變",
            fontproperties=FP, fontsize=11, color=BLACK, ha="center", va="top")
    # 淡灰＝與純 CPG 共用骨架（放上方空白帶，避開箭頭）
    ax.text(P[1][0], 3.3, "淡灰方塊＝與純 CPG 共用骨架",
            fontproperties=FP, fontsize=11, color="#555", ha="center")
    save(fig, "fig2_cpgrl_dataflow.png")


# ─── fig3：MuJoCo × MJX × Brax（三層架構）───
def fig3():
    fig, ax = new_ax()
    w, h = 9.0, 1.7
    xc = 0.0
    layers = [
        (0.0, "Brax PPO（訓練引擎）\nvmap×2048 + JIT + DR + PPO 更新"),
        (-3.0, "Go2 Env（backend=mjx）\nRL動作→CPG→IK→ctrl→mjx.step×5"),
        (-6.0, "MuJoCo 模型（scene_mjx.xml）\nGo2 連桿/關節/接觸/PD"),
    ]
    ax.set_xlim(-8.5, 8.5); ax.set_ylim(-8.7, 8.3)
    title(ax, "MuJoCo × MJX × Brax：如何結合")
    for cy, t in layers:
        rect(ax, xc, cy, t, fs=12.5, w=w, h=h)
    # Brax ↔ Env
    arr(ax, xc - 2.2, 0.0 - h / 2, xc - 2.2, -3.0 + h / 2)
    ax.text(xc - 2.5, -1.5, "step", fontproperties=FP, fontsize=11, ha="right", va="center")
    arr(ax, xc + 2.2, -3.0 + h / 2, xc + 2.2, 0.0 - h / 2)
    ax.text(xc + 2.5, -1.5, "obs, reward", fontproperties=FP, fontsize=11, ha="left", va="center")
    # Env → MuJoCo（橋樑 put_model）
    arr(ax, xc, -3.0 - h / 2, xc, -6.0 + h / 2)
    ax.text(xc + 0.3, -4.5, "mjx.put_model", fontproperties=FPB, fontsize=11.5, ha="left", va="center")
    # 左側層名
    for cy, nm in [(0.0, "訓練層"), (-3.0, "環境層"), (-6.0, "物理層")]:
        ax.text(xc - w / 2 - 0.4, cy, nm, fontproperties=FPB, fontsize=12.5, ha="right", va="center")
    save(fig, "fig3_mujoco_mjx_brax.png")


# ─── fig4：PPO 訓練迴圈（2×2 循環）───
def fig4():
    fig, ax = new_ax()
    w, h = 4.2, 1.9
    d = 3.4
    TL, TR = (-d, d), (d, d)
    BR, BL = (d, -d), (-d, -d)
    ax.set_xlim(-7.5, 7.5); ax.set_ylim(-7.5, 7.5)
    title(ax, "PPO 訓練迴圈")
    rect(ax, *TL, "① 2048 環境\n平行 rollout", fs=12.5, w=w, h=h)
    rect(ax, *TR, "② 獎勵打分", fs=12.5, w=w, h=h)
    rect(ax, *BR, "③ 算 advantage", fs=12.5, w=w, h=h)
    rect(ax, *BL, "④ PPO 更新\n策略網路", fs=12.5, w=w, h=h, shared=True)
    arr(ax, TL[0] + w / 2, TL[1], TR[0] - w / 2, TR[1])          # ①→②
    arr(ax, TR[0], TR[1] - h / 2, BR[0], BR[1] + h / 2)          # ②→③
    arr(ax, BR[0] - w / 2, BR[1], BL[0] + w / 2, BL[1])          # ③→④
    arr(ax, BL[0], BL[1] + h / 2, TL[0], TL[1] - h / 2)          # ④→①
    ax.text(0, 0, "重複至\n1.2 億步\nreward 收斂", fontproperties=FPB, fontsize=12.5,
            ha="center", va="center", color=BLACK)
    save(fig, "fig4_ppo_loop.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("done", OUT)
