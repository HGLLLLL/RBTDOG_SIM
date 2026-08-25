#!/usr/bin/env python3
"""產生「三款四足機器狗比較」論文風圖表（SVG → PNG）。

風格對齊 task6/docs/D1EDU_圖*_論文風.png：襯線中文、細黑框、白／淺灰填色、
等寬字體標示程式識別字、底部圖說。
"""
from pathlib import Path

W = 1840
SERIF = "Noto Serif CJK TC"
MONO = "Noto Sans Mono CJK TC"

INK = "#1a1a1a"
GREY = "#6b6b6b"
LINE = "#2b2b2b"
FILL_L = "#f2f1ee"   # 淺灰
FILL_M = "#dcdad5"   # 中灰
FILL_D = "#bcb9b1"   # 深灰
WHITE = "#ffffff"

out = []
A = out.append



def wrap_balanced(s, maxch):
    """把字串折成長度盡量相等的數行，並避免行首出現收尾標點。"""
    n = max(1, -(-len(s) // maxch))          # 需要幾行
    per = -(-len(s) // n)                    # 每行字數（均衡）
    lines, i = [], 0
    while i < len(s):
        j = min(len(s), i + per)
        # 不要讓下一行以收尾標點開頭 —— 把它拉到本行末尾
        while j < len(s) and s[j] in "）)。，、；：":
            j += 1
        lines.append(s[i:j])
        i = j
    return lines


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=15, font=SERIF, weight="normal", anchor="start",
         fill=INK, ls=None):
    extra = f' letter-spacing="{ls}"' if ls else ""
    A(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
      f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{extra}>{esc(s)}</text>')


def box(x, y, w, h, fill=WHITE, stroke=LINE, sw=1.2, rx=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, sw=1.0, stroke=LINE, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
      f'stroke-width="{sw}"{d}/>')


# ------------------------------------------------------------------ 版面
L = 46                 # 左邊界
GUT = 128              # 左側區塊標籤欄寬
COLW, GAP = 508, 26
CX = [L + GUT + i * (COLW + GAP) for i in range(3)]

# 高度最後才決定（見檔尾），先放佔位符
A("@@SVGOPEN@@")

# ------------------------------------------------------------------ 標題
text(L, 42, "智元三款四足機器狗：運算平台、SDK 能力與單顆馬達控制可行性", 23, weight="bold")
text(L, 68, "硬體規格為實機實測（cat /proc/device-tree/model、lscpu、free -h、dmesg）；"
            "SDK 與控制介面為官方文件＋二進位符號分析＋實機偵察", 13.5, fill=GREY)

# ------------------------------------------------------------------ 欄頭
HY = 90
HH = 76
heads = [
    ("D1 EDU", "輪足・小狗", "16 軸（12 腿＋4 輪）　20.6 kg"),
    ("D1 Max", "輪足・中狗", "16 軸（12 腿＋4 輪）　41 kg"),
    ("D1 MaxPro", "點足・大狗", "12 軸　68 kg"),
]
for i, (nm, kind, spec) in enumerate(heads):
    x = CX[i]
    box(x, HY, COLW, HH, fill=FILL_M, sw=1.4)
    text(x + 16, HY + 30, nm, 21, weight="bold")
    text(x + 16 + len(nm) * 12.5 + 24, HY + 30, kind, 15, fill=GREY)
    text(x + 16, HY + 57, spec, 15)

# ------------------------------------------------------------------ 甲：運算平台
SY = HY + HH + 22
text(L, SY + 22, "（甲）", 16.5, weight="bold")
text(L, SY + 48, "運算平台", 17.5, weight="bold")
text(L, SY + 72, "實機實測", 12.5, fill=GREY)
line(L, SY - 8, W - L, SY - 8, sw=1.4)

# 每欄的板卡描述：(板名, 副標, 明細列, 是否高算力板)
plats = [
    [("Firefly AIO-3588SJD4", "Rockchip RK3588・單板",
      ["CPU　4×A76 ＋ 4×A55", "RAM　7.7 GiB（無 swap）",
       "GPU　Mali-G610 MP4", "NPU　RKNPU driver 0.9.2",
       "OS　Ubuntu 22.04.4・eMMC 64 GB"], True)],
    [("Firefly AIO-3588SJD4", "Rockchip RK3588・運控",
      ["CPU　4×A76 ＋ 4×A55", "RAM　7.7 GiB",
       "GPU　Mali-G610 MP4", "NPU　RKNPU 0.9.2"], True),
     ("NVIDIA Jetson Orin NX 16GB", "應用・建圖／定位／導航",
      ["CPU　8×Cortex-A78AE　1.984 GHz", "RAM　15 GiB",
       "GPU　NVIDIA ga10b", "DLA　NVDLA0／NVDLA1"], True)],
    [("Firefly AIO-3588SJD4", "Rockchip RK3588・主控板",
      ["CPU　4×A76 ＋ 4×A55", "RAM　7.7 GiB（Swap 0）",
       "GPU　Mali-G610 MP4", "NPU　RKNPU 0.9.2・librknnrt／librga",
       "OS　Ubuntu 22.04.5・kernel 5.10-rt"], True),
     ("轉發板", "CAN／關節訊號轉發",
      ["非第二塊高算力板"], False)],
]

def board_h(rows):
    return 30 + 24 + len(rows) * 24 + 14


col_h = [sum(board_h(r) + 10 for (_, _, r, _) in bs) for bs in plats]
SH = max(col_h) + 38          # 內容 + 判定列

for i, boards in enumerate(plats):
    x = CX[i]
    y = SY + 6
    for (nm, sub, rows, big) in boards:
        bh = board_h(rows)
        box(x, y, COLW, bh, fill=FILL_L if big else WHITE,
            sw=1.4 if big else 1.0, dash=None if big else "5 4")
        mono = ("AIO" in nm) or ("Jetson" in nm)
        text(x + 16, y + 28, nm, 16.5, font=MONO if mono else SERIF, weight="bold")
        text(x + 16, y + 50, sub, 13.5, fill=GREY)
        for k, r in enumerate(rows):
            text(x + 16, y + 76 + k * 24, r, 14.5)
        y += bh + 10
    verdict = ["單一算力平台", "★ 兩塊獨立高算力板", "主控板＋轉發板（非第二算力板）"][i]
    text(x + 16, SY + SH + 6, verdict, 15,
         weight="bold" if i == 1 else "normal",
         fill=INK if i == 1 else GREY)

# ------------------------------------------------------------------ 乙：SDK 可寫深度
TY = SY + SH + 30
line(L, TY - 10, W - L, TY - 10, sw=1.4)
text(L, TY + 22, "（乙）", 16.5, weight="bold")
text(L, TY + 48, "SDK", 17.5, weight="bold")
text(L, TY + 72, "可寫深度", 17.5, weight="bold")
text(L, TY + 96, "官方介面", 12.5, fill=GREY)

levels = [
    "L4　單顆馬達　角度・角速度・力矩・Kp・Kd",
    "L3　關節層　位置／速度",
    "L2　姿態動作　站立・趴下・匍匐・爬階",
    "L1　整機速度　move(vx, vy, yaw)",
]
# 官方 SDK 能寫到第幾層（1-based，由 L1 往上算）
reach = [2, 2, 4]
LH = 40
for i in range(3):
    x = CX[i]
    for k, lab in enumerate(levels):
        yy = TY + 6 + k * LH
        depth = 4 - k                        # L4→4, L1→1
        ok = depth <= reach[i]
        box(x, yy, COLW, LH - 7, fill=FILL_D if ok else WHITE,
            sw=1.2 if ok else 0.9, dash=None if ok else "4 4")
        text(x + 14, yy + 22, lab, 14.5, fill=INK if ok else GREY)
    # 官方止步線（標籤放欄內右側，避免跨欄被裁切）
    stop_y = TY + 6 + (4 - reach[i]) * LH
    if reach[i] < 4:
        line(x, stop_y, x + COLW, stop_y, sw=2.6)
        lbl = "官方 SDK 止步於此"
        lw = len(lbl) * 13 + 12
        box(x + COLW - 8 - lw, stop_y - 25, lw, 21, fill=WHITE, stroke=WHITE, sw=0, rx=0)
        text(x + COLW - 12, stop_y - 9, lbl, 13.5, anchor="end", weight="bold")

# 讀取能力（三台都全開）
ry = TY + 6 + 4 * LH + 10
for i in range(3):
    x = CX[i]
    rd = ["讀取　16 軸狀態・IMU",
          "讀取　16 軸狀態＋關節溫度・IMU・光達・導航",
          "讀取　12 軸狀態・IMU・里程計・點雲"][i]
    box(x, ry, COLW, 32, fill=WHITE, sw=0.9)
    text(x + 14, ry + 22, rd, 14, fill=GREY)

sdkname = ["mc_sdk::zsl_1w::HighLevel　UDP 50 Hz",
           "robot_sdk::SDKClient　UDP :8082",
           "高層 TCP ＋ 底層 ROS 1 話題"]
for i in range(3):
    text(CX[i] + 14, ry + 58, sdkname[i], 13, font=MONO, fill=GREY)

# ------------------------------------------------------------------ 丙：單顆馬達控制
VY = ry + 74
line(L, VY - 10, W - L, VY - 10, sw=1.4)
text(L, VY + 22, "（丙）", 16.5, weight="bold")
text(L, VY + 48, "單顆馬達", 17.5, weight="bold")
text(L, VY + 72, "控制", 17.5, weight="bold")

cards = [
    # (官方態度, 判定, 路徑, 補充)
    ("官方不提供", "做得到", "/spline_shm　共享記憶體", ""),
    ("官方不提供", "極可能做得到", "/dev/shm/joint_cmd　16 軸 × 5 欄位", ""),
    ("★ 官方提供", "做得到", "ROS 1 話題　rt/lowcmd", "底層 SDK 須安裝 ROS 1"),
]
CH = 118
for i, (off, verd, path, note) in enumerate(cards):
    x = CX[i]
    strong = (i == 2)
    box(x, VY, COLW, CH, fill=FILL_L if strong else WHITE, sw=1.6 if strong else 1.2)
    text(x + 16, VY + 28, off, 15.5, weight="bold" if strong else "normal",
         fill=INK if strong else GREY)
    text(x + 16, VY + 60, verd, 21, weight="bold")
    text(x + 16, VY + 88, path, 14.5, font=MONO)
    if note:
        text(x + COLW - 16, VY + 28, note, 13, anchor="end", fill=GREY)

# ------------------------------------------------------------------ 圖說
cy = VY + CH + 46
line(L, cy - 20, W - L, cy - 20, sw=1.4)
text(L, cy, "圖 4 · 三款機型比較。", 14.5, weight="bold")
text(L + 152, cy,
     "（甲）運算平台：D1 Max 是三台中唯一的雙高算力板；D1 MaxPro 雖是大狗，"
     "算力平台與小狗 D1 EDU 同為單顆 RK3588。", 14.5)
text(L, cy + 23,
     "（乙）三台的「讀」幾乎全開，差別在「寫」——D1 EDU 與 D1 Max 的官方 SDK 只到姿態動作層，"
     "送不了關節角度或力矩；只有 D1 MaxPro 開放到單顆馬達。", 14.5)
text(L, cy + 46,
     "（丙）D1 EDU 與 D1 Max 官方雖不提供，但機上存在共享記憶體介面可繞過 SDK："
     "D1 EDU 已實機驗證三種控制模式，D1 Max 結構已解出、僅差寫入未測。", 14.5)
text(L, cy + 72,
     "註：D1 MaxPro 官方文件的 SSH 帳號為 jetson，但實測主控板為 Firefly RK3588，"
     "兩者不一致，機型批次差異待向原廠確認。", 12.5, fill=GREY)

A("</svg>")

H = cy + 96                      # 依實際內容裁切
out[0] = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
          f'viewBox="0 0 {W} {H}">\n'
          f'<rect width="{W}" height="{H}" fill="{WHITE}"/>')
svg = "\n".join(out)
p = Path(__file__).with_name("fig_dogs.svg")
p.write_text(svg, encoding="utf-8")
print("wrote", p, len(svg), "bytes")
