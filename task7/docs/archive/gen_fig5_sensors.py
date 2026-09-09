#!/usr/bin/env python3
"""產生「三款四足機器狗：感測器可讀資料與格式」論文風圖表（SVG → PNG）。

風格對齊 `gen_fig4.py` / task6 的 `D1EDU_圖*_論文風.png`：
襯線中文、細黑框、白／淺灰填色、等寬字體標示程式識別字、底部圖說。

★ 每一格的來源都可追：
  - D1 Max 高層 SDK  → `docs/D1Max_控制方式調查_2026-08-25.md` §4.3
  - D1 Max 底層 SHM  → `realbot/shm_io.py`（實機解碼＋驗證）
  - D1 Max ROS2 topic→ 同上 §7.3（官方文件列出）
  - D1 EDU 底層 SHM  → task6 `realbot/L6_imu_probe.py`、`D1EDU_輪足_lowlevel_調查與實測指南.md`
  - 三機型對照       → `docs/三機型對照表_2026-08-25.md` §2

用法：
    python3 task7/docs/gen_fig5_sensors.py
"""
from pathlib import Path

W = 1840
SERIF = "Noto Serif CJK TC"
MONO = "Noto Sans Mono CJK TC"

INK = "#1a1a1a"
GREY = "#6b6b6b"
LINE = "#2b2b2b"
FILL_L = "#f2f1ee"
FILL_M = "#dcdad5"
FILL_D = "#bcb9b1"
WHITE = "#ffffff"

out = []
A = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
L = 46
GUT = 128
COLW, GAP = 508, 26
CX = [L + GUT + i * (COLW + GAP) for i in range(3)]

A("@@SVGOPEN@@")

text(L, 42, "智元三款四足機器狗：感測器可讀資料與格式", 23, weight="bold")
text(L, 68, "底層欄位為實機共享記憶體解碼並驗證；高層 SDK 回呼與 ROS 話題為官方文件；"
            "★ 標記處為實機量測值", 13.5, fill=GREY)

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


def block(x, y, w, title, sub, rows, fill=WHITE, dash=None, star=False):
    """一個資料區塊：標題（等寬）＋副標＋明細列。回傳高度。"""
    h = 44 + (22 if sub else 0) + len(rows) * 23 + 12
    box(x, y, w, h, fill=fill, dash=dash)
    text(x + 14, y + 27, title, 15.5, font=MONO, weight="bold")
    if star:
        text(x + w - 14, y + 27, "★ 實機驗證", 12.5, anchor="end", fill=GREY)
    yy = y + 27
    if sub:
        yy += 21
        text(x + 14, yy, sub, 12.5, fill=GREY)
    for r in rows:
        yy += 23
        text(x + 20, yy, r, 13.5, font=MONO if r.startswith("·") is False else SERIF)
    return h


# ================================================================== 甲：關節
SY = HY + HH + 22
line(L, SY - 8, W - L, SY - 8, sw=1.4)
text(L, SY + 22, "（甲）", 16.5, weight="bold")
text(L, SY + 48, "關節狀態", 17.5, weight="bold")
text(L, SY + 72, "16／12 軸", 12.5, fill=GREY)

joint = [
    dict(title="/dev/shm/spline_shm", sub="共享記憶體・唯讀",
         rows=["16 軸　position / velocity / effort",
               "float32　packed struct",
               "（無溫度、無電壓）"], star=True),
    dict(title="/dev/shm/joint_state", sub="共享記憶體・1 kHz・唯讀不需 root",
         rows=["16 軸 × 5 欄位　全部 float64",
               "position  velocity  effort",
               "temp_C    voltage_V",
               "base=752  stride=120  name[64]"], star=True),
    dict(title="OnJointStateData ／ rt/lowstate", sub="高層 SDK 回呼 ／ ROS 1 話題",
         rows=["12 軸　names / positions /",
               "        velocities / efforts",
               "（底層 SDK 須安裝 ROS 1）"]),
]
jh = 0
for i, b in enumerate(joint):
    jh = max(jh, block(CX[i], SY, COLW, b["title"], b["sub"], b["rows"],
                       star=b.get("star", False)))
text(CX[0], SY + jh + 20, "task6 實機唯讀偵察確認可用", 13, fill=GREY)
text(CX[1], SY + jh + 20, "★ 每關節溫度／電壓是三台唯一　★ 實測 1 kHz", 13, fill=GREY)
text(CX[2], SY + jh + 20, "底層須 ROS 1，本專案未實測", 13, fill=GREY)

# ================================================================== 乙：IMU
SY2 = SY + jh + 40
line(L, SY2 - 8, W - L, SY2 - 8, sw=1.4)
text(L, SY2 + 22, "（乙）", 16.5, weight="bold")
text(L, SY2 + 48, "IMU", 17.5, weight="bold")
text(L, SY2 + 72, "姿態／慣性", 12.5, fill=GREY)

imu = [
    dict(title="/dev/shm/imu_shm", sub="48 bytes・packed",
         rows=["ts       size_t   奈秒",
               "acc[3]   float32  m/s²",
               "gyro[3]  float32  rad/s",
               "q[4]     float32  w,x,y,z"], star=True),
    dict(title="/dev/shm/imu_central", sub="共享記憶體・數值自 byte 824 起",
         rows=["acc[3]   float64  m/s²",
               "gyro[3]  float64  rad/s",
               "quat[4]  float64  ★ xyzw",
               "＝ 10 個 f64・6 軸＋四元數"], star=True),
    dict(title="OnImuData（含里程計）", sub="高層 SDK 回呼",
         rows=["acc / gyro / 四元數",
               "另有里程計（odometry）",
               "—"]),
]
ih = 0
for i, b in enumerate(imu):
    ih = max(ih, block(CX[i], SY2, COLW, b["title"], b["sub"], b["rows"],
                       star=b.get("star", False)))

# D1 Max 多一塊：高層 SDK 的兩個 IMU 來源
extra_y = SY2 + ih + 10
eh = block(CX[1], extra_y, COLW, "OnImuData ／ OnMcData", "高層 SDK・預設關閉，須自行開啟",
           ["OnImuData  acc/gyro/quat_x,y,z,w   0–100 Hz",
            "OnMcData   quat[4]・世界座標位置/速度/",
            "           角速度・機體速度・ns 時戳　50 Hz"],
           fill=FILL_L)

warn_y = extra_y + eh + 26
text(CX[0], SY2 + ih + 20, "6 軸 IMU ＋ 四元數（wxyz）", 13, fill=GREY)
text(CX[2], SY2 + ih + 20, "6 軸 IMU ＋ 四元數 ＋ 里程計", 13, fill=GREY)

box(CX[1], warn_y, COLW, 104, fill=WHITE, dash="5 4")
text(CX[1] + 14, warn_y + 25, "⚠ 四元數順序官方自相矛盾 —— 我們採 xyzw", 14, weight="bold")
text(CX[1] + 14, warn_y + 47,
     "ImuData 欄位名 xyzw、MotionData 註解寫 wxyz。", 12.5, fill=GREY)
text(CX[1] + 14, warn_y + 66,
     "★ M7/M8 十餘趟 roll/pitch 均合理，且重力對齊後支撐腳歸零", 12.5, fill=GREY)
text(CX[1] + 14, warn_y + 87,
     "旁證充分，惟刻意平放的 xyzw／wxyz 對照尚未執行", 12.5, fill=GREY)

# ================================================================== 丙：外感測
SY3 = warn_y + 104 + 34
line(L, SY3 - 8, W - L, SY3 - 8, sw=1.4)
text(L, SY3 + 22, "（丙）", 16.5, weight="bold")
text(L, SY3 + 48, "外部感測", 17.5, weight="bold")
text(L, SY3 + 72, "光達／影像", 12.5, fill=GREY)

ext = [
    [("光達／點雲", "✗　無"),
     ("影像", "✓　RTSP"),
     ("超音波・RTK", "✗　無")],
    [("光達／點雲", "✓　前後各 1 顆 96 線（Airy）"),
     ("", "/front_lidar  /rear_lidar"),
     ("", "PointCloud2　10 Hz　best_effort"),
     ("光達內建 IMU", "/front_lidar/imu  /rear_lidar/imu"),
     ("影像", "✓　RTSP 前後 800 萬畫素廣角"),
     ("超音波 ×2", "/uss_driver/uss_{left,right}/range  10 Hz"),
     ("RTK", "/rtk_pvh")],
    [("光達／點雲", "✓　96 線"),
     ("影像", "✓　RTSP"),
     ("導航／建圖", "✓　官方提供")],
]
eh2 = 0
for i, rows in enumerate(ext):
    h = 20 + len(rows) * 25 + 12
    eh2 = max(eh2, h)
    box(CX[i], SY3, COLW, h)
    yy = SY3 + 12
    for lab, val in rows:
        yy += 25
        if lab:
            text(CX[i] + 14, yy, lab, 13.5)
        text(CX[i] + 168, yy, val, 13, font=MONO if val.startswith("/") else SERIF)

# 感測器安裝位置（只有 D1 Max 有實測）
pos_y = SY3 + eh2 + 12
ph = block(CX[1], pos_y, COLW, "感測器安裝位置", "相對 BASE 原點（m）・URDF＋手冊交叉核對",
           ["IMU_ICM42688 運控   (0, 0, 0.0362)",
            "IMU_LUA300C  車規   (0, 0, 0.0569)",
            "光達 前/後          (±0.4043, 0, −0.0377)",
            "相機 前/後          (±0.4123, 0,  0.0378)",
            "超音波              (0.1792, ∓0.1002, 0.05)"],
           fill=FILL_L)
text(CX[1] + 14, pos_y + ph + 20,
     "⚠ URDF 的 IMU_LUA300C 寫 0.00569，手冊是 56.9 mm —— URDF 少一個零", 12.5, fill=GREY)

# ================================================================== 圖說
CAP = pos_y + ph + 42
line(L, CAP, W - L, CAP, sw=1.4)
cap = [
    ("圖 5・三款機型的感測器可讀資料。", True),
    ("（甲）關節：三台都讀得到角度／角速度／力矩。**D1 Max 是唯一附每關節溫度與電壓的**，"
     "且底層 joint_state 實測更新率 1 kHz、唯讀不需 root。", False),
    ("（乙）IMU：三台都是 6 軸（加速度 3＋角速度 3）＋四元數。"
     "D1 EDU 底層是 float32、D1 Max 底層是 float64；"
     "**D1 Max 機上其實有兩顆 IMU**（運控用 ICM42688、車規級 LUA300C）。", False),
    ("（丙）外部感測：D1 EDU 沒有光達；D1 Max 與 MaxPro 都是 96 線，"
     "但 D1 Max 是**前後各一顆**並附超音波與 RTK。", False),
]
yy = CAP + 26
for s, bold in cap:
    text(L, yy, s.replace("**", ""), 14, weight="bold" if bold else "normal")
    yy += 24
text(L, yy + 8, "註：四元數順序在 D1 EDU 與 D1 Max 上都踩過坑 —— 官方文件互相矛盾。D1 Max 目前採 xyzw，"
                "十餘趟實機旁證一致，但刻意平放的 xyzw／wxyz 對照實驗尚未執行。", 12.5, fill=GREY)

H = yy + 34
svg = "\n".join(out).replace(
    "@@SVGOPEN@@",
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="white"/>')
svg += "\n</svg>\n"

d = Path(__file__).resolve().parent
p = d / "圖5_三機型感測器_論文風.svg"
p.write_text(svg, encoding="utf-8")
print(f"✅ {p}　{W}×{H}")
