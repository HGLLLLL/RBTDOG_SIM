# Session 紀錄與 Task3 交接

> 紀錄日期：2026-07-01
> 用途：記錄本次 session 完成的內容，供**明天繼續優化 task3** 時快速接手。

---

## 0. 一分鐘接手須知（明天先看這段）

- **專案目標**：MuJoCo 中做宇樹機器狗運動模擬 + 感測器。三階段任務**已全部完成**，目前在**優化 task3**（IMU/compass 走路 demo）。
- **環境**：conda env **`rbtdog`**（Python 3.11, mujoco 3.10.0, numpy, matplotlib, imageio, imageio-ffmpeg, pillow）。
  ```bash
  mamba activate rbtdog
  cd /home/huang/rbtdog_sim/task3
  ```
- **跑任何會渲染的腳本要加** `MUJOCO_GL=egl`（Intel 內顯離屏渲染）。例：
  ```bash
  MUJOCO_GL=egl python walk_line.py
  ```
- **不要用 ROS**（初期決定，見 `01_環境架構研究報告.md`）。走路控制器是**自寫 CPG**（不是 MJPC、不是 RL），原因見下方 §3。
- **硬碟已用 90%（剩 27G）**、**core dump 未清**（見 §6 待辦）。

---

## 1. 三階段任務狀態

| 任務 | 狀態 | 主要產物 |
|------|------|----------|
| 1. 環境建置（MuJoCo，決定不用 ROS） | ✅ | `01_環境架構研究報告.md`、env `rbtdog`、`go2_task1.png` |
| 2. 機器狗走路（MJPC 互動） | ✅ | `mujoco_mpc/build/bin/mjpc`、`02_MJPC建置問題與解法.md` |
| 3. IMU/compass 航向校正 | ✅（優化中） | `task3/`（兩個 demo：正方形 + 走直線） |

---

## 2. 專案檔案地圖

```
/home/huang/rbtdog_sim/
├── 01_環境架構研究報告.md        架構決策、硬體評估、MuJoCo 教學、三任務地圖
├── 02_MJPC建置問題與解法.md      MJPC 在 Arch 編譯的 3 個雷與解法
├── 03_Session紀錄與Task3交接.md  ← 本檔
├── go2_task1.png                 任務1 驗收圖（Go2 站立）
├── mujoco_menagerie/             宇樹模型庫（含 unitree_go2 等）
├── mujoco_mpc/                   MJPC 原始碼+build（build/bin/mjpc 可執行）
└── task3/
    ├── go2_model.py              用 MjSpec 建含 9 軸 IMU 的 Go2 模型
    ├── go2_gait.py               ★核心：CPG trot 控制器（PD+雅可比IK+轉彎+羅盤解算）
    ├── walk_proto.py             早期走路調參原型
    ├── square_mission.py         Demo A：正方形（5圈），輸出 square_result.png
    ├── render_mission.py         Demo A 影片，輸出 square_video.mp4
    ├── walk_line.py              Demo B：走直線40m+干擾，輸出 line_video.mp4 + line_result.png
    ├── README.md                 task3 完整說明
    ├── square_result.png / square_video.mp4 / traj1.npy / traj2.npy
    └── line_result.png / line_video.mp4
```

---

## 3. Task3 關鍵設計決定（為什麼這樣做）

- **走路控制器用自寫 CPG，不用 MJPC / RL**：
  - MJPC 內部用**真實狀態**做全狀態回授 → 開環版也不會漂，凸顯不出感測器價值。
  - RL 走路策略**本身就需要 IMU**（陀螺+重力）才能站穩 → 做不出「零感測器」第一版。
  - CPG 開環 → 第一版真的零感測器，最貼合「有/無 IMU」對比。
- **Go2 是力矩致動器** → 軟體 PD 控制（`tau = kp*(qdes-q) - kd*qd`）。
- **CPG trot**：對角腿同相；站立相腳貼地後推、擺動相抬起前移；腳掌橢圓軌跡用**站姿數值雅可比**反解成 thigh/calf 角度。
- **轉彎**：左右腿差動步幅（turn_gain）。
- **9 軸 IMU**：`accelerometer + gyro + magnetometer` 掛在機身內建 `imu` site，用 **MjSpec** 加（跨目錄 include 會壞 meshdir，故用 MjSpec）。
- **羅盤航向**：`yaw = atan2(-mag_x, -mag_y)`（MuJoCo 預設磁場 (0,-0.5,0)），與真值誤差 0.02°。
- **重要限制（已與使用者確認接受）**：羅盤只鎖「方向」不鎖「位置」→ 有 sensor 的狗校正航向、沿平行線續走，**不會橫向壓回中心線**（要壓回需額外位置感測器=GPS）。

---

## 4. 目前兩個 Demo 的成果

### Demo A：走正方形（10m×右轉90°×5圈）— `square_mission.py`
| | 閉合誤差（終點回起點） |
|---|---|
| V1 無IMU（開環） | **10.46 m** |
| V2 IMU羅盤（閉環） | **0.14 m**（改善 75×）|
- 跑法：`MUJOCO_GL=egl python square_mission.py 5`（圖）、`python render_mission.py 5`（影片）

### Demo B：走直線 40m + 外部干擾 — `walk_line.py`
| | 終點橫向偏移 |
|---|---|
| V1 無IMU（開環） | **y=+9.6 m**（被推歪回不來）|
| V2 IMU羅盤（保持朝向）| **y=+0.5 m**（每次被推即校正）|
- 目標視覺 = 浮在狗上方、恆指 +x 的紅箭頭。
- 干擾：5 次交替左右側推(±35N)+yaw(±10Nm)，`walk_line.py` 的 `PUSHES`。
- 影片 real-time 並排，相機狗正後方45°俯視、方位角固定朝+x、隨橫偏輕微縮放。
- 跑法：`MUJOCO_GL=egl python walk_line.py`

---

## 5. 已知雷 / 調參教訓（明天優化時避免重踩）

**渲染**
- MuJoCo 離屏 framebuffer 預設上限 **640×480**（要更大需在 XML `<visual><global offwidth=.. offheight=..>`）。
- mp4 需 `imageio-ffmpeg`（已裝）；GIF 用 `iio.mimsave(..., duration=, loop=0)`。
- 腳本**勿命名 `inspect.py`**（會蓋標準庫 → 詭異 import 錯）。

**相機（Demo B 調過的）**
- `azimuth=0, elevation=-45` = 狗正後方45°俯視、+x 朝畫面上方。
- lookat **固定 y=0 + 強縮放** → 漂移的狗會變微小點甚至出框（不好）。
- 改成 **lookat 跟隨狗、方位角固定朝 +x**（狗歪時身體在框內轉向、與箭頭錯開）→ 最清楚。distance 隨 |y| 輕微放大、上限 8、平滑。

**干擾（Demo B 調過的）**
- 用 `d.xfrc_applied[base_bid] = [fx,fy,fz, tx,ty,tz]`（力+力矩）；base body id=1。
- **同向**推力 → 航向誤差累積、V1 爆走飛出畫面（y>30）。
- **交替**方向推力 → V1 留在畫面內、又能呈現「被推不自修」。tz≈10、fy≈35 是甜蜜點。

**模擬速度**
- dt=0.002；純步進約 16000 步/秒（≈32× 實時）。real-time 影片要 fps 對齊模擬時間（每幀前進 1/fps 秒）。
- 閉環轉彎迴圈**務必加超時上限**，否則羅盤雜訊會讓收斂條件永遠不滿足 → 無限迴圈（曾卡 26 分鐘）。

---

## 6. 待辦 / 未完成事項

- [ ] **清 core dump**（需 sudo，非互動 shell 無法代跑）：
      `sudo rm -f /var/lib/systemd/coredump/core.mjpc.*.zst`（約 270MB）。
- [ ] 硬碟已用 90%，注意空間；影片檔 line_video.mp4 有 17MB。
- [ ] MJPC 互動 GUI 啟動要用 `MESA_GLTHREAD=false ./build/bin/mjpc`（否則 KDE Wayland 下切視窗會崩，見 `02_...md` 雷⑤）。sudo 密碼問題使用者那邊待確認。

---

## 7. 明天可優化 Task3 的方向（候選清單）

> 以下是可選的優化點，明天可挑要做的。

**視覺 / 影片**
- 影片加「即時偏移量數字條 / 目標線殘影」讓漂移更直觀。
- Demo B 影片偏長(57s real-time)；可加速版或縮短距離另出一支。
- 在地面畫目標路徑參考線（使用者 Demo B 選了箭頭取代線，但可加淡色地面線輔助）。
- 兩隻狗改成同一場景並排（目前是兩個獨立 sim 拼接），或加第三人稱環繞鏡頭。

**控制 / 感測器真實度**
- **感測器融合**：gyro 積分（會漂）+ magnetometer 互補濾波/卡爾曼，展示「純陀螺漂移 vs 融合穩定」。
- **傾斜補償羅盤**（tilt compensation）：用 accelerometer 補償 pitch/roll，讓羅盤在身體晃動時更準（目前狗夠平所以還好）。
- 給磁力計加**磁偏/硬鐵誤差**，展示校正前後。
- **位置漂移展示**：把 V1 的「里程」也改成加速度計二次積分，展示 IMU dead-reckoning 的位置漂移。
- 走路步態優化：目前 0.68 m/s、pitch 可到 12°；可調更穩或更快。

**任務**
- 加「羅盤+里程」做真正的位置閉環（走回中心線），對比純羅盤。
- 不平地形（menagerie 有地形工具 / MJPC 有 hill task）上測感測器價值。

---

## 8. 快速指令備忘

```bash
# 啟動環境
mamba activate rbtdog && cd /home/huang/rbtdog_sim/task3

# Demo A（正方形）
MUJOCO_GL=egl python square_mission.py 5     # 圖 square_result.png + 誤差
MUJOCO_GL=egl python render_mission.py 5     # 影片 square_video.mp4

# Demo B（走直線+干擾）
MUJOCO_GL=egl python walk_line.py            # 影片 line_video.mp4 + 圖 line_result.png

# 看走路原型/調參
MUJOCO_GL=egl python walk_proto.py --stride 0.24 --lift 0.13 --freq 3.5 --secs 6

# MJPC 互動 GUI（任務2）
MESA_GLTHREAD=false /home/huang/rbtdog_sim/mujoco_mpc/build/bin/mjpc
```
