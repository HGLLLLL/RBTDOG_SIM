# D1 Max 機器狗資訊（整合參考文件）

- 整理日期：2026-09-09
- 對象：智元 AgiBot **D1 Max** 輪足四足機器狗，SDK 代號 `zsm-1w`，運控機型代號 `ZGWS` / `zg_wheels` / `zsm`
- 性質：**參考手冊，不是實驗報告。** 只放結論、數字、慣例、坑；實驗過程與敘事請看原始文件。

## 這份怎麼用

- 要查一個數字（增益、限位、offset、摩擦、延遲）→ 直接看對應章節的表。
- 每個數字後面括號裡是**來源文件與日期**。同一件事有兩個版本的，兩個都寫，並標「**矛盾，以 X 為準**」。
- 沒有標「實測」的，一律當作**文件值或推定值**，上機前要自己驗。
- 程式碼裡的單一事實來源仍然是 `task7/realbot/coord.py`（換算／限位／姿勢）與
  `task7/realbot/shm_io.py`（SHM 佈局）。**本文件與程式碼衝突時，以程式碼為準**，並回頭修這份。

### 來源清單（原始文件已移到 `task7/docs/archive/`）

| 簡稱（本文引用用） | 原始檔名 | 日期 |
|---|---|---|
| 控制方式調查 | `D1Max_控制方式調查_2026-08-25.md` | 2026-08-25 |
| MATRiX解包 | `D1Max_原廠運控參數_MATRiX解包_2026-08-25.md` | 2026-08-25 |
| 三機型對照 | `三機型對照表_2026-08-25.md` | 2026-08-25 |
| 偵察一 | `實機偵察結果_第一趟_2026-08-25.md` | 2026-08-25 |
| 偵察二 | `實機偵察結果_第二趟_2026-08-25.md` | 2026-08-25 |
| 寫入三 | `實機寫入結果_第三趟_2026-08-25.md` | 2026-08-25 |
| 單顆馬達 | `實機單顆馬達驅動結果_2026-08-25.md` | 2026-08-25 |
| 四輪驅動 | `實機四輪驅動結果_2026-08-25.md` | 2026-08-25 |
| 座標驗證 | `座標換算式驗證結果_2026-08-25.md` | 2026-08-25 |
| 腿關節16顆 | `實機腿關節與16顆全控結果_2026-08-26.md` | 2026-08-26 |
| 原廠站立 | `原廠站立實測_全程錄製_2026-08-26.md` | 2026-08-26（§5.5／§6 末為 2026-08-27 補做） |
| obs盤點 | `H_實機obs盤點_2026-09-08.md`（**未移入 archive**，仍在 `docs/`） | 2026-09-08 |
| 姿勢設計 | `腿關節與姿勢控制_設計_2026-08-26.md`（已移入 archive） | 2026-08-26 |

---

## 1. 機型與規格

| 項目 | 值 | 來源 |
|---|---|---|
| 整機重量 | **41 kg**（含電池）；URDF 慣量加總 **41.045 kg**，與規格書吻合 | 控制方式調查 |
| 站立尺寸 | 930 × 480 × 585 mm | 控制方式調查 |
| 關節數 | **16**（每腿 abad / hip / knee + 1 顆輪轂馬達） | 控制方式調查 |
| 腿關節最大扭矩 | **150 N·m** | 控制方式調查 |
| 腿關節最大轉速 | **190 RPM ≈ 19.9 rad/s** | 控制方式調查 |
| 輪馬達 | **33 N·m** / 1200 RPM ≈ **125.7 rad/s** | 控制方式調查 |
| 輪半徑 | **9 cm（0.09 m）** | 控制方式調查 |
| 最大速度 | 規格書 **6 m/s** | 控制方式調查 |
| 穩定負載 | 30 kg | 控制方式調查 |
| 爬坡 / 台階 | 45° / 連續 25 cm、極限跨越 80 cm | 控制方式調查 |
| 防護 / 溫度 | IP67、−20 ~ 55 °C | 控制方式調查 |
| 電池 | 504.9 Wh × 2，熱插拔快換；**實測母線 53–54 V**（雙電池 48 V 系統），剛充飽時量到 **61.0 V** | 控制方式調查／偵察二／寫入三 |
| 感測 | 前後各 1 顆 96 線光達（Airy）、前後 800 萬廣角相機、超音波 ×2、車規 IMU、RTK、補光燈 ×4 | 控制方式調查 |
| 輪足快換 | 支援輪足 ↔ 點足快換 | 控制方式調查 |

**⚠️ 三個「最大速度」管的是不同東西，別混用**（控制方式調查／MATRiX解包）：
規格書標稱峰值 **6 m/s**、SDK 高速檔指令上限 **3 m/s**、MPC 內部限值 `cmpc_x_vel` **2.0 m/s**。

### 1.1 質量分佈（URDF 解析，控制方式調查）

BASE 20.25 kg；每腿 ABAD 0.2696 + HIP 2.6525 + KNEE 1.5888 + FOOT 0.688 = **5.199 kg**，四腿 20.79 kg。

### 1.2 幾何（控制方式調查／`coord.py`）

| 項目 | 值 |
|---|---|
| 髖位置 x | URDF **±0.272 m**；MJCF body pos **±0.2698 m**；設定檔 `body_half_length` **0.27 m**（三者一致） |
| 髖位置 y | **±0.065 m**（`y = −0.065` 右、`y = +0.065` 左） |
| 大腿 / 小腿 | **0.26 m / 0.28 m**（合計 0.54 m） |
| 輪半徑 | 0.09 m |

### 1.3 關節限位與命名（URDF，控制方式調查；限位定案見 §6）

| 關節 | 型別 | 軸 | 限位 (rad) | URDF effort |
|---|---|---|---|---|
| ABAD（hip_roll） | revolute | `1 0 0` (x) | 右腿 **−0.697 ~ +0.523**、左腿 **−0.523 ~ +0.697**（**左右鏡像、不對稱**） | 150 |
| HIP（hip_pitch） | revolute | `0 1 0` (y) | ±2.443（±140°） | 150 |
| KNEE（knee_pitch） | revolute | `0 1 0` (y) | ±2.801（±160°） | 150 |
| FOOT（輪） | **continuous** | `0 1 0` (y) | ±3.14 | **50** |

- **矛盾**：輪 effort URDF 寫 50、規格書 33 N·m → **以 33 N·m 為準**（保守，控制方式調查）。
- **URDF 的 `velocity` 限值全部是 0**，等於沒填；轉 MJCF 要用規格書值（腿 19.9、輪 125.7 rad/s）。

### 1.4 URDF ↔ SDK 命名對應（控制方式調查）

命名：`F`/`R` = 前/後，`AR`/`BL` = 右/左（由 y 座標確認）。

| URDF | SDK 名 | 位置 |
|---|---|---|
| `FAR_*` | `fr1/2/3/4` | 右前 |
| `FBL_*` | `fl1/2/3/4` | 左前 |
| `RAR_*` | `br1/2/3/4` | 右後 |
| `RBL_*` | `bl1/2/3/4` | 左後 |

SDK 編號規則：`1` = hip_roll(abad)、`2` = hip_pitch、`3` = knee_pitch、`4` = foot(輪)。
例：`fr1_hip_roll`、`bl3_knee_pitch`。

### 1.5 感測器安裝位置（相對 BASE 原點，m，控制方式調查）

| 感測器 | 位置 |
|---|---|
| `IMU_ICM42688`（運控用） | `(0, 0, 0.0362)` |
| `IMU_LUA300C`（車規級） | `(0, 0, 0.0569)` |
| 前 / 後光達 | `(±0.4043, 0, −0.0377)` |
| 前 / 後相機 | `(±0.4123, 0, 0.0378)` |
| 超音波 | `(0.1792, ∓0.1002, 0.05)` |

**矛盾**：URDF 的 `IMU_LUA300C_JOINT` 寫 `0.00569`，手冊寫 56.9 mm = `0.0569`。
**URDF 少一個零，以手冊為準。**（控制方式調查）

### 1.6 感測器實測規格（感測器與資源實測，2026-09-22）

完整資料 `docs/results/N_感測器與資源實測_2026-09-22.md`、`outputs/recon3b_20260922/`。
**下面全是實機量到或讀自實機設定檔的，不是手冊宣稱值。**

| 感測器 | 顆數 | 規格 | 實測頻率 | 接法 |
|---|---|---|---|---|
| 光達 | 2 | RoboSense **`RSAIRY`**、**96 線**、96×900 有序點雲、86,400 點/掃（864 k 點/秒）、每點 26 B、`start/end_angle 0–360`、驅動濾 0.2–200 m | **9.907 / 9.917 Hz** | NX 的兩張 CH397 USB 網卡，UDP msop 6699 / difop 7788 / imu 6688，各 2.88 MB/s |
| 光達自帶 IMU | 2 | 在光達裡 | **前後都 200.0 Hz**（30 s 同時量＋時戳驗證、零漏收） | 同上（`imu_port 6688`） |
| 機身 IMU | 1 | 兩條路徑發同一顆 | **200.0 Hz**（NX 與 RK 兩條路徑都是，零漏收 → shm→ROS2 的橋接不掉訊息） | `robot_hal_node/imu_recv_thread` → `/dev/shm/imu_central` |
| 相機 | 2 | Sony **IMX415**，原生 **3864×2192**（8.47 MP）Bayer；對外 **H.264 1920×1080 @ 25 fps**（RTSP）、JPEG **10 Hz**（ROS2） | 9.912 / 9.994 Hz | RK 的 MIPI CSI，`imx415 6-0037`／`7-0037` |
| 超音波 | 2 | `update_rate 10 Hz`、無效值 65530；**量程與視角未知**（驅動把 `min_range`／`max_range`／`field_of_view` 全填 0） | **9.945 / 9.953 Hz** | NX `/dev/ttyCH9344USB{0,1}`、115200 |
| UWB | 1 | — | 室內量不到（沒基站） | NX `/dev/ttyTHS1` |
| GPS / RTK | 1 | `/rtk_pvh` 有發布者；`/gps/rtk`、`/gnss/data` **0 發布者** | 室內量不到 | RK `gpsd /dev/ttyS4 /dev/pps0`，**有 PPS** |

⚠️ **`/dev/video*` 有 42 個節點，不是 42 顆相機**：`rkcif` 22（MIPI 擷取）、`rkisp_*` 18
（ISP 統計／參數／raw 讀回）、`video-dec0`／`video-enc0` 2（硬體編解碼）。

⚠️ **`/laser_scan`（10 Hz）不是感測器**，是點雲壓成的 2D 掃描。
`/aligned_points`、`/body_points`、`/world_points`、`/perception_points` 也都是算出來的。

---

## 2. 三機型對照重點

| | **D1 EDU 輪足**（小狗） | **D1 Max**（中狗） | **D1 MaxPro**（大狗） |
|---|---|---|---|
| SDK 代號 | `zsl-1w` / XGW | `zsm-1w` / ZGWS | — |
| 足端 | 輪足（16 軸：12 腿 + 4 輪） | 輪足（16 軸） | **點足（12 軸）** |
| 重量 | 20.6 kg | 41 kg | 68 kg |
| SDK 命名空間 | `mc_sdk::zsl_1w::HighLevel` | `robot_sdk::SDKClient` | `high_level_remote_tcp_client` |
| repo | `zsibot/genisom_L1_sdk` | `AgibotTech/Agibot_D1_Max` | `AgibotTech/Agibot_D1_MaxPro` |
| 中介軟體 | **eCAL**（ROS2 只是橋接） | **eCAL（內部）＋ ROS2 Humble + Zenoh（對外）** ⟵ 2026-09-22 更正，見 §4.1b | **ROS 1**（底層 SDK 強制） |
| 官方底層馬達控制 | ❌ 明文不提供 | ❌ 不提供 | ✅ **官方提供** `rt/lowcmd` |
| 我們實際做到的路 | `/spline_shm`（已端到端驗證） | **`/dev/shm/joint_cmd`（已端到端驗證）** | 未實測 |
| 關節狀態可讀 | 16 軸 pos/vel/tau | 16 軸 pos/vel/tau **＋每關節溫度** | 12 軸 pos/vel/tau |

（三機型對照，2026-08-25）

**要點**（三機型對照／控制方式調查）：

1. **D1 Max 與 D1 EDU 的 SDK 完全不共用** —— 命名空間、傳輸協定、函式名、資料結構全部不同。
   task6 的 `mc_sdk` 程式碼在這台上一行都用不到。唯一共用的是「板載共享記憶體」這條思路與
   `robot-launch` 行程管理工具。
2. **41 kg / 150 N·m 是完全不同的量級**。task6 的 kp=20/kd=0.7、力矩 >5 N·m 保護門檻，
   **一個都不能照搬**。
3. D1 MaxPro 的 `rt/lowcmd` 是 Unitree 風格的 PD + 前饋介面（`q_des` / `qd_des` / `kp` / `kd` /
   `tau_ff` / `flags`，只涵蓋 abad/hip/knee 共 12 顆，**沒有輪子**），上行 `rt/lowstate` 給
   IMU + 12 馬達 `q`/`qd`/`tau` + `spi_driver_status`。代價是**必須裝 ROS 1（Ubuntu 20.04）**。
   **D1 Max 上沒有這組 topic**（偵察二實測），那是 MaxPro 專屬。
4. **待向原廠確認**：D1 MaxPro 官方文件的 SSH 帳號是 `jetson`，但實機量到的主控板是
   Firefly RK3588。**以實測為準，但值得問一句。**（三機型對照）

### 2.0b ★ 小狗（D1 EDU）與中狗（D1 Max）在 eCAL／ROS 這層差在哪

| | **D1 EDU 輪足（小狗）** | **D1 Max（中狗）** |
|---|---|---|
| eCAL | ✅ **有實測**（`ecal_mon_tui`，小狗拆機前留下的畫面）：5 個行程、**16 個發布 / 13 個訂閱**，**topic 名字與型別全都看得到**（見下表） | ✅ **2026-09-22 實測**：4 個參與者、**8 個 topic**、`/dev/shm` 30 個 `ecal_*` 段、UDP 14000–14002；但**段名不透明，topic 名字看不到**（見 §4.1b） |
| ROS 2 | 只有**薄橋接**（節點數與內容我們沒量） | **完整一層**：21 節點／56 topic、`rmw_zenoh`、`ROS_DOMAIN_ID=66`，而且是 **ros2_control**（`controller_manager` ＋ `zsi_actuator_driver` 硬體元件 ＋ **80 個 command interface**） |
| 我們用的共享記憶體 | **`/spline_shm` 一段 10240 B**（cmd 與 state 同一段）＋ `imu_shm` 1024 B | **三段各 1 MiB**：`joint_cmd`／`joint_state`／`imu_central`，Boost.Interprocess managed segment ＋ `SharedVector` |
| shm 誰建立 | 未查 | **ros2_control 端建立**，`mc_ctrl` 是寫入者（`start_motion_control.sh` 等 `joint_cmd` 出現才起 `mc_ctrl`） |
| 寫入能不能回看驗證 | ❌ 沒有這種管道 | ✅ 訂閱 `/joint_shm_controller/joint_cmd_echo` 就看得到我們寫進去的東西 |
| 接管方式 | `SIGSTOP mc_ctrl` → 寫 shm → `SIGCONT` | **一樣**（實測可凍 38.1 秒，見 §5.2／§5.3） |
| 高層 SDK | `mc_sdk::zsl_1w::HighLevel`（`zsibot/genisom_L1_sdk`） | `robot_sdk::SDKClient`（`AgibotTech/Agibot_D1_Max`），UDP 8082 |
| `mc_ctrl` | 同名行程 | 同名；字串表證實**同一份程式碼含 `Quad_Controller` 與 `Wheel_Controller` 兩套 FSM** |

**一句話**：兩台狗的**運控核心與「靠共享記憶體下指令」的範式是同一套**，
差別是**中狗在外面多包了一整層 ROS 2 ＋ ros2_control**，小狗那層只是橋接；
共享記憶體也從「一段 10 KB 混著用」變成「三段各 1 MB、職責分開」。

**對我們的實務差別**：中狗多了三個小狗沒有的東西 ——
`joint_cmd_echo` 可以回看驗證寫入、`ros2 control list_hardware_interfaces` 可以列出全部介面、
所有狀態都能用 ROS 2 旁聽。**盲寫的風險比小狗那時低得多。**

#### 小狗的 eCAL 實測表（`ecal_mon_tui`，host `firefly`；**頻率欄是 mHz，要除以 1000**）

**五個行程**：`mc_ctrl`、`dog_task`、`SPLINE Publisher`、`IMU Publisher`、**`ecal2ros`**。

| topic | 型別（protobuf） | 發布 → 訂閱 | 頻率 |
|---|---|---|---|
| `spline_leg_cmd` / `spline_leg_state` | `leg_msg.control_cmd` / `control_state` | `SPLINE Publisher` → （表上沒有訂閱者） | **904.1 Hz** |
| `attitude_data` | `attitude_data.AttitudeData` | `IMU Publisher` → — | **986.3 Hz** |
| `leg_cmd` / `leg_data` | `robot_sdk.pb.RobotCmd` / `RobotState` | `mc_ctrl` → — | **500.0 Hz** |
| `sdk_robotstate` | `robot_sdk.pb.SDKRobotState` | `mc_ctrl` → **`ecal2ros`** | **500.0 Hz** |
| `sdk_cmd` | `robot_sdk.pb.SDKCmd` | **`ecal2ros`** → `mc_ctrl` | — |
| `nav_state` | `robot_sdk.pb.NavigationState` | `mc_ctrl` → **`ecal2ros`** | **50.0 Hz** |
| `nav_cmd` | `robot_sdk.pb.NavigationCmd` | **`ecal2ros`** → `mc_ctrl` | — |
| `app_cmd` / `app_state` | `robot_sdk.pb.AppCmd` / `AppState` | `dog_task` ⟷ `mc_ctrl` | **49.7 / 50.0 Hz** |
| `mc_dtc` | `robot_sdk.pb.McDtc` | `mc_ctrl` → `dog_task` | **50.0 Hz** |
| `battery_state` | `robot_sdk.pb.BatteryState` | `dog_task` → `mc_ctrl` | **0.59 Hz** |
| `hal_fault` | `hal_fault.FaultStatus` | `IMU Publisher`／`SPLINE Publisher` → `dog_task` | — |
| `motor_log` / `motor_log_done` | `spline_messages.saveMotorLog(Done)` | `dog_task` ⟷ — | — |
| `image_ecal` / `image_h264_ecal` | `zsibot_msg.Image` | — → `ecal2ros` | — |
| `visual_data` | `robot_sdk.pb.VisualData` | — → `mc_ctrl` | — |

**三件事因此確定了**：

1. **`ecal2ros` 是一個獨立的橋接行程** —— 這就是「ROS 2 只是橋接」的直接證據。
   小狗的 ROS 那一側只能透過它跟 eCAL 講話。**中狗沒有這個東西**：
   `robot_hal_node`／`robot_remote`／`robot_roamerx` **自己就是 eCAL 參與者**，
   同時也是 ROS 2 節點，橋接被打散進節點裡。
2. **小狗的關節指令走 eCAL**（`spline_leg_cmd`／`spline_leg_state` 904 Hz、
   `leg_cmd`／`leg_data` 500 Hz）。**中狗的關節資料不在 eCAL 上** ——
   `robot_hal_node` 在 eCAL 上只發不收，下行只有 `/dev/shm/joint_cmd`（§4.1b）。
   **這是兩台最大的架構差別。**
3. **904 Hz 正好對上 task6 在 `/dev/shm/spline_shm` 量到的約 880 Hz** ——
   `SPLINE Publisher` 這個行程名也對得上。
   → 小狗的 `spline_shm` 與 eCAL 的 spline topic 很可能是同一份資料的兩個面向（推測）。

**訊息定義兩台共用**：`robot_sdk.pb.*`、`hal_fault.*`、`zsibot_msg.*` 這些 protobuf 命名空間，
與中狗 `mc_ctrl` 字串表裡的 protobuf 一致 → 同一家的訊息定義。

#### ★ 中狗機上就有完整的 eCAL 工具鏈（2026-09-22 確認）

`/usr/bin` 底下有 **eCAL 5.13.3 全套**：`ecal_mon_tui`／`ecal_mon_cli`／`ecal_mon_gui`、
`ecal_rec`／`ecal_play`（錄放）、`ecal_sys`、`ecal_config`、`ecal_mma`，
以及一整組 `ecal_sample_*` 範例。

→ **中狗那 8 個 topic 的名字可以直接問出來**，用法與小狗那張表相同：

```bash
# 唯讀：mon_cli 只是以監看身分加入，不送任何資料
ssh robot@192.168.234.1 "bash -c 'set +u
. /opt/runtime/env.bash >/dev/null 2>&1
timeout 8 ecal_mon_cli 2>&1 || timeout 8 ecal_sample_monitoring_get_topics 2>&1'" | head -80
```

⚠️ **`ecal_stop` 絕對不要跑** —— 那會把 eCAL 的行程停掉，運控就斷了。
`ecal_rec` 會寫檔到機上，也先別用。

> 版本落差值得記一筆：中狗的 eCAL 是 **5.13.3**。小狗那張 `ecal_mon_tui` 的畫面
> 沒有版本資訊，但工具名稱一致。

---

### 2.1 D1 Max 的 SDK 鏡像有兩份（控制方式調查）

- `AgibotTech/Agibot_D1_Max` —— 附**完整中文手冊**（`docs/source/*.md`）與 **URDF**
- `zsibot/genisom_robot_sdk` —— 只有 SDK 本體，但**版本較新（0.2.1）**，另附 `Protocol-1.3.0.pdf`

兩邊的 `include/robot_sdk/*.hpp` 是同一套。建議：**手冊看 AgibotTech、程式庫用 zsibot**。

---

## 3. 車載電腦與連線

### 3.1 雙板架構（控制方式調查／偵察一／三機型對照）

| | RK3588 板 | Orin NX 板 |
|---|---|---|
| 角色 | **運動控制 + 系統監控**（官方明文「勿在此板開發應用程式」） | 建圖 / 定位 / 導航 / **使用者應用**（官方指定開發板） |
| 板卡 | Firefly AIO-3588SJD4 | NVIDIA Jetson Orin NX 16GB |
| CPU | 8 核 4×A76(~2.4 GHz) + 4×A55(1.8 GHz) | 8×Cortex-A78AE @1.984 GHz |
| RAM | **7.7 GiB** | **15 GiB**（分板記錄，不可相加） |
| GPU / NPU | Mali-G610 MP4、RKNPU 0.9.2 | NVIDIA ga10b、NVDLA0/1，157 TOPS |
| OS | Ubuntu 22.04；**Python 3.10.12（aarch64）** | Ubuntu 22.04 |
| ssh | `robot@192.168.234.1`（wifi）/ `robot@192.168.168.168`（有線），密碼 `bot` | `robot@192.168.168.100`，密碼 `1` |
| ROS2 | `ROS_DOMAIN_ID=`**66**、`rmw_zenoh_cpp` | `ROS_DOMAIN_ID=`**24**、`rmw_zenoh_cpp` |
| 韌體 | **0.1.7**（2026-02-06, "A2503 RK3588 Release Image"） | **0.3.6**（2026-02-28, "A2503 Orin NX Release Image"） |

**⚠️ 官方文件只提到 `ROS_DOMAIN_ID=24`，那是 NX 那一側。運控在 RK，RK 是 66。**
從 PC 要看運控的 topic，DOMAIN 要設 66（或走 `domain_bridge` 橋出來的那份）。（偵察一）

**底層控制一定在 RK3588。** Orin NX 的 `/dev/shm` 只有 nvidia 的 `nvsci*` / `ipc_test*`，
上面跑的全是導航感知節點（`nav2_container`、`arc_lvio`、`arc_mapping`、`arc_state_machine`、
`localization`、`robot_meb`、`navigo_charging_alignment`、`navigo_error_aggregator`、
`rslidar_sdk`、`uss_driver`、`uwb_driver`、`imu_driver`、`sixents_gps_driver`）。（偵察一）

### 3.2 推論可行性（obs盤點，2026-09-08，RK3588 實測）

| | |
|---|---|
| **numpy** | **1.21.5 有** ⚠️（兩份舊文件曾矛盾，**定案：有**） |
| onnxruntime / torch | **無** |
| CPU 親和性 | 8 核，可用 **0–6**；`isolcpus=7 nohz_full=7`，**cpu7 上跑的是 `robot_hal_node/HwLoop`（綁定 `cpus=7`、16.2%）**，推論程式**絕對不能跑在 cpu7**；**cpu6 也要避開**（所有 UART 中斷綁在那，實測 60%）|
| MLP 68→256→256→128→12 前向 | **純 Python 11.3 ms、numpy 0.14 ms**（50 Hz 預算 20 ms） |

→ **推論用 numpy、權重存 `.npz`；純 Python 當備援也還在預算內。**

### 3.2b 兩塊板的實測負載（感測器與資源實測，2026-09-22）

原始資料 `outputs/recon3_20260922/`，完整分析 `docs/results/N_感測器與資源實測_2026-09-22.md`。

| | RK3588 | Orin NX |
|---|---|---|
| 整體 CPU（待機 → 走路） | **26.1% → 26.1%** | **70.4% → 72.0%** |
| 最忙的核 | cpu6 59 → 61%（UART 中斷全綁這顆） | cpu0 78 → 79% |
| 記憶體 | 1.6 / 7.9 GB（20%） | 7.6 / 15.7 GB（48%） |
| loadavg | 5.5 | **13.6（8 核，超賣）** |
| GPU | Mali 0% | GR3D 平均 12% → 29%，峰 99% |
| 最大戶 | `robot_camera_node` 70%（影像編碼） | `robot_slam` 104%、`localization` 58%、`arc_lvio` 2.2 GB |
| 隨走路變的 | **`mc_ctrl` 5.8% → 11.0%（翻倍）**、**NPU Core0 0% → 7.4%** | GPU 12% → 29–31% |

**要塞自己的東西，空間在 RK 不在 NX。** 走路本身對兩塊板的總負載幾乎沒影響
（最大戶都跟走路無關：RK 是影像編碼、NX 是 SLAM）。

**NPU：只在動作時用，而且只用一核**（`sudo cat /sys/kernel/debug/rknpu/load`，
RK 的 sudo 免密碼；驅動 `RKNPU driver: v0.9.2`）：

| | 站著 | 走路（40 s 每秒取樣） |
|---|---|---|
| Core0 | 0% | **平均 7.4%、峰 8.0%** |
| Core1 / Core2 | 0% / 0% | 0% / 0% |

→ 原廠的 RKNN 運控策略**只在動作時吃 NPU**。我們要上 NPU 推論，Core1／Core2 整個空著。
⚠️ **`/sys/class/devfreq/fdab0000.npu/load` 不能用**：兩個情境都固定 `100@1000000000Hz`，
與真實使用率無關。**一個永遠不動的讀值就是沒在讀真東西**（同 `diagnostic-tools-lie`）。
⚠️ 而且**站著量會得到「NPU 全閒」的錯誤結論** —— 只在特定狀態出現的負載要在該狀態下量。

### 3.3 連線（控制方式調查）

| 方式 | 設定 |
|---|---|
| WiFi | SSID `XG2WIFI_xxxxxx`，**密碼 `12345678`**（官方文件明載） |
| 有線 | 網線接機身拓展網口；PC 網口設 `192.168.168.x`，**不可用 .168 與 .100**（那是狗的兩塊板） |
| SDK | UDP（0.1.1 起由 WebSocket 改為 UDP）→ `192.168.234.1:8082`（實測有 listen，偵察一） |
| 影像 | RTSP `rtsp://192.168.234.1:8554/front`、`/back` |
| 內部光達 | 兩顆各走一條 NX 的 USB 網卡：前 `192.168.1.0/24`、後 `192.168.2.0/24`。`rslidar_sdk/config/config.yaml` 的欄位名寫得很清楚是 **`host_address: 192.168.1.102` / `192.168.2.102`** —— **那是 Orin NX 自己的位址，不是光達的**（舊文件寫錯，2026-09-22 更正）。驅動只綁本地位址收 UDP，**光達自己的 IP 不在設定檔裡**，要讀 difop 封包或連光達網頁才知道 |
| 從 PC 連 ROS2 graph | Ubuntu 22.04 + ROS2 humble + `ros-humble-rmw-zenoh-cpp`；`RMW_IMPLEMENTATION=rmw_zenoh_cpp`；把 `DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5` 的 `connect/endpoints` 指到 `tcp/192.168.168.100:7447` |

**⚠️ `192.168.168.100` 在 D1 Max 上是「Orin NX 的位址」。** task6 的 D1 EDU SOP 裡
那是**我們電腦自己的靜態 IP**，**直接沿用會撞位址**，接 D1 Max 時電腦要改成其他值（例如 .50）。

---

## 4. 軟體架構

### 4.1 控制鏈路（偵察二，三個獨立證據交叉確認）

```
        mc_ctrl  (運控，ROBOT_TYPE=ZGWS)
             │ 寫
             ▼
    /dev/shm/joint_cmd            16 × {position, velocity, effort, kp, kd}  float64
             │ 讀（1 kHz）
             ▼
    joint_shm_controller          ros2_control controller
      (joint_controller/JointShmController)  ← claimed 全部 80 個 command interface
             │ 寫 command interfaces
             ▼
    zsi_actuator_driver           ros2_control hardware
      (zsi_actuator_driver/ActuatorInterface)
             │  16 顆馬達
             ▼
    /dev/shm/joint_state          16 × {position, velocity, effort, temp, voltage, error}
```

三個證據（偵察二）：
1. `ros2 control list_hardware_interfaces` → 16 關節 × `{position, velocity, effort, kp, kd}`
   共 **80 個 command interface**，全部 `[available] [claimed]`。
2. `/opt/runtime/bin/start_motion_control.sh` **先等 `/dev/shm/joint_cmd` 出現才啟動 `mc_ctrl`**
   → shm 由 ros2_control 端建立，`mc_ctrl` 是寫入者。
3. SHM 二進位解碼出來的欄位，與 command interface 的名稱與數量逐項吻合。

**插入點是 `/dev/shm/joint_cmd`。** 所有 command interface 都被 `joint_shm_controller` claimed，
「自己 spawn 一個 controller 去搶介面」要先卸載活的控制路徑，風險高。**寫 shm 是原廠自己在用的同一條路。**

### 4.1b ★★★ 運控板的中介軟體是 **eCAL**（2026-09-22 實測確認）

**四個行程都是 eCAL 參與者**（`sudo ss -lunp`，eCAL v5 預設埠）：

| 行程 | 14000 註冊 | 14001 log | 14002 payload |
|---|---|---|---|
| `mc_ctrl`（閉源運控核心） | ✅ | ✅ | ✅ |
| `robot_hal_node`（ros2_control 硬體層） | ✅ | ✅ | ❌ |
| `robot_roamerx_node`（導航應用轉接） | ✅ | ✅ | ✅ |
| `robot_remote`（遙控器） | ✅ | ✅ | ✅ |

`/dev/shm` 另有 **30 個 `ecal_*` 段** —— eCAL 的本機傳輸預設走共享記憶體，
UDP 那三個埠主要是註冊與跨主機用。

**所以運控板是三層，不要混為一談**：

| 層 | 用什麼 | 誰在上面 | 我們的關係 |
|---|---|---|---|
| 對外 topic 層 | **ROS 2 Humble ＋ `rmw_zenoh_cpp`**（`ROS_DOMAIN_ID=66`） | 21 個節點、56 個 topic；`robot_hal_node` 走 ros2_control（`controller_manager`／`zsi_actuator_driver`／`zsi_imu_driver`） | 唯讀驗證用得上（`joint_cmd_echo` 可回看我們寫進去的指令） |
| **廠商內部控制匯流排** | **eCAL** | `mc_ctrl` ⟷ `robot_hal_node`／`robot_remote`／`robot_roamerx_node` | ❌ 我們沒有走這條，也沒有它的 topic 定義 |
| 高速關節資料 | **具名 POSIX 共享記憶體** `/dev/shm/{joint_cmd,joint_state,imu_central}`（各 1 MB，**不是 `ecal_*` 段**） | `mc_ctrl` ⟷ `robot_hal_node` | ✅ **我們走的就是這條**，已端到端驗證 |

**這修正了三機型對照表的說法**：原本寫「D1 EDU 用 eCAL（ROS2 只是橋接）、D1 Max 用 ROS2＋Zenoh」。
實際上 **D1 Max 內部同樣是 eCAL**，差別在它**另外疊了一層完整的 ROS 2**對外。
→ 兩台狗的運控核心是同一套範式，task6 對 eCAL 的理解沒有白費。

#### eCAL 的完整拓樸（2026-09-22 實測，**已定案**）

判定方法：段 `ecal_<雜湊>` 是**發布者**建的記憶體檔，`_<pid>_evt` 是**訂閱者**的事件握手，
`/proc/<pid>/fd` 則列出**所有開著這段的行程**。
三者一交叉，發布者＝開著但不在訂閱清單裡的那個。

| 段 | 發布者 | 訂閱者 | 方向 |
|---|---|---|---|
| `ecal_ebf608ea` | **`robot_remote`**（遙控器） | `mc_ctrl` | 遙控器指令 → 運控 |
| `ecal_d87af588` | **`robot_roamerx_node`**（導航轉接） | `mc_ctrl` | 導航指令 → 運控 |
| `ecal_a74ae66c` | **`robot_hal_node`** | `mc_ctrl` | 硬體層 → 運控 |
| `ecal_8a14b62c` | **`mc_ctrl`** | `robot_remote` ＋ `robot_roamerx_node` | 運控狀態廣播 |
| `ecal_200f526f`／`301fbb4b`／`449d205f`／`cd623317` | **`mc_ctrl`** | **無**（只有它自己開著） | 運控發出、目前沒人收 |

**共 8 個 topic**，`mc_ctrl` 是中心（8 段全開：收 3 條、發 5 條）。

**★ 最關鍵的一條：`robot_hal_node` 在 eCAL 上「只發不收」。**
它只開 `ecal_a74ae66c` 這一段，而那一段的訂閱者是 `mc_ctrl`。
→ **`mc_ctrl` → HAL 這個方向在 eCAL 上完全沒有通道**，
只能走 `/dev/shm/joint_cmd`。**這正好解釋我們直寫 `joint_cmd` 為什麼有效**：
那不是「另一條旁路」，那就是原廠下行指令的唯一通道。

（勘誤：本節先前寫「HAL 在 eCAL 上沒有訂閱任何東西 → 關節資料不在 eCAL 上」。
前半句對，後半句講得太滿 —— HAL **有發**一條 eCAL topic 給 `mc_ctrl`。
**上行**回饋因此有兩條並存：eCAL 的 `a74ae66c` 與具名段 `joint_state`，
**哪條載什麼還沒驗**。下行只有 `joint_cmd` 這一條，這點沒有變。）

**序列化是 protobuf**：`mc_ctrl` 的字串表裡有 `/usr/include/google/protobuf/repeated_field.h`。
→ 就算之後想接 eCAL，**沒有 `.proto` 定義還是解不開內容**，仍然不建議走那條。

**順便從字串表看到的**（`strings /opt/export/mc/bin/mc_ctrl`）：
- 建置路徑 `/Users/robdog/jenkins_ws_do_not_move/workspace/zsibot_mc_macmini/...`
  → 在 macOS 的 Jenkins 上交叉編譯的
- **`custom/Quad_Controller/src/FSM_States/Motion_State.cpp` 與
  `custom/Wheel_Controller/src/FSM_States/Motion_State.cpp`**
  → `mc_ctrl` 內含**四足**與**輪足兩套控制器**，都是 FSM 狀態機架構
- 動力學 `common/include/Dynamics/Quadruped.h`、線代用 **Eigen3**

> 還沒驗：那 8 個 topic 的名字（段名是雜湊）。真要查可以
> `sudo strings /opt/export/mc/bin/mc_ctrl | grep -aiE "zsibot|/cmd|/state|joint|imu" | sort -u | head -40`，
> 但架構層級已經定案，這條對報告不是必要的。

---

### 4.2 三塊共享記憶體（偵察一／偵察二／寫入三／`shm_io.py`）

- 位置 `/dev/shm/`，各 **1 MiB**，權限 `-rw-r--r-- root:root` → **非 root 可讀、寫入需 root**。
- 都是 **Boost.Interprocess managed segment**，內含名為 `SharedVector` 的容器。
  佐證：實機設定檔有 `motor_platform_type: 7 # 5 mujoco # 6 shm 7 shmContainer`。
- 三塊兩次取樣 md5 都不同 → **都是活的串流**。
- 對照：D1 EDU 的 `spline_shm` 是 10240 B、`imu_shm` 1024 B，**這台大得多**。

#### `joint_cmd`（stride **112**，第一筆 base **752**）

| 位移 | 型別 | 內容 |
|---|---|---|
| base+0 | u64 | **整幀共用的時戳／心跳**（16 筆值完全相同） |
| base+8 | char[64] | 關節名（零結尾） |
| base+72 | f64 | **position**（目標角，**馬達座標系**） |
| base+80 | f64 | **velocity**（目標角速度） |
| base+88 | f64 | **effort**（前饋力矩） |
| base+96 | f64 | **kp** |
| base+104 | f64 | **kd** |

#### `joint_state`（stride **120**，第一筆 base **752**）

同上結構，數值欄位為 `position / velocity / effort / temp_C / voltage_V`（f64），
**末尾多一個 u64 `error`**。

#### `imu_central`（單一記錄，數值自位移 **824** 起，全部 f64）

```
+824  acc_x,  acc_y,  acc_z            (m/s²)
+848  gyro_x, gyro_y, gyro_z           (rad/s)
+872  quat_x, quat_y, quat_z, quat_w   (xyzw)
```

### 4.3 關節順序與 index（`shm_io.py` / 偵察二）

SHM 的 16 顆順序（與 `robot_hal.yaml` 的 `joint_shm_controller.joints` 逐項一致）：

| idx | 名稱 | idx | 名稱 | idx | 名稱 | idx | 名稱 |
|---|---|---|---|---|---|---|---|
| 0 | `fl1_hip_roll` | 4 | `fr1_hip_roll` | 8 | `bl1_hip_roll` | 12 | `br1_hip_roll` |
| 1 | `fl2_hip_pitch` | 5 | `fr2_hip_pitch` | 9 | `bl2_hip_pitch` | 13 | `br2_hip_pitch` |
| 2 | `fl3_knee_pitch` | 6 | `fr3_knee_pitch` | 10 | `bl3_knee_pitch` | 14 | `br3_knee_pitch` |
| 3 | `fl4_foot` | 7 | `fr4_foot` | 11 | `bl4_foot` | 15 | `br4_foot` |

**★ SHM 腿序是 `FL, FR, BL, BR`；官方設定檔（`zg_wheels-motion_config.yaml` 的
`FRfootCmd / FLfootCmd / RRfootCmd / RLfootCmd`）腿序是 `FR, FL, RR, RL`。
兩個不同層用不同腿序 —— 寫程式時務必按名稱對應，不要按索引。**（偵察二，本專案反覆踩到的坑）

### 4.4 心跳 / tick 與 500 ms 逾時（寫入三，實測）

| 項目 | 值 |
|---|---|
| 心跳位置 | 每筆記錄的 base+0（u64），**整幀共用**，16 筆相同 |
| 心跳速率 | **恰好 1000/s**（實測 +3000 / 3.0 秒），等於 `controller_manager update_rate: 1000 Hz` |
| 逾時參數 | `joint_cmd_timeout` = **500 ms** |
| 逾時處置 | `joint_shm_controller` 判定指令過期 → **把 `joint_cmd` 指令區清成 0** |
| controller 的另一條退路 | 逾時後套 `estop_kd = 35` → **阻尼停止，不是失力**（姿勢設計 §3） |
| `joint_cmd` 更新率 | ≥ 1296 Hz（M0 量測，2 秒內變化 2593 次） |
| `joint_state` 更新率 | ≥ 17045 Hz（**浮點雜訊使每次取樣都不同，是過量計數，不代表寫入率**） |

**寫入語意（`shm_io.py`）**：
- 每輪把 `joint_state` 當下的 tick **抄進** `joint_cmd` 的 16 筆記錄。
  用 `joint_state` 的 tick 而不是自己遞增，理由是它由活著的 `zsi_actuator_driver` 維護、
  永遠是新的、已證實與 `joint_cmd` 共用同一個時鐘，且不必猜單位與原點。
- **順序不能反：先寫 payload，最後寫 tick。** tick 等同「這幀備妥了」的旗標。
- 寫 payload 時**先目標值、後增益**：5 個 8-byte 寫入不是原子的，撕裂讀取最壞只會拿到
  「舊增益 + 新目標」；反過來會拿到「新增益 + 舊目標」→ 意外出力。
- 歸零（`zero_gains`）時**先寫增益**：目的是讓出力盡快變 0。
- 純阻尼（`damp_only`）**先 kp/effort 歸零、最後寫 kd**。
- **只覆寫已知位移的數值欄位，絕不碰標頭與名稱**，避免破壞容器結構。
  `verify_layout()` 會先比對 16 個關節名，對不上就拒絕往下走。

**⚠️ 心跳停 → 腿約 0.5 秒後失力下垂（不是 20 ms、不是阻尼）。** 這與 D1 EDU 的
「不舉旗 → spline_daemon watchdog ~20 ms 清零」不同，**不要照搬 task6 的心智模型**。
吊掛可承受，但**腿下方不要有人或東西**。（`estop_max.sh`）

### 4.5 ROS2 拓樸與機上路徑（偵察一／偵察二）

| 項目 | 內容 |
|---|---|
| 運控行程 | `mc_ctrl r`（與 D1 EDU 同一個 binary 名稱），觀察到 PID 2422 / 2428 |
| 啟動路徑 | `robot-launch server` → `/opt/runtime/bin/start_motion_control.sh` → `mc_ctrl`（`export ROBOT_TYPE=ZGWS`） |
| HAL | `robot_hal_node`，吃 `robot_hal.yaml` + `controller_manager.yaml` → **ros2_control** |
| 設定目錄 | `/opt/export/{mc, config, rknn_model_crypto}`；設定檔一律在 `config/zsm/` 底下 |
| ROS2 工作區 | `/opt/robot/install/{robot_hal, robot_monitor, robot_remote, robot_roamerx, robot_camera, robot_manager, robot_diagnostic_analyzer}` |
| Controllers（全 active） | `joint_shm_controller`、`imu_shm_publisher`、`switch_controller`、`battery_controller`、`fill_light_controller`、`led_controller` |
| Hardware component | `zsi_actuator_driver/ActuatorInterface`（system, active） |
| 關節 topic | `/joint_shm_controller/joint_states`、`/joint_shm_controller/joint_cmd_echo`（皆 `sensor_msgs/JointState`）、`/joint_sensor`（`robot_common_interface/msg/JointSensor`：name/error/error_msg/temp/voltage） |
| 其他 topic | `/cmd_vel`（`geometry_msgs/Twist`）、`/imu_shm_publisher/imu_central`、`/estop_controller/{hw,sw}_estop/state`、`/robot_remote/knee_mode`、`/robot_manager/robot_status` |
| 感測 topic | `/front_lidar`、`/rear_lidar`（PointCloud2, 10 Hz, best_effort）、`/front_lidar/imu`、`/rear_lidar/imu`、`/imu_driver/imu_central`、`/uss_driver/uss_{left,right}/range`（10 Hz）、`/rtk_pvh` |
| 日誌 | 本次開機 `/home/robot/robot_launch_log`；最近 6 次 `/userdata/log/` |

**★ `/joint_shm_controller/joint_cmd_echo` 會把它從 shm 讀到的東西發出來**
→ 我們寫進去的指令**可以用 ROS2 訂閱驗證**，不必盲寫。task6 當年沒有這個。（偵察二）

`robot-launch`（Rust 寫的行程管理器，與 D1 EDU 相同的工具）：
`list` / `egg` / `start` / `stop` / `restart` / `add-service` / `stdout-log` / `stderr-log`。

**`/opt/export` 與 MATRiX 發布包同結構**：MATRiX 的 `assets-0.1.2.tar.gz` 解出來是
`src/robot_mc/build/export/{config,mc,onnx_model_crypto}`，實機是
`/opt/export/{config,mc,rknn_model_crypto}` —— **同一個東西**（實機用 RKNN、模擬器用 ONNX）。（偵察一）

**機上工具**（偵察二，之後可能用得到）：
`/opt/runtime/bin/`：`actuator_tool`、`pose_calib_test`、`start_pose_calibration.sh`、
`recording_csv.py`、`start_recording_topic.sh`、`mediamtx`(RTSP)、`robot_wifi_*.sh`、
`canfd_upgrade` / `mcu_upgrade` / `clockpps_upgrade`。
`/opt/export/mc/bin/`：`mc_ctrl`、`libbiomimetics.so`、`libqpOASES.so`、`libGoldfarb_Optimizer.so`
（MPC/WBC 求解器）、`librknn_model.so`、`pose_calib_test`、`ptp_ctrl`、`file_crypto_cli`。
⚠️ `start_pose_calibration.sh` 的時間戳是 2026-03-11 15:21，比其他檔案新
→ **這台可能做過姿態校正**，之後若量測對不上可以往這裡查。

### 4.6 官方高層 SDK（控制方式調查）

環境：Ubuntu **22.04**、CMake 3.8+、GCC 11+、**Boost 1.74+**、**C++ only**。
庫已預編好 `lib/x86_64/` 與 `lib/aarch64/`。

**⚠️ SDK 版本必須配狗內韌體版本**，不合的症狀是錯誤碼 **`10001 ProtocolMismatch`**。
查法：`ssh robot@192.168.234.1 'cat /opt/release/version.yaml'`（RK）、
`ssh robot@192.168.168.100 'cat /opt/release/version.yaml'`（NX）。

**能寫的指令**（從 `librobot_sdk.so.0.2.1` 導出符號實際列出，比手冊全）：
- 姿態／狀態：`StandUp` `BalanceStandUp` `LieDown` `Crawl` `CrawlWalk` `Climb`（爬高台）
  `Stair`（登階）`Slim`（穿窄縫）`Sand` `DSB`（過擋鼠板）`Gait` `SkWalk`（同膝行走）
  `Locked`（各關節保持現位）`ReverseHeadTail`
- 移動：`Move(left_right, forward_back, yaw)`（**單位是百分比 [−1, 1]，不是 m/s**）、
  `Turn(direction)`、`ControlHead(lr, ud)`、`HighLowStance(stance)`（**僅原地模式有效**）、
  `PosControl` / `PosMove`（x,y,z,roll,pitch,yaw；手冊未詳述）
- 模式：`SetMode(1 通用 / 2 原地 / 3 登階)`、`SetSpeed(1 低 / 2 中 / 3 高)`
- 安全／周邊：`SoftEmergencyStop(on)` `TakeControl` `ReleaseControl` `SwitchIdleState`
  `SwitchRemoteState` `ObstacleAvoidance` `FrontLight` `BackLight` `AutoModeLight`
  `SetLedCommand` `SetPeriphPower` `TakePhoto` `UpdateCameraBitrate`
  `StartRechargeTask` / `StopRechargeTask`（自主回充，選配）

| 速度等級 | forward_back 全幅 | left_right | yaw |
|---|---|---|---|
| 低 | ±1.0 m/s | ±0.5 m/s | ±1.5 rad/s |
| 中 | ±2.0 m/s | ±0.5 m/s（前後 >1 m/s 時鎖 0） | ±1.5 → >1 m/s 時 ±1.0 rad/s |
| 高 | ±3.0 m/s | 同上 | 前後 >2 m/s 時降到 ±0.5 rad/s |

**⚠️ `Move` 指令只維持 1 秒**，要持續走必須週期性重送。

**能讀的資料**（`IDataCallback`）：

| 回呼 | 內容 | 頻率 |
|---|---|---|
| `OnImuData` | acc / gyro / 四元數（欄位名 `quat_x,y,z,w`） | `SetImuConfig(freq)` 0–100 Hz，**預設關閉** |
| `OnMcData` | 四元數 `quat[4]`（**註解寫 `[w,x,y,z]`**）、世界座標位置/速度/角速度、機體速度/角速度、ns 時戳 | `SetMcConfig(true)` → 50 Hz，預設關閉 |
| `OnJointStateData` | `names` / `positions` / `velocities` / `efforts` | `SetJointStateConfig(true)`，v0.1.0 起 |
| `OnSpeedData` | vx / vy / yaw | `SetSpeedReportConfig(on, 1–50 Hz)` |
| `OnRobotStateData` | 運動狀態、機器狀態、電池 ×2、**每關節溫度 map**、控制來源、軟/硬急停狀態 | 主動 1 Hz |
| `OnFaultData` | 故障碼（馬達失能／編碼器／離線／過壓／過熱／CAN／IMU…） | 發生時 |
| `OnControlLost` / `OnControlAvailable` | 控制權被搶／可用 | 事件 |

**狀態機與控制權**：
- 指令必須照狀態機順序下，否則「可能造成機器摔倒、故障或不響應」（官方原話）。
- **控制權不對稱**：APP（遙控器）**可以**搶走 SDK 的控制權；SDK **不能**搶走 APP 的
  → 錯誤碼 **`10002 ControlledDenial`**。標準流程：確認遙控器 APP 沒連著 → `TakeControl()` → 下指令。
- **想在 SDK 控制期間保留遙控器急停能力，必須在 SDK 取得控制權後「點開 APP」**，
  之後才能用遙控器右上角紅色急停鈕。**這步漏掉 = SDK 跑飛時沒有遙控器急停。**
- 官方警告：「執行 SDK 時請保證系統有足夠資源，否則可能出現運動控制模組失效」。

**官方安全機制**：

| 類型 | 觸發 | 效果 |
|---|---|---|
| 硬急停 | 機身急停按鈕 | 緩慢著地、紅燈；旋開後自動恢復 |
| 軟急停 | 遙控器 or `SoftEmergencyStop(true)` | 立即停車，**不再響應任何控制指令**，保持當前姿態 |
| 內部保護（自動） | 遙控斷連／電量 <10%／關節故障／IMU 通訊中斷 | 自動停車並緩慢著地 |

---

## 5. 底層控制方式（我們自己的路）

### 5.1 已驗證到什麼程度

| 能力 | 狀態 | 來源 |
|---|---|---|
| 讀 16 關節 pos/vel/tau/temp/voltage | ✅ SDK 與 shm 皆可 | 偵察二 |
| 讀 IMU | ✅ | 偵察二 |
| 寫 `joint_cmd` 並被系統接受 | ✅ **M1 16/16 相符** | 寫入三 |
| 驅動單一顆輪馬達 | ✅ **四顆全部** | 單顆馬達／四輪驅動 |
| 驅動腿關節 | ✅ **12/12 符號正確** | 腿關節16顆 |
| **16 顆同時全控** | ✅ **S6：12 腿關節維持站姿 + 四輪同時轉** | 腿關節16顆 |
| 姿勢切換（crouch ×2 / home / knee-back） | ✅ **每組 12/12，共 48 組全對** | 腿關節16顆 |
| 落地承重 41 kg | 這 11 份文件內**未涵蓋**（吊掛只驗了「指揮得到角度」） | 姿勢設計 §8 |

**`sudo` 免密碼可用**（寫入三 M0 實測）。

### 5.2 接管時序（姿勢設計 §2，已實機驗證）

```
讀當前實測角  →  SIGSTOP mc_ctrl  →  第一幀就寫 p_des = 當前角、kp = 0、kd = 小阻尼
  →  RAMP_UP    kp 0 → 目標值（預設斜坡 2 秒）
  →  MOVE       p_des 由當前角**餘弦插值**到目標
  →  HOLD       維持並記錄（主要證據在這一段）
  →  RETURN     插值回起始角
  →  RAMP_DOWN  kp 降回 0
  →  歸零 + 補心跳  →  SIGCONT
```

- **第一幀就要有指令**：凍結後心跳停、controller 500 ms 後判過期清零，中間有空窗腿會失力下墜。
- **餘弦插值而非線性**：線性在起點與終點速度不連續，對吊在吊帶上的 41 kg 是兩次衝擊。
- **前置條件「16 顆全部洩力」不能跳過**：它同時保證 `mc_ctrl` 凍結前也是洩力狀態，
  所以事後補一個 SIGCONT 不會造成出力。

### 5.3 `mc_ctrl` 凍結（`SIGSTOP`）的實測結論

| 項目 | 值 | 來源 |
|---|---|---|
| 凍結方式 | `SIGSTOP` → 狀態 `S` 變 `T`；`SIGCONT` 解凍回 `S` | 寫入三 |
| 實測最長凍結 | **38.1 秒**（另有兩次 26.0 秒），`robot_monitor` 沒有介入、沒有被重啟、解凍後正常 | 四輪驅動 |
| 建議上限 | `--max-freeze` **90 秒**（40 秒有實測支持，更久沒有） | 四輪驅動 |
| 收工複查 | 16 顆回到洩力、`p_des` 回到原廠 offset 常數、溫度正常、`mc_ctrl` 狀態 `S` | 單顆馬達／四輪驅動 |

### 5.4 中止行為（姿勢設計 §3）

任何保護觸發 → `kp=0`、`effort=0`、`kd=--abort-kd`（**純阻尼**），補心跳，**不解凍**。

| 不採用的選項 | 為什麼 |
|---|---|
| 全部歸零 | 腿帶著載荷時等於**自由落體**撞機械停點 |
| 直接 SIGCONT | mc_ctrl 恢復後會繼續下指令去它凍結前的姿勢；腿若已移遠，**那就是一次突跳** |

所以中止一定有**人工判斷點**：印出 PID 與 `sudo kill -CONT <pid>`，由現場的人決定何時解凍。
程式整個當掉也安全 —— controller 逾時會接手套 `estop_kd=35`（阻尼停止）。

### 5.5 estop 語意（`realbot/estop_max.sh`）

`sudo ~/estop_max.sh` 做三件事，**順序不能換**：
1. 殺掉我們所有會寫 `joint_cmd` 的程式（`WRITERS` 清單逐一 `pkill -f`）
2. **確認真的死了**（含 `/proc/*/maps` 交叉檢查，排除 `mc_ctrl|ros2|controller_manager|robot_hal|zsi_actuator|spline` 這些原廠行程）
3. 才解凍原廠 `mc_ctrl`，把控制權交回去

- **殺不掉卻解凍 = 我們的程式與原廠運控同時寫同一塊指令區，比什麼都不做更糟。**
  第 2 步失敗時**拒絕解凍**並叫人切電源。
- **不用「叫程式自己優雅停止」**：按 estop 的人在另一個終端機，沒人去按 M5 的 Enter；
  而本腳本 0.3 秒後就 SIGKILL。急停要的是**快、簡單、一定會成功**，不是優雅。
- pattern 不要用會匹配到腳本自己的字串（task6 中過兩次，一次殺掉自己的 SSH）。
  **新增任何會寫入的工具必須同步加進 `WRITERS`**；`/proc/*/maps` 交叉檢查就是為了讓「漏加」被抓到。
- **唯讀工具（如 `M6_load_probe.py`）不列入清單**。

### 5.6 限位檢查用兩個不同的餘裕（姿勢設計 §6.3）

- **規劃的軌跡**（起點與目標）：用 `--margin`（預設 0.05 rad），啟動時查一次。
- **實測角**：用**硬限位**（margin=0）。若實測角也套餘裕，低 kp 故意讓它偏差的設計
  會在第一個 tick 就誤中止。
- ⚠️ 實測最遠到過 ±2.8021（編碼器誤差或停點本身有彈性），所以**起點**的檢查
  不能當硬性阻擋 —— 那是狗實際所在的位置，不是我們命令的值。

---

## 6. 座標慣例

### 6.1 換算式（座標驗證，2026-08-25 **已實機驗證**）

```
馬達角  = side_sign × 控制器角 + offset
控制器角 = (馬達角 − offset) / side_sign
速度：ω_m = s · ω_c              （offset 是常數，微分掉了）
力矩：τ_c = s · τ_m              （由功率守恆 τ_c·ω_c = τ_m·ω_m 推得）
```

**`joint_cmd` / `joint_state` 的 `position` / `velocity` / `effort` 全都是馬達座標系。**
raw `effort` 直接拿去跟 MJCF 預演比，**一半的關節會憑空反號**（姿勢設計 §6.1）。

驗證強度（座標驗證）：
- 四個姿勢、三種構型，V1（上式）勝出：RMS 殘差 `stand_knee_front` **0.0353 rad (2.02°)**、
  `stand_knee_back` **0.0417 rad (2.39°)**、`crawl` **0.0862 rad (4.94°)**；
  其他三種候選換算式（V2/V3/V4）**全部落後一個量級**。
- 決定性判準要**三項一起看**：① 對文件姿勢的 RMS 殘差（V1 2.02° vs V3 >1.9 rad）、
  ② MJCF 機構限位（V1 0/12 超限 vs V3 **12/12 超限**）、③ 四輪共面（**無鑑別力**，V1 6.5 mm、V3 4.6 mm）。
- 殘差 2–5° 的來源是位置伺服的穩態追蹤誤差，不是換算式不對。
- 腿關節16顆（2026-08-26）在**真的出力**的情況下再驗一次：
  **四種姿勢 × 12 個關節 = 48 組全部通過，沒有任何一個關節反號。**

### 6.2 side_sign 與 offset（實機 `/opt/export/config/zg_wheels-user-parameters.yaml`）

原檔腿序 **FR, FL, RR, RL**；下表已轉成 SHM 腿名。

| 關節種類 | fl | fr | bl | br |
|---|---|---|---|---|
| `1_hip_roll` sign | −1 | −1 | **+1** | **+1** |
| `2_hip_pitch` sign | −1 | **+1** | −1 | **+1** |
| `3_knee_pitch` sign | **+1** | −1 | **+1** | −1 |
| `4_foot` sign | −1 | **+1** | −1 | **+1** |
| `1_hip_roll` offset | −0.523 | +0.523 | +0.523 | −0.523 |
| `2_hip_pitch` offset | +2.443 | −2.443 | −2.443 | +2.443 |
| `3_knee_pitch` offset | +2.803 | −2.803 | −2.803 | +2.803 |
| `4_foot` offset | 0 | 0 | 0 | 0 |

（原檔寫法：`abad_side_sign [-1,-1,1,1]`、`hip_side_sign [1,-1,1,-1]`、`knee_side_sign [-1,1,-1,1]`、
`wheel_side_sign [1,-1,1,-1]`；`abad_offset [0.523,-0.523,-0.523,0.523]`、
`hip_offset [-2.443,2.443,2.443,-2.443]`、`knee_offset [-2.803,2.803,2.803,-2.803]`，腿序 FR,FL,RR,RL）

**觀察**：offset 的絕對值 = URDF 的關節限位（abad 0.523、hip 2.443、knee 2.803）
→ 「馬達編碼器零點落在關節機構限位處」的慣例。（MATRiX解包）

**獨立驗證（偵察二）**：洩力時 `joint_cmd` 的 position 欄讀出來**恰好等於**這組 offset，四組全中。
這同時證明三件事：欄位位移解對了、設定檔就是實機在用的、**`joint_cmd` 是馬達座標系**。

**⚠️ `abad_side_sign = [-1,-1,1,1]` 是前後分組（前兩腿 −1、後兩腿 +1），不是左右分組。**
task6 的 D1 EDU 是同樣形狀，但 **`knee_side_sign` 兩台正負相反**
（D1 EDU `[1,-1,1,-1]`、D1 Max `[-1,1,-1,1]`）。（MATRiX解包）

### 6.3 腿名對應（`coord.py`，MJCF 代號依 body pos 判定，非猜測）

| SHM | MJCF | 設定檔 | 位置 | MJCF `*_ABAD_LINK pos` |
|---|---|---|---|---|
| `fl` | `FBL` | `FL` | 左前 | `0.2698  0.065 0` |
| `fr` | `FAR` | `FR` | 右前 | `0.2698 −0.065 0` |
| `bl` | `RBL` | `RL` | 左後 | `−0.2698  0.065 0` |
| `br` | `RAR` | `RR` | 右後 | `−0.2698 −0.065 0` |

### 6.4 控制器座標系 == MJCF 座標系（座標驗證 §7，**已驗證**）

方法：實機站在平地上，四輪接地點必然共面 → 把實測馬達角換算成控制器角，直接餵 MJCF 正向運動學。

| 姿勢 | FR | FL | BR | BL | **四輪高度全距** | 機身離地 |
|---|---|---|---|---|---|---|
| `stand_knee_front` | 0.5627 | 0.5607 | 0.5671 | 0.5638 | **6.5 mm** | 0.535 m |
| `stand_knee_back` | 0.5551 | 0.5568 | 0.5667 | 0.5697 | 14.6 mm | 0.541 m |
| `crawl` | 0.8401 | 0.8375 | 0.8259 | 0.8260 | 14.3 mm | 0.270 m |
| `lie_knee_front` | 0.9751 | 0.9694 | 0.9839 | 0.9770 | 14.5 mm | 0.127 m |

對照組：**完全不換算**（生馬達角當 MJCF 角）→ 全距 **118.5 mm**。
殘差 6.5 mm 的來源是伺服追蹤誤差（膝 2° 誤差 → 足端 9.8 mm），不需要用「座標系有偏差」解釋。

→ **座標鏈完整、沒有未驗證的環節**：
`MJCF 關節角 = 控制器角 → × side_sign + offset → 寫進 joint_cmd`。

### 6.5 限位定案（`coord.py`，控制器座標系 == MJCF 座標系）

| 關節 | 限位 (rad) |
|---|---|
| `fr1` / `br1`（右側 ABAD） | −0.697 ~ **+0.523** |
| `fl1` / `bl1`（左側 ABAD） | **−0.523** ~ +0.697 |
| `fr2` / `fl2`（前 HIP） | −2.442 ~ +2.791 |
| `br2` / `bl2`（後 HIP） | **−2.791 ~ +2.442**（★ 後腿是反過來的） |
| 四個 KNEE | **±2.801** |
| 四個輪 | 無限位（continuous） |

- **矛盾（KNEE）**：URDF ±2.801 vs MJCF 原本 ±2.791 → **以 URDF ±2.801 為準**。
  實機癱平時膝頂在 ±2.80 的機械停點（座標驗證 §4）；2026-08-26 這 0.01 開始擋住實際操作
  （狗趴著時膝就在 ±2.80，起點限位檢查永遠不過），已同步把 `model/zgws/zgws.xml` 改成 ±2.801。
- **矛盾（ABAD）**：設定檔的保護外圍值 ±0.873（±50°）是對稱的，URDF 那組是左右鏡像且不對稱的
  → **做 IK 限位檢查一定要用 URDF 那組，而且逐腿檢查**。（MATRiX解包）
- **矛盾（HIP）**：設定檔 ±2.80、URDF ±2.443 → **用 URDF**。
- **矛盾（KNEE 設定檔）**：設定檔 ±2.775、URDF ±2.801 → 兩者接近，取小的（原文），
  但實測支持 ±2.801（見上）。

**⚠️ ABAD 的行程只有 1.22 rad，而它的 offset 正好是 ±0.523 —— 「控制器角 0」就緊貼在
行程的一端附近。微動測試不要挑這個關節，方向也不能亂選；HIP 的行程 5.2 rad 才是正確選擇。**
（姿勢設計 §5）

### 6.6 膝模式（knee_back / knee_front）

- 預設是「**後腿往前彎**」（`knee_front`）：後腿 hip/knee 與前腿**反號**。
- 「**後腿往後彎**」（`knee_back`）= **只翻後兩腿的 hip 與 knee 號，前腿完全不動**
  （實測前腿兩種模式差 < 0.03 rad）。（座標驗證 §2.2）
- 實測 `stand_knee_back` 的後腿：hip +0.587 / +0.583、knee −1.251 / −1.232（與前腿同號）；
  `stand_knee_front` 的後腿：hip −0.55 / knee +1.23。
- 相關 topic：`/robot_remote/knee_mode`。程式介面：`coord.flip_rear_knee_mode()`。

---

## 7. IMU 慣例

| 項目 | 定案 | 來源／依據 |
|---|---|---|
| **quat 順序** | **`xyzw`**（`imu_central` +872 起） | 座標驗證（四姿勢 4/4 全勝，wxyz 的 roll 每次差 11–21°）；obs盤點（跳舞 2.1° vs wxyz 141°、平放 1.5° vs 14.4°、trip14 2.2° vs 122°） |
| acc 單位 | m/s²（靜態模長 9.845–9.887） | 偵察二／obs盤點 |
| gyro 單位／軸向 | **rad/s，三軸與 MJCF 同號，無對調** | obs盤點：對四元數微分逐軸擬合 k = 0.965 / 0.976 / 0.976，相關 0.984 / 0.993 / 0.987（20 ms 平滑；**不平滑會有衰減偏差 k≈0.65**） |
| **+roll** | **右側低** | obs盤點 §2.1（跳舞第七步：22.0 s roll −7.7° 左傾、23.0 s roll +7.2° 右傾） |
| **+pitch** | **頭低** | obs盤點 §2.1（前半俯身 pitch +17.8°） |
| **+yaw / +gyro_z** | **左轉**（由上看逆時針） | obs盤點 §2.1 + I-c 原地轉 |
| **安裝偏置** | **pitch −1.24°、roll +0.63°** | obs盤點：跳舞 15847 個四輪共面樣本，IMU vs 腿 FK 相關 0.998 / 1.000，殘差中位 0.6 mm |
| gyro 偏置 | (+0.0046, −0.0127, −0.0009) rad/s（**y 軸 0.7°/s 不小，訓練要隨機化**） | obs盤點，平放 30 s |
| gyro 雜訊 | std (0.0007, 0.0011, 0.0008) rad/s | obs盤點，平放 |
| 更新率 | gyro 479 / acc 481 Hz（下界，受 500 Hz 錄製限制）；quat 377（受 1e-5 存檔捨入限制） | obs盤點 |

**★ 官方 SDK 文件自相矛盾**（控制方式調查）：`ImuData` 的欄位名是 `quat_x, quat_y, quat_z, quat_w`（xyzw），
`MotionData` 的註解卻寫 `Quaternion [w, x, y, z]`。
**矛盾，以實測的 `xyzw` 為準。** 順序若錯，CPG-RL 的 obs 前三維（重力向量）會整個翻掉，
而且**不會報錯** —— 症狀是「狗一走就往某個方向倒」，現場會被誤判成 RL 沒訓練好。

**安裝偏置是真的，不是地板或綁帶**（obs盤點 §2.2）：兩個互相獨立的參考看到同一數字同一方向
（quat vs 腿 FK → pitch −1.24°；quat vs **自己的加速度計** → 1.45° / 1.35°）。
綁帶或地板墊歪的話兩個感測器一起歪、**差值不變**，解釋不了。
**處置**：訓練時加 ±3° 常數姿態偏置隨機化即可吃掉；精修則在 `real_obs` 對 quat 套固定修正（尚未做）。

**已解掉的懸案**：2026-08-26 吊掛時「IMU roll +7.10° vs 力矩反推 +2.5°」的矛盾
——IMU 安裝偏置只有 0.6°，所以那 7° 是**狗當時真的歪著掛**，不是 IMU 錯。
弱的一方是「用 ABAD 力矩殘差反推機身姿態」那個擬合（它假設四個 ABAD 摩擦相同，
但 ABAD 靜摩擦 1.85 N·m 與殘差 0.75 同量級）。
前三個估計（四元數 +7.10°、加速度計 +6.56°、ABAD 起始角偏移 +6.7°）彼此獨立且互相吻合。
（腿關節16顆 §6 + obs盤點 §2.2）

---

## 8. 原廠運控參數

來源：`zsibot/matrix` v0.1.2 的 `assets-0.1.2.tar.gz` → `src/robot_mc/build/export/config/`，
原始檔收在 `task7/reference/matrix_zgws/`。
**已與實機 `/opt/export/config/zg_wheels-user-parameters.yaml` 逐行 diff，只差兩處**（見 §8.4）
→ 除那兩處外，這批參數就是**實機在用的值**。（MATRiX解包／偵察二）

### 8.1 增益（設定檔）

| 用途 | ABAD Kp | HIP Kp | KNEE Kp | 腿 Kd | 輪 Kp | 輪 Kd |
|---|---|---|---|---|---|---|
| **RL 策略（走路用）** | **60** | **120** | **120** | **1.0** | **60** | **0.5** |
| RL 平衡站立 | 60 | 60 | 60 | 1.0 | 20 | 0.5 |
| 關節 PD（FSM） | **250** | **250** | **250** | **5.0** | — | — |
| 動作腳本（Montion） | 160 | 160 | 160 | 5.0 | — | — |
| 自主充電 | 400 | 400 | 400 | 5.0 | — | — |
| Passive（洩力） | — | — | — | 16.0 | — | 0.1 |
| 低功耗 | — | — | — | — | 0 | 0 |

**⚠️ ABAD 與 HIP/KNEE 是不同的 Kp（60 vs 120），不是四個關節共用一個值。**
task6 的 D1 EDU 三個腿關節共用 20，這台不是。寫 MJCF actuator 時要分開設。

**⚠️ 輪子在 D1 Max 上是「真的有位置增益」的**（RL 用 Kp=60），不像 D1 EDU 幾乎只做阻尼（Kp=5）。

與 D1 EDU（`xg_wheel`）對照：RL ABAD Kp 20→60（3×）、RL HIP/KNEE 20→120（**6×**）、
RL Kd 0.7→1.0（1.4×）、RL 輪 Kp/Kd 5/0.1→60/0.5（12×/5×）、Passive Kd 4.0→16.0（4×）、
關節 PD Kp 130→250（~2×）。

### 8.2 ★ 原廠站立的實測增益排程（原廠站立 §1／§5.5，2026-08-26 錄製、2026-08-27 分析）

**增益不是「開了就不動」—— 同一次站立動作裡先後用了三組，16 顆的切換時刻完全一致。**

| 時刻 | ABAD kp | HIP/KNEE kp | 腿 kd | **輪 kp** | 輪 kd | 這一段在做什麼 |
|---|---|---|---|---|---|---|
| 0 – 5.98 s | 0 | 0 | 0 | 0 | 0 | 洩力趴著等指令 |
| **5.98 s** | 250 | 250 | 5.0 | 20 | 0.5 | 趴 → crouch → 停 → 撐起來（**硬**、輪子鎖位置） |
| **9.99 s** | **60** | **120** | **1.0** | **0** | **0.1** | ★ 站直的瞬間：**放軟 + 放開輪子**（就是設定檔的 RL 那組，一字不差） |
| **12.99 s** | 250 | 250 | 5.0 | 20 | 0.1 | 站穩，鎖死定位 |

- **增益是「跳」上去的，不是斜坡**：`t≈6.7 s` 時 16 顆 kp 從 0 一步跳到 250、kd 0 → 5.0。
  代價就在力矩包絡裡：ABAD/HIP 在那一瞬間衝到 **20–28 N·m**。
  → 反過來驗證了我們 M5 的 kp 斜坡爬升設計（`--ramp` 預設 2 秒）**比原廠溫和**。
- **前饋：腿是 0，只有輪子有。** `joint_cmd` 的 `effort` 欄：**12 個腿關節全程 ±0.000**，
  只有四顆輪非零（±0.1 ~ 0.23）—— 那個量級**剛好等於實測輪摩擦 0.15–0.20 N·m**，原廠在補輪摩擦。
- **⚠️ `kp=0` 之後 `des` 欄位會被填成 `0.000`，那是沒有意義的佔位值。**
  任何拿 `des` 做事的程式（回放、比對）都必須**先看 `kp` 再看 `des`**。
- **輪子在最吃重的三秒 kp=0 自由滾**：後兩輪各滾約 **100 mm 輪面位移**
  （`bl` −1.044 rad / `br` +1.011 rad，帶上 side_sign 是同一個世界方向），峰速 1.5–1.8 rad/s。
  → 這解釋了現場目視的「輪子微幅前後轉」：撐起 41 kg 時輪距必須改變，那一刻就把輪子放開。
  （輪 `q` 在 ±π 會繞回，**要 unwrap 才看得懂**。）

**★ 引用增益時一定要標明是哪一種模式／哪一段。**

### 8.3 原廠站立軌跡（原廠站立 §5，`fl3_knee_pitch` 的 `des`，控制器座標系）

| 起 | 訖 | 歷時 | 從 | 到 | 速率 |
|---|---|---|---|---|---|
| 5.98 | 7.18 | 1.20 s | −2.7999 | **−2.4000** | 0.333 rad/s |
| 7.19 | 8.97 | 1.78 s | −2.4000 | −2.4000 | **停住** |
| 8.98 | 10.21 | 1.23 s | −2.3976 | −1.1084 | **1.048 rad/s** |
| 10.3 – 13.0 | | ~2.8 s | −1.10 → −1.18 | | 小幅平衡微調 |
| 13.0 | 22.25 | 9.25 s | −1.1766 | −1.1766 | 定位 |

1. **起點 `des = −2.7999` 就是當下的實測角**（−2.8009）→ **無突跳接管**，跟 M5 同一個原理。
2. **中途停在 −2.4000**，正是 `liedown / crouch` 的膝角 → **原廠是「趴平 → crouch → stand」三段式**。
3. **第二段比第一段快 3 倍**（1.05 vs 0.33 rad/s）：前段把腿從機械停點收出來（慢、小心），
   後段才是真的撐起 41 kg。

→ **這是一條現成的參考軌跡，要自己站起來照這個分段與速率走即可。**

### 8.4 姿態與尺寸參數（MATRiX解包 §2）

| 參數 | 值 | 說明 |
|---|---|---|
| `body_height` | **0.48 m** | 站立機身高度（D1 EDU 是 0.38） |
| `leg_height` | **0.1 m** | 抬腿高度，直接對應 CPG 的 `g_c` |
| `body_half_length` | **0.27 m** | 與 URDF 髖 x = ±0.272 吻合 |
| `drop_height` | −0.22 m | |
| `feet_z_offset` | 0.095 m | ⚠️ **不等於輪半徑 0.09**，是運控自己的偏移慣例 |
| `controller_dt` | 0.002 → **500 Hz** | 原廠運控頻率 |
| `controller_manager update_rate` | **1000 Hz** | ros2_control 端，等於心跳速率 |

**實機 vs MATRiX 設定檔的兩處差異**（偵察二 §3）：
1. `motor_platform_type: 5`（模擬）→ `7 # 5 mujoco # 6 shm 7 shmContainer`（實機）
2. **模擬那份多了 `*_stand_pos` / `*_liedown_pos` 區塊，實機檔案裡沒有**
   （只有 `*_default_pos`）。⚠️ **但實測顯示那兩組姿勢實機確實在用**（座標驗證 §2.1），
   只是不在那個設定檔裡（多半編進 `mc_ctrl` 或別處）→ **模擬版設定檔的參考價值比原本認定的更高。**

其餘全部逐字元相同：增益、offset、side_sign、限位、`body_height`、`leg_height`、步態排程參數。

### 8.5 步態排程與運動參數（MATRiX解包 §5）

| 參數 | 值 | 說明 |
|---|---|---|
| `cmpc_x_vel` | 2.0 m/s | MPC 允許的前後最大速度指令 |
| `cmpc_y_vel` | 0.5 m/s | 側移 |
| `cmpc_yaw_vel` | 2.0 rad/s | |
| `cmpc_fmax` | 120 | 單腳最大接觸力 |
| `mpc_horizon_length` | 14 | |
| `gait_type` / `cmpc_gait` | 4 / 8 | |
| `gait_period_time` | 0.2 s | |
| `gait_switching_phase` | 0.5 | |
| `gait_max_stance_time` / `min` | 0.25 / 0.1 s | |
| `gait_max_leg_angle` | 15 | |
| 匍匐速度範圍 | x ±2.0、y ±0.5、yaw ±2.0 | |
| 關節速度上限 | 設定檔 **24.0 rad/s** | **矛盾**：規格書 190 RPM ≈ 19.9 rad/s → **保守取 19.9** |

### 8.6 原廠策略檔（不能用，但值得知道存在）（MATRiX解包 §7）

`assets-0.1.2.tar.gz` 的 `src/robot_mc/build/export/onnx_model_crypto/zg_wheels/` 底下有 20 多個 ONNX 策略：

```
policy / policy_gait_walk / policy_skwalk / policy_stair / policy_climb / policy_slim
policy_dsb / policy_snow / policy_zgws_rpy / policy_zgws_jump / policy_zgws_backflip
policy_handstand / policy_tuoluo / policy_tuoluo_singleg / policy_mix_flipover
policy_zgwt_crawl_1029        （每個 policy_* 都有對應的 odom_*）
```

**它們是加密的**（同目錄有 `libfilecrypto_shared.so` 與 `file_crypto_cli`），沒有金鑰讀不出來
→ **「直接拿原廠 policy 來跑」這條路走不通，不用花時間試。**

但這份清單有兩個用處：
1. 證實 SDK 高層那些動作（`Slim` / `DSB` / `Stair` / `SkWalk` / `Climb` / `Sand`）
   **底下都是 RL 策略**，不是寫死的軌跡 → 這台的運控本來就是學出來的。
2. `policy_gait_walk` 的存在說明官方走路也走「gait + walk」這條路。

---

## 9. 實測的物理量

### 9.1 輪馬達摩擦（動摩擦）

| 來源 | 值 |
|---|---|
| M2 左前輪（kd 關係 `τ_f = kd·(v_des − v)` = 1.5 × (0.30 − 0.187)） | 0.170 N·m |
| M2 左前輪（力矩取樣平均） | 0.141 N·m |
| run1（低速，三輪） | 0.135 ~ 0.185，平均 **0.160** |
| run2（低速，三輪） | 0.127 ~ 0.180，平均 **0.152** |
| run3（高速 v=0.8、tff=0.2，解纏後重算） | 0.123 ~ 0.145，平均 **0.135** |
| 2026-08-26 吊掛 S6（腿同時 kp=40 出力） | fl4 0.203 / fr4 0.121 / bl4 0.170 / br4 0.186，平均 **0.170** |

**定案：≈ 0.13–0.20 N·m，MJCF `frictionloss` 填 `0.15` 站得住**（可能略低一點）。
四顆分開設的話：右前 0.13、左後 0.18、右後 0.15、左前 0.15。（四輪驅動 §2）

**⚠️ 這是動摩擦。輪的「靜摩擦掙脫門檻」始終沒量到** —— 四顆都是一給指令就轉
（`tau_ff=0` 也轉，這是與 D1 EDU 最大的不同，task6 那台必須加前饋才掙脫）。要量得從很小的 kd 往上掃。

**⚠️ fl4 從 8-25 的 0.15 升到 8-26 的 0.20，是四顆裡最高的（比 fr4 高 70%）。
兩次量測還分不出是趨勢還是散佈，先記著。**（腿關節16顆 §3）

### 9.2 輪速度追蹤與前饋（單顆馬達 §3／四輪驅動 §3）

| | v_des | tau_ff | 實際速度 | 追蹤率 |
|---|---|---|---|---|
| run1/2（墊高、腿洩力） | 0.3 | 0 | ~0.19 | **~65%** |
| run3 | 0.8 | **0.2** | ~0.84 | **~105%** |
| S6（吊掛、腿 kp=40） | 0.3 | 0 | 0.165–0.219 | 55–73% |

純 kd 速度伺服的穩態誤差必然是 `v_des − v = τ_f / kd`，**不是故障**。
補上約等於摩擦的前饋即可消掉。**跑步態時輪子的前饋力矩給 0.13–0.15 N·m 是合理起點。**

M2 單顆對照（正反兩向都轉）：`--vel +0.3` → +17.18°、摩擦 0.225；`--vel −0.3` → −18.14°、摩擦 0.213。

**單顆馬達首次驅動的完整數字**（單顆馬達，2026-08-25 15:25，`fl4_foot`）：
`v_des=0.3、kd=1.5、tau_ff=0、kp=0`；起始角 1.9156 rad、角度變化 **+0.3853 rad（+22.08°）**、
最大速度 0.3960 rad/s、**最大力矩 0.3777 N·m**（規格上限 33 N·m 的 1.1%）。

### 9.3 腿關節靜摩擦（腿關節16顆 §2，力矩爬升法，**不依賴 MJCF 模型**）

方法：正向要對抗重力、負向有重力幫忙 →
`f = (|τ₊| + |τ₋|)/2`、`τ_重力 = (|τ₊| − |τ₋|)/2`。

| 關節 | 掙脫門檻 | τ_重力 | 可信度 |
|---|---|---|---|
| **HIP** | **1.505 N·m**（`|τ₊|`=1.798、`|τ₋|`=1.212） | +0.293 | **高** —— 正反兩向都三門檻一致 |
| KNEE | **1.5 – 1.7 N·m**（取 (1.340+1.75)/2 → **1.55 ± 0.15**） | — | 中 —— 正向乾淨、反向是黏滑（門檻本身 ±0.3 N·m 模糊） |
| ABAD | **1.85 ± 0.15 N·m**（`|τ₊|`≈3.50、`|τ₋|`≈0.20） | ≈1.65 | 中 —— 收回方向乾淨，外張方向只是勉強起滑 |

**★ 判讀不能只看一個持續時間門檻。** 用 25 / 100 / 300 ms 三個「滑多久才算開始滑」的門檻去判：
**三門檻一致 = 真的持續滑動；隨門檻變動或消失 = 瞬間微滑，不是掙脫。**
只印一個數字的話兩種情況長得一模一樣 —— ABAD 那個 −0.229 就會被當成「門檻只有 0.23 N·m」，
比 HIP 小一個量級，完全是假的。

**軌跡形狀確認是真掙脫、不是彈性變形**：力矩爬升時關節先完全不動（q 停在 0.1234 十幾筆），
接著微滑幾次，然後速度起飛（0.05 → 0.08 → 0.12 rad/s）而**力矩就停在 1.85~1.93 不再上升**。
彈性變形不會這樣（q 會跟 τ 成比例）。

**下界（S1 微動，僅供對照）**：`fl2_hip_pitch` kp=15 → 施加 ±0.75 N·m（峰值 0.86），
兩方向對稱地不動（+0.0008 / −0.0004 rad）→ 是庫倫摩擦不是機構受阻；
kp=20（0.962 N·m）實走 +0.0019 rad，仍未真的動起來。

### 9.4 腿關節摩擦死區（腿關節16顆 §2.1，修正後）

死區（角度）= `f / 局部重力剛度 k`：

| | 先前用的（錯，用保持殘差估） | **修正後** |
|---|---|---|
| ABAD | 0.054 | **±0.116 rad（±6.7°）**（f=1.85 / k=15.89） |
| HIP | 0.060 | **±0.106 rad（±6.1°）**（f=1.505 / k=14.24） |
| KNEE | 0.17 | **±0.314 rad（±18.0°）**（f=1.55 / k=4.94） |

先前是拿保持殘差去估，而殘差必然小於掙脫門檻 → 估得太小。
**這回頭解釋了「判斷腿有沒有自由懸掛為什麼那麼難」：關節可以停在真平衡點 ±6°（膝 ±15°）內的任何位置。**

### 9.5 腿關節保持殘差（腿關節16顆 §2.2，S4 用實測角重算 `mj_inverse`）

| | 殘差平均 | 標準差 |
|---|---|---|
| ABAD | **+0.75 N·m** | 0.57 |
| HIP | +0.43 | 0.51 |
| KNEE | **+0.84** | 0.40 |

**12 顆全部同號為正** —— 馬達撐的比重力需要的多，那正是摩擦擋在運動方向上的指紋。
量級 **0.5–0.9 N·m**（殘差是停在死區內的某一點，必然小於掙脫門檻，與 §9.3 相容）。
這也解釋了 S4 比值表裡 KNEE 系統性偏高（1.13–1.43）而 ABAD 偏低（0.75–0.93）的結構。

### 9.6 S4 / S5 的比值表（腿關節16顆，比值 = `kp·誤差 / 預演τ`，控制器座標系）

S4（`--pose stand --kp 40 --kd 1.5 --emax 0.40`）：

| | ABAD | HIP | KNEE |
|---|---|---|---|
| fl | 0.93 | 0.94 | 1.21 |
| fr | 0.82 | 0.82 | 1.43 |
| bl | 0.90 | 1.03 | 1.13 |
| br | 0.75 | 1.21 | 1.20 |

**腿之間沒有耦合**：同一顆關節在單腿(S2)／前兩腿(S3)／全身(S4) 三種情況下
（fl1 0.93/0.87/0.93、fl2 0.95/0.78/0.94、fl3 1.21/1.36/1.21）→ 吊帶晃動沒有污染量測。

S5（四種姿勢，每組 12/12）：

| 姿勢 | 比值範圍 | 峰值 τ |
|---|---|---|
| `stand` | 0.75 – 1.43 | 6.0 |
| `crouch` #1 | 0.83 – 1.31 | 6.25 |
| `crouch` #2 | 0.83 – 1.19 | 6.16 |
| `home` | 0.74 – 1.30 | 5.8 |
| `stand --knee-back` | 0.68 – 1.39 | 5.7 |

`crouch` 連跑兩次，12 個關節比值幾乎逐項相同（最大差 0.12）→ **不是碰巧跑對一次**。
事前警告 crouch 的 HIP 會逼近 8 N·m 上限，實測峰值 **6.27**（`br2_hip_pitch`），達上限 **78%**
→ 預測方向與餘裕都正確。

### 9.7 原廠站立的力矩包絡（原廠站立 §2，**單次資料，無重複性佐證**）

| 關節 | 峰值 \|τ\| | 發生於 | 穩態 τ | 峰值/穩態 |
|---|---|---|---|---|
| `fl3_knee` | 40.60 | 9.05 s | +9.68 | 4.2× |
| `fr3_knee` | 41.47 | 9.05 s | +9.98 | 4.2× |
| `bl3_knee` | 40.61 | 9.06 s | −10.29 | 3.9× |
| **`br3_knee`** | **42.45** | 9.05 s | −8.34 | 5.1× |
| `bl1_abad` | 28.15 | **6.73 s** | −0.48 | **59×** |
| `br1_abad` | 27.81 | **6.72 s** | +0.06 | **476×** |
| `fr2_hip` | 22.67 | 7.08 s | −3.30 | 6.9× |

**兩個完全不同的峰值事件**：`t≈6.7 s`（增益從 0 跳到 250 的瞬間，ABAD/HIP 20–28 N·m）
與 `t≈9.05 s`（真正把 41 kg 撐起來，KNEE 40–42 N·m）。

→ **全部腿關節的峰值 = 42.45 N·m。要做同樣的動作，力矩保護至少要 60 N·m 量級。**
（當時 M5 的門檻是 ABAD 10 / HIP 8 / KNEE 7，照吊掛訂的，**差了 6 倍**。）

**★ ABAD 的峰值/穩態比是 59× 到 476× —— 只量靜態站立永遠訂不出這個門檻。這就是要錄全程的理由。**

### 9.8 原廠穩態站立（原廠站立 §3，最後 3 秒平均，控制器座標系）

| | 實測 τ | 我們的模擬預測 | 偏差 |
|---|---|---|---|
| KNEE | **8.3 – 10.3** | 14.9 | **高估 54%** |
| HIP | 0.4 – 3.3 | 6.2 | 高估 **~4×** |
| ABAD | 0.05 – 1.06 | 4.3 | 高估 **~7×** |

**★ ABAD/HIP 高估的原因：輪子會滾。** 靜態分析把腳當固定接觸點，**實際上腳是輪子，
會滾到側向力矩消失的位置**，所以 ABAD 穩態力矩趨近 0。
**這是輪足構型特有的自我卸載機制，固定足的四足機器人沒有。**

**原廠命令的目標角不是名目 `stand` 姿勢**：

| | 名目 `stand` | 原廠實際 `p_des` | 實測落點 |
|---|---|---|---|
| ABAD | 0 | **±0.031 ~ ±0.041** | ±0.034 ~ ±0.041 |
| HIP | ±0.6 | **±0.559 ~ ±0.569** | ±0.559 ~ ±0.582 |
| KNEE | ∓1.2 | **∓1.175 ~ ∓1.190** | ∓1.215 ~ ∓1.225 |

★ 這是**只有讀 `p_des` 才看得出來**的東西：從關節角完全分不出「命令 −1.20 垂到 −1.22」
和「命令 −1.18 垂到 −1.22」。

**確認原廠腿關節就是純 PD，沒有隱藏項**（τ 與 −kp·誤差 逐項對到小數第二位）：
`fl3` +9.68 vs +9.73、`fr3` +9.98 vs +9.92、`bl3` −10.29 vs −10.30、`br3` −8.34 vs −8.39。

**穩態追蹤誤差**：膝 **2.2 – 2.4°**、髖 < 0.8°、ABAD < 0.25°
（與 2026-08-25 從 `pose_stand_knee_front.json` 挖出來的膝 0.6–2.1° 同量級）。

### 9.9 伺服延遲與 obs 雜訊（obs盤點，2026-09-08）

| 欄位 | 實機 vs 模擬 | 給訓練的數字 |
|---|---|---|
| **des→q 延遲**（髖/膝） | 實機 **30–35 ms**，模擬 10–25 ms；joint_pos 實/模 lag 中位 12.5 ms | **致動器加 ~15 ms 延遲**（50 Hz policy ≈ 1 步）。trip14 原廠增益下也是 +16 ms → **與 kp 無關** |
| `joint_pos` | 相關 0.88–0.97；範圍差 <0.05 rad | 雜訊 **0.0005–0.002 rad**（量化 0.00038） |
| `joint_vel` | 峰值 ±10–13 rad/s，兩邊同量級；相關 0.55–0.85 | 雜訊 **ABAD 0.18–0.21、HIP 0.15–0.26、KNEE 0.41–0.44 rad/s**；>25 Hz 能量 2–20%（ABAD 最高） |
| `gravity` x/y | 相關 0.84 / 0.87，幅度 ±0.08 吻合，lag 0–5 ms | 雜訊 <0.0005（受 0.01° 存檔精度限制，**是下界**） |
| `gyro` x/y | 幅度 ±1.5 / ±1.3 rad/s，相關 0.69 / 0.76 | 雜訊用 §7 的 0.001 |
| `gyro` z | **M9 log 拿不到**；模擬 std 0.56、±1.2 rad/s | 範圍取模擬值 ±1.2 rad/s |
| ABAD 站立相平均角 | 實/模差 ~0.05 rad | **ABAD 側向力矩模擬不準（已知 10×）→ ABAD 動力學隨機化加大** |

**訓練端要吃的清單**：致動器延遲 15 ms；obs 雜訊 std（joint_pos 0.001、joint_vel 0.2 (ABAD/HIP) /
0.4 (KNEE)、gyro 0.002、gravity 0.005 保守值）；常數偏置隨機化（gyro ±0.02 rad/s、姿態 ±3°）；
`joint_vel` 額外一步延遲（driver 濾波）。

### 9.10 `joint_state.velocity` 欄位的性質（★ 這條有版本更正）

**舊觀察（單顆馬達 §2.2，2026-08-25，輪子低速）**：同一段資料兩種算速度方式 ——

| | 平均 | 標準差 | 變異 |
|---|---|---|---|
| **角度差分**（`Δpos / Δt`） | 0.1867 rad/s | 0.0176 | **9.4%** |
| `joint_state` 的 `velocity` 欄位 | 0.1895 rad/s | 0.0894 | **47.2%** |

平均值幾乎相同（無偏），但瞬時值雜訊差 5 倍。看瞬時 `velocity` 會以為輪子在劇烈震盪
（0.057 ~ 0.396 rad/s 跳動），但角度是平順單調遞增的 —— **實際運動很平順，是讀數在跳**。

**新結論（obs盤點 §3，2026-09-08）**：**driver 有濾波。**
500 Hz 原廠資料裡它比角度差分**乾淨**（倍率 0.10）、200 Hz 步態資料裡略髒（倍率 0.7–1.5）、
>25 Hz 能量 2–20%；代價是**落後 12–20 ms**。

→ **「velocity 雜訊 47%」是輪子低速的觀察，腿關節在動作中不是這樣。**
policy 用它沒問題，但模擬端要加雜訊＋一步延遲。
**做輪子的控制回授、收斂判定、品質評估，一律用角度差分。**

### 9.11 溫度、電壓、error 欄位

| 項目 | 值 | 來源 |
|---|---|---|
| 關節溫度 | 24–25 °C（冷機）→ **27–30 °C**（暖機後正常上升） | 偵察二／寫入三 |
| 母線電壓 | 53–54 V（正常）；剛充飽 **61.0 V** | 偵察二／寫入三 |
| `error` 欄位 | **16 顆全部 `1` 就是正常**（`error_msg` 全空字串） | 寫入三 §5 |
| 洩力時 `joint_state` | velocity ≈ −0.001、effort ≈ ±0.03 N·m | 偵察二 |

---

## 10. 姿勢常數（控制器座標系 == MJCF 座標系）

### 10.1 官方設定檔的三組（腿序 FR / FL / RR / RL）（MATRiX解包 §2）

| 姿態 | ABAD | HIP | KNEE |
|---|---|---|---|
| **站立** `stand` | `[0, 0, 0, 0]` | `[0.6, 0.6, −0.6, −0.6]` | `[−1.2, −1.2, 1.2, 1.2]` |
| **趴下** `liedown` | `[0, 0, 0, 0]` | `[1.4, 1.4, −1.4, −1.4]` | `[−2.4, −2.4, 2.4, 2.4]` |
| **RL 預設姿** `default` | `[0, 0, 0, 0]` | `[0.8, 0.8, −0.8, −0.8]` | `[−1.5, −1.5, 1.5, 1.5]` |

**★ 這是跟 D1 EDU 最重要的結構差異：前兩腿與後兩腿反號。**
D1 EDU 四腿同號（`hip_stand_pos = [0.8, 0.8, 0.8, 0.8]`），D1 Max 站姿是**前後鏡像的 X 型**。
→ **「四條腿套同一組公式」的假設在這台會失效。** task6 的 `HOME3` 那種寫法照抄會錯。

證據（偵察二 §3，因為 `*_stand_pos` 只在模擬版設定檔裡）：
- 實機那份有的 `hip_default_pos = [0.8, 0.8, −0.8, −0.8]`、`knee_default_pos = [−1.5, −1.5, 1.5, 1.5]`
  同樣是前後反號。
- 官方 MJCF 的正向運動學：前後反號那組給出對稱的四輪 x = ±0.3398；四腿同號那組會讓後腿整條往後翹。

### 10.2 `coord.py` 的姿勢字典（前腿 hip/knee 給定，後腿自動反號）

| 名稱 | 前腿 hip | 前腿 knee | ABAD | 機身離地 |
|---|---|---|---|---|
| `stand` | **+0.6** | **−1.2** | 0 | **≈ 0.535 m**（MJCF FK） |
| `home` | **+0.8** | **−1.5** | 0 | **≈ 0.491 m**（RL 名目站姿，對上原廠 `body_height` 0.48） |
| `crouch` | **+1.4** | **−2.4** | 0 | **≈ 0.29 m**（`coord.py`）／**0.270 m**（座標驗證 §7 的 `crawl` 實測 FK）—— 兩者略有出入，取用時註明來源 |

### 10.3 `lie`（癱平姿勢，**沒有被文件記載，但真實存在**）（座標驗證 §4）

| | 實測控制器角 | `liedown` 文件值 |
|---|---|---|
| ABAD | ±0.45 ~ ±0.55 | 0 |
| HIP | ±1.15 ~ ±1.20 | ±1.40 |
| KNEE | **±2.80** | ±2.40 |

**knee 的 ±2.80 正好是機構限位**（三個關節反推後落在 −2.7998 / −2.8013 / +2.8024）。
abad 張到 ±0.5 符合「趴平時腳向外攤開」。機身離地 **0.127 m**（MJCF FK）。
→ 附帶收穫：**URDF 的 knee 限位 ±2.801 才是準的**。

### 10.4 站姿之間的關係（座標驗證 §2.1、§8）

- **遙控器的站立是 `stand`（hip ±0.6 / knee ∓1.2），不是 `rl_default`。** 匍匐對上 `liedown`。
- **遙控器的站立比「走路名目站姿」高約 45 mm**（0.535 vs 0.491 m）—— 兩者確實是不同姿勢。
- ⚠️ `rl_default` 可能是 RL 策略跑起來之後的站姿，與遙控器的靜態站立不同 —— **尚未釐清**。
- ⚠️ 0.535 m 是**模擬正向運動學**的值，沒有實機量尺佐證。

---

## 11. 已知限制與坑

### 11.1 硬體／系統層

| # | 內容 | 來源 |
|---|---|---|
| 1 | **腿關節 150 N·m、整機 41 kg**，與 D1 EDU 差一個量級；task6 的增益與保護門檻一個都不能照搬 | 控制方式調查, 2026-08-25 |
| 2 | **RK3588 官方明文「勿在此板開發應用程式」**，但底層控制只能在 RK 上做 —— 這是結構性衝突 | 控制方式調查／偵察一, 2026-08-25 |
| 3 | **`cpu7` 被核心隔離**，2026-09-22 實測上面跑的是 `robot_hal_node/HwLoop`（不是 `mc_ctrl`，`mc_ctrl` 綁 0-6）；推論程式絕不能跑在 cpu7，也要避開 cpu6（UART 中斷全綁那顆，60%） | 三機型對照／obs盤點／感測器與資源實測 |
| 4 | 狗上**沒有 torch / onnxruntime**，只有 numpy 1.21.5 | obs盤點, 2026-09-08 |
| 5 | 心跳停 → **腿約 0.5 秒後失力下垂**（不是 20 ms、不是阻尼）；腿下方不要有人或東西 | 寫入三, 2026-08-25 |
| 6 | `mc_ctrl` 凍結 **40 秒有實測支持，更久沒有**；上限保持 90 秒 | 四輪驅動, 2026-08-25 |
| 7 | **原廠對第三方寫 `/dev/shm/joint_cmd` 的態度與保固範圍未確認**；使用者決定自負風險進行（2026-08-26） | 姿勢設計, 2026-08-26 |
| 8 | **落地承重（撐得住 41 kg）在這 11 份文件內未涵蓋**；吊掛只驗了「指揮得到角度」 | 姿勢設計 §8, 2026-08-26 |
| 9 | 原廠站立力矩包絡是**單次資料**，峰值 42.45 沒有重複性佐證 | 原廠站立 §7, 2026-08-26 |
| 10 | **輪的靜摩擦掙脫門檻始終沒量到**（要從很小的 kd 往上掃） | 四輪驅動, 2026-08-25 |
| 11 | ⚠️ 零出力寫入（M1）證明「指令沒被清掉」＝ controller 認為資料是新的，但**不能區分「消費後施加零力矩」與「根本沒消費」**；要靠 M2 或訂閱 `joint_cmd_echo` 才能定論（M2 已做，等於已解） | 寫入三 §3.5, 2026-08-25 |
| 12 | 閃紅燈事件（2026-08-25 充電完開機，重開恢復）：16 顆馬達完全健康（error 全 1、error_msg 全空、溫度 24–25 °C、電壓 61.0 V）。兩個候選解釋**分不出來**：① 遙控訊號斷連（官方 2.3.3 列為內部保護觸發條件，但**沒明說燈號是閃紅**）② 剛充飽電壓偏高（比平時高 7 V，SDK 有 `ActuatorOverVoltage` 故障碼，但**沒有電池規格佐證**）。下次再閃時當場跑 `ros2 topic echo /robot_monitor/diagnostic_status --once` 與 `/joint_shm_controller/joint_sensor --once` 並記下 APP 顯示電量 | 寫入三 §5, 2026-08-25 |

### 11.2 量測與判讀層（這條線踩過七次，全部同源）

**共同點：量測與診斷工具本身沒有被驗證，失敗或成功被安靜地偽裝。**
**解方：多印一個可以互相對照的量，比多印一個結論有用。**

| # | 坑 | 來源 |
|---|---|---|
| 1 | `set -u` + `source` 別人的環境腳本（ROS/conda/SDK 的 setup 幾乎都引用未設變數）= 地雷；一律用 `( set +u; . xxx )` 包起來，**而且不要吞 stderr** | 偵察一, 2026-08-25 |
| 2 | 自動判讀的 grep 打到**腳本自己印的區段標題** → 誤報命中（結論碰巧對了，過程是錯的） | 偵察一, 2026-08-25 |
| 3 | `set -o pipefail` + `cmd \| grep -q` → grep 命中後 echo 收到 SIGPIPE(141) → **誤報失敗**；改用 herestring `grep -q "..." <<< "$BODY"` | 偵察二, 2026-08-25 |
| 4 | `tick_end` 在 `restore()` **之後**才讀（含 SIGCONT 後 0.3 秒等待），除數只用迴圈時間 → 心跳速率算錯 15%（1154 vs 1000/s） | 單顆馬達, 2026-08-25 |
| 5 | **輪關節角度讀數包裹在 [−π, π]**，沒解纏 → run3 算出的 dp **連方向都反了**（−4.90 vs 真值 +7.66 rad），摩擦被算成 1.6–2.2 N·m。修法 `shm_io.wrap_pi()` 逐筆折回；200 Hz 取樣下要 \|v\|>628 rad/s 才誤判 | 四輪驅動, 2026-08-25 |
| 6 | **保護用了雜訊訊號**：`velocity` 欄位在角度包裹瞬間噴出假尖峰（4.282 rad/s，真值 0.84）造成誤中止。修法：保護改用解纏角度差分（要求視窗至少橫跨 20 ms，否則迴圈剛啟動 Δt→0 會爆掉），`velocity` 檢查改成**連續 5 筆超標**才中止 | 四輪驅動, 2026-08-25 |
| 7 | **比較兩個量之前先確認同座標系**：拿馬達座標角去比 URDF 限位，得出「實機行程比 URDF 寬」與 8 個假超限項目。換算後 48 個角度有 45 個在限位內，剩 3 個超出 0.01 rad（膝頂在機械停點） | 座標驗證, 2026-08-25 |
| 8 | **`effort` 欄位會偶發單筆垃圾**（已觀察兩次）。決定性案例：`fr3_knee` 一筆 −33.22 N·m，但同一刻位置完全沒變（小數第四位相同）、速度 ≈ 0、我們的控制律上限只有 0.55 → 物理上不可能。**判別式：`kp·\|err\| + kd·\|v\|` 是控制律的力矩上限。** 修法：連續 `--tau-hits`(3) 筆才中止、**硬上限也改連續 `--tau-hard-hits`(2) 筆**、`\|τ\| > 3×(kp·\|e\|+kd·\|v\|)+1.0` 的取樣排除在峰值統計外但**計數＋留原始樣本＋大聲報告**（靜靜過濾會把真的外力事件也吃掉）、峰值表同時保留 `tau` 與 `tau_raw`、中止時印前 0.2 秒原始序列附 `kp·\|e\|+kd·\|v\|` 一欄 | 腿關節16顆 §5, 2026-08-26 |
| 9 | **「我的檢查重現不出來」≠「這件事不成立」。** 輸入資料錯的檢查是**沒有結論**，不是反證（撤回「左右輪互頂」撤錯了，現場目視確認有接觸） | 腿關節16顆 §4, 2026-08-26 |
| 10 | **用一個含未知量的擬合去質疑三個獨立且一致的直接量測，方向反了。** 該先問「我的擬合假設成不成立」（IMU roll +7.10° vs 力矩反推 +2.5°） | 腿關節16顆 §6, 2026-08-26 |
| 11 | **把「我算出來的」當成「實際會發生的」**：① 把輪子的前饋當成腿的前饋，還據此宣稱「這解釋了追蹤誤差為何小」（因果講反，腿前饋是 0，誤差小是因為 kp=250）；② 模擬預測站立力矩系統性高估（膝 54%、髖 4×、ABAD 7×），因為把腳當固定接觸點，忽略輪子會滾 | 原廠站立 §8, 2026-08-26 |

### 11.3 模擬／模型層

| # | 內容 | 來源 |
|---|---|---|
| 1 | **模型在「站起來」這個動作上還不可信到能拿來訂門檻**：KNEE 模擬 31.6–32.2 vs 實機 40.6–42.5（低估 0.75–0.79×）、HIP 34.8–60.2 vs 20.5–22.7（高估 1.59–2.75×）、ABAD 23.6–53.3 vs 22.2–28.2（混合）。**門檻要照實機的 42.45 訂** | 原廠站立 §6, 2026-08-26 |
| 2 | 上表「模擬左右不對稱嚴重」（`fl2` 60.2 vs `fr2` 36.0）**是回放的假象** —— `replay_standup.py` 漏抄增益排程。三根尖峰的時刻與 §8.2 三次增益切換時刻逐一對得上 | 原廠站立 §6 末, 2026-08-27 |
| 3 | **`replay_standup.py` 待修**：`KP_LEG/KD_LEG/KP_WH/KD_WH` 是寫死常數（`inference/replay_standup.py:44-45`），應改成**逐幀讀錄製檔的 `kp`/`kd`**，並在 `kp==0` 時忽略該關節的 `des`。修好之前，該支的力矩對照數字**在 t>4.01 s 之後都不可信** | 原廠站立 §6 末, 2026-08-27 |
| 4 | 回放模擬站起來了，機身終高 **532 mm**（對比實機姿勢 MJCF FK 的 512 mm）。起始姿勢有 4 個關節超出 MJCF 限位被夾住（最大 0.0115 rad），已知差異不影響結論 | 原廠站立 §6, 2026-08-26 |
| 5 | **實機不抖的兩個真正原因**：① 起點命令 = 當下實測角，誤差幾乎 0（無突跳接管）；② 真正吃重那一刻**反而變軟並放開輪子** | 原廠站立 §6 末, 2026-08-27 |
| 6 | **仍然要自己掃的參數**（原廠設定檔沒有對應項）：`x_off`（質心配平）、`mu_y`、`D_STEP` / `D_STEP_Y`、`OMEGA` 區間 | MATRiX解包 §6, 2026-08-25 |

### 11.4 尚未回答的問題

| # | 問題 | 來源 |
|---|---|---|
| 1 | **RL 段（9.99–12.99 s）那三秒，控制器實際下的 `des` 軌跡是什麼？** 是在做主動平衡（依 IMU 回授改目標角）還是只是放軟後定住？這決定我們自己站起來時要不要也做這一段 | 原廠站立 §7, 2026-08-27 |
| 2 | 逾時處置的細節未釘死：寫入後停止心跳，`joint_cmd` 的 kd 欄變 0 還是 35？ | 腿關節16顆 §9, 2026-08-26 |
| 3 | `PosControl` / `PosMove` / `Gait` / `SkWalk` 的實際參數語意（手冊沒寫全） | 控制方式調查 §9 |
| 4 | 底層介面是否對 D1 Max 開放、開放條件、是否影響保固（**要問原廠**） | 控制方式調查 §9 |
| 5 | 走路時**原始** gyro（尤其 z）沒有實機值；gravity 雜訊是存檔精度的下界不是實測；IMU 偏置修正尚未寫進 `real_obs` | obs盤點 §6, 2026-09-08 |
| 6 | `fl4` 輪摩擦兩次量測 0.15 → 0.20，分不出是趨勢還是散佈 | 腿關節16顆 §3, 2026-08-26 |

---

## 12. 其他（無法歸類但可能重要）

1. **「腿有沒有真的自由懸掛」比想像中難達成**（腿關節16顆 §4, 2026-08-26）。
   判「卡住」與判「側傾」的差別在**四腿是否同方向偏移**：**同方向 = 姿態；單腿獨大 = 卡住**。
   三個側傾的指紋：① ABAD 四腿被整體平移 −0.117 rad（全距僅 0.046）② HIP/KNEE 前後反號的 X 型
   ③ **方向不對稱的阻力**（正向 1.10 N·m 動了、反向 1.53 N·m 不動 —— 庫倫摩擦是對稱的，
   不會這樣 → 頂到東西）。⚠️ **那次「kp=30 才動」不是關節摩擦，是地面反作用力，不要引用。**
   卡住的腿**會自己鬆開**：`bl3_knee` 從偏離 +0.45 rad 掉到 +0.20（回到死區內），是多次 S4/S6 帶開的。

2. **`joint_state` 的「更新率」不能用「相鄰筆是否相同」來量**（寫入三 §1）：
   浮點雜訊使每次取樣都不同 → 量到 ≥17045 Hz 是**過量計數**，不代表寫入率。
   同理，obs盤點量到的 quat 更新率 377 Hz 是受 1e-5 存檔捨入限制的**假低值**。

3. **`/opt/export/config/` 裡有多台機型的設定檔**（偵察二 §3）：`zg-user-parameters.yaml`
   也在機器上但**那是別的機型**；`start_motion_control.sh` 的 `export ROBOT_TYPE=ZGWS`
   決定載入的是 `zg_wheels-*`。引用設定檔時要先確認檔名前綴。

4. **檔案系統搜尋 `lowcmd`/`lowstate`/`spline`/`lowlevel` 在 `/opt`、`/usr/local`、`/home/robot`
   底下都沒有命中**，但**搜尋深度只到 4 層、且沒搜 `/opt/ros`**（偵察一 §4）——
   這是「沒找到」，不是「不存在」的完整證明。（不過偵察二的 `ros2 topic list` 已獨立確認
   `rt/lowcmd` 不存在。）

5. **`nm -D` 對 `librobot_sdk.so.0.2.1` 的符號分析**（控制方式調查 §5.1）：
   **269 個導出符號全部是 `robot_sdk::SDKClient::*`**，`strings` 搜不到
   `lowcmd|lowstate|spline|/dev/shm|tau_ff|torque` 任何一個
   → **不是漏放標頭檔，是庫裡真的沒有**（但這管不到 SDK 以外的路）。

6. **MJCF 資產來源**（控制方式調查 §10）：`zsibot/matrix` 是 MuJoCo + UE5 模擬平台，
   **有支援 D1 Max**（模型標識 `zgws` = `zsm-1w`，已用外觀比對確認），
   文件表列「MJCF 資產：有、內建運控：有（Passive / Stand / Walk）」。
   repo 本身只有文件，實體檔在 GitHub Releases 的執行包裡。
   URDF 已抓進 `task7/model/max.urdf`（36 KB）；**STL 網格沒抓**（`BASE_LINK.STL` 單檔 66 MB）。

7. **低增益下的偏差量本身就是一次量測**（姿勢設計 §1）——本專案驗證換算式的核心手法：
   - 遠離平衡點時：`τ_重力 = kp × 追蹤誤差`。**差一個負號 = `side_sign` 用反了 → 立刻停，不要加大 kp。**
   - **自然下垂點不適用**（依定義就是零致動器力矩的平衡點，`τ_重力 ≈ 0`，偏差量沒有訊號）。
     微動測試改用「方向」＋「移動比例 `q_eq − q_下垂 = kp·Δ指令/(kp + k)`」，
     kp 要挑到讓比例落在 **0.3 ~ 0.7**（鑑別力最高）。
   - **低 kp 同時是安全設計**：`side_sign` 反了會變正回授，但在下垂點附近只要 `kp < k`
     系統仍穩定，錯誤只表現為「走偏」而不是「衝到限位」。

8. **兩種長得像、成因完全不同的症狀**（姿勢設計 §6.2）：
   「力矩接近上限**且**誤差不收斂」= **機構受阻**（加大 kp 只會推高力矩，要目視檢查）；
   「誤差大**但**力矩小」= **增益不足**（低 kp 階段的預期行為，不是故障）。

9. **刻意不加前饋重力力矩**（姿勢設計 §8）：前饋的符號若錯，錯誤會直接變成出力；
   而低 kp + 純回授的錯誤只會變成位置偏差。等對照通過之後再考慮加。

10. **升階用同一支程式只換參數，不寫三支腳本**（姿勢設計 §4）：
    **版本不一致的症狀會偽裝成硬體問題**；三支腳本就是三套 bug，升階時無法分辨
    行為差異來自硬體還是程式差異。

11. **`push_to_dog.sh` 做 sha256 雙向比對並清 `__pycache__`**（腿關節16顆 §8）——
    同一條「版本不一致會偽裝成硬體問題」的防線。

12. **`coord.py` 的換算式有三份副本**（`coord.py`、`M4_pose_capture.py`、
    `inference/hang_rehearsal.py`），理由是狗上不能 import numpy 而預演跑在本機需要 numpy。
    `tests/test_m5_leg_pose.py` 會**逐項比對三份**，任何一邊改動而另一邊沒跟上，測試會失敗。

13. **「第一次 S6 看到左前輪沒轉」分不出原因**（腿關節16顆 §7）：重跑（參數完全相同）四顆全轉。
    兩個候選（① 它有轉只是看不出來 —— fl4 最慢 0.165 rad/s、3 秒 HOLD 只轉 28°、輪面沒記號
    ② 當時真的卡住，中間的 M2 把它掙脫了）**分不出來，因為第一次跑時 M5 對輪子沒有任何記錄**。
    已補上輪子儀表（累積轉角逐筆 `wrap_pi` 解纏、角速度用角度差分、追蹤率、平均力矩、摩擦推估）。
    **★ 下次跑 S6 前在輪子上貼膠帶** —— 目視要有基準。

14. **工具清單**（腿關節16顆 §8 / obs盤點 §7）：
    `realbot/coord.py`（換算／限位／姿勢單一事實來源）、`realbot/shm_io.py`（SHM 讀寫，純標準函式庫）、
    `realbot/M5_leg_pose.py`（六階腿關節控制）、`realbot/M6_load_probe.py`（**唯讀**錄製）、
    `realbot/estop_max.sh`、`realbot/push_to_dog.sh`、`realbot/M_env_probe.py`、
    `realbot/M_faultwatch.py`（唯讀，開機後 1 分鐘內啟動）、`realbot/shm_decode.py`、
    `inference/hang_rehearsal.py`、`inference/replay_standup.py`、`inference/m6_rec.py`、
    `inference/m9_rec.py`、`inference/real_obs.py`、`inference/imu_check.py`、`inference/obs_compare.py`、
    `reference/hang_torque_ref.json`、`reference/matrix_zgws/`（含 `SOURCE.md`）。

15. **原始輸出目錄對照**（要回頭查生資料時用）：
    `logs/recon_20260825_111311/`（偵察一）、`logs/recon2_20260825_112721/`（偵察二，含 SHM 二進位快照）、
    `logs/m_logs_trip3/`（M0/M1）、`trip4/`（M2 單顆輪）、`trip5/`（M3 四輪）、`trip6/`（M4 姿勢擷取）、
    `trip7/`（M5 腿關節 + M6 原廠站立錄製）、`trip14/`（9/2 原廠七段 @500 Hz）、
    `trip17/`（9/3 M9 步態）、`trip18/`（9/8 obs 盤點）。

16. **參考連結**：
    `https://github.com/AgibotTech/Agibot_D1_Max`（SDK + 中文手冊 + URDF）、
    `https://github.com/zsibot/genisom_robot_sdk`（較新 0.2.1 + `docs/protocol/Protocol-1.3.0.pdf`）、
    `https://github.com/AgibotTech/Agibot_D1_MaxPro`（`docs/source/4.2底层电机控制接口.md`）、
    `https://github.com/zsibot/genisom_L1_sdk`、`https://github.com/AgibotTech/agibot_D1_Edu-Ultra`、
    `https://github.com/zsibot/matrix`（MuJoCo + UE5 模擬平台）。
