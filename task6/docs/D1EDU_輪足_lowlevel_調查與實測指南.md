# D1 EDU 輪足版 Low-Level 控制：調查結果與實測指南

- 調查日期：2026-08-10
- 對象：智元 AgiBot D1 EDU 輪足版（機型代號 `zsl-1w` / `MODEL_XGW`）
- 目的：確認能不能把自己訓練的 CPG-RL policy 部署到實機上（sim2real 的落地介面）

---

## 0. 一句話結論

**官方到今天為止沒有開放輪足版的 low-level 控制**，這點在文件和二進位檔兩個層級都驗證過了。
但機器上存在一個**不分機型的板載共享記憶體馬達介面（`/spline_shm`）**，架構上看起來輪足也走這條路。
**這是推論，不是事實** — 值得花半天做一次唯讀測試去證實或否定它。

---

## 1. 背景：機型代號對照表

看官方文件很容易迷路，因為同一台狗有三四套名字。

| 你聽到的名字 | SDK 代號 | 新 SDK 列舉值 | 模擬器代號 | 說明 |
|---|---|---|---|---|
| D1 EDU 點足 | `zsl-1` | `MODEL_XG` | `xgb` | 12 個關節（ABAD/HIP/KNEE × 4） |
| **D1 EDU 輪足** | **`zsl-1w`** | **`MODEL_XGW`** | **`xgw`** | **16 個關節（多 4 個輪子 FOOT）** |
| D1 EDU 高速輪足 | — | `MODEL_XGWHSPD` | `xgw2` | 同輪足限制 |
| D1 Max（工業輪足） | `zsm-1w` | — | — | 另一個 repo，另一套 SDK |

另外要知道：**GENISOM-AI L1 系列 == D1 EDU 系列**。`zsibot` 這個 GitHub 帳號跟 `AgibotTech` 放的是同一套 SDK，機型代號完全相同。這件事在第 3 節很重要。

---

## 2. 官方對輪足 Low-Level 的支援情形

### 2.1 文件層級：白紙黑字寫「不提供」

舊版 SDK（`AgibotTech/agibot_D1_Edu-Ultra`）的 `docs/deploy.md` 第 168 行：

> **「ZSL-1w 不提供 LowLevel 接口，因此僅有 `highlevel_demo`。」**

### 2.2 程式碼層級：標頭檔就沒有

```
include/zsl-1/highlevel.h      ← 點足：有
include/zsl-1/lowlevel.h       ← 點足：有  ★
include/zsl-1w/highlevel.h     ← 輪足：只有這個
                               ← 輪足：沒有 lowlevel.h
include/zsm-1w/highlevel.h     ← D1 Max：也只有 highlevel
include/lowlevel/lowlevel.h    ← 不分機型的共享記憶體介面（第 4 節詳談）
```

demo 目錄也對得起來：`demo/zsl-1/cpp/` 有 `lowlevel_demo.cpp`，`demo/zsl-1w/cpp/` 只有 `highlevel_demo.cpp`。

### 2.3 二進位層級：庫裡真的沒編進去（這是最硬的證據）

光看標頭檔不夠 — 有可能只是「沒公開標頭檔，但庫裡有」。所以我把三個 `.so` 抓下來看導出符號：

```bash
nm -D --defined-only libmc_sdk_<機型>_x86_64.so | c++filt | grep -i lowlevel
```

| 庫 | LowLevel 相關符號 |
|---|---|
| `zsl-1`（點足） | `mc_sdk::LowLevel::sendMotorCmd` / `getMotorState` / `haveMotorData` / `initRobot` + `LowLevelConnector::*`（UDP + checksum） |
| **`zsl-1w`（輪足）** | **0 個** |
| `zsm-1w`（D1 Max） | **0 個** |

輪足庫只導出 47 個符號，全部是 `mc_sdk::zsl_1w::HighLevel::` 開頭的高層 API。
**結論：不是漏放標頭檔，是庫裡真的沒有這個功能。**

### 2.4 額外的一個發現（重要）

我對三個 `.so` 做 `strings` 搜 `spline` / `shm` / `/dev/shm`，**一個都沒命中**。

意思是：**共享記憶體那條路根本不在 SDK 庫裡**。`create_spline_shm()` 是標頭檔裡的 `static inline` 函式，你的程式直接對機器上的 daemon 講話，**完全繞過 SDK**。

→ 所以「輪足庫沒有 LowLevel」這件事，**管不到共享記憶體那條路**。這兩件事是獨立的。

### 2.5 輪足版現在實際能做什麼

`mc_sdk::zsl_1w::HighLevel`（UDP 到 `192.168.234.1`，50 Hz）：

- **能下的指令**：`standUp` / `lieDown` / `passive` / `move(vx, vy, yaw_rate)` / `crawl` / `climb` / `shakeHand` / `attitudeControl(roll_vel, pitch_vel, yaw_vel, height_vel)`
- **不能用的**（新 SDK 文件第 372 行明列）：輪足機型禁用 跳躍 / 前跳 / 打招呼 / 後空翻 / 雙腿站立
- **能讀的（這部分是全開的）**：16 個關節的角度 / 角速度 / 力矩（`getLegAbad/Hip/Knee/FootJoint` 及其 `Vel`、`Torque` 版本）、IMU 四元數 / RPY / 加速度 / 角速度、世界座標位置與速度、機體速度、電量、控制狀態

**一句話：讀是全開的，寫只到「速度指令」這一層。** 你沒辦法送關節角度或力矩。

### 2.6 版本需求

| 項目 | 點足 zsl-1 | 輪足 zsl-1w |
|---|---|---|
| SDK 版本 | 0.2.6+ | **0.2.7+** |
| 運控版本 | 0.2.6+ | **0.3.1+** |
| 本體版本 | 0.2.0+ | **0.3.3+** |

查版本指令（SSH 上機器後執行）：

```bash
grep -oP 'motion-control_\K[^_]+' /etc/release/*[^rootfs]*.yaml
```

沒有輸出代表版本過低，要找售後升級。

---

## 3. 新 Repo 的發現

### 3.1 SDK 搬家了

`AgibotTech/agibot_D1_Edu-Ultra` 這個大家都在用的 repo，其實等同 `zsibot/genisom_l1_sdk_old`（同樣的 `zsl-1` / `zsl-1w`、同樣的文件結構）。而那個 repo 的官方描述現在寫著：

> "Legacy early-access SDK for the Genisom-AI L1 Series. **No longer maintained. Please use the latest L1 SDK.**"

新家是 **[`zsibot/genisom_L1_sdk`](https://github.com/zsibot/genisom_L1_sdk)**：

- v1.0.0，發布 2026-06-22
- MIT 授權
- 2026-08-07 還在 push（三天前）
- C++17，靜態庫 `libzsibot.a`，支援 x86_64 + aarch64

而 AgiBot 那個舊 repo 最後一次有意義的更新是 2025-11-07 的 v0.2.7；2026-05-28 那次 commit 只加了一個 `video.py` 範例檔。

**→ 你聽到的「最近有更新」，最可能就是指這個。**

### 3.2 但是 — 新 repo 裡**沒有**輪足 low-level

這點要說得很清楚，避免誤會。我同樣驗證了新 SDK 的預編譯庫：

```bash
nm -C libzsibot.a | grep -iE "lowlevel|motorcmd|sendmotor|spline|shm|joint_control"
# → 零命中
```

導出的全部是 `zsibot::ZsibotExecutor::` 的高層 API（`SetCmd` / `SetRemote` + 35 個 getter）。

| 新 repo 裡的東西 | 是不是輪足 low-level |
|---|---|
| `libzsibot.a` | ❌ 純高層 UDP |
| `example/remote_control.cpp`、`example/sdk_control.cpp` | ❌ 都是高層 |
| `include/zsibot_sdk/lowlevel/lowlevel.h` | ❌ 跟舊版**逐字相同**（只多了 `/bms_shm` 電池區塊）。是**不分機型的共享記憶體結構定義**，不是輪足專用 API，本身沒有任何實作 |
| `docs/zh/api_lowlevel.md` | ❌ 跟舊版 diff **完全一致** |

**注意一個我一開始判斷錯的地方**：新 repo 裡沒有「zsl-1w 不提供 LowLevel」那句禁止的話 — 但這是**弱證據**，因為新 repo 根本沒有 `deploy.md` 這份文件，整份都不在。**缺席不等於解禁。**

### 3.3 新 SDK 帶來的實質改變

| 項目 | 舊 SDK | 新 SDK v1.0.0 |
|---|---|---|
| 通訊 | 自訂二進位 + checksum | **UDP + JSON**（送 8081 / 收 8080） |
| 心跳 | — | 50 ms 一次，5 秒 timeout |
| 控制權 | 無 | **role-based 仲裁**（remote / roamerx / general_sdk） |
| 機型 | 各機型各一個 `.so` | 單一庫 + `MODEL_XG` / `MODEL_XGW` / `MODEL_XGWHSPD` 執行期判別 |
| 授權 | BSD 3-Clause | MIT |
| 電池 | 只有電量 | 電壓 / 電流 / 溫度 / 錯誤碼 |
| 故障 | 無 | 模組 / 子模組 / 錯誤碼 / 等級 / 描述 |
| 電機溫度 | 無 | **16 路** |

（16 路電機溫度這件事本身就再次確認了輪足是 16 顆馬達。）

### 3.4 順帶撿到的兩個寶

**A. 官方 ZSL-1W URDF** — [`zsibot/genisom_model`](https://github.com/zsibot/genisom_model)

16 個 revolute 關節，**輪子是真的驅動關節**：

| 關節 | axis | lower | upper | effort | velocity |
|---|---|---|---|---|---|
| ABAD | `1 0 0` | -0.4887 | 0.4887 | 28 | 28 |
| HIP | `0 1 0` | -1.152 | 2.967 | 28 | 28 |
| KNEE | `0 1 0` | -2.723 | -0.602 | 28 | 28 |
| **FOOT（輪）** | `0 1 0` | -999999 | 999999 | 28 | 28 |

命名是 `FBL / FAR / RAR / RBL`（不是 FL/FR/RL/RR）。FOOT 的上下限 ±999999 等於 continuous，就是輪子。

**B. 官方模擬器 MATRiX** — [`zsibot/matrix`](https://github.com/zsibot/matrix)

MuJoCo + Unreal Engine 5 + CARLA，371 stars，2026-08-03 更新。`xgw`（輪足）**有 MJCF 資產 + 內建運控**（Passive / Stand / Walk）。

它的 `docs/Motion_Control_CN.md` 列了兩種控制模式：

| 模式 | 資料鏈路 |
|---|---|
| 進程內運控 | UeSim ↔ Zenoh ↔ 內建 motion core |
| **Linux 模擬硬體** | **UeSim ↔ 共享記憶體 ↔ 外部 `mc_ctrl`** |

原文：「該模式由 UeSim 提供模擬硬體共享記憶體，再由外部 `mc_ctrl` 訪問。」

---

## 4. 為什麼值得測試一下

### 4.1 支持的理由

**理由一：共享記憶體介面在架構上不分機型。**
`include/lowlevel/lowlevel.h` 定義的 `leg_control_t` 是這樣：

```c
typedef struct leg_control_def {
    joint_control_t abad;
    joint_control_t hip;
    joint_control_t knee;
    joint_control_t foot;   // ← 就是輪子
    int32_t flags;
} leg_control_t;
```

**結構裡有 `foot` 這一欄** — 點足版根本沒有這個關節。這個結構天生就是為了涵蓋輪足設計的。

**理由二：輪足機器的啟動腳本就在等這塊記憶體。**
`docs/deploy.md` 裡輪足專用的啟動腳本 `/opt/app_launch/start_motion_control_xgw.sh`：

```bash
SHM_FILE="/dev/shm/spline_shm"
while true; do
    if [ -e "$SHM_FILE" ]; then break; fi
    sleep 1
done
...
cd /opt/export/mc/bin && taskset -c 7 ./mc_ctrl r
```

→ **輪足本體的運控程式本身就跑在這個共享記憶體架構上。** 這塊記憶體在輪足機上必然存在（否則運控起不來）。

**理由三：官方模擬器把這條 ABI 當成「硬體邊界」在模擬。**
MATRiX 的「Linux 模擬硬體」模式，就是讓 UeSim 假裝成硬體、提供共享記憶體，讓**外部的控制程式**接上去。官方明確把 `/spline_shm` 定位成硬體 ABI，而不是內部實作細節。

**理由四：介面形式正好是 RL 部署要的。**
每個關節是 `p_des, v_des, kp, kd, t_ff`（目標位置、目標速度、比例增益、微分增益、前饋力矩），1 ms 週期。這是標準的 MIT Cheetah 式 PD + 前饋力矩介面 — 你的 CPG-RL policy 輸出直接就能填進去。

（順帶一提，`spi_command_def`、`spline`（其實是 spine 的訛寫，指 MIT Cheetah 的 SPIne 板）這些命名，說明這整套架構就是 MIT Cheetah 血統。）

**理由五：光是「讀」就已經值回票價。**
即使最後不能寫，1 kHz 的 16 關節 `p/v/t` + IMU 遙測，對比 HighLevel 的 50 Hz，對你做系統辨識、驗證 MJCF 參數、量測馬達延遲，都是剛需。

### 4.2 誠實標註：以下全部是推論，沒有硬體驗證

必須把話說死，免得你抱著錯誤期待去測：

- ❓ 沒人確認過實機上 `/dev/shm/spline_shm` 到底存不存在、權限是什麼
- ❓ 沒人確認過第三方程式寫進去 daemon 會不會理你
- ❓ 沒人確認過停掉 `mc_ctrl` 之後會不會有其他保護機制擋住
- ❓ MATRiX 說的「外部 `mc_ctrl`」指的可能是**原廠自己的另一版 mc_ctrl**，不一定是「歡迎你寫自己的」。文件原話是「外部 `mc_ctrl` 的二進位、模型和參數不在當前文檔倉庫中，應以對應控制器發布包為準」
- ❓ 官方**從未在任何地方**書面表示第三方可以走這條路

**所以第 6 節的 Phase 0 才是重點：它是目前唯一能把這些問號變成句號的動作，而且零風險。**

---

## 5. 共享記憶體介面技術規格

（測試前先讀懂這節，Phase 0 的程式碼才看得懂。）

### 5.1 三塊共享記憶體

| 路徑 | 大小 | 內容 | 更新頻率 |
|---|---|---|---|
| `/spline_shm` | 10 KB | 馬達指令 + 馬達狀態 | 1 ms |
| `/imu_shm` | 1 KB | IMU（時戳 / acc / gyro / 四元數） | 1 ms |
| `/bms_shm` | 1 KB | 電池（電壓 / 電流 / 電量 / 溫度 / 工作狀態） | — |

在 Linux 上，`shm_open("/spline_shm")` 對應的檔案就是 `/dev/shm/spline_shm`，可以直接 `ls` 看到。

### 5.2 資料結構

```c
// 一個關節的指令
typedef struct {
    float p_des;   // 目標位置 rad
    float v_des;   // 目標速度 rad/s
    float kp;      // 比例增益
    float kd;      // 微分增益
    float t_ff;    // 前饋扭矩 N·m
} joint_control_t;

// 一條腿 = 4 個關節（輪足的 foot 就是輪子）
typedef struct {
    joint_control_t abad, hip, knee, foot;
    int32_t flags;   // 永遠設 1
} leg_control_t;

// 一個關節的狀態
typedef struct {
    int32_t flags;  // 見下方位元定義
    float p;        // 當前角度 rad
    float v;        // 當前角速度 rad/s
    float t;        // 當前扭矩 N·m
} joint_state_t;

typedef struct { joint_state_t abad, hip, knee, foot; } leg_state_t;

// 整塊共享記憶體的樣子
typedef struct {
    struct { leg_control_t legs[4]; uint32_t consumer_flags[2]; } cmd;
    struct { leg_state_t   legs[4]; uint32_t consumer_flags[2]; } state;
} spline_data_t;
```

`joint_state_t.flags` 的位元定義：

| 位元 | 意義 | 解碼方式 |
|---|---|---|
| bit0 | 馬達使能 | `flags & 1` |
| bit1 | 過壓故障 | `(flags >> 1) & 1` |
| bit2 | 過流故障 | `(flags >> 2) & 1` |
| bit3 | 過溫故障 | `(flags >> 3) & 1` |
| bit4 | 超速故障 | `(flags >> 4) & 1` |
| bit5 | 雙編碼器故障 | `(flags >> 5) & 1` |
| bit8~15 | 溫度（-40 ~ 215 °C） | `((flags >> 8) & 0xFF) - 40` |
| bit16~23 | 電壓（0 ~ 255 V） | `(flags >> 16) & 0xFF` |

### 5.3 旗標（consumer_flags）的握手協定 ★ 最容易搞錯的地方

```c
enum { CONSUMER_CONTROL = 0, CONSUMER_OTHER = 1, CONSUMER_MAX = 2 };
```

**規則：producer 把旗標設 1，consumer 讀完清 0。**

**下指令的路徑（1 ms 一輪）：**
1. 你（producer）把資料填進 `cmd.legs[]`
2. 你把 `cmd.consumer_flags[CONSUMER_CONTROL]` 設成 1
3. daemon 看到不是 0 → 把 `cmd.legs` 送給馬達，然後清成 0
4. 如果 daemon 連續約 10 個週期都沒看到旗標被設起來 → **自動清掉馬達指令**

→ 第 4 點就是內建的 watchdog。你的程式當掉，馬達會自己鬆掉。這是很重要的安全網。

**讀狀態的路徑（1 ms 一輪）：**
1. daemon 更新 `state.legs[]`，並把 `state.consumer_flags[0..1]` 都設成 1
2. **運控程式（mc_ctrl）用的是 `CONSUMER_CONTROL`（index 0）這一格**
3. **你這種第三方程式要用 `CONSUMER_OTHER`（index 1）**

> ⚠️ **絕對不要碰 `state.consumer_flags[CONSUMER_CONTROL]`。**
> 那是 mc_ctrl 的格子。你把它清掉，mc_ctrl 會以為「馬達狀態拿不到」，
> 超時後可能判定 SPLINE 掛掉。原文：「motion control reads
> `consumer_flags[CONSUMER_CONTROL]`; zero means motor status is
> unavailable and SPLINE may be hung after a timeout.」

### 5.4 ⚠️ 關節順序：四份文件互相矛盾

**這是最容易炸機的地方，一定要用 Phase 0 實測確認。**

| 出處 | 順序 |
|---|---|
| SHM `lowlevel.h` 的 enum | `LEG_FRONT_RIGHT=0, LEG_FRONT_LEFT=1, LEG_BACK_RIGHT=2, LEG_BACK_LEFT=3` → **FR, FL, RR, RL** |
| `docs/architecture.md`「命令順序」 | FR, FL, RR, RL（一致） |
| `include/zsl-1/lowlevel.h` 的 `motorState` 註解 | **FL, FR, RL, RR**（相反） |
| 新 SDK `protocol.md` / README | Left-Front, Right-Front, Left-Rear, Right-Rear（也是 FL 先） |

看起來是**高層 API 用 FL 開頭、SHM 底層用 FR 開頭**，兩套慣例。

關節方向定義：ABAD / HIP / KNEE 座標系是「前 X、左 Y、上 Z」（見 `docs/images/joint_orient.jpg`）。

---

## 6. 一步一步的測試指南

分三個階段，風險由零遞增。**每個階段沒過，不要進下一階段。**

---

### Phase 0：唯讀偵察（零風險，半天可完成）

**目標**：證實或否定「輪足機上有可用的 `/spline_shm`」，順便量出真正的關節順序。
**風險**：零。全程只讀，不寫任何東西。

---

#### 步驟 0-1：連上機器人

用 WiFi 連機器狗的熱點（SSID 和密碼在機器右側標籤上），然後：

```bash
ssh firefly@192.168.234.1
# 密碼：firefly
```

如果用網線，IP 是 `192.168.168.168`，而且你電腦的有線網卡要手動設成 168 網段的固定 IP（網線沒有 DHCP）。

---

#### 步驟 0-2：確認機型與版本（在機器上執行）

```bash
# 確認 CPU 架構（決定等一下怎麼編譯）
uname -m
# 預期輸出：aarch64

# 確認運控版本（輪足要 0.3.1 以上）
grep -oP 'motion-control_\K[^_]+' /etc/release/*[^rootfs]*.yaml

# 確認這台真的是輪足版（看有沒有 xgw 的啟動腳本）
ls -l /opt/app_launch/
```

**判讀**：
- 看到 `start_motion_control_xgw.sh` → 這台是輪足版 ✓
- 看到 `start_motion_control.sh`（沒有 xgw）→ 這台是點足版，那你直接有官方 LowLevel 可用，不需要走這條路

---

#### 步驟 0-3：★ 關鍵一步 — 看共享記憶體在不在

```bash
ls -l /dev/shm/
```

**這一行就決定了整條路通不通。**

| 你看到什麼 | 代表什麼 | 下一步 |
|---|---|---|
| 有 `spline_shm`，大小 10240 | ✅ 介面存在，繼續 | 步驟 0-4 |
| 有 `spline_shm` 但大小是 0 | ⚠️ 可能是空殼 | 還是繼續，但要小心 |
| 沒有 `spline_shm` | ❌ 這條路在這台機器上不通 | 停，去問原廠 |

順便看看另外兩塊：

```bash
ls -l /dev/shm/imu_shm /dev/shm/bms_shm
```

同時確認運控程式在跑：

```bash
ps aux | grep -E "mc_ctrl|spline"
```

> 🚨 **重要陷阱**：官方標頭檔的 `create_spline_shm()` 用的是 `shm_open(..., O_CREAT | O_RDWR, 0666)`。
> `O_CREAT` 的意思是「不存在就建一個」。如果你直接跑官方範例，
> 就算機器上原本沒有這塊記憶體，它也會**幫你憑空建一塊全是 0 的**，
> 然後你會讀到一堆 0，誤以為「介面在但沒資料」。
> **所以一定要先用 `ls` 肉眼確認，而且我下面的程式碼刻意不加 `O_CREAT`。**

---

#### 步驟 0-4：準備編譯環境

先看機器上有沒有編譯器：

```bash
which g++ gcc cmake
g++ --version
```

**情況 A — 機器上有 g++（最簡單）**：直接在機器上編。

**情況 B — 機器上沒有 g++**：在你的電腦上交叉編譯。

```bash
# 在你的 Ubuntu 電腦上
sudo apt install g++-aarch64-linux-gnu
aarch64-linux-gnu-g++ -O2 -o shm_probe shm_probe.cpp -lrt
scp shm_probe firefly@192.168.234.1:~/
```

---

#### 步驟 0-5：寫唯讀偵察程式

把下面存成 `shm_probe.cpp`（完整程式在附錄 A，這裡說明它做什麼）：

程式做四件事：
1. 用**唯讀**模式（`O_RDONLY` + `PROT_READ`）開啟 `/spline_shm`，**不加 `O_CREAT`**
2. 印出 4 條腿 × 4 個關節的 `p / v / t` 和解碼後的 flags（溫度、電壓、故障位）
3. 同樣唯讀開啟 `/imu_shm`，印出四元數和時戳
4. 每 100 ms 更新一次畫面

編譯（在機器上）：

```bash
g++ -O2 -o shm_probe shm_probe.cpp -lrt
```

執行：

```bash
./shm_probe
```

---

#### 步驟 0-6：判讀結果

**訊號 A — 完全通了 ✅**

- 16 個關節都有非零的 `p` 值
- IMU 的時戳在跳（每次刷新都變大）
- 你用手輕輕搖一下狗，`p` 和 `v` 的數字跟著動
- flags 解碼出來的溫度是合理值（20~50 °C），電壓合理（24V / 48V 之類）

→ **恭喜，介面是活的，進步驟 0-7。**

**訊號 B — 讀得到但全是 0 ❌**

- 所有 `p / v / t` 都是 0.0，flags 也是 0

→ 有兩個可能：(a) 你不小心建了一塊空的（回頭確認有沒有加 `O_CREAT`）；(b) 這塊記憶體不是你以為的那塊。停下來，去問原廠。

**訊號 C — 開不起來 / Permission denied ❌**

→ 試試 `sudo ./shm_probe`。如果 sudo 才行，記下來（代表要 root 權限，之後部署要考慮）。如果連 sudo 都說檔案不存在，這條路不通。

**訊號 D — 數字看起來像亂碼**

→ 可能是結構體對齊或版本不合。把原始 bytes dump 出來（程式有 `--raw` 選項），對照第 5.2 節的結構自己算一次。

---

#### 步驟 0-7：★ 量出真正的關節順序（最有價值的一步）

這一步解決第 5.4 節那個四份文件互相矛盾的問題。

**準備**：
1. 用遙控器讓狗 `passive`（卸力）或趴下，確保關節可以用手扳動
2. 開著 `./shm_probe`，畫面持續刷新

**測試**：
1. **只**用手抬起**右前腿**，慢慢彎它的膝蓋
2. 看畫面上哪一組 `legs[N].knee.p` 在變
3. 記下來：`N = ?`
4. 對左前、右後、左後各做一次

**填這張表**（測完貼在你的程式註解裡）：

| 實體腿 | `legs[?]` 索引 | 文件說的 |
|---|---|---|
| 右前 FR | ___ | SHM enum 說 0 |
| 左前 FL | ___ | SHM enum 說 1 |
| 右後 RR | ___ | SHM enum 說 2 |
| 左後 RL | ___ | SHM enum 說 3 |

**同時量正負方向**（這個一樣重要）：
- 膝蓋往「彎曲」方向扳，`p` 是變大還是變小？
- 對照 URDF 的 KNEE 範圍 `-2.723 ~ -0.602`（全負值），確認你的 MJCF 符號一致
- HIP 往前抬，`p` 變大還是變小？對照 URDF `-1.152 ~ 2.967`
- ABAD 往外張，`p` 變大還是變小？對照 URDF `±0.4887`

**測輪子**：用手轉一顆輪子，看 `legs[N].foot.p` 有沒有在累加（應該會無限增加，因為是 continuous 關節），以及 `foot.v` 有沒有反應。

---

#### 步驟 0-8：錄一段走路資料

改用 CSV 模式跑（附錄 A 的程式支援 `--csv`），用遙控器讓狗走個 30 秒：

```bash
./shm_probe --csv --rate 1000 > walk_log.csv
# 另一個終端或用遙控器讓狗走路
```

拉回你的電腦分析：

```bash
scp firefly@192.168.234.1:~/walk_log.csv .
```

**這份資料的價值**（就算後面寫指令走不通，這也已經賺到了）：
- 驗證你的 MJCF 關節限位、慣量參數對不對
- 量真實的關節力矩範圍 → 校正你 RL 訓練的 torque limit
- 量馬達的實際響應延遲
- 拿到官方運控的步態當 reference，給你的 CPG 參數當初始值

**Phase 0 到此結束。**
就算後面兩個 Phase 都不做，你也已經得到 1 kHz 全關節遙測 + 確認的關節順序 + 一份真實步態資料。

---

### Phase 1：在模擬器裡測寫入（低風險）

**目標**：在不碰實機的情況下，把「寫指令」的整套邏輯寫完測完。
**風險**：低（弄壞了大不了重跑模擬器）。
**前提**：Phase 0 的訊號 A 成立。

#### 步驟 1-1：裝 MATRiX

從 [`zsibot/matrix`](https://github.com/zsibot/matrix) 的 Releases 下載（repo 本身只有文件，執行檔在 release）。參考它的 `docs/Getting_Started_CN.md` 和 `docs/Docker_Tutorial.md`。

#### 步驟 1-2：用 xgw 機型 + 啟用模擬硬體

機器人設定檔的關鍵欄位：

```json
{
  "robot": {
    "robot_type": "xgw",
    "mujoco_model": "xgw/scene.xml",
    "inside_mc": false
  }
}
```

`inside_mc: false` 很重要 — 這樣才不會啟動內建運控來跟你搶。

按官方建議的順序：
1. 啟動 UeSim（開啟 simulated hardware）
2. **等共享記憶體與機器人狀態初始化完成**
3. 才啟動你的外部控制程式
4. 要停的時候，先讓機器人進安全狀態，再退出控制程式，最後關模擬器

#### 步驟 1-3：在模擬裡驗證這五件事

1. **旗標握手**：你寫 `cmd.consumer_flags[CONSUMER_CONTROL] = 1` 之後，它會不會被清成 0？（會清 = daemon 有收到）
2. **Watchdog**：故意停止送指令 20 ms，馬達是不是真的鬆掉？
3. **迴圈頻率**：你的迴圈能不能穩定跑 1 kHz？量抖動（jitter）
4. **零指令**：送 `kp = kd = t_ff = 0` 全零，機器人應該完全癱軟不動
5. **單關節正弦**：只給一顆膝蓋很小的 `kp`（例如 5）跟緩慢正弦的 `p_des`，看它會不會跟

**這五項全綠才進 Phase 2。**

---

### Phase 2：實機吊掛測試（有風險）

> 🚨 **這個階段可能弄壞機器，而且很可能不在保固範圍內。**
> 建議在做 Phase 2 之前，先把第 8 節那封信寄給原廠，拿到答覆再說。

#### 步驟 2-1：物理安全準備

- **把狗吊起來，四個輪子完全離地**（用吊架或掛在門框上，繩子要能承重）
- 周圍淨空 2 公尺
- **旁邊放電源開關，手指放在上面**
- 兩個人做，一個看螢幕一個看機器

#### 步驟 2-2：先確認你能還原

在動任何東西之前先確認：**重開機就能回到原狀**。

```bash
# 記錄現在的狀態，方便對照
ps aux | grep mc_ctrl > ~/before.txt
systemctl list-units | grep -i motion >> ~/before.txt
```

確認 `reboot` 之後 `mc_ctrl` 會自己起來（它就是被 `/opt/app_launch/` 的腳本拉起來的）。

#### 步驟 2-3：讓狗趴下並卸力

**先趴下再停 mc_ctrl。** 站著的時候停掉運控，狗會直接摔下去（雖然是吊著的，但關節會受衝擊）。

用遙控器或高層 SDK：`lieDown()` → `passive()`

#### 步驟 2-4：停掉 mc_ctrl

先找到它是誰在管：

```bash
ps aux | grep mc_ctrl
systemctl list-units --type=service | grep -iE "motion|mc|app_launch"
```

然後用對應的方式停（優先用 systemctl，找不到再 kill）：

```bash
# 如果是 systemd 服務
sudo systemctl stop <服務名>

# 找不到服務就直接停行程
sudo kill <mc_ctrl 的 PID>
```

確認停了：

```bash
ps aux | grep mc_ctrl   # 應該只剩 grep 自己那行
```

> ⚠️ 停掉 mc_ctrl 之後：**遙控器失效、高層 SDK 失效、所有內建保護失效。**
> 從這一刻起，機器的安全完全靠你的程式和電源開關。

#### 步驟 2-5：全零指令測試

跑一支只送全零的程式（`p_des = v_des = kp = kd = t_ff = 0`，`flags = 1`），1 kHz。

**預期**：機器完全癱軟，關節可以用手自由扳動，沒有任何馬達出力或異音。

**如果有任何抽搐、異音、發熱 → 立刻停，關電源。**

#### 步驟 2-6：Watchdog 驗證

程式跑著的時候，直接 `Ctrl+C` 砍掉它。

**預期**：機器維持癱軟（本來就是零指令），而且不會鎖死。

再做一次進階版：送一個小的持續力矩（例如某顆膝蓋 `t_ff = 0.5`），確認有輕微出力後，`Ctrl+C`。

**預期**：約 10 ms 後力矩消失，關節鬆掉。**這證明 watchdog 有效，是你之後所有實驗的救命索。**

#### 步驟 2-7：單關節微幅正弦

挑**一顆**膝蓋（用 Phase 0 量出來的索引，不要相信文件）：

```
p_des = 目前角度 + 0.05 * sin(2π * 0.5 * t)   // 振幅 0.05 rad ≈ 2.9°，頻率 0.5 Hz
v_des = 0
kp    = 5      // 從很小開始
kd    = 0.5
t_ff  = 0
```

其他 15 個關節全部保持零指令。

**預期**：那顆膝蓋緩慢小幅擺動，其他關節不動。

**逐步放大**：確認穩定後，kp 5 → 10 → 20，振幅 0.05 → 0.1 → 0.2。每次只改一個變數，每次都準備好隨時斷電。

#### 步驟 2-8：往上疊

順序建議：單關節 → 單腿三關節協調 → 四腿同步站立姿態保持 → 你的 CPG-RL policy（仍然吊掛）→ 落地低速。

**每一步都吊著測完再說。**

---

## 7. 風險清單

| 風險 | 說明 | 緩解 |
|---|---|---|
| 沒有力矩/速度限幅 | SHM 層是裸介面，你送 100 N·m 它就送 100 N·m | 自己在程式裡做 clamp（URDF 說 effort 28 N·m，velocity 28 rad/s） |
| 沒有軟啟動 | 第一幀就是全力 | 第一幀一定送 kp=kd=0，之後 ramp up |
| 停掉 mc_ctrl 後無保護 | 遙控器、緊急停止、防摔全失效 | 物理吊掛 + 手放電源開關 |
| 關節超限 | 沒有軟限位保護 | 程式裡對 `p_des` 做 URDF 限位 clamp，並留 10% 餘裕 |
| 兩個 producer 打架 | mc_ctrl 沒停乾淨就寫入 | 每次寫入前 `ps aux \| grep mc_ctrl` 確認 |
| 保固 | 撞壞燒馬達大概不在保固內 | 先問原廠（第 8 節） |
| 設定被重置 | 文件明講「設備程式更新後 `/opt` 設定會被重置」 | 所有改動寫成腳本，放版本控制 |
| 誤清 CONSUMER_CONTROL | 會害 mc_ctrl 判定 SPLINE 掛掉 | 唯讀階段用 `PROT_READ` 從根本上防止 |

---

## 8. 建議寄給原廠的信（可直接複製）

> 主旨：D1 EDU 輪足版（zsl-1w）底層馬達控制介面諮詢
>
> 您好，
>
> 我們單位使用 D1 EDU 輪足版（zsl-1w）進行四足運動控制研究，
> 目前在做強化學習步態的 sim-to-real 部署，有幾個問題想請教：
>
> 1. 官方 SDK 文件中說明「ZSL-1w 不提供 LowLevel 接口」，
>    請問輪足版未來是否有開放關節層級控制（位置/速度/力矩）的計畫？時程大約？
>
> 2. SDK 中的 `include/lowlevel/lowlevel.h` 定義了透過 `/spline_shm`
>    共享記憶體的馬達控制介面，其 `leg_control_t` 結構包含 foot（輪）關節。
>    請問這個介面在輪足機型上是否可用？是否允許第三方程式寫入該共享記憶體？
>
> 3. 若要以自行開發的控制程式取代 `mc_ctrl`，
>    官方是否有提供對應的開發套件、介面文件或安全規範？
>
> 4. MATRiX 模擬平台的「Linux 模擬硬體」模式支援外部 `mc_ctrl` 接入，
>    請問這個機制是否適用於第三方自行開發的控制器？
>
> 5. 若我們自行透過該介面進行開發，是否會影響保固？
>
> 感謝協助。

---

## 9. 參考連結

**新的官方 SDK（維護中）**
- https://github.com/zsibot/genisom_L1_sdk — v1.0.0 / 2026-06-22 / MIT
- `include/zsibot_sdk/lowlevel/lowlevel.h` — 共享記憶體結構定義
- `docs/zh/api_lowlevel.md` — 中文說明
- `docs/protocol/protocol.md` — UDP + JSON 協定

**舊的 SDK（已停止維護，但文件比較全）**
- https://github.com/AgibotTech/agibot_D1_Edu-Ultra
- https://github.com/zsibot/genisom_l1_sdk_old — 同一份，官方標示 deprecated
- `docs/deploy.md` — 部署步驟、「ZSL-1w 不提供 LowLevel」那句話在第 168 行
- `docs/architecture.md` — 關節順序、方向定義、狀態機圖
- `docs/faq.md` — lowlevel 500 Hz / highlevel 50 Hz；兩者不可同時使用

**模型與模擬器**
- https://github.com/zsibot/genisom_model — ZSL-1 / **ZSL-1W URDF**
- https://github.com/zsibot/matrix — MuJoCo + UE5 模擬平台，`docs/Motion_Control_CN.md` 有「Linux 模擬硬體」模式

**其他**
- https://github.com/AgibotTech/Agibot_D1_Max — D1 Max（zsm-1w）另一套 SDK

---

## 附錄 A：Phase 0 唯讀偵察程式

```cpp
// shm_probe.cpp — D1 EDU 輪足版 /spline_shm 唯讀偵察工具
//
// 編譯（機器上）： g++ -O2 -o shm_probe shm_probe.cpp -lrt
// 交叉編譯：       aarch64-linux-gnu-g++ -O2 -o shm_probe shm_probe.cpp -lrt
//
// 用法：
//   ./shm_probe              人眼可讀模式，10 Hz 刷新
//   ./shm_probe --csv        CSV 模式，1 kHz，輸出到 stdout
//   ./shm_probe --raw        傾印原始 bytes（結構對不上時用）
//
// ★ 安全設計：全程 O_RDONLY + PROT_READ，不加 O_CREAT。
//   讀不到就是讀不到，絕不會憑空建一塊假的騙自己。

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctime>

// ---- 結構定義（照抄官方 lowlevel.h，順序與 packed 不可改）----
typedef struct { float p_des, v_des, kp, kd, t_ff; }
    __attribute__((packed)) joint_control_t;

typedef struct { joint_control_t abad, hip, knee, foot; int32_t flags; }
    __attribute__((packed)) leg_control_t;

typedef struct { leg_control_t legs[4]; uint32_t consumer_flags[2]; }
    __attribute__((packed)) spline_cmd_data_t;

typedef struct { int32_t flags; float p, v, t; }
    __attribute__((packed)) joint_state_t;

typedef struct { joint_state_t abad, hip, knee, foot; }
    __attribute__((packed)) leg_state_t;

typedef struct { leg_state_t legs[4]; uint32_t consumer_flags[2]; }
    __attribute__((packed)) spline_state_data_t;

typedef struct { spline_cmd_data_t cmd; spline_state_data_t state; }
    __attribute__((packed)) spline_data_t;

typedef struct {
    size_t timestamp;   // 奈秒
    float acc[3];       // m/s^2
    float gyro[3];      // rad/s
    float q[4];         // w, x, y, z（旋轉順序 zyx）
} __attribute__((packed)) nav_imu_t;

// ---- 唯讀開啟共享記憶體（刻意不加 O_CREAT）----
static void* open_shm_ro(const char* path, size_t size) {
    int fd = shm_open(path, O_RDONLY, 0666);
    if (fd == -1) {
        fprintf(stderr, "[X] 開不了 %s：%s\n", path, strerror(errno));
        fprintf(stderr, "    → 先用 `ls -l /dev/shm/` 確認檔案在不在\n");
        fprintf(stderr, "    → 若是 Permission denied，試試 sudo\n");
        return nullptr;
    }
    void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[X] mmap %s 失敗：%s\n", path, strerror(errno));
        return nullptr;
    }
    return p;
}

static const char* LEG_NAME[4] = {"leg0", "leg1", "leg2", "leg3"};
// ↑ 刻意不寫 FR/FL/RR/RL —— 因為官方四份文件互相矛盾。
//   請用步驟 0-7 手動扳動關節，實測出真正的對應再改這裡。

static void decode_flags(int32_t f, char* out, size_t n) {
    int temp = ((f >> 8) & 0xFF) - 40;   // -40 ~ 215 °C
    int volt = (f >> 16) & 0xFF;         // 0 ~ 255 V
    snprintf(out, n, "en=%d %s%s%s%s%s T=%d°C V=%dV",
             f & 1,
             (f >> 1) & 1 ? "過壓 " : "",
             (f >> 2) & 1 ? "過流 " : "",
             (f >> 3) & 1 ? "過溫 " : "",
             (f >> 4) & 1 ? "超速 " : "",
             (f >> 5) & 1 ? "雙編碼器故障 " : "",
             temp, volt);
}

static void print_human(const spline_data_t* d, const nav_imu_t* imu) {
    printf("\033[2J\033[H");   // 清畫面
    printf("=== /spline_shm 唯讀偵察 ===\n\n");
    const char* JN[4] = {"ABAD", "HIP ", "KNEE", "FOOT"};
    for (int i = 0; i < 4; ++i) {
        const joint_state_t* js[4] = {
            &d->state.legs[i].abad, &d->state.legs[i].hip,
            &d->state.legs[i].knee, &d->state.legs[i].foot };
        printf("[%s]\n", LEG_NAME[i]);
        for (int j = 0; j < 4; ++j) {
            char fb[128];
            decode_flags(js[j]->flags, fb, sizeof fb);
            printf("  %s  p=%+8.4f rad  v=%+8.4f rad/s  t=%+7.3f N·m  | %s\n",
                   JN[j], js[j]->p, js[j]->v, js[j]->t, fb);
        }
        printf("\n");
    }
    printf("state.consumer_flags = [%u, %u]   (index0=mc_ctrl 的, index1=你的)\n",
           d->state.consumer_flags[0], d->state.consumer_flags[1]);
    if (imu) {
        printf("\nIMU  ts=%zu ns\n", imu->timestamp);
        printf("  quat(wxyz) = %+.4f %+.4f %+.4f %+.4f\n",
               imu->q[0], imu->q[1], imu->q[2], imu->q[3]);
        printf("  acc  = %+.3f %+.3f %+.3f m/s^2\n",
               imu->acc[0], imu->acc[1], imu->acc[2]);
        printf("  gyro = %+.3f %+.3f %+.3f rad/s\n",
               imu->gyro[0], imu->gyro[1], imu->gyro[2]);
    }
    printf("\n(Ctrl+C 離開)  ★ 用手扳單一關節，看哪個 leg 索引在動\n");
    fflush(stdout);
}

static void print_csv_header() {
    printf("t_ns");
    const char* JN[4] = {"abad", "hip", "knee", "foot"};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            printf(",leg%d_%s_p,leg%d_%s_v,leg%d_%s_t,leg%d_%s_flags",
                   i, JN[j], i, JN[j], i, JN[j], i, JN[j]);
    printf(",imu_ts,qw,qx,qy,qz,ax,ay,az,gx,gy,gz\n");
}

static void print_csv_row(const spline_data_t* d, const nav_imu_t* imu, uint64_t t) {
    printf("%llu", (unsigned long long)t);
    for (int i = 0; i < 4; ++i) {
        const joint_state_t* js[4] = {
            &d->state.legs[i].abad, &d->state.legs[i].hip,
            &d->state.legs[i].knee, &d->state.legs[i].foot };
        for (int j = 0; j < 4; ++j)
            printf(",%.6f,%.6f,%.6f,%d", js[j]->p, js[j]->v, js[j]->t, js[j]->flags);
    }
    if (imu)
        printf(",%zu,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
               imu->timestamp, imu->q[0], imu->q[1], imu->q[2], imu->q[3],
               imu->acc[0], imu->acc[1], imu->acc[2],
               imu->gyro[0], imu->gyro[1], imu->gyro[2]);
    else
        printf(",0,0,0,0,0,0,0,0,0,0,0");
    printf("\n");
}

int main(int argc, char** argv) {
    bool csv = false, raw = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv")) csv = true;
        if (!strcmp(argv[i], "--raw")) raw = true;
    }

    printf("[*] 結構大小自我檢查：spline_data_t = %zu bytes（應該遠小於 10240）\n",
           sizeof(spline_data_t));

    void* sp = open_shm_ro("/spline_shm", 1024 * 10);
    if (!sp) return 1;
    void* ip = open_shm_ro("/imu_shm", 1024 * 1);   // 沒有也不致命

    const spline_data_t* d   = (const spline_data_t*)sp;
    const nav_imu_t*     imu = (const nav_imu_t*)ip;

    if (raw) {   // 結構對不上時，肉眼看原始 bytes
        const unsigned char* b = (const unsigned char*)sp;
        for (int i = 0; i < 256; ++i) {
            printf("%02x ", b[i]);
            if (i % 16 == 15) printf("\n");
        }
        return 0;
    }

    if (csv) print_csv_header();

    // CSV 模式 1 kHz，人眼模式 10 Hz
    long period_ns = csv ? 1000000L : 100000000L;
    while (true) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        uint64_t t = (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;

        if (csv) print_csv_row(d, imu, t);
        else     print_human(d, imu);

        struct timespec ts = {0, period_ns};
        nanosleep(&ts, nullptr);
    }
    return 0;
}
```

**已驗證**：這支程式用 `g++ -O2 -Wall` 編譯無警告無錯誤；`sizeof(spline_data_t)` 算出來是 **608 bytes**（遠小於官方配置的 10240，合理）。在沒有共享記憶體的機器上執行，它會正確地印出「開不了 /spline_shm：No such file or directory」並結束 —— 也就是說它**不會**憑空建一塊假的。

> **注意這支程式做了什麼、沒做什麼**
> - ✅ 只讀，`PROT_READ` 從記憶體保護層級保證寫不進去
> - ✅ 不加 `O_CREAT`，讀不到就誠實報錯，不會建假的騙你
> - ❌ **沒有**清 `consumer_flags[CONSUMER_OTHER]`。照協定第三方讀完應該要清，
>   但清就是寫。Phase 0 的目的是偵察不是當正式 consumer，所以刻意不做。
>   等到 Phase 2 要正式接入時再改成可寫並正確清 `CONSUMER_OTHER`（**永遠不要碰 index 0**）。

---

## 附錄 B：檢查清單

**Phase 0**
- [ ] SSH 連得上機器人
- [ ] `uname -m` = aarch64
- [ ] 運控版本 ≥ 0.3.1
- [ ] `/opt/app_launch/start_motion_control_xgw.sh` 存在（確認是輪足版）
- [ ] `ls -l /dev/shm/` 看到 `spline_shm`（大小 10240）
- [ ] `imu_shm`、`bms_shm` 也在
- [ ] `shm_probe` 編譯成功
- [ ] 16 個關節讀到非零合理值
- [ ] IMU 時戳在跳
- [ ] 溫度、電壓解碼合理
- [ ] **量出真正的 leg 索引對應**（填第 0-7 步的表）
- [ ] **量出每個關節的正負方向**，跟 URDF 對照
- [ ] 輪子（foot）轉動時 `p` 會累加
- [ ] 錄到 30 秒走路 CSV

**Phase 1**
- [ ] MATRiX 裝起來，`xgw` 機型跑得動
- [ ] 模擬硬體模式下共享記憶體出現
- [ ] 旗標握手：設 1 之後會被清 0
- [ ] Watchdog：停送 20 ms 後馬達鬆掉
- [ ] 控制迴圈穩定 1 kHz
- [ ] 全零指令 → 完全癱軟
- [ ] 單關節正弦追得上

**Phase 2**（做之前先寄第 8 節那封信）
- [ ] 狗吊起來，輪子離地
- [ ] 兩個人、手放電源開關
- [ ] 確認 reboot 能還原
- [ ] 先 lieDown + passive
- [ ] mc_ctrl 停乾淨（`ps aux` 確認）
- [ ] 全零指令 → 癱軟無異音
- [ ] Watchdog 實機驗證通過
- [ ] 單關節微幅正弦（kp=5，振幅 0.05 rad）
