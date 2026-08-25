# task7 — 智元 D1 Max（zsm-1w）控制

**這是一台跟 task6 的 D1 EDU 不同的狗，SDK 完全不共用。** 開這個 task 就是為了不要把兩台的
程式碼與參數混在一起。

## 現況（2026-08-25）

> **接手請先讀 [`HANDOFF.md`](HANDOFF.md)** —— 完整現況、關鍵事實表、下一步、踩過的坑。

**底層控制鏈已實機打通。**

| 路線 | 狀態 |
|---|---|
| A. 官方 SDK 高層控制 | ✅ 可用（UDP 8082 listen 中），尚未實跑 |
| B. ROS2 `rt/lowcmd` | ❌ **不存在**，那是 D1 MaxPro 的做法 |
| B'. ros2_control command interface | ⚠️ 80 個存在，但全被 `joint_shm_controller` claimed |
| **C. `/dev/shm/joint_cmd`** | ★★★ **實機驗證通過**：寫入被接受、四顆輪都驅動過 |

已完成：寫入驗證（16/16）、四顆輪驅動、輪摩擦實測 0.15 N·m（已填進 MJCF）、
**座標換算式與座標系同框驗證**。下一個門檻是腿關節，**需要吊掛**。

**★ 額外收穫**：從 MATRiX 官方發布包解出了**原廠運控的實際設定檔**
（增益、站/趴姿、關節零點慣例、步態排程參數）。可信度有交叉驗證——
同一批檔案裡 D1 EDU 那份的增益與 task6 實機量到的原廠值逐項吻合。
見 `docs/D1Max_原廠運控參數_MATRiX解包_2026-08-25.md`。

## 檔案

### 文件（`docs/`）

| 路徑 | 內容 |
|---|---|
| **`../HANDOFF.md`** | ★ **接手入口**：現況、關鍵事實表、下一步、踩過的坑 |
| `D1Max_控制方式調查_2026-08-25.md` | 機型辨識、硬體規格、連線、SDK 完整 API、底層可行性證據 |
| `D1Max_原廠運控參數_MATRiX解包_2026-08-25.md` | 原廠增益、站姿、符號慣例、步態參數 |
| `實機偵察結果_第一趟_2026-08-25.md` | SHM 三塊、運控架構、ROS2 拓樸 |
| `實機偵察結果_第二趟_2026-08-25.md` | SHM 結構解碼、`ros2_control` 80 個介面 |
| `實機寫入結果_第三趟_2026-08-25.md` | 寫入驗證 16/16、心跳時戳的發現 |
| `實機單顆馬達驅動結果_2026-08-25.md` | 左前輪 22.08° |
| `實機四輪驅動結果_2026-08-25.md` | 四顆輪全部成功、摩擦 0.13–0.18 N·m |
| `座標換算式驗證結果_2026-08-25.md` | ★ 換算式 + 座標系同框驗證 |
| `HANDOFF_CPG步態_起步交接.md` | 給 CPG 那條線的起步交接 |
| `三機型對照表_2026-08-25.md` + `圖4_*.png/svg/pdf` | 給主管的橫向比較與論文風圖表 |
| `現場*.md`（4 份） | 上機時照著做，**設計成離線可用** |

### 工具（`realbot/`）

`shm_io.py`（底層，其他都 import 它）、`shm_decode.py`（離線解碼）、
`recon_d1max.sh` / `recon2_d1max.sh`（唯讀偵察）、
`M0_probe.py` → `M1_zero_write.py` → `M2_wheel_spin.py` → `M3_wheel_tour.py`（風險遞增）、
`M4_pose_capture.py`（姿勢擷取）、`M_faultwatch.py`（故障取證）。
用途與風險等級見 `HANDOFF.md`。

### 其他

| 路徑 | 內容 |
|---|---|
| `model/zgws/` | ★ 官方 MJCF（**已填實測輪摩擦**）+ 平地場景 + 取網格腳本 + `SOURCE.md` |
| `model/max.urdf` | 官方 URDF。與 MJCF 質量分佈不同，見 `model/zgws/SOURCE.md` |
| `reference/matrix_zgws/` | 原廠設定檔原件 + 可信度說明 |
| `logs/` | 六趟實機的原始輸出（含 SHM 二進位快照） |
| `inference/`、`tests/` | CPG 步態（由另一條線維護） |

## 下一步

見 **`HANDOFF.md`** 的「下一步」一節。摘要：

1. **[零風險]** MJCF knee 限位 ±2.791 → ±2.801（實測支持）
2. **[零風險]** 掃 `--kd` 找靜摩擦掙脫門檻
3. **[中，須吊掛]** 單一腿關節微動 ← **下一個真正的門檻**（41 kg / 150 N·m）

⚠️ 進到第 3 步前，建議先確認原廠對「第三方寫入 `/dev/shm/joint_cmd`」的態度與保固範圍。

## ✅ 官方 MJCF 已取得（不用自己從 URDF 轉了）

MATRiX（官方 MuJoCo+UE5 模擬器）的 `base-0.1.2.tar.gz` 裡有
`Content/model/zgws/zgws.xml` —— **智元官方的 D1 Max MuJoCo 模型**。
（`zgws` = `zsm-1w` = D1 Max，已用外觀比對確認。）

已收進 `model/zgws/`，**在 MuJoCo 3.10 載入通過**（nq=23 / nu=16 / 38.821 kg），
並用它驗證了原廠站姿、質心偏移、機身高度、輪半徑 —— 見 `model/zgws/SOURCE.md`。

```bash
bash task7/model/zgws/fetch_assets.sh   # 取回 54 MB 網格（未進版控）
conda run --no-capture-output -n rbtdog python -c \
  "import mujoco;m=mujoco.MjModel.from_xml_path('task7/model/zgws/scene_flat.xml');print(m.nq,m.nu)"
```

⚠️ 官方 MJCF 與官方 URDF **質量分佈對不上**（38.8 vs 41.0 kg），
且 MJCF 的致動器是純力矩、沒 keyframe、輪關節沒摩擦。三件事都要處理，見 `SOURCE.md`。

## ⚠️ 三個從 task6 帶過來會出事的東西

1. **增益與力矩門檻**：這台 41 kg、腿關節 150 N·m，是 D1 EDU 的五倍量級。
   task6 的 `kp=20/kd=0.7`、「力矩 >5 N·m 保護」全部不適用。
   （原廠 RL 用 ABAD 60 / HIP 120 / KNEE 120，**三個關節不同值**。）
2. **IP `192.168.168.100`**：在 D1 EDU 是我們電腦的靜態 IP，在 D1 Max 是**狗的 Orin NX**。
   會撞位址，電腦端要改別的（例如 .50）。
3. **SDK 程式碼**：`mc_sdk::` 那套跟這台的 `robot_sdk::SDKClient` 毫無關係，一行都不能重用。

外加一個模擬端的：**D1 Max 的站姿是前後鏡像的 X 型**
（`hip_stand_pos = [0.6, 0.6, −0.6, −0.6]`），D1 EDU 是四腿同號。
「四條腿共用一個 `HOME3`」的寫法照抄會錯。
