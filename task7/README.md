# task7 — 智元 D1 Max（zsm-1w）控制

**這是一台跟 task6 的 D1 EDU 不同的狗，SDK 完全不共用。** 開這個 task 就是為了不要把兩台的
程式碼與參數混在一起。

## 現況（2026-08-26）

> **接手請先讀 [`HANDOFF.md`](HANDOFF.md)** —— 完整現況、關鍵事實表、下一步、踩過的坑。

**底層控制鏈已實機打通。**

| 路線 | 狀態 |
|---|---|
| A. 官方 SDK 高層控制 | ✅ 可用（UDP 8082 listen 中），尚未實跑 |
| B. ROS2 `rt/lowcmd` | ❌ **不存在**，那是 D1 MaxPro 的做法 |
| B'. ros2_control command interface | ⚠️ 80 個存在，但全被 `joint_shm_controller` claimed |
| **C. `/dev/shm/joint_cmd`** | ★★★ **實機驗證通過**：寫入被接受、四顆輪都驅動過 |

已完成：寫入驗證（16/16）、四顆輪驅動、輪摩擦實測 0.15 N·m（已填進 MJCF）、
座標換算式與座標系同框驗證，以及 **2026-08-26 的 ★★ 16 顆馬達全控**
（吊帶懸吊，12 個腿關節符號全對 + 四輪同時驅動，並首次量到腿關節摩擦）。
見 `docs/實機腿關節與16顆全控結果_2026-08-26.md`。

**還沒做**：S5 姿勢切換（crouch/home/knee-back）、落地承重測試。

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
| **`實機腿關節與16顆全控結果_2026-08-26.md`** | ★★ **16 顆全控、腿關節摩擦、兩次邊界條件的教訓** |
| `腿關節與姿勢控制_設計_2026-08-26.md` | M5 的設計與理由 |
| `HANDOFF_CPG步態_起步交接.md` | 給 CPG 那條線的起步交接 |
| `三機型對照表_2026-08-25.md` + `圖4_*.png/svg/pdf` | 給主管的橫向比較與論文風圖表 |
| `現場*.md`（4 份） | 上機時照著做，**設計成離線可用** |

### 工具（`realbot/`）

`shm_io.py`（底層，其他都 import 它）、`coord.py`（★ 座標換算／限位／姿勢的單一事實來源）、
`shm_decode.py`（離線解碼）、`recon_d1max.sh` / `recon2_d1max.sh`（唯讀偵察）、
`M0_probe.py` → `M1_zero_write.py` → `M2_wheel_spin.py` → `M3_wheel_tour.py`（風險遞增）、
`M4_pose_capture.py`（姿勢擷取）、
**`M5_leg_pose.py`（腿關節與姿勢控制，★ 必須吊掛）**、
`estop_max.sh`（★ 急停）、`push_to_dog.sh`（傳檔+校驗）、`M_faultwatch.py`（故障取證）。
用途與風險等級見 `HANDOFF.md`；M5 的設計理由見
`docs/腿關節與姿勢控制_設計_2026-08-26.md`。

### 其他

| 路徑 | 內容 |
|---|---|
| `model/zgws/` | ★ 官方 MJCF（**已填實測輪摩擦**）+ 平地場景 + 取網格腳本 + `SOURCE.md` |
| `model/zgws/make_mjx_model.py` | ★ **MJX 訓練模型產生器**。官方 MJCF 的碰撞網格 98,569 頂點，MJX 會 OOM；產物 `zgws_mjx.xml` 零 STL 相依 |
| `model/max.urdf` | 官方 URDF。與 MJCF 質量分佈不同，見 `model/zgws/SOURCE.md` |
| `reference/matrix_zgws/` | 原廠設定檔原件 + 可信度說明 |
| `logs/` | 六趟實機的原始輸出（含 SHM 二進位快照） |
| `inference/`、`tests/` | CPG 步態與 CPG-RL（由另一條線維護，見下） |
| `notebooks/cpg_rl_max_colab.ipynb` | ★ **CPG-RL 訓練 notebook**，丟 Colab GPU 直接跑 |

### CPG 步態 / CPG-RL（純模擬，2026-08-27）

| 路徑 | 內容 |
|---|---|
| `inference/gait_baseline.py` | ★ **基準步態的唯一真實來源**（`walk`，180 s × 12 擾動 0 跌倒） |
| `inference/cpg_walk_max.py` | 開迴路 CPG rollout ＋ `Trace`（統計，與推論端共用） |
| `inference/cpg_sweep_max.py` | 多擾動掃描器（含記憶體守衛） |
| `inference/obs_max.py` | ★ **68 維觀測層的唯一定義**，Colab 與本機共用 |
| `inference/local_infer_max.py` | 載 RL 權重，在**原始網格模型**上回放／錄影／量指標 |
| `docs/基準步態凍結_D1Max_walk_2026-08-27.md` | 凍結參數與判準來源 |
| `docs/MJX模型對照_2026-08-27.md` | 訓練模型 vs 原始模型的落差（±2% 內）與三個踩過的坑 |
| `docs/CPG-RL_D1Max_設計_2026-08-27.md` | ★ **這條線的入口**：觀測層、reward、DR、驗收關卡 |
| `docs/現場操作卡_IMU平放複核.md` | ⚠️ **上實機前必做**（唯讀、零風險） |

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
