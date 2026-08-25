# task7 — 智元 D1 Max（zsm-1w）控制

**這是一台跟 task6 的 D1 EDU 不同的狗，SDK 完全不共用。** 開這個 task 就是為了不要把兩台的
程式碼與參數混在一起。

## 現況（2026-08-25）

文件調查完成，**尚未上實機**。

| 路線 | 狀態 |
|---|---|
| A. 官方 SDK 高層控制 | ✅ 確認可用（站立/移動/姿態/讀 16 關節狀態），尚未實跑 |
| B. ROS2 `rt/lowcmd` 底層 | ⚠️ 同家族 MaxPro 有，D1 Max **待上機驗證** |
| C. `/dev/shm/spline_shm` | ⚠️ D1 EDU 走這條，D1 Max **待上機驗證** |

**★ 額外收穫**：從 MATRiX 官方發布包解出了**原廠運控的實際設定檔**
（增益、站/趴姿、關節零點慣例、步態排程參數）。可信度有交叉驗證——
同一批檔案裡 D1 EDU 那份的增益與 task6 實機量到的原廠值逐項吻合。
見 `docs/D1Max_原廠運控參數_MATRiX解包_2026-08-25.md`。

## 檔案

| 路徑 | 內容 |
|---|---|
| `docs/D1Max_控制方式調查_2026-08-25.md` | **主文件**。機型辨識、硬體規格、連線方式、SDK 完整 API、底層可行性證據、URDF 關節表、待確認清單 |
| `docs/D1Max_原廠運控參數_MATRiX解包_2026-08-25.md` | ★ 原廠增益、站姿、符號慣例、步態參數（從官方模擬器發布包解出） |
| `docs/HANDOFF_CPG步態_起步交接.md` | 給「做 CPG 步態」那條線的起步交接（可獨立閱讀） |
| `realbot/recon_d1max.sh` | 唯讀偵察腳本（不 sudo、不寫入、不停行程）。一次回答「底層開不開放」 |
| `reference/matrix_zgws/` | 原廠設定檔原件 + `SOURCE.md`（取得方式、可信度、三個判讀陷阱） |
| `model/zgws/` | ★ **官方 MJCF** + 平地場景 + 取網格腳本 + `SOURCE.md`（驗證結果與已知落差） |
| `model/max.urdf` | 官方 URDF（41.045 kg）。與 MJCF 質量分佈不同，見 `model/zgws/SOURCE.md` |

## 下一步

```bash
# 第一步就是這個，零風險，決定後面走哪條路
bash task7/realbot/recon_d1max.sh 192.168.234.1 192.168.168.100
```

然後照主文件第 9 節的待確認清單往下做。

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
