# zgws MJCF 來源與驗證

`zgws.xml` 是 **智元官方的 D1 Max MuJoCo 模型**，不是我們從 URDF 轉的。

- 機型：`zgws` = `zsm-1w` = **D1 Max**
- 來源：`zsibot/matrix` v0.1.2 發布包 `base-0.1.2.tar.gz` → `Content/model/zgws/zgws.xml`
- sha256（tarball）：`2cbb40861e89c40735cd64b24e8b64d88d012f335bdb405b5ed52db86f8b4e38`（與 release manifest 一致）

```bash
curl -L -O https://github.com/zsibot/matrix/releases/download/v0.1.2/base-0.1.2.tar.gz
tar xzf base-0.1.2.tar.gz ./Content/model/zgws/
```

## 檔案

| 檔案 | 說明 |
|---|---|
| `zgws.xml` | ★ **官方原檔，逐字元未改**。16 個關節 + 感測器定義 |
| `assets/*LINK.STL` | 官方網格，17 個。**54 MB，不進版控** → 用 `fetch_assets.sh` 取回 |
| `scene_flat.xml` | **我們自己寫的**乾淨平地場景（`include zgws.xml` + 一塊地板） |
| `fetch_assets.sh` | 重新取回網格 |

官方原本附的 `scene_terrain_*.xml`（20 幾個地圖場景）沒有收，
它們塞滿障礙物與 hfield、依賴外部 png，不適合調步態。要用時從發布包取。

## 已做的驗證（2026-08-25，MuJoCo 3.10 / conda env `rbtdog`）

```
✅ zgws.xml 載入成功        nq=23  nv=22  nu=16  nbody=18
✅ scene_flat.xml 載入成功  同上
   總質量 38.821 kg        timestep 0.002（未指定 <option>，MuJoCo 預設值）
```

### 關節順序（qpos 位址）

```
qpos[0:7]   基座 freejoint
qpos[7,8,9]     FAR = FR  ABAD / HIP / KNEE      qpos[10] FR 輪
qpos[11,12,13]  FBL = FL  ABAD / HIP / KNEE      qpos[14] FL 輪
qpos[15,16,17]  RAR = RR  ABAD / HIP / KNEE      qpos[18] RR 輪
qpos[19,20,21]  RBL = RL  ABAD / HIP / KNEE      qpos[22] RL 輪
```

**腿序 FR, FL, RR, RL**，與 `zg_wheels-motion_config.yaml`（原廠動作腳本）一致。
腿名對應由 MJCF 自己的 `<sensor>` 段確認（`FR_hip_pos` 綁 `FAR_ABAD_JOINT`，餘類推）：
`FAR→FR`、`FBL→FL`、`RAR→RR`、`RBL→RL`。

⚠️ **12 個腿關節在 qpos 裡不連續**（每 3 個之後夾一個輪關節）。
task6 踩過這個坑：`qpos[7:19]` 這種寫法會把輪角當關節角讀進去，IK 變奇異矩陣而不報錯。
照 task6 `d1_model.LEG_QPOS_IDX` 的做法明列位址。

### ★ 站姿驗證：確認了「前後鏡像的 X 型」

把原廠設定檔的姿態餵進去做正向運動學：

| 姿態 | 結果 |
|---|---|
| 原廠站姿 `hip=[.6,.6,−.6,−.6]` `knee=[−1.2,−1.2,1.2,1.2]` | ✅ 四輪 x = **±0.3398**，前後對稱；前腿膝在輪後、後腿膝在輪前（**X 型**） |
| 四腿同號 `hip=[.6]*4` `knee=[−1.2]*4` | ❌ 後腿膝 x = −0.4753 但輪 x = −0.3172 → 後腿整條往後翹，前後**不對稱** |

→ **原廠那組（前後反號）才是對的。**
MJCF 本身也佐證：後腿的 hip 限位是鏡像的（`RAR_HIP range=[−2.791, 2.442]`
vs `FAR_HIP range=[−2.442, 2.791]`），URDF 那份則四腿都寫 ±2.443。

### ★ 質心幾乎沒有前後偏移（與 D1 EDU 差很多）

原廠站姿下：四輪 x 平均 = +0.0000、全機質心 x = −0.0006 → **偏移只有 −0.6 mm**。

對照 task6 的 D1 EDU：質心偏後 **17 mm**，逼得 CPG 要用 `x_off = −40 ~ −55 mm` 配平，
而且那個值隨步態與步頻變動、低步頻下還是雙穩態。

→ **這台可能幾乎不需要 `x_off`。** 這是個好消息，但**要用實際步態驗證**——
靜態質心對齊不保證動態俯仰也對齊。

### ★ 機身高度與原廠 `body_height` 對得上

| 姿態 | 機身離地 |
|---|---|
| 原廠站姿 hip ±0.6 / knee ∓1.2 | 0.5418 m |
| **RL 預設姿 hip ±0.8 / knee ∓1.5** | **0.4914 m** ← 對上原廠 `body_height: 0.48` |
| 趴姿 hip ±1.4 / knee ∓2.4 | 0.2916 m |

→ **走路的名目站姿是「RL 預設姿」那組**，不是「站姿」那組。

### 輪半徑

從 FOOT 碰撞網格（31730 個頂點）實算：**0.0961 m**。
規格書寫 0.09、原廠設定檔 `feet_z_offset: 0.095`。**用 0.0961。**

## ⚠️ 官方 MJCF 與官方 URDF 對不上的地方（兩份都是官方的）

| 項目 | MJCF (`zgws.xml`) | URDF (`max.urdf`) | 規格書 |
|---|---|---|---|
| **總質量** | **38.821 kg** | 41.045 kg | 41 kg |
| BASE 質量 | 17.033 | 20.25 | — |
| HIP 質量 | 2.8754 | 2.6525 | — |
| KNEE 質量 | 0.86312 | 1.5888 | — |
| FOOT（輪）質量 | 1.5113 | 0.688 | 1.275 |
| 髖 x 位置 | 0.2698 | 0.272 | — |
| HIP 限位 | 前 −2.442~2.791、**後鏡像** | 四腿都 ±2.443 | ±140° = ±2.443 |
| KNEE 限位 | ±2.791 | ±2.801 | ±160° = ±2.793 |
| 輪扭矩上限 | 40 N·m | 50 N·m | 33 N·m |

**判斷**：質量分佈以 **MJCF 為準**（它是官方模擬器實際在跑的那份，運控也是照它調的），
但要知道它比規格書輕 2.2 kg——**可能是沒算雙電池或載荷**。
扭矩上限**保守取規格書的 33 N·m（輪）/ 150 N·m（腿）**。

## ⚠️ 直接用這份 MJCF 之前要處理的三件事

1. **致動器是純力矩 `<motor>`，沒有 ctrlrange、沒有 gear、沒有位置伺服。**
   官方的 PD 是由外部 `mc_ctrl` 提供的。要做位置控制得自己加
   `<position>` 致動器，或在迴圈裡自己算 PD（task6 是後者）。
   加的時候用原廠增益：ABAD 60 / HIP 120 / KNEE 120、Kd 1.0、輪 60/0.5。
2. **沒有 keyframe。** 沒有現成的初始站姿，要自己建（用上面驗過的 RL 預設姿）。
3. **輪關節沒有 damping / frictionloss。** 實機輪馬達有靜摩擦，
   task6 在 D1 EDU 上是用實測的掙脫門檻（0.3–0.5 N·m）填 `frictionloss`。
   這台還沒量過，先留空或給小值，並在文件裡註明是猜的。

## 相關

- 原廠運控參數：`task7/docs/D1Max_原廠運控參數_MATRiX解包_2026-08-25.md`
- 設定檔原件：`task7/reference/matrix_zgws/`
- 官方 URDF：`task7/model/max.urdf`
