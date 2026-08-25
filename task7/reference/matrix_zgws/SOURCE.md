# 來源與可信度

這幾個 YAML 是從 **zsibot/matrix（MATRiX 官方 MuJoCo + UE5 模擬平台）v0.1.2 的發布包**
解出來的**原廠運動控制設定檔**，不是我們自己寫的。

## 取得方式（可重現）

```bash
curl -L -O https://github.com/zsibot/matrix/releases/download/v0.1.2/assets-0.1.2.tar.gz
# sha256 = c4af445e468bb919909176113a8b8e9b0ede5dabf80be7112a5e7e86085bb369
#   ★ 與 release 的 manifest-0.1.2.json 一致。
#   ⚠️ 同一個 release 的 checksums-0.1.2.sha256 對這個檔寫的是 3099ea01...，對不上，那份是舊的。
tar xzf assets-0.1.2.tar.gz ./src/robot_mc/build/export/config/
```

原始路徑：`src/robot_mc/build/export/config/`

## 檔案

| 檔案 | 機型 | 內容 |
|---|---|---|
| `zg_wheels-user-parameters.yaml` | **ZGWS = zsm-1w = D1 Max** | 增益、關節限位、站/趴/RL 預設姿、符號與 offset 慣例、MPC/步態排程參數 |
| `zg_wheels-motion_config.yaml` | 同上 | 動作腳本（站立、握手、扭腰），**腿序寫死為 FR / FL / RR / RL** |
| `xg_wheel-user-parameters.yaml` | XGW = zsl-1w = **D1 EDU 輪足**（task6 那台） | 同結構，**放在這裡是為了交叉驗證**，見下 |
| `robot-defaults.yaml` | 不分機型 | `controller_dt = 0.002`（500 Hz）等全域參數 |

## ★ 為什麼這些數字可以信：一個獨立的交叉驗證

`xg_wheel-user-parameters.yaml`（D1 EDU 輪足）寫著：

```
FSM_RL_ABAD_Kp : 20.0
FSM_RL_HIP_Kp  : 20.0
FSM_RL_KNEE_Kp : 20.0
FSM_RL_Kd      : 0.7
FSM_RL_Wheel_Kp: 5.0
FSM_RL_Wheel_Kd: 0.1
FSM_passive_Wheel_Kd : 0.0
```

task6 在 **2026-08-11 從 D1 EDU 實機的 `/spline_shm` 指令區量到的原廠站立增益**是
**腿 kp = 20 / kd = 0.7、輪 kp ≈ 0 / kd = 0.1** —— 與上面**逐項吻合**。

→ 這代表 MATRiX 發布包裡的 `*-user-parameters.yaml` **就是實機在跑的那組參數**，
不是模擬器自己捏的示範值。因此 `zg_wheels`（D1 Max）那份可以用同等級的信心對待。

⚠️ 但仍是**間接證據**：D1 Max 本身還沒上實機量過。標示為「高信心推定」，不是「已實測」。

## ⚠️ 三個判讀上的注意事項

1. **座標系不明**。`abad_offset` / `hip_offset` / `knee_offset` 的數值恰好等於 URDF 的關節限位
   （D1 Max：abad 0.523、hip 2.443、knee 2.803），看起來是「馬達編碼器零點落在關節限位」的慣例，
   但 `控制器角度 = side_sign × 馬達角度 + offset` 這個公式**我沒有原始碼可以證實**，
   是從數值形狀推的。**要用在 sim2real 之前必須上實機驗一次。**
2. **`JPos_limit` 比 URDF 寬**。設定檔寫 abad ±0.873 (±50°)，URDF 是 −0.697 ~ +0.523 (−39.9°~30°)，
   規格書也是 −39.9°~30°。設定檔那組應該是「保護用的外圍限值」，不是機構真限位。
   **做 IK 限位檢查請用 URDF / 規格書那組，不要用這裡的。**
3. **`feet_z_offset` 不等於輪半徑**。D1 Max 寫 0.095、規格書輪半徑 0.09；
   D1 EDU 寫 0.08、實際輪半徑 0.071。兩台都比半徑大一點，是運控自己的足端 z 偏移慣例。

## 沒有收進來的東西

- `onnx_model_crypto/zg_wheels/` 底下有 20 多個策略（`policy_gait_walk`、`policy_stair`、
  `policy_slim`、`policy_dsb`、`policy_snow`、`policy_skwalk`、`policy_climb`、`policy_zgws_rpy`、
  `policy_zgws_jump`、`policy_zgws_backflip`、`policy_handstand`、`policy_tuoluo` …），
  名稱與 D1 Max SDK 的高層動作 API 一一對應。**但它們是加密的**
  （同目錄有 `libfilecrypto_shared.so` 與 `file_crypto_cli`），沒有金鑰讀不出來，所以沒收。
- MJCF 模型不在 `assets` 包裡，另尋（見 `task7/README.md`）。
