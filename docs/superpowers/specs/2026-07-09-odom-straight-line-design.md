# odom 回授的絕對直線行走設計（CPG-RL / task4）

日期：2026-07-09
目標檔案：`task4/inference/local_infer_paper.py`（論文標準 12 維動作版）
相依：`task3/go2_gait.py`、`task3/go2_imu_scene.xml`（共用）

## 1. 目標與動機

現況：`local_infer_paper.py` 的走路底層是 CPG + RL policy，但方向回授只有
`compass_yaw()`（模擬 Xsens MTi-680G 融合航向，含慢偏移 + 白噪）。控制律只做
一件事：`yaw_rate = -GAIN·(compass_yaw - TARGET_YAW)`，也就是**只鎖航向**。

問題：
- 只鎖航向 → 狗頭方向對了，但整條路徑仍會**橫向平移飄移**（cross-track drift）。
- `TARGET_YAW` 寫死 0（+x 方向），**無法朝任意方向**走。

要達到的效果：
1. 把外掛 sensor 從 compass 換成**完美 odom**（位置 + 航向，預設零偏移、零噪聲）。
2. 走路底層維持 CPG + RL 不變。
3. 用 odom 做回授，走**絕對直線**——不只航向對，橫向位置也不能飄。
4. 目標方向任意：腳本先「右轉 45°」再「直走」，狗沿著**當下狗頭方向那條射線**走，
   橫向偏移與航向誤差都盡量壓到最小。

## 2. 可行性關鍵（已驗證）

訓練 notebook（`cpg_rl_paper_colab.ipynb`）的指令空間為 `cmd = [vx, vy, wz]`：
- `vx ∈ [0, 1]`
- **`vy ∈ [-0.3, 0.3]`**（body-frame 橫向 / 螃蟹步，**有訓練**，reward 追蹤 `blin[1]`）
- `wz ∈ [-1, 1]`（yaw rate，reward 追蹤 `gyro[2]`）

結論：可用 `wz` 鎖航向、`vy` 修橫向偏移，**兩通道解耦**，同時把側飄與航向誤差都趨近 0，
而不必靠扭頭（扭頭會犧牲航向誤差）。現有程式把 `vy` 固定為 0，白白浪費此通道。

## 3. 設計決策（brainstorm 定案）

| 項目 | 決定 |
|------|------|
| 互動形式 | 腳本化情境（turn → latch → straight），非即時遙控 |
| 目標檔案 | `local_infer_paper.py` |
| odom 實作 | XML 加 `framepos` 感測器（取代 `magnetometer`），保留 `framequat`；compass 相關程式碼保留當備份 |
| 控制律 | 方案 A：解耦——`wz` 管航向、`vy` 管 cross-track |
| 轉角約定 | `--turn_deg` 帶號相對轉角，**右轉 45° = `-45`**（世界系正 yaw = CCW = 左轉） |

## 4. 元件設計

### 4.1 odom 感測器

**`task3/go2_imu_scene.xml`**：sensor 區塊把
```xml
<magnetometer name="imu_mag" site="imu"/>
```
換成
```xml
<framepos name="odom_pos" objtype="site" objname="imu"/>
```
保留 `accelerometer` / `gyro` / `framequat`。
（副作用：未使用的備份 `mag_yaw()` 會失效，無害；`compass_yaw()` 走 `true_yaw()`
不受影響，保留當備份。）

**`task3/go2_gait.py`**：
- 建構子新增 `odom_xy_bias=(0.0, 0.0)`、`odom_yaw_bias=0.0`，**預設全 0 = 完美 odom**
  （日後可注入偏移做實驗）。
- 新增方法：
  ```python
  def odom(self):
      """完美里程計：回傳世界系 (x, y, yaw)，預設無偏移。"""
      x, y, _ = self.sensor("odom_pos")
      w, xx, yy, zz = self.sensor("imu_quat")
      yaw = np.arctan2(2*(w*zz + xx*yy), 1 - 2*(yy*yy + zz*zz))
      bx, by = self._odom_xy_bias
      return x + bx, y + by, wrap(yaw + self._odom_yaw_bias)
  ```
- `sensor()` 已可處理 dim=3（`odom_pos` 為 3 維），不需改。

### 4.2 目標線與控制律（方案 A）

目標線由 latch 時刻的 odom 定義：
- 原點 `p0 = (x0, y0)`（latch 當下 odom 位置）
- 目標航向 `psi_target`（latch 當下 odom 航向 = 真正的狗頭方向）
- 方向單位向量 `d = (cos psi_target, sin psi_target)`
- 左法向單位向量 `n = (-sin psi_target, cos psi_target)`

每個 control step（直走階段）：
```python
p = np.array(odom_xy)
e_ct  = n @ (p - p0)                       # cross-track，帶號橫向距離（+ = 偏左）
e_yaw = wrap(yaw_now - psi_target)         # 航向誤差
wz = float(np.clip(-K_YAW * e_yaw, -1.0, 1.0))     # 只鎖航向
vy = float(np.clip(-K_CT  * e_ct , -0.3, 0.3))     # 螃蟹橫移滑回線上（不動頭）
cmd = np.array([vx, vy, wz], np.float32)
```
- `K_YAW` 沿用現有 `HEADING_GAIN = 3.0`。
- `K_CT` 新增，起始 ≈ 1.5，實作時實測微調。
- `vy` 符號（body +y 是否為左）以小測試驗證後定案，不臆測。
- `--no_lateral` 旗標令 `vy ≡ 0`，一鍵重現「舊的只鎖航向」行為做 A/B 對照。

### 4.3 腳本化 mission（turn → latch → straight）

新增 CLI 參數：`--turn_deg`（帶號，預設 `-45`）、`--k_ct`（預設 1.5）、`--no_lateral`；
沿用既有 `--params --secs --vx --video --push --w_coup`。

- **Phase 0 warmup**：沿用現有 0.5 s HOME 姿勢穩定。
- **Phase 1 轉向**：`psi_goal = start_yaw + radians(turn_deg)`；
  跑 `wz = clip(-K_YAW·wrap(yaw - psi_goal), -1, 1)`、`vx = 0`、`vy = 0`，
  直到 `|wrap(yaw - psi_goal)| < radians(2)` 連續穩住（設 `turn_timeout` 秒上限保護）。
- **Latch**：`p0 = odom xy`、`psi_target = odom yaw`（實際狗頭方向）。
- **Phase 2 直走**：套用 4.2 控制律跑 `--secs` 秒。

軌跡全程記錄，並記下 latch 索引以區分兩階段。

### 4.4 輸出 / 量測

Result 印出（直走段統計）：
- 沿線前進距離（`d · (p_end - p0)`）
- **max |cross-track|**、末端 cross-track
- **航向誤差 RMS**（deg）
- 是否跌倒、FL 抬腳量（沿用現有）

存檔：
- 軌跡圖 PNG 到 `task4/outputs/`（路徑 + latch 的目標線疊圖，標示 turn/straight 兩段）。
- `--video` 沿用現有 renderer 輸出 mp4。

## 5. 測試計畫

1. **管線測試**：`--dummy --turn_deg -45 --secs 6`，驗證 odom 讀值正常、兩階段切換與
   控制律不崩、有產出結果。
2. **`vy` 符號驗證**：獨立小測試，固定命令 `+vy` 應使狗往 body 左側位移；據此定 `vy` 符號。
3. **主情境（帶權重）**：`--params ... --turn_deg -45 --secs 20`，
   驗證末端 |cross-track| 與航向 RMS 皆小。
4. **A/B 對照**：加 `--no_lateral` 重跑，量化「有 vy 修正 vs 只鎖航向」的側飄差異，
   佐證改善。

## 6. 影響範圍與相容性

- 改動檔案：`task3/go2_imu_scene.xml`、`task3/go2_gait.py`、`task4/inference/local_infer_paper.py`。
- `compass_yaw()` / `true_yaw()` 保留可用；`mag_yaw()` 因移除 magnetometer 而失效（未使用，無害）。
- 其他共用 `go2_gait` 的檔案（`walk_line.py`、`square_mission.py`、`local_infer.py`、
  `compare.py`）不改；它們不讀 `imu_mag`（僅 `mag_yaw()` 會失效，皆未使用），故不受影響。

## 7. 待實作時確認（非阻塞）

- `vy` 正負號（實測定案）。
- `K_CT` 增益數值（實測微調）。
- Phase 1 是否需小 `vx` 以穩定轉向（預設原地轉 `vx=0`，若不穩再加）。
