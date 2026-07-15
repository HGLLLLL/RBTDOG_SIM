# 真機外圈遙控直線控制器（RC + odom line controller）設計

日期：2026-07-15
狀態：已與使用者逐項確認需求，方案 A 定案

## 1. 目標

把 task4 已驗證的 odom 走直線校正算法（`line_control`，方案 A 解耦：wz 鎖航向、vy 滑回線上）包成**真機器狗的外圈控制器**。不碰底層走路，輸入「遙控器桿量 + odom」，輸出「速度指令 (vx, vy, wz)」。

遙控邏輯（使用者口述）：推前進的瞬間把當下航向鎖成目標方向、當下位置鎖成起點，定出一條參考直線；推轉向時桿量直接透傳當轉彎指令；轉向停止後用當下位置和航向重新鎖一條新直線，依此類推。

## 2. 已確認的需求決策

| 項目 | 決定 |
|---|---|
| 輸出介面 | Unitree Go2 高階 `SportClient.Move(vx, vy, vyaw)`，vy 可用 → `line_control` 照搬 |
| odom 來源 | 外接定位（SLAM/RTK 類；漂移小，但要處理延遲/掉訊/重定位跳變） |
| 遙控來源 | 自己的手把/電腦（非宇樹原廠手柄），桿量由自己的程式讀 |
| 同時推桿 | 轉向優先：轉向桿離中就暫停直線校正，vx 照給、走弧線 |
| latch 時機 | 桿回中後等航向穩定（角速度 < 門檻持續 ~0.3s）才 latch，避免鎖到還在滑的航向 |
| 前進桿 | 比例控制 vx = 桿量 × VMAX；桿量變化不重 latch，只有「中立→推下」瞬間 latch 新線 |
| fwd 回中 | 線作廢、回 MANUAL（停）；再推 fwd 重新 latch 新線 |
| 倒退 | fwd 負值 = 沿同一條線倒著走，照樣校正（航向仍鎖線方向） |
| 橫移桿 | 視為手動接管（同轉向）：透傳 vy、暫停校正、回中穩定後重 latch |
| odom 掉訊 | 退化成純手動直通（不停下），恢復且穩定後重 latch |
| 交付範圍 | 平台無關 Python 模組 + MuJoCo 模擬驗證；真機接線層（手把讀取/SDK Move）只留介面之後再寫 |

## 3. 模組邊界與資料流

```
[手把讀取層]────sticks────┐
                          ├──> RCLineController.update(sticks, odom, now) ──cmd──> [輸出層]
[外接定位層]────odom──────┘        （純 Python + numpy，無 MuJoCo/SDK 依賴）
```

- **sticks** = `(fwd, lat, turn)`，各 [-1, 1]。死區含遲滯在模組內處理（離中 0.08 / 回中 0.04），因為「離中/回中」直接驅動狀態機。
- **odom** = `(x, y, yaw, stamp)`，世界系、弧度、含時間戳。
- **cmd** = `(vx, vy, wz)`；上層自行接 `SportClient.Move()`（真機）或 RL 策略（模擬）。
- `line_control` / `line_frame` / `wrap` 從 `task4/inference/local_infer_paper.py:97-117` **原封複製**進新模組（註明來源）——不 import，避免真機拖進 MuJoCo/jax 依賴。
- 檔案位置：`task5/rc_line/`
  - `rc_line_controller.py` — 模組本體（僅依賴 numpy）
  - `tests/test_rc_line_controller.py` — 單元測試（獨立 assert 腳本，專案慣例）
  - `sim_demo.py` — MuJoCo 驗證（吃 task4 權重，重用 `odom_missions.Runner`）

## 4. 狀態機

三態，每控制週期（50Hz，同模擬 CTRL_DT=0.02）跑一次：

```
MANUAL ──(fwd 離中 且 turn/lat 回中 且 odom 新鮮)──> SETTLING ──(航向穩定≥0.3s → latch)──> TRACKING
   ^                                                                                        │
   └──────(任一成立：|turn|離中、|lat|離中、odom 逾時/NaN、fwd 回中、odom 跳變*)──────────────┘
                                                        （*跳變退回 SETTLING 而非 MANUAL）
```

- **MANUAL**：`cmd = (fwd×VMAX, lat×VYMAX, turn×WMAX)` 直通。全桿回中時輸出全零（站立）、留在 MANUAL。
- **SETTLING**：fwd 離中、接管桿已回中、odom 新鮮，但航向未穩。輸出 `(fwd×VMAX, 0, 0)`——前進照走，不透傳殘餘桿噪、不校正。
- **航向穩定判斷**：模組持續用 odom yaw 差分（一階低通）估角速度，維護「已穩定多久」計時器。從靜止直接推前進時計時器早已滿 → **推桿瞬間立即 latch**；剛轉完彎才需等 ~0.3s。
- **latch**：`p0 = (x, y)`、`psi = yaw`（當下 odom），進 TRACKING。
- **TRACKING**：`cmd = line_control(odom_xy, odom_yaw, p0, psi, vx=fwd×VMAX, K_YAW, K_CT)`。fwd 為負 = 沿線倒退（wz 仍鎖 psi、vy 校正符號與 vx 無關，控制律不用改）。

## 5. 控制律與參數

- `line_control` 原封複製（模擬已驗證：40m 末端 |y|=0.004m）。
- 參數集中一個 config dataclass，預設 = 模擬驗證值：

| 參數 | 預設 | 說明 |
|---|---|---|
| `VMAX` | 0.6 m/s | fwd 桿滿量速度 |
| `VYMAX` | 0.3 m/s | 橫移桿滿量速度（MANUAL 透傳用） |
| `WMAX` | 1.0 rad/s | 轉向桿滿量角速度 |
| `K_YAW` | 3.0 | 航向 P 增益 |
| `K_CT` | 1.5 | 橫向 P 增益 |
| `VY_LIM` | 0.3 | 校正 vy 限幅 |
| `WZ_LIM` | 1.0 | 校正 wz 限幅 |
| `DEAD_ON / DEAD_OFF` | 0.08 / 0.04 | 桿死區（遲滯） |
| `YAW_RATE_STABLE` | 0.1 rad/s | 航向穩定門檻 |
| `SETTLE_S` | 0.3 s | 穩定持續時間 |
| `STALE_S` | 0.5 s | odom 逾時 |
| `JUMP_POS_M / JUMP_YAW` | 0.5 m / 30° | 重定位跳變門檻 |
| `SLEW` | 開，可關 | vy/wz 斜率限制，平滑狀態切換跳變 |

- 角速度估計：`wrap(yaw - yaw_prev) / dt`（dt 用 odom stamp 差）過一階低通（截止 ~5Hz）。

## 6. 錯誤處理

- **odom 逾時 / NaN**：強制 MANUAL 直通（不停下，操作者保有控制權），曝露 `degraded` 旗標給上層顯示；恢復後照正常條件重 latch。
- **odom 跳變**（外接定位重定位）：相鄰兩筆位置差 > 0.5m 或 yaw 差 > 30° → 作廢目前的線、退回 SETTLING 重鎖，避免 e_ct 瞬間暴增讓狗猛拉。
- 所有輸出 clip 到限幅後才出模組。
- 尚未收到第一筆 odom 前一律 MANUAL。

## 7. 驗證與測試

**單元測試**（純 Python + numpy，餵合成 odom/桿量序列）：
1. 狀態轉移表逐條驗證（含同時推桿、fwd 回中作廢）
2. latch 時機：靜止推 fwd 立即 latch；轉彎後放桿等穩定才 latch
3. 掉訊→直通→恢復重 latch；跳變→SETTLING 重鎖
4. 倒退沿線：負 vx 下 e_ct 校正方向正確
5. 死區遲滯不顫振；輸出限幅

**MuJoCo 整合驗證**（`sim_demo.py`，task4 RL 權重 + 完美 odom）腳本化桿量時間軸：
推前進 8s → 前進中轉彎 2s → 放轉向續走 8s（重 latch 新線）→ 注入掉訊 2s → 恢復 → 倒退 5s。
輸出影片（地板畫每條 latch 線）+ 軌跡圖。

**成功標準**：每次 latch 後側偏收斂 < 0.05m（完美 odom）；狀態切換全程不跌倒；掉訊期間桿量直通行為正確。

## 8. 非目標（本次不做）

- 真機接線碼（pygame 手把讀取、unitree_sdk2 呼叫）——只留 `sticks`/`odom`/`cmd` 介面。
- odom 漂移補償/多感測融合（外接定位視為夠準）。
- 弧線／曲線路徑追蹤（只有直線段 + 手動轉彎）。
