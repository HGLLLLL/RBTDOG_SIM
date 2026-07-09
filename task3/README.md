# 任務3：IMU + compass 航向校正（走正方形對比）

> 目標：機器狗走正方形（前進 10m → 右轉 90°，共 5 圈）。
> 版本1（無 IMU）開環航位推算 → 累積誤差；版本2（IMU 羅盤）閉環校正 → 精準。

## 成果

| | 終點誤差（終點回起點） | 說明 |
|---|---|---|
| 版本1 無IMU（開環） | **7.87 m** | 每圈越漂越歪，軌跡呈亂麻狀 |
| 版本2 IMU羅盤（閉環） | **0.04 m** | 每圈精準回原點，5 圈疊成乾淨方形 |
| **改善** | **約 206 倍** | |

> 註：以上為 2026-07-02 換用新穩定步態（`walk_line.GAIT`：stride 0.32、2.6Hz、≈0.89 m/s）並改用 **Xsens MTi-680G 融合航向模擬**（見技術做法第 4 點）重跑的結果，並重新校準標稱速度/角速度。

- 軌跡/航向對比圖：`square_result.png`
- 俯視縮時影片（並排＋軌跡拖尾）：`square_video.mp4`
- 步態示意 GIF：`walk_test.gif`

## 為什麼用自寫 CPG 而非 MJPC / RL

- **MJPC** 內部用真實狀態做全狀態回授、航向永遠準 → 開環版不會漂，凸顯不出感測器價值。
- **RL 策略** 本身就需要 IMU（陀螺+重力）才能走，做不出「零感測器」的第一版。
- **CPG 開環**：第一版真的零感測器 → 最貼合「有/無 IMU」對比。

## 技術做法

1. **走路**：Go2 是力矩致動器 → 軟體 PD（`力矩 = kp·(角度目標−角度) − kd·角速度`）。
   CPG 產生 trot 步態：對角腿同相，站立相腳貼地後推、擺動相抬起前移。
   腳掌橢圓軌跡用**站姿數值雅可比**反解成 thigh/calf 角度（`go2_gait.py`）。
2. **轉彎**：左右腿差動步幅（`turn_gain`）。校準（新步態）：`turn=+1` → 右轉 58.7°/s。
3. **9 軸 IMU**：`accelerometer + gyro + magnetometer` 掛在機身 `imu` site（`go2_model.py`，用 MjSpec 加）。
4. **航向來源＝模擬 Xsens MTi-680G 的「融合航向」**：真實 Go2 上此模組內部已融合陀螺/加速度/磁力計(+GNSS)才輸出航向，故不直接用裸磁力計，而是在真實航向上疊「裝置級誤差」＝慢速零偏（Gauss-Markov, τ=20s）＋白噪，**合成 RMS≈0.5°**（符合規格書 heading 0.5° RMS）。實作在 `go2_gait.compass_yaw()`（可調 `imu_heading_rms_deg / imu_heading_tau / imu_seed`）；裸磁力計解算保留為 `mag_yaw()`。
   > 目前只做「方向」校正、未做位置控制；MTi-680G 的 RTK 位置（1cm）與陀螺/加速度原始噪聲暫未納入。
5. **版本差異**：
   - V1：直行走「10m/標稱速度」的固定時間、轉「90°/標稱角速度」的固定時間，全程不看感測器 → 自然偏擺＋轉彎誤差累積。
   - V2：直行時 `turn = Kp·(目標航向 − 融合航向)` 保持直線；轉彎閉環轉到航向=目標(±3°) → 航向不累積誤差。

## Demo B：走直線 40m + 外部干擾（`walk_line.py`）

> 展示「校正動作」：走直線 40m，途中施加 5 次側推+yaw 干擾，對比有無羅盤。

| | 結果 | 說明 |
|---|---|---|
| V1 無IMU（開環） | 終點漂到 **y=+14.0m** | 被推歪後回不來，越走越偏離目標方向 |
| V2 IMU羅盤（保持朝向） | 終點 **y=+0.24m** | 每次被推只彈一下立刻把航向 snap 回箭頭方向 |

- 步態調校：改用**長步幅、低步頻且求穩**（`freq=2.6Hz、stride=0.32m、duty=0.6、lift=0.12`，前版 3.5Hz/0.24m 走碎步），前進速度 **≈0.85 m/s**，步幅 +33%，側傾 roll 較長步幅版再降 46%、垂直彈跳降 24%，走起來平穩不搖晃、開環本身就走得直。步態姿態預覽見 `gait_preview.py` → `gait_preview.mp4`。
- 目標視覺 = **一支永遠指向前方(+x)的紅箭頭**（目標航向），浮在每隻狗上方。
- 干擾：交替左右的側推力(±35N)+yaw力矩(±10Nm)，兩隻狗同時施加。**施力當下畫面會出現橘色大箭頭**指向受力方向（長度依力量大小），並在右上角標「PUSH 35N」與方向箭頭，讓觀者一眼看出目前有外力。
- 影片 `line_video.mp4`：**real-time 播放**（≈51s），並排；相機在狗正後方 45° 俯視、**方位角固定朝 +x**（狗歪時身體在畫面內轉向、與箭頭錯開）、隨橫偏輕微縮放。
- 軌跡圖 `line_result.png`：V2 緊貼 y=0 目標線（每個小凸起=一次校正）；V1 漂到 14.0m。
- 註：羅盤只鎖「方向」不鎖「位置」→ V2 校正航向、續沿平行線直走，不會橫向壓回中心線（純 IMU 的真實能力）。

## 檔案

| 檔案 | 用途 |
|------|------|
| `go2_model.py` | 用 MjSpec 建立含 9 軸 IMU 的 Go2 模型 |
| `go2_gait.py` | CPG trot 控制器（PD + 雅可比 IK + 轉彎 + 感測 + MTi-680G 融合航向模擬 `compass_yaw()`） |
| `square_mission.py` | Demo A：正方形任務（兩版本 + 對比圖），`python square_mission.py [圈數]` |
| `render_mission.py` | Demo A 俯視縮時影片（並排 + 軌跡拖尾），`python render_mission.py [圈數]` |
| `walk_line.py` | Demo B：走直線40m+干擾（real-time 影片 + 軌跡圖），`python walk_line.py` |
| `walk_proto.py` | 早期走路調參原型 |

## 執行

```bash
mamba activate rbtdog
cd /home/huang/rbtdog_sim/task3
python square_mission.py 5     # 跑 5 圈，輸出 square_result.png + 誤差數據
python render_mission.py 5     # 輸出 square_video.mp4（需 MUJOCO_GL=egl 環境變數渲染）
```

## 可延伸

- 調整 MTi-680G 航向誤差（`imu_heading_rms_deg / imu_heading_tau`）測試校正強健性；或再補上陀螺/加速度計的原始噪聲。
- 納入 MTi-680G 的 RTK 位置（1cm）做「位置外環 + 航向內環」的雙層控制，讓位置也準（目前僅方向準）。
- 改用 gyro 積分（會漂）＋磁力計做自寫融合，對照 MTi-680G 的內建融合輸出。
