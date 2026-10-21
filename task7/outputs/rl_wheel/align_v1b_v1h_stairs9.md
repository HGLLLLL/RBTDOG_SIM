# 對齊：rl_wheel_v1b_np.npz 在 WheelEnv（MJX，平地）vs CPU rl_only_infer（real）  —  tag v1b_v1h_stairs9

MJX 46 s（3 環境 vmap）、CPU 0 s；指令 8.0 s（先 1.0 s 零指令）；統計取指令後 1 s 起。後輪 gap ＝ 後輪實體間隙近似 ＝ 輪心距 − 0.034（官方網格內側面 +0.017 ×2，V 文件口徑；傾斜時高估 ~7 mm）（mm，取最小；env 與 cpu `gap_rr` 同式）。追蹤差 ＝ |env − cpu| / |指令|，只算有下指令的軸。
cpu 欄＝rl_only_infer 預設（v3f 圓柱輪、平面地板）；cpu′＝WheelEnv 場景（膠囊輪／機身／腿、hfield）：同一支 CPU 工具改載，env vs cpu′ 只剩 MJX 與 CPU MuJoCo 的差，cpu′ vs cpu 是場景（碰撞幾何）的差。

| 指令 | 摔 env/cpu/cpu′ | vx env/cpu/cpu′ | vy env/cpu/cpu′ | 偏航 °/s env/cpu/cpu′ | 抬腳 mm env | 抬腳 mm cpu | 抬腳 mm cpu′ | 後輪 gap mm env/cpu/cpu′ | \|a\| max env/cpu | 追蹤差 env↔cpu | 追蹤差 env↔cpu′ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vx +0.0 vy +0.5 wz +0.0 | 0/0/0 | +0.02/+nan/+nan | +0.52/+nan/+nan | +0/+nan/+nan | [149.3, 216.2, 49.1, 108.1] | [] | [] | 196/nan/nan | 3.66/nan | nan 個百分點 | nan 個百分點 |
| vx +0.0 vy -0.5 wz +0.0 | 0/0/0 | -0.02/+nan/+nan | -0.43/+nan/+nan | -0/+nan/+nan | [17.2, 115.2, 13.6, 78.5] | [] | [] | 250/nan/nan | 3.83/nan | nan 個百分點 | nan 個百分點 |
| vx +0.0 vy +0.3 wz +0.0 | 0/0/0 | -0.01/+nan/+nan | +0.30/+nan/+nan | -1/+nan/+nan | [84.3, 24.4, 54.4, 11.2] | [] | [] | 173/nan/nan | 3.36/nan | nan 個百分點 | nan 個百分點 |

**最大追蹤差 env↔cpu 0.0 個百分點、env↔cpu′ 0.0 個百分點（百分比僅供參考；關卡（膠囊輪子，2026-10-13 定）：vx 列 ≤ 0.03 m/s、wz 列 ≤ 5 °/s（膠囊對圓柱的原地轉偏保守，實機 wz 0.5 ≈ 20 °/s 介於兩者）、0 摔；絕對差 env↔cpu，vy 0.3 是他的死區不計，spec §11.8）；摔 env 0／cpu 0／cpu′ 0（門檻 0）。**
