# 最終效果（2026-07-16）：terrain (RL-legheight) + odom

最新模型 `weights/cpg_rl_terrain2_1_params.pkl`（16 維、可學抬腳 + 抬腳強化）+ odom 外圈閉環，在 07-15 的爬坡/凹凸課程上的最終表現。單支影片、無對照組。

腳本：`task4/analysis/terrain_compare_final.py`
```bash
MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_final.py --exp both --solo
```

## 影片

- **`exp1_slope_final.mp4`（爬坡）**：平地→上/下坡 5/10/15° 三座山丘。
- **`exp2_rough_final.mp4`（凹凸）**：平地→崎嶇 3/5/8cm 三段。

影片左上顯示標題與地形段、左下顯示 cross-track 誤差與即時抬腳 `gc`。

## 量測

| 實驗 | 前進 | 最大側偏 | 平均側偏 | 抬腳 gc | 跌倒 |
|---|---|---|---|---|---|
| 爬坡 slope | 18.4 m | 0.07 m | 0.023 m | 0.092 | 否 |
| 凹凸 rough | 18.5 m | 0.07 m | 0.018 m | 0.092 | 否 |

→ 全向底層 + odom 外圈：爬坡/凹凸皆穩定貼線（側偏 ~7cm）、全程不跌、抬腳 ~9cm。
