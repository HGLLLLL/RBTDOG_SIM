# 最終對照實驗（2026-07-16）：並排合成影片

腳本：`task4/analysis/terrain_compare_final.py`（課程沿用 07-15 的 slope/rough）。左右各自獨立 sim、鏡頭各跟各的狗，逐幀併排。

```bash
MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_final.py --exp both
```

## 影片

- **`exp1_slope_compare.mp4`（爬坡）**：左 `terrain (開環直走)` = v1（12維、固定抬腳、直走專用，`cpg_rl_terrain_params.pkl`）‖ 右 `terrain (RL-legheight)+odom` = v2.1（16維、可學抬腳+抬腳強化，`cpg_rl_terrain2_1_params.pkl`）+ odom 閉環。
  - 註：v1 吃不下 odom 的 wz/vy 修正（會倒退），故左側用它本來會的開環直走。
- **`exp2_rough_compare.mp4`（凹凸）**：左 `terrain+odom` = v2.0（16維，`cpg_rl_terrain2_params.pkl`）‖ 右 `terrain (RL-legheight)+odom` = v2.1（16維+抬腳強化）。兩者皆 odom 閉環。

影片每側左上顯示標題與地形段、左下顯示 cross-track（odom）或側偏（開環）與即時抬腳 `gc`。

## 量測

### 實驗1 爬坡（v1 開環直走 ‖ v2.1 +odom）
| 指標 | 左 v1 開環直走 | 右 v2.1 +odom |
|---|---|---|
| 前進 | 17.7 m | 18.4 m |
| 最大側偏 | **2.44 m** | **0.07 m** |
| 平均側偏 | 1.08 m | 0.023 m |
| 抬腳 gc | 0.080（固定） | 0.092 |
| 跌倒 | 否 | 否 |

→ 兩者都爬完斜坡不跌；**差別在直線精度**：v1 開環盲走會蛇行漂 2.44m，v2.1 靠 odom 閉環牢牢貼線（0.07m）。新模型抬腳也更高。

### 實驗2 凹凸（v2.0 +odom ‖ v2.1 +odom）
| 指標 | 左 v2.0 +odom | 右 v2.1 +odom |
|---|---|---|
| 前進 | 18.7 m | 18.5 m |
| 最大側偏 | 0.05 m | 0.07 m |
| 平均側偏 | 0.011 m | 0.018 m |
| 抬腳 gc | 0.086 | **0.092** |
| 跌倒 | 否 | 否 |

→ 兩者都靠 odom 貼線、都不跌；**差別在抬腳**：v2.1 抬得略高（0.092 vs 0.086）。

## 誠實提醒（實驗2 的抬腳差異在此課程較小）

v2.0 在這條 3/5/8cm 課程上的**反應式抬腳已到 ~0.086**（比它在訓練用漸變地形上的 ~0.071 高，因為這裡是離散凸起、踩到會反應），所以和 v2.1（0.092）的**數字差距比「訓練地形上 0.071→0.092」小**。抬腳強化的效益在**更凹凸/更密的地形**上會更明顯。實際「走得順不順、卡不卡」請直接看影片的擺動腳高度與流暢度。
