# 地形實驗（2026-07-15）：terrain 版 CPG-RL 單獨測試

> 腳本：`task4/analysis/terrain_compare.py`　·　權重：`weights/cpg_rl_terrain_params.pkl`
> 模型：**terrain 版原生直走** `cmd=[vx=0.8, 0, 0]`（不套 line_control）。同 scene_mjx 課程、apply_pd 位置伺服。

## 產出

| 檔案 | 內容 |
|---|---|
| `exp1_slope_terrain.mp4` | 平地→斜坡（上/下坡 5/10/15°）。右側視角。大字顯示坡度。綠=上坡/橘=下坡/灰=平地 |
| `exp1_slope_chart.png` | 左：地形剖面+質心高度(爬過3座山丘)；右：前進距離 & 橫向漂移 vs 時間 |
| `exp2_rough_terrain.mp4` | 平地→崎嶇（3/5/8cm）。後上方45°俯視。大字顯示幅度 |
| `exp2_rough_chart.png` | 左：前進距離 vs 時間；右：橫向漂移 vs 位置(標示3/5/8cm區) |

## 結果

| 實驗 | 完成度 | 跌倒 | 橫向漂移 |
|---|---|---|---|
| slope（5/10/15° 山丘×3）| ✅ 全部爬過（質心升到 0.4/0.55/0.7m）| 否 | 2.71m |
| rough（3/5/8cm ×3）| ✅ 全部通過 | 否 | 2.91m |

## 重要限制：航向漂移（誠實記錄）

terrain 模型是**盲走**（obs 只有本體感覺、**無絕對位置/航向**）。防走歪訓練只能壓小「瞬時偏航率」，但殘留微小偏差會**隨距離累積成緩慢航向漂移**（系統性偏一側），policy 無法感測絕對 y 去修正。

- 因此本實驗把跑道**加寬到 ±6m**、用 **vx=0.8**（前進/漂移比較好），避免漂移導致掉出邊緣。
- **為何不用 odom 修正**：實測對 terrain 模型套 line_control（或只給 wz 航向修正）會讓它**停步/倒退**——它只在 `cmd=[vx,0,0]` 訓練，對任何 vy/wz 指令都是 OOD 而崩潰。故 terrain 只能原生直走。
- 真機要走直線需 odom/IMU 航向回授；要讓 terrain 模型能吃 odom，需**重訓成全向指令版**（cmd 含 vy/wz）。

## 對照原始版(paper)？

本次依指示**只做 terrain 單獨**，未跑 paper 對照（因兩隻無法公平共用 odom：paper 需 odom、terrain 被 odom 弄壞）。
