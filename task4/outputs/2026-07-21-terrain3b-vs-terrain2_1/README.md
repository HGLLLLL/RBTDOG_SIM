# terrain3b vs terrain2_1 並排對照（2026-07-21）

同今早 `2026-07-21-terrain3-vs-terrain2_1` 的爬坡/凹凸課程 + odom 閉環，把 terrain3 換成**修正版 terrain3b**（slip 懲罰 0.5→0.1 + 爬坡獎勵重訓，其餘 terrain3 設定不變）。

- **左 terrain3b（新）**：修正爬坡退步後的模型（配 cpg3，GC_MAX=0.25）。
- **右 terrain2_1（舊）**：抬腳強化模型（配 cpg2，GC_MAX=0.15）。

腳本：`task4/analysis/terrain_compare_terrain3b.py`（重用 `terrain_compare_terrain3.render_run`）
```bash
MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_terrain3b.py --exp both
```

## 影片
- **`exp1_slope_compare.mp4`（爬坡 5/10/15°）**
- **`exp2_rough_compare.mp4`（凹凸 3/5/8cm）**

## 量測

| 實驗 | 模型 | 前進 | 最大側偏 | 平均側偏 | 抬腳 gc | 跌倒 |
|---|---|---|---|---|---|---|
| 爬坡 slope | **terrain3b（新）** | **21.1 m** | 0.14 m | 0.023 m | **0.127** | 否 |
| | terrain2_1（舊） | 18.4 m | 0.07 m | 0.023 m | 0.092 | 否 |
| 凹凸 rough | **terrain3b（新）** | 18.6 m | 0.07 m | 0.016 m | **0.123** | 否 |
| | terrain2_1（舊） | 18.5 m | 0.07 m | 0.018 m | 0.092 | 否 |

## 結論：修正成功，兩全其美

對照今早失敗的 terrain3（爬坡只走 13.3m、卡在 15° 磨蹭），terrain3b：

- ✅ **爬坡力全面回復並超越**：45s 走 **21.1m**（terrain3 只有 13.3m、terrain2_1 18.4m）。診斷（`diag_slope_climb.py`，55s）顯示 terrain3b 走到 x=23.0、在 15° 段只花 10.2s（terrain3 花 27s）——不再卡住。
- ✅ **同時保住高抬腳**：gc ~0.13（≈13cm）vs terrain2_1 ~0.09，跨障礙能力更強。這正是 terrain3 想要、卻被 slip 懲罰搞砸的目標。
- ✅ 兩實驗皆全程不跌。
- ⚠️ 唯一小代價：爬坡最大側偏 0.14m（terrain2_1 0.07m）——走得更快/抬更高，貼線瞬間精度略降，但**平均側偏一樣 0.023m**、不跌，可接受。

## 根因與修法（閉環）

terrain3 爬坡退步的根因是 **`-0.5·slip` 滑動懲罰**（該指標受速度污染，等於在罰「走得快」）學成過度保守慢步態；抬腳高度不是主因（inference 夾 gc 無效已證）。terrain3b 把 `W_SLIP` 降到 0.1 + 加 `+0.6·clip(blin[0],0,cmd[0])` 爬坡獎勵，一次命中。真正的防打滑靠摩擦下限（[0.5,1.25]），不是這個 reward 項。
