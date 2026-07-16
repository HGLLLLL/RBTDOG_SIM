# 地形實驗 v2.1（2026-07-16）：抬腳強化模型 + odom 外圈閉環

模型：`weights/cpg_rl_terrain2_1_params.pkl`（在 v2 基礎上加**擺動卡住懲罰** `-0.4·scuff` + 抬腳下限 `GC_MIN` 0.03→0.05，重訓一輪）。
腳本：`task4/analysis/terrain_compare3.py`（odom 外圈閉環，同 07-15 課程）。與 `../2026-07-16/`（v2.0，無 scuff）逐檔對照。

```bash
MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare3.py \
  --params task4/weights/cpg_rl_terrain2_1_params.pkl --outdir task4/outputs/2026-07-16-v2.1 --exp both
```

## v2.0（無 scuff）vs v2.1（抬腳強化）— 皆 odom 閉環

| 指標 | v2.0（無 scuff） | **v2.1（scuff + GC_MIN 0.05）** |
|---|---|---|
| 平均抬腳 `gc` | 0.071 m | **0.092 m（+30%）** |
| slope 最大側偏 | 0.87 m | **0.07 m** |
| slope 平均 \|e_ct\| | 0.035 m | **0.023 m** |
| slope 跌倒 | @41.8s（走到跑道盡頭踏空） | **否** |
| rough 最大側偏 | 0.14 m | **0.07 m** |
| rough 平均 \|e_ct\| | 0.014 m | **0.018 m** |
| rough 跌倒 | 否 | 否 |

## 結論

- **抬腳強化成功**：平均 `gc` 從 0.071 → **0.092m（+30%）**，代表 scuff 懲罰確實逼策略在凹凸地把腳抬高。
- **slope 不再跌**：v2.0 的 slope「跌倒」是走得快、走到有限跑道盡頭（x≈23.3m）踏空；v2.1 抬得高、步伐略慢（45s 走到 x=18.4m，未達邊緣）→ 不再踏空，且線追蹤更緊（最大側偏 0.87→0.07m）。
- **rough 更順更直**：最大側偏 0.14→0.07m，全程不跌。
- **代價**：抬得高、前進略慢（slope 同 45s 走 18.4m vs v2.0 的 23.1m）——這是「抬高跨越 vs 前進速度」的正常取捨，換來凹凸不卡、更穩。

> 走順程度請直接比對 `exp2_rough_odom.mp4`（本夾 v2.1 vs `../2026-07-16/` v2.0）——v2.1 擺動腳抬得明顯高、卡頓應大幅減少。
> 若想再抬高，把 `W_scuff`（現 0.4）或 `GC_MIN` 再提高重訓即可；若覺得前進太慢，反向微調。
