# CPG-RL Terrain v3 設計 spec

> 建立：2026-07-21
> 一句話：在 v2.1（terrain2_1，抬腳強化 + odom 閉環）基礎上，**提高地形難度並根治斜坡打滑**——(1) 斜坡由 15° 漸進拉到 **30°**、(2) 障礙由 8cm 加高到 **12cm**、(3) 抬腳上限由 15cm 放寬到 **25cm**、(4) 用「摩擦下限提高 + 滑動懲罰 reward」修正上坡打滑。維度不變（obs 80 / action 16），沿用相同網路與 CPG 底層。

相關檔案：
- 參考基準：`task4/notebooks/cpg_rl_terrain2_colab.ipynb`（v2 訓練筆記本）、`task4/weights/cpg_rl_terrain2_1_params.pkl`（v2.1 權重）
- 前一版 spec：`task4/docs/superpowers/specs/2026-07-16-cpg-rl-terrain2-design.md`
- v2.1 結果：`task4/outputs/2026-07-16-v2.1/README.md`

---

## 0. 背景與動機（為什麼做這版）

v2.1 已達成「可學抬腳 + scuff 懲罰」，平均 `gc` 0.071→0.092m、斜坡側偏大減。使用者要在此基礎上加難度並修一個已知問題：

1. **地形太簡單**：斜坡最陡 15°、障礙最高 8cm。要提高到 **30° 斜坡 + 12cm 障礙**。
2. **15° 上坡打滑**（已觀察到的問題）：根因是 domain randomization 的摩擦下限 0.3 太低——15° 需 `μ≥tan15°=0.268`，走路推進還要更高餘裕，低摩擦環境（μ≈0.3）在蹬地時必滑。
3. **抬腳想更高**：為了跨 12cm 障礙，抬腳上限 15cm→**25cm**。

**明確排除／不在本次範圍**：
- 本機推論／出對比影片的 `local_infer_terrain3.py`（用新權重）是**後續另一步**，不在本筆記本內。
- **不動 repo 內任何舊檔**（`task4/inference/*.py`、舊 notebook）。新筆記本用 `%%writefile` 內嵌改版模組，Colab 內自成一體。

---

## 1. 四大改動總覽

| 項目 | v2.1（現況） | v3（本設計） |
|---|---|---|
| 斜坡最陡 | 15°（5→10→15 漸進） | **30°（5→10→15→20→25→30 漸進）** |
| 障礙最高 `AMP_MAX` | 0.08m | **0.12m** |
| 抬腳範圍 `GC_MIN/MAX` | 0.05 / 0.15 | **0.05 / 0.25** |
| 摩擦隨機範圍 | `[0.3, 1.0]` | **`[0.5, 1.25]`** |
| 打滑懲罰 | 無（僅 scuff 抬腳懲罰） | **新增 `-W_slip·slip`（支撐腳水平滑動）** |
| 地形長度 `TERR_X_MAX` | 6.0 | **7.0** |
| hfield 解析度 `ncol` | 161 | **189**（維持 ~0.075 m/格） |
| 訓練步數 `num_timesteps` | 2e8 | **3e8** |

不變：obs 80 / action 16、網路結構、PD（kp=90/kd=3）、抗推 kick、trot 相位耦合、scuff 抬腳懲罰、上/下坡各半 spawn、CPG 二階振幅動力學、其餘 PPO 超參、空間課程（中央平台 spawn、往外漸難）。

---

## 2. 地形 terrain3：斜坡漸進到 30° + 障礙 12cm

### 2.1 斜坡分段（`KNOTS_X` / `KNOTS_Z`）
分段 5→10→15→20→25→30°，每段 **1m 水平跑道**，`TERR_X_MAX: 6.0 → 7.0`。+x 側（上坡），對稱到 -x（下坡）：

| x (m) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 該段角度 | 平台邊 | 5° | 10° | 15° | 20° | 25° | 30° |
| 累積 z (m) | 0 | 0.087 | 0.264 | 0.532 | 0.896 | 1.362 | 1.939 |

每段 rise = `1.0 · tan(角度)`。`slope_z(x)` 仍用 `np.interp(x, KNOTS_X, KNOTS_Z)`，公式結構不變，只換 knots。

### 2.2 障礙幅度
`AMP_MAX: 0.08 → 0.12`。`amp_at(x)` 漸增邏輯不變（`|x|≥PLATFORM_HALF+2=3` 達滿幅 12cm），`bump(x,y)` 多正弦疊加不變。障礙疊加在斜坡上（`Hg = slope_z + amp_at·bump`），故遠處是「陡坡 + 大障礙」同時出現。

### 2.3 hfield 解析度
跑道加長（12m→14m），`build_height_grid(ncol=189, nrow=81)` 維持 ~0.075 m/格。`build_terrain2_model` 邏輯不變（平台對到 z=0、z=-10 安全網、`geom_margin/gap` 清零）；函式與 hfield 名稱在 v3 模組內改為 `build_terrain3_model` / `"terrain3"`。

---

## 3. CPG cpg3：抬腳上限放寬到 25cm

- 僅改常數：`GC_MIN, GC_MAX = 0.05, 0.15 → 0.05, 0.25`。其餘（動作映射、Hopf 振幅動力學、`dz = gc·sin(θ)`、IK）全部不變。
- ⚠️ **實作註記**：25cm 對 Go2 很高。`dz` 經 `jinv` 線性化（在 home 附近展開）再被 `actuator_ctrlrange` 夾限，**實際抬腳可能抬不滿 25cm**（尤其接近關節極限時）。不會壞，只會飽和。訓練靠 `gc_mean` metric 監控策略實際用到多少。

---

## 4. 打滑修法（兩管齊下）

### 4.1 摩擦下限提高（`domain_randomize`）
`geom_friction[:,0]` 由 `uniform(0.3, 1.0)` → **`uniform(0.5, 1.25)`**。
- 下限 0.5：靜態可撐 `atan(0.5)=26.6°`；約 11% 的環境（μ∈[0.5,0.577]）撐不住 30°，那些會在陡坡段學到「放慢別摔」——刻意保留一定低摩擦 robustness，讓新模型在偏滑的舊環境也還能走（見 §7 取捨）。
- 上限 1.25：高摩擦環境提供 30° 可靠攀爬的樣本。
- 足端 geom `priority=1`，但 domain randomize 對所有 geom 設同一 μ，故足端有效 μ = 抽樣值。

### 4.2 滑動懲罰 reward（新增）
懲罰**觸地支撐腳的世界座標水平滑動速度**（planted 腳理應原地不動，動了就是滑）：

```python
W_SLIP = 0.5                                    # 起始值，標為可調常數
# step() 內：
foot_xy   = data.geom_xpos[self._foot_gid, :2]  # (4,2) 本步足端 xy
prev_xy   = state.info["foot_xy"]               # 上一步足端 xy
slip_vel  = jnp.linalg.norm(foot_xy - prev_xy, axis=1) / CTRL_DT   # (4,) m/s
contact   = self._foot_contact(data)            # (4,) 觸地指標（沿用既有）
slip      = jnp.sum(contact * jnp.clip(slip_vel, 0.0, 1.0))        # 有界 [0,4]
reward    = ... - W_SLIP * slip                 # 併入既有 reward
```

- **有界**：`clip(slip_vel,0,1)` → `slip∈[0,4]`，`W_slip=0.5` → 罰 ≤ 2，與其他項同量級，不破壞既有 `reward∈~[-2,2.75]` 與 `-5` 下界設計。
- **state 變更**：`info` 新增 `"foot_xy"`；`reset` 時設為當前足端 xy（首步 slip=0）；`step` 每步更新。
- **監控**：`metrics` 新增 `"slip"`（每步平均滑動量）。

---

## 5. 訓練（從零 + 靠既有空間課程）

- **從零訓練**：機器狗 spawn 中央平台，往外走進漸陡斜坡／漸大障礙（空間課程已內建，不需時間課程）。
- `num_timesteps: 2e8 → 3e8`（難度顯著提高，給更多步數）。其餘 PPO 超參沿用 v2（`num_envs=2048`、`batch_size=256`、`lr=3e-4`、`entropy_cost=3e-3`、`discounting=0.97` 等）。OOM 就降 `num_envs`。
- progress 印出加 `slip`，連同既有 `gc / scuff / relh / len`。

---

## 6. 筆記本結構（交付物）

新檔 `task4/notebooks/cpg_rl_terrain3_colab.ipynb`，沿用 v2 筆記本骨架（`%%writefile → import`），cell 順序：

1. GPU 安裝 + `MUJOCO_GL=egl`
2. clone menagerie（`scene_mjx.xml`）
3. `%%writefile` 四模組：**cpg3 → terrain3 → obs3 → go2_terrain3_env**
   - `cpg3.py`：v2 cpg2 內容，`GC_MAX 0.15→0.25`
   - `terrain3.py`：v2 terrain2 內容，新 knots（到 30°）、`AMP_MAX 0.12`、`TERR_X_MAX 7.0`、`ncol 189`、函式/hfield 名 terrain3
   - `obs3.py`：與 obs2 內容相同（obs 不變），改名為 obs3
   - `go2_terrain3_env.py`：import terrain3/cpg3/obs3；`domain_randomize` 摩擦 `[0.5,1.25]`；`step`/`reset` 加滑動懲罰與 `foot_xy` state、`slip` metric
4. import 四模組
5. Smoke test（檢查 obs=80、上/下坡 spawn、reward 有限、done 不誤觸發、`gc_mean` 與新 `slip` metric 有值）
6. Brax PPO 訓練（`num_timesteps=3e8`）
7. reward 曲線
8. Rollout 影片（沿用 v2 的直走/轉向/橫移三段；地面用 `terrain3.gz_np`）
9. 存權重 `cpg_rl_terrain3_params.pkl` + 下載

⚠️ 未在 GPU 實跑過，開訓前**務必先跑 Smoke test cell**。

---

## 7. 已知取捨與風險

1. **摩擦取捨（policy 盲於摩擦）**：obs 不含摩擦係數，policy 學的是「訓練摩擦分布下的固定步態」，無法部署時自動變保守。範圍由 `[0.3,1.0]` 改 `[0.5,1.25]` 是刻意折衷：保留到 μ=0.5 的 robustness（平地 + ~20° 內仍穩、較接近舊環境），代價是最低摩擦（μ=0.3）場景不再訓練、極滑地面表現會弱於 v2.1。**平地／緩坡在任何摩擦下都正常；低摩擦 + 陡坡才是弱區。**
2. **30° 可行性**：30° 對盲走 CPG 是很陡的目標，即使摩擦足夠也可能難學或速度很慢。若 3e8 步後 30° 段仍學不起來，退路：把最陡段降到 25°、或每段跑道加長到 1.5m（更平緩）、或延長 timesteps。
3. **25cm 抬腳**：受 IK 線性化 + ctrlrange 夾限，實際可能抬不滿；靠 `gc_mean` 監控。
4. **難度一次跳很大**（15°→30°、8→12cm）：從零訓練在最高難度可能收斂慢。空間課程（近易遠難）是主要緩解；若不收斂再考慮時間課程。

---

## 8. 可調常數彙總（筆記本內標明）

| 常數 | 位置 | v3 值 | 備註 |
|---|---|---|---|
| `GC_MAX` | cpg3 | 0.25 | 抬腳上限 |
| `AMP_MAX` | terrain3 | 0.12 | 障礙最高 |
| `TERR_X_MAX` | terrain3 | 7.0 | 跑道半長 |
| 斜坡每段長 | terrain3 knots | 1.0m | 加大→更平緩 |
| 摩擦範圍 | go2_terrain3_env `domain_randomize` | [0.5, 1.25] | robustness↔陡坡取捨 |
| `W_SLIP` | go2_terrain3_env | 0.5 | 滑動懲罰權重 |
| `num_timesteps` | 訓練 cell | 3e8 | GPU 時間 |
