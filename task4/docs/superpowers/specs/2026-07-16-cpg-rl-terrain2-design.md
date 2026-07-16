# CPG-RL Terrain v2 設計 spec

> 建立：2026-07-16
> 一句話：在 v1（斜坡直走版）基礎上，做三件事——**(1) 恢復全向指令追蹤（平移/轉向）、(2) 地形加入「粗糙度漸變」的凹凸路面（統一 hfield）、(3) 抬腳高度改成每腿可學的 RL 輸出**。底層走路只靠本體感測，不引入任何外掛朝向/odom sensor。

相關檔案：
- 參考基準：`task4/notebooks/cpg_rl_paper_colab.ipynb`（論文全向版）、`task4/notebooks/cpg_rl_terrain_colab.ipynb`（v1 斜坡直走版）
- 地形研究：`task4/docs/cpg_rl_terrain_training_study.md`

---

## 0. 背景與動機（為什麼做這版）

v1（`cpg_rl_terrain_colab.ipynb`）有三個要修正的點：

1. **指令退化成只會直走**：v1 的 `_sample_cmd` 恆定回傳 `[vx,0,0]`（vy=0、wz=0），還加了絕對世界座標 `y_pen`。等於把論文版原有的**平移/轉向能力訓練掉了**。使用者真正要的是「**我下什麼方向指令，機器狗就精準往那走**」＝全向速度指令追蹤。
2. **只有斜坡、沒有凹凸**：要加入崎嶇路面（地面凹凸），且不希望平地步態過度保守。
3. **抬腳高度寫死**（`G_C=0.08`）：想改成 RL 輸出，讓策略在凹凸地形上自己學會該抬多高。

**明確排除（使用者指示）**：
- **不做朝向保持（heading-hold）**。底層走路不靠外掛 sensor，obs 不含絕對 yaw / odom。
- 「不要走歪」由**速度層**自然達成：命令 wz=0 → `r_yaw` 罰任何 yaw 角速度；命令 vy=0 → `r_lin` 罰橫向速度。長距離絕對朝向漂移交給上層（task5 odom 直線控制）修正，不是底層 policy 的責任。

---

## 1. 三大改動總覽

| 項目 | v1（現況） | v2（本設計） |
|---|---|---|
| 動作維度 | 12（每腿 μx,μy,ω） | **16（每腿 μx,μy,ω,g_c）** |
| 觀測維度 | 76 | **80**（last_action 12→16） |
| 指令取樣 | `[vx∈0.4~0.9, 0, 0]` | **`[vx∈0~1, vy∈±0.3, wz∈±1]`（全向）** |
| 抬腳高度 | 固定 `G_C=0.08` | **每腿 RL 可調 `g_c∈[0.03,0.15]`** |
| 地形 | 平台+box 斜坡 | **統一 hfield：平台+斜坡+粗糙度漸變凹凸** |
| 地面高度 `gz` | 解析 `interp(x)` | **對 H 網格雙線性內插 `gz(x,y)`** |
| reward | 速度追蹤 + `y_pen` 絕對位移罰 | **純速度追蹤，移除 `y_pen`** |

不變：PD（kp=90/kd=3）、domain randomization、抗推 kick、trot 相位耦合、PPO 超參結構、CPG 二階振幅動力學與相位耦合。

---

## 2. CPG 改動：每腿可學抬腳高度

### 2.1 動作映射（`action_to_cpg_cmd`）
動作從 12 → 16，reshape 成 `(4,4)`，每腿 4 個參數：

```python
GC_MIN, GC_MAX = 0.03, 0.15          # 抬腳高度可學範圍(m)

def action_to_cpg_cmd(action):
    a = jnp.tanh(action).reshape(4, 4)          # ★ (4,3)→(4,4)
    mux   = (a[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy   = (a[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    omega = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    gc    = (a[:, 3] + 1) / 2 * (GC_MAX - GC_MIN) + GC_MIN   # ★ 每腿抬腳高度 (4,)
    return mux, muy, omega, gc
```

- **下限 `GC_MIN=0.03`**：結構上防止在平滑段把腳壓到拖地（保留 v1「不能偷懶」的精神，但給 RL 空間）。
- **上限 `GC_MAX=0.15`**：讓策略能為較大凸起抬高。
- `G_P=0.01`（站立下壓）**維持固定**，不放給 RL。

### 2.2 腳掌偏移（`cpg_foot_offsets`）
`g_c` 從常數改成傳入的 shape-(4,) 參數：

```python
def cpg_foot_offsets(c, gc):                    # ★ 多收 gc 參數
    th = c["theta"]
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
    dx = -D_STEP * fx * jnp.cos(th)
    dy =  D_STEP * fy * jnp.cos(th)
    dz = jnp.where(jnp.sin(th) > 0, gc * jnp.sin(th), G_P * jnp.sin(th))  # ★ gc 每腿
    return jnp.stack([dx, dy, dz], axis=-1)
```

`gc` 是**每步的動作參數**（不是 CPG 內部狀態）；CPG 的 `cpg_step`/state 不變。`cpg_to_joint_targets` 多吃一個 `gc` 往下傳。

---

## 3. 地形：統一 hfield（粗糙度漸變）

### 3.1 高度場定義
整片地面用**一張 hfield**，覆蓋 x∈[−6,6]、y∈[−3,3]：

```
H(x, y) = slope(x) + amp(x) · bump(x, y)
```

- **`slope(x)`（斜坡角度 0→15°遞增）**：平台 x∈[−1,1]=0°（平）；上坡 +x **分段遞增 5°→10°→15°**（例如 x 1→2.5 為 5°、2.5→4 為 10°、4→6 為 15°，頂端約 +0.93m）；下坡 −x 鏡像。**最陡不超過 15°**，涵蓋 0–15° 的連續範圍。
- **`amp(x)`（粗糙度漸變，最高 8cm）**：`amp(x) = AMP_MAX · clip((|x|−1)/2, 0, 1)`，`AMP_MAX=0.08`。
  - `|x|<1`（平台）：amp=0 → **平滑平地**（乾淨出生/落地）。
  - `1<|x|<3`：amp 0→0.08 線性漸增 → **輕→中凹凸**。
  - `|x|>3`：amp=0.08 → **強凹凸（最高 8cm）+ 斜坡**。
- **`bump(x,y)`**：多個空間正弦疊加、正規化到 [−1,1]（波長 ~0.3–0.6m），**確定性**（幾何靜態、固定；不同步之間不變）。

效果：每回合機器人**先走平滑平台 → 輕凹凸 → 強凹凸+斜坡**，平地與難地都有代表性，步態不會過度偏保守。

### 3.2 hfield 建模（MjSpec）
- 網格解析度：`ncol`（x 向，12m）≈ 160、`nrow`（y 向，6m）≈ 80（cell ≈ 0.075m，足以解析 bump）。
- 正規化：`data = (H − Hmin)/(Hmax − Hmin) ∈ [0,1]`，row-major flatten。
- `hf.size = [6, 3, (Hmax−Hmin), 0.5]`；`floor` geom 改 `type=HFIELD`、`hfieldname`，`floor.pos.z = Hmin`，使平台（H=0）落在世界 z=0。
- 原無限地板降 `z=−10` 當安全底網。
- **MJX 相容性已驗證**：mujoco 3.10 的 MJX 支援 `hfield_sphere` 碰撞；Go2 腳為 sphere(r=0.0175) → hfield 會與腳正常碰撞。

### 3.3 地面高度查詢 `gz(x,y)`
提供 np 與 jax 兩版，對**儲存的 H 網格做雙線性內插**（與幾何同源）：
- reward 的 `rel_h`、跌倒判定、觸地布林全部改用 `gz(x,y)`（2D）。
- 觸地：`foot_z − gz(foot_x, foot_y) < FOOT_CONTACT_H(0.03)`。
- **驗證**：build 後用 `mj_ray` 在取樣點量實際表面，對照 `gz` 雙線性值，誤差 < 0.02（沿用 v1 的核對法）。

### 3.4 reset
- spawn 在平台 x=0（平滑平地），**隨機面 +x(上坡)/−x(下坡)**（與 slope 軸對齊，讓斜坡被有意義地訓練；全向能力由指令 vy/wz 提供）。
- 平台平滑處無穿透；沿用 keyframe home 姿態。

---

## 4. Reward：純速度指令追蹤（無朝向保持）

```python
def _sample_cmd(self, rng):                     # ★ 恢復全向
    k1, k2, k3 = jax.random.split(rng, 3)
    vx = jax.random.uniform(k1, (), minval=0.0, maxval=1.0)
    vy = jax.random.uniform(k2, (), minval=-0.3, maxval=0.3)
    wz = jax.random.uniform(k3, (), minval=-1.0, maxval=1.0)
    return jnp.array([vx, vy, wz])
```

```python
r_lin = jnp.exp(-((blin[0]-cmd[0])**2 + (blin[1]-cmd[1])**2) / 0.25)   # 追 vx,vy
r_yaw = jnp.exp(-((gyro[2]-cmd[2])**2) / 0.25)                          # 追 wz(yaw 角速度, IMU)
upright   = grav[0]**2 + grav[1]**2
rel_h     = data.qpos[2] - gz_j(data.qpos[0], data.qpos[1])            # ★ 2D 相對地面
height_pen= (rel_h - 0.30)**2
act_rate  = jnp.sum((action - last_action)**2)                         # 16 維

reward = (1.5*r_lin + 1.2*r_yaw - 1.0*upright
          - 0.5*height_pen - 0.05*act_rate + 0.05)                     # ★ 移除 y_pen
done   = jnp.where((rel_h < 0.18) | (grav[2] > -0.4), 1.0, 0.0)
# finite 防護沿用 v1
```

- **移除 `y_pen`**：世界 y 位移在全向指令下是合法的（下 vy/wz 就該偏移）。
- 「不要走歪」＝ wz=0 時 `r_yaw` 壓 yaw 角速度、vy=0 時 `r_lin` 壓橫向速度。都是本體感測可得（IMU 陀螺儀、機載線速度估計），**不需外掛朝向**。

---

## 5. 觀測（80 維）

```
grav(3) + blin(3) + gyro(3) + (qpos[7:19]−HOME12)(12) + qvel[6:18](12)
+ cmd(3) + last_action(16★) + foot_contact(4)
+ rx(4) + rx_d(4) + ry(4) + ry_d(4) + sin(theta)(4) + cos(theta)(4)
= 80
```

唯一相對 v1 的改變：`last_action` 12→16。**不加任何朝向/odom 觀測**。`nan_to_num + clip(-50,50)` 防護沿用。

---

## 6. 其餘（沿用 v1/論文版，不動）

- **PD**：`apply_pd` kp=90/kd=3、力矩上限（膝 45.43、其餘 23.7）。
- **Domain randomization**：摩擦 [0.3,1.0]、kp[75,105]、kd[2,4]、連桿質量 ±20%、軀幹負重 0~8kg。
- **抗推**：每 100 步（2s）注入隨機水平速度 kick（≤0.6 m/s）。
- **PPO**：policy 256/256/128、value 256³、normalize obs、`num_envs=2048`、`num_timesteps=2e8`（比 v1 稍難，OOM 就降 num_envs）、其餘超參同 v1。

---

## 7. 交付物

1. **新 notebook** `task4/notebooks/cpg_rl_terrain2_colab.ipynb`（不覆蓋 v1）。
   - 含 smoke test cell 與「跟隨不同指令（直走/轉向/橫移）在粗糙+斜坡上」的 rollout 影片 cell。
2. **權重** `task4/weights/cpg_rl_terrain2_params.pkl`（Colab 訓練後帶回）。
3. **本機推論** `task4/inference/local_infer_terrain2.py`（obs 80、action 16、每腿 gc、hfield 雙線性 gz、指令可腳本化；**底層不需 odom**）。

---

## 8. 向後相容：同時支援「抬腳寫死(舊)」與「抬腳可學(新)」兩種模型

使用者之後仍會回頭跑舊模型（抬腳寫死，action=12/obs=76）。因此**推論與實驗程式必須同時吃兩種權重**，並能在**同一套地形**上跑，才能做公平對比。

**兩版差異（只有三處）**：

| | 舊模型（fixed） | 新模型（learnable） |
|---|---|---|
| policy 輸出 | 12 | 16 |
| obs 維度 | 76 | 80 |
| CPG 抬腳 | 固定 `G_C=0.08` | 每腿 `gc = action[:,3]` 映射 |

其餘（μx/μy/ω 映射、IK、PD、CPG 相位、obs 其他欄位）**完全相同**。

**設計做法（`local_infer_terrain2.py` 與實驗腳本共用）**：
1. **自動偵測版本**：載入 policy 後看動作輸出維度（12→fixed、16→learnable），或吃顯式 `--foot-height {fixed,learnable}` 旗標覆寫。
2. **obs 建構參數化**：唯一差別是 `last_action` 長度（12 或 16）→ obs 維度隨之 76/80。其餘欄位組法一致。
3. **CPG target 參數化**：
   - fixed：`cpg_foot_offsets` 用常數 `G_C`；
   - learnable：從 `action` 解出每腿 `gc` 傳入。
   用同一份程式以 `foot_height_mode` 分支，不複製兩套。
4. **地形與模型解耦**：`gz(x,y)`／地形（flat / v1 box 斜坡 / v2 hfield）是**測試環境**屬性，與模型版本無關 → 任一模型都能在任一地形上跑。舊模型在 v2 hfield 上即為「零樣本測試」，觸地布林一律用該地形的 `gz` 計算（語意一致）。

**範圍界定**：本相容層針對 **12 維(fixed)** 與 **16 維(learnable)** 兩種。更早的 5 維實驗版（`cpg_rl_params.pkl`）動作/obs 結構不同，不在本相容層內；如需納入另議。

---

## 9. 已知風險 / 必須現場驗證

1. **MJX hfield 效能/正確性 @2048 envs**：先單環境 smoke test（reward 有限、腳不穿地、done 不誤觸發），再上 GPU；若太慢/OOM → 降網格解析度、縮 y 範圍或 num_envs。
2. **平滑段抬腳塌陷（拖地）**：靠 `GC_MIN=0.03` 下限 + 粗糙度漸變抑制；**監控指標**：平均 `gc`、FL 腳世界 z 抬腳量。若仍拖地，考慮加小額腳部離地獎勵。
3. **hfield 正規化 / pos 校正**：`mj_ray` 表面 vs 雙線性 `gz` 誤差 < 0.02 必過，否則觸地/rel_h 會錯。
4. **版本差異**：本機 mujoco 3.10 的 MJX 有 hfield；Colab `pip` 可能更新 → 務必在 Colab 先跑 smoke test。
5. **bump 振幅 vs 腳半徑/步幅**：`AMP_MAX=0.08m`（8cm）。`GC_MAX=0.15m`（15cm）對 8cm 凸起有清除餘裕；若訓練初期一直被 8cm 絆倒，可先用漸變讓遠端才到 8cm（已內建），或暫時降 `AMP_MAX` 熱身後再拉高。
6. **盲走**：obs 無地形高度圖屬預期設計；魯棒性靠 DR + 觸地反應，不靠預視。

---

## 10. 驗收標準

- Smoke test：obs shape=(80,)、上/下坡 spawn 正常、reward 有限、done 不誤觸發、腳不穿透 hfield。
- 訓練曲線收斂（eval reward 上升趨穩）。
- Rollout 影片可見：**下不同指令會往對應方向走**（直走、左右轉、橫移）、**能上下坡且跨過凹凸不拖地**、**平地步態不過度保守**。
- 監控指標：平均 `gc` 在凹凸段明顯高於平滑段（代表「有學到看地形抬腳」的間接證據——雖盲走，靠擾動反應）。
