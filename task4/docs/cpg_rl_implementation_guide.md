# CPG-RL 實作 Step-by-Step（MJX + Colab GPU → 本機推論）

> 目標：把你現有的 `Go2Gait`（固定參數的相位步態）升級成 **CPG-RL**——讓一個 RL 策略即時去「調變振盪器的振幅與頻率」，狗就能因應狀態、外力、地形自己調節步態。
>
> 你已確認的前提（本指南完全依此展開）：
> - **機器人**：Unitree Go2（`mujoco_menagerie/unitree_go2`）。
> - **訓練**：Colab 免費 GPU + **MJX（JAX 版 MuJoCo）** + **Brax PPO**。
> - **推論**：訓練完把權重帶回**本機 CPU**，接回你 task3 的 MuJoCo 迴圈跑。
> - **你已有 CPG**：`task3/go2_gait.py` 的 `Go2Gait`（相位時鐘 + 腳掌軌跡 + 常數 Jacobian IK + 軟體 PD）。
>
> 你的程度：懂片面 RL 基礎、沒 MuJoCo/JAX 深度經驗。所以本指南**每一步都白話解釋 + 給可改的程式骨架 + 給驗收標準**。JAX/MJX 是這條路最陡的地方，我會標出來，出錯時照「除錯區」查。

---

## ⚠️ 先讀：這條路的難點與心態

你選的 MJX/JAX 路線**訓練最快（分鐘級）**，代價是要用 **JAX 的思維寫程式**：
- JAX 要求「純函數 + 固定形狀陣列 + 不能用 Python 迴圈亂改狀態」。你 task3 那種 `for` 迴圈、就地改 `d.qpos` 的寫法在 MJX 裡**不能直接用**，要改成函數式。
- 好消息：**你要自己新寫的 JAX 程式其實很少**——只有「CPG 振盪器」那一小塊。其餘（物理、觀測、獎勵、訓練）都**站在 MuJoCo Playground 現成的 Go1/Go2 環境肩膀上改**，不是從零幹。

**策略：不要從白紙寫 MJX 環境。** 我們用 Playground 內建、已驗證能走的 `Go1/Go2 Joystick` 環境當模板，**只在「策略動作 → 關節目標」中間插入 CPG 這一層**。這樣把新程式和風險壓到最小。

全流程長這樣：

```
┌─────────────────────── Colab（GPU，訓練一次）───────────────────────┐
│  1. 裝 mujoco / mjx / brax / playground                             │
│  2. 寫 JAX 版 CPG（唯一的新核心，~40 行）                            │
│  3. 複製 Playground Go2 Joystick env → 插入 CPG 層                   │
│  4. Brax PPO 訓練（GPU 上幾千隻狗平行，數分鐘）                       │
│  5. 存權重 params.pkl                                                │
└─────────────────────────────────────────────────────────────────────┘
                                │  下載 params.pkl
                                ▼
┌────────────────────── 本機（CPU，永久推論）─────────────────────────┐
│  6. NumPy 版 CPG（鏡像 JAX 版）+ 載入策略權重                         │
│  7. 接回 task3 的 mj_step 迴圈：obs → 策略 → μ,ω → CPG → IK → PD     │
│  8.（加分）接上 task3 的羅盤航向控制走直線                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 第 0 步：觀念對齊——你的 `Go2Gait` 和 CPG-RL 差在哪

先看懂這張對照，後面才不會迷路。

| 元件 | 你現在的 `Go2Gait`（開迴路） | CPG-RL（閉迴路） |
|---|---|---|
| 節奏來源 | `phase = freq * t + 固定偏移`（時鐘） | **Hopf 振盪器**：狀態 `(r, θ)` 由微分方程積分出來 |
| 誰決定振幅/頻率 | 你手調的固定值（stride/freq） | **RL 策略每一步即時輸出** μ（振幅目標）、ω（頻率） |
| 有沒有回授 | 沒有（不看機器狀態） | **有**：策略讀身體姿態/速度/接觸 → 調 μ,ω |
| 腳掌→關節 | 腳掌軌跡 + 常數 Jacobian IK | **完全一樣，直接沿用你的** |
| 關節→力矩 | 軟體 PD | **完全一樣，直接沿用你的** |
| 轉彎 | 手寫 `turn_gain` 左右差動 | RL 用左右腿 μ 不對稱**自己學會**轉 |

**一句話**：CPG-RL 只是把你「固定的 freq/stride 旋鈕」換成「RL 即時轉的旋鈕」，並把節奏產生器從時鐘換成振盪器。**你的 IK 和 PD 原封不動重用。**

### CPG-RL 的三要素（RL 的標準語彙對到這個任務）

- **Observation（策略看到什麼）**：指令速度 `(vx*, vy*, yaw_rate*)`、身體姿態（重力方向/四元數）、身體線/角速度、關節角度與速度、**振盪器自身狀態 `(r, ṙ, θ)`**、上一步動作，（進階）腳觸地布林、地形高度。
- **Action（策略輸出什麼）**：每條腿的**振幅目標 μ** 與**頻率 ω**。原論文是每腿 (μx, μy, ω) 共 **12 維**；**我們起步用簡化版：4 條腿的 μ（4 維）＋ 共用頻率 ω（1 維）＝ 5 維**（最穩，之後再擴充）。
- **Reward（怎麼評分）**：追上指令速度（主項）＋ 身體別歪別掉高度 ＋ 力矩/動作平滑（省電防抖）＋ 存活 bonus。因為 CPG 保證動作平順，**獎勵可以很簡單**。

### CPG-RL 的振盪器數學（查證自論文，先看角色不用背）

每條腿 i 一個振盪器，狀態是振幅 `r_i`、相位 `θ_i`：

- **振幅動力學（二階，讓 r 平滑收斂到 RL 給的目標 μ）**：
  `r̈_i = a · ( a/4 · (μ_i − r_i) − ṙ_i )`，收斂係數 `a ≈ 50`。
  （白話：μ 是「想要的振幅」，r 是「實際振幅」，這條式子讓 r 像有阻尼的彈簧一樣平順追上 μ，不會突跳。）
- **相位動力學**：`θ̇_i = 2π·ω_i (+ 腿間耦合項)`。
  （白話：ω 是步頻，決定腳擺多快。耦合項讓四條腿相位鎖在一起，起步版我們用「固定相位偏移 + 共用 ω」取代耦合，一樣鎖成 trot。）
- **振幅/相位 → 腳掌位置**（我們用貼合你 `foot_traj` 的簡化版）：
  `dx = stride · amp · cos(θ)`（前後擺）
  `dz = max(0, lift · amp · sin(θ))`（只有擺動半週抬腳）
  其中 `amp = clip(r, 0, 1)`（振幅 0=原地踏、1=滿步幅）。
  → 再丟進**你現有的 `Jinv` IK** 得到 thigh/calf 角度。

> 論文原版的 foot 映射是 `x=-d·f(r)·cos θ`、`z=-h+g·sin θ`（swing/stance 用 `sin θ` 正負分段），`f(r)=2(r-μmin)/(μmax-μmin)-1`。我們的簡化版精神相同、但直接對齊你 task3 已調好的 stride/lift，起步更穩。想更「教科書」時可換成原式（見延伸）。

---

## 第 1 步：開好 Colab 環境（GPU）

1. 開一個新的 Colab notebook。左上 **執行階段 → 變更執行階段類型 → 硬體加速器選 GPU（T4 即可）**。
2. 第一個 cell 裝套件並確認 JAX 看得到 GPU：

```python
!pip install -q mujoco mujoco-mjx brax
!pip install -q playground   # MuJoCo Playground（套件名 playground）
import jax
print("JAX devices:", jax.devices())      # 要看到 [cuda(id=0)]，不是 cpu
```

**驗收 ✅**：`jax.devices()` 印出 `cuda`。若印 `cpu`，回去把硬體加速器改成 GPU 重開執行階段。

> 為什麼裝這些：`mujoco`=物理引擎、`mujoco-mjx`=它的 GPU/JAX 版、`brax`=Google 的 JAX RL 庫（提供 PPO）、`playground`=一堆現成機器人環境（含 Go1/Go2 走路），我們拿它當模板。

---

## 第 2 步：寫 JAX 版 CPG（唯一要自己寫的核心，約 40 行）

這是整個任務**唯一的新演算法**。放進一個 cell。重點：**純函數、輸入輸出都是 `jnp` 陣列、不可有就地修改**。

```python
import jax, jax.numpy as jnp

# ---- 常數（起步值：對齊你 task3 GAIT，之後可調）----
A_CONV     = 50.0                                   # 振幅收斂係數 a
MU_MIN, MU_MAX = 0.0, 1.0                            # 振幅目標範圍（0=踏步,1=滿步幅）
OMEGA_MIN, OMEGA_MAX = 0.5, 4.0                      # 頻率範圍 (Hz)
STRIDE     = 0.32                                    # 步幅 (m)  ← task3 stride
LIFT       = 0.12                                    # 抬腳高 (m) ← task3 lift
# trot 相位偏移 FL,FR,RL,RR（對角同相）：FL/RR=0, FR/RL=π
PHASE_OFFSET = jnp.array([0.0, jnp.pi, jnp.pi, 0.0])

def cpg_init():
    """回傳初始振盪器狀態（4 腿）。"""
    return {"r": jnp.zeros(4), "r_dot": jnp.zeros(4), "phi": jnp.array(0.0)}

def cpg_step(cpg, mu, omega, dt):
    """積分一個控制步。mu:(4,) 每腿振幅目標；omega:純量 共用頻率。"""
    r, r_dot, phi = cpg["r"], cpg["r_dot"], cpg["phi"]
    r_ddot = A_CONV * (A_CONV / 4.0 * (mu - r) - r_dot)     # 二階振幅動力學
    r_dot  = r_dot + r_ddot * dt
    r      = r + r_dot * dt
    phi    = jnp.mod(phi + 2 * jnp.pi * omega * dt, 2 * jnp.pi)
    return {"r": r, "r_dot": r_dot, "phi": phi}

def cpg_to_joint_targets(cpg, f0, Jinv, center, home):
    """振盪器狀態 → 12 個關節目標角度。f0/Jinv/center/home 是常數（見第 3 步）。"""
    theta = cpg["phi"] + PHASE_OFFSET                       # (4,)
    amp   = jnp.clip(cpg["r"], 0.0, 1.0)
    dx = -STRIDE * amp * jnp.cos(theta)                     # (4,) 前後（負號: 站立相前->後推進）
    dz = jnp.maximum(0.0, LIFT * amp * jnp.sin(theta))      # (4,) 抬腳
    foot = jnp.stack([center[0] + dx, center[1] + dz], -1) # (4,2) 腳掌(x,z)
    dq   = (foot - f0) @ Jinv.T                             # (4,2) → (dth,dca)
    q = jnp.zeros((4, 3))
    q = q.at[:, 1].set(home[1] + dq[:, 0])                  # thigh
    q = q.at[:, 2].set(home[2] + dq[:, 1])                  # calf（hip=0）
    return q.reshape(12)

def action_to_cpg_cmd(action):
    """策略輸出 action∈[-1,1]^5 → (mu:(4,), omega:純量)。"""
    a = jnp.tanh(action)                                    # 保險夾到 [-1,1]
    mu    = (a[:4] + 1.0) / 2.0 * (MU_MAX - MU_MIN) + MU_MIN
    omega = (a[4]  + 1.0) / 2.0 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mu, omega
```

**驗收 ✅**：能 `cpg = cpg_init()`、跑 `cpg = cpg_step(cpg, jnp.ones(4)*0.5, 2.0, 0.02)` 不報錯，且 `cpg["phi"]` 有在前進。

---

## 第 3 步：把你 task3 的常數 IK 搬成 JAX 常數

你的 `Go2Gait._calc_ik()` 用「在 home 姿態對真模型做有限差分」算出 `f0`（home 腳掌位置）和 `Jinv`（常數 2×2）。**它是常數**，所以我們在 **CPU 上用你現成的程式算一次**，再把數字當 JAX 常數用。

在 Colab 先把 `go2_model.py`、`go2_gait.py` 上傳（或 `git clone` 你的 repo），然後：

```python
import numpy as np
from go2_gait import Go2Gait, HOME
g = Go2Gait(**dict(freq=2.6, duty=0.60, lift=0.12, stride=0.32, kp=90, kd=3.0))
F0     = jnp.array(g.f0)               # home 腳掌 (x,z)
JINV   = jnp.array(g.Jinv)             # 常數 2x2
CENTER = jnp.array(g.center)           # 站姿中心 (f0x, z0)
HOME_J = jnp.array(HOME)               # [0, 0.9, -1.8]
KP, KD = 90.0, 3.0                      # 沿用你的 PD 增益
print("F0", F0, "CENTER", CENTER); print("JINV\n", JINV)
```

**驗收 ✅**：印出的 `JINV` 是 2×2 有限值，`F0`/`CENTER` 合理（z 約 -0.28）。這代表你的 IK 已成功變成 CPG-RL 能用的常數。

> 為什麼能重用：IK/PD 跟「用時鐘還是振盪器產生節奏」完全無關，所以你 task3 調好的這兩塊直接繼承，省掉一大堆重調。

---

## 第 4 步：以 Playground Go2 環境為模板，插入 CPG 層

**不要從零寫 MJX 環境。** Playground 內建的 Go2/Go1 Joystick 環境（繼承 `MjxEnv`，已處理物理、觀測、獎勵、domain randomization、指令取樣、終止判定）就是模板。你要做的只有兩件事：

1. **改 `action_size`**：從「12 個關節」改成「**5**」（我們的 CPG 動作維度）。
2. **在 `step()` 裡插一層**：把原本「action 直接當關節目標」改成「action → CPG → 關節目標」，並把 **CPG 狀態存進 `state.info`**（JAX 環境不能用物件屬性存狀態，必須放進會隨 `state` 傳遞的 `info` dict）。

先找到模板環境檔（在安裝路徑下 `mujoco_playground/_src/locomotion/go1/joystick.py` 之類），複製成你自己的 `go2_cpg.py`。關鍵改法示意（**只列要改的地方，其餘沿用模板**）：

```python
# --- 在 reset() 裡：初始化 CPG，塞進 info ---
def reset(self, rng):
    state = super().reset(rng)             # 沿用模板：擺好狗、取樣指令
    state.info["cpg"] = cpg_init()         # ★ 新增：振盪器初始狀態
    return state

# --- 在 step() 裡：action → CPG → 關節目標 → 再走物理 ---
def step(self, state, action):             # action shape = (5,)
    dt = self.dt                           # 控制步長（模板已定義，例如 0.02s）
    mu, omega = action_to_cpg_cmd(action)          # ★ 第2步的函數
    cpg = cpg_step(state.info["cpg"], mu, omega, dt)  # ★ 積分振盪器
    q_des = cpg_to_joint_targets(cpg, F0, JINV, CENTER, HOME_J)  # ★ 12 關節目標

    # 用 PD 把關節目標轉成力矩，或若模板致動器是 position 就直接給 q_des。
    # 你 task3 是力矩+軟體PD → 這裡也用 PD：
    qpos_j = state.data.qpos[7:19]         # 12 關節角
    qvel_j = state.data.qvel[6:18]
    torque = KP * (q_des - qpos_j) - KD * qvel_j

    data = self.pipeline_step(state.data, torque)   # ★ 用力矩走一個物理步
    # 之後：組觀測、算獎勵、判終止 —— 盡量沿用模板，只把 CPG 狀態加進觀測
    obs   = self._get_obs(data, state.info, cpg)    # 觀測裡加 r, r_dot, sin/cos(phi)
    reward, done = self._reward_and_done(data, action, state.info)
    state.info["cpg"] = cpg                          # ★ 存回更新後的振盪器
    return state.replace(data=data, obs=obs, reward=reward, done=done)
```

**觀測要記得把 CPG 狀態放進去**（策略要「知道現在振盪器在哪」才能決策）：在 `_get_obs` 的向量最後接上 `cpg["r"]`（4）、`cpg["r_dot"]`（4）、`jnp.sin(cpg["phi"])`、`jnp.cos(cpg["phi"])`。

**獎勵**先用最簡單版（沿用模板的 tracking 項 + 幾個懲罰）：
```
reward =  1.0 * 追上指令線速度 (exp(-|v_xy - v_cmd|²/σ))
        + 0.5 * 追上指令 yaw 角速度
        - 0.5 * 身體傾斜（roll/pitch 或 gravity_xy）
        - 0.1 * 動作變化率 |aₜ - aₜ₋₁|²      （防抖，關鍵）
        - 1e-4 * 力矩平方              （省電）
        + 存活每步小 bonus / 跌倒大扣分並 done
```
`done` = 身體高度過低（如 <0.15）或翻覆。

**驗收 ✅**：能在 Colab 跑
```python
env = Go2CpgEnv()                 # 你的新 class
s = jax.jit(env.reset)(jax.random.PRNGKey(0))
s = jax.jit(env.step)(s, jnp.zeros(5))
print(s.obs.shape, s.reward)      # 不報錯、形狀合理
```
> 這步是整條路**最容易卡**的地方（JAX 形狀/純函數規則）。卡住時對照 Playground 官方 `Creating Custom Environments` 文件，並看本檔「除錯區」。

---

## 第 5 步：用 Brax PPO 訓練（Colab GPU）

Playground 的訓練 pipeline 就是 Brax 的 PPO。**最省事：照官方 `learning/notebooks/locomotion.ipynb` 的訓練 cell，把 `environment` 換成你的 CPG 環境。** 代表性寫法：

```python
import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

env = Go2CpgEnv()
train_fn = functools.partial(
    ppo.train,
    num_timesteps=60_000_000,      # CPG-RL 樣本效率高，先 6e7 試（GPU 上數分鐘~十幾分）
    num_envs=4096,                 # GPU 平行環境數（記憶體不足就降到 2048）
    episode_length=1000,
    unroll_length=20, num_minibatches=32, num_updates_per_batch=4,
    learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
    batch_size=256, seed=0,
    network_factory=functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128)),  # 小網路夠用
)

def progress(step, metrics):
    print(f"step {step:>10}  reward {metrics['eval/episode_reward']:.2f}")

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
```

**驗收 ✅**：`reward` 隨步數上升並收斂到穩定正值（例如追速度獎勵接近上限）。訓完在 Colab 直接 rollout 存一段影片看狗會不會走：用 `make_inference_fn(params)` 產生策略，餵給 `env.step` 跑幾百步、`env.render`。

> 調不動時的常見對策：先把指令固定成「直走 0.5 m/s」（別一開始就學轉彎）、把獎勵只留「追速度 + 存活 + 防抖」三項、`num_timesteps` 加大。CPG 結構會幫你擋掉大部分鬼畜動作。

---

## 第 6 步：存權重帶回本機

```python
from brax.io import model
model.save_params("/content/cpg_rl_params.pkl", params)
from google.colab import files; files.download("/content/cpg_rl_params.pkl")
```

**驗收 ✅**：本機拿到 `cpg_rl_params.pkl`。訓練階段到此結束，之後**永遠不用再上雲**。

---

## 第 7 步：本機 CPU 推論——接回你的 MuJoCo 迴圈

本機**不需要 MJX**（不用 GPU 平行），只需要：跑物理用你現成的 `mujoco`（CPU），跑策略網路用 **JAX CPU**（一個小 MLP 在 CPU 上每秒跑幾百次綽綽有餘）。

本機環境（你的 `rbtdog` conda）補裝 CPU 版：
```bash
pip install jax brax        # 預設就是 CPU 版，夠跑推論
```

把 CPG 用 **NumPy 重寫一份**（和第 2 步 JAX 版**邏輯逐行對應**，只是把 `jnp` 換 `np`、`.at[].set()` 換一般賦值），然後在你 task3 的迴圈裡串起來：

```python
import numpy as np, mujoco
from go2_gait import Go2Gait, HOME
from brax.io import model

g = Go2Gait(freq=2.6, duty=0.60, lift=0.12, stride=0.32, kp=90, kd=3.0)
params = model.load_params("cpg_rl_params.pkl")
policy = make_inference_fn(params)          # 用訓練時同一個 make_inference_fn 定義

cpg = dict(r=np.zeros(4), r_dot=np.zeros(4), phi=0.0)
PHASE_OFFSET = np.array([0.0, np.pi, np.pi, 0.0])

def build_obs(g, cpg, v_cmd):
    # ★ 必須和 Colab env 的 _get_obs「同順序、同內容、同單位」，否則策略亂掉
    quat = g.sensor("imu_quat"); gyro = g.sensor("imu_gyro")
    qj = g.d.qpos[7:19]; dqj = g.d.qvel[6:18]
    return np.concatenate([v_cmd, quat, gyro, qj, dqj,
                           cpg["r"], cpg["r_dot"],
                           [np.sin(cpg["phi"]), np.cos(cpg["phi"])]]).astype(np.float32)

dt = g.m.opt.timestep * N_SUBSTEP            # 控制步長要與訓練一致
while running:
    obs = build_obs(g, cpg, v_cmd=np.array([0.5, 0.0, 0.0]))
    action = np.array(policy(obs, rng))      # 策略輸出 5 維
    mu, omega = action_to_cpg_cmd_np(action)
    cpg = cpg_step_np(cpg, mu, omega, dt)    # NumPy 版 CPG
    q_des = cpg_to_joint_targets_np(cpg, g.f0, g.Jinv, g.center, HOME)
    tau = g.kp*(q_des - g.d.qpos[7:19]) - g.kd*g.d.qvel[6:18]
    g.d.ctrl[:] = np.clip(tau, -g.flimit, g.flimit)
    for _ in range(N_SUBSTEP):
        mujoco.mj_step(g.m, g.d)
```

**驗收 ✅**：本機 viewer 裡，狗用學到的策略往前走且穩定，行為和 Colab rollout 一致（這叫 **sim-to-sim 對齊**）。

> **最容易翻車的點：觀測不一致。** 本機 `build_obs` 的欄位順序、內容、單位、以及控制頻率（`dt`）**必須和 Colab 訓練環境一模一樣**。差一個順序或縮放，策略就會亂走。除錯時印出兩邊 obs 的前幾維比對。

---

## 第 8 步（加分）：接上 task3 的羅盤走直線

你 task3 的外層羅盤 P 控制（`turn = COMPASS_GAIN * yaw誤差`）可以直接包在 CPG-RL 外面：把 `turn` 併進指令 `yaw_rate*`（obs 的指令欄位），策略就會用 CPG-RL 的方式執行轉向去修正航向。等於「RL 走路 + 你的羅盤導航」兩層合體。

**驗收 ✅**：重跑 `walk_line.py` 的對比實驗，CPG-RL 版在外力干擾下仍能鎖航向走直，且比純開迴路更抗擾。

---

## 除錯區（JAX/MJX 常見雷）

| 症狀 | 多半原因 | 解法 |
|---|---|---|
| `jax.devices()` 是 cpu | Colab 沒開 GPU | 執行階段→變更類型→GPU→重開 |
| `TracerArrayConversionError` / `Concretization` | 在 JAX 函數裡用了 Python `if`/`for` 看陣列值 | 改用 `jnp.where`、`jax.lax.cond`、向量化 |
| `.at[].set()` 忘了 | JAX 陣列不可就地改 | 一律 `x = x.at[i].set(v)` |
| 形狀不合報錯 | 動作/觀測維度沒對上 | 確認 `action_size=5`、obs 拼接維度一致 |
| MJX 物理爆炸/穿地 | MJX 接觸模型較敏感、timestep 太大 | 先用 Playground **原版 Go2/Go1** 確認能走，再改；縮 timestep；降 KP |
| 訓練 reward 不動 | 獎勵太複雜/指令太難 | 先只學直走、精簡獎勵、加大 timesteps |
| 學到原地抖/滑步 | 缺防抖懲罰 | 加大「動作變化率」「力矩」懲罰 |
| 本機推論亂走 | **obs 不一致**（順序/單位/頻率） | 逐維比對 Colab vs 本機 obs、對齊控制頻率 |

---

## 里程碑檢查表（照順序打勾）

- [ ] **M1** Colab GPU 開好，`jax.devices()` 見 cuda（第1步）
- [ ] **M2** JAX 版 CPG 三個函數能跑不報錯（第2步）
- [ ] **M3** 你的 IK 常數 `F0/JINV/CENTER` 成功轉成 JAX 常數（第3步）
- [ ] **M4** CPG 環境 `reset/step` 能 jit 跑通、形狀正確（第4步）← 最大關卡
- [ ] **M5** PPO 訓練 reward 收斂、Colab rollout 看到狗會走（第5步）
- [ ] **M6** 權重下載到本機（第6步）
- [ ] **M7** 本機 CPU 推論走路，與 Colab 一致（第7步）
- [ ] **M8** 接上羅盤走直線（第8步，加分）

---

## 延伸：從「起步版」升級到「教科書 CPG-RL」

起步跑通後，想更接近論文原版可逐步加：
1. **動作維度 5 → 12**：改成每腿 (μx, μy, ω)，加入側向 `y` 腳掌位移（需要你 IK 從 2D(x,z) 擴到含 hip 的 3D）。
2. **加腿間耦合項** `θ̇_i = 2π·ω_i + ½Σ_j (r_j) w_ij sin(θ_j − θ_i − φ_ij)`，取代固定相位偏移，讓步態在擾動下自我協調、能學不同步態。
3. **加地形高度觀測**（17×11 grid）+ Playground 的崎嶇地形，往 perceptive locomotion 走。
4. **domain randomization 加重**（摩擦、質量、延遲、外力），縮小 sim-to-real gap，為日後上真機鋪路。

---

## 名詞速查

- **MJX**：JAX 版 MuJoCo，能在 GPU 上平行跑幾千個模擬，RL 訓練用。
- **JAX**：Google 的數值庫；要求純函數、固定形狀、陣列不可就地改。
- **Brax**：Google 的 JAX RL 庫，提供本指南用的 PPO。
- **MuJoCo Playground**：一堆現成機器人 RL 環境（含 Go1/Go2 走路），我們拿來當模板。
- **Hopf 振盪器**：一種有「振幅 r、相位 θ」的非線性振盪器，CPG 的數學核心。
- **μ（mu）/ ω（omega）**：RL 要調的兩個旋鈕——振幅目標、頻率。
- **sim-to-sim**：Colab(MJX) 訓好的策略，搬到本機(一般 MuJoCo) 行為要一致。
- **sim-to-real**：再搬到真機還能動；本專案暫不做，但 DR 為它鋪路。

---

## 參考連結

- CPG-RL 原論文（振盪器/動作/觀測）：https://arxiv.org/abs/2211.00458
- Visual CPG-RL（含完整方程式 HTML）：https://arxiv.org/html/2212.14400v2
- MuJoCo Playground（模板環境、訓練 pipeline）：https://github.com/google-deepmind/mujoco_playground
- Playground 建立自訂環境文件：https://deepwiki.com/google-deepmind/mujoco_playground/6.1-creating-custom-environments
- Playground Brax PPO 訓練文件：https://deepwiki.com/google-deepmind/mujoco_playground/4.1-jax-ppo-training
- Playground locomotion 訓練 notebook：https://github.com/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb
- 你自己的 CPG：`task3/go2_gait.py`、`task3/go2_model.py`、`task3/walk_line.py`
```
