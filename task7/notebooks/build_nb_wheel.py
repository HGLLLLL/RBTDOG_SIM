#!/usr/bin/env python3
"""產生純 RL 自訓的 Colab notebook：`build(variant)` 回傳 v1 或 v1c；`main()` 同時寫 `rl_wheel_v1_colab.ipynb` 與
`rl_wheel_v1c_colab.ipynb`（env：task7/inference/rl_env_wheel.py，產生器本身不 import 它）。
安裝／版本兩格沿用 cpg_rl_max_colab.ipynb（與 build_nb_v3.py 同做法）。用法：python3 task7/notebooks/build_nb_wheel.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rl_wheel_v1_colab.ipynb"
WEIGHTS = "rl_wheel_v1_params.pkl"
OUT_C = HERE / "rl_wheel_v1c_colab.ipynb"
WEIGHTS_C = "rl_wheel_v1c_params.pkl"
OUT_D = HERE / "rl_wheel_v1d_colab.ipynb"
WEIGHTS_D = "rl_wheel_v1d_params.pkl"
OUT_F = HERE / "rl_wheel_v1f_colab.ipynb"
WEIGHTS_F = "rl_wheel_v1f_params.pkl"
OUT_G = HERE / "rl_wheel_v1g_colab.ipynb"
WEIGHTS_G = "rl_wheel_v1g_params.pkl"
OUT_H = HERE / "rl_wheel_v1h_colab.ipynb"
WEIGHTS_H = "rl_wheel_v1h_params.pkl"
OUT_I = HERE / "rl_wheel_v1i_colab.ipynb"
WEIGHTS_I = "rl_wheel_v1i_params.pkl"
OLD = json.loads((HERE / "cpg_rl_max_colab.ipynb").read_text(encoding="utf-8"))
_code = [c for c in OLD["cells"] if c["cell_type"] == "code"]
install_src, version_src = "".join(_code[0]["source"]), "".join(_code[1]["source"])
assert "brax==0.14.2" in install_src and 'assert brax.__version__ == "0.14.2"' in version_src

md0 = """# 純 RL 自訓 **v1**：智元 D1 Max · 同事 Wheel-Rough 環境的 MJX 重做 ＋ 三項改動 ＋ 低階樓梯 · Colab GPU（2026-10-13）

在同事的純 RL 第二版（Isaac Lab，10k，trip25 上機 11 趟 0 摔）之上，用我們的 MJX + Brax PPO 線重訓（spec `docs/superpowers/specs/2026-10-13-rl-wheel-v1-design.md`）：
- obs 53／動作 16／reward 十項照他的 env.yaml；地形 rough／boxes／**stairs**（階高 3–8 cm）三列 × 十級課程。
- critic 用特權 obs：無雜訊的同樣 53 ＋ 機身線速度 3（同他的 critic 56 維；policy 仍看帶雜訊的 53，匯出只取 policy）；discount 0.99（他的設定）。
- 改動一 自碰撞：輪輪 pair ＋ 實體間隙 hinge（`wgap`）＋ 接觸力（`ww`，只擋腿交叉）。改動二 小指令：追蹤 std 0.5 → 0.25、30% 回合小指令。改動三 輪摩擦：real 關節摩擦 DR。
- 開訓前關卡已過：`outputs/rl_wheel/align_10k.md`（他的 10k 在這個環境零訓練重現）、`outputs/reward_audit_wheel_v1.md`。

**流程**：第一格 throughput 測試選 `NUM_ENVS` → 訓練（每次 eval 存 pkl 到 `/content/rl_wheel_v1/`，**不用雲端硬碟**；runtime 斷線檔案就沒了，所以每個 session 跑完立刻從左側檔案面板下載 `rl_wheel_v1_params.pkl` 與最新的 `rl_wheel_v1_step*.pkl`）→ 一個 session 跑 5 億步（G4、4096 環境約 50 分鐘）→ 下個 session 把上一輪的 `rl_wheel_v1_step*.pkl` 上傳回 `/content/rl_wheel_v1/`，`RESUME` 指到它（累計步數由檔名自動解析）。G4 GPU（A100 慢 2.6 倍、TPU 不能用）。
**續訓注意**：只還原網路與 normalizer，Adam 動量與 ADAPTIVE_KL 學習率會重置，每個 session 開頭 reward 可能短暫抖動，不是退步。
**記憶體**：throughput 格的 10 億步 ETA 是下限（純 step 速率）；訓練格 OOM 就手動把 `NUM_ENVS` 降一檔。
**停損**：累計 2 億步時 `len < 600`（一直摔）、或 `lin < 0.5`（站著不動）、或 `wgap` 沒比 step 0 好（往 0 走）／`gap` 中位 < 40 mm（改動一沒被學到；`ww` 平地恆 0 不能看）→ 停，不硬跑完。
**讀法**：`lin`／`ang` 往 2.5／2.5 走；`lim`（PD 目標超軟限位，v1b 新加）與 `sat`（目標超硬限位的關節數）往 0；`wgap`、`ww`、`slip` 往 0；`level` 往上代表課程在升；`lin_small` 是小指令回合的追蹤；`fell` 往 0；len 往 1000。
"""

clone_src = '''import os, subprocess, sys

REPO = "https://github.com/HGLLLLL/RBTDOG_SIM.git"
BRANCH = "main"
DEST = "rbtdog_sim"
if not os.path.exists(DEST):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, DEST], check=True)
sys.path.insert(0, f"{DEST}/task7/inference")
print("clone 到的 commit：",
      subprocess.run(["git", "-C", DEST, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
'''

import_src = '''import jax, jax.numpy as jnp, numpy as np, mujoco, time
import rl_env_wheel as rw
env = rw.WheelEnv()
print("obs", env.observation_size, "act", env.action_size, "Z_STAND", round(env.Z_STAND, 3), "scene", env.scene)
assert env.observation_size == {"state": (53,), "privileged_state": (56,)} and env.action_size == 16
# 煙霧：4 個環境零動作 20 步，不該摔
w = rw.wrap_env(env, episode_length=rw.EPISODE_LEN)
s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), 4)); step = jax.jit(w.step)
for _ in range(20):
    s = step(s, jnp.zeros((4, 16)))
assert s.obs["state"].shape == (4, 53) and s.obs["privileged_state"].shape == (4, 56)
assert float(np.asarray(s.metrics["fell"]).max()) == 0.0, "零動作就摔：出生高或地形有問題"
print("煙霧 OK；平均 reward", float(np.asarray(s.reward).mean()))
'''

bench_src = '''# ---- throughput 測試（spec §5）：選最大且比前一檔至少快 1.3 倍的環境數（G4 實測 1024 94k／2048 145k／4096 211k）；記憶體爆就退一檔
import subprocess
bench = {}
for NUM_ENVS in (1024, 2048, 4096):
    try:
        w = rw.wrap_env(env, episode_length=rw.EPISODE_LEN)
        s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)); step = jax.jit(w.step)
        a = jnp.zeros((NUM_ENVS, 16))
        s = step(s, a); s.obs["state"].block_until_ready()                                  # 編譯
        t0 = time.time()
        for _ in range(200):
            s = step(s, a)
        s.obs["state"].block_until_ready(); dt = time.time() - t0
        bench[NUM_ENVS] = NUM_ENVS * 200 / dt
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"{NUM_ENVS} envs: {bench[NUM_ENVS]:,.0f} steps/s  GPU {mem}")
    except Exception as e:                                                          # 多半是 OOM
        print(f"{NUM_ENVS} envs: 失敗 {type(e).__name__}: {str(e)[:120]}"); break
s = w = step = a = None; del s, w, step, a                                    # 釋放 bench 的環境狀態，免得佔住 GPU 記憶體
NUM_ENVS = 1024
for n in (2048, 4096):                                                          # 遇到第一檔不合 1.3 倍就停，不跳級
    if n in bench and bench[n] >= 1.3 * bench[n // 2]:
        NUM_ENVS = n
    else:
        break
print("→ NUM_ENVS =", NUM_ENVS, "| 10 億步預估", f"{1e9 / bench[NUM_ENVS] / 3600:.1f} 小時（下限：純 step 速率，不含 PPO 更新與 eval）")
'''

train_src = f'''import functools, time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
DRIVE = "/content/rl_wheel_v1"; import os, re, glob; os.makedirs(DRIVE, exist_ok=True)   # 本機暫存（使用者決定不用雲端硬碟）；session 結束前自己下載

# 續訓：把上一輪下載的 rl_wheel_v1_step*.pkl 上傳到 /content/rl_wheel_v1/，RESUME 指到「檔名步數最大」的那個（每次 eval 存一個；(normalizer, policy, value) 三元組）。
# 不要指 rl_wheel_v1_params.pkl：它檔名沒有步數。None ＝ 從頭。
# brax 的 restore_params 不還原步數，STEP_OFFSET 從 RESUME 檔名解析累計步數，存檔名與進度列印都用累計步數；停損線「2 億步」是累計步數。
# TIMESTEPS 是「這一輪再跑幾步」；一個 session 5 億（v1 用了 5 億；v1b 從 v1 的 step 檔續訓再 5 億）。
# RESUME = "auto"：自動在 DRIVE 裡找步數最大的 rl_wheel_v1_step*.pkl（把上一輪下載的檔上傳進去就會接著訓）；None ＝ 強制從頭；也可直接給路徑。
RESUME = "auto"
if RESUME == "auto":
    _cands = sorted(glob.glob(f"{{DRIVE}}/rl_wheel_v1_step*.pkl"), key=lambda p_: int(re.search(r"step(\\d+)", p_).group(1)))
    RESUME = _cands[-1] if _cands else None
print("RESUME =", RESUME if RESUME else "None（從頭訓）")
_restore = None if RESUME is None else model.load_params(RESUME)
def _step_offset(path):
    """續訓：從 RESUME 檔名解析累計步數；檔名沒有 step<N> 就報錯（不要靜默從 0 算）。"""
    m_ = re.search(r"step(\\d+)", path)
    if m_ is None:
        raise ValueError("RESUME 要指到 rl_wheel_v1_step<N>.pkl，不是 rl_wheel_v1_params.pkl")
    return int(m_.group(1))
STEP_OFFSET = 0 if RESUME is None else _step_offset(RESUME)
TIMESTEPS = 500_000_000

env = rw.WheelEnv()
eval_env = rw.WheelEnv(noise=False, push=False)
network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                    policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
                                    policy_obs_key="state", value_obs_key="privileged_state")   # critic 特權 obs 56
_t0 = time.time(); rewards = []

def _save_ckpt(step, make_policy, params):
    model.save_params(f"{{DRIVE}}/rl_wheel_v1_step{{STEP_OFFSET + step:012d}}.pkl", params)

def progress(step, metrics):
    r = float(metrics.get("eval/episode_reward", 0.0)); total = STEP_OFFSET + step; rewards.append((total, r))   # 累計步數
    L = float(metrics.get("eval/avg_episode_length", 1.0)) or 1.0
    ps = lambda k: float(metrics.get(f"eval/episode_{{k}}", 0.0)) / L
    el = time.time() - _t0; rate = step / max(el, 1e-9)                         # 本 session 速率（ETA 用）
    print(f"step {{total:>11,}} R {{r:7.2f}} | lin {{ps('lin'):.2f}}/2.5 ang {{ps('ang'):.2f}}/2.5 small {{ps('lin_small'):.2f}} | "
          f"ww {{ps('ww'):+.3f}} wgap {{ps('wgap'):+.3f}} gap {{ps('gap_min')*1000:.0f}}mm slip {{ps('slip'):+.3f}} dev {{ps('dev'):+.3f}} rate {{ps('rate'):+.3f}} lim {{ps('lim'):+.3f}} sat {{ps('sat_n'):.2f}} | "
          f"vxerr {{ps('vx_err'):.2f}} vyerr {{ps('vy_err'):.2f}} wzerr {{ps('wz_err'):.2f}} | level {{ps('level'):.1f}} fell {{ps('fell') * L:.2f}} len {{L:.0f}} | "
          f"{{el:.0f}}s → {{TIMESTEPS / max(rate, 1) / 60:.0f}} 分")

train_fn = functools.partial(
    ppo.train, num_timesteps=TIMESTEPS, num_evals=20, episode_length=rw.EPISODE_LEN,
    num_envs=NUM_ENVS, batch_size=256 * NUM_ENVS // 2048, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.99, normalize_observations=True,
    learning_rate_schedule="ADAPTIVE_KL", desired_kl=0.01, learning_rate_schedule_min_lr=1e-5, learning_rate_schedule_max_lr=1e-3,
    network_factory=network_factory, randomization_fn=rw.make_domain_randomize(env), wrap_env_fn=rw.wrap_env,
    seed=0, eval_env=eval_env, policy_params_fn=_save_ckpt, restore_params=_restore)
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print("training done")
'''

plot_src = '''import matplotlib.pyplot as plt
plt.plot([s for s, _ in rewards], [r for _, r in rewards], marker="o"); plt.xlabel("env steps"); plt.ylabel("eval reward"); plt.grid(True); plt.show()
'''

save_src = f'''model.save_params("{WEIGHTS}", params)
import shutil; shutil.copy("{WEIGHTS}", f"{{DRIVE}}/{WEIGHTS}")
print("已存 {WEIGHTS}（/content 與 /content/rl_wheel_v1/）→ ⚠️ 立刻從檔案面板下載，runtime 斷線就沒了 → 放 task7/weights/；本機：export_policy_wheel.py → rl_only_infer.py --npz … --self-collision")
'''


md0_c = """# 純 RL 自訓 **v1c**：左右鏡像增強 ＋ roll 姿態角懲罰 ＋ 落地衝擊懲罰 ＋ 動作延遲／IMU 偏差 DR · Colab GPU（2026-10-20）

從頭訓（不從 v1b 續），環境 `WheelEnv(**rw.V1C)`（spec `docs/superpowers/specs/2026-10-20-rl-wheel-v1c-design.md`；v1b 的 reward 盤點 `docs/results/Y_v1b_reward總表_2026-10-20.md`）：
- **鏡像增強**：每回合 50% 把 obs 左右翻給 policy、動作翻回物理 → policy 本身左右對稱（v1b 左平移持續側傾 7–10°、右 1–4°，三版 policy 歪的方向隨機）。
- **`roll`**：罰真實重力 y 分量平方（w 10，roll 9° ≈ −0.25／步）；原本的 `angxy` 只罰角速度，固定側傾時是 0。
- **`impact`**：罰觸地前一步足端下落速度平方（w 3）；原本沒有任何接觸力項。
- **DR**：動作延遲每回合抽 0–2 步、IMU roll／pitch 常數偏差每回合抽 ±2°（只動 policy 的重力 obs）。
- 不動：`wgap`／`ww`／`lim`／`w_ang`／σ 0.25／`acc_leg`／`rate`／課程／推力／網路。狗端零改動，`--grav-offset 0.7 -1.5` 照舊。
- 開訓前關卡已過：`outputs/reward_audit_wheel_{10k,v1b}_v1c.md`（`impact` 10k < v1b 一半、`roll` 左 > 右）。

**流程**：第 4 格掛載雲端硬碟（會跳一次授權，按完就可以離開）→ throughput 測試選 `NUM_ENVS` → 訓練（每次 eval 存 pkl 到 `/content/rl_wheel_v1c/` **並自動複製到雲端硬碟 `MyDrive/rl_wheel_v1c/`**，人不在、runtime 斷線都不會丟）→ 一個 session 5 億步（G4、4096 環境約 50 分鐘）→ 下個 session 掛載後 `RESUME="auto"` 自動從雲端硬碟接最新的 step 檔，不用手動上傳。
**5 億步中場檢查**：session 1 結束把 step 檔拉回本機 `export_policy_wheel.py` → 跑 spec §6 全表；roll 左右差沒縮、或驗收表的「落地 vz」沒比 v1b 低 → 先調權重再跑 session 2；Colab 進度列的 `land_vz` 是 MJX 接觸口徑（v1b 左／右平移 0.43／0.70），本機驗收表的「落地 vz」是 CPU 淨空口徑（v1b 0.36／0.37），**兩者不能互比**，中場判讀以本機驗收表前後比為準。
**停損**：累計 2 億步時 `len < 600`、或 `lin < 0.5`、或 `wgap` 沒比 step 0 好／`gap` 中位 < 40 mm、或 `impact`、`roll` 沒比 step 0 往 0 走、或 `level` 明顯低於 v1b 同步數的曲線（崎嶇地 `impact` 量級大：樓梯 8 側移 −0.75／步、方格 8 −0.48，平地 −0.17；policy 可能逃避高等級地形）→ 停。
**讀法**：`lin`／`ang` 往 2.5／2.5；`roll`、`impact` 往 0；`land_vz` ＝ 每次落地平均下落速度 m/s（Σvz／Σ落地次數，不是除步數；MJX 口徑，v1b 平移 0.5 左／右 0.43／0.70，與本機驗收表的 0.36 不同口徑），往下；`mir` 應在 0.5 附近、`d` 在 1.0 附近（0–2 平均）；eval 也開鏡像／延遲／偏差，所以 eval reward 不是乾淨名目條件的數字，中場判讀以本機驗收表為準；其餘同 v1。
"""
mount_src_c = '''# ---- 雲端硬碟（v1c，使用者 2026-10-20 要求：按下去就能離開）：先授權掛載；每次 eval 的 step 檔與最後的 params 都自動複製到 MyDrive/rl_wheel_v1c/，續訓也從那裡自動找
from google.colab import drive
drive.mount("/content/drive")
import os
GDRIVE = "/content/drive/MyDrive/rl_wheel_v1c"; os.makedirs(GDRIVE, exist_ok=True)
print("雲端硬碟 OK →", GDRIVE, "｜已有：", sorted(os.listdir(GDRIVE)))
'''
import_src_c = import_src.replace("env = rw.WheelEnv()", "env = rw.WheelEnv(**rw.V1C)")
train_src_c = (train_src.replace("rl_wheel_v1", "rl_wheel_v1c")
               .replace("（v1 用了 5 億；v1b 從 v1 的 step 檔續訓再 5 億）", "（v1c 從頭訓，兩個 session 各 5 億）")
               .replace("env = rw.WheelEnv()\neval_env = rw.WheelEnv(noise=False, push=False)",
                        "env = rw.WheelEnv(**rw.V1C)\neval_env = rw.WheelEnv(noise=False, push=False, **rw.V1C)")
               .replace("    ps = lambda k: float(metrics.get(f\"eval/episode_{k}\", 0.0)) / L\n",
                        "    ps = lambda k: float(metrics.get(f\"eval/episode_{k}\", 0.0)) / L\n"
                        "    ev = lambda k: float(metrics.get(f\"eval/episode_{k}\", 0.0))                  # 回合總和（不除步數）\n"
                        "    land_vz = ev('impact_vz') / max(ev('td_n'), 1e-9)                              # 每次落地平均下落速度 m/s ＝ Σvz／Σ落地次數（除步數會被無落地的步稀釋）\n")
               .replace("# 本機暫存（使用者決定不用雲端硬碟）；session 結束前自己下載", "# 本機暫存；每個 step 檔另外複製到 GDRIVE（雲端硬碟，上一格掛載），斷線也不丟")
               .replace('glob.glob(f"{DRIVE}/rl_wheel_v1c_step*.pkl")', 'glob.glob(f"{DRIVE}/rl_wheel_v1c_step*.pkl") + glob.glob(f"{GDRIVE}/rl_wheel_v1c_step*.pkl")')
               .replace('    model.save_params(f"{DRIVE}/rl_wheel_v1c_step{STEP_OFFSET + step:012d}.pkl", params)',
                        '    p_ = f"{DRIVE}/rl_wheel_v1c_step{STEP_OFFSET + step:012d}.pkl"; model.save_params(p_, params)\n'
                        '    import shutil; shutil.copy(p_, GDRIVE)                     # 自動上傳雲端硬碟（每次 eval 一份，runtime 斷線也不丟）')
               .replace("lim {ps('lim'):+.3f} sat {ps('sat_n'):.2f} | ",
                        "lim {ps('lim'):+.3f} sat {ps('sat_n'):.2f} | roll {ps('roll'):+.3f} impact {ps('impact'):+.3f} land_vz {land_vz:.2f} mir {ps('mir'):.2f} d {ps('delay'):.1f} | "))
assert "rl_wheel_v1_step" not in train_src_c and "rw.V1C" in train_src_c and "ev('td_n')" in train_src_c and "land_vz" in train_src_c
assert "shutil.copy(p_, GDRIVE)" in train_src_c and 'glob.glob(f"{GDRIVE}/rl_wheel_v1c_step*.pkl")' in train_src_c and "不用雲端硬碟" not in train_src_c
save_src_c = (save_src.replace(WEIGHTS, WEIGHTS_C).replace("/content/rl_wheel_v1/", "/content/rl_wheel_v1c/")
              .replace('import shutil; shutil.copy("rl_wheel_v1c_params.pkl", f"{DRIVE}/rl_wheel_v1c_params.pkl")',
                       'import shutil; shutil.copy("rl_wheel_v1c_params.pkl", f"{DRIVE}/rl_wheel_v1c_params.pkl"); shutil.copy("rl_wheel_v1c_params.pkl", f"{GDRIVE}/rl_wheel_v1c_params.pkl")')
              .replace("→ ⚠️ 立刻從檔案面板下載，runtime 斷線就沒了 → 放 task7/weights/", "→ ✅ 已自動上傳雲端硬碟 MyDrive/rl_wheel_v1c/（step 檔每次 eval 也都有副本），回來從那裡拿 → 放 task7/weights/"))
assert "GDRIVE" in save_src_c and "立刻從檔案面板下載" not in save_src_c


# ---- v1d（2026-10-20 晚）：v1c 從頭訓 5 億步後 policy 完全不抬腳、側向軸死掉（lin 0.04 vs v1b 2.40；impact 把早期探索殺掉）→
#      改成 **從 v1b 熱啟動** 續訓，reward／DR 同 V1C 但 impact 改門檻形式（impact_v0 0.3）。檔名與雲端硬碟資料夾用 v1d，避免 RESUME 撿到 v1c 的 step 檔。
md0_d = md0_c.replace("# 純 RL 自訓 **v1c**：", "# 純 RL 自訓 **v1d**：v1b 熱啟動 ＋ ").replace("（2026-10-20）", "（2026-10-20 晚）").replace(
    "從頭訓（不從 v1b 續），環境 `WheelEnv(**rw.V1C)`",
    "**從 v1b 熱啟動**（`INIT_FROM`，normalizer／policy／value 都還原，步數從 0 算）續訓；v1c 從頭訓 5 億步的結果是 policy 完全不抬腳、側向軸死掉（`outputs/rl_wheel/eval_rl_only_real_v1c_s1.md`、`outputs/reward_audit_wheel_v1c_s1.md`：平移 0.5 的 lin 0.04 vs v1b 2.40），判斷是 `impact` 把早期笨拙落地的探索殺掉。環境 `WheelEnv(**rw.V1C)`").replace(
    "- **`impact`**：罰觸地前一步足端下落速度平方（w 3）；原本沒有任何接觸力項。",
    "- **`impact`**：罰觸地前一步足端下落速度**超過 0.3 m/s 的部分**的平方（w 3，`impact_v0` 0.3；10k 等級 0.10–0.19 免費、v1b 等級 0.43–0.71 仍罰）；原本沒有任何接觸力項。").replace(
    "rl_wheel_v1c", "rl_wheel_v1d").replace("MyDrive/rl_wheel_v1c/", "MyDrive/rl_wheel_v1d/").replace("/content/rl_wheel_v1c/", "/content/rl_wheel_v1d/")
md0_d = md0_d.replace("一個 session 5 億步（G4、4096 環境約 50 分鐘）", "一個 session 5 億步（G4、4096 環境約 50 分鐘；使用者：有必要可以多跑幾個 session）")
assert "v1b 熱啟動" in md0_d and "rl_wheel_v1c" not in md0_d
mount_src_d = mount_src_c.replace("rl_wheel_v1c", "rl_wheel_v1d").replace("（v1c，", "（v1d，")
train_src_d = (train_src_c.replace("rl_wheel_v1c", "rl_wheel_v1d")
               .replace('_restore = None if RESUME is None else model.load_params(RESUME)\n',
                        '_restore = None if RESUME is None else model.load_params(RESUME)\n'
                        'INIT_FROM = "rbtdog_sim/task7/weights/rl_wheel_v1b_params.pkl"     # v1d：沒有 v1d 的 step 檔時從 v1b 熱啟動（normalizer＋policy＋value 都還原；累計步數從 0 算）\n'
                        'if _restore is None and INIT_FROM:\n'
                        '    _restore = model.load_params(INIT_FROM); print("熱啟動自", INIT_FROM)\n')
               .replace("（v1c 從頭訓，兩個 session 各 5 億）", "（v1d 從 v1b 熱啟動，每 session 5 億，可多跑）"))
assert "INIT_FROM" in train_src_d and "rl_wheel_v1c" not in train_src_d and 'print("熱啟動自", INIT_FROM)' in train_src_d
save_src_d = save_src_c.replace("rl_wheel_v1c", "rl_wheel_v1d")
assert "rl_wheel_v1d_params.pkl" in save_src_d and "rl_wheel_v1c" not in save_src_d


# ---- v1f（2026-10-21，使用者定）：v1d 平移腿伸直（膝 42°）、前後晃 3 倍、直走超速 +0.1 → 拿掉延遲 DR（v1b 沒有也走得好），
#      加 w_angxy 0.5（罰 roll／pitch 角速度）與膝 hinge（比 60° 更直才罰）。其餘同 v1d：v1b 熱啟動、鏡像、roll、impact 門檻、IMU 偏差 DR。
md0_f = md0_d.replace("# 純 RL 自訓 **v1d**：v1b 熱啟動 ＋ ", "# 純 RL 自訓 **v1f**：v1b 熱啟動 ＋ ").replace("（2026-10-20 晚）", "（2026-10-21）").replace(
    "環境 `WheelEnv(**rw.V1C)`", "環境 `WheelEnv(**rw.V1F)`").replace(
    "- **DR**：動作延遲每回合抽 0–2 步、IMU roll／pitch 常數偏差每回合抽 ±2°（只動 policy 的重力 obs）。",
    "- **DR**：IMU roll／pitch 常數偏差每回合抽 ±2°（只動 policy 的重力 obs）。**不加動作延遲 DR**：v1d（有延遲 DR）平移時膝伸到 42°、機身 0.59 m、pitch 晃 3 倍、直走超速 +0.1，v1b 沒有延遲 DR 實機 24 趟走得好（使用者 10-21 定）。\n"
    "- **防「用伸直腿、前後晃躲側傾」**：`angxy` 0.05 → **0.5**（罰 roll／pitch **角速度**，樓梯上的靜態俯仰不受罰）；新項 `knee`（w 0.5）罰膝比 **60°** 更直的部分（v1b 平移膝 64–80° 不罰、v1d 的 42° 罰）。").replace(
    "rl_wheel_v1d", "rl_wheel_v1f").replace("MyDrive/rl_wheel_v1d/", "MyDrive/rl_wheel_v1f/").replace("/content/rl_wheel_v1d/", "/content/rl_wheel_v1f/")
md0_f = md0_f.replace("**讀法**：", "**讀法**：`knee` 往 0（v1b 起點應 ≈ 0）、`angxy` 看有沒有比 v1b 起點（約 −0.1）大很多；")
assert "rw.V1F" in md0_f and "rl_wheel_v1d" not in md0_f and "不加動作延遲 DR" in md0_f and "`knee`" in md0_f
mount_src_f = mount_src_d.replace("rl_wheel_v1d", "rl_wheel_v1f").replace("（v1d，", "（v1f，")
import_src_f = import_src_c.replace("rw.V1C", "rw.V1F")
train_src_f = (train_src_d.replace("rl_wheel_v1d", "rl_wheel_v1f").replace("rw.V1C", "rw.V1F")
               .replace("（v1d 從 v1b 熱啟動，每 session 5 億，可多跑）", "（v1f 從 v1b 熱啟動，每 session 5 億，可多跑）")
               .replace("roll {ps('roll'):+.3f} impact {ps('impact'):+.3f}", "roll {ps('roll'):+.3f} impact {ps('impact'):+.3f} knee {ps('knee'):+.3f} angxy {ps('angxy'):+.3f}"))
assert "rw.V1F" in train_src_f and "rl_wheel_v1d" not in train_src_f and "ps('knee')" in train_src_f and "INIT_FROM" in train_src_f
save_src_f = save_src_d.replace("rl_wheel_v1d", "rl_wheel_v1f")


# ---- v1g（2026-10-21，使用者定）：v1f 平移＝兩腿踏、兩腿被拖著橫滑 0.5 m/s（直腿外張是拖腿的幾何必然）→ 只改 w_slip 0.075 → 0.75，其餘同 v1f。
md0_g = md0_f.replace("# 純 RL 自訓 **v1f**：", "# 純 RL 自訓 **v1g**：").replace("環境 `WheelEnv(**rw.V1F)`", "環境 `WheelEnv(**rw.V1G)`").replace(
    "- **防「用伸直腿、前後晃躲側傾」**：",
    "- **防「拖腿橫滑」（v1g 的改動）**：`slip` 0.075 → **0.75**，加新項 **`stuck`**（w 1.0）：側移或旋轉指令下，任一腿足端淨空 < 5 mm 連續超過 0.6 s 就罰（超過秒數、上限 1 s）；直走、站立不罰。v1f 平移時只有兩腿踏、另外兩腿全程觸地被拖著橫滑 0.5 m/s（每步只扣 0.04，比抬腳＋重心轉移便宜），直腿外張是拖腿的幾何必然；原廠平移四腿都踏、次要腿每週期原地重放 8–10 mm（L 文件）→ 不觸發 stuck。只調 slip ×10 的關卡沒過（vy 0.3 v1f 仍勝）才加 stuck。攤帳關卡：V1G 下 v1b 四腿步態每步總分要高於 v1f 拖腿步態。\n"
    "- **防「用伸直腿、前後晃躲側傾」**：").replace("rl_wheel_v1f", "rl_wheel_v1g").replace("MyDrive/rl_wheel_v1f/", "MyDrive/rl_wheel_v1g/").replace("/content/rl_wheel_v1f/", "/content/rl_wheel_v1g/")
md0_g = md0_g.replace("**讀法**：", "**讀法**：`stuck` 起點（v1b）應 ≈ 0、全程 ≈ 0（變負＝有腿被拖）；`slip` 起點約 −0.1，往 0；")
assert "rw.V1G" in md0_g and "rl_wheel_v1f" not in md0_g and "拖腿橫滑" in md0_g
mount_src_g = mount_src_f.replace("rl_wheel_v1f", "rl_wheel_v1g").replace("（v1f，", "（v1g，")
import_src_g = import_src_f.replace("rw.V1F", "rw.V1G")
train_src_g = (train_src_f.replace("rl_wheel_v1f", "rl_wheel_v1g").replace("rw.V1F", "rw.V1G")
               .replace("（v1f 從 v1b 熱啟動，每 session 5 億，可多跑）", "（v1g 從 v1b 熱啟動，每 session 5 億，可多跑）")
               .replace("knee {ps('knee'):+.3f} angxy {ps('angxy'):+.3f}", "knee {ps('knee'):+.3f} angxy {ps('angxy'):+.3f} stuck {ps('stuck'):+.3f} slip {ps('slip'):+.3f}"))
assert "ps('stuck')" in train_src_g and "rw.V1G" in train_src_g and "rl_wheel_v1f" not in train_src_g and "INIT_FROM" in train_src_g and "ps('slip')" in train_src_g
save_src_g = save_src_f.replace("rl_wheel_v1f", "rl_wheel_v1g")


# ---- v1h（2026-10-21 下午，使用者定「1＋2」）：v1g 中場平移膝 50–58°（比 hinge 60° 更直）、機身 0.57 m；動機＝roll 懲罰在地形上逼它站直。
#      1 直接鎖姿態：膝 hinge 65°／w 2、機身高 hinge（> 0.56 m 線性 w 10）；2 減壓：w_roll 10 → 5。其餘同 v1g。
md0_h = md0_g.replace("# 純 RL 自訓 **v1g**：", "# 純 RL 自訓 **v1h**：").replace("環境 `WheelEnv(**rw.V1G)`", "環境 `WheelEnv(**rw.V1H)`").replace(
    "- **防「用伸直腿、前後晃躲側傾」**：",
    "- **鎖姿態（v1h 的改動，使用者定「1＋2」）**：v1g 中場平移膝 50–58°（比 hinge 60° 更直、每步寧願付 knee −0.24）、機身 0.57 m（v1b 0.53），動機是 roll 懲罰在地形上逼它站直（直腿垂直剛度高）。"
    "1 直接鎖：膝 hinge 60° → **65°**（70° 會讓 v1b 自己平地扣 −0.4、樓梯 −0.6，關卡 12 列輸 2 列）、w 0.5 → **2.0**；新項 **`height`**（w 10）罰機身對地形高度超過 **0.56 m** 的部分（v1b 平移 0.53–0.56）。2 減壓：`roll` 10 → **5**。關卡加地形列（rough／boxes／stairs 第 9 級側移），v1b 要每列都贏。\n"
    "- **防「用伸直腿、前後晃躲側傾」**：").replace("rl_wheel_v1g", "rl_wheel_v1h").replace("MyDrive/rl_wheel_v1g/", "MyDrive/rl_wheel_v1h/").replace("/content/rl_wheel_v1g/", "/content/rl_wheel_v1h/")
md0_h = md0_h.replace("**讀法**：", "**讀法**：`height` 往 0（v1b 起點平移時 ≈ 0）、`knee` 起點（v1b 在 65° 地板下）約 −0.1 往 0；")
assert "rw.V1H" in md0_h and "rl_wheel_v1g" not in md0_h and "`height`" in md0_h
mount_src_h = mount_src_g.replace("rl_wheel_v1g", "rl_wheel_v1h").replace("（v1g，", "（v1h，")
import_src_h = import_src_g.replace("rw.V1G", "rw.V1H")
train_src_h = (train_src_g.replace("rl_wheel_v1g", "rl_wheel_v1h").replace("rw.V1G", "rw.V1H")
               .replace("（v1g 從 v1b 熱啟動，每 session 5 億，可多跑）", "（v1h 從 v1b 熱啟動，每 session 5 億，可多跑）")
               .replace("stuck {ps('stuck'):+.3f} slip {ps('slip'):+.3f}", "stuck {ps('stuck'):+.3f} slip {ps('slip'):+.3f} height {ps('height'):+.3f}"))
assert "ps('height')" in train_src_h and "rw.V1H" in train_src_h and "rl_wheel_v1g" not in train_src_h and "INIT_FROM" in train_src_h
save_src_h = save_src_g.replace("rl_wheel_v1g", "rl_wheel_v1h")


# ---- v1i（2026-10-21 傍晚，使用者定）：v1h 平移時前腳在髖前 14–21 cm、後腳在髖後 9–17 cm（前後腳距 95–98 cm，v1b 60–75）——站高／鎖膝／拖腿都封掉後
#      policy 改用「拉長支撐面」讓踏步時機身不晃。加 `splay`：腳相對髖的機身 x 距離 > 8 cm 才罰（線性 w 3）。其餘同 v1h。
md0_i = md0_h.replace("# 純 RL 自訓 **v1h**：", "# 純 RL 自訓 **v1i**：").replace("環境 `WheelEnv(**rw.V1H)`", "環境 `WheelEnv(**rw.V1I)`").replace(
    "- **鎖姿態（v1h 的改動，使用者定「1＋2」）**：",
    "- **腳要在髖下方（v1i 的改動，使用者定）**：v1h 平移時髖角從 ±46° 開到 7–19°，前腳在髖前 14–21 cm、後腳在髖後 9–17 cm，前後腳距 95–98 cm（v1b 60–75）；站高、鎖膝、拖腿都被封後，policy 用拉長支撐面讓踏步時機身不晃，髖開＋膝 65° 的髖到腳距離仍長＝繞過膝 hinge。新項 **`splay`**（w 3）罰腳相對髖在機身前後方向超過 **8 cm** 的部分（v1b 平移 ±8 cm 內、站立 ±1 cm）。關卡：V1I 下 v1b 要贏 v1h。\n"
    "- **鎖姿態（v1h 的改動，使用者定「1＋2」）**：").replace("rl_wheel_v1h", "rl_wheel_v1i").replace("MyDrive/rl_wheel_v1h/", "MyDrive/rl_wheel_v1i/").replace("/content/rl_wheel_v1h/", "/content/rl_wheel_v1i/")
md0_i = md0_i.replace("**讀法**：", "**讀法**：`splay` 起點（v1b）≈ 0、往 0；")
assert "rw.V1I" in md0_i and "rl_wheel_v1h" not in md0_i and "`splay`" in md0_i
mount_src_i = mount_src_h.replace("rl_wheel_v1h", "rl_wheel_v1i").replace("（v1h，", "（v1i，")
import_src_i = import_src_h.replace("rw.V1H", "rw.V1I")
train_src_i = (train_src_h.replace("rl_wheel_v1h", "rl_wheel_v1i").replace("rw.V1H", "rw.V1I")
               .replace("（v1h 從 v1b 熱啟動，每 session 5 億，可多跑）", "（v1i 從 v1b 熱啟動，每 session 5 億，可多跑）")
               .replace("height {ps('height'):+.3f}", "height {ps('height'):+.3f} splay {ps('splay'):+.3f}"))
assert "ps('splay')" in train_src_i and "rw.V1I" in train_src_i and "rl_wheel_v1h" not in train_src_i and "INIT_FROM" in train_src_i
save_src_i = save_src_h.replace("rl_wheel_v1h", "rl_wheel_v1i")


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build(variant="v1"):
    """回傳 notebook dict（測試用來比對已 commit 的 ipynb）。variant：v1（內容不變）或 v1c。"""
    if variant == "v1":
        cells = [cell("markdown", md0), cell("code", install_src), cell("code", version_src), cell("code", clone_src),
                 cell("code", import_src), cell("code", bench_src), cell("code", train_src), cell("code", plot_src), cell("code", save_src)]
    elif variant == "v1c":
        cells = [cell("markdown", md0_c), cell("code", install_src), cell("code", version_src), cell("code", clone_src), cell("code", mount_src_c),
                 cell("code", import_src_c), cell("code", bench_src), cell("code", train_src_c), cell("code", plot_src), cell("code", save_src_c)]
    elif variant == "v1d":
        cells = [cell("markdown", md0_d), cell("code", install_src), cell("code", version_src), cell("code", clone_src), cell("code", mount_src_d),
                 cell("code", import_src_c), cell("code", bench_src), cell("code", train_src_d), cell("code", plot_src), cell("code", save_src_d)]
    elif variant == "v1f":
        cells = [cell("markdown", md0_f), cell("code", install_src), cell("code", version_src), cell("code", clone_src), cell("code", mount_src_f),
                 cell("code", import_src_f), cell("code", bench_src), cell("code", train_src_f), cell("code", plot_src), cell("code", save_src_f)]
    elif variant == "v1g":
        cells = [cell("markdown", md0_g), cell("code", install_src), cell("code", version_src), cell("code", clone_src), cell("code", mount_src_g),
                 cell("code", import_src_g), cell("code", bench_src), cell("code", train_src_g), cell("code", plot_src), cell("code", save_src_g)]
    elif variant == "v1h":
        cells = [cell("markdown", md0_h), cell("code", install_src), cell("code", version_src), cell("code", clone_src), cell("code", mount_src_h),
                 cell("code", import_src_h), cell("code", bench_src), cell("code", train_src_h), cell("code", plot_src), cell("code", save_src_h)]
    elif variant == "v1i":
        cells = [cell("markdown", md0_i), cell("code", install_src), cell("code", version_src), cell("code", clone_src), cell("code", mount_src_i),
                 cell("code", import_src_i), cell("code", bench_src), cell("code", train_src_i), cell("code", plot_src), cell("code", save_src_i)]
    else:
        raise ValueError(variant)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"}, "cells": cells}


def main():
    for variant, out in (("v1", OUT), ("v1c", OUT_C), ("v1d", OUT_D), ("v1f", OUT_F), ("v1g", OUT_G), ("v1h", OUT_H), ("v1i", OUT_I)):
        nb = build(variant)
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("→", out, f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
