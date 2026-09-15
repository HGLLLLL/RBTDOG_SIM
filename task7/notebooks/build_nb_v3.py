#!/usr/bin/env python3
"""產生 RL v3（雙模式）的 Colab notebook `cpg_rl_v3_colab.ipynb`。
env 在 `task7/inference/rl_env_v3.py`；安裝／版本兩格沿用 v2 notebook。
用法：python3 task7/notebooks/build_nb_v3.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cpg_rl_v3_colab.ipynb"
WEIGHTS = "cpg_rl_v3_params.pkl"
OLD = json.loads((HERE / "cpg_rl_max_colab.ipynb").read_text(encoding="utf-8"))
_code = [c for c in OLD["cells"] if c["cell_type"] == "code"]
install_src, version_src = "".join(_code[0]["source"]), "".join(_code[1]["source"])
assert "brax==0.14.2" in install_src

md0 = f"""# CPG-RL **v3.2**：智元 D1 Max · MJX · Colab GPU（2026-09-15）

模仿原廠運控：純前進＝腿站姿＋輪子；**平移＝原廠命令週期**（trip21 錄檔平均的關節命令，2.1 Hz）；原地轉＝對角小跑 1.8 Hz（原廠週期開迴路必倒，退路，spec §9.1）；
邊走邊轉＝差速＋內側前腿踏步；三軸可疊加（斜走）。
設計 `docs/superpowers/specs/2026-09-15-cpg-rl-v3.1-unified-kinematic-generator-design.md`（§0 勘誤、§8 G0 與原廠回放、§9 v3.2），原廠參考 `task7/docs/results/L_*.md`（含勘誤）。

**env 住在 repo（`task7/inference/rl_env_v3.py`），本 notebook 只有：安裝 → clone → G0 → 訓練 → 存檔。**
動作 12 維、obs 76 維；零動作＝純開迴路產生器。腿 kp 60/250/250、輪子速度伺服 kd 1.0（M11 實機已驗）。
G0 標準答案（本機 CPU 10 s，`outputs/g0_v32_final.txt`）：直走 0.49、弧線 +25°/s、原地轉 +14～28°/s、平移 0.08 → 0.062 m/s（roll std 0.3°）、斜走 0.33/0.049。
訓練完的驗收對標原廠運控（`outputs/ref_gait_dataset.md` 對標表：側傾、偏航率、速度、腿力矩峰值／RMS、抬腳）。
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

import_src = '''import jax, jax.numpy as jnp, numpy as np, mujoco
import rl_env_v3 as v3
env = v3.DualModeEnv()
print("obs", env.obs_dim, "act", env.action_size, "wheel_pos", env.wheel_pos)
assert env.wheel_pos is False
print("REF", {k: v for k, v in v3.REF.items() if not hasattr(v, "shape")})
assert (env.obs_dim, env.action_size) == (76, 12)
'''

g0_src = '''# ---- G0：零動作在五種指令下跑 10 s（同 diag/g0_v3.py --steps 500；門檻 spec §8.2／§9.1，本機結果 outputs/g0_v32_final.txt）
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
def run(cmd, steps=500):
    s = jit_reset(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10**6})
    a = jnp.zeros(v3.ACT_DIM); M = []
    for i in range(steps):
        s = jit_step(s, a); M.append({k: float(s.metrics[k]) for k in ("vx", "vy", "wz", "tau_pk", "roll", "clr_stance")} | {"done": float(s.done)})
    h = steps // 2
    return dict(vx=np.mean([m["vx"] for m in M[h:]]), vy=np.mean([m["vy"] for m in M[h:]]), yaw=np.degrees(np.mean([m["wz"] for m in M[h:]])),
                tau=max(m["tau_pk"] for m in M), roll_std=float(np.std([m["roll"] for m in M[h:]])), clr_stance=max(m["clr_stance"] for m in M[h:]),
                fell=any(m["done"] > 0 for m in M))
R = {}
for name, cmd in (("WHEEL", (0.5, 0, 0)), ("ARC", (0.5, 0, 0.5)), ("TURN", (0, 0, 1.3)), ("LAT", (0, 0.08, 0)), ("DIAG", (0.3, 0.06, 0))):
    R[name] = r = run(cmd); print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    assert not r["fell"], name
assert R["WHEEL"]["vx"] >= 0.45 and R["WHEEL"]["clr_stance"] < 10
assert R["ARC"]["yaw"] >= 20 and R["TURN"]["yaw"] >= 10
assert abs(R["LAT"]["vy"]) >= 0.05 and R["LAT"]["tau"] <= 85 and R["LAT"]["roll_std"] <= 1.5
assert R["DIAG"]["vx"] >= 0.15 and R["DIAG"]["vy"] >= 0.03
print("G0 全過 → 可以訓練")
'''

train_src = f'''import functools, time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

env = v3.DualModeEnv()
network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                    policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(256, 256, 256))
TIMESTEPS = 100_000_000   # 使用者 2026-09-15：可以訓久一點；60M 約 40 分鐘（T4/L4 視情況）
train_fn = functools.partial(
    ppo.train, num_timesteps=TIMESTEPS, num_evals=20, episode_length=1000,
    num_envs=2048, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.97, normalize_observations=True,
    network_factory=network_factory, randomization_fn=v3.domain_randomize, seed=0)
_t0 = time.time(); rewards = []

def progress(step, metrics):
    r = float(metrics.get("eval/episode_reward", 0.0)); rewards.append((step, r))
    L = float(metrics.get("eval/avg_episode_length", 1.0)) or 1.0
    ps = lambda k: float(metrics.get(f"eval/episode_{{k}}", 0.0)) / L
    el = time.time() - _t0; rate = step / max(el, 1e-9)
    print(f"step {{step:>11,}} R {{r:7.2f}} | roll {{ps('roll'):4.2f}} pitch {{ps('pitch'):4.2f}} bias {{ps('roll_bias'):+.2f}} | "
          f"vxerr {{ps('vxerr'):.3f}} vyerr {{ps('vyerr'):.3f}} yawerr {{ps('yawerr'):.3f}} | 進度 yaw {{ps('t_yawrel'):.2f}}/8 vy {{ps('t_vyrel'):.2f}}/3 | s4 {{ps('s4'):.2f}} arc {{ps('s_arc'):.2f}} clr {{ps('clr_step'):.0f}}/{{ps('clr_stance'):.0f}} | "
          f"tau_pk {{ps('tau_pk'):5.1f}} err {{ps('err_pk'):.3f}} knee_v {{ps('knee_v'):.1f}} | len {{L:.0f}} | "
          f"{{el:.0f}}s → {{TIMESTEPS / max(rate, 1) / 60:.0f}} 分")

# 讀法：進度 yaw／vy ＝ 有該軸指令時沿指令方向的相對進度（分母 8／3），要往上；vxerr/vyerr/yawerr 往下、roll_bias 往 0、tau_pk < 58、len 往 1000。
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print("training done")
'''

plot_src = '''import matplotlib.pyplot as plt
plt.plot([s for s, _ in rewards], [r for _, r in rewards], marker="o"); plt.xlabel("env steps"); plt.ylabel("eval reward"); plt.grid(True); plt.show()
'''

save_src = f'''from brax.io import model
model.save_params("{WEIGHTS}", params)
print("已存 {WEIGHTS} → 下載放 task7/weights/；本機驗收 local_infer_v3.py（待寫）對標 outputs/ref_gait_dataset.md 的原廠數字")
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"},
      "cells": [cell("markdown", md0), cell("code", install_src), cell("code", version_src), cell("code", clone_src),
                cell("code", import_src), cell("code", g0_src), cell("code", train_src), cell("code", plot_src), cell("code", save_src)]}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("→", OUT, f"({len(nb['cells'])} cells)")
