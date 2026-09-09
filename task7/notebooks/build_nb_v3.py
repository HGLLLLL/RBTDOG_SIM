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

md0 = f"""# CPG-RL **v3.0** 雙模式（輪行／原地旋轉）：智元 D1 Max · MJX · Colab GPU（2026-09-09）

模仿原廠運控：前進／後退／弧線＝腿站姿＋差速輪；原地旋轉＝對角腿同相踏步＋另一對輪反向。
設計 `docs/superpowers/specs/2026-09-09-cpg-rl-v3-dual-mode-design.md`、原廠參考 `task7/docs/results/L_*.md`、
輪子模型 `results/K_*.md`（M11）。**平移在 v3.0 不訓**（kp250 做不到原廠的單側踏步）。

**env 住在 repo（`task7/inference/rl_env_v3.py`），本 notebook 只有：安裝 → clone → G0 → 訓練 → 存檔。**
動作 12 維、obs 76 維；零動作＝純開迴路原廠模式表（G0 標準答案：直走 0.49 m/s、原地轉 27°/s、膝 70）。
輪子用位置環 kp 60（`scene_flat_mjx_v3p`，原廠形式；**實機未驗**，輪行模式上機那趟決定）。
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
print("obs", env.obs_dim, "act", env.action_size, "wheel_pos", env.wheel_pos, "kp_wheel", env.kp_wheel)
print("REF", {k: v for k, v in v3.REF.items() if not hasattr(v, "shape")})
assert (env.obs_dim, env.action_size) == (76, 12)
'''

g0_src = '''# ---- G0：零動作在三種指令下的行為（同 diag/g0_v3.py，本機 CPU 已跑過：直走 0.49、弧線 11°/s、原地轉 27°/s）
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
def run(cmd, steps=150):
    s = jit_reset(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10**6,
                        "u_mode": v3.mode_of(jnp.array(cmd))[0]})
    a = jnp.zeros(v3.ACT_DIM); M = []
    for i in range(steps):
        s = jit_step(s, a); M.append({k: float(s.metrics[k]) for k in ("vx", "wz", "tau_pk", "roll")} | {"done": float(s.done)})
    h = steps // 2
    return dict(vx=np.mean([m["vx"] for m in M[h:]]), yaw=np.degrees(np.mean([m["wz"] for m in M[h:]])),
                tau=max(m["tau_pk"] for m in M), fell=any(m["done"] > 0 for m in M))
for name, cmd in (("WHEEL 0.5", (0.5, 0, 0)), ("ARC", (0.5, 0, 0.3)), ("TURN 1.3", (0, 0, 1.3))):
    r = run(cmd); print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})
    assert not r["fell"]
'''

train_src = f'''import functools, time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

env = v3.DualModeEnv()
network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                    policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(256, 256, 256))
TIMESTEPS = 60_000_000
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
          f"vxerr {{ps('vxerr'):.3f}} yawerr {{ps('yawerr'):.3f}} | mode {{ps('mode'):.2f}} clr {{ps('clr_step'):.0f}}/{{ps('clr_stance'):.0f}} | "
          f"tau_pk {{ps('tau_pk'):5.1f}} err {{ps('err_pk'):.3f}} knee_v {{ps('knee_v'):.1f}} | len {{L:.0f}} | "
          f"{{el:.0f}}s → {{TIMESTEPS / max(rate, 1) / 60:.0f}} 分")

# 讀法：vxerr/yawerr 往下、roll_bias 往 0、tau_pk < 58、clr_stance < 10（輪行不抬腿）、len 1000。
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print("training done")
'''

plot_src = '''import matplotlib.pyplot as plt
plt.plot([s for s, _ in rewards], [r for _, r in rewards], marker="o"); plt.xlabel("env steps"); plt.ylabel("eval reward"); plt.grid(True); plt.show()
'''

save_src = f'''from brax.io import model
model.save_params("{WEIGHTS}", params)
print("已存 {WEIGHTS} → 下載放 task7/weights/；本機驗收工具 local_infer_v3.py（待寫）")
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
