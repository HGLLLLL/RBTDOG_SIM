#!/usr/bin/env python3
"""產生 RL v2 的 Colab notebook（`cpg_rl_max_colab.ipynb`）。

env 在 `task7/inference/rl_env_max.py`，這裡只有流程：安裝 → 版本斷言 → clone → import →
基準校準 → 訓練 → 曲線 → 存檔。**改 env 請改 repo、push、重新 clone，不要在 notebook 裡貼程式。**

第 1、2 格（安裝與版本斷言）沿用既有 notebook 的內容（版本鎖死那段的理由寫在那裡）。
用法：python3 task7/notebooks/build_nb_v2.py
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--preset", default="v2", help="rl_env_max.PRESETS 的鍵")
_ap.add_argument("--out", default="cpg_rl_max_colab.ipynb")
_ap.add_argument("--weights", default="cpg_rl_max_v2_params.pkl")
_a = _ap.parse_args()
PRESET, WEIGHTS = _a.preset, _a.weights
OUT = HERE / _a.out
SRC_NB = HERE / "cpg_rl_max_colab.ipynb"        # 安裝/版本兩格從 v2 notebook 沿用
OLD = json.loads(SRC_NB.read_text(encoding="utf-8"))
_code = [c for c in OLD["cells"] if c["cell_type"] == "code"]
install_src = "".join(_code[0]["source"])
version_src = "".join(_code[1]["source"])
assert "brax==0.14.2" in install_src and 'brax.__version__ == "0.14.2"' in version_src

md0 = f"""# CPG-RL **{PRESET}** 訓練：智元 D1 Max · MJX · Colab GPU（2026-09-08）

基準 `A_kp250_walk`（LS / kp250 / abad60 / kd2 / wheel_kd 0.5，實機 trip17 兩趟零中止走完）。
14 維動作（每腿 mux/muy/ω ＋ body sway x,y）、70 維 obs，隨機化與護欄依
`task7/docs/H_實機obs盤點_2026-09-08.md`；設計 `docs/superpowers/specs/2026-09-08-cpg-rl-v2-retrain-design.md`。

**env 住在 repo（`task7/inference/rl_env_max.py`），本 notebook 只有：安裝 → clone → 校準 → 訓練 → 存檔。**
reward preset：**`{PRESET}`**（權重定義與校準依據都在 `rl_env_max.PRESETS`；`diag/rl_calibrate.py` 量的）。
改 env 請改 repo、push、重新 clone（第 3 格會印 clone 到的 commit），不要在這裡貼程式。

流程：全部執行 → 第 5 格看「基準校準」（基準動作不該 done、不該碰護欄）→ 第 6 格第一個 eval
看實測步率（60M 超過 3 小時就砍 `TIMESTEPS` 到 40M 重跑）→ 下載 `{WEIGHTS}`
→ 本機 `local_infer_max.py --params … --secs 60 --perturb 12 --compare --video` 判 G3–G7。
"""

clone_src = '''import os, subprocess, sys

REPO = "https://github.com/HGLLLLL/RBTDOG_SIM.git"
BRANCH = "main"
DEST = "rbtdog_sim"          # repo 名是大寫 RBTDOG_SIM，明寫目的地

if not os.path.exists(DEST):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, DEST], check=True)
sys.path.insert(0, f"{DEST}/task7/inference")
print("clone 到的 commit：",
      subprocess.run(["git", "-C", DEST, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
# ★ 訓練模型 zgws_mjx_kp250.xml 零 STL 相依，不需要 fetch_assets.sh
'''

import_src = f'''import jax, jax.numpy as jnp, numpy as np, mujoco
import rl_env_max as re
import obs_max, gait_baseline as gb, max_model as mm

PRESET = "{PRESET}"
W = re.weights_of(PRESET)
print("preset", PRESET, "權重", W)
print("obs", re.OBS_DIM, "act", re.ACT_DIM, "scene", mm.SCENE_MJX_KP250)
print("基準", gb.BASELINE_A)
print("護欄 TAU_BAR", re.TAU_BAR, "ERR_BAR", re.ERR_BAR, "| sway ±", re.SWAY_MAX, "斜率", re.SWAY_SLEW)
assert (re.OBS_DIM, re.ACT_DIM) == (70, 14)
assert list(gb.BASELINE_A["kp3"]) == [60.0, 250.0, 250.0], "ABAD 必須是 60"
'''

calib_src = '''# ---- ★ 基準校準：A 步態是固定動作（sway=0），每一項 reward 在它身上值多少？----
# 兩個目的：(1) env 有沒有重現開迴路基準（本機 G0：speed 0.34、roll_pk 3.9°、exec 前 0.88/後 1.47）
#          (2) 護欄在基準上不該被碰到（tau_pk 平均要 < TAU_BAR、err_pk < ERR_BAR）
env = re.MaxCpgEnv(preset=PRESET)
assert env.preset == PRESET and env.w == W
assert env.sys.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
kp_xml = np.asarray(env.sys.actuator_gainprm[np.asarray(mm.LEG_ACT_IDX), 0])
assert np.allclose(kp_xml, np.tile(mm.KP3_A, 4)), f"訓練模型增益 {kp_xml[:3]} ≠ KP3_A"
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
A_BASE = jnp.array(re.baseline_action())
s = jit_reset(jax.random.PRNGKey(0))
print("reset ok, obs", s.obs.shape, "height %.4f m" % float(s.pipeline_state.qpos[2]))

import time as _t
_t0 = _t.time()
acc = {k: [] for k in re.METRIC_KEYS}
xs = []
for i in range(500):                      # 10 s
    s = jit_step(s, A_BASE)
    assert float(s.done) == 0.0, f"基準動作在第 {i} 步 done —— env 有問題，不要訓"
    if i >= 250:
        for k in re.METRIC_KEYS:
            acc[k].append(float(s.metrics[k]))
        xs.append(float(s.pipeline_state.qpos[0]))
print(f"[基準 10 s，後半段平均]  " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in acc.items())
      + f"   ({_t.time() - _t0:.0f}s)")
print(f"[護欄] tau_pk 平均 {np.mean(acc['tau_pk']):.1f} / 最大 {np.max(acc['tau_pk']):.1f}（TAU_BAR {re.TAU_BAR}）"
      f"   err_pk 平均 {np.mean(acc['err_pk']):.3f} / 最大 {np.max(acc['err_pk']):.3f}（ERR_BAR {re.ERR_BAR}）")
print("       基準步態不該常態碰到護欄 —— 若 tau_pk 平均 > TAU_BAR 或 err_pk 平均 > ERR_BAR，先回本機查再訓")
print(f"[對照] 本機 local_infer_max --dummy：speed_travel 0.34 m/s、roll_pk 3.86°、exec 前 0.88 / 後 1.47")

# ---- ★ reward 佔比檢查（權重是量出來校準的，這裡再驗一次；與本機 diag/rl_calibrate.py 同算法）----
pos = np.mean(acc["t_pos"])
share = {k: np.mean(acc[k]) / pos for k in re.TERM_KEYS if k != "t_pos"}
print("[佔比] " + "  ".join(f"{k[2:]} {100*v:.1f}%" for k, v in share.items() if v > 0.001))
roll_sh = share["t_roll"] + share["t_rollrate"]
pitch_sh = share["t_pitch"] + share["t_pitchrate"]
if PRESET != "v2":
    assert 0.10 <= roll_sh <= 0.25, f"roll 兩項佔正項 {roll_sh:.1%}，不在 10–25%：權重跟本機校準對不上，不要訓"
    assert 0.03 <= pitch_sh <= 0.15, f"pitch 兩項佔 {pitch_sh:.1%}"
    assert share["t_exec"] <= 0.50 and roll_sh > pitch_sh
assert share["t_taubar"] < 0.01 and share["t_errbar"] < 0.01, "基準動作碰到護欄 —— env 或模型有問題，不要訓"
print(f"[佔比] roll+rate {roll_sh:.1%}  pitch+rate {pitch_sh:.1%}  exec {share['t_exec']:.1%}  ✅ 與本機校準一致")
'''

train_src = '''import functools, time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

env = re.MaxCpgEnv(preset=PRESET)
# ⚠️ 網路尺寸與 normalize_observations=True 必須與本機推論端 (local_infer_max.py) 逐項相同。
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(256, 256, 128),
    value_hidden_layer_sizes=(256, 256, 256))

TIMESTEPS = 60_000_000
train_fn = functools.partial(
    ppo.train, num_timesteps=TIMESTEPS, num_evals=20, episode_length=1000,
    num_envs=2048, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.97, normalize_observations=True,
    network_factory=network_factory, randomization_fn=re.domain_randomize, seed=0)

_t0 = time.time()
rewards = []


def progress(step, metrics):
    r = float(metrics.get("eval/episode_reward", 0.0))
    rewards.append((step, r))
    L = float(metrics.get("eval/avg_episode_length", 1.0)) or 1.0

    def ps(k):
        return float(metrics.get(f"eval/episode_{k}", 0.0)) / L

    el = time.time() - _t0
    rate = step / max(el, 1e-9)
    print(f"step {step:>11,} R {r:7.2f} | roll {ps('roll'):4.2f}° pitch {ps('pitch'):4.2f}° "
          f"exec f/r {ps('exec_f'):.2f}/{ps('exec_r'):.2f} yawerr {ps('yawerr'):.3f} "
          f"vxerr {ps('vxerr'):.3f} | tau_pk {ps('tau_pk'):5.1f} err_pk {ps('err_pk'):.3f} "
          f"sway {ps('sway_x'):.0f}/{ps('sway_y'):.0f}mm len {L:.0f} | "
          f"{el:.0f}s {rate / 1e3:.0f}k步/s → {TIMESTEPS / 1e6:.0f}M 約 {TIMESTEPS / max(rate, 1) / 60:.0f} 分")


# 指標怎麼讀（每控制步平均）：roll/pitch 越小越好（基準動作在 env 裡 roll 2.35° / pitch 1.41°）；exec f/r 越接近 1 越好；
# tau_pk 要一路 < 58、err_pk < 0.45（護欄）；sway 是 policy 實際用的質心位移；len 1000 = 沒提早 done。
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print("training done —— preset", PRESET)
'''

plot_src = '''import matplotlib.pyplot as plt
plt.plot([s for s, _ in rewards], [r for _, r in rewards], marker="o")
plt.xlabel("env steps"); plt.ylabel("eval reward"); plt.grid(True); plt.show()
'''

save_src = f'''from brax.io import model
model.save_params("{WEIGHTS}", params)
print("已存 {WEIGHTS} → 下載放 task7/weights/，本機（rbtdog 環境）跑：")
print("  conda run --no-capture-output -n rbtdog python task7/inference/local_infer_max.py "
      "--params task7/weights/{WEIGHTS} --secs 60 --perturb 12 --compare --video")
print("★ 驗收在原始網格模型上做；G7（力矩×1.2<70、誤差×1.14<0.6）任一不過就不上機。")
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU"},
      "cells": [cell("markdown", md0), cell("code", install_src), cell("code", version_src),
                cell("code", clone_src), cell("code", import_src), cell("code", calib_src),
                cell("code", train_src), cell("code", plot_src), cell("code", save_src)]}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("→", OUT, f"({len(nb['cells'])} cells, preset {PRESET}, weights {WEIGHTS})")
