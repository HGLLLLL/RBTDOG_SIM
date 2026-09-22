#!/usr/bin/env python3
"""產生 RL v3（雙模式）的 Colab notebook `cpg_rl_v3_colab.ipynb`。
env 在 `task7/inference/rl_env_v3.py`；安裝／版本兩格沿用 v2 notebook。
用法：python3 task7/notebooks/build_nb_v3.py
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--gains", default="kp250", choices=("kp250", "factory"))
_ap.add_argument("--weights", default="v33", choices=("v33", "v35", "v36", "v37", "v37b"),
                 help="v35＝DualModeEnv(weights=v3.W35)（慢漂懲罰＋線性航向，spec 2026-09-16）；v36＝W36（抬腳頂點獎勵＋hinge 懲罰＋質心 DR，spec 2026-09-22）；v37＝W37（平移抬高／跨距解耦）")
ARGS = _ap.parse_args()
FACTORY = ARGS.gains == "factory"
V35 = ARGS.weights == "v35"
V37B = ARGS.weights == "v37b"
V37 = ARGS.weights == "v37" or V37B
V36 = ARGS.weights == "v36" or V37                  # v3.7 = v3.6 的一切 ＋ 解耦產生器（W37 ⊃ W36）
VNEW = V35 or V36                                   # v3.5 起共用的東西（RESUME、vy 分母 6、progress 三欄）
assert not V36 or FACTORY, "v3.6／v3.7 只跑原廠增益線"
_tag = {(False, "v33"): "v3", (True, "v33"): "v3_4f", (False, "v35"): "v3_5", (True, "v35"): "v3_5f", (True, "v36"): "v3_6f", (True, "v37"): "v3_7f", (True, "v37b"): "v3_7bf"}[(FACTORY, ARGS.weights)]
OUT = HERE / f"cpg_rl_{_tag}_colab.ipynb"
WEIGHTS = f"cpg_rl_{_tag}_params.pkl"
GAINS_KW = 'gains="factory"' if FACTORY else ''
GAINS_KW2 = 'gains="factory", ' if FACTORY else ''    # 後面還有其他參數時用
W_KW = {"v33": "", "v35": "weights=v3.W35", "v36": "weights=v3.W36", "v37": "weights=v3.W37", "v37b": "weights=v3.W37B"}[ARGS.weights]
ENV_ARGS = ", ".join(x for x in (GAINS_KW, W_KW) if x)        # '' | 'gains="factory"' | 'weights=v3.W35' | 'gains="factory", weights=v3.W35'
ENV_ARGS2 = ENV_ARGS + ", " if ENV_ARGS else ""              # 後面還有參數時用
PROG_EXTRA = ("abad {ps('abad_bias'):.1f}° drift {ps('vx_drift'):+.3f} head {ps('head_abs'):.1f}° | " if VNEW else "") + ("step {ps('t_step'):.2f} | " if V36 else "")   # 以值插進 train_src 的 f-string，不會再被展開，所以用單層大括號
VY_DEN = "6" if VNEW else "3"
RESTORE_KW = ", restore_params=_restore" if VNEW else ""
# v3.3 的 notebook 要逐字元不變（產生器是共用的），所以這兩行在 kp250 分支輸出舊文字
ENV_PRINT = ('"gains", env.gains, "wheel_space", env.wheel_space' if FACTORY else '"wheel_pos", env.wheel_pos')
REF_SRC = "env.ref" if FACTORY else "v3.REF"
DR_FN = ('v3.make_domain_randomize("factory", com_y_mm=5.0)' if V36 else f'v3.make_domain_randomize("{ARGS.gains}")') if FACTORY else "v3.domain_randomize"
OLD = json.loads((HERE / "cpg_rl_max_colab.ipynb").read_text(encoding="utf-8"))
_code = [c for c in OLD["cells"] if c["cell_type"] == "code"]
install_src, version_src = "".join(_code[0]["source"]), "".join(_code[1]["source"])
assert "brax==0.14.2" in install_src

md0 = f"""# CPG-RL **v3.3**：智元 D1 Max · MJX · Colab GPU（2026-09-15 晚）

在 v3.2b（`weights/cpg_rl_v3_params_1.pkl`，九指令 0 摔、平移達標、原地轉只有原廠 30%）之上：
- **原地轉名目改原廠命令週期**（幅度每回合隨機 0.3–0.9；開迴路 3–7 s 會倒，靠 RL 學即時平衡）
- **動作加 12 個關節目標殘差**（±0.12 rad、限速）→ 動作 24 維、obs 88 維
- ABAD 剛度隨機 ×0.4–1.0 → ×0.7–1.0；學習率 ADAPTIVE_KL；2 億步
設計與紀錄：`docs/superpowers/specs/2026-09-15-cpg-rl-v3.1-unified-kinematic-generator-design.md` §9–§10。

**停損**：`進度 yaw` 若 1 億步時仍 < 3.5、或 len < 600，代表原廠週期的原地轉學不起來 → 停掉，回 v3.2b（`REF["step_gen_turn"]="kin"`）。
**注意**：G0 格的原地轉是開迴路原廠週期，**預期會倒**，那格只 assert 其他四個指令。
"""

if FACTORY:
    md0 = """# CPG-RL **v3.4f**：智元 D1 Max · 馬達增益完全照原廠 · MJX · Colab GPU（2026-09-15）

與 v3.3 的唯一差別是**馬達增益**：腿 ABAD 60 / HIP 120 / KNEE 120、kd 1.0，輪 kv 0.1
—— 逐檔掃 trip21 17 檔確認，這就是原廠錄檔動作段在用的那組（`scene_flat_mjx_v3f.xml`）。
連帶三項：撓度 36→72 mm（`z_sag` 0.072）、ABAD 誤差護欄 0.60→1.05（原廠命令差實測 max 1.02 rad）、
輪子改力矩空間（kv 0.1 下速度殘差只剩 ±0.2 N·m，沒有權限）。
設計：`docs/superpowers/specs/2026-09-15-cpg-rl-v3.4f-factory-gains-design.md`。

**停損**：同 v3.3 —— 1 億步時 `進度 yaw` < 3.5 或 len < 600 → 停。
**注意**：G0 格的原地轉是開迴路原廠週期，**預期會倒**，那格只 assert 其他四個指令。
**⚠️ `TAU_BAR` 58 在 kp 減半後大概率不再觸發**（spec §5）：訓練曲線要盯 `tau_pk`，
如果它明顯超過原廠的 40–60 而懲罰項一直是 0，代表這一版少了一個約束，要把 `TAU_BAR` 調到 35–40 重訓。
"""

if V35:
    md0 = f"""# CPG-RL **v3.5{"f" if FACTORY else ""}**：智元 D1 Max · 慢漂懲罰＋線性航向 · MJX · Colab GPU（2026-09-16）

在 v3.3（`weights/cpg_rl_v3_params_2.pkl`：原地轉 ±75°/s 達原廠 97%、九指令 0 摔）之上修使用者看到的兩個問題：
- **原地轉一條腿慢慢內收再踏出去**：左後腿 ABAD 6–8 s 漂到 +15°（跨種子同腿）。原廠圖案本來就對角不對稱（使用者決定保留），
  踏步時沒有東西把站立腳拉回名目 → 加 `t_abadbias`：四腿 ABAD 偏離站姿的 1 s 低通平方 × 30，只在踏步時開。
- **平移「往後漂」**：機身其實只往後 0.02–0.04 m/s，七成是航向轉掉 15–23° 的投影 →
  加 `t_drift`（機身 vx/vy 低頻誤差平方 × 40）＋ `t_headlin`（線性航向項，到 29° 都有梯度，權重 1.0）。
- 平移效率：`W_VYREL` 3→6、`P_VY` 0.45→0.60。
- **右轉週期＝左轉鏡像**（兩段錄檔相位對齊後平均，`CYC_TURN_SYM`）：右轉錄檔幅度小兩成，v3.4f 右轉只 −56°/s 對左轉 +78。對角不對稱保留。
其餘（obs 88、動作 24、護欄、DR{"、原廠增益" if FACTORY else ""}）與 v3.{"4f" if FACTORY else "3"} 相同。
設計：`docs/superpowers/specs/2026-09-16-cpg-rl-v3.5-drift-penalties-design.md`；攤帳 `outputs/reward_audit_v35.md`。

**停損**：同 v3.3 —— 1 億步 `進度 yaw` < 3.5 或 len < 600 → 停；**另加**：1 億步 `abad` 沒比 step 0 低、或 `drift`／`head` 沒往 0 走 → 新 reward 沒被學到，停下來查。
v3.3 在 1 億步時 3.3 沒過線、160M 才破到 4.2 —— 沒存中繼檢查點，決定跑完就別關分頁。
**注意**：G0 格的原地轉是開迴路原廠週期，**預期會倒**，那格只 assert 其他四個指令。
"""

if V36:
    md0 = """# CPG-RL **v3.6f**：智元 D1 Max · 平移抬腳頂點獎勵＋線性懲罰＋質心 DR · MJX · Colab GPU（2026-09-22）

在 v3.5f（`weights/cpg_rl_v3_5f_params.pkl`：九指令 0 摔、左轉 105% 原廠、航向 −32.7°→+4.5°）之上修四件事：
- **平移不踏步、靠側滑**：原廠週期模式下 `t_lift`／`t_mode`／`t_stance` 恆 0，policy 把每週期抬腳壓到 5–6 mm（零動作 14／11、原廠 21）→ 加 `t_step`：
  主動側兩腿**每週期抬腳頂點**對目標（21 mm，隨指令縮）的線性分數 × 1.5，只在平移族。
- **`t_drift`／`t_abadbias` 改死區＋線性**（`PEN_SHAPE="hinge"`）：二次式在 0.038 m/s 時每步只剩 0.058、推不到目標。
- **DR 加橫向質心偏移**（整機 ±5 mm 零均值）：模型質心偏左 1.5 mm 讓 v3.5f 右轉只有左轉的 79%，policy 學的是方向專屬補償。
- 驗收工具改預設無推力（之前每 2 s 一次的訓練推力污染了 roll／航向／vx 漂）。
設計：`docs/superpowers/specs/2026-09-22-cpg-rl-v3.6-step-apex-linear-penalties-design.md`；攤帳 `outputs/reward_audit_v36.md`。

**停損**：同 v3.5f —— 1 億步 `進度 yaw` < 3.5 或 len < 600 → 停；**另加**：1 億步 `step` < 0.3（平移族約佔回合四成、零動作名目 ≈ 0.6）→ 抬腳學不起來，停下來查 `W_STEP`。
`abad`／`drift`／`head` 三欄跨指令平均不可信（v3.5 spec §9.2），只看方向。
**注意**：G0 格的原地轉是開迴路原廠週期，**預期會倒**，那格只 assert 其他四個指令。
"""

if V37:
    md0 = """# CPG-RL **v3.7f**：智元 D1 Max · 平移抬高／跨距解耦 · MJX · Colab GPU（2026-09-22）

v3.6f 訓完：漂移三項到位（vx 漂 0.014、航向 −2°、ABAD 5.6°）、0 摔，但**平移抬腳只到 10 mm**（目標 21）、vy 過衝 25%。
根因（v3.6 spec §9.2）：原廠週期表裡抬高與跨距是同一組關節圖案，抬得高速度就超，policy 折衷成腳拖著側滑。
v3.7 只改一件事（`W37` → REF `cyc_lat_decouple=True`，實驗 `outputs/v37_pretest.md` E6–E9）：
- 平移族髖膝固定 1.75× 表（零動作抬 24–30 mm，原廠 21，不隨速度縮）、ABAD 跨距倍率隨指令 k = clip(0.3 + 5.6(|vy| − 0.12), 0.3, 1.0)；
  policy 的 `amp` 只管 ABAD 跨距、`lift` 只管髖膝抬高（兩者不再耦合）。名目會往後漂 −0.065 m/s、航向 10 s 轉 −48°（E11：跟膝倍率線性），交給 `t_drift`／`t_headlin`。
- 側向指令上限 0.30 → 0.22（抬 1.75× 時 2.1 Hz 踏步的名目上限 ≈ 0.18，再高只會教它側滑；E13）。
- 原地轉不動；右轉弱（−62 對 +84）已證明是 policy 不對稱（E5／E7／E8），**部署用鏡像推論**（E10：−84±1）不靠訓練。
其餘（`t_step`、hinge 懲罰、質心 DR、obs 88／動作 24）與 v3.6f 相同。

**停損**：同 v3.6f —— 1 億步 `進度 yaw` < 3.5 或 len < 600 → 停；`step` 名目現在是 1.5（平移族約四成回合 → 混合平均 ≈ 0.6），**1 億步 `step` < 0.45 → 停**（policy 又在壓抬腳）。
**注意**：G0 格的原地轉是開迴路原廠週期，**預期會倒**，那格只 assert 其他四個指令。
"""

if V37B:
    md0 = """# CPG-RL **v3.7b-f**：智元 D1 Max · 解耦產生器＋輪前饋＋抬高只加不減 · MJX · Colab GPU（2026-09-22 夜）

v3.7f 訓到 1 億步 `step` 停在 0.36（名目 0.5–0.6）：policy 用 `lift` 通道把名目抬高縮回去，因為抬高帶來的往後漂（−0.065）與航向偏轉（−48°/10 s）
在 reward 上每步扣約 1.4，跟 `t_step` 的 1.5 打平。v3.7b 三個修正（實驗 `outputs/v37_pretest.md` E14–E16）：
- **輪前饋**（只進輪命令、不改族別）：vx 前饋 clip(0.15 − 0.43|vy|, 0.05, 0.10) 把往後漂 −0.11／−0.065 壓到 −0.02／−0.01；wz 前饋 0.4·sign(vy) 把航向 −54° 砍到 −31°（剩下的來自踏步圖案本身，交給 policy 每腿 ABAD 幅度）。
- **`lift` 通道只能加不能減**（a ≤ 0 死區＝名目 1.75×，a > 0 到 1.4 倍）。
- `W_STEP` 1.5 → 2.5。
其餘同 v3.7f（髖膝 1.75×、ABAD 隨指令、`CMD_VY` 上限 0.22、hinge 懲罰、質心 DR）。原地轉部署用鏡像推論。

**停損**：1 億步 `進度 yaw` < 3.5 或 len < 600 → 停；**`step` < 0.45** → policy 又在用關節殘差壓抬腳，停下來查。
**注意**：G0 格的原地轉是開迴路原廠週期，**預期會倒**，那格只 assert 其他四個指令。
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

import_src = f'''import jax, jax.numpy as jnp, numpy as np, mujoco
import rl_env_v3 as v3
env = v3.DualModeEnv({ENV_ARGS})
print("obs", env.obs_dim, "act", env.action_size, {ENV_PRINT})
assert env.wheel_pos is False
print("REF", {{k: v for k, v in {REF_SRC}.items() if not hasattr(v, "shape")}})
assert (env.obs_dim, env.action_size) == (88, 24)
'''

g0_src = '''# ---- G0：零動作在五種指令下跑 10 s（同 diag/g0_v3.py --steps 500；門檻 spec §8.2／§9.1，本機結果 outputs/g0_v32_final.txt）
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
def run(cmd, steps=500):
    s = jit_reset(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10**6})
    a = jnp.zeros(v3.ACT_DIM); M = []
    for i in range(steps):
        s = jit_step(s, a); M.append({k: float(s.metrics[k]) for k in ("vx", "vy", "wz", "tau_pk", "roll", "clr_stance")} | {"done": float(s.done)})
        if float(s.done) > 0: break
    h = steps // 2; seg = M[h:] or M      # 5 s 前就倒（v3.3 TURN 開迴路預期）→ 用整段算，讓 fell 能回報而不是 max() 空序列炸掉
    return dict(vx=np.mean([m["vx"] for m in seg]), vy=np.mean([m["vy"] for m in seg]), yaw=np.degrees(np.mean([m["wz"] for m in seg])),
                tau=max(m["tau_pk"] for m in M), roll_std=float(np.std([m["roll"] for m in seg])), clr_stance=max(m["clr_stance"] for m in seg),
                fell=any(m["done"] > 0 for m in M))
R = {}
for name, cmd in (("WHEEL", (0.5, 0, 0)), ("ARC", (0.5, 0, 0.5)), ("TURN", (0, 0, 1.3)), ("LAT", (0, 0.08, 0)), ("DIAG", (0.3, 0.06, 0))):
    R[name] = r = run(cmd); print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    assert name == "TURN" or not r["fell"], name          # v3.3：原地轉名目是開迴路原廠週期，預期會倒
assert R["WHEEL"]["vx"] >= 0.45 and R["WHEEL"]["clr_stance"] < 10
assert R["ARC"]["yaw"] >= 20
assert abs(R["LAT"]["vy"]) >= 0.05 and R["LAT"]["tau"] <= 85 and R["LAT"]["roll_std"] <= 1.5
assert R["DIAG"]["vx"] >= 0.15 and R["DIAG"]["vy"] >= 0.03
print("G0 全過 → 可以訓練")
'''

RESUME_SRC = '''
# 續訓：把 RESUME 指到已下載的 .pkl（存檔是 (normalizer, policy, value) 三元組，brax train.py:735 正好吃這個形狀）。
# None = 從頭訓。續訓時 TIMESTEPS 是「這一輪要再跑幾步」，不是總數。
RESUME = None
from brax.io import model as _bm
_restore = None if RESUME is None else _bm.load_params(RESUME)
''' if VNEW else ''

train_src = f'''import functools, time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
{RESUME_SRC}
env = v3.DualModeEnv({ENV_ARGS})
eval_env = v3.DualModeEnv({ENV_ARGS2}ref=dict(cyc_amp_rand=False))      # 評估用固定幅度 0.7，曲線才可比
network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                    policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(256, 256, 256))
TIMESTEPS = 200_000_000   # v3.3：2 億步（100M 約 60 分鐘）
train_fn = functools.partial(
    ppo.train, num_timesteps=TIMESTEPS, num_evals=20, episode_length=1000,
    num_envs=2048, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.97, normalize_observations=True,
    learning_rate_schedule="ADAPTIVE_KL", desired_kl=0.01, learning_rate_schedule_min_lr=1e-5, learning_rate_schedule_max_lr=1e-3,
    network_factory=network_factory, randomization_fn={DR_FN}, seed=0, eval_env=eval_env{RESTORE_KW})
_t0 = time.time(); rewards = []

def progress(step, metrics):
    r = float(metrics.get("eval/episode_reward", 0.0)); rewards.append((step, r))
    L = float(metrics.get("eval/avg_episode_length", 1.0)) or 1.0
    ps = lambda k: float(metrics.get(f"eval/episode_{{k}}", 0.0)) / L
    el = time.time() - _t0; rate = step / max(el, 1e-9)
    print(f"step {{step:>11,}} R {{r:7.2f}} | roll {{ps('roll'):4.2f}} pitch {{ps('pitch'):4.2f}} bias {{ps('roll_bias'):+.2f}} | "
          f"vxerr {{ps('vxerr'):.3f}} vyerr {{ps('vyerr'):.3f}} yawerr {{ps('yawerr'):.3f}} | 進度 yaw {{ps('t_yawrel'):.2f}}/8 vy {{ps('t_vyrel'):.2f}}/{VY_DEN} | s4 {{ps('s4'):.2f}} cyc {{ps('cyc'):.2f}} clr {{ps('clr_step'):.0f}}/{{ps('clr_stance'):.0f}} | "
          f"tau_pk {{ps('tau_pk'):5.1f}} err {{ps('err_pk'):.3f}} knee_v {{ps('knee_v'):.1f}} | {PROG_EXTRA}len {{L:.0f}} | "
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
