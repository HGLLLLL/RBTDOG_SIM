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

**流程**：第一格 throughput 測試選 `NUM_ENVS` → 訓練（每次 eval 存 pkl 到 `/content/rl_wheel_v1c/`，**不用雲端硬碟**；session 跑完立刻下載 `rl_wheel_v1c_params.pkl` 與最新的 `rl_wheel_v1c_step*.pkl`）→ 一個 session 5 億步（G4、4096 環境約 50 分鐘）→ 下個 session 把 `rl_wheel_v1c_step*.pkl` 上傳回 `/content/rl_wheel_v1c/`，`RESUME="auto"` 自動接。
**5 億步中場檢查**：session 1 結束把 step 檔拉回本機 `export_policy_wheel.py` → 跑 spec §6 全表；roll 左右差沒縮、或驗收表的「落地 vz」沒比 v1b 低 → 先調權重再跑 session 2。
**停損**：累計 2 億步時 `len < 600`、或 `lin < 0.5`、或 `wgap` 沒比 step 0 好／`gap` 中位 < 40 mm、或 `impact`、`roll` 沒比 step 0 往 0 走 → 停。
**讀法**：`lin`／`ang` 往 2.5／2.5；`roll`、`impact` 往 0；`land_vz` ＝ 每次落地平均下落速度 m/s（Σvz／Σ落地次數，不是除步數；v1b 本機驗收平移 0.5 約 0.36），往下；`mir` 應在 0.5 附近、`d` 在 1.0 附近（0–2 平均）；eval 也開鏡像／延遲／偏差，所以 eval reward 不是乾淨名目條件的數字，中場判讀以本機驗收表為準；其餘同 v1。
"""
import_src_c = import_src.replace("env = rw.WheelEnv()", "env = rw.WheelEnv(**rw.V1C)")
train_src_c = (train_src.replace("rl_wheel_v1", "rl_wheel_v1c")
               .replace("（v1 用了 5 億；v1b 從 v1 的 step 檔續訓再 5 億）", "（v1c 從頭訓，兩個 session 各 5 億）")
               .replace("env = rw.WheelEnv()\neval_env = rw.WheelEnv(noise=False, push=False)",
                        "env = rw.WheelEnv(**rw.V1C)\neval_env = rw.WheelEnv(noise=False, push=False, **rw.V1C)")
               .replace("    ps = lambda k: float(metrics.get(f\"eval/episode_{k}\", 0.0)) / L\n",
                        "    ps = lambda k: float(metrics.get(f\"eval/episode_{k}\", 0.0)) / L\n"
                        "    ev = lambda k: float(metrics.get(f\"eval/episode_{k}\", 0.0))                  # 回合總和（不除步數）\n"
                        "    land_vz = ev('impact_vz') / max(ev('td_n'), 1e-9)                              # 每次落地平均下落速度 m/s ＝ Σvz／Σ落地次數（除步數會被無落地的步稀釋）\n")
               .replace("lim {ps('lim'):+.3f} sat {ps('sat_n'):.2f} | ",
                        "lim {ps('lim'):+.3f} sat {ps('sat_n'):.2f} | roll {ps('roll'):+.3f} impact {ps('impact'):+.3f} land_vz {land_vz:.2f} mir {ps('mir'):.2f} d {ps('delay'):.1f} | "))
assert "rl_wheel_v1_step" not in train_src_c and "rw.V1C" in train_src_c and "ev('td_n')" in train_src_c and "land_vz" in train_src_c
save_src_c = save_src.replace(WEIGHTS, WEIGHTS_C).replace("/content/rl_wheel_v1/", "/content/rl_wheel_v1c/")


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
        cells = [cell("markdown", md0_c), cell("code", install_src), cell("code", version_src), cell("code", clone_src),
                 cell("code", import_src_c), cell("code", bench_src), cell("code", train_src_c), cell("code", plot_src), cell("code", save_src_c)]
    else:
        raise ValueError(variant)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"}, "cells": cells}


def main():
    for variant, out in (("v1", OUT), ("v1c", OUT_C)):
        nb = build(variant)
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("→", out, f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
