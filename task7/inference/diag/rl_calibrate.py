#!/usr/bin/env python3
"""RL reward 校準：基準固定動作在 env 裡跑 N 步，印每一項 reward 的平均值與**佔正項的比例**。

為什麼要有這支：v2 的 W_ROLL=20 在 roll 2.3° 時每步只扣 0.03 分（正項的 1%），
policy 完全有理由忽略它 —— 權重要照這裡量到的佔比訂，不能猜。

用法：conda run --no-capture-output -n rbtdog python task7/inference/diag/rl_calibrate.py --preset v2.1 [--steps 500]
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_env_max as re  # noqa: E402


CAL_CMD = (0.15, 0.0)     # 校準用固定指令：直走 0.15 m/s（隨機指令會讓 vx/yaw 兩項在基準上失真）


def calibrate(preset: str, steps: int = 500, seed: int = 0, verbose: bool = True) -> dict:
    env = re.MaxCpgEnv(preset=preset)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(seed))
    s = s.replace(info={**s.info, "cmd": jnp.array(CAL_CMD)})
    a = jnp.array(re.baseline_action())
    acc = {k: [] for k in re.METRIC_KEYS}
    for i in range(steps):
        s = step(s, a)
        assert float(s.done) == 0.0, f"基準動作在第 {i} 步 done"
        if i >= steps // 2:
            for k in re.METRIC_KEYS:
                acc[k].append(float(s.metrics[k]))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    pos = m["t_pos"]
    share = {k: m[k] / pos for k in re.TERM_KEYS if k != "t_pos"}
    if verbose:
        w = re.weights_of(preset)
        print(f"preset {preset}  cmd {CAL_CMD}  {steps} 步（統計後半）  正項總和 {pos:.3f}/步  reward {m['reward']:.3f}/步")
        print(f"  roll {m['roll']:.2f}° pitch {m['pitch']:.2f}° exec f/r {m['exec_f']:.2f}/{m['exec_r']:.2f}"
              f"  tau_pk {m['tau_pk']:.1f} err_pk {m['err_pk']:.3f}")
        print(f"  {'項':12s} {'值/步':>9s} {'佔正項':>7s}")
        for k in re.TERM_KEYS:
            if k == "t_pos":
                continue
            print(f"  {k:12s} {m[k]:9.4f} {100 * share[k]:6.1f}%")
        print(f"  權重：" + " ".join(f"{k}={v}" for k, v in w.items()
                                     if k in ("W_ROLL", "W_ROLLRATE", "W_PITCH", "W_PITCHRATE", "W_EXEC", "EXEC_SIGMA",
                                              "W_YAW", "YAW_SIG2", "YAW_EMA")))
    return {"metrics": m, "share": share, "pos": pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2.1")
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    calibrate(a.preset, a.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
