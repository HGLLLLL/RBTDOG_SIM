#!/usr/bin/env python3
"""G1（RL v2）：kp250 訓練模型 vs 原始網格模型，A 步態 20 s × 12 擾動。

過關：speed_travel / bounce / support / min_lift / roll_pk / exec_front 的中位數差 ±5%、兩邊 0 跌倒。
用法：conda run --no-capture-output -n rbtdog python task7/inference/diag/g1_kp250.py [--seeds 12] [--secs 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cpg_walk_max as cw   # noqa: E402
import gait_baseline as gb  # noqa: E402
import max_model as mm      # noqa: E402

KEYS = ("speed_travel", "bounce", "support", "min_lift", "roll_pk", "exec_front")
A = gb.BASELINE_A


def run(scene, mode, seeds, secs):
    rows = []
    for s in range(seeds):
        kw = dict(gait="walk_a", secs=secs, kp3=A["kp3"], kd3=A["kd3"],
                  kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True,
                  x_off=A["x_off"] + s * 1e-12)
        if scene:
            kw.update(scene=scene, actuator_mode=mode, solver_iters=(6, 6))
        rows.append(cw.rollout(**kw))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--secs", type=float, default=20.0)
    a_ = ap.parse_args()
    a = run(None, "torque_pd", a_.seeds, a_.secs)
    b = run(mm.SCENE_MJX_KP250, "position", a_.seeds, a_.secs)
    print(f"{'指標':14s} {'網格':>10s} {'kp250 MJX':>10s} {'差%':>7s}")
    ok = True
    for k in KEYS:
        ma, mb = np.median([r[k] for r in a]), np.median([r[k] for r in b])
        pct = 100 * (mb - ma) / max(abs(ma), 1e-9)
        flag = "" if abs(pct) <= 5 else "  ⚠️"
        ok &= abs(pct) <= 5
        print(f"{k:14s} {ma:10.4f} {mb:10.4f} {pct:+7.1f}{flag}")
    fa = sum(r["fell"] is not None for r in a)
    fb = sum(r["fell"] is not None for r in b)
    print(f"跌倒 網格 {fa}/{a_.seeds}  MJX {fb}/{a_.seeds}")
    print("G1", "✅ 通過" if ok and fa == fb == 0 else "❌ 未過")
    return 0 if (ok and fa == fb == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
