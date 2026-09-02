"""足端世界座標軌跡：舊基準 vs `kd_wheel=3.0 / x_off=−110mm`。

⚠️ 判讀方式（`diag/README.md` 的警語）：**不要看 X 跨幅**。穩態下每隻腳平均
都必須跟上機身，所以跨幅本來就會差不多。**要看的是形狀** ——
「近垂直的窄尖峰 + 貼地水平線」= 抬起來原地放下、再被拖著走；
「往前送的寬弧」= 真的在跨步。
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, 'task7/inference')
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm


def trace(x_off, kd_wheel, secs=14.0):
    """回傳後半段（穩態）的足端世界座標 (T, 4, 2) —— x 與離地高度。"""
    cfg = dict(gb.BASELINE, x_off=x_off)
    r = cw.Robot(kd_wheel=kd_wheel)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    stepf = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    q0 = cpg_max.stand_targets(ks, f0, x_off)
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q0)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    out = []
    n = int(secs / mm.CTRL_DT)
    for i in range(n):
        c = stepf(c, np.full(4, cfg["mu_x"]), np.full(4, cfg["mu_y"]),
                  np.full(4, cfg["omega"]), mm.CTRL_DT)
        q, _ = cpg_max.joint_targets(c, f0, x_off, cfg["g_c"], cfg["d_step"],
                                     cfg["d_step_y"], cfg["duty"], ks, mm.STATIC_SAG)
        r.step(q)
        if i >= n // 2:
            out.append([[float(r.d.xpos[b][0]), float(r.d.xpos[b][2]) - mm.WHEEL_RADIUS]
                        for b in r.foot_bid])
    return np.asarray(out)


old = trace(-0.040, 0.5)
new = trace(-0.110, 3.0)

fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharey=True)
for row, (dat, name) in enumerate((
        (old, "OLD   wheel kd 0.5 / x_off -40 mm"),
        (new, "NEW   wheel kd 3.0 / x_off -110 mm"))):
    for col, (k, leg) in enumerate(((0, "FR (front)"), (2, "RR (rear)"))):
        ax = axes[row][col]
        ax.plot((dat[:, k, 0] - dat[0, k, 0]) * 1000, dat[:, k, 1] * 1000,
                lw=0.9, color="#2b6cb0" if col == 0 else "#c05621")
        ax.axhline(0, color="#888", lw=0.8, ls="--")
        ax.set_title(f"{name}   —   {leg}", fontsize=10)
        ax.set_xlabel("foot world X (mm)")
        ax.grid(alpha=0.25)
    axes[row][0].set_ylabel("foot height (mm)")
fig.suptitle("Foot path in WORLD frame (7 s steady state). "
             "Look at the SHAPE, not the X span: narrow vertical spike + flat ground "
             "line = lifted and put back down, then dragged.", fontsize=10.5)
fig.tight_layout()
out = "task7/outputs/foot_path_wheelkd_vs_baseline.png"
fig.savefig(out, dpi=110)
print("[圖]", out)
