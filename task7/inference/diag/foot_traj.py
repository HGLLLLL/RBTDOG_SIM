"""足端在**世界座標**的軌跡：一眼看出「抬起來原地放下」vs「真的往前跨」。

正常的步態：腳貼地不動（站立相）→ 往前畫一個弧（擺動相）→ 再貼地不動。
壞掉的：原地上下畫一個窄的倒 U，位置幾乎沒往前移。
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, 'task7/inference')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm


def trace(cfg, kp3=None, kd3=None, z_sag=None, secs=14.0):
    r = cw.Robot(kp3=kp3, kd3=kd3)
    z = mm.STATIC_SAG if z_sag is None else z_sag
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    stepf = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    q0 = cpg_max.stand_targets(ks, f0, cfg["x_off"])
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q0)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    out = []
    n = int(secs / mm.CTRL_DT)
    for i in range(n):
        c = stepf(c, np.full(4, cfg["mu_x"]), np.full(4, 1.5),
                  np.full(4, cfg["omega"]), mm.CTRL_DT)
        q, _ = cpg_max.joint_targets(c, f0, cfg["x_off"], cfg["g_c"], cfg["d_step"],
                                     0.12, cfg["duty"], ks, z)
        r.step(q)
        if i >= n // 2:
            out.append([[float(r.d.xpos[b][0]), float(r.d.xpos[b][2]) - mm.WHEEL_RADIUS]
                        for b in r.foot_bid])
    return np.asarray(out)      # (T, 4, 2)


old = trace(gb.BASELINE)
new = trace(gb.BASELINE_KP250, kp3=gb.BASELINE_KP250["kp3"],
            kd3=gb.BASELINE_KP250["kd3"], z_sag=gb.BASELINE_KP250["z_sag"])

fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharey=True)
for row, (dat, name) in enumerate(((old, "OLD  kp120 / duty0.80 / d_step0.10"),
                                   (new, "NEW  kp250 / duty0.85 / d_step0.12"))):
    for col, (k, leg) in enumerate(((0, "FR (front)"), (2, "RR (rear)"))):
        ax = axes[row][col]
        x = (dat[:, k, 0] - dat[0, k, 0]) * 1000
        zz = dat[:, k, 1] * 1000
        ax.plot(x, zz, lw=0.9, color="#2b6cb0" if col == 0 else "#c05621")
        ax.axhline(0, color="#888", lw=0.8, ls="--")
        ax.set_title(f"{name}   —   {leg}", fontsize=10)
        ax.set_xlabel("foot world X (mm)")
        ax.grid(alpha=0.25)
        span = x.max() - x.min()
        ax.text(0.02, 0.92, f"X span {span:.0f} mm", transform=ax.transAxes,
                fontsize=11, weight="bold",
                color="#c53030" if span < 200 else "#276749")
    axes[row][0].set_ylabel("foot height (mm)")
fig.suptitle("Foot path in WORLD frame (7 s). Proper stepping = wide forward loops; "
             "stuck leg = narrow up-down in place.", fontsize=11)
fig.tight_layout()
out = "task7/outputs/foot_path_kp250_vs_baseline.png"
fig.savefig(out, dpi=110)
print("[圖]", out)
for name, dat in (("OLD", old), ("NEW", new)):
    sp = [(dat[:, k, 0].max() - dat[:, k, 0].min()) * 1000 for k in range(4)]
    print(f"{name}  7 秒內足端 X 跨幅 mm: " +
          "  ".join(f"{L}={v:.0f}" for L, v in zip(mm.LEGS, sp)))
