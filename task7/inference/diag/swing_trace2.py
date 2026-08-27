import sys
import numpy as np
sys.path.insert(0, 'task7/inference')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm
import local_infer_max as li

B = gb.BASELINE
r = cw.Robot()
ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
stepf = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
q0 = cpg_max.stand_targets(ks, f0, B['x_off'])
r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
    r.step(q0)
    if i == int(0.5 / mm.CTRL_DT):
        r.lock_wheels()

c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
n = int(12.0 / mm.CTRL_DT)
mux, muy, om = li.act_to_cmd(li.baseline_action())
T = {k: [] for k in (0, 2)}
for i in range(n):
    c = stepf(c, mux, muy, om, mm.CTRL_DT)
    tgt = cpg_max.foot_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                               B['d_step_y'], B['duty'], B['z_sag'])
    q, _ = cpg_max.joint_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                 B['d_step_y'], B['duty'], ks, B['z_sag'])
    r.step(q)
    th = cpg_max.duty_remap(c["theta"], B['duty'])
    sw = np.sin(th) > 0
    fz = r.foot_forces()
    hs = r.foot_heights()
    for k in (0, 2):
        qa = r.d.qpos[mm.LEG_QPOS_IDX][3 * k:3 * k + 3]
        fk = leg_kin.fk(k, qa)
        T[k].append((sw[k], (tgt[k, 0] - f0[k, 0]) * 1000, (fk[0] - f0[k, 0]) * 1000,
                     (tgt[k, 2] - f0[k, 2]) * 1000, (fk[2] - f0[k, 2]) * 1000,
                     hs[k] * 1000, fz[k],
                     *np.degrees(q[3 * k:3 * k + 3] - qa)))

for k in (0, 2):
    a = T[k]
    swf = [row[0] for row in a]
    # 取第 5 段完整擺動
    segs, s = [], None
    for i in range(1, len(swf)):
        if swf[i] and not swf[i - 1]:
            s = i
        elif (not swf[i]) and swf[i - 1] and s is not None:
            segs.append((s, i))
    st, en = segs[4]
    print(f"\n=== {mm.LEGS[k]}（{'前' if k < 2 else '後'}腳）第 5 次擺動，共 {en-st} 個控制步 ===")
    print(f"{'步':>3}{'指令dx':>9}{'實際dx':>9}{'指令dz':>9}{'實際dz':>9}"
          f"{'離地mm':>8}{'法向力N':>9}{'誤差 abad/hip/knee °':>26}")
    for j in range(st - 1, en + 1):
        _, cx, ax, cz, az, hz, fz, e0, e1, e2 = a[j]
        m = '←起' if j == st else ('←止' if j == en else '')
        print(f"{j-st:>3}{cx:>9.1f}{ax:>9.1f}{cz:>9.1f}{az:>9.1f}"
              f"{hz:>8.1f}{fz:>9.1f}{e0:>8.2f}{e1:>9.2f}{e2:>9.2f} {m}")
