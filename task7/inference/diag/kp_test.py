"""假設：前腳跨不出去是「擺動腿慣量 vs 位置伺服剛度」。
   證偽方式：把增益加硬，看前腳執行率會不會上去。若不動，假設就錯了。

   同時掃 omega（放慢步頻＝給更多時間），這是另一個獨立的檢查方式：
   若是慣量問題，放慢也該讓執行率上去。
"""
import sys
import numpy as np
sys.path.insert(0, 'task7/inference')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm
import local_infer_max as li

B = gb.BASELINE


def run(kp_scale=1.0, omega=None, secs=16.0):
    omega = omega if omega is not None else B['omega']
    r = cw.Robot()
    r.kp = np.tile(mm.KP3, 4) * kp_scale
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    stepf = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    q0 = cpg_max.stand_targets(ks, f0, B['x_off'])
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q0)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    n = int(secs / mm.CTRL_DT)
    mux, muy = np.full(4, B['mu_x']), np.full(4, B['mu_y'])
    omv = np.full(4, omega)
    sw_prev = np.zeros(4, bool)
    s_foot = np.zeros(4); s_body = np.zeros(4); s_cmd = np.zeros(4)
    rec = [[] for _ in range(4)]
    x0 = float(r.d.qpos[0])
    for i in range(n):
        c = stepf(c, mux, muy, omv, mm.CTRL_DT)
        tgt = cpg_max.foot_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                   B['d_step_y'], B['duty'], B['z_sag'])
        q, _ = cpg_max.joint_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                     B['d_step_y'], B['duty'], ks, B['z_sag'])
        r.step(q)
        th = cpg_max.duty_remap(c["theta"], B['duty'])
        sw = np.sin(th) > 0
        fx = np.array([r.d.xpos[b][0] for b in r.foot_bid])
        bx = float(r.d.qpos[0])
        for k in range(4):
            if sw[k] and not sw_prev[k]:
                s_foot[k], s_body[k], s_cmd[k] = fx[k], bx, tgt[k, 0]
            elif (not sw[k]) and sw_prev[k] and i > n // 3:
                rec[k].append((fx[k] - s_foot[k] - (bx - s_body[k]), tgt[k, 0] - s_cmd[k]))
        sw_prev = sw
    ex = []
    for k in range(4):
        a = np.asarray(rec[k])
        ex.append(a[:, 0].mean() / a[:, 1].mean())
    return ex, (float(r.d.qpos[0]) - x0) / secs


print(f"{'條件':<22}{'FR':>8}{'FL':>8}{'RR':>8}{'RL':>8}{'前/後':>9}{'帳面速度':>10}")
for tag, kw in (("kp ×1（基準）", {}), ("kp ×2", dict(kp_scale=2.0)),
                ("kp ×4", dict(kp_scale=4.0)), ("kp ×8", dict(kp_scale=8.0)),
                ("ω 0.7（半速）", dict(omega=0.7)), ("ω 0.35（1/4 速）", dict(omega=0.35))):
    e, v = run(**kw)
    print(f"{tag:<22}{e[0]:>8.2f}{e[1]:>8.2f}{e[2]:>8.2f}{e[3]:>8.2f}"
          f"{np.mean(e[:2])/np.mean(e[2:]):>9.2f}{v:>10.3f}")
