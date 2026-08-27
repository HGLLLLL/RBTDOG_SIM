"""kp=250/kd5 之下重掃 z_sag（它與 KP3 綁死，換增益一定要重量）。"""
import sys
import numpy as np
sys.path.insert(0, 'task7/inference')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm

B = gb.BASELINE
KP, KD = [250, 250, 250], [5, 5, 5]


def static_sag():
    r = cw.Robot(); r.kp = np.tile(KP, 4).astype(float); r.kd = np.tile(KD, 4).astype(float)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    q = cpg_max.stand_targets(ks, f0, 0.0)
    r.reset_standing(q, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(4.0 / mm.CTRL_DT)):
        r.step(q)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    return mm.NOMINAL_HEIGHT_KIN - float(r.d.qpos[2])


def run(z_sag, x_off=None, secs=20.0):
    x_off = B['x_off'] if x_off is None else x_off
    r = cw.Robot(); r.kp = np.tile(KP, 4).astype(float); r.kd = np.tile(KD, 4).astype(float)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    stepf = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    q0 = cpg_max.stand_targets(ks, f0, x_off)
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q0)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    n = int(secs / mm.CTRL_DT)
    tr = cw.Trace(r, n, secs, B['omega'], cpg_max.PHASE_WALK)
    mux, muy, omv = np.full(4, B['mu_x']), np.full(4, B['mu_y']), np.full(4, B['omega'])
    sw_prev = np.zeros(4, bool); sf = np.zeros(4); sb = np.zeros(4); sc = np.zeros(4)
    rec = [[] for _ in range(4)]; nre = 0
    for i in range(n):
        c = stepf(c, mux, muy, omv, mm.CTRL_DT)
        tgt = cpg_max.foot_targets(c, f0, x_off, B['g_c'], B['d_step'],
                                   B['d_step_y'], B['duty'], z_sag)
        q, nc = cpg_max.joint_targets(c, f0, x_off, B['g_c'], B['d_step'],
                                      B['d_step_y'], B['duty'], ks, z_sag)
        nre += nc
        r.step(q); tr.record(c["theta"])
        th = cpg_max.duty_remap(c["theta"], B['duty']); sw = np.sin(th) > 0
        fx = np.array([r.d.xpos[b][0] for b in r.foot_bid]); bx = float(r.d.qpos[0])
        for k in range(4):
            if sw[k] and not sw_prev[k]:
                sf[k], sb[k], sc[k] = fx[k], bx, tgt[k, 0]
            elif (not sw[k]) and sw_prev[k] and i > n // 3:
                rec[k].append((fx[k] - sf[k] - (bx - sb[k]), tgt[k, 0] - sc[k]))
        sw_prev = sw
    res = tr.summarize(nre)
    ex = [np.asarray(rec[k])[:, 0].mean() / np.asarray(rec[k])[:, 1].mean() for k in range(4)]
    return ex, res


print(f"kp=250/kd=5 的靜態撓度 = {static_sag()*1000:.1f} mm"
      f"（kp=120/kd=1 時是 32.5 mm）\n")
print(f"{'z_sag mm':>10}{'前腳執行':>9}{'後腳執行':>9}{'★行進m/s':>10}{'彈跳mm':>9}"
      f"{'支撐':>8}{'離地mm':>9}{'平均俯仰':>9}{'跌倒':>6}")
for z in (0.0, 0.010, 0.0175, 0.025, 0.0325, 0.045):
    ex, res = run(z)
    print(f"{z*1000:>10.1f}{np.mean(ex[:2]):>9.2f}{np.mean(ex[2:]):>9.2f}"
          f"{res['speed_travel']:>10.3f}{res['bounce']*1000:>9.1f}{res['support']:>8.2f}"
          f"{res['min_lift']*1000:>9.1f}{res['pitch_mean']:>+9.2f}"
          f"{('是' if res['fell'] else '否'):>6}")
