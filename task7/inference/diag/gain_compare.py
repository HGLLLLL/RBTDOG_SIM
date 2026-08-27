"""用有物理依據的增益組再測：原廠 RL 設定檔(60/120/120, kd1) vs 原廠站立實測(250, kd5)。
   同時看步態品質有沒有被犧牲（彈跳/支撐腳/力矩飽和/跌倒）。"""
import sys
import numpy as np
sys.path.insert(0, 'task7/inference')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm

B = gb.BASELINE


def run(tag, kp, kd, secs=20.0):
    r = cw.Robot()
    r.kp = np.tile(np.asarray(kp, float), 4)
    r.kd = np.tile(np.asarray(kd, float), 4)
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
    tr = cw.Trace(r, n, secs, B['omega'], cpg_max.PHASE_WALK)
    mux, muy, omv = np.full(4, B['mu_x']), np.full(4, B['mu_y']), np.full(4, B['omega'])
    sw_prev = np.zeros(4, bool)
    sf, sb, sc = np.zeros(4), np.zeros(4), np.zeros(4)
    rec = [[] for _ in range(4)]
    nre = 0
    for i in range(n):
        c = stepf(c, mux, muy, omv, mm.CTRL_DT)
        tgt = cpg_max.foot_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                   B['d_step_y'], B['duty'], B['z_sag'])
        q, nc = cpg_max.joint_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                      B['d_step_y'], B['duty'], ks, B['z_sag'])
        nre += nc
        r.step(q)
        tr.record(c["theta"])
        th = cpg_max.duty_remap(c["theta"], B['duty'])
        sw = np.sin(th) > 0
        fx = np.array([r.d.xpos[b][0] for b in r.foot_bid])
        bx = float(r.d.qpos[0])
        for k in range(4):
            if sw[k] and not sw_prev[k]:
                sf[k], sb[k], sc[k] = fx[k], bx, tgt[k, 0]
            elif (not sw[k]) and sw_prev[k] and i > n // 3:
                rec[k].append((fx[k] - sf[k] - (bx - sb[k]), tgt[k, 0] - sc[k]))
        sw_prev = sw
    res = tr.summarize(nre)
    ex = [np.asarray(rec[k])[:, 0].mean() / np.asarray(rec[k])[:, 1].mean()
          for k in range(4)]
    print(f"{tag:<26}{np.mean(ex[:2]):>9.2f}{np.mean(ex[2:]):>9.2f}"
          f"{res['speed_travel']:>10.3f}{res['bounce']*1000:>9.1f}{res['support']:>8.2f}"
          f"{res['min_lift']*1000:>9.1f}{res['pitch_mean']:>+9.2f}{res['tau_pct']:>8.2f}"
          f"{('是' if res['fell'] else '否'):>6}")


print(f"{'增益組':<26}{'前腳執行':>9}{'後腳執行':>9}{'★行進m/s':>10}{'彈跳mm':>9}"
      f"{'支撐':>8}{'離地mm':>9}{'平均俯仰':>9}{'飽和%':>8}{'跌倒':>6}")
run('原廠RL設定檔 120/kd1（現況）', [60, 120, 120], [1, 1, 1])
run('原廠站立實測 250/kd5', [250, 250, 250], [5, 5, 5])
run('ABAD維持60 → 60/250/250 kd5', [60, 250, 250], [5, 5, 5])
run('250/kd2', [250, 250, 250], [2, 2, 2])
run('400/kd8', [400, 400, 400], [8, 8, 8])
