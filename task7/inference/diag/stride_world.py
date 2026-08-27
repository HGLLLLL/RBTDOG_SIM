"""每一腳在**擺動相**實際往前跨多少（世界座標）—— 「向前踏」最沒歧義的定義。

擺動相判定用 duty_remap 後的 sin(th) > 0（＝軌跡公式自己的定義），
不用離地高度門檻（那在會彈跳的步態上會騙人）。
"""
import sys
import numpy as np
sys.path.insert(0, 'task7/inference')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm, obs_max
import local_infer_max as li

P = 'task7/weights/cpg_rl_max_params.pkl'
B = gb.BASELINE


def probe(tag, use_policy, secs=20.0):
    r = cw.Robot()
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    infer = li.load_policy(P) if use_policy else (lambda _o: li.baseline_action())
    q0 = cpg_max.stand_targets(ks, f0, B['x_off'])
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q0)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    n = int(secs / mm.CTRL_DT)
    cmd, last_a = np.array([0.15, 0.0]), np.zeros(12)
    sw_prev = np.zeros(4, bool)
    start_x = np.zeros(4)
    steps = [[] for _ in range(4)]      # 每次擺動的世界 x 前進量
    body_x0 = float(r.d.qpos[0])
    for i in range(n):
        a = infer(obs_max.build_obs(r.d, c, cmd, last_a))
        mux, muy, om = li.act_to_cmd(a)
        c = step(c, mux, muy, om, mm.CTRL_DT)
        q, _ = cpg_max.joint_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                     B['d_step_y'], B['duty'], ks, B['z_sag'])
        r.step(q)
        last_a = a
        th = cpg_max.duty_remap(c["theta"], B['duty'])
        sw = np.sin(th) > 0
        fx = np.array([r.d.xpos[b][0] for b in r.foot_bid])
        for k in range(4):
            if sw[k] and not sw_prev[k]:
                start_x[k] = fx[k]
            elif (not sw[k]) and sw_prev[k] and i > n // 4:
                steps[k].append(fx[k] - start_x[k])
        sw_prev = sw
    adv = float(r.d.qpos[0]) - body_x0
    print(f"\n=== {tag} ===  機身前進 {adv*1000:.0f} mm")
    print(f"{'腿':>4}{'擺動次數':>10}{'★每步前跨 mm':>16}{'標準差':>9}")
    m = []
    for k, L in enumerate(mm.LEGS):
        s = np.array(steps[k]) * 1000
        m.append(s.mean())
        print(f"{L:>4}{len(s):>10}{s.mean():>16.1f}{s.std():>9.1f}")
    print(f"  前腳平均 {np.mean(m[:2]):.1f} mm ／ 後腳平均 {np.mean(m[2:]):.1f} mm"
          f"  → 比 {np.mean(m[:2])/np.mean(m[2:]):.3f}")


probe('開迴路基準', False)
probe('RL policy', True)
