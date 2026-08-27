"""把「每步前跨」拆成兩塊：機身前進 + 足端相對機身的移動。

  足端世界位移 = 機身位移 + 足端相對機身的位移

若前腳的「相對機身位移」遠小於指令，那是**執行問題**（追蹤/接觸）；
若相對位移對得上指令，那前腳沒問題，問題在別處。
"""
import sys
import numpy as np
sys.path.insert(0, 'task7/inference')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm, obs_max
import local_infer_max as li

B = gb.BASELINE


def probe(tag, use_policy, secs=20.0):
    r = cw.Robot()
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    stepf = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    infer = (li.load_policy('task7/weights/cpg_rl_max_params.pkl') if use_policy
             else (lambda _o: li.baseline_action()))
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
    s_foot = np.zeros(4); s_body = np.zeros(4); s_cmd = np.zeros(4)
    rec = [[] for _ in range(4)]        # (足端世界, 機身, 指令相對)
    for i in range(n):
        a = infer(obs_max.build_obs(r.d, c, cmd, last_a))
        mux, muy, om = li.act_to_cmd(a)
        c = stepf(c, mux, muy, om, mm.CTRL_DT)
        tgt = cpg_max.foot_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                   B['d_step_y'], B['duty'], B['z_sag'])
        q, _ = cpg_max.joint_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                     B['d_step_y'], B['duty'], ks, B['z_sag'])
        r.step(q)
        last_a = a
        th = cpg_max.duty_remap(c["theta"], B['duty'])
        sw = np.sin(th) > 0
        fx = np.array([r.d.xpos[b][0] for b in r.foot_bid])
        bx = float(r.d.qpos[0])
        for k in range(4):
            if sw[k] and not sw_prev[k]:
                s_foot[k], s_body[k], s_cmd[k] = fx[k], bx, tgt[k, 0]
            elif (not sw[k]) and sw_prev[k] and i > n // 4:
                rec[k].append((fx[k] - s_foot[k], bx - s_body[k], tgt[k, 0] - s_cmd[k]))
        sw_prev = sw

    print(f"\n=== {tag} ===")
    print(f"{'腿':>4}{'足端世界':>10}{'= 機身':>9}{'+ 實際相對':>12}"
          f"{'｜指令相對':>12}{'執行率':>9}")
    for k, L in enumerate(mm.LEGS):
        a_ = np.asarray(rec[k]) * 1000
        foot, body, cmdrel = a_.mean(0)
        actrel = foot - body
        print(f"{L:>4}{foot:>10.1f}{body:>9.1f}{actrel:>12.1f}"
              f"{cmdrel:>12.1f}{actrel / cmdrel:>9.2f}")


probe('開迴路基準', False)
probe('RL policy', True)
