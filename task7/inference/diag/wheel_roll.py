"""逐輪淨滾動 + 逐腳法向力。驗證「前輪被拖著滾」這個推論。

預測（若成立）：前輪淨滾動明顯為正（往前滾）、後輪為負，兩者在總和裡幾乎抵銷。
若前後輪滾動量相近，推論就錯了。
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
    w_prev = r.d.qpos[mm.WHEEL_QPOS_IDX].copy()
    roll = np.zeros(4)          # 累積滾動（折回 ±π，輪角讀數是包裹的）
    forces = np.zeros(4)
    nf = 0
    for i in range(n):
        a = infer(obs_max.build_obs(r.d, c, cmd, last_a))
        mux, muy, om = li.act_to_cmd(a)
        c = step(c, mux, muy, om, mm.CTRL_DT)
        q, _ = cpg_max.joint_targets(c, f0, B['x_off'], B['g_c'], B['d_step'],
                                     B['d_step_y'], B['duty'], ks, B['z_sag'])
        r.step(q)
        last_a = a
        w = r.d.qpos[mm.WHEEL_QPOS_IDX]
        # ⚠️ 輪角讀數是包裹在 [−π, π] 的（realbot/shm_io.wrap_pi 的同一個坑）：
        #    轉超過半圈時直接相減會給出完全錯誤的值。逐步折回增量。
        roll += (w - w_prev + np.pi) % (2 * np.pi) - np.pi
        w_prev = w.copy()
        if i >= n // 2:
            forces += r.foot_forces()
            nf += 1
    adv = float(r.d.qpos[0])
    dist = -roll * mm.WHEEL_RADIUS * 1000      # 前進對應輪角減少
    print(f"\n=== {tag} ===  機身前進 {adv*1000:.0f} mm")
    print(f"{'腿':>4}{'★淨滾動 mm':>14}{'平均法向力 N':>15}")
    for k, L in enumerate(mm.LEGS):
        print(f"{L:>4}{dist[k]:>14.0f}{forces[k]/nf:>15.1f}")
    print(f"  前輪合計 {dist[:2].sum():+.0f} mm ／ 後輪合計 {dist[2:].sum():+.0f} mm"
          f" ／ 四輪平均 {dist.mean():+.0f} mm")
    print(f"  前腳/後腳 承重比 {forces[:2].sum()/forces[2:].sum():.3f}"
          f"（總 {forces.sum()/nf:.0f} N，體重 380.8 N）")


probe('開迴路基準', False)
probe('RL policy', True)
