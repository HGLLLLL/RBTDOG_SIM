"""RCLineController 單元測試（獨立 assert 腳本，conda run -n rbtdog python 執行）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rc_line_controller import wrap, line_frame, line_control, Sticks, Odom, Config


def t1_wrap():
    assert abs(wrap(2 * np.pi + 0.1) - 0.1) < 1e-9
    assert abs(wrap(-2 * np.pi - 0.1) + 0.1) < 1e-9


def t1_line_frame():
    d, n = line_frame(0.0)
    assert np.allclose(d, [1.0, 0.0]) and np.allclose(n, [0.0, 1.0])


def t1_line_control_values():
    cmd, e_ct, e_yaw = line_control((1.0, 0.5), 0.1, (0.0, 0.0), 0.0, 0.6, 3.0, 1.5)
    assert abs(e_ct - 0.5) < 1e-9
    assert abs(e_yaw - 0.1) < 1e-9
    assert abs(cmd[0] - 0.6) < 1e-6
    assert abs(cmd[1] + 0.3) < 1e-6            # -1.5*0.5=-0.75 → clip ±0.3
    assert abs(cmd[2] + 0.3) < 1e-6            # -3.0*0.1
    cmd0, _, _ = line_control((2.0, 0.0), 0.0, (0.0, 0.0), 0.0, 0.6, 3.0, 1.5)
    assert np.allclose(cmd0, [0.6, 0.0, 0.0])  # 在線上且對準 → 無修正


def t1_parity_with_task4():
    sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
    import local_infer_paper as P
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(-5, 5, 2); p0 = rng.uniform(-5, 5, 2)
        yaw = rng.uniform(-4, 4); psi = rng.uniform(-4, 4)
        a = line_control(p, yaw, p0, psi, 0.6, 3.0, 1.5)
        b = P.line_control(p, yaw, p0, psi, 0.6, 3.0, 1.5)
        assert np.allclose(a[0], b[0]) and abs(a[1] - b[1]) < 1e-12 and abs(a[2] - b[2]) < 1e-12


def t1_config_defaults():
    c = Config()
    assert c.vmax == 0.6 and c.k_yaw == 3.0 and c.k_ct == 1.5
    assert c.dead_on == 0.08 and c.dead_off == 0.04
    assert c.settle_s == 0.3 and c.stale_s == 0.5
    s = Sticks(); o = Odom(1.0, 2.0, 0.5, 3.0)
    assert s.fwd == 0.0 and o.stamp == 3.0


TESTS = [t1_wrap, t1_line_frame, t1_line_control_values, t1_parity_with_task4,
         t1_config_defaults]


from rc_line_controller import _StickGate, _YawRateEst


def t2_stick_gate_hysteresis():
    g = _StickGate(0.08, 0.04)
    assert g.update(0.05) is False           # 未過離中門檻
    assert g.update(0.09) is True
    assert g.update(0.05) is True            # 遲滯：未低於回中門檻
    assert g.update(-0.09) is True           # 負向同樣算離中
    assert g.update(0.03) is False


def t2_yaw_rate_estimator():
    e = _YawRateEst(0.03, 0.1)
    t = 0.0
    for _ in range(50):                      # 1s 以 0.5 rad/s 旋轉
        e.update(0.5 * t, t); t += 0.02
    assert abs(e.rate - 0.5) < 0.05
    assert e.stable_for(t) == 0.0            # 旋轉中不穩定
    yaw1 = 0.5 * t
    for _ in range(20):                      # 0.4s 靜止
        e.update(yaw1, t); t += 0.02
    assert abs(e.rate) < 0.1
    assert e.stable_for(t) > 0.2             # 停穩後開始累計
    e.reset()
    assert e.stable_for(t) == 0.0


def t2_yaw_rate_wrap_crossing():
    e = _YawRateEst(0.03, 0.1)
    t, yaw = 0.0, 3.1                        # 跨 ±pi 不爆
    for _ in range(50):
        e.update(wrap(yaw), t); yaw += 0.5 * 0.02; t += 0.02
    assert abs(e.rate - 0.5) < 0.05


TESTS += [t2_stick_gate_hysteresis, t2_yaw_rate_estimator, t2_yaw_rate_wrap_crossing]

from rc_line_controller import RCLineController, MANUAL, SETTLING, TRACKING

CTRL_DT = 0.02


def drive(ctrl, secs, sticks_fn, odom_fn, t0=0.0):
    """以 50Hz 餵 secs 秒；sticks_fn/odom_fn 都吃時間 t。回傳 (最後 cmd, 結束時間)。"""
    t, cmd = t0, None
    for _ in range(int(round(secs / CTRL_DT))):
        cmd = ctrl.update(sticks_fn(t), odom_fn(t), t)
        t += CTRL_DT
    return cmd, t


def still(t):
    return Odom(0.0, 0.0, 0.0, t)


def t3_neutral_stays_manual_zero():
    ctrl = RCLineController()
    cmd, _ = drive(ctrl, 1.0, lambda t: Sticks(), still)
    assert ctrl.state == MANUAL and ctrl.latch is None
    assert np.allclose(cmd, [0.0, 0.0, 0.0])


def t3_standstill_push_fwd_latch_immediate():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(), still)
    cmd = ctrl.update(Sticks(fwd=0.5), still(t), t)
    assert ctrl.state == TRACKING            # 靜止時穩定計時已滿 → 推桿瞬間 latch
    assert np.allclose(ctrl.latch[0], [0.0, 0.0]) and abs(ctrl.latch[1]) < 1e-9
    assert abs(cmd[0] - 0.3) < 1e-6 and abs(cmd[1]) < 1e-6 and abs(cmd[2]) < 1e-6


def t3_tracking_pulls_back_to_line():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)   # latch 於 y=0 線
    assert ctrl.state == TRACKING
    off = lambda tt: Odom(0.0, 0.2, 0.0, tt)                    # 向左偏 0.2m（連續）
    cmd, _ = drive(ctrl, 1.0, lambda tt: Sticks(fwd=1.0), off, t0=t)
    assert cmd[1] < -0.25                                       # vy 往右拉回（穩態 -0.3）
    assert abs(cmd[2]) < 1e-6                                   # 航向沒偏 → wz=0


def t3_turn_takeover_and_relatch():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    t_spin = t
    spin = lambda tt: Odom(0.0, 0.0, 0.8 * (tt - t_spin), tt)   # 0.8 rad/s 左轉
    cmd, t = drive(ctrl, 0.5, lambda tt: Sticks(fwd=1.0, turn=0.5), spin, t0=t)
    assert ctrl.state == MANUAL and ctrl.latch is None          # 轉向優先接管
    assert abs(cmd[0] - 0.6) < 1e-6 and abs(cmd[2] - 0.5) < 1e-6  # vx 照給、turn 透傳
    cmd, t = drive(ctrl, 0.2, lambda tt: Sticks(fwd=1.0), spin, t0=t)
    assert ctrl.state == SETTLING                               # 桿回中但航向還在滑
    assert abs(cmd[1]) < 1e-6 and abs(cmd[2]) < 1e-6            # 不透傳殘餘、不校正
    yaw1 = 0.8 * (t - t_spin)
    hold = lambda tt: Odom(0.0, 0.0, yaw1, tt)
    _, t = drive(ctrl, 0.6, lambda tt: Sticks(fwd=1.0), hold, t0=t)
    assert ctrl.state == TRACKING
    assert abs(wrap(ctrl.latch[1] - yaw1)) < 1e-9               # 新線鎖在停穩後航向


def t3_fwd_release_invalidates_line():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    assert ctrl.state == TRACKING
    ctrl.update(Sticks(), still(t), t)
    assert ctrl.state == MANUAL and ctrl.latch is None          # fwd 回中 → 線作廢
    ctrl.update(Sticks(fwd=1.0), still(t + CTRL_DT), t + CTRL_DT)
    assert ctrl.state == TRACKING                               # 再推 → 立即重 latch


def t3_backward_still_corrects():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=-1.0), still)  # 倒退也能 latch
    assert ctrl.state == TRACKING
    off = lambda tt: Odom(0.0, 0.2, 0.0, tt)
    cmd, _ = drive(ctrl, 1.0, lambda tt: Sticks(fwd=-1.0), off, t0=t)
    assert cmd[0] < -0.5                                        # vx = -0.6 倒退
    assert cmd[1] < -0.25                                       # 偏左照樣往右修


TESTS += [t3_neutral_stays_manual_zero, t3_standstill_push_fwd_latch_immediate,
          t3_tracking_pulls_back_to_line, t3_turn_takeover_and_relatch,
          t3_fwd_release_invalidates_line, t3_backward_still_corrects]


def t4_none_or_nan_degrades():
    ctrl = RCLineController()
    ctrl.update(Sticks(fwd=1.0), None, 0.0)
    assert ctrl.state == MANUAL and ctrl.degraded
    ctrl.update(Sticks(fwd=1.0), Odom(np.nan, 0.0, 0.0, 0.02), 0.02)
    assert ctrl.state == MANUAL and ctrl.degraded


def t4_stale_degrades_then_recovers():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    assert ctrl.state == TRACKING
    frozen = Odom(0.0, 0.0, 0.0, t)                             # stamp 從此不動
    cmd, t = drive(ctrl, 1.0, lambda tt: Sticks(fwd=1.0), lambda tt: frozen, t0=t)
    assert ctrl.state == MANUAL and ctrl.degraded and ctrl.latch is None
    assert abs(cmd[0] - 0.6) < 1e-6 and abs(cmd[1]) < 1e-6      # 桿量直通
    recov = lambda tt: Odom(0.3, 0.1, 0.0, tt)                  # 恢復（未觸跳變）
    _, t = drive(ctrl, 1.0, lambda tt: Sticks(fwd=1.0), recov, t0=t)
    assert ctrl.state == TRACKING and not ctrl.degraded         # 穩定後重 latch


def t4_jump_invalidates_and_relatches():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    assert ctrl.state == TRACKING
    jumped = lambda tt: Odom(3.0, 2.0, 0.0, tt)                 # 重定位跳 3.6m
    ctrl.update(Sticks(fwd=1.0), jumped(t), t)
    assert ctrl.latch is None and ctrl.state == SETTLING        # 作廢舊線、不猛拉
    _, t = drive(ctrl, 0.6, lambda tt: Sticks(fwd=1.0), jumped, t0=t)
    assert ctrl.state == TRACKING
    assert np.allclose(ctrl.latch[0], [3.0, 2.0])               # 新線鎖在跳後位置


def t4_slew_limits_vy_step():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    cmd = ctrl.update(Sticks(fwd=1.0), Odom(0.0, 0.4, 0.0, t), t)
    assert -0.035 < cmd[1] < -0.025            # 目標 -0.3，但一步只能走 1.5*0.02=0.03


def t4_finish_clips():
    ctrl = RCLineController(Config(slew_vy=0.0, slew_wz=0.0))
    out = ctrl._finish(np.array([9.0, 9.0, -9.0], np.float32), 0.0)
    assert abs(out[0] - 0.6) < 1e-6 and abs(out[1] - 0.3) < 1e-6 and abs(out[2] + 1.0) < 1e-6


TESTS += [t4_none_or_nan_degrades, t4_stale_degrades_then_recovers,
          t4_jump_invalidates_and_relatches, t4_slew_limits_vy_step, t4_finish_clips]

if __name__ == "__main__":
    for fn in TESTS:
        fn(); print("ok", fn.__name__)
    print(f"ALL {len(TESTS)} TESTS PASSED")
