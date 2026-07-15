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

if __name__ == "__main__":
    for fn in TESTS:
        fn(); print("ok", fn.__name__)
    print(f"ALL {len(TESTS)} TESTS PASSED")
