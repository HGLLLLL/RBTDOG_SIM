"""線追蹤控制律純函式單元測試。
run: conda run -n rbtdog python task4/inference/tests/test_line_control.py"""
import sys
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
from local_infer_paper import line_frame, line_control

K_YAW, K_CT = 3.0, 1.5


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def main():
    psi = np.pi / 4
    d, n = line_frame(psi)
    assert approx(np.hypot(*d), 1.0) and approx(np.hypot(*n), 1.0)
    assert approx(float(d @ n), 0.0)                       # 方向與法向正交

    p0 = np.array([0.0, 0.0])
    # 在線上且航向對齊 → 零橫向、零轉向、前進保留
    cmd, e_ct, e_yaw = line_control(p0, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert approx(e_ct, 0.0) and approx(e_yaw, 0.0)
    assert approx(float(cmd[1]), 0.0) and approx(float(cmd[2]), 0.0) and approx(float(cmd[0]), 0.6)

    # 偏左（+n 方向 0.1m）→ e_ct>0 → vy<0（螃蟹往右修回）
    cmd, e_ct, _ = line_control(p0 + 0.1 * n, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert e_ct > 0 and cmd[1] < 0 and approx(float(cmd[1]), -0.15, 1e-3), (e_ct, cmd[1])

    # 偏右 → e_ct<0 → vy>0
    cmd, e_ct, _ = line_control(p0 - 0.1 * n, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert e_ct < 0 and cmd[1] > 0

    # 大偏移 → vy 夾到 -0.3
    cmd, _, _ = line_control(p0 + 1.0 * n, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert approx(float(cmd[1]), -0.3)

    # 航向偏左（yaw>target, e_yaw>0）→ wz<0（順時針轉回）
    cmd, _, e_yaw = line_control(p0, psi + 0.1, p0, psi, 0.6, K_YAW, K_CT)
    assert e_yaw > 0 and cmd[2] < 0

    # no_lateral → vy 恆 0（重現舊的只鎖航向行為）
    cmd, _, _ = line_control(p0 + 0.5 * n, psi, p0, psi, 0.6, K_YAW, K_CT, no_lateral=True)
    assert approx(float(cmd[1]), 0.0)

    print("PASS test_line_control")


if __name__ == "__main__":
    main()
