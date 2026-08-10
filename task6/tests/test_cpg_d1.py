"""CPG / IK 純函式測試。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_d1
import d1_model


def test_cpg_init_shapes_and_trot_phase():
    c = cpg_d1.cpg_init()
    for k in ("rx", "rx_d", "ry", "ry_d", "theta"):
        assert c[k].shape == (4,), k
    # trot：FL 與 RR 同相、FR 與 RL 同相、兩組差 pi
    assert c["theta"] == pytest.approx([0.0, np.pi, np.pi, 0.0])


def test_cpg_amplitude_converges_to_mu():
    """rx 應收斂到指令 mu。"""
    c = cpg_d1.cpg_init()
    mu = np.full(4, 1.8)
    for _ in range(200):
        c = cpg_d1.cpg_step(c, mu, mu, np.zeros(4), d1_model.CTRL_DT)
    assert c["rx"] == pytest.approx(mu, abs=1e-3)
    assert c["ry"] == pytest.approx(mu, abs=1e-3)


def test_cpg_phase_advances_with_omega():
    """omega=1 Hz 跑 0.25 秒，相位應前進約 pi/2。"""
    c = cpg_d1.cpg_init()
    om = np.ones(4)
    for _ in range(int(0.25 / d1_model.CTRL_DT)):
        c = cpg_d1.cpg_step(c, np.full(4, 1.5), np.full(4, 1.5), om, d1_model.CTRL_DT)
    assert c["theta"][0] == pytest.approx(np.pi / 2, abs=0.15)


def test_act_to_cmd_saturates_into_declared_ranges():
    lo = cpg_d1.act_to_cmd(np.full(12, -50.0))
    hi = cpg_d1.act_to_cmd(np.full(12, +50.0))
    assert lo[0] == pytest.approx(np.full(4, d1_model.MU_MIN), abs=1e-6)
    assert hi[0] == pytest.approx(np.full(4, d1_model.MU_MAX), abs=1e-6)
    assert lo[2] == pytest.approx(np.full(4, d1_model.OMEGA_MIN), abs=1e-6)
    assert hi[2] == pytest.approx(np.full(4, d1_model.OMEGA_MAX), abs=1e-6)


def test_leg_ik_consts_home_wheel_is_below_hip():
    """home 姿態下輪心應在髖正下方約 0.224 m，並向外偏 0.142 m。

    實測基準：f0 = (0.0003, ±0.1423, -0.2238)，Jacobian 條件數 2.6。
    y 的 0.142 m 包含輪子相對小腿末端向外 4.5 cm 的安裝偏移，抄漏會讓步幅算錯。
    """
    m = d1_model.make_model()
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    assert f0s.shape == (4, 3)
    assert jinvs.shape == (4, 3, 3)
    for k in range(4):
        assert abs(f0s[k][0]) < 0.02, f"腿 {k} 的輪心未在髖正下方 (x={f0s[k][0]:.3f})"
        assert -0.24 < f0s[k][2] < -0.20, f"腿 {k} 髖到輪心距離異常 (z={f0s[k][2]:.3f})"
        assert 0.13 < abs(f0s[k][1]) < 0.155, f"腿 {k} 輪子橫向偏移異常 (y={f0s[k][1]:.3f})"
    left = [f0s[0][1], f0s[2][1]]
    right = [f0s[1][1], f0s[3][1]]
    assert all(v > 0 for v in left) and all(v < 0 for v in right), "左右腿的 y 偏移方向反了"


def test_ik_jacobian_moves_foot_in_requested_direction():
    """把輪心往 +x 推 2 cm，正向運動學算回來的位移應吻合（實測誤差 0.89 mm）。"""
    import mujoco

    m = d1_model.make_model()
    d = mujoco.MjData(m)
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    want = np.array([0.02, 0.0, 0.0])
    q3 = d1_model.HOME3 + jinvs[0] @ want

    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[7:10] = q3
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FL_abad")
    got = d.geom_xpos[gid] - d.xpos[hid] - f0s[0]
    assert got == pytest.approx(want, abs=3e-3)


def test_joint_targets_returns_12_and_stays_in_limits():
    import mujoco

    m = d1_model.make_model()
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    c = cpg_d1.cpg_init()
    om = np.full(4, 2.0)
    for _ in range(100):
        c = cpg_d1.cpg_step(c, np.full(4, 2.0), np.full(4, 2.0), om, d1_model.CTRL_DT)
        q = cpg_d1.joint_targets(c, f0s, jinvs)
        assert q.shape == (12,)
        assert np.all(q >= m.jnt_range[1:, 0] - 1e-6), "關節目標角低於下限"
        assert np.all(q <= m.jnt_range[1:, 1] + 1e-6), "關節目標角超過上限"


def test_w2b_rotates_gravity_into_body_frame():
    """機身繞 y 轉 90 度後，世界的 -z 重力在機身系應變成 +x 方向。"""
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    got = cpg_d1.w2b(q, np.array([0.0, 0.0, -1.0]))
    assert got == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
