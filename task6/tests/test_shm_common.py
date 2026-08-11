"""shm_common 的結構契約與純函式測試。不碰硬體。"""
import ctypes
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import shm_common as SC


def test_spline_data_size_is_608():
    """結構大小對不上 = 我們對 SHM 的理解跟 daemon 不一致，寫下去會寫到錯的欄位。"""
    assert ctypes.sizeof(SC.SplineData) == 608
    assert SC.EXPECT_SIZE == 608


def test_cmd_and_state_sub_sizes():
    assert ctypes.sizeof(SC.SplineCmd) == 344
    assert ctypes.sizeof(SC.SplineState) == 264


def test_joint_control_field_order():
    """欄位順序寫錯會靜默把 kp 寫進 v_des。"""
    assert [n for n, _ in SC.JointControl._fields_] == \
        ["p_des", "v_des", "kp", "kd", "t_ff"]
    assert [n for n, _ in SC.JointState._fields_] == ["flags", "p", "v", "t"]


def test_legname_matches_shm_leg_order():
    """SHM 腿序：0=FR 1=FL 2=RR 3=RL。實機已確認 leg0=FR、leg2=RR。"""
    assert SC.LEGNAME == {0: "FR", 1: "FL", 2: "RR", 3: "RL"}


def test_zero_all_clears_every_joint_and_sets_flags():
    d = SC.SplineData()
    d.cmd.legs[1].hip.kp = 99.0
    d.cmd.legs[3].foot.t_ff = 5.0
    SC.zero_all(d)
    for i in range(4):
        assert d.cmd.legs[i].flags == 1
        for jn in ("abad", "hip", "knee", "foot"):
            j = getattr(d.cmd.legs[i], jn)
            assert (j.p_des, j.v_des, j.kp, j.kd, j.t_ff) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_set_leg_position_leaves_wheel_at_zero_gain():
    """輪子(foot)全程零增益 —— 本專案從不對輪做位置控制。"""
    d = SC.SplineData()
    SC.zero_all(d)
    SC.set_leg_position(d, 2, 0.1, 0.2, 0.3, kp=20.0, kd=0.7)
    leg = d.cmd.legs[2]
    assert leg.abad.p_des == pytest.approx(0.1)
    assert leg.hip.p_des == pytest.approx(0.2)
    assert leg.knee.p_des == pytest.approx(0.3)
    assert leg.knee.kp == pytest.approx(20.0)
    assert leg.foot.kp == 0.0 and leg.foot.kd == 0.0


def test_check_guards_uses_passed_thresholds_and_ignores_skipped_legs():
    """被跳過的腿若故障，其 t/v 是損壞資料，不得拿來當保護判準。"""
    d = SC.SplineData()
    d.state.legs[2].foot.t = 99.0      # RR 故障中的垃圾值
    ok, why = SC.check_guards(d, (0, 1, 3), torque_abort=8.0, vel_abort=2.0)
    assert ok, why
    ok, why = SC.check_guards(d, (0, 1, 2, 3), torque_abort=8.0, vel_abort=2.0)
    assert not ok and "RR.foot" in why


def test_check_guards_threshold_is_a_parameter_not_a_constant():
    d = SC.SplineData()
    d.state.legs[0].knee.v = 5.0
    assert SC.check_guards(d, (0,), torque_abort=8.0, vel_abort=2.0)[0] is False
    assert SC.check_guards(d, (0,), torque_abort=8.0, vel_abort=20.0)[0] is True


def test_preflight_motors_healthy_flags_not_ready_and_dead_can():
    d = SC.SplineData()
    for i in range(4):
        for jn in ("abad", "hip", "knee", "foot"):
            # ready=1, 溫度 30C, 電壓 44V
            getattr(d.state.legs[i], jn).flags = 1 | (30 << 8) | (44 << 16)
    assert SC.preflight_motors_healthy(d, (0, 1, 2, 3))[0]

    d.state.legs[2].foot.flags = (29 << 8) | (44 << 16)      # ready=0
    ok, problems = SC.preflight_motors_healthy(d, (0, 1, 2, 3))
    assert not ok and any("RR.foot" in p and "ready=0" in p for p in problems)

    ok, _ = SC.preflight_motors_healthy(d, (0, 1, 3))         # 跳過 RR
    assert ok


def test_pose_stand_and_lie_are_per_leg_mirrored():
    """左右腿編碼器慣例鏡像，所以每腿一組，不是四腿共用。"""
    for pose in (SC.POSE_STAND, SC.POSE_LIE):
        assert set(pose) == {0, 1, 2, 3}
        # leg0(FR) 與 leg1(FL) 的 abad 號相反
        assert pose[0]["abad"] * pose[1]["abad"] < 0
