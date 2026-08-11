"""gait_export 的離線管線測試。不碰硬體。"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "inference"))
sys.path.insert(0, str(_ROOT / "realbot"))

import calib_map
import cpg_walk_d1 as W
import d1_model
import gait_export as GE


@pytest.fixture(scope="module")
def model():
    return d1_model.make_model()


def test_shm_limits_has_all_twelve_axes(model):
    lim = GE.shm_limits(model)
    assert set(lim) == {(leg, jn) for leg in range(4) for jn in GE.JN}


def test_shm_limits_swaps_bounds_when_sign_is_negative(model):
    """sign=-1 的關節，MJCF 的上界會變成 SHM 的下界。寫錯這裡限位檢驗就整個失效。"""
    lim = GE.shm_limits(model)
    for shm_leg in range(4):
        for jn in GE.JN:
            lo, hi = lim[(shm_leg, jn)]
            assert lo < hi, f"leg{shm_leg}.{jn} 上下界顛倒：{lo} !< {hi}"

    # leg0 = FR，其 knee 的 sign 是 -1（見 calib_map.CALIB）
    assert calib_map.CALIB[0]["knee"][0] == -1
    s, o = calib_map.CALIB[0]["knee"]
    mjcf_lo, mjcf_hi = -2.7030, -0.6220          # FR_knee 的 ctrlrange
    lo, hi = lim[(0, "knee")]
    # sign=-1：MJCF 下界映到 SHM 上界
    assert hi == pytest.approx(s * mjcf_lo + o, abs=1e-6)
    assert lo == pytest.approx(s * mjcf_hi + o, abs=1e-6)


def test_calib_map_round_trips():
    """mjcf → shm → mjcf 必須還原。sign/offset 寫錯的話這裡會先炸，
    而不是等到實機上腿往反方向甩。"""
    rng = np.random.default_rng(0)
    q12 = rng.uniform(-1.0, 1.0, 12)
    res = calib_map.mjcf12_to_shm(q12)
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(GE.JN):
            s, o = calib_map.CALIB[shm_leg][jn]
            back = (res[shm_leg][jn] - o) / s
            assert back == pytest.approx(q12[mjcf_leg * 3 + j], abs=1e-12)


def test_leg_mjcf2shm_is_a_permutation():
    """腿序重排寫錯 = 每條腿的指令都送到別條腿去，而且不會報錯。"""
    assert sorted(calib_map.LEG_MJCF2SHM) == [0, 1, 2, 3]
    # policy 腿序 (FL,FR,RL,RR) → SHM (FR,FL,RR,RL)
    assert calib_map.LEG_MJCF2SHM == [1, 0, 3, 2]


def test_captured_stand_pose_lies_inside_shm_limits(model):
    """POSE_STAND 是從這台實機擷取的，必須落在推導出來的限位內；
    否則代表 sign/offset 或限位轉換有錯。"""
    import shm_common as SC
    lim = GE.shm_limits(model)
    for shm_leg in range(4):
        for jn in GE.JN:
            lo, hi = lim[(shm_leg, jn)]
            v = SC.POSE_STAND[shm_leg][jn]
            assert lo <= v <= hi, f"leg{shm_leg}.{jn} 站姿 {v} 不在 [{lo}, {hi}]"


def test_build_trajectory_shapes_and_leg_order(model):
    q_mjcf, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=2.0)
    n = int(2.0 / d1_model.CTRL_DT)
    assert q_mjcf.shape == (n, 12)
    assert q_shm.shape == (n, 4, 3)
    # q_shm 必須是 q_mjcf 經 sign/offset + 腿序重排的結果
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(GE.JN):
            s, o = calib_map.CALIB[shm_leg][jn]
            assert q_shm[:, shm_leg, j] == pytest.approx(
                s * q_mjcf[:, mjcf_leg * 3 + j] + o, abs=1e-12)


def test_mu_y_1_5_means_abad_never_moves(model):
    """μy=1.5 → fy=0 → dy=0。abad 不動是直線走路的前提，也是限位餘裕的來源。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=5.0)
    abad = q_mjcf[:, ::3]
    assert np.ptp(abad) < 1e-9


def test_deploy_g_c_meets_margin_threshold(model):
    """DEPLOY_G_C 必須通過餘裕門檻——這是選它的唯一理由。"""
    _, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    margin, where = GE.worst_margin(q_shm, GE.shm_limits(model))
    assert margin >= GE.MARGIN_MIN, f"{where} 餘裕只有 {margin:.4f}"


def test_video_g_c_would_fail_the_margin_threshold(model):
    """釘住我們為什麼不用影片那組參數。這個測試轉綠代表門檻或校正被改動了。"""
    _, q_shm = GE.build_trajectory(model, W.GAIT_G_C, secs=20.0)
    margin, _ = GE.worst_margin(q_shm, GE.shm_limits(model))
    assert margin < GE.MARGIN_MIN


def test_worst_margin_identifies_the_knee(model):
    _, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    _, where = GE.worst_margin(q_shm, GE.shm_limits(model))
    assert "knee" in where


def test_max_joint_vel_far_exceeds_l4_threshold(model):
    """步態需要 ~13.5 rad/s，L4 的 VEL_ABORT=2.0 直接搬會一路誤中止。"""
    _, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    v = GE.max_joint_vel(q_shm)
    assert 12.0 < v < 15.0
