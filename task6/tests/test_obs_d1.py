"""73 維 observation 測試。維度與欄位順序一旦改變，訓練好的權重就失效。"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_d1
import d1_model
import obs_d1


def test_layout_sums_to_73_and_has_no_base_linear_velocity():
    total = sum(n for _, n in obs_d1.OBS_LAYOUT)
    assert total == d1_model.OBS_DIM == 73
    names = [k for k, _ in obs_d1.OBS_LAYOUT]
    assert "base_linvel" not in names, "機身線速度實機拿不到，不得放進 obs"
    assert names == ["gravity", "gyro", "joint_pos", "joint_vel",
                     "cmd", "last_action", "foot_contact", "cpg"]


def test_build_obs_shape_and_dtype():
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    knee_aid = d1_model.knee_actuator_ids(m)
    obs = obs_d1.build_obs(d, cpg_d1.cpg_init(),
                           np.array([0.6, 0.0, 0.0], np.float32),
                           np.zeros(12), knee_aid)
    assert obs.shape == (73,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_joint_pos_block_is_deviation_from_home():
    """home 姿態下關節角區塊應全為 0（存的是相對 HOME12 的偏差）。"""
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    knee_aid = d1_model.knee_actuator_ids(m)
    obs = obs_d1.build_obs(d, cpg_d1.cpg_init(),
                           np.zeros(3, np.float32), np.zeros(12), knee_aid)
    assert obs[6:18] == pytest.approx(np.zeros(12), abs=1e-9)


def test_gravity_block_points_down_when_upright():
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    knee_aid = d1_model.knee_actuator_ids(m)
    obs = obs_d1.build_obs(d, cpg_d1.cpg_init(),
                           np.zeros(3, np.float32), np.zeros(12), knee_aid)
    assert obs[0:3] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_foot_contact_uses_absolute_torque_threshold():
    knee_aid = [2, 5, 8, 11]
    force = np.zeros(12)
    force[2] = d1_model.TAU_CONTACT + 1.0      # 超過門檻 → 觸地
    force[5] = -(d1_model.TAU_CONTACT + 1.0)   # 負號同樣算觸地（取絕對值）
    force[8] = d1_model.TAU_CONTACT - 1.0      # 未達門檻 → 未觸地
    force[11] = 0.0
    got = obs_d1.foot_contact(force, knee_aid)
    assert got == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert got.dtype == np.float32
