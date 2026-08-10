"""69 維 observation 測試。維度與欄位順序一旦改變，訓練好的權重就失效。"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_d1
import d1_model
import obs_d1


def _slice(name: str) -> slice:
    """由 OBS_LAYOUT 累加推導區塊切片，測試不硬寫索引。

    如此一來 OBS_LAYOUT 與 build_obs 只要分岔（欄位對調、長度改動），
    下面的哨兵比對就會失敗。
    """
    off = 0
    for n, k in obs_d1.OBS_LAYOUT:
        if n == name:
            return slice(off, off + k)
        off += k
    raise AssertionError(f"OBS_LAYOUT 沒有欄位 {name}")


class _FakeData:
    """具備 build_obs 所需欄位的假 MjData，用來塞哨兵值。"""

    def __init__(self, qpos, qvel, actuator_force):
        self.qpos = qpos
        self.qvel = qvel
        self.actuator_force = actuator_force


# 哨兵：每個區塊一組互不相同、可辨識的值，任兩區塊對調都會被抓到。
_SEN_LINVEL = np.array([-9.1, -9.2, -9.3])          # 機身線速度：絕不該出現在 obs
_SEN_ACT_FORCE = np.arange(12, dtype=np.float64) + 500.0   # 致動器力矩：同樣不該出現
_SEN_GYRO = np.array([0.31, 0.32, 0.33])
_SEN_JOINT_POS = np.arange(12, dtype=np.float64) + 300.0
_SEN_JOINT_VEL = np.arange(12, dtype=np.float64) + 400.0
_SEN_CMD = np.full(3, 7.0)
_SEN_LAST_A = np.arange(12, dtype=np.float64) + 100.0
_SEN_CPG = {
    "rx":    np.array([1.1, 1.2, 1.3, 1.4]),
    "rx_d":  np.array([2.1, 2.2, 2.3, 2.4]),
    "ry":    np.array([3.1, 3.2, 3.3, 3.4]),
    "ry_d":  np.array([4.1, 4.2, 4.3, 4.4]),
    "theta": np.array([0.3, 1.1, 2.0, 3.5]),        # sin/cos 兩兩互異
}


def _sentinel_obs():
    qpos = np.zeros(19)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]                # 單位四元數 → gravity = (0,0,-1)
    qpos[7:19] = d1_model.HOME12 + _SEN_JOINT_POS   # build_obs 會減掉 HOME12
    qvel = np.zeros(18)
    qvel[0:3] = _SEN_LINVEL
    qvel[3:6] = _SEN_GYRO
    qvel[6:18] = _SEN_JOINT_VEL
    d = _FakeData(qpos, qvel, _SEN_ACT_FORCE.copy())
    obs = obs_d1.build_obs(d, {k: v.copy() for k, v in _SEN_CPG.items()},
                           _SEN_CMD, _SEN_LAST_A)
    return d, obs


def test_build_obs_length_matches_layout_and_obs_dim():
    _, obs = _sentinel_obs()
    assert obs.size == sum(k for _, k in obs_d1.OBS_LAYOUT)
    assert obs.size == d1_model.OBS_DIM


def test_sentinel_gravity_block():
    _, obs = _sentinel_obs()
    assert obs[_slice("gravity")] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_sentinel_gyro_block_is_angular_not_linear_velocity():
    """gyro 必須來自 qvel[3:6]；qvel[0:3]（機身線速度）實機拿不到，不得入 obs。"""
    d, obs = _sentinel_obs()
    got = obs[_slice("gyro")]
    assert got == pytest.approx(d.qvel[3:6], rel=1e-6)
    assert got != pytest.approx(d.qvel[0:3], rel=1e-6), \
        "gyro 區塊取到了機身線速度 qvel[0:3]"
    # 更強的保險：整個 obs 裡不該出現任何一個線速度分量
    for i, v in enumerate(_SEN_LINVEL):
        assert not np.any(np.isclose(obs, v, rtol=1e-5)), \
            f"obs 出現機身線速度分量 qvel[{i}]={v}"


def test_sentinel_joint_pos_and_vel_blocks_not_swapped():
    _, obs = _sentinel_obs()
    assert obs[_slice("joint_pos")] == pytest.approx(_SEN_JOINT_POS, rel=1e-6)
    assert obs[_slice("joint_vel")] == pytest.approx(_SEN_JOINT_VEL, rel=1e-6)


def test_sentinel_cmd_and_last_action_blocks_not_swapped():
    """cmd 與 last_action 填互異值：兩者對調（或與 OBS_LAYOUT 不同序）就會失敗。"""
    _, obs = _sentinel_obs()
    assert obs[_slice("cmd")] == pytest.approx(_SEN_CMD, rel=1e-6)
    assert obs[_slice("last_action")] == pytest.approx(_SEN_LAST_A, rel=1e-6)


def test_no_foot_contact_block_and_no_actuator_force_leak():
    """關卡 3：輪足版沒有可用觸地訊號，obs 不得含 foot_contact，也不得含致動器力矩。

    0.901 kg 的輪讓擺動相的膝力矩與位置誤差都大於站立相，訊號方向是反的。
    """
    assert "foot_contact" not in [k for k, _ in obs_d1.OBS_LAYOUT]
    assert not hasattr(obs_d1, "foot_contact"), "foot_contact() 應已移除"
    _, obs = _sentinel_obs()
    for i, v in enumerate(_SEN_ACT_FORCE):
        assert not np.any(np.isclose(obs, v, rtol=1e-5)), \
            f"obs 出現致動器力矩分量 actuator_force[{i}]={v}"


def test_sentinel_cpg_block_order_rx_rxd_ry_ryd_sin_cos():
    """cpg 24 維的內容順序：rx, rx_d, ry, ry_d, sin(theta), cos(theta)。

    五組哨兵互異且 sin/cos 互異，因此 rx↔ry、rx_d↔ry_d、sin↔cos 任一對調都會失敗。
    """
    _, obs = _sentinel_obs()
    cpg = obs[_slice("cpg")]
    assert cpg.size == 24
    th = _SEN_CPG["theta"]
    expected = [("rx", _SEN_CPG["rx"]), ("rx_d", _SEN_CPG["rx_d"]),
                ("ry", _SEN_CPG["ry"]), ("ry_d", _SEN_CPG["ry_d"]),
                ("sin(theta)", np.sin(th)), ("cos(theta)", np.cos(th))]
    for i, (label, want) in enumerate(expected):
        got = cpg[4 * i:4 * i + 4]
        assert got == pytest.approx(want, rel=1e-6), \
            f"cpg 第 {i} 段應為 {label}，實得 {got}"


def test_build_obs_rejects_wrong_length_cmd_or_last_action():
    """np.concatenate 不會擋長度錯誤，靠 build_obs 開頭的防呆擋。"""
    d, _ = _sentinel_obs()
    c = {k: v.copy() for k, v in _SEN_CPG.items()}
    with pytest.raises(AssertionError):
        obs_d1.build_obs(d, c, np.zeros(2), _SEN_LAST_A)
    with pytest.raises(AssertionError):
        obs_d1.build_obs(d, c, _SEN_CMD, np.zeros(11))


def test_layout_sums_to_69_and_has_no_base_linear_velocity():
    total = sum(n for _, n in obs_d1.OBS_LAYOUT)
    assert total == d1_model.OBS_DIM == 69
    names = [k for k, _ in obs_d1.OBS_LAYOUT]
    assert "base_linvel" not in names, "機身線速度實機拿不到，不得放進 obs"
    assert names == ["gravity", "gyro", "joint_pos", "joint_vel",
                     "cmd", "last_action", "cpg"]


def test_build_obs_shape_and_dtype():
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    obs = obs_d1.build_obs(d, cpg_d1.cpg_init(),
                           np.array([0.6, 0.0, 0.0], np.float32),
                           np.zeros(12))
    assert obs.shape == (69,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_joint_pos_block_is_deviation_from_home():
    """home 姿態下關節角區塊應全為 0（存的是相對 HOME12 的偏差）。"""
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    obs = obs_d1.build_obs(d, cpg_d1.cpg_init(),
                           np.zeros(3, np.float32), np.zeros(12))
    assert obs[_slice("joint_pos")] == pytest.approx(np.zeros(12), abs=1e-9)


def test_gravity_block_points_down_when_upright():
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    obs = obs_d1.build_obs(d, cpg_d1.cpg_init(),
                           np.zeros(3, np.float32), np.zeros(12))
    assert obs[_slice("gravity")] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
