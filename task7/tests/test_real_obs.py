"""錄檔 → obs 必須走同一支 build_obs，且腿序按名稱對應。"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402


def test_leg_map_matches_other_copies():
    import play_gait_traj
    assert real_obs.LEG_SHM == play_gait_traj.MM2SHM == coord._CFG2SHM
    assert real_obs.LEG_NAMES == [
        "fr1_hip_roll", "fr2_hip_pitch", "fr3_knee_pitch",
        "fl1_hip_roll", "fl2_hip_pitch", "fl3_knee_pitch",
        "br1_hip_roll", "br2_hip_pitch", "br3_knee_pitch",
        "bl1_hip_roll", "bl2_hip_pitch", "bl3_knee_pitch"]


def _sim_frames(n=6, seed=0):
    """在 MuJoCo 裡踩幾步拿到有姿態、有角速度的狀態，當「真值」。"""
    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = mm.HOME12
    d.qpos[2] = 0.6
    d.qvel[3:6] = rng.normal(0, 0.5, 3)
    frames = []
    for _ in range(n):
        d.ctrl[mm.LEG_ACT_IDX] = rng.normal(0, 5, 12)
        for _ in range(10):
            mujoco.mj_step(m, d)
        frames.append((d.qpos.copy(), d.qvel.copy()))
    return frames


def _rec_from_frames(frames, shuffle=False):
    """把模擬狀態包成 M6 錄檔：關節寫成**馬達座標**、quat 寫成 **xyzw**。"""
    n = len(frames)
    joints = {}
    shm2mj = {v: k for k, v in real_obs.LEG_SHM.items()}
    for nm in shm_io.JOINTS:
        leg, kind = nm[:2], nm[2:]
        if kind == coord.KIND_WHEEL:
            k = mm.LEGS.index(shm2mj[leg])
            qi, vi = mm.WHEEL_QPOS_IDX[k], mm.WHEEL_QVEL_IDX[k]
        else:
            idx = real_obs.LEG_NAMES.index(nm)
            qi, vi = mm.LEG_QPOS_IDX[idx], mm.LEG_QVEL_IDX[idx]
        s = coord.SIGN[kind][leg]
        joints[nm] = {"q": [coord.to_motor(nm, float(f[0][qi])) for f in frames],
                      "v": [s * float(f[1][vi]) for f in frames],
                      "tau": [0.0] * n, "des": [0.0] * n, "ff": 0.0, "kp": 0.0, "kd": 0.0}
    imu = []
    for qpos, qvel in frames:
        w, x, y, z = qpos[3:7]
        imu.append([0.0, 0.0, 9.81, *qvel[3:6], x, y, z, w])
    d = {"schema": "m6_record/1", "label": "syn", "note": "", "n": n,
         "secs": 0.005 * (n - 1), "hz_actual": 200.0,
         "t": [0.005 * i for i in range(n)], "joints": joints, "imu": imu}
    if shuffle:   # 名稱不變、dict 順序打亂 —— 按名稱對應的話結果必須一樣
        d["joints"] = dict(reversed(list(d["joints"].items())))
    return m6_rec.from_dict(d)


def test_obs_matches_build_obs_on_sensor_dims():
    frames = _sim_frames()
    rec = _rec_from_frames(frames)
    O = real_obs.obs_series(rec, quat_order="xyzw", gyro_scale=(1.0, 1.0, 1.0))
    assert O.shape == (len(frames), obs_max.OBS_DIM) and O.dtype == np.float32

    class _D:
        pass
    for i, (qpos, qvel) in enumerate(frames):
        d = _D()
        d.qpos, d.qvel = qpos, qvel
        ref = obs_max.build_obs(d, real_obs.zero_cpg(), np.zeros(2),
                                np.zeros(obs_max.ACT_DIM))
        np.testing.assert_allclose(O[i, :real_obs.SENSOR_DIM],
                                   ref[:real_obs.SENSOR_DIM], atol=1e-5)


def test_joint_order_by_name_not_position():
    frames = _sim_frames()
    a = real_obs.obs_series(_rec_from_frames(frames), "xyzw", (1, 1, 1))
    b = real_obs.obs_series(_rec_from_frames(frames, shuffle=True), "xyzw", (1, 1, 1))
    np.testing.assert_array_equal(a, b)


def test_quat_order_and_gyro_scale_are_applied():
    frames = _sim_frames()
    rec = _rec_from_frames(frames)
    a = real_obs.obs_series(rec, "xyzw", (1, 1, 1))
    b = real_obs.obs_series(rec, "wxyz", (1, 1, 1))
    assert not np.allclose(a[:, 0:3], b[:, 0:3])          # 順序錯 → 重力向量變了
    c = real_obs.obs_series(rec, "xyzw", (2.0, -1.0, 1.0))
    np.testing.assert_allclose(c[:, 3], 2.0 * a[:, 3], rtol=1e-6)
    np.testing.assert_allclose(c[:, 4], -a[:, 4], rtol=1e-6)


def test_ctrl_pos_vel_des_and_gains_shapes():
    rec = _rec_from_frames(_sim_frames(n=4))
    assert real_obs.ctrl_pos(rec).shape == (4, 12)
    assert real_obs.ctrl_vel(rec).shape == (4, 12)
    assert real_obs.ctrl_des(rec).shape == (4, 12)
    assert real_obs.kp12(rec).shape == (4, 12) and real_obs.kd12(rec).shape == (4, 12)
    assert real_obs.wheel_kp(rec).shape == (4, 4) and real_obs.wheel_kd(rec).shape == (4, 4)
