"""rl_obs（狗上純 numpy obs 組裝）：與 obs_max.build_obs 逐位元相同、腿序錯會被抓到。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
import coord  # noqa: E402
import cpg  # noqa: E402
import cpg_max  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import real_obs  # noqa: E402
import rl_obs  # noqa: E402

ACT = 10


def _rand_case(seed: int):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    quat_xyzw = q[[1, 2, 3, 0]]
    gyro = rng.normal(size=3)
    pos_motor = {n: float(rng.normal()) for n in rl_obs.LEG_NAMES}
    vel_motor = {n: float(rng.normal()) for n in rl_obs.LEG_NAMES}
    c = {k: {l: float(rng.uniform(1, 2)) for l in cpg.LEGS}
         for k in ("rx", "rx_d", "ry", "ry_d", "theta")}
    cmd = rng.normal(size=2)
    last_a = rng.normal(size=ACT)
    return quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a


def _reference(quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a):
    """本機參考路徑：real_obs 的換算 → obs_max.build_obs。"""
    f = real_obs.RealFrame()
    f.qpos[3:7] = real_obs.quat_wxyz(quat_xyzw, "xyzw")
    f.qpos[mm.LEG_QPOS_IDX] = [coord.to_ctrl(n, pos_motor[n]) for n in real_obs.LEG_NAMES]
    f.qvel[3:6] = gyro
    f.qvel[mm.LEG_QVEL_IDX] = [vel_motor[n] / real_obs._sign(n) for n in real_obs.LEG_NAMES]
    c_arr = {k: np.array([c[k][real_obs.LEG_SHM[l]] for l in mm.LEGS]) for k in c}
    return obs_max.build_obs(f, c_arr, cmd, last_a)


def test_constants_pinned():
    assert rl_obs.LEG_MJCF == mm.LEGS
    assert rl_obs.LEG_SHM == real_obs.LEG_SHM
    assert rl_obs.LEG_NAMES == real_obs.LEG_NAMES
    np.testing.assert_array_equal(rl_obs.HOME12, mm.HOME12)
    assert rl_obs.OBS_DIM == obs_max.obs_dim(ACT, False) == 66
    assert rl_obs.ACT_DIM == ACT


@pytest.mark.parametrize("seed", range(20))
def test_matches_obs_max(seed):
    quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a = _rand_case(seed)
    fr = rl_obs.Frame(pos_motor, vel_motor, quat_xyzw, gyro)
    got = rl_obs.build(fr, c, cmd, last_a)
    ref = _reference(quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a)
    assert got.dtype == np.float32 and got.shape == (66,)
    np.testing.assert_array_equal(got, ref)


def test_w2b_matches_cpg_max():
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        v = rng.normal(size=3)
        np.testing.assert_array_equal(rl_obs.w2b(q, v), cpg_max.w2b(q, v))


def test_leg_order_shuffle_detected():
    """腿序或關節序若錯，joint_pos / cpg 段一定不同（防「兩份實作各自正常」）。"""
    quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a = _rand_case(99)
    ref = _reference(quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a)
    # 把 fl / fr 對調
    swap = {n: pos_motor[n.replace("fl", "@@").replace("fr", "fl").replace("@@", "fr")]
            for n in pos_motor}
    got = rl_obs.build(rl_obs.Frame(swap, vel_motor, quat_xyzw, gyro), c, cmd, last_a)
    assert not np.array_equal(got, ref)
    c2 = {k: {**c[k], "fl": c[k]["fr"], "fr": c[k]["fl"]} for k in c}
    got2 = rl_obs.build(rl_obs.Frame(pos_motor, vel_motor, quat_xyzw, gyro), c2, cmd, last_a)
    assert not np.array_equal(got2, ref)


def test_leg_dict_roundtrip():
    arr = np.array([1.0, 2.0, 3.0, 4.0])           # FR, FL, RR, RL
    d = rl_obs.to_shm_legs(arr)
    assert d == {"fr": 1.0, "fl": 2.0, "br": 3.0, "bl": 4.0}
    np.testing.assert_array_equal(rl_obs.from_shm_legs(d), arr)


def test_build_rejects_bad_dims():
    quat_xyzw, gyro, pos_motor, vel_motor, c, cmd, last_a = _rand_case(5)
    fr = rl_obs.Frame(pos_motor, vel_motor, quat_xyzw, gyro)
    with pytest.raises(AssertionError):
        rl_obs.build(fr, c, cmd, np.zeros(14))
    with pytest.raises(AssertionError):
        rl_obs.build(fr, c, np.zeros(3), last_a)


def test_realbot_module_is_numpy_only():
    """狗上沒有 mujoco/jax，也不能 import inference/ 的模組。只看 import 行。"""
    lines = [l.strip() for l in (ROOT / "realbot" / "rl_obs.py").read_text().splitlines()
             if l.strip().startswith(("import ", "from "))]
    for l in lines:
        mod = l.split()[1].split(".")[0]
        assert mod in ("__future__", "numpy", "math", "coord", "cpg", "shm_io", "typing"), l
