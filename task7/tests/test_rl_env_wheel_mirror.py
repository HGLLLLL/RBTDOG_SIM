"""v1c 鏡像：表是恆等對合、與「把 MuJoCo 狀態真的左右翻」一致、物理左右對稱、鏡像回合的 obs／ctrl 行為。spec §2.1、§5.2。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_wheel as rw  # noqa: E402
from mujoco import mjx  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True)


def _walked_state(env, n=6):
    """從 reset 走幾步非對稱動作，得到一個左右不對稱、四輪觸地的狀態。"""
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for i in range(n):
        a = jnp.sin(jnp.arange(16) * 0.9 + i * 0.4) * 0.3
        s = step(s, a)
    return s


def test_mirror_maps_are_involutions():
    o = jnp.arange(53, dtype=jnp.float32) + 1.0
    a = jnp.arange(16, dtype=jnp.float32) + 1.0
    p = jnp.arange(56, dtype=jnp.float32) + 1.0
    assert np.allclose(rw.mirror_obs(rw.mirror_obs(o)), o) and np.allclose(rw.mirror_act(rw.mirror_act(a)), a)
    assert np.allclose(rw.mirror_priv(rw.mirror_priv(p)), p)
    assert not np.allclose(rw.mirror_obs(o), o)
    # 符號表：ωx ωz、grav_y、cmd vy wz、ABAD（jpos／jvel／last_a／act）變號；其餘 +1
    sg = np.asarray(rw.OBS_MIRROR_SGN)
    assert sg[0] == -1 and sg[1] == 1 and sg[2] == -1 and sg[4] == -1 and sg[7] == -1 and sg[8] == -1
    assert np.all(sg[9:13] == -1) and np.all(sg[13:21] == 1) and np.all(sg[21:25] == -1) and np.all(sg[25:37] == 1)
    assert np.all(sg[37:41] == -1) and np.all(sg[41:53] == 1)
    assert np.all(np.asarray(rw.ACT_MIRROR_SGN)[:4] == -1) and np.all(np.asarray(rw.ACT_MIRROR_SGN)[4:] == 1)
    idx = np.asarray(rw.OBS_MIRROR_IDX)
    assert list(idx[9:13]) == [10, 9, 12, 11] and list(idx[33:37]) == [34, 33, 36, 35]     # 腿 FR↔FL、RR↔RL（輪速也換）


def test_mirror_obs_matches_physically_mirrored_state(env):
    s = _walked_state(env)
    d = s.pipeline_state
    cmd, la = jnp.array([0.3, 0.2, -0.4]), jnp.sin(jnp.arange(16)) * 0.5
    o = env._obs(d, cmd, la)
    dm = rw.mirror_data(env, d)
    om = env._obs(dm, cmd * jnp.array([1.0, -1.0, -1.0]), rw.mirror_act(la))
    np.testing.assert_allclose(np.asarray(om), np.asarray(rw.mirror_obs(o)), atol=1e-5)
    vb = rw.quat_rot_inv(d.qpos[3:7], d.qvel[0:3]); vbm = rw.quat_rot_inv(dm.qpos[3:7], dm.qvel[0:3])
    np.testing.assert_allclose(np.asarray(vbm), np.asarray(vb) * np.array([1, -1, 1]), atol=1e-5)


def test_physics_is_left_right_equivariant(env):
    """翻過的狀態用翻過的動作跑 5 步 ＝ 原軌跡翻過去（同時驗證模型左右對稱）。"""
    s = _walked_state(env)
    d, dm = s.pipeline_state, rw.mirror_data(env, s.pipeline_state)
    phys = jax.jit(env._physics)
    for i in range(5):
        a = jnp.sin(jnp.arange(16) * 0.5 + i) * 1.5
        d = phys(d, a); dm = phys(dm, rw.mirror_act(a))
    dmm = rw.mirror_data(env, dm)
    np.testing.assert_allclose(np.asarray(dmm.qpos), np.asarray(d.qpos), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dmm.qvel), np.asarray(d.qvel), atol=1e-3)
