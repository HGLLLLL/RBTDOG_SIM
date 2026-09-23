"""rl_obs_v3（狗上 88 維 obs）：與 DualModeEnv._obs 的 numpy 重寫逐位元；鏡像對 jax 版；is_right 邊界。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import coord  # noqa: E402
import max_model as mm  # noqa: E402
import rl_env_v3 as v3  # noqa: E402
import rl_obs  # noqa: E402
import rl_obs_v3 as ro  # noqa: E402

NPZ = ROOT / "weights" / "cpg_rl_v3_6f_np.npz"


@pytest.fixture(scope="module")
def O():
    return ro.Obs(str(NPZ))


def _case(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    quat_xyzw = q[[1, 2, 3, 0]]
    gyro = rng.normal(size=3)
    q12 = rng.normal(size=12) * 0.5
    qv12 = rng.normal(size=12) * 3
    wv4 = rng.normal(size=4) * 10
    pos = {n: coord.to_motor(n, float(q12[k])) for k, n in enumerate(rl_obs.LEG_NAMES)}
    vel = {n: float(qv12[k]) * rl_obs._SIGN12[k] for k, n in enumerate(rl_obs.LEG_NAMES)}
    for k, n in enumerate(ro.WHEEL_NAMES):
        pos[n] = 0.0; vel[n] = float(wv4[k]) * ro.WHEEL_SIGN4[k]
    c = dict(theta=rng.uniform(0, 2 * np.pi, 4), amp=rng.uniform(0, 1.2, 4), wheel=rng.normal(size=4) * 5,
             cyc_abad=rng.normal(size=4) * 0.3, cyc_knee=rng.normal(size=4) * 0.3)
    cmd, last_a = rng.normal(size=3) * 0.5, rng.normal(size=24)
    head, s4, s_arc = float(rng.normal()), float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
    return dict(q=q, quat_xyzw=quat_xyzw, gyro=gyro, q12=q12, qv12=qv12, wv4=wv4, pos=pos, vel=vel, c=c, cmd=cmd, last_a=last_a,
                head=head, s4=s4, s_arc=s_arc)


def _reference(C):
    """DualModeEnv._obs 的 numpy 重寫（imu_q = 單位、gyro 偏置 0）。"""
    grav = np.asarray(v3.w2b(jnp.array(C["q"], jnp.float32), jnp.array([0.0, 0.0, -1.0])))
    stance = np.asarray(v3.STANCE_Q12_j)
    return np.concatenate([grav, C["gyro"], C["q12"] - stance, C["qv12"], C["wv4"], C["cmd"], [C["s4"], C["s_arc"]],
                          [np.clip(C["head"], -1.0, 1.0)], C["last_a"], C["c"]["amp"], np.sin(C["c"]["theta"]), np.cos(C["c"]["theta"]),
                          C["c"]["cyc_abad"], C["c"]["cyc_knee"], C["c"]["wheel"] / 5.0]).astype(np.float32)


def test_constants_pinned(O):
    assert ro.LEG_NAMES == rl_obs.LEG_NAMES
    assert ro.WHEEL_NAMES == [rl_obs.LEG_SHM[l] + coord.KIND_WHEEL for l in mm.LEGS] == ["fr4_foot", "fl4_foot", "br4_foot", "bl4_foot"]
    assert list(ro.WHEEL_SIGN4) == [coord.SIGN[coord.KIND_WHEEL][n[:2]] for n in ro.WHEEL_NAMES] == [1.0, -1.0, 1.0, -1.0]
    np.testing.assert_array_equal(O.stance12, np.asarray(v3.STANCE_Q12_j))
    assert ro.OBS_DIM == 88 and ro.ACT_DIM == 24


@pytest.mark.parametrize("seed", range(20))
def test_build_matches_env_obs(O, seed):
    C = _case(seed)
    fr = rl_obs.Frame(C["pos"], C["vel"], C["quat_xyzw"], C["gyro"])
    got = O.build(fr, C["c"], C["cmd"], C["last_a"], C["head"], C["s4"], C["s_arc"])
    ref = _reference(C)
    assert got.dtype == np.float32 and got.shape == (88,)
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)      # grav 經 jax float32 四元數旋轉，其餘位元相同
    np.testing.assert_array_equal(got[6:], ref[6:])
    np.testing.assert_allclose(O.wheel_v(fr), C["wv4"])


def test_leg_wheel_order_and_stance_errors_are_caught(O):
    C = _case(99)
    ref = _reference(C)
    pos, vel = dict(C["pos"]), dict(C["vel"])
    for a, b in (("fl", "fr"),):                                  # 腿序對調
        for k in coord.LEG_KINDS:
            pos[a + k], pos[b + k] = C["pos"][b + k], C["pos"][a + k]
    got = O.build(rl_obs.Frame(pos, C["vel"], C["quat_xyzw"], C["gyro"]), C["c"], C["cmd"], C["last_a"], C["head"], C["s4"], C["s_arc"])
    assert np.max(np.abs(got[6:18] - ref[6:18])) > 1e-3
    vel["fl4_foot"], vel["fr4_foot"] = C["vel"]["fr4_foot"], C["vel"]["fl4_foot"]   # 輪序對調
    got = O.build(rl_obs.Frame(C["pos"], vel, C["quat_xyzw"], C["gyro"]), C["c"], C["cmd"], C["last_a"], C["head"], C["s4"], C["s_arc"])
    assert np.max(np.abs(got[30:34] - ref[30:34])) > 1e-3
    assert np.max(np.abs(O.stance12 - np.asarray(mm.HOME12))) > 0.2   # 站姿不是 home


def test_mirror_obs_and_act_match_jax(O):
    rng = np.random.default_rng(7)
    for _ in range(1000):
        o = (rng.normal(size=88) * 2).astype(np.float32)
        th = rng.uniform(0, 2 * np.pi, 4); o[68:72] = np.sin(th); o[72:76] = np.cos(th)
        np.testing.assert_allclose(O.mirror_obs(o), np.asarray(v3.mirror_obs(jnp.array(o))), atol=1e-6, rtol=0)
        a = rng.normal(size=24)
        np.testing.assert_allclose(O.mirror_act(a), np.asarray(v3.mirror_act(jnp.array(a, jnp.float32))), atol=1e-6, rtol=0)


def test_mirror_is_an_involution(O):
    rng = np.random.default_rng(8)
    o = rng.normal(size=88).astype(np.float32); th = rng.uniform(0, 2 * np.pi, 4); o[68:72] = np.sin(th); o[72:76] = np.cos(th)
    np.testing.assert_allclose(O.mirror_obs(O.mirror_obs(o)), o, atol=1e-5)
    a = rng.normal(size=24)
    np.testing.assert_array_equal(O.mirror_act(O.mirror_act(a)), a)


def test_is_right_boundaries():
    assert ro.is_right((0.0, 0.0, -0.11)) and not ro.is_right((0.0, 0.0, 0.11))
    assert ro.is_right((0.0, -0.5, -0.1)) and ro.is_right((0.0, -0.021, 0.0))
    assert not ro.is_right((0.0, -0.02, 0.0)) and not ro.is_right((0.0, -0.5, 0.11)) and not ro.is_right((0.5, 0.0, 0.0))
    # 與 jax mirror_policy(turn_only=False) 的判斷式一致
    for cmd in [(0, 0, -0.11), (0, 0, 0.11), (0, -0.5, -0.1), (0, -0.021, 0), (0, -0.02, 0), (0, -0.5, 0.11), (0.5, 0, 0)]:
        c = jnp.array(cmd, jnp.float32)
        right_j = bool(jnp.where(jnp.abs(c[2]) > 0.1, c[2] < 0, c[1] < -0.02))
        assert ro.is_right(cmd) == right_j, cmd
