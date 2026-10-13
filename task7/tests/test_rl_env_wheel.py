"""WheelEnv 核心：obs 53 與 rl_only_infer.Sim 逐維相等、動作縮放、出生高度、step 跑通、重生骨架。spec §3。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_wheel as rw  # noqa: E402
import rl_only_infer as ri  # noqa: E402
import wheel_terrain as wt  # noqa: E402
from mujoco import mjx  # noqa: E402

IO_RL = ROOT.parent / "RL_only" / "policy_io.json"


@pytest.fixture(scope="module")
def env():
    return rw.WheelEnv(noise=False, push=False)


@pytest.fixture(scope="module")
def flat_env():
    return rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True)


def test_io_copy_matches_rl_only():
    if not IO_RL.exists():
        pytest.skip("RL_only/ 不在")
    assert json.load(open(IO_RL)) == rw.IO


def test_constants():
    assert (rw.OBS_DIM, rw.ACT_DIM, rw.ACT_GAIN, rw.EPISODE_LEN) == (53, 16, 4.0, 1000)
    assert rw.CTRL_DT == pytest.approx(0.02)
    assert np.allclose(np.asarray(rw.DEFAULT)[4:8], [0.8, 0.8, -0.8, -0.8]) and np.allclose(np.asarray(rw.DEFAULT)[8:12], [-1.5, -1.5, 1.5, 1.5])


def test_quat_helpers_match_numpy_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4); q /= np.linalg.norm(q); v = rng.normal(size=3)
        R = ri.quat_to_R(q)
        assert np.allclose(np.asarray(rw.quat_rot_inv(jnp.array(q), jnp.array(v))), R.T @ v, atol=1e-6)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        assert abs(float(rw.quat_yaw(jnp.array(q))) - yaw) < 1e-6
    assert np.allclose(np.asarray(rw.yaw_quat(jnp.array(0.7))), [np.cos(0.35), 0, 0, np.sin(0.35)])
    assert float(rw.wrap_pi(jnp.array(3.5))) == pytest.approx(3.5 - 2 * np.pi)


def test_obs_matches_rl_only_infer(env):
    """同一組 qpos/qvel：env._obs == Sim.obs（Sim 用圓柱平地模型，obs 不看幾何）。"""
    sim = ri.Sim(rw.IO, dt=0.005, ideal=False, wheel_kv=1.0)
    rng = np.random.default_rng(1)
    q = np.zeros(sim.m.nq); q[2] = 0.5; qq = rng.normal(size=4); q[3:7] = qq / np.linalg.norm(qq)
    q[sim.qadr] = sim.default + rng.uniform(-0.3, 0.3, 16); v = rng.uniform(-2, 2, sim.m.nv)
    sim.d.qpos[:] = q; sim.d.qvel[:] = v; sim.mj.mj_forward(sim.m, sim.d)
    last_a = rng.uniform(-3, 3, 16); sim.last_a = last_a.copy(); cmd = np.array([0.3, -0.2, 0.5])
    o_ref, _ = sim.obs(cmd)
    data = mjx.make_data(env.sys).replace(qpos=jnp.array(q), qvel=jnp.array(v))
    data = mjx.forward(env.sys, data)
    o = np.asarray(env._obs(data, jnp.array(cmd), jnp.array(last_a)))
    assert o.shape == (53,) and np.allclose(o, o_ref, atol=1e-5), np.abs(o - o_ref).max()


def test_action_scaling(env):
    a_env = jnp.ones(16)
    ctrl = np.asarray(env._ctrl(rw.ACT_GAIN * a_env))
    aid = np.asarray(env.AID)
    assert np.allclose(ctrl[aid[:12]], np.asarray(rw.DEFAULT)[:12] + 1.0)      # 0.25 × 4 = 1 rad
    assert np.allclose(ctrl[aid[12:]], 20.0)                                      # 5 × 4 = 20 rad/s


def test_spawn_height_on_each_kind():
    jit_reset = jax.jit(rw.WheelEnv(noise=False, push=False).reset)
    for ki in range(3):
        e = rw.WheelEnv(noise=False, push=False, fixed_tile=(ki, 9))
        s = jax.jit(e.reset)(jax.random.PRNGKey(ki))
        x, y, z = [float(v) for v in s.pipeline_state.qpos[:3]]
        cx, cy = wt.tile_origin(ki, 9)
        assert abs(x - cx) <= 0.5 + 1e-6 and abs(y - cy) <= 0.5 + 1e-6
        # 窗 ±0.6 m 內地形最高＝包住窗的格點最高（雙線性的極值在格點上）；9 點取樣只是下界
        j0, j1 = int(np.floor((x - 0.6 - wt.X0) / wt.DX)), int(np.ceil((x + 0.6 - wt.X0) / wt.DX))
        i0, i1 = int(np.floor((y - 0.6 - wt.Y0) / wt.DX)), int(np.ceil((y + 0.6 - wt.Y0) / wt.DX))
        h = float(e.H_np[i0:i1 + 1, j0:j1 + 1].max())
        h9 = max(wt.height_np(e.H_np, x + dx, y + dy) for dx in (-0.6, 0, 0.6) for dy in (-0.6, 0, 0.6))
        assert z == pytest.approx(h + e.Z_STAND, abs=1e-3) and z >= h9 + e.Z_STAND - 1e-4
        assert int(s.info["kind"]) == ki and int(s.info["level"]) == 9
    s = jit_reset(jax.random.PRNGKey(5))
    assert 0 <= int(s.info["level"]) <= 5 and 0 <= int(s.info["kind"]) <= 2          # 初始等級 0–5（他的 max_init_terrain_level）
    assert 0.40 < float(rw.WheelEnv(noise=False, push=False, flat=True).Z_STAND) < 0.52


def test_step_runs_and_shapes(flat_env):
    s = jax.jit(flat_env.reset)(jax.random.PRNGKey(0))
    assert s.obs["state"].shape == (53,) and s.obs["privileged_state"].shape == (56,)
    assert flat_env.observation_size == {"state": (53,), "privileged_state": (56,)} and flat_env.action_size == 16 and flat_env.backend == "mjx"
    step = jax.jit(flat_env.step)
    for _ in range(5):
        s = step(s, jnp.zeros(16))
    assert int(s.info["t"]) == 5 and float(s.done) == 0.0 and np.isfinite(float(s.reward))
    assert np.allclose(np.asarray(s.info["last_a"]), 0.0)
    # 零動作站著：機身高仍在出生附近
    assert abs(float(s.pipeline_state.qpos[2]) - (wt.BASE_H + flat_env.Z_STAND)) < 0.05


def test_respawn_at_episode_end(flat_env):
    s = jax.jit(flat_env.reset)(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "t": rw.EPISODE_LEN - 1})
    s = jax.jit(flat_env.step)(s, jnp.zeros(16))
    assert int(s.info["t"]) == 0                             # 重生：t 歸零、位置回出生點附近
    assert float(jnp.linalg.norm(s.pipeline_state.qpos[:2] - s.info["xy0"])) < 1e-5


def test_privileged_obs_and_policy_noise():
    """critic 特權 obs：前 53 ＝ 無雜訊 _obs、後 3 ＝ 機身系線速度；policy 的 "state" 帶雜訊（reset 的第一個 obs 也是）。"""
    e = rw.WheelEnv(flat=True, noise=True, push=False, fixed_cmd=True)
    s = jax.jit(e.reset)(jax.random.PRNGKey(0))
    step = jax.jit(e.step)
    for i in range(4):
        d = s.pipeline_state
        clean = np.asarray(e._obs(d, s.info["cmd"], s.info["last_a"]))
        vb = np.asarray(rw.quat_rot_inv(d.qpos[3:7], d.qvel[0:3]))
        priv, pol = np.asarray(s.obs["privileged_state"]), np.asarray(s.obs["state"])
        assert np.allclose(priv[:53], clean, atol=1e-6) and np.allclose(priv[53:], vb, atol=1e-6), i
        assert np.abs(pol - clean).max() > 1e-3, i                                       # policy 分支帶雜訊
        assert np.abs(pol[0:3] - clean[0:3]).max() <= rw.NOISE["ang"] + 1e-6 and np.allclose(pol[37:], clean[37:])
        s = step(s, jnp.full(16, 0.05))


def test_wheel_hfield_rbound_override(env):
    """hfield 子網格用的包圍半徑：輪子覆寫成真實外半徑＋2 mm（AABB 半對角 0.136 會多測一圈稜柱）。"""
    rb = np.asarray(env.sys._impl.geom_rbound_hfield)
    assert np.allclose(rb[np.asarray(env.WHEEL_GIDS)], wt.WHEEL_R + 0.002)
    assert np.all(rb[np.asarray(env.WHEEL_GIDS)] < 0.1) and rb.max() > 0.1   # 其他 geom 不動
