"""WheelEnv 訓練面：指令分布（站立 10%、小指令 30%、heading→wz）、10 s 重抽、推力、課程升降、DR、wrap_env 多回合跑通。spec §3–§5。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_wheel as rw  # noqa: E402
import wheel_terrain as wt  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return rw.WheelEnv(flat=True, noise=False, push=False)


def test_command_distribution(env):
    ks = jax.random.split(jax.random.PRNGKey(0), 2000)
    v, head, standing, small = jax.vmap(lambda k: env._sample_cmd(k, jnp.array(0.3)))(ks)
    v, standing, small = np.asarray(v), np.asarray(standing), np.asarray(small)
    assert 0.07 < standing.mean() < 0.13 and 0.26 < small.mean() < 0.34
    assert np.all(v[standing] == 0.0)
    assert np.all(np.abs(v[small & ~standing]) <= 0.4) and np.abs(v[~small & ~standing]).max() > 0.9
    h = np.asarray(head); assert np.all(np.abs(h[small] - 0.3) <= 0.5 + 1e-6) and np.abs(h[~small]).max() > 3.0


def test_heading_to_wz_and_resample(env):
    s = jax.jit(env.reset)(jax.random.PRNGKey(3))
    info = dict(s.info, standing=jnp.array(False), head_ab=jnp.array([0.2, 2.0]), cmd_ab=jnp.array([[0.3, 0.0], [0.0, 0.6]]), t=jnp.int32(0))
    d = s.pipeline_state.replace(qpos=s.pipeline_state.qpos.at[3:7].set(rw.yaw_quat(jnp.array(0.0))))
    c0 = np.asarray(env._cmd_now(d, info)); assert np.allclose(c0, [0.3, 0.0, 0.2], atol=1e-6)        # wz = 1.0 × 航向誤差
    info["t"] = jnp.int32(rw.EPISODE_LEN // 2)
    c1 = np.asarray(env._cmd_now(d, info)); assert np.allclose(c1, [0.0, 0.6, 1.0], atol=1e-6)        # 10 s 後換第二組；誤差 2.0 夾到 1
    info["standing"] = jnp.array(True); assert np.allclose(np.asarray(env._cmd_now(d, info)), 0.0)


def test_push_sets_velocity():
    e = rw.WheelEnv(flat=True, noise=False, push=True, fixed_cmd=True)
    s = jax.jit(e.reset)(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "t": s.info["push_t"]})
    s2 = jax.jit(e.step)(s, jnp.zeros(16))
    assert int(s2.info["push_t"]) > int(s.info["push_t"]) and int(s2.info["push_t"]) - int(s.info["push_t"]) in range(500, 751)
    vxy = np.asarray(s2.pipeline_state.qvel[0:2]); assert np.abs(vxy).max() > 0.05 and np.abs(vxy).max() <= 0.5 + 0.1


def test_curriculum_rule(env):
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    base = dict(s.info, level=jnp.int32(4), standing=jnp.array(False), cmd_ab=jnp.array([[0.5, 0.0], [0.5, 0.0]]))
    far = s.pipeline_state.replace(qpos=s.pipeline_state.qpos.at[0:2].set(base["xy0"] + jnp.array([4.5, 0.0])))
    near = s.pipeline_state.replace(qpos=s.pipeline_state.qpos.at[0:2].set(base["xy0"] + jnp.array([1.0, 0.0])))
    mid = s.pipeline_state.replace(qpos=s.pipeline_state.qpos.at[0:2].set(base["xy0"] + jnp.array([3.0, 0.0])))
    k = jax.random.PRNGKey(1)
    assert rw.CURR_UP_DIST == 2.0
    assert int(env._next_tile(k, far, base)[1]) == 5                                  # > 2 m 升級
    assert int(env._next_tile(k, near, base)[1]) == 3                                 # 1 m < 2 m 且 < 0.5 × 0.5 m/s × 20 s = 5 m → 降級
    assert int(env._next_tile(k, mid, base)[1]) == 5                                  # 3 m：> 2 m 也 < 5 m，升級優先 → 升
    assert int(env._next_tile(k, far, dict(base, standing=jnp.array(True)))[1]) == 4  # 站立回合不動
    top = dict(base, level=jnp.int32(9)); lv = int(env._next_tile(k, far, top)[1]); assert 0 <= lv <= 9
    assert int(env._next_tile(k, near, dict(base, level=jnp.int32(0)))[1]) == 0
    e_fixed = rw.WheelEnv(flat=True, noise=False, push=False, fixed_tile=(1, 2), scene=env.scene)
    assert tuple(int(v) for v in e_fixed._next_tile(k, far, base)) == (1, 2)


def test_domain_randomize_shapes_and_ranges(env):
    dr = rw.make_domain_randomize(env)
    sys_v, in_axes = dr(env.sys, jax.random.split(jax.random.PRNGKey(0), 16))
    assert sys_v.body_mass.shape[0] == 16 and in_axes.body_mass == 0 and in_axes.geom_friction == 0 and in_axes.dof_frictionloss == 0
    bm = np.asarray(sys_v.body_mass)[:, env.BASE_BID] - float(env.sys.body_mass[env.BASE_BID])
    assert bm.min() >= -1.0 - 1e-6 and bm.max() <= 3.0 + 1e-6 and bm.std() > 0.3
    fl = np.asarray(sys_v.dof_frictionloss); leg = np.asarray(env.VADR[:12]); wheel = np.asarray(env.VADR[12:])
    nom = np.asarray(env.sys.dof_frictionloss)[leg]
    assert np.all(fl[:, leg] >= 0.5 * nom - 1e-6) and np.all(fl[:, leg] <= 1.5 * nom + 1e-6)
    assert fl[:, wheel].min() >= 0.10 - 1e-6 and fl[:, wheel].max() <= 0.18 + 1e-6
    gf = np.asarray(sys_v.geom_friction)[:, :, 0]; assert gf.min() >= 0.6 - 1e-6 and gf.max() <= 1.0 + 1e-6
    com = np.asarray(sys_v.body_ipos)[:, env.BASE_BID] - np.asarray(env.sys.body_ipos)[env.BASE_BID]
    assert np.abs(com[:, 0]).max() <= 0.05 + 1e-6 and np.abs(com[:, 2]).max() <= 0.01 + 1e-6


def test_wrap_env_runs_two_episodes():
    e = rw.WheelEnv(flat=True, noise=True, push=True, episode_len=30)
    w = rw.wrap_env(e, episode_length=30)
    s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    step = jax.jit(w.step)
    dones = []
    for i in range(65):
        s = step(s, jnp.zeros((4, 16)))
        dones.append(np.asarray(s.done).copy())
    dones = np.stack(dones)
    assert np.all(dones[29] == 1.0) and np.all(dones[59] == 1.0) and np.all(dones[30] == 0.0)   # EpisodeWrapper 截斷
    assert np.all(np.asarray(s.info["steps"]) == 5) and np.all(np.asarray(s.info["t"]) == 5)      # 兩邊計數同步
    assert "truncation" in s.info and s.obs["state"].shape == (4, 53) and s.obs["privileged_state"].shape == (4, 56)


def test_action_delay_buffer_and_sampling(env):
    e = rw.WheelEnv(flat=True, noise=False, push=False, delay_max=2, scene=env.scene)
    ks = jax.random.split(jax.random.PRNGKey(0), 600)
    dl = np.asarray(jax.vmap(e.reset)(ks).info["delay"])
    assert set(np.unique(dl)) == {0, 1, 2} and 0.2 < (dl == 2).mean() < 0.47
    assert np.all(np.asarray(jax.vmap(rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, delay_max=2, scene=env.scene).reset)(ks[:8]).info["delay"]) == 0)   # fixed_cmd：0
    assert np.all(np.asarray(jax.vmap(env.reset)(ks[:8]).info["delay"]) == 0)                                                                                        # 預設：0
    step = jax.jit(e.step)
    acts = [jnp.full(16, 0.1 * (i + 1)) for i in range(4)]
    for d in (0, 1, 2):
        s = jax.jit(e.reset)(jax.random.PRNGKey(1))
        s = s.replace(info={**s.info, "delay": jnp.int32(d)})
        for i, a in enumerate(acts):
            s = step(s, a)
            want = rw.ACT_GAIN * acts[i - d] if i - d >= 0 else jnp.zeros(16)                 # 物理吃 d 步前的動作；緩衝初值 0
            np.testing.assert_allclose(np.asarray(s.pipeline_state.ctrl[e.AID]), np.asarray(e._ctrl(want)[e.AID]), atol=1e-6, err_msg=f"d={d} i={i}")
            np.testing.assert_allclose(np.asarray(s.obs["state"][37:53]), np.asarray(rw.ACT_GAIN * a), atol=1e-6)      # obs 的 last_action 不延遲
            assert float(s.metrics["delay"]) == d
    # 重生：緩衝清零、delay 重抽（t 推到回合末）
    s = s.replace(info={**s.info, "t": jnp.int32(rw.EPISODE_LEN - 1)})
    s2 = step(s, acts[0])
    assert int(s2.info["t"]) == 0 and np.all(np.asarray(s2.info["a_hist"]) == 0.0)
