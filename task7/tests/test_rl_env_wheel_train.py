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


def test_imu_bias_rotates_policy_gravity_only(env):
    e = rw.WheelEnv(flat=True, noise=False, push=False, grav_bias_deg=2.0, scene=env.scene)
    ks = jax.random.split(jax.random.PRNGKey(0), 400)
    rp = np.degrees(np.asarray(jax.vmap(e.reset)(ks).info["bias_rp"]))
    assert rp.shape == (400, 2) and np.abs(rp).max() <= 2.0 + 1e-6 and np.abs(rp).max() > 1.5 and np.abs(rp.mean(0)).max() < 0.4
    assert np.all(np.asarray(jax.vmap(env.reset)(ks[:8]).info["bias_rp"]) == 0.0)                                                                  # 預設：0
    assert np.all(np.asarray(jax.vmap(rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, grav_bias_deg=2.0, scene=env.scene).reset)(ks[:8]).info["bias_rp"]) == 0.0)   # fixed_cmd：0
    step = jax.jit(e.step)
    s = jax.jit(e.reset)(jax.random.PRNGKey(2))
    for rp_deg in ((2.0, 0.0), (0.0, 2.0), (1.0, -1.0)):
        s1 = s.replace(info={**s.info, "bias_rp": jnp.radians(jnp.array(rp_deg))})
        s2 = step(s1, jnp.zeros(16))
        g_pol, g_true = np.asarray(s2.obs["state"][3:6]), np.asarray(s2.obs["privileged_state"][3:6])
        ang = np.degrees(np.arccos(np.clip(np.dot(g_pol, g_true) / np.linalg.norm(g_pol) / np.linalg.norm(g_true), -1, 1)))
        want = np.degrees(np.arccos(np.cos(np.radians(rp_deg[0])) * np.cos(np.radians(rp_deg[1]))))     # Ry(p)·Rx(r) 作用在 (0,0,−1) 的夾角
        assert ang == pytest.approx(want, abs=0.1), rp_deg      # 機身不完全水平時公式有二階誤差
        np.testing.assert_allclose(np.asarray(s2.obs["state"][:3]), np.asarray(s2.obs["privileged_state"][:3]), atol=1e-6)     # 角速度不動
        np.testing.assert_allclose(np.asarray(s2.obs["state"][6:]), np.asarray(s2.obs["privileged_state"][6:53]), atol=1e-6)   # 其餘不動（noise 關）
    # 慣例與狗端 grav_fix_matrix 一致：bias (r,p) 注入後，乘回 grav_fix_matrix(r,p) 要歸零
    sys.path.insert(0, str(ROOT / "realbot")); import rl_obs_rl  # noqa: E402
    R = rl_obs_rl.grav_fix_matrix(2.0, -1.5)
    g_true = np.array([0.1, -0.2, -0.97]); g_true /= np.linalg.norm(g_true)
    g_pol = np.asarray(rw.bias_matrix(jnp.radians(2.0), jnp.radians(-1.5))).T @ g_true
    np.testing.assert_allclose(R @ g_pol, g_true, atol=1e-6)
