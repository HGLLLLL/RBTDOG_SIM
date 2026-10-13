"""WheelEnv reward 十三項（他的十項＋輪輪接觸＋std／站立倍率）、終止、接觸判定。spec §3。"""
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
from mujoco import mjx  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True)


def _standing_data(env, abad=0.0, z_extra=0.0, yaw=0.0):
    q = np.zeros(env.sys.nq); q[2] = wt.BASE_H + env.Z_STAND + z_extra; q[3:7] = np.asarray(rw.yaw_quat(jnp.array(yaw)))
    q[np.asarray(env.QADR)] = np.asarray(rw.DEFAULT)
    q[np.asarray(env.QADR)[[0, 2]]] = abad; q[np.asarray(env.QADR)[[1, 3]]] = -abad     # FR/RR +、FL/RL − ＝ 內收
    d = mjx.make_data(env.sys).replace(qpos=jnp.array(q))
    return jax.jit(mjx.forward)(env.sys, d)


def _info(env, data, a=None):
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    return dict(s.info, qvel_prev=data.qvel[env.VADR], foot_prev=data.xpos[env.FOOT_BIDS], last_a=jnp.zeros(16) if a is None else a)


def test_perfect_tracking_gives_full_weight(env):
    data = _standing_data(env)
    data = data.replace(qvel=data.qvel.at[0].set(0.5))              # 機身 vx 0.5（yaw 0 → 機身系＝世界系）
    T, aux = env._reward_terms(data, _info(env, data), jnp.zeros(16), jnp.array([0.5, 0.0, 0.0]))
    assert float(T["lin"]) == pytest.approx(2.5, abs=1e-4) and float(T["ang"]) == pytest.approx(1.25, abs=1e-4)
    assert float(aux["err_lin"]) == pytest.approx(0.0, abs=1e-6)


def test_tracking_std_shape(env):
    data = _standing_data(env)
    T, _ = env._reward_terms(data, _info(env, data), jnp.zeros(16), jnp.array([0.25, 0.0, 0.0]))
    assert float(T["lin"]) == pytest.approx(2.5 * np.exp(-(0.25 ** 2) / 0.25 ** 2), abs=1e-4)     # std 0.25：誤差 0.25 → e⁻¹
    e05 = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, std=0.5, scene=env.scene)
    T5, _ = e05._reward_terms(data, _info(e05, data), jnp.zeros(16), jnp.array([0.25, 0.0, 0.0]))
    assert float(T5["lin"]) > float(T["lin"])


def test_standing_scale_and_dev(env):
    data = _standing_data(env)
    info = _info(env, data)
    T0, _ = env._reward_terms(data, info, jnp.zeros(16), jnp.array([0.0, 0.0, 0.0]))
    assert float(T0["dev"]) == pytest.approx(0.0, abs=1e-6)                                   # 在 default pose：偏離 0
    q = data.qpos.at[env.QADR[4]].add(0.1); d2 = jax.jit(mjx.forward)(env.sys, data.replace(qpos=q))
    Ts, _ = env._reward_terms(d2, info, jnp.zeros(16), jnp.array([0.0, 0.0, 0.0]))
    Tm, _ = env._reward_terms(d2, info, jnp.zeros(16), jnp.array([0.5, 0.0, 0.0]))
    assert float(Ts["dev"]) == pytest.approx(-0.1 * 0.1 * 10.0, abs=1e-5) and float(Tm["dev"]) == pytest.approx(-0.1 * 0.1, abs=1e-5)


def test_action_rate_and_signs(env):
    data = _standing_data(env)
    a = jnp.full(16, 0.5)
    T, _ = env._reward_terms(data, _info(env, data), a, jnp.array([0.0, 0.0, 0.0]))
    assert float(T["rate"]) == pytest.approx(-0.01 * 16 * 0.25, abs=1e-6)
    for k in ("vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww"):
        assert float(T[k]) <= 1e-9, k
    assert set(T) == {"lin", "ang", "vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww"}


def test_wheel_wheel_contact_fires_when_squeezed(env):
    far = _standing_data(env, abad=0.0)
    T0, aux0 = env._reward_terms(far, _info(env, far), jnp.zeros(16), jnp.zeros(3))
    assert float(aux0["ww_force"]) == 0.0 and float(T0["ww"]) == 0.0
    near = _standing_data(env, abad=0.5)                           # 探測過：ABAD ±0.5 內收 → 前後輪各互碰
    near = jax.jit(lambda d: mjx.step(env.sys, d))(near)           # 接觸力要走一步才有
    T1, aux1 = env._reward_terms(near, _info(env, near), jnp.zeros(16), jnp.zeros(3))
    assert float(aux1["ww_force"]) > 10.0 and float(T1["ww"]) < 0.0
    assert aux1["touch"].shape == (4,)


def test_touch_and_slip(env):
    # 站姿出生輪底離地 3 mm（Z_STAND），一個 sim 步（5 ms）還落不到地；用 default 目標落地站穩 15 個控制步（0.3 s）
    data = _standing_data(env).replace(ctrl=env._ctrl(jnp.zeros(16)))
    phys = jax.jit(lambda d: env._physics(d, jnp.zeros(16)))
    for _ in range(15):
        data = phys(data)
    _, aux = env._reward_terms(data, _info(env, data), jnp.zeros(16), jnp.zeros(3))
    assert np.all(np.asarray(aux["touch"]))                         # 四輪都觸地
    info = _info(env, data); info["foot_prev"] = data.xpos[env.FOOT_BIDS] - jnp.array([0.0, 0.3 * rw.CTRL_DT, 0.0])   # 四輪沿 y（輪軸）側滑 0.3 m/s
    T, _ = env._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3))
    assert float(T["slip"]) == pytest.approx(-0.075 * 4 * 0.09, rel=0.05)
    lifted = _standing_data(env, z_extra=0.05)
    _, aux2 = env._reward_terms(lifted, _info(env, lifted), jnp.zeros(16), jnp.zeros(3))
    assert not np.any(np.asarray(aux2["touch"]))


def test_base_contact_terminates(env):
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    low = _standing_data(env, z_extra=-(env.Z_STAND - 0.06))        # 機身壓到地
    low = jax.jit(lambda d: mjx.step(env.sys, d))(low)
    s = s.replace(pipeline_state=low)
    s2 = jax.jit(env.step)(s, jnp.zeros(16))
    assert float(s2.done) == 1.0 and int(s2.info["t"]) == 0          # 摔 → done ＋ 重生
    s3 = jax.jit(env.step)(jax.jit(env.reset)(jax.random.PRNGKey(1)), jnp.zeros(16))
    assert float(s3.done) == 0.0


def test_metrics_keys(env):
    s = jax.jit(env.step)(jax.jit(env.reset)(jax.random.PRNGKey(0)), jnp.zeros(16))
    assert set(s.metrics) == set(rw.METRIC_KEYS)
    assert float(s.reward) == pytest.approx(sum(float(s.metrics[k]) for k in ("lin", "ang", "vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww")), abs=1e-5)
