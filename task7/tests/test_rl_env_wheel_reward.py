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
    for k in ("vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww", "wgap"):
        assert float(T[k]) <= 1e-9, k
    assert set(T) == {"lin", "ang", "vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww", "wgap"}


def test_gap_hinge(env):
    """實體間隙 hinge：gap ＝ 左右輪心距 − 0.034（官方網格內側面 +0.017 ×2）；gap < GAP_MIN 起罰 w_gap·(1 − gap/GAP_MIN)²。"""
    far = _standing_data(env)
    T0, aux0 = env._reward_terms(far, _info(env, far), jnp.zeros(16), jnp.zeros(3))
    g0 = np.asarray(aux0["gaps"])
    assert g0.shape == (2,) and np.all(np.abs(g0 - 0.308) < 0.02), g0                  # default 站姿輪心距 342 mm → 間隙 308 mm
    assert float(T0["wgap"]) == 0.0
    near = _standing_data(env, abad=0.5)                                                # 內收：前後輪心距 ~62 mm → 間隙 ~28 mm
    T1, aux1 = env._reward_terms(near, _info(env, near), jnp.zeros(16), jnp.zeros(3))
    g1 = np.asarray(aux1["gaps"])
    assert np.all((g1 > 0.0) & (g1 < rw.GAP_MIN)), g1
    assert float(T1["wgap"]) == pytest.approx(-env.w_gap * sum(max(0.0, 1.0 - g / rw.GAP_MIN) ** 2 for g in g1), rel=1e-5)
    assert float(T1["wgap"]) < 0.0
    e2 = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, w_gap=2 * env.w_gap, scene=env.scene)
    T2, _ = e2._reward_terms(near, _info(env, near), jnp.zeros(16), jnp.zeros(3))
    assert env.w_gap == 2.0 and float(T2["wgap"]) == pytest.approx(2.0 * float(T1["wgap"]), rel=1e-6)


def test_wheel_wheel_contact_fires_when_squeezed(env):
    far = _standing_data(env, abad=0.0)
    T0, aux0 = env._reward_terms(far, _info(env, far), jnp.zeros(16), jnp.zeros(3))
    assert float(aux0["ww_force"]) == 0.0 and float(T0["ww"]) == 0.0
    near = _standing_data(env, abad=0.5)                           # 探測過：ABAD ±0.5 內收 → 前後輪各互碰
    near = jax.jit(lambda d: mjx.step(env.sys, d))(near)           # 接觸力要走一步才有
    T1, aux1 = env._reward_terms(near, _info(env, near), jnp.zeros(16), jnp.zeros(3))
    assert float(aux1["ww_force"]) > 10.0 and float(T1["ww"]) < 0.0
    assert aux1["touch"].shape == (4,)
    # 釘公式：ww ＝ −w_ww · Σ min(fn/100, 3)²，只算兩端都是輪的接觸（ww_force 是未截斷的和）
    C = aux1["C"]
    fn, g0, g1 = (np.asarray(C[k]) for k in ("fn", "g0", "g1"))
    wg = np.asarray(env.WHEEL_GIDS)
    mask = np.isin(g0, wg) & np.isin(g1, wg)
    assert float(aux1["ww_force"]) == pytest.approx(float(np.sum(fn * mask)), rel=1e-5)
    assert float(T1["ww"]) == pytest.approx(-1.0 * float(np.sum((np.minimum(fn / 100.0, rw.WW_CAP) * mask) ** 2)), rel=1e-5)
    e2 = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, w_ww=2.0, scene=env.scene)
    T2, _ = e2._reward_terms(near, _info(env, near), jnp.zeros(16), jnp.zeros(3))
    assert float(T2["ww"]) == pytest.approx(2.0 * float(T1["ww"]), rel=1e-6)
    # 對照 CPU MuJoCo：同 qpos、mj_forward 後走一步，mj_contactForce 的法向分量加總（輪–輪接觸）
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(env.scene); wt.load_hfield(m, env.H_np)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(_standing_data(env, abad=0.5).qpos)
    mujoco.mj_forward(m, d); mujoco.mj_step(m, d)
    f6 = np.zeros(6); cpu = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in wg and c.geom2 in wg:
            mujoco.mj_contactForce(m, d, i, f6); cpu += f6[0]
    print(f"ww_force MJX={float(aux1['ww_force']):.3f} N  CPU={cpu:.3f} N")
    assert cpu > 10.0
    # 膠囊版 MJX 比 CPU 低 ~28%（網格版 15%）；此測試要抓的是「只讀第 0 列」那種 4 倍級錯誤
    assert float(aux1["ww_force"]) == pytest.approx(cpu, rel=0.35)


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


def _step_from_qpos(env, q):
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    d = jax.jit(mjx.forward)(env.sys, s.pipeline_state.replace(qpos=jnp.array(q), qvel=jnp.zeros(env.sys.nv)))
    return jax.jit(env.step)(s.replace(pipeline_state=d), jnp.zeros(16))


def test_fall_terminates(env):
    """摔倒＝幾何判定（機身／腿不碰撞）：(a) 傾 80° → tilt；(b) 機身離地 < FALL_H；(c) 正常站立不終止。"""
    base = np.asarray(_standing_data(env).qpos)
    q = base.copy(); q[3:7] = [np.cos(np.radians(40)), np.sin(np.radians(40)), 0.0, 0.0]     # 繞 x 滾 80°、站立高度
    s2 = _step_from_qpos(env, q)
    assert float(s2.done) == 1.0 and float(s2.metrics["tilt"]) == 1.0 and float(s2.metrics["fell"]) == 1.0
    assert float(s2.metrics["oob"]) == 0.0 and int(s2.info["t"]) == 0                       # 摔 → done ＋ 重生
    q = base.copy(); q[2] = wt.BASE_H + 0.15                                                 # 離地 0.15 < 0.20（輪子插進地裡，一步而已）
    s2 = _step_from_qpos(env, q)
    assert float(s2.done) == 1.0 and float(s2.metrics["tilt"]) == 0.0 and float(s2.metrics["fell"]) == 1.0
    assert int(s2.info["t"]) == 0
    s3 = jax.jit(env.step)(jax.jit(env.reset)(jax.random.PRNGKey(1)), jnp.zeros(16))
    assert float(s3.done) == 0.0 and float(s3.metrics["tilt"]) == 0.0


def test_terrain_h_matches_height_np():
    e = rw.WheelEnv(map_seed=0, noise=False, push=False, fixed_cmd=True)
    for x, y in ((-35.7, -7.9), (12.34, 3.21), (36.05, 8.3), (-70.0, 40.0)):
        assert float(e._terrain_h(jnp.float32(x), jnp.float32(y))) == pytest.approx(wt.height_np(e.H_np, x, y), abs=1e-4)


@pytest.mark.parametrize("field", ["x", "z"])
def test_out_of_map_terminates(env, field):
    """出界／掉落：fell 也靠 oob 觸發（x 超過 X_LIM，或 z 低於 0＝hfield 基準面）。"""
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    q = s.pipeline_state.qpos
    q = q.at[0].set(rw.X_LIM + 0.5) if field == "x" else q.at[2].set(-0.3)
    s = s.replace(pipeline_state=s.pipeline_state.replace(qpos=q))
    s2 = jax.jit(env.step)(s, jnp.zeros(16))
    assert float(s2.done) == 1.0 and float(s2.metrics["oob"]) == 1.0 and float(s2.metrics["fell"]) == 1.0
    assert int(s2.info["t"]) == 0                                    # 重生
    assert abs(float(s2.pipeline_state.qpos[0])) < 1.0               # flat 出生在原點附近


def test_metrics_keys(env):
    s = jax.jit(env.step)(jax.jit(env.reset)(jax.random.PRNGKey(0)), jnp.zeros(16))
    assert set(s.metrics) == set(rw.METRIC_KEYS)
    assert float(s.reward) == pytest.approx(sum(float(s.metrics[k]) for k in ("lin", "ang", "vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww", "wgap")), abs=1e-5)
    assert "gap_min" in s.metrics and 0.28 < float(s.metrics["gap_min"]) < 0.4               # 出生站姿：兩對左右輪實體間隙 ~308 mm
    assert float(s.metrics["small"]) == 0.0 and float(s.metrics["standing"]) == 0.0            # fixed_cmd：旗標固定 False
