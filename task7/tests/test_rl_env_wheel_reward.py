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
    assert float(T["lin"]) == pytest.approx(2.5, abs=1e-4) and float(T["ang"]) == pytest.approx(env.w_ang, abs=1e-4)
    assert env.w_ang == 2.5                                         # v1b：偏航追蹤權重 1.25 → 2.5
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
    for k in ("vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww", "wgap", "lim"):
        assert float(T[k]) <= 1e-9, k
    assert set(T) == set(rw.TERM_KEYS) and rw.TERM_KEYS[-3:] == ("roll", "impact", "knee")


def test_joint_limit_penalty(env):
    """lim：腿 12 關節 PD 目標 q_des＝a·ACT_SCALE＋ACT_OFFSET 超出軟限位（中點 ± 0.9 半行程）的量之和 × −w_lim；sat_n＝目標超出硬限位的顆數。
    罰目標不罰 q：v1 實際 q 從沒進軟限位，越界的是目標（vy 0.5 RL ABAD 目標 −0.73、硬限 −0.523）。"""
    lo, hi = np.asarray(env.Q_LO_HARD), np.asarray(env.Q_HI_HARD)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    assert np.allclose(np.asarray(env.Q_HI_SOFT), mid + 0.9 * half, atol=1e-6) and np.allclose(np.asarray(env.Q_LO_SOFT), mid - 0.9 * half, atol=1e-6)
    data = _standing_data(env)
    info = _info(env, data)
    T0, aux0 = env._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3))
    assert float(T0["lim"]) == 0.0 and float(aux0["sat_n"]) == 0.0                         # 零動作＝default 目標，在軟限位內
    j = 3                                                                                  # RL ABAD（v1 越界的那顆），往下限推
    sc, off = float(rw.ACT_SCALE[j]), float(rw.ACT_OFFSET[j])
    ls, lh = float(env.Q_LO_SOFT[j]), float(env.Q_LO_HARD[j])
    assert ls - 0.02 > lh
    a1 = jnp.zeros(16).at[j].set((ls - 0.02 - off) / sc)                                   # 目標＝軟限位下 0.02（仍在硬限位內）
    T1, aux1 = env._reward_terms(data, info, a1, jnp.zeros(3))
    assert float(T1["lim"]) == pytest.approx(-env.w_lim * 0.02, abs=1e-5) and float(aux1["sat_n"]) == 0.0
    a2 = jnp.zeros(16).at[j].set((lh - 0.05 - off) / sc)                                   # 目標＝硬限位下 0.05
    T2, aux2 = env._reward_terms(data, info, a2, jnp.zeros(3))
    assert float(aux2["sat_n"]) == 1.0 and float(T2["lim"]) == pytest.approx(-env.w_lim * (ls - (lh - 0.05)), abs=1e-5)
    e2 = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, w_lim=2 * env.w_lim, scene=env.scene)
    T3, _ = e2._reward_terms(data, info, a1, jnp.zeros(3))
    assert env.w_lim == 2.0 and float(T3["lim"]) == pytest.approx(2.0 * float(T1["lim"]), rel=1e-5)


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
    assert float(s.reward) == pytest.approx(sum(float(s.metrics[k]) for k in rw.TERM_KEYS), abs=1e-5)
    assert float(s.metrics["lim"]) == 0.0 and float(s.metrics["sat_n"]) == 0.0
    assert "gap_min" in s.metrics and 0.28 < float(s.metrics["gap_min"]) < 0.4               # 出生站姿：兩對左右輪實體間隙 ~308 mm
    assert float(s.metrics["small"]) == 0.0 and float(s.metrics["standing"]) == 0.0            # fixed_cmd：旗標固定 False


@pytest.fixture(scope="module")
def env_c(env):
    """v1c 權重（fixed_cmd 下鏡像／延遲／偏差自動關）。"""
    return rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, scene=env.scene, **rw.V1C)


def test_v1c_preset_values():
    assert rw.V1C == dict(mirror=True, w_roll=10.0, w_impact=3.0, impact_v0=0.3, delay_max=2, grav_bias_deg=2.0)


def test_roll_penalty_is_true_gravity_y_squared(env, env_c):
    base = np.asarray(_standing_data(env).qpos)
    info = _info(env_c, _standing_data(env))
    for deg, axis, expect in ((9.0, "x", -10.0 * np.sin(np.radians(9.0)) ** 2), (9.0, "y", 0.0), (0.0, "x", 0.0)):
        q = base.copy(); h = np.radians(deg) / 2
        q[3:7] = [np.cos(h), np.sin(h), 0.0, 0.0] if axis == "x" else [np.cos(h), 0.0, np.sin(h), 0.0]
        d = jax.jit(mjx.forward)(env_c.sys, mjx.make_data(env_c.sys).replace(qpos=jnp.array(q)))
        T, _ = env_c._reward_terms(d, info, jnp.zeros(16), jnp.zeros(3))
        assert float(T["roll"]) == pytest.approx(expect, abs=1e-5), (deg, axis)
    T0, _ = env._reward_terms(_standing_data(env), info, jnp.zeros(16), jnp.zeros(3))
    assert float(T0["roll"]) == 0.0                                           # 預設 w_roll 0


def _landed(env):
    data = _standing_data(env).replace(ctrl=env._ctrl(jnp.zeros(16)))
    phys = jax.jit(lambda d: env._physics(d, jnp.zeros(16)))
    for _ in range(15):
        data = phys(data)
    return data


def test_impact_penalty_on_touchdown_only(env, env_c):
    data = _landed(env_c)
    info = _info(env_c, data)
    _, aux = env_c._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3))
    assert np.all(np.asarray(aux["touch"]))
    info["t"] = jnp.int32(rw.IMPACT_SETTLE)                                  # _info 的 t＝0（出生後第一步）→ 先移到落定期之後
    info["touch_prev"] = jnp.array([False, True, True, True]); info["vfoot_prev"] = jnp.zeros((4, 3)).at[0, 2].set(-0.5)
    T, _ = env_c._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3))
    assert float(T["impact"]) == pytest.approx(-3.0 * (0.5 - 0.3) ** 2, abs=1e-6)   # 只有 FR 是「本步觸地、上步沒觸地」，vz_prev −0.5，門檻 0.3 → 罰 (0.2)²
    e0 = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, w_impact=3.0, impact_v0=0.0, scene=env.scene)
    Tz, _ = e0._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3)); assert float(Tz["impact"]) == pytest.approx(-3.0 * 0.25, abs=1e-6)   # 無門檻＝原式
    Tl, _ = env_c._reward_terms(data, dict(info, vfoot_prev=jnp.zeros((4, 3)).at[0, 2].set(-0.2)), jnp.zeros(16), jnp.zeros(3)); assert float(Tl["impact"]) == 0.0   # 0.2 m/s < 門檻：免費
    T, _ = env_c._reward_terms(data, dict(info, t=jnp.int32(rw.IMPACT_SETTLE - 1)), jnp.zeros(16), jnp.zeros(3))
    assert float(T["impact"]) == 0.0                                         # 落定期內的觸地不算事件
    info["touch_prev"] = jnp.ones(4, bool)
    T, _ = env_c._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3)); assert float(T["impact"]) == 0.0      # 持續觸地
    info["touch_prev"] = jnp.array([False, True, True, True]); info["vfoot_prev"] = jnp.zeros((4, 3)).at[0, 2].set(+0.5)
    T, _ = env_c._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3)); assert float(T["impact"]) == 0.0      # 往上動不算落地
    T0, _ = env._reward_terms(data, dict(info, touch_prev=jnp.array([False, True, True, True]), vfoot_prev=jnp.zeros((4, 3)).at[0, 2].set(-0.5)), jnp.zeros(16), jnp.zeros(3))
    assert float(T0["impact"]) == 0.0                                        # 預設 w_impact 0


def test_step_tracks_touch_prev_and_vfoot_prev(env_c):
    s = jax.jit(env_c.reset)(jax.random.PRNGKey(0))
    s2 = jax.jit(env_c.step)(s, jnp.zeros(16))
    assert s2.info["touch_prev"].shape == (4,) and s2.info["vfoot_prev"].shape == (4, 3)
    _, aux = env_c._reward_terms(s2.pipeline_state, s.info, jnp.zeros(16), s.info["cmd"])     # 同 data、同 foot_prev → 值必須一樣
    assert np.array_equal(np.asarray(s2.info["touch_prev"]), np.asarray(aux["touch"]))
    assert np.allclose(np.asarray(s2.info["vfoot_prev"]), np.asarray(aux["vfoot"]), atol=1e-6)
    assert "impact_vz" in s2.metrics and "td_n" in s2.metrics and "impact_fn" in s2.metrics and float(s2.metrics["impact_vz"]) >= 0.0


def test_spawn_landing_is_not_counted(env_c):
    """出生時輪子離地 3 mm（崎嶇可到數 cm）→ 前幾步的落地不是步態落地，不算 impact／td_n。"""
    s = jax.jit(env_c.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env_c.step)
    td, imp = [], []
    for _ in range(8):
        s = step(s, jnp.zeros(16))
        td.append(float(s.metrics["td_n"])); imp.append(float(s.metrics["impact"]))
    assert sum(td) == 0.0 and all(v == 0.0 for v in imp), (td, imp)
    _, aux = env_c._reward_terms(s.pipeline_state, s.info, jnp.zeros(16), jnp.zeros(3))
    assert np.all(np.asarray(aux["touch"]))                                  # 四輪都已著地：0 次是閘門擋掉，不是沒落地


def test_v1f_preset_angxy_and_knee_hinge(env):
    """V1F：不加延遲 DR；angxy 權重 0.5；膝 hinge 只罰比 60° 更直的部分。預設 w_angxy 0.05／w_knee 0 ＝ v1b。"""
    assert rw.V1F == dict(mirror=True, w_roll=10.0, w_impact=3.0, impact_v0=0.3, delay_max=0, grav_bias_deg=2.0, w_angxy=0.5, w_knee=0.5, knee_min_deg=60.0)
    ef = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, scene=env.scene, **rw.V1F)
    data = _standing_data(env); info = _info(ef, data)
    d = data.replace(qvel=data.qvel.at[3].set(1.0).at[4].set(0.5))                       # ωx 1、ωy 0.5 rad/s
    T, _ = ef._reward_terms(d, info, jnp.zeros(16), jnp.zeros(3)); T0, _ = env._reward_terms(d, info, jnp.zeros(16), jnp.zeros(3))
    assert float(T["angxy"]) == pytest.approx(-0.5 * 1.25, abs=1e-5) and float(T0["angxy"]) == pytest.approx(-0.05 * 1.25, abs=1e-5)
    assert float(T["knee"]) == 0.0                                                         # 預設站姿膝 86° > 60°：不罰
    q = data.qpos; kn = np.asarray(env.QADR)[8:12]
    q2 = q.at[kn[0]].set(-np.radians(42.0)).at[kn[2]].set(np.radians(42.0))                # FR、RR 膝伸到 42°（v1d 平移姿態）
    d2 = jax.jit(mjx.forward)(ef.sys, data.replace(qpos=q2))
    T2, _ = ef._reward_terms(d2, _info(ef, d2), jnp.zeros(16), jnp.zeros(3))
    assert float(T2["knee"]) == pytest.approx(-0.5 * 2 * np.radians(60.0 - 42.0), abs=1e-5)
    q3 = q.at[kn[1]].set(-np.radians(70.0))                                                # 膝彎更多（70° < 86°… 仍 > 60°）：不罰
    d3 = jax.jit(mjx.forward)(ef.sys, data.replace(qpos=q3))
    assert float(ef._reward_terms(d3, _info(ef, d3), jnp.zeros(16), jnp.zeros(3))[0]["knee"]) == 0.0
    assert float(env._reward_terms(d2, _info(env, d2), jnp.zeros(16), jnp.zeros(3))[0]["knee"]) == 0.0   # 預設 w_knee 0


def test_v1g_slip_weight(env):
    """V1G：只比 V1F 多 w_slip 0.75（拖腿橫滑變貴）；預設 0.075 ＝ v1b。"""
    assert rw.V1G == dict(rw.V1F, w_slip=0.75) and rw.WheelEnv.__init__.__defaults__[-1] == 0.075
    eg = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, scene=env.scene, **rw.V1G)
    data = _standing_data(env).replace(ctrl=env._ctrl(jnp.zeros(16)))
    phys = jax.jit(lambda d: env._physics(d, jnp.zeros(16)))
    for _ in range(15):
        data = phys(data)
    info = _info(eg, data); info["foot_prev"] = data.xpos[eg.FOOT_BIDS] - jnp.array([0.0, 0.3 * rw.CTRL_DT, 0.0])   # 四輪沿輪軸側滑 0.3 m/s
    Tg, _ = eg._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3)); Tb, _ = env._reward_terms(data, info, jnp.zeros(16), jnp.zeros(3))
    assert float(Tg["slip"]) == pytest.approx(-0.75 * 4 * 0.09, rel=0.05) and float(Tb["slip"]) == pytest.approx(-0.075 * 4 * 0.09, rel=0.05)
