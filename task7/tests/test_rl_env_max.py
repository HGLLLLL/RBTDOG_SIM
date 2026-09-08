"""RL v2 環境：JAX 版 CPG/IK 對 numpy 版逐點相符、14 維動作、sway 斜率、護欄、env 可跑。"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_max  # noqa: E402
import gait_baseline as gb  # noqa: E402
import leg_kin  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import rl_env_max as re  # noqa: E402

A = gb.BASELINE_A


def test_jax_cpg_and_ik_match_numpy():
    c_np = cpg_max.cpg_init(cpg_max.PHASE_WALK_LS)
    step_np = cpg_max.make_cpg_step(cpg_max.PHASE_WALK_LS)
    c_j = re.cpg_init()
    for _ in range(37):
        c_np = step_np(c_np, np.full(4, 1.8), np.full(4, 1.5), np.full(4, 1.4), mm.CTRL_DT)
        c_j = re.cpg_step(c_j, jnp.full(4, 1.8), jnp.full(4, 1.5), jnp.full(4, 1.4), mm.CTRL_DT)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    sway = (0.02, -0.03)
    q_np, _ = cpg_max.joint_targets(c_np, f0, A["x_off"], A["g_c"], A["d_step"], A["d_step_y"],
                                    A["duty"], ks, A["z_sag"], sway)
    q_j = np.asarray(re.joint_targets_j(c_j, jnp.array(sway)))
    assert np.abs(q_np - q_j).max() < 1e-4
    t_np = cpg_max.foot_targets(c_np, f0, A["x_off"], A["g_c"], A["d_step"], A["d_step_y"],
                                A["duty"], A["z_sag"], sway)
    t_j = np.asarray(re.foot_targets_j(c_j, jnp.array(sway)))
    assert np.abs(t_np - t_j).max() < 1e-5


def test_fk_j_matches_leg_kin():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.0, 1.0, 3)
        for k in range(4):
            np.testing.assert_allclose(np.asarray(re.fk_j(k, jnp.array(q))), leg_kin.fk(k, q),
                                       atol=1e-5)


def test_act_to_cmd_and_baseline_action():
    a = re.baseline_action()
    assert a.shape == (obs_max.ACT_DIM,) == (14,)
    mux, muy, om, sway = re.act_to_cmd(jnp.array(a))
    np.testing.assert_allclose(mux, A["mu_x"], atol=1e-3)
    np.testing.assert_allclose(muy, A["mu_y"], atol=1e-3)
    np.testing.assert_allclose(om, A["omega"], atol=1e-3)
    np.testing.assert_allclose(sway, 0.0, atol=1e-6)
    _, _, _, s = re.act_to_cmd(jnp.full(14, 10.0))
    np.testing.assert_allclose(s, re.SWAY_MAX, atol=1e-6)


def test_sway_slew_limits_step():
    prev = jnp.zeros(2)
    out = re.slew_sway(prev, jnp.array([0.06, -0.06]))
    np.testing.assert_allclose(out, [re.SWAY_SLEW, -re.SWAY_SLEW])
    out = re.slew_sway(jnp.array([0.05, 0.0]), jnp.array([0.052, 0.001]))
    np.testing.assert_allclose(out, [0.052, 0.001], atol=1e-7)


def test_guard_terms():
    tau = jnp.array([0.0] * 11 + [68.0])
    assert float(re.tau_barrier(tau)) == pytest.approx(100.0)       # (68−58)²
    err = jnp.array([0.0] * 11 + [0.65])
    assert float(re.err_barrier(err)) == pytest.approx(0.04, abs=1e-6)   # (0.65−0.45)²
    assert float(re.tau_barrier(jnp.full(12, 50.0))) == 0.0


def test_env_runs_baseline_without_done():
    env = re.MaxCpgEnv()
    assert env.observation_size == 70 and env.action_size == 14
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    assert s.obs.shape == (70,)
    a = jnp.array(re.baseline_action())
    for i in range(30):
        s = step(s, a)
        assert float(s.done) == 0.0, f"基準動作在第 {i} 步 done"
    assert set(re.METRIC_KEYS) <= set(s.metrics.keys())
    assert float(s.pipeline_state.qpos[2]) > 0.40


def test_domain_randomize_shapes():
    env = re.MaxCpgEnv()
    sys_r, in_axes = re.domain_randomize(env.sys, jax.random.split(jax.random.PRNGKey(1), 3))
    assert sys_r.geom_friction.shape[0] == 3
    assert sys_r.actuator_gainprm.shape[0] == 3
    # ABAD 與 HIP/KNEE 的 kp 各自隨機：三個 env 的 ABAD/HIP 比值不會全相同
    kp = np.asarray(sys_r.actuator_gainprm[:, mm.LEG_ACT_IDX, 0])
    ratio = kp[:, 0] / kp[:, 1]
    assert np.ptp(ratio) > 1e-3


def test_presets_v2_unchanged_and_v2_1_defined():
    """v2 的權重必須與模組層級常數逐項相同（舊 notebook 行為不變）；v2.1 只改六個鍵。"""
    w2 = re.weights_of("v2")
    assert w2["W_ROLL"] == re.W_ROLL == 20.0 and w2["W_EXEC"] == re.W_EXEC == 1.0
    assert w2["EXEC_SIGMA"] == re.EXEC_SIGMA == 0.03 and w2["W_TAUBAR"] == re.W_TAUBAR
    w21 = re.weights_of("v2.1")
    changed = {k for k in w21 if w21[k] != w2[k]}
    assert changed == {"W_ROLL", "W_ROLLRATE", "W_PITCH", "W_PITCHRATE", "W_EXEC", "EXEC_SIGMA"}
    assert w21["W_TAUBAR"] == w2["W_TAUBAR"] and w21["W_ERRBAR"] == w2["W_ERRBAR"]   # 護欄不動
    with pytest.raises(ValueError):
        re.weights_of("v9")


def test_v2_1_reward_shares_on_baseline():
    """★ 權重是量出來的：基準動作上 roll 兩項要佔正項 10–25%、exec ≤ 50%、護欄 ≈ 0。

    v2 的教訓：roll 兩項只佔 2.7%，policy 忽略它。這條測試讓「權重失衡」變成會失敗的東西。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "diag"))
    import rl_calibrate
    r = rl_calibrate.calibrate("v2.1", steps=300, verbose=False)
    sh = r["share"]
    roll = sh["t_roll"] + sh["t_rollrate"]
    pitch = sh["t_pitch"] + sh["t_pitchrate"]
    assert 0.10 <= roll <= 0.25, f"roll 兩項佔 {roll:.1%}"
    assert 0.03 <= pitch <= 0.15, f"pitch 兩項佔 {pitch:.1%}"
    assert sh["t_exec"] <= 0.50 and roll > pitch          # 姿態 > 執行率的優先序在權重上成立
    assert sh["t_taubar"] < 0.01 and sh["t_errbar"] < 0.01  # 基準不碰護欄
    assert r["metrics"]["reward"] > 1.0
