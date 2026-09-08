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
    assert changed == {"W_ROLL", "W_ROLLRATE", "W_PITCH", "W_PITCHRATE", "W_EXEC", "EXEC_SIGMA",
                       "W_YAW", "YAW_SIG2", "YAW_EMA"}
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


def test_yaw_reward_kernel_sees_slow_drift():
    """v2 的核（σ²=0.05）對 0.8°/s 的漂移只掉 0.4%；v2.1 的核（0.0005）掉 32%。"""
    d = 0.014                                   # rad/s = 0.8°/s（v2 實測漂移）
    v2 = float(re.yaw_reward(d, 0.0, re.weights_of("v2")["YAW_SIG2"]))
    v21 = float(re.yaw_reward(d, 0.0, re.weights_of("v2.1")["YAW_SIG2"]))
    assert v2 > 0.99 and 0.6 < v21 < 0.75
    assert re.weights_of("v2.1")["YAW_EMA"] > 0 and re.weights_of("v2")["YAW_EMA"] == 0


def test_v2_2_nomux_layout():
    """v2.2：動作 10 維（mu_x 固定＝基準）、obs 66 維；sway 在第 8–9 維。"""
    w = re.weights_of("v2.2")
    assert w["ACT_LAYOUT"] == "nomux" and w["EXEC_MODE"] == "rate"
    assert w["CMD_VX"] == (0.15, 0.40) and w["VX_SIG2"] == 0.1
    assert w["W_YAW_INST"] == 0.5 and re.weights_of("v2.1")["W_YAW_INST"] == 0.0   # v2.1 不變
    a = re.baseline_action("nomux")
    assert a.shape == (10,)
    mux, muy, om, sway = re.act_to_cmd(jnp.array(a), "nomux")
    np.testing.assert_allclose(mux, A["mu_x"])                 # 固定，不受動作影響
    np.testing.assert_allclose(muy, A["mu_y"], atol=1e-3)
    np.testing.assert_allclose(om, A["omega"], atol=1e-3)
    np.testing.assert_allclose(sway, 0.0, atol=1e-6)
    mux2, _, _, s2 = re.act_to_cmd(jnp.full(10, -10.0), "nomux")
    np.testing.assert_allclose(mux2, A["mu_x"])
    np.testing.assert_allclose(s2, -re.SWAY_MAX, atol=1e-6)
    # 護欄與 v2 相同
    assert w["W_TAUBAR"] == re.W_TAUBAR and w["W_ERRBAR"] == re.W_ERRBAR


def test_v2_2_env_runs_and_reports_true_exec_rate():
    env = re.MaxCpgEnv(preset="v2.2")
    assert env.observation_size == 66 and env.action_size == 10
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    assert s.obs.shape == (66,)
    a = jnp.array(re.baseline_action("nomux"))
    for i in range(120):                   # > 1 個步態週期，讓每腿都結算過一次執行率
        s = step(s, a)
        assert float(s.done) == 0.0
    ef, er = float(s.metrics["exec_f"]), float(s.metrics["exec_r"])
    assert 0.5 < ef < 1.3 and 0.9 < er < 2.0     # 基準量級：前 ~0.9、後 ~1.5（Trace 同量）
    assert float(s.info["rate_last"].min()) > 0.3


def test_v2_2_reward_shares_on_baseline():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "diag"))
    import rl_calibrate
    r = rl_calibrate.calibrate("v2.2", steps=300, verbose=False)
    sh, b = r["share"], re.CAL_BANDS["v2.2"]
    roll = sh["t_roll"] + sh["t_rollrate"]
    pitch = sh["t_pitch"] + sh["t_pitchrate"]
    assert b["roll"][0] <= roll <= b["roll"][1], f"roll {roll:.1%}"
    assert b["pitch"][0] <= pitch <= b["pitch"][1], f"pitch {pitch:.1%}"
    assert sh["t_exec"] <= b["exec_max"]
    assert sh["t_taubar"] < 0.01 and sh["t_errbar"] < 0.01
    assert 0.5 < r["metrics"]["exec_f"] < 1.3


def test_v2_2_symmetry_penalises_left_right_too():
    """對稱項＝前後差² + 左右差²：直接餵 rate_last 驗公式（左右差 0.5 → +0.25）。"""
    env = re.MaxCpgEnv(preset="v2.2")
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    a = jnp.array(re.baseline_action("nomux"))
    s = step(s, a)
    base_sym = float(s.metrics["t_sym"])
    # 人為把 rate_last 設成左右不對稱（前後平均相同）：左 0.75、右 1.25
    s2 = s.replace(info={**s.info, "rate_last": jnp.array([1.25, 0.75, 1.25, 0.75])})
    s2 = step(s2, a)
    # 這一步沒有腿結算（rate_last 沿用）→ sym = (f−r)² + (l−r)² = 0 + 0.25，W_SYM=1
    assert abs(float(s2.metrics["t_sym"]) - 0.25) < 1e-3 or float(s2.metrics["t_sym"]) > base_sym


def test_v2_3_preset_and_startup_ramp():
    """v2.3 = v2.2 + 動作抖振/護欄權重 + 起步淡入。淡入：第 0 步不管 policy 出什麼，都等於基準動作。"""
    w22, w23 = re.weights_of("v2.2"), re.weights_of("v2.3")
    changed = {k for k in w23 if w23[k] != w22[k]}
    assert changed == {"W_ACT", "W_TAUBAR", "RAMP_STEPS", "W_YAW", "YAW_SIG2_WIDE"}
    assert w23["RAMP_STEPS"] == 50 and w22["RAMP_STEPS"] == 0
    env = re.MaxCpgEnv(preset="v2.3")
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "delay": jnp.int32(0)})
    wild = jnp.full(10, 8.0)                       # tanh → sway +60 mm、ω 2.0
    s1 = step(s, wild)
    np.testing.assert_allclose(np.asarray(s1.info["sway"]), 0.0, atol=1e-6)   # 第 0 步 u=0 → 基準（sway 0）
    for _ in range(60):
        s1 = step(s1, wild)
    assert float(jnp.abs(s1.info["sway"]).max()) > 0.03                          # 淡入完成後 policy 生效


def test_v2_3_yaw_kernel_has_gradient_when_turning():
    """雙尺度核：轉彎誤差 0.24 rad/s 時窄核 ≈ 0（沒梯度），雙尺度核仍 > 0.02 且對誤差單調。"""
    w = re.weights_of("v2.3")
    narrow = float(re.yaw_reward(0.06, 0.30, w["YAW_SIG2"]))
    dual = float(re.yaw_reward(0.06, 0.30, w["YAW_SIG2"], w["YAW_SIG2_WIDE"]))
    assert narrow < 1e-6 and dual > 0.02
    e = np.array([0.0, 0.05, 0.1, 0.2, 0.3])
    r = [float(re.yaw_reward(x, 0.0, w["YAW_SIG2"], w["YAW_SIG2_WIDE"])) for x in e]
    assert all(r[i] > r[i + 1] for i in range(len(r) - 1))
    # 直走漂移對比仍在：0 漂 vs 0.8°/s 漂
    assert r[0] - float(re.yaw_reward(0.014, 0.0, w["YAW_SIG2"], w["YAW_SIG2_WIDE"])) > 0.1


def test_v2_4_head_err_in_obs_and_integrates_gyro():
    """v2.4：obs 67 維；head_err 隨 (gyro_z − cmd_wz) 積分；轉彎指令下若不轉，head_err 變負且獎勵下降。"""
    w = re.weights_of("v2.4")
    assert w["HEAD_OBS"] and w["W_HEAD"] == 1.0 and w["GYRO_BIAS_Z"] == 0.005
    env = re.MaxCpgEnv(preset="v2.4")
    assert env.observation_size == 67 and env.action_size == 10
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array([0.30, 0.30]), "gyro_bias": jnp.zeros(3),
                        "imu_q": jnp.array([1.0, 0.0, 0.0, 0.0])})
    a = jnp.array(re.baseline_action("nomux"))
    heads, th = [], []
    for i in range(100):                         # 基準動作不轉 → head_err ≈ −0.3·t
        s = step(s, a)
        heads.append(float(s.info["head_err"])); th.append(float(s.metrics["t_head"]))
    assert heads[-1] < -0.4 and heads[-1] > -0.8          # ≈ −0.3×2 s ＝ −0.6（加上步態偏航擾動）
    assert th[-1] < 0.1                                    # 8.6° 以上獎勵已掉到 <0.37；0.6 rad → ≈0
    assert abs(float(s.obs[32]) - max(-1.0, heads[-1])) < 1e-5   # obs 第 32 維就是 head_err（clip ±1）


def test_v2_4_reward_shares_on_baseline():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "diag"))
    import rl_calibrate
    r = rl_calibrate.calibrate("v2.4", steps=300, verbose=False)
    sh, b = r["share"], re.CAL_BANDS["v2.4"]
    roll = sh["t_roll"] + sh["t_rollrate"]
    assert b["roll"][0] <= roll <= b["roll"][1], f"roll {roll:.1%}"
    assert sh["t_head"] > 0.05 and sh["t_taubar"] < 0.01
