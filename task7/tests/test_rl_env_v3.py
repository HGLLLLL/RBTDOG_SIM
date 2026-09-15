"""v3.1 統一運動學產生器：活動度表、步向量方向、相位鏡像與插值、輪速 gate、足端目標、常數對資料集；env 形狀與短 rollout。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3  # noqa: E402

DS = ROOT / "outputs" / "ref_gait_dataset.json"
FR, FL, RR, RL = 0, 1, 2, 3
C = lambda vx, vy, wz: jnp.array([vx, vy, wz])   # noqa: E731


def A(cmd):
    return v3.activity(cmd)


def test_activity_table():
    a = A(C(0.5, 0, 0));      assert float(a["s4"]) == 0 and float(a["arc"]) == 0 and float(a["wheel"]) == 1
    a = A(C(0, 0, 1.3));      assert float(a["s4"]) == 1 and float(a["arc"]) == 0
    a = A(C(0, 0.08, 0));     assert float(a["s4"]) == 1 and float(a["arc"]) == 0
    a = A(C(0.5, 0, 0.5));    assert abs(float(a["arc"]) - 0.5 / 1.3) < 1e-6 and float(a["s4"]) == 0   # 弧線
    a = A(C(0.3, 0.06, 0));   assert abs(float(a["s4"]) - 0.75) < 1e-6 and float(a["arc"]) == 0       # 斜走
    a = A(C(0, 0, 0));        assert float(a["s4"]) == 0 and float(a["arc"]) == 0


def test_leg_activity_dominant_and_secondary():
    cmd = C(0, 0, 1.3); s = np.asarray(v3.leg_activity(cmd, A(cmd)))
    assert np.allclose(s[[FL, RR]], 1.0) and np.allclose(s[[FR, RL]], 0.7)                    # 左轉主導 FL+RR
    cmd = C(0, 0, -1.3); s = np.asarray(v3.leg_activity(cmd, A(cmd)))
    assert np.allclose(s[[FR, RL]], 1.0) and np.allclose(s[[FL, RR]], 0.7)
    cmd = C(0, 0.08, 0); s = np.asarray(v3.leg_activity(cmd, A(cmd)))
    assert np.allclose(s[[FL, RL]], 1.0) and np.allclose(s[[FR, RR]], 0.7)                    # 左移主導 FL+RL
    cmd = C(0.5, 0, 1.3); s = np.asarray(v3.leg_activity(cmd, A(cmd)))
    assert s[FL] == 1 and abs(s[RR] - 0.5) < 1e-6 and s[FR] == 0 and s[RL] == 0               # 左弧線：內前 FL、對角 RR
    cmd = C(0.5, 0, -1.3); s = np.asarray(v3.leg_activity(cmd, A(cmd)))
    assert s[FR] == 1 and abs(s[RL] - 0.5) < 1e-6 and s[FL] == 0 and s[RR] == 0
    cmd = C(0.5, 0, 0); assert float(jnp.max(v3.leg_activity(cmd, A(cmd)))) == 0


def test_kin_step_vec_directions():
    T = 0.5
    v = np.asarray(v3.kin_step_vec(C(0, 0, 1.3), T))
    assert v[FL][0] < 0 and v[FL][1] > 0 and v[RR][0] > 0 and v[RR][1] < 0          # 左轉：FL 往左後、RR 往右前
    assert v[FR][0] > 0 and v[FR][1] > 0 and v[RL][0] < 0 and v[RL][1] < 0          # ω×r：FR (+,+)、RL (−,−)
    assert abs(v[FL][1] - v3.REF["rot_step_frac"] * T * 1.3 * float(v3.FOOT_XY_BODY[FL, 0])) < 1e-6
    v = np.asarray(v3.kin_step_vec(C(0, 0.08, 0), T))
    assert np.allclose(v[:, 1], 0.04 * v3.REF["lat_cmd_gain"]) and np.allclose(v[:, 0], 0.0)   # 左移：四腿同向，40 mm × 側向命令增益
    v = np.asarray(v3.kin_step_vec(C(0.5, 0, 0), T))
    assert np.allclose(v, 0.0)                                                        # 純 vx 不踏步


def test_phase_offsets_mirror_and_blend():
    fac = dict(v3.REF, phase_set_turn="factory", phase_set_lat="factory")
    ph_t = np.asarray(v3.phase_offsets(C(0, 0, 1.3), A(C(0, 0, 1.3)), fac))
    assert np.allclose(ph_t, np.asarray(v3.PH_TURN_L))
    ph_tr = np.asarray(v3.phase_offsets(C(0, 0, -1.3), A(C(0, 0, -1.3)), fac))
    assert np.allclose(ph_tr, np.asarray(v3.PH_TURN_L)[[FL, FR, RL, RR]])
    ph_l = np.asarray(v3.phase_offsets(C(0, 0.08, 0), A(C(0, 0.08, 0)), fac))
    assert np.allclose(ph_l, np.asarray(v3.PH_LAT_L))
    # 預設：旋轉族小跑（FL+RR 0、FR+RL 0.5）、平移族原廠
    assert np.allclose(np.asarray(v3.phase_offsets(C(0, 0, 1.3), A(C(0, 0, 1.3)))), np.asarray(v3.PH_TROT))
    assert np.allclose(np.asarray(v3.phase_offsets(C(0, 0.08, 0), A(C(0, 0.08, 0)))), np.asarray(v3.PH_LAT_L))
    mix = np.asarray(v3.phase_offsets(C(0, 0.08, 1.3), A(C(0, 0.08, 1.3)), fac))
    for k in range(4):
        gap = abs((ph_t[k] - ph_l[k] + 0.5) % 1 - 0.5)
        d1 = abs((mix[k] - ph_l[k] + 0.5) % 1 - 0.5); d2 = abs((mix[k] - ph_t[k] + 0.5) % 1 - 0.5)
        assert d1 <= gap + 1e-6 and d2 <= gap + 1e-6


def test_slew_phase_is_rate_limited_and_circular():
    ph = jnp.array([0.0, 0.0, 0.0, 0.0]); tgt = jnp.array([0.5, 0.01, 0.95, 0.0])
    out = np.asarray(v3.slew_phase(ph, tgt))
    assert min(abs(out[0] - 0.02), abs(out[0] - 0.98)) < 1e-6                                   # 差 0.5 兩邊等距，走哪邊都對
    assert abs(out[1] - 0.01) < 1e-6 and abs(out[2] - 0.98) < 1e-6 and out[3] == 0


def test_wheel_cmd_gate_lateral_pattern_and_residual():
    z = jnp.zeros(4)
    w = np.asarray(v3.wheel_cmd(C(0.5, 0, 0), A(C(0.5, 0, 0)), z))
    assert np.allclose(w, 0.5 / v3.REF["r_wheel"])
    w = np.asarray(v3.wheel_cmd(C(0, 0, 1.3), A(C(0, 0, 1.3)), z))
    assert w[FR] > 4.0 and w[RL] < -4.0 and abs(w[FL]) < 1e-6 and abs(w[RR]) < 1e-6      # 左轉：主導對 FL/RR 的輪 0、FR/RL 2 倍
    w = np.asarray(v3.wheel_cmd(C(0.5, 0, 0.5), A(C(0.5, 0, 0.5)), z))
    assert w[FR] > w[FL] > 0 and w[RR] > w[RL] > 0                                       # 弧線：右快左慢、四輪都轉
    w = np.asarray(v3.wheel_cmd(C(0.05, 0, 0), A(C(0.05, 0, 0)), z))
    assert np.allclose(w, 0.05 / v3.REF["r_wheel"])                                      # 慢直走 gate 不作怪
    w = np.asarray(v3.wheel_cmd(C(0, 0.08, 0), A(C(0, 0.08, 0)), z))
    assert w[FL] > 3.0 and w[RL] < -2.0 and abs(w[FR]) < 1e-6 and abs(w[RR]) < 1e-6      # 左移圖案
    w2 = np.asarray(v3.wheel_cmd(C(0, 0.08, 0), A(C(0, 0.08, 0)), jnp.array([1.0, -1.0, 0.5, 0.0])))
    assert np.allclose(w2 - w, [1.0, -1.0, 0.5, 0.0])


def test_posture_offset_arc_only():
    p = np.asarray(v3.posture_offset(C(0.5, 0, 1.3), A(C(0.5, 0, 1.3))))
    assert p[FL] < 0 and p[RL] > 0 and p[FR] == 0 and p[RR] == 0                          # 左弧線：內側 FL 往中線(−y)、RL 往外(+y)
    p = np.asarray(v3.posture_offset(C(0.5, 0, -1.3), A(C(0.5, 0, -1.3))))
    assert p[FR] > 0 and p[RR] < 0 and p[FL] == 0
    assert np.allclose(np.asarray(v3.posture_offset(C(0, 0, 1.3), A(C(0, 0, 1.3)))), 0.0)  # 原地轉沒有


def test_act_split_wheel_residual_units():
    a = jnp.zeros(v3.ACT_DIM).at[8].set(10.0)
    assert abs(float(v3.act_split(a)["wres"][0]) - v3.WHEEL_RES) < 1e-4 and v3.WHEEL_RES == 2.0


def test_step_pattern_keys_and_foot_targets_shape():
    P = v3.step_pattern(C(0.3, 0.06, 0.4))
    assert set(P) >= {"A", "s", "g", "vec", "ph", "wheel0", "hz", "duty", "lift", "post_y"}
    # 抬高不隨指令大小縮：vy 0.04（s4 0.5）與 vy 0.08 的主導腿抬高一樣，次要腿 0.7 倍
    kin = dict(v3.REF, step_gen_lat="kin", step_gen_turn="kin")                       # 運動學路徑（v3.2 平移族預設走原廠週期）
    l1, l2 = np.asarray(v3.step_pattern(C(0, 0.04, 0), kin)["lift"]), np.asarray(v3.step_pattern(C(0, 0.08, 0), kin)["lift"])
    assert np.allclose(l1, l2) and abs(l2[FL] - v3.REF["lift_lat"]) < 1e-6 and abs(l2[FR] - 0.7 * v3.REF["lift_lat"]) < 1e-6   # 平移族用 lift_lat
    assert np.allclose(np.asarray(v3.step_pattern(C(0.5, 0, 0))["lift"]), 0.0)
    # 族別混合：純旋轉用旋轉族步頻，純平移用平移族
    assert abs(float(v3.step_pattern(C(0, 0, 1.3))["hz"]) - v3.REF["step_hz_turn"]) < 1e-6
    assert abs(float(v3.step_pattern(C(0, 0.08, 0))["hz"]) - v3.REF["step_hz_lat"]) < 1e-6
    feet = v3.foot_targets(jnp.zeros(4), jnp.ones(4), P["g"], P["vec"], P["lift"], jnp.zeros(2), 1.0, jnp.zeros(4), 0.0, P["duty"], P["post_y"])
    assert feet.shape == (4, 3)
    feet0 = v3.foot_targets(jnp.zeros(4), jnp.ones(4), jnp.zeros(4), jnp.zeros((4, 2)), jnp.zeros(4), jnp.zeros(2), 0.0, jnp.zeros(4), 0.0, 0.5, jnp.zeros(4))
    assert np.allclose(np.asarray(feet0), np.asarray(v3.STANCE_FEET_j))


@pytest.mark.skipif(not DS.exists(), reason="資料集未產生")
def test_ref_constants_match_dataset():
    d = json.loads(DS.read_text(encoding="utf-8"))["summary"]
    assert abs(v3.REF["L_eff"] - d["L_eff_m"]) < 0.02
    q = np.array(d["wheel_stance_q12"])
    assert abs(v3.REF["stance_q12"][1] - q[1]) < 0.02 and abs(v3.REF["stance_q12"][2] - q[2]) < 0.02
    V = d["v31"]
    pt, pl = V["phase_turn_left"], V["phase_lat_left"]
    for name, shm, ph in (("PH_TURN_L", pt, v3.PH_TURN_L), ("PH_LAT_L", pl, v3.PH_LAT_L)):
        for L, s in (("FR", "fr"), ("FL", "fl"), ("RR", "br"), ("RL", "bl")):
            assert abs((float(ph[v3.LEG_IDX[L]]) - shm[s] + 0.5) % 1 - 0.5) < 0.08, (name, L)
    assert abs(float(v3.W_TURN_L[FR]) - V["apex_ratio_turn"]) < 0.15
    assert abs(float(v3.W_LAT_L[FR]) - V["apex_ratio_lat"]) < 0.15
    assert abs(v3.REF["factory_hz_turn"] - d["turn_left"]["freq_hz"]) < 0.15
    g = V["gains_motion"]
    assert g["abad_kp"] == [60.0] and g["hip_kp"] == [120.0] and g["leg_kd"] == [1.0] and g["wheel_kp"] == [0.0] and g["wheel_kd"] == [0.1]


def test_obs_layout_pinned():
    """狗上 obs 組裝要照這個切；改了就要一起改 realbot。第 37–38 格是 [s4, s_arc]，40–63 是上一步 24 維動作。"""
    assert v3.ACT_DIM == 24
    assert 3 + 3 + 12 + 12 + 4 + 3 + 2 + 1 + 24 + 24 == 88


def _rollout(env, cmd, steps, jit_reset, jit_step):
    s = jit_reset(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": cmd, "cmd2": cmd, "t_switch": 10 ** 6})
    a = jnp.zeros(v3.ACT_DIM)
    M = {k: [] for k in ("vx", "vy", "wz", "tau_pk", "clr_step", "clr_stance", "s4", "s_arc")}
    done = []
    for _ in range(steps):
        s = jit_step(s, a)
        for k in M:
            M[k].append(float(s.metrics[k]))
        done.append(float(s.done))
    return s, {k: np.array(v) for k, v in M.items()}, max(done)


def test_env_shapes_defaults_and_obs_mode_slots():
    env = v3.DualModeEnv()
    assert env.obs_dim == 88 and env.action_size == 24 and env.wheel_pos is False
    s = jax.jit(env.reset)(jax.random.PRNGKey(1))
    assert s.obs.shape == (88,) and s.info["ph"].shape == (4,) and s.info["phi_cyc"].shape == () and s.info["qres"].shape == (12,)
    assert 0.3 <= float(s.info["cyc_amp"]) <= 0.9
    cmd = np.asarray(s.obs[34:37]); a = v3.activity(jnp.array(cmd))
    assert abs(float(s.obs[37]) - float(a["s4"])) < 1e-6 and abs(float(s.obs[38]) - float(a["arc"])) < 1e-6


def test_env_g0_wheel_and_turn():
    env = v3.DualModeEnv()
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    s, M, done = _rollout(env, jnp.array([0.5, 0.0, 0.0]), 80, jit_reset, jit_step)
    assert done == 0.0 and M["tau_pk"].max() < v3.TAU_KILL
    assert abs(M["vx"][40:].mean() - 0.5) < 0.08 and M["clr_stance"][40:].max() < 10.0 and M["s4"].max() == 0.0
    s, M, done = _rollout(env, jnp.array([0.0, 0.0, 1.3]), 80, jit_reset, jit_step)
    assert done == 0.0 and M["tau_pk"].max() < v3.TAU_KILL and M["s4"].max() == 1.0     # v3.3 原地轉名目是原廠週期，開迴路 3–7 s 才倒，80 步內不倒
    envk = v3.DualModeEnv(ref=dict(step_gen_turn="kin"))
    s, M, done = _rollout(envk, jnp.array([0.0, 0.0, 1.3]), 80, jax.jit(envk.reset), jax.jit(envk.step))
    assert done == 0.0 and np.degrees(M["wz"][40:].mean()) > 10.0                           # 退路（小跑）仍然可用


def test_sample_cmd_covers_axes_and_combos():
    env = v3.DualModeEnv()
    f = jax.jit(jax.vmap(env._sample_cmd))
    c = np.asarray(f(jax.random.split(jax.random.PRNGKey(0), 4000)))
    nz = np.abs(c) > 1e-9
    assert 0.55 < nz[:, 0].mean() < 0.75 and 0.22 < nz[:, 1].mean() < 0.38 and 0.42 < nz[:, 2].mean() < 0.58
    assert (nz[:, 0] & nz[:, 1]).mean() > 0.1 and (nz[:, 0] & nz[:, 2]).mean() > 0.2      # 斜走、弧線都有
    assert c[:, 0].min() >= -0.4 and c[:, 0].max() <= 0.9 and np.abs(c[nz[:, 1], 1]).min() >= 0.03 and np.abs(c[nz[:, 2], 2]).min() >= 0.2


def test_cycle_offsets_direction_amplitude_and_freq():
    A_ = A(C(0, 0.08, 0))
    ph = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    D = np.stack([np.asarray(v3.cycle_offsets(p, C(0, 0.08, 0), A_)["delta"]) for p in ph])          # (50,12)
    assert np.ptp(D[:, 3]) > 0.5 and np.ptp(D[:, 9]) > 0.5                                             # 左移：FL、RL 的 ABAD 命令偏移擺幅 ≥ 0.5 rad
    Dr = np.stack([np.asarray(v3.cycle_offsets(p, C(0, -0.08, 0), A(C(0, -0.08, 0)))["delta"]) for p in ph])
    assert np.ptp(Dr[:, 0]) > 0.5 and np.ptp(Dr[:, 6]) > 0.5                                           # 右移：FR、RR
    hz_l = float(v3.cycle_offsets(0.0, C(0, 0.08, 0), A_)["hz"]); hz_t = float(v3.cycle_offsets(0.0, C(0, 0, 1.3), A(C(0, 0, 1.3)))["hz"])
    assert 2.0 < hz_l < 2.2 and 2.4 < hz_t < 2.7
    # 幅度隨指令（G0 量的映射）：vy 0.04 → amp0、0.08 → amp0 + slope·0.04
    r = v3.REF; a4, a8 = r["cyc_lat_amp0"], r["cyc_lat_amp0"] + r["cyc_lat_slope"] * 0.04
    d4 = np.asarray(v3.cycle_offsets(1.0, C(0, 0.04, 0), A(C(0, 0.04, 0)))["delta"]); d8 = np.asarray(v3.cycle_offsets(1.0, C(0, 0.08, 0), A_)["delta"])
    assert np.allclose(d4 * a8 / a4, d8, atol=1e-5)
    # 週期族開時，四腿運動學踏步關掉、弧線保留
    P = v3.step_pattern(C(0, 0.08, 0)); assert float(jnp.max(P["g"])) < 1e-4 and float(jnp.max(P["s"])) == 1.0
    P = v3.step_pattern(C(0.5, 0, 1.3)); assert float(P["g"][FL]) == 1.0
    kin = dict(v3.REF, step_gen_lat="kin", step_gen_turn="kin")
    P = v3.step_pattern(C(0, 0, 1.3), kin); assert float(jnp.max(P["g"])) == 1.0                     # 退路：原地轉走運動學小跑
    P = v3.step_pattern(C(0, 0, 1.3)); assert float(jnp.max(P["g"])) < 1e-4                          # v3.3 預設：原地轉走原廠週期
    assert float(jnp.max(v3.step_pattern(C(0, 0.08, 0), kin)["g"])) == 1.0
    assert np.allclose(np.asarray(v3.cycle_offsets(1.0, C(0, 0.08, 0), A_, kin)["delta"]), 0.0)


def test_err_barrier_per_joint():
    e = jnp.zeros(12).at[0].set(0.55).at[1].set(0.55)      # ABAD 0.55 < 0.6 免罰；HIP 0.55 > 0.45 罰
    assert abs(float(v3.err_barrier_j(e)) - 0.10 ** 2) < 1e-6


def test_yaw_reward_has_gradient_over_full_error_range():
    """原地轉名目 0.25 rad/s 對指令 1.3：高斯核全 0；線性項要在整段誤差有斜率。"""
    w = v3.W
    for e in (0.3, 0.6, 1.0, 1.4):
        g = float(v3.yaw_reward(jnp.array(e), jnp.array(0.0), w["YAW_SIG2"], w["YAW_SIG2_WIDE"]))
        lin = 1.0 - min(e / w["YAW_LIN_E"], 1.0)
        assert lin > 0.05 and lin < 1.0
    assert (1.0 - 0.6 / w["YAW_LIN_E"]) > (1.0 - 1.0 / w["YAW_LIN_E"])       # 誤差越小獎勵越高


def test_metrics_keys_match_between_reset_and_step_under_brax_wrapper():
    """brax 訓練 wrapper 用 lax.scan：reset/step 的 metrics 與 info 結構必須一致（Colab 2026-09-15 炸過）。"""
    from brax.envs.wrappers import training as tw
    env = tw.wrap(v3.DualModeEnv(), episode_length=5, action_repeat=1)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    s = jax.jit(env.reset)(keys)
    step = jax.jit(env.step)
    for _ in range(3):
        s = step(s, jnp.zeros((2, v3.ACT_DIM)))
    assert set(s.metrics) >= set(v3.METRIC_KEYS) and s.obs.shape == (2, 88)
    T = [k for k in s.metrics if k.startswith("t_")]
    assert set(T) == set(v3.T_KEYS)


def test_qres_action_is_slewed_and_bounded():
    """關節殘差：第 12–23 維，±0.12 rad，每步最多 0.01 rad。"""
    env = v3.DualModeEnv(); jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(0)); s = s.replace(info={**s.info, "cmd": C(0.5, 0, 0), "cmd2": C(0.5, 0, 0), "t_switch": 10 ** 6})
    a = jnp.zeros(v3.ACT_DIM).at[12].set(10.0)          # tanh → 1 → 目標 +0.12
    for i in range(60):
        s = js(s, a)
        if i == 2:
            assert 0 < float(s.info["qres"][0]) <= 3 * v3.QRES_SLEW + 1e-6
    assert abs(float(s.info["qres"][0]) - v3.QRES_MAX) < 1e-5 and float(jnp.abs(s.info["qres"][1:]).max()) < 1e-6


def test_turn_cycle_amp_override_and_random():
    A_ = A(C(0, 0, 1.3))
    d5 = np.asarray(v3.cycle_offsets(1.0, C(0, 0, 1.3), A_, amp_turn=0.5)["delta"]); d9 = np.asarray(v3.cycle_offsets(1.0, C(0, 0, 1.3), A_, amp_turn=0.9)["delta"])
    assert np.allclose(d5 * 0.9 / 0.5, d9, atol=1e-5)
    env = v3.DualModeEnv(); amps = [float(jax.jit(env.reset)(jax.random.PRNGKey(k)).info["cyc_amp"]) for k in range(20)]
    assert min(amps) >= 0.3 and max(amps) <= 0.9 and (max(amps) - min(amps)) > 0.3
    env2 = v3.DualModeEnv(ref=dict(cyc_amp_rand=False)); assert abs(float(jax.jit(env2.reset)(jax.random.PRNGKey(3)).info["cyc_amp"]) - v3.REF["cyc_amp_turn"]) < 1e-6
