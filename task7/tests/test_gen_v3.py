"""gen_v3（狗上 numpy 產生器）vs rl_env_v3（jax）：逐函式、逐步釘住。jax 是 float32，容差 2e-5。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3  # noqa: E402
import gen_v3 as g  # noqa: E402

NPZ = ROOT / "weights" / "cpg_rl_v3_6f_np.npz"
ATOL = 2e-5


@pytest.fixture(scope="module")
def p():
    return g.load_params(str(NPZ))


@pytest.fixture(scope="module")
def env():
    return v3.DualModeEnv(gains="factory", weights=v3.W36, ref=dict(cyc_amp_rand=False), push=False)


def _cmds(n, seed=0):
    rng = np.random.default_rng(seed)
    C = np.stack([rng.uniform(-0.4, 0.9, n), rng.uniform(-0.3, 0.3, n), rng.uniform(-1.3, 1.3, n)], 1)
    C[::7, 1] = 0.0; C[::5, 2] = 0.0; C[::11] = 0.0          # 混些零軸與純站立
    return C


def _close(a, b, atol=ATOL, msg=""):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=0, err_msg=msg)


def _close_phase(a, b, atol=ATOL, msg=""):
    """相位 ∈[0,1) 要環狀比：正好 0 的相位在 float32 會落到 1.0、float64 落到 ~1e-23，差 1 但是同一個相位。"""
    d = np.mod(np.asarray(a, float) - np.asarray(b, float) + 0.5, 1.0) - 0.5
    np.testing.assert_allclose(d, np.zeros_like(d), atol=atol, rtol=0, err_msg=msg)


def test_params_loaded(p):
    assert p.ref["cyc_turn_sym"] is True and p.ref["wheel_space"] == "tau" and p.K["ACT_DIM"] == 24
    assert p.T["CYC_OFF"].shape == (4, 100, 12) and p.T["LO12"].shape == (12,) and p.T["MIRROR_LR"].dtype.kind == "i"


@pytest.mark.parametrize("i", range(0, 200, 10))
def test_activity_chain_matches_jax(p, env, i):
    cmd = _cmds(200)[i]
    A_n, A_j = g.activity(cmd, p), v3.activity(jnp.array(cmd, jnp.float32), env.ref)
    for k in A_j:
        _close(A_n[k], A_j[k], msg=f"activity.{k}")
    _close(g.leg_activity(cmd, A_n, p), v3.leg_activity(jnp.array(cmd, jnp.float32), A_j), msg="leg_activity")
    _close_phase(g.phase_offsets(cmd, A_n, p), v3.phase_offsets(jnp.array(cmd, jnp.float32), A_j, env.ref), msg="phase_offsets")
    _close(g.kin_step_vec(cmd, 0.4762, p), v3.kin_step_vec(jnp.array(cmd, jnp.float32), 0.4762, env.ref), msg="kin_step_vec")
    _close(g.posture_offset(cmd, A_n, p), v3.posture_offset(jnp.array(cmd, jnp.float32), A_j, env.ref), msg="posture_offset")
    res = np.array([0.3, -0.2, 0.1, 0.0])
    _close(g.wheel_cmd(cmd, A_n, res, p), v3.wheel_cmd(jnp.array(cmd, jnp.float32), A_j, jnp.array(res), env.ref), msg="wheel_cmd")
    P_n, P_j = g.step_pattern(cmd, p), v3.step_pattern(jnp.array(cmd, jnp.float32), env.ref)
    for k in ("s", "s_kin", "g", "vec", "wheel0", "hz", "duty", "cyc_on", "lift", "post_y"):
        _close(P_n[k], P_j[k], msg=f"step_pattern.{k}")
    _close_phase(P_n["ph"], P_j["ph"], msg="step_pattern.ph")


def test_act_split_and_slew(p):
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.normal(size=24) * 2
        S_n, S_j = g.act_split(a, p), v3.act_split(jnp.array(a, jnp.float32))
        for k in S_j:
            _close(S_n[k], S_j[k], msg=f"act_split.{k}")
        ph, tgt = rng.uniform(0, 1, 4), rng.uniform(0, 1, 4)
        _close(g.slew_phase(ph, tgt, p), v3.slew_phase(jnp.array(ph, jnp.float32), jnp.array(tgt, jnp.float32)))
        s = rng.uniform(0, 1, 4)                                  # 兩邊要吃同一組亂數，不能各抽一次
        _close(g.step_gain(s, p), v3.step_gain(jnp.array(s, jnp.float32)), atol=1e-6)


def test_foot_targets_and_ik_match(p, env):
    rng = np.random.default_rng(2)
    for _ in range(100):
        th = rng.uniform(0, 2 * np.pi, 4); amp = rng.uniform(0, 1.3, 4); gg = rng.uniform(0, 1, 4)
        vec = rng.normal(size=(4, 2)) * 0.05; lift4 = rng.uniform(0, 0.05, 4); sway = rng.normal(size=2) * 0.03
        u = rng.uniform(0, 1); fx = rng.normal(size=4) * 0.02; fz = rng.normal() * 0.02; duty = rng.uniform(0.5, 0.8)
        post = rng.normal(size=4) * 0.05
        F_n = g.foot_targets(th, amp, gg, vec, lift4, sway, u, fx, fz, duty, post, env.ref["z_sag"], p)
        F_j = v3.foot_targets(*[jnp.array(x, jnp.float32) for x in (th, amp, gg, vec, lift4, sway)], jnp.float32(u),
                              jnp.array(fx, jnp.float32), jnp.float32(fz), jnp.float32(duty), jnp.array(post, jnp.float32), env.ref["z_sag"])
        _close(F_n, F_j, msg="foot_targets")
        _close(g.joint_targets(F_n, p), v3.joint_targets(jnp.array(F_n, jnp.float32)), atol=1e-4, msg="joint_targets(ik)")


def test_zero_action_stance_is_identity(p):
    """cmd=0、零動作、u=0 的足端目標 == 站姿足端 → IK 回 stance_q12（起身目標就靠這條）。"""
    A = g.act_split(np.zeros(24), p)
    P = g.step_pattern(np.zeros(3), p)
    F = g.foot_targets(np.zeros(4), np.zeros(4), P["g"], P["vec"], P["lift"], np.zeros(2), 0.0, A["foot_x"], A["foot_z"],
                       P["duty"], P["post_y"], p.ref["z_sag"], p)
    _close(F, p.T["STANCE_FEET"], atol=1e-12)
    # STANCE_FEET 是 jax float32 FK 算出來再存 npz，IK 往返只能還原到 ~1e-7 rad（1e-9 是 float32 來源做不到的）
    _close(g.joint_targets(F, p), p.T["STANCE_Q12"], atol=1e-6)


def test_cycle_offsets_match_jax(p, env):
    rng = np.random.default_rng(3)
    for cmd in _cmds(60, seed=4):
        phi = rng.uniform(0, 2 * np.pi)
        A_n, A_j = g.activity(cmd, p), v3.activity(jnp.array(cmd, jnp.float32), env.ref)
        C_n = g.cycle_offsets(phi, cmd, A_n, p)
        C_j = v3.cycle_offsets(jnp.float32(phi), jnp.array(cmd, jnp.float32), A_j, env.ref, env.ref["cyc_amp_turn"])
        for k in ("delta", "wheel_tau", "hz", "on"):
            _close(C_n[k], C_j[k], atol=5e-5, msg=f"cycle_offsets.{k}")


def test_wheel_dead_matches_wheel_ctrl_tau(p, env):
    kv = env.kv_wheel
    for tau in (np.array([0.05, -0.05, 0.2, -0.2]), np.array([0.13, -0.13, 2.0, -3.5])):
        v = np.array([0.1, 0.2, 0.3, 0.4])
        ctrl = np.asarray(v3.wheel_ctrl_tau(jnp.array(tau, jnp.float32), jnp.array(v, jnp.float32), kv))
        _close(g.wheel_dead(tau, p), (ctrl - v) * kv, atol=1e-5)


def _env_rollout_ctrl(env, cmd, steps, acts, seed=0):
    """env 跑 steps 步（固定 cmd、給定動作序列），回每步的 (ctrl_leg12, ctrl_wheel4, v_wheel_meas4, delay, c)。
    q_des 就是 pipeline_state.ctrl[LEG_ACT_IDX]；輪 ctrl = v_meas + τ_dead/kv（wheel_ctrl_tau）。
    也回 reset 當下的 cmd：env.reset 的 u_mode／ph／s4／s_arc 是用「抽到的那個 cmd」算的（我們只改之後每步的 cmd），
    所以 Gen 的 cmd0 要餵 reset cmd，不是跟跑用的 cmd —— 順便讓 u_mode 從別的值斜率追過來，多測到 slew 那條路。"""
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(seed))
    cmd_reset = np.asarray(s.info["cmd"], float)
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd, jnp.float32), "cmd2": jnp.array(cmd, jnp.float32), "t_switch": 10 ** 6})
    delay = int(s.info["delay"])
    out = []
    for i in range(steps):
        v_meas = np.asarray(s.pipeline_state.qvel[v3.WHEEL_QVEL_IDX])
        s = js(s, jnp.array(acts[i], jnp.float32))
        d = s.pipeline_state
        out.append(dict(q_des=np.asarray(d.ctrl[v3.LEG_ACT_IDX]), wctrl=np.asarray(d.ctrl[v3.WHEEL_ACT_IDX]), v_meas=v_meas,
                        c={k: np.asarray(v) for k, v in s.info["c"].items()}, s4=float(s.info["s4"]), s_arc=float(s.info["s_arc"])))
    return out, delay, cmd_reset


@pytest.mark.parametrize("cmd", [(0.5, 0.0, 0.0), (0.0, 0.20, 0.0), (0.0, 0.0, 1.3), (0.0, -0.12, -0.8)])
def test_gen_step_shadows_env_ctrl(p, env, cmd):
    """300 步、固定隨機動作序列：Gen 的 q_des 對 env 的 ctrl < 1e-4 rad、τ（死區後）< 1e-3。
    env 有 1–2 步動作延遲 → 測試端餵 Gen 的是 acts[i-delay]；env 的 u_ramp 與 Gen 的 i 同步。"""
    steps = 300
    rng = np.random.default_rng(5)
    acts = rng.normal(size=(steps, 24)) * 0.5
    ref_rows, delay, cmd_reset = _env_rollout_ctrl(env, cmd, steps, acts)
    gen = g.Gen(p, cmd0=cmd_reset)
    worst_q = worst_t = 0.0
    for i, row in enumerate(ref_rows):
        act = acts[i - delay] if i >= delay else np.zeros(24)
        q_des, tau, c = gen.step(np.array(cmd), act, row["v_meas"])
        tau_d = g.wheel_dead(tau, p)
        w_ref = (row["wctrl"] - row["v_meas"]) * env.kv_wheel
        worst_q = max(worst_q, float(np.max(np.abs(q_des - row["q_des"]))))
        worst_t = max(worst_t, float(np.max(np.abs(tau_d - w_ref))))
        for k in ("theta", "amp", "cyc_abad", "cyc_knee"):
            assert np.max(np.abs(c[k] - row["c"][k])) < 1e-3, (i, k)
        assert abs(gen.s4 - row["s4"]) < 1e-6 and abs(gen.s_arc - row["s_arc"]) < 1e-6
    print(f"cmd {cmd} worst q_des {worst_q:.2e} rad, tau {worst_t:.2e} N·m, n_clamp {gen.n_clamp}")
    assert worst_q < 1e-4 and worst_t < 1e-3
    assert gen.n_clamp == 0                       # 影子跟跑全程 IK 都構得到；非 0 代表足端目標飛出工作空間


def test_gen_zero_action_cmd0_holds_stance(p):
    gen = g.Gen(p)
    for _ in range(100):
        q, tau, _ = gen.step(np.zeros(3), np.zeros(24), np.zeros(4))
    # 1e-9 是 float32 來源做不到的：STANCE_FEET／STANCE_Q12 是 jax float32 存進 npz，IK 往返只能還原到 ~1e-7 rad
    _close(q, gen.stance_q12(), atol=1e-6)
    _close(tau, np.zeros(4), atol=1e-12)
    assert gen.head_err == 0.0


def test_gen_n_sat_counts_joint_limit_saturation(p):
    """★ `np.clip(q_ik + delta + qres, LO12, HI12)` 是靜默的：橫移時 ABAD 頂在 ±0.523 的機構停點，
    產生器想要的那一段位移根本沒送出去，log 上卻完全看不出來 —— 所以要數出來（n_sat 是步數，不是軸次）。
    零動作站立（cmd 0）一定是 0；橫移 0.12 m/s 會踩到，而且只踩 ABAD（索引 0/3/6/9）。"""
    gen = g.Gen(p)
    for _ in range(100):
        gen.step(np.zeros(3), np.zeros(24), np.zeros(4))
    assert gen.n_sat == 0 and not gen.sat_mask.any()

    gen = g.Gen(p)
    seen = np.zeros(12, dtype=bool)
    for i in range(350):
        cmd = np.zeros(3) if i < 50 else np.array([0.0, 0.12, 0.0])
        gen.step(cmd, np.zeros(24), np.zeros(4))
        seen |= gen.sat_mask
    assert gen.n_sat > 0, "橫移 350 步一次都沒頂到限位 → 先確認 sat_mask 真的有在算"
    ABAD = {0, 3, 6, 9}
    hit = set(int(k) for k in np.flatnonzero(seen))
    assert hit and hit <= ABAD, f"飽和不該發生在 ABAD 以外：{sorted(hit)}"
    assert hit == {3, 9}, f"+vy（往左）頂住的是左側兩條腿的 ABAD；變成 {sorted(hit)} 代表限位或姿態偏置動過"


def test_gen_cmd_prev_is_a_copy_not_an_alias(p):
    """★ 呼叫端常常原地改同一個 cmd 陣列（遙控就是）。存參考的話 cmd_prev 會跟著變 ——
    而 obs 帶的就是 cmd_prev，「上一步的指令」會偷偷變成「這一步的」，鏡像判斷跟著錯。"""
    cmd = np.array([0.0, 0.12, 0.0])
    gen = g.Gen(p, cmd0=cmd)
    cmd[1] = -0.30                                  # 原地改：cmd_prev 不該跟著動
    assert gen.cmd_prev[1] == 0.12
    cmd2 = np.array([0.0, 0.20, 0.0])
    gen.step(cmd2, np.zeros(24), np.zeros(4))
    cmd2[1] = -0.30
    assert gen.cmd_prev[1] == 0.20


def test_gen_head_err_integrates_after_first_step(p):
    gen = g.Gen(p, cmd0=(0.0, 0.0, 1.0))
    gen.integrate_head(0.5)                         # i == 0：不積分（env reset 的 head_err 是 0）
    assert gen.head_err == 0.0
    gen.step(np.array([0.0, 0.0, 1.0]), np.zeros(24), np.zeros(4))
    gen.integrate_head(0.5)
    assert abs(gen.head_err - (0.5 - 1.0) * p.K["CTRL_DT"]) < 1e-12


def test_joint_targets_n_reports_ik_clamp(p):
    """構不到的足端目標 → kin.ik 沿徑向縮，clamped 要被數出來（靜默縮限＝動作變鈍查不出原因）。"""
    feet = np.asarray(p.T["STANCE_FEET"], float).copy()
    q_ok, n_ok = g.joint_targets_n(feet, p)
    assert n_ok == 0
    _close(q_ok, g.joint_targets(feet, p), atol=1e-12)
    feet[0, 0] += 1.0                                # 前右腿往前 1 m：遠超二連桿長度
    _, n_bad = g.joint_targets_n(feet, p)
    assert n_bad > 0


def test_params_guards_raise_valueerror(p):
    """守門是上機安全閘，要用 ValueError（python -O 會把 assert 拿掉）；npz 存的一定是 SYM 週期表。"""
    for bad in (dict(wheel_space="vel"), dict(cyc_lat_decouple=True), dict(cyc_recenter=False), dict(cyc_turn_sym=False)):
        ref = dict(p.ref, **bad)
        with pytest.raises(ValueError):
            g.Params(ref, p.K, p.T)
    g.Params(dict(p.ref), p.K, p.T)                 # 原樣要過


def test_gen_records_v_nom_and_v_meas(p):
    gen = g.Gen(p)
    v = np.array([0.5, -0.5, 1.0, 2.0])
    _, tau, _ = gen.step(np.array([0.3, 0.0, 0.0]), np.zeros(24), v)
    np.testing.assert_allclose(gen.v_nom, np.full(4, 0.3 / p.ref["r_wheel"]))
    np.testing.assert_array_equal(gen.v_meas, v)
    np.testing.assert_allclose(tau, p.ref["wheel_outer_gain"] * (gen.v_nom - v))      # 零動作、u4=0：τ 就只有外環
