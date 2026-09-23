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
