"""v3.5（spec docs/superpowers/specs/2026-09-16-cpg-rl-v3.5-drift-penalties-design.md）：ABAD 慢漂與速度漂移兩個低通懲罰。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3  # noqa: E402


def test_ema_filters_cycle_but_keeps_constant():
    """k=0.02 @ 50 Hz：2.5 Hz 週期擺被壓到 10% 以下（理論 6.4%），常值 5 s 內收斂到 1%。"""
    k, fs = 0.02, 50.0
    t = np.arange(int(5 * fs)) / fs
    e, out = 0.0, []
    for x in 0.4 * np.sin(2 * np.pi * 2.5 * t):
        e = float(v3.ema(e, x, k)); out.append(e)
    assert max(abs(v) for v in out[-int(fs):]) < 0.10 * 0.4
    e = 0.0
    for _ in range(int(5 * fs)):
        e = float(v3.ema(e, 0.26, k))
    assert abs(e - 0.26) < 0.01 * 0.26


def test_drift_terms_default_zero_and_w35_magnitudes():
    ab = jnp.array([0.26, 0.0, 0.0, 0.0]); dr = jnp.array([0.08, 0.0])     # 一腿 15°、往後漂 0.08 m/s
    ta, td = v3.drift_terms(ab, dr, v3.W)
    assert float(ta) == 0.0 and float(td) == 0.0
    ta, td = v3.drift_terms(ab, dr, v3.W35)
    assert abs(float(ta) - 30.0 * 0.26 ** 2) < 1e-6      # ≈ 2.03／步
    assert abs(float(td) - 40.0 * 0.08 ** 2) < 1e-6      # ≈ 0.256／步


def test_w35_values_pinned():
    assert v3.W["W_ABADBIAS"] == 0.0 and v3.W["W_DRIFT"] == 0.0
    assert v3.W35["W_ABADBIAS"] == 30.0 and v3.W35["W_DRIFT"] == 40.0
    assert v3.W35["W_VYREL"] == 6.0 and v3.W35["P_VY"] == 0.60
    assert v3.W["W_HEADLIN"] == 0.0 and v3.W35["W_HEADLIN"] == 1.0 and v3.W35["HEAD_LIN_E"] == 0.5
    assert v3.W["CYC_TURN_SYM"] is False and v3.W35["CYC_TURN_SYM"] is True
    changed = ("W_ABADBIAS", "W_DRIFT", "W_HEADLIN", "W_VYREL", "P_VY", "CYC_TURN_SYM")
    assert {k: v for k, v in v3.W35.items() if k not in changed} == {k: v for k, v in v3.W.items() if k not in changed}
    assert list(v3.ABAD12.tolist()) == [0, 3, 6, 9]


def _run(env, cmd, steps):
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    a = jnp.zeros(v3.ACT_DIM); out = []
    for _ in range(steps):
        s = js(s, a)
        out.append({k: float(s.metrics[k]) for k in ("t_abadbias", "t_drift", "t_headlin", "abad_bias", "vx_drift", "head_deg", "reward")})
    return out


def test_env_terms_zero_by_default_and_exactly_subtracted_in_w35():
    """零動作原地轉 30 步：預設 W 兩項精確為 0、診斷量照算；W35 下 reward 剛好少了兩項（物理同 seed 同動作完全一樣）。"""
    M = _run(v3.DualModeEnv(ref=dict(cyc_amp_rand=False)), (0.0, 0.0, 1.3), 30)
    assert all(m["t_abadbias"] == 0.0 and m["t_drift"] == 0.0 and m["t_headlin"] == 0.0 for m in M)
    assert M[-1]["abad_bias"] > 0.0 and np.isfinite(M[-1]["vx_drift"])
    # 逐項相減要物理一樣：W35 的 CYC_TURN_SYM 會換右轉週期（左轉也是平均後的），所以這裡關掉它
    M35 = _run(v3.DualModeEnv(ref=dict(cyc_amp_rand=False), weights=dict(v3.W35, CYC_TURN_SYM=False)), (0.0, 0.0, 1.3), 30)
    assert M35[-1]["t_abadbias"] > 0.0
    for m, m35 in zip(M, M35):
        assert abs(m35["reward"] - (m["reward"] - m35["t_abadbias"] - m35["t_drift"] + m35["t_headlin"])) < 1e-4     # wz 指令無 vy → t_vyrel=0，差只有這三項
    assert 0.0 <= M35[-1]["t_headlin"] <= 1.0


def test_headlin_linear_to_29deg():
    """線性航向項：0 → 1、0.25 rad → 0.5、≥ 0.5 rad → 0（W35 權重 1.0）。用 env 的公式直接算。"""
    f = lambda h: float(v3.W35["W_HEADLIN"] * (1.0 - jnp.clip(jnp.abs(h) / v3.W35["HEAD_LIN_E"], 0.0, 1.0)))   # noqa: E731
    assert f(0.0) == 1.0 and abs(f(0.25) - 0.5) < 1e-6 and f(-0.25) == f(0.25) and f(0.6) == 0.0


def test_turn_cycle_sym_is_exact_mirror_and_default_untouched():
    """cyc_turn_sym：右轉 delta(φ) 是左轉 delta(φ) 的鏡像（FL↔FR、BL↔BR、ABAD 反號）；預設 ref 仍用原始兩段錄檔（左右不鏡像）。"""
    ref_sym = dict(v3.REF, cyc_turn_sym=True, cyc_amp_rand=False)
    A_l, A_r = v3.step_pattern(jnp.array([0.0, 0.0, 1.3]), ref_sym)["A"], v3.step_pattern(jnp.array([0.0, 0.0, -1.3]), ref_sym)["A"]
    for phi in (0.0, 1.0, 2.5, 4.0, 5.5):
        dl = np.asarray(v3.cycle_offsets(jnp.array(phi), jnp.array([0.0, 0.0, 1.3]), A_l, ref_sym, 0.7)["delta"]).reshape(4, 3)
        dr = np.asarray(v3.cycle_offsets(jnp.array(phi), jnp.array([0.0, 0.0, -1.3]), A_r, ref_sym, 0.7)["delta"]).reshape(4, 3)
        mirror = dl[[1, 0, 3, 2]] * np.array([-1.0, 1.0, 1.0])
        np.testing.assert_allclose(dr, mirror, atol=1e-6)
    ref0 = dict(v3.REF, cyc_amp_rand=False)
    dl0 = np.asarray(v3.cycle_offsets(jnp.array(1.0), jnp.array([0.0, 0.0, 1.3]), A_l, ref0, 0.7)["delta"]).reshape(4, 3)
    dr0 = np.asarray(v3.cycle_offsets(jnp.array(1.0), jnp.array([0.0, 0.0, -1.3]), A_r, ref0, 0.7)["delta"]).reshape(4, 3)
    assert not np.allclose(dr0, dl0[[1, 0, 3, 2]] * np.array([-1.0, 1.0, 1.0]), atol=1e-3)     # 原始錄檔左右本來就不是鏡像
    env0, env35 = v3.DualModeEnv(), v3.DualModeEnv(weights=v3.W35)
    assert env0.ref["cyc_turn_sym"] is False and env35.ref["cyc_turn_sym"] is True
