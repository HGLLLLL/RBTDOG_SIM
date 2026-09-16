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
    changed = ("W_ABADBIAS", "W_DRIFT", "W_VYREL", "P_VY")
    assert {k: v for k, v in v3.W35.items() if k not in changed} == {k: v for k, v in v3.W.items() if k not in changed}
    assert list(v3.ABAD12.tolist()) == [0, 3, 6, 9]
