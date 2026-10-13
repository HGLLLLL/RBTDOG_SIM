"""policy_np：新鍵 act_gain（缺＝1.0，舊 npz 逐位元不變）；tanh_loc × act_gain。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))
import policy_np as pn  # noqa: E402


def _fake(act_gain=None, obs=5, act=16):
    rng = np.random.default_rng(0)
    z = dict(mean=np.zeros(obs, np.float32), std=np.ones(obs, np.float32), obs_dim=np.int64(obs), act_dim=np.int64(act),
             preset=np.str_("x"), layout=np.str_("rl16"), src_sha256=np.str_("0"),
             W0=rng.normal(size=(obs, 8)).astype(np.float32), b0=np.zeros(8, np.float32), W1=rng.normal(size=(8, 8)).astype(np.float32), b1=np.zeros(8, np.float32),
             W2=rng.normal(size=(8, 8)).astype(np.float32), b2=np.zeros(8, np.float32), W3=rng.normal(size=(8, 2 * act)).astype(np.float32), b3=np.zeros(2 * act, np.float32))
    if act_gain is not None:
        z["act_gain"] = np.float64(act_gain)
    return z


def test_default_gain_is_one_and_old_behaviour_unchanged():
    P0, P1 = pn.Policy(_fake()), pn.Policy(_fake(1.0))
    x = np.linspace(-1, 1, 5).astype(np.float32)
    assert P0.act_gain == 1.0 and np.array_equal(P0.infer(x), P1.infer(x)) and np.abs(P0.infer(x)).max() <= 1.0


def test_gain_scales_tanh_output():
    P4 = pn.Policy(_fake(4.0)); P1 = pn.Policy(_fake())
    x = np.linspace(-1, 1, 5).astype(np.float32)
    assert np.allclose(P4.infer(x), 4.0 * P1.infer(x)) and P4.act_gain == 4.0


def test_committed_rl_only_npz_unchanged():
    npz = ROOT / "weights" / "rl_only_2026-10-01_10k_np.npz"
    if not npz.exists():
        pytest.skip("rl_only npz 不在")
    P = pn.load(str(npz)); assert P.act_gain == 1.0 and P.output == "linear"
