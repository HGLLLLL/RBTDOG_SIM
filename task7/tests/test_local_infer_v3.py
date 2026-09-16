"""local_infer_v3：rollout 回傳鍵（v3.5 加 abad_bias／vx_drift／head_end）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3          # noqa: E402
import local_infer_v3 as L      # noqa: E402


def test_rollout_reports_drift_columns():
    env = v3.DualModeEnv(ref=dict(cyc_amp_rand=False)); jr, js = jax.jit(env.reset), jax.jit(env.step)
    r = L.rollout(env, jr, js, (0.0, 0.0, 1.3), 40, None, seed=0)      # 零動作 0.8 s，只驗鍵與型別
    assert {"abad_bias", "vx_drift", "head_end"} <= set(r)
    assert r["abad_bias"] >= 0.0 and isinstance(r["vx_drift"], float) and isinstance(r["head_end"], float)
