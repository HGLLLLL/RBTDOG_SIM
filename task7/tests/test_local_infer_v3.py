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


def test_rollout_reports_steps_per_second():
    env = v3.DualModeEnv(gains="factory", ref=dict(cyc_amp_rand=False), push=False); jr, js = jax.jit(env.reset), jax.jit(env.step)
    r = L.rollout(env, jr, js, (0.0, 0.20, 0.0), 250, None, seed=0)      # 零動作左平移 5 s：主動側 FL／RL 應有 ≥ 1.5 步/秒
    assert len(r["steps_s"]) == 4 and all(isinstance(x, float) for x in r["steps_s"])
    assert r["steps_s"][1] >= 1.5 and r["steps_s"][3] >= 1.5 and r["steps_s"][0] < 0.5
    r0 = L.rollout(env, jr, js, (0.5, 0.0, 0.0), 60, None, seed=0)
    assert r0["steps_s"] == [0.0, 0.0, 0.0, 0.0]


def test_cli_presets_and_push_flag():
    p = L.build_parser()
    a = p.parse_args(["--preset", "v36"])
    assert a.preset == "v36" and a.push is False
    assert L.preset_weights("v36") is v3.W36 and L.preset_weights("v35") is v3.W35 and L.preset_weights("v33") is None
