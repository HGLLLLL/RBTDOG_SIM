"""v3.6（spec docs/superpowers/specs/2026-09-22-cpg-rl-v3.6-step-apex-linear-penalties-design.md）：
抬腳頂點獎勵、死區線性懲罰、橫向質心 DR、驗收關推力。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3  # noqa: E402


def test_w36_values_pinned():
    assert v3.W["W_STEP"] == 0.0 and v3.W["STEP_APEX"] == 0.021 and "PEN_SHAPE" not in v3.W
    assert v3.W36["W_STEP"] == 1.5 and v3.W36["STEP_APEX"] == 0.021
    assert v3.W36["PEN_SHAPE"] == "hinge" and v3.W36["W_DRIFT_L"] == 8.0 and v3.W36["DRIFT_DZ"] == 0.010
    assert v3.W36["W_ABADBIAS_L"] == 6.0 and v3.W36["ABAD_DZ"] == 0.0436
    changed = ("W_STEP", "PEN_SHAPE", "W_DRIFT_L", "DRIFT_DZ", "W_ABADBIAS_L", "ABAD_DZ")
    assert {k: v for k, v in v3.W36.items() if k not in changed} == {k: v for k, v in v3.W35.items() if k not in changed}
    assert "t_step" in v3.T_KEYS and "t_step" in v3.METRIC_KEYS


def test_apex_update_hands_over_on_wrap():
    """合成 clr 序列跨兩次回繞：apex_last = 上一週期最大；回繞當步 apex_run 重置成當步 clr。"""
    run, last = jnp.zeros(4), jnp.zeros(4)
    phi = [0.0, 2.0, 4.0, 6.0, 0.5, 2.5, 4.5, 6.1, 0.2]           # 回繞在 index 4、8
    clr = [0.0, 0.010, 0.021, 0.005, 0.003, 0.015, 0.008, 0.002, 0.030]
    for i in range(1, len(phi)):
        run, last = v3.step_apex_update(jnp.float32(phi[i]), jnp.float32(phi[i - 1]), jnp.full(4, clr[i]), run, last)
        if i == 4:
            assert float(last[0]) == pytest.approx(0.021) and float(run[0]) == pytest.approx(0.003)
        if i == 7:
            assert float(last[0]) == pytest.approx(0.021) and float(run[0]) == pytest.approx(0.015)
    assert float(last[0]) == pytest.approx(0.015) and float(run[0]) == pytest.approx(0.030)


def test_apex_reward_primary_legs_only_and_saturates():
    s_lat = jnp.array([0.7, 1.0, 0.7, 1.0])
    r = v3.step_apex_reward(jnp.array([0.0, 0.021, 0.0, 0.021]), s_lat, 0.20, v3.W36)
    assert float(r) == pytest.approx(1.0)
    r = v3.step_apex_reward(jnp.array([0.0, 0.0105, 0.0, 0.021]), s_lat, 0.20, v3.W36)
    assert float(r) == pytest.approx(0.75)
    r = v3.step_apex_reward(jnp.array([0.050, 0.0, 0.050, 0.0]), s_lat, 0.20, v3.W36)      # 次要腿抬 50 mm 不算分
    assert float(r) == pytest.approx(0.0)
    r = v3.step_apex_reward(jnp.array([0.0, 0.050, 0.0, 0.050]), s_lat, 0.20, v3.W36)      # 超過目標飽和
    assert float(r) == pytest.approx(1.0)
    # 目標隨指令縮：vy 0.12 → 16.8 mm；vy 0.04 → 10.5 mm
    assert float(v3.step_apex_reward(jnp.array([0.0, 0.0168, 0.0, 0.0168]), s_lat, 0.12, v3.W36)) == pytest.approx(1.0)
    assert float(v3.step_apex_reward(jnp.array([0.0, 0.0105, 0.0, 0.0105]), s_lat, 0.04, v3.W36)) == pytest.approx(1.0)
    # 斜走：s 整體縮過（0.4×），正規化後主動腿仍是 1.0
    assert float(v3.step_apex_reward(jnp.array([0.0, 0.021, 0.0, 0.021]), s_lat * 0.4, 0.15, v3.W36)) == pytest.approx(1.0)


def _env_rollout(env, cmd, steps, seed=0):
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(seed))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    out = []
    for _ in range(steps):
        s = js(s, jnp.zeros(v3.ACT_DIM)); out.append(s)
    return out


def test_t_step_zero_by_default_and_positive_in_lateral_with_w36():
    """預設 W：t_step 恆 0。W36＋factory 零動作左平移 0.20：兩個週期後 t_step > 0（零動作主動側兩腿抬 21 mm）；原地轉 t_step = 0。"""
    env0 = v3.DualModeEnv(gains="factory", ref=dict(cyc_amp_rand=False))
    S = _env_rollout(env0, (0.0, 0.20, 0.0), 80)
    assert all(float(s.metrics["t_step"]) == 0.0 for s in S)
    assert set(S[-1].info) >= {"apex_run", "apex_last"} and S[-1].info["apex_last"].shape == (4,)
    env = v3.DualModeEnv(gains="factory", weights=v3.W36, ref=dict(cyc_amp_rand=False))
    S = _env_rollout(env, (0.0, 0.20, 0.0), 150)                           # 3 s；2.1 Hz → 6 個週期
    ts = [float(s.metrics["t_step"]) for s in S]
    assert max(ts[-50:]) > 0.5, ts[-50:]
    S = _env_rollout(env, (0.0, 0.0, 1.3), 60)
    assert all(float(s.metrics["t_step"]) == 0.0 for s in S)
