"""v3 雙模式 env：模式判定、踏步圖案（左右鏡像、對角/同側）、輪子控制律、參考常數對資料集、短 rollout 與 G0 型斷言。"""
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


def test_mode_of():
    u, t, l = v3.mode_of(jnp.array([0.5, 0.0, 0.0])); assert (float(u), bool(t), bool(l)) == (0.0, False, False)
    u, t, l = v3.mode_of(jnp.array([0.5, 0.0, 0.3])); assert float(u) == 0.0            # 邊走邊轉＝輪行
    u, t, l = v3.mode_of(jnp.array([0.0, 0.0, 1.3])); assert (float(u), bool(t)) == (1.0, True)
    u, t, l = v3.mode_of(jnp.array([0.0, 0.06, 0.0])); assert (float(u), bool(l)) == (1.0, True)
    u, t, l = v3.mode_of(jnp.array([0.0, 0.0, 0.0])); assert float(u) == 0.0


def test_step_pattern_turn_is_diagonal_in_phase_with_opposite_wheels():
    act, vec, ph, wheel, hz, duty, lift = v3.step_pattern(jnp.array([0.0, 0.0, 1.3]))
    act, vec, ph, wheel = map(np.asarray, (act, vec, ph, wheel))
    # MJCF 序 FR, FL, RR, RL：左轉踏步 FL+RR，輪 FR +Ω / RL −Ω
    assert act.tolist() == [0, 1, 1, 0]
    assert ph[1] == ph[2]
    assert wheel[0] > 3.0 and wheel[3] < -3.0 and abs(wheel[1]) < 1e-9 and abs(wheel[2]) < 1e-9
    assert vec[1][1] > 0 and vec[2][1] < 0                          # FL 往左、RR 往右 → 逆時針
    # 右轉：鏡像
    act2, vec2, ph2, wheel2, *_ = v3.step_pattern(jnp.array([0.0, 0.0, -1.3]))
    act2, vec2, wheel2 = map(np.asarray, (act2, vec2, wheel2))
    assert act2.tolist() == [1, 0, 0, 1]
    assert wheel2[1] > 3.0 and wheel2[2] < -3.0                    # FL +Ω / RR −Ω（資料 fl +3.8 / br −4.1）
    assert vec2[0][1] < 0 and vec2[3][1] > 0


def test_step_pattern_lateral_is_same_side_alternating():
    act, vec, ph, wheel, hz, duty, lift = v3.step_pattern(jnp.array([0.0, 0.06, 0.0]))
    act, vec, ph, wheel = map(np.asarray, (act, vec, ph, wheel))
    assert act.tolist() == [0, 1, 0, 1]                              # 左移：FL+RL
    assert abs((ph[1] - ph[3]) % 1.0 - 0.5) < 1e-9                   # 交替
    assert wheel[1] > 2.0 and wheel[3] < -1.0
    assert vec[1][1] > 0 and vec[3][1] > 0                           # 都往左
    assert abs(float(duty) - v3.REF["duty_lat"]) < 1e-6


def test_wheel_ctrl_deadband_and_feedforward():
    v = np.asarray(v3.wheel_ctrl(jnp.array([0.1, 2.0, -2.0, 0.0])))
    assert v[0] == 0.0 and v[3] == 0.0
    assert abs(v[1] - (2.0 + v3.TAU_FF / v3.KV_WHEEL)) < 1e-9 and abs(v[2] + (2.0 + v3.TAU_FF / v3.KV_WHEEL)) < 1e-9


def test_differential_drive():
    w = np.asarray(v3.wheel_cmd_wheelmode(jnp.array([0.5, 0.0, 0.0])))
    assert np.allclose(w, 0.5 / v3.REF["r_wheel"])
    w = np.asarray(v3.wheel_cmd_wheelmode(jnp.array([0.0, 0.0, 1.0])))
    assert w[0] > 0 and w[1] < 0 and abs(w[0] + w[1]) < 1e-9          # 左轉：右輪正、左輪負


@pytest.mark.skipif(not DS.exists(), reason="資料集未產生")
def test_ref_constants_match_dataset():
    d = json.loads(DS.read_text(encoding="utf-8"))["summary"]
    assert abs(v3.REF["L_eff"] - d["L_eff_m"]) < 0.02
    q = np.array(d["wheel_stance_q12"])            # shm 序 fl,fr,bl,br
    assert abs(v3.REF["stance_q12"][1] - q[1]) < 0.02 and abs(v3.REF["stance_q12"][2] - q[2]) < 0.02
    assert abs(v3.REF["factory_hz_turn"] - d["turn_left"]["freq_hz"]) < 0.15
    assert abs(v3.REF["factory_lift"] - d["turn_left"]["apex_m"]) < 0.004
    wp = d["turn_left"]["wheel_pattern"]
    assert wp["fr"] > 0.8 and wp["bl"] < -0.8                        # 對應 REF turn_wheel FR +1 / RL −1


def test_env_reset_step_shapes_and_g0():
    env = v3.DualModeEnv()
    assert env.obs_dim == 76 and env.action_size == 12
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    for cmd, check in ((jnp.array([0.5, 0.0, 0.0]), "wheel"), (jnp.array([0.0, 0.0, 1.3]), "turn")):
        s = jit_reset(jax.random.PRNGKey(0))
        s = s.replace(info={**s.info, "cmd": cmd, "cmd2": cmd, "t_switch": 10 ** 6, "u_mode": v3.mode_of(cmd)[0]})
        assert s.obs.shape == (76,)
        a = jnp.zeros(12)
        vx, wz, tau, done = [], [], [], []
        for i in range(80):
            s = jit_step(s, a)
            vx.append(float(s.metrics["vx"])); wz.append(float(s.metrics["wz"])); tau.append(float(s.metrics["tau_pk"])); done.append(float(s.done))
        assert max(done) == 0.0
        assert max(tau) < v3.TAU_KILL
        assert s.obs.shape == (76,) and bool(jnp.all(jnp.isfinite(s.obs)))
        if check == "wheel":
            assert abs(np.mean(vx[40:]) - 0.5) < 0.08
            assert float(s.metrics["clr_stance"]) < 10.0
        else:
            assert np.degrees(np.mean(wz[40:])) > 10.0


def test_obs_layout_pinned():
    """狗上 obs 組裝要照這個切；改了就要一起改 realbot。"""
    assert v3.ACT_DIM == 12
    assert 3 + 3 + 12 + 12 + 4 + 3 + 2 + 1 + 12 + 24 == 76
