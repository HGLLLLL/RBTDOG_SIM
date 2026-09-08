"""M9 log → M6 格式：座標轉回馬達系要可逆、增益排程要照 M9 規則、roll/pitch ↔ quat 互逆。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import imu_check as ic  # noqa: E402
import m9_rec  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402

ARGS = {"kp": 250.0, "kp_abad": 60.0, "kd": 2.0, "wheel_kd": 0.5, "kp_shift": 1.5}


def _m9(phases, hz=200.0):
    """phases: [(name, secs)] → 假 M9 log。關節角走一個慢正弦，roll/pitch 走另一個。"""
    S, t = [], 0.0
    legs = [n for n in shm_io.JOINTS if not n.endswith("4_foot")]
    wheels = [n for n in shm_io.JOINTS if n.endswith("4_foot")]
    for nm, secs in phases:
        for _ in range(int(secs * hz)):
            kp = 250.0 * min(1.0, t / 2.0) if nm == "RAMP_UP" else 250.0
            j = {n: (round(0.8 + 0.1 * np.sin(t), 4), round(0.8 + 0.1 * np.sin(t + 0.05), 4),
                     round(3.0 * np.cos(t), 2), round(0.1 * np.cos(t), 3)) for n in legs}
            w = {n: (round(0.3 * t, 4), 0.3, 0.1) for n in wheels}
            S.append({"t": round(t, 3), "phase": nm, "kp": round(kp, 1),
                      "roll": round(5 * np.sin(2 * t), 2), "pitch": round(-3 * np.cos(2 * t), 2),
                      "j": j, "w": w})
            t += 1.0 / hz
    return {"schema": "m9/1", "args": dict(ARGS), "samples": S}


def test_ctrl_coordinates_roundtrip_and_gain_schedule():
    d = _m9([("RAMP_UP", 1.0), ("READY", 0.5), ("KP_DOWN", 1.5), ("GAIT", 1.0),
             ("KP_UP", 1.5), ("RAMP_DOWN", 0.5)])
    rec = m9_rec.from_m9(d, "x.json")
    S = d["samples"]
    P = real_obs.ctrl_pos(rec)
    V = real_obs.ctrl_vel(rec)
    for i in (0, 100, len(S) - 1):
        for k, nm in enumerate(real_obs.LEG_NAMES):
            assert P[i, k] == pytest.approx(S[i]["j"][nm][0], abs=1e-9)
            assert V[i, k] == pytest.approx(S[i]["j"][nm][3], abs=1e-9)
    ph = rec.phase
    KP = real_obs.kp12(rec)
    abad, hip = KP[:, 0], KP[:, 1]
    assert np.allclose(abad[ph == "GAIT"], 60.0) and np.allclose(hip[ph == "GAIT"], 250.0)
    kd_gait = real_obs.kd12(rec)[ph == "GAIT"]
    assert np.allclose(kd_gait, 2.0)
    assert np.allclose(real_obs.kd12(rec)[ph == "READY"], 5.0)
    i0 = np.flatnonzero(ph == "KP_DOWN")
    assert abad[i0[0]] == pytest.approx(250.0, abs=2.0)
    assert abad[i0[-1]] == pytest.approx(60.0, abs=2.0)
    assert abad[i0[len(i0) // 2]] == pytest.approx(155.0, abs=5.0)
    i1 = np.flatnonzero(ph == "KP_UP")
    assert abad[i1[0]] == pytest.approx(60.0, abs=2.0) and abad[i1[-1]] == pytest.approx(250.0, abs=2.0)
    assert np.allclose(abad[ph == "RAMP_UP"], hip[ph == "RAMP_UP"])
    assert np.allclose(real_obs.wheel_kd(rec), 0.5) and np.allclose(real_obs.wheel_kp(rec), 0.0)


def test_rp_quat_roundtrip_and_gyro_matches_rp_rate():
    d = _m9([("GAIT", 3.0)])
    rec = m9_rec.from_m9(d, "x.json")
    Q = real_obs.quat_wxyz(rec.quat_raw, "xyzw")
    r, p = ic.rp_from_gravity(ic.gravity_from_quat(Q))
    S = d["samples"]
    np.testing.assert_allclose(r, [s["roll"] for s in S], atol=1e-6)
    np.testing.assert_allclose(p, [s["pitch"] for s in S], atol=1e-6)
    # roll = 5 sin 2t → ω_x ≈ 10 cos 2t (deg/s) = 0.1745 cos 2t rad/s
    t = rec.t
    mid = slice(50, -50)
    np.testing.assert_allclose(rec.gyro[mid, 0], (10 * np.pi / 180) * np.cos(2 * t[mid]),
                               atol=0.02)
    assert np.allclose(np.linalg.norm(rec.acc, axis=1), 9.81)


def test_load_any_dispatches(tmp_path):
    import json
    p = tmp_path / "M9_x.json"
    p.write_text(json.dumps(_m9([("GAIT", 0.1)])), encoding="utf-8")
    rec = m9_rec.load_any(p)
    assert rec.n == 20 and rec.label.startswith("m9:")
