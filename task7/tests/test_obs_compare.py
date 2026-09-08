"""錄檔驅動回放 + 逐欄對照：拿模擬自己產生的「假實機」驗雜訊與延遲量得回來。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import obs_compare as oc  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402

HZ = 200.0


def _rec_with_des(des12, kp_abad=60.0, kp=250.0, kd=2.0):
    """只填 des/kp/kd 的錄檔（q 先抄 des、v/imu 用零，回放不需要它們）。"""
    n = des12.shape[0]
    t = np.arange(n) / HZ
    joints = {}
    for nm in shm_io.JOINTS:
        kind = nm[2:]
        if kind == coord.KIND_WHEEL:
            joints[nm] = {"q": [0.0] * n, "v": [0.0] * n, "tau": [0.0] * n,
                          "des": [0.0] * n, "ff": 0.0, "kp": 0.0, "kd": 0.5}
        else:
            idx = real_obs.LEG_NAMES.index(nm)
            motor = [coord.to_motor(nm, float(x)) for x in des12[:, idx]]
            joints[nm] = {"q": list(motor), "v": [0.0] * n, "tau": [0.0] * n,
                          "des": list(motor), "ff": 0.0,
                          "kp": kp_abad if kind == coord.KIND_HIP_ROLL else kp, "kd": kd}
    imu = [[0.0, 0.0, 9.81, 0, 0, 0, 0, 0, 0, 1.0] for _ in range(n)]
    return m6_rec.from_dict({"schema": "m6_record/1", "label": "syn", "note": "", "n": n,
                             "secs": t[-1], "hz_actual": HZ, "t": list(t),
                             "joints": joints, "imu": imu})


def _des_sequence(secs_hold=0.8, secs_move=2.0):
    """crouch 起步 → 站到 HOME → 髖膝正弦擺動（增益是步態組，gait_segment 抓得到）。"""
    crouch = mm.CROUCH.reshape(12)
    home = mm.HOME12
    n1, n2 = int(secs_hold * HZ), int(secs_move * HZ)
    ramp = np.linspace(0, 1, n1)[:, None]
    A = crouch[None] * (1 - ramp) + home[None] * ramp
    t = np.arange(n2) / HZ
    B = np.tile(home, (n2, 1))
    for k in range(4):
        B[:, 3 * k + 1] += 0.15 * np.sin(2 * np.pi * 1.0 * t)
        B[:, 3 * k + 2] -= 0.20 * np.sin(2 * np.pi * 1.0 * t)
    return np.vstack([A, B])


def test_gait_segment_is_where_abad_kp_is_soft():
    des = _des_sequence()
    rec = _rec_with_des(des)
    for nm in real_obs.LEG_NAMES:
        if nm.endswith(coord.KIND_HIP_ROLL):
            rec.j[nm]["kp"][:100] = 250.0
    s = oc.gait_segment(rec)
    assert s.start == 100 and s.stop == rec.n


def test_replay_then_compare_recovers_noise_and_lag():
    des = _des_sequence()
    rec = _rec_with_des(des)
    sim = oc.sim_replay(rec, "xyzw", (1.0, 1.0, 1.0))
    assert sim["obs"].shape == (rec.n, 68)
    assert sim["height"][-1] > 0.35            # 站起來了、沒倒

    # 用回放結果造「假實機」：加已知雜訊、往後平移 3 筆（15 ms）
    rng = np.random.default_rng(0)
    lag = 3
    q_n, v_n, w_n = 0.002, 0.05, 0.01

    def shift(x):
        return np.concatenate([np.repeat(x[:1], lag, axis=0), x[:-lag]], axis=0)
    Pm, Vm, Qm, Gm = shift(sim["q"]), shift(sim["v"]), shift(sim["quat"]), shift(sim["gyro"])
    for nm in real_obs.LEG_NAMES:
        idx = real_obs.LEG_NAMES.index(nm)
        s = coord.SIGN[nm[2:]][nm[:2]]
        rec.j[nm]["q"] = (np.array([coord.to_motor(nm, x) for x in Pm[:, idx]])
                          + rng.normal(0, q_n, rec.n))
        rec.j[nm]["v"] = s * Vm[:, idx] + rng.normal(0, v_n, rec.n)
    rec.imu[:, 3:6] = Gm + rng.normal(0, w_n, (rec.n, 3))
    rec.imu[:, 6:10] = Qm[:, [1, 2, 3, 0]]          # 存成 xyzw
    O_real = real_obs.obs_series(rec, "xyzw", (1, 1, 1))
    seg = oc.gait_segment(rec)
    rows = oc.channel_table(O_real, sim["obs"], seg, 1.0 / HZ)
    assert len(rows) == real_obs.SENSOR_DIM
    jp = [r for r in rows if r["group"] == "joint_pos"]
    jv = [r for r in rows if r["group"] == "joint_vel"]
    gy = [r for r in rows if r["group"] == "gyro"]
    assert np.median([r["noise_est"] for r in jp]) == pytest.approx(q_n, rel=0.35)
    assert np.median([r["noise_est"] for r in jv]) == pytest.approx(v_n, rel=0.35)
    assert np.median([r["noise_est"] for r in gy]) == pytest.approx(w_n, rel=0.35)
    assert np.median([r["lag_ms"] for r in jp]) == pytest.approx(15.0, abs=5.0)
    assert all(r["corr"] > 0.9 for r in jp)

    lat = oc.des_to_q_latency(real_obs.ctrl_des(rec), real_obs.ctrl_pos(rec), 1.0 / HZ, seg)
    assert len(lat) == 12 and np.nanmedian(lat) >= 15.0     # 平移 15 ms + 伺服本身的落後（ABAD 沒動 → nan）

    jvr = oc.joint_vel_report(rec, seg)
    assert set(jvr) >= {"noise_ratio_v_over_dq", "corr_v_dq", "psd_frac_above_25hz"}


def test_lag_ms_sign_convention():
    dt = 0.005
    t = np.arange(400) * dt
    a = np.sin(2 * np.pi * 1.0 * t)
    b = np.concatenate([np.zeros(4), a[:-4]])      # b 比 a 晚 20 ms
    assert oc.lag_ms(b, a, dt) == pytest.approx(20.0, abs=2.5)
    assert oc.lag_ms(a, b, dt) == pytest.approx(-20.0, abs=2.5)
