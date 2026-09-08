"""IMU 判讀：用合成資料（已知順序、單位、軸向）驗它判得對。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord  # noqa: E402
import imu_check as ic  # noqa: E402
import leg_kin  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402

D2R = np.pi / 180


def euler_to_quat_wxyz(roll, pitch, yaw=0.0):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def synth(hz=500.0, secs=10.0, roll_amp=8.0, pitch_amp=4.0, store="wxyz",
          gyro_unit="deg/s", flip_y=True, dup_imu_every=0, yaw_rate=0.0, quiet=False):
    """合成一段跳舞：roll/pitch 正弦、四輪永遠共面貼地、gyro 用解析式。"""
    n = int(hz * secs)
    t = np.arange(n) / hz
    if quiet:
        roll = np.full(n, 1.0 * D2R)
        pitch = np.full(n, -0.5 * D2R)
        droll = dpitch = np.zeros(n)
    else:
        roll = roll_amp * D2R * np.sin(2 * np.pi * 0.5 * t)
        pitch = pitch_amp * D2R * np.sin(2 * np.pi * 0.3 * t)
        droll = roll_amp * D2R * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
        dpitch = pitch_amp * D2R * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t)
    yaw = yaw_rate * t
    Q = np.stack([euler_to_quat_wxyz(roll[i], pitch[i], yaw[i]) for i in range(n)])
    # 機身系角速度（ZYX 尤拉率 → 機身系）
    om = np.stack([droll - yaw_rate * np.sin(pitch),
                   dpitch * np.cos(roll) + yaw_rate * np.cos(pitch) * np.sin(roll),
                   -dpitch * np.sin(roll) + yaw_rate * np.cos(pitch) * np.cos(roll)], axis=1)
    g = ic.gravity_from_quat(Q)                       # (N,3) 機身系重力方向
    acc = -9.81 * g
    gyro = om * (180 / np.pi if gyro_unit == "deg/s" else 1.0)
    if flip_y:
        gyro[:, 1] *= -1
    # 四輪共面：地面法向 = −g；輪心落在該平面上
    ks = leg_kin.knee_sign_of(mm.HOME)
    origin = np.stack([mm.SIDE_X * mm.HIP_X, mm.SIDE_Y * mm.HIP_Y, np.zeros(4)], axis=1)
    hf = leg_kin.home_foot(mm.HOME) + origin       # (4,3) 機身系
    P = np.zeros((n, 12))
    for i in range(n):
        nrm = -g[i]
        c = nrm @ np.array([0.0, 0.0, -0.45])
        for k in range(4):
            x, y = hf[k, 0], hf[k, 1]
            z = (c - nrm[0] * x - nrm[1] * y) / nrm[2]
            P[i, 3 * k:3 * k + 3] = leg_kin.ik(k, np.array([x, y, z]) - origin[k], ks[k])
    joints = {}
    for nm in shm_io.JOINTS:
        leg, kind = nm[:2], nm[2:]
        if kind == coord.KIND_WHEEL:
            q = np.zeros(n)
            v = np.zeros(n)
        else:
            idx = real_obs.LEG_NAMES.index(nm)
            q = np.array([coord.to_motor(nm, x) for x in P[:, idx]])
            v = np.gradient(P[:, idx], t) * coord.SIGN[kind][leg]
        joints[nm] = {"q": list(np.round(q, 5)), "v": list(np.round(v, 4)),
                      "tau": [0.0] * n, "des": [0.0] * n, "ff": 0.0, "kp": 60.0, "kd": 1.0}
    quat = Q if store == "wxyz" else Q[:, [1, 2, 3, 0]]
    imu = np.concatenate([acc, gyro, quat], axis=1)
    if dup_imu_every:
        for i in range(1, n):
            if i % dup_imu_every:
                imu[i] = imu[i - 1]
    d = {"schema": "m6_record/1", "label": "syn", "note": "", "n": n, "secs": t[-1],
         "hz_actual": hz, "t": list(np.round(t, 4)), "joints": joints,
         "imu": [list(map(float, r)) for r in imu]}
    return m6_rec.from_dict(d)


def test_rp_from_gravity_conventions():
    # +roll = 右側低：機身繞 +x 轉 +10° 後，重力在機身系 y 分量為負
    g = ic.gravity_from_quat(euler_to_quat_wxyz(10 * D2R, 0)[None])[0]
    r, p = ic.rp_from_gravity(g)
    assert r == pytest.approx(10.0, abs=1e-6) and p == pytest.approx(0.0, abs=1e-6)
    assert g[1] < 0
    g = ic.gravity_from_quat(euler_to_quat_wxyz(0, 7 * D2R)[None])[0]
    r, p = ic.rp_from_gravity(g)
    assert p == pytest.approx(7.0, abs=1e-6) and g[0] > 0


def test_omega_body_recovers_analytic_rate():
    rec = synth(gyro_unit="rad/s", flip_y=False)
    Q = real_obs.quat_wxyz(rec.quat_raw, "wxyz")
    om = ic.omega_body(Q, rec.t)
    np.testing.assert_allclose(om, rec.gyro[:-1], atol=2e-3)


def test_decides_order_unit_axis_and_rates():
    flat = synth(store="wxyz", quiet=True, secs=3)
    dance = synth(store="wxyz", gyro_unit="deg/s", flip_y=True)
    out = ic.analyse(flat, dance, None, seen_first=None, turn_first=None)
    assert out["quat_order"] == "wxyz"
    assert out["gyro_unit"] == "deg/s"
    assert np.sign(out["gyro_scale"]).tolist() == [1.0, -1.0, 1.0]
    np.testing.assert_allclose(np.abs(out["gyro_scale"]), np.pi / 180, rtol=0.05)
    assert abs(out["rp_quat_vs_fk"]["bias_deg"][0]) < 0.5
    assert out["rp_quat_vs_fk"]["corr"][0] > 0.99
    assert out["rates"]["imu_gyro"]["hz_min"] >= 475
    assert out["first_roll"]["side"] == "右低"          # roll 先往正 → 右側低


def test_detects_xyzw_and_rad_per_s():
    flat = synth(store="xyzw", quiet=True, secs=3, gyro_unit="rad/s", flip_y=False)
    dance = synth(store="xyzw", gyro_unit="rad/s", flip_y=False)
    out = ic.analyse(flat, dance, None, None, None)
    assert out["quat_order"] == "xyzw"
    assert out["gyro_unit"] == "rad/s"
    np.testing.assert_allclose(out["gyro_scale"], [1, 1, 1], rtol=0.05)


def test_update_rate_sees_duplicated_imu():
    dance = synth(store="xyzw", gyro_unit="rad/s", flip_y=False, dup_imu_every=2)
    flat = synth(store="xyzw", quiet=True, secs=3, gyro_unit="rad/s", flip_y=False)
    out = ic.analyse(flat, dance, None, None, None)
    assert 235 <= out["rates"]["imu_gyro"]["hz_min"] <= 265


def test_turn_sign():
    turn = synth(store="xyzw", gyro_unit="rad/s", flip_y=False, yaw_rate=0.4,
                 quiet=True, secs=4)
    flat = synth(store="xyzw", quiet=True, secs=2, gyro_unit="rad/s", flip_y=False)
    dance = synth(store="xyzw", gyro_unit="rad/s", flip_y=False)
    out = ic.analyse(flat, dance, turn, None, turn_first="left")
    assert out["turn"]["gyro_z_sign"] == 1 and out["turn"]["consistent"] is True
