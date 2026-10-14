"""重力歸零（`--grav-offset`，2026-10-14）：rl_obs_rl.grav_fix_matrix 的慣例、Obs.build 只動重力三維、
模擬端 --grav-bias／--grav-offset 互相抵消、M9 旗標接到 rl16 的 Obs。背景：trip26 平移前漂 +0.13 m/s，
實機站立 IMU 讀 roll +0.7／pitch −1.5°，模擬注入同樣偏差重現前漂（W 文件 §8）。"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot")); sys.path.insert(0, str(ROOT / "inference"))
import rl_obs  # noqa: E402
import rl_obs_rl as ro  # noqa: E402

NPZ = ROOT / "weights" / "rl_wheel_v1b_np.npz"
DOWN = np.array([0.0, 0.0, -1.0])


def _rp_m7(qw, qx, qy, qz):
    """M7_standup.read_imu_rp 的公式（那支要開 shm，抄公式來對）。"""
    roll = math.degrees(math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx)))))
    return roll, pitch


def _quat_from_R(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w); y = (R[0, 2] - R[2, 0]) / (4 * w); z = (R[1, 0] - R[0, 1]) / (4 * w)
    return w, x, y, z


@pytest.mark.parametrize("roll,pitch", [(0.7, -1.5), (1.0, -1.5), (-2.0, 3.0), (0.0, 0.0)])
def test_fix_matrix_convention_matches_m7_and_sim(roll, pitch):
    """機身姿態 R = R_fix 時，M7 的 read_imu_rp 與 rl_only_infer.rp_from_R 都讀 (roll, pitch)；
    policy 看到的重力 R_fixᵀ·(0,0,−1) 乘回 R_fix 就是 (0,0,−1)。"""
    import rl_only_infer as ri
    R = ro.grav_fix_matrix(roll, pitch)
    assert np.allclose(ri.rp_from_R(R), (roll, pitch), atol=1e-9)
    assert np.allclose(_rp_m7(*_quat_from_R(R)), (roll, pitch), atol=1e-9)
    g_seen = R.T @ DOWN
    assert np.allclose(R @ g_seen, DOWN, atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_pitch_minus_means_gravity_x_negative():
    """實機 trip26 站立讀 pitch −1.5 → policy 重力 x 分量 −0.026（和 log 一致）；歸零後回 0。"""
    g = ro.grav_fix_matrix(0.0, -1.5).T @ DOWN
    assert g[0] == pytest.approx(-math.sin(math.radians(1.5)), abs=1e-9) and g[0] < 0
    assert np.allclose(ro.grav_fix_matrix(0.0, -1.5) @ g, DOWN, atol=1e-12)


def _frame(qw, qx, qy, qz):
    rng = np.random.default_rng(0)
    pos = {n: float(rng.uniform(-0.5, 0.5)) for n in rl_obs.LEG_NAMES}; vel = {n: float(rng.uniform(-1, 1)) for n in rl_obs.LEG_NAMES}
    for n in ro.WHEEL_NAMES:
        pos[n] = 0.0; vel[n] = float(rng.uniform(-3, 3))
    return rl_obs.Frame(pos, vel, np.array([qx, qy, qz, qw]), np.array([0.1, -0.2, 0.3]), 0)


@pytest.mark.skipif(not NPZ.exists(), reason="沒有 v1b npz")
def test_obs_build_offset_only_touches_gravity_slice():
    R = ro.grav_fix_matrix(0.7, -1.5)
    fr = _frame(*_quat_from_R(R))                                   # 機身姿態＝IMU 偏差：水平的狗、讀 +0.7／−1.5
    o0 = ro.Obs(str(NPZ)).build(fr, None, (0.0, 0.5, 0.0), np.zeros(16))
    o1 = ro.Obs(str(NPZ), grav_offset=(0.7, -1.5)).build(fr, None, (0.0, 0.5, 0.0), np.zeros(16))
    assert np.allclose(o0[3:6], R.T @ DOWN, atol=1e-6) and np.allclose(o1[3:6], DOWN, atol=1e-6)
    assert np.array_equal(o0[:3], o1[:3]) and np.array_equal(o0[6:], o1[6:])
    with pytest.raises(AssertionError):
        ro.Obs(str(NPZ), grav_offset=(0.0, -6.0))                   # 打錯一位數的保護


@pytest.mark.skipif(not NPZ.exists(), reason="沒有 v1b npz")
def test_sim_bias_and_offset_cancel_and_match_dog_stack():
    """模擬 --grav-bias 注入 ＋ --grav-offset 同值 ＝ 沒注入；狗端 Obs(grav_offset) 對同一姿態給同一重力。"""
    import json
    import mujoco
    import rl_only_infer as ri
    io = json.load(open(ROOT / "inference" / "rl_wheel_io.json"))
    s0 = ri.Sim(io); s1 = ri.Sim(io, grav_bias=(0.7, -1.5)); s2 = ri.Sim(io, grav_bias=(0.7, -1.5), grav_offset=(0.7, -1.5))
    for s in (s0, s1, s2):
        s.reset(); s.d.qpos[3:7] = _quat_from_R(ro.grav_fix_matrix(2.0, -3.0)); mujoco.mj_forward(s.m, s.d)   # 隨便一個真實傾斜
    o0, _ = s0.obs(np.zeros(3)); o1, _ = s1.obs(np.zeros(3)); o2, _ = s2.obs(np.zeros(3))
    assert np.allclose(o0, o2, atol=1e-6) and not np.allclose(o0[3:6], o1[3:6], atol=1e-3)
    # 狗端：同一姿態、同一 offset → 同一重力三維（偏差注入在模擬端是 R_biasᵀ·g，狗上就是 IMU 本身）
    O = ro.Obs(str(NPZ), grav_offset=(0.7, -1.5))
    qw, qx, qy, qz = s1.d.qpos[3:7]
    fr = rl_obs.Frame({n: 0.0 for n in rl_obs.LEG_NAMES + list(ro.WHEEL_NAMES)}, {n: 0.0 for n in rl_obs.LEG_NAMES + list(ro.WHEEL_NAMES)},
                      np.array([qx, qy, qz, qw]), np.zeros(3), 0)
    g_dog = O.build(fr, None, (0, 0, 0), np.zeros(16))[3:6]
    g_sim_fix_only = ro.grav_fix_matrix(0.7, -1.5) @ (ri.quat_to_R(s1.d.qpos[3:7]).T @ DOWN)
    assert np.allclose(g_dog, g_sim_fix_only, atol=1e-6)


def test_m9_passes_grav_offset_to_rl16_obs_only():
    src = (ROOT / "realbot" / "M9_gait.py").read_text(encoding="utf-8")
    assert '"--grav-offset"' in src and 'obsmod.Obs(a.v3, grav_offset=grav_off)' in src
    assert 'if policy.layout == "rl16":\n            O3 = obsmod.Obs(a.v3, grav_offset=grav_off)' in src
    assert "--grav-offset 只有 rl16 版面" in src
    assert "重力歸零 roll" in src                                        # 乾跑／實跑表頭都印，現場對 READY 讀值
    # 傾角護欄仍用原始 IMU（read_imu_rp），不經過 Obs
    assert "roll, pitch = read_imu_rp()" in src
