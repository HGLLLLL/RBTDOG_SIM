"""gen_direct（同事純 RL 的直接 PD 目標產生器）：關節序對照、縮放、限位夾、限速、淡入、零動作＝default pose。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))
NPZ = ROOT / "weights" / "rl_only_2026-09-24_np.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="npz 還沒匯出")


@pytest.fixture
def P():
    import gen_direct as gd
    return gd.load_params(str(NPZ))


def test_physx_to_mjcf_mapping(P):
    """PhysX 序（ABAD×4, HIP×4, KNEE×4；腿 FAR FBL RAR RBL）→ MJCF 序（每腿 abad hip knee；腿 FR FL RR RL）。"""
    names = P.K["JOINT_NAMES_PHYSX"]
    mj = [names[i] for i in P.K["PHYSX_TO_MJCF"]]
    assert mj == ["FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT", "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
                  "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT", "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT"]
    import rl_obs
    # MJCF 序的 shm 名：fr1.. fl1.. br1.. bl1..（FR FL RR RL）
    assert rl_obs.LEG_NAMES[0].startswith("fr") and rl_obs.LEG_NAMES[3].startswith("fl") and rl_obs.LEG_NAMES[6].startswith("br")


def test_zero_action_is_default_pose_and_limits_inside_mechanical(P):
    import gen_direct as gd, coord, rl_obs
    g = gd.Gen(P)
    q, tau, c = g.step(np.zeros(3), np.zeros(16), np.zeros(4))
    exp = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, -0.8, 1.5, 0.0, -0.8, 1.5]
    assert np.allclose(q, exp) and np.allclose(g.stance_q12(), exp) and np.allclose(tau, 0)
    for k, n in enumerate(rl_obs.LEG_NAMES):
        lo, hi = coord.limits_of(n)
        assert lo - 1e-6 <= P.T["LO12"][k] and P.T["HI12"][k] <= hi + 1e-6, n


def test_scale_order_clip_slew_ramp(P):
    import gen_direct as gd
    K = dict(P.K, RAMP_STEPS=0, Q_SLEW=10.0)
    g = gd.Gen(gd.Params(P.ref, K, P.T))
    a = np.zeros(16); a[4] = 1.0            # PhysX 4 = FAR_HIP → MJCF index 1（fr2_hip_pitch）
    q, tau, _ = g.step(np.zeros(3), a, np.zeros(4))
    assert np.isclose(q[1], 0.8 + 0.25) and np.isclose(q[0], 0.0) and np.isclose(q[2], -1.5)
    a = np.zeros(16); a[12] = 1.0           # 輪 FR
    q, tau, _ = g.step(np.zeros(3), a, np.array([1.0, 0, 0, 0]))
    assert np.allclose(g.v_nom, [5.0, 0, 0, 0]) and np.allclose(tau, [5.0 - 1.0, 0, 0, 0])
    # 限位夾與 A_CLIP：ABAD 右腿上限 0.523；a=+10 → clip 4 → 0.25×4 = 1.0 → 夾到 0.523，sat_mask 記到
    a = np.zeros(16); a[0] = 10.0
    q, _, _ = g.step(np.zeros(3), a, np.zeros(4))
    assert np.isclose(q[0], 0.523) and g.sat_mask[0] and g.n_sat >= 1
    # 限速：Q_SLEW 0.05 時一步只能動 0.05
    g2 = gd.Gen(gd.Params(P.ref, dict(K, Q_SLEW=0.05), P.T))
    a = np.zeros(16); a[4] = 1.0
    q, _, _ = g2.step(np.zeros(3), a, np.zeros(4))
    assert np.isclose(q[1], 0.8 + 0.05)
    # 淡入：RAMP_STEPS 50 時第 0 步 u=0（零動作）、第 25 步 u=0.5
    g3 = gd.Gen(gd.Params(P.ref, dict(K, RAMP_STEPS=50), P.T))
    q, _, _ = g3.step(np.zeros(3), a, np.zeros(4)); assert np.isclose(q[1], 0.8)
    for _ in range(24):
        g3.step(np.zeros(3), a, np.zeros(4))
    q, _, _ = g3.step(np.zeros(3), a, np.zeros(4)); assert np.isclose(q[1], 0.8 + 0.25 * 0.5)


def test_rejects_wrong_act_dim(P):
    import gen_direct as gd
    g = gd.Gen(P)
    with pytest.raises(ValueError):
        g.step(np.zeros(3), np.zeros(24), np.zeros(4))
