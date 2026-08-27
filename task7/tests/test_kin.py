"""`realbot/kin.py`（狗上純標準函式庫版）對 `inference/leg_kin.py`（numpy 版）的逐點比對。

★ 存在理由：同一套數學有**兩份實作**，一份在狗上跑、一份在本機跑。
  兩份漂開的話，症狀是「模擬調好的步態上機就是不一樣」，而且**兩邊各自都自洽**
  —— 正是本專案「自洽但錯誤」那一類最難查的問題。

  所以這裡不驗「kin.py 自己說得通」，而是**逐點比對另一份獨立實作**，
  外加拿 MuJoCo 的正向運動學當第三方對照。

⚠️ 需要 numpy（本機端測試）。狗上不會跑這支。
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))
sys.path.insert(0, str(ROOT / "inference"))

import coord     # noqa: E402
import kin       # noqa: E402

np = pytest.importorskip("numpy")
leg_kin = pytest.importorskip("leg_kin")
mmod = pytest.importorskip("max_model")

# SHM 腿名 → `max_model.LEGS`（FR, FL, RR, RL）的索引。
# ★ 這張表本身就是被測物之一 —— 對錯了，兩份實作才可能對得起來。
NAME2K = {"fr": 0, "fl": 1, "br": 2, "bl": 3}

TOL = 1e-9


def test_leg_index_map_matches_max_model_order():
    """腿名↔索引的對應要和 `max_model.LEGS` 一致（腿序是本專案的老陷阱）。"""
    mm2shm = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
    for k, name in enumerate(mmod.LEGS):
        assert NAME2K[mm2shm[name]] == k


def test_geometry_constants_have_not_drifted():
    """兩份實作各有一份常數副本 —— 漂開就全錯。"""
    assert kin.ABAD_TO_HIP_X == mmod.ABAD_TO_HIP_X
    assert kin.ABAD_TO_FOOT_Y == pytest.approx(mmod.ABAD_TO_FOOT_Y, abs=1e-15)
    assert kin.L_THIGH == mmod.L_THIGH
    assert kin.L_SHANK == mmod.L_SHANK
    assert kin.WHEEL_RADIUS == mmod.WHEEL_RADIUS
    for name, k in NAME2K.items():
        assert kin.SIDE[name] == (float(mmod.SIDE_X[k]), float(mmod.SIDE_Y[k])), name


def _random_q(rng):
    return (rng.uniform(-0.5, 0.5), rng.uniform(-2.0, 2.0), rng.uniform(-2.6, 2.6))


def test_fk_matches_numpy_version_pointwise():
    rng = random.Random(20260827)
    worst = 0.0
    for _ in range(4000):
        name = rng.choice(list(kin.SIDE))
        q = _random_q(rng)
        a = kin.fk(name, *q)
        b = leg_kin.fk(NAME2K[name], np.array(q))
        worst = max(worst, float(np.max(np.abs(np.array(a) - b))))
    assert worst < TOL, f"FK 最大差 {worst:.3e} m"


def test_ik_matches_numpy_version_pointwise():
    rng = random.Random(20260828)
    worst = 0.0
    for _ in range(4000):
        name = rng.choice(list(kin.SIDE))
        q = _random_q(rng)
        p = kin.fk(name, *q)
        ks = -1.0 if name in ("fl", "fr") else +1.0
        a, ca = kin.ik(name, *p, knee_sign=ks)
        b, cb = leg_kin.ik_ex(NAME2K[name], np.array(p), ks)
        assert ca == cb, f"{name} 縮限旗標不一致"
        worst = max(worst, float(np.max(np.abs(np.array(a) - b))))
    assert worst < TOL, f"IK 最大差 {worst:.3e} rad"


def _zp(q2, q3):
    """平面二連桿的 z（ABAD 轉回去之後）。IK 假設它 <= 0，也就是腳在髖下方。"""
    return -kin.L_THIGH * math.cos(q2) - kin.L_SHANK * math.cos(q2 + q3)


def test_ik_recovers_fk_round_trip():
    """IK(FK(q)) == q —— **只在 IK 的有效分支內**（腳在髖下方）。"""
    rng = random.Random(20260829)
    worst, n = 0.0, 0
    while n < 3000:
        name = rng.choice(list(kin.SIDE))
        q1 = rng.uniform(-0.4, 0.4)
        q2 = rng.uniform(-1.5, 1.5)
        q3 = rng.uniform(0.4, 2.4) * (-1 if name in ("fl", "fr") else +1)
        if _zp(q2, q3) > -0.05:        # 腳翹到髖以上 → 不在 IK 的分支內
            continue
        n += 1
        p = kin.fk(name, q1, q2, q3)
        got, clamped = kin.ik(name, *p, knee_sign=math.copysign(1, q3))
        assert not clamped
        worst = max(worst, max(abs(g - w) for g, w in zip(got, (q1, q2, q3))))
    assert worst < 1e-9, f"往返最大差 {worst:.3e} rad"


def test_ik_branch_limit_is_documented_and_far_from_m8_usage():
    """★ IK 只解「腳在髖下方」那一支 —— 這是真的限制，不是 bug。

    M8 要把腳抬高，抬過頭就會跨出這個分支（`zp` 由負轉正），
    IK 會悄悄給出另一組解。所以要知道邊界在哪：
    從 `stand` 姿勢起算，**抬 430 mm 才碰到邊界**，
    而 CPG 的 `g_c = 0.08`（80 mm）只用掉五分之一不到。
    """
    pose = coord.POSES["stand"]
    for leg in kin.SIDE:
        x, y, z = kin.foot_of(leg, pose)
        ks = kin.knee_sign_of(pose, leg)
        margin = None
        for mm_ in range(0, 600, 5):
            q, clamped = kin.ik(leg, x, y, z + mm_ / 1000.0, ks)
            if clamped or _zp(q[1], q[2]) > -0.02:
                margin = mm_ / 1000.0
                break
        assert margin is not None and margin > 0.30, (
            f"{leg} 抬腿邊界只有 {margin} m，離 g_c=0.08 太近")


def test_unreachable_target_is_clamped_not_nan():
    """構不到時要沿徑向縮限並回報，不能吐 NaN（靜默 NaN 會毀掉整段動作）。"""
    q, clamped = kin.ik("fl", 0.0, 0.0, -5.0, knee_sign=-1.0)
    assert clamped
    assert all(math.isfinite(x) for x in q)


def test_stand_pose_foot_height_matches_mujoco():
    """★ 第三方對照：拿 MuJoCo 的正向運動學驗 `stand` 姿勢的輪心高度。

    兩份 python 實作可能一起錯（它們是同一套推導）。MJCF 是獨立來源。
    """
    mujoco = pytest.importorskip("mujoco")
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    m = mujoco.MjModel.from_xml_path(mmod.SCENE)
    d = mujoco.MjData(m)
    pose = coord.POSES["stand"]
    mujoco.mj_resetData(m, d)
    d.qpos[mmod.LEG_QPOS_IDX] = [pose[lg + k] for lg in mmod.LEGS_SHM_ORDER] \
        if hasattr(mmod, "LEGS_SHM_ORDER") else \
        [pose[{"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}[lg] + k]
         for lg in mmod.LEGS for k in coord.LEG_KINDS]
    mujoco.mj_forward(m, d)
    names = {"fr": "FAR_FOOT_LINK", "fl": "FBL_FOOT_LINK",
             "br": "RAR_FOOT_LINK", "bl": "RBL_FOOT_LINK"}
    base = d.xpos[m.body("base_link").id]
    # ⚠️ kin 給的是「相對該腿 ABAD 原點」，MuJoCo 給的是世界座標，
    #    兩者差一個固定的 ABAD-在-機身上 偏移。四條腿的偏移相同，
    #    所以要比**彼此的相對高度**，不能直接比絕對值。
    zs_kin = [kin.foot_of(l, pose)[2] for l in names]
    zs_mj = [float(d.xpos[m.body(b).id][2] - base[2]) for b in names.values()]
    spread = max(abs((a - zs_kin[0]) - (b - zs_mj[0]))
                 for a, b in zip(zs_kin, zs_mj))
    assert spread < 1e-6, f"四輪相對高度與 MuJoCo 差 {spread:.3e} m"


def test_lifting_the_foot_needs_a_predictable_joint_change():
    """M8 的核心操作：把腳抬高 Δz，關節角要往可預期的方向走。"""
    pose = coord.POSES["stand"]
    for leg in kin.SIDE:
        x, y, z = kin.foot_of(leg, pose)
        ks = kin.knee_sign_of(pose, leg)
        q0, c0 = kin.ik(leg, x, y, z, ks)
        q1, c1 = kin.ik(leg, x, y, z + 0.08, ks)     # 抬 80 mm
        assert not c0 and not c1, f"{leg} 被縮限"
        # 抬腳 = 腿收短 → 膝更彎（|q3| 變大）
        assert abs(q1[2]) > abs(q0[2]), leg
        # 而且回推得到的高度差要就是 0.08
        assert kin.fk(leg, *q1)[2] - kin.fk(leg, *q0)[2] == pytest.approx(0.08, abs=1e-9)
