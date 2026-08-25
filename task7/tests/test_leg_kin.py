"""驗證解析式運動學。

核心測試是 `test_fk_matches_mujoco`：**把我們寫的 FK 拿去跟 MuJoCo 自己算的比對**。
這一項會失敗，就代表我們讀錯了官方 MJCF 的連桿尺寸或關節軸——
那是最容易「自洽但錯誤」的地方（task6 在校正層踩過同型的坑）。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import leg_kin
import max_model as mm

# 官方 MJCF 在三條腿的 KNEE body 上有 2.14e-5 rad 的殘留 quat，本檔的解析式忽略它。
# 實測影響 < 0.03 mm，容差取 0.1 mm。
TOL_M = 1e-4


@pytest.fixture(scope="module")
def mj():
    m = mm.make_model()
    return m, mujoco.MjData(m)


def _mj_foot_rel_abad(m, d, k: int, q3):
    """MuJoCo 實算：第 k 腿輪心相對 ABAD 原點。"""
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0                      # 單位四元數，機身不轉
    d.qpos[mm.LEG_QPOS_IDX[3 * k:3 * k + 3]] = q3
    mujoco.mj_forward(m, d)
    return (d.xpos[mm.foot_body_ids(m)[k]] - d.xpos[mm.abad_body_ids(m)[k]]).copy()


def test_fk_matches_mujoco(mj):
    """解析式 FK 對上 MuJoCo 的 FK —— 四條腿、關節全行程隨機取樣。"""
    m, d = mj
    rng = np.random.default_rng(0)
    rngs = mm.leg_joint_ranges(m)
    worst = 0.0
    for k in range(4):
        lo, hi = rngs[3 * k:3 * k + 3, 0], rngs[3 * k:3 * k + 3, 1]
        for _ in range(300):
            q = rng.uniform(lo, hi)
            err = np.abs(leg_kin.fk(k, q) - _mj_foot_rel_abad(m, d, k, q)).max()
            worst = max(worst, err)
            assert err < TOL_M, f"腿 {mm.LEGS[k]} q={q} 誤差 {err:.2e} m"
    print(f"\n  FK vs MuJoCo 最大誤差 {worst * 1000:.4f} mm")


def test_ik_roundtrip_at_home(mj):
    """HOME 姿態附近：IK(FK(q)) 應該回到 q 本身。"""
    m, _ = mj
    rng = np.random.default_rng(1)
    ks = leg_kin.knee_sign_of(mm.HOME)
    for k in range(4):
        for _ in range(300):
            # 在 HOME 附近擾動，涵蓋比實際步態更大的範圍
            q = np.asarray(mm.HOME[k]) + rng.uniform([-0.4, -0.5, -0.5], [0.4, 0.5, 0.5])
            back = leg_kin.ik(k, leg_kin.fk(k, q), ks[k])
            assert np.abs(back - q).max() < 1e-9, f"腿 {mm.LEGS[k]}: {q} → {back}"


def test_ik_hits_target(mj):
    """IK 解出來的角度餵回 MuJoCo，輪心要真的落在目標上（端到端，不只是自我一致）。"""
    m, d = mj
    rng = np.random.default_rng(2)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    for k in range(4):
        for _ in range(200):
            # 覆蓋比實際步態大的足端工作區：前後 ±0.20、側向 ±0.08、上下 ±0.12
            tgt = f0[k] + rng.uniform([-0.20, -0.08, -0.12], [0.20, 0.08, 0.12])
            q, clamped = leg_kin.ik_ex(k, tgt, ks[k])
            if clamped:
                continue
            got = _mj_foot_rel_abad(m, d, k, q)
            assert np.abs(got - tgt).max() < TOL_M, f"腿 {mm.LEGS[k]} 目標 {tgt} 實得 {got}"


def test_home_is_x_shaped(mj):
    """★ 站姿是前後鏡像的 X 型：四輪 x 應該前後對稱。

    這一項就是交接文件警告「四腿共用一個 HOME3 會錯」的那個結構性質。
    如果哪天有人把 HOME 改成四腿同號，這個測試會抓到。
    """
    f = leg_kin.home_foot(mm.HOME)
    hip_x = mm.SIDE_X * mm.HIP_X
    wheel_x = hip_x + f[:, 0]                       # 相對機身
    assert wheel_x[0] == pytest.approx(wheel_x[1], abs=1e-9)     # 左右前腿同 x
    assert wheel_x[2] == pytest.approx(wheel_x[3], abs=1e-9)     # 左右後腿同 x
    assert wheel_x[0] == pytest.approx(-wheel_x[2], abs=1e-6), "前後不對稱 → HOME 不是 X 型"
    assert wheel_x[0] > 0.30, "前輪應該在髖前方約 0.34 m"


def test_home_body_height(mj):
    """HOME 的純運動學機身高度應該 = NOMINAL_HEIGHT_KIN，並對得上原廠 body_height 0.48。"""
    f = leg_kin.home_foot(mm.HOME)
    h = -f[:, 2].mean() + mm.WHEEL_RADIUS
    assert h == pytest.approx(mm.NOMINAL_HEIGHT_KIN, abs=5e-4)
    assert abs(h - 0.48) < 0.02, f"與原廠 body_height 0.48 差太多：{h:.4f}"


def test_stand_pose_height(mj):
    """順帶釘住 STAND / CROUCH 的高度，數字取自 SOURCE.md 的驗證表。"""
    for pose, want in ((mm.STAND, 0.5418), (mm.CROUCH, 0.2916)):
        f = leg_kin.home_foot(pose)
        h = -f[:, 2].mean() + mm.WHEEL_RADIUS
        assert h == pytest.approx(want, abs=1e-3)


def test_home_within_joint_limits(mj):
    """HOME / STAND / CROUCH 都必須落在關節限位內 —— 逐腿檢查，不是只看一條。"""
    m, _ = mj
    rngs = mm.leg_joint_ranges(m)
    for name, pose in (("HOME", mm.HOME), ("STAND", mm.STAND), ("CROUCH", mm.CROUCH)):
        q = np.asarray(pose).reshape(12)
        assert np.all(q >= rngs[:, 0]) and np.all(q <= rngs[:, 1]), \
            f"{name} 超出限位：{q} vs {rngs}"


def test_abad_limits_are_mirrored(mj):
    """釘住「左右 ABAD 限位鏡像、前後 HIP 限位鏡像」這個容易被忽略的事實。

    交接文件特別警告過。若哪天換了模型檔而這個性質不再成立，
    「逐腿檢查限位」的必要性就變了，應該要被這個測試逼著重新看一次。
    """
    m, _ = mj
    r = mm.leg_joint_ranges(m)
    abad = {l: tuple(r[3 * k]) for k, l in enumerate(mm.LEGS)}
    hip = {l: tuple(r[3 * k + 1]) for k, l in enumerate(mm.LEGS)}
    assert abad["FR"] == abad["RR"] == (-0.697, 0.523)
    assert abad["FL"] == abad["RL"] == (-0.523, 0.697)
    assert hip["FR"] == hip["FL"] == (-2.442, 2.791)
    assert hip["RR"] == hip["RL"] == (-2.791, 2.442)
