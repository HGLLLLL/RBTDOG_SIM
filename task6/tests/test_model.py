"""關卡 1：MJCF 與 URDF 對帳。數值全部來自 zsl-1w/urdf/ZSL-1W.urdf（輪足版）。"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

XML = str(Path(__file__).resolve().parents[1] / "model" / "d1_edu_w" / "d1_edu_w.xml")
LEGS = ["FL", "FR", "RL", "RR"]

URDF_TOTAL_MASS = 20.559          # 17 個 link 質量總和（含四顆 0.9013 kg 的輪）
WHEEL_RADIUS = 0.0710             # STL 實測
LIMITS = {                        # rad，URDF <limit lower/upper>
    "abad": (-0.4887, 0.4887),
    "hip": (-1.152, 2.967),
    "knee": (-2.723, -0.602),
}


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(XML)


def test_freejoint_present_and_dof_count(model):
    # 沒有 freejoint 的話 nq=12 且 BASE_LINK 會被熔進 worldbody
    assert model.nq == 19, f"nq 應為 19（7 自由基座 + 12 關節），實得 {model.nq}"
    assert model.nv == 18
    assert model.nu == 12, f"輪子不建關節，致動器應為 12 個，實得 {model.nu}"


def test_wheels_have_no_joint(model):
    """輪子熔接鎖死：URDF 的 4 個 *_FOOT_JOINT 不得出現在 MJCF。"""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert not [n for n in names if n and ("wheel" in n or "foot" in n.lower())], \
        f"輪子不該有關節，實際關節清單：{names}"
    # 1 個 freejoint + 12 個 hinge
    assert model.njnt == 13, f"關節總數應為 13（1 freejoint + 12 hinge），實得 {model.njnt}"


def test_total_mass_matches_urdf(model):
    total = float(model.body_mass.sum())
    assert total == pytest.approx(URDF_TOTAL_MASS, rel=0.01), (
        f"總質量 {total:.3f} kg 與 URDF 的 {URDF_TOTAL_MASS} kg 差超過 1%；"
        "最常見原因是缺 freejoint 導致 BASE_LINK 的 6.716 kg 被熔進 world，"
        "或是漏掉四顆 0.9013 kg 的輪子"
    )


def test_wheel_mass_preserved(model):
    """鎖死不等於簡化掉：四顆輪的質量與慣量都要在。"""
    for leg in LEGS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_wheel")
        assert bid >= 0, f"找不到 body {leg}_wheel"
        assert float(model.body_mass[bid]) == pytest.approx(0.90130429, rel=1e-6)


@pytest.mark.parametrize("leg", LEGS)
@pytest.mark.parametrize("part", ["abad", "hip", "knee"])
def test_joint_limits_match_urdf(model, leg, part):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{part}_joint")
    assert jid >= 0, f"找不到關節 {leg}_{part}_joint"
    lo, hi = model.jnt_range[jid]
    assert (lo, hi) == pytest.approx(LIMITS[part], abs=1e-4)


@pytest.mark.parametrize("leg", LEGS)
def test_knee_axis_is_positive_y(model, leg):
    """輪足版 ZSL-1w 的膝軸是 +y（點足版 ZSL-1 是 -y）。抄錯會導致 home 姿態站不起來。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_knee_joint")
    assert model.jnt_axis[jid] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_actuator_order_and_gains(model):
    expected = [f"{leg}_{part}" for leg in LEGS for part in ("abad", "hip", "knee")]
    got = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert got == expected, "致動器順序必須是 FL,FR,RL,RR × abad,hip,knee"
    # 位置伺服：tau = kp*(ctrl - q) - kd*qd，kp=80 kd=1（原廠 demo 值）
    assert model.actuator_gainprm[:, 0] == pytest.approx(80.0)
    assert model.actuator_biasprm[:, 1] == pytest.approx(-80.0)
    assert model.actuator_biasprm[:, 2] == pytest.approx(-1.0)
    assert model.actuator_forcerange[:, 0] == pytest.approx(-28.0)
    assert model.actuator_forcerange[:, 1] == pytest.approx(28.0)


def test_named_entities_exist(model):
    for leg in LEGS:
        for suffix in ("abad", "hip", "knee", "wheel"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{suffix}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg) >= 0, f"缺輪子碰撞 geom {leg}"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu") >= 0


def test_wheel_collision_is_cylinder_with_correct_radius(model):
    """鎖死的輪子接觸地面是一條線，用 cylinder 不是 sphere。"""
    for leg in LEGS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
        assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER, f"{leg} 應為 cylinder"
        assert float(model.geom_size[gid][0]) == pytest.approx(WHEEL_RADIUS, abs=1e-6)
        assert float(model.geom_size[gid][1]) == pytest.approx(0.0240, abs=1e-6)


def test_home_keyframe(model):
    assert model.nkey >= 1
    qpos = model.key_qpos[0]
    assert qpos[7:19] == pytest.approx([0.0, 1.05, -2.00] * 4, abs=1e-6)
    assert model.key_ctrl[0] == pytest.approx([0.0, 1.05, -2.00] * 4, abs=1e-6)


def test_home_pose_wheels_are_on_the_ground(model):
    """home keyframe 的機身高度要讓四顆輪剛好觸地（誤差 < 5 mm）。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    wheel_z = [data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)][2]
               for leg in LEGS]
    clearance = np.array(wheel_z) - WHEEL_RADIUS
    assert np.abs(clearance).max() < 5e-3, f"四輪離地/陷地量 {clearance}（應接近 0）"
