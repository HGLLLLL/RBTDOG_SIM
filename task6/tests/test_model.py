"""關卡 1：MJCF 與 URDF 對帳。數值全部來自 zsl-1/urdf/ZSL-1.urdf。"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

XML = str(Path(__file__).resolve().parents[1] / "model" / "d1_edu" / "d1_edu.xml")
LEGS = ["FL", "FR", "RL", "RR"]

URDF_TOTAL_MASS = 15.186          # 17 個 link 質量總和
LIMITS = {                        # rad，URDF <limit lower/upper>
    "abad": (-0.4887, 0.4887),
    "hip": (-1.1519, 2.967),
    "knee": (-2.723, -0.602),
}


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(XML)


def test_freejoint_present_and_dof_count(model):
    # 沒有 freejoint 的話 nq=12 且 BASE_LINK 會被熔進 worldbody
    assert model.nq == 19, f"nq 應為 19（7 自由基座 + 12 關節），實得 {model.nq}"
    assert model.nv == 18


def test_total_mass_matches_urdf(model):
    total = float(model.body_mass.sum())
    assert total == pytest.approx(URDF_TOTAL_MASS, rel=0.01), (
        f"總質量 {total:.3f} kg 與 URDF 的 {URDF_TOTAL_MASS} kg 差超過 1%；"
        "最常見原因是缺 freejoint 導致 BASE_LINK 的 6.268 kg 被熔進 world"
    )


@pytest.mark.parametrize("leg", LEGS)
@pytest.mark.parametrize("part", ["abad", "hip", "knee"])
def test_joint_limits_match_urdf(model, leg, part):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{part}_joint")
    assert jid >= 0, f"找不到關節 {leg}_{part}_joint"
    lo, hi = model.jnt_range[jid]
    assert (lo, hi) == pytest.approx(LIMITS[part], abs=1e-4)


@pytest.mark.parametrize("leg", LEGS)
def test_knee_axis_is_negative_y(model, leg):
    """D1 的膝軸是 -y，與 Go2 相反。抄錯會導致 home 姿態站不起來。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_knee_joint")
    assert model.jnt_axis[jid] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


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
        for suffix in ("abad", "hip", "knee", "foot"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{suffix}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg) >= 0, f"缺足端 geom {leg}"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu") >= 0


def test_home_keyframe(model):
    assert model.nkey >= 1
    qpos = model.key_qpos[0]
    assert qpos[7:19] == pytest.approx([0.0, -0.94, -1.80] * 4, abs=1e-6)
    assert model.key_ctrl[0] == pytest.approx([0.0, -0.94, -1.80] * 4, abs=1e-6)


def test_home_pose_feet_are_on_the_ground(model):
    """home keyframe 的機身高度要讓四腳剛好觸地（誤差 < 5 mm）。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    foot_z = [data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)][2]
              for leg in LEGS]
    radius = model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL")][0]
    clearance = np.array(foot_z) - radius
    assert np.abs(clearance).max() < 5e-3, f"四腳離地/陷地量 {clearance}（應接近 0）"
