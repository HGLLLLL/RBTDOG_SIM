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


# ---------- 關卡 2：站姿穩定 ----------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))


def test_d1_model_constants_are_single_source_of_truth():
    import d1_model

    assert d1_model.LEGS == ["FL", "FR", "RL", "RR"]
    assert d1_model.HOME3 == pytest.approx([0.0, 1.05, -2.00])
    assert d1_model.HOME12 == pytest.approx([0.0, 1.05, -2.00] * 4)
    assert d1_model.OBS_DIM == 73
    assert d1_model.ACT_DIM == 12
    assert (d1_model.CTRL_DT, d1_model.SIM_DT) == (0.02, 0.004)


def test_make_model_both_variants_compile():
    import d1_model

    for mjx in (False, True):
        m = d1_model.make_model(mjx=mjx)
        assert m.nq == 19 and m.nu == 12
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_home_pose_is_statically_stable():
    """關卡 2：站姿穩定。

    量測方式是「先沉降 1 秒，再量之後 1 秒的漂移」，而不是量總位移。
    原因：kp=80 的位置伺服在 20.56 kg 下必然有靜態撓度，機身會比 keyframe 的
    幾何高度低約 2.5 cm 後停在平衡點。那是物理，不是不穩定。
    真正要測的是「沉降後有沒有停住」。
    （實測基準：沉降 2.47 cm，之後 1 秒 z 峰對峰 1.36 mm，末速 -0.0004 m/s。）
    """
    import d1_model

    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = d1_model.HOME12
    mujoco.mj_forward(m, d)
    z_start = float(d.qpos[2])

    n1 = int(1.0 / m.opt.timestep)
    for _ in range(n1):
        mujoco.mj_step(m, d)
    z_settled = float(d.qpos[2])

    zs = []
    for _ in range(n1):
        mujoco.mj_step(m, d)
        zs.append(float(d.qpos[2]))
    zs = np.array(zs)

    sag = z_start - z_settled
    assert sag < 0.04, (
        f"沉降 {sag*100:.1f} cm 過大（>4 cm），機器人可能正在塌陷。"
        "kp=80/kd=1 為原廠值不得擅改；請回報數據給使用者決定。"
    )
    assert (zs.max() - zs.min()) < 0.003, (
        f"沉降後 1 秒內 z 峰對峰 {(zs.max()-zs.min())*1000:.2f} mm（應 < 3 mm），機身仍在抖動。"
    )
    assert float(d.qvel[2]) == pytest.approx(0.0, abs=0.02), "站定後垂直速度應趨近 0"


def test_home_pose_body_stays_level():
    """站定後機身不應傾倒（重力在機身系的 z 分量應接近 -1）。"""
    import d1_model

    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = d1_model.HOME12
    mujoco.mj_forward(m, d)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    grav_z_in_body = -d.xmat[base].reshape(3, 3)[2, 2]
    assert grav_z_in_body < -0.99, f"機身傾斜過大 (重力 z 分量 {grav_z_in_body:.4f})"


# =====================================================================
# 關卡 3：慣量與幾何的逐項護欄（Task 2 審查加測）
#
# 加測動機：審查者實測發現原有 25 個測試擋不住「慣量分量反號」與
# 「body 原點張冠李戴」——把 FL_hip 的 ixy 反號、或對調兩條腿的 hip y
# 偏移，25 個測試全部照樣通過（test_total_mass_matches_urdf 的
# rel=0.01 等於 ±0.2 kg 容差，形同虛設）。以下改成對 URDF 逐項比對。
#
# 資料來源：zsibot/genisom_model 的 zsl-1w/urdf/ZSL-1W.urdf
#   git clone --depth 1 https://github.com/zsibot/genisom_model
# 期望值抄寫自該 URDF 並寫死在本檔，測試因此不依賴網路。
#
# 註：URDF 全部 17 個 link 的 <inertial><origin rpy> 皆為 0 0 0，
#     故 URDF 的六分量張量就是「body 座標系」下的張量，可與 MJCF
#     還原出來的張量直接逐分量比對。
# =====================================================================

# MJCF body 名 -> (URDF link 名, mass, ipos xyz, 慣量 [ixx, iyy, izz, ixy, ixz, iyz])
URDF_INERTIAL = {
    "base": ("BASE_LINK", 6.7155497,
             (0.0050538027, -0.000631701, 0.0050408732),
             (0.030172666, 0.088387421, 0.10431072,
              -0.00028335181, 0.000084157847, -0.00056913307)),
    "FL_abad": ("FL_ABAD_LINK", 0.45409769,
                (-0.027074453, -0.015458828, 0.0000094167931),
                (0.000251027, 0.0004874808, 0.00054102144,
                 0.000153060, 0.0000000321747, -0.0000000208886)),
    "FL_hip": ("FL_HIP_LINK", 1.34038,
               (-0.0018322311, -0.050798328, -0.01620581),
               (0.0055878127, 0.0036181265, 0.0028642756,
                0.000140269, -0.000271226, 0.0011403235)),
    "FL_knee": ("FL_KNEE_LINK", 0.76510830,          # 注意：FL/RL 是 ...830，FR/RR 是 ...852
                (0.00296222, -0.01016197, -0.18400768),
                (0.00433034, 0.00438029, 0.00042299,
                 -0.00001722, -0.00011502, -0.00027115)),
    "FL_wheel": ("FL_FOOT_LINK", 0.90130429,
                 (0, 0.04488028, 0),
                 (0.00142854, 0.00247230, 0.00142858, 0, 0, 0)),
    "FR_abad": ("FR_ABAD_LINK", 0.45409769,
                (-0.027074453, 0.015458828, -0.0000094167931),
                (0.000251027, 0.0004874808, 0.00054102144,
                 -0.000153060, -0.0000000321747, -0.0000000208886)),
    "FR_hip": ("FR_HIP_LINK", 1.34038,
               (-0.0018322311, 0.050798328, -0.01620581),
               (0.0055878127, 0.0036181265, 0.0028642756,
                -0.000140269, -0.000271226, -0.0011403235)),
    "FR_knee": ("FR_KNEE_LINK", 0.76510852,
                (0.00294408, 0.01016197, -0.18399731),
                (0.00432973, 0.00437978, 0.00042308,
                 0.00001787, -0.00011551, 0.00027077)),
    "FR_wheel": ("FR_FOOT_LINK", 0.90130429,
                 (0, -0.04488028, 0),
                 (0.00142854, 0.00247230, 0.00142858, 0, 0, 0)),
    "RL_abad": ("RL_ABAD_LINK", 0.45409769,
                (0.027074453, -0.015458828, -0.0000094167931),
                (0.000251027, 0.0004874808, 0.00054102144,
                 -0.000153060, 0.0000000321747, 0.0000000208886)),
    "RL_hip": ("RL_HIP_LINK", 1.34038,
               (-0.0018322311, -0.050798328, -0.01620581),
               (0.0055878127, 0.0036181265, 0.0028642756,
                0.000140269, -0.000271226, 0.0011403235)),
    "RL_knee": ("RL_KNEE_LINK", 0.76510830,
                (0.00296222, -0.01016197, -0.18400768),
                (0.00433034, 0.00438029, 0.00042299,
                 -0.00001722, -0.00011502, -0.00027115)),
    "RL_wheel": ("RL_FOOT_LINK", 0.90130429,
                 (0, 0.04488028, 0),
                 (0.00142854, 0.00247230, 0.00142858, 0, 0, 0)),
    "RR_abad": ("RR_ABAD_LINK", 0.45409769,
                (0.027074453, 0.015458828, 0.0000094167931),
                (0.000251027, 0.0004874808, 0.00054102144,
                 0.000153060, -0.0000000321747, 0.0000000208886)),
    "RR_hip": ("RR_HIP_LINK", 1.34038,
               (-0.0018322311, 0.050798328, -0.01620581),
               (0.0055878127, 0.0036181265, 0.0028642756,
                -0.000140269, -0.000271226, -0.0011403235)),
    "RR_knee": ("RR_KNEE_LINK", 0.76510852,
                (0.00294408, 0.01016197, -0.18399731),
                (0.00432973, 0.00437978, 0.00042308,
                 0.00001787, -0.00011551, 0.00027077)),
    "RR_wheel": ("RR_FOOT_LINK", 0.90130429,
                 (0, -0.04488028, 0),
                 (0.00142854, 0.00247230, 0.00142858, 0, 0, 0)),
}
BODIES = list(URDF_INERTIAL)

# MJCF body pos -> URDF 中「以該 link 為 child 的 joint」的 <origin xyz>。
# 四條腿的 hip y 偏移在 URDF 裡是四個不同的值，抄錯或對調不會影響總質量，
# 只能靠這張表擋下來。
URDF_BODY_POS = {
    "FL_abad": (0.17449, 0.062, 0),      "FL_hip": (0, 0.097412, 0),
    "FL_knee": (0, 0, -0.2),             "FL_wheel": (0, 0, -0.21366),
    "FR_abad": (0.1745, -0.062, 0),      "FR_hip": (0, -0.097399, 0),
    "FR_knee": (0, 0, -0.2),             "FR_wheel": (0, 0, -0.21366),
    "RL_abad": (-0.1745, 0.062, 0),      "RL_hip": (0, 0.09735, 0),
    "RL_knee": (0, 0, -0.2),             "RL_wheel": (0, 0, -0.21366),
    "RR_abad": (-0.1745, -0.062, 0),     "RR_hip": (0, -0.09735, 0),
    "RR_knee": (0, 0, -0.2),             "RR_wheel": (0, 0, -0.21366),
}

BASE_SPAWN_Z = 0.2948   # base 的 body pos 不來自 URDF，是 keyframe 的出生高度


def _mjcf_inertia_tensor(model, bid):
    """把 MJCF 編譯後的（主慣量 body_inertia + 主軸 body_iquat）還原成
    body 座標系下的完整 3x3 慣量張量，以便和 URDF 的六分量直接比對。

    為什麼不只比對主慣量（特徵值）：特徵值對某些反號是瞎的。實測
    abad link 把 ixy 反號，特徵值只變動 9.3e-8（相對），會被 rel=1e-6
    的容差放過去；還原成完整張量才抓得到每一個分量的符號。
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.body_iquat[bid])
    R = R.reshape(3, 3)
    return R @ np.diag(model.body_inertia[bid]) @ R.T


def _urdf_inertia_tensor(six):
    ixx, iyy, izz, ixy, ixz, iyz = six
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


@pytest.mark.parametrize("body", BODIES)
def test_link_mass_matches_urdf_exactly(model, body):
    """逐 link 比對質量。原本的總質量測試用 rel=0.01（±0.2 kg），這裡用 rel=1e-9。

    特別注意 KNEE：FL/RL 是 0.76510830、FR/RR 是 0.76510852，URDF 本身左右不同，
    混用不會被總質量測試發現（差 2.2e-7 kg）。
    """
    urdf_link, mass, _, _ = URDF_INERTIAL[body]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"找不到 body {body}"
    assert float(model.body_mass[bid]) == pytest.approx(mass, rel=1e-9), (
        f"{body} 質量與 URDF 的 {urdf_link} 不符"
    )


def test_link_masses_are_all_covered(model):
    """防止有人新增/刪除 body 卻沒同步上面那張表。"""
    assert model.nbody == 1 + len(BODIES), (
        f"body 數應為 1(world) + {len(BODIES)}，實得 {model.nbody}；"
        "若模型結構有變動，URDF_INERTIAL 表必須同步更新"
    )
    total = sum(v[1] for v in URDF_INERTIAL.values())
    assert float(model.body_mass.sum()) == pytest.approx(total, rel=1e-9)
    assert total == pytest.approx(URDF_TOTAL_MASS, abs=1e-3)


@pytest.mark.parametrize("body", BODIES)
def test_body_com_matches_urdf(model, body):
    """body_ipos（質心在 body 系的位置）逐項比對，抓「慣量張冠李戴」。"""
    urdf_link, _, ipos, _ = URDF_INERTIAL[body]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert model.body_ipos[bid] == pytest.approx(np.array(ipos), abs=1e-9), (
        f"{body} 質心與 URDF 的 {urdf_link} 不符"
    )


@pytest.mark.parametrize("body", BODIES)
def test_full_inertia_tensor_matches_urdf(model, body):
    """把 MJCF 的主慣量+主軸還原成完整張量，與 URDF 六分量逐項比對。

    這是唯一能抓到「單一分量反號」的測試：反號不改變質量、不改變質心、
    多數情況下也幾乎不改變特徵值。
    容差取該 link 慣量尺度的 1e-6；實測還原誤差最大只有 3.3e-8（相對），
    而最小的一個非零非對角分量反號會造成 4e-8 絕對變化，仍遠大於容差。
    """
    urdf_link, _, _, six = URDF_INERTIAL[body]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    got = _mjcf_inertia_tensor(model, bid)
    want = _urdf_inertia_tensor(six)
    tol = 1e-6 * np.abs(want).max()
    diff = np.abs(got - want)
    assert diff.max() <= tol, (
        f"{body} 慣量張量與 URDF 的 {urdf_link} 不符（最大偏差 {diff.max():.3e} > 容差 {tol:.3e}）\n"
        f"MJCF:\n{got}\nURDF:\n{want}"
    )


@pytest.mark.parametrize("body", sorted(URDF_BODY_POS))
def test_body_pos_matches_urdf_joint_origin(model, body):
    """body 原點 = URDF 中該 link 對應 joint 的 <origin xyz>。

    四個 hip 的 y 偏移刻意各不相同（FL 0.097412 / FR -0.097399 /
    RL 0.09735 / RR -0.09735），對調或抄成同一個值都會在這裡爆掉。
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert model.body_pos[bid] == pytest.approx(np.array(URDF_BODY_POS[body]), abs=1e-9), (
        f"{body} 的 body pos 與 URDF joint origin 不符"
    )


def test_hip_y_offsets_are_four_distinct_urdf_values(model):
    """把「四個 hip y 偏移互不相同」寫成獨立斷言，避免有人為了圖方便統一成 ±0.0974。"""
    got = {leg: float(model.body_pos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")][1]) for leg in LEGS}
    assert got == pytest.approx(
        {"FL": 0.097412, "FR": -0.097399, "RL": 0.09735, "RR": -0.09735}, abs=1e-9), got
    assert len(set(abs(v) for v in got.values())) == 3, (
        f"URDF 的四個 hip |y| 只有 3 個相異值（RL/RR 同為 0.09735），實得 {got}"
    )


def test_base_body_pos_is_spawn_height(model):
    """base 的 body pos 不是 URDF 值，是出生高度，且必須與 keyframe 的 z 一致。"""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert model.body_pos[bid] == pytest.approx([0.0, 0.0, BASE_SPAWN_Z], abs=1e-9)
    assert float(model.key_qpos[0][2]) == pytest.approx(BASE_SPAWN_Z, abs=1e-9)


# ---------- Task 2 審查項 2：輪子碰撞圓柱的位置 ----------

def test_wheel_collision_cylinder_sits_at_geometric_center(model):
    """碰撞圓柱要對齊輪盤的「幾何中心」y=±0.0475，不是慣量質心 ±0.04488028。

    來源：STL 實測 FL_FOOT_LINK.STL 的 y 範圍 0.0234988~0.0714999
    → 幾何中心 0.0474994、半寬 0.0240005（與 class wheel 的 size 第二項 0.0240 吻合），
    x/z 半徑 0.070394（多面體內接於 r=0.0710）。
    質心 0.04488028 比幾何中心內縮 2.6 mm，四輪合計輪距會少 5.2 mm。
    <inertial> 的 ±0.04488028 是真質心，必須維持不動。
    """
    expect = {"FL": +0.0475, "FR": -0.0475, "RL": +0.0475, "RR": -0.0475}
    for leg in LEGS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
        assert model.geom_pos[gid] == pytest.approx([0.0, expect[leg], 0.0], abs=1e-9), (
            f"{leg} 碰撞圓柱 pos 應為 y={expect[leg]}（輪盤幾何中心），"
            f"實得 {model.geom_pos[gid]}"
        )
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_wheel")
        assert abs(float(model.body_ipos[bid][1])) == pytest.approx(0.04488028, abs=1e-9), (
            f"{leg}_wheel 的 <inertial> 質心必須維持 ±0.04488028（那不是錯誤）"
        )


def test_home_pose_track_width_matches_stl(model):
    """home 姿態下左右輪碰撞中心的間距，應等於 abad 橫向鏈 + STL 幾何中心。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    y = {leg: float(data.geom_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)][1]) for leg in LEGS}
    # 0.062(abad) + 0.097412(hip) + 0.0475(輪幾何中心) = 0.206912，左右合計 0.413824
    assert (y["FL"] - y["FR"]) == pytest.approx(0.062 + 0.097412 + 0.0475
                                                + 0.062 + 0.097399 + 0.0475, abs=1e-6)
    assert (y["RL"] - y["RR"]) == pytest.approx(2 * (0.062 + 0.09735 + 0.0475), abs=1e-6)


# ---------- Task 2 審查項 3：站立高度常數 ----------

def test_nominal_height_matches_settled_simulation():
    """d1_model.NOMINAL_HEIGHT 必須是「真的站定後」的機身高度，不是 keyframe 的運動學值。

    後續訓練的高度獎勵若拿 key_qpos[2]=0.2948 當目標，會跟 kp=80 的靜態撓度對打。
    """
    import d1_model

    assert d1_model.NOMINAL_HEIGHT < float(d1_model.make_model().key_qpos[0][2]), (
        "NOMINAL_HEIGHT 應低於 keyframe 的純運動學高度"
    )
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = d1_model.HOME12
    mujoco.mj_forward(m, d)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z = float(d.qpos[2])
    assert z == pytest.approx(d1_model.NOMINAL_HEIGHT, abs=0.005), (
        f"模擬 3 秒後機身 z = {z:.5f} m，與 NOMINAL_HEIGHT "
        f"{d1_model.NOMINAL_HEIGHT} 差超過 5 mm；請重新量測並更新常數"
    )


def test_cpu_and_mjx_scenes_differ_only_in_solver_iterations():
    """MJX 的低迭代設定不得洩漏到 CPU 場景。

    實測：iterations=1 會讓開迴路走路少走 16%（3.82m vs 4.57m，理論 4.61m），
    因為接觸約束只解一次迭代會造成穿透與打滑。
    """
    import d1_model

    cpu = d1_model.make_model(mjx=False)
    mjx = d1_model.make_model(mjx=True)
    assert mjx.opt.iterations < cpu.opt.iterations, "MJX 場景應使用較少迭代"
    assert cpu.opt.iterations >= 50, f"CPU 場景迭代數 {cpu.opt.iterations} 過低，恐造成接觸打滑"
    # 其餘物理設定必須一致，否則訓練與推論會對不起來
    assert cpu.opt.timestep == mjx.opt.timestep
    assert cpu.opt.cone == mjx.opt.cone
    assert cpu.opt.impratio == mjx.opt.impratio
