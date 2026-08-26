"""CPG 的相位、軌跡與撓度補償測試。

測試的**題目**沿用 task6 `test_cpg_d1.py`（相位鎖定、左右對稱、限位），
再加上這次實際踩到的兩個坑：撓度補償的語意、以及偏航必須被量到。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_max
import leg_kin
import max_model as mm


# =============================================================================
# 相位
# =============================================================================
def test_trot_phase_is_diagonal():
    """釘住 trot 的對角關係。腿序 FR/FL/RR/RL → 對角是 (FR,RL) 與 (FL,RR)。

    腿序或相位表被就地改掉時，這個測試會抓到 —— 否則 trot 會靜默退化成別的步態。
    """
    p = cpg_max.PHASE_TROT
    assert p[0] == pytest.approx(p[3])            # FR 與 RL 同相
    assert p[1] == pytest.approx(p[2])            # FL 與 RR 同相
    assert abs(p[0] - p[1]) == pytest.approx(np.pi)


def test_walk_phase_is_lateral_sequence():
    """walk 必須是四腿均分一圈的側序走：左後→左前→右後→右前。"""
    p = np.asarray(cpg_max.PHASE_WALK)
    order = [mm.LEGS[i] for i in np.argsort(p)]
    assert order == ["RL", "FL", "RR", "FR"]
    # 四腿均分：排序後相鄰差都是 pi/2
    assert np.allclose(np.diff(np.sort(p)), np.pi / 2)


def test_phase_constants_readonly():
    for arr in (cpg_max.PHASE_TROT, cpg_max.PHASE_WALK):
        with pytest.raises(ValueError):
            arr[0] = 1.0


def test_cpg_locks_to_target_phase():
    """振盪器要把相位差拉到 PHI 定義的關係，而且要能從被打亂的初始狀態收回來。"""
    phase = cpg_max.PHASE_WALK
    step = cpg_max.make_cpg_step(phase)
    c = cpg_max.cpg_init(phase)
    c["theta"] = np.array([0.3, 1.1, 5.0, 2.2])          # 故意打亂
    for _ in range(2000):
        c = step(c, np.full(4, 1.8), np.full(4, 1.5), np.full(4, 1.4), mm.CTRL_DT)
    for k in range(4):
        err = (c["theta"][k] - c["theta"][0]) - (phase[k] - phase[0])
        err = (err + np.pi) % (2 * np.pi) - np.pi
        assert abs(np.degrees(err)) < 1.0, f"腿 {mm.LEGS[k]} 相位沒鎖回來：{np.degrees(err):.2f}°"


def test_make_cpg_step_actually_uses_its_phase():
    """換相位表必須真的換到耦合矩陣。

    只改初始相位而沒換耦合矩陣的話，耦合項會把相位拉回舊的關係 —— 這是 task6 的坑。
    """
    trot = cpg_max.make_cpg_step(cpg_max.PHASE_TROT)
    walk = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    c0 = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    ct, cw = dict(c0), dict(c0)
    for _ in range(500):
        ct = trot(ct, np.full(4, 1.8), np.full(4, 1.5), np.full(4, 1.4), mm.CTRL_DT)
        cw = walk(cw, np.full(4, 1.8), np.full(4, 1.5), np.full(4, 1.4), mm.CTRL_DT)
    assert not np.allclose(ct["theta"], cw["theta"], atol=1e-3), \
        "兩種相位表跑出同樣的結果 → 耦合矩陣沒有跟著換"


# =============================================================================
# duty_remap
# =============================================================================
def test_duty_remap_identity_at_half():
    """duty=0.5 時必須恆等於原樣（docstring 裡的證明）。"""
    th = np.linspace(0, 2 * np.pi, 501)[:-1]
    assert np.allclose(cpg_max.duty_remap(th, 0.5), th, atol=1e-12)


def test_duty_remap_swing_fraction():
    """擺動相（sin>0）佔一圈的比例必須等於 1−duty。"""
    th = np.linspace(0, 2 * np.pi, 200001)[:-1]
    for duty in (0.5, 0.6, 0.75, 0.8):
        frac = np.mean(np.sin(cpg_max.duty_remap(th, duty)) > 0)
        assert frac == pytest.approx(1.0 - duty, abs=1e-3), f"duty={duty} 實得 {frac:.4f}"


def test_duty_remap_is_monotonic():
    """重映射只能改時間分配，不能讓相位倒退。"""
    th = np.linspace(0, 2 * np.pi, 10001)[:-1]
    for duty in (0.5, 0.75, 0.8):
        assert np.all(np.diff(cpg_max.duty_remap(th, duty)) >= -1e-12)


# =============================================================================
# 足端軌跡
# =============================================================================
def _state(theta):
    return {"rx": np.full(4, 1.8), "rx_d": np.zeros(4),
            "ry": np.full(4, 1.5), "ry_d": np.zeros(4), "theta": np.asarray(theta)}


def test_mu_y_1p5_gives_zero_lateral():
    """★ mu_y=1.5 → fy=0 → 橫向偏移必須**恰好**是 0。

    這是直線走路的前提。實測 mu_y 只要偏離 1.5，四種值裡有三種直接跌倒。
    """
    f0 = leg_kin.home_foot(mm.HOME)
    for th in (0.0, 1.0, 2.5, 4.0):
        tgt = cpg_max.foot_targets(_state(np.full(4, th)), f0, 0.0, 0.08, 0.10, 0.12, 0.8)
        assert np.allclose(tgt[:, 1], f0[:, 1], atol=1e-15), "mu_y=1.5 卻產生了橫向偏移"


def test_z_sag_only_lifts_swing_leg():
    """★ 撓度補償只能加在擺動相。

    站立相就是靠位置伺服的追蹤誤差在出力撐機身的，站立相也補的話等於把機身放掉。
    """
    f0 = leg_kin.home_foot(mm.HOME)
    sag = 0.03
    for th in (0.5, 1.5, 2.5):        # sin>0，擺動相
        a = cpg_max.foot_targets(_state(np.full(4, th)), f0, 0, 0.08, 0.10, 0.12, 0.5, 0.0)
        b = cpg_max.foot_targets(_state(np.full(4, th)), f0, 0, 0.08, 0.10, 0.12, 0.5, sag)
        assert np.all(b[:, 2] > a[:, 2]), "擺動相沒有被補償抬高"
    for th in (3.5, 4.5, 5.5):        # sin<0，站立相
        a = cpg_max.foot_targets(_state(np.full(4, th)), f0, 0, 0.08, 0.10, 0.12, 0.5, 0.0)
        b = cpg_max.foot_targets(_state(np.full(4, th)), f0, 0, 0.08, 0.10, 0.12, 0.5, sag)
        assert np.allclose(a[:, 2], b[:, 2]), "站立相被補償了 → 機身會被放掉"


def test_swing_peak_equals_g_c_plus_sag():
    """擺動最高點必須是 g_c + z_sag（這就是「指令抬腿量」的定義）。"""
    f0 = leg_kin.home_foot(mm.HOME)
    g_c, sag = 0.08, mm.STATIC_SAG
    tgt = cpg_max.foot_targets(_state(np.full(4, np.pi / 2)), f0, 0, g_c, 0.10, 0.12, 0.5, sag)
    assert np.allclose(tgt[:, 2] - f0[:, 2], g_c + sag, atol=1e-12)


def test_stride_direction_is_forward():
    """站立相時足端必須由前往後掃 —— 掃反了機器人就會倒退走。

    這正是這次踩到的坑：抬腿不足時腿在擺動相被拖著走，整台往後退。
    """
    f0 = leg_kin.home_foot(mm.HOME)
    # duty=0.5 下，th=π 是觸地（最前），th=2π 是離地（最後）
    front = cpg_max.foot_targets(_state(np.full(4, np.pi)), f0, 0, 0.08, 0.10, 0.12, 0.5)
    back = cpg_max.foot_targets(_state(np.full(4, 2 * np.pi - 1e-9)), f0, 0, 0.08, 0.10, 0.12, 0.5)
    assert np.all(front[:, 0] > back[:, 0]), "站立相掃向不對，會倒退走"


def test_x_off_shifts_all_legs_equally():
    f0 = leg_kin.home_foot(mm.HOME)
    a = cpg_max.foot_targets(_state(np.full(4, 1.0)), f0, 0.00, 0.08, 0.10, 0.12, 0.8)
    b = cpg_max.foot_targets(_state(np.full(4, 1.0)), f0, -0.04, 0.08, 0.10, 0.12, 0.8)
    assert np.allclose(a[:, 0] - b[:, 0], 0.04)


# =============================================================================
# IK 串接
# =============================================================================
def test_joint_targets_within_limits_over_full_cycle():
    """★ 整個步態週期、四條腿都不能超出 jnt_range —— 逐腿檢查，不是只看一條。

    左右 ABAD 限位鏡像、前後 HIP 限位也鏡像，所以「檢查一條腿就當四條都過」會漏。
    """
    m = mm.make_model()
    rngs = mm.leg_joint_ranges(m)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    from cpg_walk_max import GAITS, D_STEP_Y
    for name, cfg in GAITS.items():
        step = cpg_max.make_cpg_step(cfg["phase"])
        c = cpg_max.cpg_init(cfg["phase"])
        worst = 0.0
        for _ in range(400):
            c = step(c, np.full(4, cfg["mu_x"]), np.full(4, 1.5),
                     np.full(4, cfg["omega"]), mm.CTRL_DT)
            q, n_clamp = cpg_max.joint_targets(
                c, f0, cfg["x_off"], cfg["g_c"], cfg["d_step"], D_STEP_Y,
                cfg["duty"], ks, mm.STATIC_SAG)
            assert n_clamp == 0, f"{name}: IK 構不到，被縮限"
            over = np.maximum(rngs[:, 0] - q, q - rngs[:, 1]).max()
            worst = max(worst, over)
        assert worst <= 0, f"步態 {name} 超出關節限位 {np.degrees(worst):.2f}°"


def test_joint_targets_left_right_symmetric():
    """直線走路時，同相位的左右腿關節角必須鏡像對稱。

    ABAD 反號、hip/knee 同號（因為左右腿的 hip/knee 軸都是 +y）。
    """
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    c = _state(np.full(4, 1.2))          # 四腿同相位
    q, _ = cpg_max.joint_targets(c, f0, -0.04, 0.08, 0.10, 0.12, 0.8, ks, mm.STATIC_SAG)
    q = q.reshape(4, 3)
    fr, fl, rr, rl = q
    for right, left, tag in ((fr, fl, "前"), (rr, rl, "後")):
        assert right[0] == pytest.approx(-left[0], abs=1e-12), f"{tag}腿 ABAD 不對稱"
        assert right[1] == pytest.approx(left[1], abs=1e-9), f"{tag}腿 HIP 不對稱"
        assert right[2] == pytest.approx(left[2], abs=1e-9), f"{tag}腿 KNEE 不對稱"


def test_joint_targets_front_rear_mirrored():
    """★ 前後腿必須是鏡像的（X 型站姿），不能同號。

    這就是交接文件警告「四腿共用一個 HOME3 會做出怪東西」的那件事。
    """
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    c = _state(np.full(4, 1.2))
    q, _ = cpg_max.joint_targets(c, f0, -0.04, 0.08, 0.10, 0.12, 0.8, ks, mm.STATIC_SAG)
    q = q.reshape(4, 3)
    assert q[0, 1] * q[2, 1] < 0, "前後 HIP 同號 → 不是 X 型"
    assert q[0, 2] * q[2, 2] < 0, "前後 KNEE 同號 → 不是 X 型"


# =============================================================================
# 工具
# =============================================================================
def test_circ_std_handles_wrap():
    """圓形統計：跨 ±180° 的一組角度，標準差不能虛胖。"""
    a = np.array([np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02, -np.pi + 0.02])
    assert cpg_max.circ_std(a) < 0.05
    assert np.std(a) > 3.0                      # 一般標準差會虛胖到 3 rad 以上


def test_circ_std_never_negative_zero():
    """完全鎖定時要回 0.0，不能是 −0.0（會印成 "-0.0" 看起來像 bug）。"""
    v = cpg_max.circ_std(np.zeros(10))
    assert v == 0.0 and not np.signbit(v)


def test_yaw_deg():
    assert cpg_max.yaw_deg(np.array([1.0, 0, 0, 0])) == pytest.approx(0.0)
    h = np.sqrt(0.5)
    assert cpg_max.yaw_deg(np.array([h, 0, 0, h])) == pytest.approx(90.0)
    assert cpg_max.yaw_deg(np.array([h, 0, 0, -h])) == pytest.approx(-90.0)
