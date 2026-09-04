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


def test_swing_order_is_not_argsort_of_phase():
    """★ 釘住「相位值大的腿先擺動」這件事。

    2026-09-03 之前這個測試寫成 `argsort(相位值)`，把順序判**反**了，
    於是 `PHASE_WALK` 是 diagonal sequence 卻被斷言成 lateral sequence，
    而測試一直通過。腿 k 的擺動起點是 `τ_k = (1 − phase_k/2π) mod 1`。
    """
    p = np.asarray(cpg_max.PHASE_WALK)
    assert [mm.LEGS[i] for i in np.argsort(p)] == ["RL", "FL", "RR", "FR"], \
        "argsort(phase) 的結果變了 —— 下面那條「兩者相反」的斷言要重新檢查"
    assert [mm.LEGS[i] for i in cpg_max.swing_order(p)] == ["RL", "FR", "RR", "FL"], \
        "PHASE_WALK 的實際擺動順序不是 diagonal sequence 了"


def test_phase_walk_is_diagonal_sequence():
    """`PHASE_WALK` ＝ diagonal sequence（左後→右前→右後→左前），四腿均分一圈。"""
    p = np.asarray(cpg_max.PHASE_WALK)
    assert [mm.LEGS[i] for i in cpg_max.swing_order(p)] == ["RL", "FR", "RR", "FL"]
    assert np.allclose(np.diff(np.sort(p)), np.pi / 2)


def test_phase_walk_ls_is_lateral_sequence():
    """`PHASE_WALK_LS` ＝ lateral sequence（左後→左前→右後→右前）。

    文獻上靜態穩定裕度最好的 crawl 序列。與 `PHASE_WALK` 只差 FR/FL 對調。
    """
    p = np.asarray(cpg_max.PHASE_WALK_LS)
    assert [mm.LEGS[i] for i in cpg_max.swing_order(p)] == ["RL", "FL", "RR", "FR"]
    assert np.allclose(np.diff(np.sort(p)), np.pi / 2)
    # 與 DS 只差 FR/FL 對調
    assert np.allclose(p[[1, 0, 2, 3]], cpg_max.PHASE_WALK)


def test_swing_order_matches_actual_trajectory():
    """★ `swing_order` 不能只是重述公式 —— 拿實際軌跡驗它。

    跑 CPG 一圈，記錄每腿第一次進入擺動相（`sin(duty_remap(θ)) > 0`）的時刻，
    照那個時刻排序，必須等於 `swing_order` 說的順序。
    """
    for phase in (cpg_max.PHASE_WALK, cpg_max.PHASE_WALK_LS):
        step = cpg_max.make_cpg_step(phase)
        c = cpg_max.cpg_init(phase)
        first = {}
        prev = np.sin(cpg_max.duty_remap(c["theta"], 0.8)) > 0
        for i in range(4000):
            c = step(c, np.full(4, 1.8), np.full(4, 1.5), np.full(4, 1.4), mm.CTRL_DT)
            now = np.sin(cpg_max.duty_remap(c["theta"], 0.8)) > 0
            for k in range(4):
                if now[k] and not prev[k] and k not in first:
                    first[k] = i
            prev = now
            if len(first) == 4:
                break
        assert len(first) == 4, "一圈之內沒有四腿都擺動過"
        got = sorted(first, key=lambda k: first[k])
        assert got == cpg_max.swing_order(phase), \
            f"實際擺動順序 {[mm.LEGS[i] for i in got]} ≠ " \
            f"swing_order {[mm.LEGS[i] for i in cpg_max.swing_order(phase)]}"


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
# 前後分離的 x_off（x_c 配平 / x_d 軸距）
# =============================================================================
def test_x_off_scalar_equals_uniform_array():
    """★ 純量與「四值相同的陣列」必須逐位元相同。

    這是整個前後分離改動的回歸護欄：純量路徑不能有任何行為變化，
    否則所有既有掃描結果（都是純量掃出來的）就無法與新結果比較。
    """
    f0 = leg_kin.home_foot(mm.HOME)
    c = _state(np.full(4, 1.0))
    a = cpg_max.foot_targets(c, f0, -0.11, 0.08, 0.10, 0.12, 0.8, mm.STATIC_SAG)
    b = cpg_max.foot_targets(c, f0, np.full(4, -0.11), 0.08, 0.10, 0.12, 0.8,
                             mm.STATIC_SAG)
    assert np.array_equal(a, b)
    ks = leg_kin.knee_sign_of(mm.HOME)
    assert np.array_equal(cpg_max.stand_targets(ks, f0, -0.11),
                          cpg_max.stand_targets(ks, f0, np.full(4, -0.11)))


def test_x_off_array_shifts_each_leg_independently():
    f0 = leg_kin.home_foot(mm.HOME)
    c = _state(np.full(4, 1.0))
    base = cpg_max.foot_targets(c, f0, 0.0, 0.08, 0.10, 0.12, 0.8)
    off = np.array([-0.11, -0.11, -0.03, -0.03])       # 腿序 FR, FL, RR, RL
    got = cpg_max.foot_targets(c, f0, off, 0.08, 0.10, 0.12, 0.8)
    assert np.allclose(got[:, 0] - base[:, 0], off)
    assert np.allclose(got[:, 1:], base[:, 1:])        # 只動 x，不碰 y/z


def test_x_off_split_leg_order():
    """`x_off_split` 的腿序必須是 FR, FL, RR, RL —— 前兩腿吃 +x_d，後兩腿吃 −x_d。"""
    got = cpg_max.x_off_split(-0.08, 0.03)
    assert np.allclose(got, [-0.05, -0.05, -0.11, -0.11])


def test_x_c_zero_gives_front_rear_symmetric_stance():
    """★ 釘住幾何事實：姿態前後對稱 ⟺ x_c = 0，與 x_d 無關。

    `f0` 本身已是前後鏡像，所以只要四腿共同的平移量是 0，
    無論 x_d 怎麼拉，前後 |hip| 都相等。這條是「配平與對稱是同一個自由度」的
    形式化 —— 有人日後把 x_c 調離 0 又期待姿態對稱時，這個測試會擋下來。
    """
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    for x_d in (-0.06, 0.0, 0.06):
        q = cpg_max.stand_targets(ks, f0, cpg_max.x_off_split(0.0, x_d)).reshape(4, 3)
        assert abs(q[0, 1]) == pytest.approx(abs(q[2, 1]), abs=1e-12), "hip 不對稱"
        assert abs(q[0, 2]) == pytest.approx(abs(q[2, 2]), abs=1e-12), "knee 不對稱"
    # 反面：x_c ≠ 0 一定不對稱（−110 mm 時實測差 30.5°）
    q = cpg_max.stand_targets(ks, f0, cpg_max.x_off_split(-0.11, 0.0)).reshape(4, 3)
    assert abs(np.degrees(abs(q[0, 1]) - abs(q[2, 1])) - 30.5) < 0.2


def test_x_c_is_support_center_shift_and_x_d_is_wheelbase():
    """★ x_c 移動支撐多邊形中心、x_d 只改軸距 —— 兩者正交。

    這是選 (x_c, x_d) 而非 (front, rear) 當掃描軸的理由；寫成測試免得日後被當成
    「只是換個寫法」而改掉參數化。
    """
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    m = mm.make_model()
    abad_x = np.array([m.body_pos[i][0] for i in mm.abad_body_ids(m)])

    def center_and_base(x_c, x_d):
        q = cpg_max.stand_targets(ks, f0, cpg_max.x_off_split(x_c, x_d)).reshape(4, 3)
        foot_x = abad_x + np.array([leg_kin.fk(k, q[k])[0] for k in range(4)])
        return 0.5 * (foot_x[0] + foot_x[2]), foot_x[0] - foot_x[2]

    c00, b00 = center_and_base(0.0, 0.0)
    for x_c, x_d in ((-0.11, 0.0), (0.0, 0.05), (-0.08, -0.03)):
        c, b = center_and_base(x_c, x_d)
        assert c - c00 == pytest.approx(x_c, abs=1e-12), "支撐中心位移 ≠ x_c"
        assert b - b00 == pytest.approx(2 * x_d, abs=1e-12), "軸距變化 ≠ 2·x_d"


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


def test_knee_peak_indices_are_really_rear_knees():
    """★ 釘住 `knee_peak_rear` 取的 8/11 真的是後腿膝。

    這條線已經因為腿序踩過坑（SHM 是 fl,fr,bl,br、設定檔是 FR,FL,RR,RL）。
    後膝峰值是掃參數的硬門檻，索引取錯會讓門檻默默守在別的關節上。
    """
    import mujoco
    m = mm.make_model()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(i))
             for i in mm.leg_joint_ids(m)]
    for i in (2, 5):
        assert "KNEE" in names[i] and names[i].startswith("F"), names[i]
    for i in (8, 11):
        assert "KNEE" in names[i] and names[i].startswith("R"), names[i]


# =============================================================================
# body sway（COG adjustment）
# =============================================================================
def test_sway_none_is_bitwise_identical():
    """★ 不給 sway 時必須與改動前逐位元相同 —— 所有既有結果的回歸護欄。"""
    f0 = leg_kin.home_foot(mm.HOME)
    c = _state(np.full(4, 1.2))
    a = cpg_max.foot_targets(c, f0, -0.02, 0.08, 0.10, 0.12, 0.8, mm.STATIC_SAG)
    b = cpg_max.foot_targets(c, f0, -0.02, 0.08, 0.10, 0.12, 0.8, mm.STATIC_SAG,
                             sway=None)
    z = cpg_max.foot_targets(c, f0, -0.02, 0.08, 0.10, 0.12, 0.8, mm.STATIC_SAG,
                             sway=(0.0, 0.0))
    assert np.array_equal(a, b) and np.array_equal(a, z)


def test_sway_shifts_all_four_legs_together():
    """sway 是**四腿共同**的偏移（＝機身相對地面移動），不是逐腿的。"""
    f0 = leg_kin.home_foot(mm.HOME)
    c = _state(np.array([0.3, 1.9, 3.1, 5.0]))      # 四腿不同相位
    base = cpg_max.foot_targets(c, f0, 0.0, 0.08, 0.10, 0.12, 0.8)
    got = cpg_max.foot_targets(c, f0, 0.0, 0.08, 0.10, 0.12, 0.8,
                               sway=(0.01, -0.03))
    assert np.allclose(got[:, 0] - base[:, 0], 0.01)
    assert np.allclose(got[:, 1] - base[:, 1], -0.03)
    assert np.allclose(got[:, 2], base[:, 2])       # 不碰 z


def test_gait_phase_is_zero_when_reference_leg_starts_swinging():
    """τ=0 必須正好是 `phase=0` 那條腿開始擺動的時刻。"""
    for ph in (cpg_max.PHASE_WALK, cpg_max.PHASE_WALK_LS):
        assert cpg_max.gait_phase(np.asarray(ph), ph) == pytest.approx(0.0, abs=1e-12)
        # 整體推進 1/4 圈 → τ = 0.25
        assert cpg_max.gait_phase(np.asarray(ph) + np.pi / 2, ph) == \
            pytest.approx(0.25, abs=1e-9)


def test_body_sway_phase_alignment_on_ls():
    """★ 釘住相位對齊：質心必須在「該腿擺動時」移到對角那一側。

    LS 序列下 τ=0.125 是 RL（左後）擺動的中點 → 質心要往**右前**
    → 足端 dx<0（機身前移）、dy>0（機身右移）。
    τ=0.375 是 FL（左前）擺動中點 → 質心往**右後** → dx>0、dy>0。
    對齊寫反的話 sway 會反而把質心推出支撐三角形，而所有診斷指標仍是乾淨的。
    """
    for tau, want_dx, want_dy, tag in ((0.125, -1, +1, "RL 左後擺動→質心右前"),
                                       (0.375, +1, +1, "FL 左前擺動→質心右後"),
                                       (0.625, -1, -1, "RR 右後擺動→質心左前"),
                                       (0.875, +1, -1, "FR 右前擺動→質心左後")):
        dx, dy = cpg_max.body_sway(tau, 0.03, 0.05)
        assert np.sign(dx) == want_dx, f"{tag}：dx 方向錯"
        assert np.sign(dy) == want_dy, f"{tag}：dy 方向錯"


def test_body_sway_lead_is_independent_per_axis():
    """★ 縱向與橫向的相位提前量必須各自獨立。

    縱向是二倍頻，綁成同一個 lead 會讓其中一軸永遠對不準 —— 實測就是靠這個
    才看清「sway_x 有害」是綁在一起造成的假象（lead=0 時執行率 0.56→1.20，
    lead=0.20 時掉到 0.09）。
    """
    tau = 0.3
    only_x = cpg_max.body_sway(tau, 0.02, 0.0, lead_x=0.13, lead_y=0.0)
    only_x2 = cpg_max.body_sway(tau, 0.02, 0.0, lead_x=0.13, lead_y=0.4)
    assert np.array_equal(only_x, only_x2), "lead_y 影響到了 x 軸"
    only_y = cpg_max.body_sway(tau, 0.0, 0.03, lead_x=0.0, lead_y=0.13)
    only_y2 = cpg_max.body_sway(tau, 0.0, 0.03, lead_x=0.4, lead_y=0.13)
    assert np.array_equal(only_y, only_y2), "lead_x 影響到了 y 軸"
    # lead 真的在移相位：x 軸提前半個「縱向週期」(0.25 步態週期) 應該反號
    a = cpg_max.body_sway(tau, 0.02, 0.0)
    b = cpg_max.body_sway(tau, 0.02, 0.0, lead_x=0.25)
    assert a[0] == pytest.approx(-b[0], abs=1e-12)


def test_body_sway_is_periodic_and_zero_mean():
    """sway 一圈的平均必須是 0 —— 否則它就變成一個常數偏移（那是 x_off 的工作）。"""
    tau = np.linspace(0, 1, 2001)[:-1]
    v = np.array([cpg_max.body_sway(t, 0.03, 0.05) for t in tau])
    assert np.allclose(v.mean(0), 0.0, atol=1e-9)
    # 縱向一圈兩次、橫向一圈一次
    assert np.sum(np.diff(np.sign(v[:, 0])) != 0) == 4
    assert np.sum(np.diff(np.sign(v[:, 1])) != 0) == 2


def test_y_off_widens_stance_symmetrically():
    """★ 站姿寬度 `y_off`：外展為正、左右鏡像，且不動 x/z。

    文獻上四足 trot 的兩個關鍵足端軌跡參數是「高度與寬度」，而寬度在這份實作裡
    一直是寫死的 212 mm。這個測試釘住它的語意，免得日後被當成「左右平移」用。
    """
    f0 = leg_kin.home_foot(mm.HOME)
    for yo in (0.03, 0.06, 0.12):
        f = f0 + np.stack([np.zeros(4), mm.SIDE_Y * yo, np.zeros(4)], -1)
        assert abs(f[0, 1] - f[1, 1]) == pytest.approx(
            abs(f0[0, 1] - f0[1, 1]) + 2 * yo, abs=1e-12), "左右足距沒有加寬 2*y_off"
        assert np.allclose(f[:, 0], f0[:, 0]) and np.allclose(f[:, 2], f0[:, 2])
        # 左右鏡像：y 的絕對值相等、符號相反
        assert f[0, 1] == pytest.approx(-f[1, 1], abs=1e-12)
        assert f[2, 1] == pytest.approx(-f[3, 1], abs=1e-12)
