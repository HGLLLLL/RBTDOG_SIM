"""M8（承重下的單腿擺動）的計畫層驗證。

★ M8 的危險不在控制迴圈（那是照抄 M7 已實機驗證的骨架），而在**計畫**：
  移重心的方向寫反 → 重心被推向要抬的那條腿 → 41 kg 往那邊倒。
  所以這裡驗的是「產生出來的路徑點在幾何上說得通」，並拿
  **支撐多邊形的靜態穩定裕度**當外部對照 —— 那是純幾何，不依賴力矩模型
  （而力矩模型在 ABAD 上實測高估 10.8 倍，正好不能拿來當判準）。

⚠️ 純標準函式庫的部分不需要 numpy；幾何對照那幾項需要。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))

import coord        # noqa: E402
import kin          # noqa: E402
import M8_swing as m8   # noqa: E402

# 2026-08-27 M7 實機量到的趴姿（控制器座標系）
Q_LIE = {
    "fl1_hip_roll": -0.014, "fl2_hip_pitch": +0.477, "fl3_knee_pitch": -2.522,
    "fr1_hip_roll": -0.021, "fr2_hip_pitch": +0.360, "fr3_knee_pitch": -2.507,
    "bl1_hip_roll": -0.039, "bl2_hip_pitch": -0.493, "bl3_knee_pitch": +2.595,
    "br1_hip_roll": +0.020, "br2_hip_pitch": -0.394, "br3_knee_pitch": +2.575,
}


class Args:
    def __init__(self, **kw):
        d = dict(stage=1, legs=list(coord.LEGS), kp=250.0, kd=5.0,
                 kp_scan=[250.0, 180.0, 120.0], gc_scan=[0.08, 0.15, 0.22],
                 shift=0.04, shift_x=0.03, shift_y=0.14, ramp=2.0, t1=1.5, t2=1.5,
                 t_shift=1.2, t_lift=1.0, hold=2.0, hold_mid=1.5, hold_shift=1.5,
                 hold_lift=1.5, settle=1.0, kp_ramp=1.0)
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)


def plan(**kw):
    a = Args(**kw)
    return a, m8.build_segments(a, dict(Q_LIE), m8.foot_ref(), m8.knee_signs())


# ════════════════════════════════════════════════════════ 凍結時間預算
# ⚠️ 這個上限的**理由換過一次**：原本是「mc_ctrl 凍結只驗到 38.1 秒」，
#    2026-08-27 `M_freezetest.py` 實測凍 200 秒完全正常 → 那個理由失效。
#    現在管的是**承重時間**（腿吃 41 kg 的發熱沒量過，最長只做過 34.6 秒）。
#    ★ 上限沒有跟著放大到 200，是因為它一直兼著第二個用途 ——
#      放寬到 210 會讓四條腿的 S3（77.6 秒）默默通過，那是 2.2 倍未測領域。
WEIGHT_BEARING_CAP = 60.0


@pytest.mark.parametrize("stage,legs,secs", [(0, ["fr"], 15.0), (1, ["fr"], 26.0),
                                             (2, ["fr"], 34.6)])
def test_each_stage_fits_the_weight_bearing_budget(stage, legs, secs):
    a, segs = plan(stage=stage, legs=legs)
    total = sum(s[1] for s in segs)
    assert total == pytest.approx(secs, abs=0.1)
    assert total < WEIGHT_BEARING_CAP


def test_m8_default_cap_matches_the_stated_reason():
    """預設值要跟它宣稱的理由一致 —— 不能文件說承重、程式卻寫 mc_ctrl 的數字。"""
    import argparse
    import M8_swing
    src = Path(M8_swing.__file__).read_text(encoding="utf-8")
    assert 'default=60.0, dest="max_freeze"' in src
    assert "這管的是承重時間，不是 mc_ctrl" in src


def test_stage3_two_legs_now_fits_but_four_does_not():
    """★ 上限放寬到 60 之後，S3 可以一次做兩條腿（43.4s），四條仍然擋下（77.6s）。"""
    one = sum(s[1] for s in plan(stage=3, legs=["fr"])[1])
    two = sum(s[1] for s in plan(stage=3, legs=["fr", "bl"])[1])
    four = sum(s[1] for s in plan(stage=3, legs=list(coord.LEGS))[1])
    assert one < WEIGHT_BEARING_CAP
    assert two < WEIGHT_BEARING_CAP
    assert four > WEIGHT_BEARING_CAP


def test_stages_are_independent_not_cumulative():
    """★ 各階不累加。累加版四條腿是 77.6 秒，等於必然超過凍結上限。"""
    names = lambda st: {s[0] for s in plan(stage=st, legs=["fr"])[1]}  # noqa: E731
    assert any(n.startswith("SAG_") for n in names(1))
    assert not any(n.startswith("SAG_") for n in names(2))
    assert not any(n.startswith("SAG_") for n in names(3))
    assert not any(n.startswith("SHIFT") or n.startswith("GO_SHIFT")
                   for n in names(3))


# ════════════════════════════════════════════════════════ 路徑點的合法性
def test_every_waypoint_is_within_mechanical_limits():
    for stage in (0, 1, 2, 3):
        for nm, dur, p0, p1, kp in plan(stage=stage, legs=["fr"])[1]:
            for j in m8.LEGS12:
                assert not coord.check_limit(j, p1[j], 0.03), f"{stage}/{nm}/{j}"


def test_no_waypoint_needs_ik_clamping():
    """IK 縮限是靜默的 —— 被縮限的路徑點會讓動作「變鈍」而查不出原因。"""
    ref, ks = m8.foot_ref(), m8.knee_signs()
    for gc in (0.08, 0.15, 0.22, 0.30):
        for lg in coord.LEGS:
            _, n = m8.pose_from_feet(m8.lift_feet(ref, lg, gc), ks)
            assert n == 0, f"抬 {lg} {gc*1000:.0f}mm 被縮限"
    for dx, dy in ((0.10, 0.0), (-0.10, 0.0), (0.0, 0.16), (0.0, -0.16)):
        _, n = m8.pose_from_feet(m8.shift_feet(ref, dx, dy), ks)
        assert n == 0, f"移 ({dx},{dy}) 被縮限"


def test_lift_only_moves_the_named_leg():
    ref = m8.foot_ref()
    out = m8.lift_feet(ref, "fr", 0.15)
    assert out["fr"][2] == pytest.approx(ref["fr"][2] + 0.15)
    for lg in ("fl", "bl", "br"):
        assert out[lg] == ref[lg]


def test_shift_sign_convention_is_body_not_feet():
    """★ `shift_feet` 的參數是**機身**位移；足端要往反方向走。

    寫反的話「往左移重心」會變成往右移 —— 抬左腿時直接翻。
    """
    ref = m8.foot_ref()
    out = m8.shift_feet(ref, 0.0, +0.10)          # 機身往左（+y）
    for lg in coord.LEGS:
        assert out[lg][1] == pytest.approx(ref[lg][1] - 0.10)


# ════════════════════════════════════════ ★ 移重心的方向（最危險的一項）
def _support_margin(lift_leg, body_dx, body_dy):
    """抬起 `lift_leg` 後，質心到臨界對角線的距離（正 = 在支撐三角形內）。

    純幾何，不依賴力矩模型 —— 這是外部對照量。
    接地點用 `kin` 的正向運動學算（x, y 與輪心相同）。
    """
    ref = m8.foot_ref()
    # 足端相對「機身原點」= ABAD 原點在機身上的位置 + 足端相對 ABAD。
    # 這裡只需要**相對關係**，而四腿的 ABAD 偏移是對稱的，
    # 所以用 ±(半軸距, 半輪距) 的號即可 —— 由 kin.SIDE 給。
    P = {}
    for lg in coord.LEGS:
        sx, sy = kin.SIDE[lg]
        P[lg] = (sx * 0.3398, sy * 0.1710)     # MuJoCo 實算的站姿接地點
    others = [l for l in coord.LEGS if l != lift_leg]
    opp = {"fr": "bl", "fl": "br", "br": "fl", "bl": "fr"}[lift_leg]
    edge = [l for l in others if l != opp]
    ax, ay = P[edge[0]]
    bx, by = P[edge[1]]
    ex, ey = bx - ax, by - ay
    nx, ny = -ey, ex
    n = math.hypot(nx, ny)
    nx, ny = nx / n, ny / n
    if nx * (P[opp][0] - ax) + ny * (P[opp][1] - ay) < 0:
        nx, ny = -nx, -ny
    # 質心（MuJoCo 實算：相對機身原點只有 −0.6 / +1.5 mm，視為 0）+ 機身位移
    return nx * (body_dx - ax) + ny * (body_dy - ay)


def test_no_shift_leaves_the_com_on_the_critical_diagonal():
    """不移重心時裕度 ≈ 0 —— 這是幾何必然，也是 M8 一定要先移重心的理由。"""
    for lg in coord.LEGS:
        assert abs(_support_margin(lg, 0.0, 0.0)) < 0.005


def test_s3_shift_direction_increases_the_stability_margin():
    """★★ M8 對每條腿算出來的位移，必須讓裕度**變大**。

    號寫反的話裕度會變成負的 —— 那就是把 41 kg 往要抬的那條腿推。
    """
    a = Args(stage=3)
    for lg in coord.LEGS:
        sx = -a.shift_x if lg in coord.FRONT_LEGS else +a.shift_x
        sy = -a.shift_y if lg in ("fl", "bl") else +a.shift_y
        before = _support_margin(lg, 0.0, 0.0)
        after = _support_margin(lg, sx, sy)
        assert after > before, f"{lg}: 位移把裕度變小了（{before:.3f} → {after:.3f}）"
        assert after > 0.06, f"{lg}: 裕度只有 {after*1000:.0f} mm，太小"


def test_shift_direction_is_away_from_the_lifted_leg():
    """再用一個獨立說法檢查同一件事：位移的號要與抬起腿的位置相反。"""
    a = Args(stage=3)
    for lg in coord.LEGS:
        sx = -a.shift_x if lg in coord.FRONT_LEGS else +a.shift_x
        sy = -a.shift_y if lg in ("fl", "bl") else +a.shift_y
        fx, fy = kin.SIDE[lg]           # 抬起腿在機身的哪一角
        assert math.copysign(1, sx) == -fx, f"{lg} 的縱向位移方向錯了"
        assert math.copysign(1, sy) == -fy, f"{lg} 的橫向位移方向錯了"


# ════════════════════════════════════════════════════════ S3 的掃描結構
def test_s3_shifts_once_then_sweeps_several_lift_heights():
    """一次移重心、掃多個高度 —— 只量一個點分不出固定偏移與增益誤差。"""
    _, segs = plan(stage=3, legs=["fr"], gc_scan=[0.08, 0.15, 0.22])
    names = [s[0] for s in segs]
    assert names.count("PRE_fr") == 1
    for gc in (80, 150, 220):
        assert f"SWING_fr_{gc}" in names
    # PRE 只有一次，而且在所有 LIFT 之前
    assert names.index("PRE_fr") < min(i for i, n in enumerate(names)
                                       if n.startswith("LIFT_"))


def test_s1_ramps_kp_instead_of_stepping():
    """★ kp 不可以階躍 —— 原廠一步從 0 跳到 250 時 ABAD/HIP 瞬間衝到 20–28 N·m。"""
    _, segs = plan(stage=1, legs=["fr"])
    names = [s[0] for s in segs]
    for kp in (250, 180, 120):
        i = names.index(f"SAG_kp{kp}")
        assert names[i - 1] == f"KPRAMP_{kp}", f"SAG_kp{kp} 前面沒有斜坡"
        assert isinstance(segs[i - 1][4], tuple) and segs[i - 1][4][0] == "ramp"
    assert "KPRAMP_back" in names       # 掃完要爬回原本的 kp


def test_every_segment_starts_where_the_previous_one_ended():
    """路徑不可以有突跳 —— 承重中的突跳就是 kp×誤差 的力矩尖峰。"""
    for stage in (0, 1, 2, 3):
        _, segs = plan(stage=stage, legs=["fr"])
        for (n0, _, _, p1, _), (n1, _, p0, _, _) in zip(segs, segs[1:]):
            for j in m8.LEGS12:
                assert p1[j] == pytest.approx(p0[j], abs=1e-12), f"{n0}→{n1} / {j}"


def test_sequence_returns_to_the_measured_lying_pose():
    """最後要回到實測的趴姿，否則解凍 mc_ctrl 會突跳。"""
    for stage in (0, 1, 2, 3):
        _, segs = plan(stage=stage, legs=["fr"])
        assert segs[-1][0] == "RAMP_DOWN"
        for j in m8.LEGS12:
            assert segs[-1][3][j] == pytest.approx(Q_LIE[j])


def test_default_t_lift_keeps_the_tallest_lift_under_vcmd_max():
    """★ 迴歸：2026-08-27 現場踩到 —— `--t-lift 0.8` 時 220 mm 那格是 2.16 rad/s，
    超過 --vcmd-max 2.0，乾跑直接擋下。預設改成 1.0（1.73，餘裕 14%）。

    ⚠️ 正確的修法是**放長時間**，不是調高門檻 —— 調高門檻等於把守衛關掉。
    """
    import M7_standup as m7
    _, segs = plan(stage=3, legs=["fr"], gc_scan=[0.08, 0.15, 0.22])
    rows = {r[0]: r[3] for r in
            m8.seg_speeds([(n, d, p0, p1) for n, d, p0, p1, _ in segs], m8.LEGS12)}
    assert rows["LIFT_fr_220"] < 2.0, f"220mm 抬腿 {rows['LIFT_fr_220']:.2f} rad/s 超標"
    assert rows["DROP_fr_220"] < 2.0
    # 而 0.8 秒確實會超標 —— 證明這個測試真的在測東西
    a = Args(stage=3, legs=["fr"], t_lift=0.8)
    segs8 = m8.build_segments(a, dict(Q_LIE), m8.foot_ref(), m8.knee_signs())
    r8 = {r[0]: r[3] for r in
          m8.seg_speeds([(n, d, p0, p1) for n, d, p0, p1, _ in segs8], m8.LEGS12)}
    assert r8["LIFT_fr_220"] > 2.0
