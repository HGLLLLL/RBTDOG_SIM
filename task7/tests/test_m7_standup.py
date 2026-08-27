"""M7（承重站立）新增的兩道保護的離線驗證。

★ 這兩道保護都是 2026-08-27 從**實機錄製檔**回推出來的，所以測試也拉實機當外部
  對照量，而不是只驗程式自我一致（task7 的老教訓：出錯的通常是工具本身）：

  | 測什麼 | 外部對照 |
  |---|---|
  | `seg_speeds` 的峰值係數 | 對 smoothstep 直接數值微分 |
  | 「走多遠」門檻抓得到 knee_back | `coord.POSES` + `flip_rear_knee_mode` |
  | 輪子 latch 鎖定的可行性 | `M6_20260826_154737.json` 實機錄製 |
  | ±π 繞回由 driver 解纏 | 同上，kp>0 期間逐筆殘差比對 |

⚠️ 純標準函式庫（不需要 numpy / mujoco）—— 被測的程式要跑在狗上，狗上沒有這些套件。
⚠️ 匯入 `M7_standup` 不會碰 /dev/shm（模組層只有常數與函式定義）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))

import coord                 # noqa: E402
import M5_leg_pose as m5     # noqa: E402
import M7_standup as m7      # noqa: E402

REC = ROOT / "logs" / "m_logs_trip7" / "M6_20260826_154737.json"

# 實機趴姿（2026-08-26 15:47，增益開啟那一刻，控制器座標系）
Q_LIE = {
    "fl1_hip_roll": +0.3889, "fl2_hip_pitch": +1.1923, "fl3_knee_pitch": -2.8009,
    "fr1_hip_roll": -0.2577, "fr2_hip_pitch": +1.1839, "fr3_knee_pitch": -2.8009,
    "bl1_hip_roll": +0.5926, "bl2_hip_pitch": -1.1126, "bl3_knee_pitch": +2.7967,
    "br1_hip_roll": -0.4583, "br2_hip_pitch": -1.1107, "br3_knee_pitch": +2.8009,
}


# ════════════════════════════════════════════════════ seg_speeds 的峰值係數
def test_smoothstep_peak_factor_matches_numeric_derivative():
    """峰值係數不是猜的：對 M5 的 smoothstep 數值微分應該就是它。

    ⚠️ M5 用的是餘弦插值（π/2），不是三次式 smoothstep（1.5）——
    第一版把係數寫成 1.5，就是這個測試抓出來的。
    """
    h = 1e-6
    peak = max((m5.smoothstep(u + h) - m5.smoothstep(u - h)) / (2 * h)
               for u in [i / 2000 for i in range(2001)])
    assert peak == pytest.approx(m7.SMOOTHSTEP_VPEAK, abs=1e-4)
    assert m7.SMOOTHSTEP_VPEAK != 1.5, "餘弦插值不是三次式，係數是 π/2"


def test_seg_speeds_uniform_move():
    """1 rad / 1 s 的區段，峰值速度應該是 π/2 rad/s，不是平均值 1.0。"""
    p0 = {j: 0.0 for j in m7.LEGS12}
    p1 = dict(p0, fl3_knee_pitch=1.0)
    (nm, j, dq, vc, dur), = m7.seg_speeds([("GO", 1.0, p0, p1)])
    assert (nm, j, dur) == ("GO", "fl3_knee_pitch", 1.0)
    assert dq == pytest.approx(1.0)
    assert vc == pytest.approx(math.pi / 2)


def test_seg_speeds_picks_the_farthest_joint():
    p0 = {j: 0.0 for j in m7.LEGS12}
    p1 = dict(p0, fl3_knee_pitch=0.4, br2_hip_pitch=-0.9)
    (_, j, dq, _, _), = m7.seg_speeds([("GO", 1.0, p0, p1)])
    assert j == "br2_hip_pitch" and dq == pytest.approx(0.9)


def test_seg_speeds_hold_segment_is_zero():
    p = {j: 0.3 for j in m7.LEGS12}
    (_, _, dq, vc, _), = m7.seg_speeds([("HOLD", 4.0, p, p)])
    assert dq == 0.0 and vc == 0.0


def test_seg_speeds_zero_duration_does_not_divide_by_zero():
    p0 = {j: 0.0 for j in m7.LEGS12}
    p1 = dict(p0, fl3_knee_pitch=1.0)
    (_, _, _, vc, _), = m7.seg_speeds([("GO", 0.0, p0, p1)])
    assert math.isfinite(vc) and vc > 0


# ════════════════════════════════════════════════ 門檻真的分得出正常 vs knee_back
def _first_leg_seg(start, target, dur=1.5):
    return m7.seg_speeds([("GO_crouch", dur, start, target)])[0]


def test_normal_lie_pose_passes_default_threshold():
    """實機那次的正常趴姿 → crouch，命令速度應該遠低於預設門檻 2.0。"""
    _, _, dq, vc, _ = _first_leg_seg(Q_LIE, coord.POSES["crouch"])
    assert dq < 1.0
    assert vc < 2.0


def test_knee_back_lie_pose_is_rejected_by_default_threshold():
    """★ 後膝反向的趴姿 → crouch，必須被門檻擋下。

    這正是加這道檢查的理由：M7 的 t1 是固定秒數，
    knee_back 的後膝要掃約 5.2 rad，同樣 1.5 秒 → 命令速度衝到 3 倍門檻以上，
    而且會撞到運轉中的 --vmax（預設 4.0）。
    """
    lie_back = coord.flip_rear_knee_mode(Q_LIE)
    _, j, dq, vc, _ = _first_leg_seg(lie_back, coord.POSES["crouch"])
    assert j.endswith("3_knee_pitch") and j[:2] in coord.REAR_LEGS
    assert dq > 5.0
    assert vc > 2.0          # 超過 --vcmd-max 預設 → 拒跑
    assert vc > 4.0          # 也超過運轉中的 --vmax 預設 → 一定會中途中止


def test_lengthening_t1_is_a_valid_escape_hatch():
    """訊息裡建議的『把 t1 放長』要真的能過關 —— 算出來的秒數不能是空話。"""
    lie_back = coord.flip_rear_knee_mode(Q_LIE)
    _, _, dq, _, _ = _first_leg_seg(lie_back, coord.POSES["crouch"])
    need = m7.SMOOTHSTEP_VPEAK * dq / 2.0
    _, _, _, vc, _ = _first_leg_seg(lie_back, coord.POSES["crouch"], dur=need)
    assert vc == pytest.approx(2.0, rel=1e-9)


def test_knee_sign_rule_stated_in_the_abort_message_is_true():
    """中止訊息叫人看『bl3/br3 是否與 fl3/fr3 反號』—— 這條規則本身要成立。"""
    for pose, same_sign in ((Q_LIE, False), (coord.flip_rear_knee_mode(Q_LIE), True)):
        front = [pose[lg + coord.KIND_KNEE] for lg in coord.FRONT_LEGS]
        rear = [pose[lg + coord.KIND_KNEE] for lg in coord.REAR_LEGS]
        pairs = [math.copysign(1, f) == math.copysign(1, r)
                 for f in front for r in rear]
        assert all(pairs) is same_sign


# ════════════════════════════════════════════════════════════ 輪子鎖定的前提
@pytest.fixture(scope="module")
def rec():
    if not REC.exists():
        pytest.skip(f"缺實機錄製檔 {REC}")
    return json.loads(REC.read_text(encoding="utf-8"))


def _series(rec, name, key):
    v = rec["joints"][name][key]
    return v if isinstance(v, list) else [v] * len(rec["t"])


def test_factory_gains_used_as_defaults_match_the_recording(rec):
    """M7 的輪鎖定預設值（kp=20 / kd=0.1）要真的是原廠站穩後用的那組。"""
    kp = _series(rec, "fl4_foot", "kp")
    kd = _series(rec, "fl4_foot", "kd")
    assert m7.WHEEL_KP_HOLD in set(kp), f"錄製檔裡的輪 kp 只有 {sorted(set(kp))}"
    assert m7.WHEEL_KD_HOLD in set(kd), f"錄製檔裡的輪 kd 只有 {sorted(set(kd))}"
    # 站穩後（最後一秒）就是這一組
    assert kp[-1] == pytest.approx(m7.WHEEL_KP_HOLD)
    assert kd[-1] == pytest.approx(m7.WHEEL_KD_HOLD)


def test_factory_relocks_wheels_by_latching_current_angle(rec):
    """★ M7 的 latch 做法（des = 鎖定當下的實測角）要與原廠一致。

    若原廠是鎖回某個絕對位置，我們照抄 latch 就是錯的。
    """
    t = [x - rec["t"][0] for x in rec["t"]]
    on = next(i for i, x in enumerate(t) if x >= 13.05)
    for w in ("fl4_foot", "fr4_foot", "bl4_foot", "br4_foot"):
        des = _series(rec, w, "des")[on:]
        assert max(des) - min(des) < 1e-9, f"{w} 鎖定後 des 不是常數"
        # 鎖定值就落在鎖定前後的實測角附近，不是某個絕對位置
        q = _series(rec, w, "q")
        assert abs(des[0] - q[on]) < 0.15, f"{w} 鎖定值離當下實測角太遠"


def test_locked_wheels_actually_hold(rec):
    """鎖定後輪子要真的停住 —— 否則 kp=20 這個值就不夠。"""
    t = [x - rec["t"][0] for x in rec["t"]]
    on = next(i for i, x in enumerate(t) if x >= 13.05)
    for w in ("fl4_foot", "fr4_foot", "bl4_foot", "br4_foot"):
        q = _series(rec, w, "q")[on:]
        # 鎖定後不會跨越 ±π，直接看首尾差即可
        assert abs(q[-1] - q[0]) < 0.05, f"{w} 鎖定後仍滾了 {q[-1]-q[0]:+.3f} rad"


def test_locked_wheel_torque_stays_under_the_guard(rec):
    """--wheel-tau-max 預設 8.0 要留有餘裕，不能一鎖就誤中止。"""
    t = [x - rec["t"][0] for x in rec["t"]]
    on = next(i for i, x in enumerate(t) if x >= 13.05)
    worst = max(abs(x) for w in ("fl4_foot", "fr4_foot", "bl4_foot", "br4_foot")
                for x in _series(rec, w, "tau")[on:])
    assert worst < 8.0 / 2, f"實機鎖定後峰值 {worst:.2f}，離門檻 8.0 太近"


def test_driver_unwraps_wheel_error_so_latching_is_safe(rec):
    """★★ 安全關鍵：寫「當下實測角」當 des，輪子滾過 ±π 會不會爆出 121 N·m？

    答案是不會 —— driver 自己解纏。這是 M7 敢鎖輪的唯一理由，
    所以這裡逐筆驗，而不是引用結論。
    """
    n_wrapped = 0
    for w in ("fl4_foot", "fr4_foot", "bl4_foot", "br4_foot"):
        kp, kd, des, q, v, tau, ff = (_series(rec, w, k) for k in
                                      ("kp", "kd", "des", "q", "v", "tau", "ff"))
        res_raw, res_wrap = [], []
        for i in range(len(kp)):
            if kp[i] <= 0:
                continue
            raw = des[i] - q[i]
            wrap = (raw + math.pi) % (2 * math.pi) - math.pi
            if abs(raw - wrap) > 1e-9:
                n_wrapped += 1
            res_raw.append(abs(kp[i] * raw - kd[i] * v[i] + ff[i] - tau[i]))
            res_wrap.append(abs(kp[i] * wrap - kd[i] * v[i] + ff[i] - tau[i]))
        rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))  # noqa: E731
        assert rms(res_wrap) < 2.0, f"{w}: 解纏誤差也對不上，控制律假設有問題"
        assert rms(res_wrap) < rms(res_raw) or rms(res_raw) < 2.0, (
            f"{w}: 原始誤差比解纏誤差更吻合 —— driver 可能沒有解纏，不可鎖輪")
    assert n_wrapped > 100, "錄製檔裡沒有足夠的 ±π 繞回取樣，這個結論站不住"


# ══════════════════════════════════ 起點比路徑點還高（吊帶撐著）的判別
def _knees_less_folded_than(pose, target):
    """M7 用的判別式：膝越彎（|角度|越大）機身越低。"""
    return [j for j in m7.LEGS12
            if j.endswith(coord.KIND_KNEE) and abs(pose[j]) < abs(target[j]) - 0.02]


def test_real_lying_pose_is_not_flagged_as_high():
    """真正趴在地上（膝頂 ±2.80）比 crouch 的 ±2.40 更彎 → 不該被擋。"""
    assert _knees_less_folded_than(Q_LIE, coord.POSES["crouch"]) == []


def test_the_2026_08_27_t1_start_pose_is_flagged():
    """★ 迴歸測試：2026-08-27 T1 實際發生的情況要被抓出來。

    吊帶掛在約 314 mm（crouch 是 292 mm），起始膝只有 ∓2.10 —— 比 crouch 的
    ∓2.40 還直。那一趟 HOLD_crouch 的膝力矩只有 3.67 N·m，原廠是 27.3–29.8，
    等於完全沒測到承重。這個檢查就是為了在**跑之前**攔下來。
    """
    real = dict(Q_LIE)
    real.update({"fl3_knee_pitch": -2.1375, "fr3_knee_pitch": -2.1017,
                 "bl3_knee_pitch": +2.1169, "br3_knee_pitch": +2.0895})
    flagged = _knees_less_folded_than(real, coord.POSES["crouch"])
    assert len(flagged) == 4, f"四條腿都該被抓到，實際 {flagged}"


def test_borderline_within_tolerance_is_not_flagged():
    """剛好等於路徑點（±0.02 容差內）不該擋 —— 否則從 crouch 重跑會被自己卡住。"""
    at_crouch = dict(Q_LIE)
    for j in m7.LEGS12:
        if j.endswith(coord.KIND_KNEE):
            at_crouch[j] = coord.POSES["crouch"][j] * 0.995
    assert _knees_less_folded_than(at_crouch, coord.POSES["crouch"]) == []


def test_flag_survives_the_real_recording_lying_pose(rec):
    """外部對照：原廠錄製檔裡增益開啟那一刻的膝角，也不該被誤擋。"""
    kp = _series(rec, "fl3_knee_pitch", "kp")
    on = next(i for i, v in enumerate(kp) if v > 0)
    pose = {}
    for j in m7.LEGS12:
        pose[j] = coord.to_ctrl(j, _series(rec, j, "q")[on])
    assert _knees_less_folded_than(pose, coord.POSES["crouch"]) == []
