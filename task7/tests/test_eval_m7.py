"""`inference/eval_m7.py`（T1→T2 的放行判定）的離線驗證。

★ 這支程式的功能就是**擋下不該往下做的情況**，所以測試的重點不是「好的會過」，
  而是**每一種壞的都真的被擋下來**，而且擋的理由是對的那一個
  （擋對事情很重要 —— 用錯理由擋下來，現場會去修錯的東西）。

⚠️ 純標準函式庫。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))

import coord  # noqa: E402

EVAL = ROOT / "inference" / "eval_m7.py"
LEGS12 = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]

# 原廠實測的 HOLD_crouch 穩態（|τ| N·m）—— 「好」的基準
GOOD_KNEE = 28.5
GOOD_PEAK = {"1_hip_roll": 28.0, "2_hip_pitch": 22.0, "3_knee_pitch": 35.0}


def make(tmp_path, *, knee_tau=GOOD_KNEE, aborted=False, reason=None,
         knee_back=False, peak_scale=1.0, err=0.02, to="crouch", name="M7_x.json"):
    q_lie, peak, row = {}, {}, {"phase": f"HOLD_{to}"}
    for j in LEGS12:
        k, leg = j[2:], j[:2]
        front = leg in ("fl", "fr")
        if k == "3_knee_pitch":
            s = 1 if (front or knee_back) else -1
            q_lie[j] = -2.80 * s
            tau = knee_tau
        elif k == "2_hip_pitch":
            q_lie[j] = 1.18 * (1 if front else -1)
            tau = 16.0
        else:
            q_lie[j] = 0.3
            tau = 2.0
        peak[j] = GOOD_PEAK[k] * peak_scale
        row[j] = [0.0, err, tau, 0.0]
    d = {"schema": "m7_standup/1", "time": "2026-08-27 10:00:00",
         "args": {"to": to, "kp": 250.0, "kd": 5.0, "wheel_lock": True},
         "aborted": aborted, "abort_reason": reason, "q_lie": q_lie,
         "peak": peak, "recent": [], "hold_samples": [dict(row) for _ in range(40)]}
    p = tmp_path / name
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def run(path):
    r = subprocess.run([sys.executable, str(EVAL), str(path)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_clean_run_is_allowed_through(tmp_path):
    rc, out = run(make(tmp_path))
    assert rc == 0, out
    assert "可以做下一步" in out
    assert "--to stand" in out          # 要明確講出下一步的指令


def test_abort_blocks(tmp_path):
    rc, out = run(make(tmp_path, aborted=True,
                       reason="br3_knee_pitch 力矩連續 3 筆超過 65（-70.2）"))
    assert rc == 1
    assert "中止" in out


def test_harness_bearing_load_blocks(tmp_path):
    """★ 這是本支程式存在的主要理由：吊帶偷偷承重，看即時列印完全看不出來。

    所有即時欄位（誤差、力矩、傾角）都會漂亮 —— 因為腿確實輕鬆。
    只有拿去和**原廠實測的 27.3–29.8** 比才看得出腿根本沒吃到載重。
    """
    rc, out = run(make(tmp_path, knee_tau=12.0))
    assert rc == 1
    assert "吊帶在幫忙承重" in out


def test_slightly_low_knee_warns_but_does_not_block(tmp_path):
    """略低於原廠只該警告 —— 門檻訂太嚴會讓現場為了過關而亂調吊帶。"""
    rc, out = run(make(tmp_path, knee_tau=25.0))     # 27.3 的 0.92 倍
    assert rc == 0
    assert "略低於原廠" in out


def test_knee_back_start_pose_blocks(tmp_path):
    rc, out = run(make(tmp_path, knee_back=True))
    assert rc == 1
    assert "knee_back" in out


def test_peak_over_budget_blocks(tmp_path):
    rc, out = run(make(tmp_path, peak_scale=1.6))
    assert rc == 1
    assert "峰值用掉" in out


def test_peak_just_under_budget_passes(tmp_path):
    """80% 是門檻本身 —— 剛好在下面要放行，否則門檻等於 79%。"""
    # 膝 35.0×scale / 65 < 0.80  → scale < 1.485；ABAD 28.0×scale / 45 < 0.80 → scale < 1.285
    rc, out = run(make(tmp_path, peak_scale=1.25))
    assert rc == 0, out


def test_tracking_error_blocks(tmp_path):
    rc, out = run(make(tmp_path, err=0.25))
    assert rc == 1
    assert "追蹤誤差" in out


def test_stand_run_uses_stand_reference_not_crouch(tmp_path):
    """★ HOLD_stand 的原廠穩態是 8.3–10.3，不是 crouch 的 27.3–29.8。

    若參考值取錯，站立那趟會被誤判成「吊帶在承重」—— 正好相反的結論。
    """
    rc, out = run(make(tmp_path, to="stand", knee_tau=9.5))
    assert rc == 0, out
    assert "8.3" in out and "腿確實在承重" in out


def test_wrong_schema_is_rejected(tmp_path):
    p = tmp_path / "M7_bad.json"
    p.write_text(json.dumps({"schema": "m5_leg_pose/1"}), encoding="utf-8")
    rc, out = run(p)
    assert rc != 0
    assert "不是 M7 的結果檔" in out


def test_thresholds_match_m7_source():
    """★ TMAX 在 eval 與 M7 各有一份副本 —— 不一致的話現場會用到錯的門檻。"""
    import M7_standup as m7
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_m7", EVAL)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    assert ev.TMAX == m7.TMAX, f"eval_m7 {ev.TMAX} != M7_standup {m7.TMAX}"


# ══════════════════════════════════════ 感測尖峰的排除（2026-08-27 T1 實際踩到）
def make_with_spike(tmp_path, *, spike=-51.49, err=0.0871, v=-0.037):
    """在 fr3 的 peak 塞一筆感測尖峰，並讓它出現在 hold_samples 裡可核對。"""
    p = make(tmp_path, err=err)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["peak"]["fr3_knee_pitch"] = spike
    # HOLD 的最後一筆帶著那個尖峰（q 沒動、速度近零 → cap 遠小於 |τ|）
    d["hold_samples"][-1]["fr3_knee_pitch"] = [-2.4843, -2.4843 + err, spike, v]
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def test_sensor_spike_is_excluded_not_blocked(tmp_path):
    """★ 迴歸：2026-08-27 T1 第二趟的 fr3 −51.49。

    那一筆是 79% 門檻，差一點就擋下一趟好的測試。判別依據是
    `kp·|err|+kd·|v|` = 21.26，而 |τ| = 51.49（2.42 倍）—— 控制律給不出來。
    """
    rc, out = run(make_with_spike(tmp_path))
    assert rc == 0, out
    assert "感測尖峰" in out
    assert "已排除" in out


def test_a_genuine_high_peak_still_blocks(tmp_path):
    """★ 反向：力矩大但**與控制律一致**（誤差也大）→ 那是真的，必須照擋。

    否則這個排除機制會變成「只要力矩大就當成雜訊」，比不檢查更危險。
    """
    # |τ|=60、誤差 0.30 → cap = 250×0.30 = 75 > 60 → 判定 real
    rc, out = run(make_with_spike(tmp_path, spike=-60.0, err=0.30, v=0.0))
    assert rc == 1, out
    assert "峰值用掉" in out


def test_spike_ratio_boundary(tmp_path):
    """剛好在 1.5 倍以內算真的 —— 門檻本身要可預測。

    ⚠️ err 用 0.09 不是 0.10：`-2.4843 + 0.10` 的浮點結果是 0.10000000000000009，
       會踩到 ERR_MAX=0.10 的追蹤誤差門檻，變成「擋下來但理由是別的」。
    """
    # cap = 250×0.09 = 22.5，|τ| = 33.0 → 1.47 倍 → real（33/65 = 51%，不擋）
    rc, out = run(make_with_spike(tmp_path, spike=-33.0, err=0.09, v=0.0))
    assert rc == 0, out
    # ⚠️「扣掉感測尖峰後的峰值」是固定會印的一行，不能拿「感測尖峰」四個字當標記
    assert "已排除" not in out
    assert "扣掉感測尖峰後的峰值 35.00" in out    # 35 是真的 peak，尖峰的 33 沒被算進去也沒被排除


def test_unverifiable_peak_is_labelled_and_still_counted(tmp_path):
    """峰值若落在 GO/BACK（沒取樣）→ 無法核對，**不可以當成尖峰放行**。"""
    p = make(tmp_path, peak_scale=1.6)      # 高峰值，但不在 hold_samples 裡
    rc, out = run(p)
    assert rc == 1
    assert "無法核對真偽" in out


def test_real_2026_08_27_t1_run_passes(tmp_path):
    """拿當天真正的檔案跑一次 —— 這是最終的外部對照。"""
    real = ROOT / "logs" / "m_logs_trip8" / "M7_20260827_101642.json"
    if not real.exists():
        pytest.skip("缺實機檔案")
    rc, out = run(real)
    assert "感測尖峰" in out and "fr3_knee_pitch" in out
    assert rc == 0, out
