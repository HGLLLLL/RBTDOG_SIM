"""v3.1 擺動事件法（滾動最低點基線、時長 0.04–0.35 s）與相位計算的合成訊號測試。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
pytest.importorskip("scipy")
import ref_extract as rx  # noqa: E402

HZ = 500.0


def _foot_z(period_s, swing_s, apex, n_steps, drift=0.0):
    """站姿 z=-0.44，每週期一次 swing_s 的正弦抬腳；drift 是慢的姿態漂移（m/s），事件法要能忽略它。"""
    n = int(n_steps * period_s * HZ) + int(HZ)
    t = np.arange(n) / HZ
    z = -0.44 + drift * t
    for k in range(n_steps):
        t0 = 0.5 + k * period_s
        m = (t >= t0) & (t < t0 + swing_s)
        z[m] += apex * np.sin(np.pi * (t[m] - t0) / swing_s)
    return z


def test_swing_events31_counts_period_and_apex_with_drift():
    z = _foot_z(period_s=0.48, swing_s=0.10, apex=0.023, n_steps=10, drift=0.004)   # 漂移 4 mm/s（實錄約 3）
    ev = rx.swing_events31(z)
    assert len(ev) == 10
    starts = np.array([s for s, _, _ in ev]) / HZ
    assert abs(np.median(np.diff(starts)) - 0.48) < 0.01
    assert all(0.018 < a < 0.026 for _, _, a in ev)


def test_swing_events31_rejects_long_posture_changes():
    z = _foot_z(period_s=1.0, swing_s=0.8, apex=0.03, n_steps=3)   # 0.8 s 的「抬」是姿態變化，不是步
    assert rx.swing_events31(z) == []


def test_phases_relative_to_ref():
    ref = np.array([100, 340, 580, 820])                            # 週期 240 tick
    other = np.array([148, 388, 628])                               # 落後 0.2 週期
    assert abs(rx.phase_rel(other, ref, 240) - 0.2) < 0.01
    assert abs(rx.phase_rel(ref + 230, ref, 240) - (230 / 240)) < 0.01
