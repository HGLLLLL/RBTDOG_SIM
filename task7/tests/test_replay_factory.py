"""原廠回放：模型增益要等於原廠動作段那組、輪致動器要關掉；有錄檔時 smoke 跑 0.2 s。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "inference" / "diag"))
import replay_factory as rf  # noqa: E402
import max_model as mm  # noqa: E402

LOG = ROOT / "logs" / "m_logs_trip21" / "M6_20260909_161316.json"   # ref_turn_left


def test_make_model_has_factory_motion_gains_and_free_wheels():
    m = rf.make_model()
    kp = m.actuator_gainprm[mm.LEG_ACT_IDX, 0]
    assert np.allclose(kp, np.tile([60.0, 120.0, 120.0], 4))
    assert np.allclose(-m.actuator_biasprm[mm.LEG_ACT_IDX, 2], 1.0)
    assert np.all(m.actuator_gainprm[mm.WHEEL_ACT_IDX] == 0.0) and np.all(m.actuator_biasprm[mm.WHEEL_ACT_IDX] == 0.0)


@pytest.mark.skipif(not LOG.exists(), reason="trip21 錄檔不在")
def test_replay_smoke():
    rec, q, des, tau_w, v_w, lift = rf.load_ctrl_frame(str(LOG))
    assert q.shape == (rec.n, 12) and tau_w.shape == (rec.n, 4) and lift.shape == (rec.n, 4)
    assert abs(rec.hz - 500.0) < 25.0
    m = rf.make_model()
    i0 = int(rec.n * 0.4)
    r = rf.replay(m, q, des, tau_w, i0, 100)
    assert r["yaw_rad_s"].shape == (100,) and r["clr"].shape == (100, 4)
    assert np.all(np.isfinite(r["clr"])) and not r["fell"]
