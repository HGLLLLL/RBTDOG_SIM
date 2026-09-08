"""M6 錄檔載入器：`_fold` 過的純量要展開、缺關節要擋。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import m6_rec  # noqa: E402
import shm_io  # noqa: E402


def make_dict(n=5, fold_kp=True):
    t = [round(0.005 * i, 4) for i in range(n)]
    joints = {}
    for nm in shm_io.JOINTS:
        joints[nm] = {"q": [0.1 * i for i in range(n)], "v": [0.0] * n,
                      "tau": [1.0] * n, "des": [0.2] * n, "ff": 0.0,
                      "kp": 60.0 if fold_kp else [60.0] * n, "kd": 1.0}
    imu = [[0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(n)]
    return {"schema": "m6_record/1", "label": "x", "note": "", "n": n,
            "secs": t[-1], "hz_actual": 200.0, "t": t, "joints": joints, "imu": imu}


def test_unfolds_scalars_to_arrays():
    r = m6_rec.from_dict(make_dict())
    assert r.n == 5
    assert r.j["fl1_hip_roll"]["kp"].shape == (5,)
    assert np.all(r.j["fl1_hip_roll"]["kp"] == 60.0)
    assert r.j["fl1_hip_roll"]["ff"].shape == (5,)
    assert r.imu.shape == (5, 10)
    assert r.quat_raw.shape == (5, 4) and r.gyro.shape == (5, 3) and r.acc.shape == (5, 3)


def test_rejects_wrong_schema_and_missing_joint():
    d = make_dict()
    d["schema"] = "m6_static/1"
    with pytest.raises(ValueError):
        m6_rec.from_dict(d)
    d = make_dict()
    del d["joints"]["br4_foot"]
    with pytest.raises(ValueError):
        m6_rec.from_dict(d)


def test_load_roundtrip(tmp_path):
    p = tmp_path / "M6_x.json"
    p.write_text(json.dumps(make_dict()), encoding="utf-8")
    r = m6_rec.load(p)
    assert r.path == str(p) and r.hz == pytest.approx(200.0)
    assert r.t[1] == pytest.approx(0.005)
