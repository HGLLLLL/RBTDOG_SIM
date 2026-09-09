"""原廠參考資料集（trip21）：抽取結果的結構與關鍵數字要在合理範圍（防止改壞抽取邏輯）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
LOGDIR = ROOT / "logs" / "m_logs_trip21"
DS = ROOT / "outputs" / "ref_gait_dataset.json"


@pytest.mark.skipif(not DS.exists(), reason="資料集未產生")
def test_dataset_structure_and_key_numbers():
    d = json.loads(DS.read_text(encoding="utf-8"))
    S = d["summary"]
    assert len(S["wheel_stance_q12"]) == 12
    assert 0.2 < S["L_eff_m"] < 0.6
    assert 0.1 < S["v_body_range"][0] < S["v_body_range"][1] < 1.2
    for k in ("turn_left", "turn_right", "lat_left", "lat_right"):
        s = S[k]
        assert 1.8 < s["freq_hz"] < 3.2, (k, s["freq_hz"])
        assert 0.015 < s["apex_m"] < 0.04, (k, s["apex_m"])
        assert 0.7 < s["duty"] < 0.9, (k, s["duty"])
        assert len(s["stepping_legs"]) >= 2
    # 旋轉：對角腿踏步、另一對輪反向；左轉偏航為正
    assert set(S["turn_left"]["stepping_legs"]) >= {"fl", "br"}
    assert S["turn_left"]["yaw_rate_deg_s"] > 40 and S["turn_right"]["yaw_rate_deg_s"] < -40
    wp = S["turn_left"]["wheel_pattern"]
    assert wp["fr"] > 0.8 and wp["bl"] < -0.8 and abs(wp["fl"]) < 0.3 and abs(wp["br"]) < 0.3
    # 平移：同側腿踏步、同側輪反向
    assert set(S["lat_right"]["stepping_legs"]) >= {"fr", "br"}
    wp = S["lat_right"]["wheel_pattern"]
    assert wp["fr"] > 0.8 and wp["br"] < -0.4
    # 過渡時間
    for k, v in d["transitions"].items():
        assert 0.3 < v["gap_s"] < 3.0, (k, v)


@pytest.mark.skipif(not LOGDIR.exists(), reason="trip21 log 不在")
def test_swing_events_require_knee_unload():
    import ref_extract as rx
    z = np.zeros(2000)
    z[500:600] = 0.03                      # 抬 30 mm
    tau = np.full(2000, 10.0)
    assert rx.swing_events(z, tau, 0, 2000) == []          # 膝沒卸載 → 不算
    tau[520:580] = 1.0
    assert rx.swing_events(z, tau, 0, 2000) == [(500, 600)]
