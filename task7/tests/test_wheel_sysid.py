"""M11 輪子系統辨識：用已知參數的模型合成 log → wheel_sysid 擬合要回到真值附近；M11 腳本的計畫與防呆。"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
import wheel_sysid as ws  # noqa: E402

TRUE = dict(J=0.004, tau_f=0.15, b=0.02, delay_s=0.010, noise=0.05)   # 合理量級：輪 + 轉子慣量、實測摩擦 0.13–0.23
WHEELS = ("fl4_foot", "fr4_foot", "bl4_foot", "br4_foot")


def _simulate(seq, kd, chirp=(1.5, 0.2, 4.0, 20.0), hz=200.0, dt_inner=0.001, seed=0):
    """用 1 kHz 內迴圈跑模型，200 Hz 取樣，寫成 M11 的 json 格式。"""
    rng = np.random.default_rng(seed)
    samples, t = [], 0.0
    v = np.zeros(4); pos = np.zeros(4)
    hist = []                                        # 指令延遲佇列
    for name, v_cmd, tff, secs in seq:
        n_out = int(secs * hz)
        for k in range(n_out):
            ts = k / hz
            if name == "chirp":
                A, f0, f1, T = chirp
                v_des = A * np.sin(2 * np.pi * (f0 * ts + 0.5 * (f1 - f0) / T * ts * ts)); ff = 0.0
            else:
                v_des, ff = v_cmd, (np.sign(v_cmd) * tff if v_cmd else 0.0)
            hist.append((v_des, ff))
            nd = int(TRUE["delay_s"] * hz)
            v_del, ff_del = hist[-1 - nd] if len(hist) > nd else (0.0, 0.0)
            for _ in range(int(1 / hz / dt_inner)):
                tau = kd * (v_del - v) + ff_del - TRUE["tau_f"] * np.tanh(v / 0.02) - TRUE["b"] * v
                v = v + tau / TRUE["J"] * dt_inner
                pos = pos + v * dt_inner
            meas_v = v + rng.normal(0, TRUE["noise"], 4)
            eff = kd * (v_del - meas_v) + ff_del
            rec = {"t": round(t, 4), "seg": name, "v_des": float(v_des), "ff": float(ff), "kd": kd}
            for i, w in enumerate(WHEELS):
                rec[w] = (float(pos[i]), float(meas_v[i]), float(eff[i]))
            samples.append(rec)
            t += 1 / hz
    return {"schema": "m11_wheel_sysid/1", "proto": seq_proto(seq), "kd": kd, "wheels": list(WHEELS),
            "seq": seq, "chirp": list(chirp), "aborted": "", "samples": samples}


def seq_proto(seq):
    names = {s[0] for s in seq}
    if "chirp" in names:
        return "chirp"
    if any(n.startswith("ff") for n in names):
        return "ff"
    if "hold0" in names:
        return "noise"
    return "step"


def _write(tmp_path, d, name):
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_step_fit_recovers_friction_and_inertia(tmp_path):
    import M11_wheel_sysid as m11
    kd = 1.0
    seq = m11.plan("step", (0.5, 1.5, 3.0, 5.0), (), 1.5, (1.5, 0.2, 4.0, 20.0))
    d = _simulate(seq, kd)
    r = ws.analyze(_write(tmp_path, d, "M11_step.json"))
    a = r["avg"]
    assert abs(a["tau_f"] - TRUE["tau_f"]) < 0.03, a
    assert abs(a["b"] - TRUE["b"]) < 0.01, a
    assert abs(a["delay"] - TRUE["delay_s"]) <= 0.006, a            # 200 Hz 解析度 5 ms
    assert a["tau_c"] <= 0.012, a                                     # 真值 3.9 ms < 解析度 → 只能得上限
    assert a["J_tc"] < 0.015, a
    rows = r["per_wheel"]["fl4_foot"]["rows"]
    # 純 kd 伺服穩態誤差 (τ_f + b v)/kd → v_des 3 → 追蹤率 ≈ 1 − (0.15+0.06)/3
    r3 = next(x for x in rows if x["v_des"] == 3.0)
    assert 0.88 < r3["ratio"] < 0.96, r3
    txt = ws.report(r, "M11_step.json")
    assert "四輪平均" in txt and "追蹤率" in txt


def test_ff_fit_matches_step_fit(tmp_path):
    import M11_wheel_sysid as m11
    kd = 0.5
    seq = m11.plan("ff", (), (0.0, 0.1, 0.15, 0.2, 0.3), 1.5, (1.5, 0.2, 4.0, 20.0))
    r = ws.analyze(_write(tmp_path, _simulate(seq, kd), "M11_ff.json"))
    x = r["per_wheel"]["br4_foot"]
    assert abs(x["tau_f"] - TRUE["tau_f"]) < 0.04, x
    assert abs(x["b"] - TRUE["b"]) < 0.02, x
    assert "kd+b" in ws.report(r, "M11_ff.json")


def test_noise_and_chirp(tmp_path):
    import M11_wheel_sysid as m11
    r = ws.analyze(_write(tmp_path, _simulate(m11.plan("noise", (), (), 1.5, (1.5, 0.2, 4.0, 20.0)), 0.5), "M11_noise.json"))
    n = r["per_wheel"]["fl4_foot"]
    assert abs(n["hold0"]["v_std"] - TRUE["noise"]) < 0.015 and abs(n["hold2"]["v_std"] - TRUE["noise"]) < 0.015
    ch = (1.5, 0.2, 4.0, 20.0)
    r = ws.analyze(_write(tmp_path, _simulate(m11.plan("chirp", (), (), 1.5, ch), 1.0, chirp=ch), "M11_chirp.json"))
    c = r["per_wheel"]["fr4_foot"]
    assert len(c["rows"]) > 10
    assert c["rows"][0]["gain"] > 0.8                                   # 低頻要跟得上
    # 一階：fc = (kd+b)/(2πJ) ≈ 40 Hz → 4 Hz 內增益不該掉到 0.707 以下
    assert np.isnan(c["bandwidth_hz"]) or c["bandwidth_hz"] > 3.0, c["bandwidth_hz"]
    assert "頻寬" in ws.report(r, "M11_chirp.json")


def test_m11_plan_and_guards():
    import M11_wheel_sysid as m11
    seq = m11.plan("step", (0.5, 3.0), (), 1.5, (1.5, 0.2, 4.0, 20.0))
    assert [s[0] for s in seq][:5] == ["zero", "+0.5", "zero", "-0.5", "zero"]
    assert all(s[2] == 0.0 for s in seq)
    seq = m11.plan("ff", (), (0.0, 0.15), 1.5, (1.5, 0.2, 4.0, 20.0))
    assert [s[0] for s in seq] == ["zero", "ff0", "zero", "zero", "ff0.15", "zero"]
    assert m11.KD_SAFE_MAX == 1.0
    # kd > 1 沒帶 --allow-kd-high 要被擋（不需要 shm）
    r = subprocess.run([sys.executable, str(ROOT / "realbot" / "M11_wheel_sysid.py"), "--kd", "2.0"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "M10" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "realbot" / "M11_wheel_sysid.py"), "--levels", "7.0"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "vmax" in r.stdout


def test_push_list_has_m11():
    src = (ROOT / "realbot" / "push_to_dog.sh").read_text(encoding="utf-8")
    assert "M11_wheel_sysid.py" in src
