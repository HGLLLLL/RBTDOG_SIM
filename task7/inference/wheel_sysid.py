"""M11 輪子系統辨識的本機擬合：M11_*.json → 模擬器要用的輪子參數（每輪＋平均）。

模型（每顆輪，速度伺服）：
    J·dv/dt = kd·(v_des − v) + τ_ff − τ_f·sign(v) − b·v
辨識：
  - step：穩態 v_ss、τ_ss → 追蹤率；(|τ_ss|, |v_ss|) 對多個速度做最小平方 → τ_f（截距）、b（斜率）；
          延遲（速度首次離開雜訊帶）、10–90% 上升時間、τ63；τ_c = τ63 − 延遲 → J = τ_c·(kd + b)（解析度 5 ms）
  - ff：固定 v_des 掃 τ_ff → v_ss 對 τ_ff 線性，斜率 1/(kd+b)、截距 → τ_f（獨立驗證）
  - noise：靜止與定速的 v、τ 標準差
  - chirp：1 s 滑窗正弦擬合（頻率已知）→ 增益比與相位落後 vs 頻率；頻寬（−3 dB）、純延遲估計

    conda run -n rbtdog python task7/inference/wheel_sysid.py task7/logs/m_logs_tripNN/M11_*.json --out task7/outputs/wheel_sysid.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WHEELS = ("fl4_foot", "fr4_foot", "bl4_foot", "br4_foot")


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _series(d: dict, w: str):
    S = d["samples"]
    t = np.array([s["t"] for s in S])
    seg = np.array([s["seg"] for s in S])
    vd = np.array([s["v_des"] for s in S])
    ff = np.array([s["ff"] for s in S])
    v = np.array([s[w][1] for s in S])
    e = np.array([s[w][2] for s in S])
    return t, seg, vd, ff, v, e


def _segments(seg: np.ndarray):
    """連續同名段 → [(name, i0, i1)]。"""
    out, i0 = [], 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[i0]:
            out.append((seg[i0], i0, i))
            i0 = i
    return out


def analyze_step(d: dict, w: str) -> dict:
    t, seg, vd, ff, v, e = _series(d, w)
    kd = float(d["kd"])
    rows = []
    for name, i0, i1 in _segments(seg):
        if name == "zero" or i1 - i0 < 20:
            continue
        tt, vv, ee, vdes = t[i0:i1] - t[i0], v[i0:i1], e[i0:i1], float(vd[i0])
        if abs(vdes) < 1e-9:
            continue
        ss = tt >= tt[-1] - 0.5
        v_ss, e_ss = float(vv[ss].mean()), float(ee[ss].mean())
        v_std = float(vv[ss].std())
        sgn = np.sign(vdes)
        vn = vv * sgn                                   # 正規化成正向
        # 10–90% 上升時間、63% 時間常數（以 v_ss 為終值）
        target = v_ss * sgn
        def t_at(frac):
            idx = np.nonzero(vn >= frac * target)[0]
            return float(tt[idx[0]]) if idx.size and target > 1e-6 else float("nan")
        t10, t90, t63 = t_at(0.1), t_at(0.9), t_at(0.632)
        # 延遲：速度首次離開雜訊帶（3σ 或 5% 終值取大）—— 200 Hz 取樣，解析度 5 ms
        thr = max(3 * v_std, 0.05 * target)
        idx = np.nonzero(vn > thr)[0]
        delay = float(tt[idx[0]]) if idx.size else float("nan")
        rows.append(dict(seg=name, v_des=vdes, ff=float(ff[i0]), v_ss=v_ss, ratio=v_ss / vdes,
                         tau_ss=e_ss, v_std=v_std, t_rise=t90 - t10, t63=t63, delay=delay))
    # 摩擦：|τ_ss| = τ_f + b·|v_ss|（用 step 的各級）
    A = np.array([[1.0, abs(r["v_ss"])] for r in rows if abs(r["v_ss"]) > 0.05])
    y = np.array([abs(r["tau_ss"]) for r in rows if abs(r["v_ss"]) > 0.05])
    tau_f, b = (np.linalg.lstsq(A, y, rcond=None)[0] if len(y) >= 2 else (float("nan"), float("nan")))
    t63s = np.array([r["t63"] for r in rows if np.isfinite(r["t63"])])
    dls = np.array([r["delay"] for r in rows if np.isfinite(r["delay"])])
    delay = float(np.nanmedian(dls)) if dls.size else float("nan")
    t63 = float(np.nanmedian(t63s)) if t63s.size else float("nan")
    # 一階＋純延遲：v 從 delay 起以時間常數 τ_c 上升 → J = τ_c·(kd + b)。τ_c 小於 5 ms 就量不出來（回報上限）
    tau_c = max(t63 - delay, 0.0)
    J_tc = float(tau_c * (kd + b)) if np.isfinite(tau_c) else float("nan")
    return dict(rows=rows, tau_f=float(tau_f), b=float(b), delay=delay, t63=t63, tau_c=float(tau_c), J_tc=J_tc, kd=kd,
                tau_c_resolution_s=0.005)


def analyze_ff(d: dict, w: str) -> dict:
    t, seg, vd, ff, v, e = _series(d, w)
    kd = float(d["kd"])
    pts = []
    for name, i0, i1 in _segments(seg):
        if not name.startswith("ff"):
            continue
        tt = t[i0:i1] - t[i0]
        ss = tt >= tt[-1] - 0.5
        pts.append((abs(float(ff[i0])), float(abs(v[i0:i1][ss]).mean()), float(abs(e[i0:i1][ss]).mean()), abs(float(vd[i0]))))
    if len(pts) < 2:
        return dict(pts=pts)
    F = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); vdes = pts[0][3]
    slope, icpt = np.polyfit(F, V, 1)          # v_ss = (kd·v_des + ff − τ_f)/(kd+b)
    kd_b = 1.0 / slope if slope > 1e-9 else float("nan")
    b = kd_b - kd
    tau_f = kd * vdes - icpt * kd_b
    return dict(pts=pts, slope=float(slope), kd_plus_b=float(kd_b), b=float(b), tau_f=float(tau_f), v_des=vdes)


def analyze_noise(d: dict, w: str) -> dict:
    t, seg, vd, ff, v, e = _series(d, w)
    out = {}
    for name, i0, i1 in _segments(seg):
        if name not in ("hold0", "hold2"):
            continue
        tt = t[i0:i1] - t[i0]
        m = tt >= 1.0
        out[name] = dict(v_mean=float(v[i0:i1][m].mean()), v_std=float(v[i0:i1][m].std()),
                         tau_std=float(e[i0:i1][m].std()), v_des=float(vd[i0]))
    return out


def analyze_chirp(d: dict, w: str, win: float = 1.0) -> dict:
    t, seg, vd, ff, v, e = _series(d, w)
    A, f0, f1, T = d["chirp"]
    k = (f1 - f0) / T
    rows = []
    for name, i0, i1 in _segments(seg):
        if name != "chirp":
            continue
        tt = t[i0:i1] - t[i0]; vv = v[i0:i1]; vdes = vd[i0:i1]
        t0 = 0.0
        while t0 + win <= tt[-1]:
            m = (tt >= t0) & (tt < t0 + win)
            fc = f0 + k * (t0 + win / 2)                  # 窗中心瞬時頻率
            ph = 2 * np.pi * (f0 * tt[m] + 0.5 * k * tt[m] ** 2)
            X = np.column_stack([np.sin(ph), np.cos(ph), np.ones(m.sum())])
            cv, *_ = np.linalg.lstsq(X, vv[m], rcond=None)
            cd, *_ = np.linalg.lstsq(X, vdes[m], rcond=None)
            amp_v, amp_d = np.hypot(cv[0], cv[1]), np.hypot(cd[0], cd[1])
            phase = np.degrees(np.arctan2(cv[1], cv[0]) - np.arctan2(cd[1], cd[0]))
            phase = (phase + 180) % 360 - 180
            rows.append(dict(f=float(fc), gain=float(amp_v / amp_d) if amp_d > 1e-6 else float("nan"), phase_deg=float(phase)))
            t0 += win / 2
    bw = next((r["f"] for r in rows if r["gain"] < 0.707), float("nan")) if rows else float("nan")
    # 純延遲估：高頻段相位落後扣掉一階的 −atan(f/fc) 後 / (360 f)
    return dict(rows=rows, bandwidth_hz=float(bw))


def report(results: dict, path: str) -> str:
    d = results
    L = [f"# 輪子系統辨識　{Path(path).name}　協定 {d['proto']}　kd {d['kd']:g}", ""]
    if d["proto"] == "step":
        L += ["| 輪 | v_des | v_ss | 追蹤率 | τ_ss | 延遲 s | 上升 10–90 % s | τ63 s | v_std |", "|---|---|---|---|---|---|---|---|---|"]
        for w, r in d["per_wheel"].items():
            for row in r["rows"]:
                L.append(f"| {w[:2]} | {row['v_des']:+.2f} | {row['v_ss']:+.2f} | {row['ratio']:.2f} | {row['tau_ss']:+.3f} | "
                         f"{row['delay']:.3f} | {row['t_rise']:.3f} | {row['t63']:.3f} | {row['v_std']:.3f} |")
        L += ["", "| 輪 | τ_f N·m | b N·m·s | 延遲 s | τ_c s（τ63−延遲） | J = τ_c·(kd+b) kg·m² |", "|---|---|---|---|---|---|"]
        for w, r in d["per_wheel"].items():
            L.append(f"| {w[:2]} | {r['tau_f']:.3f} | {r['b']:.4f} | {r['delay']:.3f} | {r['tau_c']:.3f} | {r['J_tc']:.4f} |")
        avg = d["avg"]
        L += ["", f"**四輪平均：τ_f {avg['tau_f']:.3f} N·m、b {avg['b']:.4f} N·m·s、延遲 {avg['delay']:.3f} s、τ_c {avg['tau_c']:.3f} s、J {avg['J_tc']:.4f} kg·m²**（τ_c 解析度 5 ms；小於它就只知道上限）",
              f"純 kd 伺服的穩態誤差預測 (τ_f + b·v)/kd；kd {d['kd']:g} 時 v=3 rad/s 誤差 ≈ {(avg['tau_f'] + avg['b'] * 3) / d['kd']:.2f} rad/s。", ""]
    elif d["proto"] == "ff":
        L += ["| 輪 | 斜率 1/(kd+b) | kd+b | b | τ_f |", "|---|---|---|---|---|"]
        for w, r in d["per_wheel"].items():
            if "slope" in r:
                L.append(f"| {w[:2]} | {r['slope']:.3f} | {r['kd_plus_b']:.3f} | {r['b']:.4f} | {r['tau_f']:.3f} |")
        L += ["", "各前饋級的 (τ_ff, v_ss)：" + "；".join(f"{w[:2]} " + ",".join(f"({p[0]:.2f}→{p[1]:.2f})" for p in r["pts"]) for w, r in d["per_wheel"].items()), ""]
    elif d["proto"] == "noise":
        L += ["| 輪 | 段 | v_des | v 平均 | v std | τ std |", "|---|---|---|---|---|---|"]
        for w, r in d["per_wheel"].items():
            for name, x in r.items():
                L.append(f"| {w[:2]} | {name} | {x['v_des']:.1f} | {x['v_mean']:+.3f} | {x['v_std']:.4f} | {x['tau_std']:.4f} |")
        L.append("")
    elif d["proto"] == "chirp":
        L += ["| 輪 | 頻寬 −3 dB Hz | 各頻率 (f: 增益/相位°) |", "|---|---|---|"]
        for w, r in d["per_wheel"].items():
            pts = " ".join(f"{x['f']:.1f}:{x['gain']:.2f}/{x['phase_deg']:+.0f}" for x in r["rows"][::2])
            L.append(f"| {w[:2]} | {r['bandwidth_hz']:.2f} | {pts} |")
        L.append("")
    if d.get("aborted"):
        L.append(f"⚠️ 這趟中止：{d['aborted']}")
    return "\n".join(L)


def analyze(path: str) -> dict:
    d = load(path)
    fn = {"step": analyze_step, "ff": analyze_ff, "noise": analyze_noise, "chirp": analyze_chirp}[d["proto"]]
    per = {w: fn(d, w) for w in d["wheels"]}
    out = dict(proto=d["proto"], kd=float(d["kd"]), per_wheel=per, aborted=d.get("aborted", ""))
    if d["proto"] == "step":
        keys = ("tau_f", "b", "delay", "tau_c", "J_tc")
        out["avg"] = {k: float(np.nanmean([per[w][k] for w in per])) for k in keys}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    txt = []
    for p in a.logs:
        txt.append(report(analyze(p), p))
    s = "\n\n".join(txt)
    print(s)
    if a.out:
        Path(a.out).write_text(s + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
