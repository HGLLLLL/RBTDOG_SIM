"""從 trip21 原廠錄製抽「參考資料集」：雙模式產生器與模仿獎勵要用的數字（L 文件 §4）。

輸出 `outputs/ref_gait_dataset.json`（＋ md 摘要）：
  wheel_mode:  站姿（12 關節平均角，控制器座標）、v_body 對輪速的比（有效輪半徑）、arc 的有效輪距 L_eff、腿順從擺幅
  step_mode:   每種動作（turn_left/right、lat_left/right）的 —— 步頻、擺動時長、duty、抬高、
               四腿相位表（相對 fl，單位 2π）、逐腿擺動位移 (dx, dy)、輪速圖案（正規化）、
               擺動軌跡樣板（正規化時間 0..1 的 z / dx / dy 平均曲線，每腿）
  transitions: lat_to_fwd / turn_to_fwd 的模式切換時長

    conda run -n rbtdog python task7/inference/ref_extract.py --out task7/outputs/ref_gait_dataset.json --md task7/outputs/ref_gait_dataset.md

擺動事件：足端高度（kin.fk，輪軸心，腿座標 x 前 y 左 z 上）離 10 百分位基線 > 15 mm 的連續段，
且該段膝 |τ| 最小 < 3 N·m（真的卸載）。相位：以 fl 的擺動起點為 0，其他腿取「最近一次擺動起點」相對 fl 週期的比例，取圓形平均。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realbot"))
import coord  # noqa: E402
import kin  # noqa: E402
import m6_rec  # noqa: E402
import ref_gait_analysis as rga  # noqa: E402

LEGS = rga.LEGS
HZ = 500.0
LIFT = 0.015
FILES = {
    "idle": ["160456"], "fwd": ["160740", "160825", "160911"], "back": ["161002"],
    "arc_left": ["161503", "161544"], "arc_right": ["161644", "161711"],
    "turn_left": ["161316"], "turn_right": ["161342"],
    "lat_left": ["161123"], "lat_right": ["161150", "161223"],
    "lat_to_fwd": ["161908"], "turn_to_fwd": ["162019"], "startstop": ["161817"],
}


def _path(tag: str, logdir: Path) -> str:
    return str(next(logdir.glob(f"M6_*_{tag}.json")))


def foot_xyz(q, l):
    return np.array([kin.fk(l, a, b, c) for a, b, c in
                     zip(q[l]["1_hip_roll"], q[l]["2_hip_pitch"], q[l]["3_knee_pitch"])])


def swing_events(z, knee_tau_abs, i0, i1):
    """→ [(s, e)]（索引，含 s 不含 e），只留膝真的卸載的段。"""
    base = np.percentile(z[i0:i1], 10)
    up = (z - base) > LIFT
    up[:i0] = False
    up[i1:] = False
    d = np.diff(up.astype(int))
    starts, ends = np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1
    if ends.size and starts.size and ends[0] < starts[0]:
        ends = ends[1:]
    ev = []
    for s, e in zip(starts, ends):
        if e - s >= int(0.05 * HZ) and knee_tau_abs[s:e].min() < 3.0:
            ev.append((int(s), int(e)))
    return ev


def circ_mean(ph):
    return float((np.angle(np.mean(np.exp(2j * np.pi * np.asarray(ph)))) / (2 * np.pi)) % 1.0)


def _circ_std(ph):
    r = abs(np.mean(np.exp(2j * np.pi * np.asarray(ph))))
    return float(np.sqrt(-2 * np.log(max(1e-9, r))) / (2 * np.pi))


def analyze_step_file(path: str) -> dict:
    rec, q, tau, wv = rga.load(path)
    m = rga.moving_mask(rec, q, wv)
    idx = np.nonzero(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    P = {l: foot_xyz(q, l) for l in LEGS}
    ev = {l: swing_events(P[l][:, 2], np.abs(tau[l]["3_knee_pitch"]), i0, i1) for l in LEGS}
    out = {"legs": {}, "wheel_pattern": {}, "n_swings": {l: len(ev[l]) for l in LEGS}}
    # 週期：每腿相鄰擺動起點的中位數
    periods = []
    for l in LEGS:
        st = np.array([s for s, _ in ev[l]])
        if st.size >= 3:
            periods += list(np.diff(st) / HZ)
    period = float(np.median(periods)) if periods else float("nan")
    out["period_s"] = period
    out["freq_hz"] = 1.0 / period if period == period and period > 0 else float("nan")
    # 相位表：相對「擺動最多的腿」（每種動作只有兩條腿在踏步，fl 未必是其中之一）
    ref_leg = max(LEGS, key=lambda l: len(ev[l]))
    out["ref_leg"] = ref_leg
    fl_starts = np.array([s for s, _ in ev[ref_leg]])
    for l in LEGS:
        rows, tmpl = [], []
        for s, e in ev[l]:
            z = P[l][s:e, 2]
            base = np.percentile(P[l][i0:i1, 2], 10)
            apex = float(z.max() - base)
            dx = float(P[l][e - 1, 0] - P[l][s, 0])
            dy = float(P[l][e - 1, 1] - P[l][s, 1])
            rows.append(dict(t=float(rec.t[s]), dur=float((e - s) / HZ), apex_m=apex, dx=dx, dy=dy))
            # 擺動軌跡樣板：正規化時間 20 點
            u = np.linspace(0, 1, 20)
            tt = np.linspace(0, 1, e - s)
            tmpl.append(np.stack([np.interp(u, tt, z - base), np.interp(u, tt, P[l][s:e, 0] - P[l][s, 0]),
                                  np.interp(u, tt, P[l][s:e, 1] - P[l][s, 1])], 1))
        ph = []
        if fl_starts.size >= 2 and period == period:
            for s, _ in ev[l]:
                k = np.searchsorted(fl_starts, s, side="right") - 1
                if 0 <= k < fl_starts.size:
                    ph.append((((s - fl_starts[k]) / HZ) / period) % 1.0)
        out["legs"][l] = dict(
            n=len(rows), swing_dur_s=float(np.median([r["dur"] for r in rows])) if rows else float("nan"),
            apex_m=float(np.median([r["apex_m"] for r in rows])) if rows else float("nan"),
            dx_m=float(np.median([r["dx"] for r in rows])) if rows else float("nan"),
            dy_m=float(np.median([r["dy"] for r in rows])) if rows else float("nan"),
            phase=circ_mean(ph) if ph else float("nan"),
            phase_spread=_circ_std(ph) if ph else float("nan"),   # 圓形 std（週期比例）
            duty=float(1 - np.median([r["dur"] for r in rows]) / period) if rows and period == period else float("nan"),
            template=np.mean(tmpl, 0).round(4).tolist() if tmpl else None,
            abad_mean_deg=float(np.degrees(q[l]["1_hip_roll"][i0:i1].mean())),
            abad_ptp_deg=float(np.degrees(np.ptp(q[l]["1_hip_roll"][i0:i1]))))
    # 輪速圖案：踏步區間內每輪平均，正規化到 |max| = 1
    wm = wv[i0:i1].mean(0)
    scale = float(np.abs(wm).max()) or 1.0
    out["wheel_pattern"] = {l: float(wm[k] / scale) for k, l in enumerate(LEGS)}
    out["wheel_mean_rad_s"] = {l: float(wm[k]) for k, l in enumerate(LEGS)}
    out["yaw_rate_deg_s"] = float(np.degrees(rec.gyro[i0:i1, 2].mean()))
    out["v_body"] = float(rga.R_WHEEL * wv[i0:i1].mean())
    out["secs"] = float((i1 - i0) / HZ)
    return out


def analyze_wheel_file(path: str) -> dict:
    rec, q, tau, wv = rga.load(path)
    m = rga.moving_mask(rec, q, wv)
    idx = np.nonzero(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    # 穩態：|四輪平均| ≥ 80% 的 p95
    vb = rga.R_WHEEL * wv[:, :].mean(1)
    thr = 0.8 * np.percentile(np.abs(vb[i0:i1]), 95)
    ss = np.zeros(rec.n, bool)
    ss[i0:i1] = np.abs(vb[i0:i1]) >= thr
    if ss.sum() < 100:
        ss[i0:i1] = True
    q12 = np.array([q[l][k][ss].mean() for l in LEGS for k in coord.LEG_KINDS])
    q12_std = np.array([q[l][k][ss].std() for l in LEGS for k in coord.LEG_KINDS])
    gz = rec.gyro[ss, 2]
    vl, vr = rga.R_WHEEL * wv[ss][:, [0, 2]].mean(), rga.R_WHEEL * wv[ss][:, [1, 3]].mean()
    return dict(secs=float(ss.sum() / HZ), v_body=float(vb[ss].mean()), v_left=float(vl), v_right=float(vr),
                yaw_rate=float(gz.mean()),
                L_eff=float((vr - vl) / gz.mean()) if abs(gz.mean()) > 0.05 else float("nan"),
                stance_q12=q12.round(4).tolist(), stance_q12_std_deg=np.degrees(q12_std).round(2).tolist(),
                roll_std_deg=float(np.degrees(rec.gyro[ss, 0].std())), pitch_std_deg=float(np.degrees(rec.gyro[ss, 1].std())))


def analyze_transition(path: str) -> dict:
    """踏步 → 輪行的切換：最後一次擺動結束 → 四輪都同向且 |v|>1 rad/s 的第一刻。"""
    rec, q, tau, wv = rga.load(path)
    m = rga.moving_mask(rec, q, wv)
    idx = np.nonzero(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    P = {l: foot_xyz(q, l) for l in LEGS}
    last_sw = max([e for l in LEGS for _, e in swing_events(P[l][:, 2], np.abs(tau[l]["3_knee_pitch"]), i0, i1)] or [i0])
    # 輪行＝四輪同向 > 1 rad/s 且四輪差 < 1.5 rad/s（踏步時同側／對角反向，不會滿足）
    roll = (wv.min(1) > 1.0) & (np.ptp(wv, axis=1) < 1.5)
    same = np.nonzero(roll)[0]
    same = same[same > last_sw]
    first_roll = int(same[0]) if same.size else i1
    return dict(t_last_swing=float(rec.t[last_sw]), t_first_roll=float(rec.t[first_roll]),
                gap_s=float((first_roll - last_sw) / HZ))


# ---------------------------------------------------------------- v3.1（2026-09-15）：擺動事件法、四腿相位、弧線、增益
LIFT31 = 0.012                 # 相對 1 s 滾動最低點抬 > 12 mm
DUR31 = (0.04, 0.35)           # 擺動時長（s）；更長的是姿態變化
FOOT_X_BODY = {"fl": 0.376, "fr": 0.376, "bl": -0.358, "br": -0.358}   # 站姿足端 x（機身座標，m）；旋轉運動學用


def swing_events31(z: np.ndarray) -> list:
    """→ [(s, e, apex_m)]。基線 = 1 s 滾動最低點（吃掉姿態漂移），抬 > LIFT31、時長在 DUR31 內。"""
    from scipy.ndimage import minimum_filter1d
    h = z - minimum_filter1d(z, int(HZ))
    up = h > LIFT31
    d = np.diff(up.astype(int))
    starts, ends = np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1
    if ends.size and starts.size and ends[0] < starts[0]:
        ends = ends[1:]
    lo, hi = DUR31[0] * HZ, DUR31[1] * HZ
    return [(int(s), int(e), float(h[s:e].max())) for s, e in zip(starts, ends) if lo <= e - s <= hi]


def phase_rel(starts: np.ndarray, ref_starts: np.ndarray, period_ticks: float) -> float:
    """每個 start 對最近一個 ref start 的落後（週期比例），圓形平均。"""
    ph = []
    for s in starts:
        k = np.searchsorted(ref_starts, s, side="right") - 1
        if 0 <= k < ref_starts.size:
            ph.append(((s - ref_starts[k]) / period_ticks) % 1.0)
    return circ_mean(ph) if ph else float("nan")


def gains_of(rec, i0: int, i1: int) -> dict:
    """動作段的增益（取 fl；16 顆同步切換）。"""
    def u(name, k):
        return sorted(set(np.round(rec.j[name][k][i0:i1], 2).tolist()))
    return dict(abad_kp=u("fl1_hip_roll", "kp"), hip_kp=u("fl2_hip_pitch", "kp"), knee_kp=u("fl3_knee_pitch", "kp"),
                leg_kd=u("fl2_hip_pitch", "kd"), wheel_kp=u("fl4_foot", "kp"), wheel_kd=u("fl4_foot", "kd"),
                wheel_ff_p98=float(np.percentile(np.abs(rec.j["fl4_foot"]["ff"][i0:i1]), 98)))


def analyze31(path: str) -> dict:
    rec, q, tau, wv = rga.load(path)
    m = rga.moving_mask(rec, q, wv)
    idx = np.nonzero(m)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    P = {l: foot_xyz(q, l) for l in LEGS}
    ev = {l: [e for e in swing_events31(P[l][:, 2]) if i0 <= e[0] <= i1] for l in LEGS}
    out = {"label": json.loads(Path(path).read_text(encoding="utf-8")).get("label", ""), "legs": {}, "gains": gains_of(rec, i0, i1)}
    # 穩態偏航：|gyro_z| ≥ 50% 的 p95 那些時刻
    gz = rec.gyro[:, 2]
    thr = 0.5 * np.percentile(np.abs(gz[i0:i1]), 95)
    ss = np.zeros(rec.n, bool)
    ss[i0:i1] = np.abs(gz[i0:i1]) >= max(thr, 0.05)
    if ss.sum() < 100:
        ss[i0:i1] = True
    out["yaw_rate_deg_s"] = float(np.degrees(gz[ss].mean()))
    out["wheel_mean_rad_s"] = {l: float(wv[ss, k].mean()) for k, l in enumerate(LEGS)}
    periods = []
    for l in LEGS:
        st = np.array([s for s, _, _ in ev[l]])
        if st.size >= 3:
            dp = np.diff(st) / HZ
            periods += dp[dp < 1.0].tolist()
    period = float(np.median(periods)) if periods else float("nan")
    out["period_s"], out["freq_hz"] = period, (1.0 / period if period == period and period > 0 else float("nan"))
    ref_leg = max(LEGS, key=lambda l: len(ev[l]))
    out["ref_leg"] = ref_leg
    ref_starts = np.array([s for s, _, _ in ev[ref_leg]])
    for l in LEGS:
        E = ev[l]
        out["legs"][l] = dict(
            n=len(E),
            swing_dur_s=float(np.median([(e - s) / HZ for s, e, _ in E])) if E else float("nan"),
            apex_m=float(np.median([a for _, _, a in E])) if E else float("nan"),
            dx_m=float(np.median([P[l][e - 1, 0] - P[l][s, 0] for s, e, _ in E])) if E else float("nan"),
            dy_m=float(np.median([P[l][e - 1, 1] - P[l][s, 1] for s, e, _ in E])) if E else float("nan"),
            phase=phase_rel(np.array([s for s, _, _ in E]), ref_starts, period * HZ) if E and period == period else float("nan"),
            abad_ptp_deg=float(np.degrees(np.ptp(q[l]["1_hip_roll"][i0:i1]))),
            foot_y_mean_m=float(P[l][ss, 1].mean()), foot_y_idle_m=float(P[l][:i0, 1].mean()) if i0 > 50 else float("nan"))
    # 旋轉運動學：踏步做掉的比例 = 量到的 dy / (T · ω · x_foot)
    w = abs(np.radians(out["yaw_rate_deg_s"]))
    if period == period and w > 0.3 and out["legs"][ref_leg]["n"] >= 3:
        dy_kin = period * w * abs(FOOT_X_BODY[ref_leg])
        out["rot_step_frac"] = float(abs(out["legs"][ref_leg]["dy_m"]) / dy_kin)
    else:
        out["rot_step_frac"] = float("nan")
    return out


def build(logdir: Path) -> dict:
    ds = {"source": "trip21 2026-09-09 原廠遙控錄製（M6 500 Hz）", "wheel_mode": {}, "step_mode": {}, "transitions": {}}
    for k in ("fwd", "back", "arc_left", "arc_right", "startstop"):
        ds["wheel_mode"][k] = [analyze_wheel_file(_path(t, logdir)) for t in FILES[k]]
    for k in ("turn_left", "turn_right", "lat_left", "lat_right"):
        ds["step_mode"][k] = [analyze_step_file(_path(t, logdir)) for t in FILES[k]]
    for k in ("lat_to_fwd", "turn_to_fwd"):
        ds["transitions"][k] = analyze_transition(_path(FILES[k][0], logdir))
    ds["v31"] = {k: [analyze31(_path(t, logdir)) for t in FILES[k]]
                 for k in ("turn_left", "turn_right", "lat_left", "lat_right", "arc_left", "arc_right", "fwd")}
    # 彙總：輪行站姿（fwd 三檔平均）、有效輪距（arc 四檔中位）
    fw = ds["wheel_mode"]["fwd"]
    ds["summary"] = {
        "wheel_stance_q12": np.mean([f["stance_q12"] for f in fw], 0).round(4).tolist(),
        "wheel_stance_compliance_deg": float(np.mean([np.mean(f["stance_q12_std_deg"]) for f in fw])),
        "L_eff_m": float(np.nanmedian([f["L_eff"] for k in ("arc_left", "arc_right") for f in ds["wheel_mode"][k]])),
        "v_body_range": [float(min(f["v_body"] for f in fw)), float(max(f["v_body"] for f in fw))],
    }
    for k, files in ds["step_mode"].items():
        f = files[-1] if k != "lat_right" else files[1]
        ds["summary"][k] = {"freq_hz": f["freq_hz"], "apex_m": float(np.nanmedian([f["legs"][l]["apex_m"] for l in LEGS])),
                            "swing_dur_s": float(np.nanmedian([f["legs"][l]["swing_dur_s"] for l in LEGS])),
                            "duty": float(np.nanmedian([f["legs"][l]["duty"] for l in LEGS])),
                            "phase": {l: f["legs"][l]["phase"] for l in LEGS}, "ref_leg": f["ref_leg"],
                            "stepping_legs": [l for l in LEGS if f["legs"][l]["n"] >= 5],
                            "wheel_pattern": f["wheel_pattern"], "yaw_rate_deg_s": f["yaw_rate_deg_s"]}
    v = ds["v31"]
    tl, ll = v["turn_left"][0], v["lat_left"][0]

    def _ph(f):
        # 相對 fl 的相位（fl 若不是 ref_leg 就平移）
        base = f["legs"]["fl"]["phase"]
        return {l: float((f["legs"][l]["phase"] - base) % 1.0) for l in LEGS}

    def _apex_ratio(f, dom):
        d = np.nanmean([f["legs"][l]["apex_m"] for l in dom])
        s_ = np.nanmean([f["legs"][l]["apex_m"] for l in LEGS if l not in dom])
        return float(s_ / d)
    arc = [f for k in ("arc_left", "arc_right") for f in v[k]]
    ds["summary"]["v31"] = {
        "phase_turn_left": _ph(tl), "phase_lat_left": _ph(ll),
        "apex_ratio_turn": _apex_ratio(tl, ("fl", "br")), "apex_ratio_lat": _apex_ratio(ll, ("fl", "bl")),
        "rot_step_frac_turn": float(np.nanmedian([f["rot_step_frac"] for f in v["turn_left"] + v["turn_right"]])),
        "arc_inner_front_hz": float(np.nanmedian([f["freq_hz"] for f in arc])),
        "arc_inner_front_apex_m": float(np.nanmedian([f["legs"][f["ref_leg"]]["apex_m"] for f in arc])),
        "arc_inner_front_leg": [f["ref_leg"] for f in arc],
        "lat_lead_dy_m": float(np.nanmean([ll["legs"][l]["dy_m"] for l in ("fl", "bl")])),
        "lat_trail_dy_m": float(np.nanmean([ll["legs"][l]["dy_m"] for l in ("fr", "br")])),
        "gains_motion": tl["gains"],
        "gains_fwd": v["fwd"][0]["gains"],
    }
    return ds


def md(ds: dict) -> str:
    S = ds["summary"]
    L = ["# 原廠參考資料集摘要（trip21）", "",
         f"輪行站姿（12 關節，控制器座標，rad）：{S['wheel_stance_q12']}",
         f"站姿順從 std {S['wheel_stance_compliance_deg']:.2f}°；有效輪距 L_eff {S['L_eff_m']:.3f} m；v_body 檔位 {S['v_body_range'][0]:.2f}–{S['v_body_range'][1]:.2f} m/s", ""]
    for k in ("fwd", "back", "arc_left", "arc_right"):
        for f in ds["wheel_mode"][k]:
            L.append(f"- {k}: v {f['v_body']:+.2f}（左 {f['v_left']:+.2f}／右 {f['v_right']:+.2f}）偏航 {np.degrees(f['yaw_rate']):+.1f}°/s L_eff {f['L_eff']:.3f}")
    L += ["", "| 動作 | 踏步腿 | 步頻 Hz | 擺動 s | duty | 抬高 mm | 相位（相對 ref）fl/fr/bl/br | 輪速圖案 fl/fr/bl/br | 偏航 °/s |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("turn_left", "turn_right", "lat_left", "lat_right"):
        s = S[k]
        L.append(f"| {k} | {'+'.join(s['stepping_legs'])}（ref {s['ref_leg']}） | {s['freq_hz']:.2f} | {s['swing_dur_s']:.3f} | {s['duty']:.2f} | {s['apex_m'] * 1000:.0f} | "
                 + "/".join(f"{s['phase'][l]:.2f}" for l in LEGS) + " | "
                 + "/".join(f"{s['wheel_pattern'][l]:+.2f}" for l in LEGS) + f" | {s['yaw_rate_deg_s']:+.0f} |")
    L += ["", "逐腿（最後一檔）：", "| 動作 | 腿 | 擺動數 | dx mm | dy mm | 相位散佈 | ABAD 平均° | ABAD 擺幅° |", "|---|---|---|---|---|---|---|---|"]
    for k, files in ds["step_mode"].items():
        f = files[-1]
        for l in LEGS:
            g = f["legs"][l]
            L.append(f"| {k} | {l} | {g['n']} | {g['dx_m'] * 1000:+.0f} | {g['dy_m'] * 1000:+.0f} | {g['phase_spread']:.2f} | {g['abad_mean_deg']:+.1f} | {g['abad_ptp_deg']:.1f} |")
    L += ["", "模式切換（踏步最後一次擺動結束 → 四輪同向滾）：" + "；".join(f"{k} {v['gap_s']:.2f} s" for k, v in ds["transitions"].items())]
    V = S.get("v31")
    if V:
        L += ["", "## v3.1（擺動事件法）", "",
              f"- 原地左轉相位（相對 fl）：{ {l: round(p, 2) for l, p in V['phase_turn_left'].items()} }；次要腿抬高比 {V['apex_ratio_turn']:.2f}；踏步做掉的旋轉位移比例 {V['rot_step_frac_turn']:.2f}",
              f"- 左平移相位：{ {l: round(p, 2) for l, p in V['phase_lat_left'].items()} }；次要腿抬高比 {V['apex_ratio_lat']:.2f}；移動側每步 dy {V['lat_lead_dy_m'] * 1000:+.0f} mm、另一側 {V['lat_trail_dy_m'] * 1000:+.0f} mm",
              f"- 弧線：內側前腿 {V['arc_inner_front_leg']} 踏步 {V['arc_inner_front_hz']:.2f} Hz、抬 {V['arc_inner_front_apex_m'] * 1000:.0f} mm",
              f"- 動作段增益：{V['gains_motion']}",
              f"- 前進段增益：{V['gains_fwd']}"]
        L += ["", "| 檔 | 偏航 °/s | 步頻 | fl n/apex/dy | fr | bl | br | 輪 fl/fr/bl/br |", "|---|---|---|---|---|---|---|---|"]
        for k, files in ds["v31"].items():
            for f in files:
                cells = [f"{f['legs'][l]['n']}/{f['legs'][l]['apex_m'] * 1000:.0f}/{f['legs'][l]['dy_m'] * 1000:+.0f}" for l in LEGS]
                L.append(f"| {f['label']} | {f['yaw_rate_deg_s']:+.0f} | {f['freq_hz']:.2f} | " + " | ".join(cells)
                         + " | " + "/".join(f"{f['wheel_mean_rad_s'][l]:+.1f}" for l in LEGS) + " |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=str(HERE.parent / "logs" / "m_logs_trip21"))
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "ref_gait_dataset.json"))
    ap.add_argument("--md", default=str(HERE.parent / "outputs" / "ref_gait_dataset.md"))
    a = ap.parse_args()
    ds = build(Path(a.logdir))
    Path(a.out).write_text(json.dumps(ds, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = md(ds)
    Path(a.md).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
