"""原廠參考錄製（M6 --record，trip21）的步態分析：每檔一列 —— 動作區間、輪速、里程反推速度、偏航率、
腿有沒有踏步（足端抬高、抬腿次數、步頻、膝力矩歸零比例）、ABAD 偏置與擺幅、輪速時間序列摘要。

    conda run -n rbtdog python task7/inference/ref_gait_analysis.py task7/logs/m_logs_trip21/M6_*.json --skip 162009 --out task7/outputs/ref_gait_trip21.md

慣例：輪速轉成「正＝向前滾」（馬達 v ÷ coord.SIGN）；關節角轉控制器座標；足端高度用 realbot/kin.fk。
機身速度 = 0.09 m × 四輪平均角速度（無打滑假設，純前進段可信、踏步段只當參考）。
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

R_WHEEL = 0.09
LEGS = ("fl", "fr", "bl", "br")
LIFT_MM = 15.0          # 抬腿判定：足端高度離基線 > 15 mm
HZ = 500.0


def load(path: str):
    rec = m6_rec.load(path)
    q = {l: {k: coord.to_ctrl(l + k, rec.j[l + k]["q"]) for k in coord.LEG_KINDS} for l in LEGS}
    tau = {l: {k: rec.j[l + k]["tau"] / coord.SIGN[k][l] for k in coord.LEG_KINDS} for l in LEGS}
    wv = np.stack([rec.j[l + "4_foot"]["v"] / coord.SIGN[coord.KIND_WHEEL][l] for l in LEGS], 1)
    return rec, q, tau, wv


def foot_z(q, l):
    return np.array([kin.fk(l, a, b, c)[2] for a, b, c in
                     zip(q[l]["1_hip_roll"], q[l]["2_hip_pitch"], q[l]["3_knee_pitch"])])


def moving_mask(rec, q, wv):
    gz = rec.gyro[:, 2]
    abad = np.stack([q[l]["1_hip_roll"] for l in LEGS], 1)
    return (np.abs(gz) > 0.15) | (np.abs(wv).max(1) > 1.0) | (np.abs(np.gradient(abad, axis=0)).max(1) > 0.002)


def analyze(path: str) -> dict:
    rec, q, tau, wv = load(path)
    label = json.loads(Path(path).read_text(encoding="utf-8")).get("label", "")
    t = rec.t
    m = moving_mask(rec, q, wv)
    idx = np.nonzero(m)[0]
    if idx.size < 200:
        i0, i1 = 0, rec.n - 1
    else:
        i0, i1 = int(idx[0]), int(idx[-1])
    sl = slice(i0, i1)
    gz = rec.gyro[sl, 2]
    vb = R_WHEEL * wv[sl].mean(1)
    legs = {}
    for k, l in enumerate(LEGS):
        z = foot_z(q, l)[sl]
        lift = z - np.percentile(z, 10)
        up = lift > LIFT_MM / 1000
        ev = np.nonzero(np.diff(up.astype(int)) == 1)[0]
        kt = np.abs(tau[l]["3_knee_pitch"][sl])
        legs[l] = dict(lift_p98_mm=float(np.percentile(lift, 98) * 1000), n_lift=int(len(ev)),
                       period_s=float(np.median(np.diff(ev)) / HZ) if len(ev) > 2 else float("nan"),
                       abad_mean_deg=float(np.degrees(q[l]["1_hip_roll"][sl].mean())),
                       abad_ptp_deg=float(np.degrees(np.ptp(q[l]["1_hip_roll"][sl]))),
                       hip_ptp_deg=float(np.degrees(np.ptp(q[l]["2_hip_pitch"][sl]))),
                       knee_ptp_deg=float(np.degrees(np.ptp(q[l]["3_knee_pitch"][sl]))),
                       knee_unloaded_frac=float((kt < 3.0).mean()),
                       wheel_mean=float(wv[sl, k].mean()), wheel_std=float(wv[sl, k].std()))
    stepping = sum(1 for l in LEGS if legs[l]["n_lift"] >= 3 and legs[l]["knee_unloaded_frac"] > 0.08)
    return dict(path=path, label=label, t0=float(t[i0]), t1=float(t[i1]), secs=float((i1 - i0) / HZ),
                v_body=float(vb.mean()), v_body_max=float(np.abs(vb).max()),
                yaw_rate_deg=float(np.degrees(gz.mean())), gyro_xy_std=float(np.degrees(rec.gyro[sl, :2].std())),
                legs=legs, stepping_legs=stepping)


def report(rows: list) -> str:
    L = ["# 原廠參考錄製分析（trip21）", "",
         "動作區間＝輪速 > 1 rad/s 或 |gyro_z| > 0.15 或 ABAD 在動；v_body = 0.09 × 四輪平均（無打滑假設）；",
         "抬腿＝足端高度離基線 > 15 mm；膝卸載＝|膝 τ| < 3 N·m 的時間比例（擺動相）。", "",
         "| label | 秒 | v_body m/s | 偏航 °/s | 踏步腿數 | 抬腿次數 fl/fr/bl/br | 步頻 Hz | 抬高 mm | 膝卸載 % | ABAD 平均° fl/fr/bl/br | ABAD 擺幅° | 輪速 mean fl/fr/bl/br |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["legs"]
        per = [g[l]["period_s"] for l in LEGS if np.isfinite(g[l]["period_s"])]
        f = f"{1 / np.median(per):.1f}" if per else "—"
        L.append(f"| {r['label']} | {r['secs']:.1f} | {r['v_body']:+.2f} | {r['yaw_rate_deg']:+.0f} | {r['stepping_legs']} | "
                 + "/".join(str(g[l]["n_lift"]) for l in LEGS) + f" | {f} | "
                 + f"{max(g[l]['lift_p98_mm'] for l in LEGS):.0f} | "
                 + "/".join(f"{100 * g[l]['knee_unloaded_frac']:.0f}" for l in LEGS) + " | "
                 + "/".join(f"{g[l]['abad_mean_deg']:+.0f}" for l in LEGS) + " | "
                 + "/".join(f"{g[l]['abad_ptp_deg']:.0f}" for l in LEGS) + " | "
                 + "/".join(f"{g[l]['wheel_mean']:+.1f}" for l in LEGS) + " |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--skip", default="", help="逗號分隔的時間戳片段（例如 162009）")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    skip = [s for s in a.skip.split(",") if s]
    rows = [analyze(p) for p in sorted(a.logs) if not any(s in p for s in skip)]
    txt = report(rows)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
