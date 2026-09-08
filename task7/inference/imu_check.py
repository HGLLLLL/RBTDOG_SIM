#!/usr/bin/env python3
"""階 I 判讀：IMU 的四元數順序、gyro 單位/軸向/偏置、安裝偏置、更新率。

三個**互相獨立**的參考，不靠眼睛：
  1. 加速度計：靜止時 specific force = −重力 → 重力方向 = −acc/|acc|
  2. 腿的正向運動學：狗在平地上四輪必共面 → 地面法向 → 重力方向（M4 那招）
  3. 四元數微分：q_{t+1} = q_t ⊗ Δq → 機身系角速度，與 gyro 逐軸擬合
眼睛只當決勝票（`--seen-first left|right`、`--turn-first left|right`）。

慣例（與 MJCF 一致）：x 前、y 左、z 上；+roll = 右側低；+pitch = 頭低；+yaw = 左轉。

用法：
    python imu_check.py --flat M6_flat.json --dance M6_dance.json [--turn M6_turn.json]
                        [--seen-first left|right] [--turn-first left|right]
                        [--out outputs/imu_check.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import leg_kin          # noqa: E402
import m6_rec           # noqa: E402
import max_model as mm  # noqa: E402
import real_obs         # noqa: E402
from cpg_max import qinv, w2b  # noqa: E402

DOWN = np.array([0.0, 0.0, -1.0])
R2D = 180.0 / np.pi
ORDERS = ("xyzw", "wxyz")


# ---------------------------------------------------------------- 四元數
def qmul(a, b):
    """wxyz Hamilton 積，支援 (N,4)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def gravity_from_quat(Q):
    """(N,4) wxyz → (N,3) 機身系重力方向（與 obs 的 gravity 欄位同一式子）。"""
    Q = np.asarray(Q, float)
    return np.stack([w2b(q, DOWN) for q in Q])


def gravity_from_acc(acc):
    acc = np.asarray(acc, float)
    return -acc / np.maximum(np.linalg.norm(acc, axis=-1, keepdims=True), 1e-9)


def rp_from_gravity(g):
    """機身系重力方向 → (roll°, pitch°)。
    純 roll φ：g=(0,−sinφ,−cosφ)；純 pitch θ：g=(sinθ,0,−cosθ)。"""
    g = np.asarray(g, float)
    roll = np.arctan2(-g[..., 1], -g[..., 2]) * R2D
    pitch = np.arctan2(g[..., 0], np.hypot(g[..., 1], g[..., 2])) * R2D
    return roll, pitch


def omega_body(Q, t):
    """(N,4) wxyz + t → (N−1,3) 機身系角速度 rad/s。Δq = q_t⁻¹ ⊗ q_{t+1}。"""
    Q = np.asarray(Q, float)
    Qi = np.stack([qinv(q) for q in Q[:-1]])
    dq = qmul(Qi, Q[1:])
    dq = np.where(dq[:, :1] < 0, -dq, dq)
    s = np.linalg.norm(dq[:, 1:], axis=1)
    ang = 2 * np.arctan2(s, dq[:, 0])
    axis = dq[:, 1:] / np.maximum(s, 1e-12)[:, None]
    dt = np.diff(np.asarray(t, float))
    return axis * (ang / np.maximum(dt, 1e-9))[:, None]


# ---------------------------------------------------------------- gyro 擬合
def smooth(X, win: int):
    """沿時間軸的移動平均（同長度、邊緣用 edge pad）。"""
    X = np.asarray(X, float)
    if win <= 1:
        return X
    k = np.ones(win) / win
    Xp = np.pad(X, ((win // 2, win - 1 - win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(Xp[:, i], k, mode="valid") for i in range(X.shape[1])], axis=1)


def fit_gyro(gyro, omega, win: int = 1):
    """逐軸最小平方 gyro_i = k_i · ω_i（過原點）＋ 3×3 相關矩陣（抓軸對調）。

    ★ `win`：擬合前對兩邊做同樣的移動平均。四元數微分在 500 Hz 下雜訊很大，
      雜訊在自變數（ω）那一側會把回歸斜率**系統性壓低**（衰減偏差）——
      trip14 資料不平滑時 k=0.65，平滑 20 ms 後回到 ~1。
    """
    G, W = smooth(np.asarray(gyro, float), win), smooth(np.asarray(omega, float), win)
    k = np.array([(G[:, i] @ W[:, i]) / max(W[:, i] @ W[:, i], 1e-12) for i in range(3)])
    cm = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a, b = G[:, i] - G[:, i].mean(), W[:, j] - W[:, j].mean()
            cm[i, j] = (a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return {"k": k, "corr": np.diag(cm).copy(), "corr_matrix": cm}


def classify_scale(k):
    """k = gyro/ω。回傳 (scale 讓 scale·gyro = rad/s, unit 字串)。"""
    k = np.asarray(k, float)
    ak = np.abs(k)
    if np.all((0.6 < ak) & (ak < 1.6)):
        return np.sign(k) * 1.0, "rad/s"
    if np.all((40.0 < ak) & (ak < 75.0)):
        return np.sign(k) * (np.pi / 180.0), "deg/s"
    return 1.0 / np.where(ak < 1e-9, 1.0, k), "unknown"


# ---------------------------------------------------------------- 腿 FK 參考
_ORIGIN = np.stack([mm.SIDE_X * mm.HIP_X, mm.SIDE_Y * mm.HIP_Y, np.zeros(4)], axis=1)


def foot_points_body(P12):
    """(N,12) 控制器角（MJCF 序）→ (N,4,3) 四個輪心在機身系。"""
    P = np.asarray(P12, float).reshape(-1, 4, 3)
    out = np.zeros((P.shape[0], 4, 3))
    for i in range(P.shape[0]):
        for k in range(4):
            out[i, k] = _ORIGIN[k] + leg_kin.fk(k, P[i, k])
    return out


def gravity_from_fk(P12):
    """四輪共面 → 平面法向 → 機身系重力方向。回傳 (g (N,3), 共面殘差 mm (N,))。"""
    F = foot_points_body(P12)
    N = F.shape[0]
    g = np.zeros((N, 3))
    resid = np.zeros(N)
    for i in range(N):
        A = np.column_stack([F[i, :, 0], F[i, :, 1], np.ones(4)])
        coef, *_ = np.linalg.lstsq(A, F[i, :, 2], rcond=None)
        nrm = np.array([-coef[0], -coef[1], 1.0])
        nrm /= np.linalg.norm(nrm)
        g[i] = -nrm
        resid[i] = np.abs(A @ coef - F[i, :, 2]).max() * 1000
    return g, resid


# ---------------------------------------------------------------- 更新率／量化
def update_rate(X, hz, mask=None):
    """相鄰筆完全相同的比例 → 更新率下界。回傳 {same_frac, hz_min, n}。"""
    X = np.asarray(X, float).reshape(len(X), -1)
    same = np.all(X[1:] == X[:-1], axis=1)
    if mask is not None:
        same = same[np.asarray(mask, bool)[1:]]
    frac = float(same.mean()) if same.size else float("nan")
    return {"same_frac": frac, "hz_min": float(hz * (1.0 - frac)), "n": int(same.size)}


def quant_step(x):
    """最小非零相鄰差 = 量化步階。"""
    d = np.abs(np.diff(np.asarray(x, float)))
    d = d[d > 0]
    return float(d.min()) if d.size else 0.0


# ---------------------------------------------------------------- 判定
def _static_mask(rec):
    return np.abs(np.linalg.norm(rec.acc, axis=1) - 9.81) < 0.3


def _corr(a, b):
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    return float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def decide_quat_order(rec):
    """靜態樣本上，哪種順序解出的重力方向與加速度計最接近。"""
    m = _static_mask(rec)
    if m.sum() < 10:
        m = np.ones(rec.n, bool)
    ga = gravity_from_acc(rec.acc[m])
    best, errs = None, {}
    for o in ORDERS:
        gq = gravity_from_quat(real_obs.quat_wxyz(rec.quat_raw[m], o))
        errs[o] = float(np.degrees(np.arccos(np.clip((gq * ga).sum(1), -1, 1))).mean())
        if best is None or errs[o] < errs[best]:
            best = o
    return best, errs


def analyse(flat, dance, turn, seen_first, turn_first):
    out = {"schema": "imu_check/1",
           "sources": {"flat": flat.path, "dance": dance.path,
                       "turn": turn.path if turn else None}}

    # 1. quat 順序：flat 與 dance 各自判，兩邊要一致
    order, errs = decide_quat_order(dance)
    order_f, errs_f = decide_quat_order(flat)
    out["quat_order"] = order
    out["quat_order_err_deg"] = {"dance": errs, "flat": errs_f}
    out["quat_order_consistent"] = (order == order_f)

    Qd = real_obs.quat_wxyz(dance.quat_raw, order)

    # 2. 平放：兩種順序的 roll/pitch、acc 模長/單位、gyro 偏置與雜訊、關節量化
    rp_both = {}
    for o in ORDERS:
        r, p = rp_from_gravity(gravity_from_quat(real_obs.quat_wxyz(flat.quat_raw, o)))
        rp_both[o] = [float(r.mean()), float(p.mean())]
    an = float(np.linalg.norm(flat.acc, axis=1).mean())
    r_acc, p_acc = rp_from_gravity(gravity_from_acc(flat.acc))
    out["flat"] = {
        "rp_quat_deg": rp_both,
        "rp_acc_deg": [float(r_acc.mean()), float(p_acc.mean())],
        "acc_norm": an,
        "acc_unit": ("m/s2" if 8.5 < an < 11 else ("g" if 0.9 < an < 1.1 else "unknown")),
        "gyro_bias_raw": flat.gyro.mean(0).tolist(),
        "gyro_noise_raw": flat.gyro.std(0).tolist(),
        "q_quant": float(np.median([quant_step(flat.j[n]["q"]) for n in real_obs.LEG_NAMES])),
        "v_quant": float(np.median([quant_step(flat.j[n]["v"]) for n in real_obs.LEG_NAMES])),
        "q_noise": float(np.mean([flat.j[n]["q"].std() for n in real_obs.LEG_NAMES])),
        "v_noise": float(np.mean([flat.j[n]["v"].std() for n in real_obs.LEG_NAMES])),
    }

    # 3. 跳舞：gyro 擬合
    om = omega_body(Qd, dance.t)
    bias = np.asarray(out["flat"]["gyro_bias_raw"])
    fit = fit_gyro(dance.gyro[:-1] - bias, om, win=max(1, int(round(0.02 * dance.hz))))
    scale, unit = classify_scale(fit["k"])
    scale = np.asarray(scale, float)
    out["gyro_k"] = fit["k"].tolist()
    out["gyro_corr"] = fit["corr"].tolist()
    out["gyro_corr_matrix"] = fit["corr_matrix"].tolist()
    out["gyro_axis_swap"] = bool(np.any(
        np.argmax(np.abs(fit["corr_matrix"]), axis=1) != np.arange(3)))
    out["gyro_scale"] = scale.tolist()
    out["gyro_unit"] = unit
    out["gyro_bias_rad_s"] = (bias * scale).tolist()
    out["gyro_noise_rad_s"] = (np.asarray(out["flat"]["gyro_noise_raw"]) * np.abs(scale)).tolist()

    # 4. 跳舞：IMU 姿態 vs 腿 FK 姿態（四輪共面才算）
    Pd = real_obs.ctrl_pos(dance)
    g_fk, resid = gravity_from_fk(Pd)
    ok = resid < 10.0
    r_q, p_q = rp_from_gravity(gravity_from_quat(Qd))
    r_f, p_f = rp_from_gravity(g_fk)
    out["rp_quat_vs_fk"] = {
        "n_coplanar": int(ok.sum()), "n": int(ok.size),
        "bias_deg": ([float((r_q - r_f)[ok].mean()), float((p_q - p_f)[ok].mean())]
                     if ok.any() else [float("nan")] * 2),
        "corr": ([_corr(r_q[ok], r_f[ok]), _corr(p_q[ok], p_f[ok])]
                 if ok.sum() > 10 else [float("nan")] * 2),
        "coplanar_resid_mm_median": float(np.median(resid)),
    }
    out["imu_mount_rp_deg"] = out["rp_quat_vs_fk"]["bias_deg"]

    # 5. 更新率（跳舞段；關節只看在動的樣本）
    vmask = np.abs(real_obs.ctrl_vel(dance)).max(1) > 0.05
    out["rates"] = {
        "imu_gyro": update_rate(dance.gyro, dance.hz),
        "imu_quat": update_rate(dance.quat_raw, dance.hz),
        "imu_acc": update_rate(dance.acc, dance.hz),
        "joint_q": update_rate(Pd, dance.hz, vmask),
        "joint_v": update_rate(real_obs.ctrl_vel(dance), dance.hz, vmask),
        "record_hz": dance.hz,
    }

    # 6. 第一個 roll 事件（給眼睛對）
    idx = np.flatnonzero(np.abs(r_q) > 2.0)
    if idx.size:
        i0 = int(idx[0])
        seg = r_q[i0:min(i0 + int(dance.hz), len(r_q))]
        j = i0 + int(np.argmax(np.abs(seg)))
        side = "右低" if r_q[j] > 0 else "左低"
        out["first_roll"] = {"t": float(dance.t[j]), "roll_deg": float(r_q[j]), "side": side}
        if seen_first:
            expect = "左低" if seen_first == "left" else "右低"
            out["first_roll"]["seen"] = expect
            out["first_roll"]["consistent"] = (expect == side)
    else:
        out["first_roll"] = None

    # 7. 轉向：第一段 |ω_z| > 0.15 rad/s 的平均符號
    if turn is not None:
        wz = ((turn.gyro - bias) * scale)[:, 2]
        act = np.flatnonzero(np.abs(wz) > 0.15)
        if act.size:
            i0 = int(act[0])
            seg = wz[i0:min(i0 + int(turn.hz * 0.5), len(wz))]
            sgn = int(np.sign(seg.mean()))
            out["turn"] = {"t": float(turn.t[i0]), "gyro_z_sign": sgn,
                           "dir": "左轉(逆時針)" if sgn > 0 else "右轉(順時針)"}
            if turn_first:
                out["turn"]["consistent"] = ((sgn > 0) == (turn_first == "left"))
        else:
            out["turn"] = None
    return out


def _print(out):
    f = out["flat"]
    print("═══ IMU 判讀 ═══")
    print(f"quat 順序          → {out['quat_order']}  "
          f"（靜態重力方向誤差° dance {out['quat_order_err_deg']['dance']}"
          f" / flat {out['quat_order_err_deg']['flat']}；兩段一致 {out['quat_order_consistent']}）")
    print(f"平放 roll/pitch°    xyzw {np.round(f['rp_quat_deg']['xyzw'], 2).tolist()}  "
          f"wxyz {np.round(f['rp_quat_deg']['wxyz'], 2).tolist()}  acc {np.round(f['rp_acc_deg'], 2).tolist()}")
    print(f"acc 模長 {f['acc_norm']:.3f} → {f['acc_unit']}")
    print(f"gyro k={np.round(out['gyro_k'], 3).tolist()} corr={np.round(out['gyro_corr'], 3).tolist()}"
          f" → 單位 {out['gyro_unit']}  scale {np.round(out['gyro_scale'], 5).tolist()}"
          f"  軸對調 {out['gyro_axis_swap']}")
    print(f"gyro 偏置 rad/s {np.round(out['gyro_bias_rad_s'], 4).tolist()}"
          f"  雜訊 std {np.round(out['gyro_noise_rad_s'], 4).tolist()}")
    q = out["rp_quat_vs_fk"]
    print(f"IMU vs 腿FK  偏置° {np.round(q['bias_deg'], 2).tolist()}  corr {np.round(q['corr'], 3).tolist()}"
          f"  共面樣本 {q['n_coplanar']}/{q['n']}（殘差中位 {q['coplanar_resid_mm_median']:.1f} mm）")
    print("更新率下界 Hz  " + "  ".join(f"{k} {v['hz_min']:.0f}" for k, v in out["rates"].items()
                                      if isinstance(v, dict)))
    print(f"關節量化 q {f['q_quant']:.5f} rad  v {f['v_quant']:.4f} rad/s；"
          f"靜態雜訊 q {f['q_noise']:.5f} v {f['v_noise']:.4f}")
    print(f"第一個 roll 事件    {out['first_roll']}")
    print(f"轉向                {out.get('turn')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="階 I：IMU 判讀")
    ap.add_argument("--flat", type=Path, required=True)
    ap.add_argument("--dance", type=Path, required=True)
    ap.add_argument("--turn", type=Path)
    ap.add_argument("--seen-first", choices=("left", "right"), help="眼睛看到跳舞先往哪側低")
    ap.add_argument("--turn-first", choices=("left", "right"), help="原地轉先轉哪邊")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs" / "imu_check.json")
    a = ap.parse_args()
    out = analyse(m6_rec.load(a.flat), m6_rec.load(a.dance),
                  m6_rec.load(a.turn) if a.turn else None, a.seen_first, a.turn_first)
    _print(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
