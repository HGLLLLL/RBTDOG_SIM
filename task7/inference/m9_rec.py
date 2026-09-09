"""M9 步態 log（`M9_gait.py` 的 `M9_*.json`）→ `m6_rec.Rec`，讓 obs_compare 直接吃。

★ 為什麼要有這支：階 II「走路側錄」要的東西，9/3 成功那兩趟的 M9 log **本來就記了**
  —— 每個 200 Hz tick 的 12 關節 `(q, des, tau, v)`、同一刻讀的 IMU roll/pitch、
  phase 與 kp、四輪 `(pos, vel, tau)`。差別只是格式與座標系。
  轉過來就能離線做「我們增益下的 des→q 延遲、雜訊、回放對照」，不必為此再走一趟。

M9 log 與 M6 錄檔的差異（全部在這裡處理）：
  - `j` 的四個值是**控制器座標**（M9 寫入時已用 `coord.to_ctrl` / SIGN 轉過）；
    M6 格式是馬達座標 → 這裡用 `coord.to_motor` / SIGN 轉回去，讓下游走同一條路。
  - 只記了髖膝的 `kp`；**ABAD kp、kd、輪 kd 照 M9 自己的排程規則重建**
    （`phase_kp` / `phase_gains` / `wheel_kd_of`）。KP_DOWN/KP_UP 的進度 r 由該階段起始時間
    與 `args.kp_shift` 算；GAIT_IN/GAIT/GAIT_OUT 直接用 `args.kp_abad`／`args.kd`／`args.wheel_kd`。
  - IMU 只有 roll/pitch（0.01°）：quat 由 roll/pitch（yaw=0）重建、存成 xyzw；
    acc = −9.81·g_body；**gyro 是由重建的 quat 微分得來（20 ms 平滑）**，不是原始感測值 ——
    量級與延遲可用，雜訊估計**不可用**（要用 imu_check 從 M6 跳舞段量）。
  - 輪子：M9 永遠 kp=0、只有 kd；des 抄當下 pos（M9 就是這樣寫的）。

⚠️ M9 會跳過 stale 的 tick（沒 append），所以 t 有空洞；下游用實際 dt。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import coord            # noqa: E402
import m6_rec           # noqa: E402
import real_obs         # noqa: E402
import shm_io           # noqa: E402
from imu_check import omega_body, smooth  # noqa: E402

STANDUP_KP, STANDUP_KD, WHEEL_KD_SAFE = 250.0, 5.0, 0.5   # 與 M9_gait.py 相同
GAIT_PHASES = ("GAIT_IN", "GAIT", "GAIT_OUT")
GAIT_PHASES = tuple(GAIT_PHASES) + tuple("TELEOP_" + x for x in GAIT_PHASES) + ("TELEOP_STAND",)   # 遙控段用步態增益（2026-09-09）
D2R = np.pi / 180.0


def _phase_progress(samples, args) -> np.ndarray:
    """每筆在自己階段裡走了幾成（0~1）。只對 KP_DOWN / KP_UP 有意義（dur = kp_shift）。"""
    r = np.zeros(len(samples))
    t0, prev = None, None
    for i, s in enumerate(samples):
        if s["phase"] != prev:
            t0, prev = s["t"], s["phase"]
        if s["phase"] in ("KP_DOWN", "KP_UP"):
            r[i] = min(1.0, (s["t"] - t0) / max(args["kp_shift"], 1e-9))
    return r


def gains_of(phase: str, r: float, kp_now: float, args: dict) -> tuple:
    """(kp_abad, kd, wheel_kd)。照 M9 的 phase_kp / phase_gains / wheel_kd_of 重建。"""
    gkp, gkd, gab, wkd = args["kp"], args["kd"], args["kp_abad"], args["wheel_kd"]
    if phase in GAIT_PHASES:
        return gab, gkd, wkd
    w = min(wkd, WHEEL_KD_SAFE)
    if phase in ("RAMP_UP", "RAMP_DOWN"):
        return kp_now, STANDUP_KD, w                 # 兩者都是 STANDUP_KP·r
    if phase == "KP_DOWN":
        f = r
    elif phase == "KP_UP":
        f = 1.0 - r
    elif abs(gkp - STANDUP_KP) > 1e-9:               # 非互動的 HOLD_stand / BACK_crouch
        f = min(1.0, max(0.0, (kp_now - STANDUP_KP) / (gkp - STANDUP_KP)))
    else:
        f = 0.0
    kp_abad = STANDUP_KP + (gab - STANDUP_KP) * f
    kd = STANDUP_KD if abs(gkp - STANDUP_KP) < 1e-9 else STANDUP_KD + (gkd - STANDUP_KD) * f
    return kp_abad, kd, w


def rp_to_quat_wxyz(roll_deg, pitch_deg):
    """roll/pitch（度，yaw=0）→ wxyz。與 imu_check 的 rp_from_gravity 慣例互逆。"""
    r, p = np.asarray(roll_deg, float) * D2R / 2, np.asarray(pitch_deg, float) * D2R / 2
    cr, sr, cp, sp = np.cos(r), np.sin(r), np.cos(p), np.sin(p)
    return np.stack([cr * cp, sr * cp, cr * sp, -sr * sp], axis=-1)


def from_m9(d: dict, path: str = "") -> m6_rec.Rec:
    if d.get("schema", "").startswith("m6_"):
        raise ValueError("這是 M6 錄檔，直接用 m6_rec.load")
    S, args = d["samples"], d["args"]
    # ⚠️ M9 把 t 存到 0.001 s，追趕 tick 會出現同一毫秒兩筆 → dt=0 會讓下游微分除以零。
    #    留第一筆、丟後面的（兩筆 joint_state 讀值本來就幾乎相同）。
    seen, keep = set(), []
    for s_ in S:
        if s_["t"] not in seen:
            seen.add(s_["t"])
            keep.append(s_)
    n_dup = len(S) - len(keep)
    S = keep
    n = len(S)
    t = np.array([s["t"] for s in S], float)
    kp_now = np.array([s["kp"] for s in S], float)
    r = _phase_progress(S, args)
    G = np.array([gains_of(s["phase"], r[i], kp_now[i], args) for i, s in enumerate(S)])
    kp_abad, kd, wkd = G[:, 0], G[:, 1], G[:, 2]

    joints = {}
    for nm in shm_io.JOINTS:
        leg, kind = nm[:2], nm[2:]
        sg = coord.SIGN[kind][leg]
        if kind == coord.KIND_WHEEL:
            W = np.array([s["w"][nm] for s in S], float)          # (pos, vel, tau) 馬達座標
            joints[nm] = {"q": W[:, 0], "v": W[:, 1], "tau": W[:, 2], "des": W[:, 0],
                          "ff": np.zeros(n), "kp": np.zeros(n), "kd": wkd}
        else:
            J = np.array([s["j"][nm] for s in S], float)          # (q, des, tau, v) 控制器座標
            joints[nm] = {"q": coord.to_motor(nm, J[:, 0]), "des": coord.to_motor(nm, J[:, 1]),
                          "tau": sg * J[:, 2], "v": sg * J[:, 3], "ff": np.zeros(n),
                          "kp": kp_abad if kind == coord.KIND_HIP_ROLL else kp_now, "kd": kd}

    roll = np.array([s["roll"] for s in S], float)
    pitch = np.array([s["pitch"] for s in S], float)
    Q = rp_to_quat_wxyz(roll, pitch)
    from imu_check import gravity_from_quat
    g = gravity_from_quat(Q)
    acc = -9.81 * g
    hz = n / max(t[-1] - t[0], 1e-9)
    om = omega_body(Q, t)
    om = smooth(np.vstack([om, om[-1:]]), max(1, int(round(0.02 * hz))))
    imu = np.concatenate([acc, om, Q[:, [1, 2, 3, 0]]], axis=1)     # quat 存 xyzw，同 imu_central

    dd = {"schema": m6_rec.SCHEMA, "label": f"m9:{Path(path).stem}",
          "note": f"由 M9 log 轉換；gyro 為 roll/pitch 微分（非原始感測）；args kp {args['kp']} "
                  f"kp_abad {args['kp_abad']} kd {args['kd']} wheel_kd {args['wheel_kd']}",
          "n": n, "secs": float(t[-1] - t[0]), "hz_actual": hz, "t": (t - t[0]).tolist(),
          "joints": {nm: {k: v.tolist() for k, v in jj.items()} for nm, jj in joints.items()},
          "imu": imu.tolist()}
    rec = m6_rec.from_dict(dd, path)
    rec.phase = np.array([s["phase"] for s in S])       # 額外附上，下游可用
    rec.derived_imu = True      # gyro 是 roll/pitch 微分；**gyro_z 無效**（yaw 未記錄，設 0）
    rec.n_dup_dropped = n_dup
    return rec


def load(path) -> m6_rec.Rec:
    p = Path(path)
    return from_m9(json.loads(p.read_text(encoding="utf-8")), str(p))


def load_any(path) -> m6_rec.Rec:
    """M6 或 M9 都吃。"""
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema") == m6_rec.SCHEMA:
        return m6_rec.from_dict(d, str(p))
    return from_m9(d, str(p))


# 讓 real_obs 的 LEG_NAMES 在這裡也可見（下游偶爾需要）
LEG_NAMES = real_obs.LEG_NAMES
