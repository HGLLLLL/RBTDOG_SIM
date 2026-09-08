#!/usr/bin/env python3
"""階 II：把實機錄到的 `des/kp/kd` 逐筆驅動 MuJoCo，與實機 obs 同時刻逐欄對照。

★ 為什麼是「錄檔驅動回放」而不是「模擬自己跑一遍步態」：
  M9 互動模式的步態是狗上即時算的（GaitStream），長度由人決定；重跑 CPG 要對齊相位。
  直接回放錄到的指令（`replay_standup.py` 的做法）→ 兩邊指令逐位元相同、時間軸相同，
  差異全部來自「感測器 + 物理」，正是我們要量的。站→走→趴整段都可比。

輸出：
  - `outputs/obs_noise_model.json`：訓練端直接讀（逐欄雜訊 std、延遲、範圍、quat 順序、gyro scale）
  - markdown 表（stdout / --md）：每欄 sim/real 的 mean、std、min、max、雜訊、相關、延遲

慣例：lag_ms > 0 表示**實機比模擬晚**。

用法：
    python obs_compare.py M6_walk.json --imu-check outputs/imu_check.json \
        [--md outputs/obs_compare_table.md] [--video outputs/obs_replay.mp4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco           # noqa: E402
import numpy as np      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import leg_kin          # noqa: E402
import m6_rec           # noqa: E402
import max_model as mm  # noqa: E402
import obs_max          # noqa: E402
import real_obs         # noqa: E402

TAU_MAX_LEG, TAU_MAX_WHEEL = 150.0, 40.0
GROUPS = ("gravity", "gyro", "joint_pos", "joint_vel")
_ORIGIN = np.stack([mm.SIDE_X * mm.HIP_X, mm.SIDE_Y * mm.HIP_Y, np.zeros(4)], axis=1)


# ---------------------------------------------------------------- 段落
def gait_segment(rec: m6_rec.Rec) -> slice:
    """步態段 = ABAD kp 是步態值（>0、<100）的最長連續區。站立/起身用 250，步態用 60。"""
    kp_abad = real_obs.kp12(rec)[:, 0::3].max(1)
    m = (kp_abad > 0) & (kp_abad < 100)
    best, cur = (0, 0), None
    for i, ok in enumerate(m):
        if ok and cur is None:
            cur = i
        if cur is not None and (not ok or i == len(m) - 1):
            end = i + 1 if ok else i
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    if best == (0, 0):
        return slice(0, rec.n)
    return slice(*best)


def initial_height(P0) -> float:
    """起始姿勢的機身高：最低輪心 + 輪半徑 + 1 cm 落穩餘裕。"""
    P = np.asarray(P0, float).reshape(4, 3)
    z = min(float((_ORIGIN[k] + leg_kin.fk(k, P[k]))[2]) for k in range(4))
    return -z + mm.WHEEL_RADIUS + 0.01


# ---------------------------------------------------------------- 回放
def sim_replay(rec: m6_rec.Rec, quat_order: str, gyro_scale, on_frame=None) -> dict:
    """錄到的 des/kp/kd 逐筆驅動 MuJoCo。回傳每筆的 obs、q、v、tau、quat(wxyz)、gyro、height。

    輪子：kd 照錄檔；kp>0（鎖輪）時鎖在**模擬自己**當下的輪角（實機的鎖定角是它自己的
    輪角，兩邊輪角本來就不同，抄實機的 des 會讓模擬輪子亂轉）。
    """
    des, KP, KD = real_obs.ctrl_des(rec), real_obs.kp12(rec), real_obs.kd12(rec)
    WKP, WKD = real_obs.wheel_kp(rec), real_obs.wheel_kd(rec)
    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)
    lo = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 0]
    hi = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 1]
    q0 = np.clip(real_obs.ctrl_pos(rec)[0], lo + 1e-4, hi - 1e-4)
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = q0
    d.qpos[2] = initial_height(q0)
    for _ in range(int(0.5 / m.opt.timestep)):          # 用第 0 筆的增益撐著落穩
        d.ctrl[:] = 0.0
        e = q0 - d.qpos[mm.LEG_QPOS_IDX]
        d.ctrl[mm.LEG_ACT_IDX] = np.clip(KP[0] * e - KD[0] * d.qvel[mm.LEG_QVEL_IDX],
                                         -TAU_MAX_LEG, TAU_MAX_LEG)
        mujoco.mj_step(m, d)

    n = rec.n
    dts = np.diff(rec.t)
    dts = np.append(dts, dts[-1] if dts.size else 0.005)
    c0, cmd0, a0 = real_obs.zero_cpg(), np.zeros(2), np.zeros(obs_max.ACT_DIM)
    out = {k: np.zeros((n, s)) for k, s in
           (("obs", obs_max.OBS_DIM), ("q", 12), ("v", 12), ("tau", 12),
            ("quat", 4), ("gyro", 3))}
    out["height"] = np.zeros(n)
    latch = d.qpos[mm.WHEEL_QPOS_IDX].copy()
    prev_wkp = np.zeros(4)
    for i in range(n):
        nsub = max(1, int(round(dts[i] / m.opt.timestep)))
        newly = (WKP[i] > 0) & (prev_wkp <= 0)
        latch[newly] = d.qpos[mm.WHEEL_QPOS_IDX][newly]
        prev_wkp = WKP[i]
        for _ in range(nsub):
            e = des[i] - d.qpos[mm.LEG_QPOS_IDX]
            tau = np.clip(KP[i] * e - KD[i] * d.qvel[mm.LEG_QVEL_IDX],
                          -TAU_MAX_LEG, TAU_MAX_LEG)
            d.ctrl[mm.LEG_ACT_IDX] = tau
            ew = latch - d.qpos[mm.WHEEL_QPOS_IDX]
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(
                WKP[i] * ew - WKD[i] * d.qvel[mm.WHEEL_QVEL_IDX], -TAU_MAX_WHEEL, TAU_MAX_WHEEL)
            mujoco.mj_step(m, d)
        out["obs"][i] = obs_max.build_obs(d, c0, cmd0, a0)
        out["q"][i] = d.qpos[mm.LEG_QPOS_IDX]
        out["v"][i] = d.qvel[mm.LEG_QVEL_IDX]
        out["tau"][i] = d.ctrl[mm.LEG_ACT_IDX]
        out["quat"][i] = d.qpos[3:7]
        out["gyro"][i] = d.qvel[3:6]
        out["height"][i] = d.qpos[2]
        if on_frame is not None:
            on_frame(i, m, d)
    return out


# ---------------------------------------------------------------- 統計
def lag_ms(a, b, dt, max_ms=100.0) -> float:
    """a 相對 b 的延遲（ms）。正 = a 比 b 晚。用去均值正規化互相關的峰值。"""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return float("nan")
    L = int(round(max_ms / 1000 / dt))
    best, best_l = -np.inf, 0
    for lag in range(-L, L + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        c = (x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)
        if c > best:
            best, best_l = c, lag
    return best_l * dt * 1000.0


def noise_std(x) -> float:
    """白雜訊 std 的二階差分估計：std(x[i−1] − 2x[i] + x[i+1]) / √6。

    為什麼不用一階差分或移動中位數殘差：一階差分會被訊號斜率污染（200 Hz 下關節角
    每筆變 ~0.005 rad，比雜訊還大）；含自身的移動中位數殘差會把白雜訊壓到一半。
    二階差分抵消線性趨勢，對白雜訊無偏（Var = 6σ²）。訊號的曲率項在 200 Hz 下
    是 (2π f / 200)² 量級，1–2 Hz 的步態可忽略。
    """
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    d2 = x[:-2] - 2 * x[1:-1] + x[2:]
    return float(d2.std() / np.sqrt(6.0))


def _corr(a, b) -> float:
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    return float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def channel_names():
    names = []
    for grp, dim in obs_max.OBS_LAYOUT:
        if grp not in GROUPS:
            break
        if grp in ("joint_pos", "joint_vel"):
            names += [(grp, f"{grp[-3:]}:{real_obs.LEG_NAMES[i]}") for i in range(dim)]
        else:
            names += [(grp, f"{grp}_{'xyz'[i]}") for i in range(dim)]
    return names[:real_obs.SENSOR_DIM]


def channel_table(O_real, O_sim, seg: slice, dt: float) -> list[dict]:
    R, S = np.asarray(O_real, float)[seg], np.asarray(O_sim, float)[seg]
    rows = []
    for i, (grp, nm) in enumerate(channel_names()):
        r, s = R[:, i], S[:, i]
        nr, ns = noise_std(r), noise_std(s)
        rows.append({"idx": i, "group": grp, "name": nm,
                     "mean_real": float(r.mean()), "std_real": float(r.std()),
                     "min_real": float(r.min()), "max_real": float(r.max()),
                     "mean_sim": float(s.mean()), "std_sim": float(s.std()),
                     "min_sim": float(s.min()), "max_sim": float(s.max()),
                     "noise_real": nr, "noise_sim": ns,
                     "noise_est": float(np.sqrt(max(nr * nr - ns * ns, 0.0))),
                     "corr": _corr(r, s), "lag_ms": lag_ms(r, s, dt)})
    return rows


def des_to_q_latency(des, q, dt, seg: slice) -> list[float]:
    D, Q = np.asarray(des, float)[seg], np.asarray(q, float)[seg]
    return [lag_ms(Q[:, j], D[:, j], dt) for j in range(12)]


def joint_vel_report(rec: m6_rec.Rec, seg: slice) -> dict:
    """實機 velocity 欄位 vs 角度差分：雜訊倍率、相關、25 Hz 以上能量占比。"""
    P, V = real_obs.ctrl_pos(rec)[seg], real_obs.ctrl_vel(rec)[seg]
    t = rec.t[seg]
    dt = float(np.median(np.diff(t)))
    dq = np.gradient(P, t, axis=0)
    ratio, corr, frac = [], [], []
    for j in range(12):
        ratio.append(noise_std(V[:, j]) / max(noise_std(dq[:, j]), 1e-9))
        corr.append(_corr(V[:, j], dq[:, j]))
        x = V[:, j] - V[:, j].mean()
        ps = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), dt)
        frac.append(float(ps[f > 25.0].sum() / max(ps.sum(), 1e-12)))
    return {"noise_ratio_v_over_dq": ratio, "corr_v_dq": corr,
            "psd_frac_above_25hz": frac, "record_hz": 1.0 / dt}


def build_model(rec, rows, lat_des_q, jvr, quat_order, gyro_scale, imu_check_path) -> dict:
    def grp(g, key):
        return [r[key] for r in rows if r["group"] == g]
    seg = gait_segment(rec)
    imu_lag = float(np.nanmedian(grp("gravity", "lag_ms") + grp("gyro", "lag_ms")))
    jnt_lag = float(np.nanmedian(grp("joint_pos", "lag_ms")))
    return {"schema": "obs_noise/1", "source": rec.path, "label": rec.label,
            "imu_check": str(imu_check_path) if imu_check_path else None,
            "quat_order": quat_order, "gyro_scale": list(map(float, gyro_scale)),
            "segment": {"start_s": float(rec.t[seg.start]),
                        "stop_s": float(rec.t[seg.stop - 1]), "hz": rec.hz},
            "latency_ms": {"des_to_q": lat_des_q,
                           "des_to_q_median": float(np.nanmedian(lat_des_q)),
                           "real_vs_sim_joint_pos": jnt_lag, "real_vs_sim_imu": imu_lag,
                           "imu_vs_joint": imu_lag - jnt_lag},
            "noise_std": {g: grp(g, "noise_est") for g in GROUPS},
            "range_real": {g: [grp(g, "min_real"), grp(g, "max_real")] for g in GROUPS},
            "range_sim": {g: [grp(g, "min_sim"), grp(g, "max_sim")] for g in GROUPS},
            "std_real": {g: grp(g, "std_real") for g in GROUPS},
            "std_sim": {g: grp(g, "std_sim") for g in GROUPS},
            "corr": {g: grp(g, "corr") for g in GROUPS},
            "joint_vel": jvr}


def markdown_table(rows) -> str:
    L = ["| # | 欄位 | mean r/s | std r/s | min r/s | max r/s | 雜訊 r/s/est | corr | lag ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        moving = min(r["std_real"], r["std_sim"]) > 1e-3     # 幾乎沒動的欄位相關係數無意義
        flag = " ⚠️" if (r["corr"] < 0 and moving) else ""
        L.append(f"| {r['idx']} | {r['name']}{flag} | {r['mean_real']:+.3f}/{r['mean_sim']:+.3f} "
                 f"| {r['std_real']:.3f}/{r['std_sim']:.3f} "
                 f"| {r['min_real']:+.2f}/{r['min_sim']:+.2f} "
                 f"| {r['max_real']:+.2f}/{r['max_sim']:+.2f} "
                 f"| {r['noise_real']:.4f}/{r['noise_sim']:.4f}/{r['noise_est']:.4f} "
                 f"| {r['corr']:+.2f} | {r['lag_ms']:+.0f} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="階 II：錄檔驅動回放 + obs 逐欄對照")
    ap.add_argument("rec", type=Path)
    ap.add_argument("--imu-check", type=Path, help="imu_check.json：取 quat_order 與 gyro_scale")
    ap.add_argument("--quat-order", default=None, choices=("xyzw", "wxyz"))
    ap.add_argument("--gyro-scale", type=float, nargs=3, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs" / "obs_noise_model.json")
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None, help="回放錄影 mp4")
    a = ap.parse_args()

    quat_order, gyro_scale = "xyzw", (1.0, 1.0, 1.0)
    if a.imu_check:
        ic = json.loads(a.imu_check.read_text(encoding="utf-8"))
        quat_order, gyro_scale = ic["quat_order"], tuple(ic["gyro_scale"])
    if a.quat_order:
        quat_order = a.quat_order
    if a.gyro_scale:
        gyro_scale = tuple(a.gyro_scale)

    rec = m6_rec.load(a.rec)
    print(f"錄檔 {a.rec.name}  {rec.n} 筆 @ {rec.hz:.0f} Hz  {rec.t[-1]:.1f} s；"
          f"quat {quat_order}  gyro_scale {gyro_scale}")
    seg = gait_segment(rec)
    print(f"步態段 {rec.t[seg.start]:.2f} → {rec.t[seg.stop - 1]:.2f} s（{seg.stop - seg.start} 筆）")

    frames = []
    state = {"ren": None, "cam": None}

    def on_frame(i, m, d):
        if a.video is None or i % 8:
            return
        if state["ren"] is None:
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1280)
            m.vis.global_.offheight = max(m.vis.global_.offheight, 720)
            state["ren"] = mujoco.Renderer(m, 720, 1280)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance, cam.elevation, cam.azimuth = 2.6, -12, 135
            state["cam"] = cam
        state["cam"].lookat[:] = d.qpos[:3]
        state["ren"].update_scene(d, camera=state["cam"])
        frames.append(state["ren"].render())

    sim = sim_replay(rec, quat_order, gyro_scale, on_frame)
    print(f"回放完成：末機身高 {sim['height'][-1] * 1000:.0f} mm，"
          f"最低 {sim['height'].min() * 1000:.0f} mm，最高 {sim['height'].max() * 1000:.0f} mm")
    O_real = real_obs.obs_series(rec, quat_order, gyro_scale)
    dt = float(np.median(np.diff(rec.t)))
    rows = channel_table(O_real, sim["obs"], seg, dt)
    lat = des_to_q_latency(real_obs.ctrl_des(rec), real_obs.ctrl_pos(rec), dt, seg)
    lat_sim = des_to_q_latency(real_obs.ctrl_des(rec), sim["q"], dt, seg)
    jvr = joint_vel_report(rec, seg)
    md = markdown_table(rows)
    print(md)
    print(f"\ndes→q 延遲 ms（12 關節，MJCF 序）實機 {np.round(lat, 1).tolist()}  中位 {np.nanmedian(lat):.1f}")
    print(f"                              模擬 {np.round(lat_sim, 1).tolist()}  中位 {np.nanmedian(lat_sim):.1f}")
    print(f"joint_vel 雜訊倍率 v/dq 中位 {np.median(jvr['noise_ratio_v_over_dq']):.2f}；"
          f"corr 中位 {np.median(jvr['corr_v_dq']):.3f}；"
          f">25 Hz 能量占比中位 {np.median(jvr['psd_frac_above_25hz']):.3f}")
    model = build_model(rec, rows, lat, jvr, quat_order, gyro_scale, a.imu_check)
    model["latency_ms"]["des_to_q_sim"] = lat_sim
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {a.out}")
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
        print(f"→ {a.md}")
    if a.video and frames:
        import imageio.v2 as iio
        iio.mimsave(str(a.video), frames, fps=25, codec="libx264")
        print(f"🎬 {a.video}  {len(frames)} 幀")
    return 0


if __name__ == "__main__":
    sys.exit(main())
