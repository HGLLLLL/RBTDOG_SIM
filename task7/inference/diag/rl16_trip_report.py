#!/usr/bin/env python3
"""純 RL（rl16）實機 log 全面分析（2026-10-06 起；trip25 用）。

每趟：指令／步數／退回／飽和／時序；GAIT 段 roll、pitch；gyro_z 偏航率對指令；1 s 視窗偏航率（看「飄」）；
輪速（直走的速度估計＝四輪平均 × 0.0961）；FK 足端離地（口徑同 R／T 文件：足端 z − 四足最低，> 8 mm 遲滯段）；
ABAD 用量；力矩峰（跳點已剔）、誤差峰、關節速峰對護欄；前後輪實體間隙（官方網格，負＝穿模）。
用法：python rl16_trip_report.py task7/logs/m_logs_trip25 20261006_133934:直走0.3 20261006_134203:左平移0.5 …
"""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "realbot")); sys.path.insert(0, str(ROOT / "inference"))
from kin import fk  # noqa
HIP_X, HIP_Y, WHEEL_R = 0.2698, 0.065, 0.0961
LEGS = ["fl", "fr", "bl", "br"]; SIDE = {"fl": (+1, +1), "fr": (+1, -1), "bl": (-1, +1), "br": (-1, -1)}
MJ = ["fr", "fl", "br", "bl"]   # MJCF FAR,FBL,RAR,RBL 對應 shm 腿名


def foot_body(leg, q1, q2, q3, roll=0.0, pitch=0.0):
    sx, sy = SIDE[leg]; x, y, z = fk(leg, q1, q2, q3); x += sx * HIP_X; y += sy * HIP_Y
    if roll or pitch:
        r, p = math.radians(roll), math.radians(pitch)
        z = -math.sin(p) * x + math.sin(r) * math.cos(p) * y + math.cos(r) * math.cos(p) * z
    return x, y, z


def segments(c, thr, dt, lo=None, min_s=0.06):
    lo = thr - 3.0 if lo is None else lo; out, cur = [], []
    for v in c:
        if (not cur and v > thr) or (cur and v > lo): cur.append(v)
        elif cur:
            if len(cur) * dt >= min_s: out.append(max(cur))
            cur = []
    if cur and len(cur) * dt >= min_s: out.append(max(cur))
    return out


class WheelGap:
    def __init__(self):
        import mujoco, max_model as mm
        self.mj = mujoco; self.m = mujoco.MjModel.from_xml_path(mm.SCENE); self.d = mujoco.MjData(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link"); self.LEG = ["FAR", "FBL", "RAR", "RBL"]
        self.g = {lg: [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{lg}_FOOT_LINK") and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH] for lg in self.LEG}
        self.jq = {f"{lg}_{k}": self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{lg}_{k}_JOINT")] for lg in self.LEG for k in ("ABAD", "HIP", "KNEE")}
    def _verts(self, g):
        m, d = self.m, self.d; mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]; return (w - d.xpos[self.base]) @ d.xmat[self.base].reshape(3, 3)
    def gaps(self, r):
        d = self.d; d.qpos[:] = 0; d.qpos[3] = 1
        for lg, s in zip(self.LEG, MJ):
            for k, n in (("ABAD", "1_hip_roll"), ("HIP", "2_hip_pitch"), ("KNEE", "3_knee_pitch")): d.qpos[self.jq[f"{lg}_{k}"]] = r["j"][f"{s}{n}"][0]
        self.mj.mj_forward(self.m, d); V = {lg: np.vstack([self._verts(g) for g in self.g[lg]]) for lg in self.LEG}
        def gap(right, left):
            xr, xl = V[right][:, 0], V[left][:, 0]; lo, hi = max(xr.min(), xl.min()), min(xr.max(), xl.max())
            if hi <= lo: return 999.0
            return 1000 * (V[left][(xl >= lo) & (xl <= hi), 1].min() - V[right][(xr >= lo) & (xr <= hi), 1].max())
        return gap("FAR", "FBL"), gap("RAR", "RBL")


def analyze(path, name, wg=None):
    J = json.load(open(path)); S = J["samples"]; P = J["policy"]; L = P["log"]
    out = {"name": name, "stamp": Path(path).stem, "aborted": J["aborted"], "abort": J.get("abort_reason", ""), "steps": P["steps"], "fb": P["fallback_total"],
           "n_sat": P["n_sat"], "worst_ms": P["worst_ms"], "spikes": sum(J["spikes"].values()) if isinstance(J.get("spikes"), dict) else J.get("spikes")}
    act = [l for l in L if any(abs(c) > 1e-9 for c in l["cmd"]) and not l["ol"] and l["fb"] == ""]
    out["cmd"] = act[0]["cmd"] if act else [0, 0, 0]; out["n_act"] = len(act)
    if act:
        gz = np.array([l["imu"][1][2] for l in act]); out["yaw"] = math.degrees(gz.mean()); out["yaw_std"] = math.degrees(gz.std())
        # 1 s 視窗偏航率（看飄）
        out["yaw_1s"] = [round(math.degrees(gz[i:i + 50].mean()), 1) for i in range(0, len(gz) - 49, 50)]
        a = np.array([l["a_raw"] for l in act]); out["a_abs"] = float(np.abs(a).mean()); out["a_max"] = float(np.abs(a).max())
        vw = np.array([l["v_wheel"] for l in act]); out["wheel_mean"] = vw.mean(0).round(2).tolist(); out["wheel_pk"] = float(np.abs(vw).max())
        ms = np.array([l["ms"] for l in L]); out["ms_med"] = float(np.median(ms)); out["ms_gt10"] = int((ms > 10).sum())
        out["sat_steps"] = int(sum(1 for l in act if l["sat"]))
    G = [r for r in S if r["phase"] == "GAIT"]; R = [r for r in S if r["phase"] == "READY"]; dt = 0.005
    if G:
        roll = np.array([r["roll"] for r in G]); pitch = np.array([r["pitch"] for r in G])
        out["roll"] = (roll.mean(), roll.std(), roll[np.abs(roll).argmax()]); out["pitch"] = (pitch.mean(), pitch.std(), pitch[np.abs(pitch).argmax()])
        # FK 足端：挑讓最低兩足最接近的水平校正正負號（R 文件做法）
        def feet(r, sr, sp):
            return {lg: foot_body(lg, *[r["j"][f"{lg}{k}_{n}"][0] for k, n in ((1, "hip_roll"), (2, "hip_pitch"), (3, "knee_pitch"))], sr * r["roll"], sp * r["pitch"])[2] for lg in LEGS}
        best = None
        for sr in (+1, -1):
            for sp in (+1, -1):
                gap = np.mean([np.diff(sorted(feet(r, sr, sp).values()))[0] for r in G[::4]])
                if best is None or gap < best[0]: best = (gap, sr, sp)
        _, sr, sp = best; Z = np.array([[feet(r, sr, sp)[lg] for lg in LEGS] for r in G]) * 1000
        clr = Z - Z.min(1, keepdims=True); lifts, steps = [], []
        for k in range(4):
            sg = segments(clr[:, k], 8.0, dt); lifts.append(float(np.median(sg)) if sg else 0.0); steps.append(len(sg) / (len(G) * dt))
        out["lift"] = [round(x, 0) for x in lifts]; out["steps_s"] = [round(x, 2) for x in steps]; out["level"] = (sr, sp)
        # ABAD 用量、誤差峰、速度峰、力矩峰（用 J['peak'] 已剔跳點）
        ab = {lg: np.array([r["j"][f"{lg}1_hip_roll"][0] for r in G]) for lg in LEGS}; out["abad_pk"] = {lg: float(ab[lg][np.abs(ab[lg]).argmax()]) for lg in LEGS}
        err_ab = max(abs(r["j"][f"{lg}1_hip_roll"][0] - r["j"][f"{lg}1_hip_roll"][1]) for r in G for lg in LEGS)
        err_hk = max(abs(r["j"][f"{lg}{k}"][0] - r["j"][f"{lg}{k}"][1]) for r in G for lg in LEGS for k in ("2_hip_pitch", "3_knee_pitch"))
        jv = max(abs(r["j"][f"{lg}{k}"][3]) for r in G for lg in LEGS for k in ("1_hip_roll", "2_hip_pitch", "3_knee_pitch"))
        out["err"] = (err_ab, err_hk); out["jv_pk"] = jv
        pk = J.get("peak", {}); out["tau"] = {grp: max((abs(v) for k, v in pk.items() if k[2:] == grp), default=0) for grp in ("1_hip_roll", "2_hip_pitch", "3_knee_pitch")}
        out["tau"] = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in out["tau"].items()}
        if wg is not None:
            gp = np.array([wg.gaps(r) for r in G[::2]]); out["gap_f"] = float(gp[:, 0].min()); out["gap_r"] = float(gp[:, 1].min()); out["gap_r_lt30"] = float(100 * np.mean(gp[:, 1] < 30))
    return out


def fmt(o):
    c = o["cmd"]; name = f"vx {c[0]:+.1f} vy {c[1]:+.1f} wz {c[2]:+.1f}"
    if o["aborted"] or "roll" not in o: return f"| {o['name']} | {o['stamp'][-6:]} | {name} | **中止** {o['abort']} | | | | | | | | | | |"
    r, p = o["roll"], o["pitch"]; yaw = f"{o['yaw']:+.1f}（±{o['yaw_std']:.1f}）" if "yaw" in o else "—"
    ratio = f"{100*math.radians(o['yaw'])/c[2]:.0f}%" if abs(c[2]) > 0 and "yaw" in o else ""
    vx = f"{np.mean(o['wheel_mean'])*WHEEL_R:+.2f}" if "wheel_mean" in o else ""
    return (f"| {o['name']} | {o['stamp'][-6:]} | {name} | {o['steps']}／{o['fb']}／{o['n_sat']}／{o['worst_ms']:.1f} | {yaw} {ratio} | {vx}（{o.get('wheel_mean', '')}） | "
            f"{r[1]:.2f}／{r[2]:+.1f} | {p[1]:.2f}／{p[2]:+.1f} | {o['lift']} | {o['steps_s']} | {o['err'][0]:.2f}／{o['err'][1]:.2f} | "
            f"{o['tau']['1_hip_roll']:.0f}／{o['tau']['2_hip_pitch']:.0f}／{o['tau']['3_knee_pitch']:.0f} | {o['jv_pk']:.1f}／{o.get('wheel_pk', 0):.1f} | "
            f"{o.get('gap_r', float('nan')):.0f}／{o.get('gap_f', float('nan')):.0f}（{o.get('gap_r_lt30', 0):.0f}%） | {o.get('a_max', 0):.2f} |")


HDR = ("| 段 | stamp | 指令 | 步／退回／飽和／最慢 ms | 偏航 °/s（±std）比例 | 直走速度估 m/s（四輪 rad/s） | roll std／峰 | pitch std／峰 | 抬腳 mm（fl,fr,bl,br） | 步/秒 | 誤差峰 ABAD／髖膝 | τ 峰 ABAD／髖／膝 | 關節速峰／輪速峰 | 後／前輪間隙 mm（後<30%） | \\|a\\| 峰 |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

if __name__ == "__main__":
    d = Path(sys.argv[1]); runs = [a.split(":", 1) for a in sys.argv[2:]]
    wg = WheelGap(); rows = []
    for stamp, name in runs:
        o = analyze(d / f"M9_{stamp}.json", name, wg); rows.append(o); print(fmt(o))
        if "yaw_1s" in o and abs(o["cmd"][2]) > 0: print(f"    1 s 視窗偏航率 °/s：{o['yaw_1s']}   ABAD 峰 {dict((k, round(v, 2)) for k, v in o['abad_pk'].items())}")
        elif "abad_pk" in o: print(f"    ABAD 峰 {dict((k, round(v, 2)) for k, v in o['abad_pk'].items())}  偏航 1 s：{o.get('yaw_1s')}")
