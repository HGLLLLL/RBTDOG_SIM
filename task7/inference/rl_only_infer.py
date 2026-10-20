#!/usr/bin/env python3
"""同事的純 RL 步態（Isaac Lab，`RL_only/`）在我們的 MuJoCo 模型裡重現。

介面照 `RL_only/policy_io.json`：obs 53 = ang_vel(3) | projected_gravity(3) | cmd(3) | joint_pos−default(12) | joint_vel(16) | last_action(16)，
關節序是 PhysX 的「按種類分組」（ABAD×4, HIP×4, KNEE×4, FOOT×4；腿序 FAR, FBL, RAR, RBL）；動作 16 = 0.25·a[:12]+default（腿 PD 目標）、5.0·a[12:]（輪速）。
網路：ONNX 53→512→256→128→16，ELU，無正規化，輸出直接用。

用法：
  conda run -n rbtdog --no-capture-output python task7/inference/rl_only_infer.py            # 預設一組指令、實體模型（含關節摩擦）
  ... --ideal                     # 關節摩擦／慣量歸零 ＝ 他的 IdealPDActuator（訓練條件）
  ... --cmds "0.5,0,0;0,0.5,0;0,0,1" --secs 8 --video task7/outputs/rl_only/run.mp4
  ... --npz policy_rl16.npz --self-collision   # 自訓 policy（狗端 npz，含 act_gain）＋輪輪 pair 場景；輸出到 outputs/rl_wheel/，多「後輪 gap」欄
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wheel_terrain as wt  # noqa: E402
sys.path.insert(0, str(HERE.parent / "realbot"))
from rl_obs_rl import grav_fix_matrix  # noqa: E402   # 重力歸零矩陣與狗端同一份（--grav-bias 注入、--grav-offset 歸零）
RL = HERE.parent.parent / "RL_only"
OUT = HERE.parent / "outputs" / "rl_only"


def quat_to_R(q):  # wxyz → 3×3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rp_from_R(R):
    roll = math.atan2(R[2, 1], R[2, 2]); pitch = math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))
    return math.degrees(roll), math.degrees(pitch)


def segments(c, thr, dt, lo=None, min_s=0.06):
    lo = thr - 3.0 if lo is None else lo
    out, cur = [], []
    for v in c:
        if (not cur and v > thr) or (cur and v > lo):
            cur.append(v)
        elif cur:
            if len(cur) * dt >= min_s:
                out.append(max(cur))
            cur = []
    if cur and len(cur) * dt >= min_s:
        out.append(max(cur))
    return out


def landing_vz(clr, dt, thr=0.005):
    """落地衝擊（v1c 及格線用）：每腿「淨空由 > thr 降到 ≤ thr」算一次觸地，取觸地前一步的下落速度 (clr[t−2] − clr[t−1]) / dt（m/s，負值截 0）。
    clr：(T, 4) m。回傳 (事件平均 vz, 事件數)；與 WheelEnv 的 impact 項同口徑（觸地前一步的 vz）。"""
    ev = []
    for k in range(clr.shape[1]):
        c = clr[:, k]
        for t in range(2, len(c)):
            if c[t] <= thr and c[t - 1] > thr:
                ev.append(max(0.0, float(c[t - 2] - c[t - 1]) / dt))
    return (float(np.mean(ev)) if ev else 0.0), len(ev)


class Sim:
    def __init__(self, io, dt=0.005, ideal=False, wheel_kv=1.0, scene=None, grav_bias=(0.0, 0.0), grav_offset=(0.0, 0.0)):
        import mujoco, max_model as mm
        self.mj = mujoco
        # ★ 重力向量的 sim2real 項（2026-10-14，trip26 平移前漂）：`grav_bias` 模擬 IMU 安裝偏差——機身水平時讀 (roll, pitch)，
        #   policy 看到 g = R_biasᵀ·g_true；`grav_offset` 是狗端 `--grav-offset` 的同一個修正 g' = R_fix·g。兩者同值就互相抵消。
        self.grav_bias_T = grav_fix_matrix(*grav_bias).T
        self.grav_fix = grav_fix_matrix(*grav_offset)
        self.m = mujoco.MjModel.from_xml_path(scene or mm.SCENE_MJX_V3F)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = dt
        self.sub = int(round(io["control"]["policy_dt"] / dt))
        names = io["joints"]["names"]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        assert all(j >= 0 for j in jid), "joint names not found in MJCF"
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid]); self.vadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        self.lo12 = self.m.jnt_range[jid[:12], 0]; self.hi12 = self.m.jnt_range[jid[:12], 1]
        trn = {int(self.m.actuator_trnid[a, 0]): a for a in range(self.m.nu)}
        self.aid = np.array([trn[j] for j in jid])                # actuator index in PhysX joint order
        self.default = np.array(io["joints"]["default_pos"], float)
        self.scale = np.array(io["action"]["scale"], float); self.offset = np.array(io["action"]["offset"], float)
        self.nleg, self.nwheel = 12, 4
        # 輪子 velocity actuator：kv → 他的 1.0（我們的 v3f 是 0.1）
        for a in self.aid[12:]:
            self.m.actuator_gainprm[a, 0] = wheel_kv; self.m.actuator_biasprm[a, 2] = -wheel_kv
        if ideal:
            for v in self.vadr:
                self.m.dof_frictionloss[v] = 0.0; self.m.dof_armature[v] = 0.0; self.m.dof_damping[v] = 0.0
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{p}_FOOT_LINK") for p in ("FAR", "FBL", "RAR", "RBL")]
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.base_geoms = set(np.where(self.m.geom_bodyid == self.base)[0].tolist())
        self.wheel_r = 0.0961

    def reset(self, seed=0):
        m, d, mj = self.m, self.d, self.mj
        mj.mj_resetData(m, d)
        d.qpos[:] = 0; d.qpos[3] = 1.0
        d.qpos[self.qadr] = self.default
        d.qpos[2] = 0.0; mj.mj_forward(m, d)
        zmin = min(d.xpos[b][2] for b in self.feet) - self.wheel_r
        d.qpos[2] = -zmin + 0.003
        d.qvel[:] = 0; d.ctrl[:] = 0
        d.ctrl[self.aid[:12]] = self.default[:12]
        mj.mj_forward(m, d)
        self.last_a = np.zeros(16)

    def obs(self, cmd):
        d = self.d
        R = quat_to_R(d.qpos[3:7])
        grav = self.grav_fix @ (self.grav_bias_T @ (R.T @ np.array([0.0, 0.0, -1.0])))
        ang = d.qvel[3:6].copy()                       # free joint 角速度＝機身座標
        jpos = d.qpos[self.qadr[:12]] - self.default[:12]
        jvel = d.qvel[self.vadr]
        return np.concatenate([ang, grav, np.asarray(cmd, float), jpos, jvel, self.last_a]).astype(np.float32), R

    def apply(self, a):
        tgt = a * self.scale + self.offset
        self.d.ctrl[self.aid] = tgt
        self.last_a = np.asarray(a, float).copy()
        for _ in range(self.sub):
            self.mj.mj_step(self.m, self.d)
        return tgt

    def base_contact(self):
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self.base_geoms and c.geom2 == self.floor) or (c.geom2 in self.base_geoms and c.geom1 == self.floor):
                return True
        return False


NOISE = dict(ang=0.2, grav=0.05, jpos=0.01, jvel=1.5)   # policy_io.json 的 UniformNoiseCfg（±）


def load_policy(onnx: str | None, npz: str | None):
    """同介面：obs(53,) → 他的尺度動作(16,)。npz 走狗端 policy_np（含 act_gain）；onnx 走 onnxruntime。"""
    if npz:
        sys.path.insert(0, str(HERE.parent / "realbot"))
        import policy_np as pn
        P = pn.load(npz)
        assert P.layout == "rl16" and P.obs_dim == 53, (P.layout, P.obs_dim)
        return lambda o: P.infer(o)
    if onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]); name = sess.get_inputs()[0].name
        return lambda o: sess.run(None, {name: np.asarray(o, np.float32)[None]})[0][0].astype(float)
    raise ValueError("要給 --onnx 或 --npz")


def rear_gap(sim):
    """後輪（RAR、RBL）實體間隙近似（官方網格內面 +0.017 ×2；傾斜時高估 ~7 mm）＝ 輪心距 − 0.034 m。
    與 V 文件口徑相同（link 中心距 67 mm → 實體間隙 ~33 mm）；不是 v3f 碰撞圓柱的間距（圓柱貼在外側面 y −0.047）。"""
    d = sim.d
    return float(np.linalg.norm(d.xpos[sim.feet[2]] - d.xpos[sim.feet[3]]) - 0.034)


def run(sim, pol, cmd, secs, settle, dt_pol=0.02, push=0.0, delay=0, noise=False, seed=0):
    """delay：動作延遲步數（實機 des→q 30–35 ms ≈ 1–2 步）；noise：訓練時的 obs 均勻雜訊；push：每 3 s 給機身一次 ±push m/s 的水平速度。"""
    sim.reset()
    rng = np.random.default_rng(seed)
    n_settle, n = int(settle / dt_pol), int(secs / dt_pol)
    rec = {k: [] for k in ("t", "vb", "ang", "rp", "clr", "tau", "a", "q", "v", "z", "sat", "tgt", "err", "gap")}
    fell = None
    queue = [np.zeros(16) for _ in range(delay)]
    for i in range(n_settle + n):
        c = np.zeros(3) if i < n_settle else np.asarray(cmd, float)
        o, R = sim.obs(c)
        if noise:
            o = o.copy()
            o[0:3] += rng.uniform(-NOISE["ang"], NOISE["ang"], 3); o[3:6] += rng.uniform(-NOISE["grav"], NOISE["grav"], 3)
            o[9:21] += rng.uniform(-NOISE["jpos"], NOISE["jpos"], 12); o[21:37] += rng.uniform(-NOISE["jvel"], NOISE["jvel"], 16)
        a = np.asarray(pol(o), float)
        if delay:
            queue.append(a); a_use = queue.pop(0)
        else:
            a_use = a
        if push and i >= n_settle and (i - n_settle) % int(3.0 / dt_pol) == int(1.0 / dt_pol):
            sim.d.qvel[0:2] += rng.uniform(-push, push, 2)
        tgt = sim.apply(a_use)
        sim.last_a = a.copy()                          # obs 的 last_action ＝ policy 自己上一步的原始輸出（Isaac 沒有延遲，實機 policy 也只看得到自己的輸出）
        d = sim.d
        vb = R.T @ d.qvel[0:3]
        rec["t"].append(i * dt_pol); rec["vb"].append(vb); rec["ang"].append(d.qvel[3:6].copy()); rec["rp"].append(rp_from_R(R))
        rec["clr"].append([d.xpos[b][2] - sim.wheel_r for b in sim.feet])
        rec["tau"].append(d.actuator_force[sim.aid].copy()); rec["a"].append(a.copy())
        rec["q"].append(d.qpos.copy()); rec["v"].append(d.qvel[sim.vadr].copy()); rec["z"].append(d.qpos[2])
        rec["sat"].append(int(np.sum((tgt[:12] < sim.lo12) | (tgt[:12] > sim.hi12))))
        rec["tgt"].append(tgt.copy()); rec["err"].append(np.abs(tgt[:12] - d.qpos[sim.qadr[:12]]))
        rec["gap"].append(rear_gap(sim))
        r, p = rec["rp"][-1]
        if fell is None and (sim.base_contact() or abs(r) > 60 or abs(p) > 60 or d.qpos[2] < 0.15):
            fell = i * dt_pol
    R = {k: np.array(v) for k, v in rec.items()}
    R["fell"] = fell; R["n_settle"] = n_settle
    return R


def metrics(R, cmd, dt_pol=0.02):
    s = R["n_settle"]; w = slice(s + int(1.0 / dt_pol), None)   # 指令後 1 s 起算
    vb, ang, rp, clr, tau, a = R["vb"][w], R["ang"][w], R["rp"][w], R["clr"][w], R["tau"][w], R["a"][w]
    g = dict(vx=vb[:, 0].mean(), vy=vb[:, 1].mean(), yaw=math.degrees(ang[:, 2].mean()), yaw_std=math.degrees(ang[:, 2].std()),
             roll_std=rp[:, 0].std(), roll_max=np.abs(rp[:, 0]).max(), pitch_std=rp[:, 1].std(), z=R["z"][w].mean())
    lifts, steps = [], []
    for k in range(4):
        c = clr[:, k] * 1000.0
        ground = np.min(clr, 1) * 1000.0
        sg = segments(c - ground, 8.0, dt_pol)
        lifts.append(float(np.median(sg)) if sg else 0.0); steps.append(len(sg) / (len(c) * dt_pol))
    g["lift"] = [round(x, 1) for x in lifts]; g["steps_s"] = [round(x, 2) for x in steps]
    g["tau_abad"] = np.abs(tau[:, 0:4]).max(); g["tau_hip"] = np.abs(tau[:, 4:8]).max(); g["tau_knee"] = np.abs(tau[:, 8:12]).max(); g["tau_wheel"] = np.abs(tau[:, 12:16]).max()
    g["a_abs"] = np.abs(a).mean(); g["a_max"] = np.abs(a).max(); g["a_gt1"] = 100.0 * np.mean(np.abs(a) > 1.0)
    g["wheel_v"] = np.abs(R["v"][w][:, 12:16]).max()
    g["jv_max"] = np.abs(R["v"][w][:, :12]).max()
    g["fell"] = R["fell"]; g["sat"] = 100.0 * np.mean(R["sat"][w] > 0)
    err = R["err"][w]; g["err_abad"] = err[:, 0:4].max(); g["err_hk"] = err[:, 4:12].max()
    dq = np.abs(np.diff(R["tgt"][w][:, :12], axis=0)) / dt_pol; g["vcmd"] = dq.max()
    g["land_vz"], g["land_n"] = landing_vz(R["clr"][w], dt_pol)
    g["gap_rr"] = float(R["gap"][w].min() * 1000.0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(RL / "2026-09-24_13-23-11.onnx"))
    ap.add_argument("--npz", default="", help="自訓 policy 的狗端 npz（policy_np，含 act_gain）；給了就不用 --onnx，輸出到 outputs/rl_wheel/")
    ap.add_argument("--self-collision", action="store_true", dest="self_collision", help="載入含輪輪 pair 的場景（後輪互碰才看得到）")
    ap.add_argument("--io", default="", help="policy_io.json；空＝--npz 時用 inference/rl_wheel_io.json（RL_only/ 不在 git），否則 RL_only/policy_io.json")
    ap.add_argument("--cmds", default="0,0,0;0.5,0,0;1.0,0,0;-0.5,0,0;0,0.5,0;0,-0.5,0;0,0,0.8;0,0,-0.8;0.5,0,0.5")
    ap.add_argument("--secs", type=float, default=8.0); ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005); ap.add_argument("--ideal", action="store_true")
    ap.add_argument("--wheel-kv", type=float, default=1.0)
    ap.add_argument("--delay", type=int, default=0); ap.add_argument("--noise", action="store_true"); ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--grav-bias", type=float, nargs=2, default=[0.0, 0.0], metavar=("ROLL", "PITCH"), dest="grav_bias",
                    help="注入 IMU 安裝偏差（機身水平時讀到的 roll°, pitch°；實機 trip24–26 站立 +0.7／−1.5）")
    ap.add_argument("--grav-offset", type=float, nargs=2, default=[0.0, 0.0], metavar=("ROLL", "PITCH"), dest="grav_offset",
                    help="狗端 M9 --grav-offset 的同一個歸零修正；與 --grav-bias 同值就抵消")
    ap.add_argument("--video", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--mesh", action="store_true", help="影片用官方 STL 網格模型渲染（scene_flat.xml；物理仍是 v3f 模型，只換外觀）")
    ap.add_argument("--title", default="", help="影片標題前綴")
    ap.add_argument("--video-secs", type=float, default=0.0, dest="video_secs", help="影片每個指令只放幾秒（0＝整段，含站定）")
    ap.add_argument("--video-start", type=float, default=2.0, dest="video_start", help="影片每段從第幾秒開始（含 --settle 那 1 s）")
    ap.add_argument("--names", default="", help="影片各段顯示名稱，分號分隔，與 --cmds 一一對應（空＝印指令）")
    a = ap.parse_args()
    a.io = a.io or str(HERE / "rl_wheel_io.json" if a.npz else RL / "policy_io.json")
    print("io:", a.io)
    io = json.load(open(a.io)); pol = load_policy(a.onnx if not a.npz else None, a.npz)
    if a.npz:
        a.tag = a.tag or Path(a.npz).stem
    out_dir = OUT.parent / "rl_wheel" if a.npz else OUT
    sim = Sim(io, dt=a.dt, ideal=a.ideal, wheel_kv=a.wheel_kv, scene=(wt.make_flat_pairs_scene_xml() if a.self_collision else None),
              grav_bias=a.grav_bias, grav_offset=a.grav_offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, trajs = [], {}
    for s in a.cmds.split(";"):
        cmd = [float(x) for x in s.split(",")]
        t0 = time.time(); R = run(sim, pol, cmd, a.secs, a.settle, push=a.push, delay=a.delay, noise=a.noise); g = metrics(R, cmd)
        name = f"vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f}"
        rows.append((name, cmd, g)); trajs[name] = R
        print(f"{name}: vx {g['vx']:+.2f} vy {g['vy']:+.2f} yaw {g['yaw']:+.0f}°/s roll std {g['roll_std']:.2f}/pk {g['roll_max']:.1f} z {g['z']:.3f} "
              f"lift {g['lift']} steps/s {g['steps_s']} τ abad/hip/knee/wheel {g['tau_abad']:.0f}/{g['tau_hip']:.0f}/{g['tau_knee']:.0f}/{g['tau_wheel']:.1f} "
              f"|a| {g['a_abs']:.2f} max {g['a_max']:.2f} >1 {g['a_gt1']:.0f}% wheel_v {g['wheel_v']:.1f} jv {g['jv_max']:.1f} "
              f"err abad/hk {g['err_abad']:.2f}/{g['err_hk']:.2f} vcmd {g['vcmd']:.1f} sat {g['sat']:.0f}% gap {g['gap_rr']:.0f} land {g['land_vz']:.2f}/{g['land_n']} fell {g['fell']}  ({time.time()-t0:.1f}s)")
    mode = "ideal" if a.ideal else "real"
    L = [f"# RL_only 純 RL 步態在 MuJoCo 重現（{mode}{'，' + a.tag if a.tag else ''}）", "",
         f"模型 `scene_flat_mjx_v3f.xml`（腿 kp 60/120/120 kd 1.0、輪 kv {a.wheel_kv}、forcerange 150/33）；子步 {a.dt} s × {sim.sub}；指令 {a.secs} s（先 {a.settle} s 零指令站定）；統計取指令後 1 s 起。",
         f"{'關節摩擦／慣量歸零（IdealPD）' if a.ideal else '含實機辨識的關節摩擦 1.85/1.5、輪 0.13'}；動作延遲 {a.delay} 步、obs 雜訊 {'開' if a.noise else '關'}、推力 ±{a.push} m/s；輪輪 pair {'開' if a.self_collision else '關'}；"
         f"IMU 偏差 roll {a.grav_bias[0]:+g}° pitch {a.grav_bias[1]:+g}°、重力歸零 {a.grav_offset[0]:+g}／{a.grav_offset[1]:+g}°；policy `{Path(a.npz or a.onnx).name}`。", "",
         "| 指令 | 摔 | vx | vy | 偏航 °/s（±std） | roll std/峰 ° | 機身高 | 抬腳 mm（FR,FL,RR,RL） | 步/秒 | τ ABAD/髖/膝/輪 | \\|a\\| 平均/最大/>1% | 輪速峰 | 關節速峰 | 誤差峰 ABAD/髖膝 | 命令速峰 | 飽和% | 後輪 gap mm | 落地 vz m/s（次） |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, cmd, g in rows:
        L.append(f"| {name} | {'%.1f s' % g['fell'] if g['fell'] is not None else '—'} | {g['vx']:+.2f} | {g['vy']:+.2f} | {g['yaw']:+.0f}（±{g['yaw_std']:.0f}） | {g['roll_std']:.2f}/{g['roll_max']:.1f} | {g['z']:.3f} | {g['lift']} | {g['steps_s']} | "
                 f"{g['tau_abad']:.0f}/{g['tau_hip']:.0f}/{g['tau_knee']:.0f}/{g['tau_wheel']:.1f} | {g['a_abs']:.2f}/{g['a_max']:.2f}/{g['a_gt1']:.0f} | {g['wheel_v']:.1f} | {g['jv_max']:.1f} | {g['err_abad']:.2f}/{g['err_hk']:.2f} | {g['vcmd']:.1f} | {g['sat']:.0f} | {g['gap_rr']:.0f} | {g['land_vz']:.2f}（{g['land_n']}） |")
    out = out_dir / f"eval_rl_only_{mode}{('_' + a.tag) if a.tag else ''}.md"; out.write_text("\n".join(L) + "\n", encoding="utf-8"); print("→", out)
    if a.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco, imageio
        from PIL import Image, ImageDraw, ImageFont
        import max_model as mm
        m = mujoco.MjModel.from_xml_path(mm.SCENE) if a.mesh else sim.m
        d = mujoco.MjData(m)
        # 兩個模型的 qpos 版面可能不同：按關節名把物理模型的 qpos 搬過去（free joint 7 個分量另抄）
        jmap = []
        free2 = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        for j in range(sim.m.njnt):
            if sim.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:          # 根關節多半沒名字：按類型對應（mj_name2id 吃到 None 會 segfault）
                if free2:
                    jmap.append((int(sim.m.jnt_qposadr[j]), int(m.jnt_qposadr[free2[0]]), 7))
                continue
            nm = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_JOINT, j)
            j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) if nm else -1
            if j2 < 0:
                continue
            jmap.append((int(sim.m.jnt_qposadr[j]), int(m.jnt_qposadr[j2]), 1))
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        m.vis.global_.offwidth, m.vis.global_.offheight = 960, 540
        r = mujoco.Renderer(m, 540, 960); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 3.0, 145.0, -20.0
        font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 28)
        font2 = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 20)
        wr = imageio.get_writer(a.video, fps=25, codec="libx264", quality=8, macro_block_size=None)
        labels = [x.strip() for x in a.names.split(";")] if a.names else [None] * len(rows)
        title = a.title or f"純 RL {mode}"
        for k, (name, cmd, g) in enumerate(rows):
            Q = trajs[name]["q"]; label = labels[k] or name
            i0, i1 = 0, len(Q)
            if a.video_secs > 0:
                i0 = min(int(a.video_start / 0.02), max(0, len(Q) - 2)); i1 = min(i0 + int(a.video_secs / 0.02), len(Q))
            for i in range(i0, i1, 2):
                for src, dst, n in jmap:
                    d.qpos[dst:dst + n] = Q[i][src:src + n]
                mujoco.mj_forward(m, d); cam.lookat[:] = [d.xpos[base][0], d.xpos[base][1], 0.35]
                r.update_scene(d, camera=cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im, "RGBA")
                dr.rectangle([0, 0, 960, 74], fill=(0, 0, 0, 150)); dr.text((14, 6), f"{k+1}/{len(rows)}  {title}  {label}", font=font, fill=(255, 255, 255))
                lift_txt = f"{min(g['lift']):.0f}–{max(g['lift']):.0f} mm" if max(g['lift']) > 0 else "0（純輪行）"
                dr.text((14, 42), f"t {(i-i0)*0.02:4.1f} s   {a.secs:.0f} s 平均：vx {g['vx']:+.2f}  vy {g['vy']:+.2f}  偏航 {g['yaw']:+.0f}°/s  roll std {g['roll_std']:.2f}°  膝峰 {g['tau_knee']:.0f} N·m  抬腳 {lift_txt}  摔 {'無' if g['fell'] is None else g['fell']}", font=font2, fill=(255, 235, 120))
                wr.append_data(np.asarray(im))
        wr.close(); print("→", a.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
