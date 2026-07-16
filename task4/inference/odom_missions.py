"""odom vs compass 兩組驗證實驗（CPG-RL 論文版）。

  --exp line   ：直線 40m，compass vs odom 並排；地板畫目標線 y=0；2-panel 圖表(仿 exp4)。
  --exp course ：折線任務 直5/R45/直5/L90/直10/R90/直5/L45/直5，走完應回到初始延伸線；
                 並排、地板疊理想折線、2-panel 圖表。

兩隻狗獨立模擬、同一 RL 策略；差別只在回授修正：
  compass = 只用 MTi-680G 融合航向鎖航向(有噪、vy=0)；odom = 完美 odom 做航向+橫向(line_control)。
相機：正後上方 45°(方位角跟隨航向、俯角 -45°)。重用 compare/walk_line/local_infer_paper。

用法：
  MUJOCO_GL=egl python odom_missions.py --exp line   --params ../weights/cpg_rl_paper_params.pkl
  MUJOCO_GL=egl python odom_missions.py --exp course --params ../weights/cpg_rl_paper_params.pkl
"""
import argparse, os, sys, collections
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
from go2_gait import Go2Gait
from walk_line import GAIT, add_geom, dir_mat
import local_infer_paper as P
from PIL import Image, ImageDraw

CTRL_DT = 0.02
FPS = 25
W = H = 480
VX = 0.6
K_YAW = 3.0
K_CT = 1.5
OUT = "/home/huang/rbtdog_sim/task4/outputs"
# 折線任務：每段(絕對世界航向 deg, 長度 m)。右轉=負(世界+yaw=CCW=左)。
SEGS = [(0.0, 5.0), (-45.0, 5.0), (45.0, 10.0), (-45.0, 5.0), (0.0, 5.0)]
PATH_RGBA = [0.15, 0.85, 0.25, 1.0]


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


_FONTS = {}
def font(sz):
    if sz not in _FONTS:
        import matplotlib.font_manager as fm
        from PIL import ImageFont
        _FONTS[sz] = ImageFont.truetype(fm.findfont("DejaVu Sans"), sz)
    return _FONTS[sz]


# ---------------- 幾何（純函式，可單元測試） ----------------
def ideal_waypoints(segs, origin=(0.0, 0.0)):
    """由分段(航向,長度)推理想折線航點；回傳 (N+1, 2)。"""
    pts = [np.asarray(origin, float)]
    for hd, L in segs:
        d = np.array([np.cos(np.radians(hd)), np.sin(np.radians(hd))])
        pts.append(pts[-1] + L * d)
    return np.array(pts)


def point_to_polyline(p, pts):
    """點 p 到折線(pts 連續線段)的最短距離。"""
    p = np.asarray(p, float); best = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a; L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else np.clip((p - a) @ ab / L2, 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


# ---------------- 單狗執行器 ----------------
class Runner:
    def __init__(self, mode, policy):
        self.mode = mode                       # "compass" | "odom"
        self.g = Go2Gait(**GAIT); self.g.reset()
        self.m, self.d = self.g.m, self.g.d
        self.foot_gid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in P.LEGS]
        self.n_sub = int(round(CTRL_DT / self.m.opt.timestep))
        self.f0s, self.jinvs = P.leg_ik_consts(self.m)
        self.c = P.cpg_init(); self.last_a = np.zeros(12); self.policy = policy
        self.ren = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(self.m, self.cam)
        self.cam.elevation = -45.0
        self.hist = collections.deque(maxlen=int(1.0 / CTRL_DT))
        self.fallen = False

    def apply(self, q_des):
        for _ in range(self.n_sub):
            tau = self.g.kp * (q_des - self.d.qpos[7:19]) - self.g.kd * self.d.qvel[6:18]
            self.d.ctrl[:] = np.clip(tau, -self.g.flimit, self.g.flimit)
            mujoco.mj_step(self.m, self.d)
        self.hist.append((float(self.d.time), float(self.d.qpos[0])))

    def drive(self, cmd):
        obs = P.build_obs(self.g, self.c, cmd.astype(np.float32), self.last_a, self.foot_gid)
        act = self.policy(obs)
        mux, muy, om = P.act_to_cmd(act); self.c = P.cpg_step(self.c, mux, muy, om, CTRL_DT)
        self.apply(P.joint_targets(self.c, self.f0s, self.jinvs)); self.last_a = act
        if self.g.height < 0.15:
            self.fallen = True

    @property
    def xy(self):                              # 真值位置（量測/繪圖/切段共用，兩狗公平）
        return self.g.xy

    @property
    def yaw_truth(self):
        return self.g.true_yaw()

    def yaw_meas(self):                         # 修正用的量測航向（compass 有噪 / odom 完美）
        return self.g.compass_yaw() if self.mode == "compass" else self.g.odom()[2]

    def odom_xy(self):
        return np.array(self.g.odom()[:2])

    def speed(self):
        if len(self.hist) < 2:
            return 0.0
        (t0, x0), (t1, x1) = self.hist[0], self.hist[-1]
        return (x1 - x0) / max(t1 - t0, 1e-6)


# ---------------- 影格（含地板目標軌跡） ----------------
def frame(r, label, color, waypoints, sub=""):
    x, y = r.xy
    r.cam.lookat[:] = [x, y, 0.3]; r.cam.distance = 3.2
    r.cam.azimuth = float(np.degrees(r.yaw_truth))      # 相機恆在狗正後方
    r.ren.update_scene(r.d, r.cam)
    scn = r.ren.scene
    eye = np.eye(3).flatten()
    DOT = 1.2                                                    # 目標路徑點間距(m)
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        a2 = np.array([a[0], a[1]]); b2 = np.array([b[0], b[1]])
        seg = b2 - a2; L = float(np.linalg.norm(seg))
        if L < 1e-6:
            continue
        u = seg / L
        for s in np.arange(0.0, L, DOT):                        # 沿線佈亮綠點
            p = a2 + u * s
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.09, 0.09, 0.09],
                     [p[0], p[1], 0.05], eye, PATH_RGBA)
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.16, 0.16, 0.16],   # 航點(轉角)大黃球
                 [a[0], a[1], 0.06], eye, [1.0, 0.85, 0.1, 1.0])
    wl = waypoints[-1]
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.16, 0.16, 0.16],
             [wl[0], wl[1], 0.06], eye, [1.0, 0.85, 0.1, 1.0])
    im = Image.fromarray(r.ren.render()); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 34], fill=(255, 255, 255))
    dr.text((8, 6), label, fill=color, font=font(20))
    if sub:
        dr.text((8, H - 26), sub, fill=(230, 230, 230), font=font(14))
    st = "FALLEN" if r.fallen else f"v={r.speed():.2f} m/s"
    sc = (255, 80, 80) if r.fallen else (120, 255, 120)
    dr.text((W - 150, H - 26), st, fill=sc, font=font(16))
    return np.asarray(im)


def warmup(dogs):
    for r in dogs.values():
        for _ in range(int(0.5 / CTRL_DT)):
            r.apply(P.HOME12.copy())


LABELS = {"compass": "CPG-RL (+compass)", "odom": "CPG-RL (+odom)"}
COLORS = {"compass": (200, 40, 40), "odom": (40, 90, 220)}
PLTCOL = {"compass": "#c04030", "odom": "#3060c0"}


# ---------------- 實驗 A：直線 40m ----------------
def run_line(policy, args):
    import imageio.v2 as iio
    dogs = {"compass": Runner("compass", policy), "odom": Runner("odom", policy)}
    warmup(dogs)
    p0 = {nm: r.odom_xy() for nm, r in dogs.items()}          # odom 控制基準
    gx0 = {nm: r.xy.copy() for nm, r in dogs.items()}         # 真值起點（繪圖）
    waypoints = [(gx0["compass"][0], 0.0), (gx0["compass"][0] + 44.0, 0.0)]
    traj = {nm: [] for nm in dogs}
    vid = f"{OUT}/exp_line_compare.mp4"
    writer = iio.get_writer(vid, fps=FPS, codec="libx264", quality=7)
    frame_iv = int((1.0 / FPS) / CTRL_DT)
    maxsteps = int((args.secs or 90) / CTRL_DT)
    for i in range(maxsteps):
        for nm, r in dogs.items():
            if nm == "compass":
                yaw_rate = float(np.clip(-K_YAW * wrap(r.yaw_meas() - 0.0), -1.0, 1.0))
                cmd = np.array([VX, 0.0, yaw_rate])
            else:
                cmd, _, _ = P.line_control(r.odom_xy(), r.yaw_meas(), p0[nm], 0.0,
                                           VX, K_YAW, K_CT)
            r.drive(cmd)
            traj[nm].append([r.xy[0] - gx0[nm][0], r.xy[1] - gx0[nm][1]])
        if i % frame_iv == 0:
            fr = [frame(dogs[nm], LABELS[nm], COLORS[nm], waypoints, "straight 40 m")
                  for nm in ("compass", "odom")]
            writer.append_data(np.concatenate(fr, axis=1))
        if min(dogs[nm].xy[0] - gx0[nm][0] for nm in dogs) >= 40.0:
            break
    writer.close(); print("[video]", vid)
    for nm in dogs:
        t = np.array(traj[nm]); print(f"  {nm}: 前進 {t[-1,0]:.1f}m  末端|y|={abs(t[-1,1]):.3f}m")
    chart_line(traj, f"{OUT}/exp_line_chart.png")


def chart_line(traj, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm in ("compass", "odom"):
        t = np.array(traj[nm])
        a1.plot(t[:, 0], t[:, 1], color=PLTCOL[nm], lw=1.6,
                label=f"{LABELS[nm]} (end |y|={abs(t[-1,1]):.3f} m)")
    a1.axhline(0, color="gray", ls="--", lw=1)
    a1.set_xlabel("Forward x (m)"); a1.set_ylabel("Lateral y (m)")
    a1.set_title("Straight 40 m: compass vs odom")
    a1.set_ylim(-1.0, 1.0)                      # 固定 y 範圍：0.8m 側偏才看得出來(等比例會壓成一條線)
    a1.legend(fontsize=9, loc="upper left", framealpha=0.9)
    for nm in ("compass", "odom"):
        t = np.array(traj[nm])
        a2.plot(t[:, 0], np.abs(t[:, 1]), color=PLTCOL[nm], lw=1.6, label=LABELS[nm])
    a2.set_xlabel("Forward x (m)"); a2.set_ylabel("|Lateral drift| (m)")
    a2.set_title("Absolute lateral drift vs distance"); a2.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); print("[chart]", out)


# ---------------- 實驗 B：折線任務 ----------------
def run_course(policy, args):
    import imageio.v2 as iio
    dogs = {"compass": Runner("compass", policy), "odom": Runner("odom", policy)}
    warmup(dogs)
    gx0 = {nm: r.xy.copy() for nm, r in dogs.items()}
    # 各狗以自己真值起點為原點推理想折線
    ideal = {nm: ideal_waypoints(SEGS, origin=gx0[nm]) for nm, r in dogs.items()}
    state = {nm: {"seg": 0, "phase": "turn", "settle": 0,
                  "seg_start": gx0[nm].copy(), "t_seg": 0.0}
             for nm in dogs}
    traj = {nm: [] for nm in dogs}
    vid = f"{OUT}/exp_course_compare.mp4"
    writer = iio.get_writer(vid, fps=FPS, codec="libx264", quality=7)
    frame_iv = int((1.0 / FPS) / CTRL_DT)
    maxsteps = int((args.secs or 130) / CTRL_DT)
    for i in range(maxsteps):
        for nm, r in dogs.items():
            cmd = course_cmd(r, state[nm], ideal[nm])
            r.drive(cmd)
            traj[nm].append(r.xy.copy() - gx0[nm])
        if i % frame_iv == 0:
            fr = []
            for nm in ("compass", "odom"):
                s = state[nm]; seg = min(s["seg"], len(SEGS) - 1)
                sub = f"seg {min(s['seg']+1, len(SEGS))}/{len(SEGS)}  {s['phase']}"
                if nm == "compass" and s["phase"] == "straight" and s["seg"] < len(SEGS):
                    el = dogs[nm].d.time - s["t_seg"]; tgt = SEGS[seg][1] / VX
                    sub += f"  dist by time {el:.1f}/{tgt:.1f}s"
                wp = [(p[0], p[1]) for p in ideal[nm]]
                fr.append(frame(dogs[nm], LABELS[nm], COLORS[nm], wp, sub))
            writer.append_data(np.concatenate(fr, axis=1))
        if all(state[nm]["seg"] >= len(SEGS) for nm in dogs):
            break
    writer.close(); print("[video]", vid)
    ideal_end = ideal_waypoints(SEGS)[-1]                     # 相對原點的理想終點
    for nm in dogs:
        t = np.array(traj[nm]); end = t[-1]
        print(f"  {nm}: 終點=({end[0]:.2f},{end[1]:.2f})  理想=({ideal_end[0]:.2f},{ideal_end[1]:.2f})  "
              f"終點誤差={np.linalg.norm(end-ideal_end):.3f}m  末端|y|={abs(end[1]):.3f}m")
    chart_course(traj, ideal_waypoints(SEGS), f"{OUT}/exp_course_chart.png")


def course_cmd(r, s, ideal):
    """回傳該狗此步的 cmd[vx,vy,wz]；就地更新狀態機 s。"""
    if s["seg"] >= len(SEGS):
        return np.array([0.0, 0.0, 0.0])                     # 完成後原地站立
    hd = np.radians(SEGS[s["seg"]][0]); L = SEGS[s["seg"]][1]
    yawm = r.yaw_meas()
    if s["phase"] == "turn":
        e = wrap(yawm - hd)
        s["settle"] = s["settle"] + 1 if abs(e) < np.radians(2) else 0
        if s["settle"] >= int(0.3 / CTRL_DT):
            s["phase"] = "straight"; s["settle"] = 0
            s["seg_start"] = r.xy.copy(); s["t_seg"] = float(r.d.time)  # 記直行起點(位置給odom/繪圖、時間給compass)
        return np.array([0.0, 0.0, float(np.clip(-K_YAW * e, -1.0, 1.0))])
    dseg = np.array([np.cos(hd), np.sin(hd)])
    if r.mode == "odom":
        cmd, _, _ = P.line_control(r.odom_xy(), yawm, ideal[s["seg"]], hd, VX, K_YAW, K_CT)
        done = float(dseg @ (r.xy - ideal[s["seg"]])) >= L    # odom：沿理想線的真值進度切段
    else:
        # compass：無位置信號 → 只能用「指令速度×時間」估計走了多遠，走滿 L/VX 秒即切段。
        # 若實際步速≠指令 VX，距離就會系統性偏差且無從修正，正是純羅盤的弱點。
        cmd = np.array([VX, 0.0, float(np.clip(-K_YAW * wrap(yawm - hd), -1.0, 1.0))])
        done = (float(r.d.time) - s["t_seg"]) >= L / VX       # compass：以時間估計進度切段
    if done:
        s["seg"] += 1; s["phase"] = "turn"; s["settle"] = 0
    return cmd


def chart_course(traj, ideal, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3})
    ideal_end = ideal[-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))
    a1.plot(ideal[:, 0], ideal[:, 1], "k--", lw=1.3, label="ideal course")
    a1.plot([ideal[0, 0]], [ideal[0, 1]], "ks", ms=7)
    for nm in ("compass", "odom"):
        t = np.array(traj[nm]); end = t[-1]
        a1.plot(t[:, 0], t[:, 1], color=PLTCOL[nm], lw=1.6,
                label=f"{LABELS[nm]} (end err={np.linalg.norm(end-ideal_end):.2f} m)")
    a1.axhline(0, color="gray", ls=":", lw=1)
    a1.set_xlabel("x (m)"); a1.set_ylabel("y (m)")
    a1.set_title("Course tracking: compass vs odom"); a1.set_aspect("equal", "box")
    a1.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=1, framealpha=0.9)          # 圖例移到圖下方，避免蓋住軌跡
    for nm in ("compass", "odom"):
        t = np.array(traj[nm])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))])
        ct = np.array([point_to_polyline(p, ideal) for p in t])
        a2.plot(s, ct, color=PLTCOL[nm], lw=1.6, label=LABELS[nm])
    a2.set_xlabel("Distance travelled (m)"); a2.set_ylabel("Dist. to ideal course (m)")
    a2.set_title("Cross-track error along the course"); a2.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print("[chart]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["line", "course"], required=True)
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--secs", type=float, default=0.0, help="時間上限(0=用該實驗預設)")
    ap.add_argument("--w_coup", type=float, default=8.0, help="RL 策略 CPG 耦合(耦合版8/無耦合版0)")
    args = ap.parse_args()
    P.W_COUP = args.w_coup
    if args.params:
        print(f"[info] 載入 {args.params}"); policy = P.load_policy(args.params); print("[info] ok")
    else:
        print("[warn] 無 --params，用 dummy 固定動作測管線")
        policy = lambda o: np.tile([0.69, 0.0, -0.111], 4).astype(np.float32)
    (run_line if args.exp == "line" else run_course)(policy, args)


if __name__ == "__main__":
    main()
