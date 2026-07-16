"""地形實驗（terrain 版 CPG-RL 單獨，原生直走 cmd=[vx,0,0]，不套 line_control）。

exp1 (slope)：平地→多段斜坡(上/下坡 5/10/15°)，平地與坡地不同顏色，大字顯示當前坡度。右側視角。
exp2 (rough)：平地→崎嶇(3/5/8cm)，大字顯示幅度。後上方 45° 俯視。

同一 scene_mjx 課程、apply_pd 位置伺服(kp90/kd3)。terrain 版靠自身防走歪能力直走(不需 odom)。
觸地 obs 用「相對地面 gz」(與訓練一致)。

用法：
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare.py --check
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare.py --exp both
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import local_infer_terrain as T          # CPG/IK/apply_pd/load_policy/w2b/常數
from PIL import Image, ImageDraw

SCENE = "/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene_mjx.xml"
OUT = "/home/huang/rbtdog_sim/task4/outputs/2026-07-15"
WPARAMS = {"terrain": "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain_params.pkl"}
LABELS = {"terrain": "TERRAIN CPG-RL"}
FPS, VW, VH = 25, 640, 480
RENDER_EVERY = 2                          # 每 2 控制步(=0.04s)出一格 → 25fps
VX = 0.8                                  # 較快 → 相對航向漂移較小(盲走模型無絕對位置回授)
TERR_WY, TERR_TH = 6.0, 0.5               # 跑道加寬到 ±6m，容忍盲走殘留漂移不掉出邊緣

_FONTS = {}
def font(sz):
    if sz not in _FONTS:
        import matplotlib.font_manager as fm
        from PIL import ImageFont
        _FONTS[sz] = ImageFont.truetype(fm.findfont("DejaVu Sans"), sz)
    return _FONTS[sz]


def yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def add_ramp_box(spec, xa, za, xb, zb, rgba, name):
    dx, dz = xb - xa, zb - za
    L = float(np.hypot(dx, dz)); a = float(np.arctan2(dz, dx))
    mx, mz = 0.5 * (xa + xb), 0.5 * (za + zb)
    nx, nz = -np.sin(a), np.cos(a)
    g = spec.worldbody.add_geom()
    g.name = name; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [L / 2, TERR_WY, TERR_TH]
    g.pos = [mx - nx * TERR_TH, 0.0, mz - nz * TERR_TH]
    g.quat = [np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0]
    g.rgba = rgba


# ---------------- 課程 1：平地 + 斜坡 ----------------
# 第一段平地從 x=-1.5 起（讓 spawn 在 x=0 的後腳有地面），故長 3.5 到 x=2
SEGS1 = [("flat", 3.5, 0), ("up", 1.8, 5), ("down", 1.8, 5), ("flat", 1.5, 0),
         ("up", 1.8, 10), ("down", 1.8, 10), ("flat", 1.5, 0),
         ("up", 1.8, 15), ("down", 1.8, 15), ("flat", 6.0, 0)]
SEG1_X0 = -1.5
COL1 = {"flat": [0.50, 0.50, 0.52, 1], "up": [0.30, 0.72, 0.38, 1], "down": [0.90, 0.55, 0.18, 1]}


def build_course_slopes():
    spec = mujoco.MjSpec.from_file(SCENE)
    floor = next(g for g in spec.geoms if g.name == "floor"); floor.pos = [0, 0, -10.0]
    x, z = SEG1_X0, 0.0; knots = [(x, z)]; segs = []
    for k, (typ, dx, ang) in enumerate(SEGS1):
        dz = dx * np.tan(np.radians(ang)) * (1 if typ == "up" else -1 if typ == "down" else 0)
        xb, zb = x + dx, z + dz
        add_ramp_box(spec, x, z, xb, zb, COL1[typ], f"seg{k}")
        lab = "FLAT" if typ == "flat" else (f"UPHILL {ang} deg" if typ == "up" else f"DOWNHILL {ang} deg")
        segs.append((x, xb, lab, typ)); knots.append((xb, zb)); x, z = xb, zb
    m = spec.compile()
    kx = np.array([p[0] for p in knots]); kz = np.array([p[1] for p in knots])

    def gz(xq, yq=None): return np.interp(xq, kx, kz)
    def phase(xq):
        for a, b, lab, typ in segs:
            if a <= xq < b: return lab, typ
        return segs[-1][2], segs[-1][3]
    return m, gz, phase, (kx, kz, segs)


# ---------------- 課程 2：平地 + 崎嶇 ----------------
REGIONS2 = [(2.0, 4.0, 0.03, "ROUGH 3cm"), (5.0, 7.0, 0.05, "ROUGH 5cm"),
            (8.0, 10.0, 0.08, "ROUGH 8cm")]
RX_MIN, RX_MAX, RRY, MAXA = -2.0, 20.0, 6.0, 0.08     # RRY 加寬容忍漂移


def build_course_rough():
    spec = mujoco.MjSpec.from_file(SCENE)
    floor = next(g for g in spec.geoms if g.name == "floor"); floor.pos = [0, 0, -10.0]
    ncol = int((RX_MAX - RX_MIN) / 0.08); nrow = int(2 * RRY / 0.08)
    xs = np.linspace(RX_MIN, RX_MAX, ncol); ys = np.linspace(-RRY, RRY, nrow)
    Xg, Yg = np.meshgrid(xs, ys)
    noise = (np.sin(1.5 * Xg) * np.cos(1.3 * Yg) + 0.6 * np.sin(2.9 * Xg + 1) * np.cos(3.1 * Yg + 2)
             + 0.4 * np.sin(5.0 * Xg) * np.cos(4.3 * Yg + 0.5))
    noise = noise / np.max(np.abs(noise))
    amp = np.zeros_like(Xg)
    for a, b, mag, _ in REGIONS2:
        amp[(Xg >= a) & (Xg < b)] = mag
    raw = amp * noise
    f = np.clip(0.5 + raw / (2 * MAXA), 0, 1)
    gx = 0.5 * (RX_MIN + RX_MAX); rx = 0.5 * (RX_MAX - RX_MIN)
    hf = spec.add_hfield(); hf.name = "rough"; hf.nrow = nrow; hf.ncol = ncol
    hf.size = [rx, RRY, 2 * MAXA, 0.5]; hf.userdata = f.flatten().tolist()
    g = spec.worldbody.add_geom(); g.name = "roughgeom"; g.type = mujoco.mjtGeom.mjGEOM_HFIELD
    g.hfieldname = "rough"; g.pos = [gx, 0.0, -MAXA]; g.rgba = [0.55, 0.5, 0.42, 1]
    m = spec.compile()
    hz = m.hfield_data.reshape(nrow, ncol); z_top = m.hfield_size[0, 2]; gz0 = -MAXA

    def gz(xq, yq=0.0):
        xq = np.asarray(xq, float); yq = np.asarray(yq, float) * np.ones_like(xq)
        cx = np.clip((xq - (gx - rx)) / (2 * rx) * (ncol - 1), 0, ncol - 1)
        cy = np.clip((yq - (-RRY)) / (2 * RRY) * (nrow - 1), 0, nrow - 1)
        x0 = np.floor(cx).astype(int); y0 = np.floor(cy).astype(int)
        x1 = np.minimum(x0 + 1, ncol - 1); y1 = np.minimum(y0 + 1, nrow - 1)
        fx = cx - x0; fy = cy - y0
        v = (hz[y0, x0] * (1 - fx) * (1 - fy) + hz[y0, x1] * fx * (1 - fy)
             + hz[y1, x0] * (1 - fx) * fy + hz[y1, x1] * fx * fy)
        inside = (xq >= RX_MIN) & (xq <= RX_MAX)
        return np.where(inside, gz0 + z_top * v, 0.0)

    def phase(xq):
        for a, b, mag, lab in REGIONS2:
            if a <= xq < b: return lab, "rough"
        return "FLAT", "flat"
    return m, gz, phase, REGIONS2


# ---------------- obs（相對地面 gz 觸地）----------------
def build_obs(d, c, cmd, last_a, foot_gid, gz):
    quat = d.qpos[3:7]
    grav = T.w2b(quat, np.array([0, 0, -1.0])); blin = T.w2b(quat, d.qvel[0:3])
    gyro = d.qvel[3:6]
    fxs = np.array([d.geom_xpos[g][0] for g in foot_gid])
    fys = np.array([d.geom_xpos[g][1] for g in foot_gid])
    fzs = np.array([d.geom_xpos[g][2] for g in foot_gid])
    contact = ((fzs - gz(fxs, fys)) < T.FOOT_CONTACT_H).astype(np.float32)
    return np.concatenate([grav, blin, gyro, d.qpos[7:19] - T.HOME12, d.qvel[6:18],
                           cmd, last_a, contact, c["rx"], c["rx_d"], c["ry"], c["ry_d"],
                           np.sin(c["theta"]), np.cos(c["theta"])]).astype(np.float32)


def _annotate(frame, model, ph, fell, x):
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    lab, kind = ph
    col = {"flat": (215, 215, 220), "up": (95, 225, 130), "down": (245, 165, 70),
           "rough": (240, 210, 130)}[kind]
    dr.rectangle([0, 0, VW, 74], fill=(0, 0, 0))
    dr.text((14, 6), LABELS[model], fill=(255, 255, 255), font=font(20))
    dr.text((14, 32), lab, fill=col, font=font(38))
    dr.text((VW - 175, 12), f"x = {x:+.1f} m", fill=(220, 220, 220), font=font(20))
    if fell is not None:
        dr.text((VW - 175, 42), "FALLEN", fill=(255, 90, 90), font=font(22))
    return np.asarray(im)


# ---------------- 單支實驗（terrain 單獨）----------------
def run_experiment(build_course, cam_kind, secs, out_prefix, model="terrain", video=True):
    m, gz, phase, extra = build_course()
    m.opt.timestep = T.SIM_DT; m = T.apply_pd(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in T.LEGS]
    jinvs = T.leg_ik_consts(m)
    nsub = int(round(T.CTRL_DT / T.SIM_DT))
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    infer = T.load_policy(WPARAMS[model]); c = T.cpg_init(); last = np.zeros(12)
    for _ in range(int(0.4 / T.CTRL_DT)):                 # settle 0.4s
        d.ctrl[:] = np.clip(np.array(T.HOME12), lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)

    ren = mujoco.Renderer(m, VH, VW) if video else None
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    import imageio.v2 as iio
    writer = iio.get_writer(f"{OUT}/{out_prefix}.mp4", fps=FPS, codec="libx264") if video else None
    traj = []; fell = None; n = int(secs / T.CTRL_DT)
    for i in range(n):
        x, y, yaw = float(d.qpos[0]), float(d.qpos[1]), yaw_from_quat(d.qpos[3:7])
        cmd = np.array([VX, 0.0, 0.0], np.float32)         # 原生直走
        obs = build_obs(d, c, cmd, last, foot_gid, gz)
        act = infer(obs); mux, muy, om = T.act_to_cmd(act); c = T.cpg_step(c, mux, muy, om, T.CTRL_DT)
        d.ctrl[:] = np.clip(T.joint_targets(c, jinvs), lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)
        last = act
        grav = T.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        gzx = float(gz(np.array([x]), np.array([y]))[0]) if cam_kind == "rear45" else float(gz(x))
        rel_h = float(d.qpos[2]) - gzx
        if fell is None and (rel_h < 0.15 or grav[2] > -0.4): fell = i * T.CTRL_DT
        traj.append((i * T.CTRL_DT, x, y, float(d.qpos[2]), gzx))
        if video and i % RENDER_EVERY == 0:
            if cam_kind == "side":
                cam.lookat[:] = [x, y, gzx + 0.15]; cam.distance = 3.6   # 跟隨 x,y
                cam.elevation = -8; cam.azimuth = 90                     # 世界固定右側視角
            else:
                cam.lookat[:] = [x, y, 0.3]; cam.distance = 3.2
                cam.elevation = -45; cam.azimuth = float(np.degrees(yaw))
            ren.update_scene(d, cam)
            writer.append_data(_annotate(ren.render(), model, phase(x), fell, x))
    if writer: writer.close(); print("[video]", f"{OUT}/{out_prefix}.mp4")
    t = np.array(traj)
    print(f"  [{model}] 前進到 x={t[:,1].max():.1f}m 最大側偏={np.abs(t[:,2]).max():.2f}m "
          f"跌倒={'是@%.1fs' % fell if fell else '否'}")
    return {model: t}, {model: fell}, extra


# ---------------- 圖表 ----------------
PLTCOL = {"terrain": "#1f77b4"}


def chart_slope(trajs, fell, extra, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kx, kz, segs = extra
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    xg = np.linspace(kx[0], kx[-1], 400)
    a1.fill_between(xg, np.interp(xg, kx, kz) - 0.5, np.interp(xg, kx, kz), color="0.88", zorder=0)
    a1.plot(xg, np.interp(xg, kx, kz), "k-", lw=1.2, label="terrain profile")
    for a, b, lab, typ in segs:
        if typ != "flat":
            a1.axvspan(a, b, color=(COL1[typ][0], COL1[typ][1], COL1[typ][2]), alpha=0.13)
            a1.text((a + b) / 2, np.interp((a + b) / 2, kx, kz) + 0.12,
                    lab.replace(" deg", "°").replace("UPHILL ", "↑").replace("DOWNHILL ", "↓"),
                    ha="center", fontsize=7)
    for nm, t in trajs.items():
        a1.plot(t[:, 1], t[:, 3], color=PLTCOL[nm], lw=1.8, label="CoM height")
        if fell[nm] is not None:
            k = np.argmin(np.abs(t[:, 0] - fell[nm])); a1.plot(t[k, 1], t[k, 3], "rx", ms=11, mew=3, label="fall")
    a1.set_xlabel("Forward x (m)"); a1.set_ylabel("Height z (m)")
    a1.set_title("Terrain CPG-RL over slope course"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    for nm, t in trajs.items():
        a2.plot(t[:, 0], t[:, 1], color=PLTCOL[nm], lw=1.8, label="forward x")
        a2b = a2.twinx(); a2b.plot(t[:, 0], np.abs(t[:, 2]), color="#2ca02c", lw=1.2, ls="--", label="|lateral|")
        a2b.set_ylabel("|lateral y| (m)", color="#2ca02c"); a2b.set_ylim(0, max(3.0, np.abs(t[:, 2]).max() * 1.1))
    a2.set_xlabel("Time (s)"); a2.set_ylabel("Forward x (m)")
    a2.set_title("Progress & straightness vs time"); a2.legend(fontsize=8, loc="upper left"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print("[chart]", out)


def chart_rough(trajs, fell, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for nm, t in trajs.items():
        a1.plot(t[:, 0], t[:, 1], color=PLTCOL[nm], lw=1.8, label="forward x")
        if fell[nm] is not None:
            k = np.argmin(np.abs(t[:, 0] - fell[nm])); a1.plot(t[k, 0], t[k, 1], "rx", ms=11, mew=3, label="fall")
    a1.set_xlabel("Time (s)"); a1.set_ylabel("Forward x (m)")
    a1.set_title("Terrain CPG-RL over rough course"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    for a, b, mag, lab in REGIONS2:
        a2.axvspan(a, b, color="0.85", alpha=0.8)
        a2.text((a + b) / 2, 0.92, lab.split()[1], ha="center", va="top", fontsize=8,
                transform=a2.get_xaxis_transform())
    for nm, t in trajs.items():
        a2.plot(t[:, 1], np.abs(t[:, 2]), color=PLTCOL[nm], lw=1.6, label="|lateral drift|")
    a2.set_xlabel("Forward x (m)"); a2.set_ylabel("|Lateral drift| (m)")
    a2.set_title("Straightness over rough regions"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print("[chart]", out)


def check():
    print("=== 幾何 vs gz 射線核對 ===")
    for name, builder, is2d in [("slope", build_course_slopes, False), ("rough", build_course_rough, True)]:
        m, gz, phase, _ = builder(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        bad = 0; N = 0
        for x in np.arange(0.5, 17.0, 0.5):
            if abs(x) < 0.9: continue
            gid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(m, d, np.array([x, 0.0, 5.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
            ray = 5.0 - dist if dist >= 0 else np.nan
            g = float(gz(np.array([x]), np.array([0.0]))[0]) if is2d else float(gz(x))
            N += 1
            if not np.isfinite(ray) or abs(ray - g) > 0.03: bad += 1
        print(f"  [{name}] 取樣{N}點 不一致={bad} (應為0)")
    print("=== 3s rollout smoke（不出影片）===")
    for name, builder, cam in [("slope", build_course_slopes, "side"), ("rough", build_course_rough, "rear45")]:
        trajs, fell, _ = run_experiment(builder, cam, 3.0, "smoke", video=False)
        ok = all(np.all(np.isfinite(t)) for t in trajs.values())
        print(f"  [{name}] 有限={ok}")
    print("CHECK DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["slope", "rough", "both"], default="both")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--secs", type=float, default=45.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check: check(); return
    if args.exp in ("slope", "both"):
        print("=== 實驗1：平地→斜坡（terrain）===")
        tr, fe, ex = run_experiment(build_course_slopes, "side", 45.0, "exp1_slope_terrain")
        chart_slope(tr, fe, ex, f"{OUT}/exp1_slope_chart.png")
    if args.exp in ("rough", "both"):
        print("=== 實驗2：平地→崎嶇（terrain）===")
        tr, fe, ex = run_experiment(build_course_rough, "rear45", 35.0, "exp2_rough_terrain")
        chart_rough(tr, fe, f"{OUT}/exp2_rough_chart.png")


if __name__ == "__main__":
    main()
