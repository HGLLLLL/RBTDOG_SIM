"""純CPG vs CPG-RL(論文版) 四大對比實驗。
兩隻狗同物理、同模型(kp90/kd3)、同步施加相同擾動，並排即時影片(正後上方45°) + 學術風格圖表。
框架設定(已與使用者確認)：兩者都關航向迴路(純控制器對比)、都約 0.6 m/s、推力側向左右交替。

用法:
  python compare.py --exp 1 --params ../weights/cpg_rl_paper_params.pkl   # 外力推撞
  python compare.py --exp 2 --params ...   # 摩擦力
  python compare.py --exp 3 --params ...   # 負載
  python compare.py --exp 4 --params ...   # 直線 50m
"""
import argparse, os, sys, collections
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
from go2_gait import Go2Gait
from walk_line import GAIT, dir_mat, add_geom
import local_infer_paper as P
from PIL import Image, ImageDraw, ImageFont

CTRL_DT = 0.02
FPS = 25
W = H = 480
V_CMD_CPG = 0.72          # 使純CPG≈0.6 m/s（下方會校）
RL_VX = 0.6
OUT = "/home/huang/rbtdog_sim/task4/outputs"
FRICTION_LV = [0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03]
HEADING_GAIN = 3.0        # 實驗四羅盤航向 P 增益


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


_FONTS = {}
def font(sz):
    if sz not in _FONTS:
        import matplotlib.font_manager as fm
        _FONTS[sz] = ImageFont.truetype(fm.findfont("DejaVu Sans"), sz)
    return _FONTS[sz]


def quat_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


class Dog:
    def __init__(self, kind, policy=None, compass=False):
        self.kind = kind
        self.compass = compass
        self.g = Go2Gait(**GAIT); self.g.reset()
        self.m, self.d = self.g.m, self.g.d
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.foot_gid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in P.LEGS]
        self.n_sub = int(round(CTRL_DT / self.m.opt.timestep))
        self.failed = False; self.fail_level = None
        self.peak_roll = 0.0; self.max_survived = 0.0
        if kind == "rl":
            self.f0s, self.jinvs = P.leg_ik_consts(self.m)
            self.c = P.cpg_init(); self.last_a = np.zeros(12); self.policy = policy
        # 相機
        self.ren = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(self.m, self.cam)
        self.cam.azimuth = 0.0; self.cam.elevation = -45.0
        self.hist = collections.deque(maxlen=int(1.0 / CTRL_DT))  # 1s 位置窗

    def targets(self, t):
        if self.kind == "cpg":
            return self.g.targets(t, V_CMD_CPG, 0.0)
        yaw_rate = 0.0
        if self.compass:               # 羅盤航向 P 控制 → yaw 角速度指令(維持朝 +x)
            yaw_rate = float(np.clip(-HEADING_GAIN * wrap(self.g.compass_yaw()), -1.0, 1.0))
        cmd = np.array([RL_VX, 0.0, yaw_rate], np.float32)
        obs = P.build_obs(self.g, self.c, cmd, self.last_a, self.foot_gid)
        act = self.policy(obs)
        mux, muy, om = P.act_to_cmd(act); self.c = P.cpg_step(self.c, mux, muy, om, CTRL_DT)
        self.last_a = act
        return P.joint_targets(self.c, self.f0s, self.jinvs)

    def step(self, t):
        q_des = np.tile([0, 0.9, -1.8], 4) if self.failed else self.targets(t)
        for _ in range(self.n_sub):
            if self.failed:
                self.d.ctrl[:] = 0.0
            else:
                tau = self.g.kp * (q_des - self.d.qpos[7:19]) - self.g.kd * self.d.qvel[6:18]
                self.d.ctrl[:] = np.clip(tau, -self.g.flimit, self.g.flimit)
            mujoco.mj_step(self.m, self.d)
        self.hist.append((t, self.d.qpos[0]))

    @property
    def xy(self): return self.d.qpos[0:2].copy()
    @property
    def height(self): return float(self.d.qpos[2])

    def speed(self):
        if len(self.hist) < 2: return 0.0
        (t0, x0), (t1, x1) = self.hist[0], self.hist[-1]
        return (x1 - x0) / max(t1 - t0, 1e-6)

    def is_down(self):
        r, p = quat_rpy(self.d.qpos[3:7])
        return self.height < 0.15 or abs(r) > 1.0 or abs(p) > 1.0

    def set_foot_friction(self, mu):
        for gid in self.foot_gid:
            self.m.geom_friction[gid, 0] = mu

    def set_payload(self, kg):
        self.m.body_mass[self.base] = 6.921 + kg


def render(dog, label, color, stress_txt, tr_text=None, arrow_fy=None, payload_kg=None):
    x, y = dog.xy
    dog.cam.lookat[:] = [x, y, 0.3]; dog.cam.distance = 2.6
    dog.ren.update_scene(dog.d, dog.cam)
    scn = dog.ren.scene
    # 實驗三：背上實體負載箱，隨 kg 長高（跟著軀幹姿態）
    if payload_kg is not None and payload_kg > 0.1:
        pos = dog.d.xpos[dog.base].copy(); mat = dog.d.xmat[dog.base].copy()
        h = 0.03 + 0.006 * payload_kg
        up = mat.reshape(3, 3)[:, 2]
        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.12, 0.09, h],
                 pos + up * (0.05 + h), mat, [0.55, 0.32, 0.12, 1.0])
    # 實驗一：推力箭頭（橘色，指向受力方向，長度隨力道）
    if arrow_fy is not None and abs(arrow_fy) > 1e-6:
        s = float(np.sign(arrow_fy)); L = 0.5 + 0.02 * abs(arrow_fy)
        base = np.array([x, y - s * (0.2 + L), 0.35])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.07, 0.07, L],
                 base, dir_mat([0, s, 0]), [1.0, 0.4, 0.0, 1.0])
    im = Image.fromarray(dog.ren.render()); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 34], fill=(255, 255, 255))
    dr.text((8, 6), label, fill=color, font=font(20))
    dr.text((8, H - 26), stress_txt, fill=(240, 240, 240), font=font(14))
    st = "FALLEN" if (dog.failed and dog.fail_kind == "down") else ("STALL" if dog.failed else f"v={dog.speed():.2f} m/s")
    sc = (255, 80, 80) if dog.failed else (120, 255, 120)
    dr.text((W - 150, H - 26), st, fill=sc, font=font(16))
    # 右上角大字（實驗二摩擦 / 實驗三負載）
    if tr_text is not None:
        tw = int(dr.textlength(tr_text, font=font(30)))
        dr.rectangle([W - tw - 22, 40, W, 84], fill=(0, 0, 0))
        dr.text((W - tw - 12, 46), tr_text, fill=(255, 225, 0), font=font(30))
    # 推力大小標籤（實驗一，右上角紅底）
    if arrow_fy is not None and abs(arrow_fy) > 1e-6:
        pt = f"PUSH {abs(arrow_fy):.0f} N"
        pw = int(dr.textlength(pt, font=font(28)))
        dr.rectangle([W - pw - 22, 40, W, 82], fill=(210, 60, 0))
        dr.text((W - pw - 12, 46), pt, fill=(255, 255, 255), font=font(28))
    return np.asarray(im)


def make_chart(exp, tele, out, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3})
    if exp == 1:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        names = [labels["CPG"], labels["RL"]]
        vals = [tele["fail"]["CPG"] or 0, tele["fail"]["RL"] or 0]
        bars = a1.bar(names, vals, color=["#c04030", "#3060c0"], width=0.55)
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} N", ha="center", fontweight="bold")
        a1.set_ylabel("Max lateral push force survived (N)")
        a1.set_title("Exp.1  Push recovery: max disturbance before falling")
        for nm, col in [("CPG", "#c04030"), ("RL", "#3060c0")]:
            xs, ys = tele["curve"][nm]
            lbl = labels[nm]
            a2.plot(xs, ys, "-o", color=col, ms=4, label=lbl)
            if tele["fail"][nm]:
                a2.plot(tele["fail"][nm], (ys[-1] if ys else 0), "X", color=col, ms=13, mec="k")
        a2.set_xlabel("Lateral push force (N)"); a2.set_ylabel("Peak body roll after push (deg)")
        a2.set_title("Body tilt vs push force  (X = fell)"); a2.legend()
    elif exp in (2, 3):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        xkey = "friction" if exp == 2 else "payload"
        xlab = "Foot friction coefficient" if exp == 2 else "Added payload (kg)"
        for nm, col in [("CPG", "#c04030"), ("RL", "#3060c0")]:
            xs, ys = tele["curve"][nm]
            ax.plot(xs, ys, "-o", color=col, ms=4, label=labels[nm])
        if exp == 2: ax.invert_xaxis()
        ax.axhline(0.05, color="gray", ls="--", lw=1, label="stall threshold")
        ax.set_xlabel(xlab); ax.set_ylabel("Forward speed (m/s)")
        ax.set_title(f"Exp.{exp}  Forward speed vs {'decreasing friction' if exp==2 else 'increasing payload'}")
        ax.legend()
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for nm, col in [("CPG", "#c04030"), ("RL", "#3060c0")]:
            tr = np.array(tele["traj"][nm])
            lbl = labels[nm]
            a1.plot(tr[:, 0], tr[:, 1], color=col, lw=1.6, label=f"{lbl} (end |y|={abs(tr[-1,1]):.2f} m)")
        a1.axhline(0, color="gray", ls="--", lw=1)
        a1.set_xlabel("Forward x (m)"); a1.set_ylabel("Lateral y (m)")
        a1.set_title("Exp.4  Trajectory over 30 m: CPG-RL without vs with compass")
        a1.legend(); a1.set_aspect("equal", "box")
        for nm, col in [("CPG", "#c04030"), ("RL", "#3060c0")]:
            tr = np.array(tele["traj"][nm])
            a2.plot(tr[:, 0], np.abs(tr[:, 1]), color=col, lw=1.6,
                    label=labels[nm])
        a2.set_xlabel("Forward x (m)"); a2.set_ylabel("|Lateral drift| (m)")
        a2.set_title("Absolute lateral drift vs distance"); a2.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); print("[chart]", out)


def run(args):
    P.W_COUP = args.w_coup              # 推論耦合須與策略訓練一致(耦合版8, 無耦合版0)
    policy = P.load_policy(args.params) if args.params else (lambda o: np.tile([0.69, 0., -0.111], 4).astype(np.float32))
    exp = args.exp
    if exp == 4:      # 實驗四：同一 RL 策略，左=無羅盤、右=有羅盤航向修正
        dogs = {"CPG": Dog("rl", policy, compass=False), "RL": Dog("rl", policy, compass=True)}
        labels = {"CPG": "CPG-RL (no compass)", "RL": "CPG-RL (+compass)"}
    else:
        dogs = {"CPG": Dog("cpg"), "RL": Dog("rl", policy)}
        labels = {"CPG": "Pure CPG", "RL": "CPG-RL (paper)"}
    for dd in dogs.values(): dd.fail_kind = None

    # 站穩
    for _ in range(int(0.5 / CTRL_DT)):
        for dd in dogs.values(): dd.step(0.0)

    import imageio.v2 as iio
    vid = f"{OUT}/exp{exp}_compare.mp4"
    writer = iio.get_writer(vid, fps=FPS, codec="libx264", quality=7)
    frame_iv = int((1.0 / FPS) / CTRL_DT)
    tele = {"fail": {"CPG": None, "RL": None}, "curve": {"CPG": ([], []), "RL": ([], [])},
            "traj": {"CPG": [], "RL": []}}
    x0 = {nm: dd.xy[0] for nm, dd in dogs.items()}
    stage = -1; stage_x = {}; push_sign = 1
    maxt = {1: 70, 2: 70, 3: 70, 4: 110}[exp]
    n = int(maxt / CTRL_DT)
    for i in range(n):
        t = i * CTRL_DT
        st = int(t // 5)
        new_stage = st != stage
        stage = st

        # ---- 每 5 秒升級擾動 ----
        tr_text = None; payload_kg = None
        if exp == 2:
            mu = FRICTION_LV[min(st, len(FRICTION_LV) - 1)]
            for dd in dogs.values(): dd.set_foot_friction(mu)
            level, ltxt = mu, f"foot friction = {mu:.2f}"
            tr_text = f"friction {mu:.2f}"
        elif exp == 3:
            kg = st * 2.0
            for dd in dogs.values(): dd.set_payload(kg)
            level, ltxt = kg, f"payload = +{kg:.0f} kg"
            tr_text = f"payload +{kg:.0f} kg"; payload_kg = kg
        elif exp == 1:
            level = 30 + 10 * st
            ltxt = f"push every 5s, now {level:.0f} N"
        else:
            level, ltxt = 0, "straight 30 m: compass OFF (L) vs ON (R)"

        if new_stage and exp == 2:  # 記錄每階段前的位置，算該階段速度
            stage_x = {nm: dd.xy[0] for nm, dd in dogs.items()}
        if new_stage and exp == 3:
            stage_x = {nm: dd.xy[0] for nm, dd in dogs.items()}
        if new_stage and exp == 1:
            push_sign = 1 if st % 2 == 0 else -1

        # ---- 施力(exp1)：施力 0.2s，箭頭多顯示到 0.6s 方便觀看 ----
        arrow_fy = None
        for dd in dogs.values():
            dd.d.xfrc_applied[dd.base] = 0.0
        if exp == 1:
            phase = t - st * 5
            if phase < 0.2:
                for dd in dogs.values():
                    if not dd.failed:
                        dd.d.xfrc_applied[dd.base, 1] = push_sign * level
            if phase < 0.6:
                arrow_fy = push_sign * level

        # ---- 前進一步 ----
        for nm, dd in dogs.items():
            dd.step(t)
            if not dd.failed and dd.is_down():
                dd.failed = True; dd.fail_kind = "down"; dd.fail_level = level
                tele["fail"][nm] = level

        # ---- 側傾追蹤 + 每階段承受力紀錄(exp1) ----
        if exp == 1:
            for nm, dd in dogs.items():
                if not dd.failed:
                    r, p = quat_rpy(dd.d.qpos[3:7]); dd.peak_roll = max(dd.peak_roll, abs(r))
            if int((t + CTRL_DT) // 5) != st:            # 階段最後一步
                for nm, dd in dogs.items():
                    if dd.failed: continue
                    xs, ys = tele["curve"][nm]; xs.append(level); ys.append(np.degrees(dd.peak_roll))
                    dd.max_survived = level; dd.peak_roll = 0.0
                    tele["fail"][nm] = dd.max_survived

        # ---- 失速判定(exp2,3)：每階段結束檢查該階段前進距離 ----
        if exp in (2, 3) and (int((t + CTRL_DT) // 5) != st):  # 階段最後一步
            for nm, dd in dogs.items():
                if dd.failed: continue
                adv = dd.xy[0] - stage_x.get(nm, dd.xy[0])
                spd = adv / 5.0
                xs, ys = tele["curve"][nm]; xs.append(level); ys.append(max(spd, 0))
                if spd < 0.05:
                    dd.failed = True; dd.fail_kind = "stall"; dd.fail_level = level
                    tele["fail"][nm] = level

        if exp == 4:
            for nm, dd in dogs.items():
                tele["traj"][nm].append([dd.xy[0] - x0[nm], dd.xy[1]])

        # ---- 影片 ----
        if i % frame_iv == 0:
            a = render(dogs["CPG"], labels["CPG"], (200, 40, 40), ltxt, tr_text, arrow_fy, payload_kg)
            b = render(dogs["RL"], labels["RL"], (40, 90, 220), ltxt, tr_text, arrow_fy, payload_kg)
            writer.append_data(np.concatenate([a, b], axis=1))

        # ---- 結束條件 ----
        if exp in (1, 2, 3) and all(dd.failed for dd in dogs.values()):
            # 多錄 1 秒收尾
            for _ in range(FPS):
                a = render(dogs["CPG"], labels["CPG"], (200, 40, 40), ltxt, tr_text, None, payload_kg)
                b = render(dogs["RL"], labels["RL"], (40, 90, 220), ltxt, tr_text, None, payload_kg)
                writer.append_data(np.concatenate([a, b], axis=1))
            break
        if exp == 4 and min(dogs["CPG"].xy[0] - x0["CPG"], dogs["RL"].xy[0] - x0["RL"]) >= 30:
            break

    writer.close(); print("[video]", vid)
    print("[fail levels]", tele["fail"])
    if exp == 4:
        for nm, dd in dogs.items():
            tr = np.array(tele["traj"][nm]); print(f"  {nm}: end x={tr[-1,0]:.1f} |y|={abs(tr[-1,1]):.2f}")
    make_chart(exp, tele, f"{OUT}/exp{exp}_chart.png", labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--w_coup", type=float, default=8.0,
                    help="RL 策略的 CPG 耦合強度：耦合版=8.0，無耦合版=0.0")
    run(ap.parse_args())
