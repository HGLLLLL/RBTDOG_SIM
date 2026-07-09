"""本機 CPU 推論（論文標準版）：載入 cpg_rl_paper_params.pkl，接回 task3 Go2 + odom 走絕對直線。
CPG/IK 映射與 cpg_rl_paper_colab.ipynb 逐行對應（12維動作、2D腳掌、腿間耦合、固定離地 g_c）。
用法:
  python local_infer_paper.py --dummy --secs 8 --video           # 沒權重先測管線
  python local_infer_paper.py --params cpg_rl_paper_params.pkl --secs 20 --video --push
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
import mujoco
from go2_gait import Go2Gait
from walk_line import GAIT, PUSHES

LEGS = ["FL", "FR", "RL", "RR"]
HOME3 = np.array([0.0, 0.9, -1.8])
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
# ---- 常數：必須與 notebook 一致 ----
MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 4.5
A_CONV = 50.0
D_STEP = 0.12
G_C = 0.08
G_P = 0.01
W_COUP = 8.0
N_CPG_SUB = 4
CTRL_DT = 0.02
FOOT_CONTACT_H = 0.03
OBS_DIM, ACT_DIM = 76, 12
_PO = np.array([0.0, np.pi, np.pi, 0.0])
PHI = _PO[None, :] - _PO[:, None]
# odom 走直線
HEADING_GAIN = 3.0      # 航向 P 增益（= 控制律 K_YAW 預設）
VX_CMD = 0.6


def cpg_init():
    return {"rx": np.full(4, 1.5), "rx_d": np.zeros(4),
            "ry": np.full(4, 1.5), "ry_d": np.zeros(4), "theta": _PO.copy()}


def cpg_step(c, mux, muy, omega, dt):
    rx, rxd, ry, ryd, th = (c["rx"].copy(), c["rx_d"].copy(),
                            c["ry"].copy(), c["ry_d"].copy(), c["theta"].copy())
    h = dt / N_CPG_SUB
    for _ in range(N_CPG_SUB):
        rxd += (A_CONV * (A_CONV / 4 * (mux - rx) - rxd)) * h; rx += rxd * h
        ryd += (A_CONV * (A_CONV / 4 * (muy - ry) - ryd)) * h; ry += ryd * h
        rbar = 0.5 * (rx + ry)
        diff = th[None, :] - th[:, None] - PHI
        th = th + (2 * np.pi * omega + W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th % (2 * np.pi)}


def act_to_cmd(a):
    a = np.tanh(a).reshape(4, 3)
    mux = (a[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy = (a[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    om = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, om


def joint_targets(c, f0s, jinvs):
    th = c["theta"]
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    dx = -D_STEP * fx * np.cos(th); dy = D_STEP * fy * np.cos(th)
    dz = np.where(np.sin(th) > 0, G_C * np.sin(th), G_P * np.sin(th))
    off = np.stack([dx, dy, dz], -1)
    q = np.zeros((4, 3))
    for k in range(4):
        q[k] = HOME3 + jinvs[k] @ off[k]
    return q.reshape(12)


def leg_ik_consts(m):
    d = mujoco.MjData(m); f0s, jinvs = [], []
    for k, leg in enumerate(LEGS):
        jb = 7 + 3 * k
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg)
        hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg + "_hip")
        def foot(q3):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.qpos[jb:jb + 3] = q3; mujoco.mj_forward(m, d)
            return (d.geom_xpos[gid] - d.xpos[hip]).copy()
        f0 = foot(HOME3); e = 1e-3; J = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros(3); dq[j] = e
            J[:, j] = (foot(HOME3 + dq) - foot(HOME3 - dq)) / (2 * e)
        f0s.append(f0); jinvs.append(np.linalg.inv(J))
    return np.array(f0s), np.array(jinvs)


def qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qrot(q, v):
    u = q[1:4]; t = 2 * np.cross(u, v); return v + q[0] * t + np.cross(u, t)
def w2b(q, v): return qrot(qinv(q), v)
def wrap(a): return np.arctan2(np.sin(a), np.cos(a))


def line_frame(psi):
    """目標線的方向 d 與左法向 n（世界系單位向量）。"""
    d = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-np.sin(psi), np.cos(psi)])
    return d, n


def line_control(p, yaw, p0, psi_target, vx, k_yaw, k_ct, no_lateral=False):
    """方案 A 解耦控制：wz 用航向誤差鎖航向、vy 用 cross-track 誤差滑回線上。
    p, p0 為世界系 (x,y)；回傳 (cmd[vx,vy,wz] float32, e_ct, e_yaw)。"""
    _, n = line_frame(psi_target)
    # e_ct 為世界系橫向誤差；vy 為 body 系橫移。兩者僅在 yaw≈psi_target 時等價，
    # 而 wz 主動鎖住航向確保此前提成立（大航向誤差時橫向修正方向會短暫失準）。
    e_ct = float(n @ (np.asarray(p, float) - np.asarray(p0, float)))
    e_yaw = float(wrap(yaw - psi_target))
    wz = float(np.clip(-k_yaw * e_yaw, -1.0, 1.0))
    vy = 0.0 if no_lateral else float(np.clip(-k_ct * e_ct, -0.3, 0.3))
    return np.array([vx, vy, wz], np.float32), e_ct, e_yaw


def build_obs(g, c, cmd, last_a, foot_gid):
    quat = g.d.qpos[3:7]
    grav = w2b(quat, np.array([0, 0, -1.0])); blin = w2b(quat, g.d.qvel[0:3])
    gyro = g.d.qvel[3:6]
    contact = (np.array([g.d.geom_xpos[gid][2] for gid in foot_gid]) < FOOT_CONTACT_H).astype(np.float32)
    return np.concatenate([grav, blin, gyro, g.d.qpos[7:19] - HOME12, g.d.qvel[6:18],
                           cmd, last_a, contact, c["rx"], c["rx_d"], c["ry"], c["ry_d"],
                           np.sin(c["theta"]), np.cos(c["theta"])]).astype(np.float32)


def load_policy(path):
    import jax, functools
    from brax.io import model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    factory = functools.partial(ppo_networks.make_ppo_networks,
                                policy_hidden_layer_sizes=(256, 256, 128),
                                value_hidden_layer_sizes=(256, 256, 256))
    net = factory(OBS_DIM, ACT_DIM, preprocess_observations_fn=running_statistics.normalize)
    make_policy = ppo_networks.make_inference_fn(net)
    pol = make_policy(model.load_params(path), deterministic=True)
    jpol = jax.jit(pol); key = jax.random.PRNGKey(0)
    def infer(obs):
        return np.asarray(jpol(jax.numpy.asarray(obs), key)[0])
    infer(np.zeros(OBS_DIM, np.float32))
    return infer


def run(args):
    global W_COUP
    W_COUP = args.w_coup                 # 必須與該策略訓練時的耦合一致(耦合版8, 無耦合版0)
    g = Go2Gait(**GAIT); g.reset()
    f0s, jinvs = leg_ik_consts(g.m)
    foot_gid = [mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in LEGS]
    base_id = mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_BODY, "base")
    n_sub = int(round(CTRL_DT / g.m.opt.timestep))
    print(f"[info] n_sub={n_sub} kp={g.kp} kd={g.kd}")

    if args.dummy or not args.params:
        if not args.params and not args.dummy:
            print("[warn] 無 --params，用 dummy 固定動作測管線")
        fixed = np.tile([0.69, 0.0, -0.111], 4).astype(np.float32)  # mux~1.8,muy~1.5,om~2
        infer = lambda obs: fixed
    else:
        print(f"[info] 載入 {args.params}"); infer = load_policy(args.params); print("[info] ok")

    def apply(q_des):
        for _ in range(n_sub):
            tau = g.kp * (q_des - g.d.qpos[7:19]) - g.kd * g.d.qvel[6:18]
            g.d.ctrl[:] = np.clip(tau, -g.flimit, g.flimit); mujoco.mj_step(g.m, g.d)

    def push_at(t):
        g.d.xfrc_applied[base_id] = 0.0
        if not args.push: return
        for t0, dur, fy, tz in PUSHES:
            if t0 <= t < t0 + dur:
                g.d.xfrc_applied[base_id, 1] = fy; g.d.xfrc_applied[base_id, 5] = tz

    for _ in range(int(0.5 / CTRL_DT)): apply(HOME12.copy())

    ren = cam = None; frames = []
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        ren = mujoco.Renderer(g.m, 480, 640); cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(g.m, cam)

    c = cpg_init(); last_a = np.zeros(ACT_DIM)
    fl_gid = mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    k_yaw = HEADING_GAIN

    def step_policy(cmd):
        nonlocal c, last_a
        obs = build_obs(g, c, cmd.astype(np.float32), last_a, foot_gid)
        act = infer(obs)
        mux, muy, om = act_to_cmd(act); c = cpg_step(c, mux, muy, om, CTRL_DT)
        apply(joint_targets(c, f0s, jinvs)); last_a = act

    def maybe_render(k):
        if ren is not None and k % 2 == 0:
            x, y, _ = g.odom(); cam.lookat[:] = [x, y, 0.3]; cam.distance = 2.5
            cam.elevation = -20; cam.azimuth = 90
            ren.update_scene(g.d, cam); frames.append(ren.render())

    # ---- Phase 1：轉向到 psi_goal（右轉 → turn_deg 為負）----
    _, _, start_yaw = g.odom()
    psi_goal = wrap(start_yaw + np.radians(args.turn_deg))
    settled = 0; kframe = 0; turn_t = 0.0
    for i in range(int(args.turn_timeout / CTRL_DT)):
        _, _, yaw = g.odom()
        e = wrap(yaw - psi_goal)
        wz = float(np.clip(-k_yaw * e, -1.0, 1.0))
        step_policy(np.array([args.turn_vx, 0.0, wz], np.float32))
        maybe_render(kframe); kframe += 1
        turn_t = (i + 1) * CTRL_DT
        settled = settled + 1 if abs(e) < np.radians(2) else 0
        if settled >= int(0.3 / CTRL_DT):        # 穩住 0.3s 視為到位
            break

    # ---- Latch：鎖定當下 odom 位置+航向為目標線 ----
    x0, y0, psi_target = g.odom()
    p0 = np.array([x0, y0]); d_hat, _ = line_frame(psi_target)
    print(f"[latch] 轉向完成 psi_target={np.degrees(psi_target):+.1f}° "
          f"起點=({x0:+.2f},{y0:+.2f}) 轉向耗時={turn_t:.1f}s")

    # ---- Phase 2：沿目標線直走 ----
    traj = []; ects = []; eyaws = []; fzmin = fzmax = None; fell = None
    for i in range(int(args.secs / CTRL_DT)):
        t = i * CTRL_DT; push_at(t)
        x, y, yaw = g.odom(); p = np.array([x, y])
        cmd, e_ct, e_yaw = line_control(p, yaw, p0, psi_target, args.vx,
                                        k_yaw, args.k_ct, args.no_lateral)
        step_policy(cmd)
        traj.append(p.copy()); ects.append(e_ct); eyaws.append(e_yaw)
        fz = g.d.geom_xpos[fl_gid][2]
        fzmin = fz if fzmin is None else min(fzmin, fz)
        fzmax = fz if fzmax is None else max(fzmax, fz)
        if g.height < 0.15 and fell is None: fell = t
        maybe_render(kframe); kframe += 1

    push_at(1e9)                                  # 收尾清掉殘留外力
    if not traj:
        print("[result] 直走階段 0 步（--secs 太小），略過量測"); return
    traj = np.array(traj); ects = np.array(ects); eyaws = np.array(eyaws)
    fwd = float(d_hat @ (traj[-1] - p0))
    max_ct = float(np.max(np.abs(ects))); fin_ct = float(ects[-1])
    yaw_rms = float(np.degrees(np.sqrt(np.mean(eyaws ** 2))))
    print(f"[result] 沿線前進={fwd:+.2f}m  max|側偏|={max_ct:.3f}m  末端側偏={fin_ct:+.3f}m  "
          f"航向RMS={yaw_rms:.2f}°  跌倒={'是@%.1fs' % fell if fell else '否'}")
    print(f"[result] FL 抬腳量 ≈ {fzmax - fzmin:.3f} m  (no_lateral={args.no_lateral})")

    tag = "_nolat" if args.no_lateral else ""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    L = max(fwd, 1.0) + 0.5
    seg = np.stack([p0, p0 + L * d_hat])
    ax.plot(seg[:, 0], seg[:, 1], "k--", lw=1, label="target line")
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="path")
    ax.plot([p0[0]], [p0[1]], "go", ms=8, label="latch/start")
    ax.set_aspect("equal"); ax.grid(True); ax.legend()
    ax.set_title(f"odom straight turn={args.turn_deg:+.0f}deg "
                 f"max|ct|={max_ct:.3f}m yawRMS={yaw_rms:.2f}deg"
                 + ("  [no_lateral]" if args.no_lateral else ""))
    pout = f"/home/huang/rbtdog_sim/task4/outputs/odom_line{tag}.png"
    fig.savefig(pout, dpi=120, bbox_inches="tight"); print("[result] 軌跡圖:", pout)

    if frames:
        import imageio.v2 as iio
        out = f"/home/huang/rbtdog_sim/task4/outputs/odom_line{tag}.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264"); print("[result] 影片:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--vx", type=float, default=VX_CMD)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--turn_deg", type=float, default=-45.0,
                    help="開走前的相對轉角(度)，右轉為負；預設右轉 45°")
    ap.add_argument("--turn_vx", type=float, default=0.0, help="轉向階段的前進指令(原地轉=0)")
    ap.add_argument("--turn_timeout", type=float, default=6.0, help="轉向階段秒數上限")
    ap.add_argument("--k_ct", type=float, default=1.5, help="cross-track P 增益(橫向修正)")
    ap.add_argument("--no_lateral", action="store_true",
                    help="關閉 vy 橫向修正(vy≡0)，重現舊的只鎖航向行為做 A/B 對照")
    ap.add_argument("--w_coup", type=float, default=8.0,
                    help="CPG 腿間耦合強度，須與訓練一致：耦合版=8.0，無耦合版=0.0")
    run(ap.parse_args())
