"""本機 CPU 推論：載入 Colab 訓練好的 CPG-RL 策略，接回 task3 的 Go2 + 羅盤走直線。

流程（每 50Hz 控制步）：
  讀狀態 -> 組 observation(51維) -> 策略輸出 5維動作 -> Hopf CPG 積分
  -> 腳掌軌跡 + task3 的 IK -> 關節目標角度 -> 軟體 PD(kp=90/kd=3) 逐子步 -> mj_step

關鍵：這裡的 observation 組法、CPG 參數、控制頻率、PD 增益，都必須和
Colab 的 cpg_rl_colab.ipynb 訓練環境「逐項一致」，否則策略會亂走。

用法：
  # 還沒有權重時，先用固定動作把管線跑通（驗證物理/CPG/渲染）：
  python local_infer.py --dummy --secs 8 --video

  # 拿到 Colab 下載的權重後：
  python local_infer.py --params cpg_rl_params.pkl --secs 20 --video --push

依賴：mujoco（已有）、本機推論需 `pip install jax brax`（CPU 版即可；--dummy 模式不需要）。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
import mujoco
from go2_gait import Go2Gait, HOME            # 重用 task3 的模型/PD/IK/羅盤
from walk_line import GAIT, COMPASS_GAIN, PUSHES  # 重用 task3 步態參數與干擾排程

# ============================================================
# 常數：必須與 Colab 訓練環境完全一致（見 cpg_rl_colab.ipynb 第 2 步）
# ============================================================
CTRL_DT = 0.02                       # 控制步長 50Hz
A_CONV = 50.0
MU_MIN, MU_MAX = 0.0, 1.0
OMEGA_MIN, OMEGA_MAX = 0.5, 3.5
STRIDE = 0.32
LIFT = 0.12
PHASE_OFFSET = np.array([0.0, np.pi, np.pi, 0.0])   # trot FL,FR,RL,RR
N_CPG_SUB = 4
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
OBS_DIM, ACT_DIM = 51, 5

# 羅盤航向控制
TARGET_YAW = 0.0                     # 想維持的世界航向（rad）
HEADING_GAIN = 3.0                   # 航向誤差 -> 目標 yaw 角速度 的 P 增益
VX_CMD = 0.6                         # 前進速度指令 (m/s)


# ---------------- NumPy 版 CPG（與 JAX 版邏輯逐行對應）----------------
def cpg_init():
    return {"r": np.zeros(4), "r_dot": np.zeros(4), "phi": 0.0}


def cpg_step(c, mu, omega, dt):
    r, rd, phi = c["r"].copy(), c["r_dot"].copy(), c["phi"]
    h = dt / N_CPG_SUB
    for _ in range(N_CPG_SUB):
        rdd = A_CONV * (A_CONV / 4.0 * (mu - r) - rd)
        rd = rd + rdd * h
        r = r + rd * h
        phi = phi + 2.0 * np.pi * omega * h
    return {"r": r, "r_dot": rd, "phi": phi % (2.0 * np.pi)}


def action_to_cpg_cmd(a):
    a = np.tanh(a)
    mu = (a[:4] + 1.0) / 2.0 * (MU_MAX - MU_MIN) + MU_MIN
    omega = (a[4] + 1.0) / 2.0 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mu, omega


def cpg_to_joint_targets(c, f0, jinv, center):
    theta = c["phi"] + PHASE_OFFSET
    amp = np.clip(c["r"], 0.0, 1.0)
    # 站立相(sin<=0,腳貼地)腳掌 前->後 推身體前進；擺動相(sin>0,腳抬起) 後->前 回擺。
    dx = -STRIDE * amp * np.cos(theta)
    dz = np.maximum(0.0, LIFT * amp * np.sin(theta))
    foot = np.stack([center[0] + dx, center[1] + dz], axis=-1)   # (4,2)
    dq = (foot - f0) @ jinv.T
    q = np.zeros((4, 3))
    q[:, 1] = 0.9 + dq[:, 0]
    q[:, 2] = -1.8 + dq[:, 1]
    return q.reshape(12)


# ---------------- 四元數 / observation（與訓練 _obs 逐項對應）----------------
def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    u = q[1:4]
    t = 2.0 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def world_to_body(quat, v):
    return _qrot(_qinv(quat), v)


def build_obs(g, cpg, cmd, last_action):
    q, v = g.d.qpos, g.d.qvel
    quat = q[3:7]
    grav = world_to_body(quat, np.array([0.0, 0.0, -1.0]))
    blin = world_to_body(quat, v[0:3])       # 世界線速度 -> 本體
    gyro = v[3:6]                             # 自由關節角速度即本體系
    jpos = q[7:19] - HOME12
    jvel = v[6:18]
    return np.concatenate([
        grav, blin, gyro, jpos, jvel, cmd, last_action,
        cpg["r"], cpg["r_dot"],
        [np.sin(cpg["phi"]), np.cos(cpg["phi"])],
    ]).astype(np.float32)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------- 策略載入 ----------------
def load_policy(params_path):
    """用與 Colab 相同的網路設定重建策略並載入權重。回傳 fn(obs)->action(5,)。"""
    import jax
    import functools
    from brax.io import model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics

    factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(256, 256, 256),
    )
    net = factory(OBS_DIM, ACT_DIM,
                  preprocess_observations_fn=running_statistics.normalize)
    make_policy = ppo_networks.make_inference_fn(net)
    params = model.load_params(params_path)
    policy = make_policy(params, deterministic=True)
    key = jax.random.PRNGKey(0)
    jpol = jax.jit(policy)

    def infer(obs):
        act, _ = jpol(jax.numpy.asarray(obs), key)
        return np.asarray(act)

    # 先打一次確認能跑
    _ = infer(np.zeros(OBS_DIM, np.float32))
    return infer


def dummy_policy():
    """固定動作：mu≈0.7、omega≈2Hz 的等速 trot，用來在沒有權重時測通管線。"""
    fixed = np.array([0.42, 0.42, 0.42, 0.42, 0.0], np.float32)  # ->mu~0.7, omega~2.0
    return lambda obs: fixed


# ---------------- 主流程 ----------------
def run(args):
    g = Go2Gait(**GAIT)
    g.reset()
    base_id = mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_BODY, "base")
    n_sub = int(round(CTRL_DT / g.m.opt.timestep))     # 每控制步的物理子步數（0.02/0.002=10）
    print(f"[info] timestep={g.m.opt.timestep} n_sub={n_sub} kp={g.kp} kd={g.kd} "
          f"flimit(前3)={g.flimit[:3]}")

    if args.dummy or not args.params:
        if not args.params and not args.dummy:
            print("[warn] 未提供 --params，改用 dummy 固定動作（僅測管線，不是學到的策略）")
        infer = dummy_policy()
    else:
        print(f"[info] 載入策略 {args.params} ...")
        infer = load_policy(args.params)
        print("[info] 策略載入成功")

    def apply_control(q_des):
        # 逐物理子步重算 PD（複刻訓練時位置伺服每步重算的行為）
        lo = -g.flimit
        for _ in range(n_sub):
            tau = g.kp * (q_des - g.d.qpos[7:19]) - g.kd * g.d.qvel[6:18]
            g.d.ctrl[:] = np.clip(tau, lo, g.flimit)
            mujoco.mj_step(g.m, g.d)

    def push_at(t):
        g.d.xfrc_applied[base_id] = 0.0
        if not args.push:
            return None
        for t0, dur, fy, tz in PUSHES:
            if t0 <= t < t0 + dur:
                g.d.xfrc_applied[base_id, 1] = fy
                g.d.xfrc_applied[base_id, 5] = tz
                return fy, tz
        return None

    # 先站穩 0.5s（維持 home 目標）
    for _ in range(int(0.5 / CTRL_DT)):
        apply_control(HOME12.copy())

    renderer = cam = None
    frames = []
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(g.m, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(g.m, cam)

    cpg = cpg_init()
    last_action = np.zeros(ACT_DIM)
    traj, head = [], []
    x0, y0 = g.xy
    n_steps = int(args.secs / CTRL_DT)
    fell = None
    for i in range(n_steps):
        t = i * CTRL_DT
        push = push_at(t)

        # 羅盤航向 P 控制 -> 目標 yaw 角速度指令（rad/s，夾到訓練範圍 [-1,1]）
        yaw_err = wrap(g.compass_yaw() - TARGET_YAW)
        yaw_rate_cmd = float(np.clip(-HEADING_GAIN * yaw_err, -1.0, 1.0))
        cmd = np.array([args.vx, 0.0, yaw_rate_cmd], np.float32)

        obs = build_obs(g, cpg, cmd, last_action)
        action = infer(obs)
        mu, omega = action_to_cpg_cmd(action)
        cpg = cpg_step(cpg, mu, omega, CTRL_DT)
        q_des = cpg_to_joint_targets(cpg, g.f0, g.Jinv, g.center)
        apply_control(q_des)
        last_action = action

        traj.append(g.xy.copy())
        head.append(g.true_yaw())
        if g.height < 0.15 and fell is None:
            fell = t
        if renderer is not None and i % 2 == 0:
            x, y = g.xy
            cam.lookat[:] = [x, y, 0.3]
            cam.distance = 2.5
            cam.elevation = -20
            cam.azimuth = 90
            renderer.update_scene(g.d, cam)
            frames.append(renderer.render())

    traj = np.array(traj)
    print(f"[result] 前進 x={traj[-1,0]-x0:+.2f} m  側偏 y={traj[-1,1]-y0:+.2f} m  "
          f"末端高度={g.height:.2f} m  跌倒={'是@%.1fs' % fell if fell else '否'}")

    if frames:
        import imageio.v2 as iio
        out = "/home/huang/rbtdog_sim/task4/outputs/cpg_rl_infer.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264")
        print("[result] 影片:", out)

    # 軌跡圖
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(0, color="r", ls="--", lw=1, alpha=0.6, label="target heading (+x)")
        ax.plot(traj[:, 0] - x0, traj[:, 1] - y0, "b-", lw=1.6,
                label=f"CPG-RL (end y={traj[-1,1]-y0:+.2f}m)")
        ax.plot(0, 0, "go", ms=9, label="start")
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y lateral (m)")
        ax.set_title("CPG-RL local inference: walk straight + heading hold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        out = "/home/huang/rbtdog_sim/task4/outputs/cpg_rl_infer.png"
        plt.savefig(out, dpi=110)
        print("[result] 軌跡圖:", out)
    except Exception as e:
        print("[warn] 畫圖略過:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="",
                    help="Colab 下載的權重 cpg_rl_params.pkl 路徑；省略則用 dummy")
    ap.add_argument("--dummy", action="store_true", help="強制用固定動作測管線")
    ap.add_argument("--secs", type=float, default=12.0, help="模擬秒數")
    ap.add_argument("--vx", type=float, default=VX_CMD, help="前進速度指令 m/s")
    ap.add_argument("--video", action="store_true", help="輸出影片")
    ap.add_argument("--push", action="store_true", help="加入側推+yaw 干擾（沿用 walk_line 排程）")
    run(ap.parse_args())
