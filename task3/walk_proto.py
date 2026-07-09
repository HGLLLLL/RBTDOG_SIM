"""Go2 CPG trot 走路原型 v2 —— 用腳掌軌跡 + 站姿雅可比反解關節角度。
力矩致動器 → 軟體 PD。站立相腳貼地往後推、擺動相抬起前移。
用法: python walk_proto.py [--stride .12 --lift .06 --freq 2 --duty .6 --z0 -.27 --kp 30 --kd 1 --secs 6 --video out.mp4]
"""
import argparse, numpy as np, mujoco

MODEL = "/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene.xml"
LEGS = ["FL", "FR", "RL", "RR"]
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}  # trot 對角同相
HOME = np.array([0.0, 0.9, -1.8])  # hip, thigh, calf


def foot_xz(m, d, th, ca):
    """FL 腳掌 (x,z) 相對 thigh 樞紐 (sagittal)。"""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[8] = th; d.qpos[9] = ca
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    piv = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FL_thigh")]
    p = d.geom_xpos[gid] - piv
    return np.array([p[0], p[2]])


def calc_ik(m, d):
    """回傳 home 腳掌位置 f0 與 inv(Jacobian) （thigh/calf 對 x/z）。"""
    f0 = foot_xz(m, d, 0.9, -1.8)
    e = 1e-3
    J = np.zeros((2, 2))
    J[:, 0] = (foot_xz(m, d, 0.9 + e, -1.8) - foot_xz(m, d, 0.9 - e, -1.8)) / (2 * e)
    J[:, 1] = (foot_xz(m, d, 0.9, -1.8 + e) - foot_xz(m, d, 0.9, -1.8 - e)) / (2 * e)
    return f0, np.linalg.inv(J)


def foot_traj(phase, p):
    """給定 normalized phase[0,1)，回傳腳掌 (x,z) 相對站姿中心的偏移。"""
    ph = phase % 1.0
    if ph < p.duty:                       # 站立相：貼地由前往後
        s = ph / p.duty
        x = p.stride * (0.5 - s)
        z = 0.0
    else:                                 # 擺動相：抬起由後往前
        s = (ph - p.duty) / (1 - p.duty)
        x = p.stride * (-0.5 + s)
        z = p.lift * np.sin(np.pi * s)
    return x, z


def run(p):
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    f0, Jinv = calc_ik(m, d)
    # 站姿中心 = home 腳掌位置，但 z 下移到 z0（讓腿載重）
    center = np.array([f0[0], p.z0])
    flimit = m.actuator_ctrlrange[:, 1]

    def targets(t):
        q = np.zeros(12)
        for k, leg in enumerate(LEGS):
            ph = p.freq * t + LEG_PHASE[leg]
            dx, dz = foot_traj(ph, p)
            foot = center + np.array([dx, dz])
            dth, dca = Jinv @ (foot - f0)      # 相對 home 腳掌的位移 → 角度增量
            q[3 * k:3 * k + 3] = [0.0, HOME[1] + dth, HOME[2] + dca]
        return q

    mujoco.mj_resetDataKeyframe(m, d, 0)

    def step(qdes):
        tau = p.kp * (qdes - d.qpos[7:19]) - p.kd * d.qvel[6:18]
        d.ctrl[:] = np.clip(tau, -flimit, flimit)
        mujoco.mj_step(m, d)

    for _ in range(int(0.5 / m.opt.timestep)):   # 先站穩
        step(targets(0.0))
    x0, y0 = d.qpos[0], d.qpos[1]

    renderer = None; frames = []
    if p.video:
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    log = []
    fell = None
    n = int(p.secs / m.opt.timestep)
    for i in range(n):
        t = i * m.opt.timestep
        step(targets(t))
        w, x, y, z = d.qpos[3:7]
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
        log.append((d.qpos[0]-x0, d.qpos[1]-y0, d.qpos[2], roll, pitch))
        if d.qpos[2] < 0.15 and fell is None: fell = t
        if renderer is not None and i % max(1, int(1/(30*m.opt.timestep))) == 0:
            cam.lookat[:] = d.qpos[:3]; cam.distance = 2.0; cam.elevation = -20
            renderer.update_scene(d, cam); frames.append(renderer.render())
    log = np.array(log)
    print(f"[params] stride={p.stride} lift={p.lift} freq={p.freq} duty={p.duty} z0={p.z0} kp={p.kp} kd={p.kd}")
    print(f"前進(x)= {log[-1,0]:+.2f} m   側偏(y)= {log[-1,1]:+.2f} m   末端高度= {log[-1,2]:.2f} m")
    print(f"max|roll|={np.degrees(np.abs(log[:,3]).max()):.0f}deg max|pitch|={np.degrees(np.abs(log[:,4]).max()):.0f}deg 跌倒={'是@%.1fs'%fell if fell else '否'}")
    if renderer is not None and frames:
        import imageio.v2 as iio; iio.mimsave(p.video, frames, duration=1000/30, loop=0); print("影片:", p.video)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=0.12)
    ap.add_argument("--lift", type=float, default=0.06)
    ap.add_argument("--freq", type=float, default=2.0)
    ap.add_argument("--duty", type=float, default=0.6)
    ap.add_argument("--z0", type=float, default=-0.27)
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--video", type=str, default="")
    run(ap.parse_args())
