"""CPG v2：論文 CPG + 每腿可學抬腳 gc；相容 fixed(12)/learnable(16) 兩種動作。"""
import numpy as np
import mujoco
import jax.numpy as jnp

MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 4.5
A_CONV = 50.0
D_STEP = 0.12
G_C = 0.08                       # 固定抬腳（舊模型用）
G_P = 0.01
GC_MIN, GC_MAX = 0.03, 0.15      # 可學抬腳範圍（新模型用）
W_COUP = 8.0
N_CPG_SUB = 4
LEGS = ["FL", "FR", "RL", "RR"]
HOME3 = jnp.array([0.0, 0.9, -1.8])
HOME3_np = np.array([0.0, 0.9, -1.8])
PHASE_OFFSET = jnp.array([0.0, jnp.pi, jnp.pi, 0.0])
PHI = PHASE_OFFSET[None, :] - PHASE_OFFSET[:, None]


def detect_mode(act_dim):
    if act_dim == 12:
        return "fixed"
    if act_dim == 16:
        return "learnable"
    raise ValueError(f"未知動作維度 {act_dim}（僅支援 12/16）")


def action_to_cpg_cmd(action, mode):
    if mode == "fixed":
        a = jnp.tanh(action).reshape(4, 3)
        gc = jnp.full(4, G_C)
    else:
        a4 = jnp.tanh(action).reshape(4, 4)
        a = a4[:, :3]
        gc = (a4[:, 3] + 1) / 2 * (GC_MAX - GC_MIN) + GC_MIN
    mux = (a[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy = (a[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    omega = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, omega, gc


def cpg_init():
    return {"rx": jnp.full(4, 1.5), "rx_d": jnp.zeros(4),
            "ry": jnp.full(4, 1.5), "ry_d": jnp.zeros(4), "theta": PHASE_OFFSET}


def cpg_step(c, mux, muy, omega, dt):
    rx, rxd, ry, ryd, th = c["rx"], c["rx_d"], c["ry"], c["ry_d"], c["theta"]
    h = dt / N_CPG_SUB
    for _ in range(N_CPG_SUB):
        rxd = rxd + A_CONV * (A_CONV / 4.0 * (mux - rx) - rxd) * h
        rx = rx + rxd * h
        ryd = ryd + A_CONV * (A_CONV / 4.0 * (muy - ry) - ryd) * h
        ry = ry + ryd * h
        rbar = 0.5 * (rx + ry)
        diff = th[None, :] - th[:, None] - PHI
        coup = jnp.sum(rbar[None, :] * jnp.sin(diff), axis=1)
        th = th + (2.0 * jnp.pi * omega + W_COUP * coup) * h
    th = jnp.mod(th, 2.0 * jnp.pi)
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th}


def cpg_foot_offsets(c, gc):
    th = c["theta"]
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
    dx = -D_STEP * fx * jnp.cos(th)
    dy = D_STEP * fy * jnp.cos(th)
    dz = jnp.where(jnp.sin(th) > 0, gc * jnp.sin(th), G_P * jnp.sin(th))
    return jnp.stack([dx, dy, dz], axis=-1)


def cpg_to_joint_targets(c, jinvs, gc):
    off = cpg_foot_offsets(c, gc)
    dq = jnp.einsum("kij,kj->ki", jinvs, off)
    q = HOME3[None, :] + dq
    return q.reshape(12)


def leg_ik_consts(scene_path):
    m = mujoco.MjModel.from_xml_path(scene_path); d = mujoco.MjData(m)
    jinvs = []
    for k, leg in enumerate(LEGS):
        jb = 7 + 3 * k
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg)
        hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg + "_hip")

        def foot(q3):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.qpos[jb:jb + 3] = q3; mujoco.mj_forward(m, d)
            return (d.geom_xpos[gid] - d.xpos[hip]).copy()
        e = 1e-3; J = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros(3); dq[j] = e
            J[:, j] = (foot(HOME3_np + dq) - foot(HOME3_np - dq)) / (2 * e)
        jinvs.append(np.linalg.inv(J))
    return np.array(jinvs, np.float32)
