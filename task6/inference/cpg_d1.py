"""D1 EDU 的 CPG 動力學、動作解碼與腿部 IK。純函式，不做任何 I/O。

與 task4 論文標準版逐行對應；差別只在 HOME3 與模型名稱由 d1_model 提供。
IK 用「home 姿態的數值 Jacobian 求逆」，對關節正負號自動免疫，
因此 D1 的 knee 軸為 +y、hip 正負號與 Go2 相反都不需要特別處理。

足端偏移的前後與側向用兩個不同尺度：前後維持 D_STEP=0.12（不犧牲走速），
側向改用較小的 D_STEP_Y=0.09。原因是側向偏移只由 abad 關節吸收（靈敏度約
4.47 rad/m），而本機 D1 輪足的 abad 行程僅 ±28°，遠小於 task4 目標機 Go2 的
±60°；沿用 0.12 會讓 abad 目標角達 ±0.536 rad，超出致動器 ctrlrange ±0.4687。
"""
import mujoco
import numpy as np

from d1_model import (A_CONV, D_STEP, D_STEP_Y, G_C, G_P, HOME3, LEG_QPOS_IDX,
                      LEGS, MU_MAX, MU_MIN, N_CPG_SUB, OMEGA_MAX, OMEGA_MIN,
                      PHASE_OFFSET, W_COUP)

PHI = PHASE_OFFSET[None, :] - PHASE_OFFSET[:, None]   # (4,4) 目標相位差


def cpg_init() -> dict:
    return {"rx": np.full(4, 1.5), "rx_d": np.zeros(4),
            "ry": np.full(4, 1.5), "ry_d": np.zeros(4),
            "theta": PHASE_OFFSET.copy()}


def cpg_step(c: dict, mux, muy, omega, dt: float) -> dict:
    rx, rxd, ry, ryd, th = (c["rx"].copy(), c["rx_d"].copy(),
                            c["ry"].copy(), c["ry_d"].copy(), c["theta"].copy())
    h = dt / N_CPG_SUB
    for _ in range(N_CPG_SUB):
        rxd += (A_CONV * (A_CONV / 4 * (mux - rx) - rxd)) * h
        rx += rxd * h
        ryd += (A_CONV * (A_CONV / 4 * (muy - ry) - ryd)) * h
        ry += ryd * h
        rbar = 0.5 * (rx + ry)
        diff = th[None, :] - th[:, None] - PHI
        th = th + (2 * np.pi * omega + W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th % (2 * np.pi)}


def act_to_cmd(a: np.ndarray):
    """12 維動作 → 每腿 (mux, muy, omega)。"""
    a = np.tanh(a).reshape(4, 3)
    mux = (a[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy = (a[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    om = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, om


def leg_ik_consts(m: mujoco.MjModel):
    """在 home 姿態對每腿求 3x3 數值 Jacobian（腳掌相對髖 對 三個關節角）。

    回傳 (f0s, jinvs)：f0s[k] 是第 k 腿 home 時腳掌相對髖的位置 (3,)，
    jinvs[k] 是 Jacobian 的逆 (3,3)。
    """
    d = mujoco.MjData(m)
    f0s, jinvs = [], []
    for k, leg in enumerate(LEGS):
        sl = LEG_QPOS_IDX[3 * k:3 * k + 3]   # 輪關節夾在腿關節之間，位址不連續
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg)
        hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_abad")

        def foot(q3):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.qpos[sl] = q3
            mujoco.mj_forward(m, d)
            return (d.geom_xpos[gid] - d.xpos[hip]).copy()

        f0 = foot(HOME3)
        e = 1e-3
        J = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = e
            J[:, j] = (foot(HOME3 + dq) - foot(HOME3 - dq)) / (2 * e)
        f0s.append(f0)
        jinvs.append(np.linalg.inv(J))
    return np.array(f0s), np.array(jinvs)


def joint_targets(c: dict, f0s, jinvs) -> np.ndarray:
    """CPG 狀態 → 12 個關節目標角。"""
    th = c["theta"]
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    dx = -D_STEP * fx * np.cos(th)
    dy = D_STEP_Y * fy * np.cos(th)   # 側向用較小的尺度，見 d1_model.D_STEP_Y
    dz = np.where(np.sin(th) > 0, G_C * np.sin(th), G_P * np.sin(th))
    off = np.stack([dx, dy, dz], -1)
    q = np.zeros((4, 3))
    for k in range(4):
        q[k] = HOME3 + jinvs[k] @ off[k]
    return q.reshape(12)


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    u = q[1:4]
    t = 2 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def w2b(q, v):
    """世界向量轉到機身座標系。"""
    return qrot(qinv(q), v)
