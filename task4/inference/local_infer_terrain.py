"""本機 CPU 推論（地形版）：載入 cpg_rl_terrain_params.pkl，在「平台+分段斜坡」上走上坡/下坡。

忠實重現訓練動力學：scene_mjx.xml + apply_pd（位置伺服 emulate kp=90/kd=3）、
obs 觸地布林用 gz 相對地面（與 cpg_rl_terrain_colab.ipynb 逐項一致）。

用法:
  conda run -n rbtdog python task4/inference/local_infer_terrain.py            # 上坡+下坡各一段, 出影片
  conda run -n rbtdog python task4/inference/local_infer_terrain.py --secs 12
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

SCENE = "/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene_mjx.xml"
PARAMS = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain_params.pkl"
OUTDIR = "/home/huang/rbtdog_sim/task4/outputs"

LEGS = ["FL", "FR", "RL", "RR"]
HOME3 = np.array([0.0, 0.9, -1.8])
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
# ---- 常數：與 notebook 一致 ----
MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 4.5
A_CONV, D_STEP, G_C, G_P, W_COUP, N_CPG_SUB = 50.0, 0.12, 0.08, 0.01, 8.0, 4
CTRL_DT, SIM_DT = 0.02, 0.004
FOOT_CONTACT_H = 0.03
OBS_DIM, ACT_DIM = 76, 12
KNEE_IDX = [2, 5, 8, 11]
_PO = np.array([0.0, np.pi, np.pi, 0.0])
PHI = _PO[None, :] - _PO[:, None]

# ---- 地形折點（與 notebook 同源）----
_dz1 = 2.0 * np.tan(np.radians(7.5))
_dz2 = 3.0 * np.tan(np.radians(15.0))
KNOTS_X = np.array([-6.0, -3.0, -1.0, 1.0, 3.0, 6.0], np.float32)
KNOTS_Z = np.array([-(_dz1 + _dz2), -_dz1, 0.0, 0.0, _dz1, _dz1 + _dz2], np.float32)
TERR_WY, TERR_TH = 3.0, 0.5


def gz_np(x):
    return np.interp(x, KNOTS_X, KNOTS_Z)


def build_terrain_model():
    spec = mujoco.MjSpec.from_file(SCENE)
    floor = next(g for g in spec.geoms if g.name == "floor")
    floor.pos = [0.0, 0.0, -10.0]
    for i in range(len(KNOTS_X) - 1):
        xa, za, xb, zb = KNOTS_X[i], KNOTS_Z[i], KNOTS_X[i + 1], KNOTS_Z[i + 1]
        L = float(np.hypot(xb - xa, zb - za)); a = float(np.arctan2(zb - za, xb - xa))
        mx, mz = 0.5 * (xa + xb), 0.5 * (za + zb)
        nx, nz = -np.sin(a), np.cos(a)
        g = spec.worldbody.add_geom()
        g.name = f"ramp{i}"; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [L / 2, TERR_WY, TERR_TH]
        g.pos = [mx - nx * TERR_TH, 0.0, mz - nz * TERR_TH]
        g.quat = [np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0]
        g.rgba = [0.55, 0.5, 0.45, 1.0]
    return spec.compile()


def apply_pd(m, kp=90.0, kd=3.0):
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kd
    fr = np.full(m.nu, 23.7); fr[KNEE_IDX] = 45.43
    m.actuator_forcerange[:, 0] = -fr; m.actuator_forcerange[:, 1] = fr
    m.actuator_forcelimited[:] = 1
    return m


# ---- CPG（numpy）----
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
        rbar = 0.5 * (rx + ry); diff = th[None, :] - th[:, None] - PHI
        th = th + (2 * np.pi * omega + W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th % (2 * np.pi)}


def act_to_cmd(a):
    a = np.tanh(a).reshape(4, 3)
    mux = (a[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy = (a[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    om = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, om


def joint_targets(c, jinvs):
    th = c["theta"]
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    dx = -D_STEP * fx * np.cos(th); dy = D_STEP * fy * np.cos(th)
    dz = np.where(np.sin(th) > 0, G_C * np.sin(th), G_P * np.sin(th))
    off = np.stack([dx, dy, dz], -1); q = np.zeros((4, 3))
    for k in range(4):
        q[k] = HOME3 + jinvs[k] @ off[k]
    return q.reshape(12)


def leg_ik_consts(m):
    d = mujoco.MjData(m); jinvs = []
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
            J[:, j] = (foot(HOME3 + dq) - foot(HOME3 - dq)) / (2 * e)
        jinvs.append(np.linalg.inv(J))
    return np.array(jinvs)


def qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qrot(q, v):
    u = q[1:4]; t = 2 * np.cross(u, v); return v + q[0] * t + np.cross(u, t)
def w2b(q, v): return qrot(qinv(q), v)


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
    import jax.numpy as jnp
    def infer(obs):
        return np.asarray(jpol(jnp.asarray(obs), key)[0])
    infer(np.zeros(OBS_DIM, np.float32))
    return infer


def build_obs(d, c, cmd, last_a, foot_gid):
    quat = d.qpos[3:7]
    grav = w2b(quat, np.array([0, 0, -1.0])); blin = w2b(quat, d.qvel[0:3])
    gyro = d.qvel[3:6]
    fxs = np.array([d.geom_xpos[g][0] for g in foot_gid])
    fzs = np.array([d.geom_xpos[g][2] for g in foot_gid])
    contact = ((fzs - gz_np(fxs)) < FOOT_CONTACT_H).astype(np.float32)     # 相對地面
    return np.concatenate([grav, blin, gyro, d.qpos[7:19] - HOME12, d.qvel[6:18],
                           cmd, last_a, contact, c["rx"], c["rx_d"], c["ry"], c["ry_d"],
                           np.sin(c["theta"]), np.cos(c["theta"])]).astype(np.float32)


def rollout(infer, jinvs, downhill, secs, video=True):
    m = build_terrain_model(); m.opt.timestep = SIM_DT; m = apply_pd(m)
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [0, 0, 0, 1.0] if downhill else [1.0, 0, 0, 0]          # 面 -x / +x
    mujoco.mj_forward(m, d)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in LEGS]
    fl = foot_gid[0]; nsub = int(round(CTRL_DT / SIM_DT))
    tag = "down" if downhill else "up"

    ren = cam = None; frames = []
    if video:
        ren = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)

    c = cpg_init(); last_a = np.zeros(ACT_DIM); cmd = np.array([0.6, 0.0, 0.0])
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    fz_rel_min = fz_rel_max = None; ymax = 0.0; fell = None
    n = int(secs / CTRL_DT)
    for i in range(n):
        obs = build_obs(d, c, cmd.astype(np.float32), last_a, foot_gid)
        act = infer(obs)
        mux, muy, om = act_to_cmd(act); c = cpg_step(c, mux, muy, om, CTRL_DT)
        d.ctrl[:] = np.clip(joint_targets(c, jinvs), lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)
        last_a = act
        ymax = max(ymax, abs(float(d.qpos[1]) - y0))
        grav = w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        rel_h = float(d.qpos[2]) - gz_np(float(d.qpos[0]))
        if (rel_h < 0.15 or grav[2] > -0.4) and fell is None: fell = i * CTRL_DT
        fzr = float(d.geom_xpos[fl][2]) - gz_np(float(d.geom_xpos[fl][0]))
        fz_rel_min = fzr if fz_rel_min is None else min(fz_rel_min, fzr)
        fz_rel_max = fzr if fz_rel_max is None else max(fz_rel_max, fzr)
        if ren is not None and i % 2 == 0:
            cam.lookat[:] = d.qpos[:3]; cam.distance = 2.6; cam.elevation = -18; cam.azimuth = 90
            ren.update_scene(d, cam); frames.append(ren.render())

    xf = float(d.qpos[0]); horiz = abs(xf - x0); climb = gz_np(xf) - gz_np(x0)
    rel_h_end = float(d.qpos[2]) - gz_np(xf)
    print(f"[{tag:4s}] 水平前進={horiz:.2f}m 爬升={climb:+.2f}m 最大側偏={ymax:.2f}m "
          f"抬腳={fz_rel_max - fz_rel_min:.3f}m 末端相對高={rel_h_end:.2f} "
          f"跌倒={'是@%.1fs' % fell if fell else '否'}")
    if video and frames:
        import imageio.v2 as iio
        os.makedirs(OUTDIR, exist_ok=True)
        out = f"{OUTDIR}/terrain_infer_{tag}.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264"); print(f"       影片: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--novideo", action="store_true")
    args = ap.parse_args()
    print(f"載入 {PARAMS}"); infer = load_policy(PARAMS)
    m0 = build_terrain_model(); jinvs = leg_ik_consts(m0)
    print("=== 上坡 ==="); rollout(infer, jinvs, False, args.secs, not args.novideo)
    print("=== 下坡 ==="); rollout(infer, jinvs, True, args.secs, not args.novideo)


if __name__ == "__main__":
    main()
