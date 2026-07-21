"""slope 課程爬坡力/打滑診斷：比較各模型在 15° 上坡段的前進與 slip。
落地自 terrain3 除錯（systematic-debugging）；重訓 terrain3b 後可直接複驗。"""

import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
import jax.numpy as jnp
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/analysis")
import terrain_compare as TC
import local_infer_terrain as T
import local_infer_paper as P
import cpg2, cpg3
import obs2 as O
import local_infer_terrain2 as L2

W = {
    "v2.0":       "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain2_params.pkl",
    "terrain2_1": "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain2_1_params.pkl",
    "terrain3":   "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain3_params.pkl",
    "terrain3b":  "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain3b_params.pkl",
}
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
K_YAW, K_CT = 3.0, 1.5
X15_LO, X15_HI = 12.2, 14.0     # 15° 上坡段


def run(params, C, secs=55.0, gc_cap=None):
    m, gz, phase, extra = TC.build_course_slopes()
    m.opt.timestep = T.SIM_DT; m = T.apply_pd(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS]
    jinvs = jnp.asarray(T.leg_ik_consts(m))
    nsub = int(round(T.CTRL_DT / T.SIM_DT))
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    infer, ad = L2.load_policy_any(params); mode = C.detect_mode(ad)
    c = C.cpg_init(); last = np.zeros(ad)
    for _ in range(int(0.4 / T.CTRL_DT)):
        d.ctrl[:] = np.clip(HOME12, lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)
    prev_xy = np.array([[d.geom_xpos[g][0], d.geom_xpos[g][1]] for g in foot_gid])
    fell = None; gclog = []; slip15 = []; gc15 = []
    x_at15_enter = None; x_at15_exit = None; t0_15 = t1_15 = None
    for i in range(int(secs / T.CTRL_DT)):
        x, y, yaw = float(d.qpos[0]), float(d.qpos[1]), TC.yaw_from_quat(d.qpos[3:7])
        cmd, e_ct, _ = P.line_control(np.array([x, y]), yaw, np.array([0.0, 0.0]), 0.0, TC.VX, K_YAW, K_CT)
        grav = T.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        blin = T.w2b(d.qpos[3:7], d.qvel[0:3]); gyro = d.qvel[3:6]
        fxs = np.array([d.geom_xpos[g][0] for g in foot_gid])
        fys = np.array([d.geom_xpos[g][1] for g in foot_gid])
        fzs = np.array([d.geom_xpos[g][2] for g in foot_gid])
        contact = ((fzs - gz(fxs, fys)) < 0.03).astype(np.float32)
        obs = O.build_obs(jnp.asarray(grav), jnp.asarray(blin), jnp.asarray(gyro),
                          jnp.asarray(d.qpos[7:19] - HOME12), jnp.asarray(d.qvel[6:18]),
                          jnp.asarray(cmd), jnp.asarray(last), jnp.asarray(contact), c)
        act = infer(np.asarray(obs, np.float32))
        mux, muy, om, gc = C.action_to_cpg_cmd(jnp.asarray(act), mode)
        gc = np.array(gc)
        if gc_cap is not None:
            gc = np.minimum(gc, gc_cap)
        gcm = float(np.mean(gc)); gclog.append(gcm)
        c = C.cpg_step(c, mux, muy, om, T.CTRL_DT)
        d.ctrl[:] = np.clip(np.array(C.cpg_to_joint_targets(c, jnp.asarray(jinvs), jnp.asarray(gc))), lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)
        last = act
        cur_xy = np.array([[d.geom_xpos[g][0], d.geom_xpos[g][1]] for g in foot_gid])
        slip_vel = np.linalg.norm(cur_xy - prev_xy, axis=1) / T.CTRL_DT
        slip = float(np.sum(contact * np.clip(slip_vel, 0.0, 1.0)))
        prev_xy = cur_xy
        if X15_LO <= x < X15_HI:
            slip15.append(slip); gc15.append(gcm)
            if t0_15 is None: t0_15 = i * T.CTRL_DT
            t1_15 = i * T.CTRL_DT
        grav2 = T.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        if fell is None and (float(d.qpos[2]) - float(gz(x)) < 0.15 or grav2[2] > -0.4):
            fell = i * T.CTRL_DT
    xend = float(d.qpos[0])
    return dict(xend=round(xend, 1), gc=round(float(np.mean(gclog)), 3),
                slip15=round(float(np.mean(slip15)), 3) if slip15 else None,
                gc15=round(float(np.mean(gc15)), 3) if gc15 else None,
                t_in15=round((t1_15 - t0_15), 1) if t0_15 is not None else None,
                reached15=X15_LO <= xend, fell="是@%.0fs" % fell if fell else "否")


CONFIGS = [
    ("v2.0 (低抬腳基準)",     W["v2.0"],       cpg2, None),
    ("terrain2_1",            W["terrain2_1"], cpg2, None),
    ("terrain3 (原抬腳)",     W["terrain3"],   cpg3, None),
    ("terrain3 (gc夾0.08)",   W["terrain3"],   cpg3, 0.08),
    ("terrain3 (gc夾0.10)",   W["terrain3"],   cpg3, 0.10),
    ("terrain3b (重訓修正)",  W["terrain3b"],  cpg3, None),   # 重訓後才有
]
print(f"{'配置':<22} {'走到x':>6} {'gc全程':>6} {'gc@15':>6} {'slip@15':>8} {'在15°耗時':>9} {'跌'}")
for name, w, C, cap in CONFIGS:
    if not os.path.exists(w):
        print(f"{name:<22} (權重不存在，跳過)"); continue
    r = run(w, C, gc_cap=cap)
    print(f"{name:<22} {r['xend']:>6} {r['gc']:>6} {str(r['gc15']):>6} "
          f"{str(r['slip15']):>8} {str(r['t_in15']):>9} {r['fell']}")
