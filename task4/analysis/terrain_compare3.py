"""地形實驗 v3：新 16 維(全向)模型 + task5 odom 外圈閉環，重作 slope/rough 兩實驗。

與 terrain_compare2（開環直走）唯一差別：每步的指令改由 odom 線追蹤律 `line_control` 產生
（wz 用航向誤差鎖航向、vy 用 cross-track 誤差滑回線上），目標線 = x 軸(y=0, 朝 +x)。
「odom」在 sim 即完美 framepos = 機身真值位姿(d.qpos)。因新模型是全向訓練，吃得下 wz/vy 修正
（舊 v1 給 wz/vy 會崩，故只有新模型能走這條路）。

課程/圖表/鏡頭沿用 terrain_compare.py；輸出到 task4/outputs/2026-07-16/，檔名帶 _odom 以與開環對照。

用法：
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare3.py --check
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare3.py --exp both
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import jax.numpy as jnp
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import terrain_compare as TC
import local_infer_terrain as T           # apply_pd / leg_ik_consts / w2b
import local_infer_paper as P             # line_control(odom 外圈控制律)
import cpg2 as C
import obs2 as O
import local_infer_terrain2 as L2

OUT2 = "/home/huang/rbtdog_sim/task4/outputs/2026-07-16"
NEW = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain2_params.pkl"
FOOT_CONTACT_H = 0.03
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
P0 = np.array([0.0, 0.0])                  # 目標線錨點
PSI = 0.0                                  # 目標線朝向(+x)
K_YAW, K_CT = 3.0, 1.5                     # 與 odom_missions 一致
TC.LABELS["odom"] = "TERRAIN2 + ODOM 外圈 (16D)"
TC.PLTCOL["odom"] = "#2ca02c"


def _annotate_odom(frame, ph, fell, x, e_ct):
    im = TC._annotate(frame, "odom", ph, fell, x)
    img = Image.fromarray(im); dr = ImageDraw.Draw(img)
    dr.rectangle([0, TC.VH - 30, 300, TC.VH], fill=(0, 0, 0))
    dr.text((14, TC.VH - 27), f"cross-track e = {e_ct:+.2f} m", fill=(120, 200, 255), font=TC.font(20))
    return np.asarray(img)


def run_experiment_odom(build_course, cam_kind, secs, out_prefix, video=True):
    m, gz, phase, extra = build_course()
    m.opt.timestep = T.SIM_DT; m = T.apply_pd(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS]
    jinvs = jnp.asarray(T.leg_ik_consts(m))
    nsub = int(round(T.CTRL_DT / T.SIM_DT))
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    infer, act_dim = L2.load_policy_any(NEW); assert act_dim == 16, act_dim
    c = C.cpg_init(); last = np.zeros(16)
    for _ in range(int(0.4 / T.CTRL_DT)):
        d.ctrl[:] = np.clip(HOME12, lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)

    ren = mujoco.Renderer(m, TC.VH, TC.VW) if video else None
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    import imageio.v2 as iio
    writer = iio.get_writer(f"{OUT2}/{out_prefix}.mp4", fps=TC.FPS, codec="libx264") if video else None
    traj = []; fell = None; gclog = []; ectlog = []; n = int(secs / T.CTRL_DT)
    for i in range(n):
        x, y, yaw = float(d.qpos[0]), float(d.qpos[1]), TC.yaw_from_quat(d.qpos[3:7])
        # === odom 外圈：由真值位姿(=完美 framepos odom)算修正指令 ===
        cmd, e_ct, e_yaw = P.line_control(np.array([x, y]), yaw, P0, PSI, TC.VX, K_YAW, K_CT)
        ectlog.append(abs(e_ct))
        grav = T.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        blin = T.w2b(d.qpos[3:7], d.qvel[0:3]); gyro = d.qvel[3:6]
        fxs = np.array([d.geom_xpos[g][0] for g in foot_gid])
        fys = np.array([d.geom_xpos[g][1] for g in foot_gid])
        fzs = np.array([d.geom_xpos[g][2] for g in foot_gid])
        contact = ((fzs - gz(fxs, fys)) < FOOT_CONTACT_H).astype(np.float32)
        obs = O.build_obs(jnp.asarray(grav), jnp.asarray(blin), jnp.asarray(gyro),
                          jnp.asarray(d.qpos[7:19] - HOME12), jnp.asarray(d.qvel[6:18]),
                          jnp.asarray(cmd), jnp.asarray(last), jnp.asarray(contact), c)
        act = infer(np.asarray(obs, np.float32))
        mux, muy, om, gc = C.action_to_cpg_cmd(jnp.asarray(act), "learnable")
        gclog.append(float(np.mean(np.array(gc))))
        c = C.cpg_step(c, mux, muy, om, T.CTRL_DT)
        d.ctrl[:] = np.clip(np.array(C.cpg_to_joint_targets(c, jinvs, gc)), lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)
        last = act
        grav2 = T.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        gzx = float(gz(np.array([x]), np.array([y]))[0]) if cam_kind == "rear45" else float(gz(x))
        rel_h = float(d.qpos[2]) - gzx
        if fell is None and (rel_h < 0.15 or grav2[2] > -0.4): fell = i * T.CTRL_DT
        traj.append((i * T.CTRL_DT, x, y, float(d.qpos[2]), gzx))
        if video and i % TC.RENDER_EVERY == 0:
            if cam_kind == "side":
                cam.lookat[:] = [x, y, gzx + 0.15]; cam.distance = 3.6
                cam.elevation = -8; cam.azimuth = 90
            else:
                cam.lookat[:] = [x, y, 0.3]; cam.distance = 3.2
                cam.elevation = -45; cam.azimuth = float(np.degrees(yaw))
            ren.update_scene(d, cam)
            writer.append_data(_annotate_odom(ren.render(), phase(x), fell, x, e_ct))
    if writer: writer.close(); print("[video]", f"{OUT2}/{out_prefix}.mp4")
    t = np.array(traj)
    print(f"  [odom] 前進到 x={t[:,1].max():.1f}m 最大側偏={np.abs(t[:,2]).max():.2f}m "
          f"平均|e_ct|={np.mean(ectlog):.3f}m 跌倒={'是@%.1fs' % fell if fell else '否'} "
          f"gc平均={np.mean(gclog):.3f}")
    return {"odom": t}, {"odom": fell}, extra


def check():
    print("=== 3s smoke（新16維 + odom 外圈，不出影片）===")
    for name, builder, cam in [("slope", TC.build_course_slopes, "side"),
                               ("rough", TC.build_course_rough, "rear45")]:
        trajs, fell, _ = run_experiment_odom(builder, cam, 3.0, "smoke", video=False)
        ok = all(np.all(np.isfinite(v)) for v in trajs.values())
        print(f"  [{name}] 有限={ok}")
    print("CHECK DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["slope", "rough", "both"], default="both")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT2, exist_ok=True)
    if args.check: check(); return
    if args.exp in ("slope", "both"):
        print("=== 實驗1：平地→斜坡（新16維 + odom 外圈）===")
        tr, fe, ex = run_experiment_odom(TC.build_course_slopes, "side", 45.0, "exp1_slope_odom")
        TC.chart_slope(tr, fe, ex, f"{OUT2}/exp1_slope_odom_chart.png")
    if args.exp in ("rough", "both"):
        print("=== 實驗2：平地→崎嶇（新16維 + odom 外圈）===")
        tr, fe, ex = run_experiment_odom(TC.build_course_rough, "rear45", 35.0, "exp2_rough_odom")
        TC.chart_rough(tr, fe, f"{OUT2}/exp2_rough_odom_chart.png")


if __name__ == "__main__":
    main()
