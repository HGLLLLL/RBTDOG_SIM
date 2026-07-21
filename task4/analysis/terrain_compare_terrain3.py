"""terrain3 vs terrain2_1 並排對照（同 2026-07-16-final 的爬坡/凹凸課程 + odom 閉環）。

左：terrain3（新，斜坡30°/障礙12cm+打滑修正訓練，配 cpg3 GC_MAX=0.25）
右：terrain2_1（舊，抬腳強化，配 cpg2 GC_MAX=0.15）
兩者皆 odom 外圈閉環貼線直走；左右各自獨立 sim、鏡頭各跟各的機器狗，逐幀 hstack 併排。

⚠️ 每個模型必須配「訓練時的 cpg」：terrain3 用 cpg3（抬腳映射到 [0.05,0.25]）、
   terrain2_1 用 cpg2（[0.05,0.15]）。用錯 cpg 會把新模型的抬腳幅度錯誤壓縮。

課程沿用 terrain_compare.py（同 07-15/07-16）。輸出 task4/outputs/2026-07-21-terrain3-vs-terrain2_1/。

用法：
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_terrain3.py --check
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_terrain3.py --exp both
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/analysis")
import terrain_compare as TC
import local_infer_terrain as T
import local_infer_paper as P
import cpg2                     # terrain2_1 用（GC_MAX=0.15）
import cpg3                     # terrain3 用（GC_MAX=0.25）
import obs2 as O                # obs3 與 obs2 內容相同，共用
import local_infer_terrain2 as L2

OUT = "/home/huang/rbtdog_sim/task4/outputs/2026-07-21-terrain3-vs-terrain2_1"
W_NEW = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain3_params.pkl"
W_OLD = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain2_1_params.pkl"
VW, VH = TC.VW, TC.VH
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
FOOT_CONTACT_H = 0.03
K_YAW, K_CT = 3.0, 1.5
_FP = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
F_TITLE = ImageFont.truetype(_FP, 22)
F_SUB = ImageFont.truetype(_FP, 17)
_PHCOL = {"flat": (215, 215, 220), "up": (95, 225, 130), "down": (245, 165, 70), "rough": (240, 210, 130)}


def annotate(frame, title, ph, x, fell, lines):
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    lab, kind = ph
    dr.rectangle([0, 0, VW, 62], fill=(0, 0, 0))
    dr.text((10, 4), title, fill=(255, 255, 255), font=F_TITLE)
    dr.text((10, 33), lab, fill=_PHCOL.get(kind, (220, 220, 220)), font=F_SUB)
    dr.text((VW - 150, 6), f"x={x:+.1f}m", fill=(220, 220, 220), font=F_SUB)
    if fell is not None:
        dr.text((VW - 150, 32), "FALLEN", fill=(255, 90, 90), font=F_SUB)
    y = VH - 6 - 22 * len(lines)
    for ln in lines:
        dr.rectangle([0, y - 2, 250, y + 20], fill=(0, 0, 0))
        dr.text((10, y), ln, fill=(150, 210, 255), font=F_SUB); y += 22
    return np.array(im)


def render_run(params, C, builder, cam_kind, secs, title):
    """單一模型在課程上跑 odom 閉環直走。C = 該模型對應的 cpg 模組（cpg2 或 cpg3）。"""
    m, gz, phase, extra = builder()
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

    ren = mujoco.Renderer(m, VH, VW)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    frames = []; fell = None; x0 = float(d.qpos[0]); gclog = []; latlog = []
    for i in range(int(secs / T.CTRL_DT)):
        x, y, yaw = float(d.qpos[0]), float(d.qpos[1]), TC.yaw_from_quat(d.qpos[3:7])
        cmd, e_ct, _ = P.line_control(np.array([x, y]), yaw, np.array([0.0, 0.0]), 0.0, TC.VX, K_YAW, K_CT)
        latlog.append(abs(e_ct))
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
        mux, muy, om, gc = C.action_to_cpg_cmd(jnp.asarray(act), mode)
        gcm = float(np.mean(np.array(gc))); gclog.append(gcm)
        c = C.cpg_step(c, mux, muy, om, T.CTRL_DT)
        d.ctrl[:] = np.clip(np.array(C.cpg_to_joint_targets(c, jinvs, gc)), lo, hi)
        for _ in range(nsub): mujoco.mj_step(m, d)
        last = act
        grav2 = T.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        gzx = float(gz(np.array([x]), np.array([y]))[0]) if cam_kind == "rear45" else float(gz(x))
        if fell is None and (float(d.qpos[2]) - gzx < 0.15 or grav2[2] > -0.4): fell = i * T.CTRL_DT
        if i % TC.RENDER_EVERY == 0:
            if cam_kind == "side":
                cam.lookat[:] = [x, y, gzx + 0.15]; cam.distance = 3.6; cam.elevation = -8; cam.azimuth = 90
            else:
                cam.lookat[:] = [x, y, 0.3]; cam.distance = 3.2; cam.elevation = -45; cam.azimuth = float(np.degrees(yaw))
            ren.update_scene(d, cam)
            lines = [f"cross-track e={e_ct:+.2f}m", f"抬腳 gc={gcm:.3f}m"]
            frames.append(annotate(ren.render(), title, phase(x), x, fell, lines))
    summ = dict(title=title, dist=float(d.qpos[0]) - x0, maxlat=float(max(latlog)),
                meanlat=float(np.mean(latlog)), fell=fell, gc=float(np.mean(gclog)))
    print(f"  [{title}] 前進={summ['dist']:+.1f}m 最大側偏={summ['maxlat']:.2f}m "
          f"平均側偏={summ['meanlat']:.3f}m gc={summ['gc']:.3f} 跌={'是@%.0fs' % fell if fell else '否'}")
    return frames, summ


def _combine(fl, fr, out):
    import imageio.v2 as iio
    k = min(len(fl), len(fr))
    frames = [np.hstack([fl[i], fr[i]]) for i in range(k)]
    iio.mimsave(out, frames, fps=TC.FPS, codec="libx264")
    print("[video]", out, f"({len(frames)} 幀)")


def exp1(secs=45.0):
    print("=== 實驗1 爬坡：terrain3(新) ‖ terrain2_1(舊)，皆 odom ===")
    fl, sl = render_run(W_NEW, cpg3, TC.build_course_slopes, "side", secs, "terrain3 (new)")
    fr, sr = render_run(W_OLD, cpg2, TC.build_course_slopes, "side", secs, "terrain2_1 (old)")
    _combine(fl, fr, f"{OUT}/exp1_slope_compare.mp4"); return sl, sr


def exp2(secs=35.0):
    print("=== 實驗2 凹凸：terrain3(新) ‖ terrain2_1(舊)，皆 odom ===")
    fl, sl = render_run(W_NEW, cpg3, TC.build_course_rough, "rear45", secs, "terrain3 (new)")
    fr, sr = render_run(W_OLD, cpg2, TC.build_course_rough, "rear45", secs, "terrain2_1 (old)")
    _combine(fl, fr, f"{OUT}/exp2_rough_compare.mp4"); return sl, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["1", "2", "both"], default="both")
    ap.add_argument("--check", action="store_true", help="3s smoke，不出影片")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check:
        print("=== 3s smoke（四組配置）===")
        render_run(W_NEW, cpg3, TC.build_course_slopes, "side", 3.0, "terrain3 slope")
        render_run(W_OLD, cpg2, TC.build_course_slopes, "side", 3.0, "terrain2_1 slope")
        render_run(W_NEW, cpg3, TC.build_course_rough, "rear45", 3.0, "terrain3 rough")
        render_run(W_OLD, cpg2, TC.build_course_rough, "rear45", 3.0, "terrain2_1 rough")
        print("CHECK DONE"); return
    if args.exp in ("1", "both"): exp1()
    if args.exp in ("2", "both"): exp2()


if __name__ == "__main__":
    main()
