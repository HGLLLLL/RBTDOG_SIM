"""CPG-RL 地形 v2 綜合測試影片產生器。

產出兩支影片到指定輸出夾：
1. comprehensive_new.mp4 — 新模型(16維可學抬腳)在 rough2(斜坡+8cm漸變凹凸)上的綜合指令測試：
   直走上坡 / 直走下坡 / 左轉 / 右轉 / 左橫移 / 右橫移（每段獨立 spawn，帶字幕與即時量測）。
2. compare_old_vs_new_rough.mp4 — 同地形直走上坡，舊模型(抬腳寫死12維) vs 新模型(可學抬腳16維) 並排。

run: conda run -n rbtdog python task4/inference/terrain2_demo.py --outdir task4/outputs/<date>
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, argparse
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import numpy as np
import mujoco
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as iio

import cpg2 as C
import obs2 as O
import local_infer_terrain2 as L

SCENE = L.SCENE
CTRL_DT, SIM_DT = L.CTRL_DT, L.SIM_DT
HOME12 = np.array(L.HOME12)
WIDTH, HEIGHT = 640, 480
_FONT_PATH = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
FONT = ImageFont.truetype(_FONT_PATH, 22)
FONT_S = ImageFont.truetype(_FONT_PATH, 16)

NEW = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain2_params.pkl"   # 16 維 learnable
OLD = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl"      # 12 維 fixed


def _facing_quat(face):
    # +x = 預設朝向；-x = 繞 z 轉 180°（走下坡）
    return [1.0, 0, 0, 0] if face == "+x" else [0.0, 0, 0, 1.0]


def _caption(frame, title, lines):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, WIDTH, 34], fill=(0, 0, 0))
    d.text((8, 5), title, font=FONT, fill=(255, 255, 255))
    y = HEIGHT - 6 - 18 * len(lines)
    for ln in lines:
        d.rectangle([0, y - 2, 320, y + 18], fill=(0, 0, 0))
        d.text((8, y), ln, font=FONT_S, fill=(170, 255, 170))
        y += 18
    return np.array(im)


def run_clip(infer, act_dim, mode, face, cmd, secs, title, terrain="rough2", settle=0.4):
    m = L._make_model(terrain)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = _facing_quat(face)
    mujoco.mj_forward(m, d)
    lo = m.actuator_ctrlrange[:, 0]; hi = m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS]
    jinvs = C.leg_ik_consts(SCENE)
    n_sub = int(round(CTRL_DT / SIM_DT))
    ren = mujoco.Renderer(m, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)

    def apply(q):
        d.ctrl[:] = np.clip(q, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

    for _ in range(int(settle / CTRL_DT)):
        apply(HOME12)

    c = C.cpg_init(); last_a = np.zeros(act_dim); cmdv = np.asarray(cmd, np.float32)
    frames = []; x0, y0 = float(d.qpos[0]), float(d.qpos[1]); gcs = []; fell = None
    for i in range(int(secs / CTRL_DT)):
        grav = L.w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        blin = L.w2b(d.qpos[3:7], d.qvel[0:3])
        fx = np.array([d.geom_xpos[g][0] for g in foot_gid])
        fy = np.array([d.geom_xpos[g][1] for g in foot_gid])
        fz = np.array([d.geom_xpos[g][2] for g in foot_gid])
        gzf = np.array([L._gz(terrain, fx[k], fy[k]) for k in range(4)])
        contact = ((fz - gzf) < 0.03).astype(np.float32)
        o = O.build_obs(jnp.asarray(grav), jnp.asarray(blin), jnp.asarray(d.qvel[3:6]),
                        jnp.asarray(d.qpos[7:19] - HOME12), jnp.asarray(d.qvel[6:18]),
                        jnp.asarray(cmdv), jnp.asarray(last_a), jnp.asarray(contact), c)
        act = np.array(infer(np.asarray(o, np.float32)))
        mux, muy, om, gc = C.action_to_cpg_cmd(jnp.asarray(act), mode)
        gcs.append(float(np.mean(np.array(gc))))
        c = C.cpg_step(c, mux, muy, om, CTRL_DT)
        q = np.array(C.cpg_to_joint_targets(c, jnp.asarray(jinvs), gc))
        apply(q); last_a = act
        if grav[2] > -0.4 and fell is None:
            fell = round(i * CTRL_DT, 1)
        if i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.25]
            cam.distance = 2.6; cam.elevation = -18; cam.azimuth = 90
            ren.update_scene(d, cam)
            lines = [f"cmd  vx={cmdv[0]:.2f}  vy={cmdv[1]:+.2f}  wz={cmdv[2]:+.2f}",
                     f"body vx={blin[0]:+.2f} m/s   wz={d.qvel[5]:+.2f}"]
            if mode == "learnable":
                lines.append(f"gc(抬腳)={np.mean(np.array(gc)):.3f} m")
            frames.append(_caption(ren.render(), title, lines))
    dist = float(np.hypot(d.qpos[0] - x0, d.qpos[1] - y0))
    summ = dict(title=title, dist=round(dist, 2), fell=fell,
                gc_mean=round(float(np.mean(gcs)), 3), end_h=round(float(d.qpos[2]), 2))
    return frames, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="task4/outputs/2026-07-16")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[info] 載入新模型(16維)…")
    new_infer, nad = L.load_policy_any(NEW); assert nad == 16, nad
    print("[info] 載入舊模型(12維)…")
    old_infer, oad = L.load_policy_any(OLD); assert oad == 12, oad

    # ---- 1) 綜合測試（新模型）----
    segs = [
        ("+x", (0.6, 0.0, 0.0), "直走・上坡 (rough+slope)", 3.5),
        ("-x", (0.6, 0.0, 0.0), "直走・下坡 (rough+slope)", 3.5),
        ("+x", (0.35, 0.0, 0.8), "左轉行進 wz=+0.8", 3.0),
        ("+x", (0.35, 0.0, -0.8), "右轉行進 wz=-0.8", 3.0),
        ("+x", (0.30, 0.28, 0.0), "左橫移 vy=+0.28", 3.0),
        ("+x", (0.30, -0.28, 0.0), "右橫移 vy=-0.28", 3.0),
    ]
    all_frames = []; summ_rows = []
    for face, cmd, title, secs in segs:
        print(f"[綜合] {title} …")
        fr, s = run_clip(new_infer, 16, "learnable", face, cmd, secs, title)
        all_frames += fr; summ_rows.append(s)
    out1 = os.path.join(args.outdir, "comprehensive_new.mp4")
    iio.mimsave(out1, all_frames, fps=25, codec="libx264")
    print("[ok] 影片:", out1, f"({len(all_frames)} 幀)")

    # ---- 2) 舊 vs 新 並排（同地形直走上坡）----
    print("[對比] 舊模型直走…")
    fo, so = run_clip(old_infer, 12, "fixed", "+x", (0.6, 0, 0), 6.0, "舊：抬腳寫死(12維)")
    print("[對比] 新模型直走…")
    fn, sn = run_clip(new_infer, 16, "learnable", "+x", (0.6, 0, 0), 6.0, "新：可學抬腳(16維)")
    k = min(len(fo), len(fn))
    comp = [np.hstack([fo[i], fn[i]]) for i in range(k)]
    out2 = os.path.join(args.outdir, "compare_old_vs_new_rough.mp4")
    iio.mimsave(out2, comp, fps=25, codec="libx264")
    print("[ok] 影片:", out2, f"({len(comp)} 幀)")

    # ---- 摘要表 ----
    print("\n=== 綜合測試摘要（新模型 16維）===")
    print(f"{'段落':<26}{'位移m':>7}{'跌倒':>8}{'gc抬腳':>8}{'末端高':>8}")
    for s in summ_rows:
        print(f"{s['title']:<26}{s['dist']:>7}{str(s['fell']):>8}{s['gc_mean']:>8}{s['end_h']:>8}")
    print("\n=== 舊 vs 新（直走上坡 rough2, 6s）===")
    for tag, s in [("舊12維", so), ("新16維", sn)]:
        print(f"{tag}: 位移={s['dist']}m 跌倒={s['fell']} gc={s['gc_mean']} 末端高={s['end_h']}")


if __name__ == "__main__":
    main()
