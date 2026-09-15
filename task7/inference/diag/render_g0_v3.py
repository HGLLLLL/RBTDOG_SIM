"""v3.1 零動作（純開迴路產生器）各行走模式的影片：每個指令 10 s、跟拍鏡頭、地面固定參考點、疊標題與即時數字。
    MUJOCO_GL=egl conda run -n rbtdog python task7/inference/diag/render_g0_v3.py --out task7/outputs/g0_v31_baseline.mp4
"""
import argparse, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
import jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_env_v3 as v3, max_model as mm

CASES = (("直走 vx 0.5 m/s", (0.5, 0.0, 0.0)), ("邊走邊轉（弧線）vx 0.5 + wz 0.5 rad/s", (0.5, 0.0, 0.5)),
         ("原地左轉 wz 1.3 rad/s", (0.0, 0.0, 1.3)), ("左平移 vy 0.08 m/s", (0.0, 0.08, 0.0)),
         ("斜走 vx 0.3 + vy 0.06 m/s", (0.3, 0.06, 0.0)))
FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc"


def rollout(env, cmd, steps):
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(0)); s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    Q, M = [], []
    tau_max = 0.0
    for i in range(steps):
        s = js(s, jnp.zeros(v3.ACT_DIM))
        tau_max = max(tau_max, float(s.metrics["tau_pk"]))
        Q.append(np.asarray(s.pipeline_state.qpos))
        M.append(dict(vx=float(s.metrics["vx"]), vy=float(s.metrics["vy"]), yaw=np.degrees(float(s.metrics["wz"])),
                      roll=float(s.metrics["roll"]), tau=float(s.metrics["tau_pk"]), tau_max=tau_max, done=float(s.done)))
        if float(s.done) > 0:
            break
    return Q, M


def add_ground_markers(scene, base_xy):
    """在狗附近每 0.5 m 放一顆固定於世界的小球，讓跟拍鏡頭下也看得出位移。"""
    cx, cy = np.round(base_xy / 0.5) * 0.5
    for x in np.arange(cx - 3.0, cx + 3.01, 0.5):
        for y in np.arange(cy - 2.0, cy + 2.01, 0.5):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            big = abs(x) < 1e-6 and abs(y) < 1e-6
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03 if big else 0.015] * 3),
                                np.array([x, y, 0.015]), np.eye(3).flatten(),
                                np.array([0.9, 0.2, 0.2, 1.0] if big else [0.15, 0.15, 0.15, 1.0], dtype=np.float32))
            scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "outputs" / "g0_v31_baseline.mp4"))
    ap.add_argument("--secs", type=float, default=10.0); ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="960x540")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    env = v3.DualModeEnv()
    m = mujoco.MjModel.from_xml_path(v3.SCENE_V3); d = mujoco.MjData(m)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.0, 145.0, -20.0
    base = mm._id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    f_title, f_txt = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 22)
    steps = int(a.secs / v3.CTRL_DT); every = max(1, int(round(1.0 / (a.fps * v3.CTRL_DT))))
    wr = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=None)
    for k, (title, cmd) in enumerate(CASES):
        t0 = time.time(); Q, M = rollout(env, cmd, steps); print(f"[{k+1}/{len(CASES)}] {title}: {len(Q)} 步 ({time.time()-t0:.0f}s)", flush=True)
        for i in range(0, len(Q), every):
            d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
            cam.lookat[:] = [d.xpos[base][0], d.xpos[base][1], 0.35]
            r.update_scene(d, camera=cam); add_ground_markers(r.scene, d.xpos[base][:2])
            img = Image.fromarray(r.render()); dr = ImageDraw.Draw(img, "RGBA")
            dr.rectangle([0, 0, W, 78], fill=(0, 0, 0, 150))
            dr.text((14, 6), f"{k+1}/{len(CASES)}  {title}", font=f_title, fill=(255, 255, 255))
            mi = M[i]
            dr.text((14, 46), f"t {i*v3.CTRL_DT:4.1f} s   vx {mi['vx']:+.2f}  vy {mi['vy']:+.3f} m/s   偏航 {mi['yaw']:+4.0f}°/s   側傾 {mi['roll']:4.1f}°   力矩峰(累計) {mi['tau_max']:3.0f} N·m",
                    font=f_txt, fill=(255, 235, 120))
            dr.rectangle([0, H - 30, W, H], fill=(0, 0, 0, 120))
            dr.text((14, H - 28), "CPG-RL v3.1 零動作基準（未訓練，純開迴路產生器；kp 60/250/250、輪 kd 1.0）  2026-09-15", font=ImageFont.truetype(FONT, 17), fill=(220, 220, 220))
            if mi["done"] > 0:
                dr.text((W // 2 - 60, H // 2), "終止", font=f_title, fill=(255, 80, 80))
            wr.append_data(np.asarray(img))
        if len(Q) < steps:
            print(f"   ⚠️ 在 {len(Q)*v3.CTRL_DT:.1f} s 終止", flush=True)
    wr.close(); print("→", a.out, f"{Path(a.out).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
