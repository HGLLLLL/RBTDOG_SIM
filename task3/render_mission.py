"""俯視時間縮時渲染：機器狗走正方形，地上留軌跡拖尾。V1 無IMU 與 V2 羅盤 並排輸出 mp4。"""
import numpy as np, sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
import mujoco
from go2_gait import Go2Gait
from walk_line import GAIT, COMPASS_GAIN
from square_mission import (NOMINAL_SPEED, NOMINAL_YAWRATE, SIDE_LEN, MAG_NOISE, wrap)
from PIL import Image, ImageDraw

LAPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_SIDES = LAPS * 4
RENDER_EVERY = 350         # 每 N 模擬步存一幀（時間縮時）
W, H = 480, 480
TRAIL_MAX = 800


def add_trail(scn, pts, rgba):
    for px, py in pts:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.09, 0, 0]), np.array([px, py, 0.03]),
                            np.eye(3).flatten(), np.array(rgba, np.float32))
        scn.ngeom += 1


def run_render(use_compass, rgba):
    g = Go2Gait(**GAIT)
    dt = g.m.opt.timestep
    r = mujoco.Renderer(g.m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [5, -5, 0]; cam.distance = 17; cam.elevation = -89; cam.azimuth = 90
    rng = np.random.default_rng(0)

    def compass():
        return g.compass_yaw()      # 誤差已由 MTi-680G 融合航向模型內建於 compass_yaw()

    for i in range(int(0.5 / dt)):
        g.step(0, 0, 0)
    t = 0.5; istep = 0; target = 0.0
    trail = []; frames = []

    def maybe_frame():
        if istep % RENDER_EVERY == 0:
            trail.append(g.xy.copy())
            r.update_scene(g.d, cam)
            add_trail(r.scene, trail[-TRAIL_MAX:], rgba)
            frames.append(r.render())

    for side in range(N_SIDES):
        for k in range(int((SIDE_LEN / NOMINAL_SPEED) / dt)):
            turn = np.clip(COMPASS_GAIN * wrap(compass() - target), -1, 1) if use_compass else 0.0
            g.step(t, 1.0, turn); t += dt; istep += 1; maybe_frame()
        target = wrap(target - np.pi / 2)
        if use_compass:
            settle = 0; k = 0
            while True:
                err = wrap(compass() - target)
                g.step(t, 0.0, np.clip(3.0 * err, -1, 1)); t += dt; istep += 1; k += 1; maybe_frame()
                settle = settle + 1 if abs(err) < np.radians(3.0) else 0
                if settle >= int(0.1 / dt) or k >= int(6.0 / dt):
                    break
        else:
            for k in range(int(((np.pi / 2) / NOMINAL_YAWRATE) / dt)):
                g.step(t, 0.0, 1.0); t += dt; istep += 1; maybe_frame()
        print(f"  {'V2' if use_compass else 'V1'} 邊 {side+1}/{N_SIDES}  幀數={len(frames)}", flush=True)
    return frames


def label(img_arr, text, color):
    im = Image.fromarray(img_arr)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 26], fill=(255, 255, 255))
    d.text((8, 6), text, fill=color)
    return np.asarray(im)


def main():
    print("渲染 V1 (無IMU)..."); f1 = run_render(False, [0.9, 0.1, 0.1, 1])
    print("渲染 V2 (羅盤)...");  f2 = run_render(True,  [0.1, 0.3, 0.9, 1])
    n = max(len(f1), len(f2))
    f1 += [f1[-1]] * (n - len(f1)); f2 += [f2[-1]] * (n - len(f2))
    import imageio.v2 as iio
    out = "/home/huang/rbtdog_sim/task3/square_video.mp4"
    with iio.get_writer(out, fps=25, codec="libx264", quality=7) as wtr:
        for a, b in zip(f1, f2):
            a = label(a, "V1: no IMU (open-loop, drifts)", (200, 0, 0))
            b = label(b, "V2: IMU compass (corrected)", (0, 0, 200))
            wtr.append_data(np.concatenate([a, b], axis=1))
    print("影片已存:", out, " 幀數:", n)


if __name__ == "__main__":
    main()
