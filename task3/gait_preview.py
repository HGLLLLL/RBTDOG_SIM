"""走路姿態預覽：側視(看步幅/俯仰/垂直彈跳) + 後視(看側傾搖晃) 並排短片。
只走直線、無干擾、無校正，純粹展示步態穩定度。real-time 播放。
用法: MUJOCO_GL=egl python gait_preview.py   ->  gait_preview.mp4
"""
import numpy as np, sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
import mujoco
from go2_gait import Go2Gait
from walk_line import GAIT
from PIL import Image, ImageDraw

FPS = 30
W = H = 480
T_SHOW = 8.0          # 展示秒數（暖機另計）


def rpy(g):
    w, x, y, z = g.sensor("imu_quat")
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return roll, pitch


def make_cam(az, el, dist):
    c = mujoco.MjvCamera(); c.azimuth = az; c.elevation = el; c.distance = dist
    return c


def view(g, rend, cam, label, color, extra):
    x, y = g.xy
    cam.lookat[:] = [x, y, 0.22]
    rend.update_scene(g.d, cam)
    im = Image.fromarray(rend.render()); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 22], fill=(255, 255, 255))
    dr.text((8, 5), label, fill=color)
    dr.text((8, H - 16), extra, fill=(20, 20, 20))
    return np.asarray(im)


def main():
    g = Go2Gait(**GAIT)
    dt = g.m.opt.timestep
    side = mujoco.Renderer(g.m, height=H, width=W)   # 側視相機
    rear = mujoco.Renderer(g.m, height=H, width=W)   # 後視相機
    cam_s = make_cam(90.0, -12.0, 2.2)               # 從側邊看步幅/俯仰
    cam_r = make_cam(0.0, -18.0, 1.9)                # 從正後方看側傾/擺動
    for _ in range(int(1.0 / dt)):                   # 暖機站穩
        g.step(0, 0, 0)
    import imageio.v2 as iio
    writer = iio.get_writer("/home/huang/rbtdog_sim/task3/gait_preview.mp4",
                            fps=FPS, codec="libx264", quality=8)
    frame_iv = int((1.0 / FPS) / dt)
    t, k = 1.0, 0
    rolls, pitches, zs = [], [], []
    while t < 1.0 + T_SHOW:
        g.step(t, 1.0, 0.0); t += dt; k += 1
        r, p = rpy(g); rolls.append(r); pitches.append(p); zs.append(g.height)
        if k % frame_iv == 0:
            info = f"v={GAIT['stride']*GAIT['freq']:.2f}m/s  stride={GAIT['stride']}m  {GAIT['freq']}Hz"
            a = view(g, side, cam_s, "SIDE view (stride / pitch)", (0, 0, 180),
                     f"pitch={p:+4.1f}deg  h={g.height*100:4.1f}cm")
            b = view(g, rear, cam_r, "REAR view (roll / sway)", (160, 0, 0),
                     f"roll={r:+4.1f}deg")
            writer.append_data(np.concatenate([a, b], axis=1))
    writer.close()
    print("姿態預覽已存 gait_preview.mp4")
    print(f"步態統計(展示{T_SHOW:.0f}s)：roll SD={np.std(rolls):.2f}deg  "
          f"pitch SD={np.std(pitches):.2f}deg  高度峰峰={1000*(max(zs)-min(zs)):.0f}mm  "
          f"均高={100*np.mean(zs):.1f}cm")


if __name__ == "__main__":
    main()
