"""舊基準 vs 新的 `kd_wheel=3.0 / x_off=−110mm`，並排對照影片。

兩台同時跑、鏡頭各自跟拍、橫向拼接。
**看點是前腳**：舊的是抬起來原地放下再被機身拖走（執行率 0.03），
新的是真的在空中往前送（0.79）。

用法：
    /home/huang/miniforge3/envs/rbtdog/bin/python \\
        task7/inference/diag/wheel_kd_video.py [--secs 20]
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, 'task7/inference')
import mujoco
from PIL import Image, ImageDraw

import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm

W, H = 760, 460


class Run:
    def __init__(self, label, sub, x_off, kd_wheel):
        self.label, self.sub = label, sub
        self.r = cw.Robot(kd_wheel=kd_wheel)
        self.cfg = dict(gb.BASELINE, x_off=x_off)
        self.z_sag = mm.STATIC_SAG
        self.ks = leg_kin.knee_sign_of(mm.HOME)
        self.f0 = leg_kin.home_foot(mm.HOME)
        self.step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
        q0 = cpg_max.stand_targets(self.ks, self.f0, x_off)
        self.r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
        for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
            self.r.step(q0)
            if i == int(0.5 / mm.CTRL_DT):
                self.r.lock_wheels()
        self.c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
        self.r.m.vis.global_.offwidth, self.r.m.vis.global_.offheight = W, H
        self.ren = mujoco.Renderer(self.r.m, H, W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.r.m, self.cam)
        self.x0 = float(self.r.d.qpos[0])

    def advance(self):
        g = self.cfg
        self.c = self.step(self.c, np.full(4, g["mu_x"]), np.full(4, g["mu_y"]),
                           np.full(4, g["omega"]), mm.CTRL_DT)
        q, _ = cpg_max.joint_targets(self.c, self.f0, g["x_off"], g["g_c"],
                                     g["d_step"], g["d_step_y"], g["duty"],
                                     self.ks, self.z_sag)
        self.r.step(q)

    def frame(self):
        d = self.r.d
        self.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.30]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.1, -8, 90
        self.ren.update_scene(d, self.cam)
        img = Image.fromarray(self.ren.render())
        dr = ImageDraw.Draw(img)
        adv = (float(d.qpos[0]) - self.x0) * 1000
        dr.rectangle([0, 0, W, 58], fill=(0, 0, 0))
        dr.text((12, 6), self.label, fill=(255, 255, 255))
        dr.text((12, 22), self.sub, fill=(150, 150, 150))
        dr.text((12, 40), f"forward {adv:+.0f} mm", fill=(180, 220, 255))
        return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--out", default="task7/outputs/cpg_wheelkd_vs_baseline.mp4")
    a = ap.parse_args()

    old = Run("OLD   wheel kd 0.5   x_off -40 mm",
              "front-foot exec rate 0.03  (dragged along)", -0.040, 0.5)
    new = Run("NEW   wheel kd 3.0   x_off -110 mm",
              "front-foot exec rate 0.79  (actually stepping)", -0.110, 3.0)

    frames = []
    for i in range(int(a.secs / mm.CTRL_DT)):
        old.advance()
        new.advance()
        if i % 2 == 0:
            frames.append(np.concatenate([old.frame(), new.frame()], axis=1))

    import imageio.v2 as iio
    iio.mimsave(a.out, frames, fps=25, codec="libx264")
    do = (float(old.r.d.qpos[0]) - old.x0) * 1000
    dn = (float(new.r.d.qpos[0]) - new.x0) * 1000
    print(f"[影片] {a.out}　{len(frames)} 幀")
    print(f"舊 前進 {do:.0f} mm ／ 新 前進 {dn:.0f} mm　（{dn / do:.2f}×）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
