"""舊基準 vs kp250 新參數，並排對照影片。

兩台同時跑、鏡頭各自跟拍，橫向拼接。看點是**前腳有沒有在往前跨**。
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, 'task7/inference')
import mujoco
from PIL import Image, ImageDraw
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, max_model as mm

SECS = 20.0
W, H = 760, 460


class Run:
    def __init__(self, label, cfg, kp3=None, kd3=None, z_sag=None):
        self.label = label
        self.r = cw.Robot(kp3=kp3, kd3=kd3)
        self.cfg = cfg
        self.z_sag = mm.STATIC_SAG if z_sag is None else z_sag
        self.ks = leg_kin.knee_sign_of(mm.HOME)
        self.f0 = leg_kin.home_foot(mm.HOME)
        self.step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
        q0 = cpg_max.stand_targets(self.ks, self.f0, cfg["x_off"])
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
        self.c = self.step(self.c, np.full(4, g["mu_x"]), np.full(4, 1.5),
                           np.full(4, g["omega"]), mm.CTRL_DT)
        q, _ = cpg_max.joint_targets(self.c, self.f0, g["x_off"], g["g_c"],
                                     g["d_step"], 0.12, g["duty"], self.ks, self.z_sag)
        self.r.step(q)

    def frame(self):
        d = self.r.d
        self.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.30]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.1, -8, 90
        self.ren.update_scene(d, self.cam)
        img = Image.fromarray(self.ren.render())
        dr = ImageDraw.Draw(img)
        adv = (float(d.qpos[0]) - self.x0) * 1000
        dr.rectangle([0, 0, W, 44], fill=(0, 0, 0))
        dr.text((12, 6), self.label, fill=(255, 255, 255))
        dr.text((12, 24), f"forward {adv:+.0f} mm", fill=(180, 220, 255))
        return np.asarray(img)


old = Run("OLD  kp120  duty0.80  d_step0.10", gb.BASELINE)
new = Run("NEW  kp250  duty0.85  d_step0.12", gb.BASELINE_KP250,
          kp3=gb.BASELINE_KP250["kp3"], kd3=gb.BASELINE_KP250["kd3"],
          z_sag=gb.BASELINE_KP250["z_sag"])

frames = []
n = int(SECS / mm.CTRL_DT)
for i in range(n):
    old.advance()
    new.advance()
    if i % 2 == 0:
        frames.append(np.concatenate([old.frame(), new.frame()], axis=1))

import imageio.v2 as iio
out = "task7/outputs/cpg_kp250_vs_baseline.mp4"
iio.mimsave(out, frames, fps=25, codec="libx264")
print("[影片]", out, len(frames), "frames")
print("舊 前進 %.0f mm ／ 新 前進 %.0f mm"
      % ((float(old.r.d.qpos[0]) - old.x0) * 1000,
         (float(new.r.d.qpos[0]) - new.x0) * 1000))
