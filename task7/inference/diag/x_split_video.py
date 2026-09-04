"""前後分離 `x_off` 的候選步態：側視對照影片。

看點是**後腿的大腿角度**：現行點的後大腿與機身夾 62°、前腿只有 31.5°；
分離之後後腿收到 48.6°（A）／44.2°（B），往前腿靠攏。**前腿幾乎不變**
（31.5° → 30.7°）—— 前腳足端必須往後 90 mm 以上才踩得住，那個姿態換不掉
（見 `docs/D_前後姿態對稱與x_off分離_2026-09-03.md` §4）。

⚠️ 每一路只在標示出來的參數上不同，其餘全部取 `GAITS["walk"]`
（這條線踩過「把換模型檔與換 actuator_mode 綁在同一步」的假結論）。

用法：
    PY=/home/huang/miniforge3/envs/rbtdog/bin/python
    $PY task7/inference/diag/x_split_video.py                 # 三路並排
    $PY task7/inference/diag/x_split_video.py --only A        # 現行 vs A（大畫面）
    $PY task7/inference/diag/x_split_video.py --only B --secs 30
    $PY task7/inference/diag/x_split_video.py --add " 0,-0.12,4,0.08"  # 臨時加一路
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "task7/inference")

import mujoco                                   # noqa: E402
import numpy as np                              # noqa: E402
from PIL import Image, ImageDraw                # noqa: E402

import cpg_max                                  # noqa: E402
import cpg_walk_max as cw                       # noqa: E402
import leg_kin                                  # noqa: E402
import max_model as mm                          # noqa: E402


def thigh_angles(x_c, x_d):
    """站姿的前／後大腿與機身夾角（度）—— 影片標籤上那兩個數字。"""
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    q = cpg_max.stand_targets(ks, f0, cpg_max.x_off_split(x_c, x_d)).reshape(4, 3)
    hf, hr = abs(np.degrees(q[0, 1])), abs(np.degrees(q[2, 1]))
    return 90 - hf, 90 - hr, hf - hr


class Run:
    def __init__(self, tag, x_c, x_d, kd_wheel, g_c, d_step=None,
                 omega=None, duty=None, phase=None, sway=None,
                 kp3=None, kd3=None, z_sag=None):
        f, r, dh = thigh_angles(x_c, x_d)
        off = cpg_max.x_off_split(x_c, x_d)
        self.label = (f"{tag}  x_off F{off[0] * 1000:+.0f}/R{off[2] * 1000:+.0f}  "
                      f"kd{kd_wheel:g} gc{g_c:g}")
        self.label2 = f"thigh  front {f:.1f}  rear {r:.1f}  (dHip {dh:.1f})"
        self.g = dict(cw.GAITS["walk"], g_c=g_c)
        for k, v in (("d_step", d_step), ("omega", omega), ("duty", duty)):
            if v is not None:
                self.g[k] = v
                self.label += f" {k}{v:g}"
        self.x_off = off
        self.z_sag = mm.STATIC_SAG if z_sag is None else z_sag
        if kp3 is not None:
            self.label += f" kp{kp3[1]:g}/abad{kp3[0]:g} zsag{self.z_sag:g}"
        self.r = cw.Robot(kd_wheel=kd_wheel, kp3=kp3, kd3=kd3)
        self.ks = leg_kin.knee_sign_of(mm.HOME)
        self.f0 = leg_kin.home_foot(mm.HOME)
        self.ph = cpg_max.PHASE_WALK if phase is None else phase
        self.label += "  LS" if phase is not None else "  DS"
        self.step = cpg_max.make_cpg_step(self.ph)
        q0 = cpg_max.stand_targets(self.ks, self.f0, self.x_off)
        self.r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
        for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
            self.r.step(q0, "damp")
            if i == int(0.5 / mm.CTRL_DT):
                self.r.lock_wheels()
        self.sway_p = sway            # (sway_x, sway_y, lead_x, lead_y)
        if sway:
            self.label += f" sway({sway[0]*1000:.0f},{sway[1]*1000:.0f})mm"
        self.c = cpg_max.cpg_init(self.ph)
        self.x0 = float(self.r.d.qpos[0])
        self.ren = self.cam = None

    def make_renderer(self, w, h):
        self.r.m.vis.global_.offwidth, self.r.m.vis.global_.offheight = w, h
        self.ren = mujoco.Renderer(self.r.m, h, w)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.r.m, self.cam)
        self.w, self.h = w, h

    def advance(self):
        g = self.g
        self.c = self.step(self.c, np.full(4, g["mu_x"]), np.full(4, 1.5),
                           np.full(4, g["omega"]), mm.CTRL_DT)
        sw = None
        if self.sway_p:
            sw = cpg_max.body_sway(cpg_max.gait_phase(self.c["theta"], self.ph),
                                   *self.sway_p)
        q, _ = cpg_max.joint_targets(self.c, self.f0, self.x_off, g["g_c"],
                                     g["d_step"], 0.12, g["duty"], self.ks,
                                     self.z_sag, sw)
        self.r.step(q, "damp")

    def frame(self):
        d = self.r.d
        self.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.30]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.1, -8, 90
        self.ren.update_scene(d, self.cam)
        img = Image.fromarray(self.ren.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, self.w, 58], fill=(0, 0, 0))
        dr.text((12, 4), self.label, fill=(255, 255, 255))
        dr.text((12, 21), self.label2, fill=(255, 220, 140))
        dr.text((12, 38), f"forward {(float(d.qpos[0]) - self.x0) * 1000:+.0f} mm",
                fill=(180, 220, 255))
        return np.asarray(img)


#    tag,  x_c,    x_d,     kd,  g_c,  d_step, omega, duty, phase
CAND = {
    "NOW":  (-0.110, 0.000, 3.0, 0.08),
    "A":    (-0.075, -0.045, 3.0, 0.08),
    "B":    (-0.060, -0.060, 4.0, 0.07),
    # x_c=0 ＝ 姿態完美對稱，但 DS 序列下偏航 20 秒就 +38°（影片看得到它在轉）
    "SYM":  (0.000, -0.060, 3.0, 0.08, 0.13),
    # ★★ 2026-09-03：改用 **lateral sequence**（見 cpg_max.PHASE_WALK_LS）之後的兩組。
    #   LSSYM 姿態幾乎對稱（Δhip 5.7°）、60 s 側偏 0.42 m、後膝與現行持平。
    #   LSFAST 速度 0.425（+65%）、前後執行率幾乎相等（1.16 / 1.25）。
    "LSSYM":  (-0.020, 0.0, 3.0, 0.07, None, None, None, cpg_max.PHASE_WALK_LS),
    "LSFAST": (-0.110, 0.0, 3.0, 0.08, None, None, None, cpg_max.PHASE_WALK_LS),
    # ★★ 路線 A（G 文件）：kp250 步態。z_sag 用實機錨點 0.036（kp250）——
    #   影片代表實機組態。對照請用 KP120（同樣用實機 z_sag 0.075）。
    "KP250": (-0.030, 0.0, 0.5, 0.048, None, None, None, cpg_max.PHASE_WALK_LS,
              None, [60., 250., 250.], [2., 2., 2.], 0.036),
    "KP120": (-0.030, 0.0, 0.5, 0.048, None, None, None, cpg_max.PHASE_WALK_LS,
              None, [60., 120., 120.], [1., 1., 1.], 0.075),
    # ★★ 最終建議：LS + body sway。前/後大腿 41.4°/47.1°（現行 31.5°/62.0°）
    "SWAY":  (-0.020, 0.0, 3.0, 0.07, None, None, None, cpg_max.PHASE_WALK_LS,
              (0.015, 0.010, 0.90, 0.20)),
    # 完美對稱（Dhip 0°）—— 走得動但 max 膝 +33%、偏航混沌，不可用，留著看外觀
    "SYM0":  (0.000, 0.0, 3.0, 0.07, None, None, None, cpg_max.PHASE_WALK_LS,
              (0.020, 0.010, 0.0, 0.20)),
    # ★★ 2026-09-03 下午要上機的兩組（trip17）。
    #   ⚠️ 這裡用**模擬的** z_sag（STATIC_SAG 0.0325），所以代表實機行為；
    #      軌跡檔給狗用的是實機錨點 0.075，拿到模擬裡播會多抬 42 mm。
    # trip16 實際走過的配置（DS、kd0.5、無 sway）—— 實機對照組
    "TRIP16": (-0.040, 0.0, 0.5, 0.08, None, None, None, None),
    # ★★ M10 之後的定案（kd_wheel 實機硬約束 ≤1.0，實走用 0.5）
    "WALK":  (-0.020, 0.0, 0.5, 0.048, 0.10, 1.4, None, cpg_max.PHASE_WALK_LS,
              (0.015, 0.010, 0.90, 0.20)),
    # 完美對稱組（x_off=0，kd=0.5 下可行）—— 參考用
    "WALK0": (0.000, 0.0, 0.5, 0.048, 0.10, 1.4, None, cpg_max.PHASE_WALK_LS,
              (0.015, 0.010, 0.90, 0.20)),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default=None,
                    help="只錄這一個（不加現行當對照），單格大畫面")
    ap.add_argument("--only", default=None,
                    help="只錄「現行 vs 這一個」，畫面較大（A / B / …）")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--add", action="append", default=[],
                    help='臨時加一路，格式 " x_c,x_d,kd,g_c"（公尺）')
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    for i, spec in enumerate(a.add):
        CAND[f"X{i + 1}"] = tuple(float(v) for v in spec.split(","))

    base = "KP120" if a.only == "KP250" else ("TRIP16" if a.only in ("WALK", "WALK0") else "NOW")
    tags = [a.solo] if a.solo else ([base, a.only] if a.only else list(CAND))
    w = {1: 1100, 2: 900, 3: 620}.get(len(tags), 520)
    runs = [Run(t, *CAND[t]) for t in tags]
    for r in runs:
        r.make_renderer(w, 540 if len(tags) <= 2 else 460)

    frames = []
    for i in range(int(a.secs / mm.CTRL_DT)):
        for r in runs:
            r.advance()
        if i % 2 == 0:
            frames.append(np.concatenate([r.frame() for r in runs], axis=1))

    import imageio.v2 as iio                     # noqa: E402
    out = a.out or (f"task7/outputs/x_split_{'_'.join(tags).lower()}.mp4")
    iio.mimsave(out, frames, fps=25, codec="libx264")
    print("[影片]", out, len(frames), "frames")
    for r in runs:
        print(f"  {r.label}  →  前進 {(float(r.r.d.qpos[0]) - r.x0) * 1000:+.0f} mm")
