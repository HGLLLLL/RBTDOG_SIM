#!/usr/bin/env python3
"""圖 P10-a：D1 Max（zgws）MJCF 模型的白底渲染圖。

用途是簡報／論文插圖，所以場景要乾淨：天空盒改純白、地板棋盤改淺灰、
haze 改白、反射壓到 0.05。改法是**讀 scene_flat.xml 的文字做替換後，
寫成同目錄的暫存檔**再載入 —— 必須同目錄，否則裡面的
`<include file="zgws.xml"/>` 會找不到。

姿勢用 `max_model.STAND`（承重站立那組，落穩約 530 mm），
先跑 1.5 秒 PD 迴圈讓牠落穩再拍，作法與 `inference/play_gait_traj.py` 相同。

用法：
    ~/miniforge3/envs/rbtdog/bin/python task7/docs/gen_figP10_mjcf.py

⚠️ 用 rbtdog 環境跑（要 mujoco）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco                       # noqa: E402
import numpy as np                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))

import max_model as mm              # noqa: E402

W, H = 2400, 1500
OUT_MAIN = ROOT / "docs" / "圖P10_D1Max_MJCF.png"
OUT_SIDE = ROOT / "docs" / "圖P10_D1Max_MJCF_side.png"

# 站姿用的增益 —— 取自 outputs/A_kp250_walk.json（kp 250 / kd 2.0 / wheel_kd 0.5）
KP, KD, WHEEL_KD = 250.0, 2.0, 0.5

# ---- scene_flat.xml → 白底版的文字替換（原字串 → 新字串）
SUBS = [
    # 天空盒：藍→黑漸層 改成純白
    ('<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"',
     '<texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"'),
    # 地板棋盤：深藍灰 改成淺灰
    ('rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"',
     'rgb1="0.93 0.93 0.93" rgb2="0.86 0.86 0.86" markrgb="0.78 0.78 0.78"'),
    # haze：藍 改成白
    ('<rgba haze="0.15 0.25 0.35 1"/>', '<rgba haze="1 1 1 1"/>'),
    # 反射：0.2 → 0.05
    ('reflectance="0.2"', 'reflectance="0.05"'),
]


def white_scene() -> Path:
    """把 scene_flat.xml 換成白底版，寫到同目錄的暫存檔並回傳路徑。"""
    src = Path(mm.SCENE)
    txt = src.read_text(encoding="utf-8")
    for old, new in SUBS:
        if old not in txt:
            raise SystemExit(f"❌ scene_flat.xml 找不到要替換的字串：{old[:60]}…")
        txt = txt.replace(old, new)
    dst = src.with_name("_scene_white_tmp.xml")
    dst.write_text(txt, encoding="utf-8")
    return dst


def main() -> int:
    tmp = white_scene()
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)   # 載入後就不需要了

    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)

    d = mujoco.MjData(m)
    # ★ 12 個腿關節在 qpos 裡不連續，一定要用 LEG_QPOS_IDX，不能用 qpos[7:19]
    q = mm.STAND.reshape(12)
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = q
    d.qpos[2] = 0.55

    # 落穩：與 play_gait_traj.py 同一套 PD 迴圈
    for _ in range(int(1.5 / m.opt.timestep)):
        e = q - d.qpos[mm.LEG_QPOS_IDX]
        d.ctrl[mm.LEG_ACT_IDX] = np.clip(KP * e - KD * d.qvel[mm.LEG_QVEL_IDX],
                                         -150, 150)
        d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(
            -WHEEL_KD * d.qvel[mm.WHEEL_QVEL_IDX], -40, 40)
        mujoco.mj_step(m, d)
    hz = float(d.qpos[2]) * 1000
    print(f"落穩後機身高 {hz:.1f} mm")
    if not (525.0 <= hz <= 545.0):
        print(f"⚠️ 不在預期的 530–545 mm 區間 —— 圖還是會出，但數字要重新確認")

    ren = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # 關掉 site 顯示 —— livox_imu 那顆 site 在機身上方 176 mm，
    # 在白底圖上會變成一個莫名其妙的黑點。
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.sitegroup[:] = 0
    look = [float(d.qpos[0]), float(d.qpos[1]), 0.30]

    import imageio.v2 as iio
    for out, az, el, dist in [(OUT_MAIN, 135.0, -15.0, 2.3),
                              (OUT_SIDE, 90.0, -5.0, 2.3)]:
        cam.lookat[:] = look
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        ren.update_scene(d, camera=cam, scene_option=opt)
        iio.imwrite(str(out), ren.render())
        print(f"🖼  {out.name}　{W}×{H}　azimuth {az:.0f}° elevation {el:.0f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
