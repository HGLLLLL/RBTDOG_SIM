"""產生模擬影片：【吊掛空跑】（與實機錄影並排對照）或【落地走路】。

⚠️ 這支刻意不是「一般走路」的模擬。實機測試是把狗吊起來、腳不落地跑步態，
   所以模擬也必須是同樣條件，否則並排比會是兩件不同的事：
     - 機身用 weld equality 焊到世界（吊具），不是站在地上
     - 關掉接觸
     - 位置伺服增益設成實機用的值（預設原廠 kp=20 / kd=0.7）
     - 未驅動的腿（預設 RR，該腿馬達已從 CAN 失聯）給零增益，讓它跟實機一樣垂著
     - 依 --time-scale 播放，與實機同一個倍速

--mode ground 則是一般落地走路（有地板、有接觸、機身自由、四腿全驅動），
用來預覽「RR 修好之後長什麼樣」。

⚠️ 兩種模式的物理完全不同，數字不可互相外推。實測（G_C=0.110、20 秒軌跡）：
   落地在 kp=20 下【走不動】——機身塌到 173mm（正常站姿約 270）、腳只抬 4~5mm。
   原廠站立雖然也用 kp=20，但他們把 p_des 當力矩把手（刻意留 22° 追蹤誤差）；
   本專案的架構是 p_des 就是真實目標、t_ff=0，承重時就會塌。落地要 kp≈80。

用法：
  # 吊掛空跑，對照實機
  python task6/inference/render_hanging_sim.py --time-scale 0.5 --kp 20
  # 落地走路，預覽修好之後
  python task6/inference/render_hanging_sim.py --mode ground --time-scale 0.5 --kp 80 --kd 1.0
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭錄影；須在建立 Renderer 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import d1_model
import gait_export as GE

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# MJCF 腿序 (FL, FR, RL, RR) → SHM 腿序。實機用 SHM 腿序講話。
MJCF_TO_SHM = {0: 1, 1: 0, 2: 3, 3: 2}
SHM_NAME = {0: "FR", 1: "FL", 2: "RR", 3: "RL"}

HOIST_Z = 0.80      # 吊具高度：把機身抬到腿垂下來也不穿地板的位置


def build_model(hoist_z):
    """weld 機身到世界 + 抬高到吊掛高度（吊掛模式專用）。"""
    scene = Path(d1_model.SCENE)
    xml = scene.read_text()
    body = Path(d1_model.SCENE).parent / "d1_edu_w.xml"
    src = body.read_text()
    if 'pos="0 0 0.2948"' not in src:
        raise RuntimeError("d1_edu_w.xml 的 base pos 不如預期，render 腳本要同步更新")
    # 只在暫存副本裡改，不動原始模型
    import tempfile
    tmp_body = tempfile.NamedTemporaryFile("w", suffix=".xml", dir=body.parent,
                                           delete=False)
    tmp_body.write(src.replace('pos="0 0 0.2948"', f'pos="0 0 {hoist_z}"'))
    tmp_body.close()
    xml = xml.replace('d1_edu_w.xml', Path(tmp_body.name).name)
    xml = xml.replace("</mujoco>",
                      '  <equality><weld body1="base"/></equality>\n</mujoco>')
    tmp_scene = tempfile.NamedTemporaryFile("w", suffix=".xml", dir=scene.parent,
                                            delete=False)
    tmp_scene.write(xml)
    tmp_scene.close()
    try:
        return mujoco.MjModel.from_xml_path(tmp_scene.name)
    finally:
        Path(tmp_body.name).unlink(missing_ok=True)
        Path(tmp_scene.name).unlink(missing_ok=True)


def _cjk_font():
    """問系統要一個支援中文的字型檔。找不到回 None。"""
    import subprocess
    for query in ("Noto Sans CJK TC:style=Bold", "sans-serif:lang=zh-TW"):
        try:
            p = subprocess.run(["fc-match", "-f", "%{file}", query],
                               capture_output=True, text=True, timeout=5)
            f = p.stdout.strip()
            if f and Path(f).exists():
                return f
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _caption(frame, lines):
    """在畫面左上角疊上說明文字。並排給人看時，沒有標註很容易搞混哪個是模擬。"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img, "RGBA")
    # ⚠️ 不要寫死字型路徑——各發行版擺的位置不同，找不到就會整排變成豆腐字，
    #    而且是「有輸出、但沒人看得懂」的靜默失敗。用 fc-match 問系統。
    path = _cjk_font()
    if path:
        font = ImageFont.truetype(path, 27)
        small = ImageFont.truetype(path, 20)
    else:
        print("⚠️ 找不到中文字型，字幕會是豆腐字。裝 noto-fonts-cjk 或用 --no-label")
        font = small = ImageFont.load_default()
    dr.rectangle([0, 0, img.width, 14 + 31 * len(lines)], fill=(0, 0, 0, 140))
    y = 9
    for i, t in enumerate(lines):
        dr.text((20, y), t, font=(font if i == 0 else small), fill=(255, 255, 255))
        y += 31
    import numpy as _np
    return _np.asarray(img)


def render(time_scale, secs, kp, kd, skip_shm_legs, fps, out, label=None,
           mode="hanging"):
    GE._ensure_mujoco_stack()
    ground = mode == "ground"
    if ground:
        m = mujoco.MjModel.from_xml_path(d1_model.SCENE)   # 有地板、有接觸
    else:
        m = build_model(HOIST_Z)
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    skip_mjcf = {k for k, v in MJCF_TO_SHM.items() if v in skip_shm_legs}
    for a in range(m.nu):
        leg = a // 3
        g = 0.0 if leg in skip_mjcf else kp      # 未驅動的腿零增益，跟實機一樣垂著
        dmp = 0.0 if leg in skip_mjcf else kd
        m.actuator_gainprm[a][0] = g
        m.actuator_biasprm[a][1] = -g
        m.actuator_biasprm[a][2] = -dmp

    q_mjcf, _ = GE.build_trajectory(d1_model.make_model(), GE.DEPLOY_G_C, secs=secs)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    dt = d1_model.CTRL_DT

    def sample(u):
        x = np.clip(u / dt, 0.0, len(q_mjcf) - 1)
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, len(q_mjcf) - 1)
        w = x - i0
        return q_mjcf[i0] * (1 - w) + q_mjcf[i1] * w

    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    ren = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    # az=270 = 側視、頭朝左（已用「動前腳看畫面哪端變」實測確認：把 FL/FR 的
    # knee 伸直，變的是畫面左端 → 左邊是前方）
    if ground:
        cam.distance, cam.elevation, cam.azimuth = 1.75, -6, 270   # 逐幀跟著機器人平移
    else:
        cam.lookat[:] = [0.0, 0.0, HOIST_Z - 0.26]
        cam.distance, cam.elevation, cam.azimuth = 1.45, -5, 270

    frames = []
    step_per_frame = max(1, int(round(1.0 / fps / m.opt.timestep)))

    # 先讓腿沉降/站穩到第 0 幀的穩態，避免影片一開始在抖
    for _ in range(int((0.8 if ground else 1.5) / m.opt.timestep)):
        d.ctrl[:] = np.clip(q_mjcf[0], lo, hi)
        mujoco.mj_step(m, d)

    wall = (len(q_mjcf) - 1) * dt / time_scale
    n_steps = int(wall / m.opt.timestep)
    for k in range(n_steps):
        d.ctrl[:] = np.clip(sample(k * m.opt.timestep * time_scale), lo, hi)
        mujoco.mj_step(m, d)
        if k % step_per_frame == 0:
            if ground:      # 機器人會前進，鏡頭跟著走，否則很快出畫面
                cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.20]
            ren.update_scene(d, cam)
            frames.append(ren.render())

    if label:
        frames = [_caption(f, label) for f in frames]

    import imageio.v2 as iio
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iio.mimsave(str(out), frames, fps=fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p"])
    skipped = "、".join(SHM_NAME[i] for i in sorted(skip_shm_legs)) or "無"
    print(f"[影片] {out}")
    print(f"  {len(frames)} 幀 @ {fps}fps = {len(frames) / fps:.2f}s"
          f"（實機同設定的播放段是 {wall:.2f}s）")
    cond = "落地(有接觸、機身自由)" if ground else "吊掛(weld)、無接觸"
    print(f"  條件：{cond}、kp={kp}/kd={kd}、{time_scale}× 播放、未驅動腿 {skipped}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-scale", type=float, default=0.5, dest="time_scale")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=0.7)
    ap.add_argument("--skip-legs", default=None,
                    help="不驅動的腿，SHM 腿序，逗號分隔。"
                         "預設：hanging 模式為 2（RR 失聯）、ground 模式為四腿全驅動")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-label", action="store_true", help="不要疊字幕")
    ap.add_argument("--mode", choices=("hanging", "ground"), default="hanging",
                    help="hanging=吊掛空跑（對照實機）；ground=落地走路（預覽修好後）")
    a = ap.parse_args()
    default_skip = "" if a.mode == "ground" else "2"
    skip = {int(x) for x in (a.skip_legs if a.skip_legs is not None
                             else default_skip).split(",") if x.strip()}
    out = Path(a.out) if a.out else (
        OUT_DIR / f"sim_{a.mode}_{a.time_scale:g}x_kp{a.kp:g}_side.mp4")
    if a.mode == "ground":
        lab = None if a.no_label else [
            "模擬 (MuJoCo) · 落地走路",
            f"walk_stable · {a.time_scale:g}× 播放 · kp={a.kp:g}/kd={a.kd:g} · 四腿全驅動",
            "預覽「右後腿修好之後」；尚未在實機驗證過落地",
        ]
    else:
        lab = None if a.no_label else [
            "模擬 (MuJoCo)",
            f"walk_stable 吊掛空跑 · {a.time_scale:g}× 播放 · kp={a.kp:g}/kd={a.kd:g}",
            "機身固定、腳不落地；右後腿 RR 零增益（馬達失聯，與實機相同）",
        ]
    render(a.time_scale, a.secs, a.kp, a.kd, skip, a.fps, out, lab, a.mode)
