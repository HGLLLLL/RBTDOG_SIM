#!/usr/bin/env python3
"""崎嶇地形並排比較影片（官方 STL 網格模型）：左＝舊 policy、右＝新 policy，同一塊地形（同 seed）、同指令、同步時間。

物理用 rl_only_rough_terrain 的 hfield 版 v3f 模型；畫面用 scene_flat.xml（官方網格）換 hfield 地板，qpos 按關節名搬過去（同 rl_only_infer 影片做法）。
用法：
  conda run -n rbtdog --no-capture-output python task7/inference/diag/rl_only_rough_compare_video.py \
      --left RL_only/2026-09-24_13-23-11.onnx --left-name "舊 0924（平地訓練）" \
      --right RL_only/2026-10-01_17-30-20/2026-10-01_17-30-20_10000.onnx --right-name "新 10k（崎嶇地形訓練）" \
      --segments "rough:0.04:0.5,0,0;boxes:0.10:0,-0.5,0" --secs 6 --video task7/outputs/rl_only/rough_compare_0924_vs_10k.mp4
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import rl_only_infer as ri  # noqa: E402
import rl_only_rough_terrain as rt  # noqa: E402

MODEL_DIR = rt.MODEL_DIR
W, H = 960, 540


def make_mesh_scene_xml() -> str:
    src = (MODEL_DIR / "scene_flat.xml").read_text(encoding="utf-8")
    n = int(round(2 * rt.HALF / rt.DX)) + 1
    hf = f'    <hfield name="terrain" nrow="{n}" ncol="{n}" size="{rt.HALF} {rt.HALF} {rt.ZMAX} 0.5"/>\n'
    src = src.replace("<asset>\n", "<asset>\n" + hf, 1)
    src = re.sub(r'<geom name="floor"[^>]*/>', '<geom name="floor" type="hfield" hfield="terrain" pos="0 0 0" material="groundplane"/>', src)
    src = "<!-- ⚠️ 產生檔，不要手改。產生器：inference/diag/rl_only_rough_compare_video.py（scene_flat.xml 的 hfield 版，只給渲染） -->\n" + src
    out = MODEL_DIR / "scene_rough_mesh.xml"
    if not out.exists() or out.read_text(encoding="utf-8") != src:
        out.write_text(src, encoding="utf-8")
    return str(out)


def joint_map(m_phys, m_vis, mujoco):
    jmap, free2 = [], [j for j in range(m_vis.njnt) if m_vis.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    for j in range(m_phys.njnt):
        if m_phys.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            if free2:
                jmap.append((int(m_phys.jnt_qposadr[j]), int(m_vis.jnt_qposadr[free2[0]]), 7))
            continue
        nm = mujoco.mj_id2name(m_phys, mujoco.mjtObj.mjOBJ_JOINT, j)
        j2 = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_JOINT, nm) if nm else -1
        if j2 >= 0:
            jmap.append((int(m_phys.jnt_qposadr[j]), int(m_vis.jnt_qposadr[j2]), 1))
    return jmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default=str(rt.ROOT.parent / "RL_only" / "2026-09-24_13-23-11.onnx")); ap.add_argument("--left-name", default="舊 0924（平地訓練）", dest="left_name")
    ap.add_argument("--right", default=str(rt.ROOT.parent / "RL_only" / "2026-10-01_17-30-20" / "2026-10-01_17-30-20_10000.onnx")); ap.add_argument("--right-name", default="新 10k（崎嶇地形訓練）", dest="right_name")
    ap.add_argument("--io", default=str(rt.ROOT.parent / "RL_only" / "policy_io.json"))
    ap.add_argument("--segments", default="rough:0.04:0.5,0,0;rough:0.04:0,0.5,0;rough:0.04:0,0,1.0;rough:0.06:0.5,0,0;rough:0.06:0,0,-1.0;"
                                           "boxes:0.05:0.5,0,0;boxes:0.05:0,0.5,0;boxes:0.10:0.5,0,0;boxes:0.10:1.0,0,0;boxes:0.10:0,-0.5,0",
                    help="地形:振幅:vx,vy,wz 以分號分隔")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--secs", type=float, default=6.0, help="每段影片長（從指令切入起算）")
    ap.add_argument("--video", default=str(rt.OUT / "rough_compare.mp4"))
    ap.add_argument("--no-air", action="store_true", dest="no_air", help="字幕不印「真離地」欄（給報告用；8 cm 樓梯 v1b 是滾上去的，離地 1% 會被問）")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco, imageio
    from PIL import Image, ImageDraw, ImageFont
    io = json.load(open(a.io))
    _pol = lambda path: ri.load_policy(path if path.endswith('.onnx') else None, path if path.endswith('.npz') else None)   # 副檔名決定 onnx／npz（2026-10-14）
    polL = _pol(a.left); polR = _pol(a.right)
    sim = ri.Sim(io, dt=0.005, ideal=False, wheel_kv=1.0, scene=rt.make_scene_xml())
    n = sim.m.hfield_nrow[0]
    m_vis = mujoco.MjModel.from_xml_path(make_mesh_scene_xml()); d_vis = mujoco.MjData(m_vis)
    m_vis.vis.global_.offwidth, m_vis.vis.global_.offheight = W, H
    jmap = joint_map(sim.m, m_vis, mujoco); base_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 28)
    font2 = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 21)
    wr = imageio.get_writer(a.video, fps=25, codec="libx264", quality=8, macro_block_size=None)
    segs = [s for s in a.segments.split(";") if s]
    summary = []
    for k, spec in enumerate(segs):
        kind, amp, cmd_s = spec.split(":"); amp = float(amp); cmd = [float(x) for x in cmd_s.split(",")]
        ter = rt.Terrain(rt.terrain_grid(kind, amp, a.seed, n)); ter.apply(sim.m)
        runs = []
        for pol in (polL, polR):
            R = rt.run(sim, ter, pol, cmd, max(a.secs, 8.0), seed=a.seed); runs.append((R, rt.metrics(R, cmd)))
        m_vis.hfield_data[:] = (ter.h / rt.ZMAX).ravel()
        r = mujoco.Renderer(m_vis, H, W); mujoco.mjr_uploadHField(m_vis, r._mjr_context, 0)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 2.8, 145.0, -18.0
        label = f"{kind} {amp*100:.0f} cm　vx {cmd[0]:+.1f}  vy {cmd[1]:+.1f}  wz {cmd[2]:+.1f}"
        i0 = runs[0][0]["n_settle"]; i1 = min(i0 + int(a.secs / 0.02), len(runs[0][0]["q"]))
        for i in range(i0, i1, 2):
            panels = []
            for (R, g), name in zip(runs, (a.left_name, a.right_name)):
                Q = R["q"]
                for src, dst, nq in jmap:
                    d_vis.qpos[dst:dst + nq] = Q[i][src:src + nq]
                mujoco.mj_forward(m_vis, d_vis); cam.lookat[:] = [d_vis.xpos[base_vis][0], d_vis.xpos[base_vis][1], d_vis.xpos[base_vis][2] - 0.15]
                r.update_scene(d_vis, camera=cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im, "RGBA")
                dr.rectangle([0, 0, W, 76], fill=(0, 0, 0, 150))
                fell = R["fell"]; t_now = (i - i0) * 0.02
                status = "摔倒" if (fell is not None and i * 0.02 >= fell) else ("" if fell is None else "")
                dr.text((14, 6), name + (f"　　× {status}" if status else ""), font=font, fill=(255, 120, 120) if status else (255, 255, 255))
                air = "" if a.no_air else f"  真離地 {np.mean(g['tsteps_s']):.1f} 步/秒・{np.mean(g['tair']):.0f}%"
                dr.text((14, 44), f"vx {g['vx']:+.2f}  vy {g['vy']:+.2f}  偏航 {g['yaw']:+.0f}°/s  roll 峰 {g['roll_max']:.0f}°  pitch 峰 {g['pitch_max']:.0f}°{air}  摔 {'無' if fell is None else f'{fell-1.0:.1f} s'}", font=font2, fill=(255, 235, 120))
                panels.append(im)
            canvas = Image.new("RGB", (2 * W + 6, H + 40), (20, 20, 20))
            canvas.paste(panels[0], (0, 40)); canvas.paste(panels[1], (W + 6, 40))
            dr = ImageDraw.Draw(canvas); dr.text((14, 4), f"{k+1}/{len(segs)}　{label}　　t {t_now:4.1f} s", font=font, fill=(255, 255, 255))
            wr.append_data(np.asarray(canvas))
        r.close()
        gl, gr = runs[0][1], runs[1][1]
        summary.append((label, gl, gr))
        print(f"{label}: 左 vx {gl['vx']:+.2f} 淨空 {np.mean(gl['lift']):.0f} 步/秒 {np.mean(gl['steps_s']):.1f} 摔 {gl['fell']} | 右 vx {gr['vx']:+.2f} 淨空 {np.mean(gr['lift']):.0f} 步/秒 {np.mean(gr['steps_s']):.1f} 摔 {gr['fell']}")
    wr.close(); print("→", a.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
