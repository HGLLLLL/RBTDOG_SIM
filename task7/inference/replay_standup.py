#!/usr/bin/env python3
"""把實機錄到的原廠站立過程，在 MuJoCo 裡照原參數回放並錄影。

輸入是 `M6_load_probe.py --record` 錄下來的 JSON（2026-08-26 15:47 那次）。
**不重建軌跡模型 —— 直接回放錄到的 `p_des` 時間序列**，
控制律也照實機讀到的：腿 kp=250 / kd=5.0（純 PD，無前饋）、輪 kp=20 / kd=0.5 + 前饋。

為什麼要有這支：
  - 把「實機做了什麼」與「模型會怎麼反應」放在同一個座標系裡比較
  - 2026-08-26 已經知道模擬會**系統性高估**站立力矩（髖 4 倍、ABAD 7 倍），
    這支可以量化那個差距在整個動作過程中長什麼樣，而不只是穩態一個點

用法：
    python task7/inference/replay_standup.py                      # 只算不錄
    python task7/inference/replay_standup.py --video              # 出 mp4
    python task7/inference/replay_standup.py --video --slow 2     # 半速

⚠️ 用 rbtdog 環境跑（要 mujoco）：
    /home/huang/miniforge3/envs/rbtdog/bin/python task7/inference/replay_standup.py --video
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭錄影；必須在建立 Renderer 前設定

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import coord            # noqa: E402
import max_model as mm  # noqa: E402

REC = (Path(__file__).resolve().parents[1] / "logs" / "m_logs_trip7" /
       "M6_20260826_154737.json")
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# 實機讀到的原廠增益（2026-08-26 15:47 全程錄製）
KP_LEG, KD_LEG = 250.0, 5.0
KP_WH, KD_WH = 20.0, 0.5

# max_model 的腿序是 FR,FL,RR,RL；SHM 是 fl,fr,bl,br
MM2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
LEG_SHM = [MM2SHM[l] + k for l in mm.LEGS for k in coord.LEG_KINDS]
WH_SHM = [MM2SHM[l] + coord.KIND_WHEEL for l in mm.LEGS]


def load_series(path: Path):
    """讀錄製 JSON，回傳 (t, des_leg[N,12], des_wh[N,4], ff_wh[N,4], q0_leg[12])。

    ⚠️ `des` 存的是**馬達座標系**，要換算成控制器座標系才能餵給 MJCF
       —— 這是本專案第 7 號坑（拿不同座標系的量互比）的同款陷阱。
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    T, J = d["t"], d["joints"]

    def col(n, k):
        v = J[n][k]
        return np.asarray(v if isinstance(v, list) else [v] * len(T), dtype=float)

    # 增益開啟的那一刻＝動作起點。之前那段是洩力趴著，不用回放。
    kp = col(LEG_SHM[0], "kp")
    on = int(np.argmax(kp > 0))
    sl = slice(on, len(T))

    t = np.asarray(T[on:]) - T[on]
    des_leg = np.stack([[coord.to_ctrl(n, x) for x in col(n, "des")[sl]]
                        for n in LEG_SHM], axis=1)
    q0_leg = np.array([coord.to_ctrl(n, col(n, "q")[on]) for n in LEG_SHM])
    des_wh = np.stack([col(n, "des")[sl] for n in WH_SHM], axis=1)
    ff_wh = np.stack([[coord.SIGN[coord.KIND_WHEEL][n[:2]] * x
                       for x in col(n, "ff")[sl]] for n in WH_SHM], axis=1)
    # 實機的力矩，拿來跟模擬對照（控制器座標系）
    tau_leg = np.stack([[coord.SIGN[n[2:]][n[:2]] * x for x in col(n, "tau")[sl]]
                        for n in LEG_SHM], axis=1)
    return t, des_leg, des_wh, ff_wh, q0_leg, tau_leg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, default=REC)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--slow", type=float, default=1.0, help="放慢倍率（2 = 半速）")
    ap.add_argument("--secs", type=float, default=None, help="只回放前幾秒")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()

    if not a.rec.exists():
        print(f"❌ 找不到錄製檔 {a.rec}")
        return 1
    t, des_leg, des_wh, ff_wh, q0_leg, tau_real = load_series(a.rec)
    if a.secs:
        k = int(np.searchsorted(t, a.secs))
        t, des_leg, des_wh, ff_wh, tau_real = (t[:k], des_leg[:k], des_wh[:k],
                                               ff_wh[:k], tau_real[:k])
    print(f"錄製檔 {a.rec.name}")
    print(f"  回放 {t[-1]:.2f} 秒（{len(t)} 筆 @ {len(t)/t[-1]:.0f} Hz）")
    print(f"  控制律照實機：腿 kp={KP_LEG} kd={KD_LEG}（純 PD）、"
          f"輪 kp={KP_WH} kd={KD_WH}+前饋")

    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)

    # ---- 起始姿勢：實機在增益開啟那一刻的關節角
    # ⚠️ 實機膝會頂在 ±2.80 的機械停點，而 MJCF 的限位是 ±2.791（略緊 0.01）。
    #    不夾住的話 MuJoCo 會用一個很大的限位彈力把腿彈開，整段回放從第一幀就錯。
    lo = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 0]
    hi = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 1]
    q0 = np.clip(q0_leg, lo + 1e-4, hi - 1e-4)
    n_clip = int((q0 != q0_leg).sum())
    if n_clip:
        print(f"  ⚠️ {n_clip} 個關節的起始角超出 MJCF 限位已夾住"
              f"（最大 {np.abs(q0 - q0_leg).max():.4f} rad）——"
              f"實機膝頂在 ±2.80，MJCF 限位是 ±2.791")

    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = q0
    d.qpos[2] = 0.16                      # 趴著的機身高度，先放一點點高讓它落穩
    for _ in range(int(0.5 / m.opt.timestep)):
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)              # 洩力落地，模擬「趴著等指令」
    print(f"  趴姿沉降後機身高 {d.qpos[2]*1000:.0f} mm")

    # ⚠️ MuJoCo 的 offscreen framebuffer 預設只有 640×480，超過就直接 ValueError。
    #    `scene_flat.xml` 沒有 <global offwidth=...>，所以要在這裡程式化設定。
    #    （同樣的問題也會打到 `cpg_walk_max.py --video`，它用 1000 寬。）
    W, H = 1280, 720
    if a.video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    ren = mujoco.Renderer(m, H, W) if a.video else None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 2.4, -12, 135
    frames = []
    tau_sim = np.zeros((len(t), 12))
    hgt = np.zeros(len(t))

    nsub = max(1, int(round((t[1] - t[0]) / m.opt.timestep)))
    for i in range(len(t)):
        for _ in range(nsub):
            e = des_leg[i] - d.qpos[mm.LEG_QPOS_IDX]
            tau = KP_LEG * e - KD_LEG * d.qvel[mm.LEG_QVEL_IDX]
            tau = np.clip(tau, -150.0, 150.0)      # actuatorfrcrange
            d.ctrl[mm.LEG_ACT_IDX] = tau
            ew = des_wh[i] - d.qpos[mm.WHEEL_QPOS_IDX]
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(
                KP_WH * ew - KD_WH * d.qvel[mm.WHEEL_QVEL_IDX] + ff_wh[i], -40, 40)
            mujoco.mj_step(m, d)
        tau_sim[i] = d.ctrl[mm.LEG_ACT_IDX]
        hgt[i] = d.qpos[2]
        if ren is not None and i % max(1, int(round(len(t) / t[-1] / a.fps * a.slow))) == 0:
            cam.lookat[:] = d.qpos[:3]
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())

    # ---------------------------------------------------------------- 對照
    print(f"\n{'':16s} {'實機峰值|τ|':>11s} {'模擬峰值|τ|':>11s} {'模擬/實機':>10s}")
    for j, n in enumerate(LEG_SHM):
        pr, ps = np.abs(tau_real[:, j]).max(), np.abs(tau_sim[:, j]).max()
        print(f"{n:16s} {pr:11.2f} {ps:11.2f} {ps/pr if pr > 1e-6 else 0:9.2f}x")
    print(f"\n全部腿關節峰值：實機 {np.abs(tau_real).max():.2f}　"
          f"模擬 {np.abs(tau_sim).max():.2f} N·m")
    print(f"機身高：起 {hgt[0]*1000:.0f} → 終 {hgt[-1]*1000:.0f} mm"
          f"（實機站立時 MJCF 正向運動學算約 512 mm）")
    if hgt[-1] < 0.30:
        print("⚠️ 模擬沒有站起來 —— 機身最後仍低於 300 mm。")
        print("   可能原因：起始姿勢與實機不同、地面摩擦不足、或模型的質量分佈差異。")

    if ren is not None:
        import imageio.v2 as iio
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / "standup_replay.mp4"
        iio.mimsave(str(out), frames, fps=a.fps, codec="libx264")
        print(f"\n🎬 {out}　{len(frames)} 幀 @ {a.fps} fps"
              f"（{len(frames)/a.fps:.1f} 秒{'，%gx 慢動作' % a.slow if a.slow != 1 else ''}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
