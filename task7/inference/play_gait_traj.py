#!/usr/bin/env python3
"""在 MuJoCo 裡播放 `gen_gait_traj.py` 產生的軌跡檔。

════════════════════════════════════════════════════════════════════
這支存在的理由
════════════════════════════════════════════════════════════════════

M9 的架構是「離線產生軌跡 → 狗上播放」。那條路的**唯一保證**就是：
**送去狗上的那個檔案，我們可以先在模擬裡完整播一遍**。

所以這支要回答兩個問題：

1. **軌跡檔本身正確嗎？** —— 播放它的結果，要和「直接跑 CPG」一致。
   ★ 這是在驗**檔案格式與腿序**（`max_model.LEGS` 是 FR,FL,RR,RL、
     SHM 是 fl,fr,bl,br —— 這是本專案反覆出事的地方）。
2. **這組參數在模擬裡會怎樣？** —— 跌不跌倒、傾角、力矩、觸地速度。

⚠️ 模擬**不會**告訴你動態觸地在實機上會怎樣（模型是理想力矩源，
   沒有電流環延遲與背隙），而那正是 kp=250 唯一的實質風險。

用法：
    python3 task7/inference/play_gait_traj.py traj.json
    python3 task7/inference/play_gait_traj.py traj.json --video
    python3 task7/inference/play_gait_traj.py traj.json --no-crosscheck

⚠️ 用 rbtdog 環境跑（要 mujoco）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco                       # noqa: E402
import numpy as np                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord                        # noqa: E402
import cpg_max                      # noqa: E402
import gait_baseline                # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402

MM2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
FOOT_BODY = {"fr": "FAR_FOOT_LINK", "fl": "FBL_FOOT_LINK",
             "br": "RAR_FOOT_LINK", "bl": "RBL_FOOT_LINK"}


def crosscheck(D: dict) -> int:
    """★ 拿「直接跑 CPG」重算一次，逐幀比對軌跡檔。

    驗的是**檔案格式與腿序**，不是物理 —— 兩邊用同一套 numpy 程式，
    所以只要對應關係對了就該逐位元接近。
    """
    p = D["params"]
    B = D["baseline_ref"]
    knee_sign = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    mux = np.full(4, B["mu_x"])
    muy = np.full(4, B["mu_y"])
    om = np.full(4, p["omega"])
    dt = D["dt"]
    q_stand = np.array(D["q_stand"])
    n_ramp = int(round(p["ramp"] / dt))
    n_gait = int(round(p["secs"] / dt))

    worst = 0.0
    for i in range(D["n"]):
        q_g, _ = cpg_max.joint_targets(c, f0, p["x_off"], p["g_c"], p["d_step"],
                                       B["d_step_y"], p["duty"], knee_sign,
                                       p["z_sag"])
        if i < n_ramp:
            u = i / max(n_ramp, 1)
        elif i < n_ramp + n_gait:
            u = 1.0
        else:
            u = 1.0 - (i - n_ramp - n_gait) / max(n_ramp, 1)
        u = 0.0 if u < 0 else (1.0 if u > 1 else u)
        s = 0.5 * (1.0 - math.cos(math.pi * u))
        want = (1.0 - s) * q_stand + s * q_g
        worst = max(worst, float(np.abs(want - np.array(D["q"][i])).max()))
        c = step(c, mux, muy, om, dt)

    print(f"★ 交叉比對「直接跑 CPG」vs 軌跡檔：最大差 {worst:.3e} rad")
    if worst > 1e-5:
        print("❌ 對不上 —— 檔案格式或腿序有問題，**不要送上狗**。")
        return 1
    print("✅ 逐幀吻合（含腿序對應）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="在 MuJoCo 裡播放步態軌跡檔")
    ap.add_argument("traj", type=Path)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--no-crosscheck", action="store_false", dest="crosscheck")
    ap.add_argument("--fixed-cam", action="store_true", dest="fixed_cam",
                    help="★ 攝影機不跟著機身 —— 原地踏步要看的就是牠有沒有留在原地，"
                         "跟拍會把漂移藏起來")
    a = ap.parse_args()

    D = json.loads(a.traj.read_text(encoding="utf-8"))
    if D.get("schema") != "gait_traj/1":
        print(f"❌ 不是軌跡檔（schema={D.get('schema')!r}）")
        return 1
    p = D["params"]
    print(f"軌跡檔 {a.traj.name}")
    print(f"  {'原地踏步' if p['march'] else '前進'}　duty {p['duty']} ω {p['omega']} "
          f"d_step {p['d_step']} g_c {p['g_c']} z_sag {p['z_sag']:.4f}")
    print(f"  腿 kp {D['kp']} kd {D['kd']}　輪 kp {D['wheel_kp']} kd {D['wheel_kd']}")
    print(f"  {D['n']} 幀 @ {1/D['dt']:.0f} Hz = {D['n']*D['dt']:.1f} 秒\n")

    if a.crosscheck and crosscheck(D):
        return 1

    # ---- 檔案的關節順序 → MJCF 的 qpos 索引順序
    # ★ 按**名稱**對應，不按索引。
    mj_names = [MM2SHM[l] + k for l in mm.LEGS for k in coord.LEG_KINDS]
    perm = [D["joints"].index(n) for n in mj_names]
    Q = np.array(D["q"])[:, perm]

    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)
    lo = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 0]
    hi = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 1]
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = np.clip(Q[0], lo + 1e-4, hi - 1e-4)
    d.qpos[2] = 0.55
    # 先讓它落穩（軌跡的第 0 幀就是站姿）
    for _ in range(int(1.0 / m.opt.timestep)):
        e = Q[0] - d.qpos[mm.LEG_QPOS_IDX]
        d.ctrl[mm.LEG_ACT_IDX] = np.clip(D["kp"] * e - D["kd"] * d.qvel[mm.LEG_QVEL_IDX],
                                         -150, 150)
        d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(-D["wheel_kd"] * d.qvel[mm.WHEEL_QVEL_IDX],
                                           -40, 40)
        mujoco.mj_step(m, d)
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    print(f"落穩後機身高 {d.qpos[2]*1000:.0f} mm\n")

    W, H = 1280, 720
    if a.video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    ren = mujoco.Renderer(m, H, W) if a.video else None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 2.6, -12, 135
    if a.fixed_cam:
        cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.3]
        cam.distance = 3.2
    frames = []

    bid = {l: m.body(n).id for l, n in FOOT_BODY.items()}
    nsub = max(1, int(round(D["dt"] / m.opt.timestep)))
    tau_pk = np.zeros(12)
    tilt_pk = 0.0
    fell_at = None
    vz_pk = 0.0
    prev_z = {l: float(d.xpos[bid[l]][2]) for l in bid}
    hgt = []

    for i in range(D["n"]):
        for _ in range(nsub):
            e = Q[i] - d.qpos[mm.LEG_QPOS_IDX]
            tau = np.clip(D["kp"] * e - D["kd"] * d.qvel[mm.LEG_QVEL_IDX], -150, 150)
            d.ctrl[mm.LEG_ACT_IDX] = tau
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(
                -D["wheel_kd"] * d.qvel[mm.WHEEL_QVEL_IDX], -40, 40)
            mujoco.mj_step(m, d)
            tau_pk = np.maximum(tau_pk, np.abs(tau))
        w, x, y, z = d.qpos[3:7]
        tilt_pk = max(tilt_pk, math.degrees(math.acos(max(-1, min(1, 1 - 2*(x*x+y*y))))))
        hgt.append(float(d.qpos[2]))
        for l in bid:
            zc = float(d.xpos[bid[l]][2])
            vz_pk = max(vz_pk, (prev_z[l] - zc) / D["dt"])
            prev_z[l] = zc
        if d.qpos[2] < 0.25 and fell_at is None:
            fell_at = i * D["dt"]
        if ren is not None and i % max(1, int(round(1 / D["dt"] / a.fps))) == 0:
            if not a.fixed_cam:
                cam.lookat[:] = d.qpos[:3]
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())

    dx, dy = float(d.qpos[0]) - x0, float(d.qpos[1]) - y0
    w, x, y, z = d.qpos[3:7]
    yaw = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    T = D["n"] * D["dt"]
    print(f"{'結果':22s}")
    print(f"  跌倒　　　　　{'❌ t=%.1fs' % fell_at if fell_at else '✅ 沒有'}")
    print(f"  機身位移　　　x {dx*1000:+.0f} mm　y {dy*1000:+.0f} mm"
          f"（{'原地踏步應該接近 0' if p['march'] else '前進 %.3f m/s' % (dx/T)}）")
    print(f"  偏航　　　　　{yaw:+.1f}°（{yaw/T:+.2f} °/s）")
    print(f"  機身高　　　　{np.mean(hgt)*1000:.0f} ± {np.std(hgt)*1000:.0f} mm")
    print(f"  最大傾角　　　{tilt_pk:.1f}°")
    print(f"  ★ 足端下降峰速 {vz_pk:.2f} m/s ← 動態觸地，實機唯一沒測過的風險")
    print(f"\n{'關節':16s} {'峰值|τ|':>9s}")
    for j, n in enumerate(mj_names):
        print(f"{n:16s} {tau_pk[j]:9.2f}")
    print(f"\n全部腿關節峰值 {tau_pk.max():.2f} N·m（馬達上限 150）")

    if ren is not None:
        import imageio.v2 as iio
        out = a.traj.with_suffix(".fixed.mp4" if a.fixed_cam else ".mp4")
        iio.mimsave(str(out), frames, fps=a.fps, codec="libx264")
        print(f"\n🎬 {out}　{len(frames)} 幀")
    return 1 if fell_at else 0


if __name__ == "__main__":
    sys.exit(main())
