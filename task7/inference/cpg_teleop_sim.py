"""遙控步態的模擬驗證：開迴路 A 基準 ＋ 逐腿 mu 調變（`realbot/teleop_cmd.py`），
一段一段跑 前進／原地踏步／倒退／左右平移／左右旋轉，量每段的位移、偏航、roll、膝峰、誤差峰，並輸出影片。

    conda run -n rbtdog python task7/inference/cpg_teleop_sim.py --video --out task7/outputs/teleop_sim.md

正負號驗證：+vx 要往 +x（機身前）、+vy 往機身左、+wz 讓偏航角增加（逆時針＝左轉）。
不對就改 `teleop_cmd.TURN_SIGN / VY_SIGN`，狗上沿用同一份。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realbot"))
import cpg_max  # noqa: E402
import cpg_walk_max as cw  # noqa: E402
import gait_baseline as gb  # noqa: E402
import leg_kin  # noqa: E402
import max_model as mm  # noqa: E402
import teleop_cmd as tc  # noqa: E402

A = gb.BASELINE_A
PH = cpg_max.PHASE_WALK_LS
MJ2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
KNEE, HIP = [2, 5, 8, 11], [1, 4, 7, 10]

DEFAULT_SCHEDULE = [   # (名稱, vx, vy, wz, 秒)。含「前進直接切倒退」「左轉直接切右轉」這種現場會發生的急換
    ("前進", 1, 0, 0, 4), ("原地踏步", 0, 0, 0, 4), ("倒退", -1, 0, 0, 4), ("前進", 1, 0, 0, 3),
    ("倒退", -1, 0, 0, 3), ("原地踏步", 0, 0, 0, 2),
    ("左平移", 0, 1, 0, 4), ("右平移", 0, -1, 0, 4), ("原地踏步", 0, 0, 0, 2),
    ("左轉", 0, 0, 1, 4), ("右轉", 0, 0, -1, 4), ("原地踏步", 0, 0, 0, 2),
]


def _arr(d: dict) -> np.ndarray:
    return np.array([d[MJ2SHM[l]] for l in mm.LEGS])


def run(schedule, video=False, seed=0):
    r = cw.Robot(scene=mm.SCENE, actuator_mode="torque_pd", kp3=A["kp3"], kd3=A["kd3"], kd_wheel=A["wheel_kd"])
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(PH)
    x_off = A["x_off"] + seed * 1e-12
    r.reset_standing(cpg_max.stand_targets(ks, f0, x_off), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, x_off))
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    ren = cam = None
    frames = []
    if video:
        import mujoco
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(r.m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)
    c = cpg_max.cpg_init(PH)
    shaper = tc.CmdShaper()
    om = np.full(4, A["omega"])
    rows = []
    t = 0.0
    for name, vx, vy, wz, secs in schedule:
        x0, y0 = float(r.d.qpos[0]), float(r.d.qpos[1])
        yaw0 = cpg_max.yaw_deg(r.d.qpos[3:7])
        yaw_tot, yaw_prev = 0.0, yaw0
        roll, pitch, tau_k, tau_h, err, fell = [], [], 0.0, 0.0, 0.0, False
        n = int(secs / mm.CTRL_DT)
        for i in range(n):
            mux, muy = shaper.step(vx, vy, wz)
            c = step(c, _arr(mux), _arr(muy), om, mm.CTRL_DT)
            q_des, _ = cpg_max.joint_targets(c, f0, x_off, A["g_c"], A["d_step"], A["d_step_y"],
                                             A["duty"], ks, A["z_sag"], None)
            r.step(q_des)
            g = cpg_max.w2b(r.d.qpos[3:7], np.array([0, 0, -1.0]))
            fell |= g[2] > mm.FALL_GRAV_Z
            if i >= n // 3:                                       # 每段後 2/3 當穩態
                roll.append(np.degrees(np.arcsin(np.clip(g[1], -1, 1))))
                pitch.append(np.degrees(np.arcsin(np.clip(-g[0], -1, 1))))
                tau = np.abs(r.d.actuator_force[mm.LEG_ACT_IDX])
                tau_k, tau_h = max(tau_k, tau[KNEE].max()), max(tau_h, tau[HIP].max())
                err = max(err, np.abs(q_des - r.d.qpos[mm.LEG_QPOS_IDX]).max())
            y = cpg_max.yaw_deg(r.d.qpos[3:7])
            yaw_tot += (y - yaw_prev + 180) % 360 - 180
            yaw_prev = y
            t += mm.CTRL_DT
            if ren is not None and i % 2 == 0:
                cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.30]
                cam.distance, cam.elevation, cam.azimuth = 2.2, -20, 135
                ren.update_scene(r.d, cam)
                frames.append(ren.render())
        # 位移轉到該段起點的機身系（前 x、左 y）
        dx_w, dy_w = float(r.d.qpos[0]) - x0, float(r.d.qpos[1]) - y0
        cy, sy = np.cos(np.radians(yaw0)), np.sin(np.radians(yaw0))
        fwd, left = cy * dx_w + sy * dy_w, -sy * dx_w + cy * dy_w
        rows.append(dict(name=name, vx=vx, vy=vy, wz=wz, secs=secs, fwd=fwd, left=left, yaw=yaw_tot,
                         roll_std=float(np.std(roll)), roll_pk=float(np.abs(roll).max()),
                         pitch_std=float(np.std(pitch)), knee=tau_k, hip=tau_h, err=err, fell=bool(fell)))
    return rows, frames


def report(rows) -> str:
    L = ["# 遙控步態模擬（開迴路 A ＋ 逐腿 mu 調變）", "",
         "每段位移在該段起點的機身系（前 +x、左 +y）；姿態與力矩取每段後 2/3。", "",
         "| 段 | (vx,vy,wz) | 秒 | 前進 m | 左移 m | 偏航 ° | 速度 | roll std/峰 ° | pitch std | 膝峰 | 髖峰 | 誤差峰 | 跌倒 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["wz"]:
            spd = f"{r['yaw'] / r['secs']:+.1f} °/s"
        elif r["vy"]:
            spd = f"{r['left'] / r['secs']:+.3f} m/s"
        else:
            spd = f"{r['fwd'] / r['secs']:+.3f} m/s"
        L.append(f"| {r['name']} | ({r['vx']:+.0f},{r['vy']:+.0f},{r['wz']:+.0f}) | {r['secs']} | {r['fwd']:+.2f} | {r['left']:+.2f} | "
                 f"{r['yaw']:+.1f} | {spd} | {r['roll_std']:.2f}/{r['roll_pk']:.1f} | {r['pitch_std']:.2f} | "
                 f"{r['knee']:.1f} | {r['hip']:.1f} | {r['err']:.3f} | {'❌' if r['fell'] else '否'} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--mp4", default=str(HERE.parent / "outputs" / "teleop_sim.mp4"))
    a = ap.parse_args()
    rows, frames = run(DEFAULT_SCHEDULE, video=a.video)
    txt = report(rows)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
    if frames:
        import imageio.v2 as iio
        iio.mimsave(a.mp4, frames, fps=25, codec="libx264")
        print(f"\n🎬 {a.mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
