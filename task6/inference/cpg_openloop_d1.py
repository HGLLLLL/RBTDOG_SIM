"""關卡 3：開迴路 CPG rollout（不含 RL），一次驗證 MJCF、home 姿態、IK Jacobian、
位置伺服增益、CPG 常數五件事。

用法：
  conda run --no-capture-output -n rbtdog python task6/inference/cpg_openloop_d1.py --secs 8 --video

通過標準（2026-08-10 實測校準，摩擦 0.8、求解器 100/50、8 秒、omega=2、mu=1.8）：
  前進/理論 0.85~1.20（實測 4.57/4.61 = 0.99）、不跌倒、FL 抬腳 0.05~0.12 m（實測 0.084 m）。

抬腳量的區間刻意與 task4 Go2 版的 0.03~0.06 不同：G_C = 0.08 是**指令**的離地量，
量到 0.084 代表腿把軌跡追得很準，是通過訊號而非缺陷；沿用 Go2 區間會誤判。

已知限制（不列為失敗）：摩擦超過 1.0 後步態崩潰（實測 1.5 → −0.85 m、3.0 → +0.19 m）。
訓練的摩擦 DR 範圍是 [0.3, 1.0]，實測 0.3/0.5/0.8/1.0 → 5.28/5.20/4.57/4.12 m，全範圍正常。
根因是站立相足端速度為正弦（0 → 0.905 m/s）而機身近似等速，相位內必然有滑動。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭環境錄影用；須在建立 Renderer 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import cpg_d1
import d1_model

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def theory_distance(secs: float, omega: float, mu: float) -> float:
    """開迴路 trot 的理論前進量(m)。

    足端在站立相沿 −x 掃過的行程是 2·D_STEP·fx（cos 由 +1 掃到 −1），機身因此前進同樣距離。
    trot 每個 CPG 週期有**兩個**站立相（對角腿輪流著地），故每週期前進兩個步幅。
    漏算這個 2 會把理論值低估一半，進而把正常步態誤判成打滑（2026-08-10 的首次誤判即此因）。
    """
    fx = 2 * (mu - d1_model.MU_MIN) / (d1_model.MU_MAX - d1_model.MU_MIN) - 1
    return 2 * (2 * d1_model.D_STEP * fx) * omega * secs


def rollout(secs: float = 8.0, omega: float = 2.0, mu: float = 1.8,
            video: bool = False) -> dict:
    """開迴路 CPG rollout。回傳 dist/theory/fell/foot_lift/final_z。"""
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    fl_gid = d1_model.foot_geom_ids(m)[0]
    n_sub = int(round(d1_model.CTRL_DT / m.opt.timestep))

    def apply(q_des):
        d.ctrl[:] = q_des                      # 位置伺服在 MJCF 內，不得有軟體 PD
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

    for _ in range(int(0.5 / d1_model.CTRL_DT)):   # 先用 home 姿態站穩 0.5 秒
        apply(d1_model.HOME12)

    ren = cam = None
    frames = []
    if video:
        ren = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)

    c = cpg_d1.cpg_init()
    x0 = float(d.qpos[0])
    fz = float(d.geom_xpos[fl_gid][2])
    fz_min = fz_max = fz
    fell = None

    for i in range(int(secs / d1_model.CTRL_DT)):
        c = cpg_d1.cpg_step(c, np.full(4, mu), np.full(4, mu),
                            np.full(4, omega), d1_model.CTRL_DT)
        apply(cpg_d1.joint_targets(c, f0s, jinvs))

        grav = cpg_d1.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        if grav[2] > d1_model.FALL_GRAV_Z and fell is None:   # 機身傾倒超過約 66 度
            fell = i * d1_model.CTRL_DT
        fz = float(d.geom_xpos[fl_gid][2])
        fz_min, fz_max = min(fz_min, fz), max(fz_max, fz)

        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
            cam.distance, cam.elevation, cam.azimuth = 2.5, -20, 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    res = {
        "dist": float(d.qpos[0]) - x0,
        "theory": theory_distance(secs, omega, mu),
        "fell": fell,
        "foot_lift": fz_max - fz_min,
        "final_z": float(d.qpos[2]),
    }

    print(f"[開迴路] 前進={res['dist']:+.2f} m  理論={res['theory']:.2f} m  "
          f"比值={res['dist'] / res['theory']:.2f}")
    print(f"[姿態] 跌倒={'是 @%.1fs' % fell if fell is not None else '否'}  "
          f"FL 抬腳={res['foot_lift']:.3f} m  末端機身高={res['final_z']:.3f} m")

    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "openloop_cpg.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--omega", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=1.8)
    ap.add_argument("--video", action="store_true")
    rollout(**vars(ap.parse_args()))
