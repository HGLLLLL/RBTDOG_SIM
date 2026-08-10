"""關卡 6：本機 CPU 推論，載入 Colab 訓練的權重，輸出影片與指標。

用法：
  # 沒權重時先測管線（固定動作）
  conda run --no-capture-output -n rbtdog python task6/inference/local_infer_d1.py --dummy --secs 8 --video
  # 有權重
  conda run --no-capture-output -n rbtdog python task6/inference/local_infer_d1.py \
      --params task6/weights/cpg_rl_d1w_params.pkl --secs 20 --video --push

網路結構必須與 Colab 訓練時逐項相同（policy (256,256,128)、value (256,256,256)、
normalize_observations=True），這些數字不得單方面修改。

不匹配時的後果分兩種，差別很大：

  - **隱藏層大小**對不上 → flax 會丟 `ScopeParamShapeError`，當場停住，安全。
  - **activation 函式與動作分布類型**對不上 → 參數形狀完全相同，brax
    **不會報錯**，權重照樣載入，只是 policy 行為錯亂而毫無訊息
    （實測 deterministic 動作偏差可達 0.59）。

而 activation 與分布是由 `make_ppo_networks` 的**預設值**決定的，也就是說
它們由 **brax 的版本**決定，不由本檔的參數決定。所以 Colab notebook 的安裝格
鎖死 `brax==0.14.2`（本機推論端版本）並在裝完後斷言，是這個靜默失敗的唯一防線。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭環境錄影用；須在 import mujoco 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import cpg_d1
import d1_model
import obs_d1

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
PUSHES = [(4.0, 0.2, 60.0), (9.0, 0.2, -60.0), (14.0, 0.2, 80.0)]   # (t0, 持續秒, 側向力 N)


def load_policy(path: str):
    import functools

    import jax
    from brax.io import model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    factory = functools.partial(ppo_networks.make_ppo_networks,
                                policy_hidden_layer_sizes=(256, 256, 128),
                                value_hidden_layer_sizes=(256, 256, 256))
    net = factory(d1_model.OBS_DIM, d1_model.ACT_DIM,
                  preprocess_observations_fn=running_statistics.normalize)
    pol = ppo_networks.make_inference_fn(net)(model.load_params(path), deterministic=True)
    jpol = jax.jit(pol)
    key = jax.random.PRNGKey(0)

    def infer(obs):
        return np.asarray(jpol(jax.numpy.asarray(obs), key)[0])

    infer(np.zeros(d1_model.OBS_DIM, np.float32))    # 先 warm up 編譯
    return infer


def run(args) -> dict:
    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    foot_gid = d1_model.foot_geom_ids(m)
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert base_id >= 0, "名稱契約破裂：找不到 body base"
    n_sub = int(round(d1_model.CTRL_DT / m.opt.timestep))

    if args.dummy or not args.params:
        if not args.params and not args.dummy:
            print("[warn] 無 --params，改用 dummy 固定動作測管線")
        fixed = np.tile([0.69, 0.0, -0.111], 4).astype(np.float32)
        def infer(obs): return fixed
    else:
        print(f"[info] 載入 {args.params}")
        infer = load_policy(args.params)
        print("[info] ok")

    def apply(q_des):
        d.ctrl[:] = q_des                      # 位置伺服在 MJCF 內，不得有軟體 PD
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

    for _ in range(int(0.5 / d1_model.CTRL_DT)):   # 先用 home 姿態站穩 0.5 秒
        apply(d1_model.HOME12)

    ren = cam = None
    frames = []
    if args.video:
        ren = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)

    c = cpg_d1.cpg_init()
    last_a = np.zeros(d1_model.ACT_DIM)
    cmd = np.array([args.vx, 0.0, 0.0], np.float32)
    x0 = float(d.qpos[0])
    fz_min = fz_max = float(d.geom_xpos[foot_gid[0]][2])
    fell = None

    for i in range(int(args.secs / d1_model.CTRL_DT)):
        t = i * d1_model.CTRL_DT
        d.xfrc_applied[base_id] = 0.0
        if args.push:
            for t0, dur, fy in PUSHES:
                if t0 <= t < t0 + dur:
                    d.xfrc_applied[base_id, 1] = fy

        obs = obs_d1.build_obs(d, c, cmd, last_a)
        act = infer(obs)
        mux, muy, om = cpg_d1.act_to_cmd(act)
        c = cpg_d1.cpg_step(c, mux, muy, om, d1_model.CTRL_DT)
        apply(cpg_d1.joint_targets(c, f0s, jinvs))
        last_a = act

        grav = cpg_d1.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        if grav[2] > d1_model.FALL_GRAV_Z and fell is None:   # 機身傾倒超過約 66 度
            fell = t
        fz = float(d.geom_xpos[foot_gid[0]][2])
        fz_min, fz_max = min(fz_min, fz), max(fz_max, fz)

        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
            cam.distance, cam.elevation, cam.azimuth = 2.5, -20, 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    d.xfrc_applied[base_id] = 0.0
    res = {"dist": float(d.qpos[0]) - x0, "lateral": float(d.qpos[1]),
           "fell": fell, "foot_lift": fz_max - fz_min, "final_z": float(d.qpos[2])}
    print(f"[result] 前進={res['dist']:+.2f} m  側偏={res['lateral']:+.2f} m  "
          f"跌倒={'是 @%.1fs' % fell if fell is not None else '否'}  "
          f"FL 抬腳={res['foot_lift']:.3f} m  末端機身高={res['final_z']:.3f} m"
          f"（正常值 {d1_model.NOMINAL_HEIGHT:.3f} m）")

    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "_push" if args.push else ""
        out = OUT_DIR / f"infer_d1w{tag}.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--push", action="store_true")
    run(ap.parse_args())
