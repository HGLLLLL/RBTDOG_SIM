"""本機推論（地形 v3）：新舊模型並排對比影片。

在同一張 terrain3 難地形（斜坡 30° + 障礙 12cm）上同步跑兩個模型——
預設「新模型 cpg_rl_terrain3 vs 舊模型 cpg_rl_terrain2_1」——並排渲染成一支
mp4，並印出前進距離 / 抬腳量 / gc / 打滑量，量化「這次訓練有沒有改善」。

兩個模型皆 obs80/act16，可互跑；底層走路不依賴 odom，指令腳本化。
用法（訓練出 cpg_rl_terrain3_params.pkl 後）：
  MUJOCO_GL=egl conda run -n rbtdog python task4/inference/local_infer_terrain3.py \
      --new task4/weights/cpg_rl_terrain3_params.pkl --video
  # 下坡：加 --downhill；平地對照：--terrain flat
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import numpy as np
import mujoco
import jax, jax.numpy as jnp

import terrain3 as T
import cpg3 as C
import obs3 as O

SCENE = "mujoco_menagerie/unitree_go2/scene.xml"
CTRL_DT, SIM_DT = 0.02, 0.004
HOME12 = np.array([0.0, 0.9, -1.8] * 4)
DEF_OLD = "task4/weights/cpg_rl_terrain2_1_params.pkl"
OUTDIR = "/home/huang/rbtdog_sim/task4/outputs"


# ---------- policy 載入（dims-aware，支援 80/16 新舊模型）----------
def _detect_dims(path):
    from brax.io import model
    p = model.load_params(path)
    obs_dim = int(p[0].mean.shape[0])
    pol = p[1]["params"]
    last = max(pol.keys(), key=lambda k: int(k.split("_")[1]))
    act_dim = int(pol[last]["bias"].shape[0]) // 2
    return obs_dim, act_dim


def load_policy_dims(path, obs_dim, act_dim):
    import functools
    from brax.io import model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    factory = functools.partial(ppo_networks.make_ppo_networks,
                                policy_hidden_layer_sizes=(256, 256, 128),
                                value_hidden_layer_sizes=(256, 256, 256))
    net = factory(obs_dim, act_dim, preprocess_observations_fn=running_statistics.normalize)
    make_policy = ppo_networks.make_inference_fn(net)
    pol = make_policy(model.load_params(path), deterministic=True)
    jpol = jax.jit(pol); key = jax.random.PRNGKey(0)

    def infer(obs):
        return np.asarray(jpol(jnp.asarray(obs), key)[0])
    infer(np.zeros(obs_dim, np.float32))
    return infer


def load_policy_any(path):
    obs_dim, act_dim = _detect_dims(path)
    if act_dim not in (12, 16):
        raise ValueError(f"未知動作維度 {act_dim}（僅支援 12/16）")
    return load_policy_dims(path, obs_dim, act_dim), act_dim


# ---------- 模型 / 地面 ----------
def _make_model(terrain):
    if terrain == "flat":
        m = mujoco.MjModel.from_xml_path(SCENE)
    elif terrain == "rough3":
        m = T.build_terrain3_model(SCENE)
    else:
        raise ValueError(terrain)
    m.opt.timestep = SIM_DT
    kp, kd = 90.0, 3.0
    # motor(torque) actuator → PD 位置伺服（ctrl=目標關節角）；必須設 AFFINE 否則 biasprm 被忽略。
    m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_gainprm[:, 0] = kp; m.actuator_biasprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = -kp; m.actuator_biasprm[:, 2] = -kd
    m.actuator_ctrlrange[:, 0] = -6.28; m.actuator_ctrlrange[:, 1] = 6.28
    fr = np.full(m.nu, 23.7); fr[[2, 5, 8, 11]] = 45.43
    m.actuator_forcerange[:, 0] = -fr; m.actuator_forcerange[:, 1] = fr
    m.actuator_forcelimited[:] = 1
    return m


def _gz(terrain, x, y):
    return float(T.gz_np(x, y)) if terrain == "rough3" else 0.0


def _qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def _qrot(q, v):
    u = q[1:4]; t = 2 * np.cross(u, v); return v + q[0] * t + np.cross(u, t)
def w2b(q, v): return _qrot(_qinv(q), v)


# ---------- 單一模型 rollout（回傳 frames + 統計）----------
def run_sim(infer, act_dim, terrain, secs, cmd, downhill, render):
    mode = C.detect_mode(act_dim)
    jinvs = C.leg_ik_consts(SCENE)
    m = _make_model(terrain)
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [0, 0, 0, 1.0] if downhill else [1.0, 0, 0, 0]   # 面 -x(下坡) / +x(上坡)
    mujoco.mj_forward(m, d)
    lo = m.actuator_ctrlrange[:, 0]; hi = m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS]
    n_sub = int(round(CTRL_DT / SIM_DT))
    c = C.cpg_init(); last_a = np.zeros(act_dim); cmd = np.asarray(cmd, np.float32)
    ren = cam = None
    if render:
        ren = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
    frames = []
    x0 = float(d.qpos[0]); fl = foot_gid[0]
    fzmin = fzmax = float(d.geom_xpos[fl][2])
    fell = None; gc_hist = []; slip_sum = 0.0; slip_cnt = 0
    prev_xy = np.array([[d.geom_xpos[g][0], d.geom_xpos[g][1]] for g in foot_gid])
    for i in range(int(secs / CTRL_DT)):
        grav = w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        blin = w2b(d.qpos[3:7], d.qvel[0:3])
        fx = np.array([d.geom_xpos[g][0] for g in foot_gid])
        fy = np.array([d.geom_xpos[g][1] for g in foot_gid])
        fz = np.array([d.geom_xpos[g][2] for g in foot_gid])
        gzf = np.array([_gz(terrain, fx[k], fy[k]) for k in range(4)])
        contact = ((fz - gzf) < 0.03).astype(np.float32)
        o = O.build_obs(jnp.asarray(grav), jnp.asarray(blin), jnp.asarray(d.qvel[3:6]),
                        jnp.asarray(d.qpos[7:19] - HOME12), jnp.asarray(d.qvel[6:18]),
                        jnp.asarray(cmd), jnp.asarray(last_a), jnp.asarray(contact), c)
        act = np.array(infer(jnp.asarray(o, jnp.float32)))
        mux, muy, om, gc = C.action_to_cpg_cmd(jnp.asarray(act), mode)
        gc_hist.append(float(np.mean(np.asarray(gc))))
        c = C.cpg_step(c, mux, muy, om, CTRL_DT)
        q_des = np.array(C.cpg_to_joint_targets(c, jnp.asarray(jinvs), gc))
        d.ctrl[:] = np.clip(q_des, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        last_a = act
        # 打滑量：觸地支撐腳的世界水平位移速度（與 env 的 slip 懲罰同義，越小越不打滑）
        cur_xy = np.array([[d.geom_xpos[g][0], d.geom_xpos[g][1]] for g in foot_gid])
        slip_vel = np.linalg.norm(cur_xy - prev_xy, axis=1) / CTRL_DT
        slip_sum += float(np.sum(contact * np.clip(slip_vel, 0.0, 1.0))); slip_cnt += 1
        prev_xy = cur_xy
        if grav[2] > -0.4 and fell is None:
            fell = round(i * CTRL_DT, 2)
        flz = float(d.geom_xpos[fl][2])
        fzmin = min(fzmin, flz); fzmax = max(fzmax, flz)
        if render and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]; cam.distance = 2.8
            cam.elevation = -15; cam.azimuth = 90
            ren.update_scene(d, cam); frames.append(ren.render())
    end_gz = _gz(terrain, d.qpos[0], d.qpos[1])
    stats = {"mode": mode, "dist": round(float(d.qpos[0]) - x0, 2), "fell": fell,
             "fz_lift": round(fzmax - fzmin, 3), "gc_mean": round(float(np.mean(gc_hist)), 3),
             "slip_mean": round(slip_sum / max(slip_cnt, 1), 3),
             "end_relh": round(float(d.qpos[2]) - end_gz, 2)}
    return frames, stats


# ---------- 並排標註 ----------
def _label(img, title, stats):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 40], fill=(0, 0, 0))
    dr.text((8, 4), title, fill=(255, 255, 255))
    tag = f"dist {stats['dist']}m  lift {stats['fz_lift']}m  slip {stats['slip_mean']}"
    if stats["fell"] is not None:
        tag += f"  FELL@{stats['fell']}s"
    dr.text((8, 22), tag, fill=(255, 220, 120))
    return np.asarray(im)


def compare(new_params, old_params, terrain, secs, cmd, downhill, out):
    print(f"[compare] terrain={terrain} cmd={cmd} downhill={downhill} secs={secs}")
    print(f"  new = {new_params}")
    print(f"  old = {old_params}")
    (in_new, ad_new) = load_policy_any(new_params)
    (in_old, ad_old) = load_policy_any(old_params)
    f_new, s_new = run_sim(in_new, ad_new, terrain, secs, cmd, downhill, render=True)
    f_old, s_old = run_sim(in_old, ad_old, terrain, secs, cmd, downhill, render=True)
    print(f"  new(terrain3):   {s_new}")
    print(f"  old(terrain2_1): {s_old}")
    n = min(len(f_new), len(f_old))
    frames = []
    for k in range(n):
        L = _label(f_new[k], "terrain3 (new)", s_new)
        R = _label(f_old[k], "terrain2_1 (old)", s_old)
        frames.append(np.hstack([L, R]))
    if out is None:
        tag = "down" if downhill else "up"
        out = os.path.join(OUTDIR, f"terrain3_compare_{terrain}_{tag}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import imageio.v2 as iio
    iio.mimsave(out, frames, fps=25, codec="libx264")
    print(f"  video -> {out}")
    return {"new": s_new, "old": s_old, "video": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="新模型 cpg_rl_terrain3_params.pkl")
    ap.add_argument("--old", default=DEF_OLD, help="舊模型（預設 terrain2_1）")
    ap.add_argument("--terrain", default="rough3", choices=["flat", "rough3"])
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--downhill", action="store_true", help="spawn 面 -x 下坡")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    compare(a.new, a.old, a.terrain, a.secs, (a.vx, a.vy, a.wz), a.downhill, a.out)
