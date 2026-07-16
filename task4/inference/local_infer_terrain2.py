"""本機推論（地形 v2）：自動偵測 fixed(12)/learnable(16)、可跑 flat/rough2 地形、輸出影片。
底層走路不依賴 odom；指令可腳本化。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import numpy as np
import mujoco
import jax, jax.numpy as jnp

import terrain2 as T
import cpg2 as C
import obs2 as O
# 自建 dims-aware policy loader（支援 76/12 舊模型與 80/16 新模型）

SCENE = "mujoco_menagerie/unitree_go2/scene.xml"
CTRL_DT, SIM_DT = 0.02, 0.004
HOME12 = np.array([0.0, 0.9, -1.8] * 4)


def _detect_dims(path):
    """直接從存檔 params tree 判維度：obs 取 normalizer mean，act 取 policy 末層 bias/2。"""
    from brax.io import model
    p = model.load_params(path)
    obs_dim = int(p[0].mean.shape[0])                       # normalizer running mean
    pol = p[1]["params"]                                    # policy MLP
    last = max(pol.keys(), key=lambda k: int(k.split("_")[1]))
    act_dim = int(pol[last]["bias"].shape[0]) // 2          # NormalTanh: 末層=2×act(mean+logstd)
    return obs_dim, act_dim


def load_policy_dims(path, obs_dim, act_dim):
    """依指定維度重建 brax PPO 網路(policy 256/256/128、value 256³、normalize、deterministic)。"""
    import jax, functools
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
    infer(np.zeros(obs_dim, np.float32))                    # warm-up / 驗證維度相容
    return infer


def load_policy_any(path):
    obs_dim, act_dim = _detect_dims(path)
    if act_dim not in (12, 16):
        raise ValueError(f"未知動作維度 {act_dim}（僅支援 12/16）")
    return load_policy_dims(path, obs_dim, act_dim), act_dim


def _make_model(terrain):
    if terrain == "flat":
        m = mujoco.MjModel.from_xml_path(SCENE)
    elif terrain == "rough2":
        m = T.build_terrain2_model(SCENE)
    else:
        raise ValueError(terrain)
    m.opt.timestep = SIM_DT
    kp, kd = 90.0, 3.0
    # 把預設 motor(torque) actuator 轉成 PD 位置伺服：ctrl=目標關節角。
    # 必須同時設 biastype=AFFINE，否則 biasprm 被忽略、ctrl 仍被當成力矩（robot 會癱倒）。
    m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_gainprm[:, 0] = kp; m.actuator_biasprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = -kp; m.actuator_biasprm[:, 2] = -kd
    # ctrlrange 原為 motor 力矩範圍(±23.7/45.43)，位置伺服下應為關節角範圍→放寬避免夾住目標角
    m.actuator_ctrlrange[:, 0] = -6.28; m.actuator_ctrlrange[:, 1] = 6.28
    fr = np.full(m.nu, 23.7); fr[[2, 5, 8, 11]] = 45.43
    m.actuator_forcerange[:, 0] = -fr; m.actuator_forcerange[:, 1] = fr
    m.actuator_forcelimited[:] = 1
    return m


def _gz(terrain, x, y):
    return float(T.gz_np(x, y)) if terrain == "rough2" else 0.0


def _qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def _qrot(q, v):
    u = q[1:4]; t = 2 * np.cross(u, v); return v + q[0] * t + np.cross(u, t)
def w2b(q, v): return _qrot(_qinv(q), v)


def rollout(params_path, terrain="rough2", secs=8.0, cmd=(0.6, 0.0, 0.0), video=False):
    infer, act_dim = load_policy_any(params_path)
    mode = C.detect_mode(act_dim)
    jinvs = C.leg_ik_consts(SCENE)
    m = _make_model(terrain)
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    lo = m.actuator_ctrlrange[:, 0]; hi = m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS]
    n_sub = int(round(CTRL_DT / SIM_DT))
    c = C.cpg_init(); last_a = np.zeros(act_dim)
    cmd = np.asarray(cmd, np.float32)
    frames = []; ren = cam = None
    if video:
        ren = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
    x0 = float(d.qpos[0]); fzmin = fzmax = None; fell = None
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
        c = C.cpg_step(c, mux, muy, om, CTRL_DT)
        q_des = np.array(C.cpg_to_joint_targets(c, jnp.asarray(jinvs), gc))
        d.ctrl[:] = np.clip(q_des, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        last_a = act
        if grav[2] > -0.4 and fell is None:
            fell = i * CTRL_DT
        flz = d.geom_xpos[foot_gid[0]][2]
        fzmin = flz if fzmin is None else min(fzmin, flz)
        fzmax = flz if fzmax is None else max(fzmax, flz)
        if video and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]; cam.distance = 2.5
            cam.elevation = -18; cam.azimuth = 90
            ren.update_scene(d, cam); frames.append(ren.render())
    res = {"mode": mode, "dist": float(d.qpos[0]) - x0, "fell": fell,
           "fz_lift": (fzmax - fzmin), "end_h": float(d.qpos[2])}
    if video and frames:
        import imageio.v2 as iio
        out = f"/home/huang/rbtdog_sim/task4/outputs/terrain2_{mode}_{terrain}.mp4"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        iio.mimsave(out, frames, fps=25, codec="libx264"); res["video"] = out
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--terrain", default="rough2", choices=["flat", "rough2"])
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    print(rollout(a.params, a.terrain, a.secs, (a.vx, a.vy, a.wz), a.video))
