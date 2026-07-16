"""Go2 CPG-RL 地形 v2 MJX 環境：全向指令追蹤、統一 hfield、每腿可學抬腳。"""
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State

import terrain2 as T
import cpg2 as C
import obs2 as O

SCENE_MJX = "mujoco_menagerie/unitree_go2/scene_mjx.xml"
CTRL_DT, SIM_DT = 0.02, 0.004
N_FRAMES = int(round(CTRL_DT / SIM_DT))
HOME12 = jnp.array([0.0, 0.9, -1.8] * 4)
KP_NOM, KD_NOM = 90.0, 3.0
KNEE_IDX = [2, 5, 8, 11]
FOOT_CONTACT_H = 0.03
PUSH_EVERY = 100
PUSH_VEL = 0.6
# terrain 網格常數轉 jnp（gz 用）
XS_J = jnp.asarray(T.XS); YS_J = jnp.asarray(T.YS); H_J = jnp.asarray(T.H)


def gz_j(x, y):
    return T.gz_from(jnp, XS_J, YS_J, H_J, x, y)


def apply_pd(m, kp=KP_NOM, kd=KD_NOM):
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kd
    fr = np.full(m.nu, 23.7); fr[KNEE_IDX] = 45.43
    m.actuator_forcerange[:, 0] = -fr; m.actuator_forcerange[:, 1] = fr
    m.actuator_forcelimited[:] = 1
    return m


def _qinv(q): return jnp.array([q[0], -q[1], -q[2], -q[3]])
def _qrot(q, v):
    u = q[1:4]; t = 2.0 * jnp.cross(u, v); return v + q[0] * t + jnp.cross(u, t)
def w2b(quat, v): return _qrot(_qinv(quat), v)


class Go2Terrain2Env(Env):
    def __init__(self, jinvs):
        m = T.build_terrain2_model(SCENE_MJX); m.opt.timestep = SIM_DT
        m = apply_pd(m)
        # go2_mjx.xml 的 default geom 設 margin=0.001，MJX 的 hfield-sphere
        # 碰撞未實作 margin/gap（put_model 會 raise）→ 清零讓 hfield 足端碰撞可用
        m.geom_margin[:] = 0.0
        m.geom_gap[:] = 0.0
        self._mj = m
        self.sys = mjx.put_model(m)
        self._init_q = jnp.array(m.key_qpos[0])
        self._lo = jnp.array(m.actuator_ctrlrange[:, 0])
        self._hi = jnp.array(m.actuator_ctrlrange[:, 1])
        self._jinvs = jnp.array(jinvs)
        self._foot_gid = jnp.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS])

    @property
    def observation_size(self): return 80
    @property
    def action_size(self): return 16
    @property
    def backend(self): return "mjx"

    def _sample_cmd(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        vx = jax.random.uniform(k1, (), minval=0.0, maxval=1.0)
        vy = jax.random.uniform(k2, (), minval=-0.3, maxval=0.3)
        wz = jax.random.uniform(k3, (), minval=-1.0, maxval=1.0)
        return jnp.array([vx, vy, wz])

    def _base(self, data):
        quat = data.qpos[3:7]; gyro = data.qvel[3:6]
        blin = w2b(quat, data.qvel[0:3])
        grav = w2b(quat, jnp.array([0.0, 0.0, -1.0]))
        return quat, gyro, blin, grav

    def _foot_contact(self, data):
        fx = data.geom_xpos[self._foot_gid, 0]
        fy = data.geom_xpos[self._foot_gid, 1]
        fz = data.geom_xpos[self._foot_gid, 2]
        return (fz - gz_j(fx, fy) < FOOT_CONTACT_H).astype(jnp.float32)

    def _obs(self, data, info):
        _, gyro, blin, grav = self._base(data)
        obs = O.build_obs(grav, blin, gyro,
                          data.qpos[7:19] - HOME12, data.qvel[6:18],
                          info["cmd"], info["last_action"],
                          self._foot_contact(data), info["cpg"])
        return jnp.nan_to_num(jnp.clip(obs, -50.0, 50.0), nan=0.0)

    def reset(self, rng):
        rng, crng, hrng = jax.random.split(rng, 3)
        downhill = jax.random.bernoulli(hrng, 0.5)
        quat = jnp.where(downhill, jnp.array([0.0, 0.0, 0.0, 1.0]),
                         jnp.array([1.0, 0.0, 0.0, 0.0]))
        qpos = self._init_q.at[3:7].set(quat)
        data = mjx.make_data(self.sys).replace(qpos=qpos)
        data = mjx.forward(self.sys, data)
        info = {"rng": rng, "cmd": self._sample_cmd(crng),
                "cpg": C.cpg_init(), "last_action": jnp.zeros(16),
                "step": jnp.zeros((), jnp.int32)}
        obs = self._obs(data, info)
        metrics = {"reward": jnp.zeros(()), "r_lin": jnp.zeros(()),
                   "r_yaw": jnp.zeros(()), "rel_h": jnp.zeros(()),
                   "gc_mean": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        mux, muy, omega, gc = C.action_to_cpg_cmd(action, "learnable")
        cpg = C.cpg_step(state.info["cpg"], mux, muy, omega, CTRL_DT)
        q_des = C.cpg_to_joint_targets(cpg, self._jinvs, gc)
        ctrl = jnp.clip(q_des, self._lo, self._hi)

        def one(d, _):
            return mjx.step(self.sys, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(one, state.pipeline_state, None, N_FRAMES)

        rng, krng = jax.random.split(state.info["rng"])
        step_i = state.info["step"] + 1
        do_push = jnp.mod(step_i, PUSH_EVERY) == 0
        kick = jax.random.uniform(krng, (2,), minval=-PUSH_VEL, maxval=PUSH_VEL)
        qvel = (data.qvel.at[0].add(jnp.where(do_push, kick[0], 0.0))
                          .at[1].add(jnp.where(do_push, kick[1], 0.0)))
        data = data.replace(qvel=qvel)

        info = {**state.info, "cpg": cpg, "last_action": action,
                "rng": rng, "step": step_i}
        obs = self._obs(data, info)
        _, gyro, blin, grav = self._base(data)
        cmd = info["cmd"]
        r_lin = jnp.exp(-((blin[0] - cmd[0]) ** 2 + (blin[1] - cmd[1]) ** 2) / 0.25)
        r_yaw = jnp.exp(-((gyro[2] - cmd[2]) ** 2) / 0.25)
        upright = grav[0] ** 2 + grav[1] ** 2
        gzb = gz_j(data.qpos[0], data.qpos[1])
        rel_h = data.qpos[2] - gzb
        height_pen = (rel_h - 0.30) ** 2
        act_rate = jnp.sum((action - state.info["last_action"]) ** 2)
        reward = (1.5 * r_lin + 1.2 * r_yaw - 1.0 * upright
                  - 0.5 * height_pen - 0.05 * act_rate + 0.05)   # 無 y_pen
        done = jnp.where((rel_h < 0.18) | (grav[2] > -0.4), 1.0, 0.0)
        finite = (jnp.isfinite(reward) & jnp.all(jnp.isfinite(data.qpos))
                  & jnp.all(jnp.isfinite(data.qvel)))
        reward = jnp.where(finite, reward, 0.0)
        done = jnp.where(finite, done, 1.0)
        metrics = {"reward": reward, "r_lin": r_lin, "r_yaw": r_yaw,
                   "rel_h": rel_h, "gc_mean": jnp.mean(gc)}
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=done, metrics=metrics, info=info)


_mm = T.build_terrain2_model(SCENE_MJX)
BASE_ID = mujoco.mj_name2id(_mm, mujoco.mjtObj.mjOBJ_BODY, "base")


def domain_randomize(sys, rng):
    @jax.vmap
    def per_env(rng):
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
        geom_friction = sys.geom_friction.at[:, 0].set(
            jax.random.uniform(k1, minval=0.3, maxval=1.0))
        kp = jax.random.uniform(k2, minval=75.0, maxval=105.0)
        kd = jax.random.uniform(k3, minval=2.0, maxval=4.0)
        gain = sys.actuator_gainprm.at[:, 0].set(kp)
        bias = sys.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)
        body_mass = sys.body_mass * jax.random.uniform(
            k4, (sys.nbody,), minval=0.8, maxval=1.2)
        payload = jax.random.uniform(k5, minval=0.0, maxval=8.0)
        body_mass = body_mass.at[BASE_ID].add(payload)
        return geom_friction, gain, bias, body_mass
    gf, gain, bias, bm = per_env(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.replace(geom_friction=0, actuator_gainprm=0,
                              actuator_biasprm=0, body_mass=0)
    sys = sys.replace(geom_friction=gf, actuator_gainprm=gain,
                      actuator_biasprm=bias, body_mass=bm)
    return sys, in_axes
