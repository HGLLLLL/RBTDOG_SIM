"""純 RL 自訓 v1 環境（spec docs/superpowers/specs/2026-10-13-rl-wheel-v1-design.md）。

照同事 Isaac Lab `Wheel-Rough-v0`（RL_only/*_env.yaml、policy_io.json）重做成 Brax Env（MJX 後端）：
obs 53 ＝ ang_vel(3) | projected_gravity(3) | cmd(3) | joint_pos−default(12) | joint_vel(16) | last_action(16)，關節序 PhysX 分組序；
動作 16：Brax 輸出 tanh ∈ [−1,1]，環境乘 ACT_GAIN=4 → 他的尺度 a，再 q_des = default + 0.25·a[:12]、v_des = 5·a[12:]。
三項改動：輪輪 pair ＋ 接觸懲罰（改動一）、追蹤 std 0.25 ＋ 30% 小指令回合（改動二）、real 關節摩擦 DR（改動三）；地形含低階樓梯。
Brax 的 AutoResetWrapper 只還原第一次 reset 的狀態，所以回合結束的重生（出生點、課程、指令）在 step 內自己做，訓練用 `wrap_env`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State, Wrapper
from mujoco import mjx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wheel_terrain as wt  # noqa: E402

IO = json.loads((HERE / "rl_wheel_io.json").read_text(encoding="utf-8"))
SIM_DT, N_FRAMES = 0.005, 4
CTRL_DT = SIM_DT * N_FRAMES
EPISODE_LEN = 1000                      # 20 s
ACT_GAIN = 4.0                          # tanh 輸出 → 他的尺度（|a| ≤ 4；實測他最大 3.5）
OBS_DIM, ACT_DIM = 53, 16
JOINT_NAMES = IO["joints"]["names"]
DEFAULT = jnp.array(IO["joints"]["default_pos"], dtype=jnp.float32)
ACT_SCALE = jnp.array(IO["action"]["scale"], dtype=jnp.float32)
ACT_OFFSET = jnp.array(IO["action"]["offset"], dtype=jnp.float32)
NOISE = dict(ang=0.2, grav=0.05, jpos=0.01, jvel=1.5)     # 他的 UniformNoiseCfg（±）
SPAWN_XY, SPAWN_WINDOW = 0.5, 0.6         # 出生點在方塊中心 ±0.5 m（他的 pose_range）；出生高取 ±0.6 m 窗內地形最高
FEET = ("FAR", "FBL", "RAR", "RBL")
TERM_KEYS = ("lin", "ang", "vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww")
METRIC_KEYS = TERM_KEYS + ("reward", "lin_small", "level", "fell", "dist", "ww_force", "vx_err", "vy_err", "wz_err", "touch_n", "small", "standing")


def quat_rot_inv(q, v):
    """Rᵀ v（q = wxyz，世界 → 機身）。"""
    w, qv = q[0], q[1:4]
    t = 2.0 * jnp.cross(qv, v)
    return v - w * t + jnp.cross(qv, t)


def quat_yaw(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw):
    return jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])


def wrap_pi(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class WheelEnv(Env):
    def __init__(self, scene: str | None = None, map_seed: int = 0, flat: bool = False, noise: bool = True, push: bool = True,
                 fixed_cmd: bool = False, fixed_tile: tuple[int, int] | None = None, small_frac: float = 0.3,
                 std: float = 0.25, w_ww: float = 1.0):
        """flat：全圖平地、出生在原點（對齊驗證）；fixed_cmd：指令讀 info["cmd_fix"]（不重抽、不站立、不小指令）；
        fixed_tile=(kind_idx, level)：固定出生方塊、不升降課程。std／w_ww／small_frac 是三項改動的旋鈕。"""
        self.flat, self.noise, self.push, self.fixed_cmd, self.fixed_tile = flat, noise, push, fixed_cmd, fixed_tile
        self.small_frac, self.std, self.w_ww = small_frac, std, w_ww
        self.scene = scene or wt.make_scene_xml()
        m = mujoco.MjModel.from_xml_path(self.scene)
        assert m.opt.timestep == SIM_DT, m.opt.timestep     # 場景 <option timestep="0.005">（wheel_terrain.make_scene_xml）
        self.H_np = wt.build_map(map_seed, flat=flat)
        wt.load_hfield(m, self.H_np)
        # _contacts 假設金字塔摩擦錐（cone 0）＋ condim 3：每個接觸 4 列 efc、法向力＝4 列和。
        # 換 elliptic 錐或改 condim（含 <pair>）時，_contacts 的法向力解碼就要跟著改，否則讀值錯。
        assert m.opt.cone == 0 and int(m.geom_condim.max()) == 3 and int(m.pair_dim.max() if m.npair else 3) == 3, \
            (m.opt.cone, m.geom_condim.max(), m.pair_dim)
        self._mj = m
        self.sys = mjx.put_model(m)
        self.H = jnp.array(self.H_np, dtype=jnp.float32)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        assert min(jid) >= 0, "policy_io 的關節名在 MJCF 找不到"
        self.QADR = jnp.array([m.jnt_qposadr[j] for j in jid]); self.VADR = jnp.array([m.jnt_dofadr[j] for j in jid])
        trn = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.AID = jnp.array([trn[j] for j in jid])
        self.FLOOR = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.BASE_BID = int(base)
        self.BASE_GIDS = jnp.array(np.where(m.geom_bodyid == base)[0])
        self.WHEEL_GIDS = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p}_FOOT_LINK_COLL") for p in FEET])
        self.FOOT_BIDS = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}_FOOT_LINK") for p in FEET])
        self.Z_STAND = self._z_stand(m)

    # ---------------------------------------------------------------- 幾何
    def _z_stand(self, m) -> float:
        """default pose 下，輪底離地 3 mm 時機身高（CPU 算一次）。"""
        d = mujoco.MjData(m)
        d.qpos[:] = 0; d.qpos[3] = 1.0
        d.qpos[np.asarray(self.QADR)] = np.asarray(DEFAULT)
        mujoco.mj_forward(m, d)
        zmin = min(d.xpos[int(b)][2] for b in np.asarray(self.FOOT_BIDS)) - wt.WHEEL_R
        return float(-zmin + 0.003)

    def _terrain_hmax(self, x, y):
        """出生用：(x,y) ± SPAWN_WINDOW 窗內的地形最高（含 BASE_H）。取包住整個窗的格點（起點 floor、n＋1 格），
        雙線性／三角化地形的最高點必在格點上，所以這是窗內最高的保守上界（用 round 起點會在窗邊漏掉至多 DX/2）。"""
        n = int(round(2 * SPAWN_WINDOW / wt.DX)) + 2
        j = jnp.clip(jnp.floor((x - SPAWN_WINDOW - wt.X0) / wt.DX), 0, wt.NCOL - n).astype(jnp.int32)
        i = jnp.clip(jnp.floor((y - SPAWN_WINDOW - wt.Y0) / wt.DX), 0, wt.NROW - n).astype(jnp.int32)
        return jnp.max(jax.lax.dynamic_slice(self.H, (i, j), (n, n)))

    def _tile_origin(self, kind, level):
        x = wt.X0 + (wt.NB + wt.NT / 2 + level * wt.NT) * wt.DX
        y = wt.Y0 + (wt.NB + wt.NT / 2 + kind * wt.NT) * wt.DX
        return jnp.where(self.flat, 0.0, x), jnp.where(self.flat, 0.0, y)

    # ---------------------------------------------------------------- obs／動作
    def _obs(self, data, cmd, last_a):
        q = data.qpos[3:7]
        ang = data.qvel[3:6]                                  # free joint 角速度＝機身座標
        grav = quat_rot_inv(q, jnp.array([0.0, 0.0, -1.0]))
        jpos = data.qpos[self.QADR[:12]] - DEFAULT[:12]
        jvel = data.qvel[self.VADR]
        return jnp.concatenate([ang, grav, cmd, jpos, jvel, last_a]).astype(jnp.float32)

    def _ctrl(self, a):
        """a：他的尺度（已乘 ACT_GAIN）→ 致動器 ctrl（腿 PD 目標 rad、輪速 rad/s）。"""
        return jnp.zeros(self.sys.nu).at[self.AID].set(a * ACT_SCALE + ACT_OFFSET)

    # ---------------------------------------------------------------- 指令（Task 4 擴充；這裡只有 fixed／零）
    def _sample_cmd(self, rng, yaw):
        k = jax.random.split(rng, 4)
        small = jax.random.uniform(k[0]) < self.small_frac
        standing = jax.random.uniform(k[1]) < 0.1
        v = jnp.where(small, jax.random.uniform(k[2], (2,), minval=-0.4, maxval=0.4), jax.random.uniform(k[2], (2,), minval=-1.0, maxval=1.0))
        head = jnp.where(small, yaw + jax.random.uniform(k[3], minval=-0.5, maxval=0.5), jax.random.uniform(k[3], minval=-jnp.pi, maxval=jnp.pi))
        return jnp.where(standing, 0.0, v), head, standing, small

    def _cmd_now(self, data, info):
        if self.fixed_cmd:
            return info["cmd_fix"]
        i = (info["t"] >= EPISODE_LEN // 2).astype(jnp.int32)          # 10 s 重抽
        v, head = info["cmd_ab"][i], info["head_ab"][i]
        wz = jnp.clip(wrap_pi(head - quat_yaw(data.qpos[3:7])), -1.0, 1.0)   # heading 模式：剛度 1.0、夾 ±1
        return jnp.where(info["standing"], 0.0, jnp.array([v[0], v[1], wz]))

    # ---------------------------------------------------------------- 出生／重生
    def _spawn_state(self, rng, kind, level):
        k = jax.random.split(rng, 6)
        cx, cy = self._tile_origin(kind, level)
        xy = jnp.array([cx, cy]) + jax.random.uniform(k[0], (2,), minval=-SPAWN_XY, maxval=SPAWN_XY)
        yaw = jax.random.uniform(k[1], minval=-jnp.pi, maxval=jnp.pi)
        z = self._terrain_hmax(xy[0], xy[1]) + self.Z_STAND
        qpos = jnp.zeros(self.sys.nq).at[0:2].set(xy).at[2].set(z).at[3:7].set(yaw_quat(yaw)).at[self.QADR].set(DEFAULT)
        data = mjx.make_data(self.sys).replace(qpos=qpos, ctrl=self._ctrl(jnp.zeros(ACT_DIM)))
        data = mjx.forward(self.sys, data)
        cmd_a, head_a, standing, small = self._sample_cmd(k[2], yaw)
        cmd_b, head_b, _, _ = self._sample_cmd(k[3], yaw)
        info = dict(rng=k[4], t=jnp.int32(0), kind=kind, level=level, xy0=xy,
                    cmd_ab=jnp.stack([cmd_a, cmd_b]), head_ab=jnp.stack([head_a, head_b]), standing=standing, small=small,
                    cmd_fix=jnp.zeros(3), cmd=jnp.zeros(3), last_a=jnp.zeros(ACT_DIM),
                    qvel_prev=data.qvel[self.VADR], foot_prev=data.xpos[self.FOOT_BIDS],
                    push_t=jnp.int32(EPISODE_LEN // 2) + jax.random.randint(k[5], (), 0, EPISODE_LEN // 4))   # 10–15 s 後第一次推
        info["cmd"] = self._cmd_now(data, info)
        obs = self._obs(data, info["cmd"], info["last_a"])
        z0 = jnp.zeros(())
        return State(data, obs, z0, z0, {k_: z0 for k_ in METRIC_KEYS}, info)

    def _next_tile(self, rng, data, info):
        """Task 4 換成課程規則；這裡：固定方塊或隨機。"""
        if self.fixed_tile is not None:
            return jnp.int32(self.fixed_tile[0]), jnp.int32(self.fixed_tile[1])
        k = jax.random.split(rng, 2)
        return jax.random.randint(k[0], (), 0, len(wt.KINDS)), jax.random.randint(k[1], (), 0, wt.LEVELS)

    def reset(self, rng):
        k = jax.random.split(rng, 2)
        kind, level = self._next_tile(k[0], None, None)
        return self._spawn_state(k[1], kind, level)

    # ---------------------------------------------------------------- step
    def _physics(self, data, a):
        ctrl = self._ctrl(a)

        def one(d, _):
            return mjx.step(self.sys, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(one, data, None, length=N_FRAMES)
        return data

    def _noisy(self, obs, rng):
        if not self.noise:
            return obs
        k = jax.random.split(rng, 4)
        u = lambda kk, n, s: jax.random.uniform(kk, (n,), minval=-s, maxval=s)   # noqa: E731
        return (obs.at[0:3].add(u(k[0], 3, NOISE["ang"])).at[3:6].add(u(k[1], 3, NOISE["grav"]))
                .at[9:21].add(u(k[2], 12, NOISE["jpos"])).at[21:37].add(u(k[3], 16, NOISE["jvel"])))

    def step(self, state, action):
        info = dict(state.info)
        data = state.pipeline_state
        a = ACT_GAIN * action
        rng, k_noise, k_tile, k_spawn = jax.random.split(info["rng"], 4)
        data = self._physics(data, a)
        cmd = info["cmd"]
        reward, fell, metrics = self._after_physics(data, info, a, cmd)
        t1 = info["t"] + 1
        done_env = fell | (t1 >= EPISODE_LEN)
        # 續走路徑
        info_c = dict(info, rng=rng, t=t1, last_a=a, qvel_prev=data.qvel[self.VADR], foot_prev=data.xpos[self.FOOT_BIDS])
        info_c["cmd"] = self._cmd_now(data, info_c)
        obs_c = self._obs(data, info_c["cmd"], a)
        # 重生路徑（回合結束才用；每步都算，jit 下成本固定）
        kind_n, level_n = self._next_tile(k_tile, data, info)
        new = self._spawn_state(k_spawn, kind_n, level_n)
        new_info = dict(new.info, cmd_fix=info["cmd_fix"]); new_info["cmd"] = self._cmd_now(new.pipeline_state, new_info)
        sel = lambda x, y: jnp.where(done_env, x, y)   # noqa: E731
        data_o = jax.tree_util.tree_map(sel, new.pipeline_state, data)
        obs_o = sel(self._obs(new.pipeline_state, new_info["cmd"], new_info["last_a"]), obs_c)
        info_o = {k_: (sel(new_info[k_], info_c[k_]) if k_ in new_info else info_c[k_]) for k_ in info_c}
        obs_o = self._noisy(obs_o, k_noise)
        return state.replace(pipeline_state=data_o, obs=obs_o, reward=reward, done=fell.astype(jnp.float32), metrics=metrics, info=info_o)

    # ---------------------------------------------------------------- 接觸
    def _contacts(self, data):
        """MJX 接觸表（固定長度、padding）：active、兩端 geom、法向力（走過 mjx.step 才有值）。"""
        c = data._impl.contact
        active = c.dist < 0                          # 忽略 margin（目前所有 geom／pair 的 margin 都是 0）；有 margin 要改成 dist < margin
        nefc = data._impl.efc_force.shape[0]
        rows = jnp.clip(c.efc_address[:, None] + jnp.arange(4)[None, :], 0, nefc - 1)   # 金字塔錐 condim 3：每個接觸 4 列，法向力＝4 列和（mj_contactForce 同式）；padding 由 active 遮掉
        fn = jnp.sum(jnp.maximum(data._impl.efc_force[rows], 0.0), axis=1) * active
        return dict(active=active, g0=c.geom[:, 0], g1=c.geom[:, 1], fn=fn)

    def _base_contact(self, C):
        hit = C["active"] & ((jnp.isin(C["g0"], self.BASE_GIDS) & (C["g1"] == self.FLOOR)) | (jnp.isin(C["g1"], self.BASE_GIDS) & (C["g0"] == self.FLOOR)))
        return jnp.any(hit)

    # ---------------------------------------------------------------- reward（他的十項＋輪輪接觸；權重照 env.yaml）
    def _reward_terms(self, data, info, a, cmd):
        q = data.qpos[3:7]
        vb = quat_rot_inv(q, data.qvel[0:3]); w = data.qvel[3:6]
        err_lin = jnp.sum((cmd[0:2] - vb[0:2]) ** 2); err_ang = (cmd[2] - w[2]) ** 2
        tau = data.actuator_force[self.AID]
        qv = data.qvel[self.VADR]; qacc = (qv - info["qvel_prev"]) / CTRL_DT
        C = self._contacts(data)
        is_floor0, is_floor1 = C["g0"] == self.FLOOR, C["g1"] == self.FLOOR
        other = jnp.where(is_floor0, C["g1"], C["g0"])
        floor_hit = C["active"] & (is_floor0 | is_floor1)
        touch = jnp.array([jnp.any(floor_hit & (other == self.WHEEL_GIDS[k])) for k in range(4)])
        foot = data.xpos[self.FOOT_BIDS]; vfoot = (foot - info["foot_prev"]) / CTRL_DT
        axis = data.xmat[self.FOOT_BIDS][:, :, 1]                                   # 輪軸（FOOT_LINK 的 y 軸）世界向量
        slip = jnp.sum(touch * jnp.sum(vfoot * axis, axis=1) ** 2)                 # 他的 wheel_lateral_slip（spin_axis 1）
        ww_mask = jnp.isin(C["g0"], self.WHEEL_GIDS) & jnp.isin(C["g1"], self.WHEEL_GIDS)
        ww_force = jnp.sum(C["fn"] * ww_mask)
        ww = jnp.sum((C["fn"] * ww_mask / 100.0) ** 2)                              # 法向力 / 100 N 的 L2（改動一；量級由攤帳校正）
        standing_scale = jnp.where(jnp.linalg.norm(cmd) < 0.1, 10.0, 1.0)
        dev = jnp.sum(jnp.abs(data.qpos[self.QADR[:12]] - DEFAULT[:12])) * standing_scale
        s2 = self.std ** 2
        T = dict(lin=2.5 * jnp.exp(-err_lin / s2), ang=1.25 * jnp.exp(-err_ang / s2),
                 vz=-2.0 * vb[2] ** 2, angxy=-0.05 * jnp.sum(w[0:2] ** 2),
                 tau_leg=-2e-4 * jnp.sum(tau[:12] ** 2), tau_wheel=-2e-4 * jnp.sum(tau[12:] ** 2),
                 acc_leg=-2.5e-7 * jnp.sum(qacc[:12] ** 2), acc_wheel=-2.5e-7 * jnp.sum(qacc[12:] ** 2),
                 rate=-0.01 * jnp.sum((a - info["last_a"]) ** 2),
                 slip=-0.075 * slip, dev=-0.1 * dev, ww=-self.w_ww * ww)
        aux = dict(vb=vb, w=w, touch=touch, ww_force=ww_force, err_lin=err_lin, err_ang=err_ang, C=C)
        return T, aux

    def _after_physics(self, data, info, a, cmd):
        T, aux = self._reward_terms(data, info, a, cmd)
        reward = sum(T.values())
        fell = self._base_contact(aux["C"])
        metrics = dict(T, reward=reward, lin_small=T["lin"] * info["small"], level=info["level"].astype(jnp.float32),
                       fell=fell.astype(jnp.float32), dist=jnp.linalg.norm(data.qpos[0:2] - info["xy0"]), ww_force=aux["ww_force"],
                       vx_err=jnp.abs(aux["vb"][0] - cmd[0]), vy_err=jnp.abs(aux["vb"][1] - cmd[1]), wz_err=jnp.abs(aux["w"][2] - cmd[2]),
                       touch_n=jnp.sum(aux["touch"]).astype(jnp.float32), small=info["small"].astype(jnp.float32), standing=info["standing"].astype(jnp.float32))
        return reward, fell, metrics

    @property
    def observation_size(self):
        return OBS_DIM

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"
