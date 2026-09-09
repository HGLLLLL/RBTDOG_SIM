"""CPG-RL v3：雙模式（輪行／踏步）模仿原廠步態的 MJX 訓練環境（spec `docs/superpowers/specs/2026-09-09-cpg-rl-v3-dual-mode-design.md`）。

與 v2（rl_env_max）的差別：
  - 指令 (vx, vy, wz) 決定模式：WHEEL（腿站姿＋差速輪）／STEP（原廠式踏步：對角同相旋轉、同側交替平移＋固定輪速圖案）
  - 輪子是主動的速度伺服：ctrl = v_des + sign·τ_ff/kv（等價於 kd·(v_des−v)+τ_ff，M11 定案 kd 1.0、τ_ff 0.13）
  - 動作 12 維：[ω_scale, amp×4, lift, sway×2, wheel_res×4]；零動作＝純開迴路原廠模式表（G0 有標準答案）
  - 觀測 76 維（狗上全部拿得到）：gravity 3 | gyro 3 | jpos 12 | jvel 12 | wheel_vel 4 | cmd 3 | mode 2 | head 1 | last_a 12 | cpg 24
  - reward 加 vy／wz 追蹤、靜態偏置罰（roll、sway_y 的 EMA）、模式紀律（WHEEL 不抬腿、STEP 非踏步腿不抬）
原廠參考數字來自 `outputs/ref_gait_dataset.json`（常數釘在 REF；tests 比對 json）。
本機 CPU 可跑 reset/step 做測試（慢）；訓練在 Colab GPU。
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import max_model as mm  # noqa: E402
import rl_env_max as v2  # noqa: E402
from rl_env_max import (CTRL_DT, DELAY_BASE, FALL_GRAV_Z, GYRO_BIAS, IMU_TILT_DEG, KILL_STEPS,  # noqa: E402
                        MIN_HEIGHT, N_FRAMES, NOISE_GRAV, NOISE_GYRO, NOISE_QPOS, NOISE_QVEL, PUSH_EVERY,
                        PUSH_VEL, SIM_DT, TAU_KILL, _qrot, _quat_rp, err_barrier, fk_j, ik_j,
                        tau_barrier, w2b, yaw_reward)

LEG_QPOS_IDX, LEG_QVEL_IDX = jnp.array(mm.LEG_QPOS_IDX), jnp.array(mm.LEG_QVEL_IDX)
WHEEL_QVEL_IDX = jnp.array(mm.WHEEL_QVEL_IDX)
LEG_ACT_IDX, WHEEL_ACT_IDX = jnp.array(mm.LEG_ACT_IDX), jnp.array(mm.WHEEL_ACT_IDX)
KNEE_SIGN_j = v2.KNEE_SIGN_j
SCENE_V3 = str(Path(mm.SCENE_MJX_KP250).with_name("scene_flat_mjx_v3.xml"))
SCENE_V3P = str(Path(mm.SCENE_MJX_KP250).with_name("scene_flat_mjx_v3p.xml"))

# ---------------------------------------------------------------- 原廠參考（results/L、outputs/ref_gait_dataset.json）
# 腿序 MJCF (FR, FL, RR, RL)。資料集是 shm 序 (fl, fr, bl, br)，這裡已轉。
REF = dict(
    # 輪行站姿（hip/knee 取資料集平均、ABAD 0）：前腿 hip +0.52 / knee −1.20，後腿鏡像
    stance_q12=np.array([0.0, 0.52, -1.20, 0.0, 0.52, -1.20, 0.0, -0.56, 1.20, 0.0, -0.56, 1.20]),
    L_eff=0.375, r_wheel=0.096, v_max=0.92,
    # 踏步：原廠 2.5 Hz／2.1 Hz、duty 0.8（擺動 0.08 s）、抬高 22 mm —— **kp250 位置伺服做不到**（G0 掃描：
    #   照原廠參數踏步腿只離地 4 mm、原地轉 5°/s）。名目改 2.0 Hz／duty 0.5（擺動 0.25 s）／抬 40 mm → 26°/s、膝 69 N·m；
    #   RL 可在 ω ±28%、抬高 ±40% 內調。原廠值留在 factory_* 供模仿獎勵與評估對照。
    step_hz_turn=2.0, step_hz_lat=1.5, duty=0.50, duty_lat=0.85, lift=0.040, lift_lat=0.030,
    factory_hz_turn=2.5, factory_hz_lat=2.1, factory_duty=0.80, factory_lift=0.022,
    # ⚠️ 平移（原廠式單側踏步）在 kp250 下只有 duty ≥ 0.85 站得住且膝 > 100 N·m（G0 掃描 2026-09-09）→
    #   v3.0 **不抽平移指令**（P_LAT = 0），路徑保留；之後另做四腿蟹行。
    # 旋轉（左轉 wz>0）：踏步腿 fl+br 同相；每步位移 (dx, dy) m；站姿腿的輪 fr +Ω / bl −Ω；Ω 4.1 rad/s @ 1.3 rad/s
    turn_step=dict(FL=(-0.021, 0.039), RR=(0.016, -0.025)), turn_wheel=dict(FR=1.0, RL=-1.0),
    turn_omega=4.1, turn_wz_ref=1.3,
    # 平移（左移 vy>0）：踏步腿 fl+bl 交替；每步 fl (−23, +38)、bl (+20, +29) mm；輪 fl +Ω / bl −0.67Ω；Ω 3.3 @ 0.06 m/s
    lat_step=dict(FL=(-0.023, 0.038), RL=(0.020, 0.029)), lat_wheel=dict(FL=1.0, RL=-0.67),
    lat_omega=3.3, lat_vy_ref=0.06,
    trans_s=1.0,
)
KV_WHEEL, TAU_FF, V_DEAD = 1.0, 0.13, 0.3          # M11：kd 1.0、前饋 0.13、|v_des|<0.3 rad/s 視為 0
Z_SAG = 0.036                                      # ★ kp250 承重撓度（M8 實測 36 mm）：命令抬高 = 目標抬高 + Z_SAG，否則腳離不了地
MODE_VY_THR, MODE_WZ_THR, MODE_VX_THR = 0.03, 0.3, 0.08
ACT_DIM = 12
OMEGA_NOM_SCALE, AMP_SCALE, LIFT_SCALE, SWAY_MAX, SWAY_SLEW, WHEEL_RES = 0.28, 0.5, 0.4, 0.040, 0.004, 0.20
FOOT_OFF_MAX = 0.020                    # WHEEL 模式腿的順從空間（±20 mm）
NOISE_WHEEL = 0.10
TAU_BAR, ERR_BAR = 58.0, 0.45
KNEE_V_BAR = 14.0
NOMINAL_HEIGHT = 0.54                   # 原廠 stand 站姿高度（M7 實測 533 mm）
LEG_IDX = {"FR": 0, "FL": 1, "RR": 2, "RL": 3}
LEFT_LEGS, RIGHT_LEGS = jnp.array([1, 3]), jnp.array([0, 2])
KNEE12 = jnp.array([2, 5, 8, 11])

W = dict(W_VX=2.0, W_VY=2.0, W_YAW=2.0, W_YAWI=0.5, W_HEAD=1.0, W_H=0.3, W_LIFT=0.5, W_STANCE=0.5,
         W_ROLL=150.0, W_PITCH=100.0, W_ROLLRATE=0.5, W_PITCHRATE=0.3, W_BIAS=100.0, W_SWAYBIAS=30.0,
         W_ACT=0.05, W_OMDOT=0.5, W_TAU=1e-5, W_TAUBAR=0.05, W_ERRBAR=1.0, W_KNEEV=0.02, W_MODE=2.0, W_VZ=0.05,
         VX_SIG2=0.02, VY_SIG2=0.005, YAW_SIG2=0.0005, YAW_SIG2_WIDE=0.02, YAW_INST_SIG2=0.05, HEAD_SIG=0.15,
         CMD_VX=(-0.4, 0.9), CMD_WZ_WHEEL=0.4, CMD_WZ_TURN=(0.5, 1.3), CMD_VY=(0.03, 0.08),
         P_WHEEL=0.65, P_TURN=0.35, P_SWITCH=0.4, RAMP_STEPS=50, BIAS_EMA=0.02)   # P_LAT = 1 − P_WHEEL − P_TURN = 0
METRIC_KEYS = ("height", "vx", "vy", "wz", "reward", "pitch", "roll", "mode", "clr_step", "clr_stance",
               "yawerr", "vxerr", "vyerr", "tau_pk", "err_pk", "knee_v", "omega", "sway_y", "roll_bias")


# ---------------------------------------------------------------- 幾何
def _stance_feet():
    q = REF["stance_q12"].reshape(4, 3)
    return jnp.stack([fk_j(k, jnp.array(q[k])) for k in range(4)])


STANCE_Q12_j = jnp.array(REF["stance_q12"])
STANCE_FEET_j = _stance_feet()          # (4,3) 相對各腿 ABAD


def _vec4(d: dict, default=0.0):
    out = np.full(4, default, dtype=float)
    for k, v in d.items():
        out[LEG_IDX[k]] = v
    return jnp.array(out)


def _vec42(d: dict):
    out = np.zeros((4, 2))
    for k, v in d.items():
        out[LEG_IDX[k]] = v
    return jnp.array(out)


TURN_STEP_L = _vec42(REF["turn_step"])          # 左轉：FL/RR 踏步
TURN_WHEEL_L = _vec4(REF["turn_wheel"])         # 左轉：FR +1 / RL −1
LAT_STEP_L = _vec42(REF["lat_step"])            # 左移：FL/RL 踏步
LAT_WHEEL_L = _vec4(REF["lat_wheel"])           # 左移：FL +1 / RL −0.67
MIRROR_LR = jnp.array([1, 0, 3, 2])             # FR↔FL、RR↔RL


def mode_of(cmd):
    """→ (u_step ∈ {0,1}, is_turn, is_lat)。"""
    vx, vy, wz = cmd[0], cmd[1], cmd[2]
    is_lat = jnp.abs(vy) >= MODE_VY_THR
    is_turn = (~is_lat) & (jnp.abs(wz) >= MODE_WZ_THR) & (jnp.abs(vx) < MODE_VX_THR)
    return (is_lat | is_turn).astype(jnp.float32), is_turn, is_lat


def step_pattern(cmd, ref=None):
    ref = ref or REF
    """踏步模式的四腿設定：→ (active(4)∈{0,1}, step_vec(4,2) m, phase_off(4) ∈[0,1), wheel_cmd(4) rad/s, omega_hz)。
    旋轉：對角同相；平移：同側交替。方向靠左右鏡像（x 不變、y 反號、腿對調）。"""
    vy, wz = cmd[1], cmd[2]
    u_step, is_turn, is_lat = mode_of(cmd)
    # --- 旋轉（以左轉為基準）
    turn_scale = jnp.clip(jnp.abs(wz) / ref["turn_wz_ref"], 0.2, 1.5)
    t_act = (jnp.abs(TURN_STEP_L).sum(1) > 0).astype(jnp.float32)
    t_vec = TURN_STEP_L * turn_scale
    t_wheel = TURN_WHEEL_L * ref["turn_omega"] * turn_scale
    t_ph = jnp.zeros(4)                                          # 同相
    # 右轉：鏡像
    right = wz < 0
    t_act = jnp.where(right, t_act[MIRROR_LR], t_act)
    t_vec = jnp.where(right, (t_vec[MIRROR_LR]) * jnp.array([1.0, -1.0]), t_vec)
    t_wheel = jnp.where(right, t_wheel[MIRROR_LR], t_wheel)        # 右轉 FL +Ω / RR −Ω（資料：fl +3.8、br −4.1）
    # --- 平移（以左移為基準）
    lat_scale = jnp.clip(jnp.abs(vy) / ref["lat_vy_ref"], 0.3, 1.5)
    l_act = (jnp.abs(LAT_STEP_L).sum(1) > 0).astype(jnp.float32)
    l_vec = LAT_STEP_L * lat_scale
    l_wheel = LAT_WHEEL_L * ref["lat_omega"] * lat_scale
    l_ph = jnp.array([0.0, 0.0, 0.5, 0.5])                       # 前後交替
    rightward = vy < 0
    l_act = jnp.where(rightward, l_act[MIRROR_LR], l_act)
    l_vec = jnp.where(rightward, (l_vec[MIRROR_LR]) * jnp.array([1.0, -1.0]), l_vec)
    l_wheel = jnp.where(rightward, l_wheel[MIRROR_LR], l_wheel)
    active = jnp.where(is_lat, l_act, jnp.where(is_turn, t_act, jnp.zeros(4)))
    vec = jnp.where(is_lat, l_vec, jnp.where(is_turn, t_vec, jnp.zeros((4, 2))))
    ph = jnp.where(is_lat, l_ph, t_ph)
    wheel = jnp.where(is_lat, l_wheel, jnp.where(is_turn, t_wheel, jnp.zeros(4)))
    hz = jnp.where(is_lat, ref["step_hz_lat"], ref["step_hz_turn"])
    duty = jnp.where(is_lat, ref["duty_lat"], ref["duty"])
    lift = jnp.where(is_lat, ref["lift_lat"], ref["lift"])
    return active, vec, ph, wheel, hz, duty, lift


def wheel_cmd_wheelmode(cmd):
    """差速：v_L = vx − wz·L/2、v_R = vx + wz·L/2 → 每輪 rad/s（MJCF 序 FR, FL, RR, RL）。"""
    vx, wz = cmd[0], cmd[2]
    vl = (vx - wz * REF["L_eff"] / 2) / REF["r_wheel"]
    vr = (vx + wz * REF["L_eff"] / 2) / REF["r_wheel"]
    return jnp.array([vr, vl, vr, vl])


def wheel_ctrl(v_des):
    """實機控制律 τ = kv·(v_des − v) + τ_ff·sign 的 MJX 等價：ctrl = v_des + sign·τ_ff/kv；死區 0.3。"""
    dead = jnp.abs(v_des) < V_DEAD
    return jnp.where(dead, 0.0, v_des + jnp.sign(v_des) * TAU_FF / KV_WHEEL)


def duty_remap(th, duty):
    return v2.duty_remap(th, duty)


def act_split(a):
    """tanh 後的 12 維動作 → 各尺度。"""
    a = jnp.tanh(a)
    return dict(om=1.0 + OMEGA_NOM_SCALE * a[0], amp=1.0 + AMP_SCALE * a[1:5], lift=1.0 + LIFT_SCALE * a[5],
                sway=SWAY_MAX * a[6:8], wres=1.0 + WHEEL_RES * a[8:12],
                foot_x=FOOT_OFF_MAX * a[1:5], foot_z=FOOT_OFF_MAX * a[5])


def foot_targets(theta, amp, active, vec, lift, sway, u_step, foot_x, foot_z, duty=None):
    duty = REF["duty"] if duty is None else duty
    """(4,3) 足端目標。STEP：站姿 + 踏步位移（swing 往 +vec、stance 往 −vec）＋ 抬高；WHEEL：站姿 + 小偏移。混合比例 u_step。"""
    th = duty_remap(theta, duty)
    s = jnp.sin(th)
    prog = -0.5 * jnp.cos(th)                                  # −0.5 → +0.5 across a cycle
    dxy = (vec * (amp * active)[:, None]) * prog[:, None]
    dz = jnp.where(s > 0, (lift + Z_SAG) * s, 0.0) * active        # 只加在擺動相（同 v2）
    step_off = jnp.concatenate([dxy, dz[:, None]], 1) + jnp.array([sway[0], sway[1], 0.0])
    wheel_off = jnp.stack([foot_x, jnp.zeros(4), jnp.full(4, foot_z)], 1)
    return STANCE_FEET_j + u_step * step_off + (1 - u_step) * wheel_off


def joint_targets(feet):
    return jnp.stack([ik_j(k, feet[k], KNEE_SIGN_j[k]) for k in range(4)]).reshape(12)


def foot_actual(q12):
    q = q12.reshape(4, 3)
    return jnp.stack([fk_j(k, q[k]) for k in range(4)])


# ---------------------------------------------------------------- env
class DualModeEnv(Env):
    def __init__(self, scene: str = None, wheel_pos: bool = True, ref: dict = None, weights: dict = None):
        """`wheel_pos=True`：輪子用位置＋速度伺服（scene_flat_mjx_v3p，kp 60，ctrl=累加目標角）；
        False：純速度伺服（scene_flat_mjx_v3，M11 實機已驗的 kd 1.0）。`ref`／`weights` 覆寫 REF／W（掃參數用）。"""
        self.w = dict(W, **(weights or {}))
        self.ref = dict(REF, **(ref or {}))
        self.wheel_pos = wheel_pos
        if scene is None:
            scene = SCENE_V3P if wheel_pos else SCENE_V3
        m = mujoco.MjModel.from_xml_path(scene)
        if self.ref.get("floor_mu") is not None:
            m.geom_friction[:, 0] = self.ref["floor_mu"]
        wheel_gain_kp = float(m.actuator_gainprm[mm.WHEEL_ACT_IDX[0], 0])
        wheel_bias_q = float(m.actuator_biasprm[mm.WHEEL_ACT_IDX[0], 1])
        assert (wheel_bias_q < -1e-6) == wheel_pos, "wheel_pos 與模型的輪致動器型式不符"
        self.kv_wheel = float(-m.actuator_biasprm[mm.WHEEL_ACT_IDX[0], 2])
        self.kp_wheel = -wheel_bias_q
        assert m.opt.timestep == SIM_DT
        assert m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
        self._mj = m
        self.sys = mjx.put_model(m)
        self._lo = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 0])
        self._hi = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 1])
        gids = [mm._id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{mm.PREFIX[l]}_FOOT_LINK_COLL") for l in mm.LEGS]
        self._wheel_gids = jnp.array(gids)
        self._wheel_r = float(m.geom_size[gids[0]][0])
        self.obs_dim = 3 + 3 + 12 + 12 + 4 + 3 + 2 + 1 + ACT_DIM + 24
        self._init_q = self._settled_qpos(m)

    def _settled_qpos(self, m):
        d = mujoco.MjData(m)
        q = np.asarray(STANCE_Q12_j)
        d.qpos[:3] = [0.0, 0.0, NOMINAL_HEIGHT + 0.01]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[mm.LEG_QPOS_IDX] = q
        mujoco.mj_forward(m, d)
        for _ in range(int(1.5 / SIM_DT)):
            d.ctrl[mm.LEG_ACT_IDX] = q
            d.ctrl[mm.WHEEL_ACT_IDX] = 0.0
            mujoco.mj_step(m, d)
        assert 0.42 < d.qpos[2] < 0.60, f"站定高度 {d.qpos[2]:.3f} m 不合理"
        return jnp.array(d.qpos)

    def _ctrl(self, q_des, wheel_v, wheel_theta=None):
        if self.wheel_pos:
            # 位置環：ctrl = 累加目標角；前饋等價 = 加 τ_ff/kp 的角度偏移
            dead = jnp.abs(wheel_v) < V_DEAD
            w = jnp.where(dead, wheel_theta, wheel_theta + jnp.sign(wheel_v) * TAU_FF / self.kp_wheel)
        else:
            w = wheel_ctrl(wheel_v)
        return jnp.zeros(16).at[LEG_ACT_IDX].set(q_des).at[WHEEL_ACT_IDX].set(w)

    def _wheel_clearance(self, data):
        return data.geom_xpos[self._wheel_gids, 2] - self._wheel_r

    def _gyro_obs(self, data, info):
        return _qrot(info["imu_q"], data.qvel[3:6]) + info["gyro_bias"]

    def _obs(self, data, info, last_a):
        grav = _qrot(info["imu_q"], w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0])))
        gyro = self._gyro_obs(data, info)
        c = info["c"]
        u = info["u_mode"]
        return jnp.concatenate([
            grav, gyro,
            data.qpos[LEG_QPOS_IDX] - STANCE_Q12_j,
            info["qvel_prev"],
            data.qvel[WHEEL_QVEL_IDX],
            info["cmd"], jnp.array([u, 1.0 - u]),
            jnp.clip(info["head_err"], -1.0, 1.0)[None],
            last_a,
            c["amp"], jnp.sin(c["theta"]), jnp.cos(c["theta"]), c["vec"][:, 0], c["vec"][:, 1], c["wheel"] / 5.0,
        ])

    def _sample_cmd(self, rng):
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
        u = jax.random.uniform(k1)
        vx_w = jax.random.uniform(k2, minval=self.w["CMD_VX"][0], maxval=self.w["CMD_VX"][1])
        wz_w = jnp.where(jax.random.uniform(k5) < 0.5, 0.0,
                         jax.random.uniform(k3, minval=-self.w["CMD_WZ_WHEEL"], maxval=self.w["CMD_WZ_WHEEL"]))
        wz_t = jax.random.uniform(k3, minval=self.w["CMD_WZ_TURN"][0], maxval=self.w["CMD_WZ_TURN"][1]) * jnp.sign(jax.random.uniform(k4) - 0.5)
        vy_l = jax.random.uniform(k4, minval=self.w["CMD_VY"][0], maxval=self.w["CMD_VY"][1]) * jnp.sign(jax.random.uniform(k5) - 0.5)
        wheel = jnp.array([vx_w, 0.0, wz_w])
        turn = jnp.array([0.0, 0.0, wz_t])
        lat = jnp.array([0.0, vy_l, 0.0])
        p_w, p_t = self.w["P_WHEEL"], self.w["P_TURN"]
        return jnp.where(u < p_w, wheel, jnp.where(u < p_w + p_t, turn, lat))

    def reset(self, rng):
        ks = jax.random.split(rng, 8)
        data = mjx.make_data(self.sys).replace(qpos=self._init_q)
        wheel_theta = self._init_q[jnp.array(mm.WHEEL_QPOS_IDX)]
        data = data.replace(ctrl=self._ctrl(self._init_q[LEG_QPOS_IDX], jnp.zeros(4), wheel_theta))
        data = mjx.forward(self.sys, data)
        cmd = self._sample_cmd(ks[0])
        cmd2 = self._sample_cmd(ks[1])
        do_switch = jax.random.uniform(ks[2]) < self.w["P_SWITCH"]
        t_switch = jnp.where(do_switch, jax.random.randint(ks[3], (), 150, 400), 10 ** 6)
        tilt = jax.random.uniform(ks[4], (2,), minval=-IMU_TILT_DEG, maxval=IMU_TILT_DEG) * jnp.pi / 180
        u0, _, _ = mode_of(cmd)
        c = dict(theta=jnp.zeros(4), amp=jnp.zeros(4), vec=jnp.zeros((4, 2)), wheel=jnp.zeros(4))
        z = jnp.zeros(ACT_DIM)
        info = {"rng": ks[5], "c": c, "cmd": cmd, "cmd2": cmd2, "t_switch": t_switch,
                "u_mode": u0, "gyro_bias": jax.random.uniform(ks[6], (3,), minval=-1.0, maxval=1.0) * GYRO_BIAS,
                "head_err": jnp.zeros(()), "imu_q": _quat_rp(tilt[0], tilt[1]),
                "delay": DELAY_BASE + jax.random.bernoulli(ks[7], 0.5).astype(jnp.int32),
                "a_hist": jnp.zeros((3, ACT_DIM)), "last_a": z, "sway": jnp.zeros(2),
                "qvel_prev": data.qvel[LEG_QVEL_IDX], "om_prev": jnp.zeros(()), "wheel_theta": wheel_theta,
                "roll_ema": jnp.zeros(()), "sway_ema": jnp.zeros(()),
                "kill": jnp.zeros((), jnp.int32), "step": 0}
        obs = self._obs(data, info, z)
        zz = jnp.zeros(())
        metrics = {k: zz for k in METRIC_KEYS}
        return State(data, obs, zz, zz, metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        w = self.w
        step_i = info["step"]
        cmd = jnp.where(step_i >= info["t_switch"], info["cmd2"], info["cmd"])
        # ---- 模式與混合
        u_tgt, is_turn, is_lat = mode_of(cmd)
        du = CTRL_DT / self.ref["trans_s"]
        u_mode = jnp.clip(info["u_mode"] + jnp.clip(u_tgt - info["u_mode"], -du, du), 0.0, 1.0)
        # ---- 動作（延遲、淡入）
        a_hist = jnp.concatenate([action[None], info["a_hist"][:2]], 0)
        act = a_hist[info["delay"]]
        u_ramp = jnp.clip(step_i / w["RAMP_STEPS"], 0.0, 1.0)
        act = u_ramp * act
        A = act_split(act)
        # ---- 踏步產生器
        active, vec, ph, wheel_pat, hz, duty, lift0 = step_pattern(cmd, self.ref)
        om = hz * A["om"]
        theta_free = info["c"]["theta"] + 2 * jnp.pi * om * CTRL_DT
        # 相位鎖：腿 k 的目標相位 = 共同相位 + ph[k]；用共同相位推進
        phi0 = theta_free[0] - ph[0]
        theta = jnp.where(u_mode > 0.01, phi0 + ph, info["c"]["theta"]) % (2 * jnp.pi)
        amp = info["c"]["amp"] + jnp.clip(active * A["amp"] - info["c"]["amp"], -0.05, 0.05)   # 振幅斜率
        sway = info["sway"] + jnp.clip(A["sway"] * u_mode - info["sway"], -SWAY_SLEW, SWAY_SLEW)
        feet = foot_targets(theta, amp, active, vec, lift0 * A["lift"], sway, u_mode, A["foot_x"], A["foot_z"], duty)
        q_des = jnp.clip(joint_targets(feet), self._lo, self._hi)
        # ---- 輪速：模式混合 ＋ 殘差
        wheel_v = ((1 - u_mode) * wheel_cmd_wheelmode(cmd) + u_mode * wheel_pat) * A["wres"]
        wheel_theta = info["wheel_theta"] + wheel_v * CTRL_DT              # 位置環的累加目標角
        c = dict(theta=theta, amp=amp, vec=vec * amp[:, None], wheel=wheel_v)

        data = state.pipeline_state
        rng, k_push, k_dir, k_obs = jax.random.split(info["rng"], 4)
        do_push = (step_i % PUSH_EVERY) == (PUSH_EVERY - 1)
        ang = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jnp.pi)
        mag = jax.random.uniform(k_push, minval=0.0, maxval=PUSH_VEL)
        kick = jnp.where(do_push, jnp.array([mag * jnp.cos(ang), mag * jnp.sin(ang), 0.0]), jnp.zeros(3))
        data = data.replace(qvel=data.qvel.at[0:3].add(kick))
        ctrl = self._ctrl(q_des, wheel_v, wheel_theta)

        def one(carry, _):
            d, tpk = carry
            d = mjx.step(self.sys, d.replace(ctrl=ctrl))
            return (d, jnp.maximum(tpk, jnp.abs(d.actuator_force[LEG_ACT_IDX]))), None
        (data, tau_pk12), _ = jax.lax.scan(one, (data, jnp.zeros(12)), None, length=N_FRAMES)

        grav = w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        vb = w2b(data.qpos[3:7], data.qvel[0:3])
        wz = data.qvel[5]
        head_err = info["head_err"] + (self._gyro_obs(data, info)[2] - cmd[2]) * CTRL_DT
        q12 = data.qpos[LEG_QPOS_IDX]
        err12 = q_des - q12
        clr = self._wheel_clearance(data)
        knee_v = jnp.max(jnp.abs(data.qvel[LEG_QVEL_IDX][KNEE12]))
        roll_ema = info["roll_ema"] + w["BIAS_EMA"] * (grav[1] - info["roll_ema"])
        sway_ema = info["sway_ema"] + w["BIAS_EMA"] * (sway[1] - info["sway_ema"])

        # ---- reward
        r_vx = jnp.exp(-(vb[0] - cmd[0]) ** 2 / w["VX_SIG2"])
        r_vy = jnp.exp(-(vb[1] - cmd[1]) ** 2 / w["VY_SIG2"])
        r_yaw = yaw_reward(wz, cmd[2], w["YAW_SIG2"], w["YAW_SIG2_WIDE"])
        r_yawi = yaw_reward(wz, cmd[2], w["YAW_INST_SIG2"])
        r_head = jnp.exp(-(head_err / w["HEAD_SIG"]) ** 2)
        r_h = jnp.exp(-400.0 * (data.qpos[2] - NOMINAL_HEIGHT) ** 2)
        sw = (jnp.sin(duty_remap(theta, duty)) > 0).astype(jnp.float32) * active
        clr_step = jnp.sum(sw * jnp.exp(-((clr - lift0) / 0.012) ** 2)) / jnp.maximum(jnp.sum(sw), 1.0)
        r_lift = u_mode * clr_step
        # 模式紀律：WHEEL 任何腿抬 > 10 mm；STEP 非踏步腿抬 > 10 mm
        lift_pen = jnp.sum(jnp.maximum(clr - 0.010, 0.0) ** 2 * ((1 - u_mode) + u_mode * (1 - active)))
        foot_dev = jnp.sum((foot_actual(q12) - STANCE_FEET_j) ** 2, 1)
        r_stance = (1 - u_mode) * jnp.mean(jnp.exp(-foot_dev / (0.02 ** 2)))
        c_act = jnp.sum((action - info["last_a"]) ** 2)
        c_omdot = (om - info["om_prev"]) ** 2 * u_mode
        c_tau = jnp.sum(data.actuator_force[LEG_ACT_IDX] ** 2)
        T = {
            "t_vx": w["W_VX"] * r_vx, "t_vy": w["W_VY"] * r_vy, "t_yaw": w["W_YAW"] * r_yaw, "t_yawi": w["W_YAWI"] * r_yawi,
            "t_head": w["W_HEAD"] * r_head, "t_h": w["W_H"] * r_h, "t_lift": w["W_LIFT"] * r_lift, "t_stance": w["W_STANCE"] * r_stance,
            "t_roll": w["W_ROLL"] * grav[1] ** 2, "t_pitch": w["W_PITCH"] * grav[0] ** 2,
            "t_rollrate": w["W_ROLLRATE"] * data.qvel[3] ** 2, "t_pitchrate": w["W_PITCHRATE"] * data.qvel[4] ** 2,
            "t_bias": w["W_BIAS"] * roll_ema ** 2 + w["W_SWAYBIAS"] * sway_ema ** 2,
            "t_act": w["W_ACT"] * c_act, "t_omdot": w["W_OMDOT"] * c_omdot, "t_tau": w["W_TAU"] * c_tau,
            "t_taubar": w["W_TAUBAR"] * tau_barrier(tau_pk12), "t_errbar": w["W_ERRBAR"] * err_barrier(err12),
            "t_kneev": w["W_KNEEV"] * jnp.maximum(knee_v - KNEE_V_BAR, 0.0) ** 2,
            "t_mode": w["W_MODE"] * lift_pen, "t_vz": w["W_VZ"] * data.qvel[2] ** 2,
        }
        pos = T["t_vx"] + T["t_vy"] + T["t_yaw"] + T["t_yawi"] + T["t_head"] + T["t_h"] + T["t_lift"] + T["t_stance"]
        neg = (T["t_roll"] + T["t_pitch"] + T["t_rollrate"] + T["t_pitchrate"] + T["t_bias"] + T["t_act"] + T["t_omdot"]
               + T["t_tau"] + T["t_taubar"] + T["t_errbar"] + T["t_kneev"] + T["t_mode"] + T["t_vz"])
        reward = pos - neg
        kill = jnp.where(jnp.max(tau_pk12) > TAU_KILL, info["kill"] + 1, 0)
        done = jnp.where((grav[2] > FALL_GRAV_Z) | (data.qpos[2] < MIN_HEIGHT) | (kill >= KILL_STEPS), 1.0, 0.0)

        info.update({"rng": rng, "c": c, "cmd": info["cmd"], "u_mode": u_mode, "head_err": head_err,
                     "a_hist": a_hist, "last_a": action, "sway": sway, "qvel_prev": data.qvel[LEG_QVEL_IDX],
                     "om_prev": om, "roll_ema": roll_ema, "sway_ema": sway_ema, "kill": kill, "step": step_i + 1,
                     "wheel_theta": wheel_theta})
        info_obs = dict(info, cmd=cmd)
        obs = self._obs(data, info_obs, action)
        n = jax.random.normal(k_obs, (self.obs_dim,))
        obs = (obs.at[0:3].add(NOISE_GRAV * n[0:3]).at[3:6].add(NOISE_GYRO * n[3:6])
               .at[6:18].add(NOISE_QPOS * n[6:18]).at[18:30].add(NOISE_QVEL * n[18:30])
               .at[30:34].add(NOISE_WHEEL * n[30:34]))
        metrics = {"height": data.qpos[2], "vx": vb[0], "vy": vb[1], "wz": wz, "reward": reward,
                   "pitch": jnp.abs(grav[0]) * 57.29578, "roll": jnp.abs(grav[1]) * 57.29578, "mode": u_mode,
                   "clr_step": jnp.sum(sw * clr) / jnp.maximum(jnp.sum(sw), 1.0) * 1000.0,
                   "clr_stance": jnp.max(clr * (1 - active)) * 1000.0,
                   "yawerr": jnp.abs(wz - cmd[2]), "vxerr": jnp.abs(vb[0] - cmd[0]), "vyerr": jnp.abs(vb[1] - cmd[1]),
                   "tau_pk": jnp.max(tau_pk12), "err_pk": jnp.max(jnp.abs(err12)), "knee_v": knee_v, "omega": om,
                   "sway_y": sway[1] * 1000.0, "roll_bias": roll_ema * 57.29578, **T}
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"


# ---------------------------------------------------------------- domain randomization
_BASE_ID = mm._id(mujoco.MjModel.from_xml_path(SCENE_V3), mujoco.mjtObj.mjOBJ_BODY, "base_link")
_LEG_DOF, _WHEEL_DOF = jnp.array(mm.LEG_QVEL_IDX), jnp.array(mm.WHEEL_QVEL_IDX)
_ABAD_DOF = jnp.array(mm.LEG_QVEL_IDX[::3])
_KP_NOM_j = jnp.array(np.tile(np.asarray(mm.KP3_A), 4))
_KD_NOM = float(np.asarray(mm.KD3_A)[1])
ABAD12 = jnp.array([0, 3, 6, 9])


def domain_randomize(sys, rng):
    """v2 那組 ＋ 輪 frictionloss 0.10–0.18、damping 0.005–0.03、ABAD kp ×0.4–1.0（E：靜態側傾用低 ABAD 剛度模擬）。"""
    @jax.vmap
    def per_env(rng):
        k = jax.random.split(rng, 10)
        gf = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.4, maxval=1.4))
        s_kp = jax.random.uniform(k[1], minval=0.8, maxval=1.2)
        s_ab = jax.random.uniform(k[2], minval=0.4, maxval=1.0)
        kv = jax.random.uniform(k[3], minval=0.5, maxval=2.0) * _KD_NOM
        kp_leg = _KP_NOM_j * s_kp
        kp_leg = kp_leg.at[ABAD12].set(_KP_NOM_j[ABAD12] * s_ab)
        gain = sys.actuator_gainprm.at[LEG_ACT_IDX, 0].set(kp_leg)
        bias = sys.actuator_biasprm.at[LEG_ACT_IDX, 1].set(-kp_leg).at[LEG_ACT_IDX, 2].set(-kv)
        bm = sys.body_mass * jax.random.uniform(k[4], (sys.nbody,), minval=0.9, maxval=1.1)
        bm = bm.at[_BASE_ID].add(jax.random.uniform(k[5], minval=0.0, maxval=5.0))
        fl = sys.dof_frictionloss
        fl = fl.at[_LEG_DOF].multiply(jax.random.uniform(k[6], minval=0.5, maxval=1.5))
        fl = fl.at[_ABAD_DOF].set(sys.dof_frictionloss[_ABAD_DOF] * jax.random.uniform(k[7], minval=0.6, maxval=1.4))
        fl = fl.at[_WHEEL_DOF].set(jax.random.uniform(k[8], minval=0.10, maxval=0.18))
        dmp = sys.dof_damping.at[_WHEEL_DOF].set(jax.random.uniform(k[9], minval=0.005, maxval=0.03))
        return gf, gain, bias, bm, fl, dmp

    gf, gain, bias, bm, fl, dmp = per_env(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.replace(geom_friction=0, actuator_gainprm=0, actuator_biasprm=0, body_mass=0,
                              dof_frictionloss=0, dof_damping=0)
    sys = sys.replace(geom_friction=gf, actuator_gainprm=gain, actuator_biasprm=bias, body_mass=bm,
                      dof_frictionloss=fl, dof_damping=dmp)
    return sys, in_axes
