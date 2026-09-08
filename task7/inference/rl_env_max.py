"""RL v2 的 MJX 訓練環境（JAX）。Colab notebook 直接 import 這支，本機 CPU 可跑 reset/step 做測試。

★ 為什麼從 notebook 抽出來：上一版 env 住在 notebook 裡，改一行要重跑 Colab 才知道錯；
  抽成模組後 JAX vs numpy 的逐點對照、sway 斜率、護欄數值都是本機 pytest 釘住的。

設計：docs/superpowers/specs/2026-09-08-cpg-rl-v2-retrain-design.md
  - 動作 14 維：每腿 (mux, muy, ω) ×4 ＋ body sway (x, y)，sway ±0.06 m、斜率 0.004 m/步
  - obs 70 維（obs_max.OBS_LAYOUT）；IMU 偏轉 / gyro 偏置 / 延遲 / 雜訊依 H 文件實測
  - reward 優先序：姿態 > 前腳執行率與對稱 > 航向 > 速度；力矩/誤差護欄
  - 終止：翻倒、太低、|τ|>90 N·m 連續 3 步

⚠️ 常數一律從 repo import（gait_baseline.BASELINE_A / max_model.KP3_A / obs_max），
   不在這裡重打。重打就是第二份真實來源。
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
import cpg_max                  # noqa: E402
import gait_baseline as gb      # noqa: E402
import leg_kin                  # noqa: E402
import max_model as mm          # noqa: E402
import obs_max                  # noqa: E402

# ---------------------------------------------------------------- 常數（唯一來源：repo）
A = gb.BASELINE_A
CTRL_DT, SIM_DT = mm.CTRL_DT, mm.SIM_DT
N_FRAMES = int(round(CTRL_DT / SIM_DT))          # 10
A_CONV, W_COUP, N_CPG_SUB = mm.A_CONV, mm.W_COUP, mm.N_CPG_SUB
MU_MIN, MU_MAX = mm.MU_MIN, mm.MU_MAX
G_P = mm.G_P
DUTY, X_OFF, G_C, Z_SAG = A["duty"], A["x_off"], A["g_c"], A["z_sag"]
D_STEP, D_STEP_Y = A["d_step"], A["d_step_y"]
PHASE = np.asarray(cpg_max.PHASE_WALK_LS)
OMEGA_MIN, OMEGA_MAX = 0.0, 2.0
SWAY_MAX, SWAY_SLEW = 0.060, 0.004
ACT_DIM, OBS_DIM = obs_max.ACT_DIM, obs_max.OBS_DIM            # v2/v2.1：14 / 70
ACT_DIM_NOMUX, OBS_DIM_NOMUX = obs_max.ACT_DIM_NOMUX, obs_max.OBS_DIM_NOMUX   # v2.2：10 / 66
assert (ACT_DIM, OBS_DIM, ACT_DIM_NOMUX, OBS_DIM_NOMUX) == (14, 70, 10, 66)
LAYOUT_DIMS = {"full": ACT_DIM, "nomux": ACT_DIM_NOMUX}

HOME12_np = np.array(mm.HOME12)
KNEE_SIGN_np = leg_kin.knee_sign_of(mm.HOME)
F0S_np = leg_kin.home_foot(mm.HOME)
LEG_QPOS_IDX = jnp.array(mm.LEG_QPOS_IDX)
LEG_QVEL_IDX = jnp.array(mm.LEG_QVEL_IDX)
LEG_ACT_IDX = jnp.array(mm.LEG_ACT_IDX)
WHEEL_ACT_IDX = jnp.array(mm.WHEEL_ACT_IDX)
KP_NOM = np.tile(np.asarray(mm.KP3_A), 4)          # [60,250,250]×4，MJCF 序
ABAD12 = jnp.array([0, 3, 6, 9])                   # 12 維腿關節裡的 ABAD 位置
FRONT = jnp.array([0, 1])                          # FR, FL
REAR = jnp.array([2, 3])                           # RR, RL
RIGHT = jnp.array([0, 2])                          # FR, RR
LEFT = jnp.array([1, 3])                           # FL, RL

# ---- 隨機化（H 文件 §5 的實測值）----
PUSH_EVERY, PUSH_VEL = 100, 0.6
IMU_TILT_DEG = 3.0            # IMU 安裝偏轉（實測 pitch −1.24° / roll +0.63°）
GYRO_BIAS = 0.02              # rad/s（實測 y 軸 −0.013）
DELAY_BASE = 1                # 固定 1 步（實測伺服多落後 15 ms）＋ 抽 0/1
NOISE_GRAV, NOISE_GYRO, NOISE_QPOS = 0.005, 0.002, 0.001
NOISE_QVEL = jnp.array([0.2, 0.2, 0.4] * 4)        # ABAD/HIP 0.2、KNEE 0.4 rad/s（步態工況實測）

# ---- 終止／護欄 ----
FALL_GRAV_Z = mm.FALL_GRAV_Z
MIN_HEIGHT = 0.29
TAU_BAR, ERR_BAR = 58.0, 0.45          # 實機門檻 70 N·m ÷ 1.2；M9 中止 0.6 − kp250 設計落後餘裕
TAU_KILL, KILL_STEPS = 90.0, 3
NOMINAL_HEIGHT = mm.NOMINAL_HEIGHT_WALK

# ---- reward 權重 ----
# ★ preset 機制：舊 notebook 用 "v2"（行為不變），新 notebook 用 "v2.1"。
#   每一項在基準動作上的實際值由 metrics 的 t_* 欄位吐出來（diag/rl_calibrate.py 印佔比），
#   權重是**量出來校準的**，不是猜的 —— v2 的教訓：W_ROLL=20 在 roll 2.3° 時每步只扣 0.03 分
#   （正項的 1%），policy 完全有理由忽略它，23M 步 roll 只降 12%。
W_DEFAULT = dict(
    W_ROLL=20.0, W_ROLLRATE=0.05, W_PITCH=20.0, W_PITCHRATE=0.05,
    W_EXEC=1.0, W_SYM=5.0, EXEC_SIGMA=0.03,
    W_YAW=1.0, W_VX=1.5, W_VY=0.5, W_H=0.5, W_CLR=1.5,
    W_VZ=2.0, W_OMEGA_VAR=0.5, W_ACT=0.01, W_TAU=3e-5,
    W_TAUBAR=0.01, W_ERRBAR=20.0,
    YAW_SIG2=0.05, YAW_EMA=0.0,      # 偏航 reward 的核寬；YAW_EMA=0 → 用瞬時 wz（v2）
    VX_SIG2=0.02, CMD_VX=(0.05, 0.35),  # 速度核寬與指令範圍
    W_YAW_INST=0.0, YAW_INST_SIG2=0.05,  # 瞬時偏航核（管振盪；EMA 核管漂移）。v2 的 W_YAW 就是這個
    ACT_LAYOUT="full",               # "full" 14 維 / "nomux" 10 維（mu_x 固定＝基準）
    EXEC_MODE="track",               # "track" 擺動相 x 追蹤 / "rate" 真執行率（擺動結束時結算）
    EXEC_RATE_SIG=0.35,              # rate 模式：exp(−((rate−1)/σ)²)
    RAMP_STEPS=0,                    # 起步淡入：前 N 步 policy 動作從基準線性混入（M9 上機有 3 s GAIT_IN）
)
PRESETS = {
    "v2": {},
    # v2.1（2026-09-08，diag/rl_calibrate.py 量出來的）：基準動作上各項佔正項的比例
    #   v2   ：roll+rollrate 2.7%、pitch 1.5%、exec 30%  → policy 忽略姿態（23M 步 roll 只降 12%）
    #   v2.1 ：roll+rollrate ~16%、pitch ~6%、exec ~38%  → 與使用者優先序（姿態 > 執行率）一致
    #   exec σ 放寬到 50 mm：A 步態擺動相 x 誤差本來就 ~30 mm，σ=30 在基準上只給 0.32 分。
    #   ★ 偏航：v2 權重跑出 60 s +48°（0.8°/s），但 exp(−(wz−cmd)²/0.05) 對 0.014 rad/s 只掉 0.4%
    #     —— reward 看不見漂移，而瞬時 wz 又被 1.4 Hz 步態振盪（±0.5 rad/s）淹沒。
    #     改成 4 s EMA 濾過的 wz（YAW_EMA 0.005 @50 Hz；量過：1 s 只衰減到 ±0.07、4 s 到 ±0.018）
    #     ＋ 核寬 σ=1.3°/s（YAW_SIG2 0.0005）：基準 r_yaw ≈ 0.60，漂 0.8°/s 掉到 ≈ 0.40。
    "v2.1": dict(W_ROLL=150.0, W_ROLLRATE=0.5, W_PITCH=100.0, W_PITCHRATE=0.3,
                 W_EXEC=1.5, EXEC_SIGMA=0.05,
                 W_YAW=1.5, YAW_SIG2=0.0005, YAW_EMA=0.005),
    # v2.2（2026-09-08 晚，spec 附錄 A）：mu_x 固定（動作 10 維）、指令 0.15–0.40、速度核放軟、
    #   執行率改量真的（擺動結束結算：實際前跨/指令前跨，基準 前 0.88/後 1.47）、
    #   對稱＝前後執行率差²、姿態只溫和罰（roll 角/率各 ~5%、pitch ~3%，文獻只罰角速度）。
    #   偏航兩個核都要（v2.1 訓練中發現）：EMA 核抓漂移、瞬時軟核抓 1.4 Hz 甩頭振盪
    #   （v2 的瞬時核把 yawerr 0.76→0.10，v2.1 只剩 EMA 核就卡在 0.51）。
    "v2.2": dict(ACT_LAYOUT="nomux", EXEC_MODE="rate", CMD_VX=(0.15, 0.40), VX_SIG2=0.1,
                 W_ROLL=75.0, W_ROLLRATE=0.5, W_PITCH=100.0, W_PITCHRATE=0.3,
                 W_EXEC=1.5, W_SYM=1.0,
                 W_YAW=1.5, YAW_SIG2=0.0005, YAW_EMA=0.005,
                 W_YAW_INST=0.5, YAW_INST_SIG2=0.05),
}
# v2.3（2026-09-08 夜）：v2.2 步態使用者滿意、動作空間不動。三件小事：
#   ① 驗收工具改用實機延遲（動作 1 步 + joint_vel 1 步）—— v2.2 的「偏航 +65°」是零延遲驗收的假象
#      （env 條件下 30 s 只漂 +1.4°）；② 起步淡入 1 s（解 0.22 s 那個 89 N·m 扭動）；
#   ③ W_ACT 0.01→0.05、W_TAUBAR 0.01→0.05（穩態 τ_pk 65 ×1.2 = 78 超門檻 70，動作抖 Σ(Δa)² 0.8/步）。
PRESETS["v2.3"] = dict(PRESETS["v2.2"], W_ACT=0.05, W_TAUBAR=0.05, RAMP_STEPS=50)

# 校準帶（diag/rl_calibrate.py 與 notebook 校準格都用它斷言；佔正項的比例）
CAL_BANDS = {
    "v2.1": dict(roll=(0.10, 0.25), pitch=(0.03, 0.15), exec_max=0.50),
    "v2.2": dict(roll=(0.06, 0.16), pitch=(0.02, 0.10), exec_max=0.45),
    "v2.3": dict(roll=(0.06, 0.16), pitch=(0.02, 0.10), exec_max=0.45),
}
# 模組層級常數 = v2（舊 notebook / 舊測試引用）
W_ROLL, W_ROLLRATE, W_PITCH, W_PITCHRATE = 20.0, 0.05, 20.0, 0.05
W_EXEC, W_SYM, EXEC_SIGMA = 1.0, 5.0, 0.03
W_YAW, W_VX, W_VY, W_H, W_CLR = 1.0, 1.5, 0.5, 0.5, 1.5
W_VZ, W_OMEGA_VAR, W_ACT, W_TAU = 2.0, 0.5, 0.01, 3e-5
W_TAUBAR, W_ERRBAR = 0.01, 20.0
SYM_EMA = 0.05
TERM_KEYS = ("t_pos", "t_vx", "t_yaw", "t_yawi", "t_exec", "t_roll", "t_rollrate", "t_pitch", "t_pitchrate",
             "t_sym", "t_vz", "t_omvar", "t_act", "t_tau", "t_taubar", "t_errbar")
METRIC_KEYS = ("height", "vx", "reward", "pitch", "roll", "clr", "vz", "yawerr", "vxerr",
               "exec_f", "exec_r", "tau_pk", "err_pk", "sway_x", "sway_y") + TERM_KEYS


def yaw_reward(wz_f, cmd_wz, sig2):
    """偏航 reward 核：exp(−(wz_f − cmd)²/sig2)。wz_f 可為瞬時或 EMA 濾過的角速度。"""
    return jnp.exp(-(wz_f - cmd_wz) ** 2 / sig2)


def weights_of(preset: str) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"沒有這個 preset：{preset!r}（有 {list(PRESETS)}）")
    return {**W_DEFAULT, **PRESETS[preset]}


# ---------------------------------------------------------------- 四元數（wxyz）
def _qinv(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    u = q[1:4]
    t = 2.0 * jnp.cross(u, v)
    return v + q[0] * t + jnp.cross(u, t)


def w2b(quat, v):
    return _qrot(_qinv(quat), v)


def _quat_rp(roll, pitch):
    """roll/pitch（rad，yaw=0）→ wxyz。與 m9_rec.rp_to_quat_wxyz 同式。"""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    return jnp.array([cr * cp, sr * cp, cr * sp, -sr * sp])


# ---------------------------------------------------------------- CPG（與 cpg_max 逐行同）
PHASE_j = jnp.array(PHASE)
PHI = PHASE_j[None, :] - PHASE_j[:, None]


def cpg_init():
    return {"rx": jnp.full(4, 1.5), "rx_d": jnp.zeros(4),
            "ry": jnp.full(4, 1.5), "ry_d": jnp.zeros(4), "theta": PHASE_j}


def cpg_step(c, mux, muy, omega, dt):
    rx, rxd, ry, ryd, th = c["rx"], c["rx_d"], c["ry"], c["ry_d"], c["theta"]
    h = dt / N_CPG_SUB
    for _ in range(N_CPG_SUB):
        rxd = rxd + A_CONV * (A_CONV / 4.0 * (mux - rx) - rxd) * h
        rx = rx + rxd * h
        ryd = ryd + A_CONV * (A_CONV / 4.0 * (muy - ry) - ryd) * h
        ry = ry + ryd * h
        rbar = 0.5 * (rx + ry)
        diff = th[None, :] - th[:, None] - PHI
        th = th + (2 * jnp.pi * omega
                   + W_COUP * jnp.sum(rbar[None, :] * jnp.sin(diff), 1)) * h
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": jnp.mod(th, 2 * jnp.pi)}


def duty_remap(th, duty):
    ph = jnp.mod(th, 2 * jnp.pi) / (2 * jnp.pi)
    sw = 1.0 - duty
    return jnp.where(ph < sw, jnp.pi * ph / sw, jnp.pi + jnp.pi * (ph - sw) / duty)


# ---------------------------------------------------------------- 動作
def act_to_cmd(a, layout: str = "full"):
    """動作 → (mux(4), muy(4), ω(4), sway_target(2))。
    layout "full"：14 維 (mux,muy,ω)×4 + sway；"nomux"：10 維 (muy,ω)×4 + sway，mux 固定＝基準 1.8。"""
    a = jnp.tanh(jnp.asarray(a))
    if layout == "full":
        leg = a[:12].reshape(4, 3)
        mux = (leg[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
        muy = (leg[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
        om = (leg[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
        return mux, muy, om, a[12:14] * SWAY_MAX
    if layout == "nomux":
        leg = a[:8].reshape(4, 2)
        mux = jnp.full(4, A["mu_x"])
        muy = (leg[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
        om = (leg[:, 1] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
        return mux, muy, om, a[8:10] * SWAY_MAX
    raise ValueError(f"layout {layout!r}")


def slew_sway(prev, tgt):
    """sway 斜率限制：每控制步最多變 SWAY_SLEW（承重腿不會被瞬間拉走）。"""
    return prev + jnp.clip(tgt - prev, -SWAY_SLEW, SWAY_SLEW)


def baseline_action(layout: str = "full") -> np.ndarray:
    """A 步態對應的固定動作（sway=0）。G0 的標準答案。"""
    def inv(u, lo, hi):
        return float(np.arctanh(np.clip(2 * (u - lo) / (hi - lo) - 1, -0.999, 0.999)))
    if layout == "full":
        return np.array([inv(A["mu_x"], MU_MIN, MU_MAX), inv(A["mu_y"], MU_MIN, MU_MAX),
                         inv(A["omega"], OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])
    if layout == "nomux":
        return np.array([inv(A["mu_y"], MU_MIN, MU_MAX),
                         inv(A["omega"], OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])
    raise ValueError(f"layout {layout!r}")


# ---------------------------------------------------------------- 運動學（與 leg_kin 逐行同）
SIDE_X_j, SIDE_Y_j = jnp.array(mm.SIDE_X), jnp.array(mm.SIDE_Y)
L_T, L_S = mm.L_THIGH, mm.L_SHANK
A2H_X, A2F_Y = mm.ABAD_TO_HIP_X, mm.ABAD_TO_FOOT_Y
REACH_LO, REACH_HI = abs(L_T - L_S) + 1e-6, L_T + L_S - 1e-6
F0S_j, KNEE_SIGN_j = jnp.array(F0S_np), jnp.array(KNEE_SIGN_np)


def ik_j(k, p, knee_sign):
    yp = SIDE_Y_j[k] * A2F_Y
    px, py, pz = p[0], p[1], p[2]
    zp = -jnp.sqrt(jnp.maximum(py * py + pz * pz - yp * yp, 0.0))
    q1 = jnp.arctan2(yp * pz - zp * py, yp * py + zp * pz)
    xp = px - SIDE_X_j[k] * A2H_X
    r = jnp.sqrt(xp * xp + zp * zp)
    r_new = jnp.clip(r, REACH_LO, REACH_HI)
    scale = jnp.where(r > 1e-12, r_new / r, 1.0)
    xp, zp = xp * scale, zp * scale
    cos_q3 = (r_new * r_new - L_T ** 2 - L_S ** 2) / (2 * L_T * L_S)
    q3 = jnp.sign(knee_sign) * jnp.arccos(jnp.clip(cos_q3, -1.0, 1.0))
    a = -(L_T + L_S * jnp.cos(q3))
    b = -L_S * jnp.sin(q3)
    q2 = jnp.arctan2(a * xp - b * zp, a * zp + b * xp)
    return jnp.array([q1, q2, q3])


def fk_j(k, q3):
    """第 k 腿：關節角 (3,) → 輪軸心相對 ABAD 原點 (3,)。"""
    q1, q2, q3_ = q3[0], q3[1], q3[2]
    xp = SIDE_X_j[k] * A2H_X - L_T * jnp.sin(q2) - L_S * jnp.sin(q2 + q3_)
    zp = -L_T * jnp.cos(q2) - L_S * jnp.cos(q2 + q3_)
    yp = SIDE_Y_j[k] * A2F_Y
    c, s = jnp.cos(q1), jnp.sin(q1)
    return jnp.array([xp, yp * c - zp * s, yp * s + zp * c])


def foot_targets_j(c, sway):
    """CPG 狀態 + sway(2,) → (4,3) 足端目標（相對各腿 ABAD 原點）。與 cpg_max.foot_targets 同式。"""
    th = duty_remap(c["theta"], DUTY)
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    dx = -D_STEP * fx * jnp.cos(th) + X_OFF + sway[0]
    dy = D_STEP_Y * fy * jnp.cos(th) + sway[1]
    dz = jnp.where(jnp.sin(th) > 0, (G_C + Z_SAG) * jnp.sin(th), G_P * jnp.sin(th))
    return F0S_j + jnp.stack([dx, dy, dz], -1)


def joint_targets_j(c, sway):
    tgt = foot_targets_j(c, sway)
    return jnp.stack([ik_j(k, tgt[k], KNEE_SIGN_j[k]) for k in range(4)]).reshape(12)


def foot_actual_j(q12):
    q = q12.reshape(4, 3)
    return jnp.stack([fk_j(k, q[k]) for k in range(4)])


def swing_mask_j(theta):
    return (jnp.sin(duty_remap(theta, DUTY)) > 0).astype(jnp.float32)


# ---------------------------------------------------------------- 護欄
def tau_barrier(tau12):
    return jnp.sum(jnp.maximum(jnp.abs(tau12) - TAU_BAR, 0.0) ** 2)


def err_barrier(err12):
    return jnp.sum(jnp.maximum(jnp.abs(err12) - ERR_BAR, 0.0) ** 2)


# ---------------------------------------------------------------- env
class MaxCpgEnv(Env):
    def __init__(self, scene: str = mm.SCENE_MJX_KP250, preset: str = "v2"):
        self.preset = preset
        self.w = weights_of(preset)
        self.layout = self.w["ACT_LAYOUT"]
        self.act_dim = LAYOUT_DIMS[self.layout]
        self.obs_dim = obs_max.obs_dim(self.act_dim)
        self._base_act = jnp.array(baseline_action(self.layout))
        self._ramp = int(self.w["RAMP_STEPS"])
        m = mujoco.MjModel.from_xml_path(scene)
        assert m.opt.timestep == SIM_DT, f"timestep {m.opt.timestep} ≠ {SIM_DT}"
        assert m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE, \
            "致動器不是位置伺服，ctrl 會被當力矩施加"
        self._mj = m
        self.sys = mjx.put_model(m)
        self._lo = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 0])
        self._hi = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 1])
        gids = [mm._id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{mm.PREFIX[l]}_FOOT_LINK_COLL")
                for l in mm.LEGS]
        self._wheel_gids = jnp.array(gids)
        self._wheel_r = float(m.geom_size[gids[0]][0])
        self._init_q = self._settled_qpos(m)

    def _settled_qpos(self, m):
        """CPU MuJoCo 先站定 1.5 秒，把落地後的 qpos 當 reset 起點（位置伺服有靜態撓度）。"""
        d = mujoco.MjData(m)
        q = cpg_max.stand_targets(KNEE_SIGN_np, F0S_np, X_OFF)
        d.qpos[:3] = [0.0, 0.0, mm.NOMINAL_HEIGHT_KIN + 0.005]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[mm.LEG_QPOS_IDX] = q
        mujoco.mj_forward(m, d)
        for _ in range(int(1.5 / SIM_DT)):
            d.ctrl[mm.LEG_ACT_IDX] = q
            d.ctrl[mm.WHEEL_ACT_IDX] = 0.0
            mujoco.mj_step(m, d)
        assert 0.40 < d.qpos[2] < 0.52, f"站定高度 {d.qpos[2]:.3f} m 不合理"
        return jnp.array(d.qpos)

    def _ctrl(self, q_des):
        return jnp.zeros(16).at[LEG_ACT_IDX].set(q_des).at[WHEEL_ACT_IDX].set(0.0)

    def _wheel_clearance(self, data):
        return data.geom_xpos[self._wheel_gids, 2] - self._wheel_r

    def _obs(self, data, c, cmd, last_a, info):
        """70 維，含 IMU 偏轉、gyro 偏置、joint_vel 延遲。雜訊在 step 裡加。
        ⚠️ 欄位順序必須與 obs_max.OBS_LAYOUT 逐項一致 —— 本機推論端用的是那一份。"""
        qm = info["imu_q"]
        grav = _qrot(qm, w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0])))
        gyro = _qrot(qm, data.qvel[3:6]) + info["gyro_bias"]
        return jnp.concatenate([
            grav, gyro,
            data.qpos[LEG_QPOS_IDX] - jnp.array(HOME12_np),
            info["qvel_prev"],                    # ★ joint_vel 延 1 步（driver 濾波實測）
            cmd, last_a,
            c["rx"], c["rx_d"], c["ry"], c["ry_d"], jnp.sin(c["theta"]), jnp.cos(c["theta"]),
        ])

    def reset(self, rng):
        k_cmd, k_bias, k_delay, k_tilt, k_noise = jax.random.split(rng, 5)
        data = mjx.make_data(self.sys).replace(qpos=self._init_q)
        data = data.replace(ctrl=self._ctrl(self._init_q[LEG_QPOS_IDX]))
        data = mjx.forward(self.sys, data)
        k_vx, k_wz, k_zero = jax.random.split(k_cmd, 3)
        vx = jax.random.uniform(k_vx, minval=self.w["CMD_VX"][0], maxval=self.w["CMD_VX"][1])
        wz = jnp.where(jax.random.uniform(k_zero) < 0.6, 0.0,
                       jax.random.uniform(k_wz, minval=-0.4, maxval=0.4))
        cmd = jnp.array([vx, wz])
        tilt = jax.random.uniform(k_tilt, (2,), minval=-IMU_TILT_DEG,
                                  maxval=IMU_TILT_DEG) * jnp.pi / 180
        c = cpg_init()
        z14 = jnp.zeros(self.act_dim)
        info = {"rng": k_noise, "c": c, "cmd": cmd,
                "gyro_bias": jax.random.uniform(k_bias, (3,), minval=-GYRO_BIAS, maxval=GYRO_BIAS),
                "imu_q": _quat_rp(tilt[0], tilt[1]),
                "delay": DELAY_BASE + jax.random.bernoulli(k_delay, 0.5).astype(jnp.int32),
                # 延遲佇列以**基準動作**初始化：episode 開頭等於開迴路步態（零動作 = mux 1.5 不跨步）
                "a_hist": jnp.tile(self._base_act[None], (3, 1)), "last_a": z14,
                "sway": jnp.zeros(2), "qvel_prev": data.qvel[LEG_QVEL_IDX],
                "ema_f": jnp.zeros(()), "ema_r": jnp.zeros(()),
                "wz_ema": jnp.zeros(()),
                # 真執行率（EXEC_MODE="rate"）：擺動開始時記足端 x 與目標 x，結束時結算
                "sw_prev": jnp.zeros(4, bool), "sw_x0": jnp.zeros(4), "sw_t0": jnp.zeros(4),
                "rate_last": jnp.ones(4),
                "kill": jnp.zeros((), jnp.int32), "step": 0}
        obs = self._obs(data, c, cmd, z14, info)
        z = jnp.zeros(())
        metrics = {k: z for k in METRIC_KEYS}
        metrics["height"] = data.qpos[2]
        return State(data, obs, z, z, metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        a_hist = jnp.concatenate([action[None], info["a_hist"][:2]], 0)   # [t, t−1, t−2]
        act = a_hist[info["delay"]]                                         # ★ 動作延遲
        if self._ramp > 0:                                                  # ★ 起步淡入
            u = jnp.clip(info["step"] / self._ramp, 0.0, 1.0)
            act = self._base_act + u * (act - self._base_act)
        mux, muy, om, sway_tgt = act_to_cmd(act, self.layout)
        sway = slew_sway(info["sway"], sway_tgt)
        c = cpg_step(info["c"], mux, muy, om, CTRL_DT)
        q_des = jnp.clip(joint_targets_j(c, sway), self._lo, self._hi)

        data = state.pipeline_state
        rng, k_push, k_dir, k_obs = jax.random.split(info["rng"], 4)
        do_push = (info["step"] % PUSH_EVERY) == (PUSH_EVERY - 1)
        ang = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jnp.pi)
        mag = jax.random.uniform(k_push, minval=0.0, maxval=PUSH_VEL)
        kick = jnp.where(do_push, jnp.array([mag * jnp.cos(ang), mag * jnp.sin(ang), 0.0]),
                         jnp.zeros(3))
        data = data.replace(qvel=data.qvel.at[0:3].add(kick))
        ctrl = self._ctrl(q_des)

        def one(carry, _):
            d, tpk = carry
            d = mjx.step(self.sys, d.replace(ctrl=ctrl))
            return (d, jnp.maximum(tpk, jnp.abs(d.actuator_force[LEG_ACT_IDX]))), None
        (data, tau_pk12), _ = jax.lax.scan(one, (data, jnp.zeros(12)), None, length=N_FRAMES)

        grav = w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        vb = w2b(data.qpos[3:7], data.qvel[0:3])      # reward 用真值速度（只在模擬）
        cmd = info["cmd"]
        wz = data.qvel[5]
        q12 = data.qpos[LEG_QPOS_IDX]
        err12 = q_des - q12

        # ---- 執行率：擺動相足端 x（機身系、相對 ABAD）實際 vs 指令 ----
        tgt = foot_targets_j(c, sway)
        act_ft = foot_actual_j(q12)
        sw = swing_mask_j(c["theta"])
        dx = act_ft[:, 0] - tgt[:, 0]
        w = self.w
        track = jnp.exp(-(dx / w["EXEC_SIGMA"]) ** 2)
        n_f, n_r = jnp.sum(sw[FRONT]), jnp.sum(sw[REAR])
        # ---- 真執行率：擺動開始記起點，結束時 (實際前跨)/(指令前跨)，保持到下一次擺動 ----
        # ★ 實際前跨用**世界系**足端 x 減機身 x（與 cpg_walk_max.Trace 同一個量）。
        #   機身系 FK 位移會把 pitch 擺動（±2° × 0.45 m 腿長 ≈ 30 mm）算成假的前跨。
        fx_rel = data.geom_xpos[self._wheel_gids, 0] - data.qpos[0]
        sw_b = sw > 0.5
        start = sw_b & ~info["sw_prev"]
        end = ~sw_b & info["sw_prev"]
        sw_x0 = jnp.where(start, fx_rel, info["sw_x0"])
        sw_t0 = jnp.where(start, tgt[:, 0], info["sw_t0"])
        d_cmd = tgt[:, 0] - sw_t0
        rate_new = (fx_rel - sw_x0) / jnp.where(jnp.abs(d_cmd) > 1e-3, d_cmd, 1e-3)
        rate_last = jnp.where(end & (jnp.abs(d_cmd) > 1e-3), rate_new, info["rate_last"])
        rate_f, rate_r = jnp.mean(rate_last[FRONT]), jnp.mean(rate_last[REAR])
        rate_l, rate_rt = jnp.mean(rate_last[LEFT]), jnp.mean(rate_last[RIGHT])
        if w["EXEC_MODE"] == "rate":
            r_exec = jnp.mean(jnp.exp(-((rate_last - 1.0) / w["EXEC_RATE_SIG"]) ** 2))
            # ★ 前後 + 左右都罰：v2.1 兩趟證明漂移的機制是**左右**步幅差
            #   （FL 35 / FR 61、RL 59 / RR 142 mm），mu_y 逐腿仍能做出這種不對稱。
            sym_pen = (rate_f - rate_r) ** 2 + (rate_l - rate_rt) ** 2
            ema_f, ema_r = info["ema_f"], info["ema_r"]
        else:
            r_exec = jnp.sum(sw * track) / jnp.maximum(jnp.sum(sw), 1.0)
            ef = jnp.sum(sw[FRONT] * jnp.abs(dx[FRONT])) / jnp.maximum(n_f, 1.0)
            er = jnp.sum(sw[REAR] * jnp.abs(dx[REAR])) / jnp.maximum(n_r, 1.0)
            ema_f = jnp.where(n_f > 0, (1 - SYM_EMA) * info["ema_f"] + SYM_EMA * ef, info["ema_f"])
            ema_r = jnp.where(n_r > 0, (1 - SYM_EMA) * info["ema_r"] + SYM_EMA * er, info["ema_r"])
            sym_pen = (ema_f - ema_r) ** 2

        r_vx = jnp.exp(-(vb[0] - cmd[0]) ** 2 / w["VX_SIG2"])
        r_vy = jnp.exp(-vb[1] ** 2 / 0.02)
        wz_ema = info["wz_ema"] + w["YAW_EMA"] * (wz - info["wz_ema"])
        r_yaw = yaw_reward(wz_ema if w["YAW_EMA"] > 0 else wz, cmd[1], w["YAW_SIG2"])
        r_yawi = yaw_reward(wz, cmd[1], w["YAW_INST_SIG2"])          # 瞬時（振盪）
        r_h = jnp.exp(-400.0 * (data.qpos[2] - NOMINAL_HEIGHT) ** 2)
        r_clr = jnp.mean(sw * jnp.clip(self._wheel_clearance(data) / G_C, 0.0, 1.0))
        c_act = jnp.sum((action - info["last_a"]) ** 2)
        c_tau = jnp.sum(data.actuator_force ** 2)
        # 每一項各自帶權重存起來：校準（佔比）與訓練監看都靠這些
        T = {
            "t_vx": w["W_VX"] * r_vx, "t_yaw": w["W_YAW"] * r_yaw,
            "t_yawi": w["W_YAW_INST"] * r_yawi, "t_exec": w["W_EXEC"] * r_exec,
            "t_roll": w["W_ROLL"] * grav[1] ** 2, "t_rollrate": w["W_ROLLRATE"] * data.qvel[3] ** 2,
            "t_pitch": w["W_PITCH"] * grav[0] ** 2, "t_pitchrate": w["W_PITCHRATE"] * data.qvel[4] ** 2,
            "t_sym": w["W_SYM"] * sym_pen,
            "t_vz": w["W_VZ"] * data.qvel[2] ** 2, "t_omvar": w["W_OMEGA_VAR"] * jnp.var(om),
            "t_act": w["W_ACT"] * c_act, "t_tau": w["W_TAU"] * c_tau,
            "t_taubar": w["W_TAUBAR"] * tau_barrier(tau_pk12),
            "t_errbar": w["W_ERRBAR"] * err_barrier(err12),
        }
        T["t_pos"] = (T["t_vx"] + w["W_VY"] * r_vy + T["t_yaw"] + T["t_yawi"] + w["W_H"] * r_h
                      + w["W_CLR"] * r_clr + T["t_exec"])
        reward = (T["t_pos"]
                  - T["t_sym"] - T["t_roll"] - T["t_rollrate"] - T["t_pitch"] - T["t_pitchrate"]
                  - T["t_omvar"] - T["t_vz"] - T["t_act"] - T["t_tau"]
                  - T["t_taubar"] - T["t_errbar"])

        kill = jnp.where(jnp.max(tau_pk12) > TAU_KILL, info["kill"] + 1, 0)
        fell = grav[2] > FALL_GRAV_Z
        too_low = data.qpos[2] < MIN_HEIGHT
        done = jnp.where(fell | too_low | (kill >= KILL_STEPS), 1.0, 0.0)

        obs = self._obs(data, c, cmd, action, info)
        n = jax.random.normal(k_obs, (self.obs_dim,))
        obs = (obs.at[0:3].add(NOISE_GRAV * n[0:3]).at[3:6].add(NOISE_GYRO * n[3:6])
               .at[6:18].add(NOISE_QPOS * n[6:18]).at[18:30].add(NOISE_QVEL * n[18:30]))

        info.update({"rng": rng, "c": c, "last_a": action, "a_hist": a_hist, "sway": sway,
                     "qvel_prev": data.qvel[LEG_QVEL_IDX], "ema_f": ema_f, "ema_r": ema_r,
                     "wz_ema": wz_ema, "sw_prev": sw_b, "sw_x0": sw_x0, "sw_t0": sw_t0,
                     "rate_last": rate_last, "kill": kill, "step": info["step"] + 1})
        metrics = {
            "height": data.qpos[2], "vx": vb[0], "reward": reward,
            "pitch": jnp.abs(grav[0]) * 57.29578, "roll": jnp.abs(grav[1]) * 57.29578,
            "clr": jnp.mean(self._wheel_clearance(data)) * 1000.0, "vz": jnp.abs(data.qvel[2]),
            "yawerr": jnp.abs(wz - cmd[1]), "vxerr": jnp.abs(vb[0] - cmd[0]),
            "exec_f": (rate_f if w["EXEC_MODE"] == "rate"
                       else jnp.sum(sw[FRONT] * track[FRONT]) / jnp.maximum(n_f, 1.0)),
            "exec_r": (rate_r if w["EXEC_MODE"] == "rate"
                       else jnp.sum(sw[REAR] * track[REAR]) / jnp.maximum(n_r, 1.0)),
            "tau_pk": jnp.max(tau_pk12), "err_pk": jnp.max(jnp.abs(err12)),
            "sway_x": jnp.abs(sway[0]) * 1000.0, "sway_y": jnp.abs(sway[1]) * 1000.0,
            **T,
        }
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=info)

    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return self.act_dim

    @property
    def backend(self):
        return "mjx"


# ---------------------------------------------------------------- domain randomization
_BASE_ID = mm._id(mujoco.MjModel.from_xml_path(mm.SCENE_MJX_KP250),
                  mujoco.mjtObj.mjOBJ_BODY, "base_link")
_LEG_DOF, _WHEEL_DOF = jnp.array(mm.LEG_QVEL_IDX), jnp.array(mm.WHEEL_QVEL_IDX)
_ABAD_DOF = jnp.array(mm.LEG_QVEL_IDX[::3])
_KP_NOM_j = jnp.array(KP_NOM)
_KD_NOM = float(np.asarray(mm.KD3_A)[1])


def domain_randomize(sys, rng):
    """每個 env 抽一組：地面摩擦、kp/kv（ABAD 單獨、範圍更寬）、質量+payload、關節/輪摩擦。"""
    @jax.vmap
    def per_env(rng):
        k = jax.random.split(rng, 9)
        gf = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.4, maxval=1.4))
        s_kp = jax.random.uniform(k[1], minval=0.8, maxval=1.2)
        s_ab = jax.random.uniform(k[2], minval=0.6, maxval=1.4)      # ★ ABAD 範圍加大
        kv = jax.random.uniform(k[3], minval=0.5, maxval=2.0) * _KD_NOM
        kp_leg = _KP_NOM_j * s_kp
        kp_leg = kp_leg.at[ABAD12].set(_KP_NOM_j[ABAD12] * s_ab)
        gain = sys.actuator_gainprm.at[LEG_ACT_IDX, 0].set(kp_leg)
        bias = (sys.actuator_biasprm.at[LEG_ACT_IDX, 1].set(-kp_leg)
                .at[LEG_ACT_IDX, 2].set(-kv))
        bm = sys.body_mass * jax.random.uniform(k[4], (sys.nbody,), minval=0.9, maxval=1.1)
        bm = bm.at[_BASE_ID].add(jax.random.uniform(k[5], minval=0.0, maxval=5.0))
        fl = sys.dof_frictionloss
        fl = fl.at[_LEG_DOF].multiply(jax.random.uniform(k[6], minval=0.5, maxval=1.5))
        fl = fl.at[_ABAD_DOF].set(sys.dof_frictionloss[_ABAD_DOF]
                                  * jax.random.uniform(k[7], minval=0.6, maxval=1.4))
        fl = fl.at[_WHEEL_DOF].set(jax.random.uniform(k[8], minval=0.10, maxval=0.25))
        return gf, gain, bias, bm, fl

    gf, gain, bias, bm, fl = per_env(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.replace(geom_friction=0, actuator_gainprm=0, actuator_biasprm=0,
                              body_mass=0, dof_frictionloss=0)
    sys = sys.replace(geom_friction=gf, actuator_gainprm=gain, actuator_biasprm=bias,
                      body_mass=bm, dof_frictionloss=fl)
    return sys, in_axes
