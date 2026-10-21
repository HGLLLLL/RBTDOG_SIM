"""純 RL 自訓 v1 環境（spec docs/superpowers/specs/2026-10-13-rl-wheel-v1-design.md）。

照同事 Isaac Lab `Wheel-Rough-v0`（RL_only/*_env.yaml、policy_io.json）重做成 Brax Env（MJX 後端）：
obs 53 ＝ ang_vel(3) | projected_gravity(3) | cmd(3) | joint_pos−default(12) | joint_vel(16) | last_action(16)，關節序 PhysX 分組序；
obs 是字典（同他的 env.yaml 非對稱 actor-critic）："state" 53 給 policy（帶雜訊）、"privileged_state" 56 給 critic（同樣 53 無雜訊 ＋ 機身系線速度 3）；
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
PRIV_DIM = OBS_DIM + 3                  # critic 特權 obs：乾淨 53 ＋ base_lin_vel（機身系，3）
JOINT_NAMES = IO["joints"]["names"]
DEFAULT = jnp.array(IO["joints"]["default_pos"], dtype=jnp.float32)
ACT_SCALE = jnp.array(IO["action"]["scale"], dtype=jnp.float32)
ACT_OFFSET = jnp.array(IO["action"]["offset"], dtype=jnp.float32)
NOISE = dict(ang=0.2, grav=0.05, jpos=0.01, jvel=1.5)     # 他的 UniformNoiseCfg（±）
SPAWN_XY, SPAWN_WINDOW = 0.5, 0.6         # 出生點在方塊中心 ±0.5 m（他的 pose_range）；出生高取 ±0.6 m 窗內地形最高
FEET = ("FAR", "FBL", "RAR", "RBL")
# 輪輪實體間隙（改動一的補強）：v3f 碰撞圓柱貼在 FOOT_LINK y = −0.047（輪胎外側面，半寬 0.005），但官方輪網格
# y ∈ [−0.070, +0.017]，**內側面在 +0.017**。兩個左右輪的碰撞圓盤比 link 原點還外側，間距比實體多 ~94 mm，
# 所以 <pair> 只有腿交叉才會碰（攤帳 ww 恆 0）。實機的碰撞（V 文件：link 中心距 67 mm → 實體間隙 25–26 mm）是內側面互碰，
# pair 表示不了 → 另加間隙 hinge：gap ＝ 輪心距 − 2×0.017（與 V 文件同口徑；機身／輪傾斜時高估約 7 mm）。
WHEEL_INNER_W = 0.034                   # 2 × 0.017：兩輪內側面到各自 link 原點的距離和
GAP_MIN = 0.08                          # 實體間隙 < 80 mm 開始罰（他的 10k 在 vy 0.5 ~33 mm）。0.05 時 40 mm 驗收邊界只罰 (1−0.8)² = 4%，太弱；
                                        # 0.08 配 w_gap 2.0：40 mm 時每對 (1−0.5)²×2 = 0.5，接觸（gap 0）時 2.0
LAT_PAIRS = ((0, 1), (2, 3))            # FEET 索引：前 FAR–FBL、後 RAR–RBL
X_LIM, Y_LIM = -wt.X0 - 1.0, -wt.Y0 - 1.0  # 出界終止：hfield 邊緣往內 1 m（離開地圖 _terrain_h 夾到邊緣格、輪子掉出 hfield 後自由落體）
FALL_H = 0.20                           # 摔倒：機身離地 < 0.20 m（站立 0.494；輪半徑 0.096、機身箱底離地 ~0.37；低於 0.20 表示腿已折到機身快碰地）
FALL_COS = -0.5                         # 摔倒：機身系重力 z 分量 > −0.5 ＝ 機身傾 > 60°
# 課程升級：回合位移 > 2 m（他的規則是 > 方塊一半 4 m）。v1 的 level 卡在 ~2.5：10 s 重抽指令、10% 站立回合，
# 20 s 內平均要 > 0.2 m/s 淨位移才升，指令轉向後位移互相抵銷，4 m 太嚴（W 文件 §6、spec §4）。降級規則不變。
CURR_UP_DIST = 2.0
LIM_SOFT = 0.9                          # 關節軟限位：中點 ± 0.9 × 半行程（同 Isaac soft_joint_pos_limit_factor；lim 罰 PD 目標超出它）
WW_CAP = 3.0                            # ww 每個接觸 fn/100 的上限：腿交叉時 fn 可達 2755 N，單步 −760 會拉壞 critic
IMPACT_SETTLE = 25                      # 步（0.5 s）：出生後輪子從 3 mm～數 cm 落到地面的時間，這段的落地不算衝擊事件；boxes 10 cm 自由落體 ≈ 0.14 s，25 步留餘裕
TERM_KEYS = ("lin", "ang", "vz", "angxy", "tau_leg", "tau_wheel", "acc_leg", "acc_wheel", "rate", "slip", "dev", "ww", "wgap", "lim", "roll", "impact", "knee")
METRIC_KEYS = TERM_KEYS + ("reward", "lin_small", "level", "fell", "oob", "tilt", "dist", "ww_force", "gap_min", "vx_err", "vy_err", "wz_err", "touch_n", "small", "standing", "sat_n",
                           "impact_vz", "td_n", "impact_fn", "mir", "delay")
# v1c 預設（spec 2026-10-20 §2）：鏡像增強、roll 姿態角懲罰、落地衝擊懲罰、動作延遲 0–2 步、IMU 常數偏差 ±2°。WheelEnv() 預設＝v1b。
V1C = dict(mirror=True, w_roll=10.0, w_impact=3.0, impact_v0=0.3, delay_max=2, grav_bias_deg=2.0)
# impact_v0（10-20 晚加）：只罰超過 0.3 m/s 的落地速度 → 10k 等級的輕落地（0.10–0.19）免費、v1b 等級（0.43–0.71）仍罰。
# v1c 從頭訓（w 3、無門檻）5 億步後 policy 完全不抬腳、側向軸死掉（lin 0.04 vs v1b 2.40）：早期笨拙落地每次 −3～−12 把探索殺掉。
# V1F（10-21，使用者定）：v1b 熱啟動、**不加延遲 DR**（v1d 平移腿伸直到膝 42°、pitch 晃 3 倍、直走超速 +0.1，嫌疑是延遲 DR）；
#   加 w_angxy 0.5（罰 roll／pitch 角速度：只罰 roll 讓 policy 把重心轉移改成前後晃）、膝 hinge（|q_knee| 比 knee_min_deg 60° 更直的部分；v1b 平移膝 64–80° 不罰）。
V1F = dict(mirror=True, w_roll=10.0, w_impact=3.0, impact_v0=0.3, delay_max=0, grav_bias_deg=2.0, w_angxy=0.5, w_knee=0.5, knee_min_deg=60.0)
# V1G（10-21，使用者定）：v1f 平移＝兩腿踏、兩腿被拖著橫滑 0.5 m/s（直腿外張是拖腿的幾何必然；|vy| 越大拖越多腿）。原因是 slip 0.075 太便宜：
#   拖兩輪每步只扣 0.04，比抬腳（impact）＋重心轉移（roll）便宜。只改 w_slip ×10；impact／膝 hinge 不動、不加 ABAD hinge（原廠平移 ABAD 擺幅 11–20°）。
V1G = dict(V1F, w_slip=0.75)


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


def bias_matrix(roll, pitch):
    """IMU 常數偏差 R ＝ Ry(pitch)·Rx(roll)（與 realbot/rl_obs_rl.grav_fix_matrix 同慣例）；policy 看到 g ＝ Rᵀ·g_true，狗端 --grav-offset 乘回 R 就歸零。"""
    cr, sr, cp, sp = jnp.cos(roll), jnp.sin(roll), jnp.cos(pitch), jnp.sin(pitch)
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return ry @ rx


# ---------------------------------------------------------------- 左右鏡像（v1c 鏡像增強；y → −y）
# 腿 FR↔FL、RR↔RL。四腿關節軸同向（ABAD x 軸、髖／膝／輪 y 軸），y 鏡射下繞 x／z 的量變號、繞 y 的不變：
# ABAD 位置／速度／動作、ωx／ωz、grav_y、cmd vy／wz、機身 vy 變號；髖、膝、輪速、輪動作只換腿。
MIRROR_LR = (1, 0, 3, 2)


def _mirror_maps():
    oi, os_ = np.arange(OBS_DIM), np.ones(OBS_DIM, np.float32)
    ai, as_ = np.arange(ACT_DIM), np.ones(ACT_DIM, np.float32)

    def group4(idx, sg, base, neg):
        for k in range(4):
            idx[base + k] = base + MIRROR_LR[k]
            if neg:
                sg[base + k] = -1.0
    os_[0] = os_[2] = -1.0                      # ωx、ωz
    os_[4] = -1.0                               # grav y
    os_[7] = os_[8] = -1.0                      # cmd vy、wz
    for base, neg in ((9, True), (13, False), (17, False)):                  # jpos：ABAD／HIP／KNEE
        group4(oi, os_, base, neg)
    for base, neg in ((21, True), (25, False), (29, False), (33, False)):    # jvel：＋WHEEL
        group4(oi, os_, base, neg)
    for base, neg in ((37, True), (41, False), (45, False), (49, False)):    # last_action
        group4(oi, os_, base, neg)
    for base, neg in ((0, True), (4, False), (8, False), (12, False)):       # 動作
        group4(ai, as_, base, neg)
    return jnp.array(oi), jnp.array(os_), jnp.array(ai), jnp.array(as_)


OBS_MIRROR_IDX, OBS_MIRROR_SGN, ACT_MIRROR_IDX, ACT_MIRROR_SGN = _mirror_maps()
PRIV_MIRROR_SGN = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float32)      # 機身系線速度 (vx, vy, vz)


def mirror_obs(o):
    return o[OBS_MIRROR_IDX] * OBS_MIRROR_SGN


def mirror_act(a):
    return a[ACT_MIRROR_IDX] * ACT_MIRROR_SGN


def mirror_priv(p):
    return jnp.concatenate([mirror_obs(p[:OBS_DIM]), p[OBS_DIM:] * PRIV_MIRROR_SGN])


def mirror_data(env, data, sys=None):
    """測試用：把 MJX data 真的左右翻（y → −y、四元數 (w,x,y,z) → (w,−x,y,−z)、腿互換、ABAD 變號、ctrl 同樣翻），再 mjx.forward。
    free joint 的 qvel 線速度在世界系（vy 變號）、角速度在機身系（ωx、ωz 變號）。sys：forward 用的模型（預設 env.sys；測試傳對稱化副本）。"""
    q, v = data.qpos, data.qvel
    q = q.at[1].multiply(-1.0).at[4].multiply(-1.0).at[6].multiply(-1.0)
    v = v.at[1].multiply(-1.0).at[3].multiply(-1.0).at[5].multiply(-1.0)
    QA, VA = env.QADR, env.VADR
    jq, jv = data.qpos[QA], data.qvel[VA]
    sgn16 = jnp.concatenate([-jnp.ones(4), jnp.ones(12)])
    idx16 = jnp.array([g * 4 + MIRROR_LR[k] for g in range(4) for k in range(4)])
    q = q.at[QA].set(jq[idx16] * sgn16); v = v.at[VA].set(jv[idx16] * sgn16)
    c = data.ctrl.at[env.AID].set(data.ctrl[env.AID][idx16] * sgn16)        # PD 目標／輪速也翻（ABAD offset 0，變號合法）
    return mjx.forward(env.sys if sys is None else sys, data.replace(qpos=q, qvel=v, ctrl=c))


class WheelEnv(Env):
    def __init__(self, scene: str | None = None, map_seed: int = 0, flat: bool = False, noise: bool = True, push: bool = True,
                 fixed_cmd: bool = False, fixed_tile: tuple[int, int] | None = None, small_frac: float = 0.3,
                 std: float = 0.25, w_ww: float = 1.0, w_gap: float = 2.0, episode_len: int = EPISODE_LEN,
                 w_ang: float = 2.5, w_lim: float = 2.0,
                 mirror: bool = False, w_roll: float = 0.0, w_impact: float = 0.0, impact_v0: float = 0.0, delay_max: int = 0, grav_bias_deg: float = 0.0,
                 w_angxy: float = 0.05, w_knee: float = 0.0, knee_min_deg: float = 60.0, w_slip: float = 0.075):
        """flat：全圖平地、出生在原點（對齊驗證）；fixed_cmd：指令讀 info["cmd_fix"]（不重抽、不站立、不小指令）；
        fixed_tile=(kind_idx, level)：固定出生方塊、不升降課程。std／w_ww／w_gap／small_frac 是三項改動的旋鈕（w_gap：輪輪實體間隙 hinge）。
        w_ang：偏航追蹤權重（v1 是 1.25，v1b 2.5：原地轉時 tau_leg ≈ −2／步壓過 ang 最多 1.25）；w_lim：腿關節 PD 目標超出軟限位的懲罰（v1 是 0，v1b 2.0）。
        episode_len：回合步數（內部重生、10 s 指令重抽、推力排程都以它為準；訓練時須與 wrap_env 的 episode_length 相同）。
        v1c（全部預設關，WheelEnv() ＝ v1b）：mirror 每回合 50% 左右鏡像；w_roll 罰真實重力 y 分量平方（只罰 roll）；w_impact 罰觸地前一步足端下落速度平方；
        delay_max 每回合抽 0–delay_max 步動作延遲；grav_bias_deg 每回合抽 ±deg 的 IMU roll／pitch 常數偏差（只動 policy 的重力 obs）。fixed_cmd 下三個隨機項全關。"""
        self.flat, self.noise, self.push, self.fixed_cmd, self.fixed_tile = flat, noise, push, fixed_cmd, fixed_tile
        self.small_frac, self.std, self.w_ww, self.w_gap = small_frac, std, w_ww, w_gap
        self.w_ang, self.w_lim = float(w_ang), float(w_lim)
        self.mirror, self.w_roll, self.w_impact, self.impact_v0 = bool(mirror), float(w_roll), float(w_impact), float(impact_v0)
        self.w_angxy, self.w_knee, self.knee_min = float(w_angxy), float(w_knee), float(np.radians(knee_min_deg))
        self.w_slip = float(w_slip)
        self.delay_max, self.grav_bias_deg = int(delay_max), float(grav_bias_deg)
        assert 0 <= self.delay_max <= 2, "a_hist 長度 3：延遲最多 2 步"
        self.episode_len = int(episode_len)
        self.scene = scene or wt.make_scene_xml()
        m = mujoco.MjModel.from_xml_path(self.scene)
        assert m.opt.timestep == SIM_DT, m.opt.timestep     # 場景 <option timestep="0.005">（wheel_terrain.make_scene_xml）
        self.H_np = wt.build_map(map_seed, flat=flat)
        wt.load_hfield(m, self.H_np)
        # _contacts 假設金字塔摩擦錐（cone 0）＋ condim 3：每個接觸 4 列 efc、法向力＝4 列和。
        # 換 elliptic 錐或改 condim（含 <pair>）時，_contacts 的法向力解碼就要跟著改，否則讀值錯。
        assert m.opt.cone == 0 and bool((m.geom_condim == 3).all()) and (m.npair == 0 or bool((m.pair_dim == 3).all())), \
            (m.opt.cone, np.unique(m.geom_condim), m.pair_dim)
        # _contacts 用 dist < 0 判接觸（忽略 margin）；有人改 geom／pair margin 時讀值會靜默出錯，所以在這裡擋
        assert bool((m.geom_margin == 0).all()) and (m.npair == 0 or bool((m.pair_margin == 0).all())), \
            (np.unique(m.geom_margin), m.pair_margin)
        # 只有四個輪子碰地形（wheel_terrain.make_robot_xml 把機身與 8 段腿設 contype 0）：MJX hfield 每個幾何測
        # (ceil(2r/0.1)+1)²×2 個稜柱，機身＋腿 498 個、輪子 128 個；A100 實測 hfield 6.8k vs 平面 102k env-steps/s。
        # 因此「機身碰地」不能再當摔倒訊號 → 終止改成幾何判定（_fallen：傾角 > 60° 或機身離地 < FALL_H），加上出界 oob。
        self._mj = m
        self.sys = mjx.put_model(m)
        self.H = jnp.array(self.H_np, dtype=jnp.float32)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        assert min(jid) >= 0, "policy_io 的關節名在 MJCF 找不到"
        self.QADR = jnp.array([m.jnt_qposadr[j] for j in jid]); self.VADR = jnp.array([m.jnt_dofadr[j] for j in jid])
        trn = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.AID = jnp.array([trn[j] for j in jid])
        # 腿 12 關節（PhysX 序）的硬限位與軟限位（中點 ± LIM_SOFT × 半行程）；lim 項與 sat_n 指標（PD 目標口徑）用
        rng_ = np.array([m.jnt_range[j] for j in jid[:12]], dtype=np.float64)
        assert bool(np.all(np.array([m.jnt_limited[j] for j in jid[:12]]))), "腿關節要有限位"
        mid, half = rng_.mean(1), 0.5 * (rng_[:, 1] - rng_[:, 0])
        self.Q_LO_HARD, self.Q_HI_HARD = jnp.array(rng_[:, 0], dtype=jnp.float32), jnp.array(rng_[:, 1], dtype=jnp.float32)
        self.Q_LO_SOFT = jnp.array(mid - LIM_SOFT * half, dtype=jnp.float32)
        self.Q_HI_SOFT = jnp.array(mid + LIM_SOFT * half, dtype=jnp.float32)
        self.FLOOR = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.BASE_BID = int(base)                       # DR（質量／質心）用；機身不碰撞，終止不靠接觸
        self.WHEEL_GIDS = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p}_FOOT_LINK_COLL") for p in FEET])
        # hfield 子網格用的包圍半徑：膠囊輪 rbound ＝ 半徑＋半長 0.101 → ceil(0.202/0.1)+1 ＝ 4 格；輪子真正碰地的是半徑 0.096 的圓周，
        # 覆寫成 0.098 → 3 格，每輪稜柱 4×4×2=32 → 3×3×2=18。MJX 的 geom_rbound_hfield 是給這個用的獨立欄位，不影響其他碰撞。
        rb = np.array(self.sys._impl.geom_rbound_hfield)
        rb[np.asarray(self.WHEEL_GIDS)] = wt.WHEEL_R + 0.002
        self.sys = self.sys.replace(_impl=self.sys._impl.replace(geom_rbound_hfield=rb))
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

    def _terrain_h(self, x, y):
        """(x,y) 的地形高（雙線性，與 wheel_terrain.height_np 同式；索引夾在圖內，出圖時取邊緣格）。"""
        fx = (x - wt.X0) / wt.DX; fy = (y - wt.Y0) / wt.DX
        j = jnp.clip(jnp.floor(fx), 0, wt.NCOL - 2).astype(jnp.int32); i = jnp.clip(jnp.floor(fy), 0, wt.NROW - 2).astype(jnp.int32)
        tx = jnp.clip(fx - j, 0.0, 1.0); ty = jnp.clip(fy - i, 0.0, 1.0)
        H = self.H
        return (1 - tx) * (1 - ty) * H[i, j] + tx * (1 - ty) * H[i, j + 1] + (1 - tx) * ty * H[i + 1, j] + tx * ty * H[i + 1, j + 1]

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

    def _obs_dict(self, data, cmd, last_a, rng_noise, bias_rp=None):
        """非對稱 actor-critic（同他的 env.yaml）：policy 看帶雜訊的 53（狗上拿得到的量）；
        critic 看無雜訊的同樣 53 ＋ 機身系線速度 3（實機量不到，只在訓練時給 value 估計用；匯出只取 policy 分支）。
        bias_rp：IMU 常數偏差 (roll, pitch) rad，只旋轉 policy 的重力三維（grav_bias_deg > 0 才生效）。"""
        clean = self._obs(data, cmd, last_a)
        vb = quat_rot_inv(data.qpos[3:7], data.qvel[0:3])
        pol = clean
        if self.grav_bias_deg > 0:
            pol = clean.at[3:6].set(bias_matrix(bias_rp[0], bias_rp[1]).T @ clean[3:6])
        return {"state": self._noisy(pol, rng_noise), "privileged_state": jnp.concatenate([clean, vb]).astype(jnp.float32)}

    def _maybe_mirror(self, obs, mir):
        """鏡像回合：policy（state）與 critic（privileged_state）都看左右翻過的世界。mirror=False 時原樣回傳（預設路徑不碰）。"""
        if not self.mirror:
            return obs
        return {"state": jnp.where(mir, mirror_obs(obs["state"]), obs["state"]),
                "privileged_state": jnp.where(mir, mirror_priv(obs["privileged_state"]), obs["privileged_state"])}

    def _ctrl(self, a):
        """a：他的尺度（已乘 ACT_GAIN）→ 致動器 ctrl（腿 PD 目標 rad、輪速 rad/s）。"""
        return jnp.zeros(self.sys.nu).at[self.AID].set(a * ACT_SCALE + ACT_OFFSET)

    # ---------------------------------------------------------------- 指令（站立 10%、小指令 small_frac、heading → wz、半回合重抽）
    def _sample_cmd(self, rng, yaw):
        k = jax.random.split(rng, 4)
        small = jax.random.uniform(k[0]) < self.small_frac
        standing = jax.random.uniform(k[1]) < 0.1
        if self.fixed_cmd:                       # 指令讀 cmd_fix：旗標固定 False，lin_small／small／standing metric 才不是隨機噪音
            small, standing = jnp.bool_(False), jnp.bool_(False)
        v = jnp.where(small, jax.random.uniform(k[2], (2,), minval=-0.4, maxval=0.4), jax.random.uniform(k[2], (2,), minval=-1.0, maxval=1.0))
        head = jnp.where(small, yaw + jax.random.uniform(k[3], minval=-0.5, maxval=0.5), jax.random.uniform(k[3], minval=-jnp.pi, maxval=jnp.pi))
        return jnp.where(standing, 0.0, v), head, standing, small

    def _cmd_now(self, data, info):
        if self.fixed_cmd:
            return info["cmd_fix"]
        i = (info["t"] >= self.episode_len // 2).astype(jnp.int32)     # 半回合（預設 10 s）重抽
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
        mir = jax.random.bernoulli(jax.random.fold_in(rng, 11)) if (self.mirror and not self.fixed_cmd) else jnp.bool_(False)
        delay = jax.random.randint(jax.random.fold_in(rng, 12), (), 0, self.delay_max + 1) if (self.delay_max > 0 and not self.fixed_cmd) else jnp.int32(0)
        lim = jnp.radians(self.grav_bias_deg)
        bias_rp = jax.random.uniform(jax.random.fold_in(rng, 13), (2,), minval=-lim, maxval=lim) if (self.grav_bias_deg > 0 and not self.fixed_cmd) else jnp.zeros(2)
        info = dict(rng=k[4], t=jnp.int32(0), kind=kind, level=level, xy0=xy,
                    cmd_ab=jnp.stack([cmd_a, cmd_b]), head_ab=jnp.stack([head_a, head_b]), standing=standing, small=small,
                    cmd_fix=jnp.zeros(3), cmd=jnp.zeros(3), last_a=jnp.zeros(ACT_DIM),
                    qvel_prev=data.qvel[self.VADR], foot_prev=data.xpos[self.FOOT_BIDS],
                    touch_prev=jnp.zeros(4, bool), vfoot_prev=jnp.zeros((4, 3)), mir=mir,
                    delay=delay, a_hist=jnp.zeros((3, ACT_DIM)), bias_rp=bias_rp,
                    push_t=jnp.int32(self.episode_len // 2) + jax.random.randint(k[5], (), 0, self.episode_len // 4))   # 10–15 s 後第一次推
        info["cmd"] = self._cmd_now(data, info)
        obs = self._obs_dict(data, info["cmd"], info["last_a"], jax.random.fold_in(rng, 6), info["bias_rp"])   # 第一個 obs 也帶雜訊；fold_in 不動 split(6) 的出生／指令抽樣
        obs = self._maybe_mirror(obs, mir)
        z0 = jnp.zeros(())
        return State(data, obs, z0, z0, {k_: z0 for k_ in METRIC_KEYS}, info)

    def _next_tile(self, rng, data, info):
        """他的 terrain_levels_vel（升級門檻改 CURR_UP_DIST 2 m，他的是方塊一半 4 m）：位移 > 門檻升級；位移 < 0.5 × 指令速度 × 回合時間降級；到頂隨機重抽；站立回合不動。
        reset()（data=None）時初始等級抽 0–5；到頂重抽仍是 0–9。"""
        if self.fixed_tile is not None:
            return jnp.int32(self.fixed_tile[0]), jnp.int32(self.fixed_tile[1])
        k = jax.random.split(rng, 3)
        kind = jax.random.randint(k[0], (), 0, len(wt.KINDS))
        rnd = jax.random.randint(k[1], (), 0, wt.LEVELS)
        if data is None:                                             # 初始等級 0–5（他的 max_init_terrain_level 5）
            return kind, jax.random.randint(k[1], (), 0, 6)
        dist = jnp.linalg.norm(data.qpos[0:2] - info["xy0"])
        vref = 0.5 * (jnp.linalg.norm(info["cmd_ab"][0]) + jnp.linalg.norm(info["cmd_ab"][1]))
        up = dist > CURR_UP_DIST                                     # 升級優先於降級（見下行 where 順序）
        down = dist < 0.5 * vref * (self.episode_len * CTRL_DT)
        delta = jnp.where(info["standing"], 0, jnp.where(up, 1, jnp.where(down, -1, 0)))
        lvl = info["level"] + delta
        lvl = jnp.where(lvl >= wt.LEVELS, rnd, jnp.clip(lvl, 0, wt.LEVELS - 1))
        return kind, lvl

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
        a_pol = ACT_GAIN * action
        a = jnp.where(info["mir"], mirror_act(a_pol), a_pol) if self.mirror else a_pol      # 鏡像回合：policy 的動作翻回物理側
        rng, k_noise, k_tile, k_spawn, k_push, k_next = jax.random.split(info["rng"], 6)
        if self.push:                                   # 他的 push_by_setting_velocity：機身 x/y 速度「設成」U(−0.5, 0.5)
            do_push = info["t"] == info["push_t"]
            vxy = jax.random.uniform(k_push, (2,), minval=-0.5, maxval=0.5)
            data = data.replace(qvel=data.qvel.at[0:2].set(jnp.where(do_push, vxy, data.qvel[0:2])))
            nxt = jax.random.randint(k_next, (), self.episode_len // 2, self.episode_len * 3 // 4 + 1)   # 下一次在 10–15 s 後
            info["push_t"] = jnp.where(do_push, info["push_t"] + nxt, info["push_t"])
        if self.delay_max > 0:                          # 動作延遲 DR：物理吃 delay 步前的動作（實機 des→q 約 30 ms ≈ 1–2 步）；obs 的 last_action 仍是 policy 本步輸出
            a_hist = jnp.concatenate([a[None], info["a_hist"][:-1]], axis=0)
            a_exec = a_hist[info["delay"]]
        else:
            a_hist, a_exec = info["a_hist"], a
        data = self._physics(data, a_exec)
        cmd = info["cmd"]
        reward, fell, metrics, aux = self._after_physics(data, info, a, cmd)
        t1 = info["t"] + 1
        done_env = fell | (t1 >= self.episode_len)
        # 續走路徑
        info_c = dict(info, rng=rng, t=t1, last_a=a, qvel_prev=data.qvel[self.VADR], foot_prev=data.xpos[self.FOOT_BIDS],
                      touch_prev=aux["touch"], vfoot_prev=aux["vfoot"], a_hist=a_hist)
        info_c["cmd"] = self._cmd_now(data, info_c)
        obs_c = self._obs_dict(data, info_c["cmd"], a, k_noise, info["bias_rp"])
        obs_c = self._maybe_mirror(obs_c, info["mir"])
        # 重生路徑（回合結束才用；每步都算，jit 下成本固定）
        kind_n, level_n = self._next_tile(k_tile, data, info)
        new = self._spawn_state(k_spawn, kind_n, level_n)
        new_info = dict(new.info, cmd_fix=info["cmd_fix"]); new_info["cmd"] = self._cmd_now(new.pipeline_state, new_info)
        sel = lambda x, y: jnp.where(done_env, x, y)   # noqa: E731
        data_o = jax.tree_util.tree_map(sel, new.pipeline_state, data)
        obs_n = self._obs_dict(new.pipeline_state, new_info["cmd"], new_info["last_a"], k_noise, new_info["bias_rp"])
        obs_n = self._maybe_mirror(obs_n, new_info["mir"])
        obs_o = jax.tree_util.tree_map(sel, obs_n, obs_c)
        info_o = {k_: (sel(new_info[k_], info_c[k_]) if k_ in new_info else info_c[k_]) for k_ in info_c}
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

    def _fallen(self, data):
        """摔倒（幾何判定；機身／腿不碰撞所以不能用機身接觸）：回傳 (tilt, low)。
        tilt：機身系重力 z > −0.5（傾 > 60°）；low：機身 z − 正下方地形高 < FALL_H。"""
        grav = quat_rot_inv(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        tilt = grav[2] > FALL_COS
        low = (data.qpos[2] - self._terrain_h(data.qpos[0], data.qpos[1])) < FALL_H
        return tilt, low

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
        grav_true = quat_rot_inv(q, jnp.array([0.0, 0.0, -1.0]))                   # 真實重力（不是帶偏差的 obs）
        roll = grav_true[1] ** 2                                                     # 只罰 roll：上樓梯／斜坡的俯仰是正當的
        touchdown = touch & ~info["touch_prev"] & (info["t"] >= IMPACT_SETTLE)       # 本步觸地、上步沒觸地；出生落定期（t 是本步前的計數）不算
        vz_prev = jnp.maximum(0.0, -info["vfoot_prev"][:, 2])                        # 觸地前一步的下落速度
        impact = jnp.sum(touchdown * jnp.maximum(0.0, vz_prev - self.impact_v0) ** 2)     # 門檻形式：只罰超過 impact_v0 的部分（預設 0＝原式）
        impact_vz = jnp.sum(touchdown * vz_prev)                                     # 本步落地事件的下落速度**總和**（診斷量；平均要除 td_n，不能除步數）
        td_n = jnp.sum(touchdown).astype(jnp.float32)                                # 本步落地事件數
        impact_fn = jnp.max(C["fn"] * floor_hit)                                     # 本步輪–地法向力峰（診斷量）
        axis = data.xmat[self.FOOT_BIDS][:, :, 1]                                   # 輪軸（FOOT_LINK 的 y 軸）世界向量
        slip = jnp.sum(touch * jnp.sum(vfoot * axis, axis=1) ** 2)                 # 他的 wheel_lateral_slip（spin_axis 1）
        ww_mask = jnp.isin(C["g0"], self.WHEEL_GIDS) & jnp.isin(C["g1"], self.WHEEL_GIDS)
        ww_force = jnp.sum(C["fn"] * ww_mask)
        ww = jnp.sum((jnp.minimum(C["fn"] / 100.0, WW_CAP) * ww_mask) ** 2)         # 法向力 / 100 N 的 L2（改動一）；上限 3（腿交叉時 fn 可達 2755 N，單步 −760 會拉壞 critic）
        # 實體間隙 hinge（pair 碰不到內側面，見 WHEEL_INNER_W 註解）：gap < GAP_MIN 起罰，(1 − gap/GAP_MIN)²，gap ≤ 0 繼續長
        gaps = jnp.stack([jnp.linalg.norm(foot[i] - foot[j]) for i, j in LAT_PAIRS]) - WHEEL_INNER_W
        wgap = jnp.sum(jnp.maximum(0.0, 1.0 - gaps / GAP_MIN) ** 2)
        standing_scale = jnp.where(jnp.linalg.norm(cmd) < 0.1, 10.0, 1.0)
        qleg = data.qpos[self.QADR[:12]]
        dev = jnp.sum(jnp.abs(qleg - DEFAULT[:12])) * standing_scale
        # 限位懲罰罰的是 **PD 目標**（與 _ctrl 同式；a 已乘 ACT_GAIN），不是實際 q（Isaac joint_pos_limits 罰 q）：
        # v1 攤帳顯示實際 q 從沒進軟限位，越界的是目標——vy 0.5 時 RL ABAD 目標最低 −0.73、硬限 −0.523、目標與 q 差 0.73。
        # q 口徑對這種「目標越界、q 跟不上」沒有梯度。w_lim 2.0：目標超 0.1–0.2 rad 時每步 −0.2～−0.4，與追蹤增益同量級。
        q_des12 = a[:12] * ACT_SCALE[:12] + ACT_OFFSET[:12]
        lim = jnp.sum(jnp.maximum(0.0, self.Q_LO_SOFT - q_des12) + jnp.maximum(0.0, q_des12 - self.Q_HI_SOFT))
        sat_n = jnp.sum((q_des12 < self.Q_LO_HARD) | (q_des12 > self.Q_HI_HARD)).astype(jnp.float32)   # 目標超出硬限位的顆數
        s2 = self.std ** 2
        T = dict(lin=2.5 * jnp.exp(-err_lin / s2), ang=self.w_ang * jnp.exp(-err_ang / s2),
                 vz=-2.0 * vb[2] ** 2, angxy=-self.w_angxy * jnp.sum(w[0:2] ** 2),
                 tau_leg=-2e-4 * jnp.sum(tau[:12] ** 2), tau_wheel=-2e-4 * jnp.sum(tau[12:] ** 2),
                 acc_leg=-2.5e-7 * jnp.sum(qacc[:12] ** 2), acc_wheel=-2.5e-7 * jnp.sum(qacc[12:] ** 2),
                 rate=-0.01 * jnp.sum((a - info["last_a"]) ** 2),
                 slip=-self.w_slip * slip, dev=-0.1 * dev, ww=-self.w_ww * ww, wgap=-self.w_gap * wgap, lim=-self.w_lim * lim,
                 roll=-self.w_roll * roll, impact=-self.w_impact * impact,
                 knee=-self.w_knee * jnp.sum(jnp.maximum(0.0, self.knee_min - jnp.abs(qleg[8:12]))))   # 膝比 knee_min 更直的部分（rad，4 膝和）；彎更多免費
        aux = dict(vb=vb, w=w, touch=touch, ww_force=ww_force, gaps=gaps, err_lin=err_lin, err_ang=err_ang, C=C, sat_n=sat_n,
                   vfoot=vfoot, impact_vz=impact_vz, td_n=td_n, impact_fn=impact_fn)
        return T, aux

    def _after_physics(self, data, info, a, cmd):
        T, aux = self._reward_terms(data, info, a, cmd)
        reward = sum(T.values())
        x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
        # 摔倒＝傾角 > 60° 或離地 < FALL_H（_fallen）；出界／掉落另判。z 門檻取 0（整張 hfield 的基準面）：反向樓梯第 9 級坑底在 BASE_H − 0.8 = 0.2 m，
        # 站在坑底機身約 0.67 m，舊門檻 BASE_H − 0.5 只差 0.17 m 就誤判摔；低於 0 才一定是掉出地圖
        oob = (jnp.abs(x) > X_LIM) | (jnp.abs(y) > Y_LIM) | (z < 0.0)
        tilt, low = self._fallen(data)
        fell = tilt | low | oob
        metrics = dict(T, reward=reward, lin_small=T["lin"] * info["small"], level=info["level"].astype(jnp.float32),
                       fell=fell.astype(jnp.float32), oob=oob.astype(jnp.float32), tilt=tilt.astype(jnp.float32), dist=jnp.linalg.norm(data.qpos[0:2] - info["xy0"]), ww_force=aux["ww_force"], gap_min=jnp.min(aux["gaps"]),
                       vx_err=jnp.abs(aux["vb"][0] - cmd[0]), vy_err=jnp.abs(aux["vb"][1] - cmd[1]), wz_err=jnp.abs(aux["w"][2] - cmd[2]),
                       touch_n=jnp.sum(aux["touch"]).astype(jnp.float32), small=info["small"].astype(jnp.float32), standing=info["standing"].astype(jnp.float32), sat_n=aux["sat_n"],
                       impact_vz=aux["impact_vz"], td_n=aux["td_n"], impact_fn=aux["impact_fn"], mir=info["mir"].astype(jnp.float32), delay=info["delay"].astype(jnp.float32))
        return reward, fell, metrics, aux

    @property
    def observation_size(self):
        return {"state": (OBS_DIM,), "privileged_state": (PRIV_DIM,)}

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"


# ---------------------------------------------------------------- domain randomization（每個環境一組，訓練全程固定；與 v3 線相同做法）
def make_domain_randomize(env: WheelEnv):
    """他的：地面摩擦（這裡 0.6–1.0 均勻）、基座質量 −1～+3 kg、質心 x/y ±0.05、z ±0.01。
    改動三：腿關節 frictionloss（實機辨識 1.85／1.5）× 0.5–1.5、輪 frictionloss 0.10–0.18、輪 damping 0.005–0.03。"""
    base = env.BASE_BID
    leg_dof, wheel_dof = env.VADR[:12], env.VADR[12:]

    def domain_randomize(sys, rng):
        @jax.vmap
        def per_env(rng):
            k = jax.random.split(rng, 7)
            gf = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.6, maxval=1.0))
            bm = sys.body_mass.at[base].add(jax.random.uniform(k[1], minval=-1.0, maxval=3.0))
            dcom = jnp.array([jax.random.uniform(k[2], minval=-0.05, maxval=0.05), jax.random.uniform(k[3], minval=-0.05, maxval=0.05),
                              jax.random.uniform(k[4], minval=-0.01, maxval=0.01)])
            ipos = sys.body_ipos.at[base].add(dcom)
            fl = sys.dof_frictionloss.at[leg_dof].multiply(jax.random.uniform(k[5], minval=0.5, maxval=1.5))
            fl = fl.at[wheel_dof].set(jax.random.uniform(k[6], (4,), minval=0.10, maxval=0.18))
            dmp = sys.dof_damping.at[wheel_dof].set(jax.random.uniform(jax.random.fold_in(rng, 7), (4,), minval=0.005, maxval=0.03))
            return gf, bm, ipos, fl, dmp

        gf, bm, ipos, fl, dmp = per_env(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, sys)
        in_axes = in_axes.replace(geom_friction=0, body_mass=0, body_ipos=0, dof_frictionloss=0, dof_damping=0)
        sys = sys.replace(geom_friction=gf, body_mass=bm, body_ipos=ipos, dof_frictionloss=fl, dof_damping=dmp)
        return sys, in_axes

    return domain_randomize


# ---------------------------------------------------------------- 訓練包裝器（取代 brax AutoResetWrapper：重生已在 WheelEnv.step 做完）
class RespawnWrapper(Wrapper):
    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        if "steps" in state.info:
            steps = jnp.where(state.done, jnp.zeros_like(state.info["steps"]), state.info["steps"])
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        return self.env.step(state, action)


def wrap_env(env, episode_length: int = EPISODE_LEN, action_repeat: int = 1, randomization_fn=None):
    """給 ppo.train(wrap_env_fn=rl_env_wheel.wrap_env)：Vmap／DR → Episode（截斷 done＋truncation）→ Respawn。
    截斷步與 WheelEnv 內部 t ≥ episode_len 的重生同一步發生（兩邊都從 0 數）。"""
    assert episode_length == env.episode_len, "EpisodeWrapper 截斷步必須與 WheelEnv 內部重生同步"
    from brax.envs.wrappers import training as tw
    env = tw.DomainRandomizationVmapWrapper(env, randomization_fn) if randomization_fn is not None else tw.VmapWrapper(env)
    env = tw.EpisodeWrapper(env, episode_length, action_repeat)
    return RespawnWrapper(env)
