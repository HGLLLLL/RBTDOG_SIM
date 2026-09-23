"""v3 統一運動學產生器的 numpy 移植（狗上跑，2026-09-23）。

只 import numpy、json、標準庫與 realbot/kin.py。**本檔零常數**：所有數字由 npz 帶
（`load_params`），函式一對一對應 `inference/rl_env_v3.py` 的同名函式，
正確性由 `tests/test_gen_v3.py` 對 jax 版逐函式（2e-5）／逐步（q_des 1e-4 rad）釘住。
jax 端是 float32、這裡是 float64，所以容差不是 0。

`Gen.step(cmd, act, v_wheel)` 的順序逐行照 `DualModeEnv.step`（延遲是 env 的事，這裡吃「這一步要套用的動作」；
淡入 RAMP_STEPS 在這裡做，因為它是產生器狀態 i 的函數）。
"""
from __future__ import annotations

import json

import numpy as np

import kin

LEG_SHM = ("fr", "fl", "br", "bl")          # MJCF 序 (FR, FL, RR, RL) → kin.ik 的 shm 腿名
TWO_PI = 2.0 * np.pi
TABLE_KEYS = ("PH_TURN_L", "PH_LAT_L", "PH_TROT", "W_LAT_L", "W_TURN_L", "W_ARC_L", "DOM_TURN_L", "LAT_WHEEL_L",
              "MIRROR_LR", "SIDE_Y", "FRONT", "FOOT_XY_BODY", "STANCE_Q12", "STANCE_FEET", "KNEE_SIGN4", "LO12", "HI12",
              "CYC_OFF", "CYC_TAU", "CYC_HZ", "CYC_QMEAN")


class Params:
    """npz 快照：`ref`（env.ref dict）、`K`（模組常數 dict）、`T`（表，numpy）。"""

    def __init__(self, ref: dict, K: dict, T: dict):
        self.ref, self.K, self.T = ref, K, T
        # 上機安全閘：用 ValueError 不用 assert（python -O 會把 assert 整條拿掉，守門就消失了）
        if ref["wheel_space"] != "tau":
            raise ValueError("v3.6f 只走力矩空間")
        if ref.get("cyc_lat_decouple", False):
            raise ValueError("v3.7 解耦不在本移植範圍")
        if not ref["cyc_recenter"]:
            raise ValueError("cyc_recenter=False 的絕對模式沒移植")
        if ref.get("cyc_turn_sym") is not True:     # npz 存的一定是 SYM 週期表；非 SYM 權重載進來會默默配錯表
            raise ValueError("npz 的週期表是 SYM 版，只能配 cyc_turn_sym=True 的權重")


def load_params(path: str) -> Params:
    with np.load(path, allow_pickle=False) as z:
        ref = json.loads(str(z["ref_json"]))
        K = json.loads(str(z["gen_json"]))
        T = {k: np.asarray(z[k], dtype=float) for k in TABLE_KEYS}
        T["MIRROR_LR"] = np.asarray(z["MIRROR_LR"]).astype(int)
    return Params(ref, K, T)


# ---------------------------------------------------------------- 純函式（對應 rl_env_v3）
def _mirror(v, right: bool, p: Params):
    return v[p.T["MIRROR_LR"]] if right else v


def activity(cmd, p: Params) -> dict:
    ar = p.ref["a_ref"]
    a_lat = float(np.clip(abs(cmd[1]) / ar["vy"], 0.0, 1.0))
    a_turn = float(np.clip(abs(cmd[2]) / ar["wz"], 0.0, 1.0))
    a_wheel = float(np.clip(abs(cmd[0]) / ar["vx"], 0.0, 1.0))
    return dict(lat=a_lat, turn=a_turn, wheel=a_wheel, s4=max(a_lat, a_turn * (1.0 - a_wheel)), arc=a_turn * a_wheel)


def lat_weight(A: dict) -> float:
    return A["lat"] / (A["lat"] + A["turn"] + 1e-6)


def leg_activity(cmd, A: dict, p: Params) -> np.ndarray:
    T = p.T
    w_lat = _mirror(T["W_LAT_L"], cmd[1] < 0, p)
    w_turn = _mirror(T["W_TURN_L"], cmd[2] < 0, p)
    w_arc = _mirror(T["W_ARC_L"], cmd[2] < 0, p)
    wl = lat_weight(A)
    return np.maximum(A["s4"] * (wl * w_lat + (1.0 - wl) * w_turn), A["arc"] * w_arc)


def step_gain(s, p: Params) -> np.ndarray:
    return np.clip(np.asarray(s, float) / p.K["STEP_GAIN_S"], 0.0, 1.0)


def phase_offsets(cmd, A: dict, p: Params) -> np.ndarray:
    T, ref = p.T, p.ref
    ph_t = T["PH_TROT"] if ref["phase_set_turn"] == "trot" else _mirror(T["PH_TURN_L"], cmd[2] < 0, p)
    ph_l = T["PH_TROT"] if ref["phase_set_lat"] == "trot" else _mirror(T["PH_LAT_L"], cmd[1] < 0, p)
    wl = lat_weight(A)
    z = wl * np.exp(2j * np.pi * ph_l) + (1.0 - wl) * np.exp(2j * np.pi * ph_t)
    return np.mod(np.angle(z) / TWO_PI, 1.0)


def slew_phase(ph, ph_tgt, p: Params) -> np.ndarray:
    d = np.mod(np.asarray(ph_tgt, float) - np.asarray(ph, float) + 0.5, 1.0) - 0.5
    m = p.K["PH_SLEW"]
    return np.mod(np.asarray(ph, float) + np.clip(d, -m, m), 1.0)


def kin_step_vec(cmd, T_cycle: float, p: Params) -> np.ndarray:
    ref, XY = p.ref, p.T["FOOT_XY_BODY"]
    rot = ref["rot_step_frac"] * cmd[2]
    dx = T_cycle * (-rot * XY[:, 1])
    dy = T_cycle * (ref["lat_cmd_gain"] * cmd[1] + rot * XY[:, 0])
    return np.stack([dx, dy], 1)


def posture_offset(cmd, A: dict, p: Params) -> np.ndarray:
    ref, T = p.ref, p.T
    pa = ref["posture_arc"]
    inner = np.array([0.0, 1.0, 0.0, 1.0]) if cmd[2] > 0 else np.array([1.0, 0.0, 1.0, 0.0])
    amount = T["FRONT"] * pa["front"] + (1.0 - T["FRONT"]) * pa["rear"]
    arc = A["arc"] * inner * amount * (-T["SIDE_Y"])
    lat = A["lat"] * ref.get("posture_lat", 0.0) * T["SIDE_Y"]
    return arc + lat


def wheel_cmd(cmd, A: dict, res, p: Params) -> np.ndarray:
    ref, T = p.ref, p.T
    diff = (cmd[0] - cmd[2] * T["SIDE_Y"] * ref["L_eff"] / 2) / ref["r_wheel"]
    dom = _mirror(T["DOM_TURN_L"], cmd[2] < 0, p)
    gate = 1.0 + A["turn"] * (1.0 - A["wheel"]) * (1.0 - 2.0 * dom)
    lw = _mirror(T["LAT_WHEEL_L"], cmd[1] < 0, p) * ref["lat_omega"] * float(np.clip(abs(cmd[1]) / ref["lat_vy_ref"], 0.0, 1.5))
    return diff * gate + lw + np.asarray(res, float)


def step_pattern(cmd, p: Params) -> dict:
    ref = p.ref
    cmd = np.asarray(cmd, float)
    A = activity(cmd, p)
    s = leg_activity(cmd, A, p)
    wl = lat_weight(A)
    cyc_on = wl * float(ref["step_gen_lat"] == "cycle") + (1.0 - wl) * float(ref["step_gen_turn"] == "cycle")
    A_kin = dict(A, s4=A["s4"] * (1.0 - cyc_on))
    s_kin = leg_activity(cmd, A_kin, p)
    hz = wl * ref["step_hz_lat"] + (1.0 - wl) * ref["step_hz_turn"]
    duty = wl * ref["duty_lat"] + (1.0 - wl) * ref["duty_turn"]
    lift = wl * ref["lift_lat"] + (1.0 - wl) * ref["lift"]
    return dict(A=A, s=s, s_kin=s_kin, g=step_gain(s_kin, p), vec=kin_step_vec(cmd, 1.0 / hz, p), ph=phase_offsets(cmd, A, p),
                wheel0=wheel_cmd(cmd, A, np.zeros(4), p), hz=hz, duty=duty, cyc_on=cyc_on,
                lift=lift * (s_kin / max(float(np.max(s_kin)), 1e-6)) * step_gain(s_kin, p), post_y=posture_offset(cmd, A, p))


def duty_remap(th, duty: float) -> np.ndarray:
    ph = np.mod(np.asarray(th, float), TWO_PI) / TWO_PI
    sw = 1.0 - duty
    return np.where(ph < sw, np.pi * ph / sw, np.pi + np.pi * (ph - sw) / duty)


def act_split(a, p: Params) -> dict:
    K = p.K
    a = np.tanh(np.asarray(a, float))
    return dict(om=1.0 + K["OMEGA_NOM_SCALE"] * a[0], amp=1.0 + K["AMP_SCALE"] * a[1:5], lift=1.0 + K["LIFT_SCALE"] * a[5],
                sway=K["SWAY_MAX"] * a[6:8], wres=K["WHEEL_RES"] * a[8:12], qres=K["QRES_MAX"] * a[12:24],
                foot_x=K["FOOT_OFF_MAX"] * a[1:5], foot_z=K["FOOT_OFF_MAX"] * a[5])


def foot_targets(theta, amp, g, vec, lift4, sway, u, foot_x, foot_z, duty, post_y, z_sag, p: Params) -> np.ndarray:
    th = duty_remap(theta, duty)
    s = np.sin(th)
    prog = -0.5 * np.cos(th)
    amp = np.asarray(amp, float); g = np.asarray(g, float)
    ramp = np.clip(amp, 0.0, 1.0)
    dxy = (np.asarray(vec, float) * (amp * g)[:, None]) * prog[:, None]
    dz = np.where(s > 0, (np.asarray(lift4, float) + z_sag * g) * s, 0.0) * ramp
    step_off = np.concatenate([dxy, dz[:, None]], 1) + u * np.array([sway[0], sway[1], 0.0])
    post = np.stack([np.zeros(4), np.asarray(post_y, float), np.zeros(4)], 1)
    wheel_off = np.stack([np.asarray(foot_x, float), np.zeros(4), np.full(4, float(foot_z))], 1)
    return p.T["STANCE_FEET"] + post + step_off + (1.0 - u) * wheel_off


def joint_targets_n(feet, p: Params) -> tuple[np.ndarray, int]:
    """IK 十二軸，並回報這一步有幾條腿被 `kin.ik` 縮限（構不到 → 沿徑向縮）。
    靜默的縮限在狗上表現成「動作突然變鈍」而查不出原因，所以要數出來。"""
    ks = p.T["KNEE_SIGN4"]
    out = np.zeros(12)
    n_clamped = 0
    for k in range(4):
        (q1, q2, q3), clamped = kin.ik(LEG_SHM[k], float(feet[k, 0]), float(feet[k, 1]), float(feet[k, 2]), float(ks[k]))
        out[3 * k:3 * k + 3] = (q1, q2, q3)
        n_clamped += int(clamped)
    return out, n_clamped


def joint_targets(feet, p: Params) -> np.ndarray:
    return joint_targets_n(feet, p)[0]


def cyc_interp(table, phi: float, p: Params) -> np.ndarray:
    n = p.K["CYC_N"]
    x = np.mod(phi / TWO_PI, 1.0) * n
    i0 = int(np.floor(x)) % n
    i1 = (i0 + 1) % n
    f = x - np.floor(x)
    return table[:, i0] * (1.0 - f) + table[:, i1] * f


def cycle_offsets(phi: float, cmd, A: dict, p: Params) -> dict:
    ref, T = p.ref, p.T
    off = cyc_interp(T["CYC_OFF"], phi, p)
    tau = cyc_interp(T["CYC_TAU"], phi, p)
    w_ll = 0.5 * (1.0 + float(np.clip(cmd[1] / 0.02, -1.0, 1.0)))
    w_tl = 0.5 * (1.0 + float(np.clip(cmd[2] / 0.3, -1.0, 1.0)))
    lo, hi = ref["cyc_lat_amp_clip"]
    amp_l = float(np.clip(ref["cyc_lat_amp0"] + ref["cyc_lat_slope"] * (abs(cmd[1]) - ref["cyc_lat_vy0"]), lo, hi)) * ref["cyc_amp_lat"]
    amp_t = float(np.clip(abs(cmd[2]) / ref["a_ref"]["wz"], 0.3, 1.2)) * ref["cyc_amp_turn"]
    on_l, on_t = float(ref["step_gen_lat"] == "cycle"), float(ref["step_gen_turn"] == "cycle")
    wl = lat_weight(A)

    def blend(Tb, sc_l=1.0, sc_t=1.0):
        lat = w_ll * Tb[0] + (1.0 - w_ll) * Tb[1]
        turn = w_tl * Tb[2] + (1.0 - w_tl) * Tb[3]
        return wl * amp_l * on_l * lat * sc_l + (1.0 - wl) * amp_t * on_t * turn * sc_t

    sc_lat = np.tile(np.asarray(ref["cyc_joint_scale_lat"], float), 4)
    sc_turn = np.tile(np.asarray(ref["cyc_joint_scale_turn"], float), 4)
    delta = blend(off, sc_lat, sc_turn)
    wheel_tau = blend(tau) * ref["cyc_wheel"]
    HZ = T["CYC_HZ"]
    hz = wl * (w_ll * HZ[0] + (1.0 - w_ll) * HZ[1]) + (1.0 - wl) * (w_tl * HZ[2] + (1.0 - w_tl) * HZ[3])
    return dict(delta=delta, wheel_tau=wheel_tau, hz=hz * ref["cyc_hz_scale"], on=wl * on_l + (1.0 - wl) * on_t)


def wheel_dead(tau, p: Params) -> np.ndarray:
    """τ 死區（= rl_env_v3.wheel_ctrl_tau 的力矩部分）：|τ| < TAU_FF → 0，否則 τ + TAU_FF·sign。寫進 effort 欄前才套。"""
    tau = np.asarray(tau, float)
    ff = p.K["TAU_FF"]
    return np.where(np.abs(tau) < ff, 0.0, tau + np.sign(tau) * ff)


# ---------------------------------------------------------------- 產生器狀態機（對應 DualModeEnv.reset／step 的產生器部分）
class Gen:
    """v3 產生器。狀態 = env `info` 裡產生器會用到的那幾個；不含延遲、EMA、reward。"""

    def __init__(self, p: Params, cmd0=(0.0, 0.0, 0.0)):
        self.p = p
        K = p.K
        self.dt = K["CTRL_DT"]
        cmd0 = np.asarray(cmd0, float)
        P0 = step_pattern(cmd0, p)
        self.c = dict(theta=np.zeros(4), amp=np.zeros(4), wheel=np.zeros(4), cyc_abad=np.zeros(4), cyc_knee=np.zeros(4))
        self.u_mode = float(np.max(P0["s"]))
        self.ph = P0["ph"].copy()
        self.s4, self.s_arc = P0["A"]["s4"], P0["A"]["arc"]
        self.phi_cyc = 0.0
        self.u4 = 0.0
        self.qres = np.zeros(12)
        self.sway = np.zeros(2)
        self.head_err = 0.0
        # ★ 存**複本**：呼叫端常常重複用同一個 cmd 陣列（遙控就是原地改值），存參考的話 cmd_prev 會跟著一起變，
        #   而 obs 帶的就是 cmd_prev —— 「上一步的指令」偷偷變成「這一步的指令」，鏡像判斷與 obs 都會錯。
        self.cmd_prev = np.array(cmd0, dtype=float)
        self.i = 0
        self.n_clamp = 0                            # 累計 IK 被縮限的腿次；正常跟跑應該一路是 0
        # ★ 關節限位飽和：IK 之後的 `q_ik + delta + qres` 被 LO12/HI12 夾掉的步數（每步最多算一次）。
        #   與 n_clamp 是兩件事：n_clamp 是「足端構不到、沿徑向縮」，這裡是「算得出來、但超出機構限位」。
        #   橫移時 ABAD 幾乎每步都頂在 ±0.523 的停點，policy 要的那一段位移根本沒送出去，而且完全無聲。
        self.n_sat = 0
        self.sat_mask = np.zeros(12, dtype=bool)    # 最近一步哪幾軸飽和（索引 0/3/6/9 = 四條腿的 ABAD）

    def stance_q12(self) -> np.ndarray:
        return self.p.T["STANCE_Q12"].copy()

    def integrate_head(self, gyro_z: float) -> None:
        if self.i > 0:
            self.head_err += (float(gyro_z) - float(self.cmd_prev[2])) * self.dt

    def step(self, cmd, act, v_wheel):
        p, K, ref, T = self.p, self.p.K, self.p.ref, self.p.T
        cmd = np.asarray(cmd, float); act = np.asarray(act, float); v_wheel = np.asarray(v_wheel, float)
        P = step_pattern(cmd, p)
        u_tgt = float(np.max(P["s"]))
        du = self.dt / ref["trans_s"]
        u_mode = float(np.clip(self.u_mode + np.clip(u_tgt - self.u_mode, -du, du), 0.0, 1.0))
        u_ramp = min(1.0, self.i / K["RAMP_STEPS"])
        A = act_split(u_ramp * act, p)
        om = P["hz"] * A["om"]
        ph = slew_phase(self.ph, P["ph"], p)
        theta_free = self.c["theta"] + TWO_PI * om * self.dt
        ph_rad = TWO_PI * ph
        phi0 = theta_free[0] - ph_rad[0]
        theta = np.mod(phi0 + ph_rad if u_mode > 0.01 else self.c["theta"], TWO_PI)
        amp = self.c["amp"] + np.clip(P["g"] * A["amp"] - self.c["amp"], -0.05, 0.05)
        sway = self.sway + np.clip(A["sway"] * u_mode - self.sway, -K["SWAY_SLEW"], K["SWAY_SLEW"])
        lift4 = P["lift"] * A["lift"]
        feet = foot_targets(theta, amp, P["g"], P["vec"], lift4, sway, u_mode, A["foot_x"], A["foot_z"], P["duty"], P["post_y"], ref["z_sag"], p)
        u4_tgt = float(np.clip(P["A"]["s4"] / 0.3, 0.0, 1.0)) * P["cyc_on"]
        u4 = float(np.clip(self.u4 + np.clip(u4_tgt - self.u4, -du, du), 0.0, 1.0))
        CY = cycle_offsets(self.phi_cyc, cmd, P["A"], p)
        phi_cyc = np.mod(self.phi_cyc + TWO_PI * CY["hz"] * A["om"] * self.dt * float(u4 > 0.01), TWO_PI)
        delta = (CY["delta"].reshape(4, 3) * A["amp"][:, None] * np.array([1.0, A["lift"], A["lift"]])[None, :]).reshape(12) * u4
        qres = self.qres + np.clip(A["qres"] - self.qres, -K["QRES_SLEW"], K["QRES_SLEW"])
        q_ik, n_clamped = joint_targets_n(feet, p)
        self.n_clamp += n_clamped
        q_raw = q_ik + delta + qres
        self.sat_mask = (q_raw < T["LO12"]) | (q_raw > T["HI12"])
        self.n_sat += int(self.sat_mask.any())      # 每步最多 +1（「這一步有沒有人頂住限位」），不是軸次
        q_des = np.clip(q_raw, T["LO12"], T["HI12"])
        v_nom = wheel_cmd(cmd, P["A"], np.zeros(4), p)
        tau = ref["wheel_outer_gain"] * (v_nom - v_wheel) + A["wres"] + u4 * CY["wheel_tau"]
        d3 = delta.reshape(4, 3)
        self.c = dict(theta=theta, amp=amp, wheel=tau, cyc_abad=d3[:, 0].copy(), cyc_knee=d3[:, 2].copy())
        self.u_mode, self.ph, self.u4, self.phi_cyc, self.qres, self.sway = u_mode, ph, u4, phi_cyc, qres, sway
        self.s4, self.s_arc = P["A"]["s4"], P["A"]["arc"]
        self.cmd_prev = np.array(cmd, dtype=float)   # 同 __init__：存複本，不要把呼叫端的陣列別名留下來
        self.i += 1
        return q_des, tau, self.c
