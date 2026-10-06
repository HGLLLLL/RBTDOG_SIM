"""狗上純 numpy 的「直接 PD 目標」產生器（同事的純 RL 步態，layout `rl16`，2026-09-29）。

取代 `gen_v3.Gen`：介面同名（M9 `V3GaitStream` 只透過這些成員用它），內容只有
    q_des = clip(default + scale·clip(a[:12], ±A_CLIP)·u_ramp, LO12, HI12)   （每步變化量 ≤ Q_SLEW）
    v_nom = wheel_scale·a[12:16]·u_ramp
    tau4  = G·(v_nom − v_meas)      → M9 kd1 路徑算前饋 ff = tau − G·(v_nom − v_used) = 0 → effort 恆 0
沒有 CPG、週期表、IK。常數全部從 npz 的 `gen_json` 讀（`export_policy_rl.py`）。

關節序：policy 的 12 維是 PhysX 序（ABAD×4, HIP×4, KNEE×4；腿序 FAR FBL RAR RBL），
M9 與 `rl_obs.LEG_NAMES` 是 MJCF 序（每腿 abad hip knee；腿序 FR FL RR RL）。`PHYSX_TO_MJCF` 由 npz 帶。
輪子 4 維兩邊都是 FR FL RR RL。
"""
from __future__ import annotations

import json

import numpy as np

TABLE_KEYS = ("LO12", "HI12", "STANCE_Q12")


class Params:
    """npz 快照：`ref`、`K`（gen_json）、`T`（LO12/HI12/STANCE_Q12，MJCF 序）。"""

    def __init__(self, ref: dict, K: dict, T: dict):
        self.ref, self.K, self.T = ref, K, T
        if ref.get("family") != "rl_only":
            raise ValueError("這個 npz 不是 rl_only 版面（ref.family）")
        for k in ("DEFAULT_POS_PHYSX", "ACT_SCALE_PHYSX", "PHYSX_TO_MJCF", "Q_SLEW", "A_CLIP", "RAMP_STEPS",
                  "TAU_FF", "WHEEL_TAU_MAX", "gains", "CTRL_DT", "act_dim", "obs_dim"):
            if k not in K:
                raise ValueError(f"gen_json 缺 {k!r}")


def load_params(path: str) -> Params:
    with np.load(path, allow_pickle=False) as z:
        ref = json.loads(str(z["ref_json"]))
        K = json.loads(str(z["gen_json"]))
        T = {k: np.asarray(z[k], dtype=float) for k in TABLE_KEYS}
    return Params(ref, K, T)


class Gen:
    def __init__(self, p: Params, cmd0=(0.0, 0.0, 0.0)):
        self.p = p
        K = p.K
        self.p2m = np.asarray(K["PHYSX_TO_MJCF"], int)                    # mjcf[k] = physx[p2m[k]]
        self.default_m = np.asarray(K["DEFAULT_POS_PHYSX"], float)[self.p2m]   # MJCF 序
        self.scale_m = np.asarray(K["ACT_SCALE_PHYSX"], float)[:12][self.p2m]
        self.wheel_scale = np.asarray(K["ACT_SCALE_PHYSX"], float)[12:16]
        self.lo, self.hi = p.T["LO12"], p.T["HI12"]
        self.q_slew, self.a_clip, self.ramp_steps = float(K["Q_SLEW"]), float(K["A_CLIP"]), int(K["RAMP_STEPS"])
        self.G = float(p.ref["wheel_outer_gain"])
        self.act_dim = int(K["act_dim"])
        self.i = 0
        self.q_prev = self.default_m.copy()
        self.cmd_prev = np.asarray(cmd0, float).reshape(3)
        # V3GaitStream／Obs 介面要的欄位（本產生器沒有相位／航向，全部給 0）
        self.head_err, self.s4, self.s_arc = 0.0, 0.0, 0.0
        self.c = {"theta": np.zeros(4), "amp": np.zeros(4), "wheel": np.zeros(4), "cyc_abad": np.zeros(4), "cyc_knee": np.zeros(4)}
        self.n_clamp, self.n_sat = 0, 0
        self.sat_mask = np.zeros(12, bool)
        self.v_nom, self.v_meas = np.zeros(4), np.zeros(4)

    def stance_q12(self) -> np.ndarray:
        """起身目標＝零動作目標＝他的 default pose（MJCF 序）。"""
        return self.default_m.copy()

    def integrate_head(self, gyro_z: float) -> None:      # 這個 policy 的 obs 沒有航向誤差
        pass

    def step(self, cmd, act, v_wheel):
        act = np.asarray(act, float).reshape(-1)
        if act.size != self.act_dim:
            raise ValueError(f"act 應為 {self.act_dim} 維，實得 {act.size}")
        a = np.clip(act, -self.a_clip, self.a_clip)
        u = min(1.0, self.i / self.ramp_steps) if self.ramp_steps > 0 else 1.0
        q_raw = self.default_m + self.scale_m * (u * a[:12][self.p2m])
        q_raw = self.q_prev + np.clip(q_raw - self.q_prev, -self.q_slew, self.q_slew)
        q_des = np.clip(q_raw, self.lo, self.hi)
        self.sat_mask = q_raw != q_des
        self.n_sat += int(self.sat_mask.any())
        self.v_nom = self.wheel_scale * (u * a[12:16])
        self.v_meas = np.asarray(v_wheel, float).reshape(4)
        tau4 = self.G * (self.v_nom - self.v_meas)
        self.q_prev = q_des
        self.cmd_prev = np.asarray(cmd, float).reshape(3)
        self.i += 1
        return q_des, tau4, self.c
