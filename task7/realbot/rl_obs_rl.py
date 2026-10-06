"""狗上純 numpy 的同事純 RL policy observation（53 維，layout `rl16`，2026-09-29）。

只 import numpy 與 realbot/ 內檔案。與 `inference/rl_only_infer.py`（MuJoCo 直接算的 obs）的對應由
`tests/test_policy_rl_closed_loop.py` 影子跟跑釘住。

obs 53（`RL_only/policy_io.json`）：
    ang_vel 3 ｜ projected_gravity 3 ｜ cmd 3 (vx, vy, wz) ｜ joint_pos − default 12 ｜ joint_vel 16（12 腿＋4 輪）｜ last_action 16
  - 關節：馬達座標 → 控制器座標 `coord.to_ctrl`，減 default pose；速度除 SIGN；**PhysX 序**（ABAD×4, HIP×4, KNEE×4，腿序 FR FL RR RL）
  - 輪速：`joint_state.velocity` × WHEEL_SIGN4（馬達 → 模型方向）；序 FR FL RR RL
  - 四元數 xyzw → wxyz → `w2b(q, (0,0,−1))`；gyro 直接用（IMU 偏置不修正：訓練沒隨機化，這是已知 sim2real 項）
  - 無縮放、無正規化；default pose 與 PhysX↔MJCF 對照由 npz 帶
鏡像：這個 policy 左右都訓過（指令 ±1），不用鏡像；`mirror_obs`／`mirror_act` 是恆等，M9 要帶 `--no-mirror`。
"""
from __future__ import annotations

import json

import numpy as np

import coord
import rl_obs
import rl_obs_v3
from rl_obs import Frame, w2b  # noqa: F401

LEG_NAMES = rl_obs.LEG_NAMES            # 12，MJCF 序
WHEEL_NAMES = rl_obs_v3.WHEEL_NAMES     # 4，FR FL RR RL
WHEEL_SIGN4 = rl_obs_v3.WHEEL_SIGN4
read_frame = rl_obs_v3.read_frame       # 16 關節 ＋ imu_central → Frame（同 v3）
_DOWN = np.array([0.0, 0.0, -1.0])
_SIGN12 = rl_obs._SIGN12


class Obs:
    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as z:
            K = json.loads(str(z["gen_json"]))
            self.obs_dim, self.act_dim = int(z["obs_dim"]), int(z["act_dim"])
        self.p2m = np.asarray(K["PHYSX_TO_MJCF"], int)
        self.m2p = np.argsort(self.p2m)                                   # physx[j] = mjcf[m2p[j]]
        self.default_p = np.asarray(K["DEFAULT_POS_PHYSX"], float)[:12]
        assert self.obs_dim == 53 and self.act_dim == 16 and self.p2m.shape == (12,)

    def wheel_v(self, fr: Frame) -> np.ndarray:
        return np.array([fr.vel[n] for n in WHEEL_NAMES], dtype=float) * WHEEL_SIGN4

    @staticmethod
    def gyro_z(fr: Frame) -> float:
        return float(fr.gyro[2])

    def build(self, fr: Frame, c=None, cmd=(0.0, 0.0, 0.0), last_a=None, head_err: float = 0.0, s4: float = 0.0, s_arc: float = 0.0) -> np.ndarray:
        """組 53 維 obs。`c`／`head_err`／`s4`／`s_arc` 是 V3GaitStream 介面的參數，本 policy 不用。"""
        cmd = np.asarray(cmd, dtype=np.float64).reshape(-1)
        last_a = np.zeros(16) if last_a is None else np.asarray(last_a, dtype=np.float64).reshape(-1)
        assert cmd.size == 3, f"cmd 應為 3 維 (vx, vy, wz)，實得 {cmd.size}"
        assert last_a.size == 16, f"last_a 應為 16 維，實得 {last_a.size}"
        q = np.asarray(fr.quat_xyzw, dtype=float)[[3, 0, 1, 2]]
        jpos_m = np.array([coord.to_ctrl(n, fr.pos[n]) for n in LEG_NAMES], dtype=float)
        jvel_m = np.array([fr.vel[n] for n in LEG_NAMES], dtype=float) / _SIGN12
        o = np.concatenate([
            np.asarray(fr.gyro, dtype=float), w2b(q, _DOWN), cmd,
            jpos_m[self.m2p] - self.default_p, jvel_m[self.m2p], self.wheel_v(fr),
            last_a,
        ]).astype(np.float32)
        assert o.size == self.obs_dim, f"obs 應為 {self.obs_dim} 維，實得 {o.size}"
        return o

    def mirror_obs(self, obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def mirror_act(self, a) -> np.ndarray:
        return np.asarray(a, dtype=float)
