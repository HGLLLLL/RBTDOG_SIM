"""狗上純 numpy 的同事純 RL policy observation（53 維，layout `rl16`，2026-09-29）。

只 import numpy 與 realbot/ 內檔案。與 `inference/rl_only_infer.py`（MuJoCo 直接算的 obs）的對應由
`tests/test_policy_rl_closed_loop.py` 影子跟跑釘住。

obs 53（`RL_only/policy_io.json`）：
    ang_vel 3 ｜ projected_gravity 3 ｜ cmd 3 (vx, vy, wz) ｜ joint_pos − default 12 ｜ joint_vel 16（12 腿＋4 輪）｜ last_action 16
  - 關節：馬達座標 → 控制器座標 `coord.to_ctrl`，減 default pose；速度除 SIGN；**PhysX 序**（ABAD×4, HIP×4, KNEE×4，腿序 FR FL RR RL）
  - 輪速：`joint_state.velocity` × WHEEL_SIGN4（馬達 → 模型方向）；序 FR FL RR RL
  - 四元數 xyzw → wxyz → `w2b(q, (0,0,−1))`，再乘 `grav_fix_matrix(--grav-offset)`（下）；gyro 直接用
  - 無縮放、無正規化；default pose 與 PhysX↔MJCF 對照由 npz 帶
鏡像：這個 policy 左右都訓過（指令 ±1），不用鏡像；`mirror_obs`／`mirror_act` 是恆等，M9 要帶 `--no-mirror`。

★ 重力歸零 `grav_offset=(roll_deg, pitch_deg)`（2026-10-14，trip26 平移前漂）：
  實機站好時 IMU 讀 roll +0.7°／pitch −1.5°（trip24／25／26 三趟一致），模擬站好是 −0.1／−0.3°；policy 把這 1.5° 當成
  「機身前低」去補，平移時 x 方向沒有指令就累積成前漂（實機里程 +0.12～0.17 m/s；模擬注入同樣偏差 +0.10～0.13，反號就往後漂）。
  `grav_offset` ＝「機身水平時 IMU 讀到的 (roll, pitch)」，M7 `read_imu_rp` 的慣例（＝ `rl_only_infer.rp_from_R`）。
  修正只動 obs 的重力三維：g' = R_fix · g，其中 R_fix = Ry(pitch)·Rx(roll)；站好時 g' 回到 (0,0,−1)。
  gyro 不改（常數安裝角只是座標系轉 1.5°）；M9 的傾角護欄用 IMU 原始值，不走這條。
"""
from __future__ import annotations

import json
import math

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


def grav_fix_matrix(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """R_fix = Ry(pitch)·Rx(roll)。機身水平時 IMU 讀 (roll, pitch) ⇒ policy 看到 g = R_fixᵀ·(0,0,−1)；乘回 R_fix 就歸零。
    慣例與 `M7_standup.read_imu_rp`／`rl_only_infer.rp_from_R` 同：roll = atan2(R21, R22)、pitch = atan2(−R20, √(R21²+R22²))。"""
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return ry @ rx


class Obs:
    def __init__(self, npz_path: str, grav_offset=(0.0, 0.0)):
        with np.load(npz_path, allow_pickle=False) as z:
            K = json.loads(str(z["gen_json"]))
            self.obs_dim, self.act_dim = int(z["obs_dim"]), int(z["act_dim"])
        self.p2m = np.asarray(K["PHYSX_TO_MJCF"], int)
        self.m2p = np.argsort(self.p2m)                                   # physx[j] = mjcf[m2p[j]]
        self.default_p = np.asarray(K["DEFAULT_POS_PHYSX"], float)[:12]
        assert self.obs_dim == 53 and self.act_dim == 16 and self.p2m.shape == (12,)
        self.grav_offset = (float(grav_offset[0]), float(grav_offset[1]))
        assert max(abs(x) for x in self.grav_offset) <= 5.0, f"grav_offset {self.grav_offset}° 太大：站立讀值只有 1–2°，超過 5° 多半是打錯"
        self.grav_fix = grav_fix_matrix(*self.grav_offset)

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
            np.asarray(fr.gyro, dtype=float), self.grav_fix @ w2b(q, _DOWN), cmd,
            jpos_m[self.m2p] - self.default_p, jvel_m[self.m2p], self.wheel_v(fr),
            last_a,
        ]).astype(np.float32)
        assert o.size == self.obs_dim, f"obs 應為 {self.obs_dim} 維，實得 {o.size}"
        return o

    def mirror_obs(self, obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def mirror_act(self, a) -> np.ndarray:
        return np.asarray(a, dtype=float)
