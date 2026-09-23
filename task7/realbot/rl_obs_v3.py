"""狗上純 numpy 的 v3 observation（88 維）組裝 ＋ 左右鏡像（D1 Max，2026-09-23）。

只 import numpy 與 realbot/ 內檔案。與 `rl_env_v3.DualModeEnv._obs` 的對應由 `tests/test_rl_obs_v3.py` 釘住。

obs 88：grav 3 | gyro 3 | q−stance 12 | qvel 12 | wheel_v 4 | cmd 3 | [s4, s_arc] 2 | head_err 1 | last_a 24 |
        amp 4 | sinθ 4 | cosθ 4 | cyc_abad 4 | cyc_knee 4 | wheel_τ/5 4
  - 關節：馬達座標 → 控制器座標 `coord.to_ctrl`，**減站姿 stance_q12（不是 HOME12）**；速度除 SIGN；MJCF 腿序
  - 輪速：`joint_state.velocity` × `coord.SIGN[KIND_WHEEL]`（馬達 → MJCF 方向）；MJCF 序 (FR, FL, RR, RL)
  - 四元數 xyzw → wxyz → `w2b(q, (0,0,−1))`；gyro 直接用（IMU 偏置不修正，訓練有隨機化）
  - 站姿與鏡像索引表由 npz 帶，本檔零常數（腿名與輪名除外）
鏡像（`mirror_obs`／`mirror_act`）：右向指令時把 obs 左右翻給 policy、動作翻回來 —— 等於 `rl_env_v3.mirror_policy(turn_only=False)`。
"""
from __future__ import annotations

import numpy as np

import coord
import rl_obs
from rl_obs import Frame, w2b  # noqa: F401  （M9 沿用同一個 Frame）

LEG_NAMES = rl_obs.LEG_NAMES                                              # 12，MJCF 序
WHEEL_NAMES = [rl_obs.LEG_SHM[l] + coord.KIND_WHEEL for l in rl_obs.LEG_MJCF]   # 4，MJCF 序
WHEEL_SIGN4 = np.array([coord.SIGN[coord.KIND_WHEEL][n[:2]] for n in WHEEL_NAMES])
ACT_DIM, OBS_DIM = 24, 88
_DOWN = np.array([0.0, 0.0, -1.0])
_SIGN12 = rl_obs._SIGN12


def read_frame(state_ro, imu_shm, idx: dict, state_stride: int) -> Frame:
    """狗上用：12 腿關節 ＋ 4 輪 ＋ imu_central → Frame。`idx` 要含 16 個關節名。"""
    st = state_ro.states()
    names = LEG_NAMES + WHEEL_NAMES
    pos = {n: st[idx[n]]["position"] for n in names}
    vel = {n: st[idx[n]]["velocity"] for n in names}
    imu = imu_shm.imu()
    return Frame(pos, vel, imu["quat"], imu["gyro"], state_ro.read_tick(state_stride))


def is_right(cmd) -> bool:
    """鏡像觸發：|wz| > 0.1 時看 wz < 0，否則看 vy < −0.02（= mirror_policy(turn_only=False)）。"""
    vx, vy, wz = (float(x) for x in cmd)
    return (wz < 0.0) if abs(wz) > 0.1 else (vy < -0.02)


class Obs:
    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as z:
            self.stance12 = np.asarray(z["STANCE_Q12"], dtype=float)
            self.oi = np.asarray(z["OBS_MIRROR_IDX"]).astype(int)
            self.os = np.asarray(z["OBS_MIRROR_SGN"], dtype=np.float32)
            self.ai = np.asarray(z["ACT_MIRROR_IDX"]).astype(int)
            self.as_ = np.asarray(z["ACT_MIRROR_SGN"], dtype=float)
            self.lr = np.asarray(z["MIRROR_LR"]).astype(int)
        assert self.stance12.shape == (12,) and self.oi.shape == (OBS_DIM,) and self.ai.shape == (ACT_DIM,)

    def wheel_v(self, fr: Frame) -> np.ndarray:
        return np.array([fr.vel[n] for n in WHEEL_NAMES], dtype=float) * WHEEL_SIGN4

    @staticmethod
    def gyro_z(fr: Frame) -> float:
        return float(fr.gyro[2])

    def build(self, fr: Frame, c: dict, cmd, last_a, head_err: float, s4: float, s_arc: float) -> np.ndarray:
        cmd = np.asarray(cmd, dtype=np.float64).reshape(-1)
        last_a = np.asarray(last_a, dtype=np.float64).reshape(-1)
        assert cmd.size == 3, f"cmd 應為 3 維 (vx, vy, wz)，實得 {cmd.size}"
        assert last_a.size == ACT_DIM, f"last_a 應為 {ACT_DIM} 維，實得 {last_a.size}"
        q = np.asarray(fr.quat_xyzw, dtype=float)[[3, 0, 1, 2]]
        jpos = np.array([coord.to_ctrl(n, fr.pos[n]) for n in LEG_NAMES], dtype=float)
        jvel = np.array([fr.vel[n] for n in LEG_NAMES], dtype=float) / _SIGN12
        th = np.asarray(c["theta"], dtype=float)
        o = np.concatenate([
            w2b(q, _DOWN), np.asarray(fr.gyro, dtype=float),
            jpos - self.stance12, jvel, self.wheel_v(fr),
            cmd, [float(s4), float(s_arc)], [float(np.clip(head_err, -1.0, 1.0))],
            last_a,
            np.asarray(c["amp"], float), np.sin(th), np.cos(th), np.asarray(c["cyc_abad"], float), np.asarray(c["cyc_knee"], float),
            np.asarray(c["wheel"], float) / 5.0,
        ]).astype(np.float32)
        assert o.size == OBS_DIM, f"obs 應為 {OBS_DIM} 維，實得 {o.size}"
        return o

    def mirror_obs(self, obs) -> np.ndarray:
        """照 rl_env_v3.mirror_obs：索引／符號表 ＋ θ 相位修正（腿交換後四腿一起加 d = θ0 − θ1）。"""
        obs = np.asarray(obs, dtype=np.float32)
        o = obs[self.oi] * self.os
        th = np.arctan2(obs[68:72], obs[72:76])
        d = th[0] - th[1]
        th_m = th[self.lr] + d
        o[68:72] = np.sin(th_m)
        o[72:76] = np.cos(th_m)
        return o.astype(np.float32)

    def mirror_act(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return a[self.ai] * self.as_
