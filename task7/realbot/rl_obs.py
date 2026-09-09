"""狗上純 numpy 的 CPG-RL observation 組裝（D1 Max，2026-09-09）。

只 import numpy 與 realbot/ 內檔案（狗上沒有 mujoco / jax）。
與 `inference/obs_max.build_obs` 的對應由 `tests/test_rl_obs.py` **逐位元**釘住 ——
兩份實作各自看起來正常、卻走出兩種 obs，是本專案踩過的坑。

obs 66 維（v2.3，act 10，無 head_err）：
    gravity 3 | gyro 3 | joint_pos 12 | joint_vel 12 | cmd 2 | last_action 10 | cpg 24

換算（全部沿用 real_obs / obs_max 的式子）：
  - 關節：馬達座標 → 控制器座標 `coord.to_ctrl`；速度只除 `SIGN`；**MJCF 腿序 FR, FL, RR, RL**
  - 四元數：`imu_central` 原始 **xyzw** → wxyz → 機身系重力 `w2b(q, (0,0,−1))`
  - gyro：rad/s、三軸與 MJCF 同號（H 文件定案），scale 1
  - cpg：`cpg.py` 的 dict（fl/fr/bl/br）→ MJCF 序陣列：rx, rx_d, ry, ry_d, sinθ, cosθ
IMU 姿態偏置（pitch −1.24°）**不修正** —— 訓練端已隨機化 ±3°。
"""
from __future__ import annotations

import numpy as np

import coord

LEG_MJCF = ("FR", "FL", "RR", "RL")
LEG_SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
LEG_NAMES = [LEG_SHM[l] + k for l in LEG_MJCF for k in coord.LEG_KINDS]     # 12，MJCF 序
# ⚠️ 與 max_model.HOME12 同值（測試釘住）：hip 0.8 / 膝 −1.5，後腿反號。
HOME12 = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, -0.8, 1.5, 0.0, -0.8, 1.5])
ACT_DIM = 10
OBS_DIM = 3 + 3 + 12 + 12 + 2 + ACT_DIM + 24
_DOWN = np.array([0.0, 0.0, -1.0])
_SIGN12 = np.array([coord.SIGN[n[2:]][n[:2]] for n in LEG_NAMES])


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    u = q[1:4]
    t = 2 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def w2b(q, v):
    """世界向量轉到機身座標系（q = wxyz）。與 cpg_max.w2b 同式。"""
    return qrot(qinv(q), v)


def to_shm_legs(arr4) -> dict:
    """MJCF 序 (FR, FL, RR, RL) → `{fl, fr, bl, br}`（cpg.step 要的 dict）。"""
    return {LEG_SHM[l]: float(v) for l, v in zip(LEG_MJCF, arr4)}


def from_shm_legs(d: dict) -> np.ndarray:
    return np.array([d[LEG_SHM[l]] for l in LEG_MJCF], dtype=float)


class Frame:
    """一個 50 Hz 推進點讀到的感測狀態。關節值是 **馬達座標**（shm 原樣），dict 以 shm 關節名索引。"""
    __slots__ = ("pos", "vel", "quat_xyzw", "gyro", "tick")

    def __init__(self, pos: dict, vel: dict, quat_xyzw, gyro, tick: int = -1):
        self.pos, self.vel = pos, vel
        self.quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        self.gyro = np.asarray(gyro, dtype=float)
        self.tick = tick


def read_frame(state_ro, imu_shm, idx: dict, state_stride: int) -> Frame:
    """狗上用：從 `joint_state` 與 `imu_central` 兩塊 shm 讀一幀（唯讀）。"""
    st = state_ro.states()
    pos = {n: st[idx[n]]["position"] for n in LEG_NAMES}
    vel = {n: st[idx[n]]["velocity"] for n in LEG_NAMES}
    imu = imu_shm.imu()
    return Frame(pos, vel, imu["quat"], imu["gyro"], state_ro.read_tick(state_stride))


def build(fr: Frame, c: dict, cmd, last_a) -> np.ndarray:
    """→ float32(66)。`c` 是 `cpg.py` 的狀態 dict（fl/fr/bl/br）。"""
    cmd = np.asarray(cmd, dtype=np.float64).reshape(-1)
    last_a = np.asarray(last_a, dtype=np.float64).reshape(-1)
    assert cmd.size == 2, f"cmd 應為 2 維 (vx, wz)，實得 {cmd.size}"
    assert last_a.size == ACT_DIM, f"last_a 應為 {ACT_DIM} 維，實得 {last_a.size}"
    q = np.asarray(fr.quat_xyzw, dtype=float)[[3, 0, 1, 2]]
    jpos = np.array([coord.to_ctrl(n, fr.pos[n]) for n in LEG_NAMES], dtype=float)
    jvel = np.array([fr.vel[n] for n in LEG_NAMES], dtype=float) / _SIGN12
    th = from_shm_legs(c["theta"])
    o = np.concatenate([
        w2b(q, _DOWN),                     # gravity 3
        np.asarray(fr.gyro, dtype=float),  # gyro 3
        jpos - HOME12,                     # joint_pos 12
        jvel,                              # joint_vel 12
        cmd,                               # cmd 2
        last_a,                            # last_action 10
        from_shm_legs(c["rx"]), from_shm_legs(c["rx_d"]),
        from_shm_legs(c["ry"]), from_shm_legs(c["ry_d"]),
        np.sin(th), np.cos(th),            # cpg 24
    ]).astype(np.float32)
    assert o.size == OBS_DIM, f"obs 應為 {OBS_DIM} 維，實得 {o.size}"
    return o
