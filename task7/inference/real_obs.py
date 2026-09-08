"""M6 錄檔 → 68 維 observation 序列（實機端）。

★ 不另寫一份數學：造一個只有 `qpos`/`qvel` 的假 `MjData`（`RealFrame`），
  餵**同一支** `obs_max.build_obs`。兩份實作各自看起來正常、卻走出兩種 obs，
  是本專案踩過的坑（M9 的 `--traj` vs `--live`）。

換算（全部沿用既有式子）：
  - 關節：馬達座標 → 控制器座標 `coord.to_ctrl`；速度只除 `SIGN`
  - 腿序：SHM `fl,fr,bl,br` → MJCF `FR,FL,RR,RL`，按**名稱**
  - 四元數：`imu_central` 原始 `xyzw`（由 imu_check 定案）→ `w2b` 要的 `wxyz`
  - gyro：乘 `gyro_scale`（imu_check 定案的單位/軸向）→ `qvel[3:6]`（機身系角速度）
`cmd`/`last_action`/`cpg` 38 維是自產欄位，填零；對照只看前 30 維感測欄位。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import coord            # noqa: E402
import m6_rec           # noqa: E402
import max_model as mm  # noqa: E402
import obs_max          # noqa: E402

# MJCF 腿名 → SHM 腿名。與 play_gait_traj.MM2SHM / coord._CFG2SHM 相同（測試釘住）。
LEG_SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
LEG_NAMES = [LEG_SHM[l] + k for l in mm.LEGS for k in coord.LEG_KINDS]   # 12，MJCF 序
WHEEL_NAMES = [LEG_SHM[l] + coord.KIND_WHEEL for l in mm.LEGS]           # 4，MJCF 序
SENSOR_DIM = 3 + 3 + 12 + 12     # gravity, gyro, joint_pos, joint_vel
NQ, NV = 23, 22


def _sign(nm: str) -> float:
    return coord.SIGN[nm[2:]][nm[:2]]


def ctrl_pos(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([coord.to_ctrl(n, rec.j[n]["q"]) for n in LEG_NAMES], axis=1)


def ctrl_des(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([coord.to_ctrl(n, rec.j[n]["des"]) for n in LEG_NAMES], axis=1)


def ctrl_vel(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["v"] / _sign(n) for n in LEG_NAMES], axis=1)


def ctrl_tau(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["tau"] / _sign(n) for n in LEG_NAMES], axis=1)


def kp12(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["kp"] for n in LEG_NAMES], axis=1)


def kd12(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["kd"] for n in LEG_NAMES], axis=1)


def wheel_kp(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["kp"] for n in WHEEL_NAMES], axis=1)


def wheel_kd(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["kd"] for n in WHEEL_NAMES], axis=1)


def wheel_vel(rec: m6_rec.Rec) -> np.ndarray:
    return np.stack([rec.j[n]["v"] / _sign(n) for n in WHEEL_NAMES], axis=1)


def quat_wxyz(raw: np.ndarray, order: str) -> np.ndarray:
    """`imu_central` 原始 4 值 → MuJoCo 的 wxyz。"""
    raw = np.asarray(raw, dtype=float)
    if order == "xyzw":
        return raw[..., [3, 0, 1, 2]]
    if order == "wxyz":
        return raw
    raise ValueError(f"quat order 只能是 xyzw / wxyz，不是 {order!r}")


def zero_cpg() -> dict:
    z = np.zeros(4)
    return {"rx": z, "rx_d": z, "ry": z, "ry_d": z, "theta": z}


class RealFrame:
    """長得像 MjData 的殼：只有 build_obs 會碰的 qpos / qvel。"""
    __slots__ = ("qpos", "qvel")

    def __init__(self):
        self.qpos = np.zeros(NQ)
        self.qvel = np.zeros(NV)


def frame_at(rec: m6_rec.Rec, i: int, quat_order: str, gyro_scale) -> RealFrame:
    f = RealFrame()
    f.qpos[3:7] = quat_wxyz(rec.quat_raw[i], quat_order)
    f.qpos[mm.LEG_QPOS_IDX] = ctrl_pos(rec)[i]
    f.qvel[3:6] = rec.gyro[i] * np.asarray(gyro_scale, dtype=float)
    f.qvel[mm.LEG_QVEL_IDX] = ctrl_vel(rec)[i]
    return f


def obs_series(rec: m6_rec.Rec, quat_order: str = "xyzw",
               gyro_scale=(1.0, 1.0, 1.0)) -> np.ndarray:
    """(N, 68) float32。前 30 維是感測欄位，其餘 38 維填零。"""
    Q = quat_wxyz(rec.quat_raw, quat_order)
    P, V = ctrl_pos(rec), ctrl_vel(rec)
    G = rec.gyro * np.asarray(gyro_scale, dtype=float)
    c0, cmd0, a0 = zero_cpg(), np.zeros(2), np.zeros(obs_max.ACT_DIM)
    f = RealFrame()
    out = np.zeros((rec.n, obs_max.OBS_DIM), dtype=np.float32)
    for i in range(rec.n):
        f.qpos[3:7] = Q[i]
        f.qpos[mm.LEG_QPOS_IDX] = P[i]
        f.qvel[3:6] = G[i]
        f.qvel[mm.LEG_QVEL_IDX] = V[i]
        out[i] = obs_max.build_obs(f, c0, cmd0, a0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 錄檔 → 68 維 obs 序列（.npz）")
    ap.add_argument("rec", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quat-order", default="xyzw", choices=("xyzw", "wxyz"))
    ap.add_argument("--gyro-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    a = ap.parse_args()
    rec = m6_rec.load(a.rec)
    O = obs_series(rec, a.quat_order, a.gyro_scale)
    np.savez(a.out, obs=O, t=rec.t, layout=np.array([n for n, _ in obs_max.OBS_LAYOUT]))
    print(f"{a.rec.name}: {O.shape} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
