"""真機外圈遙控直線控制器（spec: docs/superpowers/specs/2026-07-15-rc-odom-line-controller-design.md）。

輸入：遙控桿量 Sticks(fwd, lat, turn) + 外接定位 Odom(x, y, yaw, stamp)
輸出：cmd = np.array([vx, vy, wz], float32)；上層自行接 SportClient.Move() 或模擬。
僅依賴 numpy，無 MuJoCo/SDK。時間一律用呼叫端傳入的 now/stamp（秒、同時基）。
"""
from dataclasses import dataclass
import numpy as np

MANUAL, SETTLING, TRACKING = "manual", "settling", "tracking"


# ---- 核心純函式：原封複製自 task4/inference/local_infer_paper.py:97-117（勿改） ----
def wrap(a): return np.arctan2(np.sin(a), np.cos(a))


def line_frame(psi):
    """目標線的方向 d 與左法向 n（世界系單位向量）。"""
    d = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-np.sin(psi), np.cos(psi)])
    return d, n


def line_control(p, yaw, p0, psi_target, vx, k_yaw, k_ct, no_lateral=False):
    """方案 A 解耦控制：wz 用航向誤差鎖航向、vy 用 cross-track 誤差滑回線上。
    p, p0 為世界系 (x,y)；回傳 (cmd[vx,vy,wz] float32, e_ct, e_yaw)。"""
    _, n = line_frame(psi_target)
    e_ct = float(n @ (np.asarray(p, float) - np.asarray(p0, float)))
    e_yaw = float(wrap(yaw - psi_target))
    wz = float(np.clip(-k_yaw * e_yaw, -1.0, 1.0))
    vy = 0.0 if no_lateral else float(np.clip(-k_ct * e_ct, -0.3, 0.3))
    return np.array([vx, vy, wz], np.float32), e_ct, e_yaw


# ---- 介面型別 ----
@dataclass
class Sticks:
    fwd: float = 0.0    # 前進 [-1,1]，負=倒退
    lat: float = 0.0    # 橫移 [-1,1]，左正
    turn: float = 0.0   # 轉向 [-1,1]，左(CCW)正


@dataclass
class Odom:
    x: float
    y: float
    yaw: float
    stamp: float        # 量測時間戳（秒，與 update 的 now 同時基）


@dataclass
class Config:
    vmax: float = 0.6                       # fwd 滿桿速度 m/s
    vymax: float = 0.3                      # 橫移滿桿速度（MANUAL 透傳）
    wmax: float = 1.0                       # 轉向滿桿角速度 rad/s
    k_yaw: float = 3.0
    k_ct: float = 1.5
    vy_lim: float = 0.3                     # 輸出限幅
    wz_lim: float = 1.0
    dead_on: float = 0.08                   # 死區遲滯：離中門檻
    dead_off: float = 0.04                  # 回中門檻
    yaw_rate_stable: float = 0.1            # 航向穩定門檻 rad/s
    settle_s: float = 0.3                   # 穩定持續時間 s
    lpf_tau: float = 0.03                   # 角速度低通時間常數（≈5Hz）
    stale_s: float = 0.5                    # odom 逾時 s
    jump_pos_m: float = 0.5                 # 重定位跳變門檻
    jump_yaw_rad: float = float(np.radians(30))
    slew_vy: float = 1.5                    # 輸出斜率限制 m/s²（0=關）
    slew_wz: float = 5.0                    # rad/s²（0=關）
