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


class _StickGate:
    """單軸死區遲滯：|v|>on 進入 active，|v|<off 才離開，避免門檻邊緣顫振。"""

    def __init__(self, on, off):
        self.on, self.off = on, off
        self.active = False

    def update(self, v):
        if self.active:
            if abs(v) < self.off:
                self.active = False
        elif abs(v) > self.on:
            self.active = True
        return self.active


class _YawRateEst:
    """odom yaw 差分 + 一階低通估角速度；stable_for(now) 回傳連續穩定秒數。"""

    def __init__(self, tau, thresh):
        self.tau, self.thresh = tau, thresh
        self.reset()

    def reset(self):
        self._prev = None                    # (yaw, stamp)
        self.rate = 0.0
        self._stable_since = None

    def update(self, yaw, stamp):
        if self._prev is not None:
            dt = stamp - self._prev[1]
            if dt > 1e-6:
                raw = wrap(yaw - self._prev[0]) / dt
                a = dt / (self.tau + dt)
                self.rate += a * (raw - self.rate)
        self._prev = (yaw, stamp)
        if abs(self.rate) < self.thresh:
            if self._stable_since is None:
                self._stable_since = stamp
        else:
            self._stable_since = None

    def stable_for(self, now):
        return 0.0 if self._stable_since is None else max(0.0, now - self._stable_since)


class RCLineController:
    """遙控外圈直線控制器。每控制週期呼叫 update(sticks, odom, now) → cmd。

    狀態機（spec §4）：
      MANUAL   手動直通（任一接管條件：turn/lat 離中、odom 不可用、fwd 回中）
      SETTLING fwd 離中且接管桿回中，等航向穩定
      TRACKING 沿 latch 直線做 line_control 校正
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.state = MANUAL
        self.degraded = True                 # 尚未收到 odom
        self.latch = None                    # (p0: np.ndarray(2), psi) | None
        self._gate_fwd = _StickGate(c.dead_on, c.dead_off)
        self._gate_lat = _StickGate(c.dead_on, c.dead_off)
        self._gate_turn = _StickGate(c.dead_on, c.dead_off)
        self._yr = _YawRateEst(c.lpf_tau, c.yaw_rate_stable)
        self._last_odom = None               # 最後接受的 Odom
        self._prev_out = None                # (cmd, now)，slew 用

    def update(self, sticks, odom, now):
        c = self.cfg
        fwd_on = self._gate_fwd.update(sticks.fwd)
        lat_on = self._gate_lat.update(sticks.lat)
        turn_on = self._gate_turn.update(sticks.turn)
        fwd = sticks.fwd if fwd_on else 0.0

        fresh = self._ingest_odom(odom, now)
        self.degraded = not fresh

        if turn_on or lat_on or not fresh or not fwd_on:   # 接管 → 手動直通
            self.state = MANUAL
            self.latch = None
            lat_v = sticks.lat if lat_on else 0.0
            turn_v = sticks.turn if turn_on else 0.0
            cmd = np.array([fwd * c.vmax, lat_v * c.vymax, turn_v * c.wmax], np.float32)
            return self._finish(cmd, now)

        if self.latch is None:                             # 等航向穩才鎖線
            if self._yr.stable_for(now) >= c.settle_s:
                self.latch = (np.array([self._last_odom.x, self._last_odom.y]),
                              float(self._last_odom.yaw))
            else:
                self.state = SETTLING
                return self._finish(np.array([fwd * c.vmax, 0.0, 0.0], np.float32), now)

        self.state = TRACKING
        p0, psi = self.latch
        cmd, _, _ = line_control((self._last_odom.x, self._last_odom.y),
                                 self._last_odom.yaw, p0, psi,
                                 fwd * c.vmax, c.k_yaw, c.k_ct)
        return self._finish(cmd, now)

    def _ingest_odom(self, odom, now):
        """接收 odom：None/NaN 拒收；跳變作廢 latch；回傳「數據可用」（新鮮）。"""
        c = self.cfg
        if odom is None or not np.all(np.isfinite([odom.x, odom.y, odom.yaw, odom.stamp])):
            return False
        if self._last_odom is None or odom.stamp > self._last_odom.stamp:
            if self._last_odom is not None:
                dp = float(np.hypot(odom.x - self._last_odom.x, odom.y - self._last_odom.y))
                dyaw = abs(float(wrap(odom.yaw - self._last_odom.yaw)))
                if dp > c.jump_pos_m or dyaw > c.jump_yaw_rad:  # 外接定位重定位
                    self.latch = None
                    self._yr.reset()
            self._yr.update(odom.yaw, odom.stamp)
            self._last_odom = odom
        return (now - self._last_odom.stamp) <= c.stale_s

    def _finish(self, cmd, now):
        """安全網：限幅 + vy/wz 斜率限制（平滑狀態切換跳變），最後才出模組。"""
        c = self.cfg
        cmd = np.asarray(cmd, np.float32).copy()
        cmd[0] = np.clip(cmd[0], -c.vmax, c.vmax)
        cmd[1] = np.clip(cmd[1], -max(c.vymax, c.vy_lim), max(c.vymax, c.vy_lim))
        cmd[2] = np.clip(cmd[2], -max(c.wmax, c.wz_lim), max(c.wmax, c.wz_lim))
        if self._prev_out is not None:
            prev, t_prev = self._prev_out
            dt = max(now - t_prev, 1e-6)
            for i, rate in ((1, c.slew_vy), (2, c.slew_wz)):
                if rate > 0:
                    cmd[i] = np.clip(cmd[i], prev[i] - rate * dt, prev[i] + rate * dt)
        self._prev_out = (cmd.copy(), now)
        return cmd
