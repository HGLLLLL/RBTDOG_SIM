# 真機外圈遙控直線控制器（task5/rc_line）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 task4 已驗證的 odom 走直線校正算法包成平台無關的遙控外圈控制器（桿量+odom 進、(vx,vy,wz) 出），並在 MuJoCo 完成整合驗證。

**Architecture:** 純 Python 模組 `RCLineController`，三態狀態機 MANUAL→SETTLING→TRACKING；TRACKING 用原封複製的 `line_control` 沿 latch 直線校正；掉訊/跳變退化處理。模擬驗證重用 `task4/inference/odom_missions.Runner`。

**Tech Stack:** Python + numpy（模組本體）；MuJoCo + task4 論文版 CPG-RL 權重（僅 sim_demo）。

**Spec:** `docs/superpowers/specs/2026-07-15-rc-odom-line-controller-design.md`

## Global Constraints

- 模組 `rc_line_controller.py` 僅依賴 numpy，**不得** import MuJoCo/jax/SDK。
- `wrap` / `line_frame` / `line_control` 原封複製自 `task4/inference/local_infer_paper.py:97-117`，邏輯一行不改（docstring 註明來源）。
- 測試是獨立 assert 腳本（專案無 pytest），執行：`conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`，成功須印 `ALL n TESTS PASSED`。
- 模組不得寫死控制週期；時間一律來自呼叫端傳入的 `now` 與 `odom.stamp`（同一時基、秒）。
- 座標約定：世界系、公尺、弧度、yaw 以 CCW（左轉）為正；桿量 `fwd/lat/turn ∈ [-1,1]`，turn 正=左轉。
- 參數預設值 = spec §5 表（VMAX 0.6、K_YAW 3.0、K_CT 1.5、死區 0.08/0.04、穩定 0.1 rad/s×0.3s、逾時 0.5s、跳變 0.5m/30°）。
- 合成 odom 測試序列必須位置/航向連續（單步位移 <0.5m、轉角 <30°），否則會誤觸 Task 4 的跳變防護。
- Commit 訊息用 `feat(rc): ...` 繁中，結尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

---

### Task 1: 模組骨架——核心純函式 + 資料型別 + Config

**Files:**
- Create: `task5/rc_line/rc_line_controller.py`
- Create: `task5/rc_line/tests/test_rc_line_controller.py`

**Interfaces:**
- Produces: `wrap(a)->float-like`、`line_frame(psi)->(d,n)`、`line_control(p,yaw,p0,psi_target,vx,k_yaw,k_ct,no_lateral=False)->(np.ndarray[3] float32, e_ct, e_yaw)`、`@dataclass Sticks(fwd,lat,turn)`、`@dataclass Odom(x,y,yaw,stamp)`、`@dataclass Config(...)`、常數 `MANUAL="manual", SETTLING="settling", TRACKING="tracking"`。後續所有 Task 都 import 這些。

- [ ] **Step 1: 寫失敗測試**

建立 `task5/rc_line/tests/test_rc_line_controller.py`：

```python
"""RCLineController 單元測試（獨立 assert 腳本，conda run -n rbtdog python 執行）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rc_line_controller import wrap, line_frame, line_control, Sticks, Odom, Config


def t1_wrap():
    assert abs(wrap(2 * np.pi + 0.1) - 0.1) < 1e-9
    assert abs(wrap(-2 * np.pi - 0.1) + 0.1) < 1e-9


def t1_line_frame():
    d, n = line_frame(0.0)
    assert np.allclose(d, [1.0, 0.0]) and np.allclose(n, [0.0, 1.0])


def t1_line_control_values():
    cmd, e_ct, e_yaw = line_control((1.0, 0.5), 0.1, (0.0, 0.0), 0.0, 0.6, 3.0, 1.5)
    assert abs(e_ct - 0.5) < 1e-9
    assert abs(e_yaw - 0.1) < 1e-9
    assert abs(cmd[0] - 0.6) < 1e-6
    assert abs(cmd[1] + 0.3) < 1e-6            # -1.5*0.5=-0.75 → clip ±0.3
    assert abs(cmd[2] + 0.3) < 1e-6            # -3.0*0.1
    cmd0, _, _ = line_control((2.0, 0.0), 0.0, (0.0, 0.0), 0.0, 0.6, 3.0, 1.5)
    assert np.allclose(cmd0, [0.6, 0.0, 0.0])  # 在線上且對準 → 無修正


def t1_parity_with_task4():
    sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
    import local_infer_paper as P
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(-5, 5, 2); p0 = rng.uniform(-5, 5, 2)
        yaw = rng.uniform(-4, 4); psi = rng.uniform(-4, 4)
        a = line_control(p, yaw, p0, psi, 0.6, 3.0, 1.5)
        b = P.line_control(p, yaw, p0, psi, 0.6, 3.0, 1.5)
        assert np.allclose(a[0], b[0]) and abs(a[1] - b[1]) < 1e-12 and abs(a[2] - b[2]) < 1e-12


def t1_config_defaults():
    c = Config()
    assert c.vmax == 0.6 and c.k_yaw == 3.0 and c.k_ct == 1.5
    assert c.dead_on == 0.08 and c.dead_off == 0.04
    assert c.settle_s == 0.3 and c.stale_s == 0.5
    s = Sticks(); o = Odom(1.0, 2.0, 0.5, 3.0)
    assert s.fwd == 0.0 and o.stamp == 3.0


TESTS = [t1_wrap, t1_line_frame, t1_line_control_values, t1_parity_with_task4,
         t1_config_defaults]

if __name__ == "__main__":
    for fn in TESTS:
        fn(); print("ok", fn.__name__)
    print(f"ALL {len(TESTS)} TESTS PASSED")
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ModuleNotFoundError: No module named 'rc_line_controller'`

- [ ] **Step 3: 最小實作**

建立 `task5/rc_line/rc_line_controller.py`：

```python
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ALL 5 TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add task5/rc_line/rc_line_controller.py task5/rc_line/tests/test_rc_line_controller.py
git commit -m "feat(rc): task5 外圈控制器骨架：複製 line_control 純函式+介面型別+Config

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: 桿量死區遲滯 `_StickGate` + 航向角速度估計 `_YawRateEst`

**Files:**
- Modify: `task5/rc_line/rc_line_controller.py`（檔尾附加）
- Modify: `task5/rc_line/tests/test_rc_line_controller.py`（附加測試）

**Interfaces:**
- Consumes: Task 1 的 `wrap`。
- Produces: `_StickGate(on, off)`，方法 `update(v)->bool`（active 狀態）；`_YawRateEst(tau, thresh)`，屬性 `rate`，方法 `update(yaw, stamp)`、`stable_for(now)->float`（連續穩定秒數）、`reset()`。Task 3/4 直接使用。

- [ ] **Step 1: 寫失敗測試（附加到測試檔，TESTS 一併擴充）**

```python
from rc_line_controller import _StickGate, _YawRateEst


def t2_stick_gate_hysteresis():
    g = _StickGate(0.08, 0.04)
    assert g.update(0.05) is False           # 未過離中門檻
    assert g.update(0.09) is True
    assert g.update(0.05) is True            # 遲滯：未低於回中門檻
    assert g.update(-0.09) is True           # 負向同樣算離中
    assert g.update(0.03) is False


def t2_yaw_rate_estimator():
    e = _YawRateEst(0.03, 0.1)
    t = 0.0
    for _ in range(50):                      # 1s 以 0.5 rad/s 旋轉
        e.update(0.5 * t, t); t += 0.02
    assert abs(e.rate - 0.5) < 0.05
    assert e.stable_for(t) == 0.0            # 旋轉中不穩定
    yaw1 = 0.5 * t
    for _ in range(20):                      # 0.4s 靜止
        e.update(yaw1, t); t += 0.02
    assert abs(e.rate) < 0.1
    assert e.stable_for(t) > 0.2             # 停穩後開始累計
    e.reset()
    assert e.stable_for(t) == 0.0


def t2_yaw_rate_wrap_crossing():
    e = _YawRateEst(0.03, 0.1)
    t, yaw = 0.0, 3.1                        # 跨 ±pi 不爆
    for _ in range(50):
        e.update(wrap(yaw), t); yaw += 0.5 * 0.02; t += 0.02
    assert abs(e.rate - 0.5) < 0.05


TESTS += [t2_stick_gate_hysteresis, t2_yaw_rate_estimator, t2_yaw_rate_wrap_crossing]
```

（`TESTS += [...]` 放在原 `TESTS = [...]` 之後、`if __name__` 之前。）

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ImportError: cannot import name '_StickGate'`

- [ ] **Step 3: 實作（附加到模組檔尾）**

```python
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
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ALL 8 TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add task5/rc_line/rc_line_controller.py task5/rc_line/tests/test_rc_line_controller.py
git commit -m "feat(rc): 桿量死區遲滯與航向角速度估計器

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `RCLineController` 三態狀態機（latch／接管／倒退）

**Files:**
- Modify: `task5/rc_line/rc_line_controller.py`（檔尾附加）
- Modify: `task5/rc_line/tests/test_rc_line_controller.py`（附加測試）

**Interfaces:**
- Consumes: Task 1 的 `line_control/Config/Sticks/Odom/MANUAL/SETTLING/TRACKING`、Task 2 的 `_StickGate/_YawRateEst`。
- Produces: `RCLineController(cfg: Config | None = None)`；方法 `update(sticks: Sticks, odom: Odom | None, now: float) -> np.ndarray[3] float32`；可觀察屬性 `state`（三常數之一）、`latch`（`(p0: np.ndarray(2), psi: float)` 或 `None`）、`degraded: bool`；內部掛點 `_ingest_odom(odom, now)->bool`、`_finish(cmd, now)->np.ndarray`（Task 4 會替換這兩個）。

- [ ] **Step 1: 寫失敗測試（附加）**

```python
from rc_line_controller import RCLineController, MANUAL, SETTLING, TRACKING

CTRL_DT = 0.02


def drive(ctrl, secs, sticks_fn, odom_fn, t0=0.0):
    """以 50Hz 餵 secs 秒；sticks_fn/odom_fn 都吃時間 t。回傳 (最後 cmd, 結束時間)。"""
    t, cmd = t0, None
    for _ in range(int(round(secs / CTRL_DT))):
        cmd = ctrl.update(sticks_fn(t), odom_fn(t), t)
        t += CTRL_DT
    return cmd, t


def still(t):
    return Odom(0.0, 0.0, 0.0, t)


def t3_neutral_stays_manual_zero():
    ctrl = RCLineController()
    cmd, _ = drive(ctrl, 1.0, lambda t: Sticks(), still)
    assert ctrl.state == MANUAL and ctrl.latch is None
    assert np.allclose(cmd, [0.0, 0.0, 0.0])


def t3_standstill_push_fwd_latch_immediate():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(), still)
    cmd = ctrl.update(Sticks(fwd=0.5), still(t), t)
    assert ctrl.state == TRACKING            # 靜止時穩定計時已滿 → 推桿瞬間 latch
    assert np.allclose(ctrl.latch[0], [0.0, 0.0]) and abs(ctrl.latch[1]) < 1e-9
    assert abs(cmd[0] - 0.3) < 1e-6 and abs(cmd[1]) < 1e-6 and abs(cmd[2]) < 1e-6


def t3_tracking_pulls_back_to_line():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)   # latch 於 y=0 線
    assert ctrl.state == TRACKING
    off = lambda tt: Odom(0.0, 0.2, 0.0, tt)                    # 向左偏 0.2m（連續）
    cmd, _ = drive(ctrl, 1.0, lambda tt: Sticks(fwd=1.0), off, t0=t)
    assert cmd[1] < -0.25                                       # vy 往右拉回（穩態 -0.3）
    assert abs(cmd[2]) < 1e-6                                   # 航向沒偏 → wz=0


def t3_turn_takeover_and_relatch():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    t_spin = t
    spin = lambda tt: Odom(0.0, 0.0, 0.8 * (tt - t_spin), tt)   # 0.8 rad/s 左轉
    cmd, t = drive(ctrl, 0.5, lambda tt: Sticks(fwd=1.0, turn=0.5), spin, t0=t)
    assert ctrl.state == MANUAL and ctrl.latch is None          # 轉向優先接管
    assert abs(cmd[0] - 0.6) < 1e-6 and abs(cmd[2] - 0.5) < 1e-6  # vx 照給、turn 透傳
    cmd, t = drive(ctrl, 0.2, lambda tt: Sticks(fwd=1.0), spin, t0=t)
    assert ctrl.state == SETTLING                               # 桿回中但航向還在滑
    assert abs(cmd[1]) < 1e-6 and abs(cmd[2]) < 1e-6            # 不透傳殘餘、不校正
    yaw1 = 0.8 * (t - t_spin)
    hold = lambda tt: Odom(0.0, 0.0, yaw1, tt)
    _, t = drive(ctrl, 0.6, lambda tt: Sticks(fwd=1.0), hold, t0=t)
    assert ctrl.state == TRACKING
    assert abs(wrap(ctrl.latch[1] - yaw1)) < 1e-9               # 新線鎖在停穩後航向


def t3_fwd_release_invalidates_line():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    assert ctrl.state == TRACKING
    ctrl.update(Sticks(), still(t), t)
    assert ctrl.state == MANUAL and ctrl.latch is None          # fwd 回中 → 線作廢
    ctrl.update(Sticks(fwd=1.0), still(t + CTRL_DT), t + CTRL_DT)
    assert ctrl.state == TRACKING                               # 再推 → 立即重 latch


def t3_backward_still_corrects():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=-1.0), still)  # 倒退也能 latch
    assert ctrl.state == TRACKING
    off = lambda tt: Odom(0.0, 0.2, 0.0, tt)
    cmd, _ = drive(ctrl, 1.0, lambda tt: Sticks(fwd=-1.0), off, t0=t)
    assert cmd[0] < -0.5                                        # vx = -0.6 倒退
    assert cmd[1] < -0.25                                       # 偏左照樣往右修


TESTS += [t3_neutral_stays_manual_zero, t3_standstill_push_fwd_latch_immediate,
          t3_tracking_pulls_back_to_line, t3_turn_takeover_and_relatch,
          t3_fwd_release_invalidates_line, t3_backward_still_corrects]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ImportError: cannot import name 'RCLineController'`

- [ ] **Step 3: 實作（附加到模組檔尾）**

```python
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
        self._prev_out = None                # (cmd, now)，Task 4 slew 用

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

    # Task 4 會把下面兩個換成含逾時/NaN/跳變與限幅/slew 的完整版
    def _ingest_odom(self, odom, now):
        if odom is None:
            return False
        if self._last_odom is None or odom.stamp > self._last_odom.stamp:
            self._yr.update(odom.yaw, odom.stamp)
            self._last_odom = odom
        return True

    def _finish(self, cmd, now):
        cmd = np.asarray(cmd, np.float32)
        self._prev_out = (cmd.copy(), now)
        return cmd
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ALL 14 TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add task5/rc_line/rc_line_controller.py task5/rc_line/tests/test_rc_line_controller.py
git commit -m "feat(rc): RCLineController 三態狀態機（latch/接管/倒退沿線）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: odom 防護（逾時／NaN／跳變）＋輸出限幅與 slew 平滑

**Files:**
- Modify: `task5/rc_line/rc_line_controller.py`（替換 `_ingest_odom` 與 `_finish`）
- Modify: `task5/rc_line/tests/test_rc_line_controller.py`（附加測試）

**Interfaces:**
- Consumes: Task 3 的 `RCLineController` 及其 `_ingest_odom/_finish` 掛點。
- Produces: 對外介面不變；新行為——odom None/NaN/逾時 → `degraded=True` 強制 MANUAL；相鄰接受樣本位置差 >`jump_pos_m` 或 yaw 差 >`jump_yaw_rad` → 作廢 latch＋重置角速度估計（之後照常 SETTLING 重鎖）；輸出 clip 到 `±vmax/±max(vymax,vy_lim)/±max(wmax,wz_lim)` 並對 vy/wz 做斜率限制。

- [ ] **Step 1: 寫失敗測試（附加）**

```python
def t4_none_or_nan_degrades():
    ctrl = RCLineController()
    ctrl.update(Sticks(fwd=1.0), None, 0.0)
    assert ctrl.state == MANUAL and ctrl.degraded
    ctrl.update(Sticks(fwd=1.0), Odom(np.nan, 0.0, 0.0, 0.02), 0.02)
    assert ctrl.state == MANUAL and ctrl.degraded


def t4_stale_degrades_then_recovers():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    assert ctrl.state == TRACKING
    frozen = Odom(0.0, 0.0, 0.0, t)                             # stamp 從此不動
    cmd, t = drive(ctrl, 1.0, lambda tt: Sticks(fwd=1.0), lambda tt: frozen, t0=t)
    assert ctrl.state == MANUAL and ctrl.degraded and ctrl.latch is None
    assert abs(cmd[0] - 0.6) < 1e-6 and abs(cmd[1]) < 1e-6      # 桿量直通
    recov = lambda tt: Odom(0.3, 0.1, 0.0, tt)                  # 恢復（未觸跳變）
    _, t = drive(ctrl, 1.0, lambda tt: Sticks(fwd=1.0), recov, t0=t)
    assert ctrl.state == TRACKING and not ctrl.degraded         # 穩定後重 latch


def t4_jump_invalidates_and_relatches():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    assert ctrl.state == TRACKING
    jumped = lambda tt: Odom(3.0, 2.0, 0.0, tt)                 # 重定位跳 3.6m
    ctrl.update(Sticks(fwd=1.0), jumped(t), t)
    assert ctrl.latch is None and ctrl.state == SETTLING        # 作廢舊線、不猛拉
    _, t = drive(ctrl, 0.6, lambda tt: Sticks(fwd=1.0), jumped, t0=t)
    assert ctrl.state == TRACKING
    assert np.allclose(ctrl.latch[0], [3.0, 2.0])               # 新線鎖在跳後位置


def t4_slew_limits_vy_step():
    ctrl = RCLineController()
    _, t = drive(ctrl, 1.0, lambda t: Sticks(fwd=1.0), still)
    cmd = ctrl.update(Sticks(fwd=1.0), Odom(0.0, 0.4, 0.0, t), t)
    assert -0.035 < cmd[1] < -0.025            # 目標 -0.3，但一步只能走 1.5*0.02=0.03


def t4_finish_clips():
    ctrl = RCLineController(Config(slew_vy=0.0, slew_wz=0.0))
    out = ctrl._finish(np.array([9.0, 9.0, -9.0], np.float32), 0.0)
    assert abs(out[0] - 0.6) < 1e-6 and abs(out[1] - 0.3) < 1e-6 and abs(out[2] + 1.0) < 1e-6


TESTS += [t4_none_or_nan_degrades, t4_stale_degrades_then_recovers,
          t4_jump_invalidates_and_relatches, t4_slew_limits_vy_step, t4_finish_clips]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: t4 系列 AssertionError（例如 `t4_none_or_nan_degrades` 過但 `t4_stale_degrades_then_recovers` 在 stale 後仍 TRACKING 而失敗；t3 全數仍過）

- [ ] **Step 3: 實作——整段替換 Task 3 版的 `_ingest_odom` 與 `_finish`**

```python
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
```

- [ ] **Step 4: 跑全部測試確認通過（含 t1–t3 未回歸）**

Run: `conda run -n rbtdog python task5/rc_line/tests/test_rc_line_controller.py`
Expected: `ALL 19 TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add task5/rc_line/rc_line_controller.py task5/rc_line/tests/test_rc_line_controller.py
git commit -m "feat(rc): odom 逾時/NaN/跳變防護＋輸出限幅與 slew 平滑

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: MuJoCo 整合驗證 `sim_demo.py`（影片＋圖表＋量測）

**Files:**
- Create: `task5/rc_line/sim_demo.py`
- Test: 實跑 demo（整合驗證，無單元測試）

**Interfaces:**
- Consumes: Task 1–4 的 `RCLineController/Sticks/Odom/line_frame/TRACKING`；`task4/inference/odom_missions.py` 的 `Runner/frame/CTRL_DT/FPS`；`task4/inference/local_infer_paper.py` 的 `load_policy/HOME12`；權重 `task4/weights/cpg_rl_paper_params.pkl`。
- Produces: `task5/rc_line/outputs/rc_demo.mp4`、`outputs/rc_demo.png`、console 量測報告。

- [ ] **Step 1: 寫 `task5/rc_line/sim_demo.py`**

```python
"""RC 外圈直線控制器 MuJoCo 整合驗證（spec §7）。

腳本化桿量時間軸驅動 RCLineController；狗 = task4 論文版 CPG-RL + 完美 odom。
時間軸：推前進(立即latch) → 前進中右轉(手動弧線) → 放桿(等穩重latch) →
        odom 掉訊 2s(退化直通) → 恢復(跳變防護→重latch) → 倒退沿線。
注意：策略訓練指令 vx∈[0,1]，倒退段超出訓練分佈，只驗證「不跌倒、e_ct 不發散」。

用法：
  MUJOCO_GL=egl conda run -n rbtdog python task5/rc_line/sim_demo.py \
      --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
from rc_line_controller import RCLineController, Sticks, Odom, line_frame, TRACKING
import odom_missions as OM
import local_infer_paper as P

CTRL_DT = OM.CTRL_DT
OUT = os.path.join(HERE, "outputs")

# (t0, t1, fwd, lat, turn, odom_ok)
TIMELINE = [
    (0.0,  8.0,  1.0, 0.0,  0.0, True),    # 推前進：立即 latch、走直線
    (8.0,  10.0, 1.0, 0.0, -0.5, True),    # 前進中右轉：手動接管走弧線
    (10.0, 18.0, 1.0, 0.0,  0.0, True),    # 放轉向：等穩 → latch 新線
    (18.0, 20.0, 1.0, 0.0,  0.0, False),   # odom 掉訊：退化直通
    (20.0, 26.0, 1.0, 0.0,  0.0, True),    # 恢復：跳變防護 → 重 latch
    (26.0, 30.0, -0.5, 0.0, 0.0, True),    # 倒退沿線（超出訓練分佈，寬鬆驗證）
]


def timeline_at(t):
    for t0, t1, f, l, tr, ok in TIMELINE:
        if t0 <= t < t1:
            return Sticks(f, l, tr), ok
    return Sticks(0.0, 0.0, 0.0), True


def cur_waypoints(latch):
    """目前 latch 線的頭尾兩點（給 OM.frame 畫地板目標線）。"""
    if latch is None:
        return [(0.0, 0.0), (0.0, 0.0)]
    p0, psi = latch
    d, _ = line_frame(psi)
    return [tuple(p0), tuple(p0 + 8.0 * d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import imageio.v2 as iio
    print("[info] 載入策略", args.params)
    policy = P.load_policy(args.params)
    r = OM.Runner("odom", policy)
    for _ in range(int(0.5 / CTRL_DT)):                          # 站立 warmup
        r.apply(P.HOME12.copy())

    writer = iio.get_writer(f"{OUT}/rc_demo.mp4", fps=OM.FPS, codec="libx264", quality=7)
    frame_iv = int((1.0 / OM.FPS) / CTRL_DT)
    ctrl = RCLineController()
    last_odom = None
    latches = []                                                 # 每次 latch 的 (p0,psi)
    log = {"t": [], "x": [], "y": [], "e_ct": [], "state": []}
    for i in range(int(30.0 / CTRL_DT)):
        t = i * CTRL_DT
        sticks, ok = timeline_at(t)
        if ok or last_odom is None:
            x, y, yaw = r.g.odom()
            last_odom = Odom(float(x), float(y), float(yaw), t)  # 掉訊時沿用舊樣本(stamp不動)
        prev_latch = ctrl.latch
        cmd = ctrl.update(sticks, last_odom, t)
        if ctrl.latch is not None and ctrl.latch is not prev_latch:
            latches.append(ctrl.latch)
        r.drive(np.asarray(cmd, np.float32))
        e_ct = np.nan
        if ctrl.latch is not None:                               # 真值位置量 e_ct（量測用）
            _, n = line_frame(ctrl.latch[1])
            e_ct = float(n @ (np.asarray(r.xy) - ctrl.latch[0]))
        log["t"].append(t); log["x"].append(float(r.xy[0])); log["y"].append(float(r.xy[1]))
        log["e_ct"].append(e_ct); log["state"].append(ctrl.state)
        if i % frame_iv == 0:
            sub = f"state={ctrl.state}" + ("  ODOM LOST" if not ok else "")
            writer.append_data(OM.frame(r, "RC line controller", (40, 90, 220),
                                        cur_waypoints(ctrl.latch), sub))
        if r.fallen:
            print("[fail] 狗跌倒於 t=%.1fs" % t); break
    writer.close(); print("[video]", f"{OUT}/rc_demo.mp4")
    report(log, latches)
    chart(log, latches)


def report(log, latches):
    e = np.array(log["e_ct"], float)
    st = np.array(log["state"])
    runs, i = [], 0
    while i < len(st):                                           # 切出連續 TRACKING 段
        if st[i] == TRACKING:
            j = i
            while j < len(st) and st[j] == TRACKING:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    print(f"[result] latch 次數={len(latches)}  tracking 段數={len(runs)}")
    ok = True
    for k, (i, j) in enumerate(runs):
        seg, dur = e[i:j], (j - i) * CTRL_DT
        tail = seg[len(seg) // 2:]
        tmax = float(np.nanmax(np.abs(tail)))
        print(f"  段{k+1}: t={log['t'][i]:.1f}s 起  長{dur:.1f}s  "
              f"max|e_ct|={np.nanmax(np.abs(seg)):.3f}m  後半 max|e_ct|={tmax:.3f}m")
        if dur >= 2.0 and log["t"][i] < 26.0 and tmax > 0.05:    # 倒退段(26s後)寬鬆
            ok = False
    print("[result]", "PASS：各段後半 |e_ct|<0.05m" if ok else "FAIL：有段落未收斂 <0.05m")


def chart(log, latches):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(log["x"], log["y"], lw=1.5, color="#3060c0", label="trajectory")
    for k, (p0, psi) in enumerate(latches):
        d, _ = line_frame(psi)
        seg = np.array([p0 - 1.0 * d, p0 + 8.0 * d])
        a1.plot(seg[:, 0], seg[:, 1], "k--", lw=1.0,
                label="latched line" if k == 0 else None)
        a1.plot([p0[0]], [p0[1]], "k^", ms=6)
    a1.set_xlabel("x (m)"); a1.set_ylabel("y (m)"); a1.set_aspect("equal", "box")
    a1.set_title("RC line controller trajectory"); a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2.plot(log["t"], log["e_ct"], lw=1.2, color="#c04030")
    a2.axhline(0, color="gray", ls=":", lw=1)
    a2.set_xlabel("t (s)"); a2.set_ylabel("e_ct (m)")
    a2.set_title("cross-track error (NaN = not tracking)"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/rc_demo.png", dpi=120)
    print("[chart]", f"{OUT}/rc_demo.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 實跑 demo**

Run:
```bash
MUJOCO_GL=egl conda run -n rbtdog python task5/rc_line/sim_demo.py \
    --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl
```
Expected: 印出 `[video] .../rc_demo.mp4`、`[chart] .../rc_demo.png`、`[result] PASS：各段後半 |e_ct|<0.05m`，且全程無 `[fail] 狗跌倒`。latch 次數應為 3（~0.3s：冷啟動需先累積穩定判斷、非 0s 立即；~10.3s 轉完；~20.3s 恢復後觸發跳變防護再重鎖；倒退段沿用第 3 條線不重 latch）。

- [ ] **Step 3: 人工檢視輸出**

看 `outputs/rc_demo.png`：軌跡應貼合三條虛線 latch 線；e_ct 在轉彎/掉訊段為 NaN（斷線）、TRACKING 段收斂到 0。影片抽查：轉彎段字幕 `state=manual`、掉訊段顯示 `ODOM LOST`。若倒退段策略走不動（訓練分佈外），只要不跌倒、e_ct 不發散即可接受，並在 commit 訊息註明。

- [ ] **Step 4: Commit**

```bash
git add task5/rc_line/sim_demo.py
git commit -m "feat(rc): MuJoCo 整合驗證 demo（桿量時間軸+影片+e_ct 量測）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 驗收總表（對照 spec §7）

| spec 要求 | 對應 |
|---|---|
| 狀態轉移表 | t3_* 全系列 |
| 靜止推 fwd 立即 latch / 轉後等穩 | t3_standstill_push_fwd_latch_immediate / t3_turn_takeover_and_relatch |
| 掉訊→直通→恢復重 latch | t4_stale_degrades_then_recovers |
| 跳變→SETTLING 重鎖 | t4_jump_invalidates_and_relatches |
| 倒退沿線符號 | t3_backward_still_corrects |
| 死區遲滯 / 輸出限幅 | t2_stick_gate_hysteresis / t4_finish_clips、t4_slew_limits_vy_step |
| MuJoCo 整合 + 成功標準 | Task 5（|e_ct| 後半 <0.05m、不跌倒、掉訊直通） |
