# odom 絕對直線行走 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 CPG-RL（`local_infer_paper.py`）用完美 odom 回授走絕對直線——腳本先右轉 45°，latch 當下位置+航向為目標線，再沿該射線直走，側飄與航向誤差同時趨近 0。

**Architecture:** 把感測器從 magnetometer 換成 `framepos`（完美 odom，預設零偏移）；新增解耦控制律——`wz` 用航向誤差鎖航向、`vy`（policy 已訓練的螃蟹步 ±0.3）用 cross-track 誤差滑回線上；腳本化 mission 分「轉向 → latch → 直走」兩階段。

**Tech Stack:** Python, MuJoCo 3.10（透過 `MjSpec` 建模）, NumPy, JAX/Brax（載入 PPO 權重）, Matplotlib（軌跡圖）。conda 環境 `rbtdog`。

## Global Constraints

- 執行環境固定用 conda：所有指令前綴 `conda run -n rbtdog`；需要渲染時加 `MUJOCO_GL=egl`。專案無 pytest，測試一律寫成獨立 `assert` 腳本，用 `conda run -n rbtdog python <file>` 執行，PASS 時印 `PASS <name>`。
- 模型由 `task3/go2_model.py::make_model()` 以 `MjSpec` 程式化建立；`task3/go2_imu_scene.xml` 為未被任何程式引用的宣告式鏡像檔（功能性變更必須改 `go2_model.py`，xml 僅同步以保文件一致）。
- policy 指令空間（訓練時）：`vx∈[0,1]`、`vy∈[-0.3,0.3]`、`wz∈[-1,1]`。所有指令 clip 必須落在此範圍。
- 轉角約定：`--turn_deg` 為帶號相對轉角，**右轉 45° = `-45`**（世界系正 yaw = CCW = 左轉）。
- odom 預設完美：`odom_xy_bias=(0.0,0.0)`、`odom_yaw_bias=0.0`。
- 控制律預設增益：`K_YAW=3.0`（沿用現有 `HEADING_GAIN`）、`K_CT=1.5`；`vy` clip ±0.3、`wz` clip ±1.0。
- 沿用既有座標/工具函式：`wrap()`、`build_obs()`、`act_to_cmd()`、`cpg_step()`、`joint_targets()`、`apply()`、`push_at()` 不改變其介面。

---

## File Structure

- **Modify** `task3/go2_model.py` — `make_model()` 把 `imu_mag`(magnetometer) 換成 `odom_pos`(framepos)。
- **Modify** `task3/go2_gait.py` — 建構子加 odom bias 參數；新增 `odom()` 方法。
- **Modify** `task3/go2_imu_scene.xml` — 同步宣告（非功能性，僅保文件一致）。
- **Modify** `task4/inference/local_infer_paper.py` — 新增純函式 `line_frame()`/`line_control()`；重寫 `run()` 為兩階段 mission + 量測 + 軌跡圖；新增 CLI 參數。
- **Create** `task3/tests/test_odom.py` — odom 感測器與 `odom()` 單元測試。
- **Create** `task4/inference/tests/test_line_control.py` — 控制律純函式單元測試。
- **Create** `task4/inference/tests/check_vy_sign.py` — （需權重）`vy` 符號整合驗證。

---

## Task 1: odom 感測器 + `Go2Gait.odom()`

**Files:**
- Modify: `task3/go2_model.py`（`make_model()` 內 `add("imu_mag", …)` 那行）
- Modify: `task3/go2_gait.py`（`__init__` 簽章與內文；新增 `odom()`）
- Modify: `task3/go2_imu_scene.xml`（sensor 區塊，文件同步）
- Test: `task3/tests/test_odom.py`

**Interfaces:**
- Produces:
  - 感測器 `odom_pos`（framepos, dim 3, objtype=site, objname=imu），移除 `imu_mag`。
  - `Go2Gait.__init__(..., odom_xy_bias=(0.0,0.0), odom_yaw_bias=0.0)`
  - `Go2Gait.odom() -> (x: float, y: float, yaw: float)`，世界系，含 bias（預設 0）。
- Consumes: 既有 `self.sensor(name)`、`self.true_yaw()`、模組級 `wrap()`。

- [ ] **Step 1: 寫失敗測試**

Create `task3/tests/test_odom.py`:

```python
"""odom 感測器與 Go2Gait.odom() 單元測試（無 pytest，直接 assert）。
run: conda run -n rbtdog python task3/tests/test_odom.py"""
import sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
from go2_gait import Go2Gait, wrap
from walk_line import GAIT


def main():
    g = Go2Gait(**GAIT); g.reset()
    assert "odom_pos" in g._sadr, "odom_pos 感測器不存在"
    assert "imu_mag" not in g._sadr, "magnetometer 應已移除"

    x, y, yaw = g.odom()
    # 航向與 true_yaw（同一顆四元數）應完全一致
    assert abs(wrap(yaw - g.true_yaw())) < 1e-9, (yaw, g.true_yaw())
    # 零偏移下位置貼近 base（imu site 僅約 2.6cm 偏移）
    assert abs(x - g.d.qpos[0]) < 0.1 and abs(y - g.d.qpos[1]) < 0.1, (x, y, g.d.qpos[:2])

    # 偏移注入：差值等於 bias
    g2 = Go2Gait(**GAIT, odom_xy_bias=(1.0, -2.0), odom_yaw_bias=0.1); g2.reset()
    x2, y2, yaw2 = g2.odom()
    assert abs((x2 - x) - 1.0) < 1e-6 and abs((y2 - y) + 2.0) < 1e-6, (x2, y2)
    assert abs(wrap(yaw2 - (yaw + 0.1))) < 1e-6, (yaw2, yaw)
    print("PASS test_odom")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `conda run -n rbtdog python task3/tests/test_odom.py`
Expected: FAIL — `AssertionError: odom_pos 感測器不存在`（或 `AttributeError: 'Go2Gait' object has no attribute 'odom'`）。

- [ ] **Step 3: `go2_model.py` 換上 framepos 感測器**

在 `task3/go2_model.py` 的 `make_model()` 內，將：

```python
    add("imu_mag",  mujoco.mjtSensor.mjSENS_MAGNETOMETER)
```

改為：

```python
    add("odom_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS)     # 完美里程計位置（取代羅盤）
```

- [ ] **Step 4: `go2_gait.py` 加 odom bias 參數與 `odom()` 方法**

4a. 修改 `__init__` 簽章（`task3/go2_gait.py` 第 18-20 行）為：

```python
    def __init__(self, freq=3.0, duty=0.55, lift=0.12, stride=0.22,
                 turn_gain=0.10, z0=-0.28, kp=70.0, kd=1.5,
                 imu_heading_rms_deg=0.5, imu_heading_tau=20.0, imu_seed=0,
                 odom_xy_bias=(0.0, 0.0), odom_yaw_bias=0.0):
```

4b. 在 `__init__` 內、`self.reset()` 之前（緊接 `self.imu_seed = imu_seed` 那行後）新增：

```python
        self._odom_xy_bias = np.asarray(odom_xy_bias, dtype=float)
        self._odom_yaw_bias = float(odom_yaw_bias)
```

4c. 在 `true_yaw()` 方法（第 126-129 行）之後新增：

```python
    def odom(self):
        """完美里程計（取代羅盤）：回傳世界系 (x, y, yaw)。
        位置取自 imu site 的 framepos，航向由 framequat 解算；bias 預設 0，可注入偏移做實驗。"""
        x, y, _ = self.sensor("odom_pos")
        w, xx, yy, zz = self.sensor("imu_quat")
        yaw = np.arctan2(2 * (w * zz + xx * yy), 1 - 2 * (yy * yy + zz * zz))
        bx, by = self._odom_xy_bias
        return float(x + bx), float(y + by), wrap(yaw + self._odom_yaw_bias)
```

- [ ] **Step 5: 執行測試確認通過**

Run: `conda run -n rbtdog python task3/tests/test_odom.py`
Expected: PASS — 輸出 `PASS test_odom`。

- [ ] **Step 6: 同步宣告式 xml（文件一致，非功能性）**

在 `task3/go2_imu_scene.xml` 把：

```xml
    <magnetometer  name="imu_mag"  site="imu"/>   <!-- 3 軸磁力計 = 羅盤 -->
```

改為：

```xml
    <framepos      name="odom_pos" objtype="site" objname="imu"/>   <!-- 完美里程計位置（取代羅盤） -->
```

並把第 5 行註解 `<!-- 9 軸 IMU（含 compass）掛在機身內建的 imu site 上 -->` 改為
`<!-- IMU + 完美 odom（framepos）掛在機身內建的 imu site 上；本檔為 go2_model.py 的宣告式鏡像 -->`。

- [ ] **Step 7: Commit**

```bash
git add task3/go2_model.py task3/go2_gait.py task3/go2_imu_scene.xml task3/tests/test_odom.py
git commit -m "feat(odom): framepos 完美 odom 取代 magnetometer + Go2Gait.odom()"
```

---

## Task 2: 線追蹤控制律純函式

**Files:**
- Modify: `task4/inference/local_infer_paper.py`（在 `def wrap(a): …` 第 98 行之後新增兩個純函式）
- Test: `task4/inference/tests/test_line_control.py`

**Interfaces:**
- Produces:
  - `line_frame(psi) -> (d: np.ndarray[2], n: np.ndarray[2])`：目標線方向 `d` 與左法向 `n`（世界系單位向量）。
  - `line_control(p, yaw, p0, psi_target, vx, k_yaw, k_ct, no_lateral=False) -> (cmd: np.ndarray[3] float32, e_ct: float, e_yaw: float)`：方案 A 解耦控制，`cmd=[vx, vy, wz]`。
- Consumes: 模組級 `wrap()`（已存在，第 98 行）。

- [ ] **Step 1: 寫失敗測試**

Create `task4/inference/tests/test_line_control.py`:

```python
"""線追蹤控制律純函式單元測試。
run: conda run -n rbtdog python task4/inference/tests/test_line_control.py"""
import sys
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
from local_infer_paper import line_frame, line_control

K_YAW, K_CT = 3.0, 1.5


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def main():
    psi = np.pi / 4
    d, n = line_frame(psi)
    assert approx(np.hypot(*d), 1.0) and approx(np.hypot(*n), 1.0)
    assert approx(float(d @ n), 0.0)                       # 方向與法向正交

    p0 = np.array([0.0, 0.0])
    # 在線上且航向對齊 → 零橫向、零轉向、前進保留
    cmd, e_ct, e_yaw = line_control(p0, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert approx(e_ct, 0.0) and approx(e_yaw, 0.0)
    assert approx(float(cmd[1]), 0.0) and approx(float(cmd[2]), 0.0) and approx(float(cmd[0]), 0.6)

    # 偏左（+n 方向 0.1m）→ e_ct>0 → vy<0（螃蟹往右修回）
    cmd, e_ct, _ = line_control(p0 + 0.1 * n, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert e_ct > 0 and cmd[1] < 0 and approx(float(cmd[1]), -0.15, 1e-3), (e_ct, cmd[1])

    # 偏右 → e_ct<0 → vy>0
    cmd, e_ct, _ = line_control(p0 - 0.1 * n, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert e_ct < 0 and cmd[1] > 0

    # 大偏移 → vy 夾到 -0.3
    cmd, _, _ = line_control(p0 + 1.0 * n, psi, p0, psi, 0.6, K_YAW, K_CT)
    assert approx(float(cmd[1]), -0.3)

    # 航向偏左（yaw>target, e_yaw>0）→ wz<0（順時針轉回）
    cmd, _, e_yaw = line_control(p0, psi + 0.1, p0, psi, 0.6, K_YAW, K_CT)
    assert e_yaw > 0 and cmd[2] < 0

    # no_lateral → vy 恆 0（重現舊的只鎖航向行為）
    cmd, _, _ = line_control(p0 + 0.5 * n, psi, p0, psi, 0.6, K_YAW, K_CT, no_lateral=True)
    assert approx(float(cmd[1]), 0.0)

    print("PASS test_line_control")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `conda run -n rbtdog python task4/inference/tests/test_line_control.py`
Expected: FAIL — `ImportError: cannot import name 'line_frame'`。

- [ ] **Step 3: 新增純函式**

在 `task4/inference/local_infer_paper.py` 的 `def wrap(a): return np.arctan2(np.sin(a), np.cos(a))`（第 98 行）之後新增：

```python
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
    e_yaw = wrap(yaw - psi_target)
    wz = float(np.clip(-k_yaw * e_yaw, -1.0, 1.0))
    vy = 0.0 if no_lateral else float(np.clip(-k_ct * e_ct, -0.3, 0.3))
    return np.array([vx, vy, wz], np.float32), e_ct, e_yaw
```

- [ ] **Step 4: 執行測試確認通過**

Run: `conda run -n rbtdog python task4/inference/tests/test_line_control.py`
Expected: PASS — 輸出 `PASS test_line_control`。

- [ ] **Step 5: Commit**

```bash
git add task4/inference/local_infer_paper.py task4/inference/tests/test_line_control.py
git commit -m "feat(control): 解耦線追蹤控制律純函式 line_frame/line_control"
```

---

## Task 3: 兩階段 mission 接線 + 量測 + 軌跡圖

**Files:**
- Modify: `task4/inference/local_infer_paper.py`（頂部常數；`run()` 主體第 166 行起到結尾；`argparse` 區塊）
- Test: `task4/inference/tests/check_vy_sign.py`（新建，需權重的整合驗證）

**Interfaces:**
- Consumes: Task 1 的 `g.odom()`；Task 2 的 `line_frame()`/`line_control()`；既有 `build_obs/act_to_cmd/cpg_step/joint_targets/apply/push_at`。
- Produces: `local_infer_paper.py` 新 CLI 參數 `--turn_deg`(-45.0)、`--turn_vx`(0.0)、`--turn_timeout`(6.0)、`--k_ct`(1.5)、`--no_lateral`；輸出軌跡圖 `task4/outputs/odom_line[_nolat].png` 與（`--video` 時）`task4/outputs/odom_line[_nolat].mp4`。

- [ ] **Step 1: 移除只鎖航向的舊常數**

在 `task4/inference/local_infer_paper.py` 頂部把（第 31-34 行）：

```python
# 羅盤走直線
TARGET_YAW = 0.0
HEADING_GAIN = 3.0
VX_CMD = 0.6
```

改為：

```python
# odom 走直線
HEADING_GAIN = 3.0      # 航向 P 增益（= 控制律 K_YAW 預設）
VX_CMD = 0.6
```

- [ ] **Step 2: 重寫 `run()` 的推論主體（第 166 行到 `run()` 結尾）**

把從 `c = cpg_init(); last_a = np.zeros(ACT_DIM); traj = []; x0, y0 = g.xy`（第 166 行）
一直到 `iio.mimsave(...); print("[result] 影片:", out)`（第 191 行）**整段**替換為：

```python
    c = cpg_init(); last_a = np.zeros(ACT_DIM)
    fl_gid = mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    k_yaw = HEADING_GAIN

    def step_policy(cmd):
        nonlocal c, last_a
        obs = build_obs(g, c, cmd.astype(np.float32), last_a, foot_gid)
        act = infer(obs)
        mux, muy, om = act_to_cmd(act); c = cpg_step(c, mux, muy, om, CTRL_DT)
        apply(joint_targets(c, f0s, jinvs)); last_a = act

    def maybe_render(k):
        if ren is not None and k % 2 == 0:
            x, y, _ = g.odom(); cam.lookat[:] = [x, y, 0.3]; cam.distance = 2.5
            cam.elevation = -20; cam.azimuth = 90
            ren.update_scene(g.d, cam); frames.append(ren.render())

    # ---- Phase 1：轉向到 psi_goal（右轉 → turn_deg 為負）----
    _, _, start_yaw = g.odom()
    psi_goal = wrap(start_yaw + np.radians(args.turn_deg))
    settled = 0; kframe = 0; turn_t = 0.0
    for i in range(int(args.turn_timeout / CTRL_DT)):
        _, _, yaw = g.odom()
        e = wrap(yaw - psi_goal)
        wz = float(np.clip(-k_yaw * e, -1.0, 1.0))
        step_policy(np.array([args.turn_vx, 0.0, wz], np.float32))
        maybe_render(kframe); kframe += 1
        turn_t = (i + 1) * CTRL_DT
        settled = settled + 1 if abs(e) < np.radians(2) else 0
        if settled >= int(0.3 / CTRL_DT):        # 穩住 0.3s 視為到位
            break

    # ---- Latch：鎖定當下 odom 位置+航向為目標線 ----
    x0, y0, psi_target = g.odom()
    p0 = np.array([x0, y0]); d_hat, _ = line_frame(psi_target)
    print(f"[latch] 轉向完成 psi_target={np.degrees(psi_target):+.1f}° "
          f"起點=({x0:+.2f},{y0:+.2f}) 轉向耗時={turn_t:.1f}s")

    # ---- Phase 2：沿目標線直走 ----
    traj = []; ects = []; eyaws = []; fzmin = fzmax = None; fell = None
    for i in range(int(args.secs / CTRL_DT)):
        t = i * CTRL_DT; push_at(t)
        x, y, yaw = g.odom(); p = np.array([x, y])
        cmd, e_ct, e_yaw = line_control(p, yaw, p0, psi_target, args.vx,
                                        k_yaw, args.k_ct, args.no_lateral)
        step_policy(cmd)
        traj.append(p.copy()); ects.append(e_ct); eyaws.append(e_yaw)
        fz = g.d.geom_xpos[fl_gid][2]
        fzmin = fz if fzmin is None else min(fzmin, fz)
        fzmax = fz if fzmax is None else max(fzmax, fz)
        if g.height < 0.15 and fell is None: fell = t
        maybe_render(kframe); kframe += 1

    push_at(1e9)                                  # 收尾清掉殘留外力
    traj = np.array(traj); ects = np.array(ects); eyaws = np.array(eyaws)
    fwd = float(d_hat @ (traj[-1] - p0))
    max_ct = float(np.max(np.abs(ects))); fin_ct = float(ects[-1])
    yaw_rms = float(np.degrees(np.sqrt(np.mean(eyaws ** 2))))
    print(f"[result] 沿線前進={fwd:+.2f}m  max|側偏|={max_ct:.3f}m  末端側偏={fin_ct:+.3f}m  "
          f"航向RMS={yaw_rms:.2f}°  跌倒={'是@%.1fs' % fell if fell else '否'}")
    print(f"[result] FL 抬腳量 ≈ {fzmax - fzmin:.3f} m  (no_lateral={args.no_lateral})")

    tag = "_nolat" if args.no_lateral else ""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    L = max(fwd, 1.0) + 0.5
    seg = np.stack([p0, p0 + L * d_hat])
    ax.plot(seg[:, 0], seg[:, 1], "k--", lw=1, label="target line")
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="path")
    ax.plot([p0[0]], [p0[1]], "go", ms=8, label="latch/start")
    ax.set_aspect("equal"); ax.grid(True); ax.legend()
    ax.set_title(f"odom straight turn={args.turn_deg:+.0f}deg "
                 f"max|ct|={max_ct:.3f}m yawRMS={yaw_rms:.2f}deg"
                 + ("  [no_lateral]" if args.no_lateral else ""))
    pout = f"/home/huang/rbtdog_sim/task4/outputs/odom_line{tag}.png"
    fig.savefig(pout, dpi=120, bbox_inches="tight"); print("[result] 軌跡圖:", pout)

    if frames:
        import imageio.v2 as iio
        out = f"/home/huang/rbtdog_sim/task4/outputs/odom_line{tag}.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264"); print("[result] 影片:", out)
```

- [ ] **Step 3: 新增 CLI 參數**

在 `argparse` 區塊（`ap.add_argument("--push", ...)` 之後、`ap.add_argument("--w_coup", ...)` 之前）新增：

```python
    ap.add_argument("--turn_deg", type=float, default=-45.0,
                    help="開走前的相對轉角(度)，右轉為負；預設右轉 45°")
    ap.add_argument("--turn_vx", type=float, default=0.0, help="轉向階段的前進指令(原地轉=0)")
    ap.add_argument("--turn_timeout", type=float, default=6.0, help="轉向階段秒數上限")
    ap.add_argument("--k_ct", type=float, default=1.5, help="cross-track P 增益(橫向修正)")
    ap.add_argument("--no_lateral", action="store_true",
                    help="關閉 vy 橫向修正(vy≡0)，重現舊的只鎖航向行為做 A/B 對照")
```

- [ ] **Step 4: 管線 smoke test（dummy，不需權重）**

Run: `conda run -n rbtdog bash -c "MUJOCO_GL=egl python task4/inference/local_infer_paper.py --dummy --turn_deg -45 --secs 6"`
Expected: 正常結束，印出一行 `[latch] 轉向完成 psi_target=...°` 與一行 `[result] 沿線前進=... max|側偏|=... 航向RMS=...°`，並印 `[result] 軌跡圖: .../odom_line.png`；`task4/outputs/odom_line.png` 檔案存在。（dummy 為固定動作、不理會 cmd，故不會真的走直線，只驗證兩階段管線與量測不崩。）

- [ ] **Step 5: 加權主情境驗證（需權重）**

Run:
```bash
conda run -n rbtdog bash -c "MUJOCO_GL=egl python task4/inference/local_infer_paper.py \
  --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl \
  --turn_deg -45 --secs 20 --video"
```
Expected: `[latch]` 顯示 `psi_target≈-45°`（右轉到位）；`[result]` 的 `max|側偏|` 應偏小、`航向RMS` 應偏小、`跌倒=否`；產出 `odom_line.png` 與 `odom_line.mp4`。
判讀：若 `max|側偏|` 隨時間**發散**（狗越走越偏離目標線）而非收斂，代表 policy 的 `vy` 方向與假設相反 → 進行 Step 6 翻轉符號；若側偏收斂在小範圍即為正確，跳過 Step 6。

- [ ] **Step 6:（條件式）`vy` 符號驗證與修正**

只有在 Step 5 觀察到側偏發散時才做。先建 `task4/inference/tests/check_vy_sign.py`：

```python
"""vy 符號整合驗證（需權重）：固定命令 +vy，量測 body 左向位移。
+vy 應使機身往 body 左側(+n)移動 → 我們的控制律 vy=-k_ct·e_ct 才會收斂。
run: conda run -n rbtdog bash -c "MUJOCO_GL=egl python task4/inference/tests/check_vy_sign.py \
       --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl"
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import mujoco
from go2_gait import Go2Gait
from walk_line import GAIT
import local_infer_paper as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True)
    args = ap.parse_args()
    g = Go2Gait(**GAIT); g.reset()
    f0s, jinvs = L.leg_ik_consts(g.m)
    foot_gid = [mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in L.LEGS]
    n_sub = int(round(L.CTRL_DT / g.m.opt.timestep))
    infer = L.load_policy(args.params)

    def apply(q):
        for _ in range(n_sub):
            tau = g.kp * (q - g.d.qpos[7:19]) - g.kd * g.d.qvel[6:18]
            g.d.ctrl[:] = np.clip(tau, -g.flimit, g.flimit); mujoco.mj_step(g.m, g.d)

    c = L.cpg_init(); last_a = np.zeros(L.ACT_DIM)
    for _ in range(int(0.5 / L.CTRL_DT)): apply(L.HOME12.copy())
    x0, y0, psi0 = g.odom()
    for _ in range(int(3.0 / L.CTRL_DT)):
        cmd = np.array([0.0, 0.3, 0.0], np.float32)   # 只命令 +vy
        obs = L.build_obs(g, c, cmd, last_a, foot_gid); act = infer(obs)
        mux, muy, om = L.act_to_cmd(act); c = L.cpg_step(c, mux, muy, om, L.CTRL_DT)
        apply(L.joint_targets(c, f0s, jinvs)); last_a = act
    x1, y1, _ = g.odom()
    n = np.array([-np.sin(psi0), np.cos(psi0)])       # body 左 ≈ 世界左法向
    left = float(n @ np.array([x1 - x0, y1 - y0]))
    print(f"[vy-sign] +vy 3s 後 body 左向位移={left:+.3f}m "
          f"→ {'左(符合假設, 控制律不需改)' if left > 0.02 else '右(需翻轉: line_control 內 vy 改為 +k_ct*e_ct)'}")


if __name__ == "__main__":
    main()
```

Run 之。若輸出為「右(需翻轉)」，則在 `local_infer_paper.py` 的 `line_control()` 內把
`vy = 0.0 if no_lateral else float(np.clip(-k_ct * e_ct, -0.3, 0.3))`
改為 `... float(np.clip(k_ct * e_ct, -0.3, 0.3))`，並同步更新 `test_line_control.py` 的
偏左/偏右斷言方向，重跑 Task 2 Step 4 與本 Task Step 5 確認收斂。

- [ ] **Step 7: A/B 對照驗證（需權重）**

Run:
```bash
conda run -n rbtdog bash -c "MUJOCO_GL=egl python task4/inference/local_infer_paper.py \
  --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl \
  --turn_deg -45 --secs 20 --no_lateral"
```
Expected: 產出 `odom_line_nolat.png`；其 `max|側偏|` 應**明顯大於** Step 5（開啟 vy）的值，量化證明 `vy` 橫向修正壓下了側飄。將兩者 `max|側偏|` 記入 commit message 或報告。

- [ ] **Step 8: Commit**

```bash
git add task4/inference/local_infer_paper.py task4/inference/tests/check_vy_sign.py
git commit -m "feat(mission): odom 兩階段直線行走(turn→latch→straight)+量測+軌跡圖"
```

---

## Self-Review

**Spec coverage：**
- odom 感測器（framepos, 完美, bias 預設 0）→ Task 1 ✓（含 spec §4.1 修正：功能改在 `go2_model.py`，xml 同步）。
- 控制律方案 A（wz 鎖航向 / vy 修 cross-track）→ Task 2（純函式）+ Task 3（接線）✓。
- 腳本化 mission turn→latch→straight、`--turn_deg -45` 帶號 → Task 3 ✓。
- 量測（max/末端 cross-track、航向 RMS）+ 軌跡圖 + `--video` → Task 3 ✓。
- `--no_lateral` A/B 對照 → Task 2/Task 3 + Step 7 ✓。
- `vy` 符號不臆測、實測定案 → Task 3 Step 5/6 ✓。
- spec §7 待確認項：`vy` 符號（Step 6）、`K_CT`（`--k_ct` 預設 1.5，可調）、Phase 1 `vx`（`--turn_vx` 預設 0）✓。

**Placeholder scan：** 無 TBD/TODO；每個 code step 皆含完整程式碼；條件式 Step 6 有明確觸發條件與完整腳本。

**Type consistency：** `odom()` 回傳 `(x,y,yaw)` 三元組，Task 3 全程以此解包；`line_control()` 回傳 `(cmd, e_ct, e_yaw)`，Task 2 測試與 Task 3 主體一致；`line_frame()` 回傳 `(d, n)`，`run()` 取 `d_hat, _`、`line_control` 取 `_, n`，一致。
