# CPG-RL Terrain v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 訓練一個新的 CPG-RL 地形版模型，支援全向指令追蹤（平移/轉向）、崎嶇路面（粗糙度漸變凹凸）+ 0–15° 斜坡、每腿可學抬腳高度；並確保新（抬腳可學）與舊（抬腳寫死）兩種模型都能在同一套地形上跑實驗。

**Architecture:** 把「可本機測試的核心邏輯」抽成 `task4/inference/` 下的模組（terrain2 / cpg2 / obs2 / go2_terrain2_env），用 assert 腳本在本機 CPU（含 MJX）驗證；`local_infer_terrain2.py` 直接 import 這些模組並自動偵測模型版本；最後把這些**已驗證的模組原始碼原封嵌入** Colab notebook（Colab 需自包含）加上 DR/PPO 進行 GPU 訓練。

**Tech Stack:** MuJoCo 3.10 / MJX / JAX（本機 CPU、Colab GPU）/ Brax PPO / numpy / imageio。conda env `rbtdog`。

## Global Constraints

- 執行環境一律 `conda run -n rbtdog python ...`；本機 JAX 僅 CPU。
- 測試風格：standalone assert 腳本，含 `main()` 與 `if __name__=="__main__"`，成功印 `PASS <name>`；放 `task4/inference/tests/`；run: `conda run -n rbtdog python task4/inference/tests/<file>.py`。
- 動作/觀測：新模型 **action=16 / obs=80**；舊模型 **action=12 / obs=76**。兩者只差 (a) policy 輸出維度、(b) `last_action` 長度、(c) CPG 抬腳固定 `G_C=0.08` vs 每腿 `gc=action[:,3]`。其餘完全相同。
- CPG 常數：`MU_MIN,MU_MAX=1,2`；`OMEGA_MIN,OMEGA_MAX=0,4.5`；`A_CONV=50`；`D_STEP=0.12`；`G_P=0.01`；`W_COUP=8`；`N_CPG_SUB=4`；`PHASE_OFFSET=[0,π,π,0]`。
- 抬腳可學範圍：`GC_MIN,GC_MAX=0.03,0.15`；固定值 `G_C=0.08`。
- 地形：x∈[−6,6]、y∈[−3,3]；平台 |x|<1 平滑平地；斜坡 0→15°遞增（5°→10°→15°，最陡 15°）；凹凸 `AMP_MAX=0.08`（8cm），`amp(x)=AMP_MAX·clip((|x|−1)/2,0,1)`。
- 觸地/rel_h/跌倒判定一律用 2D 地面高度 `gz(x,y)`（雙線性內插，與幾何同源，`mj_ray` 誤差 < 0.02）。
- reward：純速度指令追蹤，**移除 y_pen**，**不引入任何朝向/odom 觀測**。
- 控制：`CTRL_DT=0.02, SIM_DT=0.004`；PD kp=90/kd=3，力矩上限（膝 45.43、其餘 23.7）。
- 場景：訓練 `mujoco_menagerie/unitree_go2/scene_mjx.xml`；本機推論 `mujoco_menagerie/unitree_go2/scene.xml`。
- 不動 v1 檔案（`cpg_rl_terrain_colab.ipynb` / `cpg_rl_terrain_params.pkl` / `local_infer_terrain.py`）。
- Spec：`task4/docs/superpowers/specs/2026-07-16-cpg-rl-terrain2-design.md`。

---

### Task 1: 地形模組 terrain2（hfield 建模 + 雙線性 gz）

**Files:**
- Create: `task4/inference/terrain2.py`
- Test: `task4/inference/tests/test_terrain2.py`

**Interfaces:**
- Produces:
  - `KNOTS_X, KNOTS_Z: np.ndarray` — 斜坡折點。
  - `slope_z(x)`, `amp_at(x)`, `bump(x,y)` — 純函式（np broadcast）。
  - `XS, YS: np.ndarray`、`H: np.ndarray(shape=(len(YS),len(XS)))` — 高度網格（模組載入時建好）。
  - `gz_np(x, y) -> float|np.ndarray` — 對 `H` 雙線性內插（np）。
  - `gz_from(xp, xs, ys, Hg, x, y)` — array-agnostic 版（傳 `np` 或 `jax.numpy`），供 env 用 jnp 呼叫。
  - `build_terrain2_model(scene_path) -> mujoco.MjModel` — floor 改 hfield、加安全底網。

- [ ] **Step 1: 寫失敗測試** `task4/inference/tests/test_terrain2.py`

```python
"""terrain2 幾何/gz 測試。run: conda run -n rbtdog python task4/inference/tests/test_terrain2.py"""
import sys, numpy as np, mujoco
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import terrain2 as T


def main():
    # 平台平滑：|x|<1 高度=0、振幅=0
    assert abs(T.slope_z(0.0)) < 1e-9 and abs(T.slope_z(0.5)) < 1e-9
    assert T.amp_at(0.0) == 0.0 and T.amp_at(0.5) == 0.0
    # 粗糙度漸變：中段~一半、遠端=AMP_MAX(0.08)
    assert abs(T.amp_at(2.0) - 0.04) < 1e-6, T.amp_at(2.0)
    assert abs(T.amp_at(3.0) - 0.08) < 1e-6 and abs(T.amp_at(6.0) - 0.08) < 1e-6
    # 斜坡最陡 15°：相鄰折點最大斜率 tan(15°)
    dz = np.diff(T.KNOTS_Z); dx = np.diff(T.KNOTS_X)
    assert abs(np.max(np.abs(dz/dx)) - np.tan(np.radians(15.0))) < 1e-6
    # gz 在平台≈0
    assert abs(float(T.gz_np(0.0, 0.0))) < 1e-6
    # ★ 幾何 == gz：mj_ray 打表面對照雙線性 gz
    m = T.build_terrain2_model("mujoco_menagerie/unitree_go2/scene_mjx.xml")
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    bad = 0
    for x in np.arange(-5.5, 5.6, 0.5):
        for y in [-2.0, 0.0, 2.0]:
            gid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(m, d, np.array([x, y, 5.0]), np.array([0, 0, -1.0]),
                                 None, 1, -1, gid)
            surf = 5.0 - dist
            if abs(surf - float(T.gz_np(x, y))) > 0.02:
                bad += 1
    assert bad == 0, f"幾何/gz 不一致點數={bad}"
    print("PASS test_terrain2")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task4/inference/tests/test_terrain2.py`
Expected: FAIL（`ModuleNotFoundError: No module named 'terrain2'`）

- [ ] **Step 3: 實作 `task4/inference/terrain2.py`**

```python
"""地形 v2：統一 hfield（平台 + 0–15° 斜坡 + 粗糙度漸變凹凸）與雙線性 gz。"""
import numpy as np
import mujoco

# --- 幾何參數（見 spec §3）---
PLATFORM_HALF = 1.0
TERR_X_MAX = 6.0
TERR_WY = 3.0
AMP_MAX = 0.08
_d1 = 1.5 * np.tan(np.radians(5.0))
_d2 = 1.5 * np.tan(np.radians(10.0))
_d3 = 2.0 * np.tan(np.radians(15.0))
KNOTS_X = np.array([-6.0, -4.0, -2.5, -1.0, 1.0, 2.5, 4.0, 6.0], np.float64)
KNOTS_Z = np.array([-(_d1 + _d2 + _d3), -(_d1 + _d2), -_d1, 0.0,
                    0.0, _d1, _d1 + _d2, _d1 + _d2 + _d3], np.float64)


def slope_z(x):
    return np.interp(x, KNOTS_X, KNOTS_Z)


def amp_at(x):
    return AMP_MAX * np.clip((np.abs(x) - PLATFORM_HALF) / 2.0, 0.0, 1.0)


def bump(x, y):
    # 多正弦疊加，正規化到 ~[-1,1]；確定性（幾何靜態）
    s = (np.sin(2.1 * x) * np.cos(1.7 * y)
         + 0.5 * np.sin(3.7 * x + 1.0) * np.cos(2.9 * y + 0.5)
         + 0.3 * np.sin(5.3 * x + 2.0) * np.cos(4.1 * y))
    return s / 1.8


def build_height_grid(ncol=161, nrow=81):
    xs = np.linspace(-TERR_X_MAX, TERR_X_MAX, ncol)
    ys = np.linspace(-TERR_WY, TERR_WY, nrow)
    X, Y = np.meshgrid(xs, ys)                      # (nrow, ncol)
    Hg = slope_z(X) + amp_at(X) * bump(X, Y)
    return xs, ys, Hg


XS, YS, H = build_height_grid()


def gz_from(xp, xs, ys, Hg, x, y):
    """array-agnostic 雙線性內插；xp = numpy 或 jax.numpy。均勻網格→直接算索引。"""
    nx = xs.shape[0]; ny = ys.shape[0]
    fx = (x - xs[0]) / (xs[-1] - xs[0]) * (nx - 1)
    fy = (y - ys[0]) / (ys[-1] - ys[0]) * (ny - 1)
    fx = xp.clip(fx, 0.0, nx - 1 - 1e-6)
    fy = xp.clip(fy, 0.0, ny - 1 - 1e-6)
    ix = xp.floor(fx).astype(xp.int32); iy = xp.floor(fy).astype(xp.int32)
    tx = fx - ix; ty = fy - iy
    h00 = Hg[iy, ix]; h01 = Hg[iy, ix + 1]
    h10 = Hg[iy + 1, ix]; h11 = Hg[iy + 1, ix + 1]
    return (h00 * (1 - tx) * (1 - ty) + h01 * tx * (1 - ty)
            + h10 * (1 - tx) * ty + h11 * tx * ty)


def gz_np(x, y):
    return gz_from(np, XS, YS, H, np.asarray(x, np.float64), np.asarray(y, np.float64))


def build_terrain2_model(scene_path):
    spec = mujoco.MjSpec.from_file(scene_path)
    floor = next(g for g in spec.geoms if g.name == "floor")
    hmin = float(H.min()); hmax = float(H.max())
    data01 = ((H - hmin) / (hmax - hmin)).astype(np.float64)   # [0,1] row-major
    hf = spec.add_hfield()
    hf.name = "terrain2"
    hf.nrow = H.shape[0]; hf.ncol = H.shape[1]
    hf.size = [TERR_X_MAX, TERR_WY, (hmax - hmin), 0.5]
    hf.userdata = data01.flatten().tolist()
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "terrain2"
    floor.pos = [0.0, 0.0, hmin]                # data=0(最低) 對到世界 z=hmin → 平台(H=0) 落在 z=0
    # 安全底網：加一塊大 plane 在 z=-10
    net = spec.worldbody.add_geom()
    net.name = "safety_net"; net.type = mujoco.mjtGeom.mjGEOM_PLANE
    net.size = [0.0, 0.0, 0.05]; net.pos = [0.0, 0.0, -10.0]
    net.rgba = [0.3, 0.3, 0.3, 0.0]
    return spec.compile()
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run -n rbtdog python task4/inference/tests/test_terrain2.py`
Expected: `PASS test_terrain2`
（若幾何/gz 不一致：檢查 `floor.pos.z=hmin` 與 hfield row/col 對應 x/y 的方向；`userdata` row-major 對應 `(nrow=y, ncol=x)`。）

- [ ] **Step 5: Commit**

```bash
git add task4/inference/terrain2.py task4/inference/tests/test_terrain2.py
git commit -m "feat(terrain2): 統一 hfield 地形(0-15度斜坡+8cm漸變凹凸)+雙線性gz+幾何核對測試"
```

---

### Task 2: CPG 模組 cpg2（每腿可學抬腳 + 新舊版本相容）

**Files:**
- Create: `task4/inference/cpg2.py`
- Test: `task4/inference/tests/test_cpg2.py`

**Interfaces:**
- Consumes: `terrain2`（無）。
- Produces（皆用 `jax.numpy`，本機 CPU 亦可跑）：
  - 常數 `MU_MIN…, G_C=0.08, GC_MIN=0.03, GC_MAX=0.15, D_STEP, G_P, W_COUP, N_CPG_SUB, PHASE_OFFSET, LEGS, HOME3`。
  - `detect_mode(act_dim:int) -> str`：12→`"fixed"`、16→`"learnable"`。
  - `action_to_cpg_cmd(action, mode) -> (mux, muy, omega, gc)`，`gc` shape (4,)。
  - `cpg_init()`, `cpg_step(c, mux, muy, omega, dt)`（同 v1）。
  - `cpg_foot_offsets(c, gc)`、`cpg_to_joint_targets(c, jinvs, gc)`。
  - `leg_ik_consts(scene_path) -> jinvs(np,(4,3,3))`。

- [ ] **Step 1: 寫失敗測試** `task4/inference/tests/test_cpg2.py`

```python
"""cpg2：新舊版本動作映射 + 每腿抬腳。run: conda run -n rbtdog python task4/inference/tests/test_cpg2.py"""
import sys, numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import jax.numpy as jnp
import cpg2 as C


def main():
    assert C.detect_mode(12) == "fixed" and C.detect_mode(16) == "learnable"
    # fixed：action 12 → gc 全部 == G_C(0.08)
    mux, muy, om, gc = C.action_to_cpg_cmd(jnp.zeros(12), "fixed")
    assert gc.shape == (4,) and np.allclose(np.array(gc), C.G_C)
    assert mux.shape == (4,) and muy.shape == (4,) and om.shape == (4,)
    # learnable：action 16、gc 落在 [GC_MIN,GC_MAX]
    a = jnp.zeros(16)
    _, _, _, gc0 = C.action_to_cpg_cmd(a, "learnable")
    assert np.allclose(np.array(gc0), (C.GC_MIN + C.GC_MAX) / 2, atol=1e-6)  # tanh(0)=0→中點
    big = jnp.array(([0, 0, 0, 10.0]) * 4, dtype=jnp.float32)   # 第4欄 tanh→1 → GC_MAX
    _, _, _, gcmax = C.action_to_cpg_cmd(big, "learnable")
    assert np.allclose(np.array(gcmax), C.GC_MAX, atol=1e-4)
    # foot offsets：擺動相(sin>0) dz == gc*sinθ
    c = C.cpg_init()
    c = {**c, "theta": jnp.full(4, jnp.pi / 2)}                 # sinθ=1
    gc_test = jnp.array([0.05, 0.10, 0.03, 0.15])
    off = C.cpg_foot_offsets(c, gc_test)
    assert np.allclose(np.array(off[:, 2]), np.array(gc_test), atol=1e-6), off[:, 2]
    print("PASS test_cpg2")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task4/inference/tests/test_cpg2.py`
Expected: FAIL（`No module named 'cpg2'`）

- [ ] **Step 3: 實作 `task4/inference/cpg2.py`**

```python
"""CPG v2：論文 CPG + 每腿可學抬腳 gc；相容 fixed(12)/learnable(16) 兩種動作。"""
import numpy as np
import mujoco
import jax.numpy as jnp

MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 4.5
A_CONV = 50.0
D_STEP = 0.12
G_C = 0.08                       # 固定抬腳（舊模型用）
G_P = 0.01
GC_MIN, GC_MAX = 0.03, 0.15      # 可學抬腳範圍（新模型用）
W_COUP = 8.0
N_CPG_SUB = 4
LEGS = ["FL", "FR", "RL", "RR"]
HOME3 = jnp.array([0.0, 0.9, -1.8])
HOME3_np = np.array([0.0, 0.9, -1.8])
PHASE_OFFSET = jnp.array([0.0, jnp.pi, jnp.pi, 0.0])
PHI = PHASE_OFFSET[None, :] - PHASE_OFFSET[:, None]


def detect_mode(act_dim):
    if act_dim == 12:
        return "fixed"
    if act_dim == 16:
        return "learnable"
    raise ValueError(f"未知動作維度 {act_dim}（僅支援 12/16）")


def action_to_cpg_cmd(action, mode):
    if mode == "fixed":
        a = jnp.tanh(action).reshape(4, 3)
        gc = jnp.full(4, G_C)
    else:
        a4 = jnp.tanh(action).reshape(4, 4)
        a = a4[:, :3]
        gc = (a4[:, 3] + 1) / 2 * (GC_MAX - GC_MIN) + GC_MIN
    mux = (a[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy = (a[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    omega = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, omega, gc


def cpg_init():
    return {"rx": jnp.full(4, 1.5), "rx_d": jnp.zeros(4),
            "ry": jnp.full(4, 1.5), "ry_d": jnp.zeros(4), "theta": PHASE_OFFSET}


def cpg_step(c, mux, muy, omega, dt):
    rx, rxd, ry, ryd, th = c["rx"], c["rx_d"], c["ry"], c["ry_d"], c["theta"]
    h = dt / N_CPG_SUB
    for _ in range(N_CPG_SUB):
        rxd = rxd + A_CONV * (A_CONV / 4.0 * (mux - rx) - rxd) * h
        rx = rx + rxd * h
        ryd = ryd + A_CONV * (A_CONV / 4.0 * (muy - ry) - ryd) * h
        ry = ry + ryd * h
        rbar = 0.5 * (rx + ry)
        diff = th[None, :] - th[:, None] - PHI
        coup = jnp.sum(rbar[None, :] * jnp.sin(diff), axis=1)
        th = th + (2.0 * jnp.pi * omega + W_COUP * coup) * h
    th = jnp.mod(th, 2.0 * jnp.pi)
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th}


def cpg_foot_offsets(c, gc):
    th = c["theta"]
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
    dx = -D_STEP * fx * jnp.cos(th)
    dy = D_STEP * fy * jnp.cos(th)
    dz = jnp.where(jnp.sin(th) > 0, gc * jnp.sin(th), G_P * jnp.sin(th))
    return jnp.stack([dx, dy, dz], axis=-1)


def cpg_to_joint_targets(c, jinvs, gc):
    off = cpg_foot_offsets(c, gc)
    dq = jnp.einsum("kij,kj->ki", jinvs, off)
    q = HOME3[None, :] + dq
    return q.reshape(12)


def leg_ik_consts(scene_path):
    m = mujoco.MjModel.from_xml_path(scene_path); d = mujoco.MjData(m)
    jinvs = []
    for k, leg in enumerate(LEGS):
        jb = 7 + 3 * k
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg)
        hip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg + "_hip")

        def foot(q3):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.qpos[jb:jb + 3] = q3; mujoco.mj_forward(m, d)
            return (d.geom_xpos[gid] - d.xpos[hip]).copy()
        e = 1e-3; J = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros(3); dq[j] = e
            J[:, j] = (foot(HOME3_np + dq) - foot(HOME3_np - dq)) / (2 * e)
        jinvs.append(np.linalg.inv(J))
    return np.array(jinvs, np.float32)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run -n rbtdog python task4/inference/tests/test_cpg2.py`
Expected: `PASS test_cpg2`

- [ ] **Step 5: Commit**

```bash
git add task4/inference/cpg2.py task4/inference/tests/test_cpg2.py
git commit -m "feat(cpg2): 每腿可學抬腳gc + fixed(12)/learnable(16)動作相容 + 單元測試"
```

---

### Task 3: 觀測模組 obs2（76/80 相容）

**Files:**
- Create: `task4/inference/obs2.py`
- Test: `task4/inference/tests/test_obs2.py`

**Interfaces:**
- Consumes: `cpg2`。
- Produces: `build_obs(grav, blin, gyro, dq, dqvel, cmd, last_action, contact, c) -> jnp.ndarray`
  - 欄位順序：`grav(3)+blin(3)+gyro(3)+dq(12)+dqvel(12)+cmd(3)+last_action(len)+contact(4)+rx(4)+rx_d(4)+ry(4)+ry_d(4)+sin(theta)(4)+cos(theta)(4)`。
  - `last_action` 長度 12→obs 76、16→obs 80。

- [ ] **Step 1: 寫失敗測試** `task4/inference/tests/test_obs2.py`

```python
"""obs2：76/80 維相容。run: conda run -n rbtdog python task4/inference/tests/test_obs2.py"""
import sys, numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import jax.numpy as jnp
import cpg2 as C
import obs2 as O


def _dummy(last_len):
    z3 = jnp.zeros(3); z12 = jnp.zeros(12); z4 = jnp.zeros(4)
    return O.build_obs(z3, z3, z3, z12, z12, jnp.zeros(3),
                       jnp.zeros(last_len), z4, C.cpg_init())


def main():
    assert _dummy(12).shape == (76,), _dummy(12).shape
    assert _dummy(16).shape == (80,), _dummy(16).shape
    # 欄位順序：前 3 = grav
    grav = jnp.array([0.1, 0.2, 0.3])
    o = O.build_obs(grav, jnp.zeros(3), jnp.zeros(3), jnp.zeros(12), jnp.zeros(12),
                    jnp.zeros(3), jnp.zeros(16), jnp.zeros(4), C.cpg_init())
    assert np.allclose(np.array(o[:3]), np.array(grav))
    print("PASS test_obs2")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task4/inference/tests/test_obs2.py`
Expected: FAIL（`No module named 'obs2'`）

- [ ] **Step 3: 實作 `task4/inference/obs2.py`**

```python
"""觀測建構：欄位順序與 v1 一致，last_action 長度決定 76/80。"""
import jax.numpy as jnp


def build_obs(grav, blin, gyro, dq, dqvel, cmd, last_action, contact, c):
    return jnp.concatenate([
        grav, blin, gyro,
        dq, dqvel,
        cmd, last_action, contact,
        c["rx"], c["rx_d"], c["ry"], c["ry_d"],
        jnp.sin(c["theta"]), jnp.cos(c["theta"]),
    ])
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run -n rbtdog python task4/inference/tests/test_obs2.py`
Expected: `PASS test_obs2`

- [ ] **Step 5: Commit**

```bash
git add task4/inference/obs2.py task4/inference/tests/test_obs2.py
git commit -m "feat(obs2): 觀測建構(76/80維相容)+維度測試"
```

---

### Task 4: MJX 訓練環境 go2_terrain2_env + 本機 CPU 冒煙測試

**Files:**
- Create: `task4/inference/go2_terrain2_env.py`
- Test: `task4/inference/tests/test_env2_smoke.py`

**Interfaces:**
- Consumes: `terrain2`, `cpg2`, `obs2`。
- Produces:
  - `Go2Terrain2Env(jinvs)`：Brax `Env`，`observation_size=80`、`action_size=16`、`backend="mjx"`。
  - `apply_pd(m)`、`domain_randomize(sys, rng)`（供 notebook 沿用）。

- [ ] **Step 1: 寫失敗測試** `task4/inference/tests/test_env2_smoke.py`

```python
"""MJX 地形環境本機 CPU 冒煙測試。run: conda run -n rbtdog python task4/inference/tests/test_env2_smoke.py"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import cpg2 as C
import go2_terrain2_env as E


def main():
    jinvs = C.leg_ik_consts("mujoco_menagerie/unitree_go2/scene_mjx.xml")
    env = E.Go2Terrain2Env(jinvs)
    assert env.observation_size == 80 and env.action_size == 16
    for seed in [0, 1, 2]:
        s = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        assert s.obs.shape == (80,), s.obs.shape
        assert float(s.done) == 0.0, "reset 不應立即 done"
        s2 = jax.jit(env.step)(s, jnp.zeros(16))
        r = float(s2.reward)
        assert np.isfinite(r), f"reward 非有限 {r}"
        # 站立零動作數步後仍在地形上（rel_h 合理、未穿透）
        st = s
        for _ in range(20):
            st = jax.jit(env.step)(st, jnp.zeros(16))
        rel_h = float(st.metrics["rel_h"])
        assert 0.10 < rel_h < 0.45, f"rel_h 異常(可能穿透/飛起) {rel_h}"
    print("PASS test_env2_smoke")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task4/inference/tests/test_env2_smoke.py`
Expected: FAIL（`No module named 'go2_terrain2_env'`）

- [ ] **Step 3: 實作 `task4/inference/go2_terrain2_env.py`**

```python
"""Go2 CPG-RL 地形 v2 MJX 環境：全向指令追蹤、統一 hfield、每腿可學抬腳。"""
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State

import terrain2 as T
import cpg2 as C
import obs2 as O

SCENE_MJX = "mujoco_menagerie/unitree_go2/scene_mjx.xml"
CTRL_DT, SIM_DT = 0.02, 0.004
N_FRAMES = int(round(CTRL_DT / SIM_DT))
HOME12 = jnp.array([0.0, 0.9, -1.8] * 4)
KP_NOM, KD_NOM = 90.0, 3.0
KNEE_IDX = [2, 5, 8, 11]
FOOT_CONTACT_H = 0.03
PUSH_EVERY = 100
PUSH_VEL = 0.6
# terrain 網格常數轉 jnp（gz 用）
XS_J = jnp.asarray(T.XS); YS_J = jnp.asarray(T.YS); H_J = jnp.asarray(T.H)


def gz_j(x, y):
    return T.gz_from(jnp, XS_J, YS_J, H_J, x, y)


def apply_pd(m, kp=KP_NOM, kd=KD_NOM):
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kd
    fr = np.full(m.nu, 23.7); fr[KNEE_IDX] = 45.43
    m.actuator_forcerange[:, 0] = -fr; m.actuator_forcerange[:, 1] = fr
    m.actuator_forcelimited[:] = 1
    return m


def _qinv(q): return jnp.array([q[0], -q[1], -q[2], -q[3]])
def _qrot(q, v):
    u = q[1:4]; t = 2.0 * jnp.cross(u, v); return v + q[0] * t + jnp.cross(u, t)
def w2b(quat, v): return _qrot(_qinv(quat), v)


class Go2Terrain2Env(Env):
    def __init__(self, jinvs):
        m = T.build_terrain2_model(SCENE_MJX); m.opt.timestep = SIM_DT
        m = apply_pd(m)
        self._mj = m
        self.sys = mjx.put_model(m)
        self._init_q = jnp.array(m.key_qpos[0])
        self._lo = jnp.array(m.actuator_ctrlrange[:, 0])
        self._hi = jnp.array(m.actuator_ctrlrange[:, 1])
        self._jinvs = jnp.array(jinvs)
        self._foot_gid = jnp.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS])

    @property
    def observation_size(self): return 80
    @property
    def action_size(self): return 16
    @property
    def backend(self): return "mjx"

    def _sample_cmd(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        vx = jax.random.uniform(k1, (), minval=0.0, maxval=1.0)
        vy = jax.random.uniform(k2, (), minval=-0.3, maxval=0.3)
        wz = jax.random.uniform(k3, (), minval=-1.0, maxval=1.0)
        return jnp.array([vx, vy, wz])

    def _base(self, data):
        quat = data.qpos[3:7]; gyro = data.qvel[3:6]
        blin = w2b(quat, data.qvel[0:3])
        grav = w2b(quat, jnp.array([0.0, 0.0, -1.0]))
        return quat, gyro, blin, grav

    def _foot_contact(self, data):
        fx = data.geom_xpos[self._foot_gid, 0]
        fy = data.geom_xpos[self._foot_gid, 1]
        fz = data.geom_xpos[self._foot_gid, 2]
        return (fz - gz_j(fx, fy) < FOOT_CONTACT_H).astype(jnp.float32)

    def _obs(self, data, info):
        _, gyro, blin, grav = self._base(data)
        obs = O.build_obs(grav, blin, gyro,
                          data.qpos[7:19] - HOME12, data.qvel[6:18],
                          info["cmd"], info["last_action"],
                          self._foot_contact(data), info["cpg"])
        return jnp.nan_to_num(jnp.clip(obs, -50.0, 50.0), nan=0.0)

    def reset(self, rng):
        rng, crng, hrng = jax.random.split(rng, 3)
        downhill = jax.random.bernoulli(hrng, 0.5)
        quat = jnp.where(downhill, jnp.array([0.0, 0.0, 0.0, 1.0]),
                         jnp.array([1.0, 0.0, 0.0, 0.0]))
        qpos = self._init_q.at[3:7].set(quat)
        data = mjx.make_data(self.sys).replace(qpos=qpos)
        data = mjx.forward(self.sys, data)
        info = {"rng": rng, "cmd": self._sample_cmd(crng),
                "cpg": C.cpg_init(), "last_action": jnp.zeros(16),
                "step": jnp.zeros((), jnp.int32)}
        obs = self._obs(data, info)
        metrics = {"reward": jnp.zeros(()), "r_lin": jnp.zeros(()),
                   "r_yaw": jnp.zeros(()), "rel_h": jnp.zeros(()),
                   "gc_mean": jnp.zeros(())}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state, action):
        mux, muy, omega, gc = C.action_to_cpg_cmd(action, "learnable")
        cpg = C.cpg_step(state.info["cpg"], mux, muy, omega, CTRL_DT)
        q_des = C.cpg_to_joint_targets(cpg, self._jinvs, gc)
        ctrl = jnp.clip(q_des, self._lo, self._hi)

        def one(d, _):
            return mjx.step(self.sys, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(one, state.pipeline_state, None, N_FRAMES)

        rng, krng = jax.random.split(state.info["rng"])
        step_i = state.info["step"] + 1
        do_push = jnp.mod(step_i, PUSH_EVERY) == 0
        kick = jax.random.uniform(krng, (2,), minval=-PUSH_VEL, maxval=PUSH_VEL)
        qvel = (data.qvel.at[0].add(jnp.where(do_push, kick[0], 0.0))
                          .at[1].add(jnp.where(do_push, kick[1], 0.0)))
        data = data.replace(qvel=qvel)

        info = {**state.info, "cpg": cpg, "last_action": action,
                "rng": rng, "step": step_i}
        obs = self._obs(data, info)
        _, gyro, blin, grav = self._base(data)
        cmd = info["cmd"]
        r_lin = jnp.exp(-((blin[0] - cmd[0]) ** 2 + (blin[1] - cmd[1]) ** 2) / 0.25)
        r_yaw = jnp.exp(-((gyro[2] - cmd[2]) ** 2) / 0.25)
        upright = grav[0] ** 2 + grav[1] ** 2
        gzb = gz_j(data.qpos[0], data.qpos[1])
        rel_h = data.qpos[2] - gzb
        height_pen = (rel_h - 0.30) ** 2
        act_rate = jnp.sum((action - state.info["last_action"]) ** 2)
        reward = (1.5 * r_lin + 1.2 * r_yaw - 1.0 * upright
                  - 0.5 * height_pen - 0.05 * act_rate + 0.05)   # 無 y_pen
        done = jnp.where((rel_h < 0.18) | (grav[2] > -0.4), 1.0, 0.0)
        finite = (jnp.isfinite(reward) & jnp.all(jnp.isfinite(data.qpos))
                  & jnp.all(jnp.isfinite(data.qvel)))
        reward = jnp.where(finite, reward, 0.0)
        done = jnp.where(finite, done, 1.0)
        metrics = {"reward": reward, "r_lin": r_lin, "r_yaw": r_yaw,
                   "rel_h": rel_h, "gc_mean": jnp.mean(gc)}
        return state.replace(pipeline_state=data, obs=obs, reward=reward,
                             done=done, metrics=metrics, info=info)


_mm = T.build_terrain2_model(SCENE_MJX)
BASE_ID = mujoco.mj_name2id(_mm, mujoco.mjtObj.mjOBJ_BODY, "base")


def domain_randomize(sys, rng):
    @jax.vmap
    def per_env(rng):
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
        geom_friction = sys.geom_friction.at[:, 0].set(
            jax.random.uniform(k1, minval=0.3, maxval=1.0))
        kp = jax.random.uniform(k2, minval=75.0, maxval=105.0)
        kd = jax.random.uniform(k3, minval=2.0, maxval=4.0)
        gain = sys.actuator_gainprm.at[:, 0].set(kp)
        bias = sys.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)
        body_mass = sys.body_mass * jax.random.uniform(
            k4, (sys.nbody,), minval=0.8, maxval=1.2)
        payload = jax.random.uniform(k5, minval=0.0, maxval=8.0)
        body_mass = body_mass.at[BASE_ID].add(payload)
        return geom_friction, gain, bias, body_mass
    gf, gain, bias, bm = per_env(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.replace(geom_friction=0, actuator_gainprm=0,
                              actuator_biasprm=0, body_mass=0)
    sys = sys.replace(geom_friction=gf, actuator_gainprm=gain,
                      actuator_biasprm=bias, body_mass=bm)
    return sys, in_axes
```

- [ ] **Step 4: 跑測試確認通過（本機 CPU MJX，可能較慢，數十秒）**

Run: `conda run -n rbtdog python task4/inference/tests/test_env2_smoke.py`
Expected: `PASS test_env2_smoke`
（若 `rel_h` 異常：檢查 hfield `floor.pos.z` 校正與 spawn 是否穿透；若 hfield 碰撞報錯：確認 mujoco-mjx 版本支援 `hfield_sphere`。這一步就是把 spec §9 風險 1/3 在本機先擋掉。）

- [ ] **Step 5: Commit**

```bash
git add task4/inference/go2_terrain2_env.py task4/inference/tests/test_env2_smoke.py
git commit -m "feat(env2): MJX 地形v2環境(全向指令+每腿抬腳+統一hfield)+本機CPU冒煙測試"
```

---

### Task 5: 本機推論 local_infer_terrain2（新舊模型 × 任意地形 + 影片）

**Files:**
- Create: `task4/inference/local_infer_terrain2.py`
- Test: `task4/inference/tests/test_infer2.py`

**Interfaces:**
- Consumes: `terrain2`, `cpg2`, `obs2`；既有 `local_infer_paper.load_policy`（policy 256/256/128、value 256³、normalize、deterministic）。
- Produces:
  - `load_policy_any(path) -> (infer, act_dim)`：載入權重並回傳推論函式與動作維度。
  - `rollout(params_path, terrain="rough2", secs=8.0, video=False) -> dict`：自動偵測版本，於指定地形跑 rollout，回傳 `{"mode","dist","fell","fz_lift","end_h"}`；`video=True` 輸出 mp4。
  - `terrain="flat"|"rough2"`（`rough2`＝本設計 hfield；`flat`＝原始 scene）。

- [ ] **Step 1: 寫失敗測試** `task4/inference/tests/test_infer2.py`

```python
"""local_infer_terrain2：舊模型(12維)零樣本跑 v2 地形不崩、能前進。
run: conda run -n rbtdog python task4/inference/tests/test_infer2.py"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import local_infer_terrain2 as L

OLD = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl"   # 12維 fixed


def main():
    infer, act_dim = L.load_policy_any(OLD)
    assert act_dim == 12, act_dim
    r = L.rollout(OLD, terrain="rough2", secs=4.0, video=False)
    assert r["mode"] == "fixed"
    assert r["fell"] is None or r["fell"] > 1.0, f"不應一開始就跌 {r}"
    # 平台起步應能往前一點（零樣本、地形難，門檻放寬）
    assert r["dist"] > 0.1, f"前進距離過小 {r['dist']}"
    print("PASS test_infer2", r)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run -n rbtdog python task4/inference/tests/test_infer2.py`
Expected: FAIL（`No module named 'local_infer_terrain2'`）

- [ ] **Step 3: 實作 `task4/inference/local_infer_terrain2.py`**

```python
"""本機推論（地形 v2）：自動偵測 fixed(12)/learnable(16)、可跑 flat/rough2 地形、輸出影片。
底層走路不依賴 odom；指令可腳本化。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import numpy as np
import mujoco
import jax, jax.numpy as jnp

import terrain2 as T
import cpg2 as C
import obs2 as O
from local_infer_paper import load_policy   # 沿用既有 policy 建構/normalize/deterministic

SCENE = "mujoco_menagerie/unitree_go2/scene.xml"
CTRL_DT, SIM_DT = 0.02, 0.004
HOME12 = np.array([0.0, 0.9, -1.8] * 4)


def load_policy_any(path):
    infer = load_policy(path)
    # 用一個測試 obs 探測動作維度：先試 80，失敗再試 76
    for od, ad in [(80, 16), (76, 12)]:
        try:
            a = np.array(infer(jnp.zeros(od, jnp.float32)))
            if a.shape[-1] == ad:
                return infer, ad
        except Exception:
            continue
    raise RuntimeError("無法判定模型維度")


def _make_model(terrain):
    if terrain == "flat":
        m = mujoco.MjModel.from_xml_path(SCENE)
    elif terrain == "rough2":
        m = T.build_terrain2_model(SCENE)
    else:
        raise ValueError(terrain)
    m.opt.timestep = SIM_DT
    kp, kd = 90.0, 3.0
    m.actuator_gainprm[:, 0] = kp; m.actuator_biasprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = -kp; m.actuator_biasprm[:, 2] = -kd
    fr = np.full(m.nu, 23.7); fr[[2, 5, 8, 11]] = 45.43
    m.actuator_forcerange[:, 0] = -fr; m.actuator_forcerange[:, 1] = fr
    m.actuator_forcelimited[:] = 1
    return m


def _gz(terrain, x, y):
    return float(T.gz_np(x, y)) if terrain == "rough2" else 0.0


def _qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def _qrot(q, v):
    u = q[1:4]; t = 2 * np.cross(u, v); return v + q[0] * t + np.cross(u, t)
def w2b(q, v): return _qrot(_qinv(q), v)


def rollout(params_path, terrain="rough2", secs=8.0, cmd=(0.6, 0.0, 0.0), video=False):
    infer, act_dim = load_policy_any(params_path)
    mode = C.detect_mode(act_dim)
    jinvs = C.leg_ik_consts(SCENE)
    m = _make_model(terrain)
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    lo = m.actuator_ctrlrange[:, 0]; hi = m.actuator_ctrlrange[:, 1]
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in C.LEGS]
    n_sub = int(round(CTRL_DT / SIM_DT))
    c = C.cpg_init(); last_a = np.zeros(act_dim)
    cmd = np.asarray(cmd, np.float32)
    frames = []; ren = cam = None
    if video:
        ren = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
    x0 = float(d.qpos[0]); fzmin = fzmax = None; fell = None
    for i in range(int(secs / CTRL_DT)):
        grav = w2b(d.qpos[3:7], np.array([0, 0, -1.0]))
        blin = w2b(d.qpos[3:7], d.qvel[0:3])
        fx = np.array([d.geom_xpos[g][0] for g in foot_gid])
        fy = np.array([d.geom_xpos[g][1] for g in foot_gid])
        fz = np.array([d.geom_xpos[g][2] for g in foot_gid])
        gzf = np.array([_gz(terrain, fx[k], fy[k]) for k in range(4)])
        contact = ((fz - gzf) < 0.03).astype(np.float32)
        o = O.build_obs(jnp.asarray(grav), jnp.asarray(blin), jnp.asarray(d.qvel[3:6]),
                        jnp.asarray(d.qpos[7:19] - HOME12), jnp.asarray(d.qvel[6:18]),
                        jnp.asarray(cmd), jnp.asarray(last_a), jnp.asarray(contact), c)
        act = np.array(infer(jnp.asarray(o, jnp.float32)))
        mux, muy, om, gc = C.action_to_cpg_cmd(jnp.asarray(act), mode)
        c = C.cpg_step(c, mux, muy, om, CTRL_DT)
        q_des = np.array(C.cpg_to_joint_targets(c, jnp.asarray(jinvs), gc))
        d.ctrl[:] = np.clip(q_des, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        last_a = act
        if grav[2] > -0.4 and fell is None:
            fell = i * CTRL_DT
        flz = d.geom_xpos[foot_gid[0]][2]
        fzmin = flz if fzmin is None else min(fzmin, flz)
        fzmax = flz if fzmax is None else max(fzmax, flz)
        if video and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]; cam.distance = 2.5
            cam.elevation = -18; cam.azimuth = 90
            ren.update_scene(d, cam); frames.append(ren.render())
    res = {"mode": mode, "dist": float(d.qpos[0]) - x0, "fell": fell,
           "fz_lift": (fzmax - fzmin), "end_h": float(d.qpos[2])}
    if video and frames:
        import imageio.v2 as iio
        out = f"/home/huang/rbtdog_sim/task4/outputs/terrain2_{mode}_{terrain}.mp4"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        iio.mimsave(out, frames, fps=25, codec="libx264"); res["video"] = out
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--terrain", default="rough2", choices=["flat", "rough2"])
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    print(rollout(a.params, a.terrain, a.secs, (a.vx, a.vy, a.wz), a.video))
```

- [ ] **Step 4: 跑測試確認通過（用既有 12 維舊權重零樣本跑 v2 地形）**

Run: `conda run -n rbtdog python task4/inference/tests/test_infer2.py`
Expected: `PASS test_infer2 {...}`
（這步同時驗證「向後相容」：舊 12 維模型能在 v2 hfield 上跑不崩。若 `load_policy` 對維度不符報錯，調整 `load_policy_any` 探測順序。）

- [ ] **Step 5: Commit**

```bash
git add task4/inference/local_infer_terrain2.py task4/inference/tests/test_infer2.py
git commit -m "feat(infer2): 本機推論自動偵測新舊模型(12/16)×任意地形+影片+相容測試"
```

---

### Task 6: 組裝 Colab 訓練 notebook cpg_rl_terrain2_colab.ipynb

**Files:**
- Create: `task4/notebooks/cpg_rl_terrain2_colab.ipynb`

**Interfaces:**
- Consumes: 已驗證模組 `terrain2.py` / `cpg2.py` / `obs2.py` / `go2_terrain2_env.py` 的原始碼（**原封嵌入 cell**，Colab 需自包含）。
- Produces: `cpg_rl_terrain2_params.pkl`（16 維 learnable 權重）。

> 說明：notebook 無法用 assert 腳本自動測試；其正確性繼承自 Task 1–4 的模組（cell 內容為模組原始碼複本），並以 notebook 內建 smoke cell 於 Colab 再驗一次。**每個嵌入 cell 頂端註明來源模組**以利日後同步。

- [ ] **Step 1: 建立 notebook 骨架與安裝 cell**

以 v1 `cpg_rl_terrain_colab.ipynb` 為模板複製，改標題為「CPG-RL 地形 v2（全向指令 + 粗糙度漸變凹凸 + 每腿可學抬腳）」。保留：
- 安裝 cell（`pip install -q mujoco mujoco-mjx brax mediapy`、`MUJOCO_GL=egl`）。
- clone menagerie cell（`SCENE = "mujoco_menagerie/unitree_go2/scene_mjx.xml"`）。

- [ ] **Step 2: 嵌入已驗證模組（四個 code cell）**

依序建立 4 個 code cell，內容分別 = `cpg2.py`、`terrain2.py`、`obs2.py`、`go2_terrain2_env.py` 的**原始碼原封貼入**（cell 頂端加註 `# === 來源: task4/inference/<mod>.py（已本機測試，勿隨意改，改則同步回模組）===`）。
注意 import 順序：cpg2 → terrain2 → obs2 → go2_terrain2_env（env 依賴前三者）。因 Colab 為單一 namespace，將 env cell 內的 `import terrain2 as T / cpg2 as C / obs2 as O` 改為直接引用已在 namespace 的名稱（把 `T.`/`C.`/`O.` 前綴保留的話，改成在 cell 頂 `import sys, types` 建 module alias，或最簡：把三模組也放進可 import 的檔案並 `%%writefile`）。**採用 `%%writefile` 法**：四個 cell 各用 `%%writefile terrain2.py` 等寫出檔案，再一個 cell `import cpg2, terrain2, obs2` 與 `from go2_terrain2_env import Go2Terrain2Env, domain_randomize, apply_pd`。這樣與模組零差異、免改前綴。

- [ ] **Step 3: Smoke test cell（Colab 上先擋風險）**

```python
import cpg2 as C, go2_terrain2_env as E
import jax, jax.numpy as jnp
jinvs = C.leg_ik_consts("mujoco_menagerie/unitree_go2/scene_mjx.xml")
env = E.Go2Terrain2Env(jinvs)
for seed in [0, 1, 2, 3]:
    s = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    s2 = jax.jit(env.step)(s, jnp.zeros(16))
    face = "下坡(-x)" if float(s.pipeline_state.qpos[6]) > 0.5 else "上坡(+x)"
    print(f"[seed {seed}] obs={s.obs.shape} {face} reward={float(s2.reward):+.3f} "
          f"done={float(s2.done):.0f} rel_h={float(s2.metrics['rel_h']):.3f} "
          f"gc_mean={float(s2.metrics['gc_mean']):.3f}")
print("PASSED" if s.obs.shape == (80,) else "CHECK OBS SIZE")
```

- [ ] **Step 4: PPO 訓練 cell**

複製 v1 的 PPO cell，改兩處：`environment=Go2Terrain2Env(jinvs)`、`num_timesteps=200_000_000`（其餘超參不變：`num_envs=2048, batch_size=256, num_minibatches=32, unroll_length=20, num_updates_per_batch=4, lr=3e-4, entropy_cost=1e-2, discounting=0.97, normalize_observations=True, policy=(256,256,128), value=(256,256,256), randomization_fn=domain_randomize`）。保留 `progress` 印 reward 與 `plt` 曲線 cell。

- [ ] **Step 5: Rollout 影片 cell（展示指令跟隨）**

新增 rollout cell：對訓練好的 policy，用 numpy 迴圈（obs 組法呼叫嵌入的 `obs2.build_obs`、CPG 呼叫 `cpg2`、地面用 `terrain2.gz_np`）跑三段並各出一支影片，命令分別：
- 直走上坡：`cmd=[0.6,0,0]`、spawn 面 +x；
- 轉向：`cmd=[0.4,0,0.8]`（邊走邊左轉）；
- 橫移：`cmd=[0.3,0.25,0]`（前進兼右移）。
每段印 `前進/末端高/FL抬腳量/gc_mean`，用 `media.show_video`。（可直接改寫 `local_infer_terrain2.rollout` 的迴圈；notebook 版用 `make_inference_fn(params, deterministic=True)`。）

- [ ] **Step 6: 存權重 cell**

```python
from brax.io import model
model.save_params("cpg_rl_terrain2_params.pkl", params)
try:
    from google.colab import files; files.download("cpg_rl_terrain2_params.pkl")
except Exception as e:
    print("左側檔案面板右鍵下載 cpg_rl_terrain2_params.pkl。", e)
```

- [ ] **Step 7: 本機驗證 notebook 的 env 區塊（用模組測試代跑）**

因 notebook cell = 模組原始碼，直接重跑 Task 4 測試即證明嵌入邏輯正確：
Run: `conda run -n rbtdog python task4/inference/tests/test_env2_smoke.py`
Expected: `PASS test_env2_smoke`

- [ ] **Step 8: Commit**

```bash
git add task4/notebooks/cpg_rl_terrain2_colab.ipynb
git commit -m "feat(notebook): cpg_rl_terrain2 訓練notebook(全向指令+漸變凹凸+每腿抬腳,嵌入已測模組)"
```

---

## 完訓後（Colab 產出權重帶回本機，非本計畫自動步驟）

1. 把 `cpg_rl_terrain2_params.pkl` 放到 `task4/weights/`。
2. 出對比影片（新模型 learnable vs 舊模型 fixed，同一 rough2 地形）：
   ```bash
   conda run -n rbtdog python task4/inference/local_infer_terrain2.py \
     --params task4/weights/cpg_rl_terrain2_params.pkl --terrain rough2 --video --secs 8
   conda run -n rbtdog python task4/inference/local_infer_terrain2.py \
     --params task4/weights/cpg_rl_paper_params.pkl --terrain rough2 --video --secs 8
   ```
3. 對照 spec §10 驗收標準（指令跟隨、上下坡+凹凸不拖地、平地步態不過度保守、凹凸段 `gc_mean` 較高）。

---

## Self-Review（對照 spec）

- **§2 每腿抬腳**：Task 2（cpg2 `action_to_cpg_cmd` learnable、`cpg_foot_offsets(c,gc)`）✓
- **§3 統一 hfield + 漸變凹凸 + 0–15°斜坡 + 雙線性 gz + mj_ray 核對**：Task 1 ✓
- **§4 全向指令 + 純速度追蹤 + 移除 y_pen**：Task 4 env `_sample_cmd`/`step` reward ✓
- **§5 obs 80**：Task 3 + Task 4 ✓
- **§6 PD/DR/push/PPO 沿用**：Task 4（apply_pd/domain_randomize）+ Task 6 PPO cell ✓
- **§8 向後相容（12/16 自動偵測、同地形跑兩版）**：Task 2 `detect_mode` + Task 5 `load_policy_any`/`rollout` + test_infer2 用舊權重零樣本 ✓
- **§9 風險（hfield 效能/正確、拖地監控、正規化校正、盲走）**：Task 1 mj_ray、Task 4 本機 CPU MJX 冒煙、`gc_mean` metric ✓
- **§10/§11 驗收**：Task 6 rollout cell + 完訓後對比 ✓
- Placeholder 掃描：無 TBD/TODO；每步含實碼與指令。
- 型別一致：`gz_from(xp,...)`/`gz_np`/`gz_j`、`action_to_cpg_cmd(action,mode)`、`build_obs(...)`、`Go2Terrain2Env`、`load_policy_any`/`rollout` 跨 task 命名一致。
