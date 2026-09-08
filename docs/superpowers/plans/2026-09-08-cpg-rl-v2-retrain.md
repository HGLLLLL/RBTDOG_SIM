# CPG-RL v2 重訓 —— 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 `A_kp250_walk` 為基準，建立 14 維動作（加 body sway）、70 維 obs、含實測延遲/雜訊/偏置隨機化與力矩護欄的 MJX 訓練環境，在 Colab 訓練，本機以 G0–G8 驗收。

**Architecture:** env 從 notebook 抽成 repo 模組 `task7/inference/rl_env_max.py`（JAX：CPG、IK、env、DR），本機 CPU 可跑 reset/step 做測試與 G0；notebook 只剩安裝、clone、import、基準校準、訓練、存檔。`local_infer_max.py` 升到 14 維＋sway＋12 擾動＋G3–G7 判定。訓練模型由 `make_mjx_model.py` 參數化產生 kp250 版。

**Tech Stack:** Python 3.11、numpy、mujoco 3.10.0、mujoco-mjx、jax 0.10.2（本機 CPU）／jax<0.10+cuda（Colab）、brax 0.14.2、pytest。

## Global Constraints

- 測試一律 `conda run -n rbtdog python -m pytest task7/tests -q`；既有 741 項不可變紅
- brax 0.14.2 / mujoco 3.10.0 版本鎖死；policy (256,256,128)、value (256,256,256)、normalize_observations=True 在 notebook 與 `local_infer_max.py` 必須逐項相同
- `obs_max.OBS_LAYOUT` 只改 `last_action` 12→14（`ACT_DIM=14`、`OBS_DIM=70`）；其他欄位順序不動
- 基準常數唯一來源 `gait_baseline.BASELINE_A`；增益 `max_model.KP3_A=[60,250,250]`、`KD3_A=[2,2,2]`、輪 kd `KD_WHEEL=0.5`；不改 `KP3/KD3/BASELINE/BASELINE_KP250`
- 動作範圍：mu 1.0–2.0、ω 0–2.0、sway ±0.060 m、sway 斜率 0.004 m/步
- 隨機化：IMU 偏轉 ±3°、gyro 偏置 ±0.02、延遲 1+{0,1} 步、joint_vel 延 1 步；雜訊 joint_pos 0.001、joint_vel [0.2,0.2,0.4]、gyro 0.002、gravity 0.005
- 護欄：`TAU_BAR=58`、`ERR_BAR=0.45`；終止 |τ|>90 連續 3 步；翻倒 `grav_z>−0.4`；機身高 <0.29
- 驗收在原始網格模型 `max_model.SCENE`；G4 roll 峰值與 std 都降 ≥60%；G5 exec_front ≥0.9 且 |前−後|<0.15；G6 60 s 總偏航 <5°；G7 峰值力矩×1.2<70、誤差×1.14<0.6
- Colab 從 GitHub `main` clone → 訓練前 `git push origin main`
- 機身座標 x 前 y 左 z 上；四元數 wxyz（MuJoCo）

## File Structure

| 檔案 | 責任 |
|---|---|
| `task7/inference/max_model.py`（改） | 加 `KP3_A`、`KD3_A`、`SCENE_MJX_KP250` |
| `task7/inference/gait_baseline.py`（改） | 加 `BASELINE_A`（含 phase 名、增益）；不動舊常數 |
| `task7/inference/cpg_walk_max.py`（改） | `GAITS["walk_a"]`；`Robot.err_peak`；`Trace.summarize` 加 `roll_pk/roll_std/err_peak` |
| `task7/model/zgws/make_mjx_model.py`（改） | `build(kp3=, kd3=)`；`build_all()` 多產 `zgws_mjx_kp250.xml` + `scene_flat_mjx_kp250.xml` |
| `task7/inference/obs_max.py`（改） | `ACT_DIM=14` |
| `task7/inference/rl_env_max.py`（新） | JAX：常數、cpg_step(LS)、ik_j/fk_j、joint_targets_j(sway)、act_to_cmd(14)、sway 斜率、`MaxCpgEnv`、`domain_randomize`、`baseline_action()`、`reward_terms()` |
| `task7/inference/local_infer_max.py`（改） | 14 維、sway、LS、A 增益、`--perturb N`、`--compare`、G3–G7 判定、影片 v2 |
| `task7/notebooks/cpg_rl_max_colab.ipynb`（改） | 由 `task7/notebooks/build_nb_v2.py` 重新產生 |
| `task7/tests/test_rl_env_max.py`（新）、`test_local_infer_max.py`（新）；`test_obs_max.py`、`test_gait_baseline.py`、`test_mjx_model.py`（改） | 測試 |
| `task7/docs/CPG-RL_v2_設計_2026-09-08.md`（新）、`task7/HANDOFF.md`（改） | 文件 |

---

### Task 1: 基準常數與增益（`BASELINE_A`、`KP3_A`）

**Files:**
- Modify: `task7/inference/max_model.py`（`KP3/KD3` 定義處之後）
- Modify: `task7/inference/gait_baseline.py`（檔尾）
- Modify: `task7/inference/cpg_walk_max.py`（`GAITS` 字典）
- Test: `task7/tests/test_gait_baseline.py`

**Interfaces:**
- Produces: `max_model.KP3_A: np.ndarray([60,250,250])`、`max_model.KD3_A: np.ndarray([2,2,2])`、`max_model.SCENE_MJX_KP250: str`；`gait_baseline.BASELINE_A: dict`（鍵：gait, seq, duty, omega, mu_x, mu_y, d_step, d_step_y, x_off, g_c, z_sag, kp3, kd3, wheel_kd, wheel_mode）；`cpg_walk_max.GAITS["walk_a"]`

- [ ] **Step 1: 寫測試（加在 `test_gait_baseline.py` 檔尾）**

```python
def test_baseline_a_matches_A_kp250_walk_json():
    """BASELINE_A 是 RL v2 的基準與對照組，必須與實機走過的軌跡檔逐欄相等。"""
    import json
    p = Path(__file__).resolve().parents[1] / "outputs" / "A_kp250_walk.json"
    D = json.loads(p.read_text(encoding="utf-8"))
    P = D["params"]
    A = gb.BASELINE_A
    assert A["seq"] == D["seq"] == "ls"
    for k in ("duty", "omega", "d_step", "x_off", "g_c", "z_sag"):
        assert A[k] == P[k], k
    assert A["mu_x"] == 1.80 and A["mu_y"] == 1.50 and A["d_step_y"] == 0.12
    assert A["kp3"] == [D["kp_abad"], D["kp"], D["kp"]] == [60.0, 250.0, 250.0]
    assert A["kd3"] == [D["kd"]] * 3 == [2.0, 2.0, 2.0]
    assert A["wheel_kd"] == D["wheel_kd"] == 0.5 and A["wheel_mode"] == "damp"
    assert list(mm.KP3_A) == A["kp3"] and list(mm.KD3_A) == A["kd3"]
    # 舊常數不可被順手改掉
    assert gb.BASELINE["x_off"] == -0.040 and list(mm.KP3) == [60.0, 120.0, 120.0]


def test_walk_a_gait_uses_ls_phase_and_baseline_a():
    g = cw.GAITS["walk_a"]
    assert np.allclose(g["phase"], cpg_max.PHASE_WALK_LS)
    for k in ("duty", "omega", "mu_x", "x_off", "g_c", "d_step"):
        assert g[k] == gb.BASELINE_A[k], k
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_gait_baseline.py -q -k "baseline_a or walk_a"`
Expected: FAIL（`AttributeError: BASELINE_A`）

- [ ] **Step 3: 實作**

`max_model.py`，在 `KD3 = ...` 之後加：

```python
# ★ RL v2（2026-09-08）用的增益 —— 就是實機 A_kp250_walk 走過的那組。
#   ABAD 60 是原廠值，HIP/KNEE 250 是原廠站立值（M7/M8/trip17 實機驗證）。
#   不改 KP3/KD3：kp120 那條線（BASELINE）仍引用它們。
KP3_A = np.array([60.0, 250.0, 250.0])
KD3_A = np.array([2.0, 2.0, 2.0])
```

在 `SCENE_MJX = ...` 之後加：

```python
SCENE_MJX_KP250 = str(_MODEL_DIR / "scene_flat_mjx_kp250.xml")   # RL v2 訓練模型（Task 2 產生）
```

`gait_baseline.py` 檔尾加：

```python
# =============================================================================
# ★ RL v2 基準（2026-09-08）：實機 trip17 兩趟零中止走完的 A_kp250_walk
# =============================================================================
# 每個數字直接對應 `outputs/A_kp250_walk.json` 的 params（測試釘住逐欄相等）。
# 與 BASELINE 的差異：LS 相位序列（E 文件）、kp250/abad60/kd2（H 文件）、
# x_off −30 / g_c 0.048 / z_sag 0.036（G 文件，kp250 重掃 + 實機錨點）。
BASELINE_A = {
    "gait": "walk_a",
    "seq": "ls",
    "duty": 0.80,
    "omega": 1.4,
    "mu_x": 1.80,
    "mu_y": 1.50,
    "d_step": 0.10,
    "d_step_y": 0.12,
    "x_off": -0.030,
    "g_c": 0.048,
    "z_sag": 0.036,
    "kp3": [60.0, 250.0, 250.0],
    "kd3": [2.0, 2.0, 2.0],
    "wheel_kd": 0.5,
    "wheel_mode": "damp",
}
```

`cpg_walk_max.py`：找到 `GAITS = {` 字典，在 `"walk_kp250"` 那筆之後加一筆（欄位名照既有 GAITS 的格式；若既有筆有 `phase` 鍵就給 `cpg_max.PHASE_WALK_LS`，若沒有就加上）：

```python
    # ★ RL v2 基準：實機 A_kp250_walk。用它 rollout 時**必須同時給**
    #   kp3=BASELINE_A["kp3"], kd3=BASELINE_A["kd3"], kd_wheel=0.5, z_sag=0.036
    #   （增益不在 GAITS 裡；local_infer_max 與 G1 腳本都這樣呼叫）。
    "walk_a": {"phase": cpg_max.PHASE_WALK_LS,
               "duty": gb.BASELINE_A["duty"], "omega": gb.BASELINE_A["omega"],
               "mu_x": gb.BASELINE_A["mu_x"], "x_off": gb.BASELINE_A["x_off"],
               "g_c": gb.BASELINE_A["g_c"], "d_step": gb.BASELINE_A["d_step"]},
```

（`cpg_walk_max` 已 `import gait_baseline as gb`；若 `GAITS` 定義在 import 之前，把該筆改用字面值並由測試釘住。）

- [ ] **Step 4: 跑，確認通過；全套不變紅**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_gait_baseline.py -q`
Expected: 全部 passed（含新 2 項）

- [ ] **Step 5: Commit**

```bash
git add task7/inference/max_model.py task7/inference/gait_baseline.py task7/inference/cpg_walk_max.py task7/tests/test_gait_baseline.py
git commit -m "feat(task7): BASELINE_A / KP3_A / GAITS walk_a —— RL v2 基準凍結（= A_kp250_walk）"
```

---

### Task 2: kp250 訓練模型（`make_mjx_model` 參數化）＋ G1

**Files:**
- Modify: `task7/model/zgws/make_mjx_model.py`（`build()` 簽名、致動器段、`build_all()`）
- Create（產生物）: `task7/model/zgws/zgws_mjx_kp250.xml`、`task7/model/zgws/scene_flat_mjx_kp250.xml`
- Create: `task7/inference/diag/g1_kp250.py`
- Test: `task7/tests/test_mjx_model.py`

**Interfaces:**
- Produces: `build(..., kp3=None, kd3=None)`；`max_model.SCENE_MJX_KP250` 可載入；`diag/g1_kp250.py` 印 A 步態在兩模型 12 擾動的對照表

- [ ] **Step 1: 寫測試（加在 `test_mjx_model.py` 檔尾）**

```python
def test_kp250_model_has_A_gains():
    m = mujoco.MjModel.from_xml_path(mm.SCENE_MJX_KP250)
    kp = m.actuator_gainprm[mm.LEG_ACT_IDX, 0]
    kv = -m.actuator_biasprm[mm.LEG_ACT_IDX, 2]
    np.testing.assert_allclose(kp, np.tile(mm.KP3_A, 4))
    np.testing.assert_allclose(kv, np.tile(mm.KD3_A, 4))
    wkv = -m.actuator_biasprm[mm.WHEEL_ACT_IDX, 2]
    np.testing.assert_allclose(wkv, mm.KD_WHEEL)
    # 舊模型不受影響
    m0 = mujoco.MjModel.from_xml_path(mm.SCENE_MJX)
    np.testing.assert_allclose(m0.actuator_gainprm[mm.LEG_ACT_IDX, 0], np.tile(mm.KP3, 4))


def test_kp250_model_walks_A_gait_like_mesh():
    """G1 的縮小版（單次、8 s）：速度差 10% 內、不跌倒。完整 12 擾動由 diag/g1_kp250.py 做。"""
    import gait_baseline as gb
    A = gb.BASELINE_A
    kw = dict(gait="walk_a", secs=8.0, kp3=A["kp3"], kd3=A["kd3"], kd_wheel=A["wheel_kd"],
              z_sag=A["z_sag"], quiet=True)
    a = cw.rollout(**kw)
    b = cw.rollout(scene=mm.SCENE_MJX_KP250, actuator_mode="position",
                   solver_iters=(6, 6), **kw)
    assert a["fell"] is None and b["fell"] is None
    assert abs(b["speed_travel"] - a["speed_travel"]) < 0.10 * max(a["speed_travel"], 1e-3)
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_mjx_model.py -q -k kp250`
Expected: FAIL（找不到 `scene_flat_mjx_kp250.xml`）

- [ ] **Step 3: 實作**

`make_mjx_model.py`：
1. import 改成 `from max_model import (KD3, KD3_A, KD_WHEEL, KP3, KP3_A, LEGS, PREFIX, ...`
2. `build()` 簽名加 `kp3=None, kd3=None`，致動器段改：

```python
    kp3 = KP3 if kp3 is None else np.asarray(kp3, float)
    kd3 = KD3 if kd3 is None else np.asarray(kd3, float)
    ...
            for j, (kp, kd, tau) in enumerate(zip(kp3, kd3, TAU_MAX3)):
```
3. header 的「組合」那行加 `  kp={kp3.tolist()} kd={kd3.tolist()}`
4. `build_all()` 檔尾加：

```python
    # ★ RL v2 訓練模型：與 zgws_mjx.xml 唯一差異是增益（A_kp250_walk 那組）
    xml = HERE / "zgws_mjx_kp250.xml"
    build(dst=str(xml), kp3=KP3_A, kd3=KD3_A)
    build_scene(str(HERE / "scene_flat_mjx_kp250.xml"), xml.name)
    print(f"[產生] {xml.name}  kp={KP3_A.tolist()} kd={KD3_A.tolist()}")
```

執行產生器：`conda run -n rbtdog python task7/model/zgws/make_mjx_model.py`，確認 `git diff` 只有 header 的 kp/kd 註解與新增兩檔（舊 xml 內容不變）。

`diag/g1_kp250.py`：

```python
#!/usr/bin/env python3
"""G1（RL v2）：kp250 訓練模型 vs 原始網格模型，A 步態 20 s × 12 擾動。

過關：speed_travel / bounce / support / min_lift / roll_pk 的中位數差 ±5%。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cpg_walk_max as cw   # noqa: E402
import gait_baseline as gb  # noqa: E402
import max_model as mm      # noqa: E402

KEYS = ("speed_travel", "bounce", "support", "min_lift", "roll_pk", "exec_front")
A = gb.BASELINE_A


def run(scene, mode, seeds, secs):
    rows = []
    for s in range(seeds):
        kw = dict(gait="walk_a", secs=secs, kp3=A["kp3"], kd3=A["kd3"],
                  kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True,
                  x_off=A["x_off"] + s * 1e-12)
        if scene:
            kw.update(scene=scene, actuator_mode=mode, solver_iters=(6, 6))
        rows.append(cw.rollout(**kw))
    return rows


def main():
    seeds, secs = 12, 20.0
    a = run(None, "torque_pd", seeds, secs)
    b = run(mm.SCENE_MJX_KP250, "position", seeds, secs)
    print(f"{'指標':14s} {'網格':>10s} {'kp250 MJX':>10s} {'差%':>7s}")
    ok = True
    for k in KEYS:
        ma, mb = np.median([r[k] for r in a]), np.median([r[k] for r in b])
        pct = 100 * (mb - ma) / max(abs(ma), 1e-9)
        flag = "" if abs(pct) <= 5 else "  ⚠️"
        ok &= abs(pct) <= 5
        print(f"{k:14s} {ma:10.4f} {mb:10.4f} {pct:+7.1f}{flag}")
    fa, fb = sum(r["fell"] is not None for r in a), sum(r["fell"] is not None for r in b)
    print(f"跌倒 網格 {fa}/{seeds}  MJX {fb}/{seeds}")
    print("G1", "✅ 通過" if ok and fa == fb == 0 else "❌ 未過")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑測試與 G1**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_mjx_model.py -q` → 全 passed
Run: `conda run --no-capture-output -n rbtdog python task7/inference/diag/g1_kp250.py` → 記下表格（進文件）；若某指標超 5%，回 `make_mjx_model` 查（不放寬門檻）。

- [ ] **Step 5: Commit**

```bash
git add task7/model/zgws/make_mjx_model.py task7/model/zgws/zgws_mjx_kp250.xml task7/model/zgws/scene_flat_mjx_kp250.xml task7/model/zgws/zgws_mjx.xml task7/inference/diag/g1_kp250.py task7/tests/test_mjx_model.py task7/inference/max_model.py
git commit -m "feat(task7): kp250 訓練模型（make_mjx_model 參數化）+ G1 腳本"
```

---

### Task 3: obs 70 維（`ACT_DIM=14`）

**Files:**
- Modify: `task7/inference/obs_max.py`
- Test: `task7/tests/test_obs_max.py`

- [ ] **Step 1: 改測試**：`test_layout_order_is_frozen` 的 `("last_action", 12)` 改 `("last_action", 14)`；`test_layout_sums_to_dim` 的 `== 68` 改 `== 70`。其他用到 `obs_max.ACT_DIM`/`OBS_DIM` 的測試不必改（`test_real_obs` 用符號）。

- [ ] **Step 2: 跑，確認失敗**：`conda run -n rbtdog python -m pytest task7/tests/test_obs_max.py -q` → 2 failed

- [ ] **Step 3: 實作**：`obs_max.py` 的 `ACT_DIM = 12` 改為

```python
ACT_DIM = 14          # 每腿 (mux, muy, omega) ×4 ＋ body sway (x, y)。2026-09-08 RL v2 起
```
docstring 表格 `last_action | 12` 改 `14`，並在「⚠️ 欄位順序一旦改動」那段加一句：「2026-09-08：`last_action` 12→14（加 sway），舊權重 `cpg_rl_max_params.pkl` 自此報廢（它本來就是 kp120 訓的）。」

- [ ] **Step 4: 全套測試**：`conda run -n rbtdog python -m pytest task7/tests -q` → 全 passed（`test_real_obs` 因用符號自動適應；`test_obs_compare` 的 `(rec.n, 68)` 斷言改成 `obs_max.OBS_DIM`）

- [ ] **Step 5: Commit**

```bash
git add task7/inference/obs_max.py task7/tests/test_obs_max.py task7/tests/test_obs_compare.py
git commit -m "feat(task7): obs 70 維 —— last_action 12→14（加 body sway）"
```

---

### Task 4: `cpg_walk_max` 補指標（`err_peak`、`roll_pk`、`roll_std`）

**Files:**
- Modify: `task7/inference/cpg_walk_max.py`（`Robot.__init__`、`Robot.step`、`Trace.summarize`）
- Test: `task7/tests/test_gait_baseline.py`（檔尾加）

**Interfaces:**
- Produces: `Robot.err_peak: np.ndarray(12)`（|q_des−q| 逐關節峰值，控制步粒度）；`summarize()` 新鍵 `roll_pk`（後半段 |roll| p99，度）、`roll_std`（度）、`err_peak`（list 12）、`err_peak_max`

- [ ] **Step 1: 測試**

```python
def test_summarize_has_roll_and_err_peaks():
    A = gb.BASELINE_A
    r = cw.rollout(gait="walk_a", secs=6.0, kp3=A["kp3"], kd3=A["kd3"],
                   kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True)
    assert 0.0 < r["roll_pk"] < 30.0 and 0.0 < r["roll_std"] <= r["roll_pk"]
    assert len(r["err_peak"]) == 12 and 0.0 < r["err_peak_max"] < 1.0
```

- [ ] **Step 2: 跑，確認失敗**：`-k roll_and_err` → KeyError

- [ ] **Step 3: 實作**

`Robot.__init__`（`self.tau_peak = np.zeros(12)` 之後）：
```python
        # 逐關節追蹤誤差峰值（控制步粒度，用 clip 後的 q_des）。實機 M9 的中止門檻是 0.6 rad，
        # RL v2 的護欄 0.45 —— 沒有這個量就沒辦法在模擬端做 G7。
        self.err_peak = np.zeros(12)
```
`Robot.step`，`q_des = np.clip(q_des, lo, hi)` 之後：
```python
        np.maximum(self.err_peak, np.abs(q_des - d.qpos[mm.LEG_QPOS_IDX]), out=self.err_peak)
```
`Trace.summarize` 的回傳字典，在 `"roll_mean"` 那行後加：
```python
            "roll_pk": float(np.percentile(np.abs(self.roll), 99)),
            "roll_std": float(np.std(self.roll)),
```
在 `"tau_peak"` 那行後加：
```python
            "err_peak": [float(v) for v in r.err_peak],
            "err_peak_max": float(r.err_peak.max()),
```

- [ ] **Step 4: 跑**：`conda run -n rbtdog python -m pytest task7/tests/test_gait_baseline.py -q` → 全 passed；`--gait walk --secs 60` 回歸（`+8.90/−0.25/−7.0`）不變：`conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --gait walk --secs 60 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add task7/inference/cpg_walk_max.py task7/tests/test_gait_baseline.py
git commit -m "feat(task7): Robot.err_peak、Trace roll_pk/roll_std —— G4/G7 要的量"
```

---

### Task 5: `rl_env_max.py` —— JAX 環境模組

**Files:**
- Create: `task7/inference/rl_env_max.py`
- Test: `task7/tests/test_rl_env_max.py`

**Interfaces:**
- Consumes: `gait_baseline.BASELINE_A`、`max_model.{KP3_A,KD3_A,SCENE_MJX_KP250,…}`、`cpg_max.PHASE_WALK_LS`、`obs_max.{OBS_DIM,ACT_DIM}`、`leg_kin`
- Produces: `cpg_init()`、`cpg_step(c,mux,muy,om,dt)`、`duty_remap`、`act_to_cmd(a)->(mux,muy,om,sway_tgt)`、`slew_sway(prev, tgt)`、`joint_targets_j(c, sway)`、`foot_targets_j(c, sway)`、`fk_j(k,q3)`、`baseline_action()->np(14)`、`MaxCpgEnv(scene=SCENE_MJX_KP250)`、`domain_randomize(sys, rng)`、常數 `W_*`, `TAU_BAR`, `ERR_BAR`, `SWAY_MAX`, `SWAY_SLEW`, `NOISE_*`, `METRIC_KEYS`

- [ ] **Step 1: 寫測試**

```python
# task7/tests/test_rl_env_max.py
"""RL v2 環境：JAX 版 CPG/IK 對 numpy 版逐點相符、14 維動作、sway 斜率、護欄、env 可跑。"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_max  # noqa: E402
import gait_baseline as gb  # noqa: E402
import leg_kin  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import rl_env_max as re  # noqa: E402

A = gb.BASELINE_A


def test_jax_cpg_and_ik_match_numpy():
    c_np = cpg_max.cpg_init(cpg_max.PHASE_WALK_LS)
    step_np = cpg_max.make_cpg_step(cpg_max.PHASE_WALK_LS)
    c_j = re.cpg_init()
    for _ in range(37):
        c_np = step_np(c_np, np.full(4, 1.8), np.full(4, 1.5), np.full(4, 1.4), mm.CTRL_DT)
        c_j = re.cpg_step(c_j, jnp.full(4, 1.8), jnp.full(4, 1.5), jnp.full(4, 1.4), mm.CTRL_DT)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    sway = (0.02, -0.03)
    q_np, _ = cpg_max.joint_targets(c_np, f0, A["x_off"], A["g_c"], A["d_step"], A["d_step_y"],
                                    A["duty"], ks, A["z_sag"], sway)
    q_j = np.asarray(re.joint_targets_j(c_j, jnp.array(sway)))
    assert np.abs(q_np - q_j).max() < 1e-4
    t_np = cpg_max.foot_targets(c_np, f0, A["x_off"], A["g_c"], A["d_step"], A["d_step_y"],
                                A["duty"], A["z_sag"], sway)
    t_j = np.asarray(re.foot_targets_j(c_j, jnp.array(sway)))
    assert np.abs(t_np - t_j).max() < 1e-5


def test_fk_j_matches_leg_kin():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.0, 1.0, 3)
        for k in range(4):
            np.testing.assert_allclose(np.asarray(re.fk_j(k, jnp.array(q))), leg_kin.fk(k, q),
                                       atol=1e-5)


def test_act_to_cmd_and_baseline_action():
    a = re.baseline_action()
    assert a.shape == (obs_max.ACT_DIM,) == (14,)
    mux, muy, om, sway = re.act_to_cmd(jnp.array(a))
    np.testing.assert_allclose(mux, A["mu_x"], atol=1e-3)
    np.testing.assert_allclose(muy, A["mu_y"], atol=1e-3)
    np.testing.assert_allclose(om, A["omega"], atol=1e-3)
    np.testing.assert_allclose(sway, 0.0, atol=1e-6)
    _, _, _, s = re.act_to_cmd(jnp.full(14, 10.0))
    np.testing.assert_allclose(s, re.SWAY_MAX, atol=1e-6)


def test_sway_slew_limits_step():
    prev = jnp.zeros(2)
    out = re.slew_sway(prev, jnp.array([0.06, -0.06]))
    np.testing.assert_allclose(out, [re.SWAY_SLEW, -re.SWAY_SLEW])
    out = re.slew_sway(jnp.array([0.05, 0.0]), jnp.array([0.052, 0.001]))
    np.testing.assert_allclose(out, [0.052, 0.001])


def test_guard_terms():
    tau = jnp.array([0.0] * 11 + [68.0])
    assert float(re.tau_barrier(tau)) == pytest.approx(100.0)       # (68−58)²
    err = jnp.array([0.0] * 11 + [0.65])
    assert float(re.err_barrier(err)) == pytest.approx(0.04)        # (0.65−0.45)²
    assert float(re.tau_barrier(jnp.full(12, 50.0))) == 0.0


def test_env_runs_baseline_without_done():
    env = re.MaxCpgEnv()
    assert env.observation_size == 70 and env.action_size == 14
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    assert s.obs.shape == (70,)
    a = jnp.array(re.baseline_action())
    for i in range(30):
        s = step(s, a)
        assert float(s.done) == 0.0, f"基準動作在第 {i} 步 done"
    assert set(re.METRIC_KEYS) <= set(s.metrics.keys())
    assert float(s.pipeline_state.qpos[2]) > 0.40


def test_domain_randomize_shapes():
    env = re.MaxCpgEnv()
    sys_r, in_axes = re.domain_randomize(env.sys, jax.random.split(jax.random.PRNGKey(1), 3))
    assert sys_r.geom_friction.shape[0] == 3
    assert sys_r.actuator_gainprm.shape[0] == 3
```

- [ ] **Step 2: 跑，確認失敗**：`conda run -n rbtdog python -m pytest task7/tests/test_rl_env_max.py -q` → `No module named 'rl_env_max'`

- [ ] **Step 3: 實作**

```python
# task7/inference/rl_env_max.py
"""RL v2 的 MJX 訓練環境（JAX）。Colab notebook 直接 import 這支，本機 CPU 可跑 reset/step 做測試。

★ 為什麼從 notebook 抽出來：上一版 env 住在 notebook 裡，改一行要重跑 Colab 才知道錯；
  抽成模組後 JAX vs numpy 的逐點對照、sway 斜率、護欄數值都是本機 pytest 釘住的。

設計：docs/superpowers/specs/2026-09-08-cpg-rl-v2-retrain-design.md
  - 動作 14 維：每腿 (mux, muy, ω) ×4 ＋ body sway (x, y)，sway ±0.06 m、斜率 0.004 m/步
  - obs 70 維（obs_max.OBS_LAYOUT），IMU 偏轉/gyro 偏置/延遲/雜訊依 H 文件實測
  - reward 優先序：姿態 > 前腳執行率與對稱 > 航向 > 速度；力矩/誤差護欄
  - 終止：翻倒、太低、|τ|>90 連續 3 步
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpg_max                  # noqa: E402
import gait_baseline as gb      # noqa: E402
import leg_kin                  # noqa: E402
import max_model as mm          # noqa: E402
import obs_max                  # noqa: E402

# ---------------------------------------------------------------- 常數（唯一來源：repo）
A = gb.BASELINE_A
CTRL_DT, SIM_DT = mm.CTRL_DT, mm.SIM_DT
N_FRAMES = int(round(CTRL_DT / SIM_DT))          # 10
A_CONV, W_COUP, N_CPG_SUB = mm.A_CONV, mm.W_COUP, mm.N_CPG_SUB
MU_MIN, MU_MAX = mm.MU_MIN, mm.MU_MAX
G_P = mm.G_P
DUTY, X_OFF, G_C, Z_SAG = A["duty"], A["x_off"], A["g_c"], A["z_sag"]
D_STEP, D_STEP_Y = A["d_step"], A["d_step_y"]
PHASE = np.asarray(cpg_max.PHASE_WALK_LS)
OMEGA_MIN, OMEGA_MAX = 0.0, 2.0
SWAY_MAX, SWAY_SLEW = 0.060, 0.004
ACT_DIM, OBS_DIM = obs_max.ACT_DIM, obs_max.OBS_DIM
assert ACT_DIM == 14 and OBS_DIM == 70

HOME12_np = np.array(mm.HOME12)
KNEE_SIGN_np = leg_kin.knee_sign_of(mm.HOME)
F0S_np = leg_kin.home_foot(mm.HOME)
LEG_QPOS_IDX = jnp.array(mm.LEG_QPOS_IDX)
LEG_QVEL_IDX = jnp.array(mm.LEG_QVEL_IDX)
LEG_ACT_IDX = jnp.array(mm.LEG_ACT_IDX)
WHEEL_ACT_IDX = jnp.array(mm.WHEEL_ACT_IDX)
KP_NOM = np.tile(np.asarray(mm.KP3_A), 4)
ABAD_IDX = jnp.array([0, 3, 6, 9])
FRONT = jnp.array([0, 1])
REAR = jnp.array([2, 3])

# ---- 隨機化（H 文件 §5）----
PUSH_EVERY, PUSH_VEL = 100, 0.6
IMU_TILT_DEG = 3.0
GYRO_BIAS = 0.02
DELAY_BASE = 1                     # 固定 1 步（實測 +15 ms）＋ 抽 0/1
NOISE_GRAV, NOISE_GYRO, NOISE_QPOS = 0.005, 0.002, 0.001
NOISE_QVEL = jnp.array([0.2, 0.2, 0.4] * 4)

# ---- 終止／護欄 ----
FALL_GRAV_Z = mm.FALL_GRAV_Z
MIN_HEIGHT = 0.29
TAU_BAR, ERR_BAR = 58.0, 0.45
TAU_KILL, KILL_STEPS = 90.0, 3
NOMINAL_HEIGHT = mm.NOMINAL_HEIGHT_WALK

# ---- reward 權重（起始值；notebook 校準格印每一項在基準動作上的值）----
W_ROLL, W_ROLLRATE, W_PITCH, W_PITCHRATE = 20.0, 0.05, 20.0, 0.05
W_EXEC, W_SYM, EXEC_SIGMA = 1.0, 5.0, 0.03
W_YAW, W_VX, W_VY, W_H, W_CLR = 1.0, 1.5, 0.5, 0.5, 1.5
W_VZ, W_OMEGA_VAR, W_ACT, W_TAU = 2.0, 0.5, 0.01, 3e-5
W_TAUBAR, W_ERRBAR = 0.01, 20.0
SYM_EMA = 0.05
METRIC_KEYS = ("height", "vx", "reward", "pitch", "roll", "clr", "vz", "yawerr", "vxerr",
               "exec_f", "exec_r", "tau_pk", "err_pk", "sway_x", "sway_y")


# ---------------------------------------------------------------- 四元數
def _qinv(q): return jnp.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    u = q[1:4]
    t = 2.0 * jnp.cross(u, v)
    return v + q[0] * t + jnp.cross(u, t)


def w2b(quat, v): return _qrot(_qinv(quat), v)


def _quat_rp(roll, pitch):
    """roll/pitch（rad，yaw=0）→ wxyz。與 m9_rec.rp_to_quat_wxyz 同式。"""
    cr, sr, cp, sp = jnp.cos(roll / 2), jnp.sin(roll / 2), jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    return jnp.array([cr * cp, sr * cp, cr * sp, -sr * sp])


# ---------------------------------------------------------------- CPG（與 cpg_max 逐行同）
PHASE_j = jnp.array(PHASE)
PHI = PHASE_j[None, :] - PHASE_j[:, None]


def cpg_init():
    return {"rx": jnp.full(4, 1.5), "rx_d": jnp.zeros(4),
            "ry": jnp.full(4, 1.5), "ry_d": jnp.zeros(4), "theta": PHASE_j}


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
        th = th + (2 * jnp.pi * omega + W_COUP * jnp.sum(rbar[None, :] * jnp.sin(diff), 1)) * h
    return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": jnp.mod(th, 2 * jnp.pi)}


def duty_remap(th, duty):
    ph = jnp.mod(th, 2 * jnp.pi) / (2 * jnp.pi)
    sw = 1.0 - duty
    return jnp.where(ph < sw, jnp.pi * ph / sw, jnp.pi + jnp.pi * (ph - sw) / duty)


# ---------------------------------------------------------------- 動作
def act_to_cmd(a):
    """14 維 → (mux(4), muy(4), ω(4), sway_target(2))。"""
    a = jnp.tanh(a)
    leg = a[:12].reshape(4, 3)
    mux = (leg[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    muy = (leg[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
    om = (leg[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, om, a[12:14] * SWAY_MAX


def slew_sway(prev, tgt):
    return prev + jnp.clip(tgt - prev, -SWAY_SLEW, SWAY_SLEW)


def baseline_action() -> np.ndarray:
    """A 步態對應的固定動作（sway=0）。G0 的標準答案。"""
    def inv(u, lo, hi):
        return float(np.arctanh(np.clip(2 * (u - lo) / (hi - lo) - 1, -0.999, 0.999)))
    return np.array([inv(A["mu_x"], MU_MIN, MU_MAX), inv(A["mu_y"], MU_MIN, MU_MAX),
                     inv(A["omega"], OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])


# ---------------------------------------------------------------- 運動學（與 leg_kin 逐行同）
SIDE_X_j, SIDE_Y_j = jnp.array(mm.SIDE_X), jnp.array(mm.SIDE_Y)
L_T, L_S = mm.L_THIGH, mm.L_SHANK
A2H_X, A2F_Y = mm.ABAD_TO_HIP_X, mm.ABAD_TO_FOOT_Y
REACH_LO, REACH_HI = abs(L_T - L_S) + 1e-6, L_T + L_S - 1e-6
F0S_j, KNEE_SIGN_j = jnp.array(F0S_np), jnp.array(KNEE_SIGN_np)


def ik_j(k, p, knee_sign):
    yp = SIDE_Y_j[k] * A2F_Y
    px, py, pz = p[0], p[1], p[2]
    zp = -jnp.sqrt(jnp.maximum(py * py + pz * pz - yp * yp, 0.0))
    q1 = jnp.arctan2(yp * pz - zp * py, yp * py + zp * pz)
    xp = px - SIDE_X_j[k] * A2H_X
    r = jnp.sqrt(xp * xp + zp * zp)
    r_new = jnp.clip(r, REACH_LO, REACH_HI)
    scale = jnp.where(r > 1e-12, r_new / r, 1.0)
    xp, zp = xp * scale, zp * scale
    cos_q3 = (r_new * r_new - L_T ** 2 - L_S ** 2) / (2 * L_T * L_S)
    q3 = jnp.sign(knee_sign) * jnp.arccos(jnp.clip(cos_q3, -1.0, 1.0))
    a = -(L_T + L_S * jnp.cos(q3))
    b = -L_S * jnp.sin(q3)
    q2 = jnp.arctan2(a * xp - b * zp, a * zp + b * xp)
    return jnp.array([q1, q2, q3])


def fk_j(k, q3):
    q1, q2, q3_ = q3[0], q3[1], q3[2]
    xp = SIDE_X_j[k] * A2H_X - L_T * jnp.sin(q2) - L_S * jnp.sin(q2 + q3_)
    zp = -L_T * jnp.cos(q2) - L_S * jnp.cos(q2 + q3_)
    yp = SIDE_Y_j[k] * A2F_Y
    c, s = jnp.cos(q1), jnp.sin(q1)
    return jnp.array([xp, yp * c - zp * s, yp * s + zp * c])


def foot_targets_j(c, sway):
    th = duty_remap(c["theta"], DUTY)
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    dx = -D_STEP * fx * jnp.cos(th) + X_OFF + sway[0]
    dy = D_STEP_Y * fy * jnp.cos(th) + sway[1]
    dz = jnp.where(jnp.sin(th) > 0, (G_C + Z_SAG) * jnp.sin(th), G_P * jnp.sin(th))
    return F0S_j + jnp.stack([dx, dy, dz], -1)


def joint_targets_j(c, sway):
    tgt = foot_targets_j(c, sway)
    return jnp.stack([ik_j(k, tgt[k], KNEE_SIGN_j[k]) for k in range(4)]).reshape(12)


def foot_actual_j(q12):
    q = q12.reshape(4, 3)
    return jnp.stack([fk_j(k, q[k]) for k in range(4)])


def swing_mask_j(theta):
    return (jnp.sin(duty_remap(theta, DUTY)) > 0).astype(jnp.float32)


# ---------------------------------------------------------------- 護欄
def tau_barrier(tau12):
    return jnp.sum(jnp.maximum(jnp.abs(tau12) - TAU_BAR, 0.0) ** 2)


def err_barrier(err12):
    return jnp.sum(jnp.maximum(jnp.abs(err12) - ERR_BAR, 0.0) ** 2)


# ---------------------------------------------------------------- env
class MaxCpgEnv(Env):
    def __init__(self, scene: str = mm.SCENE_MJX_KP250):
        m = mujoco.MjModel.from_xml_path(scene)
        assert m.opt.timestep == SIM_DT
        assert m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE, "致動器不是位置伺服"
        self._mj = m
        self.sys = mjx.put_model(m)
        self._lo = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 0])
        self._hi = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 1])
        gids = [mm._id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{mm.PREFIX[l]}_FOOT_LINK_COLL") for l in mm.LEGS]
        self._wheel_gids = jnp.array(gids)
        self._wheel_r = float(m.geom_size[gids[0]][0])
        self._init_q = self._settled_qpos(m)

    def _settled_qpos(self, m):
        d = mujoco.MjData(m)
        q = cpg_max.stand_targets(KNEE_SIGN_np, F0S_np, X_OFF)
        d.qpos[:3] = [0.0, 0.0, mm.NOMINAL_HEIGHT_KIN + 0.005]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[mm.LEG_QPOS_IDX] = q
        mujoco.mj_forward(m, d)
        for _ in range(int(1.5 / SIM_DT)):
            d.ctrl[mm.LEG_ACT_IDX] = q
            d.ctrl[mm.WHEEL_ACT_IDX] = 0.0
            mujoco.mj_step(m, d)
        assert 0.40 < d.qpos[2] < 0.52, f"站定高度 {d.qpos[2]:.3f} m 不合理"
        return jnp.array(d.qpos)

    def _ctrl(self, q_des):
        return jnp.zeros(16).at[LEG_ACT_IDX].set(q_des).at[WHEEL_ACT_IDX].set(0.0)

    def _wheel_clearance(self, data):
        return data.geom_xpos[self._wheel_gids, 2] - self._wheel_r

    def _obs(self, data, c, cmd, last_a, info):
        """70 維，含 IMU 偏轉、gyro 偏置、joint_vel 延遲。雜訊在 step 裡加。"""
        qm = info["imu_q"]
        grav = _qrot(qm, w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0])))
        gyro = _qrot(qm, data.qvel[3:6]) + info["gyro_bias"]
        return jnp.concatenate([
            grav, gyro,
            data.qpos[LEG_QPOS_IDX] - jnp.array(HOME12_np),
            info["qvel_prev"],                    # ★ joint_vel 延 1 步（driver 濾波實測）
            cmd, last_a,
            c["rx"], c["rx_d"], c["ry"], c["ry_d"], jnp.sin(c["theta"]), jnp.cos(c["theta"]),
        ])

    def reset(self, rng):
        k_cmd, k_bias, k_delay, k_tilt, k_noise = jax.random.split(rng, 5)
        data = mjx.make_data(self.sys).replace(qpos=self._init_q)
        data = data.replace(ctrl=self._ctrl(self._init_q[LEG_QPOS_IDX]))
        data = mjx.forward(self.sys, data)
        k_vx, k_wz, k_zero = jax.random.split(k_cmd, 3)
        vx = jax.random.uniform(k_vx, minval=0.05, maxval=0.35)
        wz = jnp.where(jax.random.uniform(k_zero) < 0.6, 0.0,
                       jax.random.uniform(k_wz, minval=-0.4, maxval=0.4))
        cmd = jnp.array([vx, wz])
        tilt = jax.random.uniform(k_tilt, (2,), minval=-IMU_TILT_DEG, maxval=IMU_TILT_DEG) * jnp.pi / 180
        c = cpg_init()
        z14 = jnp.zeros(ACT_DIM)
        info = {"rng": k_noise, "c": c, "cmd": cmd,
                "gyro_bias": jax.random.uniform(k_bias, (3,), minval=-GYRO_BIAS, maxval=GYRO_BIAS),
                "imu_q": _quat_rp(tilt[0], tilt[1]),
                "delay": DELAY_BASE + jax.random.bernoulli(k_delay, 0.5).astype(jnp.int32),
                "a_hist": jnp.zeros((3, ACT_DIM)), "last_a": z14,
                "sway": jnp.zeros(2), "qvel_prev": data.qvel[LEG_QVEL_IDX],
                "ema_f": jnp.zeros(()), "ema_r": jnp.zeros(()), "kill": jnp.zeros((), jnp.int32),
                "step": 0}
        obs = self._obs(data, c, cmd, z14, info)
        z = jnp.zeros(())
        metrics = {k: z for k in METRIC_KEYS}
        metrics["height"] = data.qpos[2]
        return State(data, obs, z, z, metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        a_hist = jnp.concatenate([action[None], info["a_hist"][:2]], 0)   # [now, t−1, t−2]
        act = a_hist[info["delay"]]
        mux, muy, om, sway_tgt = act_to_cmd(act)
        sway = slew_sway(info["sway"], sway_tgt)
        c = cpg_step(info["c"], mux, muy, om, CTRL_DT)
        q_des = jnp.clip(joint_targets_j(c, sway), self._lo, self._hi)

        data = state.pipeline_state
        rng, k_push, k_dir, k_obs = jax.random.split(info["rng"], 4)
        do_push = (info["step"] % PUSH_EVERY) == (PUSH_EVERY - 1)
        ang = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jnp.pi)
        mag = jax.random.uniform(k_push, minval=0.0, maxval=PUSH_VEL)
        kick = jnp.where(do_push, jnp.array([mag * jnp.cos(ang), mag * jnp.sin(ang), 0.0]), jnp.zeros(3))
        data = data.replace(qvel=data.qvel.at[0:3].add(kick))
        ctrl = self._ctrl(q_des)

        def one(carry, _):
            d, tpk = carry
            d = mjx.step(self.sys, d.replace(ctrl=ctrl))
            return (d, jnp.maximum(tpk, jnp.abs(d.actuator_force[LEG_ACT_IDX]))), None
        (data, tau_pk12), _ = jax.lax.scan(one, (data, jnp.zeros(12)), None, length=N_FRAMES)

        grav = w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        vb = w2b(data.qpos[3:7], data.qvel[0:3])
        cmd = info["cmd"]
        wz = data.qvel[5]
        q12 = data.qpos[LEG_QPOS_IDX]
        err12 = q_des - q12

        # ---- 執行率：擺動相足端 x（機身系、相對 ABAD）vs 指令
        tgt = foot_targets_j(c, sway)
        act_ft = foot_actual_j(q12)
        sw = swing_mask_j(c["theta"])
        dx = act_ft[:, 0] - tgt[:, 0]
        track = jnp.exp(-(dx / EXEC_SIGMA) ** 2)
        r_exec = jnp.sum(sw * track) / jnp.maximum(jnp.sum(sw), 1.0)
        ef = jnp.sum(sw[FRONT] * jnp.abs(dx[FRONT])) / jnp.maximum(jnp.sum(sw[FRONT]), 1.0)
        er = jnp.sum(sw[REAR] * jnp.abs(dx[REAR])) / jnp.maximum(jnp.sum(sw[REAR]), 1.0)
        ema_f = jnp.where(jnp.sum(sw[FRONT]) > 0, (1 - SYM_EMA) * info["ema_f"] + SYM_EMA * ef, info["ema_f"])
        ema_r = jnp.where(jnp.sum(sw[REAR]) > 0, (1 - SYM_EMA) * info["ema_r"] + SYM_EMA * er, info["ema_r"])

        r_vx = jnp.exp(-(vb[0] - cmd[0]) ** 2 / 0.02)
        r_vy = jnp.exp(-vb[1] ** 2 / 0.02)
        r_yaw = jnp.exp(-(wz - cmd[1]) ** 2 / 0.05)
        r_h = jnp.exp(-400.0 * (data.qpos[2] - NOMINAL_HEIGHT) ** 2)
        r_clr = jnp.mean(sw * jnp.clip(self._wheel_clearance(data) / G_C, 0.0, 1.0))
        c_act = jnp.sum((action - info["last_a"]) ** 2)
        c_tau = jnp.sum(data.actuator_force ** 2)
        reward = (W_VX * r_vx + W_VY * r_vy + W_YAW * r_yaw + W_H * r_h + W_CLR * r_clr
                  + W_EXEC * r_exec
                  - W_SYM * (ema_f - ema_r) ** 2
                  - W_ROLL * grav[1] ** 2 - W_ROLLRATE * data.qvel[3] ** 2
                  - W_PITCH * grav[0] ** 2 - W_PITCHRATE * data.qvel[4] ** 2
                  - W_OMEGA_VAR * jnp.var(om) - W_VZ * data.qvel[2] ** 2
                  - W_ACT * c_act - W_TAU * c_tau
                  - W_TAUBAR * tau_barrier(tau_pk12) - W_ERRBAR * err_barrier(err12))

        kill = jnp.where(jnp.max(tau_pk12) > TAU_KILL, info["kill"] + 1, 0)
        done = jnp.where((grav[2] > FALL_GRAV_Z) | (data.qpos[2] < MIN_HEIGHT) | (kill >= KILL_STEPS), 1.0, 0.0)

        obs = self._obs(data, c, cmd, action, info)
        n = jax.random.normal(k_obs, (OBS_DIM,))
        obs = (obs.at[0:3].add(NOISE_GRAV * n[0:3]).at[3:6].add(NOISE_GYRO * n[3:6])
                  .at[6:18].add(NOISE_QPOS * n[6:18]).at[18:30].add(NOISE_QVEL * n[18:30]))

        info.update({"rng": rng, "c": c, "last_a": action, "a_hist": a_hist, "sway": sway,
                     "qvel_prev": data.qvel[LEG_QVEL_IDX], "ema_f": ema_f, "ema_r": ema_r,
                     "kill": kill, "step": info["step"] + 1})
        metrics = {"height": data.qpos[2], "vx": vb[0], "reward": reward,
                   "pitch": jnp.abs(grav[0]) * 57.29578, "roll": jnp.abs(grav[1]) * 57.29578,
                   "clr": jnp.mean(self._wheel_clearance(data)) * 1000.0, "vz": jnp.abs(data.qvel[2]),
                   "yawerr": jnp.abs(wz - cmd[1]), "vxerr": jnp.abs(vb[0] - cmd[0]),
                   "exec_f": jnp.sum(sw[FRONT] * track[FRONT]) / jnp.maximum(jnp.sum(sw[FRONT]), 1.0),
                   "exec_r": jnp.sum(sw[REAR] * track[REAR]) / jnp.maximum(jnp.sum(sw[REAR]), 1.0),
                   "tau_pk": jnp.max(tau_pk12), "err_pk": jnp.max(jnp.abs(err12)),
                   "sway_x": jnp.abs(sway[0]) * 1000.0, "sway_y": jnp.abs(sway[1]) * 1000.0}
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=info)

    @property
    def observation_size(self): return OBS_DIM

    @property
    def action_size(self): return ACT_DIM

    @property
    def backend(self): return "mjx"


# ---------------------------------------------------------------- domain randomization
def domain_randomize(sys, rng):
    m = mujoco.MjModel.from_xml_path(mm.SCENE_MJX_KP250)
    base_id = mm._id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    leg_dof, wheel_dof = jnp.array(mm.LEG_QVEL_IDX), jnp.array(mm.WHEEL_QVEL_IDX)
    abad_dof = jnp.array(mm.LEG_QVEL_IDX[::3])
    leg_act, abad_act = LEG_ACT_IDX, jnp.array(mm.LEG_ACT_IDX[::3])
    kp_nom = jnp.array(KP_NOM)

    @jax.vmap
    def per_env(rng):
        k = jax.random.split(rng, 9)
        gf = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.4, maxval=1.4))
        s_kp = jax.random.uniform(k[1], minval=0.8, maxval=1.2)
        s_ab = jax.random.uniform(k[2], minval=0.6, maxval=1.4)      # ★ ABAD 範圍加大
        kv = jax.random.uniform(k[3], minval=0.5, maxval=2.0) * jnp.asarray(mm.KD3_A)[1]
        kp_leg = kp_nom * s_kp
        kp_leg = kp_leg.at[jnp.array([0, 3, 6, 9])].set(kp_nom[jnp.array([0, 3, 6, 9])] * s_ab)
        gain = sys.actuator_gainprm.at[leg_act, 0].set(kp_leg)
        bias = sys.actuator_biasprm.at[leg_act, 1].set(-kp_leg).at[leg_act, 2].set(-kv)
        bm = sys.body_mass * jax.random.uniform(k[4], (sys.nbody,), minval=0.9, maxval=1.1)
        bm = bm.at[base_id].add(jax.random.uniform(k[5], minval=0.0, maxval=5.0))
        fl = sys.dof_frictionloss
        fl = fl.at[leg_dof].multiply(jax.random.uniform(k[6], minval=0.5, maxval=1.5))
        fl = fl.at[abad_dof].multiply(jax.random.uniform(k[7], minval=0.6, maxval=1.4) / 1.0)
        fl = fl.at[wheel_dof].set(jax.random.uniform(k[8], minval=0.10, maxval=0.25))
        return gf, gain, bias, bm, fl

    gf, gain, bias, bm, fl = per_env(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.replace(geom_friction=0, actuator_gainprm=0, actuator_biasprm=0,
                              body_mass=0, dof_frictionloss=0)
    sys = sys.replace(geom_friction=gf, actuator_gainprm=gain, actuator_biasprm=bias,
                      body_mass=bm, dof_frictionloss=fl)
    return sys, in_axes
```

（`domain_randomize` 的 `abad_act` 未用可刪；ABAD kp 用索引 [0,3,6,9] 直接設。）

- [ ] **Step 4: 跑測試**：`conda run -n rbtdog python -m pytest task7/tests/test_rl_env_max.py -q -x`
Expected: 7 passed（env 測試在 CPU 約 1–3 分鐘：mjx jit）。若 `test_env_runs_baseline_without_done` 的高度斷言失敗，印 `qpos[2]` 判斷是站定高度問題（調 `_settled_qpos` 的 assert 範圍）還是模型問題。

- [ ] **Step 5: Commit**

```bash
git add task7/inference/rl_env_max.py task7/tests/test_rl_env_max.py
git commit -m "feat(task7): rl_env_max —— RL v2 JAX 環境（14 維+sway、70 維 obs、實測隨機化、力矩護欄）"
```

---

### Task 6: `local_infer_max.py` v2（14 維、sway、擾動、G3–G7）

**Files:**
- Modify: `task7/inference/local_infer_max.py`
- Test: `task7/tests/test_local_infer_max.py`（新）

**Interfaces:**
- Produces: `baseline_action()->(14,)`、`act_to_cmd(a)->(mux,muy,om,sway_tgt)`、`slew_sway(prev,tgt)`（numpy，與 rl_env_max 同數值）、`run_once(args, seed)->dict`、`run(args)->dict`（含 `perturb` 彙總與 `gates`）、CLI `--perturb N --compare --secs --vx --wz --video --scene`

- [ ] **Step 1: 測試**

```python
# task7/tests/test_local_infer_max.py
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cpg_walk_max as cw  # noqa: E402
import gait_baseline as gb  # noqa: E402
import local_infer_max as li  # noqa: E402
import rl_env_max as re  # noqa: E402


def test_baseline_action_and_act_to_cmd_match_env_module():
    a = li.baseline_action()
    np.testing.assert_allclose(a, re.baseline_action())
    mux, muy, om, sw = li.act_to_cmd(np.full(14, 0.3))
    mux2, muy2, om2, sw2 = re.act_to_cmd(np.full(14, 0.3))
    np.testing.assert_allclose(mux, np.asarray(mux2), atol=1e-6)
    np.testing.assert_allclose(sw, np.asarray(sw2), atol=1e-6)
    np.testing.assert_allclose(li.slew_sway(np.zeros(2), np.array([0.06, -0.01])),
                               [re.SWAY_SLEW, -0.004])


def test_dummy_matches_open_loop_walk_a():
    """G0：固定動作（sway=0）必須與開迴路 A 步態 rollout 逐位相同。"""
    A = gb.BASELINE_A
    args = SimpleNamespace(params="", dummy=True, secs=6.0, vx=0.15, wz=0.0, video=False,
                           scene=None, perturb=1, compare=False)
    res = li.run(args)
    ref = cw.rollout(gait="walk_a", secs=6.0, kp3=A["kp3"], kd3=A["kd3"],
                     kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True)
    for k in ("speed_travel", "bounce", "support", "min_lift", "roll_pk", "exec_front"):
        assert abs(res[k] - ref[k]) < 1e-9, k


def test_gates_report_keys():
    args = SimpleNamespace(params="", dummy=True, secs=4.0, vx=0.15, wz=0.0, video=False,
                           scene=None, perturb=2, compare=True)
    res = li.run(args)
    assert set(res["gates"]) >= {"G3", "G4", "G5", "G6", "G7"}
    assert res["n_perturb"] == 2 and "baseline" in res
```

- [ ] **Step 2: 跑，確認失敗**：`-k baseline_action` → shape 12 vs 14 或 unpack 錯

- [ ] **Step 3: 實作** —— 重寫 `local_infer_max.py` 的常數與 `run`：

```python
# 取代原本的 baseline_action / OMEGA_* / act_to_cmd：
OMEGA_MIN, OMEGA_MAX = 0.0, 2.0
SWAY_MAX, SWAY_SLEW = 0.060, 0.004            # ⚠️ 與 rl_env_max 同值（測試釘住）
A = gb.BASELINE_A
PHASE = cpg_max.PHASE_WALK_LS
GATE_TAU, GATE_ERR = 70.0, 0.6                # 實機 M9 門檻
SIM2REAL_TAU, SIM2REAL_ERR = 1.2, 1.14


def baseline_action() -> np.ndarray:
    def inv(u, lo, hi):
        return float(np.arctanh(np.clip(2 * (u - lo) / (hi - lo) - 1, -0.999, 0.999)))
    return np.array([inv(A["mu_x"], mm.MU_MIN, mm.MU_MAX), inv(A["mu_y"], mm.MU_MIN, mm.MU_MAX),
                     inv(A["omega"], OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])


def act_to_cmd(a: np.ndarray):
    a = np.tanh(np.asarray(a, dtype=float))
    leg = a[:12].reshape(4, 3)
    mux = (leg[:, 0] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
    muy = (leg[:, 1] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
    om = (leg[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, om, a[12:14] * SWAY_MAX


def slew_sway(prev, tgt):
    return prev + np.clip(np.asarray(tgt) - prev, -SWAY_SLEW, SWAY_SLEW)


def run_once(args, infer, seed: int = 0) -> dict:
    """一次 rollout。`seed` 只用來給 x_off 加 1e-12·seed 的皮米擾動（同 cpg_sweep_max）。"""
    import mujoco
    scene = args.scene or DEFAULT_SCENE
    mode = "position" if scene != mm.SCENE else "torque_pd"
    r = cw.Robot(scene=scene, actuator_mode=mode, kp3=A["kp3"], kd3=A["kd3"],
                 kd_wheel=A["wheel_kd"], solver_iters=(6, 6) if mode == "position" else None)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(PHASE)
    x_off = A["x_off"] + seed * 1e-12
    r.reset_standing(cpg_max.stand_targets(ks, f0, x_off), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, x_off))
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    ren = cam = None
    frames = []
    if args.video and seed == 0:
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(r.m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)
    c = cpg_max.cpg_init(PHASE)
    n = int(args.secs / mm.CTRL_DT)
    cmd = np.array([args.vx, args.wz])
    last_a = np.zeros(obs_max.ACT_DIM)
    sway = np.zeros(2)
    n_reach, om_hist, sway_hist = 0, [], []
    tr = cw.Trace(r, n, args.secs, A["omega"], PHASE, A["duty"])
    for i in range(n):
        obs = obs_max.build_obs(r.d, c, cmd, last_a)
        a = infer(obs)
        mux, muy, om, sw_t = act_to_cmd(a)
        sway = slew_sway(sway, sw_t)
        om_hist.append(om.copy()); sway_hist.append(sway.copy())
        c = step(c, mux, muy, om, mm.CTRL_DT)
        sw_arg = None if not np.any(sway) else (float(sway[0]), float(sway[1]))
        tgt = cpg_max.foot_targets(c, f0, x_off, A["g_c"], A["d_step"], A["d_step_y"], A["duty"],
                                   A["z_sag"], sw_arg)
        q_des, nc = cpg_max.joint_targets(c, f0, x_off, A["g_c"], A["d_step"], A["d_step_y"],
                                          A["duty"], ks, A["z_sag"], sw_arg)
        n_reach += nc
        r.step(q_des)
        tr.record(c["theta"], tgt[:, 0])
        last_a = a
        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.30]
            cam.distance, cam.elevation, cam.azimuth = 2.0, -10, 90
            ren.update_scene(r.d, cam)
            frames.append(ren.render())
    om_arr, sw_arr = np.asarray(om_hist), np.asarray(sway_hist)
    res = tr.summarize(n_reach, extra={
        "scene": scene, "seed": seed, "cmd_vx": args.vx, "cmd_wz": args.wz, "dummy": bool(args.dummy),
        "omega_mean": float(om_arr.mean()), "omega_min": float(om_arr.min()), "omega_max": float(om_arr.max()),
        "sway_x_abs": float(np.abs(sw_arr[:, 0]).mean() * 1000), "sway_y_abs": float(np.abs(sw_arr[:, 1]).mean() * 1000)})
    res["yaw_rate"] = res["yaw_total"] / args.secs
    res["_frames"] = frames
    return res


def _med(rs, k):
    return float(np.median([r[k] for r in rs]))


def run(args) -> dict:
    """跑 `perturb` 個擾動（policy），可選同擾動的開迴路對照；回傳中位數彙總與 G3–G7 判定。"""
    fixed = baseline_action()
    infer = (lambda _o: fixed) if args.dummy else load_policy(args.params)
    rs = [run_once(args, infer, s) for s in range(max(1, args.perturb))]
    res = dict(rs[0]); res.pop("_frames", None)
    for k in ("speed_travel", "bounce", "support", "min_lift", "roll_pk", "roll_std",
              "exec_front", "exec_rear", "yaw_total", "yaw_rate", "err_peak_max"):
        res[k] = _med(rs, k)
    res["n_perturb"] = len(rs)
    res["fell_n"] = sum(r["fell"] is not None for r in rs)
    res["tau_peak_max"] = float(max(max(r["tau_peak"]) for r in rs))
    res["err_peak_max"] = float(max(r["err_peak_max"] for r in rs))
    base = None
    if args.compare:
        bargs = argparse.Namespace(**{**vars(args), "dummy": True, "video": False})
        brs = [run_once(bargs, lambda _o: fixed, s) for s in range(len(rs))]
        base = {k: _med(brs, k) for k in ("roll_pk", "roll_std", "exec_front", "yaw_total", "speed_travel")}
        res["baseline"] = base
    g = {}
    g["G3"] = res["fell_n"] == 0
    g["G4"] = (base is not None and res["roll_pk"] <= 0.4 * base["roll_pk"]
               and res["roll_std"] <= 0.4 * base["roll_std"])
    g["G5"] = res["exec_front"] >= 0.9 and abs(res["exec_front"] - res["exec_rear"]) < 0.15
    g["G6"] = abs(res["yaw_total"]) * (60.0 / args.secs) < 5.0 if args.wz == 0 else None
    g["G7"] = (res["tau_peak_max"] * SIM2REAL_TAU < GATE_TAU) and (res["err_peak_max"] * SIM2REAL_ERR < GATE_ERR)
    res["gates"] = g

    src = "基準固定動作" if args.dummy else Path(args.params).name
    cw.report(res, f"[推論 v2] {src}  cmd=(vx {args.vx:.2f}, wz {args.wz:+.2f})  擾動 {len(rs)}"
                   f"  ω {res['omega_min']:.2f}~{res['omega_max']:.2f}  sway |x| {res['sway_x_abs']:.0f} |y| {res['sway_y_abs']:.0f} mm")
    print(f"[G3] 跌倒 {res['fell_n']}/{len(rs)} → {'✅' if g['G3'] else '❌'}")
    if base:
        print(f"[G4] roll 峰值 {res['roll_pk']:.2f}° (基準 {base['roll_pk']:.2f}) std {res['roll_std']:.2f} (基準 {base['roll_std']:.2f})"
              f" → {'✅' if g['G4'] else '❌'}（要兩者都降 ≥60%）")
    print(f"[G5] 執行率 前 {res['exec_front']:.2f} 後 {res['exec_rear']:.2f} → {'✅' if g['G5'] else '❌'}")
    if g["G6"] is not None:
        print(f"[G6] 總偏航 {res['yaw_total']:+.1f}° / {args.secs:.0f}s（換算 60 s {res['yaw_total']*60/args.secs:+.1f}°）→ {'✅' if g['G6'] else '❌'}")
    print(f"[G7] 峰值力矩 {res['tau_peak_max']:.1f}×{SIM2REAL_TAU}={res['tau_peak_max']*SIM2REAL_TAU:.1f} (<{GATE_TAU})"
          f"  誤差 {res['err_peak_max']:.3f}×{SIM2REAL_ERR}={res['err_peak_max']*SIM2REAL_ERR:.3f} (<{GATE_ERR}) → {'✅' if g['G7'] else '❌ 不上機'}")
    frames = rs[0].get("_frames") or []
    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "cpg_rl_max_v2.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res
```

`main()` 加 `--perturb`（int，預設 1）、`--compare`（store_true）；docstring 更新（14 維、sway、v2、G3–G7）。`load_policy` 不變（`obs_max.OBS_DIM/ACT_DIM` 已是 70/14）。

- [ ] **Step 4: 跑**：`conda run -n rbtdog python -m pytest task7/tests/test_local_infer_max.py -q` → 3 passed；`conda run --no-capture-output -n rbtdog python task7/inference/local_infer_max.py --dummy --secs 20 --perturb 3 --compare` → 印 G3–G7（G4 對基準必為 ❌，正常）

- [ ] **Step 5: Commit**

```bash
git add task7/inference/local_infer_max.py task7/tests/test_local_infer_max.py
git commit -m "feat(task7): local_infer_max v2 —— 14 維+sway、LS、A 增益、12 擾動、G3–G7 判定"
```

---

### Task 7: notebook v2（由腳本產生）

**Files:**
- Create: `task7/notebooks/build_nb_v2.py`
- Modify（產生物）: `task7/notebooks/cpg_rl_max_colab.ipynb`
- Test: `task7/tests/test_notebook_v2.py`

- [ ] **Step 1: 測試**

```python
# task7/tests/test_notebook_v2.py
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "cpg_rl_max_colab.ipynb"


def test_notebook_v2_contract():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'BRANCH = "main"' in src
    assert "import rl_env_max" in src and "MaxCpgEnv" in src
    assert "brax==0.14.2" in src and "mujoco==3.10.0" in src
    assert "cpg_rl_max_v2_params.pkl" in src
    assert "class MaxCpgEnv" not in src          # env 不再住在 notebook 裡
    assert "gb.BASELINE[" not in src             # 只能用 BASELINE_A（經 rl_env_max）
```

- [ ] **Step 2: 跑，確認失敗**

- [ ] **Step 3: 實作 `build_nb_v2.py`**（`python task7/notebooks/build_nb_v2.py` 重寫 ipynb；保留原 cell 0–2 的安裝/版本文字，把 BRANCH 改 main）

```python
#!/usr/bin/env python3
"""產生 RL v2 的 Colab notebook。env 在 task7/inference/rl_env_max.py，這裡只有流程。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cpg_rl_max_colab.ipynb"
OLD = json.loads(OUT.read_text(encoding="utf-8"))
install_src = "".join(OLD["cells"][1]["source"])
version_src = "".join(OLD["cells"][2]["source"])

md0 = """# CPG-RL **v2** 訓練：智元 D1 Max · MJX · Colab GPU（2026-09-08）

基準 `A_kp250_walk`（LS / kp250 / abad60 / kd2），14 維動作（每腿 mux/muy/ω ＋ body sway x,y），
70 維 obs，隨機化與護欄依 `task7/docs/H_實機obs盤點_2026-09-08.md` 與
`docs/superpowers/specs/2026-09-08-cpg-rl-v2-retrain-design.md`。

**env 住在 repo（`task7/inference/rl_env_max.py`），本 notebook 只有：安裝 → clone → 校準 → 訓練 → 存檔。**
改 env 請改 repo、push、重新 clone，不要在這裡貼程式。
"""
clone_src = '''import os, subprocess, sys
REPO = "https://github.com/HGLLLLL/RBTDOG_SIM.git"
BRANCH = "main"
DEST = "rbtdog_sim"
if not os.path.exists(DEST):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, DEST], check=True)
sys.path.insert(0, f"{DEST}/task7/inference")
print(subprocess.run(["git", "-C", DEST, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
'''
import_src = '''import jax, jax.numpy as jnp, numpy as np, mujoco
import rl_env_max as re
import obs_max, gait_baseline as gb, max_model as mm
print("obs", re.OBS_DIM, "act", re.ACT_DIM, "scene", mm.SCENE_MJX_KP250)
print("基準", gb.BASELINE_A)
assert (re.OBS_DIM, re.ACT_DIM) == (70, 14)
'''
calib_src = '''# ---- 基準校準：A 步態是固定動作（sway=0），每一項 reward 在它身上值多少？----
env = re.MaxCpgEnv()
assert env.sys.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
A_BASE = jnp.array(re.baseline_action())
s = jit_reset(jax.random.PRNGKey(0))
print("reset ok, obs", s.obs.shape, "height %.4f" % float(s.pipeline_state.qpos[2]))
import time as _t; _t0 = _t.time()
acc = {k: [] for k in re.METRIC_KEYS}
for i in range(500):
    s = jit_step(s, A_BASE)
    assert float(s.done) == 0.0, f"基準動作第 {i} 步 done"
    if i >= 250:
        for k in re.METRIC_KEYS: acc[k].append(float(s.metrics[k]))
print(f"[基準 10 s] " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in acc.items()) + f"  ({_t.time()-_t0:.0f}s)")
print("[對照] 本機 local_infer_max --dummy 的 speed_travel / roll_pk / exec_front 要與此同量級")
print(f"[護欄] tau_pk 平均 {np.mean(acc['tau_pk']):.1f}（TAU_BAR {re.TAU_BAR}）  err_pk 平均 {np.mean(acc['err_pk']):.3f}（ERR_BAR {re.ERR_BAR}）")
print("     基準步態不該碰到護欄 —— 若 tau_pk 平均 > 50 或 err_pk > 0.4，先回本機查再訓")
'''
train_src = '''import functools, time
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
env = re.MaxCpgEnv()
network_factory = functools.partial(ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(256, 256, 256))
TIMESTEPS = 60_000_000
train_fn = functools.partial(ppo.train, num_timesteps=TIMESTEPS, num_evals=20, episode_length=1000,
    num_envs=2048, batch_size=256, num_minibatches=32, unroll_length=20, num_updates_per_batch=4,
    learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97, normalize_observations=True,
    network_factory=network_factory, randomization_fn=re.domain_randomize, seed=0)
_t0 = time.time(); rewards = []
def progress(step, metrics):
    r = float(metrics.get("eval/episode_reward", 0.0)); rewards.append((step, r))
    L = float(metrics.get("eval/avg_episode_length", 1.0)) or 1.0
    ps = lambda k: float(metrics.get(f"eval/episode_{k}", 0.0)) / L
    el = time.time() - _t0; rate = step / max(el, 1e-9)
    print(f"step {step:>11,} R {r:7.2f} | roll {ps('roll'):4.2f}° pitch {ps('pitch'):4.2f}° "
          f"exec f/r {ps('exec_f'):.2f}/{ps('exec_r'):.2f} yawerr {ps('yawerr'):.3f} vxerr {ps('vxerr'):.3f} "
          f"tau_pk {ps('tau_pk'):5.1f} err_pk {ps('err_pk'):.3f} sway {ps('sway_x'):.0f}/{ps('sway_y'):.0f}mm "
          f"len {L:.0f} | {el:.0f}s {rate/1e3:.0f}k/s → {TIMESTEPS/max(rate,1)/60:.0f} min")
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print("training done")
'''
plot_src = '''import matplotlib.pyplot as plt
plt.plot([s for s, _ in rewards], [r for _, r in rewards], marker="o"); plt.xlabel("env steps"); plt.ylabel("eval reward"); plt.grid(True); plt.show()
'''
save_src = '''from brax.io import model
model.save_params("cpg_rl_max_v2_params.pkl", params)
print("已存 cpg_rl_max_v2_params.pkl → 下載放 task7/weights/，本機跑：")
print("conda run --no-capture-output -n rbtdog python task7/inference/local_infer_max.py --params task7/weights/cpg_rl_max_v2_params.pkl --secs 60 --perturb 12 --compare --video")
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU"},
      "cells": [cell("markdown", md0), cell("code", install_src), cell("code", version_src),
                cell("code", clone_src), cell("code", import_src), cell("code", calib_src),
                cell("code", train_src), cell("code", plot_src), cell("code", save_src)]}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("→", OUT)
```

- [ ] **Step 4: 產生並測**：`python3 task7/notebooks/build_nb_v2.py && conda run -n rbtdog python -m pytest task7/tests/test_notebook_v2.py -q`
（`version_src` 原本斷言 `brax.__version__ == "0.14.2"`，保留。）

- [ ] **Step 5: Commit**

```bash
git add task7/notebooks/build_nb_v2.py task7/notebooks/cpg_rl_max_colab.ipynb task7/tests/test_notebook_v2.py
git commit -m "feat(task7): Colab notebook v2 —— env 改 import rl_env_max，校準格印每項 reward 與護欄"
```

---

### Task 8: 訓練前收尾：G0/G1/G2 全跑、設計文件、HANDOFF、push

**Files:**
- Create: `task7/docs/CPG-RL_v2_設計_2026-09-08.md`
- Modify: `task7/HANDOFF.md`

- [ ] **Step 1: G0–G2**

```bash
conda run -n rbtdog python -m pytest task7/tests -q                         # G2 全綠
conda run --no-capture-output -n rbtdog python task7/inference/local_infer_max.py --dummy --secs 20 --perturb 3 --compare   # G0
conda run --no-capture-output -n rbtdog python task7/inference/diag/g1_kp250.py   # G1
```
把三個輸出的關鍵數字抄進設計文件 §驗收。

- [ ] **Step 2: 寫 `task7/docs/CPG-RL_v2_設計_2026-09-08.md`**：內容＝spec 的 §1–§9 ＋「G0/G1/G2 結果」＋「Colab 使用流程」（push → 開 notebook → 全部執行 → 第 6 格看基準校準 → 第 7 格第一個 eval 看步率 → 下載 pkl → 本機 `local_infer_max --perturb 12 --compare --video --secs 60`，再跑 `--secs 180 --perturb 12` 與 `--wz 0.3`）。

- [ ] **Step 3: HANDOFF 最上面加一節**（RL v2 工具鏈完成、等 Colab 訓練；列 G3–G8 指令與過關門檻；提醒 G7 不過不上機；狗上推論另開 spec）

- [ ] **Step 4: Commit + push**

```bash
git add task7/docs/CPG-RL_v2_設計_2026-09-08.md task7/HANDOFF.md
git commit -m "docs(task7): CPG-RL v2 設計文件 + G0/G1/G2 結果 + HANDOFF；可開訓"
git push origin main
```

---

### Task 9（訓練後）：G3–G8 驗收與結果文件

**Files:**
- Create: `task7/weights/cpg_rl_max_v2_params.pkl`（Colab 下載）
- Create: `task7/docs/CPG-RL_v2_結果_<日期>.md`
- Modify: `task7/HANDOFF.md`

- [ ] **Step 1**：`conda run --no-capture-output -n rbtdog python task7/inference/local_infer_max.py --params task7/weights/cpg_rl_max_v2_params.pkl --secs 60 --perturb 12 --compare --video` → G3–G7 一次印完
- [ ] **Step 2**：`--secs 180 --perturb 12`（G3 長時）、`--wz 0.3 --secs 20`、`--wz -0.3 --secs 20`、`--vx 0.30 --secs 20`
- [ ] **Step 3**：結果文件（表格照 8/27 §6.1 的格式，加 roll_pk/roll_std/exec/tau_pk/err_pk；每個 G 的 ✅/❌；未過的寫原因與下一輪要動的權重）；HANDOFF；commit。
- [ ] **Step 4**：G7 ❌ → **不上機**，回 Task 5 調權重重訓；G7 ✅ 且 G3–G6 ✅ → 開「狗上推論」spec。

---

## Self-review

- **Spec coverage**：§1 動作→Task 5/6；§2 obs 與隨機化→Task 3/5；§3 reward→Task 5（權重常數＋校準格 Task 7）；§4 終止→Task 5；§5 模型與增益→Task 1/2；§6 訓練→Task 7/8；§7 G0–G8→Task 2（G1）、6（G3–G7 工具）、8（G0–G2 執行）、9（G3–G8 執行）；§8 交付物全部對到；§9 風險寫進設計文件（Task 8）。
- **Placeholder**：無 TBD；Task 9 依賴權重，步驟是具體指令。
- **型別一致**：`baseline_action()` 皆回 (14,)；`act_to_cmd` 皆回 4-tuple；`slew_sway(prev,tgt)`；`BASELINE_A` 鍵名在 Task 1/5/6 一致（`kp3/kd3/wheel_kd/z_sag/x_off/g_c/d_step/d_step_y/duty/omega/mu_x/mu_y`）；`METRIC_KEYS` 在 Task 5 定義、Task 7 使用；`SCENE_MJX_KP250` 在 Task 1 定義、Task 2 產生、Task 5 使用。
- **注意**：Task 5 的 `domain_randomize` 內 `kv` 取 `KD3_A[1]`（=2.0）乘 0.5–2.0；ABAD 用索引 `[0,3,6,9]`。
