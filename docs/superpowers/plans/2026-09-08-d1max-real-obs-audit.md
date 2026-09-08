# D1 Max 實機 obs 盤點 —— 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M6 錄檔轉成 68 維 obs、判定 IMU 順序/單位/軸向、把實機錄到的指令逐筆回放進 MuJoCo 並逐欄對照，產出訓練端可讀的雜訊模型與延遲；另加一支狗上電腦探測。

**Architecture:** 四支離線工具共用一個 M6 錄檔載入器（`m6_rec.py`）；`real_obs.py` 造假 `MjData` 餵**同一支** `obs_max.build_obs`；`imu_check.py` 用加速度計（重力）、腿 FK（四輪共面）、四元數微分（角速度）三個獨立參考判定 IMU；`obs_compare.py` 用錄到的 `des/kp/kd` 驅動 MuJoCo（同 `replay_standup.py` 的做法）→ 兩邊 obs 逐筆同時刻可比，不需要對齊。

**Tech Stack:** Python 3、numpy、mujoco 3.10（rbtdog conda 環境）、pytest。狗上程式只用標準函式庫。

## Global Constraints

- 執行測試一律：`conda run -n rbtdog python -m pytest task7/tests -q`（本機系統 python 沒有 mujoco）
- 不改 `M9_gait.py`、不改 `M6_load_probe.py`、不寫 `joint_cmd`、不改 `obs_max.OBS_LAYOUT`
- 既有 719 項測試必須維持全綠
- 四元數：`imu_central` 原始順序 `xyzw`（待 ② 定案）；`cpg_max.w2b` 吃 `wxyz`
- 座標：`joint_state`/`joint_cmd` 是**馬達座標**；MJCF/obs 是**控制器座標**；換算只用 `coord.to_ctrl` / `coord.SIGN`
- 腿序：SHM `fl,fr,bl,br`；MJCF `FR,FL,RR,RL`；對應一律按名稱 `{"FR":"fr","FL":"fl","RR":"br","RL":"bl"}`
- 機身座標 x 前、y 左、z 上；**+roll = 右側低**、**+pitch = 頭低**、**+yaw = 左轉（由上看逆時針）**
- 加速度計靜止時讀 **+9.8 在 z**（specific force），重力方向 = `−acc/|acc|`
- 檔案路徑：工具在 `task7/inference/`，狗上探測在 `task7/realbot/`，測試在 `task7/tests/`

## File Structure

| 檔案 | 責任 |
|---|---|
| `task7/inference/m6_rec.py`（新） | 載入 `m6_record/1` JSON：展開 `_fold` 過的純量、驗 schema 與 16 關節齊全、提供 `(N,)` 陣列 |
| `task7/inference/real_obs.py`（新） | 錄檔 → 控制器座標的 q/v/des（MJCF 順序）、quat 順序轉換、`RealFrame`、`obs_series` |
| `task7/inference/imu_check.py`（新） | 階 I 判讀：quat 順序、gyro 單位/軸向/偏置、安裝偏置、更新率、量化、第一個 roll 事件、轉向符號；寫 `outputs/imu_check.json` |
| `task7/inference/obs_compare.py`（新） | 階 II：錄檔驅動 MuJoCo 回放、逐欄統計、雜訊/延遲、`joint_vel` 分析；寫 `outputs/obs_noise_model.json` + markdown 表 |
| `task7/realbot/M_env_probe.py`（新） | 狗上：numpy/onnxruntime 有無、CPU 親和性、純 Python / numpy MLP 前向時間；存 `~/m_logs/ENV_*.json` |
| `task7/tests/test_m6_rec.py`、`test_real_obs.py`、`test_imu_check.py`、`test_obs_compare.py`、`test_m_env_probe.py`（新） | 對應測試 |

---

### Task 1: `m6_rec.py` —— M6 錄檔載入器

**Files:**
- Create: `task7/inference/m6_rec.py`
- Test: `task7/tests/test_m6_rec.py`

**Interfaces:**
- Produces: `Rec` dataclass（`t (N,)`, `hz`, `j[name][field] (N,)`, `imu (N,10)`, `acc/gyro/quat_raw` 屬性, `n`）、`load(path) -> Rec`、`from_dict(d, path="") -> Rec`、`FIELDS`

- [ ] **Step 1: 寫測試**

```python
# task7/tests/test_m6_rec.py
"""M6 錄檔載入器：`_fold` 過的純量要展開、缺關節要擋。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import m6_rec  # noqa: E402
import shm_io  # noqa: E402


def make_dict(n=5, fold_kp=True):
    t = [round(0.005 * i, 4) for i in range(n)]
    joints = {}
    for nm in shm_io.JOINTS:
        joints[nm] = {"q": [0.1 * i for i in range(n)], "v": [0.0] * n,
                      "tau": [1.0] * n, "des": [0.2] * n, "ff": 0.0,
                      "kp": 60.0 if fold_kp else [60.0] * n, "kd": 1.0}
    imu = [[0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(n)]
    return {"schema": "m6_record/1", "label": "x", "note": "", "n": n,
            "secs": t[-1], "hz_actual": 200.0, "t": t, "joints": joints, "imu": imu}


def test_unfolds_scalars_to_arrays():
    r = m6_rec.from_dict(make_dict())
    assert r.n == 5
    assert r.j["fl1_hip_roll"]["kp"].shape == (5,)
    assert np.all(r.j["fl1_hip_roll"]["kp"] == 60.0)
    assert r.j["fl1_hip_roll"]["ff"].shape == (5,)
    assert r.imu.shape == (5, 10)
    assert r.quat_raw.shape == (5, 4) and r.gyro.shape == (5, 3) and r.acc.shape == (5, 3)


def test_rejects_wrong_schema_and_missing_joint():
    d = make_dict()
    d["schema"] = "m6_static/1"
    with pytest.raises(ValueError):
        m6_rec.from_dict(d)
    d = make_dict()
    del d["joints"]["br4_foot"]
    with pytest.raises(ValueError):
        m6_rec.from_dict(d)


def test_load_roundtrip(tmp_path):
    p = tmp_path / "M6_x.json"
    p.write_text(json.dumps(make_dict()), encoding="utf-8")
    r = m6_rec.load(p)
    assert r.path == str(p) and r.hz == pytest.approx(200.0)
    assert r.t[1] == pytest.approx(0.005)
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_m6_rec.py -q`
Expected: FAIL（`No module named 'm6_rec'`）

- [ ] **Step 3: 實作**

```python
# task7/inference/m6_rec.py
"""`M6_load_probe.py --record` 錄檔（schema `m6_record/1`）的載入器。

三支離線工具（real_obs / imu_check / obs_compare）共用。存檔時 `_fold()` 會把
整段不變的欄位存成純量（例如 kp 全程 60.0），這裡一律展開成 `(N,)` 陣列，
讓下游不用到處判斷「這是純量還是陣列」。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import shm_io  # noqa: E402

SCHEMA = "m6_record/1"
FIELDS = ("q", "v", "tau", "des", "ff", "kp", "kd")


@dataclass
class Rec:
    path: str
    label: str
    note: str
    t: np.ndarray            # (N,) 秒，從 0 起
    hz: float                # 實際達到的取樣率
    j: dict                  # name -> {field: (N,) float64}，馬達座標系（原樣）
    imu: np.ndarray          # (N, 10)：acc[3] gyro[3] quat[4]（quat 原始存放順序）

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def acc(self) -> np.ndarray:
        return self.imu[:, 0:3]

    @property
    def gyro(self) -> np.ndarray:
        return self.imu[:, 3:6]

    @property
    def quat_raw(self) -> np.ndarray:
        return self.imu[:, 6:10]


def from_dict(d: dict, path: str = "") -> Rec:
    if d.get("schema") != SCHEMA:
        raise ValueError(f"不是 M6 錄檔（schema={d.get('schema')!r}，需要 {SCHEMA}）")
    t = np.asarray(d["t"], dtype=float)
    n = len(t)
    missing = [nm for nm in shm_io.JOINTS if nm not in d["joints"]]
    if missing:
        raise ValueError(f"錄檔缺關節 {missing}")
    j = {}
    for nm in shm_io.JOINTS:
        j[nm] = {}
        for f in FIELDS:
            v = d["joints"][nm][f]
            arr = np.full(n, float(v)) if not isinstance(v, list) else np.asarray(v, dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{nm}.{f} 長度 {arr.shape} ≠ N={n}")
            j[nm][f] = arr
    imu = np.asarray(d["imu"], dtype=float)
    if imu.shape != (n, 10):
        raise ValueError(f"imu 形狀 {imu.shape} ≠ ({n}, 10)")
    hz = float(d.get("hz_actual") or (n / max(t[-1], 1e-9)))
    return Rec(path=path, label=d.get("label", ""), note=d.get("note", ""),
               t=t, hz=hz, j=j, imu=imu)


def load(path) -> Rec:
    p = Path(path)
    return from_dict(json.loads(p.read_text(encoding="utf-8")), str(p))
```

- [ ] **Step 4: 跑，確認通過**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_m6_rec.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add task7/inference/m6_rec.py task7/tests/test_m6_rec.py
git commit -m "feat(task7): m6_rec —— M6 錄檔載入器（展開 _fold 純量、驗 16 關節）"
```

---

### Task 2: `real_obs.py` —— 錄檔 → 68 維 obs

**Files:**
- Create: `task7/inference/real_obs.py`
- Test: `task7/tests/test_real_obs.py`

**Interfaces:**
- Consumes: `m6_rec.Rec`、`coord.to_ctrl/SIGN/LEG_KINDS/KIND_WHEEL`、`max_model.LEGS/LEG_QPOS_IDX/LEG_QVEL_IDX/WHEEL_*`、`obs_max.build_obs`
- Produces: `LEG_SHM`、`LEG_NAMES`(12, MJCF 序)、`WHEEL_NAMES`(4)、`ctrl_pos(rec)`、`ctrl_vel(rec)`、`ctrl_des(rec)`、`kp12(rec)`、`kd12(rec)`、`quat_wxyz(raw, order)`、`RealFrame`、`frame_at(rec, i, quat_order, gyro_scale)`、`obs_series(rec, quat_order, gyro_scale)`、`SENSOR_DIM=30`、`zero_cpg()`

- [ ] **Step 1: 寫測試**

```python
# task7/tests/test_real_obs.py
"""錄檔 → obs 必須走同一支 build_obs，且腿序按名稱對應。"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402


def test_leg_map_matches_other_copies():
    import play_gait_traj
    assert real_obs.LEG_SHM == play_gait_traj.MM2SHM == coord._CFG2SHM
    assert real_obs.LEG_NAMES == [
        "fr1_hip_roll", "fr2_hip_pitch", "fr3_knee_pitch",
        "fl1_hip_roll", "fl2_hip_pitch", "fl3_knee_pitch",
        "br1_hip_roll", "br2_hip_pitch", "br3_knee_pitch",
        "bl1_hip_roll", "bl2_hip_pitch", "bl3_knee_pitch"]


def _sim_frames(n=6, seed=0):
    """在 MuJoCo 裡踩幾步拿到有姿態、有角速度的狀態，當「真值」。"""
    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = mm.HOME12
    d.qpos[2] = 0.6
    d.qvel[3:6] = rng.normal(0, 0.5, 3)
    frames = []
    for _ in range(n):
        d.ctrl[mm.LEG_ACT_IDX] = rng.normal(0, 5, 12)
        for _ in range(10):
            mujoco.mj_step(m, d)
        frames.append((d.qpos.copy(), d.qvel.copy()))
    return frames


def _rec_from_frames(frames, shuffle=False):
    """把模擬狀態包成 M6 錄檔：關節寫成**馬達座標**、quat 寫成 **xyzw**。"""
    n = len(frames)
    joints = {}
    names = list(shm_io.JOINTS)
    for nm in names:
        leg, kind = nm[:2], nm[2:]
        if kind == coord.KIND_WHEEL:
            k = mm.LEGS.index({v: kk for kk, v in real_obs.LEG_SHM.items()}[leg])
            qi, vi = mm.WHEEL_QPOS_IDX[k], mm.WHEEL_QVEL_IDX[k]
        else:
            idx = real_obs.LEG_NAMES.index(nm)
            qi, vi = mm.LEG_QPOS_IDX[idx], mm.LEG_QVEL_IDX[idx]
        s = coord.SIGN[kind][leg]
        joints[nm] = {"q": [coord.to_motor(nm, float(f[0][qi])) for f in frames],
                      "v": [s * float(f[1][vi]) for f in frames],
                      "tau": [0.0] * n, "des": [0.0] * n, "ff": 0.0, "kp": 0.0, "kd": 0.0}
    imu = []
    for qpos, qvel in frames:
        w, x, y, z = qpos[3:7]
        imu.append([0.0, 0.0, 9.81, *qvel[3:6], x, y, z, w])
    d = {"schema": "m6_record/1", "label": "syn", "note": "", "n": n, "secs": 0.005 * (n - 1),
         "hz_actual": 200.0, "t": [0.005 * i for i in range(n)], "joints": joints, "imu": imu}
    if shuffle:   # 名稱不變、dict 順序打亂 —— 按名稱對應的話結果必須一樣
        d["joints"] = dict(reversed(list(d["joints"].items())))
    return m6_rec.from_dict(d)


def test_obs_matches_build_obs_on_sensor_dims():
    frames = _sim_frames()
    rec = _rec_from_frames(frames)
    O = real_obs.obs_series(rec, quat_order="xyzw", gyro_scale=(1.0, 1.0, 1.0))
    assert O.shape == (len(frames), obs_max.OBS_DIM) and O.dtype == np.float32

    class _D:
        pass
    for i, (qpos, qvel) in enumerate(frames):
        d = _D(); d.qpos, d.qvel = qpos, qvel
        ref = obs_max.build_obs(d, real_obs.zero_cpg(), np.zeros(2), np.zeros(obs_max.ACT_DIM))
        np.testing.assert_allclose(O[i, :real_obs.SENSOR_DIM], ref[:real_obs.SENSOR_DIM],
                                   atol=1e-5)


def test_joint_order_by_name_not_position():
    frames = _sim_frames()
    a = real_obs.obs_series(_rec_from_frames(frames), "xyzw", (1, 1, 1))
    b = real_obs.obs_series(_rec_from_frames(frames, shuffle=True), "xyzw", (1, 1, 1))
    np.testing.assert_array_equal(a, b)


def test_quat_order_and_gyro_scale_are_applied():
    frames = _sim_frames()
    rec = _rec_from_frames(frames)
    a = real_obs.obs_series(rec, "xyzw", (1, 1, 1))
    b = real_obs.obs_series(rec, "wxyz", (1, 1, 1))
    assert not np.allclose(a[:, 0:3], b[:, 0:3])          # 順序錯 → 重力向量變了
    c = real_obs.obs_series(rec, "xyzw", (2.0, -1.0, 1.0))
    np.testing.assert_allclose(c[:, 3], 2.0 * a[:, 3], rtol=1e-6)
    np.testing.assert_allclose(c[:, 4], -a[:, 4], rtol=1e-6)


def test_ctrl_pos_vel_des_and_gains_shapes():
    rec = _rec_from_frames(_sim_frames(n=4))
    assert real_obs.ctrl_pos(rec).shape == (4, 12)
    assert real_obs.ctrl_vel(rec).shape == (4, 12)
    assert real_obs.ctrl_des(rec).shape == (4, 12)
    assert real_obs.kp12(rec).shape == (4, 12) and real_obs.kd12(rec).shape == (4, 12)
    assert real_obs.wheel_kp(rec).shape == (4, 4) and real_obs.wheel_kd(rec).shape == (4, 4)
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_real_obs.py -q`
Expected: FAIL（`No module named 'real_obs'`）

- [ ] **Step 3: 實作**

```python
# task7/inference/real_obs.py
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
```

- [ ] **Step 4: 跑，確認通過**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_real_obs.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add task7/inference/real_obs.py task7/tests/test_real_obs.py
git commit -m "feat(task7): real_obs —— M6 錄檔經同一支 build_obs 轉 68 維 obs"
```

---

### Task 3: `imu_check.py` —— 階 I 判讀

**Files:**
- Create: `task7/inference/imu_check.py`
- Test: `task7/tests/test_imu_check.py`

**Interfaces:**
- Consumes: `m6_rec.load`、`real_obs.quat_wxyz/ctrl_pos/ctrl_vel`、`cpg_max.qinv/w2b`、`leg_kin.fk/ik/knee_sign_of`、`max_model.HOME/HIP_X/HIP_Y/SIDE_X/SIDE_Y/WHEEL_RADIUS`
- Produces: `qmul(a,b)`、`rp_from_gravity(g)->(roll_deg,pitch_deg)`、`gravity_from_quat(Q_wxyz)->(N,3)`、`gravity_from_acc(acc)->(N,3)`、`omega_body(Q_wxyz,t)->(N-1,3)`、`fit_gyro(gyro, omega)->dict(k, corr, corr_matrix)`、`foot_points_body(P12)->(N,4,3)`、`gravity_from_fk(P12)->(g (N,3), resid_mm (N,))`、`update_rate(X, hz, mask)`、`quant_step(x)`、`decide_quat_order(rec)`、`analyse(flat, dance, turn, seen_first, turn_first)->dict`、`main()`

- [ ] **Step 1: 寫測試**

```python
# task7/tests/test_imu_check.py
"""IMU 判讀：用合成資料（已知順序、單位、軸向）驗它判得對。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord  # noqa: E402
import imu_check as ic  # noqa: E402
import leg_kin  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402

D2R = np.pi / 180


def euler_to_quat_wxyz(roll, pitch, yaw=0.0):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def synth(hz=500.0, secs=10.0, roll_amp=8.0, pitch_amp=4.0, store="wxyz",
          gyro_unit="deg/s", flip_y=True, dup_imu_every=0, yaw_rate=0.0, quiet=False):
    """合成一段跳舞：roll/pitch 正弦、四輪永遠共面貼地、gyro 用解析式。"""
    n = int(hz * secs)
    t = np.arange(n) / hz
    if quiet:
        roll = np.full(n, 1.0 * D2R); pitch = np.full(n, -0.5 * D2R)
        droll = dpitch = np.zeros(n)
    else:
        roll = roll_amp * D2R * np.sin(2 * np.pi * 0.5 * t)
        pitch = pitch_amp * D2R * np.sin(2 * np.pi * 0.3 * t)
        droll = roll_amp * D2R * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
        dpitch = pitch_amp * D2R * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t)
    yaw = yaw_rate * t
    Q = np.stack([euler_to_quat_wxyz(roll[i], pitch[i], yaw[i]) for i in range(n)])
    # 機身系角速度（ZYX 尤拉率 → 機身系；yaw 率只進 z 附近）
    om = np.stack([droll - yaw_rate * np.sin(pitch),
                   dpitch * np.cos(roll) + yaw_rate * np.cos(pitch) * np.sin(roll),
                   -dpitch * np.sin(roll) + yaw_rate * np.cos(pitch) * np.cos(roll)], axis=1)
    g = ic.gravity_from_quat(Q)                       # (N,3) 機身系重力方向
    acc = -9.81 * g
    gyro = om * (180 / np.pi if gyro_unit == "deg/s" else 1.0)
    if flip_y:
        gyro[:, 1] *= -1
    # 四輪共面：地面法向 = −g；輪心落在該平面上
    ks = leg_kin.knee_sign_of(mm.HOME)
    origin = np.stack([mm.SIDE_X * mm.HIP_X, mm.SIDE_Y * mm.HIP_Y, np.zeros(4)], axis=1)
    hf = leg_kin.home_foot(mm.HOME) + origin       # (4,3) 機身系
    P = np.zeros((n, 12))
    for i in range(n):
        nrm = -g[i]
        c = nrm @ np.array([0.0, 0.0, -0.45])
        for k in range(4):
            x, y = hf[k, 0], hf[k, 1]
            z = (c - nrm[0] * x - nrm[1] * y) / nrm[2]
            P[i, 3 * k:3 * k + 3] = leg_kin.ik(k, np.array([x, y, z]) - origin[k], ks[k])
    joints = {}
    for nm in shm_io.JOINTS:
        leg, kind = nm[:2], nm[2:]
        if kind == coord.KIND_WHEEL:
            q = np.zeros(n); v = np.zeros(n)
        else:
            idx = real_obs.LEG_NAMES.index(nm)
            q = np.array([coord.to_motor(nm, x) for x in P[:, idx]])
            v = np.gradient(P[:, idx], t) * coord.SIGN[kind][leg]
        joints[nm] = {"q": list(np.round(q, 5)), "v": list(np.round(v, 4)), "tau": [0.0] * n,
                      "des": [0.0] * n, "ff": 0.0, "kp": 60.0, "kd": 1.0}
    quat = Q if store == "wxyz" else Q[:, [1, 2, 3, 0]]
    imu = np.concatenate([acc, gyro, quat], axis=1)
    if dup_imu_every:
        for i in range(1, n):
            if i % dup_imu_every:
                imu[i] = imu[i - 1]
    d = {"schema": "m6_record/1", "label": "syn", "note": "", "n": n, "secs": t[-1],
         "hz_actual": hz, "t": list(np.round(t, 4)), "joints": joints,
         "imu": [list(map(float, r)) for r in imu]}
    return m6_rec.from_dict(d)


def test_rp_from_gravity_conventions():
    # +roll = 右側低：機身繞 +x 轉 +10° 後，重力在機身系 y 分量為負
    g = ic.gravity_from_quat(euler_to_quat_wxyz(10 * D2R, 0)[None])[0]
    r, p = ic.rp_from_gravity(g)
    assert r == pytest.approx(10.0, abs=1e-6) and p == pytest.approx(0.0, abs=1e-6)
    assert g[1] < 0
    g = ic.gravity_from_quat(euler_to_quat_wxyz(0, 7 * D2R)[None])[0]
    r, p = ic.rp_from_gravity(g)
    assert p == pytest.approx(7.0, abs=1e-6) and g[0] > 0


def test_omega_body_recovers_analytic_rate():
    rec = synth(gyro_unit="rad/s", flip_y=False)
    Q = real_obs.quat_wxyz(rec.quat_raw, "wxyz")
    om = ic.omega_body(Q, rec.t)
    np.testing.assert_allclose(om, rec.gyro[:-1], atol=2e-3)


def test_decides_order_unit_axis_and_rates():
    flat = synth(store="wxyz", quiet=True, secs=3)
    dance = synth(store="wxyz", gyro_unit="deg/s", flip_y=True)
    out = ic.analyse(flat, dance, None, seen_first=None, turn_first=None)
    assert out["quat_order"] == "wxyz"
    assert out["gyro_unit"] == "deg/s"
    assert np.sign(out["gyro_scale"]).tolist() == [1.0, -1.0, 1.0]
    np.testing.assert_allclose(np.abs(out["gyro_scale"]), np.pi / 180, rtol=0.05)
    assert abs(out["rp_quat_vs_fk"]["bias_deg"][0]) < 0.5
    assert out["rp_quat_vs_fk"]["corr"][0] > 0.99
    assert out["rates"]["imu_gyro"]["hz_min"] >= 475
    assert out["first_roll"]["side"] == "右低"          # roll 先往正 → 右側低


def test_detects_xyzw_and_rad_per_s():
    flat = synth(store="xyzw", quiet=True, secs=3, gyro_unit="rad/s", flip_y=False)
    dance = synth(store="xyzw", gyro_unit="rad/s", flip_y=False)
    out = ic.analyse(flat, dance, None, None, None)
    assert out["quat_order"] == "xyzw"
    assert out["gyro_unit"] == "rad/s"
    np.testing.assert_allclose(out["gyro_scale"], [1, 1, 1], rtol=0.05)


def test_update_rate_sees_duplicated_imu():
    dance = synth(store="xyzw", gyro_unit="rad/s", flip_y=False, dup_imu_every=2)
    flat = synth(store="xyzw", quiet=True, secs=3, gyro_unit="rad/s", flip_y=False)
    out = ic.analyse(flat, dance, None, None, None)
    assert 235 <= out["rates"]["imu_gyro"]["hz_min"] <= 265


def test_turn_sign():
    turn = synth(store="xyzw", gyro_unit="rad/s", flip_y=False, yaw_rate=0.4, quiet=True, secs=4)
    flat = synth(store="xyzw", quiet=True, secs=2, gyro_unit="rad/s", flip_y=False)
    dance = synth(store="xyzw", gyro_unit="rad/s", flip_y=False)
    out = ic.analyse(flat, dance, turn, None, turn_first="left")
    assert out["turn"]["gyro_z_sign"] == 1 and out["turn"]["consistent"] is True
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_imu_check.py -q`
Expected: FAIL（`No module named 'imu_check'`）

- [ ] **Step 3: 實作**

```python
# task7/inference/imu_check.py
"""階 I 判讀：IMU 的四元數順序、gyro 單位/軸向/偏置、安裝偏置、更新率。

三個**互相獨立**的參考，不靠眼睛：
  1. 加速度計：靜止時 specific force = −重力 → 重力方向 = −acc/|acc|
  2. 腿的正向運動學：狗在平地上四輪必共面 → 地面法向 → 重力方向（M4 那招）
  3. 四元數微分：q_{t+1} = q_t ⊗ Δq → 機身系角速度，與 gyro 逐軸擬合
眼睛只當決勝票（`--seen-first left|right`、`--turn-first left|right`）。

慣例（與 MJCF 一致）：x 前、y 左、z 上；+roll = 右側低；+pitch = 頭低；+yaw = 左轉。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import leg_kin          # noqa: E402
import m6_rec           # noqa: E402
import max_model as mm  # noqa: E202,E402
import real_obs         # noqa: E402
from cpg_max import qinv, w2b  # noqa: E402

DOWN = np.array([0.0, 0.0, -1.0])
R2D = 180.0 / np.pi
ORDERS = ("xyzw", "wxyz")


# ---------------------------------------------------------------- 四元數
def qmul(a, b):
    """wxyz Hamilton 積，支援 (N,4)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def gravity_from_quat(Q):
    """(N,4) wxyz → (N,3) 機身系重力方向（與 obs 的 gravity 欄位同一式子）。"""
    Q = np.asarray(Q, float)
    return np.stack([w2b(q, DOWN) for q in Q])


def gravity_from_acc(acc):
    acc = np.asarray(acc, float)
    return -acc / np.maximum(np.linalg.norm(acc, axis=-1, keepdims=True), 1e-9)


def rp_from_gravity(g):
    """機身系重力方向 → (roll°, pitch°)。純 roll φ：g=(0,−sinφ,−cosφ)；純 pitch θ：g=(sinθ,0,−cosθ)。"""
    g = np.asarray(g, float)
    roll = np.arctan2(-g[..., 1], -g[..., 2]) * R2D
    pitch = np.arctan2(g[..., 0], np.hypot(g[..., 1], g[..., 2])) * R2D
    return roll, pitch


def omega_body(Q, t):
    """(N,4) wxyz + t → (N−1,3) 機身系角速度 rad/s。Δq = q_t⁻¹ ⊗ q_{t+1}。"""
    Q = np.asarray(Q, float)
    Qi = np.stack([qinv(q) for q in Q[:-1]])
    dq = qmul(Qi, Q[1:])
    dq = np.where(dq[:, :1] < 0, -dq, dq)
    s = np.linalg.norm(dq[:, 1:], axis=1)
    ang = 2 * np.arctan2(s, dq[:, 0])
    axis = dq[:, 1:] / np.maximum(s, 1e-12)[:, None]
    dt = np.diff(np.asarray(t, float))
    return axis * (ang / np.maximum(dt, 1e-9))[:, None]


# ---------------------------------------------------------------- gyro 擬合
def fit_gyro(gyro, omega):
    """逐軸最小平方 gyro_i = k_i · ω_i（過原點）＋ 3×3 相關矩陣（抓軸對調）。"""
    G, W = np.asarray(gyro, float), np.asarray(omega, float)
    k = np.array([(G[:, i] @ W[:, i]) / max(W[:, i] @ W[:, i], 1e-12) for i in range(3)])
    cm = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a, b = G[:, i] - G[:, i].mean(), W[:, j] - W[:, j].mean()
            cm[i, j] = (a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return {"k": k, "corr": np.diag(cm).copy(), "corr_matrix": cm}


def classify_scale(k):
    """k = gyro/ω。回傳 (scale 讓 scale·gyro = rad/s, unit 字串)。"""
    ak = np.abs(k)
    if np.all((0.6 < ak) & (ak < 1.6)):
        return np.sign(k) * 1.0, "rad/s"
    if np.all((40.0 < ak) & (ak < 75.0)):
        return np.sign(k) * (np.pi / 180.0), "deg/s"
    return 1.0 / np.where(ak < 1e-9, 1.0, k), "unknown"


# ---------------------------------------------------------------- 腿 FK 參考
_ORIGIN = np.stack([mm.SIDE_X * mm.HIP_X, mm.SIDE_Y * mm.HIP_Y, np.zeros(4)], axis=1)


def foot_points_body(P12):
    """(N,12) 控制器角（MJCF 序）→ (N,4,3) 四個輪心在機身系。"""
    P = np.asarray(P12, float).reshape(-1, 4, 3)
    out = np.zeros((P.shape[0], 4, 3))
    for i in range(P.shape[0]):
        for k in range(4):
            out[i, k] = _ORIGIN[k] + leg_kin.fk(k, P[i, k])
    return out


def gravity_from_fk(P12):
    """四輪共面 → 平面法向 → 機身系重力方向。回傳 (g (N,3), 共面殘差 mm (N,))。"""
    F = foot_points_body(P12)
    N = F.shape[0]
    g = np.zeros((N, 3)); resid = np.zeros(N)
    for i in range(N):
        A = np.column_stack([F[i, :, 0], F[i, :, 1], np.ones(4)])
        coef, *_ = np.linalg.lstsq(A, F[i, :, 2], rcond=None)
        nrm = np.array([-coef[0], -coef[1], 1.0]); nrm /= np.linalg.norm(nrm)
        g[i] = -nrm
        resid[i] = np.abs(A @ coef - F[i, :, 2]).max() * 1000
    return g, resid


# ---------------------------------------------------------------- 更新率／量化
def update_rate(X, hz, mask=None):
    """相鄰筆完全相同的比例 → 更新率下界。回傳 {same_frac, hz_min, n}。"""
    X = np.asarray(X, float).reshape(len(X), -1)
    same = np.all(X[1:] == X[:-1], axis=1)
    if mask is not None:
        same = same[np.asarray(mask, bool)[1:]]
    frac = float(same.mean()) if same.size else float("nan")
    return {"same_frac": frac, "hz_min": float(hz * (1.0 - frac)), "n": int(same.size)}


def quant_step(x):
    """最小非零相鄰差 = 量化步階。"""
    d = np.abs(np.diff(np.asarray(x, float)))
    d = d[d > 0]
    return float(d.min()) if d.size else 0.0


# ---------------------------------------------------------------- 判定
def _static_mask(rec):
    return np.abs(np.linalg.norm(rec.acc, axis=1) - 9.81) < 0.3


def decide_quat_order(rec):
    """靜態樣本上，哪種順序解出的重力方向與加速度計最接近。"""
    m = _static_mask(rec)
    ga = gravity_from_acc(rec.acc[m])
    best, errs = None, {}
    for o in ORDERS:
        gq = gravity_from_quat(real_obs.quat_wxyz(rec.quat_raw[m], o))
        errs[o] = float(np.degrees(np.arccos(np.clip((gq * ga).sum(1), -1, 1))).mean())
        if best is None or errs[o] < errs[best]:
            best = o
    return best, errs


def analyse(flat, dance, turn, seen_first, turn_first):
    out = {"schema": "imu_check/1",
           "sources": {"flat": flat.path, "dance": dance.path, "turn": turn.path if turn else None}}

    # 1. quat 順序：flat + dance 的靜態樣本一起判
    order, errs = decide_quat_order(dance)
    order_f, errs_f = decide_quat_order(flat)
    out["quat_order"] = order
    out["quat_order_err_deg"] = {"dance": errs, "flat": errs_f}
    out["quat_order_consistent"] = (order == order_f)

    Qf = real_obs.quat_wxyz(flat.quat_raw, order)
    Qd = real_obs.quat_wxyz(dance.quat_raw, order)

    # 2. 平放：兩種順序的 roll/pitch、acc 模長/單位、gyro 偏置與雜訊、關節量化
    rp_both = {}
    for o in ORDERS:
        r, p = rp_from_gravity(gravity_from_quat(real_obs.quat_wxyz(flat.quat_raw, o)))
        rp_both[o] = [float(r.mean()), float(p.mean())]
    an = float(np.linalg.norm(flat.acc, axis=1).mean())
    r_acc, p_acc = rp_from_gravity(gravity_from_acc(flat.acc))
    out["flat"] = {
        "rp_quat_deg": rp_both, "rp_acc_deg": [float(r_acc.mean()), float(p_acc.mean())],
        "acc_norm": an, "acc_unit": "m/s2" if 8.5 < an < 11 else ("g" if 0.9 < an < 1.1 else "unknown"),
        "gyro_bias_raw": flat.gyro.mean(0).tolist(), "gyro_noise_raw": flat.gyro.std(0).tolist(),
        "q_quant": float(np.median([quant_step(flat.j[n]["q"]) for n in real_obs.LEG_NAMES])),
        "v_quant": float(np.median([quant_step(flat.j[n]["v"]) for n in real_obs.LEG_NAMES])),
        "q_noise": float(np.mean([flat.j[n]["q"].std() for n in real_obs.LEG_NAMES])),
        "v_noise": float(np.mean([flat.j[n]["v"].std() for n in real_obs.LEG_NAMES])),
    }

    # 3. 跳舞：gyro 擬合
    om = omega_body(Qd, dance.t)
    bias = np.asarray(out["flat"]["gyro_bias_raw"])
    fit = fit_gyro(dance.gyro[:-1] - bias, om)
    scale, unit = classify_scale(fit["k"])
    out["gyro_k"] = fit["k"].tolist()
    out["gyro_corr"] = fit["corr"].tolist()
    out["gyro_corr_matrix"] = fit["corr_matrix"].tolist()
    out["gyro_axis_swap"] = bool(np.any(np.argmax(np.abs(fit["corr_matrix"]), axis=1) != np.arange(3)))
    out["gyro_scale"] = np.asarray(scale, float).tolist()
    out["gyro_unit"] = unit
    out["gyro_bias_rad_s"] = (bias * np.asarray(scale)).tolist()
    out["gyro_noise_rad_s"] = (np.asarray(out["flat"]["gyro_noise_raw"]) * np.abs(scale)).tolist()

    # 4. 跳舞：IMU 姿態 vs 腿 FK 姿態（四輪共面才算）
    Pd = real_obs.ctrl_pos(dance)
    g_fk, resid = gravity_from_fk(Pd)
    ok = resid < 10.0
    r_q, p_q = rp_from_gravity(gravity_from_quat(Qd))
    r_f, p_f = rp_from_gravity(g_fk)
    def _corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    out["rp_quat_vs_fk"] = {
        "n_coplanar": int(ok.sum()), "n": int(ok.size),
        "bias_deg": [float((r_q - r_f)[ok].mean()), float((p_q - p_f)[ok].mean())],
        "corr": [_corr(r_q[ok], r_f[ok]), _corr(p_q[ok], p_f[ok])] if ok.sum() > 10 else [float("nan")] * 2,
    }
    out["imu_mount_rp_deg"] = out["rp_quat_vs_fk"]["bias_deg"]

    # 5. 更新率（跳舞段，關節只看在動的樣本）
    vmask = np.abs(real_obs.ctrl_vel(dance)).max(1) > 0.05
    out["rates"] = {
        "imu_gyro": update_rate(dance.gyro, dance.hz),
        "imu_quat": update_rate(dance.quat_raw, dance.hz),
        "imu_acc": update_rate(dance.acc, dance.hz),
        "joint_q": update_rate(Pd, dance.hz, vmask),
        "joint_v": update_rate(real_obs.ctrl_vel(dance), dance.hz, vmask),
        "record_hz": dance.hz,
    }

    # 6. 第一個 roll 事件（給眼睛對）
    idx = np.flatnonzero(np.abs(r_q) > 2.0)
    if idx.size:
        i0 = idx[0]
        seg = r_q[i0:min(i0 + int(dance.hz), len(r_q))]
        j = i0 + int(np.argmax(np.abs(seg)))
        side = "右低" if r_q[j] > 0 else "左低"
        out["first_roll"] = {"t": float(dance.t[j]), "roll_deg": float(r_q[j]), "side": side}
        if seen_first:
            expect = "左低" if seen_first == "left" else "右低"
            out["first_roll"]["seen"] = expect
            out["first_roll"]["consistent"] = (expect == side)
    else:
        out["first_roll"] = None

    # 7. 轉向：第一段 |ω_z| > 0.15 的平均符號
    if turn is not None:
        wz = (turn.gyro - bias) @ np.diag(np.asarray(scale))
        wz = wz[:, 2]
        act = np.flatnonzero(np.abs(wz) > 0.15)
        if act.size:
            i0 = act[0]
            seg = wz[i0:min(i0 + int(turn.hz * 0.5), len(wz))]
            sgn = int(np.sign(seg.mean()))
            out["turn"] = {"t": float(turn.t[i0]), "gyro_z_sign": sgn,
                           "dir": "左轉(逆時針)" if sgn > 0 else "右轉(順時針)"}
            if turn_first:
                out["turn"]["consistent"] = ((sgn > 0) == (turn_first == "left"))
        else:
            out["turn"] = None
    return out


def _print(out):
    f = out["flat"]
    print("═══ IMU 判讀 ═══")
    print(f"quat 順序          → {out['quat_order']}  （靜態重力方向誤差° dance {out['quat_order_err_deg']['dance']}"
          f" / flat {out['quat_order_err_deg']['flat']}；兩段一致 {out['quat_order_consistent']}）")
    print(f"平放 roll/pitch°    xyzw {f['rp_quat_deg']['xyzw']}  wxyz {f['rp_quat_deg']['wxyz']}  acc {f['rp_acc_deg']}")
    print(f"acc 模長 {f['acc_norm']:.3f} → {f['acc_unit']}")
    print(f"gyro k={np.round(out['gyro_k'], 3).tolist()} corr={np.round(out['gyro_corr'], 3).tolist()}"
          f" → 單位 {out['gyro_unit']}  scale {np.round(out['gyro_scale'], 5).tolist()}  軸對調 {out['gyro_axis_swap']}")
    print(f"gyro 偏置 rad/s {np.round(out['gyro_bias_rad_s'], 4).tolist()}  雜訊 std {np.round(out['gyro_noise_rad_s'], 4).tolist()}")
    q = out["rp_quat_vs_fk"]
    print(f"IMU vs 腿FK  偏置° {np.round(q['bias_deg'], 2).tolist()}  corr {np.round(q['corr'], 3).tolist()}"
          f"  共面樣本 {q['n_coplanar']}/{q['n']}")
    print("更新率下界 Hz  " + "  ".join(f"{k} {v['hz_min']:.0f}" for k, v in out["rates"].items() if isinstance(v, dict)))
    print(f"關節量化 q {f['q_quant']:.5f} rad  v {f['v_quant']:.4f} rad/s；靜態雜訊 q {f['q_noise']:.5f} v {f['v_noise']:.4f}")
    print(f"第一個 roll 事件    {out['first_roll']}")
    print(f"轉向                {out.get('turn')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="階 I：IMU 判讀")
    ap.add_argument("--flat", type=Path, required=True)
    ap.add_argument("--dance", type=Path, required=True)
    ap.add_argument("--turn", type=Path)
    ap.add_argument("--seen-first", choices=("left", "right"), help="眼睛看到跳舞先往哪側低")
    ap.add_argument("--turn-first", choices=("left", "right"), help="原地轉先轉哪邊")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "imu_check.json")
    a = ap.parse_args()
    out = analyse(m6_rec.load(a.flat), m6_rec.load(a.dance),
                  m6_rec.load(a.turn) if a.turn else None, a.seen_first, a.turn_first)
    _print(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑，確認通過**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_imu_check.py -q`
Expected: 6 passed

- [ ] **Step 5: 用既有 trip14 資料試跑（無 flat 時拿 xlate_left 兼當 flat 看是否會跑）**

Run: `conda run -n rbtdog python task7/inference/imu_check.py --flat task7/logs/m_logs_trip14/M6_20260902_155126.json --dance task7/logs/m_logs_trip14/M6_20260902_155223.json --out /tmp/claude-1000/-home-huang-rbtdog-sim/557339f6-03ee-4b5d-89d2-243698d236cf/scratchpad/imu_check_trip14.json`
Expected: 印出判定；`quat_order` 應為 `xyzw`、`gyro_unit` 應判得出來（rad/s 或 deg/s）。把結果數字記下來給文件用。

- [ ] **Step 6: Commit**

```bash
git add task7/inference/imu_check.py task7/tests/test_imu_check.py
git commit -m "feat(task7): imu_check —— quat 順序/gyro 單位軸向/安裝偏置/更新率，三個獨立參考"
```

---

### Task 4: `M_env_probe.py` —— 狗上電腦探測

**Files:**
- Create: `task7/realbot/M_env_probe.py`
- Test: `task7/tests/test_m_env_probe.py`

**Interfaces:**
- Produces: `make_weights(sizes, seed) -> list[(W rows as list[list], b list)]`、`forward_py(layers, x) -> list`、`forward_np(layers, x)`、`bench(fn, n) -> (mean_ms, max_ms)`、`main()`

- [ ] **Step 1: 寫測試**

```python
# task7/tests/test_m_env_probe.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import M_env_probe as ep  # noqa: E402


def test_pure_python_forward_matches_numpy():
    layers = ep.make_weights((68, 256, 256, 128, 12), seed=1)
    x = [0.01 * i for i in range(68)]
    y_py = ep.forward_py(layers, x)
    y_np = ep.forward_np(layers, x)
    np.testing.assert_allclose(y_py, y_np, rtol=1e-9, atol=1e-9)
    assert len(y_py) == 12


def test_bench_returns_positive_ms():
    layers = ep.make_weights((8, 8, 4), seed=0)
    mean_ms, max_ms = ep.bench(lambda: ep.forward_py(layers, [0.0] * 8), n=5)
    assert 0 < mean_ms <= max_ms
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_m_env_probe.py -q`
Expected: FAIL（`No module named 'M_env_probe'`）

- [ ] **Step 3: 實作**

```python
#!/usr/bin/env python3
"""M_env_probe —— 狗上電腦（RK3588）探測：numpy 有沒有、policy 前向幾 ms、CPU 能用幾核。

零風險：不開任何 /dev/shm、不碰馬達、不需要 root。狗趴著即可。

為什麼要有這支：CPG-RL 訓練完的網路要在狗上每 20 ms 出一次指令。
兩份自家文件對「狗上有沒有 numpy」說法相反，從沒查過；純 Python 前向在本機要 6 ms，
RK3588 若慢 4 倍就是 24 ms —— 那要在**訓練前**知道，不是訓練完才發現要重寫推論。

用法（狗上）：python3 M_env_probe.py
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time

SIZES = (68, 256, 256, 128, 12)


def make_weights(sizes=SIZES, seed=0):
    rng = random.Random(seed)
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        s = 1.0 / math.sqrt(a)
        W = [[rng.uniform(-s, s) for _ in range(a)] for _ in range(b)]
        bias = [rng.uniform(-s, s) for _ in range(b)]
        layers.append((W, bias))
    return layers


def forward_py(layers, x):
    h = list(x)
    n = len(layers)
    for li, (W, b) in enumerate(layers):
        out = []
        for row, bb in zip(W, b):
            s = bb
            for w, v in zip(row, h):
                s += w * v
            out.append(s if li == n - 1 else (s if s > 0 else 0.0))   # ReLU，最後一層線性
        h = out
    return h


def forward_np(layers, x):
    import numpy as np
    h = np.asarray(x, float)
    n = len(layers)
    for li, (W, b) in enumerate(layers):
        h = np.asarray(W) @ h + np.asarray(b)
        if li != n - 1:
            h = np.maximum(h, 0.0)
    return h


def bench(fn, n=100):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts), max(ts)


def main() -> int:
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        info["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity"] = None
    for mod in ("numpy", "onnxruntime", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    layers = make_weights()
    x = [0.0] * SIZES[0]
    forward_py(layers, x)
    info["mlp_py_ms"] = bench(lambda: forward_py(layers, x), 50)
    if info["numpy"]:
        import numpy as np
        Wn = [(np.asarray(W), np.asarray(b)) for W, b in layers]

        def f():
            h = np.asarray(x)
            for li, (W, b) in enumerate(Wn):
                h = W @ h + b
                if li != len(Wn) - 1:
                    h = np.maximum(h, 0.0)
            return h
        f()
        info["mlp_np_ms"] = bench(f, 200)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"\n純 Python 前向 {info['mlp_py_ms'][0]:.2f} ms（最大 {info['mlp_py_ms'][1]:.2f}）"
          f"；50 Hz 預算 20 ms")
    if info.get("mlp_np_ms"):
        print(f"numpy 前向 {info['mlp_np_ms'][0]:.3f} ms（最大 {info['mlp_np_ms'][1]:.3f}）")
    else:
        print("numpy：無 → 推論只能純 Python 或自帶 wheel")
    d = os.path.expanduser("~/m_logs")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, time.strftime("ENV_%Y%m%d_%H%M%S.json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"→ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑，確認通過；本機也跑一次 main 看輸出**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_m_env_probe.py -q && python3 task7/realbot/M_env_probe.py`
Expected: 2 passed；本機印出 py/np 前向毫秒（本機純 Python 約 5–10 ms）

- [ ] **Step 5: Commit**

```bash
git add task7/realbot/M_env_probe.py task7/tests/test_m_env_probe.py
git commit -m "feat(task7): M_env_probe —— 狗上 numpy/MLP 前向時間/CPU 親和性探測（零風險）"
```

---

### Task 5: `obs_compare.py` —— 階 II 錄檔驅動回放與逐欄對照

**Files:**
- Create: `task7/inference/obs_compare.py`
- Test: `task7/tests/test_obs_compare.py`

**Interfaces:**
- Consumes: `m6_rec`、`real_obs.*`、`obs_max.build_obs/OBS_LAYOUT`、`leg_kin.fk`、`max_model.*`、`mujoco`
- Produces: `gait_segment(rec) -> slice`、`initial_height(P0) -> float`、`sim_replay(rec, quat_order, gyro_scale) -> dict(obs (N,68), q (N,12), v (N,12), tau (N,12), quat (N,4), gyro (N,3), height (N,))`、`lag_ms(a, b, dt, max_ms=100) -> float`、`noise_std(x, k=5) -> float`、`channel_table(O_real, O_sim, seg, dt) -> list[dict]`、`joint_vel_report(rec, seg) -> dict`、`des_to_q_latency(des, q, dt, seg) -> list[float]`、`build_model(...)->dict`、`main()`

- [ ] **Step 1: 寫測試**

```python
# task7/tests/test_obs_compare.py
"""錄檔驅動回放 + 逐欄對照：拿模擬自己產生的「假實機」驗雜訊與延遲量得回來。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import coord  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import obs_compare as oc  # noqa: E402
import real_obs  # noqa: E402
import shm_io  # noqa: E402

HZ = 200.0


def _rec_with_des(des12, kp_abad=60.0, kp=250.0, kd=2.0):
    """只填 des/kp/kd 的錄檔（q/v/imu 先用零，回放不需要它們）。"""
    n = des12.shape[0]
    t = np.arange(n) / HZ
    joints = {}
    for nm in shm_io.JOINTS:
        kind = nm[2:]
        if kind == coord.KIND_WHEEL:
            joints[nm] = {"q": [0.0] * n, "v": [0.0] * n, "tau": [0.0] * n, "des": [0.0] * n,
                          "ff": 0.0, "kp": 0.0, "kd": 0.5}
        else:
            idx = real_obs.LEG_NAMES.index(nm)
            joints[nm] = {"q": [coord.to_motor(nm, float(x)) for x in des12[:, idx]],
                          "v": [0.0] * n, "tau": [0.0] * n,
                          "des": [coord.to_motor(nm, float(x)) for x in des12[:, idx]],
                          "ff": 0.0, "kp": kp_abad if kind == coord.KIND_HIP_ROLL else kp, "kd": kd}
    imu = [[0.0, 0.0, 9.81, 0, 0, 0, 0, 0, 0, 1.0] for _ in range(n)]
    return m6_rec.from_dict({"schema": "m6_record/1", "label": "syn", "note": "", "n": n,
                             "secs": t[-1], "hz_actual": HZ, "t": list(t),
                             "joints": joints, "imu": imu})


def _des_sequence(secs_hold=0.8, secs_move=2.0):
    """crouch 起步 → 站到 HOME → 髖做正弦擺動（增益是步態組，gait_segment 抓得到）。"""
    crouch = np.tile(mm.CROUCH.reshape(-1)[:3], 4) if mm.CROUCH.shape == (4, 3) else mm.CROUCH.reshape(12)
    home = mm.HOME12
    n1, n2 = int(secs_hold * HZ), int(secs_move * HZ)
    ramp = np.linspace(0, 1, n1)[:, None]
    A = crouch[None] * (1 - ramp) + home[None] * ramp
    t = np.arange(n2) / HZ
    B = np.tile(home, (n2, 1))
    for k in range(4):
        B[:, 3 * k + 1] += 0.15 * np.sin(2 * np.pi * 1.0 * t)
        B[:, 3 * k + 2] -= 0.20 * np.sin(2 * np.pi * 1.0 * t)
    return np.vstack([A, B])


def test_gait_segment_is_where_abad_kp_is_soft():
    des = _des_sequence()
    rec = _rec_with_des(des)
    for nm in real_obs.LEG_NAMES:
        if nm.endswith(coord.KIND_HIP_ROLL):
            rec.j[nm]["kp"][:100] = 250.0
    s = oc.gait_segment(rec)
    assert s.start == 100 and s.stop == rec.n


def test_replay_then_compare_recovers_noise_and_lag():
    des = _des_sequence()
    rec = _rec_with_des(des)
    sim = oc.sim_replay(rec, "xyzw", (1.0, 1.0, 1.0))
    assert sim["obs"].shape == (rec.n, 68)
    assert sim["height"][-1] > 0.35            # 站起來了、沒倒

    # 用回放結果造「假實機」：加已知雜訊、往後平移 3 筆（15 ms）
    rng = np.random.default_rng(0)
    lag = 3
    q_n, v_n, g_n, w_n = 0.002, 0.05, 0.005, 0.01
    def shift(x):
        return np.concatenate([np.repeat(x[:1], lag, axis=0), x[:-lag]], axis=0)
    Pm = shift(sim["q"]); Vm = shift(sim["v"]); Qm = shift(sim["quat"]); Gm = shift(sim["gyro"])
    for nm in real_obs.LEG_NAMES:
        idx = real_obs.LEG_NAMES.index(nm)
        s = coord.SIGN[nm[2:]][nm[:2]]
        rec.j[nm]["q"] = np.array([coord.to_motor(nm, x) for x in Pm[:, idx]]) + rng.normal(0, q_n, rec.n)
        rec.j[nm]["v"] = s * Vm[:, idx] + rng.normal(0, v_n, rec.n)
    rec.imu[:, 3:6] = Gm + rng.normal(0, w_n, (rec.n, 3))
    rec.imu[:, 6:10] = Qm[:, [1, 2, 3, 0]]          # 存成 xyzw
    O_real = real_obs.obs_series(rec, "xyzw", (1, 1, 1))
    seg = oc.gait_segment(rec)
    rows = oc.channel_table(O_real, sim["obs"], seg, 1.0 / HZ)
    assert len(rows) == real_obs.SENSOR_DIM
    jp = [r for r in rows if r["group"] == "joint_pos"]
    jv = [r for r in rows if r["group"] == "joint_vel"]
    gy = [r for r in rows if r["group"] == "gyro"]
    assert np.median([r["noise_est"] for r in jp]) == pytest.approx(q_n, rel=0.35)
    assert np.median([r["noise_est"] for r in jv]) == pytest.approx(v_n, rel=0.35)
    assert np.median([r["noise_est"] for r in gy]) == pytest.approx(w_n, rel=0.35)
    assert np.median([r["lag_ms"] for r in jp]) == pytest.approx(15.0, abs=5.0)
    assert all(r["corr"] > 0.9 for r in jp)

    lat = oc.des_to_q_latency(real_obs.ctrl_des(rec), real_obs.ctrl_pos(rec), 1.0 / HZ, seg)
    assert len(lat) == 12 and np.median(lat) >= 15.0        # 平移 15 ms + 伺服本身的落後

    jvr = oc.joint_vel_report(rec, seg)
    assert set(jvr) >= {"noise_ratio_v_over_dq", "corr_v_dq", "psd_frac_above_25hz"}


def test_lag_ms_sign_convention():
    dt = 0.005
    t = np.arange(400) * dt
    a = np.sin(2 * np.pi * 1.0 * t)
    b = np.concatenate([np.zeros(4), a[:-4]])      # b 比 a 晚 20 ms
    assert oc.lag_ms(b, a, dt) == pytest.approx(20.0, abs=2.5)
    assert oc.lag_ms(a, b, dt) == pytest.approx(-20.0, abs=2.5)
```

- [ ] **Step 2: 跑，確認失敗**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_obs_compare.py -q`
Expected: FAIL（`No module named 'obs_compare'`）

- [ ] **Step 3: 實作**

```python
#!/usr/bin/env python3
"""階 II：把實機錄到的 `des/kp/kd` 逐筆驅動 MuJoCo，與實機 obs 同時刻逐欄對照。

★ 為什麼是「錄檔驅動回放」而不是「模擬自己跑一遍步態」：
  M9 互動模式的步態是狗上即時算的（GaitStream），長度由人決定；重跑 CPG 要對齊相位。
  直接回放錄到的指令（`replay_standup.py` 的做法）→ 兩邊指令逐位元相同、時間軸相同，
  差異全部來自「感測器 + 物理」，正是我們要量的。站→走→趴整段都可比。

輸出：
  - `outputs/obs_noise_model.json`：訓練端直接讀（逐欄雜訊 std、延遲、範圍、quat 順序、gyro scale）
  - markdown 表（stdout / --md）：每欄 sim/real 的 mean、std、min、max、雜訊、相關、延遲

慣例：lag_ms > 0 表示**實機比模擬晚**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco           # noqa: E402
import numpy as np      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import leg_kin          # noqa: E402
import m6_rec           # noqa: E402
import max_model as mm  # noqa: E402
import obs_max          # noqa: E402
import real_obs         # noqa: E402

TAU_MAX_LEG, TAU_MAX_WHEEL = 150.0, 40.0
_ORIGIN = np.stack([mm.SIDE_X * mm.HIP_X, mm.SIDE_Y * mm.HIP_Y, np.zeros(4)], axis=1)


# ---------------------------------------------------------------- 段落
def gait_segment(rec: m6_rec.Rec) -> slice:
    """步態段 = ABAD kp 是步態值（<100、>0）的最長連續區。站立/起身用 250，步態用 60。"""
    kp_abad = real_obs.kp12(rec)[:, 0::3].max(1)
    m = (kp_abad > 0) & (kp_abad < 100)
    best, cur = (0, 0), None
    for i, ok in enumerate(m):
        if ok and cur is None:
            cur = i
        if (not ok or i == len(m) - 1) and cur is not None:
            end = i + 1 if ok else i
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    if best == (0, 0):
        return slice(0, rec.n)
    return slice(*best)


def initial_height(P0) -> float:
    """起始姿勢的機身高：最低輪心 + 輪半徑 + 1 cm 落穩餘裕。"""
    P = np.asarray(P0, float).reshape(4, 3)
    z = min(float((_ORIGIN[k] + leg_kin.fk(k, P[k]))[2]) for k in range(4))
    return -z + mm.WHEEL_RADIUS + 0.01


# ---------------------------------------------------------------- 回放
def sim_replay(rec: m6_rec.Rec, quat_order: str, gyro_scale, on_frame=None) -> dict:
    """錄到的 des/kp/kd 逐筆驅動 MuJoCo。回傳每筆的 obs、q、v、tau、quat(wxyz)、gyro、height。"""
    des, KP, KD = real_obs.ctrl_des(rec), real_obs.kp12(rec), real_obs.kd12(rec)
    WKP, WKD = real_obs.wheel_kp(rec), real_obs.wheel_kd(rec)
    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)
    lo = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 0]
    hi = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 1]
    q0 = np.clip(real_obs.ctrl_pos(rec)[0], lo + 1e-4, hi - 1e-4)
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = q0
    d.qpos[2] = initial_height(q0)
    for _ in range(int(0.5 / m.opt.timestep)):          # 洩力落穩
        d.ctrl[:] = 0.0
        e = q0 - d.qpos[mm.LEG_QPOS_IDX]
        d.ctrl[mm.LEG_ACT_IDX] = np.clip(KP[0] * e - KD[0] * d.qvel[mm.LEG_QVEL_IDX],
                                         -TAU_MAX_LEG, TAU_MAX_LEG)
        mujoco.mj_step(m, d)

    n = rec.n
    dts = np.diff(rec.t)
    dts = np.append(dts, dts[-1] if dts.size else 0.005)
    c0, cmd0, a0 = real_obs.zero_cpg(), np.zeros(2), np.zeros(obs_max.ACT_DIM)
    out = {k: np.zeros((n, s)) for k, s in
           (("obs", obs_max.OBS_DIM), ("q", 12), ("v", 12), ("tau", 12), ("quat", 4), ("gyro", 3))}
    out["height"] = np.zeros(n)
    latch = d.qpos[mm.WHEEL_QPOS_IDX].copy()
    prev_wkp = np.zeros(4)
    for i in range(n):
        nsub = max(1, int(round(dts[i] / m.opt.timestep)))
        newly = (WKP[i] > 0) & (prev_wkp <= 0)
        latch[newly] = d.qpos[mm.WHEEL_QPOS_IDX][newly]      # 鎖輪：鎖在模擬自己當下的輪角
        prev_wkp = WKP[i]
        for _ in range(nsub):
            e = des[i] - d.qpos[mm.LEG_QPOS_IDX]
            tau = np.clip(KP[i] * e - KD[i] * d.qvel[mm.LEG_QVEL_IDX], -TAU_MAX_LEG, TAU_MAX_LEG)
            d.ctrl[mm.LEG_ACT_IDX] = tau
            ew = latch - d.qpos[mm.WHEEL_QPOS_IDX]
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(WKP[i] * ew - WKD[i] * d.qvel[mm.WHEEL_QVEL_IDX],
                                               -TAU_MAX_WHEEL, TAU_MAX_WHEEL)
            mujoco.mj_step(m, d)
        out["obs"][i] = obs_max.build_obs(d, c0, cmd0, a0)
        out["q"][i] = d.qpos[mm.LEG_QPOS_IDX]
        out["v"][i] = d.qvel[mm.LEG_QVEL_IDX]
        out["tau"][i] = d.ctrl[mm.LEG_ACT_IDX]
        out["quat"][i] = d.qpos[3:7]
        out["gyro"][i] = d.qvel[3:6]
        out["height"][i] = d.qpos[2]
        if on_frame is not None:
            on_frame(i, m, d)
    return out


# ---------------------------------------------------------------- 統計
def lag_ms(a, b, dt, max_ms=100.0) -> float:
    """a 相對 b 的延遲（ms）。正 = a 比 b 晚。用去均值正規化互相關的峰值。"""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return float("nan")
    L = int(round(max_ms / 1000 / dt))
    best, best_l = -np.inf, 0
    for l in range(-L, L + 1):
        if l >= 0:
            x, y = a[l:], b[:len(b) - l]
        else:
            x, y = a[:len(a) + l], b[-l:]
        c = (x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)
        if c > best:
            best, best_l = c, l
    return best_l * dt * 1000.0


def noise_std(x, k=5) -> float:
    """對 k 點移動中位數的殘差 std —— 抓高頻雜訊，不抓訊號本身。"""
    x = np.asarray(x, float)
    if len(x) < k + 1:
        return 0.0
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    med = np.array([np.median(xp[i:i + k]) for i in range(len(x))])
    return float((x - med).std())


def _corr(a, b) -> float:
    a, b = np.asarray(a, float) - np.mean(a), np.asarray(b, float) - np.mean(b)
    return float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def channel_names():
    names = []
    for grp, dim in obs_max.OBS_LAYOUT:
        if grp in ("joint_pos", "joint_vel"):
            names += [(grp, real_obs.LEG_NAMES[i]) for i in range(dim)]
        else:
            names += [(grp, f"{grp}_{'xyz'[i]}") for i in range(dim)]
    return names[:real_obs.SENSOR_DIM]


def channel_table(O_real, O_sim, seg: slice, dt: float) -> list[dict]:
    R, S = np.asarray(O_real, float)[seg], np.asarray(O_sim, float)[seg]
    rows = []
    for i, (grp, nm) in enumerate(channel_names()):
        r, s = R[:, i], S[:, i]
        nr, ns = noise_std(r), noise_std(s)
        rows.append({"idx": i, "group": grp, "name": nm,
                     "mean_real": float(r.mean()), "std_real": float(r.std()),
                     "min_real": float(r.min()), "max_real": float(r.max()),
                     "mean_sim": float(s.mean()), "std_sim": float(s.std()),
                     "min_sim": float(s.min()), "max_sim": float(s.max()),
                     "noise_real": nr, "noise_sim": ns,
                     "noise_est": float(np.sqrt(max(nr * nr - ns * ns, 0.0))),
                     "corr": _corr(r, s), "lag_ms": lag_ms(r, s, dt)})
    return rows


def des_to_q_latency(des, q, dt, seg: slice) -> list[float]:
    D, Q = np.asarray(des, float)[seg], np.asarray(q, float)[seg]
    return [lag_ms(Q[:, j], D[:, j], dt) for j in range(12)]


def joint_vel_report(rec: m6_rec.Rec, seg: slice) -> dict:
    """實機 velocity 欄位 vs 角度差分：雜訊倍率、相關、25 Hz 以上能量占比。"""
    P, V = real_obs.ctrl_pos(rec)[seg], real_obs.ctrl_vel(rec)[seg]
    t = rec.t[seg]
    dt = float(np.median(np.diff(t)))
    dq = np.gradient(P, t, axis=0)
    ratio, corr, frac = [], [], []
    for j in range(12):
        ratio.append(noise_std(V[:, j]) / max(noise_std(dq[:, j]), 1e-9))
        corr.append(_corr(V[:, j], dq[:, j]))
        x = V[:, j] - V[:, j].mean()
        ps = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), dt)
        frac.append(float(ps[f > 25.0].sum() / max(ps.sum(), 1e-12)))
    return {"noise_ratio_v_over_dq": ratio, "corr_v_dq": corr, "psd_frac_above_25hz": frac,
            "record_hz": 1.0 / dt}


def build_model(rec, rows, lat_des_q, jvr, quat_order, gyro_scale, imu_check_path) -> dict:
    def grp(g, key):
        return [r[key] for r in rows if r["group"] == g]
    imu_lag = float(np.nanmedian(grp("gravity", "lag_ms") + grp("gyro", "lag_ms")))
    jnt_lag = float(np.nanmedian(grp("joint_pos", "lag_ms")))
    return {"schema": "obs_noise/1", "source": rec.path, "label": rec.label,
            "imu_check": str(imu_check_path) if imu_check_path else None,
            "quat_order": quat_order, "gyro_scale": list(map(float, gyro_scale)),
            "segment": {"start_s": float(rec.t[gait_segment(rec).start]),
                        "stop_s": float(rec.t[gait_segment(rec).stop - 1]), "hz": rec.hz},
            "latency_ms": {"des_to_q": lat_des_q, "des_to_q_median": float(np.nanmedian(lat_des_q)),
                           "real_vs_sim_joint_pos": jnt_lag, "real_vs_sim_imu": imu_lag,
                           "imu_vs_joint": imu_lag - jnt_lag},
            "noise_std": {g: grp(g, "noise_est") for g in ("gravity", "gyro", "joint_pos", "joint_vel")},
            "range_real": {g: [grp(g, "min_real"), grp(g, "max_real")] for g in ("gravity", "gyro", "joint_pos", "joint_vel")},
            "range_sim": {g: [grp(g, "min_sim"), grp(g, "max_sim")] for g in ("gravity", "gyro", "joint_pos", "joint_vel")},
            "std_real": {g: grp(g, "std_real") for g in ("gravity", "gyro", "joint_pos", "joint_vel")},
            "std_sim": {g: grp(g, "std_sim") for g in ("gravity", "gyro", "joint_pos", "joint_vel")},
            "corr": {g: grp(g, "corr") for g in ("gravity", "gyro", "joint_pos", "joint_vel")},
            "joint_vel": jvr}


def markdown_table(rows) -> str:
    L = ["| # | 欄位 | mean r/s | std r/s | min r/s | max r/s | 雜訊 r/s/est | corr | lag ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " ⚠️" if r["corr"] < 0 else ""
        L.append(f"| {r['idx']} | {r['name']}{flag} | {r['mean_real']:+.3f}/{r['mean_sim']:+.3f} "
                 f"| {r['std_real']:.3f}/{r['std_sim']:.3f} | {r['min_real']:+.2f}/{r['min_sim']:+.2f} "
                 f"| {r['max_real']:+.2f}/{r['max_sim']:+.2f} | {r['noise_real']:.4f}/{r['noise_sim']:.4f}/{r['noise_est']:.4f} "
                 f"| {r['corr']:+.2f} | {r['lag_ms']:+.0f} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="階 II：錄檔驅動回放 + obs 逐欄對照")
    ap.add_argument("rec", type=Path)
    ap.add_argument("--imu-check", type=Path, help="imu_check.json：取 quat_order 與 gyro_scale")
    ap.add_argument("--quat-order", default=None, choices=("xyzw", "wxyz"))
    ap.add_argument("--gyro-scale", type=float, nargs=3, default=None)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "obs_noise_model.json")
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None, help="回放錄影 mp4")
    a = ap.parse_args()

    quat_order, gyro_scale = "xyzw", (1.0, 1.0, 1.0)
    if a.imu_check:
        ic = json.loads(a.imu_check.read_text(encoding="utf-8"))
        quat_order, gyro_scale = ic["quat_order"], tuple(ic["gyro_scale"])
    if a.quat_order:
        quat_order = a.quat_order
    if a.gyro_scale:
        gyro_scale = tuple(a.gyro_scale)

    rec = m6_rec.load(a.rec)
    print(f"錄檔 {a.rec.name}  {rec.n} 筆 @ {rec.hz:.0f} Hz  {rec.t[-1]:.1f} s；quat {quat_order}  gyro_scale {gyro_scale}")
    seg = gait_segment(rec)
    print(f"步態段 {rec.t[seg.start]:.2f} → {rec.t[seg.stop - 1]:.2f} s（{seg.stop - seg.start} 筆）")

    frames = []
    ren = cam = None
    if a.video:
        pass  # 由 on_frame 內部建

    def on_frame(i, m, d):
        nonlocal ren, cam
        if a.video is None or i % 8:
            return
        if ren is None:
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1280)
            m.vis.global_.offheight = max(m.vis.global_.offheight, 720)
            ren = mujoco.Renderer(m, 720, 1280)
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam.distance, cam.elevation, cam.azimuth = 2.6, -12, 135
        cam.lookat[:] = d.qpos[:3]
        ren.update_scene(d, camera=cam)
        frames.append(ren.render())

    sim = sim_replay(rec, quat_order, gyro_scale, on_frame)
    print(f"回放完成：末機身高 {sim['height'][-1] * 1000:.0f} mm，最低 {sim['height'].min() * 1000:.0f} mm")
    O_real = real_obs.obs_series(rec, quat_order, gyro_scale)
    dt = float(np.median(np.diff(rec.t)))
    rows = channel_table(O_real, sim["obs"], seg, dt)
    lat = des_to_q_latency(real_obs.ctrl_des(rec), real_obs.ctrl_pos(rec), dt, seg)
    jvr = joint_vel_report(rec, seg)
    md = markdown_table(rows)
    print(md)
    print(f"\ndes→q 延遲 ms（12 關節）{np.round(lat, 1).tolist()}  中位 {np.nanmedian(lat):.1f}")
    print(f"joint_vel 雜訊倍率 v/dq 中位 {np.median(jvr['noise_ratio_v_over_dq']):.2f}；"
          f"corr 中位 {np.median(jvr['corr_v_dq']):.3f}；>25 Hz 能量占比中位 {np.median(jvr['psd_frac_above_25hz']):.3f}")
    model = build_model(rec, rows, lat, jvr, quat_order, gyro_scale, a.imu_check)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {a.out}")
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
        print(f"→ {a.md}")
    if a.video and frames:
        import imageio.v2 as iio
        iio.mimsave(str(a.video), frames, fps=25, codec="libx264")
        print(f"🎬 {a.video}  {len(frames)} 幀")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑，確認通過**

Run: `conda run -n rbtdog python -m pytest task7/tests/test_obs_compare.py -q`
Expected: 3 passed（回放測試約 10–20 秒）

- [ ] **Step 5: 用既有 trip14 錄檔（原廠平移）試跑整條管線**

Run: `conda run -n rbtdog python task7/inference/obs_compare.py task7/logs/m_logs_trip14/M6_20260902_155126.json --out /tmp/claude-1000/-home-huang-rbtdog-sim/557339f6-03ee-4b5d-89d2-243698d236cf/scratchpad/obs_noise_trip14.json`
Expected: 跑完、印表；原廠資料增益是 60/120/120，`gait_segment` 會抓到整段。這一步驗的是管線不會炸，數字不當結論（原廠的增益與我們不同）。

- [ ] **Step 6: Commit**

```bash
git add task7/inference/obs_compare.py task7/tests/test_obs_compare.py
git commit -m "feat(task7): obs_compare —— 錄檔驅動 MuJoCo 回放 + obs 逐欄雜訊/延遲對照"
```

---

### Task 6: 現場操作卡 + 全套測試 + HANDOFF

**Files:**
- Create: `task7/docs/現場操作卡_obs盤點_2026-09-08.md`
- Modify: `task7/HANDOFF.md`（最上面加一節）

- [ ] **Step 1: 寫操作卡**

```markdown
# 現場操作卡：實機 obs 盤點（2026-09-08）

**目的**：CPG-RL 重訓前把 68 維 obs 的感測欄位在實機上量清楚。**全部唯讀**，只多一支背景錄製程序。
**前置**：`bash task7/realbot/push_to_dog.sh`（要推 `M_env_probe.py`）；筆電連狗 WiFi；`ssh robot@192.168.234.1`。
錄檔自動存在狗的 `~/m_logs/M6_*.json`，結束後 `bash task7/realbot/pull_from_dog.sh` 拉回 `task7/logs/m_logs_trip18/`。

| 階 | 狗上指令（`cd ~/rbtdog/task7/realbot`） | 狗的狀態 | 你記 |
|---|---|---|---|
| I-a | `python3 M6_load_probe.py flat --record --secs 30 --hz 200 --note "平放"` | 趴平不動；地面用手機水平儀看 | — |
| I-b | `python3 M6_load_probe.py dance --record --secs 90 --hz 500 --note "跳舞"` | 起錄 5 秒後按遙控器跳舞 | **先往左低還是右低** |
| I-c | `python3 M6_load_probe.py turn --record --secs 30 --hz 200 --note "左轉停右轉"` | 起錄後左轉 5 s、停 3 s、右轉 5 s（沒原地轉就跳過） | 先左 |
| I-d | `python3 M_env_probe.py` | 趴著 | — |
| II | ssh#2：`python3 M6_load_probe.py walk --record --secs 120 --hz 200 --note "A_kp250_walk"`；ssh#1 照 `現場操作卡_互動式前進_trip17_2026-09-03.md` 跑 | 站→走 10 s→停→趴 | 同 9/3 |

⚠️ II 的 M6 要**先於** M9 啟動、且 M9 結束後才 Ctrl-C（要錄到趴著→站→走→趴的全程）。
⚠️ I-b 若 500 Hz 印出的 `hz_actual` 明顯低於 450，改 `--hz 400` 重錄一次。

**回來之後**（本機、rbtdog 環境）：
```
python task7/inference/imu_check.py --flat logs/m_logs_trip18/M6_<flat>.json --dance logs/m_logs_trip18/M6_<dance>.json [--turn ...] --seen-first left|right [--turn-first left]
python task7/inference/obs_compare.py logs/m_logs_trip18/M6_<walk>.json --imu-check outputs/imu_check.json --md outputs/obs_compare_table.md --video outputs/obs_replay.mp4
```
產出 `outputs/imu_check.json`、`outputs/obs_noise_model.json`、表格；結果整理到 `docs/H_實機obs盤點_2026-09-08.md`。
```

- [ ] **Step 2: 全套測試**

Run: `conda run -n rbtdog python -m pytest task7/tests -q`
Expected: 719 + 19 = 738 passed（數字以實際為準，重點是既有 719 全綠）

- [ ] **Step 3: HANDOFF 最上面加一節**

在 `task7/HANDOFF.md` 的第一個 `---` 之後插入：

```markdown
## ▶ 2026-09-08 上午：obs 盤點工具鏈完成，等上機

決策：**IMU 進 obs（重力 + 角速度），先驗證再重訓**。今天一輪量齊，不回頭補。
工具：`inference/{m6_rec,real_obs,imu_check,obs_compare}.py`、`realbot/M_env_probe.py`，
操作卡 `docs/現場操作卡_obs盤點_2026-09-08.md`，spec `docs/superpowers/specs/2026-09-08-d1max-real-obs-audit-design.md`。
★ 意外收穫：trip14（9/2 原廠平移）500 Hz 錄檔已含 gyro，IMU 更新率 ≥500 Hz 已從那裡看出來。
產出（上機後）：`outputs/imu_check.json`、`outputs/obs_noise_model.json`、`docs/H_實機obs盤點_2026-09-08.md`。

---
```

- [ ] **Step 4: Commit**

```bash
git add task7/docs/現場操作卡_obs盤點_2026-09-08.md task7/HANDOFF.md
git commit -m "docs(task7): obs 盤點操作卡 + HANDOFF"
```

---

## Self-review

- **Spec coverage**：①→Task 2、②→Task 3、③→Task 5、④→Task 4、載入器→Task 1、操作卡/文件→Task 6；spec 的每階「回答什麼」都對到 `analyse()`/`channel_table()`/`joint_vel_report()`/`des_to_q_latency()`。`docs/H_*.md` 是上機後才能寫的結果文件，不在本計畫。
- **Placeholder**：無 TBD；每個 Step 都有完整程式碼。
- **型別一致**：`real_obs.LEG_NAMES/ctrl_pos/ctrl_vel/ctrl_des/kp12/kd12/wheel_kp/wheel_kd/quat_wxyz/zero_cpg/obs_series/SENSOR_DIM` 在 Task 2 定義、Task 3/5 使用；`m6_rec.Rec.{t,hz,j,imu,acc,gyro,quat_raw,n,path,label}` 在 Task 1 定義、後面使用；`imu_check.analyse` 回傳鍵與測試斷言一致。
