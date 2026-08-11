# walk_stable 吊掛空跑部署 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `walk_stable` 步態的關節軌跡，在吊掛、腳不落地、不驅動右後腿的條件下於 D1 EDU 輪足實機重現，並量出逐軸追蹤誤差。

**Architecture:** 三層。開發機的 `gait_export.py` 負責產生軌跡、離線檢驗、事後分析；狗上的 `L7_gait_shm.py` 負責把軌跡以 500 Hz 寫進 `/dev/shm/spline_shm`；`shm_common.py` 是兩邊共用的 SHM 結構與安全骨架，從既有的 `L4_standup_shm.py` 抽出。離線檔（npz）是黃金標準，狗上即時計算的路徑必須逐幀吻合才准使用。

**Tech Stack:** Python 3、numpy、MuJoCo（僅開發機）、ctypes + mmap（SHM）、pytest。

**設計文件：** `docs/superpowers/specs/2026-08-11-d1-walk-stable-hanging-deploy-design.md`

## Global Constraints

- **狗上（`task6/realbot/`）只能依賴 python3 標準庫 + numpy。** 不得 `import mujoco` —— 車載電腦沒有。`shm_common.py` 與 `L7_gait_shm.py` 都受此限制。`gait_export.py` 在開發機跑，可以用 mujoco。
- **不得修改 `task6/inference/d1_model.py` 的任何常數。** 那是訓練與推論的共用契約，改了會讓既有權重作廢。
- **不得修改 `task6/inference/cpg_walk_d1.py` 的 `GAIT_G_C`、`GAITS` 或任何步態參數。** 那支腳本產生了影片，是對照基準。部署用的 `G_C` 放在 `gait_export.py` 自己的常數裡。
- **不得複製 `cpg_walk_d1.py` 的軌跡邏輯 —— 但有一個明確的例外。** `gait_export.py` 必須 `import` 它的 `make_cpg_step` / `joint_targets` / `GAITS` / `MU_Y` / `SETTLE_S`。兩份會漂移，而漂移是靜默的。

  **例外（使用者決策，2026-08-11）**：`L7_gait_shm.py` 的 `--source live` 模式必須在狗上重新實作一份 CPG 積分 + IK。這不是疏忽，是需求 —— 使用者要「一個模式直接播寫死的軌跡、一個模式在狗上自己算」，而狗上沒有 mujoco（`leg_ik_consts` 要跑 forward kinematics），無法 import 開發機那份。

  漂移防線是 `test_live_trajectory_matches_the_file_frame_by_frame`：兩份的輸出逐幀比對，容忍 1e-9。那個測試轉紅就代表兩份漂移了。**審查時請把這份重複當作已決策事項，但仍應檢查防線是否真的有效**（例如測試是否只在某個特例下成立）。
- **SHM 結構定義只能有一份**，在 `shm_common.py`。`SplineData` 大小必須是 608 bytes。
- **腿序有兩套，不可混用**：policy/MJCF 腿序是 `(FL, FR, RL, RR)`；SHM 腿序是 `legs[0]=FR, [1]=FL, [2]=RR, [3]=RL`。轉換只能透過 `calib_map.LEG_MJCF2SHM`。
- **部署步態參數**（`walk_stable`，除 `G_C` 外全部沿用）：`duty=0.80, ω=1.4, μx=1.90, μy=1.5, x_off=-0.055`。
- **部署 `G_C = 0.110`**（Task 3 驗證後定案），不是影片版的 0.12。
- **限位餘裕門檻 0.05 rad。**
- **實機增益預設 `kp=20 / kd=0.7`**（原廠站立實測值）。
- 測試指令：`conda run --no-capture-output -n rbtdog python -m pytest task6/tests -q`。基準：164 passed。
- 狗上執行一律 dry-run 預設，`--confirm` 才碰硬體，且需 root。

---

### Task 1: 抽出 `shm_common.py`

把 SHM 結構與安全骨架從 `L4_standup_shm.py` 抽出，讓 L4 與之後的 L7 共用。L4 的行為必須完全不變。

**Files:**
- Create: `task6/realbot/shm_common.py`
- Modify: `task6/realbot/L4_standup_shm.py`（刪掉被抽走的定義，改成 import）
- Test: `task6/tests/test_shm_common.py`

**Interfaces:**
- Produces（給 Task 6、7 用）：
  - `SplineData` / `SplineCmd` / `SplineState` / `LegControl` / `LegState` / `JointControl` / `JointState`（ctypes.Structure，`_pack_=1`）
  - `SHM_PATH: str`、`SHM_SIZE: int`、`EXPECT_SIZE: int = 608`、`CONSUMER_CONTROL: int = 0`
  - `CTRL_HZ: int = 500`、`DT: float`
  - `LEGNAME: dict[int, str]`、`FAULT_BITS: tuple`
  - `POSE_STAND: dict[int, dict[str, float]]`、`POSE_LIE`、`POSES`、`POSE_LABEL`
  - `open_shm() -> tuple[SplineData, mmap.mmap]`
  - `zero_all(d) -> None`
  - `set_leg_position(d, i, abad, hip, knee, kp, kd) -> None`
  - `publish(d) -> None`
  - `read_leg_q(d, i) -> tuple[float, float, float]`
  - `preflight_mc_stopped(d) -> tuple[bool, int]`
  - `preflight_motors_healthy(d, active_legs) -> tuple[bool, list[str]]`
  - `report_legs(d, active_legs) -> None`
  - `check_guards(d, active_legs, torque_abort, vel_abort) -> tuple[bool, str]`
  - `passive_stop(d, active_legs, cycles=1500, stop_kd=3.0) -> None`
  - `check_struct_size() -> None`（大小不符就 `sys.exit(1)`）

> ⚠️ `check_guards` 與 `passive_stop` 的簽章與 L4 現況**不同**：門檻與 `stop_kd` 改成參數傳入，因為 L7 要用不同的值（見設計文件 §5）。L4 呼叫端要補上 `TORQUE_ABORT` / `VEL_ABORT` / `STOP_KD` 引數。

- [ ] **Step 1: 寫失敗的測試**

建立 `task6/tests/test_shm_common.py`：

```python
"""shm_common 的結構契約與純函式測試。不碰硬體。"""
import ctypes
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import shm_common as SC


def test_spline_data_size_is_608():
    """結構大小對不上 = 我們對 SHM 的理解跟 daemon 不一致，寫下去會寫到錯的欄位。"""
    assert ctypes.sizeof(SC.SplineData) == 608
    assert SC.EXPECT_SIZE == 608


def test_cmd_and_state_sub_sizes():
    assert ctypes.sizeof(SC.SplineCmd) == 344
    assert ctypes.sizeof(SC.SplineState) == 264


def test_joint_control_field_order():
    """欄位順序寫錯會靜默把 kp 寫進 v_des。"""
    assert [n for n, _ in SC.JointControl._fields_] == \
        ["p_des", "v_des", "kp", "kd", "t_ff"]
    assert [n for n, _ in SC.JointState._fields_] == ["flags", "p", "v", "t"]


def test_legname_matches_shm_leg_order():
    """SHM 腿序：0=FR 1=FL 2=RR 3=RL。實機已確認 leg0=FR、leg2=RR。"""
    assert SC.LEGNAME == {0: "FR", 1: "FL", 2: "RR", 3: "RL"}


def test_zero_all_clears_every_joint_and_sets_flags():
    d = SC.SplineData()
    d.cmd.legs[1].hip.kp = 99.0
    d.cmd.legs[3].foot.t_ff = 5.0
    SC.zero_all(d)
    for i in range(4):
        assert d.cmd.legs[i].flags == 1
        for jn in ("abad", "hip", "knee", "foot"):
            j = getattr(d.cmd.legs[i], jn)
            assert (j.p_des, j.v_des, j.kp, j.kd, j.t_ff) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_set_leg_position_leaves_wheel_at_zero_gain():
    """輪子(foot)全程零增益 —— 本專案從不對輪做位置控制。"""
    d = SC.SplineData()
    SC.zero_all(d)
    SC.set_leg_position(d, 2, 0.1, 0.2, 0.3, kp=20.0, kd=0.7)
    leg = d.cmd.legs[2]
    assert leg.abad.p_des == pytest.approx(0.1)
    assert leg.hip.p_des == pytest.approx(0.2)
    assert leg.knee.p_des == pytest.approx(0.3)
    assert leg.knee.kp == pytest.approx(20.0)
    assert leg.foot.kp == 0.0 and leg.foot.kd == 0.0


def test_check_guards_uses_passed_thresholds_and_ignores_skipped_legs():
    """被跳過的腿若故障，其 t/v 是損壞資料，不得拿來當保護判準。"""
    d = SC.SplineData()
    d.state.legs[2].foot.t = 99.0      # RR 故障中的垃圾值
    ok, why = SC.check_guards(d, (0, 1, 3), torque_abort=8.0, vel_abort=2.0)
    assert ok, why
    ok, why = SC.check_guards(d, (0, 1, 2, 3), torque_abort=8.0, vel_abort=2.0)
    assert not ok and "RR.foot" in why


def test_check_guards_threshold_is_a_parameter_not_a_constant():
    d = SC.SplineData()
    d.state.legs[0].knee.v = 5.0
    assert SC.check_guards(d, (0,), torque_abort=8.0, vel_abort=2.0)[0] is False
    assert SC.check_guards(d, (0,), torque_abort=8.0, vel_abort=20.0)[0] is True


def test_preflight_motors_healthy_flags_not_ready_and_dead_can():
    d = SC.SplineData()
    for i in range(4):
        for jn in ("abad", "hip", "knee", "foot"):
            # ready=1, 溫度 30C, 電壓 44V
            getattr(d.state.legs[i], jn).flags = 1 | (30 << 8) | (44 << 16)
    assert SC.preflight_motors_healthy(d, (0, 1, 2, 3))[0]

    d.state.legs[2].foot.flags = (29 << 8) | (44 << 16)      # ready=0
    ok, problems = SC.preflight_motors_healthy(d, (0, 1, 2, 3))
    assert not ok and any("RR.foot" in p and "ready=0" in p for p in problems)

    ok, _ = SC.preflight_motors_healthy(d, (0, 1, 3))         # 跳過 RR
    assert ok


def test_pose_stand_and_lie_are_per_leg_mirrored():
    """左右腿編碼器慣例鏡像，所以每腿一組，不是四腿共用。"""
    for pose in (SC.POSE_STAND, SC.POSE_LIE):
        assert set(pose) == {0, 1, 2, 3}
        # leg0(FR) 與 leg1(FL) 的 abad 號相反
        assert pose[0]["abad"] * pose[1]["abad"] < 0
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_shm_common.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'shm_common'`

- [ ] **Step 3: 建立 `shm_common.py`**

把 `L4_standup_shm.py` 的下列區塊**原樣搬過去**（不要改邏輯，只改兩個簽章）：

- 第 1 節全部 ctypes 結構定義、`CONSUMER_CONTROL`/`CONSUMER_OTHER`/`SHM_PATH`/`SHM_SIZE`/`EXPECT_SIZE`
- `CTRL_HZ`/`DT`/`STOP_KD`
- `POSE_STAND`/`POSE_LIE`/`POSES`/`POSE_LABEL`
- `LEGNAME`/`FAULT_BITS`
- 第 3 節全部函式

檔頭寫清楚它的角色：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shm_common.py —— D1 EDU 輪足實機 SHM 讀寫的共用底層。

由 L4_standup_shm.py 於 2026-08-11 抽出，供 L4（姿勢序列）與 L7（步態串流）共用。

★ 為什麼要抽出：SHM 結構定義絕對不能有兩份。欄位順序漂掉是靜默的 ——
  不會報錯，只會把 kp 寫進 v_des、把指令寫到錯的關節。

★ 依賴限制：本檔在【車載電腦】上執行，只能用 python3 標準庫。
  不得 import mujoco / torch / scipy。
"""
```

兩個簽章要改（其餘原樣）：

```python
def check_guards(d, active_legs, torque_abort, vel_abort):
    """回傳 (ok, 說明)。被驅動的腿任一關節力矩/速度超限 → not ok。

    門檻改成參數傳入（原本是模組常數）：L4 的站姿 ramp 與 L7 的步態串流
    需求差一個數量級（步態膝關節要 ~13.5 rad/s，L4 的 ramp 只要 0.63），
    共用常數會讓其中一邊不是誤中止就是形同虛設。

    只檢查 active_legs：被跳過的腿若是故障中，其 t/v 欄位可能是損壞資料
    （實測 2026-08-11 的 RR.foot 會在 0/28/44/95 之間亂跳），拿來當保護判準
    會造成無意義的誤中止。被跳過的腿全程零增益，本來就不會出力。
    """
    for i in active_legs:
        s = d.state.legs[i]
        for jn in ("abad", "hip", "knee", "foot"):
            js = getattr(s, jn)
            if abs(js.t) > torque_abort:
                return False, f"{LEGNAME[i]}.{jn} 力矩 {js.t:.2f} > {torque_abort}"
            if abs(js.v) > vel_abort:
                return False, f"{LEGNAME[i]}.{jn} 速度 {js.v:.2f} > {vel_abort}"
    return True, ""


def passive_stop(d, active_legs, cycles=1500, stop_kd=3.0):
    """卸力收尾：被驅動的腿 kp=0、kd=stop_kd，軟軟停住；被跳過的腿維持全零。"""
    for _ in range(cycles):
        zero_all(d)
        for i in active_legs:
            for jn in ("abad", "hip", "knee"):
                getattr(d.cmd.legs[i], jn).kd = stop_kd
        publish(d)
        time.sleep(DT)
    zero_all(d)
    publish(d)
```

再加一個 L4 的 main() 本來就在做、但值得共用的檢查：

```python
def check_struct_size():
    """結構大小對不上 = 這支程式對 SHM 的理解跟 daemon 不一致，寫下去會寫到錯的欄位。

    用 sys.exit 而非 assert：assert 在 python -O 下會被拿掉，這道關卡不能被關掉。
    """
    size = ctypes.sizeof(SplineData)
    print(f"[*] SplineData 結構大小 = {size} bytes（應為 {EXPECT_SIZE}）")
    if size != EXPECT_SIZE:
        print(f"✗ 結構大小不符（{size} != {EXPECT_SIZE}）→ 拒絕執行，避免寫壞共享記憶體。")
        sys.exit(1)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_shm_common.py -q`
Expected: PASS（10 passed）

- [ ] **Step 5: 改 `L4_standup_shm.py` 改用 shm_common**

刪掉第 1 節結構定義、第 3 節函式、`POSE_*`、`LEGNAME`、`FAULT_BITS`、`CTRL_HZ`/`DT`/`STOP_KD` 的定義，改成：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shm_common import (CTRL_HZ, DT, EXPECT_SIZE, FAULT_BITS, LEGNAME, POSE_LABEL,
                        POSE_LIE, POSE_STAND, POSES, SHM_PATH, SplineData,
                        check_guards, check_struct_size, open_shm,
                        passive_stop, preflight_mc_stopped,
                        preflight_motors_healthy, publish, read_leg_q,
                        report_legs, set_leg_position, zero_all)

STOP_KD = 3.0
```

`L4` 保留自己的 `TORQUE_ABORT = 8.0` / `VEL_ABORT = 2.0`（那是站姿 ramp 的門檻，不是 L7 的），並把三個呼叫端補上引數：

```python
        return check_guards(d, active_legs, TORQUE_ABORT, VEL_ABORT)
```

```python
    def abort(why):
        print(f"⚠️ 保護觸發：{why} → 卸力中止")
        passive_stop(d, active_legs, 300, STOP_KD)
```

`main()` 裡把手寫的大小檢查換成 `check_struct_size()`，並刪掉 `import ctypes`（若已無其他用途）。`passive_stop(d, active_legs, 800)` 兩處補成 `passive_stop(d, active_legs, 800, STOP_KD)`。

- [ ] **Step 6: 確認 L4 dry-run 行為不變**

先存下改動前的輸出當基準（改檔前若沒存，用 `git stash` 取回舊版跑一次）：

Run:
```bash
cd /home/huang/rbtdog_sim/task6/realbot
git stash && python3 L4_standup_shm.py --sequence lie,stand > /tmp/l4_before.txt 2>&1; git stash pop
python3 L4_standup_shm.py --sequence lie,stand > /tmp/l4_after.txt 2>&1
diff /tmp/l4_before.txt /tmp/l4_after.txt && echo "IDENTICAL"
```
Expected: `IDENTICAL`

- [ ] **Step 7: 全套測試**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests -q`
Expected: `174 passed`（原 164 + 新增 10）

- [ ] **Step 8: Commit**

```bash
git add task6/realbot/shm_common.py task6/realbot/L4_standup_shm.py task6/tests/test_shm_common.py
git commit -m "refactor(task6): 抽出 shm_common，L4 與 L7 共用 SHM 骨架

SHM 結構定義不能有兩份——欄位順序漂掉是靜默的，會把 kp 寫進 v_des。
check_guards / passive_stop 的門檻改成參數：L4 的站姿 ramp 要 0.63 rad/s，
L7 的步態要 ~13.5，共用常數會讓其中一邊誤中止或形同虛設。
L4 dry-run 輸出與改動前逐字元相同。"
```

> ⚠️ **實機重驗**：本任務改動了 L4。在 L7 上機之前，必須用實機重跑一次
> `sudo python3 L4_standup_shm.py --sequence lie,stand --confirm --skip-legs 2`
> 確認趴下／站立行為不變。這是設計文件 §8 的硬性要求。

---

### Task 2: SHM 限位轉換

`gait_export.py` 的第一塊：把 MJCF 的 `ctrlrange` 轉成 SHM 慣例的限位。這是整個管線最容易寫錯的一步。

**Files:**
- Create: `task6/inference/gait_export.py`
- Test: `task6/tests/test_gait_export.py`

**Interfaces:**
- Consumes: `calib_map.CALIB`、`calib_map.LEG_MJCF2SHM`、`d1_model.make_model`
- Produces:
  - `shm_limits(m) -> dict[tuple[int, str], tuple[float, float]]` —— key 是 `(shm_leg, joint_name)`，value 是 `(lo, hi)`，已按 SHM 慣例排序
  - `JN: tuple[str, str, str] = ("abad", "hip", "knee")`

- [ ] **Step 1: 寫失敗的測試**

建立 `task6/tests/test_gait_export.py`：

```python
"""gait_export 的離線管線測試。不碰硬體。"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "inference"))
sys.path.insert(0, str(_ROOT / "realbot"))

import calib_map
import d1_model
import gait_export as GE


@pytest.fixture(scope="module")
def model():
    return d1_model.make_model()


def test_shm_limits_has_all_twelve_axes(model):
    lim = GE.shm_limits(model)
    assert set(lim) == {(leg, jn) for leg in range(4) for jn in GE.JN}


def test_shm_limits_swaps_bounds_when_sign_is_negative(model):
    """sign=-1 的關節，MJCF 的上界會變成 SHM 的下界。寫錯這裡限位檢驗就整個失效。"""
    lim = GE.shm_limits(model)
    for shm_leg in range(4):
        for jn in GE.JN:
            lo, hi = lim[(shm_leg, jn)]
            assert lo < hi, f"leg{shm_leg}.{jn} 上下界顛倒：{lo} !< {hi}"

    # leg0 = FR，其 knee 的 sign 是 -1（見 calib_map.CALIB）
    assert calib_map.CALIB[0]["knee"][0] == -1
    s, o = calib_map.CALIB[0]["knee"]
    mjcf_lo, mjcf_hi = -2.7030, -0.6220          # FR_knee 的 ctrlrange
    lo, hi = lim[(0, "knee")]
    # sign=-1：MJCF 下界映到 SHM 上界
    assert hi == pytest.approx(s * mjcf_lo + o, abs=1e-6)
    assert lo == pytest.approx(s * mjcf_hi + o, abs=1e-6)


def test_calib_map_round_trips():
    """mjcf → shm → mjcf 必須還原。sign/offset 寫錯的話這裡會先炸，
    而不是等到實機上腿往反方向甩。"""
    rng = np.random.default_rng(0)
    q12 = rng.uniform(-1.0, 1.0, 12)
    res = calib_map.mjcf12_to_shm(q12)
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(GE.JN):
            s, o = calib_map.CALIB[shm_leg][jn]
            back = (res[shm_leg][jn] - o) / s
            assert back == pytest.approx(q12[mjcf_leg * 3 + j], abs=1e-12)


def test_leg_mjcf2shm_is_a_permutation():
    """腿序重排寫錯 = 每條腿的指令都送到別條腿去，而且不會報錯。"""
    assert sorted(calib_map.LEG_MJCF2SHM) == [0, 1, 2, 3]
    # policy 腿序 (FL,FR,RL,RR) → SHM (FR,FL,RR,RL)
    assert calib_map.LEG_MJCF2SHM == [1, 0, 3, 2]


def test_captured_stand_pose_lies_inside_shm_limits(model):
    """POSE_STAND 是從這台實機擷取的，必須落在推導出來的限位內；
    否則代表 sign/offset 或限位轉換有錯。"""
    import shm_common as SC
    lim = GE.shm_limits(model)
    for shm_leg in range(4):
        for jn in GE.JN:
            lo, hi = lim[(shm_leg, jn)]
            v = SC.POSE_STAND[shm_leg][jn]
            assert lo <= v <= hi, f"leg{shm_leg}.{jn} 站姿 {v} 不在 [{lo}, {hi}]"
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'gait_export'`

- [ ] **Step 3: 建立 `gait_export.py` 的骨架與 `shm_limits`**

```python
"""gait_export.py —— 把 walk_stable 步態導出成實機可播放的關節軌跡，並做上機前的離線檢驗。

角色：本檔在【開發機】執行（要用 MuJoCo）。狗上執行的是 task6/realbot/L7_gait_shm.py。

管線：
    cpg_walk_d1.make_cpg_step + joint_targets     ← import，不複製
      → 每 20 ms 的 12 軸 MJCF 角
      → calib_map.mjcf12_to_shm
      → 五項離線檢驗
      → gait_walk_stable.npz

⚠️ 本檔【不修改】cpg_walk_d1 的任何步態參數 —— 那支腳本產生了對照影片。
   部署用的 G_C 是本檔自己的 DEPLOY_G_C（見 §為什麼不是 0.12）。

================================================================================
§ 為什麼部署用 G_C=0.110 而不是影片的 0.12
================================================================================
影片版 G_C=0.12 的膝關節在擺動相會折到距 ctrlrange 只剩 0.0114 rad（0.65°）。
而 calib_map 的 offset 誤差量級恰好就在這個尺度上（以 POSE_LIE 對照推導限位，
leg0 abad 超出 0.0089、leg2 超出 0.0138 rad）。也就是說實機的膝很可能會頂到
機構限位，而模擬顯示 clip 0.000%，離線完全看不出來。

頂限位時馬達會持續對機構硬推，靠事後力矩中止來處理是把可預防的問題留到現場，
所以改成事前用餘裕門檻擋掉。掃描結果（--sweep 可重現）：

    G_C     膝餘裕(rad)   (度)    最大角速度    離地量 FL/FR/RL/RR (mm)
    0.100     0.1266     7.25      12.27       50.6  49.9  46.2  45.9  ✓
    0.105     0.0978     5.60      12.88       54.7  54.2  49.4  49.4  ✓
    0.110     0.0690     3.95      13.49       59.5  59.0  53.1  53.4  ✓  ← 採用
    0.115     0.0402     2.30      14.10       66.3  64.7  56.3  56.0  ✗
    0.120     0.0114     0.65      14.72       76.7  77.0  56.6  55.6  ✗  ← 影片版

代價：前腳離地量從 77 掉到 59 mm（−23%）。後腳幾乎不變（56.6 → 53.1）。
副作用是前後更均勻了（比值 1.37 → 1.12）。
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "realbot"))

import calib_map
import cpg_d1
import cpg_walk_d1 as W
import d1_model

JN = ("abad", "hip", "knee")

GAIT = "walk_stable"
DEPLOY_G_C = 0.110          # 見檔頭 §。影片版是 cpg_walk_d1.GAIT_G_C = 0.12
MARGIN_MIN = 0.05           # 限位餘裕門檻(rad) = 2.9°


def shm_limits(m):
    """MJCF ctrlrange → SHM 慣例限位。回傳 {(shm_leg, joint_name): (lo, hi)}。

    ⚠️ 轉換是 shm = sign * mjcf + offset，所以 **sign = -1 時上下界會對調**。
       calib_map 裡 FR/RR 的 hip 與 knee 都是 -1。忘了 sorted() 的話限位檢驗
       會恆為「超限」或恆為「通過」，兩種都是靜默失效。
    """
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = {}
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(JN):
            col = mjcf_leg * 3 + j
            s, o = calib_map.CALIB[shm_leg][jn]
            out[(shm_leg, jn)] = tuple(sorted((s * lo[col] + o, s * hi[col] + o)))
    return out
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add task6/inference/gait_export.py task6/tests/test_gait_export.py
git commit -m "feat(task6): gait_export 骨架與 SHM 限位轉換

sign=-1 時上下界要對調（FR/RR 的 hip、knee 都是 -1），忘了 sorted()
限位檢驗會恆真或恆假，兩種都是靜默失效，所以有測試釘住。"
```

---

### Task 3: 軌跡產生與 G_C 掃描

**Files:**
- Modify: `task6/inference/gait_export.py`
- Test: `task6/tests/test_gait_export.py`

**Interfaces:**
- Consumes: Task 2 的 `shm_limits`、`JN`
- Produces:
  - `build_trajectory(m, g_c, secs=20.0) -> tuple[np.ndarray, np.ndarray]` —— 回傳 `(q_mjcf, q_shm)`，形狀 `(N, 12)` 與 `(N, 4, 3)`。`q_mjcf` 是 policy 腿序展平，`q_shm` 第一維索引是 SHM 腿序。
  - `worst_margin(q_shm, lim) -> tuple[float, str]` —— 最小餘裕與是哪一軸
  - `max_joint_vel(q_shm) -> float` —— rad/s
  - `sweep(m, g_c_values) -> None` —— 印出掃描表

- [ ] **Step 1: 寫失敗的測試**

追加到 `task6/tests/test_gait_export.py`：

```python
def test_build_trajectory_shapes_and_leg_order(model):
    q_mjcf, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=2.0)
    n = int(2.0 / d1_model.CTRL_DT)
    assert q_mjcf.shape == (n, 12)
    assert q_shm.shape == (n, 4, 3)
    # q_shm 必須是 q_mjcf 經 sign/offset + 腿序重排的結果
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(GE.JN):
            s, o = calib_map.CALIB[shm_leg][jn]
            assert q_shm[:, shm_leg, j] == pytest.approx(
                s * q_mjcf[:, mjcf_leg * 3 + j] + o, abs=1e-12)


def test_mu_y_1_5_means_abad_never_moves(model):
    """μy=1.5 → fy=0 → dy=0。abad 不動是直線走路的前提，也是限位餘裕的來源。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=5.0)
    abad = q_mjcf[:, ::3]
    assert np.ptp(abad) < 1e-9


def test_deploy_g_c_meets_margin_threshold(model):
    """DEPLOY_G_C 必須通過餘裕門檻——這是選它的唯一理由。"""
    _, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    margin, where = GE.worst_margin(q_shm, GE.shm_limits(model))
    assert margin >= GE.MARGIN_MIN, f"{where} 餘裕只有 {margin:.4f}"


def test_video_g_c_would_fail_the_margin_threshold(model):
    """釘住我們為什麼不用影片那組參數。這個測試轉綠代表門檻或校正被改動了。"""
    _, q_shm = GE.build_trajectory(model, W.GAIT_G_C, secs=20.0)
    margin, _ = GE.worst_margin(q_shm, GE.shm_limits(model))
    assert margin < GE.MARGIN_MIN


def test_worst_margin_identifies_the_knee(model):
    _, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    _, where = GE.worst_margin(q_shm, GE.shm_limits(model))
    assert "knee" in where


def test_max_joint_vel_far_exceeds_l4_threshold(model):
    """步態需要 ~13.5 rad/s，L4 的 VEL_ABORT=2.0 直接搬會一路誤中止。"""
    _, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    v = GE.max_joint_vel(q_shm)
    assert 12.0 < v < 15.0
```

需要在 test 檔頂端補 `import cpg_walk_d1 as W`。

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: FAIL —— `AttributeError: module 'gait_export' has no attribute 'build_trajectory'`

- [ ] **Step 3: 實作**

加到 `gait_export.py`：

```python
def build_trajectory(m, g_c, secs=20.0):
    """產生步態的關節軌跡。回傳 (q_mjcf (N,12), q_shm (N,4,3))。

    軌跡邏輯全部來自 cpg_walk_d1（import，不複製）。本函式只負責：
    指定 g_c、跑滿 secs、把結果轉成 SHM 慣例。
    """
    cfg = W.GAITS[GAIT]
    step = W.make_cpg_step(cfg["phase"])
    f0s, jinvs = cpg_d1.leg_ik_consts(m)

    c = cpg_d1.cpg_init()
    c["theta"] = cfg["phase"].copy()      # cpg_init 用的是 d1_model 的 trot 相位
    n = int(secs / d1_model.CTRL_DT)
    q_mjcf = np.zeros((n, 12))
    for i in range(n):
        c = step(c, np.full(4, cfg["mu_x"]), np.full(4, W.MU_Y),
                 np.full(4, cfg["omega"]), d1_model.CTRL_DT)
        q_mjcf[i] = W.joint_targets(c, f0s, jinvs, cfg["x_off"], g_c, cfg["duty"])

    q_shm = np.zeros((n, 4, 3))
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(JN):
            s, o = calib_map.CALIB[shm_leg][jn]
            q_shm[:, shm_leg, j] = s * q_mjcf[:, mjcf_leg * 3 + j] + o
    return q_mjcf, q_shm


def worst_margin(q_shm, lim):
    """最小限位餘裕(rad) 與發生在哪一軸。負值代表已經超限。"""
    from shm_common import LEGNAME
    worst, where = np.inf, ""
    for shm_leg in range(4):
        for j, jn in enumerate(JN):
            lo, hi = lim[(shm_leg, jn)]
            col = q_shm[:, shm_leg, j]
            for margin, side in ((col.min() - lo, "下界"), (hi - col.max(), "上界")):
                if margin < worst:
                    worst, where = margin, f"leg{shm_leg}({LEGNAME[shm_leg]}).{jn} {side}"
    return float(worst), where


def max_joint_vel(q_shm):
    """逐幀差分的最大關節角速度(rad/s)。用來設 L7 的 VEL_ABORT。"""
    return float(np.abs(np.diff(q_shm, axis=0) / d1_model.CTRL_DT).max())


def sweep(m, g_c_values):
    """印出 G_C 掃描表。用來重現檔頭 § 的那張表。"""
    lim = shm_limits(m)
    print(f"{'G_C':>6} {'膝餘裕(rad)':>12} {'(度)':>7} {'最大角速度':>11}  ")
    for g_c in g_c_values:
        _, q_shm = build_trajectory(m, g_c)
        margin, _ = worst_margin(q_shm, lim)
        mark = "✓" if margin >= MARGIN_MIN else "✗"
        print(f"{g_c:>6.3f} {margin:>12.4f} {np.degrees(margin):>7.2f} "
              f"{max_joint_vel(q_shm):>11.2f}  {mark}")
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: PASS（11 passed）

- [ ] **Step 5: 用 `--sweep` 重現掃描表**

先加最小的 CLI（完整 CLI 在 Task 4 補齊）：

```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="印出 G_C 掃描表")
    args = ap.parse_args()
    if args.sweep:
        sweep(d1_model.make_model(),
              (0.080, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120))
```

Run: `conda run --no-capture-output -n rbtdog python task6/inference/gait_export.py --sweep`
Expected: 0.110 那列餘裕 0.0690、標 ✓；0.115 與 0.120 標 ✗。與檔頭 § 的表一致。

- [ ] **Step 6: Commit**

```bash
git add task6/inference/gait_export.py task6/tests/test_gait_export.py
git commit -m "feat(task6): 軌跡產生與 G_C 掃描，部署值定為 0.110

影片版 0.12 的膝距 ctrlrange 只剩 0.65 度，而 calib_map 的 offset
誤差量級就是 0.5~0.8 度，實機很可能頂限位而離線看不出來。
0.110 給 3.95 度餘裕，代價是前腳離地 77->59mm，但前後比值
1.37->1.12 反而更均勻。有測試釘住「0.12 過不了門檻」。"
```

---

### Task 4: 五項離線檢驗與 npz 匯出

**Files:**
- Modify: `task6/inference/gait_export.py`
- Test: `task6/tests/test_gait_export.py`

**Interfaces:**
- Consumes: Task 3 的 `build_trajectory`、`worst_margin`、`max_joint_vel`
- Produces:
  - `calib_hash() -> str` —— `calib_map` 內容的 sha256 前 16 碼
  - `run_checks(m, q_mjcf, q_shm) -> tuple[bool, list[str], dict]` —— `(ok, 問題清單, 統計)`
  - `export(m, out_path, g_c=DEPLOY_G_C, secs=20.0) -> Path`
  - npz 欄位契約：`t (N,)`、`q_mjcf (N,12)`、`q_shm (N,4,3)`、`meta_json` (0-d str)
  - `meta_json` 內的 key：`gait, g_c, omega, mu_x, mu_y, x_off, duty, ctrl_dt, secs, calib_hash, max_joint_vel, worst_margin, start_offset_from_stand`

- [ ] **Step 1: 寫失敗的測試**

追加到 `task6/tests/test_gait_export.py`：

```python
def test_calib_hash_changes_when_calibration_changes(monkeypatch):
    """npz 帶著校正雜湊，是為了防止『改了校正卻拿舊軌跡去跑』。"""
    h0 = GE.calib_hash()
    patched = {k: dict(v) for k, v in calib_map.CALIB.items()}
    patched[0]["hip"] = (+1, +1.166)          # 把暫定的 hip 號翻過來
    monkeypatch.setattr(calib_map, "CALIB", patched)
    assert GE.calib_hash() != h0


def test_run_checks_passes_for_deploy_g_c(model):
    q_mjcf, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    ok, problems, stats = GE.run_checks(model, q_mjcf, q_shm)
    assert ok, problems
    assert stats["clip_pct"] == 0.0
    assert stats["worst_margin"] >= GE.MARGIN_MIN


def test_run_checks_rejects_video_g_c_on_margin(model):
    q_mjcf, q_shm = GE.build_trajectory(model, W.GAIT_G_C, secs=20.0)
    ok, problems, _ = GE.run_checks(model, q_mjcf, q_shm)
    assert not ok
    assert any("餘裕" in p for p in problems)


def test_run_checks_catches_a_discontinuity(model):
    """跨幀跳變檢驗：注入一個階躍，必須被抓到。"""
    q_mjcf, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=5.0)
    q_shm = q_shm.copy()
    q_shm[100:, 1, 2] += 0.9
    ok, problems, _ = GE.run_checks(model, q_mjcf, q_shm)
    assert not ok
    assert any("跳變" in p for p in problems)


def test_export_writes_npz_with_the_agreed_schema(model, tmp_path):
    out = GE.export(model, tmp_path / "g.npz", secs=2.0)
    z = np.load(out, allow_pickle=False)
    assert set(z.files) == {"t", "q_mjcf", "q_shm", "meta_json"}
    n = int(2.0 / d1_model.CTRL_DT)
    assert z["t"].shape == (n,)
    assert z["q_mjcf"].shape == (n, 12)
    assert z["q_shm"].shape == (n, 4, 3)
    assert z["t"][1] - z["t"][0] == pytest.approx(d1_model.CTRL_DT)

    import json
    meta = json.loads(str(z["meta_json"]))
    for k in ("gait", "g_c", "omega", "mu_x", "mu_y", "x_off", "duty", "ctrl_dt",
              "secs", "calib_hash", "max_joint_vel", "worst_margin",
              "start_offset_from_stand"):
        assert k in meta, k
    assert meta["g_c"] == pytest.approx(GE.DEPLOY_G_C)
    assert meta["gait"] == "walk_stable"
    assert meta["calib_hash"] == GE.calib_hash()


def test_export_refuses_when_checks_fail(model, tmp_path):
    out = tmp_path / "bad.npz"
    with pytest.raises(SystemExit):
        GE.export(model, out, g_c=W.GAIT_G_C, secs=2.0)
    assert not out.exists(), "檢驗沒過就不該留下檔案"


def test_start_offset_from_stand_matches_measured_value(model):
    """起步位移決定 ramp 時間。實測最大 0.4553 rad（leg1 hip）。"""
    q_mjcf, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=2.0)
    _, _, stats = GE.run_checks(model, q_mjcf, q_shm)
    assert 0.2 < stats["start_offset_from_stand"] < 0.7
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: FAIL —— `AttributeError: module 'gait_export' has no attribute 'calib_hash'`

- [ ] **Step 3: 實作**

加到 `gait_export.py`（頂端補 `import hashlib`、`import json`）：

```python
# 跨幀跳變門檻(rad)。步態本身單 tick 最大約 0.27 rad（13.5 rad/s × 0.02s），
# 取 0.40 留餘裕：正常軌跡不會碰到，注入式的階躍會被抓到。
JUMP_MAX = 0.40


def calib_hash():
    """calib_map 內容的雜湊。npz 帶著它，L7 上機前比對，不符就拒跑。"""
    payload = json.dumps(
        {"legs": calib_map.LEG_MJCF2SHM,
         "calib": {str(k): {jn: list(v[jn]) for jn in JN}
                   for k, v in sorted(calib_map.CALIB.items())}},
        sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def run_checks(m, q_mjcf, q_shm):
    """五項離線檢驗。回傳 (ok, 問題清單, 統計)。"""
    from shm_common import POSE_STAND
    lim = shm_limits(m)
    problems = []

    # 1) 限位餘裕
    margin, where = worst_margin(q_shm, lim)
    if margin < MARGIN_MIN:
        problems.append(f"限位餘裕不足：{where} 只剩 {margin:.4f} rad "
                        f"（{np.degrees(margin):.2f}°），門檻 {MARGIN_MIN}")

    # 2) 角速度（不擋，只記錄——L7 拿它設 VEL_ABORT）
    vmax = max_joint_vel(q_shm)

    # 3) 跨幀跳變
    jump = np.abs(np.diff(q_shm, axis=0))
    if jump.max() > JUMP_MAX:
        idx = np.unravel_index(jump.argmax(), jump.shape)
        problems.append(f"跨幀跳變過大：{jump.max():.4f} rad @ 幀{idx[0]} "
                        f"leg{idx[1]}.{JN[idx[2]]}，門檻 {JUMP_MAX}")

    # 4) 起步位移（不擋，只記錄——L7 拿它算 ramp 時間）
    start_off = max(abs(q_shm[0, leg, j] - POSE_STAND[leg][jn])
                    for leg in range(4) for j, jn in enumerate(JN))

    # 5) MJCF 側 ctrlrange clip 率必須為 0
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    clipped = int(np.sum((q_mjcf < lo - 1e-9) | (q_mjcf > hi + 1e-9)))
    clip_pct = 100.0 * clipped / q_mjcf.size
    if clipped:
        problems.append(f"MJCF ctrlrange clip {clip_pct:.3f}%（{clipped} 個指令），必須為 0")

    stats = {"worst_margin": float(margin), "worst_margin_at": where,
             "max_joint_vel": float(vmax), "max_frame_jump": float(jump.max()),
             "start_offset_from_stand": float(start_off), "clip_pct": clip_pct}
    return not problems, problems, stats


def export(m, out_path, g_c=DEPLOY_G_C, secs=20.0):
    """產生軌跡、跑檢驗、寫 npz。檢驗沒過就 sys.exit(1) 且不留檔案。"""
    cfg = W.GAITS[GAIT]
    q_mjcf, q_shm = build_trajectory(m, g_c, secs)
    ok, problems, stats = run_checks(m, q_mjcf, q_shm)

    print(f"[檢驗] G_C={g_c}  最小餘裕 {stats['worst_margin']:.4f} rad "
          f"（{np.degrees(stats['worst_margin']):.2f}°）@ {stats['worst_margin_at']}")
    print(f"       最大角速度 {stats['max_joint_vel']:.2f} rad/s  "
          f"最大跨幀跳變 {stats['max_frame_jump']:.4f} rad  "
          f"起步位移 {stats['start_offset_from_stand']:.4f} rad  "
          f"clip {stats['clip_pct']:.3f}%")
    if not ok:
        for p in problems:
            print(f"  ✗ {p}")
        print("→ 拒絕匯出。修正參數後重跑。")
        sys.exit(1)

    meta = {"gait": GAIT, "g_c": float(g_c), "omega": cfg["omega"],
            "mu_x": cfg["mu_x"], "mu_y": W.MU_Y, "x_off": cfg["x_off"],
            "duty": cfg["duty"], "ctrl_dt": d1_model.CTRL_DT, "secs": float(secs),
            "calib_hash": calib_hash(), **stats}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             t=np.arange(len(q_mjcf)) * d1_model.CTRL_DT,
             q_mjcf=q_mjcf, q_shm=q_shm,
             meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"[匯出] {out_path}  {len(q_mjcf)} 幀  calib_hash={meta['calib_hash']}")
    return out_path
```

把 `__main__` 區塊換成：

```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="walk_stable 軌跡導出與離線檢驗")
    ap.add_argument("--sweep", action="store_true", help="印出 G_C 掃描表")
    ap.add_argument("--export", metavar="PATH",
                    default=None, help="匯出 npz 到指定路徑")
    ap.add_argument("--g-c", type=float, default=DEPLOY_G_C, dest="g_c")
    ap.add_argument("--secs", type=float, default=20.0)
    args = ap.parse_args()

    model = d1_model.make_model()
    if args.sweep:
        sweep(model, (0.080, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120))
    if args.export:
        export(model, args.export, args.g_c, args.secs)
    if not args.sweep and not args.export:
        ap.error("要 --sweep 或 --export 其中之一")
```

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: PASS（18 passed）

- [ ] **Step 5: 產生正式的部署軌跡檔**

Run:
```bash
conda run --no-capture-output -n rbtdog python task6/inference/gait_export.py \
  --export task6/weights/gait_walk_stable.npz --secs 20
```
Expected: 印出 `最小餘裕 0.0690 rad（3.95°）`、`clip 0.000%`、`[匯出] ... 1000 幀`

- [ ] **Step 6: Commit**

```bash
git add task6/inference/gait_export.py task6/tests/test_gait_export.py task6/weights/gait_walk_stable.npz
git commit -m "feat(task6): 五項離線檢驗與 npz 匯出

npz 帶 calib_hash，L7 上機前比對，防止改了校正卻拿舊軌跡去跑。
檢驗沒過 sys.exit 且不留檔案——半個檔案比沒有檔案更危險。
max_joint_vel 與 start_offset 寫進 meta，L7 拿來設保護門檻與 ramp 時間。"
```

---

### Task 5: L7 的離線核心（時間縮放、上取樣、live 一致性）

先把 L7 裡**不碰硬體**的部分做完並測透。這一塊決定寫進 SHM 的每一個數字，值得單獨的測試迴圈。

**Files:**
- Create: `task6/realbot/L7_gait_shm.py`
- Test: `task6/tests/test_L7_gait.py`

**Interfaces:**
- Consumes: Task 4 的 npz 契約、Task 1 的 `shm_common`
- Produces:
  - `load_trajectory(npz_path) -> tuple[np.ndarray, dict]` —— `(q_shm (N,4,3), meta)`；`calib_hash` 不符就 `sys.exit(1)`
  - `sample_at(q_shm, ctrl_dt, u) -> np.ndarray` —— 在時間 `u`（秒，標量或陣列）線性內插，回傳 `(4,3)` 或 `(len(u),4,3)`
  - `playback_times(n_frames, ctrl_dt, time_scale, hz) -> np.ndarray` —— 播放時每個 500 Hz 週期對應的軌跡時間
  - `live_trajectory(npz_path, secs) -> np.ndarray` —— 狗上純 numpy 重算，形狀同 npz 的 `q_shm`

> 保護門檻 `guard_thresholds` 由 Task 6 提供 —— 它需要 Task 6 才算得出來的空中模擬表。

> ⚠️ `live_trajectory` 不能 import mujoco。IK 常數 `f0s`/`jinvs` 由 npz 一併帶上狗 —— 因此 Task 4 的 npz 要**追加兩個欄位**：`f0s (4,3)`、`jinvs (4,3,3)`。回到 `gait_export.export()` 補上，並更新 schema 測試。

- [ ] **Step 1: 先補 npz 的 IK 常數欄位**

在 `gait_export.py` 的 `export()` 裡，`np.savez` 加兩個欄位：

```python
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    np.savez(out_path,
             t=np.arange(len(q_mjcf)) * d1_model.CTRL_DT,
             q_mjcf=q_mjcf, q_shm=q_shm, f0s=f0s, jinvs=jinvs,
             meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
```

並把 `test_export_writes_npz_with_the_agreed_schema` 的 `z.files` 斷言改成：

```python
    assert set(z.files) == {"t", "q_mjcf", "q_shm", "f0s", "jinvs", "meta_json"}
    assert z["f0s"].shape == (4, 3)
    assert z["jinvs"].shape == (4, 3, 3)
```

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -q`
Expected: PASS（18 passed）

- [ ] **Step 2: 寫失敗的測試**

建立 `task6/tests/test_L7_gait.py`：

```python
"""L7 的離線核心測試。不碰硬體，不需要狗。"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "inference"))
sys.path.insert(0, str(_ROOT / "realbot"))

import d1_model
import gait_export as GE
import L7_gait_shm as L7


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    m = d1_model.make_model()
    return GE.export(m, tmp_path_factory.mktemp("w") / "gait.npz", secs=6.0)


def test_load_trajectory_returns_shape_and_meta(npz):
    q, meta = L7.load_trajectory(npz)
    assert q.shape == (int(6.0 / d1_model.CTRL_DT), 4, 3)
    assert meta["gait"] == "walk_stable"
    assert meta["calib_hash"] == GE.calib_hash()


def test_load_trajectory_refuses_stale_calibration(npz, monkeypatch):
    """改了校正卻拿舊軌跡去跑 = 每個關節都下錯指令。必須擋下來。"""
    monkeypatch.setattr(L7, "expected_calib_hash", lambda: "deadbeefdeadbeef")
    with pytest.raises(SystemExit):
        L7.load_trajectory(npz)


def test_sample_at_reproduces_frames_exactly_on_grid(npz):
    q, meta = L7.load_trajectory(npz)
    dt = meta["ctrl_dt"]
    for i in (0, 7, 42, len(q) - 1):
        assert L7.sample_at(q, dt, i * dt) == pytest.approx(q[i], abs=1e-12)


def test_sample_at_midpoint_is_the_average(npz):
    q, meta = L7.load_trajectory(npz)
    dt = meta["ctrl_dt"]
    mid = L7.sample_at(q, dt, 3.5 * dt)
    assert mid == pytest.approx((q[3] + q[4]) / 2, abs=1e-12)


def test_sample_at_clamps_past_the_end(npz):
    q, meta = L7.load_trajectory(npz)
    dt = meta["ctrl_dt"]
    assert L7.sample_at(q, dt, 1e6) == pytest.approx(q[-1], abs=1e-12)


def test_playback_times_scale_duration_inversely(npz):
    q, meta = L7.load_trajectory(npz)
    dt, n = meta["ctrl_dt"], len(q)
    full = L7.playback_times(n, dt, time_scale=1.0, hz=500)
    quarter = L7.playback_times(n, dt, time_scale=0.25, hz=500)
    # 四分之一速 → 播放週期數變四倍，但走過的軌跡時間相同
    assert len(quarter) == pytest.approx(4 * len(full), rel=0.01)
    assert quarter[-1] == pytest.approx(full[-1], rel=1e-6)


def test_upsampling_shrinks_per_step_jump_by_ten(npz):
    """500 Hz 內插不是效能優化，是安全需求：50 Hz 直寫單 tick 跳 0.29 rad，
    kp=20 下瞬間 5.9 N·m，逼近中止門檻。"""
    q, meta = L7.load_trajectory(npz)
    dt = meta["ctrl_dt"]
    raw = np.abs(np.diff(q, axis=0)).max()
    u = L7.playback_times(len(q), dt, 1.0, 500)
    fine = np.abs(np.diff(L7.sample_at(q, dt, u), axis=0)).max()
    assert fine < raw / 8


def test_live_trajectory_matches_the_file_frame_by_frame(npz):
    """離線檔是黃金標準。live 不吻合就不准用。"""
    q_file, meta = L7.load_trajectory(npz)
    q_live = L7.live_trajectory(npz, meta["secs"])
    assert q_live.shape == q_file.shape
    assert q_live == pytest.approx(q_file, abs=1e-9)


```

- [ ] **Step 3: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_L7_gait.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'L7_gait_shm'`

- [ ] **Step 4: 實作 L7 的離線核心**

建立 `task6/realbot/L7_gait_shm.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L7_gait_shm.py —— 把 walk_stable 步態以 500 Hz 串流寫進 /dev/shm/spline_shm。

⚠️ 這是【吊掛空跑】用的。狗必須吊起來、四腳離地。不是落地行走工具。
   右後腿(leg2)整條已從 CAN 失聯，一律用 --skip-legs 2 排除。

三種模式：
  jog    單關節 ±0.10 rad 三角波。用來確認 calib_map 的正負號。
         calib_map 自己標注 hip 的號是「暫定」——號反了腿會往反方向甩到限位，
         而離線檢驗完全看不出來，因為數字本身很正常。所以這關必須先過。
  leg    只驅動一條腿跑完整步態，其餘零增益。把風險限制在單腿。
  gait   三條腿同步跑。

依賴限制：本檔在【車載電腦】上執行，只能用 python3 標準庫 + numpy。
不得 import mujoco —— 狗上沒有。IK 常數由 npz 帶上來。

用法：
  python3 L7_gait_shm.py --mode gait --traj gait_walk_stable.npz --time-scale 0.25
  sudo python3 L7_gait_shm.py --mode gait --traj gait_walk_stable.npz \
       --time-scale 0.25 --skip-legs 2 --confirm
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib_map
import shm_common as SC

# --- live 模式重列的常數 ---
# 必須與 d1_model / cpg_walk_d1 一致。狗上不能 import d1_model（它 import mujoco），
# 所以在此重列。提到模組層級是為了讓 test_live_constants_match_d1_model 讀得到值
# 逐項比對 —— 比對數值，不是比對原始碼字串。
MU_MIN, MU_MAX = 1.0, 2.0
A_CONV, W_COUP, N_CPG_SUB = 50.0, 8.0, 4
D_STEP, D_STEP_Y, G_P = 0.12, 0.09, 0.01
HOME3 = np.array([0.0, 1.05, -2.00])
PHASE_WALK = np.array([0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi])

LEG_KP, LEG_KD = 20.0, 0.7     # 原廠站立實測值
STOP_KD = 3.0
CATCH_SEC = 0.5
RAMP_MIN_SEC = 2.0


def expected_calib_hash():
    """本機 calib_map 的雜湊。與 gait_export.calib_hash() 必須用同一套算法。

    不 import gait_export——那支要 mujoco，狗上沒有。所以這裡重算一份。
    ⚠️ 兩邊的算法必須逐字相同，有測試 test_calib_hash_agrees_across_modules 釘住。
    """
    import hashlib
    payload = json.dumps(
        {"legs": calib_map.LEG_MJCF2SHM,
         "calib": {str(k): {jn: list(v[jn]) for jn in ("abad", "hip", "knee")}
                   for k, v in sorted(calib_map.CALIB.items())}},
        sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_trajectory(npz_path):
    """讀 npz。回傳 (q_shm (N,4,3), meta)。校正雜湊不符就拒跑。"""
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    here = expected_calib_hash()
    if meta["calib_hash"] != here:
        print(f"✗ 校正雜湊不符：軌跡檔是 {meta['calib_hash']}、本機是 {here}")
        print("  代表 calib_map 改過但軌跡沒重產。每個關節都會下錯指令，拒絕執行。")
        print("  重新產生：python gait_export.py --export <path>")
        sys.exit(1)
    return z["q_shm"], meta


def sample_at(q_shm, ctrl_dt, u):
    """在軌跡時間 u（秒）線性內插。u 可為標量或陣列。超過尾端就夾住。

    500 Hz 上取樣與 --time-scale 是同一個操作：都只是「在什麼時間點取樣」。
    """
    u = np.asarray(u, dtype=float)
    n = len(q_shm)
    x = np.clip(u / ctrl_dt, 0.0, n - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (x - i0)[..., None, None]
    out = q_shm[i0] * (1.0 - w) + q_shm[i1] * w
    return out


def playback_times(n_frames, ctrl_dt, time_scale, hz=SC.CTRL_HZ):
    """播放時每個控制週期對應的軌跡時間(秒)。

    time_scale 是播放倍率：0.25 = 四分之一速，走完同一條軌跡要花四倍時間。
    """
    total = (n_frames - 1) * ctrl_dt / time_scale      # 實際牆鐘秒數
    n_steps = int(round(total * hz)) + 1
    return np.arange(n_steps) / hz * time_scale


def live_trajectory(npz_path, secs):
    """狗上純 numpy 自己算軌跡，不讀 npz 的 q_shm。**離線檔是黃金標準**，
    不逐幀吻合就不准用。

    ★ CPG 一律在【未縮放的 50 Hz 網格】上積分，與 --time-scale 無關。
      時間縮放純粹是播放層的事（sample_at + playback_times），兩個模式共用。
      這樣 live 與 file 在任何倍速下都產生完全相同的路點序列。

      不要改成「用縮放後的 dt 即時積分」：那會在慢速下產生更密的路點，
      在擺動→站立交界處（duty_remap 讓 dθ/dt 跳 4 倍的那個折點）與 file
      差到 7.5°。兩者都不算錯，但就不再是同一條指令串流，黃金標準也就沒了。
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return _cpg_rollout(meta, z["jinvs"], secs)   # f0s 只給診斷用，這裡不需要


def _cpg_rollout(meta, jinvs, secs):
    """CPG 積分 + IK + calib_map，純 numpy。逐行對應 cpg_walk_d1 的實作。

    ⚠️ 這是 cpg_walk_d1 的第二份實作。這是【使用者決策的例外】而非疏忽 ——
       需求是「一個模式播寫死的軌跡、一個模式在狗上自己算」，而狗上沒有
       mujoco（leg_ik_consts 要跑 forward kinematics），無法 import 那份。
       折衷是把 jinvs 預先算好由 npz 帶上來，讓這裡只剩純算術，並用
       test_live_trajectory_matches_the_file_frame_by_frame 逐幀釘住
       （容忍 1e-9）。那個測試轉紅就代表兩份漂移了。

    ★ 一律用 meta["ctrl_dt"]（50 Hz）積分，不吃 time_scale —— 見 live_trajectory。
    """
    dt = meta["ctrl_dt"]
    mux, muy = np.full(4, meta["mu_x"]), np.full(4, meta["mu_y"])
    omega = np.full(4, meta["omega"])
    duty, x_off, g_c = meta["duty"], meta["x_off"], meta["g_c"]

    PHI = PHASE_WALK[None, :] - PHASE_WALK[:, None]
    rx, rxd = np.full(4, 1.5), np.zeros(4)
    ry, ryd = np.full(4, 1.5), np.zeros(4)
    th = PHASE_WALK.copy()

    n = int(secs / dt)
    out = np.zeros((n, 4, 3))
    for i in range(n):
        h = dt / N_CPG_SUB
        for _ in range(N_CPG_SUB):
            rxd += (A_CONV * (A_CONV / 4 * (mux - rx) - rxd)) * h
            rx += rxd * h
            ryd += (A_CONV * (A_CONV / 4 * (muy - ry) - ryd)) * h
            ry += ryd * h
            rbar = 0.5 * (rx + ry)
            diff = th[None, :] - th[:, None] - PHI
            th = th + (2 * np.pi * omega + W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
        th %= 2 * np.pi

        ph = (th % (2 * np.pi)) / (2 * np.pi)
        sw = 1.0 - duty
        thr = np.where(ph < sw, np.pi * ph / sw, np.pi + np.pi * (ph - sw) / duty)

        fx = 2 * (rx - MU_MIN) / (MU_MAX - MU_MIN) - 1
        fy = 2 * (ry - MU_MIN) / (MU_MAX - MU_MIN) - 1
        dx = -D_STEP * fx * np.cos(thr) + x_off
        dy = D_STEP_Y * fy * np.cos(thr)
        dz = np.where(np.sin(thr) > 0, g_c * np.sin(thr), G_P * np.sin(thr))
        off = np.stack([dx, dy, dz], -1)

        for mjcf_leg in range(4):
            q3 = HOME3 + jinvs[mjcf_leg] @ off[mjcf_leg]
            shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
            for j, jn in enumerate(("abad", "hip", "knee")):
                s, o = calib_map.CALIB[shm_leg][jn]
                out[i, shm_leg, j] = s * q3[j] + o
    return out


```

- [ ] **Step 5: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_L7_gait.py -q`
Expected: PASS（8 passed）

- [ ] **Step 6: 加兩個防漂移的測試**

`_cpg_rollout` 重列了常數、`expected_calib_hash` 重算了雜湊。兩處都必須釘住。追加到 `task6/tests/test_L7_gait.py`：

```python
def test_calib_hash_agrees_across_modules():
    """L7 不能 import gait_export（那支要 mujoco），所以雜湊算了兩份。
    兩份不一致的話，每次都會誤判成『校正過期』而拒跑。"""
    assert L7.expected_calib_hash() == GE.calib_hash()


def test_live_constants_match_d1_model():
    """_cpg_rollout 重列了 d1_model 的常數（狗上不能 import d1_model）。
    任何一個漂掉，live 路線就會安靜地產生不同的軌跡。

    ⚠️ 比對【實際數值】，不要比對原始碼字串。用 inspect.getsource 比字串
       會在無害的排版改動上誤報，又抓不到「宣告沒變但別處覆寫了」的情況。
       把常數從函式裡提到模組層級，就能直接讀值比對。
    """
    assert (L7.MU_MIN, L7.MU_MAX) == (d1_model.MU_MIN, d1_model.MU_MAX)
    assert L7.A_CONV == d1_model.A_CONV
    assert L7.W_COUP == d1_model.W_COUP
    assert L7.N_CPG_SUB == d1_model.N_CPG_SUB
    assert L7.D_STEP == d1_model.D_STEP
    assert L7.D_STEP_Y == d1_model.D_STEP_Y
    assert L7.G_P == d1_model.G_P
    assert list(L7.HOME3) == list(d1_model.HOME3)
    import cpg_walk_d1 as W
    assert list(L7.PHASE_WALK) == list(W.PHASE_WALK)


def test_live_matches_file_at_every_playback_speed(npz):
    """時間縮放是播放層的事，CPG 一律在未縮放的 50 Hz 網格上積分。
    所以 live 與 file 在任何倍速下都必須產生相同的 500 Hz 指令串流。"""
    q_file, meta = L7.load_trajectory(npz)
    q_live = L7.live_trajectory(npz, meta["secs"])
    dt = meta["ctrl_dt"]
    for s in (0.25, 0.5, 1.0):
        u = L7.playback_times(len(q_file), dt, s)
        assert L7.sample_at(q_live, dt, u) == pytest.approx(
            L7.sample_at(q_file, dt, u), abs=1e-9), f"{s}× 不吻合"
```

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_L7_gait.py -q`
Expected: PASS（11 passed）

- [ ] **Step 7: 確認狗上的依賴限制沒被違反**

Run:
```bash
cd /home/huang/rbtdog_sim/task6/realbot
grep -nE "^\s*(import|from)\s+(mujoco|torch|scipy|onnx)" shm_common.py L7_gait_shm.py calib_map.py \
  && echo "✗ 違反狗上依賴限制" || echo "✓ 狗上依賴 OK"
```
Expected: `✓ 狗上依賴 OK`

- [ ] **Step 8: Commit**

```bash
git add task6/realbot/L7_gait_shm.py task6/tests/test_L7_gait.py task6/inference/gait_export.py task6/tests/test_gait_export.py
git commit -m "feat(task6): L7 離線核心——時間縮放、500Hz 上取樣、live 一致性

sample_at 讓上取樣與 --time-scale 變成同一個操作：都只是選取樣時間點。
live 路線在狗上重算（沒有 mujoco），jinvs 由 npz 帶上去，
兩份用逐幀 1e-9 的測試釘死，並加測試釘住重列的常數與雜湊算法。"
```

---

### Task 6: 吊掛空跑模擬 → 保護門檻與誤差預測表

保護門檻不能用猜的，也**不能用逆動力學算**。改成直接模擬這次要跑的東西：機身固定、無接觸、位置伺服設成實際要用的 `kp/kd`，沿軌跡跑一遍，取致動器力矩與關節速度峰值。

> **為什麼不用逆動力學**：對這條軌跡跑 `mj_inverse` 會得到 45.8 N·m，超過 URDF 的 28 N·m effort 上限。原因是 `duty_remap` 在擺動→站立交界處讓 dθ/dt 差 4 倍（`duty/(1-duty)`），指令軌跡在該處有速度折點，二階差分的加速度會爆掉。那個數字描述「完美追蹤所需的力矩」，而完美追蹤本來就不會發生。實際力矩由 `τ = kp·err + kd·(0−v)` 決定。

副產品是每一級的**追蹤誤差預測值** —— 實機量到的數字可以直接對照。

**Files:**
- Modify: `task6/inference/gait_export.py`、`task6/realbot/L7_gait_shm.py`
- Test: `task6/tests/test_gait_export.py`、`task6/tests/test_L7_gait.py`

**Interfaces:**
- Produces:
  - `air_servo_sim(m, q_mjcf, kp, kd, time_scale) -> dict` —— key：`tau_peak`、`err_peak_deg`、`err_rms_deg`、`vel_peak`
  - `AIR_SIM_GRID: tuple` —— 要預先算的 `((kp, kd), ...)` 組合
  - `AIR_SIM_SCALES: tuple = (0.25, 0.5, 1.0)`
  - npz `meta["air_sim"]`：`{"20.0/0.7": {"0.25": {...}, ...}, ...}`
- Consumes（L7 端改寫）：`guard_thresholds(meta, kp, kd, time_scale)`，查不到組合就 `sys.exit(1)`

- [ ] **Step 1: 寫失敗的測試**

追加到 `task6/tests/test_gait_export.py`：

```python
def test_air_servo_sim_matches_the_measured_baseline(model):
    """原廠增益 1.0× 的基準值。這幾個數字是保護門檻與誤差預測的來源，
    偏離超過 20% 代表模型或軌跡被改動了，要重新確認而不是改門檻。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=8.0)
    r = GE.air_servo_sim(model, q_mjcf, kp=20.0, kd=0.7, time_scale=1.0)
    assert r["tau_peak"] == pytest.approx(10.18, rel=0.20)
    assert r["err_peak_deg"] == pytest.approx(39.20, rel=0.20)
    assert r["err_rms_deg"] == pytest.approx(9.30, rel=0.20)
    assert r["vel_peak"] == pytest.approx(12.96, rel=0.20)


def test_air_servo_sim_gets_easier_when_slowed_down(model):
    """--time-scale 存在的理由：放慢之後力矩與誤差都要顯著下降。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=8.0)
    fast = GE.air_servo_sim(model, q_mjcf, 20.0, 0.7, 1.0)
    slow = GE.air_servo_sim(model, q_mjcf, 20.0, 0.7, 0.25)
    assert slow["tau_peak"] < fast["tau_peak"] / 3
    assert slow["err_rms_deg"] < fast["err_rms_deg"] / 3


def test_full_speed_torque_exceeds_l4_ceiling(model):
    """釘住『L4 的 8.0 不能照搬』這個結論。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=8.0)
    r = GE.air_servo_sim(model, q_mjcf, 20.0, 0.7, 1.0)
    assert r["tau_peak"] > 8.0


def test_export_embeds_the_air_sim_table(model, tmp_path):
    import json
    out = GE.export(model, tmp_path / "g.npz", secs=8.0)
    meta = json.loads(str(np.load(out, allow_pickle=False)["meta_json"]))
    table = meta["air_sim"]
    assert "20.0/0.7" in table
    for s in ("0.25", "0.5", "1.0"):
        entry = table["20.0/0.7"][s]
        assert set(entry) == {"tau_peak", "err_peak_deg", "err_rms_deg", "vel_peak"}
```

追加到 `task6/tests/test_L7_gait.py`（**取代** Task 5 寫的兩個 `guard_thresholds` 測試）：

```python
def test_guard_thresholds_come_from_the_embedded_table(npz):
    _, meta = L7.load_trajectory(npz)
    entry = meta["air_sim"]["20.0/0.7"]["1.0"]
    t, v = L7.guard_thresholds(meta, 20.0, 0.7, 1.0)
    assert t == pytest.approx(entry["tau_peak"] * L7.TORQUE_SAFETY)
    assert v == pytest.approx(entry["vel_peak"] * L7.VEL_SAFETY)


def test_guard_thresholds_are_looser_at_full_speed_than_quarter(npz):
    _, meta = L7.load_trajectory(npz)
    t_slow, v_slow = L7.guard_thresholds(meta, 20.0, 0.7, 0.25)
    t_fast, v_fast = L7.guard_thresholds(meta, 20.0, 0.7, 1.0)
    assert t_slow < t_fast and v_slow < v_fast


def test_guard_thresholds_refuse_an_untabulated_combination(npz):
    """門檻不能用猜的。沒算過的 kp/倍速組合就拒跑。"""
    _, meta = L7.load_trajectory(npz)
    with pytest.raises(SystemExit):
        L7.guard_thresholds(meta, 33.0, 0.7, 1.0)
    with pytest.raises(SystemExit):
        L7.guard_thresholds(meta, 20.0, 0.7, 0.75)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests -k "air_servo or guard_thresholds or air_sim" -q`
Expected: FAIL —— `AttributeError: module 'gait_export' has no attribute 'air_servo_sim'`

- [ ] **Step 3: 實作 `air_servo_sim`**

加到 `gait_export.py`（頂端補 `import mujoco`）：

```python
AIR_SIM_GRID = ((20.0, 0.7), (40.0, 1.0))     # (kp, kd)。20/0.7 是原廠站立實測值
AIR_SIM_SCALES = (0.25, 0.5, 1.0)


def air_servo_sim(m, q_mjcf, kp, kd, time_scale, settle_sec=1.0):
    """吊掛空跑的直接模擬：機身固定、無接觸、位置伺服 kp/kd。

    量的是【實際會發生的事】——致動器力矩、追蹤誤差、關節速度峰值。

    ⚠️ 不要改用 mj_inverse。逆動力學會得到 45.8 N·m（超過 URDF 的 28 N·m
       effort 上限），因為 duty_remap 在擺動→站立交界處讓 dθ/dt 差 4 倍，
       指令軌跡有速度折點，二階差分的加速度會爆掉。那描述的是「完美追蹤
       所需的力矩」，而完美追蹤本來就不會發生——位置伺服必然落後。
    """
    m2 = mujoco.MjModel.from_xml_path(d1_model.SCENE)
    m2.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for a in range(m2.nu):                     # 覆寫位置伺服增益
        m2.actuator_gainprm[a][0] = kp
        m2.actuator_biasprm[a][1] = -kp
        m2.actuator_biasprm[a][2] = -kd

    d = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d, 0)
    base_qpos = d.qpos[:7].copy()              # 吊具：機身固定在初始位姿
    lo, hi = m2.actuator_ctrlrange[:, 0], m2.actuator_ctrlrange[:, 1]
    idx, vidx = d1_model.LEG_QPOS_IDX, d1_model.LEG_QVEL_IDX
    dt = d1_model.CTRL_DT

    def sample(u):
        x = np.clip(u / dt, 0.0, len(q_mjcf) - 1)
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, len(q_mjcf) - 1)
        w = x - i0
        return q_mjcf[i0] * (1 - w) + q_mjcf[i1] * w

    def pin():
        d.qpos[:7] = base_qpos
        d.qvel[:6] = 0.0

    for _ in range(int(settle_sec / m2.opt.timestep)):    # 先靜置到第 0 幀
        d.ctrl[:] = np.clip(q_mjcf[0], lo, hi)
        pin()
        mujoco.mj_step(m2, d)

    wall = (len(q_mjcf) - 1) * dt / time_scale
    tau_pk = vel_pk = 0.0
    errs = []
    for k in range(int(wall / m2.opt.timestep)):
        tgt = sample(k * m2.opt.timestep * time_scale)
        d.ctrl[:] = np.clip(tgt, lo, hi)
        pin()
        mujoco.mj_step(m2, d)
        tau_pk = max(tau_pk, float(np.abs(d.actuator_force).max()))
        vel_pk = max(vel_pk, float(np.abs(d.qvel[vidx]).max()))
        errs.append(np.abs(d.qpos[idx] - tgt))
    errs = np.asarray(errs)
    return {"tau_peak": tau_pk,
            "err_peak_deg": float(np.degrees(errs.max())),
            "err_rms_deg": float(np.degrees(np.sqrt((errs ** 2).mean()))),
            "vel_peak": vel_pk}


def air_sim_table(m, q_mjcf):
    """對 AIR_SIM_GRID × AIR_SIM_SCALES 全部算一遍。寫進 npz 的 meta。"""
    table = {}
    for kp, kd in AIR_SIM_GRID:
        key = f"{kp}/{kd}"
        table[key] = {}
        for s in AIR_SIM_SCALES:
            r = air_servo_sim(m, q_mjcf, kp, kd, s)
            table[key][str(s)] = r
            print(f"  kp={kp} kd={kd} {s:>5.2f}×  力矩 {r['tau_peak']:6.2f} N·m  "
                  f"誤差峰值 {r['err_peak_deg']:6.2f}°  RMS {r['err_rms_deg']:5.2f}°  "
                  f"速度 {r['vel_peak']:5.2f} rad/s")
    return table
```

- [ ] **Step 4: 把表寫進 npz**

在 `export()` 裡，`meta` 組出來之前插入：

```python
    print("[空中模擬] 機身固定、無接觸，量實際力矩與追蹤誤差：")
    air = air_sim_table(m, q_mjcf)
```

並把 `air_sim` 加進 `meta`：

```python
    meta = {"gait": GAIT, "g_c": float(g_c), "omega": cfg["omega"],
            "mu_x": cfg["mu_x"], "mu_y": W.MU_Y, "x_off": cfg["x_off"],
            "duty": cfg["duty"], "ctrl_dt": d1_model.CTRL_DT, "secs": float(secs),
            "calib_hash": calib_hash(), "air_sim": air, **stats}
```

- [ ] **Step 5: 改寫 L7 的 `guard_thresholds`**

在 `L7_gait_shm.py` 新增（Task 5 刻意沒寫這塊 —— 它需要本任務才算得出來的模擬表）：

```python
# 保護門檻的安全係數。門檻 = 模擬峰值 × 係數。
# 模擬是「一切正常時會發生什麼」，超過它兩倍就不是正常了。
VEL_SAFETY = 1.5
TORQUE_SAFETY = 2.0


def guard_thresholds(meta, kp, kd, time_scale):
    """回傳 (torque_abort, vel_abort)，查 npz 內建的空中模擬表。

    ⚠️ 沒有 fallback。查不到就拒跑——門檻不能用猜的。
       L4 的 8.0 N·m 更不能照搬：原廠增益 1.0× 的模擬力矩峰值就有 10.18 N·m。
    """
    table = meta.get("air_sim", {})
    key, skey = f"{kp}/{kd}", str(time_scale)
    if key not in table or skey not in table[key]:
        print(f"✗ 軌跡檔沒有 kp={kp}/kd={kd} @ {time_scale}× 的模擬結果，拒絕執行。")
        print(f"  檔內有的組合：{ {k: sorted(v) for k, v in table.items()} }")
        print("  要用別的組合，先把它加進 gait_export.AIR_SIM_GRID/AIR_SIM_SCALES 再重產。")
        sys.exit(1)
    e = table[key][skey]
    return float(e["tau_peak"] * TORQUE_SAFETY), float(e["vel_peak"] * VEL_SAFETY)
```

本任務只提供這個函式；使用它的 `run_gait` 在 Task 7 才寫。`run_jog` 也在 Task 7，它**不查表** —— jog 是 ±0.10 rad 的慢速微動，用 L4 的保守值 8.0 就對。

- [ ] **Step 6: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests -q`
Expected: 全綠

- [ ] **Step 7: 重產正式軌跡檔（現在含模擬表）**

Run:
```bash
conda run --no-capture-output -n rbtdog python task6/inference/gait_export.py \
  --export task6/weights/gait_walk_stable.npz --secs 20
```
Expected: 印出六行模擬結果，`kp=20.0 kd=0.7  1.00×` 那行力矩約 10.2 N·m、誤差峰值約 39°。

- [ ] **Step 8: Commit**

```bash
git add task6/inference/gait_export.py task6/realbot/L7_gait_shm.py task6/tests/ task6/weights/gait_walk_stable.npz
git commit -m "feat(task6): 保護門檻改由吊掛空跑模擬得出，不用逆動力學

逆動力學給 45.8 N·m，超過 URDF 的 28——因為 duty_remap 交界處有速度
折點，二階差分爆掉，那描述的是完美追蹤所需力矩而完美追蹤不會發生。
改成機身固定、無接觸、位置伺服 kp/kd 直接跑一遍，量實際力矩與誤差。

結果：原廠增益 1.0x 力矩峰值 10.18 N·m > L4 的 8.0，照搬會誤中止。
表寫進 npz，L7 查表，查不到就拒跑。副產品是每一級的誤差預測值，
實機量到的數字可以直接對照。"
```

---

### Task 7: L7 的硬體執行路徑

**Files:**
- Modify: `task6/realbot/L7_gait_shm.py`
- Test: `task6/tests/test_L7_gait.py`

**Interfaces:**
- Consumes: Task 5 的全部、Task 1 的 `shm_common`
- Produces:
  - `jog_targets(start_q, joint_idx, amp, secs, hz) -> np.ndarray` —— `(N, 3)` 三角波
  - `write_log(path, log, meta) -> None`（在 `shm_common`）
  - log npz 契約：`t (N,)`、`cmd (N,4,3)`、`p (N,4,3)`、`v (N,4,3)`、`tau (N,4,3)`、`overrun (N,) bool`、`meta_json`
  - `run_gait(d, traj, meta, ...) -> bool`
  - `run_jog(d, ...) -> bool`

- [ ] **Step 1: 寫失敗的測試（純函式部分）**

追加到 `task6/tests/test_L7_gait.py`：

```python
def test_jog_targets_is_a_triangle_starting_and_ending_at_rest():
    start = np.array([0.5, -2.2, 1.25])
    q = L7.jog_targets(start, joint_idx=1, amp=0.10, secs=4.0, hz=500)
    assert q.shape == (2000, 3)
    assert q[0] == pytest.approx(start)
    assert q[-1] == pytest.approx(start, abs=1e-6)
    # 只動指定的那一軸
    assert np.ptp(q[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(q[:, 2]) == pytest.approx(0.0, abs=1e-12)
    # 幅度剛好 ±amp，兩個來回
    assert q[:, 1].max() == pytest.approx(start[1] + 0.10, abs=1e-3)
    assert q[:, 1].min() == pytest.approx(start[1] - 0.10, abs=1e-3)


def test_jog_targets_never_steps_more_than_a_safe_increment():
    """jog 是用來驗號的，不能自己變成危險動作。"""
    start = np.zeros(3)
    q = L7.jog_targets(start, joint_idx=1, amp=0.10, secs=4.0, hz=500)
    assert np.abs(np.diff(q, axis=0)).max() < 0.002


def test_write_and_read_log_roundtrip(tmp_path):
    import shm_common as SC
    n = 7
    log = {"t": np.arange(n) * SC.DT,
           "cmd": np.zeros((n, 4, 3)), "p": np.ones((n, 4, 3)),
           "v": np.zeros((n, 4, 3)), "tau": np.zeros((n, 4, 3)),
           "overrun": np.zeros(n, dtype=bool)}
    path = tmp_path / "log.npz"
    SC.write_log(path, log, meta={"mode": "gait", "time_scale": 0.25})
    z = np.load(path, allow_pickle=False)
    assert set(z.files) == {"t", "cmd", "p", "v", "tau", "overrun", "meta_json"}
    assert z["p"].shape == (n, 4, 3)
    assert z["overrun"].dtype == bool
    import json
    assert json.loads(str(z["meta_json"]))["time_scale"] == 0.25
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_L7_gait.py -k "jog or log" -q`
Expected: FAIL —— `AttributeError: module 'L7_gait_shm' has no attribute 'jog_targets'`

- [ ] **Step 3: 在 `shm_common.py` 加 log 寫入**

```python
def write_log(path, log, meta):
    """把一次執行的 cmd/state 記錄寫成 npz。由 gait_export --analyze 讀。

    欄位契約（改動要同步改 gait_export 的讀取端與 test_write_and_read_log_roundtrip）：
      t (N,)  cmd (N,4,3)  p (N,4,3)  v (N,4,3)  tau (N,4,3)  overrun (N,) bool
    索引是 SHM 腿序 (0=FR 1=FL 2=RR 3=RL)、關節序 (abad, hip, knee)。
    """
    import json
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
             **{k: np.asarray(v) for k, v in log.items()})
    print(f"[記錄] {path}  {len(log['t'])} 筆")
```

頂端補 `from pathlib import Path`。

> ⚠️ `shm_common` 現在要 `import numpy`。狗上有 numpy，不違反依賴限制；但 `write_log` 是唯一用到它的地方，import 放在函式內，讓沒有 numpy 的環境仍能載入模組跑其他功能。

- [ ] **Step 4: 實作 `jog_targets` 與執行迴圈**

加到 `L7_gait_shm.py`：

```python
def jog_targets(start_q, joint_idx, amp, secs, hz=SC.CTRL_HZ):
    """單關節三角波，兩個來回，起點與終點都回到 start_q。

    用來確認 calib_map 的正負號：人眼看腿往哪邊動，對照 MJCF 的正向定義
    （+knee→伸直、+abad→外張、+hip→後擺）。號反了就停在這關修映射。

    amp 刻意小（0.10 rad = 5.7°）：驗號只需要看得出方向，不需要大動作。
    """
    n = int(secs * hz)
    ph = np.arange(n) / n * 2.0                      # 0..2，兩個來回
    tri = np.where(ph < 0.5, ph * 2,
                   np.where(ph < 1.5, 2 - ph * 2, ph * 2 - 4))
    q = np.tile(np.asarray(start_q, dtype=float), (n, 1))
    q[:, joint_idx] += amp * tri
    return q


def _stream(d, targets_iter, active_legs, kp, kd, torque_abort, vel_abort,
            log, dry, label):
    """核心串流迴圈：每個 500 Hz 週期寫一組目標、檢查保護、記錄 state。

    回傳 (ok, 原因)。ok=False 時呼叫端負責卸力。

    ⚠️ log 的時間戳用 len(log["t"]) 而非本次呼叫的迴圈索引 —— 一次執行會分成
       接住／到起始姿／播放／回站姿四段，每段各呼叫一次本函式。用迴圈索引的話
       時間戳會每段從 0 重來，事後分析對齊指令時會整個錯位。
    """
    import time
    for k, tgt in enumerate(targets_iter):
        t_start = time.monotonic()
        if not dry:
            SC.zero_all(d)
            for i in active_legs:
                SC.set_leg_position(d, i, tgt[i][0], tgt[i][1], tgt[i][2], kp, kd)
            SC.publish(d)
            ok, why = SC.check_guards(d, active_legs, torque_abort, vel_abort)
            if not ok:
                return False, why
            log["t"].append(len(log["t"]) * SC.DT)
            log["cmd"].append(np.asarray(tgt, dtype=float).copy())
            log["p"].append(np.array([[getattr(d.state.legs[i], jn).p
                                       for jn in ("abad", "hip", "knee")]
                                      for i in range(4)]))
            log["v"].append(np.array([[getattr(d.state.legs[i], jn).v
                                       for jn in ("abad", "hip", "knee")]
                                      for i in range(4)]))
            log["tau"].append(np.array([[getattr(d.state.legs[i], jn).t
                                         for jn in ("abad", "hip", "knee")]
                                        for i in range(4)]))
            # 超時要記錄：Python 在 RK3588 上被 GC 打斷是正常的，但不記的話
            # 超時造成的軌跡失真會被誤讀成追蹤誤差。
            spent = time.monotonic() - t_start
            log["overrun"].append(spent > SC.DT)
            if spent < SC.DT:
                time.sleep(SC.DT - spent)
        elif k % (SC.CTRL_HZ // 2) == 0:
            print(f"  [{label}] t={k * SC.DT:5.2f}s  "
                  f"leg{active_legs[0]} 目標 " +
                  " ".join(f"{v:+.3f}" for v in tgt[active_legs[0]]))
    return True, ""
```

`ramp` 與主流程（`run_gait`）：

```python
def _ramp_frames(a, b, secs, hz=SC.CTRL_HZ):
    """從姿勢 a 線性內插到姿勢 b，兩者都是 (4,3)。回傳 (N,4,3)。"""
    n = int(secs * hz)
    w = np.linspace(0.0, 1.0, n)[:, None, None]
    return np.asarray(a)[None] * (1 - w) + np.asarray(b)[None] * w


def run_gait(d, traj, meta, active_legs, time_scale, kp, kd, dry, log_path):
    """catch → ramp 到第 0 幀 → 播放 → ramp 回站姿 → 卸力。"""
    torque_abort, vel_abort = guard_thresholds(meta, kp, kd, time_scale)
    pred = meta["air_sim"][f"{kp}/{kd}"][str(time_scale)]
    print(f"\n[*] 保護門檻：力矩 {torque_abort:.2f} N·m、速度 {vel_abort:.2f} rad/s")
    print(f"[*] 模擬預測：誤差峰值 {pred['err_peak_deg']:.2f}°、"
          f"RMS {pred['err_rms_deg']:.2f}° —— 實機量到的值拿來跟這個比")

    stand = np.array([[SC.POSE_STAND[i][jn] for jn in ("abad", "hip", "knee")]
                      for i in range(4)])
    if dry:
        init = stand.copy()
        print("[dry-run] 假設起點為站姿（真機會讀 state.legs[*]）")
    else:
        ok, trans = SC.preflight_mc_stopped(d)
        if not ok:
            print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。先 SIGSTOP mc_ctrl。")
            return False
        SC.report_legs(d, active_legs)
        ok, problems = SC.preflight_motors_healthy(d, active_legs)
        if not ok:
            print(f"\n✗ 中止：被驅動的腿有 {len(problems)} 個馬達問題，拒絕寫入 ——")
            for p in problems:
                print(f"    • {p}")
            return False
        init = np.array([SC.read_leg_q(d, i) for i in range(4)])

    log = {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun")}
    frame0 = traj[0]
    ramp_sec = max(RAMP_MIN_SEC, meta["start_offset_from_stand"] / 0.25)
    u = playback_times(len(traj), meta["ctrl_dt"], time_scale)
    print(f"[*] 播放 {len(traj)} 幀 @ {time_scale}×  "
          f"→ {u[-1] / time_scale:.1f}s 牆鐘、{len(u)} 個控制週期")

    def stage(label, frames, kp_seq=None):
        """跑一段。kp_seq 給定時逐幀套用不同增益（接住段用）。回傳 True/False。"""
        print(f"\n[*] {label}")
        if kp_seq is None:
            ok, why = _stream(d, frames, active_legs, kp, kd,
                              torque_abort, vel_abort, log, dry, label)
        else:
            ok, why = True, ""
            for r in kp_seq:
                ok, why = _stream(d, frames[:1], active_legs, kp * r, kd * r,
                                  torque_abort, vel_abort, log, dry, label)
                if not ok:
                    break
        if not ok:
            print(f"⚠️ 保護觸發：{why} → 卸力中止")
            SC.passive_stop(d, active_legs, 300, STOP_KD)
        return ok

    # 接住：p_des 固定在當前實際角度，kp/kd 由 0 平滑升到設定值。
    # 凍結 mc_ctrl 後腿會因重力垂下，先用漸入增益接住，避免力矩突跳。
    n_catch = int(CATCH_SEC * SC.CTRL_HZ)
    if not stage("接住", init[None], np.linspace(0.0, 1.0, n_catch + 1)):
        return False
    if not stage("到起始姿", _ramp_frames(init, frame0, ramp_sec)):
        return False
    if not stage("播放步態", sample_at(traj, meta["ctrl_dt"], u)):
        return False
    if not stage("回站姿", _ramp_frames(traj[-1], stand, RAMP_MIN_SEC)):
        return False

    if not dry:
        SC.passive_stop(d, active_legs, 800, STOP_KD)
        n_over = int(np.sum(log["overrun"]))
        print(f"[*] 完成。500 Hz 週期超時 {n_over} / {len(log['t'])} "
              f"（{100.0 * n_over / max(1, len(log['t'])):.2f}%）")
        SC.write_log(log_path, {k: np.asarray(v) for k, v in log.items()},
                     meta={"mode": "gait", "time_scale": time_scale,
                           "kp": kp, "kd": kd,
                           "active_legs": list(active_legs), **meta})
    return True
```

- [ ] **Step 5: 實作 CLI 與 signal handler**

```python
def main():
    ap = argparse.ArgumentParser(description="D1 EDU 輪足：步態串流（吊掛空跑用）")
    ap.add_argument("--mode", choices=("jog", "leg", "gait"), required=True)
    ap.add_argument("--traj", default=None, help="gait_export 產生的 npz（leg/gait 模式必填）")
    ap.add_argument("--source", choices=("file", "live"), default="file")
    ap.add_argument("--time-scale", type=float, default=0.25, dest="time_scale",
                    help="播放倍率。0.25=四分之一速（預設，先慢再快）。file/live 皆適用")
    ap.add_argument("--secs", type=float, default=5.0, help="播放秒數（從軌跡頭開始）")
    ap.add_argument("--kp", type=float, default=LEG_KP)
    ap.add_argument("--kd", type=float, default=LEG_KD)
    ap.add_argument("--skip-legs", default="2",
                    help="不驅動的腿（0=FR 1=FL 2=RR 3=RL）。預設 2 —— RR 整條已失聯")
    ap.add_argument("--only-leg", type=int, default=None,
                    help="leg 模式：只驅動這一條（SHM 腿序）")
    ap.add_argument("--jog-leg", type=int, default=0)
    ap.add_argument("--jog-joint", choices=("abad", "hip", "knee"), default="hip")
    ap.add_argument("--log", default="l7_log.npz")
    ap.add_argument("--confirm", action="store_true", help="真的驅動硬體")
    args = ap.parse_args()

    if args.mode in ("leg", "gait") and not args.traj:
        ap.error("--mode leg/gait 需要 --traj")

    try:
        skip = {int(x) for x in args.skip_legs.split(",") if x.strip() != ""}
    except ValueError:
        print(f"✗ --skip-legs 格式錯誤：{args.skip_legs!r}")
        sys.exit(1)
    if not skip <= {0, 1, 2, 3}:
        print(f"✗ --skip-legs 只能是 0~3，收到 {sorted(skip)}")
        sys.exit(1)
    active = tuple(i for i in range(4) if i not in skip)
    if args.mode == "leg":
        if args.only_leg is None:
            ap.error("--mode leg 需要 --only-leg")
        if args.only_leg in skip:
            ap.error(f"--only-leg {args.only_leg} 同時被 --skip-legs 排除了")
        active = (args.only_leg,)
    if not active:
        print("✗ 四條腿都被跳過了，沒事可做。")
        sys.exit(1)

    SC.check_struct_size()

    traj = meta = None
    if args.traj:
        if args.source == "file":
            traj, meta = load_trajectory(args.traj)
        else:
            _, meta = load_trajectory(args.traj)
            traj = live_trajectory(args.traj, meta["secs"])
        n = int(args.secs / meta["ctrl_dt"])
        traj = traj[:n]

    print(f"\n[*] 驅動的腿：{', '.join(f'leg{i}({SC.LEGNAME[i]})' for i in active)}")
    skipped = [i for i in range(4) if i not in active]
    if skipped:
        print(f"[*] ⚠️ 跳過的腿：{', '.join(f'leg{i}({SC.LEGNAME[i]})' for i in skipped)}"
              f" —— 全程零增益，完全不出力")

    if not args.confirm:
        print("=" * 66)
        print("DRY-RUN：不開啟、不寫入共享記憶體，只印出動作計畫。")
        print("要真的驅動硬體請加 --confirm（且需 sudo）。")
        print("=" * 66)
        if args.mode == "gait":
            run_gait(None, traj, meta, active, args.time_scale,
                     args.kp, args.kd, dry=True, log_path=args.log)
        print("\n⚠️ 跑真機前必讀：狗要吊掛、四腳離地、mc_ctrl 已 SIGSTOP、estop 隨手可按。")
        return

    print("=" * 66)
    print("⚠️ 真機模式：即將驅動【腿關節】。確認：狗已吊掛四腳離地、mc_ctrl 已停。")
    print("=" * 66)
    if __import__("os").geteuid() != 0:
        print("✗ 需要 root：請用 sudo 執行。")
        sys.exit(1)
    try:
        d, _buf = SC.open_shm()
    except FileNotFoundError:
        print(f"✗ 找不到 {SC.SHM_PATH}（機器人運控沒起來？）")
        sys.exit(1)
    except PermissionError:
        print("✗ 權限不足：請用 sudo。")
        sys.exit(1)

    try:
        if args.mode == "gait" or args.mode == "leg":
            run_gait(d, traj, meta, active, args.time_scale,
                     args.kp, args.kd, dry=False, log_path=args.log)
        else:
            run_jog(d, active[0], args.jog_joint, args.kp, args.kd, args.log)
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C → 卸力收尾")
        SC.passive_stop(d, active, 800, STOP_KD)
    finally:
        SC.zero_all(d)
        SC.publish(d)
        print("[*] 已歸零收尾，watchdog 兜底。測完 SIGCONT 解凍 mc_ctrl 還原。")


if __name__ == "__main__":
    main()
```

`run_jog`：

```python
def run_jog(d, leg, joint_name, kp, kd, log_path):
    """單關節微動驗號。只驅動一條腿的一個關節。"""
    ji = ("abad", "hip", "knee").index(joint_name)
    ok, trans = SC.preflight_mc_stopped(d)
    if not ok:
        print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。")
        return False
    SC.report_legs(d, (leg,))
    ok, problems = SC.preflight_motors_healthy(d, (leg,))
    if not ok:
        for p in problems:
            print(f"    • {p}")
        return False

    start = np.array(SC.read_leg_q(d, leg))
    print(f"\n[*] jog：leg{leg}({SC.LEGNAME[leg]}).{joint_name} "
          f"起點 {start[ji]:+.4f} rad，±0.10 rad 兩個來回")
    print("    ⚠️ 盯著腿看。對照 MJCF 正向定義：+knee→伸直、+abad→外張、+hip→後擺。")
    print("    方向不符就停下來修 calib_map，不要往下走。")

    frames = jog_targets(start, ji, amp=0.10, secs=8.0)
    full = np.tile(start, (len(frames), 4, 1))
    full[:, leg, :] = frames
    log = {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun")}
    # jog 不查空中模擬表：±0.10 rad 的慢速微動，用 L4 的保守值就對。
    ok, why = _stream(d, full, (leg,), kp, kd,
                      torque_abort=8.0, vel_abort=1.0,
                      log=log, dry=False, label="jog")
    if not ok:
        print(f"⚠️ 保護觸發：{why} → 卸力中止")
    SC.passive_stop(d, (leg,), 800, STOP_KD)
    SC.write_log(log_path, {k: np.asarray(v) for k, v in log.items()},
                 meta={"mode": "jog", "leg": leg, "joint": joint_name,
                       "kp": kp, "kd": kd, "start": start.tolist()})
    return ok
```

- [ ] **Step 6: 跑測試與 dry-run**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests -q`
Expected: 全綠

Run:
```bash
cd /home/huang/rbtdog_sim/task6/realbot
python3 L7_gait_shm.py --mode gait --traj ../weights/gait_walk_stable.npz \
  --time-scale 0.25 --secs 5
```
Expected: 印出保護門檻（速度約 `13.5 × 0.25 × 1.5 ≈ 5.06` rad/s）、四個階段的計畫、以及吊掛提醒。不得有例外。

- [ ] **Step 7: 確認狗上依賴仍乾淨**

Run:
```bash
cd /home/huang/rbtdog_sim/task6/realbot
grep -nE "^\s*(import|from)\s+(mujoco|torch|scipy|onnx)" shm_common.py L7_gait_shm.py \
  && echo "✗ 違反" || echo "✓ OK"
```
Expected: `✓ OK`

- [ ] **Step 8: Commit**

```bash
git add task6/realbot/L7_gait_shm.py task6/realbot/shm_common.py task6/tests/test_L7_gait.py
git commit -m "feat(task6): L7 硬體執行路徑——jog/leg/gait 三模式

沿用 L4 已實機驗證的骨架（兩道預檢、catch 漸入、卸力收尾），
只有軌跡來源不同。保護門檻由 npz meta 算出而非硬編。
500Hz 週期超時逐筆記錄——不記的話超時造成的失真會被誤讀成追蹤誤差。
--time-scale 預設 0.25，先慢再快。"
```

---

### Task 8: `--analyze` 追蹤誤差分析

**Files:**
- Modify: `task6/inference/gait_export.py`
- Test: `task6/tests/test_gait_export.py`

**Interfaces:**
- Consumes: Task 7 的 log npz 契約
- Produces: `analyze(log_path) -> dict` —— 逐軸 `rms_deg`、`max_deg`、`lag_ms`，以及 `overrun_pct`

- [ ] **Step 1: 寫失敗的測試**

追加到 `task6/tests/test_gait_export.py`：

```python
def _synthetic_log(tmp_path, lag_steps=0, err_rad=0.0):
    """造一份假的 state log：實際角 = 指令角延遲 lag_steps 再加固定偏差。"""
    import json
    import numpy as np
    n, dt = 2000, 1.0 / 500
    t = np.arange(n) * dt
    cmd = np.zeros((n, 4, 3))
    cmd[:, :, 1] = np.sin(2 * np.pi * 1.4 * t)[:, None]
    p = np.roll(cmd, lag_steps, axis=0) + err_rad
    p[:lag_steps] = cmd[:lag_steps]
    path = tmp_path / "log.npz"
    np.savez(path, t=t, cmd=cmd, p=p, v=np.zeros((n, 4, 3)),
             tau=np.zeros((n, 4, 3)), overrun=np.zeros(n, dtype=bool),
             meta_json=np.array(json.dumps({"mode": "gait", "time_scale": 1.0,
                                            "active_legs": [0, 1, 3]})))
    return path


def test_analyze_reports_zero_error_for_perfect_tracking(tmp_path):
    r = GE.analyze(_synthetic_log(tmp_path))
    assert r["axes"][(0, "hip")]["rms_deg"] == pytest.approx(0.0, abs=1e-9)
    assert r["overrun_pct"] == pytest.approx(0.0)


def test_analyze_recovers_a_known_constant_offset(tmp_path):
    r = GE.analyze(_synthetic_log(tmp_path, err_rad=np.radians(3.0)))
    assert r["axes"][(0, "hip")]["rms_deg"] == pytest.approx(3.0, abs=0.01)


def test_analyze_recovers_a_known_lag(tmp_path):
    """延遲 10 個 500 Hz 週期 = 20 ms。"""
    r = GE.analyze(_synthetic_log(tmp_path, lag_steps=10))
    assert r["axes"][(0, "hip")]["lag_ms"] == pytest.approx(20.0, abs=2.0)


def test_analyze_skips_legs_that_were_not_driven(tmp_path):
    """RR 沒被驅動，它的誤差是無意義的，不該出現在報告裡。"""
    r = GE.analyze(_synthetic_log(tmp_path))
    assert (2, "hip") not in r["axes"]
    assert (0, "hip") in r["axes"]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests/test_gait_export.py -k analyze -q`
Expected: FAIL —— `AttributeError: module 'gait_export' has no attribute 'analyze'`

- [ ] **Step 3: 實作**

加到 `gait_export.py`：

```python
def analyze(log_path):
    """讀實機 log，算逐軸追蹤誤差。回傳 {'axes': {...}, 'overrun_pct': float}。

    相位延遲用互相關求：把實際角相對指令角平移，找誤差平方和最小的位移量。
    只分析被驅動的腿——跳過的腿全程零增益，它的「誤差」沒有意義。
    """
    from shm_common import LEGNAME
    z = np.load(log_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    cmd, p, t = z["cmd"], z["p"], z["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 500
    active = meta.get("active_legs", [0, 1, 2, 3])

    axes, rows = {}, []
    for leg in active:
        for j, jn in enumerate(JN):
            c, a = cmd[:, leg, j], p[:, leg, j]
            err = a - c
            rms = float(np.degrees(np.sqrt((err ** 2).mean())))
            mx = float(np.degrees(np.abs(err).max()))
            # 相位延遲：平移實際角，找誤差最小的位移
            best, best_k = np.inf, 0
            for k in range(0, min(200, len(c) // 4)):
                d_ = (a[k:] - c[:len(c) - k]) if k else (a - c)
                s = float((d_ ** 2).mean())
                if s < best:
                    best, best_k = s, k
            axes[(leg, jn)] = {"rms_deg": rms, "max_deg": mx,
                               "lag_ms": best_k * dt * 1000.0}
            rows.append((leg, jn, rms, mx, best_k * dt * 1000.0))

    over = float(100.0 * np.sum(z["overrun"]) / max(1, len(z["overrun"])))
    print(f"\n[分析] {log_path}  模式={meta.get('mode')}  "
          f"time_scale={meta.get('time_scale')}  kp={meta.get('kp')}")
    print(f"{'軸':<12} {'RMS(°)':>9} {'最大(°)':>9} {'延遲(ms)':>10}")
    for leg, jn, rms, mx, lag in rows:
        print(f"leg{leg}({LEGNAME[leg]}).{jn:<5} {rms:>9.2f} {mx:>9.2f} {lag:>10.1f}")
    print(f"\n500 Hz 週期超時 {over:.2f}%"
          + ("  ⚠️ 超過 1% 時追蹤誤差的解讀要打折" if over > 1.0 else ""))
    return {"axes": axes, "overrun_pct": over, "meta": meta}
```

CLI 加：

```python
    ap.add_argument("--analyze", metavar="LOGPATH", default=None,
                    help="分析實機 log，輸出逐軸追蹤誤差")
```

```python
    if args.analyze:
        analyze(args.analyze)
```

並把 `ap.error("要 --sweep 或 --export 其中之一")` 改成同時接受 `--analyze`：

```python
    if not any((args.sweep, args.export, args.analyze)):
        ap.error("要 --sweep / --export / --analyze 其中之一")
```

`--analyze` 不需要建 model，所以把 `model = d1_model.make_model()` 移到需要它的分支裡。

- [ ] **Step 4: 跑測試確認通過**

Run: `conda run --no-capture-output -n rbtdog python -m pytest task6/tests -q`
Expected: 全綠

- [ ] **Step 5: Commit**

```bash
git add task6/inference/gait_export.py task6/tests/test_gait_export.py
git commit -m "feat(task6): --analyze 逐軸追蹤誤差

RMS/最大誤差/互相關求相位延遲，只分析被驅動的腿。
超時率超過 1% 會提醒追蹤誤差的解讀要打折。
測試用合成 log 反推已知的偏差與延遲，不需要實機。"
```

---

### Task 9: 操作文件與關卡紀錄表

**Files:**
- Create: `task6/docs/L7_吊掛空跑操作手冊.md`
- Modify: `task6/realbot/README.md`

- [ ] **Step 1: 寫操作手冊**

建立 `task6/docs/L7_吊掛空跑操作手冊.md`，內容必須包含：

**上機前檢查清單**（逐項打勾）
1. 狗吊起來，四腳離地，用手撥動確認完全不觸地
2. RR（右後腿）垂下來不會撞到吊具、其他腿、或地面
3. `mc_ctrl` 已 SIGSTOP（`sudo kill -STOP $(pgrep mc_ctrl)`）
4. `estop.sh` 開在另一個終端，隨手可按
5. 車載電腦有 numpy：`python3 -c "import numpy; print(numpy.__version__)"`
6. 軌跡檔已複製到狗上，且 `calib_hash` 與狗上的 `calib_map` 一致

**四道關卡與往下走的條件**

| 關卡 | 指令 | 通過條件 |
|---|---|---|
| G0 離線 | `pytest task6/tests -q` + `gait_export.py --export` | 全綠、npz 產出 |
| G1 jog | `sudo python3 L7_gait_shm.py --mode jog --jog-leg 0 --jog-joint hip --confirm` | 方向與 MJCF 正向定義一致 |
| G2 leg | `--mode leg --only-leg 1 --time-scale 0.25 --secs 5 --confirm` | 不中止 |
| G3 gait | `--mode gait --skip-legs 2 --time-scale 0.25 --secs 5 --confirm` | 不中止 |
| G3-live | 同上加 `--source live` | 不中止，且追蹤誤差與 `--source file` 同一級的數字一致 |

`--source file` 是播放寫死的軌跡檔，`--source live` 是狗上自己算 CPG + IK。兩者在任何倍速下都應該送出相同的指令，所以 G3-live 是對「狗上算得對不對、算得夠不夠快」的獨立驗證 —— 特別看 500 Hz 週期超時率有沒有比 file 模式高。

G1 要對 `leg0` 的 abad / hip / knee 三軸各跑一次。**hip 是重點** —— `calib_map.py` 明寫它的號是暫定的。

MJCF 正向定義（人眼比對用）：`+knee → 伸直`、`+abad → 外張`、`+hip → 後擺`。

**號驗錯了怎麼辦**：停在 G1，改 `calib_map.CALIB` 對應項的 sign，重跑 `gait_export.py --export` 產新的 npz（`calib_hash` 會變，舊 npz 會被 L7 拒絕，這是刻意的），然後重跑 G1。

**倍速階梯**：每一關都先 `--time-scale 0.25`，過了才 0.5，再 1.0。每一級跑完用 `--analyze` 看追蹤誤差再決定要不要往上加。

**收尾**：測完 `sudo kill -CONT $(pgrep mc_ctrl)` 解凍運控。

**每一級的預測值**（從 npz 的 `air_sim` 表抄，原廠增益 kp=20/kd=0.7）：

| 倍速 | 力矩峰值 | 誤差峰值 | 誤差 RMS | 速度峰值 |
|---|---|---|---|---|
| 0.25× | 2.33 N·m | 10.89° | 2.33° | 3.62 rad/s |
| 0.50× | 5.28 N·m | 23.30° | 5.02° | 7.47 rad/s |
| 1.00× | 10.18 N·m | 39.20° | 9.30° | 12.96 rad/s |

實機量到的誤差跟這一欄比。**顯著大於預測**代表實機比模型軟（增益沒真的生效、或馬達到了速度上限）；**顯著小於預測**代表軌跡沒真的送出去，要先確認寫入有生效再高興。

**已知限制**（從設計文件 §11 抄過來，一字不改）：吊掛結果不能外推到落地；上機跑的是 `G_C=0.110` 不是影片版的 0.12；實機馬達速度上限未知，1.0× 可能達不到；1.0× 的預測誤差峰值 39.20°，實機動作會與影片有明顯差異，這是預期內的。

**後續（等 RR 修好才能做）**：用 `L0_cmd_probe` 唯讀錄下原廠遙控走路時的指令區，取得原廠步態的關節速度與增益。那是回答「15 rad/s 合不合理」最直接的證據，也是落地階段調增益的基準。見設計文件 §10。

- [ ] **Step 2: 更新 `task6/realbot/README.md`**

在既有的 L0～L6 清單後面加 L7 與 `shm_common`：

```markdown
| `shm_common.py` | SHM 結構與安全骨架，L4/L7 共用。結構定義只能有這一份。 |
| `L7_gait_shm.py` | 步態串流（吊掛空跑）。三模式 jog/leg/gait。操作見 `docs/L7_吊掛空跑操作手冊.md`。 |
```

- [ ] **Step 3: Commit**

```bash
git add task6/docs/L7_吊掛空跑操作手冊.md task6/realbot/README.md
git commit -m "docs(task6): L7 吊掛空跑操作手冊與關卡紀錄表"
```

---

## 執行順序與阻塞點

Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9，全部可在開發機完成，不需要實機。

**兩個實機阻塞點，都在程式碼寫完之後：**

1. Task 1 改動了 L4，上機前要重跑一次 `L4 --sequence lie,stand --confirm --skip-legs 2` 確認行為不變。
2. G1～G3 需要機器可用。報修報告記錄機器開機後約 3 分鐘就會閃紅燈並 disable 全部馬達 —— 每次上機只有約 3 分鐘的窗口，所以 `--secs 5` 加 `--time-scale 0.25` 的單次執行約 20 秒是刻意的：一次開機做得完一到兩關。
