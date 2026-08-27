"""`realbot/cpg.py`（狗上純標準函式庫版）對 `inference/cpg_max.py`（numpy 版）的逐步比對。

★ 存在理由與 `test_kin.py` 相同：同一套數學有**兩份實作**，一份跑在狗上、
  一份跑在本機。兩份漂開的症狀是「模擬調好的步態上機就是不一樣」，
  而且**兩邊各自都自洽** —— 本專案「自洽但錯誤」那一類最難查的問題。

  所以這裡不驗「cpg.py 自己說得通」，而是**逐步比對另一份獨立實作**，
  而且要跑夠久 —— 相位耦合會累積，只比一步看不出偏差。

⚠️ 需要 numpy（本機端測試）。狗上不會跑這支。
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))
sys.path.insert(0, str(ROOT / "inference"))

import coord     # noqa: E402
import cpg       # noqa: E402

np = pytest.importorskip("numpy")
cpg_max = pytest.importorskip("cpg_max")
leg_kin = pytest.importorskip("leg_kin")
mmod = pytest.importorskip("max_model")

MM2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
# numpy 版的 4 維向量索引 ← SHM 腿名
K = {MM2SHM[l]: k for k, l in enumerate(mmod.LEGS)}

TOL = 1e-12


# ════════════════════════════════════════════════════════════ 常數與相位
def test_constants_have_not_drifted():
    assert cpg.MU_MIN == mmod.MU_MIN and cpg.MU_MAX == mmod.MU_MAX
    assert cpg.A_CONV == mmod.A_CONV
    assert cpg.G_P == mmod.G_P
    assert cpg.W_COUP == mmod.W_COUP
    assert cpg.N_CPG_SUB == mmod.N_CPG_SUB


def test_phase_tables_match_the_numpy_version():
    """★ 相位表是照 `max_model.LEGS`（FR,FL,RR,RL）排的，這裡改用腿名 —— 要對得上。"""
    for name, arr in (("PHASE_WALK", cpg_max.PHASE_WALK),
                      ("PHASE_TROT", cpg_max.PHASE_TROT)):
        mine = getattr(cpg, name)
        for lg, k in K.items():
            assert mine[lg] == pytest.approx(float(arr[k]), abs=1e-15), f"{name}/{lg}"


# ════════════════════════════════════════════════════════════ 振盪器
def _run_both(n_steps, omega=1.4, mu_x=1.8, mu_y=1.5, dt=None, phase="walk"):
    dt = dt if dt is not None else mmod.CTRL_DT
    ph_np = cpg_max.PHASE_WALK if phase == "walk" else cpg_max.PHASE_TROT
    ph_py = cpg.PHASE_WALK if phase == "walk" else cpg.PHASE_TROT
    step_np = cpg_max.make_cpg_step(ph_np)
    c_np = cpg_max.cpg_init(ph_np)
    step_py = cpg.make_step(ph_py)
    c_py = cpg.init(ph_py)
    mux_np, muy_np = np.full(4, mu_x), np.full(4, mu_y)
    om_np = np.full(4, omega)
    mux_py = {l: mu_x for l in cpg.LEGS}
    muy_py = {l: mu_y for l in cpg.LEGS}
    om_py = {l: omega for l in cpg.LEGS}
    worst = 0.0
    for _ in range(n_steps):
        c_np = step_np(c_np, mux_np, muy_np, om_np, dt)
        c_py = step_py(c_py, mux_py, muy_py, om_py, dt)
        for key in ("rx", "rx_d", "ry", "ry_d", "theta"):
            for lg, k in K.items():
                worst = max(worst, abs(float(c_np[key][k]) - c_py[key][lg]))
    return worst, c_np, c_py


def test_oscillator_matches_step_by_step_over_20_seconds():
    """★ 要跑夠久 —— 相位耦合會累積，只比一步看不出偏差。"""
    worst, _, _ = _run_both(int(20.0 / mmod.CTRL_DT))
    assert worst < TOL, f"1000 步後最大差 {worst:.3e}"


def test_oscillator_matches_for_trot_phase_too():
    worst, _, _ = _run_both(500, phase="trot")
    assert worst < TOL, f"trot 相位差 {worst:.3e}"


@pytest.mark.parametrize("omega,mu_x", [(0.5, 1.8), (1.4, 1.8), (2.0, 1.2)])
def test_oscillator_matches_across_parameters(omega, mu_x):
    worst, _, _ = _run_both(300, omega=omega, mu_x=mu_x)
    assert worst < TOL


def test_phase_coupling_actually_locks():
    """耦合要真的把相位拉回設定的關係 —— 否則「兩份一致」只是一起壞掉。"""
    _, c_np, c_py = _run_both(int(10.0 / mmod.CTRL_DT))
    want = cpg.PHASE_WALK
    ref = c_py["theta"]["bl"] - want["bl"]
    for lg in cpg.LEGS:
        d = (c_py["theta"][lg] - want[lg] - ref + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < 0.05, f"{lg} 相位沒鎖住（偏 {d:.3f} rad）"


# ════════════════════════════════════════════════════════════ 軌跡
def test_duty_remap_matches():
    for duty in (0.5, 0.7, 0.8, 0.85):
        for i in range(200):
            th = i / 200 * 2 * math.pi
            a = cpg.duty_remap(th, duty)
            b = float(cpg_max.duty_remap(np.array([th]), duty)[0])
            assert abs(a - b) < 1e-14, f"duty={duty} th={th}"


def test_duty_remap_is_identity_at_half():
    """duty=0.5 時應該恆等 —— 那是原公式的隱含假設，寫成測試釘住。"""
    for i in range(100):
        th = i / 100 * 2 * math.pi
        assert cpg.duty_remap(th, 0.5) == pytest.approx(th, abs=1e-12)


def _f0_ks():
    """兩邊的基準足端與膝分支（用 max_model.HOME）。"""
    f0_np = leg_kin.home_foot(mmod.HOME)
    ks_np = leg_kin.knee_sign_of(mmod.HOME)
    home_pose = {}
    for k, l in enumerate(mmod.LEGS):
        for j, kd in enumerate(coord.LEG_KINDS):
            home_pose[MM2SHM[l] + kd] = float(mmod.HOME[k, j])
    return f0_np, ks_np, cpg.home_foot(home_pose), cpg.knee_signs(home_pose)


def test_home_foot_and_knee_sign_match():
    f0_np, ks_np, f0_py, ks_py = _f0_ks()
    for lg, k in K.items():
        assert ks_py[lg] == pytest.approx(float(ks_np[k]))
        for i in range(3):
            assert f0_py[lg][i] == pytest.approx(float(f0_np[k][i]), abs=1e-12)


def test_joint_targets_match_over_a_full_gait():
    """★ 端到端：CPG 狀態 → 足端 → IK → 12 個關節角，跑滿 20 秒逐幀比。"""
    import gait_baseline as gb
    B = gb.BASELINE
    f0_np, ks_np, f0_py, ks_py = _f0_ks()
    dt = mmod.CTRL_DT
    step_np = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    c_np = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    step_py = cpg.make_step(cpg.PHASE_WALK)
    c_py = cpg.init(cpg.PHASE_WALK)
    mux_np, muy_np = np.full(4, B["mu_x"]), np.full(4, B["mu_y"])
    om_np = np.full(4, B["omega"])
    mux_py = {l: B["mu_x"] for l in cpg.LEGS}
    muy_py = {l: B["mu_y"] for l in cpg.LEGS}
    om_py = {l: B["omega"] for l in cpg.LEGS}
    names = [MM2SHM[l] + kd for l in mmod.LEGS for kd in coord.LEG_KINDS]
    worst = 0.0
    for _ in range(int(20.0 / dt)):
        q_np, n_np = cpg_max.joint_targets(c_np, f0_np, B["x_off"], B["g_c"],
                                           B["d_step"], B["d_step_y"], B["duty"],
                                           ks_np, B["z_sag"])
        q_py, n_py = cpg.joint_targets(c_py, f0_py, ks_py, B["x_off"], B["g_c"],
                                       B["d_step"], B["d_step_y"], B["duty"],
                                       B["z_sag"])
        assert n_np == n_py
        for i, nm in enumerate(names):
            worst = max(worst, abs(float(q_np[i]) - q_py[nm]))
        c_np = step_np(c_np, mux_np, muy_np, om_np, dt)
        c_py = step_py(c_py, mux_py, muy_py, om_py, dt)
    assert worst < 1e-11, f"20 秒逐幀最大差 {worst:.3e} rad"


def test_stand_targets_match():
    import gait_baseline as gb
    f0_np, ks_np, f0_py, ks_py = _f0_ks()
    names = [MM2SHM[l] + kd for l in mmod.LEGS for kd in coord.LEG_KINDS]
    for x_off in (0.0, -0.04, +0.04):
        a = cpg_max.stand_targets(ks_np, f0_np, x_off)
        b = cpg.stand_targets(f0_py, ks_py, x_off)
        for i, nm in enumerate(names):
            assert b[nm] == pytest.approx(float(a[i]), abs=1e-12), f"{x_off}/{nm}"


# ════════════════════════════════════════════════════════════ 已知的坑
def test_low_g_c_makes_the_foot_not_clear_and_that_is_documented():
    """★ 迴歸：`g_c` 小於撓度時腳不離地。2026-08-27 在 MuJoCo 又重現一次
    （g_c=0.04 + z_sag=0.0156 → 實際離地只有 4.5 mm，而狗被拖著前進 118 mm）。

    這裡只驗**指令**：擺動相的抬起量就是 `g_c + z_sag`，
    所以要判斷會不會離地，比的是它和實測撓度的大小。
    """
    f0_np, ks_np, f0_py, ks_py = _f0_ks()
    c = cpg.init(cpg.PHASE_WALK)
    # 找一個擺動相中點（sin(remap)≈1）的相位
    c["theta"] = {l: 0.5 * (1 - 0.85) * 2 * math.pi for l in cpg.LEGS}
    for g_c, z_sag in ((0.04, 0.0156), (0.12, 0.0156)):
        t = cpg.foot_targets(c, f0_py, 0.0, g_c, 0.0, 0.0, 0.85, z_sag)
        lift = t["fl"][2] - f0_py["fl"][2]
        assert lift == pytest.approx(g_c + z_sag, rel=1e-6), "抬起量應該就是 g_c+z_sag"
    # 實機在 kp=250 量到的擺動離地損失是 36 mm
    assert 0.04 + 0.0156 < 0.036 + 0.02, "g_c=0.04 的餘裕小到不該當第一趟參數"


def test_stdlib_cpg_is_fast_enough_for_50hz():
    """★ 狗上要即時算。實測 0.043 ms/週期（本機），RK3588 慢 4 倍也只有 0.9% 預算。"""
    import time
    f0_np, ks_np, f0_py, ks_py = _f0_ks()
    step = cpg.make_step(cpg.PHASE_WALK)
    c = cpg.init(cpg.PHASE_WALK)
    mux = {l: 1.8 for l in cpg.LEGS}
    muy = {l: 1.5 for l in cpg.LEGS}
    om = {l: 1.4 for l in cpg.LEGS}
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 0.3:
        cpg.joint_targets(c, f0_py, ks_py, -0.04, 0.08, 0.10, 0.12, 0.85, 0.016)
        c = step(c, mux, muy, om, mmod.CTRL_DT)
        n += 1
    ms = (time.perf_counter() - t0) / n * 1000
    # 本機 ~0.04 ms。留 10 倍餘裕當上限，避免哪天寫成 O(n²) 沒人發現。
    assert ms < 0.5, f"每個控制週期 {ms:.3f} ms —— 太慢，狗上會來不及"
