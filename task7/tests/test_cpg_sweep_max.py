"""共用模型快取與多擾動掃描器的測試。

題目全部來自這次真的踩到的兩個坑：

  1. **記憶體**：`Robot()` 原本每次重建整個 MJCF（約 1.15 GB），
     一次平行掃描就把 16 GB 的開發機 OOM 弄當機（2026-08-26）。
  2. **快取汙染**：模型改成共用之後，`--friction 0.3` 之類的覆寫如果沒有
     在下一次建構時還原，會**靜默滲進**後面每一次 rollout，
     而四個診斷指標（超限／飽和／IK縮限／相位鎖定）全都看不出來。
     ★ 這正是 task7 一路上最貴的那種 bug：自洽但錯誤。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_sweep_max as cs
import cpg_walk_max as cw
import max_model as mm


# =============================================================================
# 模型快取
# =============================================================================
def test_model_is_shared():
    """同一個進程裡只該有一份模型 —— 這條就是省下那 1.15 GB 的全部理由。"""
    assert cw.Robot().m is cw.Robot().m


def test_data_is_not_shared():
    """模型共用，但狀態不共用。兩個 Robot 共用 MjData 會讓 rollout 互相汙染。"""
    a, b = cw.Robot(), cw.Robot()
    assert a.d is not b.d


@pytest.mark.parametrize("field, idx, override", [
    ("geom_friction", None, dict(friction=0.31)),
    ("dof_frictionloss", mm.WHEEL_QVEL_IDX, dict(wheel_friction=0.77)),
    ("dof_frictionloss", mm.LEG_QVEL_IDX, dict(leg_friction=2.9)),
    ("dof_frictionloss", mm.LEG_QVEL_IDX[::3], dict(abad_friction=4.2)),
])
def test_override_does_not_leak_to_next_robot(field, idx, override):
    """★ 主要測試：覆寫過的欄位，下一個 Robot 必須拿回 MJCF 的原值。

    沒有這條，上一格掃描的摩擦會滲進下一格，掃出來的趨勢是假的。
    """
    pristine = getattr(cw.Robot().m, field).copy()
    cw.Robot(**override)                       # 汙染
    after = getattr(cw.Robot().m, field)       # 下一個 Robot
    assert np.array_equal(after, pristine), f"{field} 沒有被還原：{override}"


def test_override_actually_applies():
    """還原不能還原過頭 —— 覆寫在**當次**必須真的生效。"""
    r = cw.Robot(friction=0.31, wheel_friction=0.77, leg_friction=2.9)
    assert r.m.geom_friction[:, 0] == pytest.approx(0.31)
    assert r.m.dof_frictionloss[mm.WHEEL_QVEL_IDX] == pytest.approx(0.77)
    assert r.m.dof_frictionloss[mm.LEG_QVEL_IDX] == pytest.approx(2.9)


def test_abad_override_applies_after_leg_friction():
    """ABAD 是 leg_friction 的子集，順序反了會被整組蓋掉而**靜默失效**。"""
    r = cw.Robot(leg_friction=2.9, abad_friction=4.2)
    fl = r.m.dof_frictionloss[mm.LEG_QVEL_IDX]
    assert fl[::3] == pytest.approx(4.2), "ABAD 沒吃到覆寫"
    assert np.delete(fl, np.arange(0, 12, 3)) == pytest.approx(2.9), "HIP/KNEE 被誤改"


def test_mjcf_leg_friction_defaults_are_nonzero():
    """MJCF 自帶的實測靜摩擦不該在重構中掉回 0（那會靜默改變所有掃描結果）。"""
    m = cw.Robot().m
    assert (m.dof_frictionloss[mm.LEG_QVEL_IDX] > 0).all()
    assert (m.dof_frictionloss[mm.WHEEL_QVEL_IDX] > 0).all()


# =============================================================================
# 記憶體守衛
# =============================================================================
def test_safe_workers_is_bounded_by_memory(monkeypatch):
    """worker 數必須由**記憶體**反推，不是由核數。這台是記憶體綁死的工作。"""
    monkeypatch.setattr(cs, "mem_available_gb", lambda: 8.0)
    monkeypatch.setattr(cs.os, "cpu_count", lambda: 64)
    # (8.0 − 2.5) // 1.5 = 3
    assert cs.safe_workers(None) == 3
    assert cs.safe_workers(2) == 2, "要求值比記憶體上限小時要聽要求值"
    assert cs.safe_workers(99) == 3, "要求值比記憶體上限大時要被壓下來"


def test_safe_workers_refuses_when_memory_too_low(monkeypatch):
    """放不下一個 worker 就當場擋下來，不要開下去然後把機器弄當。"""
    monkeypatch.setattr(cs, "mem_available_gb", lambda: 3.0)
    with pytest.raises(SystemExit):
        cs.safe_workers(4)


def test_mem_available_is_plausible():
    v = cs.mem_available_gb()
    assert 0.0 < v < 4096.0


# =============================================================================
# 掃描計畫與聚合
# =============================================================================
def test_plan_expands_every_cell_to_all_seeds():
    jobs = cs.build_plan("base", secs=1.0, nseed=4)
    cells = {tuple(j["cell"]) for j in jobs}
    assert len(jobs) == len(cells) * 4
    for c in cells:
        seeds = sorted(j["seed"] for j in jobs if tuple(j["cell"]) == c)
        assert seeds == [0, 1, 2, 3]


def test_full_plan_covers_the_handoff_next_steps():
    """交接文件 §6 點名要重掃的東西，計畫裡一項都不能漏。"""
    sweeps = {j["cell"][0] for j in cs.build_plan("full", secs=1.0, nseed=1)}
    for need in ("x_off/walk", "x_off/walk_fast", "duty", "mu_y",
                 "地面摩擦", "ABAD摩擦", "腿關節摩擦", "預設值"):
        assert need in sweeps, f"掃描計畫漏了 {need}"


def _fake(**over):
    """`_agg` 需要的完整欄位。

    ⚠️ 這裡故意用 `rollout` 的**真實鍵集**當來源，而不是手寫一份。
       手寫過一次，結果 `rollout` 加了 `speed_travel` 之後測試才發現漏欄位 ——
       假資料與真資料脫節，測到的就不是同一個東西。
    """
    base = dict(fell=None, speed=0.2, speed_path=0.25, speed_travel=0.15,
                speed_net=0.14, net_disp=3.0, yaw=0.0, bounce=0.02,
                min_lift=0.09, support=3.0, pitch_mean=0.0, pitch_cycle=1.0,
                height=0.48, lateral=0.0, net_roll=0.0, path_len=5.0, dist=2.9,
                lim_pct=0.0, tau_pct=0.0, reach_pct=0.0, _secs=20.0)
    base.update(over)
    return base


def test_fake_covers_every_key_agg_reads():
    """假資料一漏欄位，_agg 就 KeyError —— 讓這件事在這裡爆，不是在掃到一半時爆。"""
    cs._agg([_fake()])          # 不丟 KeyError 就算過


def test_agg_reports_range_not_just_median():
    """★ 全距是這支存在的理由。少了它就退化成「單次數字」，等於沒改。"""
    a = cs._agg([_fake(speed_travel=0.2 + 0.1 * i, yaw=10.0 * i) for i in range(3)])
    assert a["speed_travel_rng"] == pytest.approx(0.2)
    assert a["yaw_rng"] == pytest.approx(20.0)
    assert a["n"] == 3 and a["n_fell"] == 0


def test_agg_counts_falls():
    assert cs._agg([_fake(fell=None if i else 3.2) for i in range(4)])["n_fell"] == 1


def test_agg_yaw_rate_uses_each_cell_own_secs():
    """不同 secs 的格要能互比 —— 用總偏航直接比會把 60 s 那格算成漂三倍。"""
    a = cs._agg([_fake(yaw=-30.0, _secs=60.0)])
    b = cs._agg([_fake(yaw=-10.0, _secs=20.0)])
    assert a["yaw_rate"] == pytest.approx(-0.5)
    assert b["yaw_rate"] == pytest.approx(-0.5)


def test_rollout_result_has_every_key_agg_needs():
    """★ 真正的防線：拿一次**真的** rollout 餵進 _agg。

    假資料只能證明 _agg 自洽；這條證明它跟 `rollout` 實際吐出來的東西對得上。
    """
    r = cw.rollout(gait="walk", secs=2.0, quiet=True)
    r["_secs"] = 2.0
    cs._agg([r])


def test_jitter_is_physically_negligible():
    """擾動必須小到「不可能是物理效果」，否則分不出混沌與真趨勢。

    1 皮米 ≈ 步幅的 1e-11 倍。指標若跟著變，變的原因只可能是混沌。
    """
    assert cs.JITTER < 1e-9 * cw.GAITS["walk"]["d_step"]
