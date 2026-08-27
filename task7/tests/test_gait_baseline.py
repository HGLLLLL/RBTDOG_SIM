"""基準步態凍結的回歸測試。

為什麼要有這支：掃描期間曾發生「跑到一半共用 MJCF 被別條線改掉」
（commit 533e91a 改了膝關節 range），事後只因為 `lim_pct` 這個診斷欄一直有印
才判定得出來。步態常數比 MJCF 更容易被順手改掉，而改掉之後 RL 的基準就悄悄變了。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_max  # noqa: E402
import cpg_walk_max as cw  # noqa: E402
import gait_baseline as gb  # noqa: E402
import max_model as mm  # noqa: E402


def test_baseline_values_frozen():
    """每一個數字都有判準來源，改動必須連同文件一起改。"""
    assert gb.BASELINE == {
        "gait": "walk",
        "duty": 0.80,
        "omega": 1.4,
        "mu_x": 1.80,
        "mu_y": 1.50,
        "d_step": 0.10,
        "d_step_y": 0.12,
        "x_off": -0.040,
        "g_c": 0.08,
        "z_sag": 0.0325,
        "wheel_mode": "damp",
    }


def test_cpg_walk_max_imports_gait_baseline():
    """★ 必須是**引用**，不是「數值剛好一樣的第二份抄本」。

    只比對數值是擋不住複製的：複製出來的抄本一開始當然一樣，
    問題出在半年後只改了其中一份。所以這裡直接驗模組層級的接線。
    """
    assert cw.gb is gb


def test_cpg_walk_max_uses_baseline():
    """cpg_walk_max 的 walk 必須是引用而不是另一份抄本。"""
    g = cw.GAITS["walk"]
    assert g["duty"] == gb.BASELINE["duty"]
    assert g["omega"] == gb.BASELINE["omega"]
    assert g["mu_x"] == gb.BASELINE["mu_x"]
    assert g["x_off"] == gb.BASELINE["x_off"]
    assert g["d_step"] == gb.BASELINE["d_step"]
    assert g["g_c"] == gb.BASELINE["g_c"]
    assert g["phase"] is cpg_max.PHASE_WALK
    assert cw.MU_Y == gb.BASELINE["mu_y"]
    assert cw.D_STEP_Y == gb.BASELINE["d_step_y"]


def test_z_sag_matches_static_sag():
    """z_sag 與 max_model.STATIC_SAG 綁死，不可以各寫各的。"""
    assert gb.BASELINE["z_sag"] == mm.STATIC_SAG


def test_baseline_walks_without_falling():
    """8 秒煙霧測試：不跌倒、四個診斷指標全 0。

    只跑 8 秒是因為完整驗收在 `cpg_sweep_max --plan base` 做（12 擾動 × 180 秒）。
    這裡只擋「有人把常數改到步態當場垮掉」這種等級的錯。
    """
    r = cw.rollout(gait="walk", secs=8.0, quiet=True)
    assert r["fell"] is None
    assert r["lim_pct"] == 0.0
    assert r["tau_pct"] == 0.0
    assert r["reach_pct"] == 0.0
    assert r["support"] > 3.0
    assert np.isfinite(r["speed_travel"]) and r["speed_travel"] > 0.10
