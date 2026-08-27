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


def test_exec_rate_catches_front_legs_not_stepping():
    """★ 執行率必須抓得到「前腳抬起來、原地放下」。

    2026-08-27：使用者看影片才發現前腳幾乎不往前踏，而當時**全部指標都是乾淨的**
    （離地 93–111 mm 四腿很平均、支撐腳 3.20、速度正常、三個診斷 0.00%）。
    這條測試釘住「這個缺陷會被量到」——它現在是已知狀態，不是回歸失敗。
    ⚠️ 若哪天基準改好了（前腳執行率上去），**這條要跟著改**，
       而且必須連同 `docs/CPG步態_完整結果_2026-08-27.md` 一起更新。
    """
    r = cw.rollout(gait="walk", secs=12.0, quiet=True)
    assert r["exec_rear"] > 1.0, f"後腳執行率不該低：{r['exec_rear']}"
    assert r["exec_front"] < 0.2, (
        f"前腳執行率變成 {r['exec_front']:.2f} —— 若這是刻意改好的，"
        "請更新本測試與 docs/CPG步態_完整結果_2026-08-27.md")
    # 世界前跨與「腿自走」必須分開：前腳世界有 30 mm 但腿自己只走 2–5 mm
    assert r["step_world"][0] > 20.0 and r["step_self"][0] < 10.0


def test_gain_override_changes_front_exec():
    """增益覆寫要真的生效 —— 加硬增益前腳執行率必須上去。

    這同時擋住「kp3 參數被接錯而靜默沿用預設」：那種錯誤下整組掃描會得到
    「增益沒有影響」的結論，而那結論看起來完全合理。
    """
    soft = cw.rollout(gait="walk", secs=12.0, quiet=True)
    stiff = cw.rollout(gait="walk", secs=12.0, kp3=[250.0, 250.0, 250.0],
                       kd3=[5.0, 5.0, 5.0], quiet=True)
    assert stiff["kp3"] == [250.0, 250.0, 250.0]
    assert stiff["exec_front"] > soft["exec_front"] + 0.3, (
        f"加硬增益後前腳執行率只從 {soft['exec_front']:.2f} 到 "
        f"{stiff['exec_front']:.2f} —— kp3 可能沒接進去")


def test_kp250_baseline_frozen():
    """kp=250 那組（給實機 M9 用）的參數釘住。

    ⚠️ 它**不是**新的凍結基準 —— 實機線的決定是「下一階段測試用 250，
       最終增益之後再議」。`BASELINE` 維持 kp=120 那組不動。
    """
    assert gb.BASELINE_KP250["duty"] == 0.85
    assert gb.BASELINE_KP250["d_step"] == 0.12
    assert gb.BASELINE_KP250["x_off"] == -0.050
    assert gb.BASELINE_KP250["z_sag"] == 0.036       # ★ 實機錨點，不是模擬掃出來的
    assert gb.BASELINE_KP250["kp3"] == [250.0, 250.0, 250.0]
    assert gb.BASELINE_KP250["kd3"] == [5.0, 5.0, 5.0]
    # 舊基準不可以被順手改掉
    assert gb.BASELINE["duty"] == 0.80 and gb.BASELINE["x_off"] == -0.040


def test_kp250_gait_actually_fixes_front_legs():
    """kp=250 那組必須真的把前腳修好 —— 這是它存在的唯一理由。

    ⚠️ 用它**必須同時給 kp3/kd3/z_sag**，只改 gait 名稱是不夠的。
       這條測試順便釘住那件事：少給增益的話前腳執行率會掉回 0.0x。
    """
    b = gb.BASELINE_KP250
    good = cw.rollout(gait="walk_kp250", secs=12.0, kp3=b["kp3"], kd3=b["kd3"],
                      z_sag=b["z_sag"], quiet=True)
    assert good["fell"] is None
    assert good["exec_front"] > 0.5, f"前腳執行率只有 {good['exec_front']:.2f}"
    assert good["step_self"][0] > 70.0, "前腿每步自走應該 > 70 mm"

    # 只改 gait 名稱、不給增益 → 前腳仍然不跨步（這是常見的誤用）
    bad = cw.rollout(gait="walk_kp250", secs=12.0, quiet=True)
    assert bad["exec_front"] < 0.3, (
        "沒給 kp3 時前腳執行率竟然是好的 —— 表示增益從別的地方漏進來了")
