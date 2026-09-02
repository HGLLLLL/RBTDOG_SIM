"""M9（步態）的計畫層驗證。

★ M9 的核心風險不在控制迴圈（那是照抄 M7/M8 已實機驗證的骨架），而在：

  1. **兩種模式必須產生一樣的東西** —— `--traj`（播放離線檔）與 `--live`
     （狗上算 CPG）如果不一致，「模擬驗過的軌跡」這個保證就是假的。
  2. **保護門檻和 M7/M8 不同** —— 步態擺動相本來就要 6~8 rad/s，
     照搬 M7/M8 的 `--vmax 4.0` 會在第一個擺動相就誤中止。
  3. **`g_c` 不能調小** —— 小於撓度時腳不離地，被地面拖著走。

⚠️ 純標準函式庫的部分不需要 numpy；跨模式比對那項需要。
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))

import coord            # noqa: E402
import cpg              # noqa: E402
import M9_gait as m9    # noqa: E402

Q_LIE = {
    "fl1_hip_roll": -0.014, "fl2_hip_pitch": +0.477, "fl3_knee_pitch": -2.522,
    "fr1_hip_roll": -0.021, "fr2_hip_pitch": +0.360, "fr3_knee_pitch": -2.507,
    "bl1_hip_roll": -0.039, "bl2_hip_pitch": -0.493, "bl3_knee_pitch": +2.595,
    "br1_hip_roll": +0.020, "br2_hip_pitch": -0.394, "br3_knee_pitch": +2.575,
}


class Args:
    def __init__(self, **kw):
        d = dict(ramp_kp=2.0, t1=1.5, t2=1.5, hold=2.0, hold_mid=1.5)
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)


# ════════════════════════════════════════════════════ 站起來／坐回去的路徑
def test_standup_and_sitdown_are_continuous():
    """路徑不可以有突跳 —— 承重中的突跳就是 kp×誤差 的力矩尖峰。"""
    a = Args()
    q0 = cpg.stand_targets(cpg.home_foot(coord.POSES["home"]),
                           cpg.knee_signs(coord.POSES["home"]), 0.04)
    pre = m9.build_standup(a, dict(Q_LIE), q0)
    post = m9.build_sitdown(a, q0, dict(Q_LIE))
    for segs in (pre, post):
        for (n0, _, _, p1), (n1, _, p0, _) in zip(segs, segs[1:]):
            for j in m9.LEGS12:
                assert p1[j] == pytest.approx(p0[j], abs=1e-12), f"{n0}→{n1}/{j}"
    assert pre[0][2] == Q_LIE and post[-1][3] == Q_LIE


def test_standup_ends_at_the_gait_start_pose():
    """★ 站起來的終點必須就是步態第一幀 —— 否則進入步態時會突跳。"""
    a = Args()
    q0 = {j: 0.1 * i for i, j in enumerate(m9.LEGS12)}
    pre = m9.build_standup(a, dict(Q_LIE), q0)
    assert pre[-1][3] == q0 and pre[-1][0] == "HOLD_stand"


def test_sitdown_starts_at_the_gait_end_pose():
    a = Args()
    qN = {j: 0.2 * i for i, j in enumerate(m9.LEGS12)}
    post = m9.build_sitdown(a, qN, dict(Q_LIE))
    assert post[0][2] == qN


# ════════════════════════════════════ ★★ 兩種模式必須產生一樣的東西
def test_live_cpg_matches_the_generated_trajectory_file(tmp_path):
    """★★ `--live` 與 `--traj` 對同一組參數要逐幀吻合。

    這是「離線產生 + 模擬驗過 + 狗上播放」那條路的**唯一保證** ——
    如果 live 模式算出別的東西，那個保證就不成立。
    """
    pytest.importorskip("numpy")
    gen = ROOT / "inference" / "gen_gait_traj.py"
    out = tmp_path / "t.json"
    r = subprocess.run(
        [sys.executable, str(gen), "--march", "--secs", "4", "--omega", "0.5",
         "--g-c", "0.12", "--x-off", "0.04", "--ramp", "1.0", "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    D = json.loads(out.read_text(encoding="utf-8"))
    p = D["params"]

    # live 端：完全照 M9 的算法重跑一次
    f0 = cpg.home_foot(coord.POSES["home"])
    ks = cpg.knee_signs(coord.POSES["home"])
    stp = cpg.make_step(cpg.PHASE_WALK)
    c = cpg.init(cpg.PHASE_WALK)
    mux = {l: D["baseline_ref"]["mu_x"] for l in cpg.LEGS}
    muy = {l: D["baseline_ref"]["mu_y"] for l in cpg.LEGS}
    om = {l: p["omega"] for l in cpg.LEGS}
    dt = D["dt"]
    q_stand = cpg.stand_targets(f0, ks, p["x_off"])
    n_ramp = int(round(p["ramp"] / dt))
    n_body = int(round(p["secs"] / dt))
    worst = 0.0
    for i in range(D["n"]):
        qg, ncl = cpg.joint_targets(c, f0, ks, p["x_off"], p["g_c"], p["d_step"],
                                    D["baseline_ref"]["d_step_y"], p["duty"],
                                    p["z_sag"])
        assert ncl == 0
        if i < n_ramp:
            u = i / max(n_ramp, 1)
        elif i < n_ramp + n_body:
            u = 1.0
        else:
            u = 1.0 - (i - n_ramp - n_body) / max(n_ramp, 1)
        u = 0.0 if u < 0 else (1.0 if u > 1 else u)
        s = 0.5 * (1.0 - math.cos(math.pi * u))
        for k, j in enumerate(D["joints"]):
            want = (1 - s) * q_stand[j] + s * qg[j]
            worst = max(worst, abs(want - D["q"][i][k]))
        # ⚠️ 這一行漏掉的話 CPG 狀態不會推進，整個比對就沒有意義
        #   （第 0 幀仍會過，因為那時狀態本來就一樣）—— 2026-08-27 我漏過一次。
        c = stp(c, mux, muy, om, dt)
    # 檔案存到小數第 6 位 → 1e-6 是捨入下限
    assert worst < 2e-6, f"live vs traj 最大差 {worst:.3e} rad"


# ════════════════════════════════════ ★ 保護門檻與 M7/M8 不同（刻意的）
def test_m9_speed_guards_are_much_looser_than_m7m8():
    """★ 步態擺動相要 6~8 rad/s。照搬 M7/M8 的 2.0/4.0 會在第一個擺動相誤中止。"""
    import M7_standup as m7
    import M8_swing as m8
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert 'default=16.0' in src and '"--vmax"' in src
    assert 'default=14.0, dest="vcmd_max"' in src
    # M8 的護欄
    s8 = Path(m8.__file__).read_text(encoding="utf-8")
    assert 'default=2.0, dest="vcmd_max"' in s8
    assert m7.SMOOTHSTEP_VPEAK > 1.5   # 餘弦插值，不是三次式


def test_gait_command_speed_really_exceeds_the_m8_guard():
    """把「為什麼要放寬」量化：同樣的步態在 M8 的 2.0 rad/s 下根本跑不了。"""
    f0 = cpg.home_foot(coord.POSES["home"])
    ks = cpg.knee_signs(coord.POSES["home"])
    stp = cpg.make_step(cpg.PHASE_WALK)
    c = cpg.init(cpg.PHASE_WALK)
    mux = {l: 1.8 for l in cpg.LEGS}
    muy = {l: 1.5 for l in cpg.LEGS}
    om = {l: 1.4 for l in cpg.LEGS}
    dt, prev, vmax = 0.02, None, 0.0
    for _ in range(400):
        q, _ = cpg.joint_targets(c, f0, ks, -0.04, 0.08, 0.10, 0.12, 0.80, 0.0156)
        if prev is not None:
            vmax = max(vmax, max(abs(q[j] - prev[j]) / dt for j in q))
        prev = q
        c = stp(c, mux, muy, om, dt)
    assert vmax > 2.0, "步態命令速度應該遠超 M8 的 2.0 rad/s"
    # ★ 基準步態實測 13.1 rad/s —— 我第一版把 --vmax 訂成 12，會誤中止。
    #   基準步態（HOME 基準、ω=1.4）實測 10.9 rad/s ＝ 馬達 19.9 的 55%。
    assert 10.0 < vmax < 12.0, f"基準步態命令峰值應在 11 附近（實測 {vmax:.1f}）"


# ════════════════════════════════════════════════════ 已知的坑
def test_default_g_c_is_large_enough_to_clear():
    """★ `g_c` 預設不可以小到腳不離地。

    2026-08-27 在 MuJoCo 重現：g_c=0.04 + z_sag=0.0156 → 實際離地 4.5 mm，
    而狗被拖著前進 118 mm。實機在 kp=250 量到的擺動離地損失是 36 mm。
    """
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert 'default=0.12, dest="g_c"' in src
    z_sag = 0.0325 * 120.0 / 250.0
    assert 0.12 + z_sag - 0.036 > 0.05, "預設 g_c 扣掉實機損失後餘裕太小"


def test_default_x_off_is_the_kp250_march_trim_not_the_kp120_baseline():
    """★ 原地踏步在 kp=250 的配平點是 +0.04，和基準的 −0.04 **符號相反**。"""
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert 'default=0.04, dest="x_off"' in src


def test_wheel_position_is_recorded():
    """★★ M8 只記了輪子的 effort，導致實機資料無法拆「觸地滾動 vs 懸空空轉」，
    而那是判斷「前後腿有沒有互相對抗」的唯一方法。M9 必須記 position。"""
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert 'stt[wi]["position"]' in src and '"w": wrec' in src


def test_wheels_are_never_position_held_during_the_gait():
    """★ 輪子全程 kp=0。設定檔的 FSM_RL_Wheel_Kp=60 是配「每步重給目標角」的 RL，
    開迴路套上去實測偏航失控 +39°/12s。"""
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert "kp=0.0, kd=a.wheel_kd" in src
    assert "wheel_kp" not in src.split("def main")[1].split("--wheel-kd")[0]


def test_live_uses_home_pose_not_stand():
    """★★ 迴歸：CPG 的基準足端是 **home**（hip 0.8/膝 −1.5），不是 stand（0.6/−1.2）。

    我第一版寫成 stand，兩種模式差了 **1.006 rad** ——
    正是 `test_live_cpg_matches_the_generated_trajectory_file` 抓出來的。
    這也是「兩份實作要互相比對」這個模式的價值。
    """
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert 'cpg.home_foot(coord.POSES["home"])' in src
    assert 'cpg.home_foot(coord.POSES["stand"])' not in src
    # 而 coord.POSES["home"] 必須就是 max_model.HOME
    pytest.importorskip("numpy")
    sys.path.insert(0, str(ROOT / "inference"))
    import max_model as mm
    mm2 = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
    for k, l in enumerate(mm.LEGS):
        for j, kd in enumerate(coord.LEG_KINDS):
            assert coord.POSES["home"][mm2[l] + kd] == pytest.approx(
                float(mm.HOME[k, j]), abs=1e-12)


# ────────────────────────────────────────────────────────────────────────
# commit_peak：孤立反號跳點
# ────────────────────────────────────────────────────────────────────────

def _feed(seq):
    """把 (tau, cap) 序列餵進 commit_peak，回傳 (peak, 剔除筆數)。"""
    peak, spikes, win = {"j": 0.0}, {"j": 0}, []
    for x in seq:
        win.append(x)
        if len(win) > 3:
            win.pop(0)
        m9.commit_peak(win, peak, spikes, "j")
    return peak["j"], spikes["j"]


def test_isolated_sign_flipped_spike_is_rejected():
    """★★ 迴歸：trip10 的 `fl3_knee_pitch -51.52`。

    真實資料（2026-09-02 t=16.11 附近）：只有中間那筆反號，而且**同一筆的 v
    也一起跳**，把 `kp|e|+kd|v|` 上限灌大到 34.3 —— ×1.5+1 = 52.5 > 51.52，
    所以舊的上限判別式**放它過**。鄰居才擋得住。
    """
    peak, n = _feed([(27.77, 27.8), (-51.52, 34.3), (28.01, 33.1), (29.90, 31.0)])
    assert n == 1
    # 跳點沒有變成峰值。取 28.01 而不是 29.90 —— 最後一筆永遠當不成「中間點」，
    # 那筆由 main() 收尾時補提交（見 M9_gait.py 的「最後一筆」註解）。
    assert peak == pytest.approx(28.01)


def test_a_real_sign_change_across_two_samples_is_kept():
    """★ 反例：真的換向（連續兩筆同號）**不可以**被剔除。

    擺動相結束、腿開始承重時力矩本來就會換號。誤殺這個會把真峰值藏起來。
    """
    peak, n = _feed([(30.0, 30.0), (-40.0, 40.0), (-42.0, 42.0), (-35.0, 35.0)])
    assert n == 0
    assert peak == pytest.approx(-42.0)


def test_small_signal_chatter_near_zero_is_not_treated_as_a_spike():
    """★ 反例：0 附近的抖動天天反號，不該被算成跳點（也不可能是峰值）。"""
    peak, n = _feed([(1.0, 5.0), (-2.0, 5.0), (1.5, 5.0), (-1.2, 5.0)])
    assert n == 0


def test_spike_must_be_larger_than_both_neighbours():
    """★ 反例：反號但**比鄰居小**的，是正常過零，不是跳點。"""
    peak, n = _feed([(30.0, 30.0), (-8.0, 30.0), (25.0, 30.0)])
    assert n == 0


def test_fast_abort_path_uses_raw_tau_never_the_filtered_count():
    """★★ 兩條中止路徑對「過濾」的取捨相反，這個測試釘住那條界線。

    **快速路徑**（`--tau-hits` 連續 N 筆、`TAU_HARD` 連續 2 筆）必須用**原始
    `tau`** —— 它要在持續超載發生的當下就跳，而 `commit_peak` 的鄰居判定要等
    下一筆，延遲一拍。它靠「連續」本身就對單筆雜訊免疫，不需要過濾。

    **慢速路徑**（`--tau-total` 整趟累計）反過來：本來就累積好幾秒才觸發，
    慢一拍無所謂；而不過濾的話幾筆感測跳點就會把累計值灌上去
    （trip13 的 `fl1` 未濾 3 筆、濾後只剩 1 筆）。
    """
    src = Path(m9.__file__).read_text(encoding="utf-8")
    body = src.split("def main()", 1)[1]
    # 快速路徑的三行條件都必須直接比 abs(tau)
    assert "if abs(tau) > TAU_HARD:" in body
    assert "elif abs(tau) > lim:" in body
    assert "tau_hot[j] >= a.tau_hits" in body
    assert "tau_hot[j] >= 2" in body
    # 慢速路徑用的是 commit_peak 累計的 over[]，不是 over_raw[]
    assert "over[j] >= a.tau_total" in body
    assert "over_raw[j] >= a.tau_total" not in body


def test_slow_abort_path_counts_only_filtered_exceedances():
    """★★ 慢速路徑必須用濾掉跳點後的計數，否則感測跳點會誤觸發。

    trip13 的 `fl1_hip_roll` 有兩筆假的 84.5 / 80.9（都在 50 門檻之上、
    都是孤立反號）。不過濾的話它們會佔掉 `--tau-total` 額度的 40%。
    """
    peak = {"fl1_hip_roll": 0.0}
    spikes = {"fl1_hip_roll": 0}
    over = {"fl1_hip_roll": 0}
    win = []
    # 真實形狀：+3.8 → −84.5（跳點）→ +7.0
    for x in [(3.76, 4.0), (3.66, 4.0), (-84.50, 4.0), (6.97, 8.0), (2.13, 4.0)]:
        win.append(x)
        if len(win) > 3:
            win.pop(0)
        m9.commit_peak(win, peak, spikes, "fl1_hip_roll", over)
    assert spikes["fl1_hip_roll"] == 1
    assert over["fl1_hip_roll"] == 0, "跳點不可以計入 --tau-total"


def test_slow_abort_path_does_count_real_impact_peaks():
    """★ 反例：真的衝擊尖峰**必須**被算進去，否則這條路徑等於沒有。

    trip13 `bl3_knee_pitch` 的真實形狀（t=12.70 附近）：連續好幾筆同號、
    乾淨衰減 —— 不是孤立反號，過濾器不該碰它。
    """
    peak = {"bl3_knee_pitch": 0.0}
    spikes = {"bl3_knee_pitch": 0}
    over = {"bl3_knee_pitch": 0}
    win = []
    for x in [(-32.10, 104.0), (-80.03, 76.5), (-69.79, 76.0),
              (-56.89, 77.7), (-47.58, 83.0)]:
        win.append(x)
        if len(win) > 3:
            win.pop(0)
        m9.commit_peak(win, peak, spikes, "bl3_knee_pitch", over)
    assert spikes["bl3_knee_pitch"] == 0
    # 只有 −80.03 超過膝門檻 70 —— −69.79 是 69.79，**差 0.21 就不算**。
    # ★ 這正是為什麼單看「超標筆數」還不夠：真實的衝擊尖峰有一整串接近門檻的
    #   樣本，計數對門檻極度敏感。要判讀嚴重程度還是得看峰值。
    assert over["bl3_knee_pitch"] == 1
    assert peak["bl3_knee_pitch"] == pytest.approx(-80.03)


def test_tau_total_default_separates_the_four_real_trips():
    """★ 預設值要能分開「已知安全」與「已知超載」的實測資料。

    實測累計筆數（濾後）：trip10 = 0、trip11 = 0、trip12 = 1、
    trip13 = 13（bl3）/ 5（fr3）。預設 5 落在 1 與 5 之間，且靠近後者。
    """
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert 'default=5, dest="tau_total"' in src


def test_report_peaks_never_raises():
    """★★ `report_peaks()` 跑在 `keeper.start()` 之後，而 Keepalive 是 daemon ——
    這裡一拋例外，process 就結束、心跳停、指令區約 0.5 秒後被清零，
    **狗會在站姿失力**。所以要把各種畸形輸入都打過一遍。
    """
    full = {j: [] for j in m9.LEGS12}
    cases = [
        # (tau_win, peak, spikes, 說明)
        ({j: [] for j in m9.LEGS12}, {j: 0.0 for j in m9.LEGS12},
         {j: 0 for j in m9.LEGS12}),                       # 一筆都沒跑到就中止
        ({j: [(1.0, 1.0)] for j in m9.LEGS12}, {j: 0.0 for j in m9.LEGS12},
         {j: 0 for j in m9.LEGS12}),                       # 只有一筆
        ({j: [(5.0, 5.0), (-70.0, 90.0)] for j in m9.LEGS12},
         {j: -12.5 for j in m9.LEGS12}, {j: 3 for j in m9.LEGS12}),
        ({}, {j: 0.0 for j in m9.LEGS12}, {}),             # 視窗/計數整個沒建起來
        (full, {j: 0.0 for j in m9.LEGS12}, {j: 0 for j in m9.LEGS12}),
    ]
    for win, peak, spikes in cases:
        m9.report_peaks(win, peak, spikes)                 # 不炸就是通過


def test_report_peaks_prints_every_joint(capsys):
    """峰值表要 12 顆全印 —— 少印一顆＝現場少一個判準。"""
    m9.report_peaks({j: [] for j in m9.LEGS12},
                    {j: 0.0 for j in m9.LEGS12},
                    {j: 0 for j in m9.LEGS12})
    out = capsys.readouterr().out
    for j in m9.LEGS12:
        assert j in out


def test_report_peaks_commits_the_final_sample():
    """★ 最後一筆永遠當不成「中間點」，必須由 report_peaks 補提交。

    漏掉它 = 「中止當下那一筆」的力矩不會出現在峰值表裡，
    而中止時最想看的就是那一筆。
    """
    peak = {j: 0.0 for j in m9.LEGS12}
    win = {j: [(10.0, 10.0), (44.0, 40.0)] for j in m9.LEGS12}
    m9.report_peaks(win, peak, {j: 0 for j in m9.LEGS12})
    assert peak["fl3_knee_pitch"] == pytest.approx(44.0)
