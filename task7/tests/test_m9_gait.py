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
import time
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
    # 2026-09-03 起輪 kd 按階段排程（wheel_kd_of），但 kp 恆 0 這件事不變
    assert "kp=0.0, kd=wheel_kd" in src
    assert "def wheel_kd_of" in src
    assert "wheel_kp" not in src.split("def main")[1].split("--wheel-kd")[0]


def test_gain_guard_compares_wheel_kd_not_just_kp_and_kd():
    """★★ 「說兩次」的防呆必須涵蓋**三個**增益。

    2026-09-03 發現的缺口：原本只比 `kp`/`kd`。而輪阻尼送給狗的是**命令列**的值
    （預設 0.5），不是軌跡檔裡的值 —— 所以「用 `--wheel-kd 3.0` 產了檔、
    跑的時候忘了帶那個旗標」會**靜默用回 0.5**，而 0.5 正是「前腳不跨步」那組
    （模擬前腳執行率 0.03 vs 3.0 的 0.79）。
    症狀是「模擬明明好了、實機還是老樣子」，**而所有診斷指標都乾淨**。
    """
    # 用**真的要送上狗的那個軌跡檔**測，不是自己捏的字典。
    traj = json.loads((ROOT / "outputs/gait/walk_kp120_first.json")
                      .read_text(encoding="utf-8"))
    assert traj["wheel_kd"] == 0.5 and traj["kp"] == 120.0 and traj["kd"] == 1.0

    # 三個都對上 → 放行
    assert m9.gain_mismatches(traj, 120.0, 1.0, 0.5) == []

    # ★ 這一格就是缺口本身：kp/kd 都對，只有 wheel_kd 沒帶到 → 必須擋下來
    assert [k for k, _, _ in m9.gain_mismatches(traj, 120.0, 1.0, 3.0)] == ["wheel_kd"]

    # 檔案裡缺欄位也算不一致（不可以當成「沒寫就用預設」放行）
    assert m9.gain_mismatches({k: v for k, v in traj.items() if k != "wheel_kd"},
                              120.0, 1.0, 0.5)

    # 錯誤訊息要給得出可以直接貼上去的完整指令（含 --wheel-kd），
    # 不然操作者站在狗前面還要自己拼旗標。
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert "--wheel-kd" in src.split("if bad:", 1)[1].split("return 1", 1)[0]


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


# ────────────────────────────────────────────────────────────────────────
# phase_gains：站起來用 M7 驗證過的 250，只有步態用 --kp
# ────────────────────────────────────────────────────────────────────────

def test_standup_never_uses_the_gait_gain():
    """★★ 安全：`--kp 120` 不可以讓狗用沒測過的增益從趴姿站起來。

    M7 只在 kp=250 驗證過承重站立（四趟）。承重站起來是整個序列風險最高的
    一段，不該拿它試新增益。原本 `STANDUP_KP = 250` 定義了卻從沒被用過 ——
    在 kp 一直是 250 的時候沒差，一旦要學原廠降到 120 就變成安全問題。
    """
    for nm in ("GO_crouch", "HOLD_crouch", "GO_stand",
               "HOLDB_crouch", "BACK_LIE"):
        kp, kd = m9.phase_gains(nm, 0.5, 120.0, 1.0)
        assert kp == pytest.approx(m9.STANDUP_KP), nm
        assert kd == pytest.approx(m9.STANDUP_KD), nm


def test_only_the_gait_phase_uses_the_gait_gain():
    kp, kd = m9.phase_gains("GAIT", 0.5, 120.0, 1.0)
    assert (kp, kd) == pytest.approx((120.0, 1.0))


def test_gain_change_is_ramped_never_stepped():
    """★★ 承重中 kp 不可以階躍下降 —— 撐機身的力矩會瞬間砍半，狗會掉下去。

    降的那一段（`HOLD_stand`）狗是靜止的；升的那一段（`BACK_crouch`）是往上收。
    **兩個方向都往安全的那一邊。**
    """
    prev = m9.phase_gains("GO_stand", 1.0, 120.0, 1.0)[0]
    seen = [prev]
    for r in [i / 20 for i in range(21)]:
        seen.append(m9.phase_gains("HOLD_stand", r, 120.0, 1.0)[0])
    seen.append(m9.phase_gains("GAIT", 0.0, 120.0, 1.0)[0])
    assert seen[0] == pytest.approx(250.0) and seen[-1] == pytest.approx(120.0)
    assert max(abs(b - a) for a, b in zip(seen, seen[1:])) < 15.0, "有階躍"
    # 出步態：BACK_crouch 前半升回 250
    up = [m9.phase_gains("BACK_crouch", r / 20, 120.0, 1.0)[0] for r in range(21)]
    assert up[0] == pytest.approx(120.0) and up[-1] == pytest.approx(250.0)
    assert max(abs(b - a) for a, b in zip(up, up[1:])) < 15.0, "有階躍"


def test_damping_ratio_stays_sane_through_the_transition():
    """★ kd 要跟 kp 同比例走。先降 kp 再降 kd 會短暫過阻尼，反之欠阻尼。"""
    for r in [i / 20 for i in range(21)]:
        kp, kd = m9.phase_gains("HOLD_stand", r, 120.0, 1.0)
        # 站立是 250/5.0（比 0.020）、步態是 120/1.0（比 0.0083）——
        # 過程中不可以跑到區間外
        assert 0.0083 - 1e-6 <= kd / kp <= 0.0200 + 1e-6, (r, kp, kd)


def test_kp_equals_250_behaves_exactly_as_before():
    """★ 迴歸：kp=250 時（前四趟的設定）排程必須和舊行為完全一致。"""
    for nm in ("RAMP_UP", "GO_crouch", "HOLD_stand", "GAIT", "BACK_crouch",
               "BACK_LIE", "RAMP_DOWN"):
        for r in (0.0, 0.5, 1.0):
            kp, kd = m9.phase_gains(nm, r, 250.0, 5.0)
            want = 250.0 * (r if nm == "RAMP_UP"
                            else (1 - r) if nm == "RAMP_DOWN" else 1.0)
            assert kp == pytest.approx(want), (nm, r)
            assert kd == pytest.approx(5.0)


def test_sitdown_after_enter_uses_standup_gains_not_gait_gains():
    """★★ 按 Enter 坐回趴姿是**承重動作** —— 用 kp=120 撐不住。"""
    src = Path(m9.__file__).read_text(encoding="utf-8")
    tail = src.split("SIT_crouch", 1)[1]
    assert "max(held_kp, STANDUP_KP), STANDUP_KD" in tail
    assert "max(held_kp, a.kp)" not in tail


def test_gain_mismatch_error_prints_the_correct_command():
    """★ 護欄擋下來的時候，要直接給出可以貼上去的指令。

    2026-09-02 現場實例：操作卡漏了 `--kp 120 --kd 1.0`，人站在狗前面
    看到「不一致」但不知道該打什麼。護欄本身是對的（刻意的「說兩次」防呆，
    軌跡檔說一次、命令列說一次，這樣拿錯檔案會被擋住）——
    **不可以改成自動採用檔案值**，那等於把唯一一道防呆拿掉。
    要改的是訊息，不是行為。

    ⚠️ 2026-09-03：訊息改成逐項列出（因為要涵蓋第三個增益 `wheel_kd`），
    所以錨點從「軌跡檔的增益」換成組指令的那一行。判準本身沒有變。
    """
    src = Path(m9.__file__).read_text(encoding="utf-8")
    blk = src.split("if bad:", 1)[1].split("return 1", 1)[0]
    assert "--kp {D['kp']:g}" in blk and "--kd {D['kd']:g}" in blk, \
        "要印出正確的 --kp/--kd"
    assert "--wheel-kd {D.get('wheel_kd', 0.5):g}" in blk, "也要印出 --wheel-kd"
    assert "--confirm" in blk, "真跑的指令也要給"
    # 行為不可以變成自動採用
    assert 'a.kp = D["kp"]' not in src and "a.kp = D['kp']" not in src


# ══════════════════════════ 2026-09-03：LS 序列 + body sway 的實機路徑
def test_live_cpg_matches_traj_for_the_recommended_gait(tmp_path):
    """★★ **下午要上機的那一組**：LS ＋ body sway，`--live` 與 `--traj` 逐幀吻合。

    既有的同型測試只涵蓋 DS、無 sway。新參數如果只進了其中一條路徑，
    症狀會是「離線驗過的軌跡與狗上即時算的不一樣」，而兩邊各自都自洽。
    """
    pytest.importorskip("numpy")
    gen = ROOT / "inference" / "gen_gait_traj.py"
    out = tmp_path / "ls.json"
    r = subprocess.run(
        [sys.executable, str(gen), "--march", "--secs", "4", "--omega", "0.5",
         "--seq", "ls", "--kp", "120", "--kd", "1.0", "--wheel-kd", "3.0",
         "--x-off", "-0.020", "--g-c", "0.07",
         "--sway-x", "0.015", "--sway-y", "0.010",
         "--sway-lead-x", "0.90", "--sway-lead-y", "0.20",
         "--ramp", "1.0", "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    D = json.loads(out.read_text(encoding="utf-8"))
    p = D["params"]
    assert D["seq"] == "ls" and D["sway_x"] == 0.015, "頂層參數沒寫進檔案 → M9 比對不到"

    phase = cpg.PHASES["ls"]
    f0 = cpg.home_foot(coord.POSES["home"])
    ks = cpg.knee_signs(coord.POSES["home"])
    stp = cpg.make_step(phase)
    c = cpg.init(phase)
    mux = {l: D["baseline_ref"]["mu_x"] for l in cpg.LEGS}
    muy = {l: D["baseline_ref"]["mu_y"] for l in cpg.LEGS}
    om = {l: p["omega"] for l in cpg.LEGS}
    dt = D["dt"]
    x_off_legs = cpg.x_off_split(p["x_off"], p["x_d"])
    q_stand = cpg.stand_targets(f0, ks, x_off_legs)
    n_ramp = int(round(p["ramp"] / dt))
    n_body = int(round(p["secs"] / dt))
    worst = 0.0
    for i in range(D["n"]):
        sway = cpg.body_sway(cpg.gait_phase(c["theta"], phase),
                             p["sway_x"], p["sway_y"],
                             p["sway_lead_x"], p["sway_lead_y"])
        qg, ncl = cpg.joint_targets(c, f0, ks, x_off_legs, p["g_c"], p["d_step"],
                                    D["baseline_ref"]["d_step_y"], p["duty"],
                                    p["z_sag"], sway)
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
            worst = max(worst, abs((1 - s) * q_stand[j] + s * qg[j] - D["q"][i][k]))
        c = stp(c, mux, muy, om, dt)
    assert worst < 2e-6, f"live vs traj（建議組）最大差 {worst:.3e} rad"


def test_param_mismatch_catches_forgotten_seq_and_sway():
    """★ 忘了帶 `--seq ls` / `--sway-*` 必須被擋下來。

    這正是 `wheel_kd` 那個坑的同型：忘了帶旗標 → 靜默跑回預設值 →
    「模擬明明好了、實機還是老樣子」，而所有診斷指標都乾淨。
    """
    D = {"kp": 120.0, "kd": 1.0, "wheel_kd": 3.0, "seq": "ls",
         "x_off": -0.020, "g_c": 0.07, "sway_x": 0.015, "sway_y": 0.010,
         "sway_lead_x": 0.90, "sway_lead_y": 0.20}
    ok = dict(D)
    assert m9.param_mismatches(D, ok) == []
    # 忘了 --seq ls（預設 ds）
    assert [k for k, _, _ in m9.param_mismatches(D, dict(ok, seq="ds"))] == ["seq"]
    # 忘了 --sway-lead-x（預設 0）—— 幅度對了但相位沒提前，等於跑成純擾動
    assert [k for k, _, _ in
            m9.param_mismatches(D, dict(ok, sway_lead_x=0.0))] == ["sway_lead_x"]
    # 檔案裡沒有那一項也要報（舊檔配新旗標）
    assert [k for k, _, _ in
            m9.param_mismatches({k: v for k, v in D.items() if k != "seq"},
                                ok)] == ["seq"]


# ══════════════════════════ 互動模式（--interactive）的兩個元件
def test_gait_stream_matches_offline_trajectory(tmp_path):
    """★★ `GaitStream`（即時算）必須與 `gen_gait_traj`（離線）逐幀吻合。

    互動模式不能播固定長度的軌跡檔（走多久由現場決定），所以步態要在狗上即時算。
    這是那條路徑的**唯一保證** —— 算出別的東西的話，離線驗過的一切都不算數。

    ⚠️ 特別驗「50 Hz 推進 + 內插」這件事：若改成每個寫入 tick 都推進 CPG，
    積分步長就變了，跑出來不是同一個步態。
    """
    pytest.importorskip("numpy")
    gen = ROOT / "inference" / "gen_gait_traj.py"
    out = tmp_path / "s.json"
    r = subprocess.run(
        [sys.executable, str(gen), "--secs", "6", "--omega", "1.4",
         "--seq", "ls", "--kp", "120", "--kd", "1.0", "--wheel-kd", "3.0",
         "--x-off", "-0.020", "--g-c", "0.048", "--d-step", "0.10",
         "--sway-x", "0.015", "--sway-y", "0.010",
         "--sway-lead-x", "0.90", "--sway-lead-y", "0.20",
         "--ramp", "1.0", "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    D = json.loads(out.read_text(encoding="utf-8"))
    p = dict(D["params"], mu_x=D["baseline_ref"]["mu_x"],
             mu_y=D["baseline_ref"]["mu_y"], d_step_y=D["baseline_ref"]["d_step_y"])

    f0 = cpg.home_foot(coord.POSES["home"])
    ks = cpg.knee_signs(coord.POSES["home"])
    gs = m9.GaitStream(p, f0, ks)
    dt = D["dt"]
    n_ramp = int(round(p["ramp"] / dt))
    worst = 0.0
    # 只比 ramp 之後（s=1.0，檔案裡就是純步態）
    for i in range(n_ramp, D["n"] - n_ramp):
        q = gs.sample(i * dt)
        for k, j in enumerate(D["joints"]):
            worst = max(worst, abs(q[j] - D["q"][i][k]))
    assert gs.n_clamp == 0
    assert worst < 2e-6, f"GaitStream vs 離線軌跡最大差 {worst:.3e} rad"


def test_gait_stream_interpolates_between_50hz_frames():
    """★ 兩個 50 Hz 幀之間必須是線性內插，不是零階保持也不是提前推進 CPG。"""
    p = dict(seq="ls", x_off=-0.020, x_d=0.0, g_c=0.048, d_step=0.10,
             d_step_y=0.12, duty=0.80, mu_x=1.80, mu_y=1.50, omega=1.4,
             z_sag=0.075, sway_x=0.015, sway_y=0.010,
             sway_lead_x=0.90, sway_lead_y=0.20)
    f0 = cpg.home_foot(coord.POSES["home"])
    ks = cpg.knee_signs(coord.POSES["home"])
    gs = m9.GaitStream(p, f0, ks)
    a = gs.sample(0.0)
    mid = gs.sample(0.01)
    b = gs.sample(0.02)
    for j in a:
        assert mid[j] == pytest.approx(0.5 * (a[j] + b[j]), abs=1e-12), j
    # 單調不減的 t 可以連續取樣，不會跳步
    gs2 = m9.GaitStream(p, f0, ks)
    for i in range(200):
        gs2.sample(i * 0.005)
    assert gs2.sample(1.0) == pytest.approx(
        {j: v for j, v in m9.GaitStream(p, f0, ks).sample(1.0).items()}, abs=1e-12)


def test_keywatch_never_blocks_without_tty():
    """★ 沒有 tty（例如 nohup／腳本背景執行）時必須直接回 False，不能卡住。

    卡住 = 心跳停 = controller 500 ms 後清零 = 承重中的狗塌下去。
    """
    kw = m9.KeyWatch(enabled=True)
    t0 = time.monotonic()
    for _ in range(50):
        kw.pressed()
    assert time.monotonic() - t0 < 0.2, "KeyWatch 會阻塞 —— 那會讓心跳停掉"
    assert m9.KeyWatch(enabled=False).pressed() is False


# ══════════════════════════ InteractivePlan：三段流程的狀態機
def _plan(**kw):
    """建一個 InteractivePlan（用假的 GaitStream，只驗時序與增益）。"""
    import types
    a = types.SimpleNamespace(
        ramp_kp=2.0, t1=1.5, t2=1.5, hold=2.0, hold_mid=1.5, ramp=3.0,
        kp=120.0, kp_abad=60.0, kd=1.0, hold_max=25.0, walk_max=20.0, kp_shift=1.5)
    for k, v in kw.items():
        setattr(a, k, v)
    q_lie = {j: 0.0 for j in m9.LEGS12}
    q_stand = {j: 1.0 for j in m9.LEGS12}

    class FakeGS:
        def sample(self, t):
            return {j: 2.0 for j in m9.LEGS12}
    return m9.InteractivePlan(a, q_lie, q_stand, FakeGS()), a, q_lie, q_stand


def test_interactive_phase_order():
    """★ 階段順序，以及三個等待點的位置。"""
    plan, a, _, _ = _plan()
    names = [x[0] for x in plan.segs]
    assert names == ["RAMP_UP", "GO_crouch", "HOLD_crouch", "GO_stand",
                     "READY", "KP_DOWN", "GAIT_IN", "GAIT", "GAIT_OUT",
                     "KP_UP", "STOPPED", "SIT_crouch", "SIT_hold", "SIT_LIE",
                     "RAMP_DOWN"]
    # 等待階段（dur=None）正好三個，就是使用者要按的三次
    assert [n for n, d, *_ in plan.segs if d is None] == ["READY", "GAIT", "STOPPED"]


def test_interactive_enter_advances_only_at_wait_phases():
    """按鍵只在等待階段生效；其他階段按了要被忽略（不能跳過站起來）。"""
    plan, a, _, _ = _plan()
    t = 0.0
    # RAMP_UP 期間狂按 —— 不能推進
    for _ in range(10):
        nm, *_ = plan.update(t, True)
        t += 0.005
    assert nm == "RAMP_UP"
    # 走到 READY
    t = a.ramp_kp + a.t1 + a.hold_mid + a.t2 + 0.01
    nm, *_ = plan.update(t, False)
    assert nm == "READY"
    nm, *_ = plan.update(t + 0.01, True)          # 按下 → 進 KP_DOWN
    assert nm == "KP_DOWN"


def test_interactive_wait_times_out_and_continues():
    """★ 等待逾時是「自動往下走」，不是中止 —— 人分心時狗不能一直承重站著。"""
    plan, a, _, _ = _plan(hold_max=5.0)
    t = a.ramp_kp + a.t1 + a.hold_mid + a.t2 + 0.01
    nm, *_ = plan.update(t, False)
    assert nm == "READY"
    nm, *_ = plan.update(t + 5.01, False)
    assert nm == "KP_DOWN", "逾時之後沒有自動往下走"


def test_interactive_kp_never_steps_while_loaded():
    """★★ 承重期間 kp 不可以階躍。

    走完整條流程逐 tick 檢查相鄰兩幀的 kp 變化量。
    kp 從 250 砍到 120 是 −52%，靜態撓度 8.9→18 mm，機身會掉下去。
    """
    plan, a, _, _ = _plan(hold_max=1.0, walk_max=1.0)
    t, prev_kp, worst, where = 0.0, None, 0.0, ""
    dt = 1.0 / 200
    for _ in range(int(60 / dt)):
        nm, des, kp, kd, kpa, done = plan.update(t, False)
        if prev_kp is not None:
            d = abs(kp - prev_kp)
            if d > worst:
                worst, where = d, nm
        prev_kp = kp
        t += dt
        if done:
            break
    # 每個 tick 最多變 (250-120)/kp_shift/200 ≈ 0.43；抓 2.0 當門檻
    assert worst < 2.0, f"kp 在 {where} 有階躍：單 tick 變化 {worst:.1f}"


def test_interactive_ends_at_lie_with_zero_kp():
    """流程走完必須回到趴姿、kp 歸零 —— 否則狗會帶著增益留在原地。"""
    plan, a, q_lie, _ = _plan(hold_max=1.0, walk_max=1.0)
    t, dt = 0.0, 1.0 / 200
    last = None
    for _ in range(int(60 / dt)):
        last = plan.update(t, False)
        t += dt
        if last[-1]:
            break
    nm, des, kp, kd, kpa, done = last
    assert done, "流程沒有結束"
    assert nm == "RAMP_DOWN"
    assert kp == pytest.approx(0.0, abs=1e-9), f"收工時 kp = {kp}"
    for j in m9.LEGS12:
        assert des[j] == pytest.approx(q_lie[j], abs=1e-9), f"{j} 沒回到趴姿"


def test_interactive_gait_out_returns_to_stand_pose():
    """★ 淡出結束必須落在站姿 —— 接下來 SIT_crouch 是從站姿起算的。"""
    plan, a, _, q_stand = _plan(hold_max=0.5, walk_max=0.5)
    t, dt = 0.0, 1.0 / 200
    seen = None
    for _ in range(int(60 / dt)):
        nm, des, kp, kd, kpa, done = plan.update(t, False)
        if nm == "KP_UP" and seen is None:
            seen = des          # 進 KP_UP 的第一幀 = 淡出剛結束
        t += dt
        if done:
            break
    assert seen is not None, "沒有走到 KP_UP"
    for j in m9.LEGS12:
        assert seen[j] == pytest.approx(q_stand[j], abs=1e-6), f"{j} 沒回到站姿"


def test_interactive_can_stop_during_gait_ramp_in():
    """★ 起步淡入期間按 Enter 也要能停 —— 使用者要的是「隨時」。

    而且淡出必須從**當下的混合比例**接續，不能先跳回全步態再淡出。
    """
    plan, a, _, q_stand = _plan(hold_max=1.0, walk_max=5.0)
    t, dt = 0.0, 1.0 / 200
    # 走到 GAIT_IN 中段
    while plan.name != "GAIT_IN":
        plan.update(t, False)
        t += dt
    for _ in range(int(1.2 / dt)):          # 淡入 3 秒，走 1.2 秒
        nm, des_before, *_ = plan.update(t, False)
        t += dt
    assert plan.name == "GAIT_IN"
    u_before = (des_before[m9.LEGS12[0]] - q_stand[m9.LEGS12[0]]) / \
               (2.0 - q_stand[m9.LEGS12[0]])          # FakeGS 回 2.0
    nm, des_after, *_ = plan.update(t, True)
    assert nm == "GAIT_OUT", "淡入期間按 Enter 沒有生效"
    # 接續：按下前後那一幀不能跳
    for j in m9.LEGS12:
        assert des_after[j] == pytest.approx(des_before[j], abs=0.02), \
            f"{j} 在按停瞬間跳了 {des_after[j] - des_before[j]:+.3f}"
    assert plan.u_out0 == pytest.approx(u_before, abs=0.05)
    # 淡出結束仍要落在站姿
    t_end = t + plan.segs[plan.i][1] + dt
    while plan.name == "GAIT_OUT":
        nm, des, *_ = plan.update(t, False)
        t += dt
        if t > t_end + 1.0:
            break
    assert plan.name == "KP_UP"


def test_interactive_stop_during_gait_still_ramps_out():
    """走路中按 Enter 是「開始減速」不是急停 —— 淡出時間仍然完整。"""
    plan, a, _, _ = _plan(hold_max=1.0, walk_max=20.0)
    t, dt = 0.0, 1.0 / 200
    while plan.name != "GAIT":
        plan.update(t, False)
        t += dt
    t_press = t + 2.0
    while t < t_press:
        plan.update(t, False)
        t += dt
    nm, *_ = plan.update(t, True)
    assert nm == "GAIT_OUT"
    assert plan.segs[plan.i][1] == pytest.approx(a.ramp), "淡出時間被縮短了"


# ══════════════════════════ trip17 事故（2026-09-03）後加的兩個偵測器
def test_chatter_watch_fires_on_oscillation_not_rolling():
    """★ 抖振（高幅正負翻轉）要抓；正常滾動與雜訊不可誤報。

    門檻對過六趟實機資料：事故 t=2.23s 觸發（比使用者按急停早 1.7 秒）、
    五趟正常資料最長連續 0 筆。
    """
    cw = m9.ChatterWatch()
    # 事故型：±10 rad/s 交替
    fired = False
    for i in range(40):
        fired = cw.feed("w", 10.0 if i % 2 == 0 else -10.0) or fired
    assert fired, "事故等級的抖振沒被抓到"
    # 正常滾動：單向 10 rad/s
    cw2 = m9.ChatterWatch()
    assert not any(cw2.feed("w", 10.0) for _ in range(200))
    # 雜訊：小幅翻轉（velocity 欄位雜訊 47%，但幅度小）
    cw3 = m9.ChatterWatch()
    assert not any(cw3.feed("w", (1.5 if i % 2 == 0 else -1.5)) for i in range(200))
    # 偶發單次反向（換方向）後繼續單向 —— 分數要衰減回去
    cw4 = m9.ChatterWatch()
    seq = [8.0] * 20 + [-8.0] * 20 + [8.0] * 20
    assert not any(cw4.feed("w", v) for v in seq)


def test_wheel_kd_safe_constant():
    """起身／坐下的輪阻尼上限 0.5 —— trip16 實機驗證值；3.0 會抖振（trip17）。"""
    assert m9.WHEEL_KD_SAFE == 0.5


# ══════════════════════════ M10（輪阻尼抖振測試）—— 可離機測的部分
def test_m10_imports_and_reuses_m9_detectors():
    """★ M10 必須重用 M9 的 ChatterWatch/KeyWatch —— 同一套偵測器兩份實作
    是這條線最痛的一類問題（門檻對過六趟實機資料的是 M9 那份）。"""
    import M10_wheel_kd_chatter as m10
    assert m10.ChatterWatch is m9.ChatterWatch
    assert m10.KeyWatch is m9.KeyWatch
    # 保護門檻：kd·|v| 上限要低於輪力矩上限 33
    assert m10.TAU_IMPLIED_MAX < 33.0


def test_m10_requires_ascending_kds():
    """逐級由小到大 —— 一抖就停，順序反了會漏掉安全級。"""
    src = Path(__import__("M10_wheel_kd_chatter").__file__).read_text(encoding="utf-8")
    assert 'kds == sorted(kds)' in src


# ══════════════ ABAD 逐關節 kp（2026-09-03 兩趟步態失敗的根因修正）
def test_abad_kp_schedule_matches_factory_and_sim():
    """★★ 步態段 ABAD=60、hip/knee=120；站立段一律 250 —— 與原廠及模擬一致。

    修正前 M9 把 --kp 套到 12 顆全關節：實機 ABAD 比所有模擬驗證硬一倍，
    重現實驗證實那正是「前腳不跨＋搖擺 11°」的根因。
    """
    for nm, r, want_hipknee, want_abad in (
            ("GO_stand", 0.5, 250.0, 250.0),     # 站立段：原廠 ABAD 也是 250
            ("READY", 0.5, 250.0, 250.0),
            ("KP_DOWN", 1.0, 120.0, 60.0),       # 過渡終點
            ("GAIT", 0.5, 120.0, 60.0),          # ★ 步態＝模擬驗證的 [60,120,120]
            ("KP_UP", 1.0, 250.0, 250.0)):
        assert m9.phase_kp(nm, r, 120.0) == pytest.approx(want_hipknee), nm
        assert m9.phase_kp(nm, r, 60.0) == pytest.approx(want_abad), nm


def test_abad_kp_transition_is_continuous():
    """KP_DOWN 全程 ABAD 從 250 平滑到 60，不許階躍（承重中階躍會掉機身）。"""
    prev = None
    for i in range(101):
        v = m9.phase_kp("KP_DOWN", i / 100, 60.0)
        if prev is not None:
            assert abs(v - prev) < 3.0, f"r={i/100} 跳了 {abs(v-prev):.1f}"
        prev = v
    assert m9.phase_kp("KP_DOWN", 0.0, 60.0) == pytest.approx(250.0)
    assert m9.phase_kp("KP_DOWN", 1.0, 60.0) == pytest.approx(60.0)


def test_interactive_plan_returns_abad_kp():
    plan, a, _, _ = _plan(hold_max=0.5, walk_max=0.5)
    t, dt = 0.0, 1.0 / 200
    seen = {}
    for _ in range(int(60 / dt)):
        nm, des, kp, kd, kpa, done = plan.update(t, False)
        seen.setdefault(nm, (kp, kpa))
        t += dt
        if done:
            break
    assert seen["GAIT"] == (120.0, 60.0)
    assert seen["READY"] == (250.0, 250.0)


def test_param_mismatch_catches_forgotten_kp_abad():
    """忘了帶 --kp-abad（或用舊檔）要被「說兩次」抓到。"""
    D = {"kp": 120.0, "kd": 1.0, "wheel_kd": 0.5, "kp_abad": 60.0}
    ok = {"kp": 120.0, "kd": 1.0, "wheel_kd": 0.5, "kp_abad": 60.0}
    assert m9.param_mismatches(D, ok) == []
    assert [k for k, _, _ in m9.param_mismatches(D, dict(ok, kp_abad=120.0))] == ["kp_abad"]


def test_m9_rt_loop_hardening_markers():
    """★ 16:47 假中止的三個修正必須都在：GC 關閉、舊幀防護、追趕重同步。

    事故：t=19.37 決定性 45ms 停頓（兩趟同時刻＝GC）→ 12 個 <1ms 追趕 tick
    讀同一幀 → 「連續 3 筆」把一筆實體樣本數成 3 筆 → 假中止。
    這在本機無法整合測試（要 /dev/shm），只能釘住原始碼標記。
    """
    src = Path(m9.__file__).read_text(encoding="utf-8")
    assert "gc.disable()" in src and "gc.enable()" in src
    assert "stale = (tick_state == last_state_tick)" in src
    assert "nxt = time.monotonic() + 1.0 / a.hz" in src, "停頓後沒有重新對時"
    # 保護計數與紀錄都必須在 stale 防護之後
    assert src.index("stale = (tick_state") < src.index("chatter.feed")


def test_all_write_frame_calls_have_five_args():
    """★★ 16:47 之後的鐵律：`write_frame` 簽名改了就要掃**全部**呼叫點。

    漏掉的後果是塌狗等級：Keepalive 的 payload 每 tick TypeError，
    吞 50 次後執行緒自殺 → 心跳停 → controller 500ms 清零 → 承重中的狗失力。
    用 AST 驗所有呼叫都是 5 個引數，不靠人眼。
    """
    import ast as _ast
    tree = _ast.parse(Path(m9.__file__).read_text(encoding="utf-8"))
    calls = [n for n in _ast.walk(tree)
             if isinstance(n, _ast.Call)
             and isinstance(n.func, _ast.Name) and n.func.id == "write_frame"]
    assert len(calls) >= 4, "呼叫點數量異常 —— 有人改結構了，重新檢查這個測試"
    for c in calls:
        assert len(c.args) + len(c.keywords) == 5, \
            f"write_frame 在第 {c.lineno} 行只有 {len(c.args)} 個引數"
