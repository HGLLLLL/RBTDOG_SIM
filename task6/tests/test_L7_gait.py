"""L7 的離線核心測試。不碰硬體，不需要狗。"""
import json
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


def test_jog_log_meta_carries_active_legs_so_analysis_skips_dead_legs(tmp_path):
    """jog 的 log 若缺 active_legs，gait_export.analyze() 會 fallback 成四條腿全分析，
    把從未驅動、已知故障的 RR 當成有效資料，產出虛構的追蹤誤差數字。"""
    import json
    import shm_common as SC
    log = {k: np.zeros((5, 4, 3)) for k in ("cmd", "p", "v", "tau")}
    log["t"] = np.arange(5) * SC.DT
    log["overrun"] = np.zeros(5, dtype=bool)
    path = tmp_path / "jog.npz"
    SC.write_log(path, log, meta={"mode": "jog", "leg": 0, "joint": "hip",
                                  "kp": 20.0, "kd": 0.7, "time_scale": 1.0,
                                  "active_legs": [0], "start": [0.0, 0.0, 0.0]})
    meta = json.loads(str(np.load(path, allow_pickle=False)["meta_json"]))
    for k in ("mode", "time_scale", "kp", "kd", "active_legs"):
        assert k in meta, f"log meta 缺 {k}——analyze() 的讀取端依賴它"
    assert meta["active_legs"] == [0]


def test_run_jog_writes_active_legs_and_time_scale_into_log(tmp_path, monkeypatch):
    """直接檢查 run_jog 本體的契約：write_log 收到的 meta 必須帶 active_legs/time_scale。

    不吊真機、不開 SHM——用記憶體版 SplineData，monkeypatch 掉 SC.write_log
    來攔截實際傳入的 meta，並停用 time.sleep 避免 500 Hz 迴圈真的跑很久。
    """
    import time as _time
    import shm_common as SC
    d = SC.SplineData()
    for i in range(4):
        for jn in ("abad", "hip", "knee", "foot"):
            getattr(d.state.legs[i], jn).flags = 1 | (30 << 8) | (44 << 16)
        for jn in ("abad", "hip", "knee"):
            # 初始化成站姿：留 0.0 會落在機構限位上，被行程餘裕檢查正確拒跑
            getattr(d.state.legs[i], jn).p = SC.POSE_STAND[i][jn]
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    captured = {}

    def fake_write_log(path, log, meta):
        captured["meta"] = meta

    monkeypatch.setattr(SC, "write_log", fake_write_log)
    L7.run_jog(d, leg=0, joint_name="hip", kp=20.0, kd=0.7,
               log_path=tmp_path / "jog.npz")
    assert captured["meta"]["active_legs"] == [0]
    assert captured["meta"]["time_scale"] == 1.0


# ============================================================
# 真機執行路徑（_stream / run_gait / main）—— 記憶體版 SplineData，不需要硬體
# ============================================================

@pytest.fixture
def fake_shm(monkeypatch):
    """記憶體版 SplineData + 停用 sleep。讓真機執行路徑可測，不需要硬體。

    ⚠️ state.p 要初始化成【站姿】而不是留 0.0。0.0 在新校正下換算成 MJCF
       會落在機構限位上（例如 hip 的 0.0 → MJCF +2.9558，上限是 +2.967），
       run_jog 的行程餘裕檢查會正確地拒跑，讓測試在一個不真實的前提下失敗。
    """
    import time as _time
    import shm_common as SC
    d = SC.SplineData()
    for i in range(4):
        for jn in ("abad", "hip", "knee", "foot"):
            getattr(d.state.legs[i], jn).flags = 1 | (30 << 8) | (44 << 16)
        for jn in ("abad", "hip", "knee"):
            getattr(d.state.legs[i], jn).p = SC.POSE_STAND[i][jn]
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    return d


def _fresh_log():
    return {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun")}


def test_stream_timestamps_continue_across_segments(fake_shm):
    """一次執行分四段，每段各呼叫一次 _stream。時間戳若用本次呼叫的迴圈索引，
    會每段從 0 重來，事後分析對齊指令時整個錯位。"""
    import shm_common as SC
    log = _fresh_log()
    frames = np.zeros((10, 4, 3))
    for _ in range(3):                      # 模擬三個段落
        ok, why = L7._stream(fake_shm, frames, (0, 1, 3), 20.0, 0.7,
                             8.0, 20.0, log, False, "t")
        assert ok, why
    t = np.asarray(log["t"])
    assert len(t) == 30
    assert np.all(np.diff(t) > 0), "時間戳沒有跨段累積"
    assert t[-1] == pytest.approx(29 * SC.DT)


def test_stream_writes_every_active_leg_not_just_the_last(fake_shm):
    """zero_all 必須在腿迴圈【外】。搬進迴圈的話會抹掉前一條腿剛設好的指令，
    最後只剩最後一條腿有非零增益——而狗會只有一條腿動。"""
    log = _fresh_log()
    frames = np.full((1, 4, 3), 0.3)
    ok, _ = L7._stream(fake_shm, frames, (0, 1, 3), 20.0, 0.7,
                       8.0, 20.0, log, False, "t")
    assert ok
    for i in (0, 1, 3):
        for jn in ("abad", "hip", "knee"):
            j = getattr(fake_shm.cmd.legs[i], jn)
            assert j.kp == pytest.approx(20.0), f"leg{i}.{jn} 沒有被設定"
            assert j.p_des == pytest.approx(0.3)
    # 被跳過的 RR 必須全零
    for jn in ("abad", "hip", "knee", "foot"):
        assert getattr(fake_shm.cmd.legs[2], jn).kp == 0.0


def test_stream_stops_immediately_when_a_guard_trips(fake_shm):
    """保護觸發必須立刻 return，不能繼續送指令。"""
    log = _fresh_log()
    fake_shm.state.legs[1].knee.t = 99.0          # 遠超力矩門檻
    frames = np.zeros((50, 4, 3))
    ok, why = L7._stream(fake_shm, frames, (0, 1, 3), 20.0, 0.7,
                         8.0, 20.0, log, False, "t")
    assert not ok
    assert "FL" in why and "knee" in why
    assert len(log["t"]) < 5, "保護觸發後還繼續跑了很多週期"


def test_stream_records_one_log_row_per_cycle(fake_shm):
    log = _fresh_log()
    frames = np.zeros((7, 4, 3))
    L7._stream(fake_shm, frames, (0,), 20.0, 0.7, 8.0, 20.0, log, False, "t")
    for k in ("t", "cmd", "p", "v", "tau", "overrun"):
        assert len(log[k]) == 7, k


def test_run_gait_ramps_gain_in_during_catch(fake_shm, npz, monkeypatch):
    """接住階段必須從 0 漸入增益。凍結 mc_ctrl 後腿會因重力垂下，
    直接滿增益接住會有力矩突跳、可能撞機構限位。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    seen = []
    orig = L7._stream

    def spy(d, frames, active, kp, kd, ta, va, log, dry, label):
        seen.append((label, kp))
        return orig(d, frames, active, kp, kd, ta, va, log, dry, label)

    monkeypatch.setattr(L7, "_stream", spy)
    traj, meta = L7.load_trajectory(npz)
    L7.run_gait(fake_shm, traj[:5], meta, (0, 1, 3), 1.0, 20.0, 0.7,
                dry=False, log_path=None)
    catch_kps = [kp for label, kp in seen if label == "接住"]
    assert catch_kps, "沒有接住階段"
    assert min(catch_kps) == pytest.approx(0.0), "接住沒有從 0 開始漸入"
    assert max(catch_kps) == pytest.approx(20.0), "接住沒有升到設定增益"


def test_run_gait_disarms_the_legs_when_a_guard_trips(fake_shm, npz, monkeypatch):
    """保護觸發必須卸力。不卸力的話腿會維持在觸發當下的指令上硬撐。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    called = []
    monkeypatch.setattr(SC, "passive_stop",
                        lambda d, legs, cycles=1500, stop_kd=3.0: called.append(cycles))
    fake_shm.state.legs[0].hip.t = 99.0
    traj, meta = L7.load_trajectory(npz)
    r = L7.run_gait(fake_shm, traj[:5], meta, (0, 1, 3), 1.0, 20.0, 0.7,
                    dry=False, log_path=None)
    assert r is False
    assert called, "保護觸發卻沒有卸力"


def test_run_gait_records_the_actual_mode_not_always_gait(fake_shm, npz, monkeypatch):
    """--mode leg 呼叫 run_gait 時，log meta 的 mode 必須記成 'leg'，不能恆為 'gait'
    ——不然事後分析分不清這次到底驅動了幾條腿是刻意的還是步態模式本來就這樣。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    captured = {}

    def fake_write_log(path, log, meta):
        captured["meta"] = meta

    monkeypatch.setattr(SC, "write_log", fake_write_log)
    traj, meta = L7.load_trajectory(npz)
    ok = L7.run_gait(fake_shm, traj[:5], meta, (1,), 1.0, 20.0, 0.7,
                     dry=False, log_path="ignored.npz", mode="leg")
    assert ok
    assert captured["meta"]["mode"] == "leg"


def test_confirm_requires_root(monkeypatch, capsys):
    """--confirm 會驅動真實關節，必須是 root。這道檢查失效的話，
    非 root 執行會一路走到 open_shm 才失敗，錯誤訊息不明確。"""
    import os
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "jog", "--confirm"])
    with pytest.raises(SystemExit):
        L7.main()
    assert "root" in capsys.readouterr().out


def test_leg_mode_dry_run_previews_like_gait_mode(npz, monkeypatch, capsys):
    """leg 模式是『先驗證單腿、再上三腿』的降風險步驟，卻是唯一沒辦法先排練的
    真機路徑——main() 只在 mode=='gait' 時做 dry-run 預覽，leg 模式被漏掉了。"""
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "leg", "--only-leg", "1",
                         "--traj", str(npz), "--time-scale", "1.0", "--secs", "1.0"])
    L7.main()
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "保護門檻" in out, "leg 模式的 dry-run 沒有走到 run_gait 的預覽輸出"


def test_gait_mode_dry_run_catch_stage_prints_one_summary_line(npz, monkeypatch, capsys):
    """接住階段在 dry-run 下必須只印一行摘要，不能因為 kp_seq 逐比例呼叫 _stream
    而洗出 251 行重複訊息（CATCH_SEC=0.5s @ 500Hz → 251 個比例）。"""
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "gait", "--skip-legs", "2",
                         "--traj", str(npz), "--time-scale", "1.0", "--secs", "1.0"])
    L7.main()
    out = capsys.readouterr().out
    catch_idx = out.index("[*] 接住")
    next_stage_idx = out.index("[*] 到起始姿")
    catch_block = out[catch_idx:next_stage_idx]
    # 接住段內只允許一行帶「接住」字樣的細節列印（外加上面的 [*] 接住 標頭）
    detail_lines = [ln for ln in catch_block.splitlines() if ln.strip().startswith("[接住]")]
    assert len(detail_lines) == 1, f"接住階段印了 {len(detail_lines)} 行細節，應為 1 行"


def test_only_leg_out_of_range_is_rejected_at_cli(npz, monkeypatch, capsys):
    """打錯 --only-leg 不能一路跑到 d.state.legs[i] 才丟 ctypes IndexError
    ——且那只在 --confirm 真機路徑才現形。比照 --skip-legs 在 CLI 階段就擋下來。"""
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "leg", "--only-leg", "7",
                         "--traj", str(npz)])
    with pytest.raises(SystemExit):
        L7.main()
    assert "0~3" in capsys.readouterr().err


def test_jog_leg_out_of_range_is_rejected_at_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "jog", "--jog-leg", "-1"])
    with pytest.raises(SystemExit):
        L7.main()
    assert "0~3" in capsys.readouterr().err


# ============================================================
# 修正 1（Critical）：run_jog 必須在 MJCF 空間下指令，不是 SHM 空間
# ============================================================

@pytest.mark.parametrize("leg,joint,jidx", [(0, "hip", 1), (1, "hip", 1),
                                            (0, "abad", 0), (3, "knee", 2)])
def test_jog_commands_in_mjcf_space_so_the_manual_criterion_holds(
        fake_shm, monkeypatch, leg, joint, jidx):
    """驗號是唯一能抓到 sign 錯誤的關卡，判準必須與手冊的 MJCF 正向定義同一個空間。

    jog 在 MJCF 空間下 +0.10 rad，所以 SHM 指令要往 sign 的方向偏離起點：
    sign=-1 的關節往負、sign=+1 的往正。若 jog 直接在 SHM 空間加正值，
    sign=-1 的關節在校正正確時會讓操作者看到相反的方向，進而把對的校正改壞。

    ⚠️ 不寫死某一顆關節的 sign——2026-08-12 重新校正時 leg0.hip 從 -1 變成 +1，
       寫死會讓測試在校正變動時失敗而非在真的有 bug 時失敗。這裡涵蓋
       sign=-1 與 sign=+1 兩種情形。
    """
    import calib_map
    import shm_common as SC
    s, _o = calib_map.CALIB[leg][joint]

    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    start_shm = SC.POSE_STAND[leg][joint]
    setattr(getattr(fake_shm.state.legs[leg], joint), "p", start_shm)

    sent = []
    orig = L7._stream

    def spy(d, frames, active, kp, kd, ta, va, log, dry, label):
        sent.extend(np.asarray(frames)[:, leg, jidx].tolist())
        return orig(d, frames, active, kp, kd, ta, va, log, dry, label)

    monkeypatch.setattr(L7, "_stream", spy)
    L7.run_jog(fake_shm, leg, joint, 20.0, 0.7, log_path=None)

    # SHM 指令方向 = sign × jog 實際選用的 MJCF 方向。
    # 方向由 pick_jog_dir 自動選（行程較充裕的一邊），所以這裡也用同一支函式
    # 算出預期值——不要另外寫死，否則測的是「兩份寫死值一不一樣」而非行為。
    mjcf0 = (start_shm - calib_map.CALIB[leg][joint][1]) / s
    d_mjcf, _room = L7.pick_jog_dir(leg, joint, mjcf0, 0.10)
    expect = s * d_mjcf
    sent = np.asarray(sent)
    if expect < 0:
        assert sent.min() < start_shm - 0.09, "指令沒有往預期的 SHM 負向走"
        assert sent.max() < start_shm + 0.01, "指令往正向跑了，座標空間搞反"
    else:
        assert sent.max() > start_shm + 0.09, "指令沒有往預期的 SHM 正向走"
        assert sent.min() > start_shm - 0.01, "指令往負向跑了，座標空間搞反"
    # 幅度恆為 0.10 rad（sign 與方向都只改正負，不改大小）
    assert (sent.max() - sent.min()) == pytest.approx(0.10, abs=5e-3)


def test_jog_returns_to_the_starting_angle(fake_shm, monkeypatch):
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    start_shm = SC.POSE_STAND[0]["knee"]
    fake_shm.state.legs[0].knee.p = start_shm
    sent = []
    orig = L7._stream
    monkeypatch.setattr(L7, "_stream", lambda d, f, a, kp, kd, ta, va, lg, dry, lb: (
        sent.extend(np.asarray(f)[:, 0, 2].tolist()) or orig(d, f, a, kp, kd, ta, va, lg, dry, lb)))
    L7.run_jog(fake_shm, 0, "knee", 20.0, 0.7, log_path=None)
    # abs=1e-6 而非更緊：JointState.p 是 ctypes c_float（單精度），寫入時
    # 就已經把 start_shm 從 python float64 截成 float32，讀回來本身就帶著
    # ~1e-8 rad 的截斷差；1e-9 是量錯了精度，不是量到真的漂移。
    assert sent[0] == pytest.approx(start_shm, abs=1e-6)
    assert sent[-1] == pytest.approx(start_shm, abs=1e-6)


@pytest.mark.parametrize("leg", [0, 1, 3])
def test_jog_works_for_every_leg_g1_needs_to_check(fake_shm, monkeypatch, leg):
    """G1 要對 leg 0/1/3 各驗一次——CALIB 的 abad 號不是左右鏡像而是對角
    （FR/RL +1，FL/RR -1），不能只驗過 leg0 就假設其他腿同理。leg2(RR)
    整條已失聯，不驗。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    ok = L7.run_jog(fake_shm, leg=leg, joint_name="abad", kp=20.0, kd=0.7,
                    log_path=None)
    assert ok is True


def test_jog_mode_has_a_dry_run_preview(monkeypatch, capsys):
    """第三條 dry-run 命令之前沒有預覽路徑——跑真機前至少要能先看到這條
    路徑不會馬上炸掉，理由跟當初補 leg 模式 dry-run 一樣。"""
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "jog",
                         "--jog-leg", "0", "--jog-joint", "hip"])
    L7.main()
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "jog 計畫" in out
    assert "sign=" in out


# ============================================================
# 修正 2：保護觸發／例外／Ctrl-C 都要留下 log
# ============================================================

def test_run_gait_writes_aborted_log_with_reason_when_a_guard_trips(
        fake_shm, npz, monkeypatch, tmp_path):
    """保護觸發不能把整份 log 丟掉——3 分鐘操作窗口內，觸發前那幾個週期的
    cmd/p/v/tau 是診斷價值最高的一次。

    保護門檻要在跑了幾個週期之後才觸發（不是一開始就壞），才能驗到「觸發
    前的資料真的被留下來」而不是巧合地一筆都還沒寫。
    """
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    calls = {"n": 0}

    def flaky_guard(d, active_legs, torque_abort, vel_abort):
        calls["n"] += 1
        if calls["n"] > 20:
            return False, "FR.hip 力矩 99.00 > 20.00"
        return True, ""

    monkeypatch.setattr(SC, "check_guards", flaky_guard)
    traj, meta = L7.load_trajectory(npz)
    log_path = tmp_path / "run.npz"
    r = L7.run_gait(fake_shm, traj[:5], meta, (0, 1, 3), 1.0, 20.0, 0.7,
                    dry=False, log_path=log_path)
    assert r is False

    aborted = tmp_path / "run_ABORTED.npz"
    assert aborted.exists(), "保護觸發後 log 檔沒有留下來"
    z = np.load(aborted, allow_pickle=False)
    m = json.loads(str(z["meta_json"]))
    assert m["aborted"] is True
    assert "FR" in m["abort_reason"] and "hip" in m["abort_reason"]
    assert m["aborted_stage"] == "接住"
    assert len(z["t"]) > 0, "應該至少留下觸發前那幾筆"
    assert len(z["stage"]) == len(z["t"])


def test_run_gait_writes_aborted_log_on_keyboard_interrupt(
        fake_shm, npz, monkeypatch, tmp_path):
    """Ctrl-C 中止也不能把 log 丟掉，且要重新拋出讓上層知道發生了中斷。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(L7, "_stream", boom)
    traj, meta = L7.load_trajectory(npz)
    log_path = tmp_path / "run.npz"
    with pytest.raises(KeyboardInterrupt):
        L7.run_gait(fake_shm, traj[:5], meta, (0, 1, 3), 1.0, 20.0, 0.7,
                    dry=False, log_path=log_path)

    aborted = tmp_path / "run_ABORTED.npz"
    assert aborted.exists()
    m = json.loads(str(np.load(aborted, allow_pickle=False)["meta_json"]))
    assert m["aborted"] is True
    assert "Ctrl" in m["abort_reason"] or "中止" in m["abort_reason"]


def test_run_jog_writes_aborted_log_on_guard_trip(fake_shm, monkeypatch, tmp_path):
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    fake_shm.state.legs[0].hip.t = 99.0   # 遠超 jog 的保護門檻 8.0 N·m
    log_path = tmp_path / "jog.npz"
    ok = L7.run_jog(fake_shm, leg=0, joint_name="hip", kp=20.0, kd=0.7,
                    log_path=log_path)
    assert ok is False
    aborted = tmp_path / "jog_ABORTED.npz"
    assert aborted.exists()
    m = json.loads(str(np.load(aborted, allow_pickle=False)["meta_json"]))
    assert m["aborted"] is True
    assert "hip" in m["abort_reason"]


# ============================================================
# 修正 3：log 要有階段標記，meta 要有 source，secs 要是實際播放秒數
# ============================================================

def test_run_gait_log_stage_field_marks_each_phase(fake_shm, npz, monkeypatch, tmp_path):
    """stage 沒有標記的話，事後分析會把接住／ramp 段混進『播放步態』的統計，
    稀釋 RMS（總審實測 1.0× 配 --secs 5 稀釋 27%）。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    traj, meta = L7.load_trajectory(npz)
    log_path = tmp_path / "run.npz"
    r = L7.run_gait(fake_shm, traj[:5], meta, (0, 1, 3), 1.0, 20.0, 0.7,
                    dry=False, log_path=log_path)
    assert r is True

    z = np.load(log_path, allow_pickle=False)
    stage = z["stage"]
    assert stage.dtype == np.int8
    assert len(stage) == len(z["t"])
    assert set(np.unique(stage).tolist()) == {0, 1, 2, 3}, \
        "四段（接住/到起始姿/播放步態/回站姿）都要出現在 stage 裡"
    play_idx = np.where(stage == 2)[0]
    assert play_idx.size > 0
    assert stage[play_idx[0] - 1] == 1, "播放步態前應緊接著到起始姿"
    assert stage[play_idx[-1] + 1] == 3, "播放步態後應緊接著回站姿"


def test_run_gait_log_meta_has_source_and_actual_played_secs(
        fake_shm, npz, monkeypatch, tmp_path):
    """G3-live 的通過條件是『追蹤誤差與同一倍速的 --source file 一致』，log
    沒記 source 就事後分不出哪份是哪份；meta['secs'] 之前被 **meta 帶進來
    的是軌跡全長（本 fixture 是 6.0），不是這次實際播的秒數。"""
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    traj, meta = L7.load_trajectory(npz)
    assert meta["secs"] == pytest.approx(6.0)   # npz fixture 的軌跡全長

    n = int(1.0 / meta["ctrl_dt"])              # 模擬 --secs 1.0 的截斷
    log_path = tmp_path / "run.npz"
    r = L7.run_gait(fake_shm, traj[:n], meta, (0, 1, 3), 1.0, 20.0, 0.7,
                    dry=False, log_path=log_path, source="live")
    assert r is True

    m = json.loads(str(np.load(log_path, allow_pickle=False)["meta_json"]))
    assert m["source"] == "live"
    assert m["secs"] == pytest.approx(1.0, abs=0.05), \
        "meta['secs'] 沿用了軌跡全長，沒有換成這次實際播放的秒數"
    assert m["secs"] != pytest.approx(6.0)


def test_run_jog_log_meta_source_is_jog(fake_shm, monkeypatch, tmp_path):
    import shm_common as SC
    monkeypatch.setattr(SC, "preflight_mc_stopped", lambda d: (True, 0))
    log_path = tmp_path / "jog.npz"
    L7.run_jog(fake_shm, leg=0, joint_name="hip", kp=20.0, kd=0.7,
              log_path=log_path)
    m = json.loads(str(np.load(log_path, allow_pickle=False)["meta_json"]))
    assert m["source"] == "jog"


def test_log_default_filename_encodes_mode_source_scale(npz, monkeypatch):
    """--log 不給時，預設檔名要能區分不同執行，不能每次都覆蓋同一個
    l7_log.npz（G3-live 要跟同一倍速的 --source file 比對，第二次執行
    才不會把第一次的 log 蓋掉）。"""
    captured = {}

    def fake_run_gait(d, traj, meta, active, time_scale, kp, kd, dry, log_path,
                      mode="gait", source="file"):
        captured["log_path"] = str(log_path)
        return True

    monkeypatch.setattr(L7, "run_gait", fake_run_gait)
    monkeypatch.setattr(sys, "argv",
                        ["L7_gait_shm.py", "--mode", "gait", "--skip-legs", "2",
                         "--traj", str(npz), "--time-scale", "0.5",
                         "--secs", "1.0", "--source", "file"])
    L7.main()
    name = captured["log_path"]
    assert name != "l7_log.npz"
    assert "gait" in name and "file" in name and "0.5" in name
