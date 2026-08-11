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
