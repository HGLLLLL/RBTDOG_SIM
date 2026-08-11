"""gait_export 的離線管線測試。不碰硬體。"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "inference"))
sys.path.insert(0, str(_ROOT / "realbot"))

import calib_map
import cpg_walk_d1 as W
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


def test_worst_margin_detects_a_lower_bound_violation(model):
    """只違反下界的軌跡必須被抓到。

    ⚠️ 不能只靠真實軌跡測這件事：FR/FL、RR/RL 是鏡像腿，leg0.knee 的上界餘裕
       與 leg1.knee 的下界餘裕數值相同，光看上界會意外得到正確答案，
       下界分支整個刪掉也測不出來（審查者實測 11 測試全過）。
    """
    lim = GE.shm_limits(model)
    # 造一條「每一軸都停在自己區間正中央」的軌跡當乾淨底稿
    q = np.zeros((10, 4, 3))
    for leg in range(4):
        for j, jn in enumerate(GE.JN):
            lo, hi = lim[(leg, jn)]
            q[:, leg, j] = (lo + hi) / 2
    clean, _ = GE.worst_margin(q, lim)
    assert clean > 0.1, "底稿本身就該有很大的餘裕"

    # 只把 leg2.hip 壓到下界之外，其他不動
    lo2, _ = lim[(2, "hip")]
    q[5, 2, 1] = lo2 - 0.03
    margin, where = GE.worst_margin(q, lim)
    assert margin == pytest.approx(-0.03, abs=1e-9)
    assert "leg2" in where and "hip" in where and "下界" in where


def test_worst_margin_detects_an_upper_bound_violation(model):
    """對稱的上界版本，確認兩個方向都真的有在看。"""
    lim = GE.shm_limits(model)
    q = np.zeros((10, 4, 3))
    for leg in range(4):
        for j, jn in enumerate(GE.JN):
            lo, hi = lim[(leg, jn)]
            q[:, leg, j] = (lo + hi) / 2
    _, hi3 = lim[(3, "abad")]
    q[7, 3, 0] = hi3 + 0.02
    margin, where = GE.worst_margin(q, lim)
    assert margin == pytest.approx(-0.02, abs=1e-9)
    assert "leg3" in where and "abad" in where and "上界" in where


def test_x_off_shows_up_as_a_real_foot_offset(model):
    """x_off 是足端前後基準偏移（配平機身抬頭），不是可有可無的參數。

    驗法與 build_trajectory 的實作路徑獨立：把關節角逆推回足端位置。
    IK 是 q = HOME3 + jinv @ offset，所以 offset = inv(jinv) @ (q - HOME3)。
    前後擺動項在整個步態週期上平均為 0，所以足端 x 的週期平均 = x_off。
    """
    import cpg_d1
    cfg = W.GAITS[GE.GAIT]
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=20.0)
    _, jinvs = cpg_d1.leg_ik_consts(model)

    for leg in range(4):
        q3 = q_mjcf[:, leg * 3:leg * 3 + 3] - d1_model.HOME3
        offs = q3 @ np.linalg.inv(jinvs[leg]).T        # (N,3) 足端偏移
        mean_x = offs[:, 0].mean()
        # 5e-4：2e-3 太鬆，x_off 被打折成 -0.054 也會過（誤差 1e-3）。
        # 收緊到只容得下數值積分的殘差，讓「小幅度打折」這種隱蔽迴歸也擋得住。
        assert mean_x == pytest.approx(cfg["x_off"], abs=5e-4), (
            f"leg{leg} 足端 x 週期平均 {mean_x:.4f}，應為 x_off={cfg['x_off']}")


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
    assert set(z.files) == {"t", "q_mjcf", "q_shm", "f0s", "jinvs", "meta_json"}
    n = int(2.0 / d1_model.CTRL_DT)
    assert z["t"].shape == (n,)
    assert z["q_mjcf"].shape == (n, 12)
    assert z["q_shm"].shape == (n, 4, 3)
    assert z["f0s"].shape == (4, 3)
    assert z["jinvs"].shape == (4, 3, 3)
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
