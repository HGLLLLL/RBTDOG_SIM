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

    # 找一個 sign=-1 的關節來驗上下界對調。不寫死是哪一顆——2026-08-12 重新校正
    # 之後 leg0.knee 從 -1 變成 +1，寫死會讓這個測試在校正變動時失敗而非在
    # 真的有 bug 時失敗。
    neg = [(leg, jn) for leg in range(4) for jn in GE.JN
           if calib_map.CALIB[leg][jn][0] == -1]
    assert neg, "沒有任何 sign=-1 的關節，這個測試就失去意義了"
    MJCF_RANGE = {"abad": (-0.4687, +0.4687), "hip": (-1.1320, +2.9470),
                  "knee": (-2.7030, -0.6220)}      # ctrlrange，四條腿相同
    for leg, jn in neg:
        s, o = calib_map.CALIB[leg][jn]
        mjcf_lo, mjcf_hi = MJCF_RANGE[jn]
        lo, hi = lim[(leg, jn)]
        # sign=-1：MJCF 下界映到 SHM 上界
        assert hi == pytest.approx(s * mjcf_lo + o, abs=1e-6), f"leg{leg}.{jn}"
        assert lo == pytest.approx(s * mjcf_hi + o, abs=1e-6), f"leg{leg}.{jn}"


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


def test_calib_hash_changes_when_leg_order_changes(monkeypatch):
    """腿序重排也是校正的一部分。它改了但雜湊沒變，等於拒跑機制對
    「四條腿指令互換」這種最嚴重的錯誤完全失明。"""
    h0 = GE.calib_hash()
    monkeypatch.setattr(calib_map, "LEG_MJCF2SHM", [0, 1, 2, 3])
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
    """起步位移決定 ramp 時間。

    2026-08-12 重新校正後從 0.44 變成 0.87 rad——因為舊校正把「實機站姿」
    當成 MJCF home，新校正實測出兩者差 knee 34°、hip 20°。位移變大是正確的，
    ramp 時間會跟著從 2.0s 拉到約 3.5s（0.87/0.25）。

    上界 1.5 rad 是安全性斷言而非精度斷言：超過那個量代表校正又跑掉了，
    L7 會用很長的時間把腿拉到一個很遠的地方，是先前 Critical 的失效樣態。
    """
    q_mjcf, q_shm = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=2.0)
    _, _, stats = GE.run_checks(model, q_mjcf, q_shm)
    assert 0.5 < stats["start_offset_from_stand"] < 1.5


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


def test_air_servo_sim_keeps_the_base_pinned_and_contacts_off(model):
    """吊掛空跑的兩個前提：機身固定（模擬吊具）、接觸關閉。
    這兩件事若失效，量到的力矩就不是吊掛情境的力矩——但力矩本身的變化
    小到會被 20% 容忍帶蓋掉，所以直接斷言不變量。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=3.0)
    r = GE.air_servo_sim(model, q_mjcf, kp=20.0, kd=0.7, time_scale=1.0)
    assert r["base_drift"] == pytest.approx(0.0, abs=1e-12), "機身沒有被固定住"
    assert r["max_contacts"] == 0, "接觸沒有被關掉"


def test_air_sim_table_does_not_leak_diagnostics_into_npz(model):
    """診斷欄位不進 npz——schema 已被測試與既有的 gait_walk_stable.npz 釘住。
    per_axis 是新加的必留欄位（逐軸預測值的來源），base_drift/max_contacts 仍要濾掉。"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=3.0)
    table = GE.air_sim_table(model, q_mjcf)
    for gains, per_scale in table.items():
        for scale, entry in per_scale.items():
            assert set(entry) == {"tau_peak", "err_peak_deg", "err_rms_deg",
                                  "vel_peak", "per_axis"}, f"{gains} {scale}"
            assert set(entry["per_axis"]) == {"rms_deg", "peak_deg"}
            assert np.array(entry["per_axis"]["rms_deg"]).shape == (4, 3)
            assert np.array(entry["per_axis"]["peak_deg"]).shape == (4, 3)


def test_mjcf_to_shm_per_axis_reorders_legs_only(model):
    """逐軸預測值存進 npz 前要換成 SHM 腿序，關節序 (abad,hip,knee) 不變。

    這是總審點名的索引順序問題：air_servo_sim 在 MJCF 腿序 (FL,FR,RL,RR) 工作，
    log 是 SHM 腿序，兩者對不上就會把某條腿的誤差錯記到另一條腿頭上。"""
    arr = np.arange(12, dtype=float).reshape(4, 3)   # mjcf 腿序，每列 = [abad,hip,knee]
    out = GE._mjcf_to_shm_per_axis(arr)
    assert out.shape == (4, 3)
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        assert np.array_equal(out[shm_leg], arr[mjcf_leg]), (
            f"mjcf_leg{mjcf_leg} 沒有正確映射到 shm_leg{shm_leg}")


def test_air_servo_sim_per_axis_rms_is_consistent_with_the_aggregate(model):
    """逐軸 RMS 與既有的 12 軸 aggregate RMS 必須是同一組底層誤差算出來的——
    腿序重排不改變數值集合，只改變哪個位置是哪條腿，所以兩者換算後要相等。
    （腿序方向本身由 test_mjcf_to_shm_per_axis_reorders_legs_only 釘住。）"""
    q_mjcf, _ = GE.build_trajectory(model, GE.DEPLOY_G_C, secs=3.0)
    r = GE.air_servo_sim(model, q_mjcf, kp=20.0, kd=0.7, time_scale=1.0)
    per_axis_rad = np.radians(np.array(r["per_axis"]["rms_deg"]))
    recombined = float(np.degrees(np.sqrt((per_axis_rad ** 2).mean())))
    assert recombined == pytest.approx(r["err_rms_deg"], rel=1e-6)


def test_export_embeds_the_air_sim_table(model, tmp_path):
    import json
    out = GE.export(model, tmp_path / "g.npz", secs=8.0)
    meta = json.loads(str(np.load(out, allow_pickle=False)["meta_json"]))
    table = meta["air_sim"]
    assert "20.0/0.7" in table
    for s in ("0.25", "0.5", "1.0"):
        entry = table["20.0/0.7"][s]
        assert set(entry) == {"tau_peak", "err_peak_deg", "err_rms_deg",
                              "vel_peak", "per_axis"}


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


def test_analyze_reports_no_lag_for_a_motionless_axis(tmp_path):
    """本步態的 abad 依設計恆定不動（mu_y=1.5 → fy=0）。
    靜止軸的相位延遲沒有定義，必須回報 None 而不是一個假數字。"""
    r = GE.analyze(_synthetic_log(tmp_path, lag_steps=10))
    assert r["axes"][(0, "abad")]["lag_ms"] is None
    assert r["axes"][(0, "hip")]["lag_ms"] == pytest.approx(20.0, abs=2.0)


def test_analyze_does_not_invent_lag_from_rounding_noise(tmp_path):
    """靜止軸不能因為浮點噪訊被挑出假位移。這是原演算法的實際失效模式
    （實測對完全靜止的訊號給出 220 ms）。"""
    import json
    n, dt = 2000, 1.0 / 500
    t = np.arange(n) * dt
    cmd = np.zeros((n, 4, 3))
    cmd[:, :, 1] = 0.3                      # 完全不動，但不是 0
    p = cmd + 1e-9 * np.sin(np.arange(n))[:, None, None]   # 只有量化級別的噪訊
    path = tmp_path / "flat.npz"
    np.savez(path, t=t, cmd=cmd, p=p, v=np.zeros((n, 4, 3)),
             tau=np.zeros((n, 4, 3)), overrun=np.zeros(n, dtype=bool),
             meta_json=np.array(json.dumps(
                 {"mode": "gait", "time_scale": 1.0, "active_legs": [0]})))
    r = GE.analyze(path)
    for jn in GE.JN:
        assert r["axes"][(0, jn)]["lag_ms"] is None, f"{jn} 憑空生出延遲"


def test_analyze_prints_without_crashing_on_none_lag(tmp_path, capsys):
    """列印路徑要能處理 None，不能 format 炸掉。"""
    GE.analyze(_synthetic_log(tmp_path))
    out = capsys.readouterr().out
    assert "abad" in out


# ---------------------------------------------------------------------------
# 修正 1：統計視窗只看 stage==2（播放步態），接住/ramp/回站姿不該混進 RMS。
# ---------------------------------------------------------------------------

def _synthetic_log_with_stages(tmp_path, err_by_stage, lag_steps=0,
                                active_legs=(0, 1, 3), fname="log_stage.npz"):
    """依 stage 分四等分（各占 n//4）合成 log，每段可以指定各自的固定誤差。"""
    import json
    n, dt = 2000, 1.0 / 500
    t = np.arange(n) * dt
    cmd = np.zeros((n, 4, 3))
    cmd[:, :, 1] = np.sin(2 * np.pi * 1.4 * t)[:, None]
    stage = np.zeros(n, dtype=np.int8)
    quarter = n // 4
    for s in range(4):
        stage[s * quarter:(s + 1) * quarter] = s
    err = np.zeros(n)
    for s, e in err_by_stage.items():
        err[stage == s] = e
    p = np.roll(cmd, lag_steps, axis=0) + err[:, None, None]
    p[:lag_steps] = cmd[:lag_steps]
    path = tmp_path / fname
    np.savez(path, t=t, cmd=cmd, p=p, v=np.zeros((n, 4, 3)),
             tau=np.zeros((n, 4, 3)), overrun=np.zeros(n, dtype=bool), stage=stage,
             meta_json=np.array(json.dumps({"mode": "gait", "time_scale": 1.0,
                                            "active_legs": list(active_legs),
                                            "source": "live"})))
    return path


def test_analyze_windows_stats_to_stage2_only(tmp_path, capsys):
    """接住/ramp/回站姿段的大誤差不能稀釋播放段的 RMS——這是總審抓到的
    『手冊預設配置下播放段只佔 53%，RMS 被稀釋 27%』那題。"""
    path = _synthetic_log_with_stages(
        tmp_path, {0: np.radians(30.0), 1: np.radians(30.0),
                   2: np.radians(3.0), 3: np.radians(30.0)})
    r = GE.analyze(path)
    assert r["axes"][(0, "hip")]["rms_deg"] == pytest.approx(3.0, abs=0.05), (
        "RMS 被非播放段稀釋了——沒有只窗到 stage==2")
    assert r["n_total"] == 2000
    assert r["n_play"] == 500
    out = capsys.readouterr().out
    assert "播放步態 500" in out and "25.0%" in out


def test_analyze_falls_back_to_whole_log_without_stage_field(tmp_path, capsys):
    """舊 log 沒有 stage 欄位：退回整份分析，且要印明顯警告，不能悄悄算錯。"""
    r = GE.analyze(_synthetic_log(tmp_path, err_rad=np.radians(3.0)))
    assert r["n_total"] == r["n_play"] == 2000
    out = capsys.readouterr().out
    assert "⚠️" in out and "舊格式" in out


# ---------------------------------------------------------------------------
# 修正 2：靜止軸的假延遲——ramp 段的極小位移不能打敗播放段的靜止判定。
# ---------------------------------------------------------------------------

def test_analyze_windowed_lag_ignores_ramp_drift_on_static_axis(tmp_path):
    """POSE_STAND 與軌跡起點的 abad 差約 1e-4 rad——ramp 段(stage 1)因此有個
    1e-4 的爬升，但播放段(stage 2)依設計完全不動。統計視窗沒切到播放段的話，
    這個 1e-4 的 ptp 會打敗 LAG_CONST_EPS，讓每條腿各自因浮點捨入雜訊挑出
    不同的假延遲（總審實測 leg0=80ms／leg1=—／leg3=0.0，三條腿三個答案）。"""
    import json
    n, dt = 2000, 1.0 / 500
    t = np.arange(n) * dt
    cmd = np.zeros((n, 4, 3))
    stage = np.zeros(n, dtype=np.int8)
    quarter = n // 4
    for s in range(4):
        stage[s * quarter:(s + 1) * quarter] = s
    ramp = np.linspace(0.0, 1e-4, quarter)          # 只有 ramp 段(stage 1)的 abad 在爬升
    cmd[quarter:2 * quarter, :, 0] = ramp[:, None]
    p = cmd.copy()                                    # 完美追蹤：只測窗口切得對不對
    path = tmp_path / "ramp_drift.npz"
    np.savez(path, t=t, cmd=cmd, p=p, v=np.zeros((n, 4, 3)),
             tau=np.zeros((n, 4, 3)), overrun=np.zeros(n, dtype=bool), stage=stage,
             meta_json=np.array(json.dumps({"mode": "gait", "time_scale": 1.0,
                                            "active_legs": [0, 1, 3]})))
    r = GE.analyze(path)
    for leg in (0, 1, 3):
        assert r["axes"][(leg, "abad")]["lag_ms"] is None, (
            f"leg{leg} abad 從 ramp 段的 1e-4 位移憑空生出延遲")


# ---------------------------------------------------------------------------
# 修正 3：--traj 查表印逐軸預測值，且必須是逐軸而不是拿 aggregate 硬比。
# ---------------------------------------------------------------------------

def _synthetic_log_with_gains(tmp_path, kp, kd, time_scale, active_legs=(0, 1, 3),
                              fname="log_gains.npz"):
    import json
    n, dt = 2000, 1.0 / 500
    t = np.arange(n) * dt
    cmd = np.zeros((n, 4, 3))
    cmd[:, :, 1] = np.sin(2 * np.pi * 1.4 * t)[:, None]
    p = cmd.copy()
    path = tmp_path / fname
    np.savez(path, t=t, cmd=cmd, p=p, v=np.zeros((n, 4, 3)),
             tau=np.zeros((n, 4, 3)), overrun=np.zeros(n, dtype=bool),
             meta_json=np.array(json.dumps({"mode": "gait", "time_scale": time_scale,
                                            "kp": kp, "kd": kd,
                                            "active_legs": list(active_legs)})))
    return path


def test_analyze_traj_arg_looks_up_predicted_per_axis_rms(model, tmp_path, capsys):
    """--traj 給了才印預測欄；查表鍵是 log meta 的 kp/kd/time_scale，
    印出來的必須是逐軸預測，不是拿 12 軸 aggregate 硬比。"""
    import json
    traj = GE.export(model, tmp_path / "traj.npz", secs=3.0)
    traj_meta = json.loads(str(np.load(traj, allow_pickle=False)["meta_json"]))
    log = _synthetic_log_with_gains(tmp_path, kp=20.0, kd=0.7, time_scale=1.0)
    r = GE.analyze(log, traj_path=traj)
    assert r["predicted"] is not None
    expected = traj_meta["air_sim"]["20.0/0.7"]["1.0"]["per_axis"]
    assert r["predicted"] == expected
    out = capsys.readouterr().out
    assert "預測RMS" in out


def test_analyze_without_traj_arg_has_no_predicted_column(tmp_path, capsys):
    """沒給 --traj 就只印量測值，不能無中生有印出一欄。"""
    r = GE.analyze(_synthetic_log(tmp_path))
    assert r["predicted"] is None
    out = capsys.readouterr().out
    assert "預測RMS" not in out


def test_analyze_traj_arg_warns_when_gains_not_in_table(model, tmp_path, capsys):
    """log 的 kp/kd 不在 npz 的 air_sim 表裡：印警告，不當掉，量測值照印。"""
    traj = GE.export(model, tmp_path / "traj.npz", secs=3.0)
    log = _synthetic_log_with_gains(tmp_path, kp=999.0, kd=1.0, time_scale=1.0)
    r = GE.analyze(log, traj_path=traj)
    assert r["predicted"] is None
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "hip" in out, "查無預測值時，量測值仍要照印"


# ---------------------------------------------------------------------------
# 修正 4：--analyze 要能在沒有 mujoco 的車載電腦上跑。
# ---------------------------------------------------------------------------

def test_analyze_works_without_mujoco(tmp_path):
    """--analyze 要能在機器狗的車載電腦上跑，那台沒有 mujoco。"""
    import subprocess
    import sys as _sys
    log = _synthetic_log(tmp_path)
    code = (
        "import sys;"
        "sys.modules['mujoco'] = None;"      # 讓 import mujoco 失敗
        "sys.path.insert(0, %r); sys.path.insert(0, %r);"
        "import gait_export as GE;"
        "r = GE.analyze(%r);"
        "print('OK', len(r['axes']))"
        % (str(_ROOT / "inference"), str(_ROOT / "realbot"), str(log))
    )
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert "OK" in r.stdout, f"analyze 需要 mujoco：{r.stdout!r} {r.stderr!r}"
