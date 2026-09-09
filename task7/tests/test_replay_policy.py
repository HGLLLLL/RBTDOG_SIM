"""replay_policy：狗端堆疊回放實機 log 必須與本機堆疊一致；CPG 重建與 GaitStream 同步。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
import policy_np as pn  # noqa: E402
import replay_policy as rp  # noqa: E402

LOG = ROOT / "logs" / "m_logs_trip17" / "M9_20260903_170013.json"


@pytest.mark.skipif(not LOG.exists(), reason="trip17 log 不在")
def test_trip17_replay_dog_stack_matches_local_stack():
    import local_infer_max as li
    pol = pn.load(str(ROOT / "weights" / "cpg_rl_max_v2_3_np.npz"))
    infer = li.load_policy(str(ROOT / "weights" / "cpg_rl_max_v2_3_params.pkl"), act_dim=10, head=False)
    r = rp.replay(str(LOG), pol, infer, np.array([0.30, 0.0]))
    assert r["n"] > 1000
    assert r["worst_obs"].max() < 1e-5
    assert r["worst_act"] < 1e-4
    # 走路段的重力向量應該接近 (0,0,−1)：z 分量平均 < −0.95
    assert r["obs"][:, 2].mean() < -0.95
    txt = rp.report([r], np.array([0.30, 0.0]))
    assert "全部通過" in txt


def test_cpg_reconstruction_matches_gait_stream():
    """重建的 CPG 狀態要和 M9 GaitStream 在同一時刻的推進前狀態一致。"""
    import json
    import coord
    import cpg
    import M9_gait as m9
    D = json.loads((ROOT / "outputs" / "A_kp250_walk.json").read_text(encoding="utf-8"))
    p = dict(D["params"], mu_x=D["baseline_ref"]["mu_x"], mu_y=D["baseline_ref"]["mu_y"],
             d_step_y=D["baseline_ref"]["d_step_y"])
    gs = m9.GaitStream(p, cpg.home_foot(coord.POSES["home"]), cpg.knee_signs(coord.POSES["home"]))

    # ⚠️ 取樣時刻刻意**不落在 20 ms 整數邊界**：邊界上浮點兩邊都可能歧義
    #   （GaitStream 累加的 t_next 是 1.0000000000000004）。實機 log 的 t 相對 t_gait0
    #   （Enter 按下的 monotonic 時刻）本來就不會剛好在邊界；歧義最多 5 ms = 1/4 步。
    OFF = 0.0025
    class R:
        t = np.array([10.0 + OFF + k / 200 for k in range(400)])
    idx = np.arange(400)
    cs = rp.cpg_states_for(R, idx, p)
    # GaitStream 在 t 時 c 是「推進到 t 為止」的狀態；重建用推進前 → 再推一步應相同
    worst = 0.0
    for k in (0, 1, 3, 4, 5, 200, 399):
        t = OFF + k / 200
        gs.sample(t)
        c_gs = gs.c                       # 已推進 floor(t/0.02)+1 次
        # cs[k] 推進了 floor(t/0.02) 次 → 比 gs 少一步；再推一步應相同
        c_re = gs.step(cs[k], gs.mux, gs.muy, gs.om, gs.GAIT_DT)
        worst = max(worst, max(abs(c_gs["theta"][l] - c_re["theta"][l]) for l in cpg.LEGS))
    assert worst < 1e-12, worst


def test_policy_log_summary_reads_m9_policy_block(tmp_path):
    """上機後的摘要工具要吃得下 M9 `--policy` 寫出的 json（欄位名對得上 PolicyGaitStream.log）。"""
    import json
    import coord
    import cpg
    import M9_gait as m9
    import policy_log_summary as pls
    import rl_obs
    D = json.loads((ROOT / "outputs" / "A_kp250_walk.json").read_text(encoding="utf-8"))
    p = dict(D["params"], mu_x=D["baseline_ref"]["mu_x"], mu_y=D["baseline_ref"]["mu_y"],
             d_step_y=D["baseline_ref"]["d_step_y"])
    pol = pn.load(str(ROOT / "weights" / "cpg_rl_max_v2_3_np.npz"))
    tick = [0]

    def reader():
        tick[0] += 1
        pose = coord.POSES["home"]
        return rl_obs.Frame({j: coord.to_motor(j, pose[j]) for j in rl_obs.LEG_NAMES},
                            {j: 0.0 for j in rl_obs.LEG_NAMES}, [0, 0, 0, 1], [0, 0, 0], tick[0])
    log = []
    gs = m9.PolicyGaitStream(p, cpg.home_foot(coord.POSES["home"]), cpg.knee_signs(coord.POSES["home"]),
                             pol, (0.30, 0.0), 1.0, reader, log)
    for k in range(400):
        gs.sample(k / 200)
    out = {"abort_reason": None, "peak": {"fl3_knee_pitch": 55.0, "fr3_knee_pitch": 50.0},
           "policy": {"path": "x.npz", "sha256": pol.src_sha256, "preset": "v2.3", "vx": 0.3, "wz": 0.0,
                      "gain": 1.0, "steps": gs.i, "fallback_total": gs.n_fallback_total,
                      "open_loop": gs.open_loop, "open_loop_why": gs.open_loop_why,
                      "worst_ms": round(gs.worst_ms, 3), "log": gs.log}}
    f = tmp_path / "M9_x.json"
    f.write_text(json.dumps(out), encoding="utf-8")
    txt = pls.summarize(str(f))
    assert "步數 100" in txt and "退回 0" in txt and "gravity" in txt and "ω 範圍" in txt
    assert gs.worst_ms < 10.0, gs.worst_ms       # 本機都超過預算的話狗上（RK3588）更慘
