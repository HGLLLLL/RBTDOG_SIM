"""local_infer_max v2：與訓練端常數一致、--dummy 逐位重現 A 基準、G3–G7 判定鍵齊全。"""
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cpg_walk_max as cw  # noqa: E402
import gait_baseline as gb  # noqa: E402
import local_infer_max as li  # noqa: E402
import rl_env_max as re  # noqa: E402


def test_baseline_action_and_act_to_cmd_match_env_module():
    a = li.baseline_action()
    np.testing.assert_allclose(a, re.baseline_action())
    mux, muy, om, sw = li.act_to_cmd(np.full(14, 0.3))
    mux2, muy2, om2, sw2 = re.act_to_cmd(np.full(14, 0.3))
    np.testing.assert_allclose(mux, np.asarray(mux2), atol=1e-6)
    np.testing.assert_allclose(om, np.asarray(om2), atol=1e-6)
    np.testing.assert_allclose(sw, np.asarray(sw2), atol=1e-6)
    np.testing.assert_allclose(li.slew_sway(np.zeros(2), np.array([0.06, -0.01])),
                               [re.SWAY_SLEW, -0.004])


def test_dummy_matches_open_loop_walk_a():
    """G0：固定動作（sway=0）必須與開迴路 A 步態 rollout 逐位相同。"""
    A = gb.BASELINE_A
    args = SimpleNamespace(params="", dummy=True, secs=6.0, vx=0.15, wz=0.0, video=False,
                           scene=None, perturb=1, compare=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res = li.run(args)
        ref = cw.rollout(gait="walk_a", secs=6.0, kp3=A["kp3"], kd3=A["kd3"],
                         kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True)
    for k in ("speed_travel", "bounce", "support", "min_lift", "roll_pk", "exec_front"):
        assert abs(res[k] - ref[k]) < 1e-9, k


def test_gates_report_keys():
    args = SimpleNamespace(params="", dummy=True, secs=4.0, vx=0.15, wz=0.0, video=False,
                           scene=None, perturb=2, compare=True)
    with contextlib.redirect_stdout(io.StringIO()):
        res = li.run(args)
    assert set(res["gates"]) >= {"G3", "G4", "G5", "G6", "G7"}
    assert res["n_perturb"] == 2 and "baseline" in res


def test_v2_2_layout_dummy_matches_open_loop_walk_a():
    """v2.2（10 維、mu_x 固定）的 --dummy 也必須逐位重現 A 基準。"""
    A = gb.BASELINE_A
    np.testing.assert_allclose(li.baseline_action("nomux"), re.baseline_action("nomux"))
    mux, muy, om, sw = li.act_to_cmd(np.full(10, 0.2), "nomux")
    mux2, muy2, om2, sw2 = re.act_to_cmd(np.full(10, 0.2), "nomux")
    np.testing.assert_allclose(mux, np.asarray(mux2))
    np.testing.assert_allclose(om, np.asarray(om2), atol=1e-6)
    np.testing.assert_allclose(sw, np.asarray(sw2), atol=1e-6)
    args = SimpleNamespace(params="", dummy=True, secs=6.0, vx=0.30, wz=0.0, video=False,
                           scene=None, perturb=1, compare=False, preset="v2.2")
    with contextlib.redirect_stdout(io.StringIO()):
        res = li.run(args)
        ref = cw.rollout(gait="walk_a", secs=6.0, kp3=A["kp3"], kd3=A["kd3"],
                         kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True)
    for k in ("speed_travel", "bounce", "support", "roll_pk", "exec_front"):
        assert abs(res[k] - ref[k]) < 1e-9, k


def test_latency_and_ramp_defaults_and_dummy_still_bit_exact():
    """預設延遲 1 步（實機條件）；固定動作不受延遲/淡入影響，G0 仍逐位相同。"""
    assert li.DEFAULT_LATENCY == 1 and li.PRESET_RAMP["v2.3"] == 50 and li.PRESET_RAMP["v2.2"] == 0
    A = gb.BASELINE_A
    args = SimpleNamespace(params="", dummy=True, secs=4.0, vx=0.30, wz=0.0, video=False,
                           scene=None, perturb=1, compare=False, preset="v2.3", latency=1, ramp=50)
    with contextlib.redirect_stdout(io.StringIO()):
        res = li.run(args)
        ref = cw.rollout(gait="walk_a", secs=4.0, kp3=A["kp3"], kd3=A["kd3"],
                         kd_wheel=A["wheel_kd"], z_sag=A["z_sag"], quiet=True)
    for k in ("speed_travel", "bounce", "roll_pk"):
        assert abs(res[k] - ref[k]) < 1e-9, k
    assert res["latency"] == 1 and res["ramp"] == 50
