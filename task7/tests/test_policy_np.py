"""policy_np（狗上純 numpy 推論）：與 brax 前向逐點相同、常數與 local_infer_max 釘死。

★ 這是 RL 上機的第一道硬門：正規化式子或 activation 對錯，brax 不會報錯，
  policy 只會靜默偏掉。所以不從記憶寫式子，用 1000 筆隨機 obs 對 brax 逐點比。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
import export_policy_np as ex  # noqa: E402
import gait_baseline as gb  # noqa: E402
import local_infer_max as li  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import policy_np as pn  # noqa: E402

PKL = ROOT / "weights" / "cpg_rl_max_v2_3_params.pkl"
NPZ = ROOT / "weights" / "cpg_rl_max_v2_3_np.npz"


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    out = tmp_path_factory.mktemp("w") / "p.npz"
    ex.export(str(PKL), str(out), preset="v2.3")
    return out


def test_constants_pinned():
    assert (pn.MU_MIN, pn.MU_MAX) == (mm.MU_MIN, mm.MU_MAX)
    assert (pn.OMEGA_MIN, pn.OMEGA_MAX) == (li.OMEGA_MIN, li.OMEGA_MAX)
    assert (pn.SWAY_MAX, pn.SWAY_SLEW) == (li.SWAY_MAX, li.SWAY_SLEW)
    assert pn.A_MU_X == gb.BASELINE_A["mu_x"]
    assert pn.A_MU_Y == gb.BASELINE_A["mu_y"]
    assert pn.A_OMEGA == gb.BASELINE_A["omega"]
    assert pn.LAYOUT_DIMS == li.LAYOUT_DIMS
    assert pn.RAMP_STEPS == li.PRESET_RAMP["v2.3"] == 50


def test_act_to_cmd_and_baseline_match_local_infer():
    rng = np.random.default_rng(0)
    for layout in ("nomux", "full"):
        n = li.LAYOUT_DIMS[layout]
        np.testing.assert_array_equal(pn.baseline_action(layout), li.baseline_action(layout))
        for _ in range(100):
            a = rng.normal(size=n) * 2
            for x, y in zip(pn.act_to_cmd(a, layout), li.act_to_cmd(a, layout)):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    np.testing.assert_array_equal(pn.slew_sway(np.zeros(2), np.array([0.06, -0.001])),
                                  li.slew_sway(np.zeros(2), np.array([0.06, -0.001])))


def test_npz_meta(npz):
    p = pn.load(str(npz))
    assert (p.obs_dim, p.act_dim, p.preset, p.layout) == (66, 10, "v2.3", "nomux")
    for k in ("omega", "duty", "d_step", "d_step_y", "x_off", "g_c", "z_sag", "mu_x", "mu_y", "seq"):
        assert p.baseline[k] == gb.BASELINE_A[k], k
    assert p.baseline["kp3"] == list(gb.BASELINE_A["kp3"])
    assert p.baseline["wheel_kd"] == gb.BASELINE_A["wheel_kd"]
    assert len(p.src_sha256) == 64


def test_forward_matches_brax(npz):
    p = pn.load(str(npz))
    infer = li.load_policy(str(PKL), act_dim=10, head=False)
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(1000, obs_max.obs_dim(10, False))).astype(np.float32) * 3
    worst = 0.0
    for o in obs:
        a_np = p.infer(o)
        a_jx = infer(o)
        assert a_np.shape == (10,)
        worst = max(worst, float(np.max(np.abs(a_np - a_jx))))
    assert worst < 1e-5, worst
    assert np.all(np.abs(a_np) <= 1.0)


def test_infer_rejects_wrong_dim(npz):
    p = pn.load(str(npz))
    with pytest.raises(AssertionError):
        p.infer(np.zeros(70))


def test_committed_npz_is_current_export(npz):
    """repo 裡的 npz 必須是這支匯出器對這顆 pkl 產生的（防止換了 pkl 沒重匯）。"""
    assert NPZ.exists(), "先跑 export_policy_np.py 產生 weights/cpg_rl_max_v2_3_np.npz"
    a, b = np.load(str(npz)), np.load(str(NPZ))
    assert str(a["src_sha256"]) == str(b["src_sha256"])
    for k in ("mean", "std", "W0", "b3"):
        np.testing.assert_array_equal(a[k], b[k])
    assert json.loads(str(b["baseline_json"]))["mu_x"] == gb.BASELINE_A["mu_x"]
