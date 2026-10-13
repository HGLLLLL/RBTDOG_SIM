"""export_policy_wheel：Brax (normalizer, policy, value) pkl → rl16 npz（swish、tanh_loc、act_gain 4）；npz 推論 == brax 推論 × 4；gen_json 與 rl_only 匯出同欄位。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference")); sys.path.insert(0, str(ROOT / "realbot"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import export_policy_wheel as ew  # noqa: E402
import policy_np as pn  # noqa: E402


@pytest.fixture(scope="module")
def pkl(tmp_path_factory):
    from brax.io import model
    p = tmp_path_factory.mktemp("w") / "rl_wheel_test_params.pkl"
    model.save_params(str(p), ew.make_params_for_test(jax.random.PRNGKey(0)))
    return p


def test_export_matches_brax_inference(pkl, tmp_path):
    out = tmp_path / "rl_wheel_test_np.npz"
    z = ew.export(str(pkl), str(out))
    P = pn.load(str(out))
    assert (P.layout, P.obs_dim, P.act_dim, P.activation, P.output, P.act_gain) == ("rl16", 53, 16, "swish", "tanh_loc", 4.0)
    assert P.W[0].shape == (53, 512) and P.W[3].shape == (128, 32)
    assert not np.allclose(P.mean, 0.0) and not np.allclose(P.std, 1.0)        # 測試參數帶非平凡正規化
    infer = ew.brax_inference_fn(str(pkl))
    rng = np.random.default_rng(0)
    scale = np.concatenate([np.full(3, 3.0), np.ones(3), np.ones(3), np.ones(12), np.full(16, 15.0), np.full(16, 3.0)]).astype(np.float32)
    X = (rng.uniform(-1, 1, (500, 53)) * scale).astype(np.float32)
    ref = np.asarray(infer(jnp.array(X)))                                        # brax deterministic：tanh(loc) ∈ [−1,1]
    got = np.stack([P.infer(x) for x in X])
    assert np.abs(got - 4.0 * ref).max() < 1e-4
    K = json.loads(str(z["gen_json"]))
    for k in ("DEFAULT_POS_PHYSX", "ACT_SCALE_PHYSX", "PHYSX_TO_MJCF", "Q_SLEW", "A_CLIP", "RAMP_STEPS", "TAU_FF", "WHEEL_TAU_MAX", "gains", "obs_slices", "noise", "CTRL_DT"):
        assert k in K, k
    assert K["A_CLIP"] == 4.0 and K["source"]["family"] == "rl_wheel"
    ref_json = json.loads(str(z["ref_json"])); assert ref_json["family"] == "rl_wheel" and ref_json["nominal_height"] == 0.485


def test_rl_only_gen_tables_unchanged():
    """抽出 gen_tables 後，rl_only 匯出的 gen_json／LO12／HI12 要與已 commit 的 npz 逐字同（test_policy_rl_np 也守這條）。"""
    import export_policy_rl as er
    npz = ROOT / "weights" / "rl_only_2026-10-01_10k_np.npz"
    io_path = ROOT.parent / "RL_only" / "policy_io.json"
    if not (npz.exists() and io_path.exists()):
        pytest.skip("RL_only／npz 不在")
    io = json.load(open(io_path)); z = np.load(str(npz))
    T = er.gen_tables(io, source=json.loads(str(z["gen_json"]))["source"])
    assert json.dumps(T["K"], sort_keys=True) == str(z["gen_json"])
    assert np.array_equal(T["lo12"], z["LO12"]) and np.array_equal(T["hi12"], z["HI12"])
