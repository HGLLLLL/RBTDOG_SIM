"""同事純 RL policy：狗端 numpy 推論（policy_np，layout rl16）== onnxruntime，2000 筆隨機 obs 逐點比對。
npz 由 `inference/export_policy_rl.py` 產；ONNX 在 repo 根 `RL_only/`。"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
ort = pytest.importorskip("onnxruntime")
# 換 policy 測：RL_ONLY_ONNX=… RL_ONLY_NPZ=… pytest …（2026-10-06；預設仍是 09-24 那顆）
ONNX = Path(os.environ.get("RL_ONLY_ONNX", ROOT.parent / "RL_only" / "2026-09-24_13-23-11.onnx"))
IO = ROOT.parent / "RL_only" / "policy_io.json"
NPZ = Path(os.environ.get("RL_ONLY_NPZ", ROOT / "weights" / "rl_only_2026-09-24_np.npz"))
pytestmark = pytest.mark.skipif(not (ONNX.exists() and IO.exists()), reason="RL_only/ 不在")


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    import export_policy_rl as ex
    out = tmp_path_factory.mktemp("rl") / "rl.npz"
    ex.export(str(ONNX), str(IO), str(out))
    return out


def test_numpy_matches_onnxruntime(npz):
    import policy_np as pn
    P = pn.load(str(npz))
    assert (P.layout, P.obs_dim, P.act_dim, P.activation, P.output) == ("rl16", 53, 16, "elu", "linear")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # 量級照 obs 各段：gyro ±3、重力 ±1、cmd ±1、q−default ±1、qvel ±15、last_a ±3
    scale = np.concatenate([np.full(3, 3.0), np.ones(3), np.ones(3), np.ones(12), np.full(16, 15.0), np.full(16, 3.0)]).astype(np.float32)
    X = (rng.uniform(-1, 1, (2000, 53)) * scale).astype(np.float32)
    ref = np.stack([sess.run(None, {"obs": x[None]})[0][0] for x in X])   # ONNX 的 batch 維固定 1
    got = np.stack([P.infer(x) for x in X])
    d = np.abs(got - ref).max()
    assert d < 2e-5, d
    assert np.abs(ref).max() > 1.0        # 輸出無界，確認測試真的踩到 tanh 外的區域


def test_committed_npz_matches_source(npz):
    """repo 裡的 npz（push_to_dog 推的那個）要和 ONNX 當下匯出的一致。"""
    if not NPZ.exists():
        pytest.skip("weights/rl_only_*_np.npz 還沒匯出")
    a, b = np.load(str(npz)), np.load(str(NPZ))
    for k in ("W0", "W1", "W2", "W3", "b0", "b1", "b2", "b3", "LO12", "HI12", "STANCE_Q12"):
        assert np.array_equal(a[k], b[k]), k
    assert str(a["src_sha256"]) == str(b["src_sha256"])
    assert str(a["gen_json"]) == str(b["gen_json"])
