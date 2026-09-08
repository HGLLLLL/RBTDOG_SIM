import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import M_env_probe as ep  # noqa: E402


def test_pure_python_forward_matches_numpy():
    layers = ep.make_weights((68, 256, 256, 128, 12), seed=1)
    x = [0.01 * i for i in range(68)]
    y_py = ep.forward_py(layers, x)
    y_np = ep.forward_np(layers, x)
    np.testing.assert_allclose(y_py, y_np, rtol=1e-9, atol=1e-9)
    assert len(y_py) == 12


def test_bench_returns_positive_ms():
    layers = ep.make_weights((8, 8, 4), seed=0)
    mean_ms, max_ms = ep.bench(lambda: ep.forward_py(layers, [0.0] * 8), n=5)
    assert 0 < mean_ms <= max_ms
