"""MJX 訓練模型與 Robot 的替代模型支援。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_walk_max as cw  # noqa: E402
import max_model as mm  # noqa: E402


def test_model_cache_same_scene_is_shared():
    """同一個場景要拿到同一份模型（快取是擋 OOM 的，不是微優化）。

    跨場景的隔離另外由 `test_model_cache_is_per_scene` 驗（需要 MJX 場景）。
    """
    a = cw._model()
    b = cw._model(mm.SCENE)
    assert a is b


def test_solver_iters_override_is_restored():
    """solver 迭代數覆寫必須在下一次建 Robot 時還原，否則會滲進後續 rollout。

    這是 `--friction 0.3` 滲透那個坑的同一類：上一格的設定悄悄留在快取模型上，
    而四個診斷指標全是乾淨的，事後看不出來。
    """
    r1 = cw.Robot(solver_iters=(6, 6))
    assert (r1.m.opt.iterations, r1.m.opt.ls_iterations) == (6, 6)
    r2 = cw.Robot()
    assert (r2.m.opt.iterations, r2.m.opt.ls_iterations) == (100, 50)
