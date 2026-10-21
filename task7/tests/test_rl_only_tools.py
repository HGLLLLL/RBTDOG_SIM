"""驗收工具：load_policy（onnx／npz 同介面）、run 吃 callable、gap_rr 欄、stairs 地形、--self-collision 場景。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference")); sys.path.insert(0, str(ROOT / "inference" / "diag")); sys.path.insert(0, str(ROOT / "realbot"))
import rl_only_infer as ri  # noqa: E402
import rl_only_rough_terrain as rt  # noqa: E402
import wheel_terrain as wt  # noqa: E402

ONNX = ROOT.parent / "RL_only" / "2026-10-01_17-30-20" / "2026-10-01_17-30-20_10000.onnx"


def ri_io():
    return json.load(open(ROOT / "inference" / "rl_wheel_io.json"))


def test_load_policy_npz_matches_policy_np(tmp_path):
    jax = pytest.importorskip("jax")
    import export_policy_wheel as ew, policy_np as pn
    from brax.io import model
    p = tmp_path / "t_params.pkl"; model.save_params(str(p), ew.make_params_for_test(jax.random.PRNGKey(1)))
    npz = tmp_path / "t_np.npz"; ew.export(str(p), str(npz))
    pol = ri.load_policy(None, str(npz)); P = pn.load(str(npz))
    o = np.random.default_rng(0).normal(size=53).astype(np.float32)
    assert np.allclose(pol(o), P.infer(o)) and np.abs(pol(o)).max() <= 4.0
    with pytest.raises(ValueError):
        ri.load_policy(None, None)


@pytest.mark.skipif(not ONNX.exists(), reason="RL_only/ 不在")
def test_load_policy_onnx_and_run_gap():
    pytest.importorskip("onnxruntime")
    pol = ri.load_policy(str(ONNX), None)
    sim = ri.Sim(ri_io(), dt=0.005, ideal=False, wheel_kv=1.0, scene=wt.make_flat_pairs_scene_xml())
    # metrics 從指令後 1 s 起算：secs 要 > 1 s 窗口才非空
    R = ri.run(sim, pol, [0.0, 0.0, 0.0], secs=1.4, settle=0.2); g = ri.metrics(R, [0, 0, 0])
    assert "gap_rr" in g and 50 < g["gap_rr"] < 400 and g["fell"] is None


def test_stairs_terrain_grid():
    n = int(round(2 * rt.HALF / rt.DX)) + 1
    h = rt.terrain_grid("stairs", 0.08, seed=0, n=n)
    c = (n - 1) // 2
    at = lambda x, y: h[c + int(round(y / rt.DX)), c + int(round(x / rt.DX))]   # noqa: E731
    assert h.min() >= 0.0 and at(0, 0) == pytest.approx(0.8) and at(1.15, 0) == pytest.approx(0.72) and at(3.5, 0) == pytest.approx(0.08)
    assert at(5.0, 0) == 0.0                                                   # 10 階後平地
    h2 = rt.terrain_grid("stairs", 0.08, seed=1, n=n)                            # 反向：中央是坑，整體平移到 ≥ 0
    assert h2.min() == 0.0 and h2[c, c] == pytest.approx(0.0)                     # 中央最低
    assert h2[c, c + 12] == pytest.approx(0.08) and h2[c, c + 50] == pytest.approx(0.8)   # 1.2 m 第 1 階、5 m 外平地＝最高


def test_rough_scene_with_pairs():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(rt.make_scene_xml(pairs=True))
    assert m.npair == 4 and m.nhfield == 1


def test_landing_vz_counts_touchdowns_and_uses_pre_touch_velocity():
    dt = 0.02
    clr = np.zeros((40, 4))                     # 四腿：三腿一直在地上，FR 抬一次再落
    c = np.zeros(40); c[10:20] = [0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.016, 0.008, 0.0]   # 0.016→0.008 是觸地前一步：vz ＝ 0.4 m/s
    clr[:, 0] = c
    mean_vz, n = ri.landing_vz(clr, dt)
    assert n == 1 and mean_vz == pytest.approx(0.4, rel=1e-6)
    clr[:, 1] = c                               # 兩腿同樣 → 兩個事件、平均不變
    assert ri.landing_vz(clr, dt) == (pytest.approx(0.4, rel=1e-6), 2)
    assert ri.landing_vz(np.zeros((40, 4)), dt) == (0.0, 0)           # 從沒離地：0 事件
    up = np.zeros((40, 4)); up[5:, 2] = 0.05                           # 抬起不落：0 事件
    assert ri.landing_vz(up, dt) == (0.0, 0)


def test_landing_counts_per_leg():
    dt = 0.02
    clr = np.zeros((40, 4)); c = np.zeros(40); c[10:20] = [0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.016, 0.008, 0.0]
    clr[:, 1] = c; clr[:, 3] = c; clr[25:30, 3] = 0.02                       # FL 一次、RL 兩次、FR／RR 從沒離地
    assert ri.landing_counts(clr, dt) == [0, 1, 0, 2]
    assert sum(ri.landing_counts(clr, dt)) == ri.landing_vz(clr, dt)[1]       # 總數與 landing_vz 一致
