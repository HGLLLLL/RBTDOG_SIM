"""68 維觀測層。

觀測層是 sim2real 最容易靜默壞掉的地方：`np.concatenate` 對長度錯誤的輸入
不會報錯，會安靜產生錯誤維度的 obs，而錯誤維度的 obs 會讓訓練好的權重直接失效。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_max  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402


def test_layout_sums_to_dim():
    assert sum(d for _, d in obs_max.OBS_LAYOUT) == obs_max.OBS_DIM == 68


def test_layout_order_is_frozen():
    """欄位順序改了，舊權重就報廢。改順序必須是一個刻意的、會失敗的決定。"""
    assert obs_max.OBS_LAYOUT == [
        ("gravity", 3), ("gyro", 3), ("joint_pos", 12), ("joint_vel", 12),
        ("cmd", 2), ("last_action", 12), ("cpg", 24)]


def test_slice_of():
    assert obs_max.slice_of("gravity") == slice(0, 3)
    assert obs_max.slice_of("cmd") == slice(30, 32)
    assert obs_max.slice_of("cpg") == slice(44, 68)
    with pytest.raises(KeyError):
        obs_max.slice_of("base_lin_vel")


@pytest.fixture(scope="module")
def level_data():
    """機身水平、關節在 HOME 的 MjData。"""
    m = mm.make_model()
    d = mujoco.MjData(m)
    d.qpos[mm.LEG_QPOS_IDX] = mm.HOME12
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[2] = mm.NOMINAL_HEIGHT_KIN
    mujoco.mj_forward(m, d)
    return d


def _c():
    return cpg_max.cpg_init(cpg_max.PHASE_WALK)


def test_build_obs_dim_and_dtype(level_data):
    o = obs_max.build_obs(level_data, _c(), np.array([0.15, 0.0]), np.zeros(12))
    assert o.shape == (68,)
    assert o.dtype == np.float32


def test_gravity_is_down_when_level(level_data):
    """機身水平時重力向量應為 (0, 0, -1)。四元數順序弄錯這一條會當場失敗。"""
    o = obs_max.build_obs(level_data, _c(), np.array([0.0, 0.0]), np.zeros(12))
    np.testing.assert_allclose(o[obs_max.slice_of("gravity")],
                               [0.0, 0.0, -1.0], atol=1e-6)


def test_joint_pos_is_zero_at_home(level_data):
    """HOME 姿態時 joint_pos 應全 0（它是相對 HOME12 的偏差）。"""
    o = obs_max.build_obs(level_data, _c(), np.array([0.0, 0.0]), np.zeros(12))
    np.testing.assert_allclose(o[obs_max.slice_of("joint_pos")], np.zeros(12), atol=1e-9)


def test_cmd_and_last_action_land_where_expected(level_data):
    """欄位真的放在 slice_of 說的位置 —— 順序錯了這裡會抓到。"""
    a = np.arange(12, dtype=float)
    o = obs_max.build_obs(level_data, _c(), np.array([0.31, -0.17]), a)
    np.testing.assert_allclose(o[obs_max.slice_of("cmd")], [0.31, -0.17], rtol=1e-6)
    np.testing.assert_allclose(o[obs_max.slice_of("last_action")], a, rtol=1e-6)


def test_wrong_size_is_rejected(level_data):
    """維度錯必須當場擋下來，不可以讓 concatenate 靜默產生 67 或 69 維。"""
    with pytest.raises(AssertionError):
        obs_max.build_obs(level_data, _c(), np.array([0.1, 0.0, 0.0]), np.zeros(12))
    with pytest.raises(AssertionError):
        obs_max.build_obs(level_data, _c(), np.array([0.1, 0.0]), np.zeros(16))


def test_leg_indices_are_not_contiguous():
    """釘住「腿關節位址不連續」這件事本身 —— 有人改成 qpos[7:19] 會在這裡失敗。"""
    assert list(mm.LEG_QPOS_IDX) != list(range(7, 19))
    assert set(mm.LEG_QPOS_IDX).isdisjoint(set(mm.WHEEL_QPOS_IDX))


def test_obs_excludes_base_linear_velocity(level_data):
    """機身線速度**不可以**進 obs —— 實機底層沒有這個量。

    這條看起來多餘，但它擋的是「訓練時順手加一欄讓 reward 好學」那種改動：
    加了會訓練得更快，然後在實機上發現餵不出來。
    """
    d = level_data
    d.qvel[0:3] = [0.37, -0.11, 0.05]      # 給一個很特別的機身線速度
    o = obs_max.build_obs(d, _c(), np.array([0.0, 0.0]), np.zeros(12))
    d.qvel[0:3] = 0.0
    assert not np.any(np.isclose(o, 0.37, atol=1e-6)), "obs 裡出現了機身線速度"
    assert not np.any(np.isclose(o, -0.11, atol=1e-6)), "obs 裡出現了機身線速度"


def test_local_infer_uses_mesh_scene_by_default():
    """預設必須跑**原始網格模型**，不是訓練用的圓盤模型。

    訓練模型是為了 MJX 才簡化的。驗收如果也跑簡化模型，
    等於用同一個近似去驗證那個近似 —— 落差永遠量不到。
    """
    import local_infer_max
    assert local_infer_max.DEFAULT_SCENE == mm.SCENE


def test_local_infer_omega_range_matches_notebook():
    """ω 的映射範圍必須與 notebook 第 4 格同值。

    不同值的話 policy 輸出的同一個數字會被解成不同的頻率，而且**不會報錯**——
    症狀是「權重在 Colab 好好的，拿回本機就走不動」。
    """
    import json
    import local_infer_max
    nb = json.loads((Path(__file__).resolve().parents[1]
                     / "notebooks" / "cpg_rl_max_colab.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    want = f"OMEGA_MIN, OMEGA_MAX = {local_infer_max.OMEGA_MIN}, {local_infer_max.OMEGA_MAX}"
    assert want in src, f"notebook 裡找不到 `{want}`"


def test_dummy_reproduces_open_loop_baseline_bit_exactly():
    """★ `--dummy` 跑出來的指標必須與開迴路基準**逐位相同**。

    基準步態在這個動作空間裡是一個固定動作（mux/muy/ω 用 atanh 反推），
    所以整條推論鏈（obs → act_to_cmd → CPG → 解析 IK → PD）有標準答案可對。
    這比「跑起來沒炸就算過」強得多：任何一個係數、順序、範圍接錯都會在這裡露餡。
    """
    import argparse
    import contextlib
    import io

    import cpg_walk_max as cw
    import local_infer_max

    args = argparse.Namespace(params="", dummy=True, secs=6.0, vx=0.15, wz=0.0,
                              video=False, scene=None)
    with contextlib.redirect_stdout(io.StringIO()):
        got = local_infer_max.run(args)
        want = cw.rollout(gait="walk", secs=6.0, quiet=True)
    for k in ("speed_travel", "bounce", "pitch_mean", "support", "yaw",
              "net_roll", "min_lift", "height", "fell"):
        assert got[k] == want[k], f"{k}: 推論端 {got[k]} vs 開迴路 {want[k]}"
    # ω 沒被動過 → 平均值就是基準值
    assert abs(got["omega_mean"] - 1.4) < 1e-12
