"""純 RL 自訓 v1 地形：方塊圖案（rough／boxes／stairs）、大圖拼接、hfield 慣例、場景檔（膠囊碰撞＋pair）。spec §4。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
import wheel_terrain as wt  # noqa: E402


def test_layout_constants():
    assert (wt.NT, wt.NB, wt.NCOL, wt.NROW) == (80, 200, 1201, 641)      # 外框 20 m（他的 border_width 200）
    assert (wt.X0, wt.Y0) == (-60.0, -32.0)
    assert wt.tile_origin(0, 0) == (-36.0, -8.0) and wt.tile_origin(2, 9) == (36.0, 8.0)


def test_level_amp_endpoints():
    assert wt.level_amp("rough", 0) == pytest.approx(0.01) and wt.level_amp("rough", 9) == pytest.approx(0.06)
    assert wt.level_amp("boxes", 0) == pytest.approx(0.025) and wt.level_amp("boxes", 9) == pytest.approx(0.10)
    assert wt.level_amp("stairs", 0) == pytest.approx(0.03) and wt.level_amp("stairs", 9) == pytest.approx(0.08)


def _at(h, x, y):
    n = wt.NT; i = int(round(y / wt.DX + n / 2)); j = int(round(x / wt.DX + n / 2))
    return h[i, j]


def test_stairs_pyramid_geometry():
    h = wt.tile_pattern("stairs", 9, seed=1)        # seed 1 → 正金字塔（default_rng(1).integers(2)==0）
    assert h.shape == (wt.NT + 1, wt.NT + 1)
    assert _at(h, 0.0, 0.0) == pytest.approx(0.8)              # 中央平台 10 階 × 0.08
    assert _at(h, 0.95, 0.0) == pytest.approx(0.8)             # |x| ≤ 1 仍是平台
    assert _at(h, 1.15, 0.0) == pytest.approx(0.72)            # 第 1 階
    assert _at(h, 1.15, 1.15) == pytest.approx(0.72)           # 以 max(|x|,|y|) 算環
    assert _at(h, 3.5, 0.0) == pytest.approx(0.08)             # 第 9 階
    assert _at(h, 3.9, 0.0) == 0.0                             # 邊緣 0.25 m 壓平
    h_inv = wt.tile_pattern("stairs", 9, seed=0)   # seed 0 → 反向（default_rng(0).integers(2)==1）
    assert _at(h_inv, 0.0, 0.0) == pytest.approx(-0.8) and _at(h_inv, 3.9, 0.0) == 0.0


def test_boxes_and_rough_ranges():
    hb = wt.tile_pattern("boxes", 9, seed=3)
    assert _at(hb, 0.0, 0.0) == 0.0 and _at(hb, 0.9, 0.9) == 0.0          # 中央 2 m 平台
    assert hb.max() <= 0.10 + 1e-9 and hb.min() >= -0.10 - 1e-9 and hb.max() > 0.05
    hr = wt.tile_pattern("rough", 9, seed=3)
    assert hr.min() >= 0.0 and hr.max() <= 0.06 + 1e-9
    assert np.allclose(hr, np.round(hr / 0.01) * 0.01, atol=1e-9)   # 0.01 步階
    assert wt.tile_pattern("flat", 0, 0).max() == 0.0


def test_build_map_tiles_and_border():
    H = wt.build_map(seed=0)
    assert H.shape == (wt.NROW, wt.NCOL)
    assert np.all(H[:wt.NB, :] == wt.BASE_H) and np.all(H[:, :wt.NB] == wt.BASE_H)        # 外框平地
    assert np.all(H[-wt.NB:, :] == wt.BASE_H) and np.all(H[:, -wt.NB:] == wt.BASE_H)
    for ki in range(3):
        for lv in range(wt.LEVELS):
            x, y = wt.tile_origin(ki, lv)
            if ki != 0:                                                                      # boxes／stairs 中央是平台或平台高度
                assert abs(wt.height_np(H, x + 0.3, y - 0.3) - wt.height_np(H, x, y)) < 1e-9
    assert H.min() >= wt.BASE_H - 0.8 - 1e-9 and H.max() <= wt.BASE_H + 0.8 + 1e-9
    Hf = wt.build_map(flat=True); assert np.all(Hf == wt.BASE_H)


def test_height_np_bilinear():
    H = np.full((wt.NROW, wt.NCOL), wt.BASE_H); H[wt.NB + 5, wt.NB + 7] = wt.BASE_H + 0.1
    x = wt.X0 + (wt.NB + 7) * wt.DX; y = wt.Y0 + (wt.NB + 5) * wt.DX
    assert wt.height_np(H, x, y) == pytest.approx(wt.BASE_H + 0.1)
    assert wt.height_np(H, x + 0.05, y) == pytest.approx(wt.BASE_H + 0.05)


def test_scene_files_load_and_match_mjx_requirements():
    mujoco = pytest.importorskip("mujoco")
    scene = wt.make_scene_xml()
    m = mujoco.MjModel.from_xml_path(scene)
    assert m.opt.timestep == 0.005
    assert m.opt.iterations == 6 and m.opt.ls_iterations == 6
    assert m.npair == 4 and m.nhfield == 1 and m.hfield_nrow[0] == wt.NROW and m.hfield_ncol[0] == wt.NCOL
    for p in ("FAR", "FBL", "RAR", "RBL"):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p}_FOOT_LINK_COLL")
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and m.geom_size[g, 0] == pytest.approx(wt.WHEEL_R)
        assert m.geom_size[g, 1] == pytest.approx(wt.WHEEL_HW)
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{p}_FOOT_LINK")
        assert m.actuator_gainprm[a, 0] == pytest.approx(1.0) and m.actuator_biasprm[a, 2] == pytest.approx(-1.0)
    # 全部碰撞幾何改膠囊（MJX hfield–膠囊閉式解）：14 ＝ hfield 地板 ＋ 機身 ＋ 8 段腿 ＋ 4 輪，除地板外全是膠囊
    assert m.ngeom == 14 and int(((m.geom_contype != 0) | (m.geom_conaffinity != 0)).sum()) == 14
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert m.geom_type[floor] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert all(m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE for i in range(m.ngeom) if i != floor)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "BASE_LINK_COLL")
    assert np.allclose(m.geom_size[b], (0.105, 0.31, 0.0))
    legs = [i for i in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) is None]
    assert len(legs) == 8 and sorted(round(float(m.geom_size[i, 0]), 3) for i in legs) == [0.015] * 4 + [0.03] * 4
    assert abs(m.hfield_size[0][0] - 60.0) < 1e-9 and abs(m.hfield_size[0][1] - 32.0) < 1e-9 and m.hfield_size[0][2] == wt.ZMAX
    # CPU 工具的 pair 場景：圓柱輪子不動、只加 pair
    m2 = mujoco.MjModel.from_xml_path(wt.make_flat_pairs_scene_xml())
    g = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "FAR_FOOT_LINK_COLL")
    assert m2.npair == 4 and m2.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
    first = Path(scene).read_text(encoding="utf-8").splitlines()[0]; assert "不要手改" in first


def test_hfield_convention_with_ray():
    """row ↔ y、col ↔ x、高度 = data × ZMAX：用 mj_ray 從上往下打對 height_np。"""
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(wt.make_scene_xml()); H = wt.build_map(seed=0); wt.load_hfield(m, H)
    d = mujoco.MjData(m); d.qpos[2] = 5.0; d.qpos[3] = 1.0; mujoco.mj_forward(m, d)
    worst = 0.0
    for ki in range(3):
        for lv in (0, 9):
            x0, y0 = wt.tile_origin(ki, lv)
            for dx, dy in ((0.33, -0.21), (-1.47, 1.9), (2.6, 2.2)):
                x, y = x0 + dx, y0 + dy
                geomid = np.zeros(1, dtype=np.int32)
                dist = mujoco.mj_ray(m, d, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
                assert dist >= 0
                worst = max(worst, abs((5.0 - dist) - wt.height_np(H, x, y)))
    assert worst < 0.02, worst


def test_mjx_put_model_accepts_scene():
    jax = pytest.importorskip("jax"); mujoco = pytest.importorskip("mujoco")
    from mujoco import mjx
    m = mujoco.MjModel.from_xml_path(wt.make_scene_xml()); wt.load_hfield(m, wt.build_map(0))
    sys_ = mjx.put_model(m)
    assert sys_.hfield_data.shape == (wt.NROW * wt.NCOL,)
