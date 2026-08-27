"""MJX 訓練模型與 Robot 的替代模型支援。"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_walk_max as cw  # noqa: E402
import max_model as mm  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
MJX_XML = _ROOT / "model" / "zgws" / "zgws_mjx.xml"
MJX_SCENE = _ROOT / "model" / "zgws" / "scene_flat_mjx.xml"


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


def test_model_cache_is_per_scene():
    """兩個場景必須拿到不同的模型物件。

    `_model()` 原本只快取一份。加了第二個場景之後若沿用單一快取，
    第二次呼叫會**靜默拿到上一個場景的模型** —— 而所有診斷指標都會是乾淨的，
    G1 對照會變成「同一個模型跟自己比」而看起來完美通過。
    """
    a = cw._model(mm.SCENE)
    b = cw._model(mm.SCENE_MJX)
    assert a is not b
    assert a is cw._model(mm.SCENE)      # 快取沒有被第二個場景擠掉


@pytest.fixture(scope="module")
def pair():
    a = mujoco.MjModel.from_xml_path(mm.SCENE)
    b = mujoco.MjModel.from_xml_path(str(MJX_SCENE))
    return a, b


def test_mjx_model_has_no_mesh(pair):
    """訓練模型必須零網格相依。

    網格要靠 fetch_assets.sh 從官方發布包抓 2.1 GB 才拿得到，
    Colab 每次開機都要重抓。拿掉網格之後 XML 自己就是完整模型，clone 完即可用。
    """
    _, b = pair
    assert b.nmesh == 0


def test_mass_and_inertia_unchanged(pair):
    """換碰撞幾何**不可以**動到質量與慣量。

    每個 body 都有明寫的 <inertial>，所以 geom 的 density 本來就不參與計算——
    但這是一個「以為如此」的推論，必須變成會失敗的測試。
    """
    a, b = pair
    np.testing.assert_allclose(a.body_mass, b.body_mass, rtol=0, atol=0)
    np.testing.assert_allclose(a.body_inertia, b.body_inertia, rtol=0, atol=0)
    np.testing.assert_allclose(a.body_ipos, b.body_ipos, rtol=0, atol=0)


def test_joints_identical(pair):
    """關節數量、限位、frictionloss、armature 一項都不能變。"""
    a, b = pair
    assert (a.nq, a.nv) == (b.nq, b.nv)
    np.testing.assert_allclose(a.jnt_range, b.jnt_range)
    np.testing.assert_allclose(a.dof_frictionloss, b.dof_frictionloss)
    np.testing.assert_allclose(a.dof_armature, b.dof_armature)


def test_actuators_are_affine_servos(pair):
    """12 腿是 position、4 輪是 velocity，且全部 affine，順序與原模型一致。

    biastype 不是 affine 時 ctrl 會被當力矩直接施加，機器人當場塌掉且不報錯。
    """
    a, b = pair
    assert b.nu == a.nu == 16
    # 致動器名稱與順序必須逐項相同，否則 LEG_ACT_IDX / WHEEL_ACT_IDX 會指到別的關節
    for i in range(16):
        na = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        nb = mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        assert na == nb, f"致動器 {i} 名稱不同：{na} vs {nb}"
    assert np.all(b.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
    leg = b.actuator_biasprm[mm.LEG_ACT_IDX]
    wheel = b.actuator_biasprm[mm.WHEEL_ACT_IDX]
    # position 致動器：biasprm = [0, -kp, -kv]
    np.testing.assert_allclose(-leg[:, 1], np.tile(mm.KP3, 4))
    np.testing.assert_allclose(-leg[:, 2], np.tile(mm.KD3, 4))
    # velocity 致動器：biasprm = [0, 0, -kv]
    np.testing.assert_allclose(-wheel[:, 2], np.full(4, mm.KD_WHEEL))
    np.testing.assert_allclose(wheel[:, 1], np.zeros(4))


def test_wheel_collision_is_cylinder_about_joint_axis(pair):
    """四輪碰撞體是圓柱、半徑對得上實算值，而且**軸與輪關節軸同向**。

    ⚠️ 軸的方向是最容易靜默錯掉的地方：MuJoCo 把網格頂點重新表示在主慣量軸座標系，
       `geom_aabb` 因此軸被置換（實測輪子的 aabb 說輪軸在 x，但關節軸是 y）。
       照 aabb 建圓柱會做出一個**躺倒 90° 的輪子**——它照樣有接觸力、
       四個診斷指標照樣乾淨，只是接觸的是輪緣側面。
    """
    _, b = pair
    d = mujoco.MjData(b)
    mujoco.mj_forward(b, d)
    cyl = [i for i in range(b.ngeom)
           if b.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
           and (b.geom_contype[i] or b.geom_conaffinity[i])]
    assert len(cyl) == 4
    for i in cyl:
        assert abs(b.geom_size[i][0] - mm.WHEEL_RADIUS) < 5e-4
        # 圓柱的區域 z 軸（geom_xmat 第三行）必須與該腿輪關節的世界軸平行
        axis = d.geom_xmat[i].reshape(3, 3)[:, 2]
        jid = [j for j in range(b.njnt)
               if b.jnt_bodyid[j] == b.geom_bodyid[i]]
        assert len(jid) == 1, "輪 body 應該只有一個關節"
        jaxis = d.xaxis[jid[0]]
        assert abs(abs(float(np.dot(axis, jaxis))) - 1.0) < 1e-6, \
            f"圓柱軸與輪關節軸不平行（內積 {np.dot(axis, jaxis):.4f}）—— 輪子躺倒了"


def test_wheel_bottom_matches_mesh_model(pair):
    """站在同一組關節角時，兩個模型的輪最低點高度差必須小於 0.5 mm。

    這一條擋的是「圓柱位置偏了」——輪心的 y 偏移若漏掉，輪子會歪在一邊；
    半徑若取錯框，輪子會埋進地裡或浮在空中。兩者都不會報錯。
    """
    a, b = pair
    out = []
    for m in (a, b):
        d = mujoco.MjData(m)
        d.qpos[mm.LEG_QPOS_IDX] = mm.HOME12
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        lo = []
        for i in range(m.ngeom):
            if not (m.geom_contype[i] or m.geom_conaffinity[i]):
                continue
            bid = m.geom_bodyid[i]
            if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "").endswith("_FOOT_LINK"):
                continue
            lo.append(float(d.geom_xpos[i][2]) - mm.WHEEL_RADIUS)
        out.append(sorted(lo))
    np.testing.assert_allclose(out[0], out[1], atol=5e-4)


def test_mjx_model_loads_into_mjx(pair):
    """能被 mjx.put_model 吃下去，且不再有網格碰撞體。"""
    from mujoco import mjx
    _, b = pair
    assert mjx.put_model(b) is not None
    ncoll_mesh = sum(1 for i in range(b.ngeom)
                     if b.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                     and (b.geom_contype[i] or b.geom_conaffinity[i]))
    assert ncoll_mesh == 0


def test_solver_iterations_lowered(pair):
    """訓練模型必須帶低迭代數。MJX 沒有提早收斂，100/50 是固定成本。"""
    a, b = pair
    assert (a.opt.iterations, a.opt.ls_iterations) == (100, 50)
    assert (b.opt.iterations, b.opt.ls_iterations) == (6, 6)


def test_generator_is_reproducible(tmp_path):
    """重跑產生器要得到逐字元相同的檔案 —— 產生物才可以進版控。"""
    sys.path.insert(0, str(_ROOT / "model" / "zgws"))
    import make_mjx_model
    out = tmp_path / "zgws_mjx.xml"
    make_mjx_model.build(str(make_mjx_model.SRC), str(out))
    assert out.read_text() == MJX_XML.read_text()
