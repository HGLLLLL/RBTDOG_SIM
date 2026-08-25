"""驗證 max_model 裡寫死的位址表與模型實際狀況一致。

寫死位址是刻意的（見 max_model 的註解：`qpos[7:19]` 那種切片會靜默把輪角當關節角），
但寫死就必須有測試釘住，否則模型檔一改就無聲對不上。
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import max_model as mm


@pytest.fixture(scope="module")
def m():
    return mm.make_model()


def test_model_shape(m):
    assert (m.nq, m.nv, m.nu) == (23, 22, 16)
    assert m.opt.timestep == pytest.approx(mm.SIM_DT)


def test_leg_qpos_idx_matches_names(m):
    """LEG_QPOS_IDX 必須逐項等於「用名稱查到的」位址。"""
    want = []
    for leg in mm.LEGS:
        for j in ("ABAD", "HIP", "KNEE"):
            jid = mm._id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{mm.PREFIX[leg]}_{j}_JOINT")
            want.append(m.jnt_qposadr[jid])
    assert list(mm.LEG_QPOS_IDX) == want


def test_leg_qvel_idx_matches_names(m):
    want = []
    for leg in mm.LEGS:
        for j in ("ABAD", "HIP", "KNEE"):
            jid = mm._id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{mm.PREFIX[leg]}_{j}_JOINT")
            want.append(m.jnt_dofadr[jid])
    assert list(mm.LEG_QVEL_IDX) == want


def test_wheel_idx_matches_names(m):
    qp, qv = [], []
    for leg in mm.LEGS:
        jid = mm._id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{mm.PREFIX[leg]}_FOOT_JOINT")
        qp.append(m.jnt_qposadr[jid])
        qv.append(m.jnt_dofadr[jid])
    assert list(mm.WHEEL_QPOS_IDX) == qp
    assert list(mm.WHEEL_QVEL_IDX) == qv


def test_leg_and_wheel_idx_disjoint():
    """腿關節與輪關節的位址不得重疊 —— 重疊就是那個經典靜默錯誤。"""
    assert not (set(mm.LEG_QPOS_IDX) & set(mm.WHEEL_QPOS_IDX))
    assert not (set(mm.LEG_QVEL_IDX) & set(mm.WHEEL_QVEL_IDX))
    assert len(set(mm.LEG_QPOS_IDX)) == 12


def test_actuator_idx_matches_names(m):
    """致動器位址表對上名稱查詢，並確認每個致動器綁的是對的關節。"""
    leg_act, wheel_act = [], []
    for leg in mm.LEGS:
        for j in ("ABAD", "HIP", "KNEE"):
            aid = mm._id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{mm.PREFIX[leg]}_{j}_LINK")
            jid = mm._id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{mm.PREFIX[leg]}_{j}_JOINT")
            assert m.actuator_trnid[aid, 0] == jid
            leg_act.append(aid)
        aid = mm._id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{mm.PREFIX[leg]}_FOOT_LINK")
        wheel_act.append(aid)
    assert list(mm.LEG_ACT_IDX) == leg_act
    assert list(mm.WHEEL_ACT_IDX) == wheel_act


def test_actuators_are_pure_torque(m):
    """釘住「官方致動器是純力矩、沒有 ctrlrange」這個前提。

    PD 是我們自己在迴圈裡算的。哪天模型換成帶位置伺服的版本，
    這個測試會失敗 —— 那時 cpg_walk_max 的控制迴圈必須跟著改，不能靜默疊加兩層 PD。
    """
    assert not m.actuator_ctrllimited.any(), "出現 ctrlrange：控制迴圈的假設要重新檢查"
    assert np.allclose(m.actuator_gainprm[:, 0], 1.0)
    assert np.allclose(m.actuator_biasprm[:, :3], 0.0), "biasprm 非零代表已內建位置伺服"


def test_foot_body_is_wheel_axle(m):
    """輪 body 原點應該就是輪軸心：四腿在 HOME 下的 z 一致、y 左右對稱。"""
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    d.qpos[mm.LEG_QPOS_IDX] = mm.HOME12
    mujoco.mj_forward(m, d)
    fb = np.array([d.xpos[i] for i in mm.foot_body_ids(m)])
    base = d.xpos[mm._id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")]
    rel = fb - base
    assert np.allclose(rel[:, 2], rel[0, 2], atol=1e-9), "四輪不等高"
    assert rel[0, 1] == pytest.approx(-rel[1, 1], abs=1e-9), "FR/FL 的 y 不對稱"
    assert rel[2, 1] == pytest.approx(-rel[3, 1], abs=1e-9), "RR/RL 的 y 不對稱"


def test_constants_are_readonly():
    """常數必須是唯讀的 —— 就地改寫在 numpy 上會成功且不報錯，然後全域擴散。"""
    for name in ("HOME", "STAND", "CROUCH", "KP3", "TAU_MAX3", "LEG_QPOS_IDX", "SIDE_Y"):
        arr = getattr(mm, name)
        with pytest.raises(ValueError):
            arr[0] = 99
