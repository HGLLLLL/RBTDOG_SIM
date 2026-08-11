"""CPG / IK 純函式測試。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cpg_d1
import d1_model


def test_cpg_init_shapes_and_trot_phase():
    c = cpg_d1.cpg_init()
    for k in ("rx", "rx_d", "ry", "ry_d", "theta"):
        assert c[k].shape == (4,), k
    # trot：FL 與 RR 同相、FR 與 RL 同相、兩組差 pi
    assert c["theta"] == pytest.approx([0.0, np.pi, np.pi, 0.0])


def test_cpg_amplitude_converges_to_mu():
    """rx 應收斂到指令 mu。"""
    c = cpg_d1.cpg_init()
    mu = np.full(4, 1.8)
    for _ in range(200):
        c = cpg_d1.cpg_step(c, mu, mu, np.zeros(4), d1_model.CTRL_DT)
    assert c["rx"] == pytest.approx(mu, abs=1e-3)
    assert c["ry"] == pytest.approx(mu, abs=1e-3)


def test_cpg_phase_advances_with_omega():
    """omega=1 Hz 跑 0.25 秒，相位應前進約 pi/2。"""
    c = cpg_d1.cpg_init()
    om = np.ones(4)
    for _ in range(int(0.25 / d1_model.CTRL_DT)):
        c = cpg_d1.cpg_step(c, np.full(4, 1.5), np.full(4, 1.5), om, d1_model.CTRL_DT)
    assert c["theta"][0] == pytest.approx(np.pi / 2, abs=0.15)


def test_act_to_cmd_saturates_into_declared_ranges():
    lo = cpg_d1.act_to_cmd(np.full(12, -50.0))
    hi = cpg_d1.act_to_cmd(np.full(12, +50.0))
    assert lo[0] == pytest.approx(np.full(4, d1_model.MU_MIN), abs=1e-6)
    assert hi[0] == pytest.approx(np.full(4, d1_model.MU_MAX), abs=1e-6)
    assert lo[2] == pytest.approx(np.full(4, d1_model.OMEGA_MIN), abs=1e-6)
    assert hi[2] == pytest.approx(np.full(4, d1_model.OMEGA_MAX), abs=1e-6)


def test_leg_ik_consts_home_wheel_is_below_hip():
    """home 姿態下輪心應在髖正下方約 0.224 m，並向外偏 0.142 m。

    實測基準：f0 = (0.0003, ±0.1423, -0.2238)，Jacobian 條件數 2.6。
    y 的 0.142 m 包含輪子相對小腿末端向外 4.5 cm 的安裝偏移，抄漏會讓步幅算錯。
    """
    m = d1_model.make_model()
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    assert f0s.shape == (4, 3)
    assert jinvs.shape == (4, 3, 3)
    for k in range(4):
        assert abs(f0s[k][0]) < 0.02, f"腿 {k} 的輪心未在髖正下方 (x={f0s[k][0]:.3f})"
        assert -0.24 < f0s[k][2] < -0.20, f"腿 {k} 髖到輪心距離異常 (z={f0s[k][2]:.3f})"
        assert 0.13 < abs(f0s[k][1]) < 0.155, f"腿 {k} 輪子橫向偏移異常 (y={f0s[k][1]:.3f})"
    left = [f0s[0][1], f0s[2][1]]
    right = [f0s[1][1], f0s[3][1]]
    assert all(v > 0 for v in left) and all(v < 0 for v in right), "左右腿的 y 偏移方向反了"


def test_ik_jacobian_moves_foot_in_requested_direction():
    """把輪心往 +x 推 2 cm，正向運動學算回來的位移應吻合（實測誤差 0.89 mm）。"""
    import mujoco

    m = d1_model.make_model()
    d = mujoco.MjData(m)
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    want = np.array([0.02, 0.0, 0.0])
    q3 = d1_model.HOME3 + jinvs[0] @ want

    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[d1_model.LEG_QPOS_IDX[0:3]] = q3   # FL abad/hip/knee（輪關節不連續，別用 7:10）
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FL_abad")
    got = d.geom_xpos[gid] - d.xpos[hid] - f0s[0]
    assert got == pytest.approx(want, abs=3e-3)


def test_joint_targets_returns_12_and_stays_in_limits():
    import mujoco

    m = d1_model.make_model()
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    c = cpg_d1.cpg_init()
    om = np.full(4, 2.0)
    for _ in range(100):
        c = cpg_d1.cpg_step(c, np.full(4, 2.0), np.full(4, 2.0), om, d1_model.CTRL_DT)
        q = cpg_d1.joint_targets(c, f0s, jinvs)
        assert q.shape == (12,)
        # jnt_range 現在還含 4 個輪關節（無限位），不能再用 [1:] 一把抓，改用名稱取 12 個腿關節
        leg_jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{j}_joint")
                    for leg in d1_model.LEGS for j in ("abad", "hip", "knee")]
        assert np.all(q >= m.jnt_range[leg_jids, 0] - 1e-6), "關節目標角低於下限"
        assert np.all(q <= m.jnt_range[leg_jids, 1] + 1e-6), "關節目標角超過上限"
        # 還要在致動器 ctrlrange 內才不會被 clip（abad 的 ctrlrange 比 jnt_range 更緊）
        assert np.all(q >= m.actuator_ctrlrange[:, 0] - 1e-6), "關節目標角低於 ctrlrange"
        assert np.all(q <= m.actuator_ctrlrange[:, 1] + 1e-6), "關節目標角超過 ctrlrange"


def test_w2b_rotates_gravity_into_body_frame():
    """機身繞 y 轉 90 度後，世界的 -z 重力在機身系應變成 +x 方向。"""
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    got = cpg_d1.w2b(q, np.array([0.0, 0.0, -1.0]))
    assert got == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_openloop_walks_forward_without_falling():
    """關卡 3 的回歸測試（較慢，約 10 秒）。

    判準依 2026-08-10 實測校準：名目摩擦 0.8 下前進/理論 = 0.99、抬腳 0.084 m。
    抬腳區間刻意寬於 Go2 版——G_C=0.08 是指令值，量到 ~0.084 代表追蹤良好。
    """
    import cpg_openloop_d1

    res = cpg_openloop_d1.rollout(secs=6.0, video=False)
    assert res["fell"] is None, f"開迴路 CPG 跌倒於 {res['fell']} s"
    ratio = res["dist"] / res["theory"]
    assert 0.85 < ratio < 1.20, (
        f"前進 {res['dist']:.2f} m / 理論 {res['theory']:.2f} m = {ratio:.2f}，超出 0.85~1.20"
    )
    assert 0.05 < res["foot_lift"] < 0.12, f"抬腳量 {res['foot_lift']:.3f} m 不合理"


# =====================================================================
# CPG 常數釘住（Task 6 審查加測）
#
# 加測動機：審查者實測把 D_STEP 改成 0.99、把 PHASE_OFFSET 改成 pace 步態，
# 既有 29 個測試全部照樣通過。CPG 常數是訓練與推論之間的隱形契約——
# 訓練時用 D_STEP=0.12 學到的策略，推論時若變成 0.99，權重直接作廢。
# 期望值抄自 task6/inference/d1_model.py 現行值（＝ task4 論文標準版）。
# =====================================================================

# 常數名 -> 期望值（與 d1_model.py 逐項對照，不得由 d1_model 反推）
CPG_CONSTANTS = {
    "MU_MIN": 1.0,
    "MU_MAX": 2.0,
    "OMEGA_MIN": 0.0,
    "OMEGA_MAX": 2.5,
    "A_CONV": 50.0,
    "D_STEP": 0.12,
    "D_STEP_Y": 0.09,
    "G_C": 0.08,
    "G_P": 0.01,
    "W_COUP": 8.0,
    "N_CPG_SUB": 4,
}


@pytest.mark.parametrize("name", sorted(CPG_CONSTANTS))
def test_cpg_constant_is_pinned(name):
    """逐一釘住 CPG 常數。改動任一個都必須是有意識的決定（含同步更新本表與權重）。"""
    got = getattr(d1_model, name)
    want = CPG_CONSTANTS[name]
    assert got == pytest.approx(want, abs=1e-12), (
        f"d1_model.{name} = {got}，期望 {want}。"
        "若這是刻意調整，請一併確認訓練用的權重是否需要重跑——"
        "CPG 常數在訓練與推論之間對不上，等於整批 GPU 時數白燒。"
    )


def test_n_cpg_sub_is_an_int_not_a_float():
    """N_CPG_SUB 拿來當 range() 的次數，變成 float 會在推論時才炸。"""
    assert isinstance(d1_model.N_CPG_SUB, int) and not isinstance(d1_model.N_CPG_SUB, bool)


def test_cpg_ranges_are_ordered():
    assert d1_model.MU_MIN < d1_model.MU_MAX
    assert d1_model.OMEGA_MIN < d1_model.OMEGA_MAX


def test_d_step_y_is_smaller_than_d_step():
    """側向擺幅必須小於前後步幅。

    原因：本機 abad 行程僅 ±28°（Go2 是 ±60°），沿用前後的 0.12 會超出 abad 限位約 14%，
    IK 解出來的目標角會被 ctrlrange clip 掉，步態靜默走樣。
    """
    assert d1_model.D_STEP_Y < d1_model.D_STEP, (
        f"D_STEP_Y={d1_model.D_STEP_Y} 應小於 D_STEP={d1_model.D_STEP}；"
        "abad 行程僅 ±28°，側向沿用前後尺度會超限"
    )


def test_phase_offset_is_trot_not_any_other_gait():
    """PHASE_OFFSET 必須是 trot：對角腿同相，兩組相差 pi。

    腿序是 LEGS = ("FL", "FR", "RL", "RR")，所以對角組是 index (0, 3) 與 (1, 2)。
    這條特別重要：若日後有人改動腿序卻忘了同步改 PHASE_OFFSET，步態會從 trot
    靜默變成 pace 或 bound——機器人照樣會動，只是走得爛，而且沒有任何測試會響。
    """
    legs = list(d1_model.LEGS)
    assert legs == ["FL", "FR", "RL", "RR"], (
        f"腿序改成了 {legs}；PHASE_OFFSET 的對角配對是綁在腿序上的，必須同步更新"
    )
    ph = np.asarray(d1_model.PHASE_OFFSET, dtype=float)
    assert ph.shape == (4,)

    def diff(a, b):
        """相位差摺回 [0, pi]，避免 0 與 2pi 被判為不同。"""
        return abs(np.angle(np.exp(1j * (a - b))))

    i_fl, i_fr, i_rl, i_rr = (legs.index(n) for n in ("FL", "FR", "RL", "RR"))
    assert diff(ph[i_fl], ph[i_rr]) == pytest.approx(0.0, abs=1e-9), (
        f"對角腿 FL/RR 應同相，實得相位差 {diff(ph[i_fl], ph[i_rr]):.4f} rad（PHASE_OFFSET={ph}）"
    )
    assert diff(ph[i_fr], ph[i_rl]) == pytest.approx(0.0, abs=1e-9), (
        f"對角腿 FR/RL 應同相，實得相位差 {diff(ph[i_fr], ph[i_rl]):.4f} rad（PHASE_OFFSET={ph}）"
    )
    assert diff(ph[i_fl], ph[i_fr]) == pytest.approx(np.pi, abs=1e-9), (
        f"兩組對角腿應相差 pi（trot），實得 {diff(ph[i_fl], ph[i_fr]):.4f} rad；"
        f"若同相則是 bound、若 FL/FR 同相則是 pace。PHASE_OFFSET={ph}"
    )
    # 逐項釘住現行值，順便擋下「整體平移一個常數」這種等價但會改變落地相位的改動
    assert ph == pytest.approx([0.0, np.pi, np.pi, 0.0], abs=1e-12)


def test_cpg_init_uses_phase_offset_directly():
    """cpg_init 的初始相位必須來自 PHASE_OFFSET，不能另外寫死一份。"""
    c = cpg_d1.cpg_init()
    assert c["theta"] == pytest.approx(np.asarray(d1_model.PHASE_OFFSET), abs=1e-12)
    c["theta"][0] += 1.0
    assert d1_model.PHASE_OFFSET[0] == pytest.approx(0.0), "cpg_init 回傳的相位不得是常數本體的別名"
