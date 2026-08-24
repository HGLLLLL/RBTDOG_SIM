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


# =====================================================================
# 開迴路「可用走路步態」的回歸測試（cpg_walk_d1.py）
#
# 與上面 test_openloop_walks_forward_without_falling 的分工：
#   那支驗的是 cpg_openloop_d1 —— 關卡 3 的驗收腳本，只確認 MJCF/IK/伺服接得起來，
#   它的步態其實不能看（後腳實測只抬 6.3~16.9 mm，指令是 80 mm）。
#   這支驗的是 cpg_walk_d1 —— 調過參數、四條腿都真的抬起來的走路步態。
# 兩支都要留：前者是模型正確性的回歸，後者是步態品質的回歸。
# =====================================================================


def test_cpg_walk_all_four_legs_actually_clear_the_ground():
    """四條腿都要真的離地，而且左右對稱。

    判準依 2026-08-11 實測（8 秒、名目摩擦 0.6、kp=80）：
    FL 79.9 / FR 80.9 / RL 95.8 / RR 98.3 mm。下限取 40 mm——
    這個門檻刻意設在「舊 openloop 的最差腿 6.3 mm」與「本步態最差腿 79.9 mm」
    之間很寬的位置，只擋真正的退化（例如有人改了 X_OFF 或 MU_Y 卻沒發現後腳又趴回去）。
    """
    import cpg_walk_d1

    res = cpg_walk_d1.rollout(secs=8.0, video=False, quiet=True)

    assert res["fell"] is None, f"步態跌倒於 {res['fell']} s"

    lifts_mm = [v * 1000 for v in res["peak_lift"]]
    assert res["min_lift"] * 1000 >= 40.0, (
        f"最弱的腿只離地 {res['min_lift'] * 1000:.1f} mm（四腿 {lifts_mm}）。"
        "四條腿都要抬起來才算走路——這正是舊 openloop 步態的病（最差腿 6.3 mm）。"
    )

    # 左右對稱：同一端的左右腿離地量不應差超過 1.5 倍。
    # MU_Y 若被改回 1.8（橫向擺動打開），左右會立刻失衡，這條就是用來擋那件事的。
    front = sorted(lifts_mm[0:2])
    rear = sorted(lifts_mm[2:4])
    assert front[1] / front[0] < 1.5, f"前腳左右不對稱：FL/FR = {lifts_mm[0:2]}"
    assert rear[1] / rear[0] < 1.5, f"後腳左右不對稱：RL/RR = {lifts_mm[2:4]}"


def test_cpg_walk_goes_forward_and_roughly_straight():
    """要往前走，而且不能畫弧。

    實測 8 秒：前進 +6.04 m、側偏 +0.73 m。側偏上限取前進量的 40%，
    足以擋住 MU_Y 被改回 1.8 的情況（那時側偏會到 −2.45 m / 10 秒）。
    """
    import cpg_walk_d1

    res = cpg_walk_d1.rollout(secs=8.0, video=False, quiet=True)
    assert res["dist"] > 3.0, f"8 秒只前進 {res['dist']:.2f} m"
    assert abs(res["lateral"]) < 0.4 * res["dist"], (
        f"側偏 {res['lateral']:+.2f} m 相對前進 {res['dist']:.2f} m 太大，步態在畫弧"
    )


def test_cpg_walk_does_not_saturate_actuator_ctrlrange():
    """指令不得撞到 ctrlrange。

    撞到就代表 IK 解出來的角度超出致動器行程，指令會被靜默 clip、步態走樣而不報錯。
    實測本步態全程 0%。
    """
    import cpg_walk_d1

    res = cpg_walk_d1.rollout(secs=8.0, video=False, quiet=True)
    assert res["clip_pct"] < 1.0, f"{res['clip_pct']:.1f}% 的關節指令被 ctrlrange clip 掉"


def test_cpg_walk_does_not_mutate_shared_d1_model_constants():
    """cpg_walk_d1 的步態參數必須是本檔私有，不得汙染 d1_model。

    d1_model.G_C / MU_* / D_STEP 是 RL 訓練與推論的共用契約——
    本檔用的是 G_C=0.12（RL 是 0.08），若哪天有人圖方便直接改 d1_model，
    既有權重會全部靜默作廢。
    """
    import cpg_walk_d1

    assert cpg_walk_d1.GAIT_G_C != d1_model.G_C, "步態的 G_C 應與 d1_model 分離"
    assert d1_model.G_C == pytest.approx(0.08), "d1_model.G_C 被改動了"
    cpg_walk_d1.rollout(secs=1.0, video=False, quiet=True)
    assert d1_model.G_C == pytest.approx(0.08), "跑完之後 d1_model.G_C 被就地改寫"
    assert d1_model.MU_MAX == pytest.approx(2.0)


def test_cpg_walk_four_beat_gait_keeps_three_legs_on_the_ground():
    """四拍 walk 的核心價值就是佔空比 0.75 → 任一時刻約三腳著地。

    這是它在低步頻能贏過 trot 的唯一原因：trot 永遠只有兩腳著地，
    兩腳支撐的空檔身體會下沉，步頻越低下沉越久（實測彈跳 ω=2.0 的 15.7 mm
    → ω=1.2 的 39.4 mm）。walk 沒有那個空檔。
    若哪天 duty 被改回 0.5 或相位被改成 trot，這條會響。
    """
    import cpg_walk_d1

    res = cpg_walk_d1.rollout(gait="walk", secs=10.0, video=False, quiet=True)
    assert res["support"] > 2.5, (
        f"平均支撐腳只有 {res['support']:.2f}，四拍 walk 應接近 3。"
        "檢查 GAITS['walk'] 的 duty 與 PHASE_WALK 有沒有被改動。"
    )
    assert res["fell"] is None


def test_cpg_walk_trot_mode_duty_remap_is_identity():
    """duty=0.5 時 `duty_remap` 必須恆等——否則 trot 模式會與改版前的實測值脫鉤。

    數學上：ph<0.5 → pi*ph/0.5 = 2pi*ph = theta；
            ph>=0.5 → pi + pi*(ph-0.5)/0.5 = 2pi*ph = theta。
    """
    import cpg_walk_d1

    th = np.linspace(0.0, 2 * np.pi, 401)
    assert cpg_walk_d1.duty_remap(th, 0.5) == pytest.approx(th % (2 * np.pi), abs=1e-9)
    # duty > 0.5 時擺動相要被壓縮：相位走到一圈的 25% 時就該完成整個擺動（theta'≈pi）
    assert cpg_walk_d1.duty_remap(np.array([0.25 * 2 * np.pi]), 0.75)[0] == pytest.approx(
        np.pi, abs=1e-9)


def test_cpg_walk_does_not_touch_d1_model_phase_offset():
    """walk 需要非 trot 的相位，但只能自帶，不得改動 d1_model.PHASE_OFFSET。

    那份是 RL 訓練與推論的共用契約，而且 test_phase_offset_is_trot_not_any_other_gait
    釘死它是 trot。本檔的耦合矩陣必須來自自己的 PHASE_WALK。
    """
    import cpg_walk_d1

    assert not np.allclose(cpg_walk_d1.PHASE_WALK, d1_model.PHASE_OFFSET), \
        "PHASE_WALK 不應等於 trot"
    assert np.allclose(cpg_walk_d1.PHASE_TROT, d1_model.PHASE_OFFSET), \
        "PHASE_TROT 應與 d1_model 的 trot 相位一致"
    cpg_walk_d1.rollout(gait="walk", secs=2.0, video=False, quiet=True)
    assert np.allclose(d1_model.PHASE_OFFSET, [0.0, np.pi, np.pi, 0.0]), \
        "跑完 walk 之後 d1_model.PHASE_OFFSET 被就地改寫了"
