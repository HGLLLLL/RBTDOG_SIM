"""閉迴路「說兩次」：MuJoCo 內同一趟 rollout，jax 堆疊（obs_max + brax + cpg_max）驅動機器人，
狗端堆疊（rl_obs + policy_np + realbot/cpg + realbot/kin）以自己的 CPG 狀態影子跟跑，
逐步比 obs、動作、12 個關節目標角。含 v2.3 的延遲 1 步、淡入 50 步、sway 斜率。

★ 這是上機前唯一能同時驗到「腿序 × 座標換算 × 正規化 × 動作映射 × CPG 調變 × IK」整條鏈的測試。
狗端 CPG 是 stdlib math，與 numpy 版差 ~1e-11（test_cpg_port），所以容差不是 0。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
import coord  # noqa: E402
import cpg  # noqa: E402
import cpg_max  # noqa: E402
import cpg_walk_max as cw  # noqa: E402
import gait_baseline as gb  # noqa: E402
import leg_kin  # noqa: E402
import local_infer_max as li  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import policy_np as pn  # noqa: E402
import rl_obs  # noqa: E402

PKL = ROOT / "weights" / "cpg_rl_max_v2_3_params.pkl"
NPZ = ROOT / "weights" / "cpg_rl_max_v2_3_np.npz"
N_STEPS, LAT, RAMP = 200, 1, 50


def _dog_home():
    pose = {}
    for k, l in enumerate(mm.LEGS):
        for j, kd in enumerate(coord.LEG_KINDS):
            pose[rl_obs.LEG_SHM[l] + kd] = float(mm.HOME[k, j])
    return cpg.home_foot(pose), cpg.knee_signs(pose)


def test_dog_stack_shadows_jax_stack():
    A = gb.BASELINE_A
    infer = li.load_policy(str(PKL), act_dim=10, head=False)
    pol = pn.load(str(NPZ))
    cmd = np.array([0.30, 0.0])
    base = li.baseline_action("nomux")
    np.testing.assert_array_equal(base, pn.baseline_action("nomux"))

    # ---- 機器人（同 local_infer_max.run_once，原始網格模型 torque_pd）
    r = cw.Robot(scene=mm.SCENE, actuator_mode="torque_pd", kp3=A["kp3"], kd3=A["kd3"],
                 kd_wheel=A["wheel_kd"])
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    r.reset_standing(cpg_max.stand_targets(ks, f0, A["x_off"]), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, A["x_off"]))
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    # ---- jax 堆疊狀態
    step_j = cpg_max.make_cpg_step(cpg_max.PHASE_WALK_LS)
    c_j = cpg_max.cpg_init(cpg_max.PHASE_WALK_LS)
    last_j, sway_j = np.zeros(10), np.zeros(2)
    a_hist_j = [base.copy()] * 3
    # ---- 狗端堆疊狀態
    f0_d, ks_d = _dog_home()
    step_d = cpg.make_step(cpg.PHASES["ls"])
    c_d = cpg.init(cpg.PHASES["ls"])
    xo_d = cpg.x_off_split(A["x_off"], 0.0)
    last_d, sway_d = np.zeros(10), np.zeros(2)
    a_hist_d = [base.copy()] * 3
    names = rl_obs.LEG_NAMES
    qv_hist = [r.d.qvel[mm.LEG_QVEL_IDX].copy()] * 3

    worst = dict(obs=0.0, act=0.0, q=0.0, cpg=0.0)
    for i in range(N_STEPS):
        qv_del = qv_hist[LAT]
        # jax 路徑 obs
        d_ = SimpleNamespace(qpos=r.d.qpos, qvel=r.d.qvel.copy())
        d_.qvel[mm.LEG_QVEL_IDX] = qv_del
        obs_j = obs_max.build_obs(d_, c_j, cmd, last_j)
        # 狗端 obs：把模擬狀態「裝成」shm 讀值（馬達座標、xyzw 四元數）
        qctrl = r.d.qpos[mm.LEG_QPOS_IDX]
        pos_m = {n: coord.to_motor(n, float(qctrl[k])) for k, n in enumerate(names)}
        vel_m = {n: float(qv_del[k]) * rl_obs._SIGN12[k] for k, n in enumerate(names)}
        fr = rl_obs.Frame(pos_m, vel_m, r.d.qpos[3:7][[1, 2, 3, 0]], r.d.qvel[3:6])
        obs_d = rl_obs.build(fr, c_d, cmd, last_d)
        worst["obs"] = max(worst["obs"], float(np.max(np.abs(obs_j - obs_d))))

        a_j, a_d = infer(obs_j), pol.infer(obs_d)
        worst["act"] = max(worst["act"], float(np.max(np.abs(a_j - a_d))))
        a_hist_j = [a_j] + a_hist_j[:2]
        a_hist_d = [a_d] + a_hist_d[:2]
        qv_hist = [r.d.qvel[mm.LEG_QVEL_IDX].copy()] + qv_hist[:2]
        u = min(1.0, i / RAMP)
        act_j = base + u * (a_hist_j[LAT] - base)
        act_d = base + u * (a_hist_d[LAT] - base)

        mux, muy, om, sw_t = li.act_to_cmd(act_j, "nomux")
        sway_j = li.slew_sway(sway_j, sw_t)
        c_j = step_j(c_j, mux, muy, om, mm.CTRL_DT)
        sw_arg = None if not np.any(sway_j) else (float(sway_j[0]), float(sway_j[1]))
        q_j, _ = cpg_max.joint_targets(c_j, f0, A["x_off"], A["g_c"], A["d_step"], A["d_step_y"],
                                       A["duty"], ks, A["z_sag"], sw_arg)

        mux_d, muy_d, om_d, sw_td = pn.act_to_cmd(act_d, "nomux")
        sway_d = pn.slew_sway(sway_d, sw_td)
        c_d = step_d(c_d, rl_obs.to_shm_legs(mux_d), rl_obs.to_shm_legs(muy_d),
                     rl_obs.to_shm_legs(om_d), mm.CTRL_DT)
        sw_d = None if not np.any(sway_d) else (float(sway_d[0]), float(sway_d[1]))
        q_d, _ = cpg.joint_targets(c_d, f0_d, ks_d, xo_d, A["g_c"], A["d_step"], A["d_step_y"],
                                   A["duty"], A["z_sag"], sw_d)
        for key in ("rx", "ry", "theta"):
            worst["cpg"] = max(worst["cpg"], float(np.max(np.abs(
                np.asarray(c_j[key]) - rl_obs.from_shm_legs(c_d[key])))))
        worst["q"] = max(worst["q"], max(abs(float(q_j[k]) - q_d[n]) for k, n in enumerate(names)))

        r.step(q_j)                       # 只有 jax 路徑驅動機器人；狗端是影子
        last_j, last_d = a_j, a_d

    print("worst", worst)
    assert worst["obs"] < 1e-5, worst
    assert worst["act"] < 1e-4, worst
    assert worst["cpg"] < 1e-6, worst
    assert worst["q"] < 1e-6, worst
    assert np.any(sway_j), "200 步後 sway 仍全零 —— policy 沒在調變，測試沒測到東西"
