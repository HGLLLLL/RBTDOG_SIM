"""v1c 鏡像：表是恆等對合、與「把 MuJoCo 狀態真的左右翻」一致、物理左右對稱、鏡像回合的 obs／ctrl 行為。spec §2.1、§5.2。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_wheel as rw  # noqa: E402
import mujoco  # noqa: E402
from mujoco import mjx  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True)


def _walked_state(env, n=6):
    """從 reset 走幾步非對稱動作，得到一個左右不對稱、四輪觸地的狀態。"""
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for i in range(n):
        a = jnp.sin(jnp.arange(16) * 0.9 + i * 0.4) * 0.3
        s = step(s, a)
    return s


def test_mirror_maps_are_involutions():
    o = jnp.arange(53, dtype=jnp.float32) + 1.0
    a = jnp.arange(16, dtype=jnp.float32) + 1.0
    p = jnp.arange(56, dtype=jnp.float32) + 1.0
    assert np.allclose(rw.mirror_obs(rw.mirror_obs(o)), o) and np.allclose(rw.mirror_act(rw.mirror_act(a)), a)
    assert np.allclose(rw.mirror_priv(rw.mirror_priv(p)), p)
    assert not np.allclose(rw.mirror_obs(o), o)
    # 符號表：ωx ωz、grav_y、cmd vy wz、ABAD（jpos／jvel／last_a／act）變號；其餘 +1
    sg = np.asarray(rw.OBS_MIRROR_SGN)
    assert sg[0] == -1 and sg[1] == 1 and sg[2] == -1 and sg[4] == -1 and sg[7] == -1 and sg[8] == -1
    assert np.all(sg[9:13] == -1) and np.all(sg[13:21] == 1) and np.all(sg[21:25] == -1) and np.all(sg[25:37] == 1)
    assert np.all(sg[37:41] == -1) and np.all(sg[41:53] == 1)
    assert np.all(np.asarray(rw.ACT_MIRROR_SGN)[:4] == -1) and np.all(np.asarray(rw.ACT_MIRROR_SGN)[4:] == 1)
    idx = np.asarray(rw.OBS_MIRROR_IDX)
    assert list(idx[9:13]) == [10, 9, 12, 11] and list(idx[33:37]) == [34, 33, 36, 35]     # 腿 FR↔FL、RR↔RL（輪速也換）


def test_mirror_obs_matches_physically_mirrored_state(env):
    s = _walked_state(env)
    d = s.pipeline_state
    cmd, la = jnp.array([0.3, 0.2, -0.4]), jnp.sin(jnp.arange(16)) * 0.5
    o = env._obs(d, cmd, la)
    dm = rw.mirror_data(env, d)
    om = env._obs(dm, cmd * jnp.array([1.0, -1.0, -1.0]), rw.mirror_act(la))
    np.testing.assert_allclose(np.asarray(om), np.asarray(rw.mirror_obs(o)), atol=1e-5)
    vb = rw.quat_rot_inv(d.qpos[3:7], d.qvel[0:3]); vbm = rw.quat_rot_inv(dm.qpos[3:7], dm.qvel[0:3])
    np.testing.assert_allclose(np.asarray(vbm), np.asarray(vb) * np.array([1, -1, 1]), atol=1e-5)


LINKS = ("ABAD", "HIP", "KNEE", "FOOT")
LR_PAIRS = (("FAR", "FBL"), ("RAR", "RBL"))
YFLIP = np.array([1.0, -1.0, 1.0])


def _bid(env, name):
    return int(mujoco.mj_name2id(env._mj, mujoco.mjtObj.mjOBJ_BODY, name))


def _quat_mirror(q):
    w, x, y, z = q
    return np.array([w, -x, y, -z])


def _symmetrised_sys(env):
    """env.sys 的左右對稱副本（只給等變性測試用；訓練模型不改）。
    機身：質心 y 歸 0、慣量主軸四元數去掉 w、y 分量（鏡射後只差正負號＝同一旋轉）；
    四個膝質量取平均；左腿（FBL、RBL）每段 link 的 ipos／iquat／inertia／pos／quat 由對應右腿鏡射。"""
    ipos, mass, iquat = np.array(env.sys.body_ipos), np.array(env.sys.body_mass), np.array(env.sys.body_iquat)
    inertia, pos, quat = np.array(env.sys.body_inertia), np.array(env.sys.body_pos), np.array(env.sys.body_quat)
    b = env.BASE_BID
    ipos[b, 1] = 0.0
    qb = iquat[b].copy(); qb[0] = qb[2] = 0.0; iquat[b] = qb / np.linalg.norm(qb)
    kb = [_bid(env, f"{p}_KNEE_LINK") for p in rw.FEET]
    mass[kb] = mass[kb].mean()
    for R, L in LR_PAIRS:
        for link in LINKS:
            r, l = _bid(env, f"{R}_{link}_LINK"), _bid(env, f"{L}_{link}_LINK")
            mass[l], inertia[l] = mass[r], inertia[r]
            ipos[l], pos[l] = ipos[r] * YFLIP, pos[r] * YFLIP
            iquat[l], quat[l] = _quat_mirror(iquat[r]), _quat_mirror(quat[r])
    return env.sys.replace(body_ipos=jnp.array(ipos), body_mass=jnp.array(mass), body_iquat=jnp.array(iquat),
                           body_inertia=jnp.array(inertia), body_pos=jnp.array(pos), body_quat=jnp.array(quat))


def test_physics_is_left_right_equivariant(env):
    """翻過的狀態用翻過的動作跑 4 步 ＝ 原軌跡翻過去（驗 mirror_data／mirror_act 與物理一致）。
    為什麼用對稱化副本：原模型本身左右不對稱（機身質心 y +3.45 mm、RAR 膝 0.8598 vs 0.8631 kg、髖／膝慣量主軸差 ~5e-4、
    機身慣量主軸有小的傾斜），在原模型上 5 步 qpos 差 1.47e-2（ABAD 最敏感）；這些量在 DR 範圍內，見
    test_model_lr_asymmetry_is_within_dr。只改機身質心 y＋膝質量仍差 6e-4／qvel 3e-2，所以副本把左腿整段由右腿鏡射、機身慣量主軸對稱化。
    為什麼 4 步：完全對稱的模型上第 5 步起會因 MJX hfield 碰撞的接觸點數不左右等變（某輪 3 點、鏡像輪 1 點）而跳到 ~1.5e-3，
    那是離散事件不是鏡像錯誤（2026-10-20 task-2 報告）。物理照抄 WheelEnv._physics（_ctrl → mjx.step × N_FRAMES）。"""
    sys_sym = _symmetrised_sys(env)

    @jax.jit
    def phys(d, a):
        ctrl = env._ctrl(a)

        def one(dd, _):
            return mjx.step(sys_sym, dd.replace(ctrl=ctrl)), None
        return jax.lax.scan(one, d, None, length=rw.N_FRAMES)[0]

    s = _walked_state(env)
    d = mjx.forward(sys_sym, s.pipeline_state)
    dm = rw.mirror_data(env, d, sys=sys_sym)
    for i in range(4):
        a = jnp.sin(jnp.arange(16) * 0.5 + i) * 1.5
        d = phys(d, a); dm = phys(dm, rw.mirror_act(a))
        dmm = rw.mirror_data(env, dm, sys=sys_sym)
        eq = np.abs(np.asarray(dmm.qpos) - np.asarray(d.qpos)); ev = np.abs(np.asarray(dmm.qvel) - np.asarray(d.qvel))
        print(f"step {i}: qpos max {eq.max():.3e} @ {eq.argmax()}  qvel max {ev.max():.3e} @ {ev.argmax()}")
    np.testing.assert_allclose(np.asarray(dmm.qpos), np.asarray(d.qpos), atol=1e-4)
    np.testing.assert_allclose(np.asarray(dmm.qvel), np.asarray(d.qvel), atol=1e-3)


def test_model_lr_asymmetry_is_within_dr(env):
    """模型左右不對稱量（2026-10-20 量到：機身質心 y +3.45 mm、RAR 膝 0.8598 vs 0.8631 kg）在 DR 範圍內，訓練模型不改；鏡像增強把它當雜訊。"""
    ipos, mass = np.asarray(env.sys.body_ipos), np.asarray(env.sys.body_mass)
    com_y = float(ipos[env.BASE_BID, 1])
    km = np.array([mass[_bid(env, f"{p}_KNEE_LINK")] for p in rw.FEET])
    print(f"機身質心 y = {com_y * 1e3:+.2f} mm（DR ±50 mm）；膝質量 { {p: round(float(k), 5) for p, k in zip(rw.FEET, km)} }，"
          f"分散 {(km.max() - km.min()) / km.mean() * 100:.2f}%")
    assert abs(com_y) < 0.005
    assert (km.max() - km.min()) / km.mean() < 0.01
    for R, L in LR_PAIRS:
        for link in LINKS:
            r, l = _bid(env, f"{R}_{link}_LINK"), _bid(env, f"{L}_{link}_LINK")
            assert abs(mass[l] - mass[r]) / mass[r] < 0.01, (R, L, link, mass[r], mass[l])
            np.testing.assert_allclose(ipos[l], ipos[r] * YFLIP, atol=1e-3, err_msg=f"{R}/{L} {link} ipos")


@pytest.fixture(scope="module")
def env_m(env):
    return rw.WheelEnv(flat=True, noise=False, push=False, mirror=True, scene=env.scene)


def test_mir_is_sampled_half_and_off_under_fixed_cmd(env, env_m):
    ks = jax.random.split(jax.random.PRNGKey(0), 400)
    mir = np.asarray(jax.vmap(env_m.reset)(ks).info["mir"])
    assert mir.dtype == bool and 0.4 < mir.mean() < 0.6
    assert not np.any(np.asarray(jax.vmap(env.reset)(ks).info["mir"]))          # fixed_cmd：恆 False
    assert not np.any(np.asarray(jax.vmap(rw.WheelEnv(flat=True, noise=False, push=False, scene=env.scene).reset)(ks[:8]).info["mir"]))   # 預設 mirror=False


def test_mirror_episode_obs_and_ctrl(env_m):
    s = jax.jit(env_m.reset)(jax.random.PRNGKey(3))
    step = jax.jit(env_m.step)
    a = jnp.sin(jnp.arange(16) * 0.7) * 0.4
    for mir in (False, True):
        s1 = s.replace(info={**s.info, "mir": jnp.bool_(mir)})
        s2 = step(s1, a)
        d = s2.pipeline_state
        a_phys = rw.mirror_act(rw.ACT_GAIN * a) if mir else rw.ACT_GAIN * a
        np.testing.assert_allclose(np.asarray(d.ctrl[env_m.AID]), np.asarray(env_m._ctrl(a_phys)[env_m.AID]), atol=1e-6)     # 動作鏡像回物理
        np.testing.assert_allclose(np.asarray(s2.info["last_a"]), np.asarray(a_phys), atol=1e-6)                   # last_a 存物理側
        clean = env_m._obs(d, s2.info["cmd"], s2.info["last_a"])
        want = rw.mirror_obs(clean) if mir else clean
        np.testing.assert_allclose(np.asarray(s2.obs["state"]), np.asarray(want), atol=1e-6)                      # policy 看鏡像 obs
        np.testing.assert_allclose(np.asarray(s2.obs["state"][37:53]), np.asarray(rw.ACT_GAIN * a), atol=1e-6)   # policy 看到自己的上一步輸出
        vb = rw.quat_rot_inv(d.qpos[3:7], d.qvel[0:3]); priv = jnp.concatenate([clean, vb])
        np.testing.assert_allclose(np.asarray(s2.obs["privileged_state"]), np.asarray(rw.mirror_priv(priv) if mir else priv), atol=1e-6)
        assert float(s2.metrics["mir"]) == float(mir)
        assert bool(s2.info["mir"]) == mir                                                                           # 回合內不變


def test_mirror_episode_reward_equals_mirrored_world(env_m):
    """鏡像回合的 reward ＝ 把真實狀態翻過去、用翻過的動作算出的 reward（reward 對稱 ⇒ 鏡像資料是同一套物理的另一邊）。"""
    s = jax.jit(env_m.reset)(jax.random.PRNGKey(5))
    step = jax.jit(env_m.step)
    for i in range(4):
        s = step(s, jnp.sin(jnp.arange(16) * 0.9 + i) * 0.3)
    # mirror_data 末尾 mjx.forward 會用步後 qpos／qvel 重算 actuator_force 與接觸；step 留下的 actuator_force 是最後一個子步「積分前」的值。
    # 兩邊都走同一次 forward 才是同一個時刻（不 forward 時 tau_leg 差 0.10、tau_wheel 差 9e-4，全是這個時間錯位）。
    d, info = mjx.forward(env_m.sys, s.pipeline_state), dict(s.info)
    a = jnp.sin(jnp.arange(16)) * 0.6
    cmd = jnp.array([0.3, 0.4, 0.2])
    T, _ = env_m._reward_terms(d, info, a, cmd)
    dm = rw.mirror_data(env_m, d)
    LR = jnp.array(rw.MIRROR_LR); ym = jnp.array([1.0, -1.0, 1.0])
    idx16 = jnp.array([g * 4 + rw.MIRROR_LR[k] for g in range(4) for k in range(4)]); sgn16 = jnp.concatenate([-jnp.ones(4), jnp.ones(12)])
    info_m = dict(info, last_a=rw.mirror_act(info["last_a"]), qvel_prev=info["qvel_prev"][idx16] * sgn16,
                  foot_prev=info["foot_prev"][LR] * ym, touch_prev=info["touch_prev"][LR], vfoot_prev=info["vfoot_prev"][LR] * ym)
    Tm, _ = env_m._reward_terms(dm, info_m, rw.mirror_act(a), cmd * jnp.array([1.0, -1.0, -1.0]))
    for k in rw.TERM_KEYS:
        tol = 1e-3 if k in ("slip", "ww", "impact", "tau_leg", "tau_wheel") else 1e-4     # 接觸表排序與求解器容差
        assert float(Tm[k]) == pytest.approx(float(T[k]), abs=tol), k
