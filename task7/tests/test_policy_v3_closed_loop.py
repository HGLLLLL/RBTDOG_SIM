"""閉迴路「說兩次」（v3.6f，2026-09-23）：MJX 內同一趟，jax 堆疊（DualModeEnv ＋ brax policy ＋ mirror_policy）驅動機器人，
狗端堆疊（rl_obs_v3 ＋ policy_np ＋ gen_v3 ＋ kin）影子跟跑，逐步比 obs(88)、動作(24)、q_des(12)、輪 τ(4)。

env 的隨機項全部關掉（雜訊 0、IMU 傾角 0、gyro 偏置 0、push=False）；動作延遲由測試端重現（狗端吃 a_hist[delay]）。
五個指令各 300 步：直走 0.5、左平移 0.20、原地左轉 1.3、原地右轉 1.3、右平移 0.20（後兩個走鏡像）。
另外印每個指令的 err_pk／knee_v／τ 峰 → 操作卡護欄定值。本機 MJX CPU 一步 0.11 s，整個測試約 3 分鐘。

注意：`env.reset` 會先抽一個隨機指令，並用**那個**指令初始化 u_mode／ph／s4／s_arc（也就是 obs[34:39]），
之後測試才覆寫 `info["cmd"]`。所以影子產生器要用 reset 抽到的指令建（`cmd0=cmd_reset`），
每一步的 `Gen.step` 才吃測試指令（env 從第 0 步起用的就是覆寫後的指令）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import coord  # noqa: E402
import gen_v3 as g  # noqa: E402
import local_infer_v3 as li3  # noqa: E402
import max_model as mm  # noqa: E402
import policy_np as pn  # noqa: E402
import rl_env_v3 as v3  # noqa: E402
import rl_obs  # noqa: E402
import rl_obs_v3 as ro  # noqa: E402

PKL = ROOT / "weights" / "cpg_rl_v3_6f_params.pkl"
NPZ = ROOT / "weights" / "cpg_rl_v3_6f_np.npz"
CASES = [("直走 0.5", (0.5, 0.0, 0.0)), ("左平移 0.20", (0.0, 0.20, 0.0)), ("原地左轉 1.3", (0.0, 0.0, 1.3)),
         ("原地右轉 1.3", (0.0, 0.0, -1.3)), ("右平移 0.20", (0.0, -0.20, 0.0))]
N_STEPS = 300
ABAD, KNEE = [0, 3, 6, 9], [2, 5, 8, 11]


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def stack(monkeypatch_module):
    mp = monkeypatch_module
    mp.setattr(v3, "NOISE_GRAV", 0.0); mp.setattr(v3, "NOISE_GYRO", 0.0); mp.setattr(v3, "NOISE_QPOS", 0.0)
    mp.setattr(v3, "NOISE_QVEL", jnp.zeros(12)); mp.setattr(v3, "NOISE_WHEEL", 0.0)
    mp.setattr(v3, "IMU_TILT_DEG", 0.0); mp.setattr(v3, "GYRO_BIAS", 0.0)
    env = v3.DualModeEnv(gains="factory", weights=v3.W36, ref=dict(cyc_amp_rand=False), push=False)
    pol = jax.jit(v3.mirror_policy(li3.load_policy(str(PKL), 88), turn_only=False))
    return env, jax.jit(env.reset), jax.jit(env.step), pol, g.load_params(str(NPZ)), pn.load(str(NPZ)), ro.Obs(str(NPZ))


def _dog_frame(d, qv12, wv4):
    """把 MJX 狀態裝成 shm 讀值：馬達座標、xyzw 四元數、gyro = 機身角速度（imu_q 單位、偏置 0）。"""
    qctrl = np.asarray(d.qpos[v3.LEG_QPOS_IDX])
    pos = {n: coord.to_motor(n, float(qctrl[k])) for k, n in enumerate(rl_obs.LEG_NAMES)}
    vel = {n: float(qv12[k]) * rl_obs._SIGN12[k] for k, n in enumerate(rl_obs.LEG_NAMES)}
    for k, n in enumerate(ro.WHEEL_NAMES):
        pos[n] = float(d.qpos[mm.WHEEL_QPOS_IDX[k]]); vel[n] = float(wv4[k]) * ro.WHEEL_SIGN4[k]
    return rl_obs.Frame(pos, vel, np.asarray(d.qpos[3:7])[[1, 2, 3, 0]], np.asarray(d.qvel[3:6]))


@pytest.mark.parametrize("name,cmd", CASES)
def test_dog_stack_shadows_jax_stack(stack, name, cmd):
    env, jr, js, pol, P, polnp, O = stack
    s = jr(jax.random.PRNGKey(0))
    cmd_reset = np.asarray(s.info["cmd"])            # reset 抽到的指令：obs[34:39] 與產生器初值都是它
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd, jnp.float32), "cmd2": jnp.array(cmd, jnp.float32), "t_switch": 10 ** 6})
    delay = int(s.info["delay"])
    gen = g.Gen(P, cmd0=cmd_reset)
    last_d = np.zeros(24)
    a_hist_d = [np.zeros(24)] * 3
    worst = dict(obs=0.0, act=0.0, q=0.0, tau=0.0)
    pk = dict(err_abad=0.0, err_hk=0.0, knee_v=0.0, tau_leg=0.0, tau_wheel=0.0, ms=0.0)
    right_seen = False
    for i in range(N_STEPS):
        d = s.pipeline_state
        obs_j = np.asarray(s.obs)
        fr = _dog_frame(d, np.asarray(d.qvel[v3.LEG_QVEL_IDX]), np.asarray(d.qvel[v3.WHEEL_QVEL_IDX]))
        gen.integrate_head(O.gyro_z(fr))
        obs_d = O.build(fr, gen.c, gen.cmd_prev, last_d, gen.head_err, gen.s4, gen.s_arc)
        worst["obs"] = max(worst["obs"], float(np.max(np.abs(obs_j - obs_d))))
        a_j = np.asarray(pol(jnp.array(obs_j)))
        # 鏡像判斷要照 mirror_policy：它讀的是 **obs 裡的指令**。第 0 步的 obs 還帶著 reset 抽到的指令
        # （測試是 reset 之後才覆寫 info["cmd"]），所以這裡也得讀 obs，否則第 0 步兩邊的鏡像決定會不一致。
        # 狗上 M9_gait 現在也是讀 `is_right(gen.cmd_prev)`（＝ obs 帶的那個指令），與這裡同源；
        # 差別只在 cmd 0→指令切換的那一步（obs 還帶著 0），讀 cmd 會多鏡像一步、和 jax 對不起來。
        right = ro.is_right(obs_d[34:37])
        right_seen |= right
        a_d = O.mirror_act(polnp.infer(O.mirror_obs(obs_d))) if right else polnp.infer(obs_d)
        worst["act"] = max(worst["act"], float(np.max(np.abs(a_j - a_d))))
        a_hist_d = [a_d] + a_hist_d[:2]
        v_meas = O.wheel_v(fr)
        q_d, tau_d, _ = gen.step(np.array(cmd), a_hist_d[delay], v_meas)
        s = js(s, jnp.array(a_j, jnp.float32))
        dn = s.pipeline_state
        worst["q"] = max(worst["q"], float(np.max(np.abs(q_d - np.asarray(dn.ctrl[v3.LEG_ACT_IDX])))))
        tau_ref = (np.asarray(dn.ctrl[v3.WHEEL_ACT_IDX]) - v_meas) * env.kv_wheel
        worst["tau"] = max(worst["tau"], float(np.max(np.abs(g.wheel_dead(tau_d, P) - tau_ref))))
        last_d = a_d
        if i >= 50:                                                        # 淡入後才算護欄峰值
            err = np.abs(q_d - np.asarray(dn.qpos[v3.LEG_QPOS_IDX]))
            pk["err_abad"] = max(pk["err_abad"], float(err[ABAD].max())); pk["err_hk"] = max(pk["err_hk"], float(np.delete(err, ABAD).max()))
            pk["knee_v"] = max(pk["knee_v"], float(np.abs(np.asarray(dn.qvel[v3.LEG_QVEL_IDX])[KNEE]).max()))
            pk["tau_leg"] = max(pk["tau_leg"], float(s.metrics["tau_pk"])); pk["tau_wheel"] = max(pk["tau_wheel"], float(np.abs(tau_d).max()))
        assert float(s.done) == 0.0, f"{name} 第 {i} 步摔了"
    print(f"\n[{name}] right={right_seen} delay={delay} worst obs {worst['obs']:.2e} act {worst['act']:.2e} q {worst['q']:.2e} tau {worst['tau']:.2e}")
    print(f"[{name}] 護欄定值（×1.3 寫操作卡）：err ABAD {pk['err_abad']:.3f} 髖膝 {pk['err_hk']:.3f} rad、膝速 {pk['knee_v']:.1f} rad/s、"
          f"腿 τ 峰 {pk['tau_leg']:.0f}、輪 τ 峰 {pk['tau_wheel']:.2f} N·m")
    assert right_seen == (cmd[2] < 0 or (abs(cmd[2]) <= 0.1 and cmd[1] < -0.02))
    assert worst["obs"] < 2e-4, worst
    assert worst["act"] < 1e-4, worst
    assert worst["q"] < 1e-4, worst
    assert worst["tau"] < 1e-3, worst
    assert gen.n_clamp == 0, f"{name} IK 被縮限 {gen.n_clamp} 次"
