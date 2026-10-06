"""閉迴路「說兩次」（同事純 RL，2026-09-29）：MuJoCo 內同一趟，參考堆疊（`inference/rl_only_infer.Sim` 直接從 MuJoCo 狀態算 obs
＋ onnxruntime）驅動機器人；狗端堆疊（rl_obs_rl ＋ policy_np ＋ gen_direct ＋ coord）影子跟跑，
逐步比 obs(53)、動作(16)、q_des(12, MJCF 序)、輪速目標(4)。

兩邊是獨立實作：參考端用 MuJoCo 關節名直接索引 PhysX 序、quat 轉旋轉矩陣；狗端走 shm 名 → `coord.to_ctrl`／`_SIGN12` →
npz 的 PHYSX_TO_MJCF → `w2b`。輸入端的馬達座標由 `coord.to_motor` 構造（與 v3 影子跟跑同法）。
另外印每個指令的誤差峰／關節速峰／命令速峰／τ 峰 → 操作卡護欄定值（×1.3）。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))
ort = pytest.importorskip("onnxruntime")
mujoco = pytest.importorskip("mujoco")
import coord  # noqa: E402
import gen_direct as gd  # noqa: E402
import policy_np as pn  # noqa: E402
import rl_obs  # noqa: E402
import rl_obs_rl as ro  # noqa: E402
import rl_only_infer as ri  # noqa: E402

# 換 policy 測：RL_ONLY_ONNX=… RL_ONLY_NPZ=… pytest …（2026-10-06；預設仍是 09-24 那顆）
ONNX = Path(os.environ.get("RL_ONLY_ONNX", ROOT.parent / "RL_only" / "2026-09-24_13-23-11.onnx"))
IO = ROOT.parent / "RL_only" / "policy_io.json"
NPZ = Path(os.environ.get("RL_ONLY_NPZ", ROOT / "weights" / "rl_only_2026-09-24_np.npz"))
pytestmark = pytest.mark.skipif(not (ONNX.exists() and IO.exists() and NPZ.exists()), reason="RL_only/ 或 npz 不在")
CASES = [("直走 0.5", (0.5, 0.0, 0.0)), ("左平移 0.5", (0.0, 0.5, 0.0)), ("原地左轉 1.0", (0.0, 0.0, 1.0)),
         ("右平移 0.5", (0.0, -0.5, 0.0)), ("弧線 0.5+0.5", (0.5, 0.0, 0.5))]
N_STEPS = 300


@pytest.fixture(scope="module")
def stack():
    import json
    io = json.load(open(IO))
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    sim = ri.Sim(io, dt=0.005, ideal=False, wheel_kv=1.0)
    P = gd.load_params(str(NPZ))
    P = gd.Params(P.ref, dict(P.K, RAMP_STEPS=0, Q_SLEW=1e9), P.T)     # 參考端沒有淡入／限速；只留限位夾（參考端的 tgt 另外夾）
    return sim, sess, io, P, pn.load(str(NPZ)), ro.Obs(str(NPZ))


def _dog_frame(sim):
    """把 MuJoCo 狀態裝成 shm 讀值：馬達座標、xyzw 四元數、gyro = 機身角速度。"""
    d = sim.d
    jid_mjcf = [mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}_{k}_JOINT") for p in ("FAR", "FBL", "RAR", "RBL") for k in ("ABAD", "HIP", "KNEE")]
    qctrl = np.array([d.qpos[sim.m.jnt_qposadr[j]] for j in jid_mjcf]); vctrl = np.array([d.qvel[sim.m.jnt_dofadr[j]] for j in jid_mjcf])
    pos = {n: coord.to_motor(n, float(qctrl[k])) for k, n in enumerate(rl_obs.LEG_NAMES)}
    vel = {n: float(vctrl[k]) * rl_obs._SIGN12[k] for k, n in enumerate(rl_obs.LEG_NAMES)}
    wj = [mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}_FOOT_JOINT") for p in ("FAR", "FBL", "RAR", "RBL")]
    for k, n in enumerate(ro.WHEEL_NAMES):
        pos[n] = float(d.qpos[sim.m.jnt_qposadr[wj[k]]]); vel[n] = float(d.qvel[sim.m.jnt_dofadr[wj[k]]]) * ro.WHEEL_SIGN4[k]
    return rl_obs.Frame(pos, vel, np.asarray(d.qpos[3:7])[[1, 2, 3, 0]], np.asarray(d.qvel[3:6]).copy(), 0)


@pytest.mark.parametrize("name,cmd", CASES)
def test_dog_stack_shadows_reference(stack, name, cmd):
    sim, sess, io, P, pol, O = stack
    sim.reset()
    g = gd.Gen(P)
    last_a = np.zeros(16)
    p2m = np.asarray(P.K["PHYSX_TO_MJCF"], int)
    d_obs = d_act = d_q = d_v = d_obs_a = 0.0
    err_pk = jv_pk = vcmd_pk = 0.0; tau_pk = np.zeros(3); sat = 0
    tgt_prev = None
    for i in range(N_STEPS):
        c = np.zeros(3) if i < 50 else np.asarray(cmd, float)
        o_ref, _ = sim.obs(c)                                   # 參考 obs（MuJoCo 直接算）
        fr = _dog_frame(sim)
        o_dog = O.build(fr, g.c, c, last_a, g.head_err, g.s4, g.s_arc)
        d_obs = max(d_obs, float(np.abs(o_dog[:37] - o_ref[:37]).max()))      # 感測段嚴格比；last_action 段承接動作差（見下）
        d_obs_a = float(np.abs(o_dog[37:] - o_ref[37:]).max())
        a_ref = sess.run(None, {"obs": o_ref[None]})[0][0].astype(float)
        a_dog = pol.infer(o_dog)
        d_act = max(d_act, float(np.abs(a_dog - a_ref).max()))
        v_wheel = O.wheel_v(fr)
        q_dog, tau_dog, _ = g.step(c, a_dog, v_wheel)
        tgt = sim.apply(a_ref); sim.last_a = a_ref.copy()      # 參考端驅動機器人
        q_ref_m = np.clip(tgt[:12][p2m], P.T["LO12"], P.T["HI12"])
        d_q = max(d_q, float(np.abs(q_dog - q_ref_m).max()))
        d_v = max(d_v, float(np.abs(g.v_nom - tgt[12:16]).max()))
        last_a = a_dog
        # 護欄用峰值（實際 vs 命令）
        qnow = np.array([coord.to_ctrl(n, _dog_frame(sim).pos[n]) for n in rl_obs.LEG_NAMES])
        err_pk = max(err_pk, float(np.abs(q_dog - qnow).max()))
        jv_pk = max(jv_pk, float(np.abs(sim.d.qvel[sim.vadr[:12]]).max()))
        if tgt_prev is not None:
            vcmd_pk = max(vcmd_pk, float(np.abs(tgt[:12] - tgt_prev[:12]).max() / 0.02))
        tgt_prev = tgt
        f = np.abs(sim.d.actuator_force[sim.aid])
        tau_pk = np.maximum(tau_pk, [f[0:4].max(), f[4:8].max(), f[8:12].max()])
        sat += int(g.sat_mask.any())
    print(f"\n{name}: Δobs {d_obs:.2e} Δact {d_act:.2e} Δq_des {d_q:.2e} Δv_nom {d_v:.2e} | err_pk {err_pk:.3f} jv_pk {jv_pk:.1f} "
          f"vcmd_pk {vcmd_pk:.1f} τ abad/hip/knee {tau_pk[0]:.0f}/{tau_pk[1]:.0f}/{tau_pk[2]:.0f} sat {sat}/{N_STEPS} fell {sim.base_contact()}")
    # last_action 段：狗端餵自己的 numpy 動作、參考端餵 onnxruntime 的，差就是兩個推論的浮點差（≤ d_act），不另設門檻
    # 門檻（2026-10-06 放寬）：感測段 Δobs 仍嚴格（< 5e-6，這才是 obs 組裝對不對）。動作段兩端各餵自己的 last_action，
    # 同一 obs 下 numpy vs onnxruntime 的單步差只有 ~1e-6，但 10-01 的兩顆 policy 對 last_action 較敏感，300 步沿軌跡放大到
    # 2e-5～1.6e-4（＝ q_des 差 4e-5 rad，0.002°）；09-24 那顆 ≤ 1.3e-5。真的接錯線（序／符號／縮放）差會是 1e-2 以上。
    assert d_obs < 5e-6 and d_act < 1e-3 and d_q < 2.5e-4 and d_v < 5e-3, (d_obs, d_act, d_q, d_v, d_obs_a)
    assert not sim.base_contact()
