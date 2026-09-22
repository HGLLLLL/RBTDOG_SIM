"""v3.7 候選（2026-09-22 E6–E10）：平移抬高／跨距解耦旗標、obs／動作左右鏡像。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3  # noqa: E402


def test_mirror_maps_are_involutions_and_flip_expected_entries():
    o = jax.random.normal(jax.random.PRNGKey(0), (88,)); a = jax.random.normal(jax.random.PRNGKey(1), (v3.ACT_DIM,))
    th = jax.random.uniform(jax.random.PRNGKey(2), (4,), minval=-3.0, maxval=3.0)
    o = o.at[68:72].set(jnp.sin(th)).at[72:76].set(jnp.cos(th))                       # sinθ／cosθ 要是單位向量（鏡像會經 atan2 重建）
    assert np.allclose(v3.mirror_obs(v3.mirror_obs(o)), o) and np.allclose(v3.mirror_act(v3.mirror_act(a)), a)
    m = np.asarray(v3.mirror_obs(o)); o = np.asarray(o)
    assert m[1] == -o[1] and m[3] == -o[3] and m[5] == -o[5] and m[4] == o[4]          # grav y、gyro x/z 變號
    assert m[6] == -o[9] and m[7] == o[10] and m[8] == o[11]                             # FR ABAD ← −FL ABAD、髖膝直接換
    assert m[30] == o[31] and m[35] == -o[35] and m[36] == -o[36] and m[34] == o[34] and m[39] == -o[39]
    am = np.asarray(v3.mirror_act(a)); a = np.asarray(a)
    assert am[0] == a[0] and am[5] == a[5] and am[1] == a[2] and am[7] == -a[7] and am[6] == a[6] and am[8] == a[9] and am[12] == -a[15]


def _rollout_sym(env, cmd, steps):
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(0))
    P0 = v3.step_pattern(jnp.array(cmd), env.ref)
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6, "ph": P0["ph"],   # 相位偏移直接到位，不從隨機初值滑
                        "imu_q": jnp.array([1.0, 0.0, 0.0, 0.0]), "gyro_bias": jnp.zeros(3)})
    O, Q = [], []
    for _ in range(steps):
        s = js(s, jnp.zeros(v3.ACT_DIM)); O.append(np.asarray(s.obs)); Q.append(np.asarray(s.pipeline_state.qpos[v3.LEG_QPOS_IDX]))
    return np.array(O), np.array(Q)


@pytest.mark.parametrize("cmd,steps", [((0.0, 0.20, 0.0), 40), ((0.0, 0.0, 1.3), 25), ((0.3, 0.15, 0.0), 40)])
def test_mirrored_command_gives_mirrored_obs(cmd, steps):
    """模型對稱（質心 1.5 mm 除外）＋ 產生器鏡像 → 左指令的 obs 鏡像 ≈ 右指令的 obs。驗的是映射本身；
    原地轉名目開迴路 3 s 會倒，窗口取短一點（左右軌跡本來就會分歧）。產生器特徵欄（40:）與物理無關，要幾乎精確。"""
    env = v3.DualModeEnv(gains="factory", weights=v3.W36, ref=dict(cyc_amp_rand=False), push=False)   # W36：右轉表＝左轉鏡像
    OL, QL = _rollout_sym(env, cmd, steps); OR, QR = _rollout_sym(env, (cmd[0], -cmd[1], -cmd[2]), steps)
    q_mir = np.asarray(jnp.stack([v3.mirror_obs(jnp.zeros(88).at[6:18].set(q))[6:18] for q in QL]))
    assert np.abs(q_mir - QR).max() < 0.08, np.abs(q_mir - QR).max()
    o_mir = np.asarray(jnp.stack([v3.mirror_obs(o) for o in OL]))
    d = np.abs(o_mir - OR)
    assert d[:, 64:84].max() < 0.02, d[:, 64:84].max(axis=0)                     # amp／sinθ／cosθ／cyc_abad／cyc_knee：與物理無關，要精確
    assert d[:, 34:40].max() < 0.01                                               # cmd／s4／s_arc／head_err
    assert d[:, :6].max() < 0.1 and np.percentile(d[:, 6:34], 90) < 0.8           # 感測欄：觸地尖峰讓 max 沒意義，看 p90
    assert np.percentile(d[:, 84:88], 90) < 0.05                                  # 輪命令


def test_decouple_flag_default_off_and_scales_as_designed():
    assert v3.REF["cyc_lat_decouple"] is False
    ref = dict(v3.REF, **v3.REF_FACTORY, cyc_lat_decouple=True)
    for vy, k_exp in ((0.05, 0.3), (0.12, 0.3), (0.16, 0.524), (0.20, 0.748), (0.30, 1.0)):
        cmd = jnp.array((0.0, vy, 0.0)); A = v3.step_pattern(cmd, ref)["A"]
        d = np.asarray(v3.cycle_offsets(jnp.float32(1.0), cmd, A, ref)["delta"]).reshape(4, 3)
        ref0 = dict(ref, cyc_lat_decouple=False, cyc_lat_amp_clip=(1.0, 1.0), cyc_amp_lat=1.0)    # 表本身（amp_l = 1）
        d0 = np.asarray(v3.cycle_offsets(jnp.float32(1.0), cmd, v3.step_pattern(cmd, ref0)["A"], ref0)["delta"]).reshape(4, 3)
        nz = np.abs(d0) > 1e-4
        ratio = d[nz] / d0[nz]; col = np.broadcast_to(np.arange(3), (4, 3))[nz]
        assert np.allclose(ratio[col == 0], k_exp, atol=0.02), (vy, ratio[col == 0])
        assert np.allclose(ratio[col > 0], 1.75, atol=0.02), (vy, ratio[col > 0])


def test_w37b_values_and_lift_nonneg_mapping():
    assert v3.W37["CYC_LAT_DECOUPLE"] is True and v3.W37["CMD_VY"] == (0.04, 0.22) and "LIFT_NONNEG" not in v3.W37
    assert v3.W37B["LIFT_NONNEG"] is True and v3.W37B["W_STEP"] == 2.5
    assert v3.REF["cyc_lat_wz_ff"] == 0.4 and v3.REF["cyc_lat_vx_ff"] == (0.15, -0.43, 0.05, 0.10)
    # lift 通道：a ≤ 0 → 1.0（死區＝名目）、a = +∞ → 1.4；預設路徑 a=−∞ → 0.6
    for a5, exp in ((-3.0, 1.0), (0.0, 1.0), (3.0, 1.0 + v3.LIFT_SCALE * float(jnp.tanh(3.0)))):
        A = v3.act_split(jnp.zeros(v3.ACT_DIM).at[5].set(a5))
        lift = 1.0 + jnp.maximum(A["lift"] - 1.0, 0.0)
        assert abs(float(lift) - exp) < 1e-6
    assert abs(float(v3.act_split(jnp.zeros(v3.ACT_DIM).at[5].set(-3.0))["lift"]) - (1.0 - v3.LIFT_SCALE * float(jnp.tanh(3.0)))) < 1e-6


@pytest.mark.parametrize("weights,vx_tol,head_tol", [(v3.W37, 0.10, 70.0), (v3.W37B, 0.03, 45.0)])
def test_decouple_env_zero_action_lifts_tracks_and_feedforward(weights, vx_tol, head_tol):
    """W37／W37B 都走解耦產生器（前饋是 REF 預設，兩者都有）；W37B 零動作＝名目（lift 死區）。"""
    env = v3.DualModeEnv(gains="factory", weights=weights, ref=dict(cyc_amp_rand=False), push=False)
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    s = jr(jax.random.PRNGKey(0)); s = s.replace(info={**s.info, "cmd": jnp.array((0.0, 0.20, 0.0)), "cmd2": jnp.array((0.0, 0.20, 0.0)), "t_switch": 10 ** 6})
    VY, VX, AP = [], [], []
    for i in range(500):
        s = js(s, jnp.zeros(v3.ACT_DIM)); VY.append(float(s.metrics["vy"])); VX.append(float(s.metrics["vx"])); AP.append(np.asarray(s.info["apex_last"]) * 1000)
        assert float(s.done) == 0.0
    assert 0.14 < np.mean(VY[100:]) < 0.26 and np.mean(AP[200:], 0)[1] > 15 and np.mean(AP[200:], 0)[3] > 15
    assert abs(np.mean(VX[100:])) < vx_tol and abs(float(s.metrics["head_deg"])) < head_tol
