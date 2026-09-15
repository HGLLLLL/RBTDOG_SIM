"""釘住 v3.3（gains="kp250"）的 rollout 數值。

v3.4f 的重構把 scene／撓度／護欄／輪空間參數化，**預設路徑必須逐位元不變**。
這支測試是那件事唯一的證據。golden 檔不存在時會自己產生並 skip，
產生後要 commit；之後任何一步讓它失敗，就是真的動到 v3.3 了 —— 先查清楚再決定，不要放寬容差。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_v3 as v3  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "rl_env_v3_golden.npz"
CMDS = ((0.5, 0.0, 0.0), (0.0, 0.08, 0.0), (0.0, 0.0, 1.3))   # 輪行／平移／原地轉，三條路徑都踩到
STEPS = 12


def _rollout():
    """固定 seed、固定動作序列跑三個指令，回傳 (obs, reward, qpos, ctrl) 四個 array。"""
    env = v3.DualModeEnv(ref=dict(cyc_amp_rand=False))
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    O, R, Q, C = [], [], [], []
    for ci, cmd in enumerate(CMDS):
        s = jit_reset(jax.random.PRNGKey(ci))
        s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
        o, r, q, c = [], [], [], []
        for i in range(STEPS):
            # 非零、可重現的動作：三條路徑的殘差／sway／輪殘差都要被踩到
            a = jnp.sin(jnp.arange(v3.ACT_DIM) * 0.7 + i * 0.3) * 0.5
            s = jit_step(s, a)
            o.append(np.asarray(s.obs)); r.append(float(s.reward))
            q.append(np.asarray(s.pipeline_state.qpos)); c.append(np.asarray(s.pipeline_state.ctrl))
        O.append(o); R.append(r); Q.append(q); C.append(c)
    return np.array(O), np.array(R), np.array(Q), np.array(C)


def test_v33_rollout_is_bitwise_unchanged():
    obs, rew, qpos, ctrl = _rollout()
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GOLDEN, obs=obs, reward=rew, qpos=qpos, ctrl=ctrl)
        pytest.skip(f"golden 不存在，已產生 {GOLDEN} —— 請 commit 後重跑")
    g = np.load(GOLDEN)
    for name, got in (("obs", obs), ("reward", rew), ("qpos", qpos), ("ctrl", ctrl)):
        np.testing.assert_allclose(got, g[name], rtol=0, atol=0,
                                   err_msg=f"{name} 與 golden 不同 —— v3.3 預設路徑被動到了")


def test_golden_covers_all_three_wheel_paths():
    """golden 的三個指令必須真的讓輪子動，否則它擋不住輪控制律的回歸。"""
    if not GOLDEN.exists():
        pytest.skip("golden 尚未產生")
    g = np.load(GOLDEN)
    ctrl = g["ctrl"][:, :, np.asarray(v3.WHEEL_ACT_IDX)]
    assert np.abs(ctrl).max(axis=(1, 2)).min() > 0.1, f"有指令的輪 ctrl 全是 0：{np.abs(ctrl).max(axis=(1, 2))}"
