"""釘住 v1b 的 WheelEnv() 預設路徑（obs／reward／qpos／ctrl 逐位元）。

v1c 把鏡像／roll／impact／延遲／IMU 偏差都做成建構子旗標，**預設路徑必須逐位元不變**。
golden 檔不存在時會自己產生並 skip，產生後要 commit；之後任何一步讓它失敗，就是真的動到 v1b 了 —— 先查清楚再決定，不要放寬容差。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
jnp = jax.numpy
import rl_env_wheel as rw  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "rl_env_wheel_golden.npz"
SEEDS = (0, 1, 2)      # 三個 reset：出生方塊、指令、雜訊都不同
STEPS = 12


def _rollout():
    """WheelEnv() 全預設（地形、雜訊、推力、隨機指令），固定 seed、固定動作序列。"""
    env = rw.WheelEnv()
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    O, P, R, Q, C = [], [], [], [], []
    for sd in SEEDS:
        s = jit_reset(jax.random.PRNGKey(sd))
        o, p, r, q, c = [], [], [], [], []
        for i in range(STEPS):
            a = jnp.sin(jnp.arange(rw.ACT_DIM) * 0.7 + i * 0.3 + sd) * 0.5     # 非零、可重現、四腿都動
            s = jit_step(s, a)
            o.append(np.asarray(s.obs["state"])); p.append(np.asarray(s.obs["privileged_state"])); r.append(float(s.reward))
            q.append(np.asarray(s.pipeline_state.qpos)); c.append(np.asarray(s.pipeline_state.ctrl))
        O.append(o); P.append(p); R.append(r); Q.append(q); C.append(c)
    return np.array(O), np.array(P), np.array(R), np.array(Q), np.array(C)


def test_v1b_default_rollout_is_bitwise_unchanged():
    obs, priv, rew, qpos, ctrl = _rollout()
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GOLDEN, obs=obs, priv=priv, reward=rew, qpos=qpos, ctrl=ctrl)
        pytest.skip(f"golden 不存在，已產生 {GOLDEN} —— 請 commit 後重跑")
    g = np.load(GOLDEN)
    for name, got in (("obs", obs), ("priv", priv), ("reward", rew), ("qpos", qpos), ("ctrl", ctrl)):
        np.testing.assert_allclose(got, g[name], rtol=0, atol=0, err_msg=f"{name} 與 golden 不同 —— v1b 預設路徑被動到了")


def test_golden_actually_moves():
    """三個 seed 的輪 ctrl 都非零、obs 有雜訊差異，否則擋不住回歸。"""
    if not GOLDEN.exists():
        pytest.skip("golden 尚未產生")
    g = np.load(GOLDEN)
    assert np.abs(g["ctrl"][:, :, -4:]).max(axis=(1, 2)).min() > 0.1
    assert np.abs(g["obs"][:, :, :53] - g["priv"][:, :, :53]).max() > 1e-3     # state 帶雜訊、priv 乾淨
