"""釘住 v1b 的 WheelEnv() 預設路徑（obs／priv／reward／qpos／ctrl／done 逐位元）。

v1c 把鏡像／roll／impact／延遲／IMU 偏差都做成建構子旗標，**預設路徑必須逐位元不變**。
golden 檔不存在時會自己產生並 skip，產生後要 commit；之後任何一步讓它失敗，就是真的動到 v1b 了 —— 先查清楚再決定，不要放寬容差。

兩段 rollout：
- 長回合：WheelEnv() 全預設、12 步（初段的 obs／reward／動力學）。
- 短回合：WheelEnv(episode_len=16)、20 步 —— 走過 t=8 的指令 B 切換、push_t∈[8,11] 的推力、
  t=16 的內部重生（done_env → _next_tile／_spawn_state／new_info 合併／sel）。直接呼叫 env.step，不經 EpisodeWrapper。

golden 產生環境（2026-10-20）：jax 0.10.2（backend cpu）、mujoco 3.10.0、brax 0.14.2、numpy 2.4.6。
換平台／換版本造成的失敗：確認 env 沒改之後重新產生 golden，不要放寬容差。
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
SHORT_LEN, SHORT_STEPS = 16, 20     # 短回合：t 跨過 16 → 內部重生
NAMES = ("obs", "priv", "reward", "qpos", "ctrl", "done")


def _run(env, steps):
    """固定 seed、固定動作序列；回傳 NAMES 順序的陣列，形狀 (seed, step, ...)。"""
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    out = {n: [] for n in NAMES}
    for sd in SEEDS:
        s = jit_reset(jax.random.PRNGKey(sd))
        rec = {n: [] for n in NAMES}
        for i in range(steps):
            a = jnp.sin(jnp.arange(rw.ACT_DIM) * 0.7 + i * 0.3 + sd) * 0.5     # 非零、可重現、四腿都動
            s = jit_step(s, a)
            rec["obs"].append(np.asarray(s.obs["state"])); rec["priv"].append(np.asarray(s.obs["privileged_state"]))
            rec["reward"].append(float(s.reward)); rec["done"].append(float(s.done))
            rec["qpos"].append(np.asarray(s.pipeline_state.qpos)); rec["ctrl"].append(np.asarray(s.pipeline_state.ctrl))
        for n in NAMES:
            out[n].append(rec[n])
    return {n: np.array(v) for n, v in out.items()}


def _rollout():
    """長回合（WheelEnv() 全預設：地形、雜訊、推力、隨機指令）＋短回合（episode_len=16，走到重生）。"""
    long_ = _run(rw.WheelEnv(), STEPS)
    short = _run(rw.WheelEnv(episode_len=SHORT_LEN), SHORT_STEPS)
    return {**long_, **{f"{n}_s": v for n, v in short.items()}}


def test_v1b_default_rollout_is_bitwise_unchanged():
    got = _rollout()
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GOLDEN, **got)
        pytest.skip(f"golden 不存在，已產生 {GOLDEN} —— 請 commit 後重跑")
    g = np.load(GOLDEN)
    assert set(g.files) == set(got), f"golden 欄位 {sorted(g.files)} ≠ 測試欄位 {sorted(got)}"
    for name, arr in got.items():
        np.testing.assert_allclose(arr, g[name], rtol=0, atol=0, err_msg=f"{name} 與 golden 不同 —— v1b 預設路徑被動到了")


def test_golden_actually_moves():
    """機身真的在動、短回合真的重生、obs 有雜訊差異，否則擋不住回歸。"""
    if not GOLDEN.exists():
        pytest.skip("golden 尚未產生")
    g = np.load(GOLDEN)
    assert np.ptp(g["qpos"][:, :, :3], axis=1).max() > 1e-3                      # 12 步內機身位置有變
    jump = np.linalg.norm(g["qpos_s"][:, SHORT_LEN - 1, :2] - g["qpos_s"][:, SHORT_LEN - 2, :2], axis=-1)
    assert (jump > 0.05).all(), f"短回合 t=16 沒看到重生跳躍：{jump}"            # 每個 seed 都在第 16 步重生到別的方塊
    assert np.abs(g["obs"][:, :, :53] - g["priv"][:, :, :53]).max() > 1e-3     # state 帶雜訊、priv 乾淨
