"""狗上純 numpy 的 CPG-RL policy 推論（D1 Max，2026-09-09）。

狗上只有 Python 3.10 ＋ numpy 1.21.5，沒有 jax/brax/mujoco。本檔只 import numpy 與標準庫。

網路（brax PPO，`export_policy_np.py` 匯出）：
    obs → (obs − mean) / std → [Linear → swish] ×3 → Linear(2·act) → 取前 act 為 loc → tanh(loc)
deterministic 動作就是 tanh(loc)（brax `NormalTanhDistribution.mode`）。
式子的正確性**不靠這段註解**，靠 `tests/test_policy_np.py` 對 brax 前向 1000 筆逐點比對。

動作 → CPG 指令（`act_to_cmd`）與 `inference/local_infer_max.py` 逐行同義，常數由測試釘住同值。
回傳的 mux/muy/om 是 **MJCF 腿序（FR, FL, RR, RL）**，餵狗上 `cpg.step` 前要經
`rl_obs.to_shm_legs()` 轉成 `{fl, fr, bl, br}`。
"""
from __future__ import annotations

import json

import numpy as np

# ⚠️ 必須與 inference/{max_model,local_infer_max,gait_baseline} 同值（test_policy_np 釘住）。
#    不同值的話 policy 輸出的同一個數字會被解成不同的頻率／位移，而且不會報錯。
MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 2.0
SWAY_MAX, SWAY_SLEW = 0.060, 0.004
A_MU_X, A_MU_Y, A_OMEGA = 1.80, 1.50, 1.4          # BASELINE_A
LAYOUT_DIMS = {"full": 14, "nomux": 10}
RAMP_STEPS = 50                                    # v2.3 起步淡入（= rl_env_max RAMP_STEPS）
INFER_BUDGET_MS = 10.0                             # 超過就退回基準動作（50 Hz 預算 20 ms）


def _inv(u, lo, hi):
    return float(np.arctanh(np.clip(2 * (u - lo) / (hi - lo) - 1, -0.999, 0.999)))


def baseline_action(layout: str = "nomux") -> np.ndarray:
    """開迴路 A 基準步態對應的固定動作（`act_to_cmd` 的反函式；sway=0）。"""
    if layout == "full":
        return np.array([_inv(A_MU_X, MU_MIN, MU_MAX), _inv(A_MU_Y, MU_MIN, MU_MAX),
                         _inv(A_OMEGA, OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])
    if layout == "nomux":
        return np.array([_inv(A_MU_Y, MU_MIN, MU_MAX),
                         _inv(A_OMEGA, OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])
    raise ValueError(layout)


def act_to_cmd(a: np.ndarray, layout: str = "nomux"):
    """動作 → (mux(4), muy(4), omega(4), sway_target(2))，MJCF 腿序。"""
    a = np.tanh(np.asarray(a, dtype=float))
    if layout == "full":
        leg = a[:12].reshape(4, 3)
        mux = (leg[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
        muy = (leg[:, 1] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
        om = (leg[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
        return mux, muy, om, a[12:14] * SWAY_MAX
    if layout == "nomux":
        leg = a[:8].reshape(4, 2)
        mux = np.full(4, A_MU_X)
        muy = (leg[:, 0] + 1) / 2 * (MU_MAX - MU_MIN) + MU_MIN
        om = (leg[:, 1] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
        return mux, muy, om, a[8:10] * SWAY_MAX
    raise ValueError(layout)


def slew_sway(prev, tgt):
    return prev + np.clip(np.asarray(tgt, dtype=float) - prev, -SWAY_SLEW, SWAY_SLEW)


def _swish(x):
    # sigmoid 用 tanh 形式：大負值時 exp(−x) 會溢位出警告（結果對、但吵）
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))


class Policy:
    def __init__(self, z):
        self.obs_dim = int(z["obs_dim"])
        self.act_dim = int(z["act_dim"])
        self.preset = str(z["preset"])
        self.layout = str(z["layout"])
        self.baseline = json.loads(str(z["baseline_json"]))
        self.src_sha256 = str(z["src_sha256"])
        self.mean = np.asarray(z["mean"], dtype=np.float32)
        self.std = np.asarray(z["std"], dtype=np.float32)
        self.W = [np.asarray(z[f"W{i}"], dtype=np.float32) for i in range(4)]
        self.b = [np.asarray(z[f"b{i}"], dtype=np.float32) for i in range(4)]
        assert self.mean.shape == (self.obs_dim,) and self.W[0].shape[0] == self.obs_dim
        assert self.W[3].shape[1] == 2 * self.act_dim
        assert LAYOUT_DIMS[self.layout] == self.act_dim

    def infer(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        assert obs.size == self.obs_dim, f"obs 應為 {self.obs_dim} 維，實得 {obs.size}"
        x = (obs - self.mean) / self.std
        for i in range(3):
            x = _swish(x @ self.W[i] + self.b[i])
        loc = (x @ self.W[3] + self.b[3])[: self.act_dim]
        return np.tanh(loc).astype(float)


def load(path: str) -> Policy:
    with np.load(path, allow_pickle=False) as z:
        return Policy({k: z[k] for k in z.files})
