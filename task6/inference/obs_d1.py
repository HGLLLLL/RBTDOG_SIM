"""D1 EDU 的 73 維 observation。

設計約束：每一個欄位都必須是實機 LowLevel SDK 拿得到的量。
（官方 FAQ：Highlevel 與 Lowlevel 不可同時使用，所以只能用 LowLevel 的 getter。）

| 欄位 | 維度 | 實機來源 |
|---|---|---|
| gravity      | 3  | getQuaternion() → 轉機身系 |
| gyro         | 3  | getBodyGyro() |
| joint_pos    | 12 | motorState.q_*  減 HOME12 |
| joint_vel    | 12 | motorState.qd_* |
| cmd          | 3  | 自產 |
| last_action  | 12 | 自存 |
| foot_contact | 4  | |motorState.tau_knee_fb| > TAU_CONTACT |
| cpg          | 24 | 自算 |

刻意不含機身線速度：LowLevel 沒有這個量（getBodyVelocity 只存在於 HighLevel）。
reward 仍可使用模擬真值速度，因為 reward 只在模擬計算。
"""
import numpy as np

from cpg_d1 import w2b
from d1_model import ACT_DIM, HOME12, TAU_CONTACT

OBS_LAYOUT = [
    ("gravity", 3),
    ("gyro", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("cmd", 3),
    ("last_action", 12),
    ("foot_contact", 4),
    ("cpg", 24),
]

_DOWN = np.array([0.0, 0.0, -1.0])


def foot_contact(actuator_force: np.ndarray, knee_aid) -> np.ndarray:
    """膝關節力矩門檻判觸地。

    取絕對值是因為 D1 的 knee 軸為 +y，站立相力矩符號與 Go2 相反；
    用絕對值可免除符號約定爭議，且模擬（actuator_force）與實機（tau_knee_fb）公式一致。
    """
    tau = np.abs(np.asarray(actuator_force)[knee_aid])
    return (tau > TAU_CONTACT).astype(np.float32)


def build_obs(d, c: dict, cmd: np.ndarray, last_a: np.ndarray, knee_aid) -> np.ndarray:
    """組 73 維 observation。d 為 mujoco.MjData（或具備同名欄位的物件）。"""
    # np.concatenate 對長度錯誤的輸入不會報錯，會靜默產生錯誤維度的 obs，
    # 而錯誤維度的 obs 會讓訓練好的權重直接失效，故在入口先擋掉。
    cmd = np.asarray(cmd, dtype=np.float64).reshape(-1)
    last_a = np.asarray(last_a, dtype=np.float64).reshape(-1)
    assert cmd.size == 3, f"cmd 應為 3 維，實得 {cmd.size}"
    assert last_a.size == ACT_DIM, f"last_a 應為 {ACT_DIM} 維，實得 {last_a.size}"

    quat = d.qpos[3:7]
    return np.concatenate([
        w2b(quat, _DOWN),            # gravity 3
        d.qvel[3:6],                 # gyro 3
        d.qpos[7:19] - HOME12,       # joint_pos 12
        d.qvel[6:18],                # joint_vel 12
        cmd,                         # cmd 3
        last_a,                      # last_action 12
        foot_contact(d.actuator_force, knee_aid),   # foot_contact 4
        c["rx"], c["rx_d"], c["ry"], c["ry_d"],     # cpg 16
        np.sin(c["theta"]), np.cos(c["theta"]),     # cpg 8
    ]).astype(np.float32)
