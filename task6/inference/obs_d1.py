"""D1 EDU 的 69 維 observation。

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
| cpg          | 24 | 自算 |

總計 69 維。

刻意不含兩項：

1. 機身線速度——LowLevel 沒有這個量（getBodyVelocity 只存在於 HighLevel）。
   reward 仍可使用模擬真值速度，因為 reward 只在模擬計算。
2. 腳觸地布林——關卡 3 實測證明這台輪足版沒有可用的觸地訊號。
   0.901 kg 的輪子（點足版只有 0.063 kg）讓擺動相的膝力矩與位置誤差都**大於**
   站立相，訊號方向是反的：以 MuJoCo 接觸為真值、摩擦 0.8、1600 樣本測得
   膝力矩站立相 p05=1.55 N·m 而擺動相 p95=8.76 N·m，膝位置誤差站立相
   p05=0.017 rad 而擺動相 p95=0.144 rad。膝角速度重疊，Jacobian 反推足端 Fz
   尾部重疊、僅約 85% 準確。四個候選訊號全部不可用，故整段移除。
"""
import numpy as np

from cpg_d1 import w2b
from d1_model import ACT_DIM, HOME12

OBS_LAYOUT = [
    ("gravity", 3),
    ("gyro", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("cmd", 3),
    ("last_action", 12),
    ("cpg", 24),
]

_DOWN = np.array([0.0, 0.0, -1.0])


def build_obs(d, c: dict, cmd: np.ndarray, last_a: np.ndarray) -> np.ndarray:
    """組 69 維 observation。d 為 mujoco.MjData（或具備同名欄位的物件）。"""
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
        c["rx"], c["rx_d"], c["ry"], c["ry_d"],     # cpg 16
        np.sin(c["theta"]), np.cos(c["theta"]),     # cpg 8
    ]).astype(np.float32)
