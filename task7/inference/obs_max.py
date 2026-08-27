"""D1 Max 的 68 維 observation —— Colab 訓練端與本機推論端的**共用定義**。

## 設計約束：每一欄都必須是實機底層真的拿得到的量

實機的路徑是「凍結 `mc_ctrl` → 直接讀寫 `/dev/shm`」，高層 SDK 在那之後就死了。

| 欄位 | 維度 | 實機來源 | 驗證程度 |
|---|---|---|---|
| `gravity` | 3 | `/dev/shm/imu_central` quat（**xyzw**）轉機身系 | ⚠️ 見下 |
| `gyro` | 3 | `/dev/shm/imu_central` gyro | ⚠️ 見下 |
| `joint_pos` | 12 | `/dev/shm/joint_state` pos − `HOME12` | ★ 實機驗過是 1 kHz 活的串流 |
| `joint_vel` | 12 | `/dev/shm/joint_state` vel | ★ 同上 |
| `cmd` | 2 | `(vx, wz)`，自產 | — |
| `last_action` | 12 | 自存 | — |
| `cpg` | 24 | `rx, rx_d, ry, ry_d, sin θ, cos θ`，自算 | — |

⚠️ **IMU 那兩欄尚未驗證**：`imu_central` 目前只有離線快照解碼過，**沒驗過是不是活的
串流**；而且 `xyzw` 的順序是「取樣當下機身剛好水平」推出來的，不是刻意做的平放實驗。
四元數順序若錯 → 重力向量翻掉 → policy 直接廢掉。
上實機前必須先跑 `task7/docs/現場操作卡_IMU平放複核.md`。

## 刻意不含的三項

1. **機身線速度** —— 底層沒有這個量（高層 SDK 才有，接管後就死了）。
   reward 仍可使用模擬真值速度，因為 reward 只在模擬計算、訓練完就丟棄：
   訓練產物只有 policy 網路權重，而 policy 是純函式 obs → action，
   推論時根本不會呼叫到 reward 的任何一行。
2. **足端觸地布林** —— 沒有感測器。task6 在同型輪足（D1 EDU）上實測四個候選訊號
   全部不可用：0.9 kg 的輪子讓**擺動相**的膝力矩與位置誤差**大於站立相**，
   訊號方向是反的（以 MuJoCo 接觸為真值、1600 樣本：膝力矩站立相 p05=1.55 N·m
   而擺動相 p95=8.76 N·m）。D1 Max 的輪更重（1.51 kg），只會更嚴重。
3. **輪子角速度** —— 實機讀得到，但實機從未走過路，輪速在真實地面（地毯 vs 磁磚）
   的行為沒有任何資料可對照，而 policy 又控制不了它（輪子固定阻尼）。
   放進去等於押一個沒驗證過的量。

## 為什麼 cmd 是 2 維而不是 3 維

只放實際會變的兩個量 `(vx, wz)`。側移 `vy` 不在本輪的動作範圍內：
ABAD 行程只有 −0.697 ~ +0.523 rad 且左右鏡像，全向移動另案處理。
放一個恆為 0 的欄位只是讓網路多學一個常數。

⚠️ **欄位順序一旦改動，既有權重全部報廢。** 有 `test_obs_max.py` 釘住。
"""
import numpy as np

from cpg_max import w2b
from max_model import HOME12, LEG_QPOS_IDX, LEG_QVEL_IDX

ACT_DIM = 12          # 每腿 (mux, muy, omega)

OBS_LAYOUT = [
    ("gravity", 3),
    ("gyro", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("cmd", 2),
    ("last_action", ACT_DIM),
    ("cpg", 24),
]
OBS_DIM = sum(d for _, d in OBS_LAYOUT)

_DOWN = np.array([0.0, 0.0, -1.0])


def slice_of(name: str) -> slice:
    """某個欄位在 obs 向量裡的位置。給測試與除錯用，推論路徑不需要。"""
    i = 0
    for n, d in OBS_LAYOUT:
        if n == name:
            return slice(i, i + d)
        i += d
    raise KeyError(f"沒有這個欄位：{name}")


def build_obs(d, c: dict, cmd, last_a) -> np.ndarray:
    """組 68 維 observation。`d` 為 `mujoco.MjData`（或具備同名欄位的物件）。

    ⚠️ 入口一定要擋維度。`np.concatenate` 對長度錯誤的輸入**不會報錯**，
       會靜默產生錯誤維度的 obs，而錯誤維度的 obs 會讓訓練好的權重直接失效。
    """
    cmd = np.asarray(cmd, dtype=np.float64).reshape(-1)
    last_a = np.asarray(last_a, dtype=np.float64).reshape(-1)
    assert cmd.size == 2, f"cmd 應為 2 維 (vx, wz)，實得 {cmd.size}"
    assert last_a.size == ACT_DIM, f"last_a 應為 {ACT_DIM} 維，實得 {last_a.size}"

    o = np.concatenate([
        w2b(d.qpos[3:7], _DOWN),           # gravity 3
        d.qvel[3:6],                       # gyro 3
        # ⚠️ 輪關節夾在腿關節中間，位址不連續。`qpos[7:19]` 會把輪角當關節角讀進去，
        #    IK 變奇異矩陣而不報錯（task6 踩過）。
        d.qpos[LEG_QPOS_IDX] - HOME12,     # joint_pos 12
        d.qvel[LEG_QVEL_IDX],              # joint_vel 12
        cmd,                               # cmd 2
        last_a,                            # last_action 12
        c["rx"], c["rx_d"], c["ry"], c["ry_d"],       # cpg 16
        np.sin(c["theta"]), np.cos(c["theta"]),       # cpg 8
    ]).astype(np.float32)
    assert o.size == OBS_DIM, f"obs 應為 {OBS_DIM} 維，實得 {o.size}"
    return o
