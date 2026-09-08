"""D1 Max 基準步態（`walk`）的凍結參數 —— **唯一真實來源**。

這組參數是 RL 的基準與對照組。`cpg_walk_max.GAITS["walk"]` 與
Colab 訓練 notebook 都必須引用這裡，不得各自寫死。

## 為什麼要凍結

`walk` 在多擾動掃描（`task7/outputs/sweep_*.json`）中 20 / 60 / 120 / **180 秒**
每格 12 個擾動**全部 0 跌倒**，關節超限 / 力矩飽和 / IK 縮限全部 0.00%。
也就是「平地不跌倒」已經達成且複驗過。RL 要拿它當起點與對照，
所以它必須是一個**不會被順手改掉**的常數集合，而不是散在腳本裡的預設值。

掃描期間就發生過一次「跑到一半共用 MJCF 被別條線改掉」（`533e91a` 改了膝關節
`range`），事後之所以判定得出來，只因為 `lim_pct` 這個診斷欄一直有印。
步態常數比 MJCF 更容易被順手改掉，而改掉之後 RL 的基準就悄悄變了。

## 每個數字的判準來源（全部是多擾動掃描的結論，不是單次樣本）

| 參數 | 判準 |
|---|---|
| `duty=0.80` | ≤0.70 是 **12/12 全跌**；0.80 彈跳最小（17.2 mm，0.85 是 21.1、0.90 是 19.9） |
| `omega=1.4` | 再高就帶進 0.4–0.9 °/s 的偏航，且符號不可預測 |
| `mu_y=1.50` | → `fy = 0`，橫向偏移恰為 0（有測試釘到 1e-15）；1.75 是 **12/12 全跌** |
| `x_off=-0.040` | 平均俯仰**單調過零**在 −40.8 mm（12 擾動，格內全距 0.01–0.02°、雜訊僅佔 2%） |
| `g_c=0.08` | 實際離地約 100 mm，對上原廠 `leg_height=0.10`；再大俯仰開始劣化 |
| `z_sag` | ＝ `max_model.STATIC_SAG`，位置伺服的靜態撓度補償，**只加在擺動相** |
| `wheel_mode="damp"` | 輪子給位置增益實測會造成 **+39° 偏航失控**（原廠那個 Kp=60 是搭配「每步重給目標角」的 RL 策略用的） |

⚠️ **配平點 `x_off` 會隨三件事移動**：輪摩擦、腿關節摩擦、`d_step`。
   動到任何一個都必須用 `cpg_sweep_max.py --plan trim` 重掃，不可以沿用。

⚠️ **不要用 `speed_path` 引用行進速度**，它把機身左右搖擺算成前進，實測高估 68%。
   行進速度一律看 `speed_travel`（以一個步態週期為步長重算的路徑長）。
"""
import cpg_max
from max_model import STATIC_SAG

# 凍結值。純量字典，好比對、好序列化、好在 notebook 裡 assert。
BASELINE = {
    "gait": "walk",
    "duty": 0.80,
    "omega": 1.4,
    "mu_x": 1.80,
    "mu_y": 1.50,
    "d_step": 0.10,
    "d_step_y": 0.12,
    "x_off": -0.040,
    "g_c": 0.08,
    "z_sag": STATIC_SAG,
    "wheel_mode": "damp",
}

MU_Y = BASELINE["mu_y"]
D_STEP_Y = BASELINE["d_step_y"]


# =============================================================================
# kp=250 的重掃結果（2026-08-27，應實機線要求）
# =============================================================================
# ⚠️ 這**不是**新的凍結基準。實機線的決定是「下一階段的實機測試用 kp=250，
#    最終步態的增益之後再議」，所以 `BASELINE` 維持 kp=120 那組不動。
#    這組是給實機 M9（軌跡播放）用的，數據見
#    `docs/CPG步態_完整結果_2026-08-27.md`。
#
# 與 BASELINE 的差異全部是重掃出來的，不是換算的：
#   duty   0.80 → 0.85     x_off  −40 → **−50 mm**（俯仰單調過零於 −49.7）
#   d_step 0.10 → **0.12**  z_sag  32.5 → **36 mm**（★ 用實機錨點，不是模擬掃出來的）
#
# ★ z_sag 為什麼不用模擬值：實機線實測**模擬在 z 方向系統性高估順從性約 1.9 倍**
#   （靜態撓度 2.13×、擺動離地 1.81×），但 **x 方向是準的（1.01×）**。
#   所以 d_step / x_off 信模擬，z_sag 用實機量到的 36 mm。
BASELINE_KP250 = {
    "gait": "walk",
    "duty": 0.85,
    "omega": 1.4,
    "mu_x": 1.80,
    "mu_y": 1.50,
    "d_step": 0.12,
    "d_step_y": 0.12,
    "x_off": -0.050,
    "g_c": 0.08,
    "z_sag": 0.036,
    "kp3": [250.0, 250.0, 250.0],
    "kd3": [5.0, 5.0, 5.0],
    "wheel_mode": "damp",
}


def kp250_gait() -> dict:
    """`cpg_walk_max.GAITS["walk_kp250"]` 用的 dict。"""
    b = BASELINE_KP250
    return dict(phase=cpg_max.PHASE_WALK, duty=b["duty"], omega=b["omega"],
                mu_x=b["mu_x"], x_off=b["x_off"], d_step=b["d_step"], g_c=b["g_c"])


def walk_gait() -> dict:
    """`cpg_walk_max.GAITS["walk"]` 用的 dict（含相位物件）。

    相位是 ndarray，不放進 `BASELINE` 是為了讓 `BASELINE` 保持可直接 `==` 比對。
    """
    return dict(phase=cpg_max.PHASE_WALK, duty=BASELINE["duty"],
                omega=BASELINE["omega"], mu_x=BASELINE["mu_x"],
                x_off=BASELINE["x_off"], d_step=BASELINE["d_step"],
                g_c=BASELINE["g_c"])


# =============================================================================
# ★ RL v2 基準（2026-09-08）：實機 trip17 兩趟零中止走完的 A_kp250_walk
# =============================================================================
# 每個數字直接對應 `outputs/A_kp250_walk.json` 的 params（測試釘住逐欄相等）。
# 與 BASELINE 的差異：LS 相位序列（E 文件）、kp250/abad60/kd2（H 文件）、
# x_off −30 / g_c 0.048 / z_sag 0.036（G 文件，kp250 重掃 + 實機錨點）。
BASELINE_A = {
    "gait": "walk_a",
    "seq": "ls",
    "duty": 0.80,
    "omega": 1.4,
    "mu_x": 1.80,
    "mu_y": 1.50,
    "d_step": 0.10,
    "d_step_y": 0.12,
    "x_off": -0.030,
    "g_c": 0.048,
    "z_sag": 0.036,
    "kp3": [60.0, 250.0, 250.0],
    "kd3": [2.0, 2.0, 2.0],
    "wheel_kd": 0.5,
    "wheel_mode": "damp",
}


def walk_a_gait() -> dict:
    """`cpg_walk_max.GAITS["walk_a"]` 用的 dict。

    ⚠️ 用它 rollout 時**必須同時給** kp3/kd3/kd_wheel/z_sag（增益不在 GAITS 裡）：
        rollout(gait="walk_a", kp3=BASELINE_A["kp3"], kd3=BASELINE_A["kd3"],
                kd_wheel=BASELINE_A["wheel_kd"], z_sag=BASELINE_A["z_sag"])
    """
    b = BASELINE_A
    return dict(phase=cpg_max.PHASE_WALK_LS, duty=b["duty"], omega=b["omega"],
                mu_x=b["mu_x"], x_off=b["x_off"], d_step=b["d_step"], g_c=b["g_c"])
