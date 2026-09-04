"""D1 Max 的 CPG 振盪器與足端軌跡。純函式，不碰 MuJoCo，不做 I/O。

振盪器數學與 task6 `cpg_d1.py` 逐行相同（再往上是 task4 論文標準版）。
**改掉的只有兩件事**，都是因為 D1 Max 的結構不同：

1. `joint_targets` 改用 `leg_kin` 的解析式 IK，不是 home 附近的線性化 Jacobian。
2. 足端基準是**每腿各自的** `home_foot[k]`，因為站姿是前後鏡像的 X 型。
   task6 那種「四腿共用一組 HOME3 + 共用一個 jinv 形狀」的寫法在這台會做出
   前後腿方向相反的怪東西。

相位／佔空比的處理沿用 task6：`duty_remap` 把相位重映射成指定佔空比，
duty=0.5 時恆等於原樣（見該函式的證明）。
"""
import numpy as np

import leg_kin
from max_model import (A_CONV, G_P, HOME, MU_MAX, MU_MIN, N_CPG_SUB, W_COUP)

# 相位偏移，腿序 FR, FL, RR, RL。
#
# ⚠️⚠️ **相位值小的腿不是先擺動的那一隻，是相反。** 2026-09-03 才發現。
#   腿 k 的相位是 `φ_k = phase_k/2π + ωt`，擺動發生在 `φ_k mod 1 ∈ [0, 1−duty)`，
#   所以**相位值大的腿先繞回 0、先擺動**：
#
#       開始擺動的時刻  τ_k = (1 − phase_k/2π) mod 1
#
#   照這個算，`PHASE_WALK` 的順序是 **RL → FR → RR → FL（左後→右前→右後→左前）
#   ＝ diagonal sequence walk**，而不是註解原本寫的 lateral sequence。
#   `test_walk_phase_is_lateral_sequence` 用 `argsort(相位值)` 判順序，
#   剛好把順序判反了，於是這個錯誤被一個「通過的測試」保護了兩週。
#   ★ 教訓：**測試斷言的是「我以為的語意」時，它保護的是那個誤解，不是程式。**
PHASE_TROT = np.array([0.0, np.pi, np.pi, 0.0])       # 兩拍：對角腿 FR+RL / FL+RR 同相

# 四拍 **diagonal sequence**（左後→右前→右後→左前）。這是 2026-09-03 之前
# 一直在用的那組，所有既有結果都是它跑出來的，所以保留原名不動。
PHASE_WALK = np.array([1.5 * np.pi, 0.5 * np.pi, np.pi, 0.0])

# 四拍 **lateral sequence**（左後→左前→右後→右前）——文獻上靜態穩定裕度最好的
# crawl 序列（McGhee & Frank 的經典結果：六種 creeping gait 只有三種能全程靜態穩定）。
# 與 `PHASE_WALK` 只差 FR / FL 對調。實測（同參數、唯一變因是序列，60 s）：
#     前腳執行率 0.76 → **1.15**、行進速度 0.257 → **0.425**、後膝 48.5 → 48.3（不變）
PHASE_WALK_LS = np.array([0.5 * np.pi, 1.5 * np.pi, np.pi, 0.0])

for _c in (PHASE_TROT, PHASE_WALK, PHASE_WALK_LS):
    _c.flags.writeable = False
del _c


def swing_order(phase) -> list[int]:
    """回傳四腿**實際開始擺動**的先後（索引，腿序 FR,FL,RR,RL）。

    ★ 不要用 `argsort(phase)` —— 那會得到相反的順序，見上面的警語。
    """
    tau = (1.0 - np.asarray(phase, dtype=float) / (2 * np.pi)) % 1.0
    return [int(i) for i in np.argsort(tau)]


def cpg_init(phase: np.ndarray) -> dict:
    return {"rx": np.full(4, 1.5), "rx_d": np.zeros(4),
            "ry": np.full(4, 1.5), "ry_d": np.zeros(4),
            "theta": np.array(phase, dtype=float)}


def make_cpg_step(phase: np.ndarray):
    """回傳一個綁定指定相位偏移的 cpg_step。

    耦合項 `W_COUP·Σ sin(θj−θi−Φ)` 會把相位拉回 Φ 定義的關係，
    所以**只改初始相位是無效的**，必須連耦合矩陣一起換（task6 的教訓）。
    """
    PHI = np.asarray(phase)[None, :] - np.asarray(phase)[:, None]

    def step(c: dict, mux, muy, omega, dt: float) -> dict:
        rx, rxd, ry, ryd, th = (c["rx"].copy(), c["rx_d"].copy(),
                                c["ry"].copy(), c["ry_d"].copy(), c["theta"].copy())
        h = dt / N_CPG_SUB
        for _ in range(N_CPG_SUB):
            rxd += (A_CONV * (A_CONV / 4 * (mux - rx) - rxd)) * h
            rx += rxd * h
            ryd += (A_CONV * (A_CONV / 4 * (muy - ry) - ryd)) * h
            ry += ryd * h
            rbar = 0.5 * (rx + ry)
            diff = th[None, :] - th[:, None] - PHI
            th = th + (2 * np.pi * omega + W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
        return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th % (2 * np.pi)}

    return step


def duty_remap(th: np.ndarray, duty: float) -> np.ndarray:
    """把相位重映射成「擺動相佔一圈的 (1−duty)、站立相佔 duty」。

    軌跡公式用 `sin θ > 0` 判擺動，等於佔空比固定 0.5（永遠只有兩腳著地）。
    重映射之後軌跡形狀不變、只有時間分配變。
    duty=0.5 時本函式恆等於原樣：ph<0.5 → π·ph/0.5 = 2π·ph = θ；
    ph≥0.5 → π + π(ph−0.5)/0.5 = 2π·ph = θ。
    """
    ph = (np.asarray(th) % (2 * np.pi)) / (2 * np.pi)
    sw = 1.0 - duty
    return np.where(ph < sw,
                    np.pi * ph / sw,                     # 擺動 → 0~π
                    np.pi + np.pi * (ph - sw) / duty)    # 站立 → π~2π


def x_off_split(x_c: float, x_d: float) -> np.ndarray:
    """把配平量 `x_c` 與軸距量 `x_d` 組成逐腿的 `x_off`（腿序 FR, FL, RR, RL）。

    前兩腿 `x_c + x_d`、後兩腿 `x_c − x_d`。兩個軸是正交的，各自有物理意義
    （`test_x_c_is_support_center_shift_and_x_d_is_wheelbase` 釘住）：

    - **`x_c` = 支撐多邊形中心相對機身的位移**，也就是配平點。
    - **`x_d` = 半軸距增量**，足端 wheelbase 變化 `2·x_d`，不動配平。

    ★ 由此得到一條硬約束：`f0` 本身已是前後鏡像，所以
    **前後姿態對稱 ⟺ `x_c = 0`**，與 `x_d` 無關。
    「保留配平又要前後對稱」在這個構型下不存在 —— 它們是同一個自由度。
    """
    return np.array([x_c + x_d, x_c + x_d, x_c - x_d, x_c - x_d], dtype=float)


def gait_phase(theta, phase) -> float:
    """全域步態相位 τ ∈ [0,1)。τ=0 是 `phase=0` 那條腿開始擺動的時刻。

    `τ = (θ_k − phase_k)/2π`，四腿相位鎖定後對每個 k 都相同。
    ⚠️ 用**圓平均**而不是隨便取一腿 —— 起步或擾動時相位還沒鎖定，
    取單腿會讓 sway 的相位跳動，而那正是 sway 最需要準的時候。
    """
    d = (np.asarray(theta, float) - np.asarray(phase, float)) / (2 * np.pi)
    ang = 2 * np.pi * d
    return float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                  / (2 * np.pi)) % 1.0)


def body_sway(tau: float, sway_x: float, sway_y: float,
              lead_x: float = 0.0, lead_y: float = 0.0) -> np.ndarray:
    """四腿共同的足端偏移 (dx, dy)，用來把質心送進當下的支撐三角形。

    這是文獻上 crawl gait 的另一半（COG adjustment / body sway）。
    我們的 CPG 只規劃足端相對機身的位置，所以「機身往左」＝「四腿足端往右」——
    因此這裡回傳的是**足端**偏移，與質心偏移反向。

    相位對齊（以 LS 序列 `PHASE_WALK_LS` 為準，τ 的定義見 `gait_phase`）：

        τ∈[0,   0.2)  RL 擺動（左後）→ 質心要往**右前**
        τ∈[0.25,0.45) FL 擺動（左前）→ 質心要往**右後**
        τ∈[0.5, 0.7)  RR 擺動（右後）→ 質心要往**左前**
        τ∈[0.75,0.95) FR 擺動（右前）→ 質心要往**左後**

    → 橫向一圈一次（左腿擺動時往右）、**縱向一圈兩次**（後腿擺動時往前）：

        質心   x = +sway_x·cos(4πτ)      y = −sway_y·sin(2πτ)
        足端   dx = −sway_x·cos(4πτ)     dy = +sway_y·sin(2πτ)

    `lead_x` / `lead_y` 是相位提前量（步態週期的比例）：機身要在腿抬起**之前**
    就移好，否則腿已經離地了質心才開始移過去。

    ⚠️ **兩個方向必須分開給。** 縱向是二倍頻，同一個 lead 值對縱向相當於
    橫向的兩倍相位量。實測（LS、`x_c=−20`）：橫向的最佳 lead ≈ 0.20，
    而縱向在 lead=0 時前腳執行率 0.56 → **1.20**、lead=0.20 時掉到 0.09。
    先前把兩者綁成同一個參數時，掃出來的結論是「sway_x 有害」——**那是綁在一起的假象**。
    """
    tx = (tau + lead_x) % 1.0
    ty = (tau + lead_y) % 1.0
    return np.array([-sway_x * np.cos(4 * np.pi * tx),
                     sway_y * np.sin(2 * np.pi * ty)])


def foot_targets(c: dict, f0: np.ndarray, x_off, g_c: float,
                 d_step: float, d_step_y: float, duty: float,
                 z_sag: float = 0.0, sway=None) -> np.ndarray:
    """CPG 狀態 → (4, 3) 足端目標（輪軸心相對各腿 ABAD 原點）。

    `f0` 是四腿的基準足端位置，通常取 `leg_kin.home_foot(HOME)`。

    `x_off` 可以是純量（四腿共用，原本的用法）或 (4,) 的逐腿值
    （用 `x_off_split` 產生）。純量走的是同一條算式，結果逐位元相同。

    ★ `z_sag`：位置伺服的靜態撓度補償，**這台一定要給**。

    站立時 kp 是有限的，要撐住機身就必須留追蹤誤差，所以**指令的足端基準會比實際
    站立的足端低 `z_sag`**（D1 Max 實測 32.5 mm，見 `max_model.STATIC_SAG`）。
    擺動相的腿是空載的、會確實走到指令位置，於是：

        實際離地 = g_c − z_sag

    不補的後果是「命令抬 100 mm、實際只抬 67 mm」，而 g_c 小於 z_sag 時
    **腿根本不會離地**（實測 g_c=0.02 時四腳全程貼地）。抬不夠的腿在擺動相被
    地面往前拖，反而把機身往後推 —— 症狀是「明明在走路卻倒退」，
    而且四個診斷指標（超限／飽和／IK 縮限／相位鎖定）**全都是乾淨的**，
    完全看不出問題在哪。task6 在 D1 EDU 上踩過同一個坑。

    補上之後 `g_c` 的語意才是「實際離地高度」，也才對得上原廠 `leg_height`。
    """
    th = duty_remap(c["theta"], duty)
    fx = 2 * (c["rx"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    fy = 2 * (c["ry"] - MU_MIN) / (MU_MAX - MU_MIN) - 1
    sx, sy = (0.0, 0.0) if sway is None else (float(sway[0]), float(sway[1]))
    dx = -d_step * fx * np.cos(th) + np.asarray(x_off, dtype=float) + sx
    dy = d_step_y * fy * np.cos(th) + sy
    # 擺動相加上 z_sag；站立相不加 —— 站立相就是靠那個追蹤誤差在出力撐機身的。
    dz = np.where(np.sin(th) > 0, (g_c + z_sag) * np.sin(th), G_P * np.sin(th))
    return np.asarray(f0) + np.stack([dx, dy, dz], -1)


def joint_targets(c: dict, f0: np.ndarray, x_off, g_c: float,
                  d_step: float, d_step_y: float, duty: float,
                  knee_sign: np.ndarray, z_sag: float = 0.0,
                  sway=None) -> tuple[np.ndarray, int]:
    """CPG 狀態 → 12 個關節目標角，外加「因構不到而被縮限的腿數」。

    回傳 `(q12, n_reach_clamped)`。縮限計數要往上報 —— 靜默的 IK 縮限會表現成
    「步態突然變鈍」而找不到原因。
    """
    tgt = foot_targets(c, f0, x_off, g_c, d_step, d_step_y, duty, z_sag, sway)
    q = np.zeros((4, 3))
    n_clamped = 0
    for k in range(4):
        q[k], clamped = leg_kin.ik_ex(k, tgt[k], knee_sign[k])
        n_clamped += int(clamped)
    return q.reshape(12), n_clamped


def stand_targets(knee_sign: np.ndarray, f0: np.ndarray, x_off=0.0) -> np.ndarray:
    """帶 x_off 的靜態站姿關節角（開走前先站穩用）。

    `x_off` 純量或 (4,)，語意與 `foot_targets` 相同 —— 兩邊必須一致，
    否則站姿與步態的基準會差一個偏移，第一步會從偏掉的地方跳過來。
    """
    xo = np.broadcast_to(np.asarray(x_off, dtype=float), (4,))
    q = np.zeros((4, 3))
    for k in range(4):
        q[k] = leg_kin.ik(k, f0[k] + np.array([xo[k], 0.0, 0.0]), knee_sign[k])
    return q.reshape(12)


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    u = q[1:4]
    t = 2 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def w2b(q, v):
    """世界向量轉到機身座標系。"""
    return qrot(qinv(q), v)


def yaw_deg(q) -> float:
    """四元數 → 偏航角（度）。

    ⚠️ 一定要量偏航。開迴路步態最大的失效模式是**走弧線**，
    而只看「側偏」看不出來：走弧線與平移側滑在 y 位移上長得一模一樣，
    但成因與解法完全不同（前者是偏航力矩，後者是橫向滑動）。
    """
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def circ_std(angles: np.ndarray) -> float:
    """圓形統計的標準差（rad）。

    相位差要用圓形統計：±180° 的包裹會讓一般標準差虛胖，
    看起來像相位沒鎖定，其實鎖得很好（task6 §7-2 的坑）。
    """
    a = np.asarray(angles, dtype=float)
    r = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    r = min(max(r, 1e-12), 1.0)
    # r 完全等於 1 時 log 給 −0.0，sqrt(−0.0) = −0.0，會印成 "-0.0"。夾成 0。
    return float(np.sqrt(max(0.0, -2.0 * np.log(r))))
