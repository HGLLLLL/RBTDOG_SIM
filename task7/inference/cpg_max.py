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

# 相位偏移，腿序 FR, FL, RR, RL。相位小的先擺動。
PHASE_TROT = np.array([0.0, np.pi, np.pi, 0.0])       # 兩拍：對角腿 FR+RL / FL+RR 同相
# 四拍側序走（lateral sequence walk）：左後 → 左前 → 右後 → 右前，四腿均分一圈。
PHASE_WALK = np.array([1.5 * np.pi, 0.5 * np.pi, np.pi, 0.0])
for _c in (PHASE_TROT, PHASE_WALK):
    _c.flags.writeable = False
del _c


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


def foot_targets(c: dict, f0: np.ndarray, x_off: float, g_c: float,
                 d_step: float, d_step_y: float, duty: float,
                 z_sag: float = 0.0) -> np.ndarray:
    """CPG 狀態 → (4, 3) 足端目標（輪軸心相對各腿 ABAD 原點）。

    `f0` 是四腿的基準足端位置，通常取 `leg_kin.home_foot(HOME)`。

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
    dx = -d_step * fx * np.cos(th) + x_off
    dy = d_step_y * fy * np.cos(th)
    # 擺動相加上 z_sag；站立相不加 —— 站立相就是靠那個追蹤誤差在出力撐機身的。
    dz = np.where(np.sin(th) > 0, (g_c + z_sag) * np.sin(th), G_P * np.sin(th))
    return np.asarray(f0) + np.stack([dx, dy, dz], -1)


def joint_targets(c: dict, f0: np.ndarray, x_off: float, g_c: float,
                  d_step: float, d_step_y: float, duty: float,
                  knee_sign: np.ndarray, z_sag: float = 0.0) -> tuple[np.ndarray, int]:
    """CPG 狀態 → 12 個關節目標角，外加「因構不到而被縮限的腿數」。

    回傳 `(q12, n_reach_clamped)`。縮限計數要往上報 —— 靜默的 IK 縮限會表現成
    「步態突然變鈍」而找不到原因。
    """
    tgt = foot_targets(c, f0, x_off, g_c, d_step, d_step_y, duty, z_sag)
    q = np.zeros((4, 3))
    n_clamped = 0
    for k in range(4):
        q[k], clamped = leg_kin.ik_ex(k, tgt[k], knee_sign[k])
        n_clamped += int(clamped)
    return q.reshape(12), n_clamped


def stand_targets(knee_sign: np.ndarray, f0: np.ndarray, x_off: float = 0.0) -> np.ndarray:
    """帶 x_off 的靜態站姿關節角（開走前先站穩用）。"""
    q = np.zeros((4, 3))
    for k in range(4):
        q[k] = leg_kin.ik(k, f0[k] + np.array([x_off, 0.0, 0.0]), knee_sign[k])
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
    return float(np.sqrt(-2.0 * np.log(r)))
