#!/usr/bin/env python3
"""CPG 振盪器與足端軌跡 —— **狗上可用的純標準函式庫版本**。

`inference/cpg_max.py` 是同一套數學的 numpy 版，跑在本機（模擬／調參）。
這一份是為了讓 **M9 能在狗上即時算步態**（狗上沒有 numpy）。

⚠️ **兩份實作必須逐點一致**，`tests/test_cpg_port.py` 拿 numpy 版當外部對照
   逐步比對（含相位耦合的長時間累積），容差 1e-12。改了任何一邊都會被抓到。

★ 實測：純 Python 的「CPG step + 4 條腿 IK」是 **0.043 ms / 控制週期**，
  50 Hz 預算 20 ms 的 0.2%。就算 RK3588 慢 4 倍也只有 0.9%。
  （對照：RL policy (256,256,128) 純 Python 要 6 ms —— 那個才需要 numpy。）

★ 為什麼要在狗上跑而不是播放離線檔案：
  文件記載**唯一擋住「能直線走遠」的問題是偏航慢漂 −0.5~−0.8°/s**，
  開迴路沒有航向回授。播放固定檔案永遠解不了；狗上跑 CPG 才有機會
  接 IMU 做航向修正（`/dev/shm/imu_central` 我們已經在讀了）。

★ 腿序：本檔一律用 **SHM 腿名 `(fl, fr, bl, br)`**，不用數字索引。
  `max_model.LEGS` 是 `(FR, FL, RR, RL)` —— 那是本專案反覆出事的地方。

⚠️ 純標準函式庫。
"""
from __future__ import annotations

import math

import coord
import kin

# ---------------------------------------------------------------- 常數
# 與 `inference/max_model.py` 同源；`tests/test_cpg_port.py` 會比對沒有漂移。
MU_MIN, MU_MAX = 1.0, 2.0
A_CONV = 50.0       # 振幅收斂速率
G_P = 0.01          # 站立相的下壓量
W_COUP = 8.0        # 相位耦合強度
N_CPG_SUB = 4       # 每個控制週期的 CPG 次步數
TWO_PI = 2.0 * math.pi

# 四拍側序走：左後 → 左前 → 右後 → 右前。
# ★ `cpg_max.PHASE_WALK` 是照 `max_model.LEGS = (FR, FL, RR, RL)` 排的
#   `[1.5π, 0.5π, π, 0]`，這裡改成用腿名，語意一樣但不會弄錯。
PHASE_WALK = {"fr": 1.5 * math.pi, "fl": 0.5 * math.pi,
              "br": math.pi, "bl": 0.0}
PHASE_TROT = {"fr": 0.0, "fl": math.pi, "br": math.pi, "bl": 0.0}

LEGS = coord.LEGS          # ("fl", "fr", "bl", "br")


def init(phase: dict) -> dict:
    """CPG 初始狀態。四條腿的振幅都從 1.5 起（= MU 的中點）。"""
    return {"rx": {l: 1.5 for l in LEGS}, "rx_d": {l: 0.0 for l in LEGS},
            "ry": {l: 1.5 for l in LEGS}, "ry_d": {l: 0.0 for l in LEGS},
            "theta": {l: float(phase[l]) for l in LEGS}}


def make_step(phase: dict):
    """回傳綁定指定相位關係的 `step`。

    ★ 耦合項 `W_COUP·Σ sin(θj−θi−Φij)` 會把相位拉回 Φ 定義的關係，
      所以**只改初始相位是無效的**，必須連耦合矩陣一起換（task6 的教訓）。
    """
    PHI = {i: {j: phase[j] - phase[i] for j in LEGS} for i in LEGS}

    def step(c: dict, mux: dict, muy: dict, omega: dict, dt: float) -> dict:
        rx = dict(c["rx"])
        rxd = dict(c["rx_d"])
        ry = dict(c["ry"])
        ryd = dict(c["ry_d"])
        th = dict(c["theta"])
        h = dt / N_CPG_SUB
        for _ in range(N_CPG_SUB):
            # ⚠️ 更新順序必須和 numpy 版一致：先 rxd（用舊 rx），再 rx（用新 rxd）。
            for l in LEGS:
                rxd[l] += (A_CONV * (A_CONV / 4.0 * (mux[l] - rx[l]) - rxd[l])) * h
                rx[l] += rxd[l] * h
                ryd[l] += (A_CONV * (A_CONV / 4.0 * (muy[l] - ry[l]) - ryd[l])) * h
                ry[l] += ryd[l] * h
            rbar = {l: 0.5 * (rx[l] + ry[l]) for l in LEGS}
            # ⚠️ θ 要**整批**更新（用舊的 th 算完再一起寫回），不能邊算邊改。
            nt = {}
            for i in LEGS:
                s = 0.0
                for j in LEGS:
                    s += rbar[j] * math.sin(th[j] - th[i] - PHI[i][j])
                nt[i] = th[i] + (TWO_PI * omega[i] + W_COUP * s) * h
            th = nt
        return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd,
                "theta": {l: th[l] % TWO_PI for l in LEGS}}

    return step


def duty_remap(theta: float, duty: float) -> float:
    """相位重映射：擺動相佔一圈的 (1−duty)、站立相佔 duty。

    軌跡公式用 `sin θ > 0` 判擺動，等於佔空比固定 0.5（永遠只有兩腳著地）。
    重映射之後**軌跡形狀不變、只有時間分配變**。
    """
    ph = (theta % TWO_PI) / TWO_PI
    sw = 1.0 - duty
    if ph < sw:
        return math.pi * ph / sw                       # 擺動 → 0~π
    return math.pi + math.pi * (ph - sw) / duty        # 站立 → π~2π


def foot_targets(c: dict, f0: dict, x_off: float, g_c: float, d_step: float,
                 d_step_y: float, duty: float, z_sag: float = 0.0) -> dict:
    """CPG 狀態 → 每條腿的足端目標（輪軸心相對該腿 ABAD 原點）。

    ★ `z_sag` 是位置伺服的撓度補償，**只加在擺動相** ——
      站立相就是靠那個追蹤誤差在出力撐機身。
      不補的後果：`g_c` 小於撓度時**腿根本不離地**，被地面拖著走，
      而超限／飽和／IK 縮限／相位鎖定四個診斷指標**全是乾淨的**。
      2026-08-27 在 MuJoCo 又重現一次：`g_c=0.04` 時實際離地只有 4.5 mm。

    ⚠️ `z_sag` 與 kp 綁定。實機兩點：kp=120 → 72 mm、kp=250 → 36 mm，
      正比於 1/kp。改增益就必須重算。
    """
    out = {}
    for l in LEGS:
        th = duty_remap(c["theta"][l], duty)
        fx = 2.0 * (c["rx"][l] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
        fy = 2.0 * (c["ry"][l] - MU_MIN) / (MU_MAX - MU_MIN) - 1.0
        dx = -d_step * fx * math.cos(th) + x_off
        dy = d_step_y * fy * math.cos(th)
        s = math.sin(th)
        dz = (g_c + z_sag) * s if s > 0 else G_P * s
        x, y, z = f0[l]
        out[l] = (x + dx, y + dy, z + dz)
    return out


def joint_targets(c: dict, f0: dict, knee_sign: dict, x_off: float, g_c: float,
                  d_step: float, d_step_y: float, duty: float,
                  z_sag: float = 0.0) -> tuple[dict, int]:
    """CPG 狀態 → 12 個關節目標角（SHM 關節名 → 角度），外加被縮限的腿數。

    ★ 縮限數要往上報 —— **靜默的 IK 縮限會表現成「步態突然變鈍」而查不出原因。**
    """
    tgt = foot_targets(c, f0, x_off, g_c, d_step, d_step_y, duty, z_sag)
    out, n = {}, 0
    for l in LEGS:
        q, clamped = kin.ik(l, *tgt[l], knee_sign=knee_sign[l])
        n += int(clamped)
        for kd, v in zip(coord.LEG_KINDS, q):
            out[l + kd] = v
    return out, n


def stand_targets(f0: dict, knee_sign: dict, x_off: float = 0.0) -> dict:
    """帶 `x_off` 的靜態站姿關節角（開走前先站穩、以及淡入淡出的端點）。"""
    out = {}
    for l in LEGS:
        x, y, z = f0[l]
        q, _ = kin.ik(l, x + x_off, y, z, knee_sign=knee_sign[l])
        for kd, v in zip(coord.LEG_KINDS, q):
            out[l + kd] = v
    return out


def home_foot(pose: dict) -> dict:
    """從一組姿勢算四條腿的基準足端位置。"""
    return {l: kin.foot_of(l, pose) for l in LEGS}


def knee_signs(pose: dict) -> dict:
    return {l: kin.knee_sign_of(pose, l) for l in LEGS}
