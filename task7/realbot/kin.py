#!/usr/bin/env python3
"""D1 Max 單腿運動學 —— **狗上可用的純標準函式庫版本**。

`inference/leg_kin.py` 是同一套數學的 numpy 版，跑在本機（模擬／分析）。
這一份是為了讓 **M8 能在狗上就地算出「把腳抬高 Δz」要下什麼關節角** ——
狗上沒有 numpy，所以不能直接用那一份。

⚠️ **兩份實作必須逐點一致**，`tests/test_kin.py` 拿 `inference/leg_kin.py`
   當外部對照逐點比對（隨機取樣 + 邊界），容差 1e-9 m。
   改了任何一邊都會被那個測試抓到。

★ **這一份用「腿名」當索引，不用數字 index。**
  `inference/leg_kin.py` 的 `k` 是 `max_model.LEGS = (FR, FL, RR, RL)` 的序，
  而 SHM 是 `(fl, fr, bl, br)` —— 這是本專案反覆出事的腿序陷阱
  （`docs/實機偵察結果_第二趟` 明列「按名稱對應，別按索引」）。
  在狗上唯一拿得到的識別是 SHM 名稱，所以這裡就用它。

座標系（與 `inference/leg_kin.py`、MJCF 相同）：
  原點 = 該腿的 ABAD 關節；x 前、y 左、z 上；回傳的是**輪軸心**位置。
  ⚠️ 不是接地點 —— 接地點在輪軸心下方 `WHEEL_RADIUS`。
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- 幾何常數
# 與 `inference/max_model.py` 同源；`tests/test_kin.py` 會比對兩邊沒有漂移。
ABAD_TO_HIP_X = 0.0587                      # ABAD → HIP，x（後腿用 SIDE_X 變號）
ABAD_TO_FOOT_Y = 0.045 + 0.0522 + 0.0088    # = 0.1060，ABAD 之後的 y 偏移總和
L_THIGH = 0.26                              # HIP → KNEE
L_SHANK = 0.28                              # KNEE → FOOT（輪軸心）
WHEEL_RADIUS = 0.0961

# 腿名 → (x 號, y 號)。前腿 x=+1、左腿 y=+1。
SIDE = {"fl": (+1.0, +1.0), "fr": (+1.0, -1.0),
        "bl": (-1.0, +1.0), "br": (-1.0, -1.0)}


def fk(leg: str, q1: float, q2: float, q3: float) -> tuple[float, float, float]:
    """正向運動學：三個關節角（控制器座標系）→ 輪軸心相對 ABAD 原點。"""
    sx, sy = SIDE[leg]
    xp = sx * ABAD_TO_HIP_X - L_THIGH * math.sin(q2) - L_SHANK * math.sin(q2 + q3)
    zp = -L_THIGH * math.cos(q2) - L_SHANK * math.cos(q2 + q3)
    yp = sy * ABAD_TO_FOOT_Y
    c, s = math.cos(q1), math.sin(q1)
    return (xp, yp * c - zp * s, yp * s + zp * c)


def ik(leg: str, px: float, py: float, pz: float,
       knee_sign: float) -> tuple[tuple[float, float, float], bool]:
    """逆向運動學：輪軸心目標 → (關節角, 有沒有被縮限)。

    `knee_sign` 選膝的分支：**前腿 −1、後腿 +1**（原廠 X 型站姿那一支）。
    構不到時沿徑向縮到可達邊界再解，並把 `clamped` 回報出來 ——
    靜默的縮限會表現成「動作突然變鈍」而查不出原因。
    """
    sx, sy = SIDE[leg]
    yp = sy * ABAD_TO_FOOT_Y
    clamped = False

    # q1：y-z 平面上 (py,pz) 是 (yp,zp) 繞 x 轉 q1 的結果，且 zp <= 0
    zp2 = py * py + pz * pz - yp * yp
    if zp2 < 0.0:
        zp2, clamped = 0.0, True
    zp = -math.sqrt(zp2)
    q1 = math.atan2(yp * pz - zp * py, yp * py + zp * pz)

    # q2, q3：ABAD 轉回去之後的平面二連桿
    xp = px - sx * ABAD_TO_HIP_X
    r = math.sqrt(xp * xp + zp * zp)
    lo = abs(L_THIGH - L_SHANK) + 1e-6
    hi = L_THIGH + L_SHANK - 1e-6
    if r > hi or r < lo:
        r_new = min(max(r, lo), hi)
        if r > 1e-12:
            xp, zp = xp * r_new / r, zp * r_new / r
        r = r_new
        clamped = True

    cos_q3 = (r * r - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    cos_q3 = min(1.0, max(-1.0, cos_q3))
    q3 = math.copysign(math.acos(cos_q3), knee_sign)

    a = -(L_THIGH + L_SHANK * math.cos(q3))
    b = -L_SHANK * math.sin(q3)
    q2 = math.atan2(a * xp - b * zp, a * zp + b * xp)
    return (q1, q2, q3), clamped


def foot_of(leg: str, pose: dict) -> tuple[float, float, float]:
    """從一組姿勢 dict（SHM 關節名 → 角度）取某條腿的輪軸心位置。"""
    return fk(leg, pose[leg + "1_hip_roll"], pose[leg + "2_hip_pitch"],
              pose[leg + "3_knee_pitch"])


def knee_sign_of(pose: dict, leg: str) -> float:
    """從姿勢取膝分支的號。前腿與後腿相反是正常的（X 型站姿）。"""
    return math.copysign(1.0, pose[leg + "3_knee_pitch"])
