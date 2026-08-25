"""D1 Max 單腿的解析式正／逆運動學。純函式，不碰 MuJoCo，不做 I/O。

**與 task6 的差別：task6 用「home 姿態的數值 Jacobian 求逆」做線性化 IK，這裡改成解析式。**

換掉的理由有三個：

1. task6 那套只在 home 附近準。這台的抬腿指令是 0.10 m、連桿 0.26+0.28 m，
   足端偏移不算小，線性化誤差會直接變成「我以為抬了 100 mm、實際抬了別的數字」——
   而抬腿量正是我們要評估的指標，用一個有偏差的 IK 去量它會自己騙自己。
2. ABAD 行程只有 −0.697~+0.523 rad，而且**左右鏡像**。要逐腿檢查超限，
   就得有準確的角度，不能是線性近似值。
3. 解析式可以直接對 MuJoCo 自己的 FK 做逐點比對（見 test_leg_kin），
   等於把「我有沒有讀錯 MJCF」這件事變成一個會失敗的測試。

座標系：足端位置一律表示為**輪軸心相對該腿 ABAD 關節原點**，機身座標系。
x 前、y 左、z 上。

機構鏈（以 MJCF 的 body pos 為準）：
    ABAD 關節（軸 +x, 角 q1）
      └ +（sx·0.0587, sy·0.045, 0）      sx = 前腿 +1 / 後腿 −1；sy = 左腿 +1 / 右腿 −1
        HIP 關節（軸 +y, 角 q2）
          └ +（0, sy·0.0522, −0.26）
            KNEE 關節（軸 +y, 角 q3）
              └ +（0, sy·0.0088, −0.28） = 輪軸心

因為兩個 y 軸關節不會改變 y 分量，ABAD 之後的 y 偏移可以先加總成一個常數
`ABAD_TO_FOOT_Y = 0.1060`，整條腿在 ABAD 旋轉前是一個平面二連桿問題。

⚠️ 官方 MJCF 在 FBL/RAR/RBL 的 KNEE body 上有一個 2.14e-5 rad 的 quat（約 0.0012°），
   本檔忽略它。實測對輪心位置的影響 < 0.03 mm，見 test_leg_kin 的容差。
"""
import numpy as np

from max_model import (ABAD_TO_FOOT_Y, ABAD_TO_HIP_X, L_SHANK, L_THIGH, SIDE_X,
                       SIDE_Y)


def fk(k: int, q3: np.ndarray) -> np.ndarray:
    """第 k 腿（LEGS 順序）的正向運動學：關節角 (3,) → 輪軸心相對 ABAD 原點 (3,)。"""
    q1, q2, q3_ = float(q3[0]), float(q3[1]), float(q3[2])
    # ABAD 旋轉前的平面二連桿（x-z 平面）
    xp = SIDE_X[k] * ABAD_TO_HIP_X - L_THIGH * np.sin(q2) - L_SHANK * np.sin(q2 + q3_)
    zp = -L_THIGH * np.cos(q2) - L_SHANK * np.cos(q2 + q3_)
    yp = SIDE_Y[k] * ABAD_TO_FOOT_Y
    # 繞 +x 轉 q1
    c, s = np.cos(q1), np.sin(q1)
    return np.array([xp, yp * c - zp * s, yp * s + zp * c])


def ik(k: int, p: np.ndarray, knee_sign: float) -> np.ndarray:
    """第 k 腿的逆向運動學：輪軸心目標 (3,) → 關節角 (3,)。

    `knee_sign` 選膝關節的分支（兩解）：**前腿 −1、後腿 +1**，也就是原廠 X 型站姿
    那一支。用 `max_model.HOME[k, 2]` 的正負號取即可（見 `knee_sign_of`）。

    無解時（目標超出腿能構到的球殼）會把目標**沿徑向縮到可達邊界**再解，
    而不是丟 NaN——步態掃參數時偶爾會掃到邊界外，靜默 NaN 會讓整段模擬變成
    nan 而只發一個 RuntimeWarning。回傳的 `reach_clamped` 旗標讓呼叫端知道發生過。
    """
    q, _ = ik_ex(k, p, knee_sign)
    return q


def ik_ex(k: int, p: np.ndarray, knee_sign: float) -> tuple[np.ndarray, bool]:
    """同 `ik`，但一併回傳「有沒有因為構不到而被縮限」。"""
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    yp = SIDE_Y[k] * ABAD_TO_FOOT_Y

    # --- q1：繞 x 軸。y-z 平面上，(py, pz) 是 (yp, zp) 轉了 q1 的結果，且 zp <= 0。
    #     |(py,pz)|^2 = yp^2 + zp^2 → zp 直接解出來。
    r_yz2 = py * py + pz * pz
    zp2 = r_yz2 - yp * yp
    clamped = False
    if zp2 < 0.0:                      # 目標離髖太近，連 y 偏移都吃不下
        zp2, clamped = 0.0, True
    zp = -np.sqrt(zp2)                 # 腿在下方，取負根
    # 由 (py, pz) = (yp·cos q1 − zp·sin q1, yp·sin q1 + zp·cos q1) 解 q1
    q1 = np.arctan2(yp * pz - zp * py, yp * py + zp * pz)

    # --- q2, q3：ABAD 轉回去之後的平面二連桿
    xp = px - SIDE_X[k] * ABAD_TO_HIP_X
    r2 = xp * xp + zp * zp
    r = np.sqrt(r2)
    reach_lo = abs(L_THIGH - L_SHANK) + 1e-6
    reach_hi = L_THIGH + L_SHANK - 1e-6
    if r > reach_hi or r < reach_lo:   # 沿徑向縮到可達邊界（見 docstring）
        r_new = min(max(r, reach_lo), reach_hi)
        if r > 1e-12:
            xp, zp = xp * r_new / r, zp * r_new / r
        r, r2 = r_new, r_new * r_new
        clamped = True

    # 餘弦定理：q3 是膝的內角補角
    cos_q3 = (r2 - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    q3 = np.sign(knee_sign) * np.arccos(np.clip(cos_q3, -1.0, 1.0))

    # 由 xp = −L2·sin q2 − L3·sin(q2+q3)、zp = −L2·cos q2 − L3·cos(q2+q3) 解 q2。
    # 展開後是 xp = A·sin q2 + B·cos q2 的形式（A、B 由 q3 決定），用 atan2 一次解掉。
    a = -(L_THIGH + L_SHANK * np.cos(q3))
    b = -L_SHANK * np.sin(q3)
    #  xp = a·sin q2 + b·cos q2
    #  zp = a·cos q2 − b·sin q2
    q2 = np.arctan2(a * xp - b * zp, a * zp + b * xp)
    return np.array([q1, q2, q3]), clamped


def knee_sign_of(home: np.ndarray) -> np.ndarray:
    """從 HOME 姿態取每腿的膝分支號（前腿 −1、後腿 +1）。"""
    return np.sign(home[:, 2])


def home_foot(home: np.ndarray) -> np.ndarray:
    """(4, 3)：四腿在 HOME 姿態的輪軸心（相對各自 ABAD 原點）。"""
    return np.array([fk(k, home[k]) for k in range(4)])
