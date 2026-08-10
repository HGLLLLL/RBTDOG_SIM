#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib_map.py —— MJCF/policy 角度 → 實機 SHM 指令角度 的映射（校正層第二層）。

由來（2026-08-10）：
  - MJCF 慣例：inference/d1_model.py 的 HOME3=[0, 1.05, -2.00]，LEGS=(FL,FR,RL,RR)，四腿同號。
  - 實機慣例：calib_capture.py 擷取的站姿(calib_stand.json) + 趴下(calib_lie.json)，每腿鏡像。
  - MJCF 幾何方向：MuJoCo forward kinematics 探測（+knee→伸直、+abad→外張、+hip→後擺）。

映射公式（逐關節）：  shm_cmd = SIGN * mjcf_angle + OFFSET
再加腿序重排：policy 腿序 (FL,FR,RL,RR) → SHM 腿序 (FR,FL,RR,RL)。

★ 信心等級（務必知道）：
  - 腿序重排 LEG_MJCF2SHM：✅ 確定（實測 leg0=FR）
  - abad / knee 的 SIGN：✅ 高（兩姿勢方向 + 幾何交叉驗證）
  - hip 的 SIGN：⚠️ 暫定（只知左右鏡像，絕對號待吊掛時單關節微動確認）
  - 所有 OFFSET：⚠️ 建立在「實機站姿 ≈ MJCF home」假設上；吊掛掃描時要複核
  → 未經吊掛驗證前，不可用於閉環部署。先只拿來做「單關節、微幅、可中止」的核對。
"""

import numpy as np

# policy 腿序 index → SHM legs[] index。 (FL,FR,RL,RR) → legs[1],legs[0],legs[3],legs[2]
LEG_MJCF2SHM = [1, 0, 3, 2]

# 以下都用【SHM 腿序】索引：0=FR, 1=FL, 2=RR, 3=RL
# 每腿每關節的 (sign, offset)，順序 (abad, hip, knee)
#   sign：+1 表示編碼器與 MJCF 同向，-1 反向
#   offset：MJCF 角度為該關節 home 值時，編碼器讀到的值（rad）
CALIB = {
    # SHM leg 0 = FR（右前）
    0: {"abad": (+1, +0.506), "hip": (-1, -1.166), "knee": (-1, -0.744)},
    # SHM leg 1 = FL（左前）
    1: {"abad": (-1, -0.511), "hip": (+1, +1.157), "knee": (+1, +0.717)},
    # SHM leg 2 = RR（右後）
    2: {"abad": (-1, -0.504), "hip": (-1, -1.099), "knee": (-1, -0.781)},
    # SHM leg 3 = RL（左後）
    3: {"abad": (+1, +0.513), "hip": (+1, +1.103), "knee": (+1, +0.743)},
}
# ⚠️ hip 的 sign 為暫定（右腿 -1 / 左腿 +1，僅左右鏡像已確定，絕對號待驗）。

JN = ("abad", "hip", "knee")


def mjcf_leg_to_shm(shm_leg, abad, hip, knee):
    """把某條腿的 MJCF 角度 (abad,hip,knee) 轉成該 SHM 腿的指令角度。"""
    c = CALIB[shm_leg]
    out = {}
    for jn, val in (("abad", abad), ("hip", hip), ("knee", knee)):
        s, o = c[jn]
        out[jn] = s * val + o
    return out


def mjcf12_to_shm(q12):
    """
    q12: policy 順序(FL,FR,RL,RR)×(abad,hip,knee) 的 12 維 MJCF 角度。
    回傳 dict：shm_leg -> {abad,hip,knee}（實機指令角度）。
    """
    q12 = np.asarray(q12, dtype=float).reshape(4, 3)
    result = {}
    for mjcf_leg in range(4):
        shm_leg = LEG_MJCF2SHM[mjcf_leg]
        a, h, k = q12[mjcf_leg]
        result[shm_leg] = mjcf_leg_to_shm(shm_leg, a, h, k)
    return result


if __name__ == "__main__":
    # 自我檢查：把 MJCF home 丟進去，應該還原成擷取的站姿（因 offset 就是這樣定的）
    from math import isclose
    HOME3 = (0.0, 1.05, -2.00)
    q12 = np.tile(HOME3, 4)   # FL,FR,RL,RR 都用 home
    res = mjcf12_to_shm(q12)
    print("MJCF home → 實機指令（應 ≈ calib_stand.json）：")
    STAND = {0: (+0.506, -2.216, +1.256), 1: (-0.511, +2.207, -1.283),
             2: (-0.504, -2.149, +1.219), 3: (+0.513, +2.153, -1.257)}
    ok = True
    for shm_leg in range(4):
        r = res[shm_leg]
        a, h, k = r["abad"], r["hip"], r["knee"]
        ra, rh, rk = STAND[shm_leg]
        good = all(isclose(x, y, abs_tol=0.02) for x, y in ((a, ra), (h, rh), (k, rk)))
        ok &= good
        print(f"  SHM leg{shm_leg}: abad={a:+.3f} hip={h:+.3f} knee={k:+.3f}  "
              f"{'✓' if good else '✗ 與站姿不符'}")
    print("\n自我檢查", "通過（映射與擷取站姿自洽）" if ok else "失敗")
    print("⚠️ 提醒：hip 絕對號與所有 offset 仍需吊掛時單關節微動最終確認。")
