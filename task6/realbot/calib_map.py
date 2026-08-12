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

★ 2026-08-12 吊掛實測全面改寫。舊版（2026-08-10）的 12 個項目裡有 11 個是錯的。

  舊版怎麼錯的：
  1. SIGN 假設「三個軸都左右鏡像」——實測 abad 是【前後】分組不是左右
     （前腳 FR/FL 都是 -1、後腳 RR/RL 都是 +1；hip/knee 才是左右分組）
  2. OFFSET 全部由「實機站姿 ≈ MJCF home」反推——這個假設【不成立】。
     實機站姿換算成 MJCF 是 (abad +0.010, hip +0.709, knee -1.413)，
     與 home (0, +1.05, -2.00) 差到 knee 34°、hip 20°。
  3. 舊版自檢「MJCF home → 應還原成 calib_stand.json」永遠會過，因為 offset
     就是那樣定義的——自洽不等於正確。

  現在怎麼來的（`L8_sign_probe.py --stops`，全程唯讀）：
  對 leg0/1/3 的每個關節用手推到【兩端機構限位】、記錄 SHM，對應到 MJCF 的
  joint range。sign、offset、刻度比一次全定，不依賴任何姿態假設。
  刻度比實測 0.91~0.99（接近 1:1，無減速比問題；小於 1 是沒完全推到底）。

★ 信心等級：
  - 腿序重排 LEG_MJCF2SHM：✅ 確定（實測 leg0=FR、leg2=RR）
  - leg0 / leg1 / leg3 的 sign 與 offset：✅ 兩端限位實測
  - leg2(RR)：⚠️ 推論值。該腿整條馬達從 CAN 失聯、編碼器讀 0V，量不到。
    sign 由 abad 前後分組 + hip/knee 左右分組的結構推得（並經站姿/趴姿交叉驗證）；
    offset 由「四條腿站姿應為同一 MJCF 姿態」的共識反推。
    本專案全程不驅動 leg2，所以它只影響離線檢驗的掃描範圍，不影響實機安全。

★ 三項獨立驗證（都不是用來推導校正的資料）：
  1. 站姿在三條實測腿上換算成同一 MJCF 姿態，離散度 ±0.75°~±1.40°
  2. 趴姿（完全未參與推導）四條腿一致，且全部落在機構範圍內：
     abad 貼近 ±0.489 限位（外張到底、左右鏡像）、knee -2.63~-2.67（彎到底附近）
  3. 舊校正把趴姿的足端算成比站姿【還低】157mm（趴著比站著高），物理上不可能；
     新校正的趴姿是足端收到髖部高度、膝彎到底、腿外張到底 —— 那才是趴下的樣子
"""

import numpy as np

# policy 腿序 index → SHM legs[] index。 (FL,FR,RL,RR) → legs[1],legs[0],legs[3],legs[2]
LEG_MJCF2SHM = [1, 0, 3, 2]

# 以下都用【SHM 腿序】索引：0=FR, 1=FL, 2=RR, 3=RL
# 每腿每關節的 (sign, offset)，順序 (abad, hip, knee)
#   sign：+1 表示編碼器與 MJCF 同向，-1 反向
#   offset：MJCF 角度為 0 時，編碼器讀到的值（rad）
#
# ★ 結構（2026-08-12 兩端限位實測）：
#     abad —— 【前後】分組：前腳 FR/FL = -1、後腳 RR/RL = +1
#     hip  —— 【左右】分組：右腿 FR/RR = +1、左腿 FL/RL = -1
#     knee —— 【左右】分組：同 hip
#   abad 不隨左右鏡像，是因為四顆 abad 馬達同軸裝在機身上、轉向一致；
#   hip/knee 裝在左右互為鏡像的腿組件上，所以鏡像。舊版誤以為三軸都左右鏡像。
CALIB = {
    # SHM leg 0 = FR（右前）—— 兩端限位實測
    0: {"abad": (-1, +0.5153), "hip": (+1, -2.9558), "knee": (+1, +2.6744)},
    # SHM leg 1 = FL（左前）—— 兩端限位實測
    1: {"abad": (-1, -0.4808), "hip": (-1, +2.9036), "knee": (-1, -2.6793)},
    # SHM leg 2 = RR（右後）—— ⚠️ 推論值，該腿失聯無法實測，見檔頭信心等級
    2: {"abad": (+1, -0.5141), "hip": (+1, -2.8586), "knee": (+1, +2.6323)},
    # SHM leg 3 = RL（左後）—— 兩端限位實測
    3: {"abad": (+1, +0.5218), "hip": (-1, +2.8443), "knee": (-1, -2.6800)},
}

# 實機站姿換算成 MJCF 的共識值（三條實測腿平均，離散度 ±0.75°~±1.40°）。
# ⚠️ 這【不等於】MJCF home (0, +1.05, -2.00) —— 差 knee 34°、hip 20°。
#    舊版把兩者當成同一件事，那正是 offset 全錯的根因。
#    步態是繞著 MJCF home 生成的，所以部署時機器會蹲得比原廠站姿低一截。
STAND_MJCF = (0.0103, 0.7093, -1.4129)

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
    # ⚠️ 舊版的自我檢查是「MJCF home → 應還原成擷取的站姿」。那個檢查【永遠會通過】，
    #    因為 offset 就是那樣定義的——它驗的是自洽，不是正確，所以在 offset 全錯了
    #    34° 的情況下照樣印「通過」。已刪除。
    #
    #    現在改成驗【獨立於推導過程】的性質：機構範圍與姿態一致性。
    from math import isclose
    RANGE = {"abad": (-0.4887, 0.4887), "hip": (-1.1520, 2.9670),
             "knee": (-2.7230, -0.6020)}
    STAND_SHM = {0: (+0.5061, -2.2156, +1.2557), 1: (-0.5110, +2.2073, -1.2825),
                 2: (-0.5039, -2.1493, +1.2194), 3: (+0.5132, +2.1528, -1.2570)}
    LIE_SHM = {0: (+0.9836, -1.9568, +0.0027), 1: (-0.9756, +1.9507, -0.0044),
               2: (-0.9865, -1.9396, +0.0067), 3: (+0.9783, +1.9251, -0.0068)}
    NAME = {0: "FR", 1: "FL", 2: "RR", 3: "RL"}

    def to_mjcf(shm_leg, vals):
        return [(v - CALIB[shm_leg][jn][1]) / CALIB[shm_leg][jn][0]
                for jn, v in zip(JN, vals)]

    ok = True
    for label, table in (("站姿", STAND_SHM), ("趴姿", LIE_SHM)):
        print(f"{label} → MJCF（四條腿應一致，且都在機構範圍內）：")
        cols = {jn: [] for jn in JN}
        for shm_leg in range(4):
            q = to_mjcf(shm_leg, table[shm_leg])
            bad = [jn for jn, v in zip(JN, q)
                   if not (RANGE[jn][0] - 0.03 <= v <= RANGE[jn][1] + 0.03)]
            ok &= not bad
            for jn, v in zip(JN, q):
                cols[jn].append(v)
            print(f"  leg{shm_leg}({NAME[shm_leg]}) abad={q[0]:+.4f} hip={q[1]:+.4f} "
                  f"knee={q[2]:+.4f}  {'✗ 超出機構 ' + str(bad) if bad else '✓'}")
        # hip/knee 在對稱姿態下四腿應【相等】；abad 應【左右鏡像】——趴下時腿往外張，
        # 右腿負、左腿正，直接比大小會看到 55° 的假離散度。
        spread = {jn: (max(cols[jn]) - min(cols[jn])) for jn in ("hip", "knee")}
        ab = cols["abad"]
        mirrored = [ab[0], -ab[1], ab[2], -ab[3]]      # 右腿原樣、左腿取負
        spread["abad"] = max(mirrored) - min(mirrored)
        worst = max(spread.values())
        ok &= worst < 0.10
        print(f"  四腿離散度  " + "  ".join(f"{jn}={spread[jn] * 57.3:.2f}°" for jn in JN)
              + "   （abad 取左右鏡像後比較）"
              + f"   {'✓' if worst < 0.10 else '✗ 太大，校正不一致'}\n")

    # 結構檢查：abad 前後分組、hip/knee 左右分組（2026-08-12 實測結論）
    FRONT, REAR = (0, 1), (2, 3)
    RIGHT, LEFT = (0, 2), (1, 3)
    structure = [
        ("abad 前腳同號", {CALIB[i]["abad"][0] for i in FRONT} == {-1}),
        ("abad 後腳同號", {CALIB[i]["abad"][0] for i in REAR} == {+1}),
        ("hip 右腿 +1", {CALIB[i]["hip"][0] for i in RIGHT} == {+1}),
        ("hip 左腿 -1", {CALIB[i]["hip"][0] for i in LEFT} == {-1}),
        ("knee 右腿 +1", {CALIB[i]["knee"][0] for i in RIGHT} == {+1}),
        ("knee 左腿 -1", {CALIB[i]["knee"][0] for i in LEFT} == {-1}),
    ]
    print("結構檢查（abad 前後分組、hip/knee 左右分組）：")
    for name, good in structure:
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}")

    print("\n自我檢查", "通過" if ok else "失敗")
    print("⚠️ leg2(RR) 是推論值（該腿失聯無法實測）。修好之後要用 "
          "L8_sign_probe.py --leg 2 --stops 實測補上。")
