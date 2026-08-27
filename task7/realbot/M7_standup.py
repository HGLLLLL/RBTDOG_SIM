#!/usr/bin/env python3
"""M7 —— 站起來（★★ 承重，這是目前為止風險最高的一步）。

★★★ 這支跟 M5 的安全語意**相反**，不要混用
     M5 是吊掛、腿不承重，中止＝切成純阻尼（放手）。
     **M7 是承重，中止時放手＝狗塌下去。** 所以 M7 的中止是「凍結目標角、
     維持增益、原地撐住」，等人工決定要繼續還是坐回去。

════════════════════════════════════════════════════════════════════
前置條件（少一項就不要跑）
════════════════════════════════════════════════════════════════════
  - **狗趴在地上**（不是吊掛），16 顆洩力
  - **吊帶掛著但鬆弛** —— 不承重，純粹當接住用。
    ⚠️ 這是唯一的物理安全網：程式若整個死掉，controller 500 ms 後清零 →
       狗會從站姿塌下來。**軟體沒有辦法保證這件事不發生。**
  - 地面淨空、四周無人、第二個終端機備著 `sudo ~/estop_max.sh`
  - M5 的 S0–S6 全部通過（換算式、腿序、增益符號都已驗證）

════════════════════════════════════════════════════════════════════
軌跡：照抄原廠（2026-08-26 15:47 全程錄製）
════════════════════════════════════════════════════════════════════

原廠不是一步站起來，是**經過中間姿勢的分段線性斜坡**：

    趴平（實測角） --1.2s--> crouch --停 1.8s--> stand
                   0.33rad/s          1.05rad/s

中途那個路徑點**正是文件裡的 `crouch`**（12 個關節 RMS 0.0000，完全吻合）。
起點 `des` 就是當下的實測角 → **無突跳接管**，原廠也是這樣做。

增益也照抄：**kp=250 / kd=5.0，純 PD 無前饋**（實測 12 個腿關節的 effort 全程 0）。
⚠️ 這跟設定檔的 ABAD 60 / HIP 120 / KNEE 120 不同 —— 那組是 RL 用的。

★ 但**增益的上法不照抄**：原廠是一步從 0 跳到 250，代價是 ABAD/HIP 在那一瞬間
  衝到 20–28 N·m。我們用斜坡爬升，避開這個開啟衝擊。

════════════════════════════════════════════════════════════════════
力矩門檻（實機分段峰值 × 1.5）
════════════════════════════════════════════════════════════════════

    階段              ABAD    HIP   KNEE
    趴→crouch        28.15  22.70  35.41
    crouch 停住      22.23  21.34  33.62   ← 光是停著就要 33.6
    crouch→stand     18.26  18.99  42.45   ← 全程峰值
    站穩              1.25   3.71  11.38

⚠️ **「只做到 crouch」在力矩上幾乎沒有比較安全**（35.4 vs 42.45）。
   它降低的是**塌下來的高度**（0.29 m vs 0.51 m），不是力矩。

════════════════════════════════════════════════════════════════════
兩道乾跑就會擋下來的保護（2026-08-27 加）
════════════════════════════════════════════════════════════════════

1. **「走多遠」檢查（`--vcmd-max`，預設 2.0 rad/s）**
   M7 只管「幾秒到」，不管「離多遠」—— `t1`/`t2` 是固定秒數，起點卻是狗當下
   的姿勢。正常趴姿最遠的關節只要轉 0.4 rad；**若狗是 knee_back（後膝往反
   方向彎），後膝要掃 5.2 rad ＝ 300°**，同樣秒數 → 命令速度 13 倍，中途整條
   腿會打直掃過地面，而且一定撞到 `--vmax`。承重中的中止是「凍結原地」，
   會卡在腿甩到一半的位置。→ 乾跑就列表擋下，並告訴你要喬姿勢還是放長 t1。

2. **輪子到位後鎖定（`--no-wheel-lock` 可關）**
   照原廠：移動中純阻尼讓它滾、到 HOLD 區段就 latch 鎖住。見下面 WHEEL_* 的註解。

用法：
    # 乾跑（一定先做）—— 看「趴平(實測)」欄的 bl3/br3 是否與 fl3/fr3 反號
    python3 M7_standup.py --to crouch

    # T1：趴 → crouch → 趴（塌下來的高度較低，先做這個）
    sudo python3 M7_standup.py --to crouch --confirm

    # T2：趴 → crouch → stand → crouch → 趴
    sudo python3 M7_standup.py --to stand --confirm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

import coord
import shm_io
from M5_leg_pose import Keepalive, mc_ctrl_pid, proc_state, smoothstep

# 實機讀到的原廠站立增益（2026-08-26 15:47）
KP_DEF, KD_DEF = 250.0, 5.0
# 輪子：**移動中純阻尼、到位後鎖定** —— 照原廠的兩段式（2026-08-27 改）。
#
# 移動中純阻尼（kp=0）：實測起身過程後兩輪各滾約 100 mm，
#   **輪子必須能滾，鎖死的話腿伸不開。**
#   （實測 kd 0.1~1.0 對腿峰值力矩幾乎無影響：膝 23.1 vs 23.2 N·m。）
#
# ★ 到位後鎖定（kp=20）：原廠在 12.99 s 站穩後就是這樣做的 —— des 鎖在
#   當下的實測角（latch，之後是常數），鎖完 9 秒內四輪只再動 0.3~2.1 mm。
#   不鎖的代價（MuJoCo，站穩後 15 秒）：
#       平地        kd=0.1 飄 0.8 mm ／ 鎖定 1.2 mm   → 無差別
#       2° 斜坡     kd=0.1 飄 2170 mm、kd=0.5 飄 411 mm ／ **鎖定 21 mm**
#   斜地不鎖＝狗會像溜冰一樣慢慢滑走，而且 M7 的中止是「凍結腿」，擋不住這個。
#
# ⚠️ 曾經擔心的 ±π 繞回問題 —— **已驗證是 driver 自己解纏，我們不必處理**：
#   `fl4` @ t=6.37s，des=+3.0543 q=−3.1401，原始誤差 +6.1944 rad
#     用原始誤差預測 τ = +123.73；用解纏誤差預測 τ = −1.93；**實測 −1.93**。
#   kp>0 的 1328 筆逐筆比對：解纏 RMS 殘差 0.96，原始 65.6（362 筆有繞回）。
#   → 直接寫「當下實測角」當 des 是安全的，driver 會處理繞回。
WHEEL_KD_DEF = 0.5          # 移動中
WHEEL_KP_HOLD = 20.0        # 到位後鎖定（原廠值）
WHEEL_KD_HOLD = 0.1         # 到位後鎖定（原廠值）

# 力矩門檻 = 實機分段峰值 × 1.5
TMAX = {"1_hip_roll": 45.0, "2_hip_pitch": 40.0, "3_knee_pitch": 65.0}
TAU_HARD = 120.0        # 絕對硬上限（馬達規格 150）

LEGS12 = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]

# M5 的 `smoothstep` 是**餘弦插值** f(u)=½(1−cos πu)，不是三次式 3u²−2u³。
# 峰值速度 = f'(½) × 平均速度 = **π/2 ≈ 1.5708**（三次式才是 1.5）。
# ⚠️ 我第一版寫成 1.5，被 test_smoothstep_peak_factor_matches_numeric_derivative
#    抓出來 —— 差 4.7%，會讓「走多遠」檢查低估命令速度。
SMOOTHSTEP_VPEAK = math.pi / 2


def seg_speeds(segs, joints=None):
    """每個區段「最遠的那個關節要轉多少、峰值命令速度多快」。

    segs 是 (名稱, 秒數, 起點姿勢dict, 終點姿勢dict) 的序列。
    回傳 [(名稱, 關節, 位移rad, 峰值速度rad/s, 秒數), ...]，與 segs 同序。

    ★ 存在理由：M7 只管「幾秒到」，不管「離多遠」—— t1/t2 是固定秒數，
      起點卻是狗當下的姿勢。起點離路徑點越遠，同樣秒數就要走越快。
    """
    joints = list(joints or LEGS12)
    out = []
    for nm, dur, p0, p1 in segs:
        dq = {j: abs(p1[j] - p0[j]) for j in joints}
        j = max(dq, key=dq.get)
        out.append((nm, j, dq[j], SMOOTHSTEP_VPEAK * dq[j] / max(dur, 1e-6), dur))
    return out


def read_imu_rp():
    """回傳 (roll_deg, pitch_deg)，四元數 xyzw。"""
    with shm_io.Shm("imu_central") as si:
        v = [shm_io._F8.unpack_from(si.mm, 824 + 8 * k)[0] for k in range(10)]
    qx, qy, qz, qw = v[6:10]
    roll = math.degrees(math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx)))))
    return roll, pitch


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="M7 —— 站起來（承重，風險最高）")
    ap.add_argument("--to", choices=("crouch", "stand"), default="crouch",
                    help="做到哪一階。crouch 塌下來的高度較低，先做它")
    ap.add_argument("--confirm", action="store_true", help="不帶就是乾跑")
    ap.add_argument("--kp", type=float, default=KP_DEF)
    ap.add_argument("--kd", type=float, default=KD_DEF)
    ap.add_argument("--wheel-kd", type=float, default=WHEEL_KD_DEF, dest="wheel_kd",
                    help="移動中的輪阻尼。實測 0.1~1.0 對站起來幾乎無差別")
    ap.add_argument("--wheel-kp", type=float, default=WHEEL_KP_HOLD, dest="wheel_kp",
                    help="★ 到位後鎖輪的 kp（原廠 20）")
    ap.add_argument("--wheel-kd-hold", type=float, default=WHEEL_KD_HOLD,
                    dest="wheel_kd_hold", help="鎖定期間的輪 kd（原廠 0.1）")
    ap.add_argument("--wheel-tau-max", type=float, default=8.0, dest="wheel_tau_max",
                    help="鎖定期間的輪力矩保護（原廠實測鎖定後 <1.6）")
    ap.add_argument("--no-wheel-lock", action="store_false", dest="wheel_lock",
                    help="★ 到位後也不鎖輪（M7 舊行為）。平地可以，斜地會滑走")
    ap.add_argument("--vcmd-max", type=float, default=2.0, dest="vcmd_max",
                    help="★ 乾跑檢查：命令速度上限 rad/s。起點離路徑點太遠會超過")
    ap.add_argument("--allow-high-start", action="store_true", dest="allow_high_start",
                    help="★ 起點比第一個路徑點還高（吊帶撐著）仍照跑。"
                         "★★ 那一趟不是承重測試")
    ap.add_argument("--ramp", type=float, default=2.0, help="kp 斜坡秒數（原廠是一步跳）")
    ap.add_argument("--t1", type=float, default=1.5, help="趴 ↔ crouch 的移動秒數")
    ap.add_argument("--t2", type=float, default=1.5, help="crouch ↔ stand 的移動秒數")
    ap.add_argument("--hold", type=float, default=4.0, help="到達目標後維持秒數")
    ap.add_argument("--hold-mid", type=float, default=2.0, dest="hold_mid",
                    help="停在 crouch 的秒數（原廠是 1.8）")
    ap.add_argument("--stay", action="store_true",
                    help="★ 不回程，停在目標姿勢等人工結束（★★ 腿會一直承重）")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--emax", type=float, default=0.50, help="追蹤誤差保護 rad")
    ap.add_argument("--vmax", type=float, default=4.0, help="關節速度保護 rad/s")
    ap.add_argument("--tilt-max", type=float, default=25.0, dest="tilt_max",
                    help="機身傾角保護（度）。站立時翻倒的第一個徵兆")
    ap.add_argument("--temp-max", type=float, default=70.0, dest="temp_max")
    ap.add_argument("--tau-hits", type=int, default=3, dest="tau_hits")
    a = ap.parse_args()

    logp = shm_io.start_log("M7")
    print("M7 —— 站起來（★★ 承重）\n")
    print("⚠️⚠️ 確認：狗趴在地上、16 顆洩力、**吊帶掛著但鬆弛**、地面淨空、"
          "第二個終端機備著 estop。\n")

    # ---------------------------------------------------------------- 前置
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節帶著非零增益：{', '.join(live)}")
        print("   狗必須是洩力趴著。遙控器關閉運控／趴下之後再跑。")
        return 1
    print("✅ 16 顆全部洩力中")

    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")

    roll0, pitch0 = read_imu_rp()
    print(f"✅ 機身姿態 roll {roll0:+.1f}° pitch {pitch0:+.1f}°")

    st0 = shm_io.read_joint_state()
    by = {r["name"]: r for r in st0}
    q_lie = {j: coord.to_ctrl(j, by[j]["position"]) for j in LEGS12}

    # ---- 路徑點
    WPT = [("crouch", coord.POSES["crouch"])]
    if a.to == "stand":
        WPT.append(("stand", coord.POSES["stand"]))

    print(f"\n目標 {a.to}　kp={a.kp} kd={a.kd}　輪子純阻尼 kd={a.wheel_kd}")
    print(f"路徑：趴平(實測) → " + " → ".join(n for n, _ in WPT)
          + ("" if a.stay else " → " + " → ".join(
              n for n, _ in reversed(WPT[:-1])) + (" → " if len(WPT) > 1 else "")
             + "趴平"))
    print(f"\n{'關節':16s} {'趴平(實測)':>11s} " +
          " ".join(f"{n:>10s}" for n, _ in WPT) + f" {'限位':>18s}")
    # ★ 起點與路徑點的檢查**語意不同**：
    #   路徑點是我們命令的值 → 超出就拒跑（硬性）。
    #   起點是狗**實際所在的位置** → 我們改變不了它，只能警告。
    #   實機趴著時膝就頂在 ±2.80 的機械停點，實測最遠到 ±2.8021 ——
    #   比 URDF 的 ±2.801 還多一點（編碼器誤差或停點有彈性）。
    #   把它當硬性阻擋的話，M7 永遠跑不起來。
    bad, warn = [], []
    for j in LEGS12:
        lo, hi = coord.limits_of(j)
        cells = " ".join(f"{p[j]:10.4f}" for _, p in WPT)
        m1 = coord.check_limit(j, q_lie[j], 0.0)
        m2 = next((coord.check_limit(j, p[j], 0.03) for _, p in WPT
                   if coord.check_limit(j, p[j], 0.03)), "")
        if m2:
            bad.append(f"{j}（路徑點）: {m2}")
        if m1:
            warn.append(f"{j}（起點）: {m1}")
        print(f"{j:16s} {q_lie[j]:11.4f} {cells} {f'[{lo:+.3f},{hi:+.3f}]':>18s}")
    if bad:
        print("\n❌ **路徑點**超出機構限位 —— 那是我們命令的值，拒跑：")
        for b in bad:
            print("   " + b)
        return 1
    # 起點超出模型限位多少 —— 迴圈內的檢查以此為基準，只在「**更往外跑**」時才中止。
    #   否則趴著的膝（頂在機械停點）會在第一個 tick 就誤中止。
    slack = {}
    for j in LEGS12:
        lo, hi = coord.limits_of(j)
        slack[j] = max(0.0, lo - q_lie[j], q_lie[j] - hi) + 0.01
    if warn:
        print(f"\n⚠️ {len(warn)} 個關節的**起點**略微超出模型限位（狗實際就在那）：")
        for b in warn[:4]:
            print("   " + b)
        mx = max(abs(q_lie[j]) - max(abs(x) for x in coord.limits_of(j))
                 for j in LEGS12 if coord.check_limit(j, q_lie[j], 0.0))
        print(f"   最大超出 {mx:.4f} rad。趴著時膝頂在機械停點，這是預期的。")
        if mx > 0.05:
            print("   ❌ 但超出 0.05 rad 太多了，不像機械停點 —— 停下來檢查。")
            return 1
    print("\n✅ 所有路徑點都在機構限位內")

    # ---------------------------------------------------------------- 時序
    segs = []          # (名稱, 秒數, 起點姿勢, 終點姿勢)
    prev = q_lie
    segs.append(("RAMP_UP", a.ramp, prev, prev))
    for i, (nm, p) in enumerate(WPT):
        segs.append((f"GO_{nm}", a.t1 if i == 0 else a.t2, prev, p))
        segs.append((f"HOLD_{nm}", a.hold if i == len(WPT) - 1 else a.hold_mid, p, p))
        prev = p
    if not a.stay:
        for i in range(len(WPT) - 2, -1, -1):
            nm, p = WPT[i]
            segs.append((f"BACK_{nm}", a.t2, prev, p))
            segs.append((f"HOLD_{nm}2", a.hold_mid, p, p))
            prev = p
        segs.append(("BACK_LIE", a.t1, prev, q_lie))
        segs.append(("RAMP_DOWN", a.ramp, q_lie, q_lie))
    bounds, tt = [], 0.0
    for nm, d_, p0, p1 in segs:
        bounds.append((tt, tt + d_, nm, p0, p1))
        tt += d_
    T_END = tt
    print(f"\n總時長 {T_END:.1f} 秒：" + " → ".join(f"{n}({d:.1f}s)" for n, d, _, _ in segs))

    # ---- ★ 「走多遠」檢查（2026-08-27 加）
    # M7 只管「幾秒到」，不管「離多遠」—— t1/t2 是固定秒數，起點卻是狗當下的姿勢。
    # 起點離路徑點越遠，同樣的秒數就要走越快：
    #   正常趴姿（後膝往前彎）  最遠的關節只要轉 0.4 rad → 0.4 rad/s，溫和
    #   knee_back（後膝反向）  後膝要掃 5.2 rad → 5.2 rad/s，快 13 倍
    # 5.2 rad = 300°，那條腿會從完全折起甩到反方向折，**中途整條腿打直掃過地面**，
    # 而且一定撞到 --vmax 保護；承重中的中止是「凍結原地」，
    # 會卡在腿甩到一半、打直的那個最不穩的位置。
    # smoothstep 的峰值速度 = 1.5 × 平均速度。
    print(f"\n{'區段':>13s} {'最大位移':>12s} {'峰值命令速度':>14s} {'哪個關節':>16s}")
    rows = seg_speeds(segs)
    too_fast = [r for r in rows if r[3] > a.vcmd_max]
    for nm, j, dq_, vc, d_ in rows:
        print(f"{nm:>13s} {dq_:8.3f} rad {vc:11.2f} rad/s {j:>16s}"
              f"{'  ⚠️' if vc > a.vcmd_max else ''}")
    if too_fast:
        need = max(SMOOTHSTEP_VPEAK * x[2] / a.vcmd_max for x in too_fast)
        print(f"\n❌ 有區段的命令速度超過 --vcmd-max {a.vcmd_max} rad/s —— 拒跑。")
        for nm, j, dq_, vc, d_ in too_fast:
            print(f"   {nm}：{j} 要掃 {dq_:.3f} rad（{math.degrees(dq_):.0f}°），"
                  f"{d_:.1f} 秒走完 → 峰值 {vc:.2f} rad/s")
        print(f"\n   運轉中的保護是 --vmax {a.vmax} rad/s，這樣一定會中途中止，")
        print("   而承重中的中止是**凍結原地** —— 會卡在腿甩到一半的位置。")
        print("\n   兩個處理方式：")
        print("   (1) ★ 先確認膝模式 —— 上面「趴平(實測)」那欄，`bl3`/`br3` 應該")
        print("       和 `fl3`/`fr3` **反號**（後腿往前彎，原廠預設）。")
        print("       同號＝狗現在是 knee_back，先用 M5 把姿勢喬回來再跑 M7。")
        print(f"   (2) 姿勢確實就是要這樣走的話，把 --t1/--t2 放長到 ≥{need:.1f} 秒。")
        return 1
    print(f"✅ 所有區段的命令速度都在 {a.vcmd_max} rad/s 以內")

    # ---- ★ 「起點比第一個路徑點還高」檢查（2026-08-27 加，T1 實際踩到才補的）
    # 膝越彎（|角度|越大）機身越低。若起點的膝**比 crouch 還直**，代表狗被吊帶
    # 撐在比 crouch 更高的位置 —— 那麼整段「站起來」其實是「把腿收上來」，
    # 腿完全沒有承重，這一趟量到的力矩沒有意義。
    # 2026-08-27 T1 就是這樣：起始膝 ∓2.10（吊帶約 314 mm）< crouch 的 ∓2.40（292 mm），
    # HOLD_crouch 膝力矩只有 3.67 N·m，原廠實測是 27.3–29.8。
    w0 = WPT[0][1]
    too_high = [(j, abs(q_lie[j]), abs(w0[j])) for j in LEGS12
                if j.endswith(coord.KIND_KNEE) and abs(q_lie[j]) < abs(w0[j]) - 0.02]
    if too_high:
        print(f"\n❌ **起點比 {WPT[0][0]} 還高 —— 吊帶把狗撐起來了。**")
        print(f"{'膝關節':16s} {'起點|角度|':>11s} {WPT[0][0]+'|角度|':>13s}")
        for j, a_, b_ in too_high:
            print(f"{j:16s} {a_:11.3f} {b_:13.3f}   ← 比目標還直")
        print("\n   膝越彎（|角度|越大）機身越低。起點的膝比目標還直，")
        print(f"   代表狗現在被吊在**比 {WPT[0][0]} 更高**的位置 ——")
        print("   這一趟會變成「把腿收上來」，**腿完全不承重，量到的力矩沒有意義**。")
        print("\n   處理：**把吊帶調低**，讓狗真的趴到地上（膝應接近 ±2.80），")
        print("   或至少低於目標姿勢。調完重跑乾跑確認。")
        print("\n   （確定要在這個高度跑，加 --allow-high-start）")
        if not a.allow_high_start:
            return 1
        print("\n   ⚠️ --allow-high-start：照跑，但這一趟不是承重測試。")

    print(f"\n力矩門檻（實機分段峰值 ×1.5）："
          f"ABAD {TMAX['1_hip_roll']:.0f} / HIP {TMAX['2_hip_pitch']:.0f}"
          f" / KNEE {TMAX['3_knee_pitch']:.0f} N·m，硬上限 {TAU_HARD:.0f}")
    print(f"傾角保護 ±{a.tilt_max:.0f}°（現在 roll {roll0:+.1f} pitch {pitch0:+.1f}）")
    print(f"輪子：移動中純阻尼 kd={a.wheel_kd}；" + (
        f"到位後鎖定 kp={a.wheel_kp} kd={a.wheel_kd_hold}"
        f"（原廠做法，力矩保護 {a.wheel_tau_max:.0f}）"
        if a.wheel_lock else "★ 全程不鎖 —— 地面若有斜度會慢慢滑走 ⚠️"))

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結、沒有寫入。")
        print(f"\n📄 {logp}")
        return 0
    if os.geteuid() != 0:
        print("❌ 需要 root：請加 sudo")
        return 1

    idx = {j: shm_io.idx_of(j) for j in LEGS12}
    widx = {w: shm_io.idx_of(w) for w in shm_io.WHEELS}
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    abort = ""
    peak = {j: 0.0 for j in LEGS12}
    tau_hot = {j: 0 for j in LEGS12}
    recent: list = []
    samples: list = []
    kp_now = 0.0
    des_now = dict(q_lie)
    wlock = None            # 目前鎖定的輪角（None = 純阻尼）
    wlock_seg = None        # 已為哪個區段 latch 過
    wtau_hot = 0

    def write_frame(des, kp, wlock=None):
        """wlock=None → 輪子純阻尼（移動中）；wlock={輪名: 馬達角} → 鎖在該角度。"""
        for j in LEGS12:
            shm.write_cmd(idx[j], position=coord.to_motor(j, des[j]),
                          velocity=0.0, effort=0.0, kp=kp, kd=a.kd)
        st_w = None if wlock else state_ro.states()
        for w, wi in widx.items():
            if wlock:
                # 到位後：鎖在進入該區段當下的角度（原廠 12.99 s 的 latch 做法）
                shm.write_cmd(wi, position=wlock[w], velocity=0.0, effort=0.0,
                              kp=a.wheel_kp, kd=a.wheel_kd_hold)
            else:
                # 移動中純阻尼：kp=0，只給 kd。輪子必須能自由滾（實測會滾約 100 mm）
                shm.write_cmd(wi, position=st_w[wi]["position"],
                              velocity=0.0, effort=0.0, kp=0.0, kd=a.wheel_kd)

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.15)
        print(f"\n✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")
        print(f"{'t':>6s} {'階段':>13s} {'kp':>6s} {'最大|誤差|':>10s} {'最大|τ|':>8s}"
              f" {'關節':>16s} {'roll':>6s} {'pitch':>6s}")
        t0 = time.monotonic()
        nxt = t0
        last = -1.0
        while True:
            t = time.monotonic() - t0
            if t >= T_END:
                break
            seg = next(b for b in bounds if b[0] <= t < b[1])
            s0, s1, nm, p0, p1 = seg
            u = smoothstep((t - s0) / max(s1 - s0, 1e-6))
            if nm == "RAMP_UP":
                kp_now = a.kp * ((t - s0) / max(s1 - s0, 1e-6))
            elif nm == "RAMP_DOWN":
                kp_now = a.kp * max(0.0, 1 - (t - s0) / max(s1 - s0, 1e-6))
            else:
                kp_now = a.kp
            des_now = {j: p0[j] + u * (p1[j] - p0[j]) for j in LEGS12}

            stt = state_ro.states()

            # ---- ★ 輪子：到 HOLD 區段就 latch 鎖住，離開就放回純阻尼
            if a.wheel_lock and nm.startswith("HOLD_"):
                if wlock_seg != nm:
                    wlock = {w: stt[wi]["position"] for w, wi in widx.items()}
                    wlock_seg = nm
                    print(f"       ↳ 輪子鎖定於 {nm}（kp={a.wheel_kp:g}）")
            else:
                wlock, wlock_seg = None, None
            if wlock:
                for w, wi in widx.items():
                    if abs(stt[wi]["effort"]) > a.wheel_tau_max:
                        wtau_hot += 1
                        if wtau_hot >= a.tau_hits:
                            abort = (f"{w} 鎖定中力矩 {stt[wi]['effort']:+.2f} 連續"
                                     f" {wtau_hot} 筆超過 {a.wheel_tau_max}")
                        break
                else:
                    wtau_hot = 0
            else:
                wtau_hot = 0
            if abort:
                break

            we = (0.0, ""); wt = (0.0, "")
            tick = {}
            for j in LEGS12:
                sg = coord.SIGN[j[2:]][j[:2]]
                r = stt[idx[j]]
                q = coord.to_ctrl(j, r["position"])
                v = sg * r["velocity"]
                tau = sg * r["effort"]
                err = q - des_now[j]
                tick[j] = (round(q, 4), round(des_now[j], 4), round(tau, 2), round(v, 3))
                cap = kp_now * abs(err) + a.kd * abs(v)
                if abs(tau) <= 3.0 * cap + 1.0 and abs(tau) > abs(peak[j]):
                    peak[j] = tau
                if abs(err) > we[0]:
                    we = (abs(err), j)
                if abs(tau) > wt[0]:
                    wt = (abs(tau), j)
                lim = TMAX[j[2:]]
                if abs(tau) > TAU_HARD:
                    tau_hot[j] += 1
                    if tau_hot[j] >= 2:
                        abort = f"{j} 力矩連續 2 筆超過硬上限 {TAU_HARD}（{tau:+.1f}）"
                elif abs(tau) > lim:
                    tau_hot[j] += 1
                    if tau_hot[j] >= a.tau_hits:
                        abort = f"{j} 力矩連續 {tau_hot[j]} 筆超過 {lim}（{tau:+.1f}）"
                else:
                    tau_hot[j] = 0
                if abs(err) > a.emax:
                    abort = f"{j} 追蹤誤差 {err:+.3f} 超過 {a.emax}"
                if abs(v) > a.vmax:
                    abort = f"{j} 速度 {v:+.2f} 超過 {a.vmax}"
                # ★ 以起點的超出量為基準（見 slack）。趴著時膝就頂在機械停點，
                #   拿模型限位硬比會在第一個 tick 誤中止。
                lo_j, hi_j = coord.limits_of(j)
                if q < lo_j - slack[j] or q > hi_j + slack[j]:
                    abort = (f"{j} 比起點更往限位外跑：{q:+.4f}"
                             f"（限位 [{lo_j:+.3f},{hi_j:+.3f}]，起點容差 {slack[j]:.3f}）")
                if r["temp_C"] > a.temp_max:
                    abort = f"{j} 溫度 {r['temp_C']:.1f}°C 超過 {a.temp_max}"
                if abort:
                    break
            roll, pitch = read_imu_rp()
            if not abort and max(abs(roll), abs(pitch)) > a.tilt_max:
                abort = f"機身傾角 roll {roll:+.1f}° pitch {pitch:+.1f}° 超過 ±{a.tilt_max}°"
            recent.append({"t": round(t, 3), "phase": nm, "kp": round(kp_now, 1),
                           "roll": round(roll, 2), "pitch": round(pitch, 2), "j": tick})
            if len(recent) > 60:
                recent.pop(0)
            if nm.startswith("HOLD_"):
                samples.append({"phase": nm, **{j: tick[j] for j in LEGS12}})
            if abort:
                break

            write_frame(des_now, kp_now, wlock)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

            if t - last >= 0.25:
                print(f"{t:6.2f} {nm:>13s} {kp_now:6.0f} {we[0]:10.4f} {wt[0]:8.2f}"
                      f" {wt[1]:>16s} {roll:+6.1f} {pitch:+6.1f}")
                last = t
            nxt += 1.0 / a.hz
            d_ = nxt - time.monotonic()
            if d_ > 0:
                time.sleep(d_)
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    except Exception as e:
        abort = f"未預期的例外：{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- 收尾
    # ★★ 承重時中止**不能放手**。凍結目標角、維持增益、原地撐住。
    held_des, held_kp = dict(des_now), (kp_now if abort else 0.0)
    # ★ 凍結時也把輪子鎖住。腿凍結擋不住「輪子在斜地慢慢滑走」——
    #   那是兩個獨立的自由度，中止當下 latch 在哪就鎖在哪。
    held_wlock = None
    if a.wheel_lock and held_kp > 0:
        try:
            st_now = state_ro.states()
            held_wlock = {w: st_now[wi]["position"] for w, wi in widx.items()}
        except Exception as e:
            print(f"⚠️ 中止時讀不到輪角，輪子維持純阻尼：{e}")
    keeper = Keepalive(shm, state_ro,
                       (lambda: write_frame(held_des, held_kp, held_wlock)), a.hz,
                       "凍結目標角、維持增益" if abort else "零增益保持")
    keeper.start()

    print("\n" + "=" * 76)
    if abort:
        print(f"⛔ 中止：{abort}")
        if held_kp >= 0.3 * a.kp:
            print(f"\n★★ **已凍結目標角並維持 kp={held_kp:.0f} —— 狗還撐著，沒有放手。**")
            print("   （M5 的中止是切成純阻尼；那在承重時等於讓狗塌下去，所以 M7 不那樣做。）")
        else:
            print(f"\n★ 中止時 kp 只有 {held_kp:.0f}（還在 RAMP_UP 階段，尚未真正出力）。")
            print("   狗應該還在原本的趴姿，沒有被撐起來過。")
        if recent:
            hurt = abort.split()[0]
            if hurt in LEGS12:
                print(f"\n中止前最後 12 筆 —— {hurt}")
                print(f"  {'t':>7s} {'階段':>13s} {'q':>9s} {'des':>9s} {'τ':>8s}"
                      f" {'v':>7s} {'kp|e|+kd|v|':>11s}")
                for rr in recent[-12:]:
                    q_, d_, tau_, v_ = rr["j"][hurt]
                    cap = rr["kp"] * abs(q_ - d_) + a.kd * abs(v_)
                    print(f"  {rr['t']:7.3f} {rr['phase']:>13s} {q_:9.4f} {d_:9.4f}"
                          f" {tau_:8.2f} {v_:7.3f} {cap:11.1f}")
    else:
        print("✅ 序列完整跑完")

    print(f"\n{'關節':16s} {'峰值τ':>9s} {'門檻':>7s} {'用掉':>7s}")
    for j in LEGS12:
        lim = TMAX[j[2:]]
        print(f"{j:16s} {peak[j]:+9.2f} {lim:7.0f} {100*abs(peak[j])/lim:6.0f}%")
    print(f"\n全部腿關節峰值 {max(abs(v) for v in peak.values()):.2f} N·m"
          f"（實機原廠做同樣動作是 42.45）")

    # ---- 收工：等人工，而不是自己放手
    print("\n" + "=" * 76)
    if abort or a.stay:
        print("★ 現在腿還在承重。接下來由你決定：")
        print("   [Enter] → 依原路徑**慢慢坐回趴姿**，再降增益（建議）")
        print("   [Ctrl-C] → 立刻放手（★ 狗會塌下去，只有在確定安全時才用）")
        try:
            if sys.stdin.isatty():
                input("\n   > ")
            else:
                print("   非互動模式 → 直接執行坐回去")
            keeper.stop()
            # 反向走回趴姿
            back = [("SIT_" + n, a.t2 if i else a.t1, p)
                    for i, (n, p) in enumerate(reversed(WPT))]
            print("\n坐回趴姿中…")
            cur = dict(held_des)
            t0 = time.monotonic()
            for nm, dur, _ in back[1:] + [("LIE", a.t1, None)]:
                tgt = q_lie if nm == "LIE" else dict(WPT[0][1])
                s = time.monotonic()
                while (e := time.monotonic() - s) < dur:
                    u = smoothstep(e / dur)
                    dd = {j: cur[j] + u * (tgt[j] - cur[j]) for j in LEGS12}
                    write_frame(dd, held_kp)
                    shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
                    time.sleep(1.0 / a.hz)
                cur = dict(tgt)
            # 增益歸零
            s = time.monotonic()
            while (e := time.monotonic() - s) < a.ramp:
                write_frame(cur, held_kp * max(0.0, 1 - e / a.ramp))
                shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
                time.sleep(1.0 / a.hz)
            for i in range(len(shm_io.JOINTS)):
                shm.zero_gains(i)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
            print("✅ 已坐回趴姿並降到零增益")
            keeper = Keepalive(shm, state_ro,
                               (lambda: [shm.zero_gains(i)
                                         for i in range(len(shm_io.JOINTS))]),
                               a.hz, "零增益保持")
            keeper.start()
        except KeyboardInterrupt:
            print("\n   （Ctrl-C：放手）")
    keeper.stop()

    out = {"schema": "m7_standup/1", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "args": vars(a), "aborted": bool(abort), "abort_reason": abort or None,
           "q_lie": q_lie, "peak": peak, "recent": recent,
           "hold_samples": samples[:6000]}
    jp = (logp[:-4] if logp.endswith(".log") else logp) + ".json"
    try:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        if os.geteuid() == 0 and os.getenv("SUDO_USER"):
            import pwd
            pw = pwd.getpwnam(os.environ["SUDO_USER"])
            try:
                os.chown(jp, pw.pw_uid, pw.pw_gid)
            except OSError:
                pass
        print(f"\n📊 {jp}")
    except Exception as e:
        print(f"\n⚠️ 結果檔寫入失敗：{e}")

    try:
        shm.close()
        state_ro.close()
    except Exception:
        pass
    if frozen:
        print(f"\n⏸ mc_ctrl 仍在凍結中（PID {pid}）。確認狗安全後交還控制權：")
        print(f"      sudo kill -CONT {pid}")
    print(f"\n📄 {logp}")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
