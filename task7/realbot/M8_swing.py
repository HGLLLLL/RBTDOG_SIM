#!/usr/bin/env python3
"""M8 —— 承重站立下的**靜態撓度量測**與**單腿擺動**（原地，不前進）。

這是「跑自己的步態」之前的最後一塊實機拼圖。M7 證明了狗撐得住 41 kg，
但**步態要求一腿懸空、三腿承重**，那件事完全沒測過。

════════════════════════════════════════════════════════════════════
M8 要回答的四個問題
════════════════════════════════════════════════════════════════════

1. **`STATIC_SAG` 在實機是多少？**（`max_model.py` 的 0.0325 是模擬量的）
   位置伺服撐著機身時必須留追蹤誤差，那個誤差換算成足端就是撓度。
   ★ 這是 CPG 線的頭號坑：`cpg_max.foot_targets` 明寫
     「實際離地 = g_c − 撓度，g_c 小於撓度時**腿根本不離地**，
      而超限／飽和／IK 縮限／相位鎖定**四個診斷指標全是乾淨的**」。
   撓度隨 kp 變（≈ τ/kp），所以 **S1 會掃多個 kp**。

2. **抬腿指令 g_c 下去，實際離地幾 mm？**（模擬說 80 → 102.7 mm）

3. **三腿承重時的力矩峰值多少？** M7 的門檻是四腿平均分攤時訂的。

4. **抬腿時機身傾多少？** 決定步態需不需要先移重心。

════════════════════════════════════════════════════════════════════
分階段（跟 M5 一樣，每一階可以單獨停）
════════════════════════════════════════════════════════════════════

  S0  乾跑           印計畫與預測，不寫入、不凍結、不需 sudo
  S1  撓度掃描       站起來，**不抬腿**，掃 kp 量靜態撓度      ← 零新風險
  S2  重心橫移       四腳不離地，機身左右／前後移動            ← 低
  S3  單腿擺動       先移重心 → 抬一條腿 → 放下 → 換下一條      ← ★ 真正的新風險

⚠️ **S3 之前一定要先看 S2 的結果**：三輪支撐時重心恰好落在支撐三角形的邊上，
   不先移重心就抬腿，物理上就是臨界穩定。S2 量的正是「移得動多少」。

════════════════════════════════════════════════════════════════════
★★ S3 會不會翻？（2026-08-27 在 MuJoCo 量過，不是推論）
════════════════════════════════════════════════════════════════════

**不會翻倒，但會「坐回抬起的那條腿」。**

抬起的腳只離地 80~150 mm，機身 roll 到 **15~17°** 時那隻腳就自己碰回地面
變成支撐 —— 它是自己的後擋。實測推它：roll 在 16~20° 飽和、質心高度不變、
不會繼續倒下去。（幾何自洽：抬起的輪離傾覆軸 305 mm，asin(80/305)=15.2°。）

**但「不會翻」不等於「安全」**，橫向推力的耐受度差很多：

| `--shift-y` | 對角腿分到的重量 | 撐得住的橫推 | 傾覆角 |
|---|---|---|---|
| 50 mm | **17 N（4.5%）** | **50 N 就坐回去** | 8.3° |
| 100 mm | 57 N（15%） | 100 N，200 N 坐回去 | 14.4° |
| **140 mm（預設）** | **89 N（23%）** | **>200 N** | **19.1°** |

→ 預設用 140 mm。`--tilt-max` 設 **15°**，在自己碰地（15~17°）之前就停。

⚠️ **roll 不完全收斂**：SWING 拉長到 10 秒，roll 由 5.6° 緩慢爬到 7.2°、
   對角腿的力由 71 N 掉到 51 N。`--hold-lift` 預設 1.5 秒沒問題，
   **不要為了「看清楚」把它拉長到 5 秒以上。**

════════════════════════════════════════════════════════════════════
前置條件（與 M7 相同）
════════════════════════════════════════════════════════════════════
  - **狗趴在地上**（不是吊掛），16 顆洩力
  - **吊帶掛在 crouch（292 mm）以下**，鬆弛但接得住
  - 地面淨空平坦、四周無人、第二個終端機備著 `sudo ~/estop_max.sh`
  - M7 的 T1／T2 都通過（`eval_m7.py` 放行）

★★ **中止語意與 M7 相同**（承重）：凍結目標角、維持增益、原地撐住。
   放手＝狗塌下去。**不要按 Ctrl-C**，除非確定放手是安全的。

用法：
    python3 M8_swing.py --stage 1                      # 乾跑
    sudo python3 M8_swing.py --stage 1 --confirm       # S1 撓度掃描
    sudo python3 M8_swing.py --stage 2 --confirm       # S2 重心橫移
    sudo python3 M8_swing.py --stage 3 --legs fr --confirm    # S3 只抬一條腿
    sudo python3 M8_swing.py --stage 3 --confirm              # S3 四條腿

⚠️ **凍結時間**：`mc_ctrl` 側 2026-08-27 已實測到 **200 秒完全正常**
   （`M_freezetest.py`，`logs/m_logs_trip9/MF_*`）。所以 `--max-freeze` 現在管的是
   **承重時間** —— 腿長時間吃 41 kg 的發熱**還沒量過**，M7/M8 最長只撐了 30 秒。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import coord
import kin
import shm_io
from M5_leg_pose import Keepalive, mc_ctrl_pid, proc_state, smoothstep
from M7_standup import (KD_DEF, KP_DEF, TAU_HARD, WHEEL_KD_DEF, WHEEL_KD_HOLD,
                        WHEEL_KP_HOLD, read_imu_rp, seg_speeds)

LEGS12 = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]

# 要鎖輪 + 要取樣的區段字首。
# ★ LIFT/DROP/PRE/POST 也鎖 —— 模擬顯示輪子純阻尼時「移重心」會被輪子滾掉
#   （同樣 shift 純阻尼實際離地 −1.2 mm、鎖輪 13.1 mm）。
#   輪足構型要移重心，必須先把輪子釘住。
# ★ S4 的 FWD_/BACKX_ 也要取樣 —— **峰值落後發生在移動中，不在停住時**。
#   只取樣 HOLD 會量到「讓它慢慢追上之後的穩態」，那是錯的量。
HOLDISH = ("HOLD", "SAG_", "SWING_", "PREHOLD_", "SETTLE_",
           "PRE_", "POST_", "LIFT", "DROP", "FWD_", "BACKX_")

# 力矩門檻。★ 三腿承重時單腿分攤變大，所以比 M7 的（45/40/65）再放寬一點，
#   但仍遠低於馬達規格 150。M7 實測四腿站立峰值只有 27 → 這裡留了 2.4 倍。
TMAX = {"1_hip_roll": 50.0, "2_hip_pitch": 50.0, "3_knee_pitch": 70.0}

# S1 掃描的增益。250 是 M7 驗證過的站立增益、120 是 CPG／RL 用的那組。
KP_SCAN = (250.0, 180.0, 120.0)

# 站立姿勢的足端基準（由 coord.POSES["stand"] 的正向運動學算出）
STAND = coord.POSES["stand"]


def foot_ref() -> dict:
    """四腿在 `stand` 姿勢下的輪軸心位置（相對各腿 ABAD 原點）。"""
    return {lg: kin.foot_of(lg, STAND) for lg in coord.LEGS}


def knee_signs() -> dict:
    return {lg: kin.knee_sign_of(STAND, lg) for lg in coord.LEGS}


def pose_from_feet(feet: dict, ks: dict) -> tuple[dict, int]:
    """足端目標 → 12 個關節角。回傳 (姿勢, 被縮限的腿數)。"""
    out, n = {}, 0
    for lg in coord.LEGS:
        q, clamped = kin.ik(lg, *feet[lg], knee_sign=ks[lg])
        n += int(clamped)
        for kind, v in zip(coord.LEG_KINDS, q):
            out[lg + kind] = v
    return out, n


def shift_feet(ref: dict, dx: float, dy: float, dz: float = 0.0) -> dict:
    """把足端整體平移。

    ★ 注意號：要把**機身**往 +x 移，四個**足端**要往 −x 移。
      `dx`/`dy` 這裡是**機身**的位移，函式內部自己變號 —— 現場講的都是機身。
    """
    return {lg: (ref[lg][0] - dx, ref[lg][1] - dy, ref[lg][2] + dz)
            for lg in coord.LEGS}


def move_feet(ref: dict, leg: str, dx: float, dz: float = 0.0) -> dict:
    """只把某一條腿的足端往前移 dx（並可同時抬 dz），其餘不動。

    ★ +x 是機身的前方，四條腿都一樣（`kin.SIDE` 的 sx 是 ABAD→HIP 的偏移方向，
      不是運動方向 —— 別拿它去變號）。
    """
    return {lg: (ref[lg][0] + (dx if lg == leg else 0.0), ref[lg][1],
                 ref[lg][2] + (dz if lg == leg else 0.0)) for lg in coord.LEGS}


def lift_feet(ref: dict, leg: str, dz: float) -> dict:
    """只把某一條腿的足端抬高 dz（其餘不動）。"""
    return {lg: (ref[lg][0], ref[lg][1], ref[lg][2] + (dz if lg == leg else 0.0))
            for lg in coord.LEGS}


def build_segments(a, q_lie: dict, ref: dict, ks: dict):
    """(名稱, 秒數, 起點姿勢, 終點姿勢, kp) 的序列。kp=None 代表用 a.kp。

    ★ 各階**獨立**、不累加：每一階都是「站起來 → 做該階的事 → 坐回去」。
      累加會把凍結時間吃爆 —— 實測支持的上限只有 38 秒
      （四條腿全做的累加版是 77.6 秒）。想做完整套就分幾趟跑。
    """
    stand_pose, _ = pose_from_feet(ref, ks)
    crouch = dict(coord.POSES["crouch"])
    segs = [("RAMP_UP", a.ramp, q_lie, q_lie, None),
            ("GO_crouch", a.t1, q_lie, crouch, None),
            ("HOLD_crouch", a.hold_mid, crouch, crouch, None),
            ("GO_stand", a.t2, crouch, stand_pose, None),
            ("HOLD_stand", a.hold, stand_pose, stand_pose, None)]

    if a.stage == 1:
        # S1：站著不動，把 kp 換到掃描值各停一段。撓度 = 追蹤誤差 → 足端。
        # ★ kp 用斜坡切換，不要階躍 —— 原廠在 5.98s 一步從 0 跳到 250，
        #   代價就是 ABAD/HIP 瞬間衝到 20–28 N·m（`原廠站立實測` §1）。
        #   降 kp 比升 kp 溫和，但升回去那一次一定要爬。
        for kp in a.kp_scan:
            segs.append((f"KPRAMP_{kp:.0f}", a.kp_ramp, stand_pose, stand_pose,
                         ("ramp", kp)))
            segs.append((f"SAG_kp{kp:.0f}", a.hold, stand_pose, stand_pose, kp))
        segs.append(("KPRAMP_back", a.kp_ramp, stand_pose, stand_pose,
                     ("ramp", a.kp)))
        segs.append(("HOLD_stand2", a.settle, stand_pose, stand_pose, None))

    if a.stage == 2:
        # S2：四腳不離地，機身平移。先橫向、再縱向，各來回一次。
        for nm, dx, dy in (("SHIFT_L", 0.0, +a.shift), ("SHIFT_R", 0.0, -a.shift),
                           ("SHIFT_F", +a.shift, 0.0), ("SHIFT_B", -a.shift, 0.0)):
            p, _ = pose_from_feet(shift_feet(ref, dx, dy), ks)
            segs.append((f"GO_{nm}", a.t_shift, None, p, None))
            segs.append((f"HOLD_{nm}", a.hold_shift, p, p, None))
            segs.append((f"BACK_{nm}", a.t_shift, p, stand_pose, None))
            segs.append((f"SETTLE_{nm}", a.settle, stand_pose, stand_pose, None))

    if a.stage == 3:
        # S3：移一次重心 → **掃多個抬腿高度** → 收回。
        #
        # ★ 為什麼要掃高度而不是只抬一次：模擬顯示「實際離地 = 指令 − 固定損失」，
        #   損失約 68 mm 而且**與指令高度無關**（80→13、120→52、160→91、200→130）。
        #   只量一個點分不出「固定偏移」和「增益誤差」；量三個點才有斜率。
        #   斜率≈1 → 純偏移，補償就是加一個常數（CPG 的 z_sag 就是這樣用的）。
        #
        # ⚠️ CPG 基準的 g_c=0.08 在這個情境下**幾乎不離地（13 mm）**，
        #   所以掃描要從能離地的高度開始，否則量到的全是 0。
        for lg in a.legs:
            sx = -a.shift_x if lg in coord.FRONT_LEGS else +a.shift_x
            sy = -a.shift_y if lg in ("fl", "bl") else +a.shift_y
            shifted = shift_feet(ref, sx, sy)
            p_shift, _ = pose_from_feet(shifted, ks)
            segs.append((f"PRE_{lg}", a.t_shift, None, p_shift, None))
            segs.append((f"PREHOLD_{lg}", a.hold_shift, p_shift, p_shift, None))
            for gc in a.gc_scan:
                p_lift, _ = pose_from_feet(lift_feet(shifted, lg, gc), ks)
                tag = f"{lg}_{gc*1000:.0f}"
                segs.append((f"LIFT_{tag}", a.t_lift, p_shift, p_lift, None))
                segs.append((f"SWING_{tag}", a.hold_lift, p_lift, p_lift, None))
                segs.append((f"DROP_{tag}", a.t_lift, p_lift, p_shift, None))
            segs.append((f"POST_{lg}", a.t_shift, p_shift, stand_pose, None))
            segs.append((f"SETTLE_{lg}", a.settle, stand_pose, stand_pose, None))

    if a.stage == 4:
        # S4：擺動腿的**前跨**量測。抬起來 → 往前移 Δx → 量實際走了多少。
        #
        # ★★ 為什麼是「固定 Δx、掃秒數」而不是「掃 Δx」：
        #   模擬顯示前腳落後有一個固定成分（自走 ≈ 指令 − 109 mm，斜率 1.04），
        #   但那 109 mm 是**擺動相 107 ms 之內的動態落後**，不是穩態誤差。
        #   移完停住讓它慢慢追，量到的會是穩態值 —— 那是錯的量。
        #   所以固定 Δx，掃移動秒數 → 得到「落後 vs 速度」，那才是步態要的。
        #
        # ⚠️ 步態擺動相的關節命令速度是 **6.59 rad/s**，是 --vcmd-max 2.0 的 3.3 倍。
        #   預設掃 1.5 / 1.0 / 0.5 秒（0.47 / 0.71 / 1.41 rad/s）全在護欄內，
        #   從慢往快逼近。要再快必須自己明確調高 --vcmd-max 並承擔。
        for lg in a.legs:
            sx = -a.shift_x if lg in coord.FRONT_LEGS else +a.shift_x
            sy = -a.shift_y if lg in ("fl", "bl") else +a.shift_y
            shifted = shift_feet(ref, sx, sy)
            p_shift, _ = pose_from_feet(shifted, ks)
            p_up, _ = pose_from_feet(lift_feet(shifted, lg, a.gc_x), ks)
            p_fwd, _ = pose_from_feet(move_feet(shifted, lg, a.dx, a.gc_x), ks)
            segs.append((f"PRE_{lg}", a.t_shift, None, p_shift, None))
            segs.append((f"PREHOLD_{lg}", a.hold_shift, p_shift, p_shift, None))
            segs.append((f"LIFTX_{lg}", a.t_lift, p_shift, p_up, None))
            for dur in a.t_fwd_scan:
                tag = f"{lg}_{dur*1000:.0f}"
                segs.append((f"FWD_{tag}", dur, p_up, p_fwd, None))
                segs.append((f"HOLDX_{tag}", a.hold_lift, p_fwd, p_fwd, None))
                segs.append((f"BACKX_{tag}", dur, p_fwd, p_up, None))
            segs.append((f"DROPX_{lg}", a.t_lift, p_up, p_shift, None))
            segs.append((f"POST_{lg}", a.t_shift, p_shift, stand_pose, None))
            segs.append((f"SETTLE_{lg}", a.settle, stand_pose, stand_pose, None))

    segs += [("BACK_crouch", a.t2, stand_pose, crouch, None),
             ("HOLDB_crouch", a.hold_mid, crouch, crouch, None),
             ("BACK_LIE", a.t1, crouch, q_lie, None),
             ("RAMP_DOWN", a.ramp, q_lie, q_lie, None)]

    # 把 p0=None 的區段接上前一段的終點
    prev = q_lie
    out = []
    for nm, dur, p0, p1, kp in segs:
        p0 = prev if p0 is None else p0
        out.append((nm, dur, p0, p1, kp))
        prev = p1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="M8 —— 靜態撓度量測與單腿擺動（承重，原地）")
    ap.add_argument("--stage", type=int, choices=(0, 1, 2, 3, 4), default=1,
                    help="★ 各階獨立不累加。0=只站起來 1=撓度掃描 "
                         "2=重心橫移 3=單腿抬高 4=單腿前跨")
    ap.add_argument("--legs", nargs="*", default=list(coord.LEGS),
                    help="S3 要抬哪幾條腿（預設四條）")
    ap.add_argument("--confirm", action="store_true", help="不帶就是乾跑")
    ap.add_argument("--kp", type=float, default=KP_DEF)
    ap.add_argument("--kd", type=float, default=KD_DEF)
    ap.add_argument("--kp-scan", nargs="*", type=float, default=list(KP_SCAN),
                    dest="kp_scan", help="S1 要掃哪幾個 kp")
    ap.add_argument("--gc-scan", nargs="*", type=float, dest="gc_scan",
                    default=[0.08, 0.15, 0.22],
                    help="★ S3 要掃哪幾個抬腿指令高度 m。"
                         "0.08 是 CPG 基準（模擬預測幾乎不離地），"
                         "後兩個是為了拿到斜率")
    ap.add_argument("--shift", type=float, default=0.04, help="S2 機身平移量 m")
    ap.add_argument("--dx", type=float, default=0.12,
                    help="★ S4 的前跨指令 m。步態在 d_step=0.10 時是 0.118")
    ap.add_argument("--gc-x", type=float, default=0.15, dest="gc_x",
                    help="S4 前跨時的抬腿高度 m（實測 0.15 → 真實離地約 84 mm）")
    ap.add_argument("--t-fwd-scan", nargs="*", type=float, dest="t_fwd_scan",
                    default=[1.5, 1.0, 0.5],
                    help="★ S4 的前跨秒數（由慢到快）。步態擺動相是 0.107 秒，"
                         "那需要 6.59 rad/s —— 遠超 --vcmd-max，不要一步跳過去")
    ap.add_argument("--shift-x", type=float, default=0.03, dest="shift_x",
                    help="S3 抬腿前的機身縱向位移 m")
    ap.add_argument("--shift-y", type=float, default=0.14, dest="shift_y",
                    help="★ S3 抬腿前的機身橫向位移 m。50/100/140 的橫推耐受度是"
                         "50 N / 100 N / >200 N —— 50 mm 明顯不夠")
    ap.add_argument("--wheel-kd", type=float, default=WHEEL_KD_DEF, dest="wheel_kd")
    ap.add_argument("--wheel-kp", type=float, default=WHEEL_KP_HOLD, dest="wheel_kp")
    ap.add_argument("--wheel-kd-hold", type=float, default=WHEEL_KD_HOLD,
                    dest="wheel_kd_hold")
    ap.add_argument("--no-wheel-lock", action="store_false", dest="wheel_lock")
    ap.add_argument("--ramp", type=float, default=2.0)
    ap.add_argument("--t1", type=float, default=1.5)
    ap.add_argument("--t2", type=float, default=1.5)
    ap.add_argument("--t-shift", type=float, default=1.2, dest="t_shift")
    ap.add_argument("--t-lift", type=float, default=1.0, dest="t_lift",
                    help="★ 抬腿／放下的秒數。0.8 會讓 220 mm 那格衝到 2.16 rad/s "
                         "撞到 --vcmd-max；1.0 是 1.73，餘裕 14%%")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--hold-mid", type=float, default=1.5, dest="hold_mid")
    ap.add_argument("--hold-shift", type=float, default=1.5, dest="hold_shift")
    ap.add_argument("--hold-lift", type=float, default=1.5, dest="hold_lift")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--kp-ramp", type=float, default=1.0, dest="kp_ramp",
                    help="S1 切換 kp 的斜坡秒數（不要階躍）")
    ap.add_argument("--max-freeze", type=float, default=60.0, dest="max_freeze",
                    help="★ 總時長上限。**這管的是承重時間，不是 mc_ctrl** —— "
                         "mc_ctrl 已實測凍 200 秒正常，但腿長時間吃 41 kg 的發熱"
                         "沒量過，目前最長只做過 34.6 秒")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--gap-max", type=float, default=0.25, dest="gap_max",
                    help="★ 單次迴圈間隔上限（秒）。controller 的 joint_cmd_timeout "
                         "是 500 ms —— 超過就會把指令區清成 0，承重中等於放手。"
                         "預設取一半當警戒線")
    ap.add_argument("--emax", type=float, default=0.50)
    ap.add_argument("--vmax", type=float, default=4.0)
    ap.add_argument("--vcmd-max", type=float, default=2.0, dest="vcmd_max")
    ap.add_argument("--tilt-max", type=float, default=15.0, dest="tilt_max",
                    help="★ 比 M7 嚴很多（15 vs 25）。S3 正常是 6.4°，"
                         "而 15~17° 抬起的腳就會自己碰地 —— 要在那之前停")
    ap.add_argument("--temp-max", type=float, default=70.0, dest="temp_max")
    ap.add_argument("--tau-hits", type=int, default=3, dest="tau_hits")
    ap.add_argument("--wheel-tau-max", type=float, default=8.0, dest="wheel_tau_max")
    a = ap.parse_args()

    bad_legs = [l for l in a.legs if l not in coord.LEGS]
    if bad_legs:
        print(f"❌ 不認得的腿名 {bad_legs}，只能是 {list(coord.LEGS)}")
        return 1

    logp = shm_io.start_log("M8")
    print(f"M8 —— 靜態撓度量測與單腿擺動（★★ 承重）　stage={a.stage}\n")
    print("⚠️⚠️ 確認：狗趴在地上、16 顆洩力、**吊帶掛在 292 mm 以下且鬆弛**、"
          "地面淨空、第二個終端機備著 estop。\n")

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
        print(f"❌ 這些關節目前帶著非零增益，先處理再跑：{live}")
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

    ref, ks = foot_ref(), knee_signs()
    print(f"\n站姿足端基準（輪軸心相對 ABAD 原點，m）　膝分支 "
          f"{ {lg: int(ks[lg]) for lg in coord.LEGS} }")
    for lg in coord.LEGS:
        print(f"  {lg}: x{ref[lg][0]:+.4f} y{ref[lg][1]:+.4f} z{ref[lg][2]:+.4f}")

    # ---- 起點檢查（沿用 M7 的兩條）
    crouch = coord.POSES["crouch"]
    too_high = [(j, abs(q_lie[j]), abs(crouch[j])) for j in LEGS12
                if j.endswith(coord.KIND_KNEE) and abs(q_lie[j]) < abs(crouch[j]) - 0.02]
    if too_high:
        print("\n❌ **起點比 crouch 還高 —— 吊帶把狗撐起來了。**")
        for j, x, y in too_high:
            print(f"   {j}: 起點 |{x:.3f}| < crouch |{y:.3f}|")
        print("   把吊帶調低（見 M7 的說明），這一趟量到的撓度會是假的。")
        return 1

    front = [q_lie[lg + coord.KIND_KNEE] for lg in coord.FRONT_LEGS]
    rear = [q_lie[lg + coord.KIND_KNEE] for lg in coord.REAR_LEGS]
    if all(math.copysign(1, f) == math.copysign(1, r) for f in front for r in rear):
        print("\n❌ 前後膝同號 = knee_back 模式，先用 M5 喬回來")
        return 1
    print("✅ 前後膝反號（後腿往前彎，原廠預設）")

    # ---------------------------------------------------------------- 時序
    segs = build_segments(a, q_lie, ref, ks)
    T_END = sum(s[1] for s in segs)
    print(f"\n總時長 {T_END:.1f} 秒　{len(segs)} 段")
    print("  " + " → ".join(f"{n}({d:.1f})" for n, d, _, _, _ in segs))

    if T_END > a.max_freeze:
        print(f"\n❌ **預估凍結 {T_END:.1f} 秒超過上限 {a.max_freeze:.0f} 秒。**")
        print("   ⚠️ `mc_ctrl` 側已實測到 200 秒沒問題（`M_freezetest.py`，2026-08-27），")
        print("   **所以這個上限管的是承重時間** —— 腿長時間吃 41 kg 的發熱還沒量過，")
        print("   目前做過最長的是 34.6 秒（S2）。要更久請自己明確承擔並調高。")
        print("   處理：減少 --legs、降低 --stage、縮短 --hold/--settle，")
        print("   或明確承擔風險把 --max-freeze 調高。")
        return 1
    print(f"✅ 凍結時間 {T_END:.1f}s 在上限 {a.max_freeze:.0f}s 之內")

    # ---- 走多遠檢查
    rows = seg_speeds([(n, d, p0, p1) for n, d, p0, p1, _ in segs], LEGS12)
    too_fast = [r for r in rows if r[3] > a.vcmd_max]
    print(f"\n{'區段':>14s} {'最大位移':>12s} {'峰值命令速度':>14s} {'哪個關節':>16s}")
    for nm, j, dq, vc, dur in rows:
        if dq < 1e-9:
            continue
        print(f"{nm:>14s} {dq:8.3f} rad {vc:11.2f} rad/s {j:>16s}"
              f"{'  ⚠️' if vc > a.vcmd_max else ''}")
    if too_fast:
        print(f"\n❌ 有區段超過 --vcmd-max {a.vcmd_max} rad/s：")
        for nm, j, dq, vc, dur in too_fast:
            print(f"   {nm}：{j} 掃 {dq:.3f} rad / {dur:.1f}s → {vc:.2f} rad/s")
        print("   把對應的 --t-shift / --t-lift 放長。")
        return 1
    print(f"✅ 所有區段的命令速度都在 {a.vcmd_max} rad/s 以內")

    # ---- 限位檢查（路徑點是我們命令的值 → 超出就拒跑）
    bad = []
    for nm, dur, p0, p1, kp in segs:
        for j in LEGS12:
            msg = coord.check_limit(j, p1[j], 0.03)
            if msg:
                bad.append(f"{nm} / {j}: {msg}")
    if bad:
        print("\n❌ 路徑點超出機構限位，拒跑：")
        for b in bad[:8]:
            print("   " + b)
        return 1
    print("✅ 所有路徑點都在機構限位內")

    print(f"\n力矩門檻：ABAD {TMAX['1_hip_roll']:.0f} / HIP {TMAX['2_hip_pitch']:.0f}"
          f" / KNEE {TMAX['3_knee_pitch']:.0f} N·m，硬上限 {TAU_HARD:.0f}")
    print(f"傾角保護 ±{a.tilt_max:.0f}°（現在 roll {roll0:+.1f} pitch {pitch0:+.1f}）")
    print(f"輪子：移動中純阻尼 kd={a.wheel_kd}；" + (
        f"HOLD 段鎖定 kp={a.wheel_kp}" if a.wheel_lock else "全程不鎖 ⚠️"))
    if a.stage >= 3:
        print(f"\n★ S3 會抬 {a.legs}，指令高度掃 "
              f"{[f'{g*1000:.0f}' for g in a.gc_scan]} mm，"
              f"抬腿前機身先移 x{a.shift_x*1000:+.0f} y{a.shift_y*1000:+.0f} mm")
        print("  ⚠️ MuJoCo 預測（kp=250、鎖輪、shift(30,100)）：")
        print("     指令  80 → 實際  13 mm　　指令 150 → 實際 ~82 mm")
        print("     指令 120 → 實際  52 mm　　指令 220 → 實際 ~150 mm")
        print("     **損失約 68 mm 且與指令高度無關** —— 是 `STATIC_SAG=32.5` 的兩倍多。")
        print("     若實機也是這樣，CPG 基準的 g_c=0.08 在實機上腿幾乎不會離地。")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結、沒有寫入。")
        print(f"\n📄 {logp}")
        return 0
    if os.geteuid() != 0:
        print("❌ 需要 root：請加 sudo")
        return 1

    # ---------------------------------------------------------------- 執行
    bounds, tt = [], 0.0
    for nm, dur, p0, p1, kp in segs:
        bounds.append((tt, tt + dur, nm, p0, p1, kp))
        tt += dur

    idx = {j: shm_io.idx_of(j) for j in LEGS12}
    widx = {w: shm_io.idx_of(w) for w in shm_io.WHEELS}
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    abort = ""
    peak = {j: 0.0 for j in LEGS12}
    tau_hot = {j: 0 for j in LEGS12}
    wtau_hot = 0
    recent: list = []
    samples: list = []
    kp_now = 0.0
    des_now = dict(q_lie)
    worst_gap = 0.0          # 最長的單次迴圈間隔（不是平均 —— 平均漂亮也可能有一次長停頓）
    worst_gap_t = 0.0
    n_tick = 0
    t_prev = None
    wlock = None
    wlock_seg = None
    cur_seg = None
    kp_seg_start = 0.0

    def write_frame(des, kp, wl=None):
        for j in LEGS12:
            shm.write_cmd(idx[j], position=coord.to_motor(j, des[j]),
                          velocity=0.0, effort=0.0, kp=kp, kd=a.kd)
        st_w = None if wl else state_ro.states()
        for w, wi in widx.items():
            if wl:
                shm.write_cmd(wi, position=wl[w], velocity=0.0, effort=0.0,
                              kp=a.wheel_kp, kd=a.wheel_kd_hold)
            else:
                shm.write_cmd(wi, position=st_w[wi]["position"],
                              velocity=0.0, effort=0.0, kp=0.0, kd=a.wheel_kd)


    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.15)
        print(f"\n✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")
        print(f"{'t':>6s} {'階段':>14s} {'kp':>6s} {'最大|誤差|':>10s} {'最大|τ|':>8s}"
              f" {'關節':>16s} {'roll':>6s} {'pitch':>6s}")
        t0 = time.monotonic()
        nxt = t0
        last = -1.0
        while True:
            t = time.monotonic() - t0
            if t >= T_END:
                break
            # ---- ★ 迴圈間隔監看（2026-08-27 加）
            # M5 有查送出速率，M7/M8 原本沒有。狗上 CPU 一忙（開機、別的行程），
            # 我們的 python 迴圈就會被排程延遲；單次停頓超過 500 ms →
            # controller 判定指令過期 → **把指令區清成 0** → 承重中就是放手。
            # ⚠️ 要看**最長單次間隔**，不是平均：平均 200 Hz 也可能藏一次 600 ms 停頓。
            if t_prev is not None:
                gap = t - t_prev
                if gap > worst_gap:
                    worst_gap, worst_gap_t = gap, t
                if gap > a.gap_max:
                    abort = (f"迴圈間隔 {gap*1000:.0f} ms 超過 {a.gap_max*1000:.0f} ms"
                             f"（controller 逾時是 500 ms，超過就會清零）")
            t_prev = t
            n_tick += 1
            if abort:
                break
            s0, s1, nm, p0, p1, kp_seg = next(b for b in bounds if b[0] <= t < b[1])
            u = smoothstep((t - s0) / max(s1 - s0, 1e-6))
            if nm != cur_seg:
                cur_seg, kp_seg_start = nm, kp_now
            frac = (t - s0) / max(s1 - s0, 1e-6)
            if nm == "RAMP_UP":
                kp_now = a.kp * frac
            elif nm == "RAMP_DOWN":
                kp_now = a.kp * max(0.0, 1 - frac)
            elif isinstance(kp_seg, tuple):          # ("ramp", 目標 kp)
                kp_now = kp_seg_start + (kp_seg[1] - kp_seg_start) * min(frac, 1.0)
            else:
                kp_now = a.kp if kp_seg is None else kp_seg
            des_now = {j: p0[j] + u * (p1[j] - p0[j]) for j in LEGS12}

            stt = state_ro.states()

            if a.wheel_lock and nm.startswith(HOLDISH):
                if wlock_seg != nm:
                    wlock = {w: stt[wi]["position"] for w, wi in widx.items()}
                    wlock_seg = nm
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

            we = (0.0, "")
            wt = (0.0, "")
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
                if abs(tau) <= 1.5 * cap + 1.0 and abs(tau) > abs(peak[j]):
                    peak[j] = tau       # ★ 1.5 倍：與 eval 的感測尖峰判別式同一把尺
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
                if r["temp_C"] > a.temp_max:
                    abort = f"{j} 溫度 {r['temp_C']:.1f}°C 超過 {a.temp_max}"
                if abort:
                    break

            roll, pitch = read_imu_rp()
            if not abort and max(abs(roll), abs(pitch)) > a.tilt_max:
                abort = f"機身傾角 roll {roll:+.1f}° pitch {pitch:+.1f}° 超過 ±{a.tilt_max}°"
            rec = {"t": round(t, 3), "phase": nm, "kp": round(kp_now, 1),
                   "roll": round(roll, 2), "pitch": round(pitch, 2), "j": tick}
            recent.append(rec)
            if len(recent) > 60:
                recent.pop(0)
            if nm.startswith(HOLDISH):
                samples.append({"phase": nm, "t": round(t, 3),
                                "roll": round(roll, 2), "pitch": round(pitch, 2),
                                **{j: tick[j] for j in LEGS12},
                                **{w: round(stt[wi]["effort"], 2)
                                   for w, wi in widx.items()}})
            if abort:
                break

            write_frame(des_now, kp_now, wlock)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

            if t - last >= 0.25:
                print(f"{t:6.2f} {nm:>14s} {kp_now:6.0f} {we[0]:10.4f} {wt[0]:8.2f}"
                      f" {wt[1]:>16s} {roll:+6.1f} {pitch:+6.1f}")
                last = t
            nxt += 1.0 / a.hz
            dly = nxt - time.monotonic()
            if dly > 0:
                time.sleep(dly)
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    except Exception as e:
        abort = f"未預期的例外：{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- 收尾
    held_des, held_kp = dict(des_now), (kp_now if abort else 0.0)
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
        hurt = abort.split()[0]
        if hurt in LEGS12 and recent:
            print(f"\n中止前最後 12 筆 —— {hurt}")
            print(f"  {'t':>7s} {'階段':>14s} {'q':>9s} {'des':>9s} {'τ':>8s}"
                  f" {'v':>7s} {'kp|e|+kd|v|':>11s}")
            for rr in recent[-12:]:
                q_, d_, tau_, v_ = rr["j"][hurt]
                cap = rr["kp"] * abs(q_ - d_) + a.kd * abs(v_)
                print(f"  {rr['t']:7.3f} {rr['phase']:>14s} {q_:9.4f} {d_:9.4f}"
                      f" {tau_:8.2f} {v_:7.3f} {cap:11.1f}")
    else:
        print("✅ 序列完整跑完")

    print(f"\n{'關節':16s} {'峰值τ':>9s} {'門檻':>7s} {'用掉':>7s}")
    for j in LEGS12:
        lim = TMAX[j[2:]]
        print(f"{j:16s} {peak[j]:+9.2f} {lim:7.0f} {100*abs(peak[j])/lim:6.0f}%")

    # ★ 迴圈健康度 —— 狗上 CPU 忙的時候，這是唯一會露餡的地方
    el = min(t, T_END) if n_tick else 0.0
    hz = n_tick / el if el > 0 else 0.0
    print(f"\n迴圈：{n_tick} 次 / {el:.2f}s = {hz:.0f} Hz（目標 {a.hz:.0f}）"
          f"　最長單次間隔 {worst_gap*1000:.0f} ms @ t={worst_gap_t:.2f}s"
          f"（警戒 {a.gap_max*1000:.0f}、controller 逾時 500）")
    if worst_gap > 0.5:
        print("  ❌ **有一次間隔超過 500 ms —— 那一刻指令區很可能被清成 0。**")
        print("     承重中發生就是短暫放手。這一趟的力矩資料要打折看。")
    elif worst_gap > a.gap_max:
        print("  ⚠️ 超過警戒線。狗上 CPU 是不是有別的東西在跑？先 uptime／top 看一下。")
    elif hz < 0.8 * a.hz:
        print("  ⚠️ 平均速率偏低 —— 雖然沒有長停頓，但 CPU 可能吃緊。")

    if abort:
        print("\n★ 現在腿還在承重。[Enter] 依原路徑坐回趴姿；[Ctrl-C] 立刻放手（會塌）")
        try:
            if sys.stdin.isatty():
                input("\n   > ")
            else:
                print("   非互動模式 → 直接執行坐回去")
            keeper.stop()
            cur = dict(held_des)
            for nm, dur, tgt in (("SIT_crouch", a.t2, dict(coord.POSES["crouch"])),
                                 ("SIT_LIE", a.t1, q_lie)):
                s = time.monotonic()
                while (e := time.monotonic() - s) < dur:
                    u = smoothstep(e / dur)
                    write_frame({j: cur[j] + u * (tgt[j] - cur[j]) for j in LEGS12},
                                held_kp)
                    shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
                    time.sleep(1.0 / a.hz)
                cur = dict(tgt)
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

    out = {"schema": "m8_swing/1", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "args": vars(a), "aborted": bool(abort), "abort_reason": abort or None,
           "q_lie": q_lie, "foot_ref": {k: list(v) for k, v in ref.items()},
           "knee_sign": {k: float(v) for k, v in ks.items()},
           "peak": peak, "recent": recent, "hold_samples": samples[:20000],
           "loop": {"ticks": n_tick, "hz": round(hz, 1),
                    "worst_gap_s": round(worst_gap, 4),
                    "worst_gap_t": round(worst_gap_t, 3)}}
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
        print(f"\n⏸ mc_ctrl 仍在凍結中（PID {pid}）。確認狗安全後：")
        print(f"      sudo kill -CONT {pid}")
    print(f"\n📄 {logp}")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
