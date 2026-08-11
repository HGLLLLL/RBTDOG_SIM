#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4_standup_shm.py —— 把官方點足範例 lowlevel_demo.py 改成「共享記憶體」寫法，讓輪足也能用。

移植來源：
  demo/zsl-1/python/examples/lowlevel_demo.py（點足 zsl-1，走 UDP 的 mc_sdk_zsl_1_py.LowLevel）
本檔改為：
  直接讀寫 /dev/shm/spline_shm（不透過 SDK、不透過 UDP），因此在【輪足】上也能用。

保留範例的三個範式：
  1) 讀「當前角度」當起點           → 避免 p_des 一開始就跟實際差很多造成力矩突跳
  2) 用 ratio 把 p_des 從起點內插到目標 → 平滑站姿
  3) 結束用 kp=0、kd 大的「卸力收尾」   → 軟軟停住

新增（來自我們 L1~L3 的實機經驗）：
  - 預檢 1：偵測 cmd 旗標仍在跳動（mc_ctrl 沒停）就中止，不寫
  - 預檢 2：16 顆馬達任一 ready=0 / 已失聯 / 有故障位就中止，不寫（2026-08-11 硬體故障後補上）
  - 每關節力矩/速度超限即中止並卸力
  - 輪子(foot) 全程零增益（站姿不控制輪子）
  - 預設 dry-run（不碰硬體，只印出動作計畫）；要真的驅動硬體必須加 --confirm

動作分兩階段：
  階段 1「接住」(CATCH_SEC)：p_des 固定在當前實際角度，kp/kd 由 0 平滑升到設定值。
      凍結 mc_ctrl 後腿會因重力垂下（模型估：hip −60°、knee +79° 直到頂住機構限位），
      先用漸入增益把腿接住，避免撞限位，也避免一上來就有力矩突跳。
  階段 2「站起」(RAMP_SEC)：在滿增益下把 p_des 從接住點內插到 POSE_STAND。

======================= ⚠️ 跑真機前必讀 =======================
(1) 這會驅動【腿關節】→ 狗會站起來、身體會動 → 【必須把狗吊起來】，不能平躺/墊肚子。
(2) POSE_STAND 已由 calib_capture.py 從【這台實機】的站姿擷取（每腿各自、輪足編碼器慣例）。
    但它只對「當時擺的那個站姿」有效；換了狗或重新校正就要重抓。真機前建議再擷取一次確認。
(3) 需要 root（spline_shm 是 root 擁有）：sudo。
(4) 需先凍結/停 mc_ctrl（本程式會預檢；建議用 SIGSTOP）。
==============================================================

用法：
  python3 L4_standup_shm.py                 # dry-run：離線印出動作計畫，不碰硬體（預設）
  sudo python3 L4_standup_shm.py --confirm   # 真的驅動硬體（前提：狗吊掛、POSE 已校正、mc_ctrl 已停）
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shm_common import (DT, LEGNAME, POSE_LABEL,
                        POSE_LIE, POSE_STAND, POSES, SHM_PATH,
                        check_guards, check_struct_size, open_shm,
                        passive_stop, preflight_mc_stopped,
                        preflight_motors_healthy, publish, read_leg_q,
                        report_legs, set_leg_position, zero_all)

STOP_KD = 3.0

# ============================================================
# 2) 參數（對應範例；⚠️ POSE 需校正、gains 可調）
# ============================================================
CATCH_SEC = 0.5            # 階段 1：增益漸入「接住」腿的時間
RAMP_SEC  = 2.0            # 階段 2：內插到站姿的時間（可用 --ramp 覆寫）
HOLD_SEC  = 3.0            # 階段 3：到位後維持站姿的時間（可用 --hold；0=不維持，負數=無限）

# ✅ 2026-08-11 改用【原廠實測值】：L0_cmd_probe 量到輪足 D1 EDU 原廠 mc_ctrl 站立時，
#    16 個腿關節恆定 kp=20.000 / kd=0.700（輪關節 kp=0/kd=0.1，本程式輪子直接零增益）。
#    先前的 kp=80/kd=1 是從【點足版】lowlevel_demo.py 借來的，對輪足沒有依據。
#    代價：重力造成的穩態下垂由 1.9° 變成 7.6°（MJCF 算得 knee 重力 2.67 N·m ÷ kp），
#    這是可預測的系統性偏差，不是校正誤差。要更緊的追蹤再往上調。
LEG_KP    = 20.0          # 原廠站立值
LEG_KD    = 0.7           # 原廠站立值

# 安全保護（真機模式才作用）
# 依 MJCF (task6/model/d1_edu_w) 計算：吊掛站姿下重力力矩最大僅 knee 2.67 / abad 2.46 N·m，
# 2 秒 ramp 的最大角速度約 0.63 rad/s。門檻取「實際需求的 3 倍上下」，出事才攔得住。
TORQUE_ABORT = 8.0        # 任一關節力矩超過 → 中止（重力只需 ~2.7，超過 8 代表卡住或打架）
VEL_ABORT    = 2.0        # 任一關節速度超過 rad/s → 中止（ramp 只需 ~0.63）


# ============================================================
# 4) 站姿主邏輯（移植 run()：讀起點 → 兩段內插）
# ============================================================
def interpolate(a, b, ratio):
    return a * (1.0 - ratio) + b * ratio

def run_standup(d, dry=True, active_legs=(0, 1, 2, 3), sequence=("stand",)):
    active_legs = tuple(sorted(active_legs))
    skipped = [i for i in range(4) if i not in active_legs]
    print(f"\n[*] 驅動的腿：{', '.join(f'leg{i}({LEGNAME[i]})' for i in active_legs)}")
    if skipped:
        print(f"[*] ⚠️ 跳過的腿：{', '.join(f'leg{i}({LEGNAME[i]})' for i in skipped)}"
              f" —— 全程零增益，完全不出力")

    # --- 起點 ---
    if dry:
        init = [(0.0, 0.0, 0.0)] * 4          # 離線用 0 當起點示意
        print("[dry-run] 假設起點角度全為 0（真機會讀 state.legs[*]）\n")
    else:
        # 預檢 1：mc_ctrl 必須已凍結
        ok, trans = preflight_mc_stopped(d)
        if not ok:
            print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。先 SIGSTOP mc_ctrl。")
            return False

        report_legs(d, active_legs)

        # 預檢 2：被驅動的腿其馬達必須健康（見 preflight_motors_healthy 的說明）
        ok, problems = preflight_motors_healthy(d, active_legs)
        if not ok:
            print(f"\n✗ 中止：被驅動的腿有 {len(problems)} 個馬達問題，拒絕寫入 ——")
            for p in problems:
                print(f"    • {p}")
            print("  在馬達故障狀態下寫入 = 繞過原廠唯一在動作的安全機制。")
            print("  若該腿本來就要跳過，請用 --skip-legs 排除它。")
            return False
        print(f"\n[*] 預檢通過：mc_ctrl 已凍結；被驅動的 {len(active_legs)} 條腿"
              f"（共 {len(active_legs)*4} 顆馬達）全部 ready 且無故障位。")

        init = [read_leg_q(d, i) for i in range(4)]
        print("[*] 讀到起點角度：")
        for i, (a, h, k) in enumerate(init):
            tag = "★" if i in active_legs else "跳過"
            print(f"    {tag} leg{i} {LEGNAME[i]}: abad={a:+.3f} hip={h:+.3f} knee={k:+.3f}")

    def drive(targets, kp, kd):
        """把「被驅動的腿」的目標寫進 cmd 並送出。回傳 (ok, 原因)。

        ⚠️ zero_all() 一定要在「腿的迴圈之外」——它會清掉全部四條腿，
           放進迴圈裡會把前一條腿剛設好的指令抹掉，最後只剩最後一條有指令。
           被跳過的腿因為 zero_all 已經歸零、之後不再碰，所以維持零增益。
        """
        zero_all(d)                                   # 先壓全零（輪子也歸零）
        for i in active_legs:
            a, h, k = targets[i]
            set_leg_position(d, i, a, h, k, kp, kd)
        publish(d)
        return check_guards(d, active_legs, TORQUE_ABORT, VEL_ABORT)

    def abort(why):
        print(f"⚠️ 保護觸發：{why} → 卸力中止")
        passive_stop(d, active_legs, 300, STOP_KD)

    # --- 階段 1：增益漸入「接住」腿（p_des 固定在當前角度，kp/kd 由 0 升上來）---
    n_catch = int(CATCH_SEC / DT)
    print(f"\n[*] 階段 1：接住（{CATCH_SEC}s，p_des 保持不動，kp 0→{LEG_KP}）")
    for step in range(n_catch + 1):
        ratio = min(step / n_catch, 1.0)
        kp, kd = LEG_KP * ratio, LEG_KD * ratio
        if not dry:
            ok, why = drive(init, kp, kd)
            if not ok:
                abort(why)
                return False
            time.sleep(DT)
        elif step % max(1, int(0.25 / DT)) == 0:
            print(f"  t={step*DT:4.2f}s  kp={kp:5.1f} kd={kd:4.2f}  "
                  f"leg0 目標 abad={init[0][0]:+.3f} hip={init[0][1]:+.3f} knee={init[0][2]:+.3f}（不動）")

    # --- 階段 2+：依序走過 --sequence 指定的每一個姿勢 ---
    # 每個姿勢都是「ramp 內插過去 → hold 維持」。ramp 的起點用【上一個姿勢的指令值】
    # 而不是重讀實際角度，否則重力造成的穩態下垂會在每段開頭變成一個階躍。
    cur = list(init)
    for si, pose_name in enumerate(sequence, start=2):
        pose = POSES[pose_name]
        label = POSE_LABEL[pose_name]
        goal = [(pose[i]["abad"], pose[i]["hip"], pose[i]["knee"]) for i in range(4)]

        # ---- ramp ----
        n = int(RAMP_SEC / DT)
        print(f"\n[*] 階段 {si}a：{label}（內插 {RAMP_SEC}s，kp={LEG_KP} kd={LEG_KD}）")
        for step in range(n + 1):
            ratio = min(step / n, 1.0)
            targets = [(interpolate(cur[i][0], goal[i][0], ratio),
                        interpolate(cur[i][1], goal[i][1], ratio),
                        interpolate(cur[i][2], goal[i][2], ratio)) for i in range(4)]
            if not dry:
                ok, why = drive(targets, LEG_KP, LEG_KD)
                if not ok:
                    abort(why)
                    return False
                time.sleep(DT)
            elif step % int(0.5 / DT) == 0:
                a, h, k = targets[0]
                print(f"  t={step*DT:4.1f}s ratio={ratio:.2f}  "
                      f"leg0 目標 abad={a:+.3f} hip={h:+.3f} knee={k:+.3f}")

        # ---- hold ----
        if HOLD_SEC != 0:
            n_hold = int(HOLD_SEC / DT) if HOLD_SEC > 0 else None
            hlabel = f"{HOLD_SEC}s" if HOLD_SEC > 0 else "無限（按 Ctrl+C 進入下一步）"
            print(f"\n[*] 階段 {si}b：維持{label}（{hlabel}）"
                  f"  每 0.5s 回報追蹤誤差（絕對值才可比，左右腿編碼器慣例鏡像）")
            if dry:
                print(f"  （dry-run：實機會回報 {len(active_legs)} 條腿的追蹤誤差）")
            else:
                step = 0
                try:
                    while n_hold is None or step <= n_hold:
                        ok, why = drive(goal, LEG_KP, LEG_KD)
                        if not ok:
                            abort(why)
                            return False
                        time.sleep(DT)
                        if step % int(0.5 / DT) == 0:
                            parts = []
                            for i in active_legs:
                                a, h, k = read_leg_q(d, i)
                                ta, th, tk = goal[i]
                                parts.append(f"{LEGNAME[i]}[{(a-ta)*57.3:+5.1f} "
                                             f"{(h-th)*57.3:+5.1f} {(k-tk)*57.3:+5.1f}]")
                            print(f"  t={step*DT:5.1f}s  誤差°(abad hip knee)  " + "  ".join(parts))
                        step += 1
                except KeyboardInterrupt:
                    if n_hold is None:
                        print("\n  （Ctrl+C：結束本段維持，繼續下一步）")
                    else:
                        raise

        cur = goal          # 下一段從這個姿勢接著走

    print(f"\n[*] 動作序列完成：{' → '.join(POSE_LABEL[x] for x in sequence)} → 洩力停止")
    if not dry:
        print("[*] 進行卸力收尾 ...")
        passive_stop(d, active_legs, 800, STOP_KD)
    return True


# ============================================================
# 5) 進入點
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="輪足 D1 EDU：SHM 版站姿（移植自點足 lowlevel_demo.py）")
    ap.add_argument("--confirm", action="store_true",
                    help="真的驅動硬體（否則只做 dry-run，不碰硬體）")
    ap.add_argument("--sequence", default="stand",
                    help="要依序走過的姿勢，逗號分隔。可用：lie（趴下）、stand（站立）。"
                         "例如 --sequence lie,stand 表示先趴下再站起來，最後一律洩力停止。")
    ap.add_argument("--ramp", type=float, default=RAMP_SEC,
                    help=f"階段 2 內插到站姿的秒數（預設 {RAMP_SEC}）。動作太快就調大。")
    ap.add_argument("--hold", type=float, default=HOLD_SEC,
                    help=f"階段 3 維持站姿的秒數（預設 {HOLD_SEC}）。"
                         "0 = 到位後立刻卸力；負數 = 一直維持到你按 Ctrl+C。")
    ap.add_argument("--skip-legs", default="",
                    help="不驅動的腿，逗號分隔的 index（0=FR 1=FL 2=RR 3=RL）。"
                         "例如 --skip-legs 2 表示右後腿全程零增益。"
                         "被跳過的腿不列入預檢與保護判準。")
    args = ap.parse_args()

    try:
        skip = {int(x) for x in args.skip_legs.split(",") if x.strip() != ""}
    except ValueError:
        print(f"✗ --skip-legs 格式錯誤：{args.skip_legs!r}（要像 2 或 0,2）")
        sys.exit(1)
    if not skip <= {0, 1, 2, 3}:
        print(f"✗ --skip-legs 只能是 0~3，收到 {sorted(skip)}")
        sys.exit(1)
    globals()["RAMP_SEC"] = args.ramp
    globals()["HOLD_SEC"] = args.hold
    sequence = tuple(x.strip() for x in args.sequence.split(",") if x.strip())
    bad = [x for x in sequence if x not in POSES]
    if bad or not sequence:
        print(f"✗ --sequence 不認得 {bad or '(空的)'}；可用：{', '.join(POSES)}")
        sys.exit(1)

    active_legs = tuple(i for i in range(4) if i not in skip)
    if not active_legs:
        print("✗ 四條腿都被跳過了，沒事可做。")
        sys.exit(1)

    check_struct_size()

    if not args.confirm:
        print("="*66)
        print("DRY-RUN 模式：不開啟、不寫入共享記憶體，只印出動作計畫。")
        print("要真的驅動硬體請加 --confirm（且需 sudo）。")
        print("="*66 + "\n")
        run_standup(None, dry=True, active_legs=active_legs, sequence=sequence)
        print("\n⚠️ 提醒：跑真機前必讀檔頭四點 —— 尤其【狗要吊掛】與【POSE 需校正】。")
        return

    # ---- 真機模式 ----
    print("="*66)
    print("⚠️ 真機模式：即將驅動【腿關節】。確認：狗已吊掛、POSE 已校正、mc_ctrl 已停。")
    print("="*66)
    if os.geteuid() != 0:
        print("✗ 需要 root：請用 sudo 執行。")
        sys.exit(1)
    try:
        d, buf = open_shm()
    except FileNotFoundError:
        print(f"✗ 找不到 {SHM_PATH}（機器人運控沒起來？）")
        sys.exit(1)
    except PermissionError:
        print("✗ 權限不足：請用 sudo。")
        sys.exit(1)

    print(f"[*] 已映射 {SHM_PATH}")
    try:
        run_standup(d, dry=False, active_legs=active_legs, sequence=sequence)
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C → 卸力收尾")
        passive_stop(d, active_legs, 800, STOP_KD)
    finally:
        zero_all(d); publish(d)
        print("[*] 已歸零收尾，watchdog 兜底。測完 SIGCONT 解凍 mc_ctrl 還原。")


if __name__ == "__main__":
    main()
