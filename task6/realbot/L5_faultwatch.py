#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L5_faultwatch.py —— 唯讀故障監看：從開機到「leg 掉線」全程記錄 16 顆馬達狀態。

用途：診斷 2026-08-11 發現的「開機後數分鐘 leg2(RR) 的 hip/knee/wheel 三顆馬達
      突然整組從 CAN 上消失 → 運控 latch motor_disenable → 閃紅燈 + 整隻軟掉」。
      重開機才會恢復，故推測與溫度或該分支接頭有關。此程式要回答：
        1) 掉線前 temp 欄位有沒有在爬（→ 溫度問題）
        2) 三顆是同時消失還是有先後（→ 定位 CAN 拓樸上的哪個接頭）
        3) 掉線前一刻有沒有出現過流/過溫等故障位

安全性：全程 O_RDONLY + PROT_READ，不寫入、不需 sudo、不必停 mc_ctrl。

用法（在狗上）：
  nohup python3 ~/L5_faultwatch.py --out ~/faultwatch.csv --hz 10 > ~/faultwatch.log 2>&1 &
  tail -f ~/faultwatch.log        # 看即時事件（掉線會印出來）
"""

import argparse
import ctypes
import mmap
import os
import time

SHM_PATH = "/dev/shm/spline_shm"
SHM_SIZE = 1024 * 10
STATE_OFFSET = 344                      # cmd 區 344 bytes，state 從這裡開始


class JointControl(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("p_des", ctypes.c_float), ("v_des", ctypes.c_float),
                ("kp", ctypes.c_float), ("kd", ctypes.c_float), ("t_ff", ctypes.c_float)]


class LegControl(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("abad", JointControl), ("hip", JointControl),
                ("knee", JointControl), ("foot", JointControl), ("flags", ctypes.c_int32)]


class JointState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("flags", ctypes.c_int32), ("p", ctypes.c_float),
                ("v", ctypes.c_float), ("t", ctypes.c_float)]


class LegState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("abad", JointState), ("hip", JointState),
                ("knee", JointState), ("foot", JointState)]


JOINTS = ("abad", "hip", "knee", "foot")
LEGNAME = {0: "FR", 1: "FL", 2: "RR", 3: "RL"}   # leg0=FR 已實測確認；其餘為推論
FAULT_BITS = ((1, "過壓"), (2, "過流"), (3, "過溫"), (4, "超速"), (5, "編碼器"))


def decode(flags):
    """回傳 (ready, temp_raw, volt_raw, 故障字串)。
    temp/volt 的絕對值不可信（實測室溫下讀到 5C / -10C，公式對不上），
    但【相對變化】仍然有診斷價值，所以原樣記錄 raw 欄位。"""
    ready = flags & 1
    temp = (flags >> 8) & 0xFF
    volt = (flags >> 16) & 0xFF
    faults = "|".join(n for b, n in FAULT_BITS if (flags >> b) & 1)
    return ready, temp, volt, faults


def uptime_sec():
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except OSError:
        return float("nan")


def snapshot(buf):
    """印出 16 顆馬達的當下狀態表（--once 模式）。"""
    legs_s = (LegState * 4).from_buffer_copy(buf, STATE_OFFSET)
    legs_c = (LegControl * 4).from_buffer_copy(buf, 0)
    up = uptime_sec()

    print(f"開機後 {up:.0f}s（{up/60:.1f} 分）    時間 {time.strftime('%H:%M:%S')}")
    print()
    print(f"{'關節':<10}{'ready':>6}{'溫度':>7}{'電壓':>7}{'位置':>10}{'速度':>9}"
          f"{'力矩':>9}{'指令kp':>8}{'指令kd':>8}  故障")
    print("-" * 88)

    dead, alive = [], 0
    for i in range(4):
        for jn in JOINTS:
            js = getattr(legs_s[i], jn)
            jc = getattr(legs_c[i], jn)
            ready, temp, volt, faults = decode(js.flags)
            tag = f"{LEGNAME[i]}.{jn}"
            if ready:
                alive += 1
            else:
                dead.append(tag)
            flag = "✅" if ready else "❌"
            print(f"{tag:<10}{flag:>6}{temp:>6}C{volt:>6}V{js.p:>10.4f}{js.v:>9.4f}"
                  f"{js.t:>9.3f}{jc.kp:>8.2f}{jc.kd:>8.2f}  {faults or '-'}")
        print()

    print(f"存活 {alive}/16" + ("   ⚠️ 掉線: " + ", ".join(dead) if dead else "   ✅ 全部正常"))
    kps = [getattr(getattr(legs_c[i], jn), "kp") for i in range(4) for jn in JOINTS]
    print("原廠控制狀態：" + ("passive（所有指令增益為 0）" if not any(kps) else "正在控制（有非零 kp）"))
    print()
    print("註：溫度是原始值直接為攝氏度（L0_shm_probe.cpp 的 -40 偏移是 bug）。"
          "腿關節 42–47°C、輪子 29–31°C 為正常。")


def main():
    ap = argparse.ArgumentParser(description="唯讀監看 16 顆馬達，記錄掉線事件")
    ap.add_argument("--once", action="store_true",
                    help="只印一次當下 16 顆馬達的狀態表就結束（不寫 CSV）")
    ap.add_argument("--out", default=os.path.expanduser("~/faultwatch.csv"))
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--secs", type=float, default=0.0, help="0 = 一直錄到被中止")
    args = ap.parse_args()

    fd = os.open(SHM_PATH, os.O_RDONLY)                       # 唯讀
    buf = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)

    if args.once:
        snapshot(buf)
        return

    cols = ["wall", "uptime"]
    for i in range(4):
        for jn in JOINTS:
            tag = f"{LEGNAME[i]}_{jn}"
            cols += [f"{tag}_ready", f"{tag}_temp", f"{tag}_volt",
                     f"{tag}_fault", f"{tag}_p", f"{tag}_v", f"{tag}_t", f"{tag}_cmdkp"]

    dt = 1.0 / args.hz
    t0 = time.monotonic()
    prev_ready = None
    n = 0

    print(f"[*] 開始錄製 → {args.out}   {args.hz}Hz   開機已 {uptime_sec():.0f}s", flush=True)
    print("[*] 唯讀模式，不影響狗的運行。掉線事件會即時印在下面。", flush=True)

    with open(args.out, "w", buffering=1) as fo:
        fo.write(",".join(cols) + "\n")
        while True:
            legs_s = (LegState * 4).from_buffer_copy(buf, STATE_OFFSET)
            legs_c = (LegControl * 4).from_buffer_copy(buf, 0)
            up = uptime_sec()
            row = [f"{time.time():.3f}", f"{up:.1f}"]
            ready_now = {}
            for i in range(4):
                for jn in JOINTS:
                    js = getattr(legs_s[i], jn)
                    jc = getattr(legs_c[i], jn)
                    r, tp, vt, fl = decode(js.flags)
                    ready_now[(i, jn)] = r
                    row += [str(r), str(tp), str(vt), fl,
                            f"{js.p:.5f}", f"{js.v:.5f}", f"{js.t:.4f}", f"{jc.kp:.2f}"]
            fo.write(",".join(row) + "\n")

            # --- 即時偵測 ready 位變化（掉線 / 恢復）---
            if prev_ready is not None:
                for k, v in ready_now.items():
                    if v != prev_ready[k]:
                        i, jn = k
                        what = "恢復 ✅" if v else "掉線 ❌"
                        js = getattr(legs_s[i], jn)
                        r, tp, vt, fl = decode(js.flags)
                        print(f"[!] {time.strftime('%H:%M:%S')} 開機後 {up:7.1f}s  "
                              f"{LEGNAME[i]}.{jn} {what}  temp_raw={tp} volt_raw={vt} "
                              f"fault={fl or '-'}", flush=True)
            prev_ready = ready_now

            n += 1
            if n % (int(args.hz) * 60) == 0:
                alive = sum(ready_now.values())
                print(f"[.] 開機後 {up:7.1f}s  存活馬達 {alive}/16  已記錄 {n} 筆", flush=True)

            if args.secs and time.monotonic() - t0 > args.secs:
                break
            time.sleep(dt)

    print(f"[*] 結束，共 {n} 筆 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
