#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib_capture.py —— 校正層第一步：唯讀擷取「當前姿勢」的實機關節角度。

用途：
  用遙控器（高階控制）把狗擺到某個標準姿勢（例如站好），再跑這支把 16 個關節的
  實機角度讀下來、平均、輸出成可直接貼進 L4_standup_shm.py 的「每腿 POSE」格式。

★ 完全唯讀（O_RDONLY / PROT_READ），不寫、不驅動馬達、不需停 mc_ctrl、不需 sudo。

重點：左右腿角度會相反（鏡像），所以輸出是【每條腿各自】的角度，不是 4 腿共用一組。

用法：
  python3 calib_capture.py --label stand            # 擷取當前姿勢，命名為 stand，印出並存檔
  python3 calib_capture.py --label lie --secs 2      # 平均 2 秒
"""

import argparse
import ctypes
import json
import os
import mmap
import time

# ---- 共享記憶體結構（與 lowlevel.h 一致，packed）----
class JointControl(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("p_des", ctypes.c_float), ("v_des", ctypes.c_float),
                ("kp", ctypes.c_float), ("kd", ctypes.c_float), ("t_ff", ctypes.c_float)]
class LegControl(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("abad", JointControl), ("hip", JointControl),
                ("knee", JointControl), ("foot", JointControl), ("flags", ctypes.c_int32)]
class SplineCmd(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("legs", LegControl * 4), ("consumer_flags", ctypes.c_uint32 * 2)]
class JointState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("flags", ctypes.c_int32), ("p", ctypes.c_float),
                ("v", ctypes.c_float), ("t", ctypes.c_float)]
class LegState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("abad", JointState), ("hip", JointState),
                ("knee", JointState), ("foot", JointState)]
class SplineState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("legs", LegState * 4), ("consumer_flags", ctypes.c_uint32 * 2)]
class SplineData(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("cmd", SplineCmd), ("state", SplineState)]

SHM_PATH = "/dev/shm/spline_shm"
SHM_SIZE = 1024 * 10
LEG_NAME = {0: "FR(右前)", 1: "FL(左前)", 2: "RR(右後)", 3: "RL(左後)"}
JOINTS = ("abad", "hip", "knee", "foot")


def main():
    ap = argparse.ArgumentParser(description="唯讀擷取實機關節角度（校正用）")
    ap.add_argument("--label", default="pose", help="這個姿勢的名字（如 stand / lie）")
    ap.add_argument("--secs", type=float, default=1.0, help="平均時間（秒）")
    ap.add_argument("--out", default=None, help="輸出檔（預設 calib_<label>.json 與 .py）")
    args = ap.parse_args()

    try:
        fd = os.open(SHM_PATH, os.O_RDONLY)                       # 唯讀
    except FileNotFoundError:
        print(f"✗ 找不到 {SHM_PATH}（運控沒起來？）"); return
    except PermissionError:
        print("✗ 沒有讀權限（理論上應是 -rw-r--r-- 全域可讀）"); return
    buf = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)

    print(f"[*] 擷取姿勢「{args.label}」，平均 {args.secs:.1f} 秒（唯讀）...")

    # 累加 16 關節的角度與速度
    sum_p = [[0.0]*4 for _ in range(4)]
    sum_v = [[0.0]*4 for _ in range(4)]
    n = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.secs:
        snap = SplineData.from_buffer_copy(buf)                   # 一致快照
        for i in range(4):
            s = snap.state.legs[i]
            for jn, jc in enumerate(JOINTS):
                js = getattr(s, jc)
                sum_p[i][jn] += js.p
                sum_v[i][jn] += js.v
        n += 1
        time.sleep(0.002)

    avg_p = [[sum_p[i][j]/n for j in range(4)] for i in range(4)]
    avg_v = [[sum_v[i][j]/n for j in range(4)] for i in range(4)]

    # 檢查狗有沒有在動（速度太大代表沒站穩，擷取不準）
    max_v = max(abs(avg_v[i][j]) for i in range(4) for j in range(4))
    print(f"[*] 取樣 {n} 次，最大平均關節速度 {max_v:.4f} rad/s "
          f"{'（穩定，擷取可信）' if max_v < 0.05 else '⚠️（狗在動，建議擺穩再擷取）'}\n")

    # 印出每腿角度
    print(f"{'腿':10s} {'abad':>10s} {'hip':>10s} {'knee':>10s} {'foot(輪)':>10s}")
    for i in range(4):
        a, h, k, f = avg_p[i]
        print(f"{LEG_NAME[i]:10s} {a:+10.4f} {h:+10.4f} {k:+10.4f} {f:+10.4f}")

    # 組出「每腿 POSE」——注意左右腿角度相反，所以每腿各自一組
    pose = {i: {"abad": round(avg_p[i][0], 4),
                "hip":  round(avg_p[i][1], 4),
                "knee": round(avg_p[i][2], 4)} for i in range(4)}

    # 可貼進 L4 的 Python 片段
    print(f"\n# ==== 貼進 L4_standup_shm.py 的每腿 POSE（label={args.label}）====")
    print(f"POSE_{args.label} = {{")
    for i in range(4):
        p = pose[i]
        print(f"    {i}: {{\"abad\": {p['abad']:+.4f}, \"hip\": {p['hip']:+.4f}, "
              f"\"knee\": {p['knee']:+.4f}}},  # leg{i} {LEG_NAME[i]}")
    print("}")

    # 存檔
    out_json = args.out or f"calib_{args.label}.json"
    with open(out_json, "w") as fp:
        json.dump({"label": args.label, "avg_pos": avg_p, "avg_vel": avg_v,
                   "pose_legs": pose, "samples": n, "secs": args.secs}, fp,
                  ensure_ascii=False, indent=2)
    print(f"\n[*] 已存 {out_json}（含完整 16 關節角度/速度）")


if __name__ == "__main__":
    main()
