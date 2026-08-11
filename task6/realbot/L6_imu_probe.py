#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L6_imu_probe.py —— 唯讀驗證 /dev/shm/imu_shm 的 IMU 資料是否可用於 sim2real。

推定結構（來自官方 header 與先前 shm_probe 觀察）：
    size_t ts;        # 8B  時戳
    float  acc[3];    # 12B 加速度 m/s^2
    float  gyro[3];   # 12B 角速度 rad/s
    float  q[4];      # 16B 四元數 (w,x,y,z)
  = 48 bytes，struct fmt "<Q3f3f4f"

本程式做四項驗證（不只是把數字印出來）：
  1) 時戳是否在跳 → 資料是活的，且量出實際更新率
  2) 四元數範數是否 ≈ 1 → 確認欄位順序與型別解讀正確（wxyz vs xyzw 都試）
  3) 加速度向量長度是否 ≈ 9.81 → 靜止時應該只有重力，可反推單位與方向
  4) 靜止時角速度是否 ≈ 0 → 量出 gyro 零偏（部署時要扣掉）

安全性：O_RDONLY + PROT_READ，不寫入、不需 sudo、不必停 mc_ctrl。

用法：
  python3 L6_imu_probe.py              # 取樣 3 秒做完整驗證
  python3 L6_imu_probe.py --secs 10    # 取樣更久（零偏估得更準）
  python3 L6_imu_probe.py --raw        # 額外印出前 64 bytes 的原始十六進位
"""

import argparse
import math
import mmap
import os
import struct
import time

SHM_PATH = "/dev/shm/imu_shm"
SHM_SIZE = 1024
FMT = "<Q3f3f4f"
REC = struct.calcsize(FMT)          # 48


def read_once(buf):
    ts, ax, ay, az, gx, gy, gz, q0, q1, q2, q3 = struct.unpack_from(FMT, buf, 0)
    return ts, (ax, ay, az), (gx, gy, gz), (q0, q1, q2, q3)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def main():
    ap = argparse.ArgumentParser(description="唯讀驗證 IMU 共享記憶體")
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--raw", action="store_true", help="印出原始位元組")
    args = ap.parse_args()

    print(f"[*] struct fmt {FMT} = {REC} bytes")
    fd = os.open(SHM_PATH, os.O_RDONLY)
    buf = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)

    if args.raw:
        print("[*] 前 64 bytes 原始資料：")
        raw = bytes(buf[:64])
        for off in range(0, 64, 16):
            hexs = " ".join(f"{b:02x}" for b in raw[off:off + 16])
            print(f"    {off:04d}  {hexs}")
        print()

    # --- 取樣 ---
    samples, seen_ts = [], []
    t0 = time.monotonic()
    last_ts = None
    while time.monotonic() - t0 < args.secs:
        rec = read_once(buf)
        if rec[0] != last_ts:
            samples.append(rec)
            seen_ts.append((time.monotonic() - t0, rec[0]))
            last_ts = rec[0]
        time.sleep(0.0005)

    if not samples:
        print("✗ 完全沒讀到資料")
        return

    print(f"[*] 取樣 {args.secs:.1f}s，收到 {len(samples)} 個「時戳不同」的樣本")

    # --- 1) 時戳是否在跳 + 更新率 ---
    print("\n=== 1) 時戳 / 更新率 ===")
    if len(seen_ts) < 2:
        print("  ✗ 時戳沒有變化 → 資料是死的（daemon 沒在寫？）")
    else:
        hz = (len(seen_ts) - 1) / (seen_ts[-1][0] - seen_ts[0][0])
        d_ts = [b[1] - a[1] for a, b in zip(seen_ts, seen_ts[1:])]
        d_ts = [d for d in d_ts if d > 0]
        print(f"  ✅ 時戳在跳，實測更新率 ≈ {hz:.1f} Hz")
        print(f"     時戳首/末 = {seen_ts[0][1]} → {seen_ts[-1][1]}")
        if d_ts:
            med = sorted(d_ts)[len(d_ts) // 2]
            print(f"     時戳增量中位數 = {med}"
                  f"（若 ≈ {1e9/hz:.0f} 則單位是 ns；≈ {1e6/hz:.0f} 則是 µs；≈ {1e3/hz:.0f} 則是 ms）")

    # --- 2) 四元數範數 ---
    print("\n=== 2) 四元數（判斷欄位順序）===")
    qs = [s[3] for s in samples]
    n_all = [norm(q) for q in qs]
    print(f"  四元數範數：min {min(n_all):.5f}  max {max(n_all):.5f}  平均 {sum(n_all)/len(n_all):.5f}")
    if abs(sum(n_all) / len(n_all) - 1.0) < 0.02:
        print("  ✅ 範數 ≈ 1 → 這 4 個 float 確實是單位四元數，結構解讀正確")
        q = qs[-1]
        # wxyz 假設下的 roll/pitch/yaw
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        print(f"     最新值 (假設 wxyz) = ({w:+.4f}, {x:+.4f}, {y:+.4f}, {z:+.4f})")
        print(f"     → roll {math.degrees(roll):+7.2f}°  pitch {math.degrees(pitch):+7.2f}°  "
              f"yaw {math.degrees(yaw):+7.2f}°")
        # xyzw 假設
        x2, y2, z2, w2 = q
        roll2 = math.atan2(2 * (w2 * x2 + y2 * z2), 1 - 2 * (x2 * x2 + y2 * y2))
        pitch2 = math.asin(max(-1, min(1, 2 * (w2 * y2 - z2 * x2))))
        print(f"     → 若改成 xyzw：roll {math.degrees(roll2):+7.2f}°  "
              f"pitch {math.degrees(pitch2):+7.2f}°")
        print("     （狗現在是平放/趴姿的話，roll 和 pitch 都該接近 0；哪個假設對就看哪個接近 0）")
    else:
        print("  ⚠️ 範數不等於 1 → 欄位順序或位移可能不對，需要重新對齊 struct")

    # --- 3) 加速度 ---
    print("\n=== 3) 加速度（靜止時應只有重力）===")
    accs = [s[1] for s in samples]
    mags = [norm(a) for a in accs]
    avg = tuple(sum(a[i] for a in accs) / len(accs) for i in range(3))
    print(f"  平均向量 = ({avg[0]:+.4f}, {avg[1]:+.4f}, {avg[2]:+.4f})")
    print(f"  向量長度：min {min(mags):.4f}  max {max(mags):.4f}  平均 {sum(mags)/len(mags):.4f}")
    m = sum(mags) / len(mags)
    if abs(m - 9.81) < 0.6:
        print("  ✅ 長度 ≈ 9.81 → 單位是 m/s²，且靜止時只受重力")
    elif abs(m - 1.0) < 0.08:
        print("  ✅ 長度 ≈ 1.0 → 單位是 g（重力加速度倍數），部署時要 ×9.81")
    else:
        print(f"  ⚠️ 長度 {m:.3f} 不符 9.81 也不符 1.0 → 單位待確認，或狗當下不是靜止")
    dom = max(range(3), key=lambda i: abs(avg[i]))
    print(f"  重力主要落在第 {dom} 軸（{'xyz'[dom]}），符號 {'+' if avg[dom] > 0 else '−'}"
          f" → 這決定了 MJCF 的重力軸對應")

    # --- 4) 角速度零偏 ---
    print("\n=== 4) 角速度（靜止時的零偏，部署要扣掉）===")
    gyros = [s[2] for s in samples]
    gavg = tuple(sum(g[i] for g in gyros) / len(gyros) for i in range(3))
    gmax = tuple(max(abs(g[i]) for g in gyros) for i in range(3))
    for i, ax in enumerate("xyz"):
        print(f"  gyro_{ax}: 平均 {gavg[i]:+.5f}  絕對值最大 {gmax[i]:.5f} rad/s")
    if max(abs(v) for v in gavg) < 0.05:
        print("  ✅ 零偏很小（<0.05 rad/s），靜止判定合理")
    else:
        print("  ⚠️ 零偏偏大 → 部署時務必做靜止校正並扣除")

    print("\n[*] 驗證結束（全程唯讀，狗未受影響）")


if __name__ == "__main__":
    main()
