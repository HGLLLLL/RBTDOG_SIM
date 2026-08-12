#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L8_sign_probe.py —— 用手扳腿，量出 calib_map 每個關節的正負號。【全程唯讀】

為什麼需要這支：
  `calib_map.CALIB` 的 sign 描述「編碼器座標 ↔ 物理現實」的關係。這件事
  **讀編碼器讀不出來** —— 你下指令、馬達跟著走、編碼器回報跟著變，永遠自洽，
  不管 sign 對錯。所以 L7 的 --mode jog 收尾那行「指令方向 vs 實測方向 ✓ 一致」
  是套套邏輯，對驗號沒有價值（2026-08-12 實機發現）。

  唯一的一手證據是「物理上把腿往某個方向扳，編碼器數字往哪走」。這支就是
  把那件事做成有紀錄、不靠目視判讀的量測。

  ⚠️ 2026-08-12 實機用這個方法發現 leg0(FR) 的 hip sign 是錯的：手往後擺
     hip 從 −2.6 變大到 −2.0，代表物理後擺對應 SHM 遞增，而 MJCF 的 +hip
     正是後擺 → sign 應為 +1，但檔案裡寫 −1。

安全性：
  用 O_RDONLY + PROT_READ 開共享記憶體，**物理上不可能寫入**。
  不碰 mc_ctrl、不下任何馬達指令。馬達應該是卸力狀態（手扳得動）。

用法（在狗上）：
  sudo python3 L8_sign_probe.py --leg 0
  sudo python3 L8_sign_probe.py --leg 0 --joint hip     # 只量單軸

MJCF 正向的物理意義（由 MuJoCo forward kinematics 實測，+x=前 +y=左）：
  +abad → 足端往【左】移（四條腿都一樣，不是各自外張！
          對右腿 FR/RR 來說往左是【內收】）
  +hip  → 足端往【後】移（22 mm）
  +knee → 足端往【下】伸（腿打直）
"""
import argparse
import ctypes
import mmap
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib_map
import shm_common as SC

JN = ("abad", "hip", "knee")
HOME_MJCF = {"abad": 0.0, "hip": 1.05, "knee": -2.00}   # d1_model.HOME3

# 每個關節「MJCF 正向」對應的物理動作，給操作者照著扳。
# 措辭刻意不用「外張」——那對右腿是錯的，見檔頭。
MOVE = {
    "abad": "把整條腿往【狗的左邊】平移（腿往左擺，不是往外）",
    "hip":  "把【大腿】往【後】擺（膝關節往尾巴方向移動）",
    "knee": "把【膝蓋打直】（小腿往下伸展）",
}
MIN_DELTA = 0.05        # rad。低於此視為沒扳動，不做判定


def open_readonly():
    """唯讀映射共享記憶體。O_RDONLY + PROT_READ = 寫不進去。"""
    fd = os.open(SC.SHM_PATH, os.O_RDONLY)
    buf = mmap.mmap(fd, SC.SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)
    return buf


def read_leg(buf, leg):
    s = SC.SplineData.from_buffer_copy(buf)
    j = s.state.legs[leg]
    return {"abad": j.abad.p, "hip": j.hip.p, "knee": j.knee.p}


def live(buf, leg, joint, secs=0.0):
    """即時顯示，按 Enter 結束並回傳當下讀值。"""
    import select
    print(f"    現在值：", end="", flush=True)
    while True:
        v = read_leg(buf, leg)[joint]
        print(f"\r    現在值：{v:+.4f} rad   （扳到底後按 Enter）      ", end="", flush=True)
        if select.select([sys.stdin], [], [], 0.15)[0]:
            sys.stdin.readline()
            print()
            return read_leg(buf, leg)[joint]


def probe(buf, leg, joint):
    print(f"\n── leg{leg}({SC.LEGNAME[leg]}).{joint} " + "─" * 40)
    print(f"  [1] 先把腿放到大致中間的位置，別靠在限位上。")
    start = live(buf, leg, joint)
    print(f"  [2] {MOVE[joint]}")
    end = live(buf, leg, joint)

    delta = end - start
    print(f"  位移 {start:+.4f} → {end:+.4f}  (Δ = {delta:+.4f} rad)")
    if abs(delta) < MIN_DELTA:
        print(f"  ⚠️ 位移太小（< {MIN_DELTA}），無法判定。再扳大一點重來。")
        return None

    measured = 1 if delta > 0 else -1
    cur_sign, cur_off = calib_map.CALIB[leg][joint]
    # 由站姿反推該 sign 下的 offset：shm_stand = sign*home + offset
    new_off = SC.POSE_STAND[leg][joint] - measured * HOME_MJCF[joint]

    print(f"  → 物理正向對應 SHM {'遞增' if delta > 0 else '遞減'}，"
          f"量到的 sign = {measured:+d}")
    if measured == cur_sign:
        print(f"  ✅ 與 calib_map 一致（{cur_sign:+d}, offset {cur_off:+.4f}）")
    else:
        print(f"  ❌ 與 calib_map 不符！檔案是 {cur_sign:+d}，量到 {measured:+d}")
        print(f"     若要更正：CALIB[{leg}]['{joint}'] = ({measured:+d}, {new_off:+.4f})")
        print(f"     （offset 由站姿 {SC.POSE_STAND[leg][joint]:+.4f} 反推，"
              f"讓 MJCF home {HOME_MJCF[joint]:+.2f} 仍對回站姿）")
    return measured, new_off


# 機構限位（MJCF joint range，來自 d1_edu_w.xml）。兩端底校正的對應基準。
# 不用 ctrlrange——那比 joint range 窄 0.02 rad，是留給致動器的餘裕，不是實體停點。
JOINT_RANGE = {"abad": (-0.4887, +0.4887),
               "hip":  (-1.1520, +2.9670),
               "knee": (-2.7230, -0.6020)}

# 各關節「往 MJCF 負向 / 正向」的物理動作，給操作者推到底
STOP_MOVE = {
    "abad": ("把整條腿往【狗的右邊】推到底", "把整條腿往【狗的左邊】推到底"),
    "hip":  ("把【大腿】往【前】擺到底", "把【大腿】往【後】擺到底"),
    "knee": ("把【膝蓋彎】到底", "把【膝蓋打直】到底"),
}


def probe_stops(buf, leg, joint):
    """量兩端機構限位，一次定出 sign、offset、刻度。不依賴任何姿態假設。

    這是取代「由站姿反推 offset」的正解。舊作法假設「實機站姿 ≈ MJCF home」，
    2026-08-12 實機證明該假設不成立（knee 差約 0.6 rad / 34°）。
    """
    lo_mjcf, hi_mjcf = JOINT_RANGE[joint]
    neg_move, pos_move = STOP_MOVE[joint]
    print(f"\n── leg{leg}({SC.LEGNAME[leg]}).{joint} 兩端限位 " + "─" * 28)
    print(f"  機構行程 {hi_mjcf - lo_mjcf:.4f} rad = {(hi_mjcf - lo_mjcf) * 57.3:.1f}°")
    print(f"  ⚠️ 輕輕推到【感覺頂住】就好，不要硬壓。")

    print(f"  [1] {neg_move}")
    a = live(buf, leg, joint)
    print(f"  [2] {pos_move}")
    b = live(buf, leg, joint)

    span_shm = abs(b - a)
    span_mjcf = hi_mjcf - lo_mjcf
    if span_shm < 0.5 * span_mjcf:
        print(f"  ⚠️ 量到的跨距 {span_shm:.3f} rad 遠小於機構行程 {span_mjcf:.3f}，"
              f"應該沒推到底。重來。")
        return None

    scale = span_shm / span_mjcf
    sign = 1 if b > a else -1          # MJCF 正向端對應 SHM 較大 → sign=+1
    shm_at_lo, shm_at_hi = (a, b) if sign > 0 else (b, a)
    # shm = sign*mjcf + offset → offset 由兩端各算一次取平均
    o1 = shm_at_lo - sign * lo_mjcf
    o2 = shm_at_hi - sign * hi_mjcf
    offset = 0.5 * (o1 + o2)

    cur_sign, cur_off = calib_map.CALIB[leg][joint]
    print(f"  SHM 兩端 {a:+.4f} / {b:+.4f}   跨距 {span_shm:.4f} rad")
    print(f"  刻度比 {scale:.3f}（1.000 表示編碼器與關節角 1:1，偏離太多代表有減速比）")
    print(f"  兩端各自反推的 offset：{o1:+.4f} / {o2:+.4f}（差 {abs(o1 - o2):.4f}）")
    print(f"  → 量到 ({sign:+d}, {offset:+.4f})   檔案是 ({cur_sign:+d}, {cur_off:+.4f})")
    if abs(o1 - o2) > 0.1:
        print(f"  ⚠️ 兩端反推的 offset 差太多，可能有一端沒推到底、或刻度不是 1:1")
    same = sign == cur_sign and abs(offset - cur_off) < 0.05
    print("  ✅ 與檔案一致" if same else "  ❌ 與檔案不符")
    return sign, offset, scale


def main():
    ap = argparse.ArgumentParser(description="用手扳腿量 calib_map 的 sign/offset（唯讀）")
    ap.add_argument("--leg", type=int, required=True, help="SHM 腿序 0=FR 1=FL 2=RR 3=RL")
    ap.add_argument("--joint", choices=JN, default=None, help="只量單一關節（預設三軸都量）")
    ap.add_argument("--stops", action="store_true",
                    help="量兩端機構限位，一次定出 sign+offset+刻度（推薦；"
                         "取代由站姿反推 offset 的舊作法）")
    args = ap.parse_args()
    if args.leg not in (0, 1, 2, 3):
        print("✗ --leg 只能是 0~3")
        sys.exit(1)

    print("=" * 62)
    print("L8 正負號量測 —— 全程唯讀，不下任何馬達指令")
    print("前提：馬達已卸力（手扳得動）、狗吊掛四腳離地")
    print("=" * 62)
    print(f"目標：leg{args.leg}({SC.LEGNAME[args.leg]})")

    buf = open_readonly()
    joints = [args.joint] if args.joint else list(JN)

    if args.stops:
        out = {}
        for jn in joints:
            r = probe_stops(buf, args.leg, jn)
            if r:
                out[jn] = r
        print("\n" + "=" * 62)
        print(f"leg{args.leg}({SC.LEGNAME[args.leg]}) 兩端限位校正結果")
        print("直接貼進 calib_map.CALIB 的形式：")
        print(f"    {args.leg}: {{", end="")
        parts = []
        for jn in JN:
            if jn in out:
                s, o, _ = out[jn]
                parts.append(f'"{jn}": ({s:+d}, {o:+.4f})')
            else:
                s, o = calib_map.CALIB[args.leg][jn]
                parts.append(f'"{jn}": ({s:+d}, {o:+.4f})  # 未量，沿用舊值')
        print(", ".join(parts) + "},")
        print("\n⚠️ 改了 calib_map 之後：重跑 gait_export.py --export → "
              "deploy_to_dog.sh → 重跑量測確認")
        return

    results = {}
    for jn in joints:
        r = probe(buf, args.leg, jn)
        if r:
            results[jn] = r

    print("\n" + "=" * 62)
    print(f"leg{args.leg}({SC.LEGNAME[args.leg]}) 量測摘要")
    bad = []
    for jn in joints:
        cur = calib_map.CALIB[args.leg][jn]
        if jn not in results:
            print(f"  {jn:5s} 未判定")
            continue
        sign, off = results[jn]
        ok = sign == cur[0]
        print(f"  {jn:5s} 量到 {sign:+d} / 檔案 {cur[0]:+d}  {'✅' if ok else '❌ 應改為 ' + f'({sign:+d}, {off:+.4f})'}")
        if not ok:
            bad.append((jn, sign, off))
    if bad:
        print(f"\n⚠️ 有 {len(bad)} 個關節的 sign 不符。改 calib_map 之後【必須】：")
        print("   1) 在開發機重跑 gait_export.py --export（calib_hash 會變）")
        print("   2) 重新 deploy_to_dog.sh（狗上舊的 npz 會被 L7 拒絕，這是刻意的）")
        print("   3) 重跑 G1 確認")
    else:
        print("\n✅ 全部一致。")


if __name__ == "__main__":
    main()
