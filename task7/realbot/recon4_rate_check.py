#!/usr/bin/env python3
"""recon4_rate_check —— 多個 topic **同時**量頻率，並分辨「感測器真的慢」與「訊息掉了」。

為什麼需要這支（2026-09-22 的疑問 6）：
`ros2 topic hz` 量到前光達 IMU **206.5 Hz**、後光達 IMU **175.0 Hz**，差 15%。
但那兩個數字是**先後**各用 7 秒的窗口量的，而且 `topic hz` 只看訊息**到達**的時間 ——
到達變慢有三種完全不同的原因，報告要寫哪一個必須分清楚：

  1. 感測器／驅動真的送得少      → header 時戳的間隔也會變大
  2. 訊息在路上掉了（QoS、網路） → header 時戳間隔正常，但**缺號**（到達間隔出現整數倍跳躍）
  3. 訂閱端自己來不及收          → 兩邊都正常，但我們的 callback 有長時間空窗

光達設定是 `use_lidar_clock: true`，所以 header 時戳是**光達自己的時鐘**
→ 時戳間隔就是感測器真實的取樣週期，跟我們收得快不快無關。

★ 同時訂閱（而不是先後量）才能排除「兩次量測時機不同」這個混淆。

用法（在狗上，要先 source ROS 環境）：
    python3 recon4_rate_check.py <秒數> <topic> [<topic> ...]

從 PC 餵進去、狗上不留檔：
    ssh robot@192.168.168.100 "bash -c 'set +u
      . /opt/runtime/env.bash >/dev/null 2>&1
      exec python3 - 30 /front_lidar/imu /rear_lidar/imu'" < recon4_rate_check.py

輸出：stdout = JSON、stderr = 人讀表格。

⚠️ 不要拿這支量 PointCloud2：2 MB 的訊息每秒 10 筆，Python 反序列化會變成瓶頸，
   量到的就不是感測器的頻率而是我們自己的速度。點雲的頻率用 `ros2 topic hz` 就夠。
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time

WARMUP_S = 2.0          # 丟掉的前置時間：discovery 與第一批訊息的抖動不算
HEAVY = ("PointCloud2", "Image", "LaserScan")


def main():
    if len(sys.argv) < 3:
        print("用法: python3 recon4_rate_check.py <秒數> <topic> [<topic> ...]", file=sys.stderr)
        return 2
    secs = float(sys.argv[1])
    topics = sys.argv[2:]

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message
    except Exception as e:
        print(json.dumps({"error": "rclpy 匯入失敗: %r" % (e,)}, ensure_ascii=False))
        print("❌ 這台機器的 python3 匯不進 rclpy：%r" % (e,), file=sys.stderr)
        print("   有 source /opt/runtime/env.bash 嗎？沒有 rclpy 就改用 "
              "`ros2 topic hz -w 500 <topic>`（但那量不出掉包）。", file=sys.stderr)
        return 3

    rclpy.init(args=None)
    node = Node("recon4_rate_check")

    # ---- 解析型別（等 discovery，最多 8 秒）----
    types = {}
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        table = dict(node.get_topic_names_and_types())
        for t in topics:
            if t in table and table[t]:
                types.setdefault(t, table[t][0])
        if len(types) == len(topics):
            break
        rclpy.spin_once(node, timeout_sec=0.2)

    missing = [t for t in topics if t not in types]
    for t in missing:
        print("⚠️ %s：graph 裡看不到這個 topic（名稱錯？DOMAIN 錯？）" % t, file=sys.stderr)

    data = {t: {"arrive": [], "stamp_ns": []} for t in types}
    npub = {}
    collecting = {"on": False}

    def make_cb(t):
        def cb(msg):
            if not collecting["on"]:
                return
            d = data[t]
            d["arrive"].append(time.monotonic())
            hdr = getattr(msg, "header", None)
            if hdr is not None:
                d["stamp_ns"].append(hdr.stamp.sec * 1_000_000_000 + hdr.stamp.nanosec)
        return cb

    subs = []
    for t, ty in types.items():
        if any(h in ty for h in HEAVY):
            print("⚠️ %s 是 %s —— 重訊息，Python 收不快，量到的可能是我們自己的速度"
                  % (t, ty), file=sys.stderr)
        # 發布者是 BEST_EFFORT 而我們用 RELIABLE 訂閱 → QoS 不相容，一筆都收不到。
        # 所以照發布者的 reliability 來配。
        rel = QoSReliabilityPolicy.RELIABLE
        npub[t] = None
        try:
            infos = node.get_publishers_info_by_topic(t)
            npub[t] = len(infos)
            for info in infos:
                if info.qos_profile.reliability == QoSReliabilityPolicy.BEST_EFFORT:
                    rel = QoSReliabilityPolicy.BEST_EFFORT
                    break
        except Exception:
            pass
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=500, reliability=rel)
        try:
            subs.append(node.create_subscription(get_message(ty), t, make_cb(t), qos))
            print("→ 訂閱 %-34s %-38s %s" % (t, ty, rel.name), file=sys.stderr)
        except Exception as e:
            print("⚠️ %s 訂閱失敗（型別載不進來？）: %r" % (t, e), file=sys.stderr)

    if not subs:
        print(json.dumps({"error": "沒有任何 topic 訂閱成功"}, ensure_ascii=False))
        node.destroy_node(); rclpy.shutdown()
        return 4

    # ---- 暖機（discovery 與第一批抖動不算）----
    t_end = time.monotonic() + WARMUP_S
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.02)

    print("[recon4] 開始量 %.0f 秒（%d 個 topic 同時）…" % (secs, len(subs)), file=sys.stderr)
    collecting["on"] = True
    t0 = time.monotonic()
    t_end = t0 + secs
    spins = 0
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.02)
        spins += 1
    collecting["on"] = False
    elapsed = time.monotonic() - t0

    # ---- 統計 ----
    out = {"secs": round(elapsed, 2), "spins": spins, "topics": {}, "missing": missing}
    for t, d in data.items():
        a = d["arrive"]
        r = {"type": types[t], "n": len(a), "publishers": npub.get(t)}
        if len(a) >= 3:
            gaps = [b - x for x, b in zip(a, a[1:])]
            med = st.median(gaps)
            r.update({
                "rate_hz": round((len(a) - 1) / (a[-1] - a[0]), 3),
                "gap_ms": {"mean": round(1000 * st.fmean(gaps), 3),
                           "std": round(1000 * st.pstdev(gaps), 3),
                           "min": round(1000 * min(gaps), 3),
                           "max": round(1000 * max(gaps), 3),
                           "median": round(1000 * med, 3)},
                # 到達間隔 > 3 倍中位數 = 有一段空窗（掉包或卡頓）
                "big_gaps": sum(1 for g in gaps if g > 3 * med),
                "big_gap_worst_x": round(max(gaps) / med, 1) if med > 0 else None,
            })
        s = d["stamp_ns"]
        if len(s) >= 3 and s[-1] != s[0]:
            sg = [(b - x) / 1e6 for x, b in zip(s, s[1:])]   # ms
            smed = st.median(sg)
            r["stamp"] = {
                "rate_hz": round((len(s) - 1) / ((s[-1] - s[0]) / 1e9), 3),
                "dt_ms": {"mean": round(st.fmean(sg), 3), "std": round(st.pstdev(sg), 3),
                          "min": round(min(sg), 3), "max": round(max(sg), 3),
                          "median": round(smed, 3)},
                "big_gaps": sum(1 for g in sg if g > 3 * smed),
                # 時戳間隔是中位數的整數倍 → 中間那幾筆「本來有、但我們沒收到」
                "missing_est": int(round(sum((g / smed) - 1 for g in sg if g > 1.5 * smed)))
                if smed > 0 else None,
            }
        out["topics"][t] = r

    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()

    # ---- 人讀表 ----
    e = sys.stderr
    print("", file=e)
    print("---- %.0f 秒，%d 個 topic ----" % (elapsed, len(data)), file=e)
    print("%-32s %7s %9s %9s %9s %8s %8s"
          % ("topic", "筆數", "到達Hz", "時戳Hz", "時戳dt中位", "空窗數", "推估漏收"), file=e)
    for t, r in out["topics"].items():
        s = r.get("stamp") or {}
        print("%-32s %7d %9s %9s %9s %8s %8s"
              % (t, r["n"],
                 r.get("rate_hz", "—"),
                 s.get("rate_hz", "—"),
                 (s.get("dt_ms") or {}).get("median", "—"),
                 r.get("big_gaps", "—"),
                 s.get("missing_est", "—")), file=e)
    zero = [t for t, r in out["topics"].items() if r["n"] == 0]
    if zero:
        print("", file=e)
        for t in zero:
            print("⚠️ %s：%s 秒內 0 筆（發布者數 %s）。有發布者卻收不到 → 先用 "
                  "`ros2 topic hz %s` 對照，再看是不是 QoS 或該感測器真的沒資料。"
                  % (t, int(elapsed), out["topics"][t].get("publishers"), t), file=e)
    print("", file=e)
    print("怎麼讀：", file=e)
    print("  到達Hz 低、**時戳Hz 也低**            → 感測器／驅動真的送得少（是規格，寫進報告）", file=e)
    print("  到達Hz 低、**時戳Hz 正常**、推估漏收>0 → 訊息在路上掉了（不是感測器的規格）", file=e)
    print("  兩個都正常但空窗數多                   → 我們自己的 callback 卡住，換條件重量", file=e)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
