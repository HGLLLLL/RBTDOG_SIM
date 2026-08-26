#!/usr/bin/env python3
"""M6 —— 承重狀態的唯讀擷取（**零風險**：不寫入、不凍結行程、不需 sudo）。

要回答的問題：**狗站在地上、承受全部體重時，每個關節實際在做什麼？**

到 2026-08-26 為止的所有實機測試都是**吊掛**的 —— 關節只撐自己的腿重。
模擬算出站立承重時膝關節要 **14.9 N·m**（吊掛時只要 2.9），是 5 倍。
在我們自己去撐那個載荷之前，先把實機的真實數字讀回來。

★ 關鍵：`joint_cmd` 是**可讀**的。狗在原廠控制下站著時，我們可以直接讀到

    原廠用的 kp / kd / 目標角  ＋  實際產生的力矩 ＋ 實際的關節角

也就是一份完整的「我們要複製什麼」的規格 —— 而且完全不用碰它。

用法（在狗上；遙控器讓狗站好、停穩再跑）：
    python3 M6_load_probe.py stand_ground --secs 5
    python3 M6_load_probe.py hang_free  --secs 5 --note "吊掛、洩力"

⚠️ 這支**只讀不寫**，可以在任何狀態下安全執行，包括狗站著、走著、故障中。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import coord
import shm_io

REF_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hang_torque_ref.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "reference", "hang_torque_ref.json"),
)

LEG_JOINTS = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]


def capture(secs: float, hz: float) -> dict:
    """同時擷取 joint_state 與 joint_cmd。

    ⚠️ 兩塊 shm 要在**同一輪**讀完再進下一輪 —— 分兩趟讀會讓
    「力矩」與「當時的指令」對不上，那種錯位在靜態下看不出來，
    一旦狗在動就會給出完全錯誤的因果推論。
    """
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)

    acc = {n: {"q": [], "tau": [], "v": [], "kp": [], "kd": [], "des": [], "temp": []}
           for n in shm_io.JOINTS}
    imu_acc = []
    period, t0 = 1.0 / hz, time.monotonic()
    tick0 = None
    with shm_io.Shm("joint_state") as ss, shm_io.Shm("joint_cmd") as sc, \
            shm_io.Shm("imu_central") as si:
        tick0 = ss.read_tick(shm_io.STATE_STRIDE)
        while time.monotonic() - t0 < secs:
            st = ss.states()
            cm = sc.read_records(shm_io.CMD_STRIDE, 5)
            for i, nm in enumerate(shm_io.JOINTS):
                a = acc[nm]
                a["q"].append(st[i]["position"])
                a["tau"].append(st[i]["effort"])
                a["v"].append(st[i]["velocity"])
                a["temp"].append(st[i]["temp_C"])
                a["des"].append(cm[i][0])      # position
                a["kp"].append(cm[i][3])       # kp
                a["kd"].append(cm[i][4])       # kd
            imu_acc.append([shm_io._F8.unpack_from(si.mm, 824 + 8 * k)[0]
                            for k in range(10)])
            time.sleep(period)
        tick1 = ss.read_tick(shm_io.STATE_STRIDE)
    dt = time.monotonic() - t0
    return {"acc": acc, "imu": imu_acc, "secs": dt, "n": len(imu_acc),
            "tick_rate": (tick1 - tick0) / dt if dt > 0 else 0.0}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="這次擷取叫什麼，例如 stand_ground / hang_free")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--note", default="", help="自由文字：狗當下的狀態、遙控器按了什麼")
    a = ap.parse_args()

    logp = shm_io.start_log("M6")
    print("M6 —— 承重狀態唯讀擷取（不寫入、不凍結、不需 sudo）\n")
    print(f"標籤 {a.label}　{a.secs:.1f} 秒 @ {a.hz:.0f} Hz")
    if a.note:
        print(f"備註：{a.note}")
    print("⚠️ 擷取期間請不要碰狗、不要動遙控器。\n")

    r = capture(a.secs, a.hz)
    acc = r["acc"]
    print(f"取樣 {r['n']} 筆 / {r['secs']:.2f} 秒　"
          f"joint_state 心跳 {r['tick_rate']:.0f}/s（實機應接近 1000）\n")

    # ---- 原廠指令：這是「我們要複製什麼」的規格
    print("=" * 78)
    print("原廠當下在下什麼指令（讀 joint_cmd）")
    print("=" * 78)
    live = [n for n in shm_io.JOINTS
            if abs(mean(acc[n]["kp"])) + abs(mean(acc[n]["kd"])) > 1e-9]
    if not live:
        print("16 顆全部零增益 —— 狗是洩力的，不是在承重。")
        print("（若你預期它站著，代表遙控器沒有真的把運控打開。）")
    else:
        print(f"{'關節':16s} {'kp':>7s} {'kd':>7s} {'目標(馬達)':>11s} {'目標(控制器)':>12s}")
        for n in shm_io.JOINTS:
            kp, kd, des = mean(acc[n]["kp"]), mean(acc[n]["kd"]), mean(acc[n]["des"])
            if abs(kp) + abs(kd) < 1e-9:
                continue
            ctrl = coord.to_ctrl(n, des) if not n.endswith(coord.KIND_WHEEL) else float("nan")
            print(f"{n:16s} {kp:7.1f} {kd:7.2f} {des:11.4f} {ctrl:12.4f}")

    # ---- 實際狀態：角度與力矩（換算到控制器座標系）
    print("\n" + "=" * 78)
    print("實際狀態（角度／速度／力矩已換算到**控制器座標系**，與 MJCF 同框）")
    print("=" * 78)
    print(f"{'關節':16s} {'實測角':>9s} {'目標角':>9s} {'誤差':>9s} {'誤差°':>7s}"
          f" {'力矩':>8s} {'kp·誤差':>8s} {'溫度':>6s} {'角度σ':>8s}")
    rows = {}
    for n in LEG_JOINTS:
        sgn = coord.SIGN[n[2:]][n[:2]]
        q = coord.to_ctrl(n, mean(acc[n]["q"]))
        des = coord.to_ctrl(n, mean(acc[n]["des"]))
        tau = sgn * mean(acc[n]["tau"])
        kp = mean(acc[n]["kp"])
        err = q - des
        sq = std([coord.to_ctrl(n, x) for x in acc[n]["q"]])
        rows[n] = {"q": q, "des": des, "err": err, "tau": tau, "kp": kp,
                   "kd": mean(acc[n]["kd"]), "temp": mean(acc[n]["temp"]),
                   "q_std": sq, "tau_std": std(acc[n]["tau"])}
        print(f"{n:16s} {q:9.4f} {des:9.4f} {err:+9.4f} {math.degrees(err):+7.2f}"
              f" {tau:+8.2f} {-kp * err:+8.2f} {mean(acc[n]['temp']):6.1f} {sq:8.5f}")

    # ★ 交叉檢核：力矩應該 ≈ −kp·誤差（原廠也是 PD）
    #   兩者對不上，代表 (a) 原廠還有前饋項，或 (b) 我們讀錯欄位。
    #   多印一個可以互相對照的量，比多印一個結論有用。
    bad = [n for n, v in rows.items()
           if abs(v["kp"]) > 1e-9 and abs(v["tau"] + v["kp"] * v["err"]) >
           max(1.5, 0.4 * abs(v["tau"]))]
    print(f"\n★ 交叉檢核：實測力矩 vs −kp·誤差")
    if not any(abs(v["kp"]) > 1e-9 for v in rows.values()):
        print("   （零增益，不適用）")
    elif bad:
        print(f"   ⚠️ {len(bad)} 個關節對不上：{', '.join(bad)}")
        print("   → 可能原廠除了 PD 還有**前饋力矩**（joint_cmd 的 effort 欄），")
        print("     也可能是我們讀錯欄位。看 effort 欄的值再判。")
        ff = [n for n in bad if abs(mean(acc[n]["q"])) >= 0]   # 佔位，下面統一印
        print(f"   joint_cmd 的 effort 欄（前饋）：")
        with shm_io.Shm("joint_cmd") as sc:
            cm = sc.read_records(shm_io.CMD_STRIDE, 5)
        for n in bad:
            i = shm_io.idx_of(n)
            print(f"     {n:16s} effort = {cm[i][2]:+.3f}")
    else:
        print("   ✅ 全部相符 → 原廠就是純 PD，沒有額外前饋。我們的控制律形式一樣。")

    # ---- 與吊掛預演對照（如果拿得到）
    ref = {}
    for p in REF_PATHS:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    ref = json.load(f)
                break
            except Exception:
                pass
    tau_max = max(abs(v["tau"]) for v in rows.values())
    print(f"\n最大 |力矩| {tau_max:.2f} N·m")
    print("  對照：吊掛 STAND 姿勢預演 6.09（ABAD）／2.9（KNEE）")
    print("        站立承重的模擬預測 KNEE **14.9**、HIP 6.2、ABAD 4.3")
    print("  → 若這次是站在地上，膝的力矩應該落在 14~17 而不是 3")

    # ---- IMU
    im = [mean([x[k] for x in r["imu"]]) for k in range(10)]
    ax, ay, az = im[0:3]
    qx, qy, qz, qw = im[6:10]
    roll = math.degrees(math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx)))))
    print(f"\nIMU（xyzw）roll {roll:+.2f}°  pitch {pitch:+.2f}°"
          f"　|a| = {math.sqrt(ax*ax+ay*ay+az*az):.3f} m/s²")

    # ---- 存 JSON
    out = {"schema": "m6_load/1", "label": a.label, "note": a.note,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n": r["n"], "secs": r["secs"], "tick_rate": r["tick_rate"],
           "joints": rows,
           "wheels": {w: {"q": mean(acc[w]["q"]), "tau": mean(acc[w]["tau"]),
                          "v": mean(acc[w]["v"]), "kp": mean(acc[w]["kp"]),
                          "kd": mean(acc[w]["kd"])} for w in shm_io.WHEELS},
           "imu": {"roll_deg": roll, "pitch_deg": pitch, "raw": im}}
    jp = (logp[:-4] if logp.endswith(".log") else logp) + ".json"
    try:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n📊 機器可讀結果已存到 {jp}")
    except Exception as e:
        print(f"\n⚠️ 結果檔寫入失敗：{e}")
    print(f"📄 完整輸出已存到 {logp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
