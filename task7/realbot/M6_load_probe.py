#!/usr/bin/env python3
"""M6 —— 承重狀態的唯讀擷取（**零風險**：不寫入、不凍結行程、不需 sudo）。

要回答的問題：**狗站在地上、承受全部體重時，每個關節實際在做什麼？**

到 2026-08-26 為止的所有實機測試都是**吊掛**的 —— 關節只撐自己的腿重。
模擬算出站立承重時膝關節要 **14.9 N·m**（吊掛時只要 2.9），是 5 倍。
在我們自己去撐那個載荷之前，先把實機的真實數字讀回來。

★ 關鍵：`joint_cmd` 是**可讀**的。狗在原廠控制下站著時，我們可以直接讀到

    原廠用的 kp / kd / 目標角  ＋  實際產生的力矩 ＋ 實際的關節角

也就是一份完整的「我們要複製什麼」的規格 —— 而且完全不用碰它。

兩種模式：

  **靜態擷取**（預設）—— 狗停穩後跑，取多秒平均：
      python3 M6_load_probe.py stand_ground --secs 5

  **★ 全程錄製**（`--record`）—— 逐筆記錄時間序列，錄原廠自己做動作的全程：
      python3 M6_load_probe.py standup --record --secs 30
      # 開始跑之後，再用遙控器讓狗從趴下站起來

★ 為什麼錄製比靜態快照有價值得多：
  **原廠控制器已經解決了我們正要解決的問題。** 錄下它的全程等於拿到
  它用的軌跡（p_des 隨時間）、增益排程（kp/kd 有沒有分段）、
  有沒有重力前饋（effort 欄），以及**真正的力矩包絡**。
  靜態站著只給你終點，而**峰值力矩發生在起身的過程中**，
  那才是決定我們保護門檻的數字。

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


def record(secs: float, hz: float) -> dict:
    """逐筆記錄時間序列。Ctrl-C 提前結束會保留已錄到的部分。

    ⚠️ 與靜態擷取同樣的鐵則：兩塊 shm 在**同一輪**讀完再進下一輪。
       分兩趟讀會讓「力矩」與「當時的指令」錯位 —— 靜態下看不出來，
       但這個模式專門用來錄**動作中**的資料，錯位會直接毀掉因果推論。
    """
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)

    T, IMU = [], []
    J = {n: {"q": [], "des": [], "tau": [], "v": [], "kp": [], "kd": [], "ff": []}
         for n in shm_io.JOINTS}
    period = 1.0 / hz
    interrupted = False
    print(f"● 錄製中 —— 最長 {secs:.0f} 秒。**現在可以動遙控器了。**")
    print("  （Ctrl-C 可提前結束並保留資料）\n")
    with shm_io.Shm("joint_state") as ss, shm_io.Shm("joint_cmd") as sc, \
            shm_io.Shm("imu_central") as si:
        tick0 = ss.read_tick(shm_io.STATE_STRIDE)
        t0 = nxt = time.monotonic()
        last = -1.0
        try:
            while True:
                t = time.monotonic() - t0
                if t >= secs:
                    break
                st = ss.states()
                cm = sc.read_records(shm_io.CMD_STRIDE, 5)
                T.append(round(t, 4))
                for i, nm in enumerate(shm_io.JOINTS):
                    d = J[nm]
                    d["q"].append(round(st[i]["position"], 5))
                    d["v"].append(round(st[i]["velocity"], 4))
                    d["tau"].append(round(st[i]["effort"], 4))
                    d["des"].append(round(cm[i][0], 5))
                    d["ff"].append(round(cm[i][2], 4))
                    d["kp"].append(round(cm[i][3], 3))
                    d["kd"].append(round(cm[i][4], 3))
                IMU.append([round(shm_io._F8.unpack_from(si.mm, 824 + 8 * k)[0], 5)
                            for k in range(10)])
                if t - last >= 1.0:
                    # 即時回饋：讓操作者知道在錄、而且看得到動作有沒有被記到
                    mt = max(abs(x["effort"]) for x in st)
                    mv = max(abs(x["velocity"]) for x in st)
                    on = sum(1 for c in cm if abs(c[3]) + abs(c[4]) > 1e-9)
                    print(f"  {t:5.1f}s  最大|τ| {mt:6.2f}  最大|v| {mv:5.2f}  "
                          f"帶增益的關節 {on:2d}/16")
                    last = t
                nxt += period
                dly = nxt - time.monotonic()
                if dly > 0:
                    time.sleep(dly)
        except KeyboardInterrupt:
            interrupted = True
            print("\n  （Ctrl-C：提前結束，保留已錄到的資料）")
        tick1 = ss.read_tick(shm_io.STATE_STRIDE)
    dt = T[-1] if T else 0.0
    return {"t": T, "j": J, "imu": IMU, "secs": dt, "n": len(T),
            "interrupted": interrupted,
            "tick_rate": (tick1 - tick0) / dt if dt > 0 else 0.0}


def _fold(xs):
    """整段都一樣就存成純量，否則存陣列。

    ★ 這不只是為了省空間 —— **「這個欄位是常數還是有變」本身就是答案**。
      kp 存成純量 = 原廠全程用同一組增益；存成陣列 = 它有增益排程。
      看 JSON 就知道，不必另外分析。
    """
    return xs[0] if xs and all(x == xs[0] for x in xs) else xs


def analyse_record(r: dict) -> None:
    T, J = r["t"], r["j"]
    if not T:
        print("沒有錄到任何資料。")
        return
    print(f"\n錄到 {r['n']} 筆 / {r['secs']:.2f} 秒 @ {r['n']/max(r['secs'],1e-9):.0f} Hz"
          f"　joint_state 心跳 {r['tick_rate']:.0f}/s"
          + ("　（提前結束）" if r["interrupted"] else ""))

    # ---- 增益有沒有變？這是「原廠有沒有增益排程」的直接答案
    print("\n" + "=" * 78)
    print("① 原廠的增益：全程固定，還是有排程？")
    print("=" * 78)
    print(f"{'關節':16s} {'kp':>22s} {'kd':>18s} {'前饋 effort':>20s}")
    sched = []
    for n in shm_io.JOINTS:
        d = J[n]
        def desc(xs, fmt="{:.1f}"):
            lo, hi = min(xs), max(xs)
            return fmt.format(lo) if lo == hi else f"{fmt.format(lo)} → {fmt.format(hi)} ★變"
        if max(d["kp"]) != min(d["kp"]) or max(d["kd"]) != min(d["kd"]):
            sched.append(n)
        print(f"{n:16s} {desc(d['kp']):>22s} {desc(d['kd'],'{:.2f}'):>18s}"
              f" {desc(d['ff'],'{:+.3f}'):>20s}")
    if sched:
        print(f"\n   ★ {len(sched)} 個關節的增益在過程中**變過** → 原廠有增益排程，")
        print("     我們的單一 kp 複製不了它。要看時間序列才知道怎麼排。")
    else:
        print("\n   ✅ 增益全程固定 → 沒有排程，單一 kp/kd 就能複製。")
    # ⚠️ **腿關節與輪關節要分開講**。2026-08-26 第一版把兩者混在一起報
    #    「有前饋力矩 → 這解釋了追蹤誤差為何比純 PD 小」，但非零的那 4 個
    #    全是**輪子**（前饋約 ±0.2，剛好等於實測的輪摩擦），腿關節是 0。
    #    腿的誤差小是因為 kp 高，不是因為前饋 —— 那句話把因果講反了。
    ff_leg = [n for n in LEG_JOINTS if any(abs(x) > 1e-6 for x in J[n]["ff"])]
    ff_wh = [n for n in shm_io.WHEELS if any(abs(x) > 1e-6 for x in J[n]["ff"])]
    if ff_leg:
        print(f"   ★★ **腿關節有前饋力矩**（{len(ff_leg)} 個）——")
        print("      那是重力補償，我們的純 PD 複製不了，要一起實作。")
    else:
        print("   ✅ **腿關節的前饋 effort 全程為 0 → 原廠是純 PD。**")
        print("      承重時追蹤誤差之所以小，是因為 kp 高，不是因為有前饋。")
    if ff_wh:
        print(f"   ℹ️ 輪子有前饋（{len(ff_wh)} 顆），量級可對照實測輪摩擦 0.15~0.20 N·m")

    # ---- 力矩包絡：這才是保護門檻的依據
    print("\n" + "=" * 78)
    print("② 力矩包絡（控制器座標系）—— **保護門檻要照這個訂，不是照靜態值**")
    print("=" * 78)
    print(f"{'關節':16s} {'峰值|τ|':>9s} {'發生於':>8s} {'結束時τ':>9s} {'峰值/結束':>10s}")
    for n in LEG_JOINTS:
        sgn = coord.SIGN[n[2:]][n[:2]]
        taus = [sgn * x for x in J[n]["tau"]]
        i = max(range(len(taus)), key=lambda k: abs(taus[k]))
        fin = mean(taus[-max(1, len(taus) // 20):])
        ratio = abs(taus[i]) / abs(fin) if abs(fin) > 1e-6 else float("inf")
        print(f"{n:16s} {abs(taus[i]):9.2f} {T[i]:7.2f}s {fin:+9.2f}"
              f" {ratio:9.1f}x")
    allpk = max(abs(coord.SIGN[n[2:]][n[:2]] * x)
                for n in LEG_JOINTS for x in J[n]["tau"])
    print(f"\n   全部腿關節的峰值 |τ| = **{allpk:.2f} N·m**")
    print(f"   → 我們自己做同樣動作時，力矩保護至少要 {allpk*1.5:.0f} N·m")
    print("   （目前 M5 的門檻是 ABAD 10 / HIP 8 / KNEE 7，是照**吊掛**訂的）")

    # ---- 粗略時間軸
    print("\n" + "=" * 78)
    print("③ 時間軸（每 0.5 秒一列）")
    print("=" * 78)
    print(f"{'t':>6s} {'機身pitch':>9s} {'最大|τ|':>8s} {'在哪顆':>16s}"
          f" {'最大|v|':>8s} {'帶增益':>7s}")
    step = max(1, int(len(T) / max(1, r["secs"] / 0.5)))
    for k in range(0, len(T), step):
        qx, qy, qz, qw = r["imu"][k][6:10]
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx)))))
        pairs = [(abs(coord.SIGN[n[2:]][n[:2]] * J[n]["tau"][k]), n) for n in LEG_JOINTS]
        mt, mn = max(pairs)
        mv = max(abs(J[n]["v"][k]) for n in shm_io.JOINTS)
        on = sum(1 for n in shm_io.JOINTS
                 if abs(J[n]["kp"][k]) + abs(J[n]["kd"][k]) > 1e-9)
        print(f"{T[k]:6.2f} {pitch:+9.2f} {mt:8.2f} {mn:>16s} {mv:8.2f} {on:6d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="這次擷取叫什麼，例如 stand_ground / hang_free")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--note", default="", help="自由文字：狗當下的狀態、遙控器按了什麼")
    ap.add_argument("--record", action="store_true",
                    help="★ 逐筆記錄時間序列（錄原廠做動作的全程）。"
                         "Ctrl-C 可提前結束並保留已錄到的資料")
    a = ap.parse_args()

    logp = shm_io.start_log("M6")
    print("M6 —— 承重狀態唯讀擷取（不寫入、不凍結、不需 sudo）\n")
    print(f"標籤 {a.label}　{'全程錄製' if a.record else '靜態擷取'}"
          f"　{a.secs:.1f} 秒 @ {a.hz:.0f} Hz")
    if a.note:
        print(f"備註：{a.note}")

    # ---------------------------------------------------------------- 錄製模式
    if a.record:
        rec = record(a.secs, a.hz)
        analyse_record(rec)
        out = {"schema": "m6_record/1", "label": a.label, "note": a.note,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n": rec["n"], "secs": rec["secs"], "hz_actual":
                   rec["n"] / max(rec["secs"], 1e-9),
               "tick_rate": rec["tick_rate"], "interrupted": rec["interrupted"],
               "t": rec["t"],
               # ★ 每個欄位若整段不變就存純量 —— 見 _fold()：
               #   「是常數還是有變」本身就是答案，看 JSON 就知道有沒有增益排程。
               "joints": {n2: {k: _fold(v) for k, v in d.items()}
                          for n2, d in rec["j"].items()},
               "imu": rec["imu"]}
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
            print(f"\n📊 時間序列已存到 {jp}"
                  f"（{os.path.getsize(jp)/1024:.0f} KB）")
        except Exception as e:
            print(f"\n⚠️ 結果檔寫入失敗：{e}")
        print(f"📄 完整輸出已存到 {logp}")
        return 0

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
