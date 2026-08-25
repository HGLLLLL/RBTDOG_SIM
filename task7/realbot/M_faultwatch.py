#!/usr/bin/env python3
"""M_faultwatch —— D1 Max 閃紅燈故障的唯讀取證工具。

對應 task6 在 D1 EDU 上的 `L5_faultwatch.py`，但資料來源改成 D1 Max 的
`/dev/shm/joint_state`（含 error / 溫度 / 電壓）與 `/dev/shm/joint_cmd`
（看原廠運控有沒有把馬達 disable 掉）。

★ 全程唯讀、不需 sudo、不碰任何行程。可以放心在故障狀態下跑。

⚠️ **開機後 1 分鐘內就要啟動。**
   task6 的經驗：D1 EDU 的故障約在開機 3 分鐘後出現，晚啟動 1 分 49 秒就錯過了
   `ready` 1→0 的轉換瞬間，那次白跑。轉換點才是證據，穩態值沒有用。

它會標記的事件：
  - 任何一顆的 `error` 欄位變化（D1 EDU 那次就是某顆馬達的 ready 位掉了）
  - 電壓或溫度掉到 0（= 該顆真的從匯流排上消失，D1 EDU 惡化時的特徵）
  - `joint_cmd` 的增益出現變化（原廠 disable / 套用急停阻尼時會看到）

用法（在狗上）：
    python3 M_faultwatch.py                 # 10 Hz，Ctrl-C 結束
    python3 M_faultwatch.py --hz 20 --secs 600
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time

import shm_io


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--secs", type=float, default=1800.0, help="最長記錄時間")
    ap.add_argument("--out", default="~/m_logs", help="輸出目錄")
    a = ap.parse_args()

    logp = shm_io.start_log("Mfault", dirname=a.out)
    d = os.path.dirname(logp)
    csvp = logp.replace(".log", ".csv")

    print("M_faultwatch —— D1 Max 故障取證（唯讀）")
    print(f"開始 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   {a.hz:.0f} Hz   最長 {a.secs:.0f}s")
    print(f"CSV → {csvp}")
    print("Ctrl-C 可隨時結束並印摘要\n")

    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過\n")

    fh = open(csvp, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["t", "wall"] +
               [f"{n}.{f}" for n in shm_io.JOINTS
                for f in ("pos", "vel", "tau", "temp", "volt", "err")] +
               [f"{n}.{f}" for n in shm_io.JOINTS for f in ("kp", "kd", "tff")])

    prev_err: list[int] | None = None
    prev_gain: list[tuple] | None = None
    events: list[str] = []
    n = 0
    period = 1.0 / a.hz
    t0 = time.monotonic()

    def note(msg: str):
        line = f"[{now()}  t={time.monotonic()-t0:7.2f}s] {msg}"
        events.append(line)
        print(line)

    try:
        # error 欄位是 u64，read_joint_state() 沒回傳，這裡自己開一份直接讀
        with shm_io.Shm("joint_state") as ss, shm_io.Shm("joint_cmd") as cs:
            import struct
            U8 = struct.Struct("<Q")
            nxt = t0
            while time.monotonic() - t0 < a.secs:
                t = time.monotonic() - t0
                st = shm_io.read_joint_state()
                cm = shm_io.read_joint_cmd()
                err = []
                for i in range(len(shm_io.JOINTS)):
                    o = shm_io.BASE + i * shm_io.STATE_STRIDE + shm_io.STATE_STRIDE - 8
                    err.append(U8.unpack_from(ss.mm, o)[0])

                row = [f"{t:.3f}", now()]
                for i, r in enumerate(st):
                    row += [f"{r['position']:.5f}", f"{r['velocity']:.5f}",
                            f"{r['effort']:.5f}", f"{r['temp_C']:.1f}",
                            f"{r['voltage_V']:.1f}", err[i]]
                for c in cm:
                    row += [f"{c['kp']:.2f}", f"{c['kd']:.2f}", f"{c['effort']:.4f}"]
                w.writerow(row)
                if n % 50 == 0:
                    fh.flush()

                # ---- 事件偵測 ----
                if prev_err is not None and err != prev_err:
                    for i, (o_, n_) in enumerate(zip(prev_err, err)):
                        if o_ != n_:
                            note(f"★ error 變化  {shm_io.JOINTS[i]}: {o_} → {n_}")
                dead = [shm_io.JOINTS[i] for i, r in enumerate(st)
                        if r["voltage_V"] == 0.0 or r["temp_C"] == 0.0]
                if dead and (prev_err is None or n % 100 == 0):
                    note(f"⚠️ 電壓或溫度為 0（疑似失聯）：{', '.join(dead)}")

                gains = [(round(c["kp"], 2), round(c["kd"], 2), round(c["effort"], 3))
                         for c in cm]
                if prev_gain is not None and gains != prev_gain:
                    ch = [shm_io.JOINTS[i] for i in range(16) if gains[i] != prev_gain[i]]
                    note(f"joint_cmd 增益變化：{', '.join(ch[:6])}"
                         + (f" 等 {len(ch)} 顆" if len(ch) > 6 else ""))
                prev_err, prev_gain = err, gains

                if n % int(a.hz * 30) == 0:            # 每 30 秒回報一次還活著
                    tmax = max(r["temp_C"] for r in st)
                    vmin = min(r["voltage_V"] for r in st)
                    print(f"[{now()}  t={t:7.2f}s] 取樣 {n:6d}  最高溫 {tmax:.0f}°C  "
                          f"最低電壓 {vmin:.1f}V  error={sorted(set(err))}")

                n += 1
                nxt += period
                dt = nxt - time.monotonic()
                if dt > 0:
                    time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[Ctrl-C] 結束記錄")
    finally:
        fh.flush()
        fh.close()

    # ---------------------------------------------------------------- 摘要
    el = time.monotonic() - t0
    print("\n" + "=" * 60)
    print(f"記錄 {el:.1f} 秒、{n} 筆取樣")
    if events:
        print(f"\n★ 偵測到 {len(events)} 個事件：")
        for e in events[:40]:
            print("   " + e)
        if len(events) > 40:
            print(f"   …另有 {len(events)-40} 筆，見 log")
    else:
        print("\n沒有偵測到 error / 增益 / 失聯 的變化。")
        print("若紅燈在記錄期間就亮著，代表故障發生在我們開始記錄之前 ——")
        print("請斷電重開，並在**開機後 1 分鐘內**重跑本工具，才抓得到轉換瞬間。")
    print(f"\n📄 log → {logp}")
    print(f"📄 CSV → {csvp}")
    print(f"\n帶回來：scp -r robot@192.168.234.1:{d} ./")
    return 0


if __name__ == "__main__":
    sys.exit(main())
