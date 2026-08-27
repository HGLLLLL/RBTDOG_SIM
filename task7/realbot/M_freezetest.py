#!/usr/bin/env python3
"""M_freezetest —— `mc_ctrl` 到底能凍結多久？**零風險**觀察實驗。

════════════════════════════════════════════════════════════════════
為什麼要有這支
════════════════════════════════════════════════════════════════════

我們的每一支寫入工具（M1/M2/M3/M5/M7/M8）都靠 `SIGSTOP` 凍結 `mc_ctrl`，
才能獨占 `/dev/shm/joint_cmd`。目前**實測支持的最長凍結是 38.1 秒**
（`docs/實機四輪驅動結果_2026-08-25` §5 的 M3 run3），而且那次完全正常 ——

★ **38.1 不是量到的極限，只是「我們試過最久的一次」。**
  M3 的 `--max-freeze 90` 也是隨手訂的護欄，不是機器的限制。

而基準步態是 **180 秒**。「凍 180 秒會不會出事」是個**沒人問過的問題**，
不是已知會出事。這支就是去問它。

════════════════════════════════════════════════════════════════════
為什麼是零風險
════════════════════════════════════════════════════════════════════

  - **本程式一個位元組都不寫 `joint_cmd`**（連開檔都是唯讀）
  - 狗趴在地上、16 顆洩力；`mc_ctrl` 一凍，controller 逾時 500 ms 後
    會把指令區清成 0 → 馬達本來就沒有出力路徑
  - 不需要吊掛、不需要墊高、不會有任何動作

⚠️ **唯一真正的風險**：如果某個看門狗**重啟了 `mc_ctrl`**，
   新的那個會開始寫指令區，而舊的還被我們凍著。
   → 本程式每個取樣週期都在看 PID 集合，一發現有新的就**立刻停止並且不解凍**
     （解凍會變成兩個 mc_ctrl 同時寫同一塊記憶體，比什麼都糟），
     然後叫你人工處理。

════════════════════════════════════════════════════════════════════
要觀察什麼
════════════════════════════════════════════════════════════════════

| 觀察量 | 為什麼 |
|---|---|
| `mc_ctrl` 的 PID 集合 | 有沒有被重啟（**最重要**） |
| `mc_ctrl` 的行程狀態 | 應該一直是 `T`（stopped） |
| **`joint_state` 的時戳有沒有繼續遞增** | ★ 那是 **driver** 維護的，與 mc_ctrl 無關 —— 它停了代表底層死了 |
| `joint_cmd` 的 kp/kd/effort | 應該全 0 且維持 0。**變成非 0 = 有別人在寫** |
| 馬達溫度／電壓 | 有沒有異常 |
| 關節角 | 狗應該一動不動 |
| `/proc/loadavg` | 凍結有沒有讓系統負載改變 |

建議階梯（每一階跑完看報告再往下）：**60 → 120 → 200 秒**。

用法：
    python3 M_freezetest.py --secs 60                  # 乾跑，只做前置檢查
    sudo python3 M_freezetest.py --secs 60 --confirm   # 真的凍 60 秒
    sudo python3 M_freezetest.py --secs 200 --confirm  # 目標值

⚠️ 純標準函式庫。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import shm_io


def mc_ctrl_pids() -> list[int]:
    """★ 回傳**全部**的 mc_ctrl PID，不是只有第一個。

    `M5_leg_pose.mc_ctrl_pid()` 只取第一個 —— 那對「凍結誰」夠用，
    但本測試要偵測的正是「多出一個」，所以必須看整個集合。
    """
    r = subprocess.run(["pgrep", "-x", "mc_ctrl"], capture_output=True, text=True)
    return sorted(int(x) for x in r.stdout.split())


def proc_state(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0]
    except Exception:
        return "?"


def loadavg() -> tuple:
    try:
        with open("/proc/loadavg") as f:
            p = f.read().split()
        return (float(p[0]), float(p[1]), float(p[2]), p[3])
    except Exception:
        return (0.0, 0.0, 0.0, "?")


def snapshot(cmd_shm, state_shm) -> dict:
    """一次取樣。全部唯讀。"""
    st = state_shm.states()
    cmd = cmd_shm.read_records(shm_io.CMD_STRIDE, 5)
    gains = max(abs(r[3]) + abs(r[4]) + abs(r[2]) for r in cmd)   # kp+kd+effort
    la = loadavg()
    return {
        "pids": mc_ctrl_pids(),
        "state_tick": state_shm.read_tick(shm_io.STATE_STRIDE),
        "cmd_tick": cmd_shm.read_tick(shm_io.CMD_STRIDE),
        "gain_sum": round(gains, 6),
        "temp_max": round(max(r["temp_C"] for r in st), 1),
        "volt_min": round(min(r["voltage_V"] for r in st), 2),
        "q": [round(r["position"], 4) for r in st],
        "load1": la[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="mc_ctrl 長時間凍結的零風險觀察實驗")
    ap.add_argument("--secs", type=float, default=60.0,
                    help="凍結秒數。建議階梯 60 → 120 → 200")
    ap.add_argument("--interval", type=float, default=0.5, help="取樣週期（秒）")
    ap.add_argument("--confirm", action="store_true", help="不帶就是乾跑")
    a = ap.parse_args()

    logp = shm_io.start_log("MF")
    print(f"M_freezetest —— mc_ctrl 凍結 {a.secs:.0f} 秒的觀察實驗\n")
    print("★ 本程式**完全不寫入** joint_cmd。狗趴在地上洩力即可，不需吊掛、不需墊高。")
    print("  已知安全範圍：38.1 秒（M3 run3，2026-08-25，完全正常）。\n")

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
        print("   （代表有東西在出力，凍結期間的觀察會不乾淨）")
        return 1
    print("✅ 16 顆全部洩力中")

    pids0 = mc_ctrl_pids()
    if len(pids0) != 1:
        print(f"❌ 找到 {len(pids0)} 個 mc_ctrl：{pids0}")
        print("   要正好一個才能做這個測試。")
        return 1
    pid = pids0[0]
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")
    la = loadavg()
    print(f"✅ 系統負載 {la[0]} {la[1]} {la[2]}　行程 {la[3]}")

    with shm_io.Shm("joint_cmd") as c, shm_io.Shm("joint_state") as st:
        base = snapshot(c, st)
    print(f"✅ joint_state 時戳 {base['state_tick']}　"
          f"最高溫 {base['temp_max']}°C　最低電壓 {base['volt_min']} V")

    print(f"\n計畫：SIGSTOP → 觀察 {a.secs:.0f} 秒（每 {a.interval}s 取樣）→ SIGCONT")
    print("觀察：PID 集合／行程狀態／joint_state 時戳／指令區增益／溫度電壓／關節角")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結、沒有寫入。")
        print(f"\n📄 {logp}")
        return 0
    if os.geteuid() != 0:
        print("❌ 需要 root：請加 sudo")
        return 1

    # ---------------------------------------------------------------- 觀察
    cmd_shm = shm_io.Shm("joint_cmd")
    state_shm = shm_io.Shm("joint_state")
    samples = []
    alarm = ""
    safe_to_cont = True          # ★ 發現第二個 mc_ctrl 時會變 False
    frozen = False
    t0 = 0.0

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        t0 = time.monotonic()
        time.sleep(0.15)
        print(f"\n✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")
        print(f"{'t':>7s} {'狀態':>4s} {'PID數':>5s} {'state時戳':>11s} {'Δ時戳/s':>9s} "
              f"{'增益和':>7s} {'溫度':>6s} {'電壓':>6s} {'關節最大移動':>12s} {'load1':>6s}")
        last_print = -99.0
        prev = dict(base)
        prev_t = 0.0
        while True:
            t = time.monotonic() - t0
            if t >= a.secs:
                break
            s = snapshot(cmd_shm, state_shm)
            s["t"] = round(t, 2)
            s["proc"] = proc_state(pid)
            samples.append(s)

            # ---- 警報判定
            new = [p for p in s["pids"] if p not in pids0]
            if new:
                alarm = (f"★★ 出現新的 mc_ctrl PID {new} —— 有東西把它重啟了。"
                         f"**不解凍**（兩個一起寫同一塊記憶體比什麼都糟）")
                safe_to_cont = False
            elif pid not in s["pids"]:
                alarm = f"★★ 被凍結的 PID {pid} 不見了 —— 有東西殺掉它"
                safe_to_cont = False
            elif s["proc"] not in ("T", "t"):
                alarm = f"行程狀態變成 {s['proc']}（應該是 T）"
            elif s["gain_sum"] > 1e-9:
                alarm = (f"★★ 指令區的增益變成非 0（和={s['gain_sum']}）"
                         f" —— **有別人在寫 joint_cmd**")
            elif s["state_tick"] == prev["state_tick"] and t - prev_t > 2.0:
                alarm = ("★★ joint_state 的時戳停了 2 秒沒動 —— "
                         "那是 driver 維護的，代表底層可能死了")
            elif s["temp_max"] > 70.0:
                alarm = f"馬達溫度 {s['temp_max']}°C 超過 70"

            if s["state_tick"] != prev["state_tick"]:
                prev, prev_t = dict(s), t

            if t - last_print >= 2.0 or alarm:
                dtick = ((s["state_tick"] - base["state_tick"]) / max(t, 1e-6))
                dq = max(abs(x - y) for x, y in zip(s["q"], base["q"]))
                print(f"{t:7.1f} {s['proc']:>4s} {len(s['pids']):5d} "
                      f"{s['state_tick']:11d} {dtick:9.0f} {s['gain_sum']:7.3f} "
                      f"{s['temp_max']:5.1f}° {s['volt_min']:5.2f}V "
                      f"{dq:11.4f}  {s['load1']:6.2f}")
                last_print = t
            if alarm:
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        alarm = "使用者 Ctrl-C"
    except Exception as e:
        alarm = f"未預期的例外：{type(e).__name__}: {e}"

    held = time.monotonic() - t0 if frozen else 0.0

    # ---------------------------------------------------------------- 解凍
    print("\n" + "=" * 76)
    if alarm:
        print(f"⛔ 提前結束（凍了 {held:.1f} 秒）：{alarm}")
    else:
        print(f"✅ 凍結 {held:.1f} 秒，全程沒有異常")

    if frozen and safe_to_cont:
        os.kill(pid, signal.SIGCONT)
        time.sleep(0.5)
        print(f"✅ 已 SIGCONT 解凍，狀態={proc_state(pid)}")
    elif frozen:
        print(f"\n❌❌ **沒有解凍**（PID {pid} 仍是 {proc_state(pid)}）。")
        print("   理由見上面的警報。現在請人工處理：")
        print(f"     pgrep -x mc_ctrl            # 看現在有幾個")
        print(f"     ps -o pid,stat,lstart,cmd -p $(pgrep -x mc_ctrl | tr '\\n' ',')")
        print("   ★ **不要盲目 SIGCONT** —— 若已經有新的在跑，會變成兩個一起寫指令區。")
        print("   最保守的收法是整台重開機（SIGSTOP/SIGCONT 不會留下永久改變）。")

    # ---------------------------------------------------------------- 事後
    time.sleep(1.0)
    post = None
    try:
        post = snapshot(cmd_shm, state_shm)
        print(f"\n解凍後：PID {post['pids']}　state 時戳 {post['state_tick']}"
              f"（凍結期間 +{post['state_tick'] - base['state_tick']}）")
        print(f"        最高溫 {post['temp_max']}°C（起始 {base['temp_max']}）　"
              f"最低電壓 {post['volt_min']} V（起始 {base['volt_min']}）")
        dq = max(abs(x - y) for x, y in zip(post["q"], base["q"]))
        print(f"        關節最大移動 {dq:.4f} rad（狗應該一動不動）")
        cmd1 = shm_io.read_joint_cmd()
        live1 = [c["name"] for c in cmd1
                 if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
        print(f"        指令區帶增益的關節：{live1 if live1 else '無 ✅'}")
    except Exception as e:
        print(f"\n⚠️ 事後取樣失敗：{e}")

    # ---- 結論
    print("\n" + "=" * 76)
    if not alarm and safe_to_cont:
        print(f"★ **凍結 {held:.0f} 秒沒有問題。**")
        nxt = {60: 120, 120: 200}.get(int(round(held)), None)
        if held < 190:
            print(f"  下一階：sudo python3 M_freezetest.py --secs "
                  f"{nxt or int(held) * 2} --confirm")
        else:
            print("  ★★ 已達步態需要的量級（180 秒）。M9 可以不必設計成分段。")
    else:
        print("★ **這一趟有異常，不要據此放寬凍結時間。** 先弄清楚上面的警報。")

    out = {"schema": "m_freezetest/1", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "args": vars(a), "pid": pid, "held_s": round(held, 2),
           "alarm": alarm or None, "resumed": bool(frozen and safe_to_cont),
           "base": base, "post": post, "samples": samples}
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
        cmd_shm.close()
        state_shm.close()
    except Exception:
        pass
    print(f"\n📄 {logp}")
    return 1 if alarm else 0


if __name__ == "__main__":
    sys.exit(main())
