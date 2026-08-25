#!/usr/bin/env python3
"""M1 —— 零出力寫入測試：證明「我們寫進 joint_cmd 的東西會被消費」。

★ 為什麼這是安全的（兩層保險，缺一都成立）：
  1. 我們寫的 kp = kd = effort = 0 → 力矩 = kp·(p_des−p) + kd·(v_des−v) + tau_ff = 0，
     **數學上恆為零**，與 p_des 寫什麼無關。
  2. 即使第 1 點的力矩公式假設有誤，我們把 p_des 寫成「當下的實測角度」，
     追蹤誤差 ≈ 0，任何合理的控制律算出來也接近零出力。

  兩者疊加，這一步不會讓馬達動。

驗證方式：mc_ctrl 平常會把 p_des 寫成各關節的 offset 常數（±0.523 / ±2.443 / ±2.803）。
我們改寫成實測角度後，數值會明顯不同 —— 用另一台電腦訂閱
`/joint_shm_controller/joint_cmd_echo` 就能看到我們的值被 controller 讀走並轉發。

流程：
    SIGSTOP 凍結 mc_ctrl  →  以 200 Hz 持續寫零增益指令  →  SIGCONT 還原

⚠️ 一定要凍結 mc_ctrl，否則它會以自己的頻率蓋掉我們寫的東西，測不出結果。
⚠️ 用 SIGSTOP 不用 kill：kill 掉會被 robot-launch / robot-monitor 重啟，反而更亂。

在狗上執行（需 root）：
    sudo python3 M1_zero_write.py --confirm
    sudo python3 M1_zero_write.py --confirm --secs 5 --hz 200
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import shm_io


def mc_ctrl_pid() -> int | None:
    # ⚠️ 用 pgrep -x（精確比對執行檔名），不要用 -f：
    #    -f 會匹配到我們自己的命令列字串，把自己算進去。
    r = subprocess.run(["pgrep", "-x", "mc_ctrl"], capture_output=True, text=True)
    out = r.stdout.strip()
    return int(out.split("\n")[0]) if out else None


def proc_state(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0]
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="沒帶這個旗標就只做乾跑")
    ap.add_argument("--secs", type=float, default=3.0, help="接管時間（秒）")
    ap.add_argument("--hz", type=float, default=200.0, help="寫入頻率")
    a = ap.parse_args()

    if os.geteuid() != 0:
        print("❌ 需要 root：sudo python3 M1_zero_write.py --confirm")
        return 1

    # ---------------------------------------------------------------- 預檢
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過（16 顆關節名稱與順序正確）")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"⚠️ 這 {len(live)} 顆關節目前帶著非零增益（馬達在出力）：{', '.join(live)}")
        print("   凍結 mc_ctrl 後它們會失去指令來源。請先讓狗趴下／洩力再跑本測試。")
        if a.confirm:
            print("   ❌ 為安全起見拒絕執行。")
            return 1

    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl，先確認運控有起來")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止，沒有凍結任何行程、沒有寫入。")
        return 0

    # ---------------------------------------------------------------- 接管
    period = 1.0 / a.hz
    n_written = 0
    frozen = False
    shm = shm_io.Shm("joint_cmd", write=True)

    def restore(*_):
        """把指令歸零並解凍 —— 不論正常結束、Ctrl-C 或例外都會走到這裡。"""
        try:
            for i in range(len(shm_io.JOINTS)):
                shm.zero_gains(i)
        except Exception as e:
            print(f"⚠️ 歸零失敗：{e}")
        try:
            shm.close()
        except Exception:
            pass
        if frozen:
            os.kill(pid, signal.SIGCONT)
            time.sleep(0.3)
            print(f"✅ 已 SIGCONT 解凍 mc_ctrl，狀態={proc_state(pid)}")

    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.2)
        print(f"✅ 已 SIGSTOP 凍結 mc_ctrl，狀態={proc_state(pid)}（T=已凍結）")

        st = shm_io.read_joint_state()
        print(f"\n以 {a.hz:.0f} Hz 寫入零增益指令 {a.secs:.1f} 秒 "
              f"（p_des = 實測角度、v=0、tau_ff=0、kp=0、kd=0）…")

        t0 = time.monotonic()
        nxt = t0
        while time.monotonic() - t0 < a.secs:
            for i in range(len(shm_io.JOINTS)):
                # 先目標、後增益（見 shm_io.write_cmd 的說明）
                shm.write_cmd(i, position=st[i]["position"], velocity=0.0,
                              effort=0.0, kp=0.0, kd=0.0)
            n_written += 1
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)

        # ---------------------------------------------------------------- 驗證
        after = shm_io.read_joint_cmd()
        st2 = shm_io.read_joint_state()
        print(f"\n共寫入 {n_written} 輪")
        print(f"\n{'關節':16s} {'原本 p_des':>11s} {'我們寫的':>10s} {'現在讀回':>10s} "
              f"{'kp':>5s} {'kd':>5s} | {'最大|速度|':>10s} {'最大|力矩|':>10s}")
        ok = 0
        for i, nm in enumerate(shm_io.JOINTS):
            want = st[i]["position"]
            got = after[i]["position"]
            match = abs(got - want) < 1e-9
            ok += match
            print(f"{nm:16s} {cmd0[i]['position']:11.4f} {want:10.4f} {got:10.4f} "
                  f"{after[i]['kp']:5.1f} {after[i]['kd']:5.1f} | "
                  f"{abs(st2[i]['velocity']):10.4f} {abs(st2[i]['effort']):10.4f}")

        vmax = max(abs(x["velocity"]) for x in st2)
        tmax = max(abs(x["effort"]) for x in st2)
        print(f"\n寫入後讀回相符：{ok}/16")
        print(f"全機最大速度 {vmax:.4f} rad/s、最大力矩 {tmax:.4f} N·m")
        if ok == 16 and tmax < 0.5:
            print("\n★ 通過：我們的指令確實寫進去了，而且馬達沒有出力。")
            print("  下一步可做 M2_wheel_spin.py（需先墊高、四輪離地）。")
        elif ok < 16:
            print("\n⚠️ 有欄位沒寫進去或被別人蓋掉 —— 檢查 mc_ctrl 是否真的凍住了。")
        else:
            print("\n⚠️ 力矩偏高，與預期的零出力不符，先停下來查清楚再繼續。")

        print("\n（要確認 controller 有把我們的值讀走，請在另一台電腦同時跑：")
        print("  ros2 topic echo /joint_shm_controller/joint_cmd_echo --once ）")

    except KeyboardInterrupt:
        print("\n[中斷] Ctrl-C，執行還原…")
    except Exception as e:
        print(f"\n❌ 例外：{e}，執行還原…")
        raise
    finally:
        restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
