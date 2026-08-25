#!/usr/bin/env python3
"""M2 —— 驅動單一顆輪馬達（第一次真的讓 D1 Max 的馬達動）。

★ 前置條件（不符就不要跑）：
  - **狗肚子墊高、四輪完全離地**。輪子轉起來不會推動機身
  - M0 全部 ✅、M1 已通過（證明寫入會被消費且零出力）
  - 手邊有電源開關或急停鈕

為什麼挑輪子而不是腿關節：輪子離地空轉不承載機身重量，
即使控制律理解錯了，最壞情況也只是輪子亂轉，不會讓 41 kg 的機身動作。
腿關節（abad/hip/knee）要等吊掛之後再說。

任何一項超標就立刻歸零並解凍：
  - |速度| > --vmax（預設 3.0 rad/s）
  - |力矩| > --tmax（預設 5.0 N·m；規格書輪馬達上限 33 N·m，這裡取極保守值）
  - 逾時 --secs

task6 在 D1 EDU 上的經驗（這台未必相同，但值得先知道）：
  純速度控制（只給 kd）推不動輪子 —— 力矩卡在靜摩擦上不去。
  加一點前饋力矩 tau_ff 才會轉，但 tau_ff 掙脫靜摩擦時會造成速度過衝。
  所以本檔預設 kd 與 tau_ff 都給，且 tau_ff 起始值取小。

在狗上執行（需 root）：
    sudo python3 M2_wheel_spin.py --confirm                      # 預設 fl4_foot、最保守
    sudo python3 M2_wheel_spin.py --joint fr4_foot --confirm
    sudo python3 M2_wheel_spin.py --confirm --vel 0.5 --kd 2.0 --tff 0.8
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
    # pgrep -x 精確比對執行檔名；-f 會匹配到自己的命令列
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
    ap.add_argument("--joint", default="fl4_foot", choices=shm_io.WHEELS,
                    help="要驅動的輪馬達（只允許四顆輪）")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--vel", type=float, default=0.3, help="目標角速度 rad/s")
    ap.add_argument("--kd", type=float, default=1.5, help="速度增益")
    ap.add_argument("--tff", type=float, default=0.0, help="前饋力矩 N·m（掙脫靜摩擦用）")
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--vmax", type=float, default=3.0, help="速度保護")
    ap.add_argument("--tmax", type=float, default=5.0, help="力矩保護")
    a = ap.parse_args()

    logp = shm_io.start_log("M2")
    if os.geteuid() != 0:
        print("❌ 需要 root：sudo python3 M2_wheel_spin.py --confirm")
        return 1

    idx = shm_io.idx_of(a.joint)

    print("M2 —— 單顆輪馬達驅動")
    print(f"   目標關節 {a.joint}（index {idx}）")
    print(f"   v_des={a.vel} rad/s  kd={a.kd}  tau_ff={a.tff} N·m  kp=0（不做位置控制）")
    print(f"   保護：|v|>{a.vmax} rad/s 或 |tau|>{a.tmax} N·m 或 {a.secs}s 逾時 → 立刻歸零")
    print("\n   ⚠️ 確認狗肚子已墊高、四輪完全離地，且手邊有急停。")

    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("\n✅ 結構檢查通過")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節目前帶著非零增益：{', '.join(live)}")
        print("   請先讓狗趴下／洩力再跑。")
        return 1
    print("✅ 16 顆全部洩力中")

    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結行程、沒有寫入。")
        return 0

    period = 1.0 / a.hz
    frozen = False
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")      # 常駐 handle，避免 200 Hz 迴圈裡重複 open/close
    trace: list[tuple[float, float, float]] = []
    abort = ""
    tick_end = None          # ⚠️ 必須在 restore() 之前取，否則會把解凍的 0.3s 等待算進去
    loop_elapsed = 0.0

    def restore():
        try:
            for i in range(len(shm_io.JOINTS)):
                shm.zero_gains(i)        # 先增益歸零 → 立即停止出力
            # ★ 歸零後補一次心跳：讓「零增益」這一幀立刻生效，
            #   而不是等 joint_cmd_timeout(500ms) 逾時才被 controller 清掉。
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
        except Exception as e:
            print(f"⚠️ 歸零失敗：{e}")
        try:
            shm.close()
        except Exception:
            pass
        try:
            state_ro.close()
        except Exception:
            pass
        if frozen:
            os.kill(pid, signal.SIGCONT)
            time.sleep(0.3)
            print(f"✅ 已 SIGCONT 解凍 mc_ctrl，狀態={proc_state(pid)}")

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.2)
        print(f"✅ 已凍結 mc_ctrl（狀態={proc_state(pid)}）\n")

        st0 = state_ro.states()
        p_start = st0[idx]["position"]
        tick_start = state_ro.read_tick(shm_io.STATE_STRIDE)
        print(f"起始角度 {p_start:.4f} rad")
        print("每輪 payload 寫完後同步心跳時戳（缺心跳會被 controller 判定過期而清零）\n")
        print(f"{'t(s)':>6s} {'角度':>9s} {'速度':>9s} {'力矩':>9s}")

        t0 = time.monotonic()
        nxt = t0
        last_print = -1.0
        while True:
            t = time.monotonic() - t0
            if t >= a.secs:
                break

            st = state_ro.states()
            v, tau = st[idx]["velocity"], st[idx]["effort"]
            trace.append((t, v, tau))
            if abs(v) > a.vmax:
                abort = f"速度 {v:.3f} 超過 {a.vmax}"
                break
            if abs(tau) > a.tmax:
                abort = f"力矩 {tau:.3f} 超過 {a.tmax}"
                break

            # 其餘 15 顆每輪都壓零，確保只有目標關節會動
            for i in range(len(shm_io.JOINTS)):
                if i == idx:
                    continue
                shm.zero_gains(i)
            # 目標關節：先目標值、後增益（見 shm_io.write_cmd）
            shm.write_cmd(idx, position=st[idx]["position"], velocity=a.vel,
                          effort=a.tff, kp=0.0, kd=a.kd)
            # ★ 整幀 payload 都寫完了才寫心跳 —— 它是「這幀備妥了」的旗標
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

            if t - last_print >= 0.2:
                print(f"{t:6.2f} {st[idx]['position']:9.4f} {v:9.4f} {tau:9.4f}")
                last_print = t

            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)

        tick_end = state_ro.read_tick(shm_io.STATE_STRIDE)
        loop_elapsed = time.monotonic() - t0
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    finally:
        restore()

    # ---------------------------------------------------------------- 結果
    st1 = shm_io.read_joint_state()      # state_ro 已在 restore() 關閉，這裡另開一次
    print("\n" + "=" * 56)
    if abort:
        print(f"⛔ 提前中止：{abort}")
    if trace:
        if tick_end is not None and loop_elapsed > 0:
            print(f"心跳時戳 {tick_start} → {tick_end}"
                  f"（+{tick_end - tick_start} / {loop_elapsed:.3f}s"
                  f" = 約 {(tick_end - tick_start) / loop_elapsed:.0f}/s，實機應接近 1000/s）")
        else:
            print("心跳時戳：提前中止，未取到結束值")
        vmax = max(abs(v) for _, v, _ in trace)
        # ★ 速度用「角度差分」而非 joint_state 的 velocity 欄位 —— 後者雜訊很大。
        #   2026-08-25 實測：同一段資料，角度差分的變異 9.4%，velocity 欄位 47.2%，
        #   但兩者平均幾乎相同（0.1867 vs 0.1895）→ velocity 無偏但雜訊高。
        if len(trace) > 1:
            v_mean = (st1[idx]["position"] - p_start) / loop_elapsed if loop_elapsed else 0.0
            print(f"平均角速度（由角度差分）{v_mean:.4f} rad/s"
                  f"　追蹤率 {100 * v_mean / a.vel if a.vel else 0:.0f}%（目標 {a.vel}）")
            if a.kd > 0:
                print(f"穩態摩擦力矩推估 τ_f = kd·(v_des − v) = "
                      f"{a.kd * (a.vel - v_mean):.4f} N·m")
        tmax = max(abs(x) for _, _, x in trace)
        dp = st1[idx]["position"] - p_start
        print(f"最大速度 {vmax:.4f} rad/s   最大力矩 {tmax:.4f} N·m")
        print(f"角度變化 {dp:+.4f} rad（{dp * 57.2958:+.2f}°）")
        if abs(dp) > 0.05:
            print("\n★★★ 輪子轉了 —— D1 Max 的底層單顆馬達控制實機驗證成功。")
        elif tmax > 0.05:
            print("\n⚠️ 有出力但沒轉起來 → 多半卡在靜摩擦。")
            print("   照 task6 在 D1 EDU 的經驗，加一點前饋力矩試試：")
            print(f"   sudo python3 M2_wheel_spin.py --joint {a.joint} --confirm "
                  f"--tff 0.5 --kd {a.kd} --vel {a.vel}")
        else:
            print("\n⚠️ 力矩幾乎為零 → 指令可能沒被接受。")
            print("   先確認 joint_cmd 的增益有沒有被清成 0（心跳沒跟上就會這樣），")
            print("   再回頭看 M1 是否仍是 16/16。")
    else:
        print("沒有取到任何取樣。")
    print(f"\n📄 完整輸出已存到 {logp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
