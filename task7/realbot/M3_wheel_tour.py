#!/usr/bin/env python3
"""M3 —— 依序驅動多顆輪馬達（一鏡到底，方便錄影）。

M2 已驗證 `fl4_foot`（左前輪）可驅動。這支把**其餘三顆**依序跑一遍，
凍結 mc_ctrl 一次、中間不解凍，所以錄影不必中斷重下指令。

★ 前置條件（與 M2 相同，不符就不要跑）：
  - **狗肚子墊高、四輪完全離地**
  - M0 全部 ✅、M1 16/16、M2 已成功
  - 手邊有急停

每顆輪子之間會有 `--gap` 秒的靜止間隔（全部歸零但**維持心跳**），
畫面上看得出「換下一顆」的分界，剪片也好抓點。

安全設計與 M2 相同，另外多兩層：
  - `--joints` 只接受四顆輪，腿關節選不到
  - `--max-freeze` 總凍結時間上限（預設 90 秒），超過直接收工

⚠️ **mc_ctrl 長時間凍結是未測領域。** M1/M2 只凍了 2–3 秒，
   這支預設會凍約 43 秒（3 顆 × (10s 轉 + 3s 倒數) + 2 × 2s 間隔）。機上有 `robot_monitor` / `robot_self_test_manager`，
   凍太久會不會被判定異常沒人知道。**建議第一次先用 `--secs 4` 試一輪**，
   確認解凍後 mc_ctrl 狀態回到 `S`、遙控器仍正常，再拉長。

在狗上執行（需 root）：
    sudo python3 M3_wheel_tour.py --confirm                    # 其餘三顆，各 10 秒
    sudo python3 M3_wheel_tour.py --confirm --secs 4           # 先短試一輪
    sudo python3 M3_wheel_tour.py --confirm --joints all --secs 15
    sudo python3 M3_wheel_tour.py --confirm --tff 0.2          # 想轉快一點
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import shm_io

REMAINING = ["fr4_foot", "bl4_foot", "br4_foot"]     # M2 已做過 fl4_foot
LABEL = {"fl4_foot": "左前輪", "fr4_foot": "右前輪",
         "bl4_foot": "左後輪", "br4_foot": "右後輪"}


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
    ap.add_argument("--joints", default="remaining",
                    help="remaining（預設，其餘三顆）／all（四顆）／逗號分隔的輪名")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--secs", type=float, default=10.0, help="每顆轉多久")
    ap.add_argument("--gap", type=float, default=2.0, help="每顆之間的靜止間隔")
    ap.add_argument("--vel", type=float, default=0.3, help="目標角速度 rad/s")
    ap.add_argument("--kd", type=float, default=1.5)
    ap.add_argument("--tff", type=float, default=0.0, help="前饋力矩 N·m")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--vmax", type=float, default=3.0)
    ap.add_argument("--tmax", type=float, default=5.0)
    ap.add_argument("--countdown", type=int, default=3,
                    help="每顆輪子之前的倒數秒數（錄影用的分界；0 = 不倒數）")
    ap.add_argument("--max-freeze", type=float, default=90.0,
                    dest="max_freeze", help="總凍結時間上限（秒）")
    a = ap.parse_args()

    if a.joints == "remaining":
        joints = list(REMAINING)
    elif a.joints == "all":
        joints = list(shm_io.WHEELS)
    else:
        joints = [j.strip() for j in a.joints.split(",") if j.strip()]
    bad = [j for j in joints if j not in shm_io.WHEELS]
    if bad:
        print(f"❌ 只允許四顆輪：{', '.join(shm_io.WHEELS)}\n   不接受：{', '.join(bad)}")
        return 1

    logp = shm_io.start_log("M3")
    if os.geteuid() != 0:
        print("❌ 需要 root：sudo python3 M3_wheel_tour.py --confirm")
        return 1

    # ⚠️ 估算要把倒數也算進去，否則守衛會低估實際凍結時間
    total = (len(joints) * (a.secs + a.countdown)
             + max(0, len(joints) - 1) * a.gap)
    print("M3 —— 多顆輪馬達巡迴（錄影用）")
    print(f"   順序　　{' → '.join(f'{LABEL[j]}({j})' for j in joints)}")
    print(f"   每顆　　{a.secs:.1f} 秒　倒數 {a.countdown} 秒　間隔 {a.gap:.1f} 秒"
          f"　→ 預估總凍結 {total:.1f} 秒")
    print(f"   參數　　v_des={a.vel} rad/s  kd={a.kd}  tau_ff={a.tff}  kp=0")
    print(f"   保護　　|v|>{a.vmax} rad/s、|tau|>{a.tmax} N·m、總凍結 >{a.max_freeze:.0f}s")
    print("\n   ⚠️ 狗肚子墊高、四輪完全離地、急停在手邊。")

    if total > a.max_freeze:
        print(f"\n❌ 預估凍結 {total:.1f}s 超過上限 {a.max_freeze:.0f}s，請縮短 --secs 或提高 --max-freeze")
        return 1

    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("\n✅ 結構檢查通過")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節目前帶著非零增益：{', '.join(live)}\n   請先讓狗趴下／洩力再跑。")
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
    state_ro = shm_io.Shm("joint_state")
    results: list[dict] = []
    aborted = ""

    def hold_all():
        """全部歸零 + 維持心跳。間隔期間用，避免 controller 逾時清空。"""
        for i in range(len(shm_io.JOINTS)):
            shm.zero_gains(i)
        shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

    def restore():
        try:
            hold_all()
        except Exception as e:
            print(f"⚠️ 歸零失敗：{e}")
        for h in (shm, state_ro):
            try:
                h.close()
            except Exception:
                pass
        if frozen:
            os.kill(pid, signal.SIGCONT)
            time.sleep(0.3)
            print(f"\n✅ 已 SIGCONT 解凍 mc_ctrl，狀態={proc_state(pid)}")

    def spin(idx: int, name: str) -> dict:
        """驅動單一顆輪 a.secs 秒，回傳量測結果。

        ★ 角度必須逐筆解纏累加：輪關節讀數包裹在 [−π, π]，
          直接 end−start 在轉超過半圈時會給出錯誤的量值甚至反向。
        """
        st = state_ro.states()
        p_prev = st[idx]["position"]
        total = 0.0                      # 解纏後的累積轉角
        hist: list[tuple[float, float]] = [(0.0, 0.0)]   # (t, total)，供保護用
        taus, t0 = [], time.monotonic()
        nxt, last_print = t0, -1.0
        stop = ""
        raw_hot = 0                      # velocity 欄位連續超標次數
        while True:
            t = time.monotonic() - t0
            if t >= a.secs:
                break
            st = state_ro.states()
            p_raw, v_raw, tau = st[idx]["position"], st[idx]["velocity"], st[idx]["effort"]
            total += shm_io.wrap_pi(p_raw - p_prev)
            p_prev = p_raw
            hist.append((t, total))
            taus.append(tau)

            # 保護①：由解纏角度算的速度（wrap-safe，取 ~50 ms 視窗降雜訊）
            # ⚠️ 視窗太短時 v = Δθ/Δt 會爆掉（迴圈剛啟動時 Δt 可能趨近 0）→ 誤中止。
            #    要求視窗至少橫跨 20 ms 才採信。
            w = [h for h in hist if t - h[0] <= 0.05]
            if len(w) >= 3 and (t - w[0][0]) >= 0.02:
                v_ang = (total - w[0][1]) / (t - w[0][0])
                if abs(v_ang) > a.vmax:
                    stop = f"速度 {v_ang:.3f} rad/s（角度差分）超過 {a.vmax}"
                    break
            # 保護②：velocity 欄位。它雜訊大又會在 wrap 時噴尖峰，
            #         所以要求**連續 5 筆**超標才中止，避免單一尖峰誤殺。
            raw_hot = raw_hot + 1 if abs(v_raw) > a.vmax else 0
            if raw_hot >= 5:
                stop = f"velocity 欄位連續 5 筆超過 {a.vmax}（最後 {v_raw:.3f}）"
                break
            if abs(tau) > a.tmax:
                stop = f"力矩 {tau:.3f} 超過 {a.tmax}"
                break
            for i in range(len(shm_io.JOINTS)):
                if i != idx:
                    shm.zero_gains(i)
            shm.write_cmd(idx, position=st[idx]["position"], velocity=a.vel,
                          effort=a.tff, kp=0.0, kd=a.kd)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
            if t - last_print >= 1.0:
                print(f"     t={t:5.1f}s  累積轉角 {total:+8.4f} rad  "
                      f"力矩 {tau:7.4f}")
                last_print = t
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)

        el = time.monotonic() - t0
        for i in range(len(shm_io.JOINTS)):
            shm.zero_gains(i)
        shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
        time.sleep(0.15)                       # 等它停下來再量最終角度
        total += shm_io.wrap_pi(state_ro.states()[idx]["position"] - p_prev)
        dp = total                             # ★ 解纏後的累積轉角
        # ★ 平均速度用角度差分，不用 joint_state 的 velocity 欄位 ——
        #   2026-08-25 實測後者變異 47% vs 前者 9%（無偏但雜訊高 5 倍）
        v_mean = dp / el if el else 0.0
        tau_mean = sum(taus) / len(taus) if taus else 0.0
        return {"name": name, "secs": el, "dpos": dp, "v_mean": v_mean,
                "tau_mean": tau_mean, "tau_max": max((abs(x) for x in taus), default=0.0),
                "friction": a.kd * (a.vel - v_mean), "stop": stop}

    t_freeze = None
    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        t_freeze = time.monotonic()
        time.sleep(0.2)
        print(f"✅ 已凍結 mc_ctrl（狀態={proc_state(pid)}）")
        hold_all()

        for k, name in enumerate(joints):
            if time.monotonic() - t_freeze > a.max_freeze:
                aborted = "總凍結時間超過上限"
                break
            idx = shm_io.idx_of(name)
            print("\n" + "─" * 56)
            print(f"  [{k+1}/{len(joints)}]  {LABEL[name]}　{name}")
            print("─" * 56)
            for c in range(a.countdown, 0, -1):   # 錄影用的倒數，畫面上好抓點
                print(f"     {c}…", flush=True)
                time.sleep(1.0)
                hold_all()
            r = spin(idx, name)
            results.append(r)
            print(f"     → 轉了 {r['dpos']:+.4f} rad（{r['dpos']*57.2958:+.2f}°）"
                  f"　平均 {r['v_mean']:.4f} rad/s")
            if r["stop"]:
                print(f"     ⛔ 提前中止：{r['stop']}")

            if k < len(joints) - 1:
                t_gap = time.monotonic()
                while time.monotonic() - t_gap < a.gap:
                    hold_all()
                    time.sleep(period)

    except KeyboardInterrupt:
        aborted = "使用者 Ctrl-C"
    except Exception as e:
        aborted = f"例外 {e}"
        raise
    finally:
        restore()

    # ---------------------------------------------------------------- 摘要
    print("\n" + "=" * 72)
    if t_freeze:
        print(f"總凍結時間 {time.monotonic() - t_freeze:.1f} 秒")
    if aborted:
        print(f"⛔ 中止：{aborted}")
    if results:
        print(f"\n{'輪子':10s} {'秒':>6s} {'角度變化':>12s} {'平均速度':>11s} "
              f"{'平均力矩':>10s} {'最大力矩':>10s} {'摩擦推估':>10s}")
        for r in results:
            print(f"{LABEL[r['name']]:10s} {r['secs']:6.1f} "
                  f"{r['dpos']:+9.4f} rad {r['v_mean']:11.4f} "
                  f"{r['tau_mean']:10.4f} {r['tau_max']:10.4f} {r['friction']:10.4f}")
        turned = [r for r in results if abs(r["dpos"]) > 0.05]
        print(f"\n轉起來的：{len(turned)}/{len(results)}")
        if len(turned) == len(results):
            print("★★★ 全部驅動成功。")
        else:
            dead = [LABEL[r['name']] for r in results if abs(r['dpos']) <= 0.05]
            print(f"⚠️ 沒轉的：{', '.join(dead)} —— 多半卡靜摩擦，試 --tff 0.5")
        fr = [r["friction"] for r in turned]
        if len(fr) > 1:
            print(f"\n摩擦推估 {min(fr):.3f} ~ {max(fr):.3f} N·m"
                  f"（平均 {sum(fr)/len(fr):.3f}）→ 可拿去填 MJCF 的 frictionloss")
    print(f"\n📄 完整輸出已存到 {logp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
