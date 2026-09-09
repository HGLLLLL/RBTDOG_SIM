#!/usr/bin/env python3
"""M11 —— 輪子速度伺服的系統辨識（2026-09-09，為「輪行模式」的模擬器建模）。

★ 前置條件：**狗肚子墊高、四輪完全離地可自由轉**、腿全程洩力、手邊有急停。
  風險等級同 M10：輪子空轉不推動機身；最壞情況是輪子亂轉，保護會歸零。

要量的東西（每一項都對應模擬器裡的一個參數，見 inference/wheel_sysid.py）：
  1. 速度階躍響應：上升時間、時間常數 → 轉動慣量 J 與有效頻寬
  2. 穩態誤差 vs 速度、vs 前饋 → 庫侖摩擦 τ_f、黏滯 b（純 kd 伺服的穩態誤差 = (τ_f + b·v)/kd）
  3. 掃頻（chirp）→ 頻寬與相位落後（含 driver 1 kHz 離散與通訊延遲）
  4. 速度量測雜訊（靜止與定速下的 std）

實機硬約束：kd ≥ 2 會讓 driver 迴路自激（M10 定案）→ 本檔 kd 上限 1.0，要更高得帶 --allow-kd-high。
速度指令的執行方式與 M9 步態相同：kp=0、velocity=v_des、kd、effort=τ_ff → τ = kd·(v_des − v) + τ_ff。

用法（狗上，需 root）：
    python3 M11_wheel_sysid.py                              # 乾跑：只印計畫
    sudo python3 M11_wheel_sysid.py --proto step --kd 0.5 --confirm
    sudo python3 M11_wheel_sysid.py --proto step --kd 1.0 --confirm
    sudo python3 M11_wheel_sysid.py --proto ff   --kd 0.5 --confirm
    sudo python3 M11_wheel_sysid.py --proto chirp --kd 1.0 --confirm
    sudo python3 M11_wheel_sysid.py --proto noise --kd 0.5 --confirm
輸出 ~/m_logs/M11_*.json（每 tick 四輪 pos/v/eff ＋ 指令），本機跑 inference/wheel_sysid.py 擬合。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import shm_io
from M2_wheel_spin import mc_ctrl_pid, proc_state
from M9_gait import ChatterWatch, KeyWatch

KD_SAFE_MAX = 1.0        # M10：kd 2.0 懸空靜置自激；1.0 乾淨
V_LEVELS = (0.5, 1.5, 3.0, 5.0)     # rad/s；r=0.09 m → 0.045/0.135/0.27/0.45 m/s
FF_LEVELS = (0.0, 0.10, 0.15, 0.20, 0.30)   # N·m；實測滾動摩擦 0.13–0.23
STEP_HOLD = 2.0          # 每個階躍保持秒數（時間常數預期 < 0.3 s）
ZERO_HOLD = 1.5


def plan(proto: str, levels, ff_levels, v_ff: float, chirp):
    """→ [(name, v_des, tau_ff, secs)]。tau_ff 的符號跟 v_des 同向。"""
    seq = []
    if proto == "step":
        for v in levels:
            seq += [("zero", 0.0, 0.0, 1.0), (f"+{v:g}", v, 0.0, STEP_HOLD), ("zero", 0.0, 0.0, ZERO_HOLD),
                    (f"-{v:g}", -v, 0.0, STEP_HOLD), ("zero", 0.0, 0.0, ZERO_HOLD)]
    elif proto == "ff":
        for f in ff_levels:
            seq += [("zero", 0.0, 0.0, 1.0), (f"ff{f:g}", v_ff, f, STEP_HOLD), ("zero", 0.0, 0.0, 1.0)]
    elif proto == "noise":
        seq += [("hold0", 0.0, 0.0, 5.0), ("hold2", 2.0, 0.15, 6.0), ("zero", 0.0, 0.0, 1.5)]
    elif proto == "chirp":
        A, f0, f1, T = chirp
        seq += [("zero", 0.0, 0.0, 1.0), ("chirp", A, 0.0, T), ("zero", 0.0, 0.0, 1.5)]
    else:
        raise ValueError(proto)
    return seq


def chirp_v(t: float, A: float, f0: float, f1: float, T: float) -> float:
    """線性掃頻：瞬時頻率 f0 → f1。"""
    k = (f1 - f0) / T
    return A * math.sin(2 * math.pi * (f0 * t + 0.5 * k * t * t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--proto", choices=("step", "ff", "noise", "chirp"), default="step")
    ap.add_argument("--kd", type=float, default=0.5, help="速度伺服增益（上限 1.0，見 M10）")
    ap.add_argument("--allow-kd-high", action="store_true", dest="allow_kd_high",
                    help="允許 kd > 1.0（M10 已證明 2.0 會自激；除非要重現，不要帶）")
    ap.add_argument("--levels", default=",".join(f"{v:g}" for v in V_LEVELS), help="step 的速度階梯 rad/s")
    ap.add_argument("--ff-levels", default=",".join(f"{v:g}" for v in FF_LEVELS), dest="ff_levels")
    ap.add_argument("--v-ff", type=float, default=1.5, dest="v_ff", help="ff 協定的固定速度")
    ap.add_argument("--chirp", default="1.5,0.2,4.0,20", help="A,f0,f1,T（rad/s, Hz, Hz, s）")
    ap.add_argument("--wheels", default="all", help="all 或逗號分隔（fl4_foot,...）")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--vmax", type=float, default=8.0, help="量測速度保護 rad/s（馬達上限 125）")
    ap.add_argument("--tmax", type=float, default=5.0, help="力矩保護 N·m（規格 33）")
    a = ap.parse_args()

    if a.kd > KD_SAFE_MAX and not a.allow_kd_high:
        print(f"❌ kd {a.kd} > {KD_SAFE_MAX}：M10 定案 kd≥2 自激。要重現請帶 --allow-kd-high。")
        return 1
    levels = [float(x) for x in a.levels.split(",")]
    ff_levels = [float(x) for x in a.ff_levels.split(",")]
    chirp = tuple(float(x) for x in a.chirp.split(","))
    assert len(chirp) == 4
    if max(levels) > a.vmax * 0.8 or chirp[0] > a.vmax * 0.8:
        print(f"❌ 指令速度太接近保護 --vmax {a.vmax}（要 ≤ 80%）")
        return 1
    wheels = list(shm_io.WHEELS) if a.wheels == "all" else [w.strip() for w in a.wheels.split(",")]
    for w in wheels:
        assert w in shm_io.WHEELS, w
    seq = plan(a.proto, levels, ff_levels, a.v_ff, chirp)
    total = sum(s[3] for s in seq)

    logp = shm_io.start_log("M11")
    print(f"M11 —— 輪子速度伺服系統辨識　協定 {a.proto}　kd {a.kd:g}　輪 {wheels}")
    print(f"   序列 {len(seq)} 段、共 {total:.1f} s：" + " → ".join(f"{n}({s:g}s)" for n, _, _, s in seq))
    print(f"   保護：|v| > {a.vmax} rad/s 或 |τ| > {a.tmax} N·m 或抖振 → 歸零；腿全程洩力")
    print("\n   ⚠️ 確認：狗肚子墊高、**四輪完全離地可自由轉**、手邊有急停。\n")

    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過")
    live = [c["name"] for c in shm_io.read_joint_cmd()
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節帶著非零增益，先洩力：{live}")
        return 1
    print("✅ 16 顆全部洩力中")
    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")
    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結、沒有寫入。")
        return 0
    if os.geteuid() != 0:
        print("❌ 需要 root：加 sudo")
        return 1

    kw = KeyWatch()
    widx = {w: shm_io.idx_of(w) for w in wheels}
    leg_idx = [shm_io.idx_of(j) for j in shm_io.JOINTS if j not in shm_io.WHEELS]
    other_wheels = {w: shm_io.idx_of(w) for w in shm_io.WHEELS if w not in widx}
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    samples: list = []
    abort = ""
    period = 1.0 / a.hz
    cw = ChatterWatch()

    def restore():
        try:
            for i in range(len(shm_io.JOINTS)):
                shm.zero_gains(i)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
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
            print(f"✅ 已解凍 mc_ctrl，狀態={proc_state(pid)}")

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.2)
        print(f"✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")
        t_run0 = time.monotonic()
        nxt = t_run0
        for name, v_cmd, tff, secs in seq:
            print(f"── {name:>7s}  v_des {v_cmd:+.2f} rad/s  τ_ff {tff:+.2f}  {secs:g} s")
            t_seg0 = time.monotonic()
            last_print = -1.0
            while True:
                t = time.monotonic() - t_seg0
                if t >= secs:
                    break
                if kw.pressed():
                    abort = "使用者按 Enter"
                    break
                if name == "chirp":
                    v_des = chirp_v(t, *chirp)
                    ff = 0.0
                else:
                    v_des = v_cmd
                    ff = math.copysign(tff, v_cmd) if v_cmd else 0.0
                st = state_ro.states()
                tick = state_ro.read_tick(shm_io.STATE_STRIDE)
                rec = {"t": round(time.monotonic() - t_run0, 4), "seg": name, "v_des": round(v_des, 4),
                       "ff": round(ff, 3), "kd": a.kd}
                for w, wi in widx.items():
                    p, v, e = st[wi]["position"], st[wi]["velocity"], st[wi]["effort"]
                    rec[w] = (round(p, 4), round(v, 4), round(e, 3))
                    if abs(v) > a.vmax:
                        abort = f"{w} 速度 {v:+.1f} 超過 {a.vmax}"
                    if abs(e) > a.tmax:
                        abort = f"{w} 力矩 {e:+.1f} 超過 {a.tmax}"
                    if cw.feed(w, v):
                        abort = f"{w} 抖振（高頻正負翻轉）"
                samples.append(rec)
                if abort:
                    break
                for ji in leg_idx:
                    shm.zero_gains(ji)
                for w, wi in other_wheels.items():          # 沒選的輪子：純阻尼 0.5 靜止
                    shm.write_cmd(wi, position=st[wi]["position"], velocity=0.0, effort=0.0, kp=0.0, kd=0.5)
                for w, wi in widx.items():
                    shm.write_cmd(wi, position=st[wi]["position"], velocity=v_des, effort=ff, kp=0.0, kd=a.kd)
                shm.write_tick(tick)
                if t - last_print >= 0.5:
                    vs = "  ".join(f"{w[:2]} {st[wi]['velocity']:+5.2f}/{st[wi]['effort']:+5.2f}" for w, wi in widx.items())
                    print(f"   t={t:4.1f}  des {v_des:+5.2f}  v/τ: {vs}")
                    last_print = t
                nxt += period
                d = nxt - time.monotonic()
                if d > 0:
                    time.sleep(d)
                elif d < -3 * period:
                    nxt = time.monotonic()
            if abort:
                print(f"\n⛔ 中止：{abort}")
                break
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    finally:
        restore()

    out = {"schema": "m11_wheel_sysid/1", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a),
           "proto": a.proto, "kd": a.kd, "wheels": wheels, "seq": seq, "chirp": chirp,
           "aborted": abort, "samples": samples}
    dp = logp.replace(".log", ".json")
    with open(dp, "w") as fp:
        json.dump(out, fp)
    n = len(samples)
    print(f"\n{'⚠️ 中止：' + abort if abort else '✅ 序列跑完'}　{n} 筆（{n / max(1e-6, samples[-1]['t'] if samples else 1):.0f} Hz）")
    print(f"📊 {dp}\n📄 {logp}")
    print("本機：conda run -n rbtdog python task7/inference/wheel_sysid.py task7/logs/m_logs_tripNN/M11_*.json")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
