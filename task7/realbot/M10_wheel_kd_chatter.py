#!/usr/bin/env python3
"""M10 —— 輪阻尼抖振測試（trip17 事故的分辨實驗）。

═══════════════════════════════════════════════════════════════════
為什麼要有這支（2026-09-03）
═══════════════════════════════════════════════════════════════════
trip17：`wheel_kd=3.0` 起身時四輪 ~64 Hz 抖振（刺耳聲、急停收場）。
抖振發生在「承重＋滾動」。兩個候選機制，處置完全不同：

  H1 **driver 離散阻尼迴路不穩**（velocity 雜訊 47% × kd 放大 × 1 kHz 離散）
     → 懸空也會抖 → kd=3.0 **直接出局**，回 kd≤1.0 重掃步態
  H2 **地面黏滑（stick-slip）** → 懸空不抖 → 「滾動中」的步態或許還有機會，
     但要從 1.0 逐步上探

本測試：**墊高離地、腿全程洩力、只給輪阻尼**，從 kd=0.5 逐級升到 3.0。
每一級先靜置看會不會**自激**，再請操作者**用手撥輪**看受擾後會不會**自持振盪**。
懸空、無載、逐級升 —— 這是零風險版的 trip17 復現實驗。

用法：
    python3 M10_wheel_kd_chatter.py                 # 乾跑（唯讀）
    sudo python3 M10_wheel_kd_chatter.py --confirm
    sudo python3 M10_wheel_kd_chatter.py --confirm --kds 0.5,1.0,2.0,3.0,4.0

⚠️ 撥輪**輕撥即可**（轉起來 3~5 rad/s 就夠）。kd=3.0 時煞車力矩 = 3×v，
   甩到 10 rad/s 就是 30 N·m（輪上限 33）。保護：kd·|v| > 25 立即歸零該級中止。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import shm_io
from M2_wheel_spin import mc_ctrl_pid, proc_state
from M9_gait import ChatterWatch, KeyWatch

QUIET_S = 5.0        # 每級靜置觀察（自激偵測）
FLICK_S = 20.0       # 每級撥輪窗口（Enter 可提早結束）
TAU_IMPLIED_MAX = 25.0   # kd·|v| 超過就歸零（輪上限 33，留 25%）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--kds", type=str, default="0.5,1.0,2.0,3.0",
                    help="逐級測的 kd（由小到大；一抖就停止升級）")
    ap.add_argument("--hz", type=float, default=200.0)
    a = ap.parse_args()
    kds = [float(v) for v in a.kds.split(",")]
    assert kds == sorted(kds), "--kds 必須由小到大（一抖就停，順序反了會漏掉安全級）"

    logp = shm_io.start_log("M10")
    print("M10 —— 輪阻尼抖振測試（trip17 分辨實驗：driver 迴路不穩 vs 地面黏滑）")
    print(f"   逐級 kd：{kds}　每級 靜置 {QUIET_S:.0f}s ＋ 撥輪 {FLICK_S:.0f}s")
    print(f"   保護：kd·|v| > {TAU_IMPLIED_MAX} N·m 立即歸零；腿全程洩力")
    print("\n   ⚠️ 確認：狗肚子墊高、**四輪完全離地可自由轉**、手邊有急停。")
    print("   ⚠️ 撥輪輕撥（3~5 rad/s 就夠），不要用力甩。\n")

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
    widx = {w: shm_io.idx_of(w) for w in shm_io.WHEELS}
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    period = 1.0 / a.hz
    samples: list = []
    results: list = []
    abort = ""

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

    def run_phase(kd: float, name: str, secs: float,
                  allow_skip: bool) -> dict:
        """跑一個觀察窗。回傳 {chatter, per-wheel 統計}。全程維持心跳。"""
        cw = ChatterWatch()
        stat = {w: {"v_peak": 0.0, "flips": 0, "score_max": 0} for w in shm_io.WHEELS}
        prev_v = {w: 0.0 for w in shm_io.WHEELS}
        chattered = None
        t0 = time.monotonic()
        nxt = t0
        last = -1.0
        while True:
            t = time.monotonic() - t0
            if t >= secs:
                break
            if allow_skip and kw.pressed():
                print("  （提早結束此窗口）")
                break
            st = state_ro.states()
            for w, wi in widx.items():
                v = st[wi]["velocity"]
                s_ = stat[w]
                s_["v_peak"] = max(s_["v_peak"], abs(v))
                if v * prev_v[w] < 0:
                    s_["flips"] += 1
                prev_v[w] = v
                if cw.feed(w, v) and chattered is None:
                    chattered = (w, t)
                s_["score_max"] = max(s_["score_max"], cw.score.get(w, 0))
                # 保護：這個 kd 配這個速度，隱含煞車力矩太大 → 立刻收
                if kd * abs(v) > TAU_IMPLIED_MAX and chattered is None:
                    chattered = (w, t)
                    print(f"  ⚠️ {w} kd·|v| = {kd * abs(v):.0f} N·m 超過"
                          f" {TAU_IMPLIED_MAX} —— 歸零保護")
                samples.append({"t": round(time.monotonic(), 4), "kd": kd,
                                "phase": name, "w": w,
                                "pos": round(st[wi]["position"], 4),
                                "v": round(v, 3),
                                "eff": round(st[wi]["effort"], 2)})
            if chattered:
                break
            # 寫入：腿 12 顆壓零、四輪只給阻尼（des = 當下角，kp=0）
            for j, ji in ((j, shm_io.idx_of(j)) for j in shm_io.JOINTS
                          if j not in shm_io.WHEELS):
                shm.zero_gains(ji)
            for w, wi in widx.items():
                shm.write_cmd(wi, position=st[wi]["position"], velocity=0.0,
                              effort=0.0, kp=0.0, kd=kd)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

            if t - last >= 1.0:
                vs = "  ".join(f"{w[:3]} {stat[w]['v_peak']:4.1f}"
                               for w in shm_io.WHEELS)
                print(f"  {name:>6s} kd={kd:g} t={t:4.1f}s  |v|峰: {vs}")
                last = t
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
        return {"chatter": chattered, "stat": stat}

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.2)
        print(f"✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")

        for kd in kds:
            print(f"═══ kd = {kd:g} ═══")
            print(f"  ① 靜置 {QUIET_S:.0f}s（看會不會自激）——不要碰輪子")
            q = run_phase(kd, "QUIET", QUIET_S, allow_skip=False)
            if q["chatter"]:
                w, t = q["chatter"]
                results.append({"kd": kd, "verdict": "SELF_EXCITED",
                                "wheel": w, "t": round(t, 2)})
                print(f"\n  ⛔ kd={kd:g} **靜置自激**（{w} @ {t:.1f}s）")
                print("  → H1 成立：driver 離散阻尼迴路不穩。停止升級。")
                break
            print(f"  ② 撥輪 {FLICK_S:.0f}s —— 依序輕撥四顆輪，看停不停得下來。"
                  f"撥完按 Enter 提早結束")
            f = run_phase(kd, "FLICK", FLICK_S, allow_skip=True)
            if f["chatter"]:
                w, t = f["chatter"]
                results.append({"kd": kd, "verdict": "RING_SUSTAINED",
                                "wheel": w, "t": round(t, 2)})
                print(f"\n  ⛔ kd={kd:g} **受擾後自持振盪**（{w} @ {t:.1f}s）")
                print("  → 邊界不穩。停止升級。")
                break
            results.append({"kd": kd, "verdict": "OK",
                            "stat": {w: f["stat"][w] for w in shm_io.WHEELS}})
            print(f"  ✅ kd={kd:g} 通過（|v|峰 " +
                  " ".join(f"{w[:3]}={f['stat'][w]['v_peak']:.1f}"
                           for w in shm_io.WHEELS) + "）\n")
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    finally:
        restore()

    print("\n" + "=" * 60)
    print("結論")
    print("=" * 60)
    ok_kds = [r["kd"] for r in results if r["verdict"] == "OK"]
    bad = [r for r in results if r["verdict"] != "OK"]
    if bad:
        b = bad[0]
        print(f"  懸空最高穩定 kd = {max(ok_kds) if ok_kds else '（一級都沒過）'}")
        print(f"  kd={b['kd']:g} {b['verdict']}（{b['wheel']}）")
        print("  → **H1：driver 迴路在該 kd 不穩，與地面無關。**")
        print("    該 kd 出局；步態要用 kd ≤ 懸空最高穩定值，並重掃 x_off。")
    elif ok_kds:
        print(f"  懸空全部通過（最高測到 kd={max(ok_kds):g}）")
        print("  → **H2：trip17 的抖振需要地面（黏滑）才發生。**")
        print("    下一步：墊高放回地面、輪貼地但不承重 → 逐級重測；")
        print("    或直接在起身流程用階段排程（M9 已改）從 kd=1.0 上探。")
    if abort:
        print(f"  ⚠️ 測試被中斷：{abort}")

    out = {"schema": "m10_chatter/1", "kds": kds, "results": results,
           "aborted": abort, "samples": samples}
    dp = logp.replace(".log", ".json")
    with open(dp, "w") as fp:
        json.dump(out, fp)
    print(f"\n📊 {dp}\n📄 {logp}")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
