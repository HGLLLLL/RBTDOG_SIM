#!/usr/bin/env python3
"""讀 M7 的結果 JSON，判定「可不可以做下一步」。

★ 為什麼要有這支：T1（crouch）跑完到 T2（stand）之間那個決策，
  不該靠肉眼掃終端機。要看的東西有五項，其中兩項（吊帶有沒有偷偷承重、
  力矩用掉幾成）**只看螢幕滾過去的即時列印是看不出來的**。

用法：
    bash task7/realbot/pull_from_dog.sh trip8 'M7_*'      # 先收檔
    python3 task7/inference/eval_m7.py task7/logs/m_logs_trip8/M7_*.json

離開碼：0 = 可以往下做；1 = 不要往下做（原因會印出來）。

⚠️ 純標準函式庫。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# 力矩門檻（與 M7_standup.py 的 TMAX 相同；分開寫一份是為了「兩份獨立副本」
# 對照 —— 若哪天 M7 改了門檻而這裡沒改，下面的 check 會印出不一致）
TMAX = {"1_hip_roll": 45.0, "2_hip_pitch": 40.0, "3_knee_pitch": 65.0}

# 原廠實機的參考值（2026-08-26 15:47 全程錄製，|τ| N·m）
FACTORY = {
    "HOLD_crouch": {"knee": (27.3, 29.8), "hip": (15.0, 17.9)},
    "HOLD_stand":  {"knee": (8.3, 10.3),  "hip": (0.4, 3.3)},
}
# 分段峰值（原廠實測）
FACTORY_PEAK = {"crouch": 35.41, "stand": 42.45}

# 判定門檻
PEAK_BUDGET = 0.80        # 峰值用掉超過這個比例 → 不要往下做
ERR_MAX = 0.10            # HOLD 期間的追蹤誤差 rad
HARNESS_FRAC = 0.70       # HOLD_crouch 膝力矩低於原廠下限的這個比例 → 吊帶在承重


def kind(j):
    return j[2:]


def load(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema") != "m7_standup/1":
        raise SystemExit(f"❌ {path.name} 不是 M7 的結果檔（schema={d.get('schema')!r}）")
    return d


# 感測尖峰判別：|τ| 超過控制律上限 kp·|err| + kd·|v| 的這個倍數 → 不是我們下的力矩
SPIKE_RATIO = 1.5


def classify_peak(d, joint, value):
    """把 `peak` 裡的一筆分成 真實 / 感測尖峰 / 無法核對。

    ★ 判別式：`kp·|err| + kd·|v|` 是我們的 PD 律**所能產生的力矩上限**。
      實測 τ 遠大於它 → 那力矩不是我們下的（外力或感測尖峰）。

    2026-08-27 T1 第二趟：`fr3_knee` 吐一筆 −51.49，同刻 cap 只有 21.26（2.42 倍）、
    位置前後六筆完全相同、方向與穩態相反、其他三個膝沒跟著跳
    → 單筆感測垃圾。同款事件 2026-08-26 已在 RAMP_DOWN 抓過一次（也是 fr3）。

    回傳 ("real"|"spike"|"unknown", 說明)。
    """
    kp = d["args"].get("kp", 250.0)
    kd = d["args"].get("kd", 5.0)
    for s in d.get("hold_samples", []):
        if joint not in s or abs(s[joint][2] - value) > 0.01:
            continue
        q, des, tau, v = s[joint]
        cap = kp * abs(q - des) + kd * abs(v)
        if abs(tau) > SPIKE_RATIO * cap:
            return "spike", (f"同刻控制律上限只有 {cap:.2f}"
                             f"（{abs(tau)/max(cap,1e-6):.2f} 倍）")
        return "real", f"同刻控制律上限 {cap:.2f}，對得上"
    return "unknown", "發生在未取樣的區段（GO/BACK），無法核對"


def hold_stats(d, phase):
    """回傳 {關節: (平均|τ|, 最大|誤差|)}，只取該 HOLD 區段的後半段（已穩定）。"""
    rows = [s for s in d.get("hold_samples", []) if s.get("phase") == phase]
    if not rows:
        return {}
    rows = rows[len(rows) // 2:]           # 後半段＝穩態
    out = {}
    for j in rows[0]:
        if j == "phase":
            continue
        taus = [abs(r[j][2]) for r in rows]
        errs = [abs(r[j][0] - r[j][1]) for r in rows]
        out[j] = (sum(taus) / len(taus), max(errs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="M7 結果評估：可不可以做下一步")
    ap.add_argument("path", nargs="+", help="M7_*.json（可用萬用字元）")
    a = ap.parse_args()

    paths = sorted({Path(p) for pat in a.path for p in glob.glob(pat)} or
                   {Path(p) for p in a.path})
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"❌ 找不到檔案：{a.path}")
    p = max(paths, key=lambda x: x.stat().st_mtime)
    if len(paths) > 1:
        print(f"（找到 {len(paths)} 個，用最新的）")
    d = load(p)

    target = d["args"].get("to", "?")
    print(f"檔案　　{p}")
    print(f"時間　　{d.get('time')}")
    print(f"目標　　{target}　kp={d['args'].get('kp')} kd={d['args'].get('kd')}"
          f"　輪鎖定={'開' if d['args'].get('wheel_lock', False) else '關'}")

    blockers, warnings = [], []

    # ── 1. 有沒有中止 ────────────────────────────────────────────────
    print("\n" + "─" * 68)
    if d["aborted"]:
        print(f"⛔ **中止**：{d['abort_reason']}")
        blockers.append("這一趟中止了 —— 先弄清楚原因，不要直接做下一步")
    else:
        print("✅ 序列完整跑完，沒有中止")

    # ── 2. 峰值力矩用掉幾成 ─────────────────────────────────────────
    print(f"\n{'關節':16s} {'峰值τ':>9s} {'門檻':>7s} {'用掉':>7s}  判別")
    worst, real_pk, spikes = 0.0, 0.0, []
    for j, v in d["peak"].items():
        lim = TMAX[kind(j)]
        frac = abs(v) / lim
        verdict, why = classify_peak(d, j, v)
        note = {"real": "", "spike": f"  ⚡感測尖峰（{why}）",
                "unknown": ""}[verdict]
        if verdict == "spike":
            spikes.append((j, v, why))
        else:
            worst = max(worst, frac)
            real_pk = max(real_pk, abs(v))
            if frac > PEAK_BUDGET:
                note += "  ⚠️"
                blockers.append(f"{j} 峰值用掉 {100*frac:.0f}%"
                                f"（>{100*PEAK_BUDGET:.0f}%）" +
                                ("　※ 未取樣區段，無法核對真偽"
                                 if verdict == "unknown" else ""))
        print(f"{j:16s} {v:+9.2f} {lim:7.0f} {100*frac:6.0f}%{note}")

    if spikes:
        print(f"\n⚡ {len(spikes)} 筆判定為**感測尖峰，已排除**"
              f"（`kp·|err|+kd·|v|` 是控制律的力矩上限，超過就不是我們下的）：")
        for j, v, why in spikes:
            print(f"   {j} {v:+.2f} —— {why}")
        warnings.append(f"{len(spikes)} 筆感測尖峰已排除 —— `effort` 欄位會偶發單筆垃圾，"
                        f"這是已知現象（2026-08-26 起第 2 次）")

    ref = FACTORY_PEAK.get(target)
    print(f"\n扣掉感測尖峰後的峰值 {real_pk:.2f} N·m" +
          (f"（原廠做同樣動作 {ref:.2f}，比值 {real_pk/ref:.2f}×）" if ref else ""))
    if ref and real_pk > 1.5 * ref:
        warnings.append(f"峰值 {real_pk:.1f} 是原廠 {ref:.1f} 的 "
                        f"{real_pk/ref:.1f} 倍 —— 動作不一樣或有卡住")

    # ── 3. HOLD 期間：吊帶有沒有偷偷承重 ────────────────────────────
    phase = f"HOLD_{target}"
    st = hold_stats(d, phase)
    if not st:
        warnings.append(f"結果檔裡沒有 {phase} 的取樣 —— 沒走到那一段？")
    else:
        fref = FACTORY.get(phase)
        print(f"\n── {phase} 穩態（後半段平均）" + ("─" * 40))
        print(f"{'關節':16s} {'平均|τ|':>9s} {'最大|誤差|':>11s}")
        knees = []
        for j in sorted(st):
            tau, err = st[j]
            print(f"{j:16s} {tau:9.2f} {err:11.4f}")
            if kind(j) == "3_knee_pitch":
                knees.append(tau)
            if err > ERR_MAX:
                blockers.append(f"{j} 在 {phase} 的追蹤誤差 {err:.3f} rad 超過 {ERR_MAX}")
        if fref and knees:
            lo, hi = fref["knee"]
            avg = sum(knees) / len(knees)
            print(f"\n膝平均 {avg:.2f} N·m　原廠實測 {lo:.1f}–{hi:.1f}")
            if avg < lo * HARNESS_FRAC:
                blockers.append(
                    f"★ {phase} 膝力矩只有 {avg:.1f}，原廠是 {lo:.1f}–{hi:.1f}"
                    f" → **吊帶在幫忙承重，腿沒有真的吃到載重**")
            elif avg < lo:
                warnings.append(
                    f"{phase} 膝力矩 {avg:.1f} 略低於原廠 {lo:.1f}"
                    f" —— 吊帶可能輕微吃力，看得到的話確認一下鬆緊")
            else:
                print("✅ 與原廠同量級 → 腿確實在承重，吊帶是鬆的")

    # ── 4. 起始姿勢：膝模式對不對 ───────────────────────────────────
    q = d.get("q_lie", {})
    if q:
        fr = [q[j] for j in q if kind(j) == "3_knee_pitch" and j[:2] in ("fl", "fr")]
        rr = [q[j] for j in q if kind(j) == "3_knee_pitch" and j[:2] in ("bl", "br")]
        if fr and rr:
            same = all((f > 0) == (r > 0) for f in fr for r in rr)
            print(f"\n起始膝角　前 {['%+.3f' % x for x in fr]}"
                  f"　後 {['%+.3f' % x for x in rr]}")
            if same:
                blockers.append("★ 前後膝同號 = knee_back 模式，先用 M5 喬回來")
            else:
                print("✅ 前後反號（後腿往前彎，原廠預設）")

    # ── 判定 ────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    for w in warnings:
        print(f"⚠️ {w}")
    if blockers:
        print("\n❌ **不要做下一步。** 原因：")
        for b in blockers:
            print(f"   - {b}")
        return 1
    nxt = {"crouch": "T2：sudo python3 M7_standup.py --to stand --confirm",
           "stand": "站立已完成 —— 接下來是 --stay 長時間站立、或自己的步態"}
    print(f"\n✅ **可以做下一步**（峰值最多用掉 {100*worst:.0f}%）")
    print(f"   → {nxt.get(target, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
