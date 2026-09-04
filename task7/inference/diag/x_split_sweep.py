#!/usr/bin/env python3
"""前後分離的 `x_off`：姿態對稱度 ↔ 步態效果的取捨曲線。

2026-09-03 的問題：昨天選定的 `kd_wheel=3.0 / x_off=−110 mm` 讓狗真的走起來了，
但**前後腿姿態明顯不對稱** —— 站姿前腿 hip +58.5°、後腿 −28.0°，差 30.5°
（目視就是「前腳大腿與機身的夾角比較小」）。

把 `x_off` 拆成兩個正交的軸（`cpg_max.x_off_split`）：

    x_c   四腿共同平移 ＝ 支撐多邊形中心相對機身的位移 ＝ **配平點**
    x_d   前後差動     ＝ 半軸距增量（足端 wheelbase 變化 2·x_d）＝ **軸距**

★ 幾何上 `f0` 本身已是前後鏡像，所以**姿態對稱 ⟺ x_c = 0**，與 `x_d` 無關
（`test_x_c_zero_gives_front_rear_symmetric_stance` 釘住）。
也就是說「保留配平又要對稱」不存在 —— 它們是同一個自由度。

**所以本腳本要回答的是：`x_d` 能不能在 `x_c` 往 0 收的時候把效果買回來。**
⚠️ 誠實的先驗：`x_d` 不動配平，所以它很可能買不回來。掃它是為了讓
「不對稱是配平量的必然代價」這句話有證據，而不是靠推論。

判準（使用者 2026-09-03 選定，三項都要過）：

    速度 ≥ 0.234 m/s      現行 0.26，容許掉 10%
    前腳執行率 ≥ 0.70     現行 0.79；舊基準 0.03 就是「前腳幾乎不跨步」
    後膝峰值 ≤ 48.2 N·m   現行值。⚠️ 模擬對觸地衝擊一路低估 1.46–1.52 倍，
                          實機門檻 70 —— 這一項不能放寬

⚠️ 三個一定要一起看的陷阱（都在這條線上踩過）：

1. **`kd_wheel` 與 `x_off` 是耦合的**：單獨動任一個都幾乎沒效果（+15% / +38%），
   一起動 +76%。所以候選點必須**重掃 `kd_wheel`**，不能假設 3.0 還是最佳。
2. **單次數字會選到混沌點**：`kd=8` 的 20 s 速度最好（0.271），但 1e-12 擾動下
   偏航全距 31.8°。候選點一律用皮米級擾動複驗。
3. **指標乾淨 ≠ 系統正確**：超限／飽和／IK 縮限全 0 的情況下，前腳仍可能
   完全不跨步。執行率是唯一會露餡的那一項，必須印。

用法：
    PY=/home/huang/miniforge3/envs/rbtdog/bin/python
    $PY task7/inference/diag/x_split_sweep.py            # 階段 1：5×5 粗掃
    $PY task7/inference/diag/x_split_sweep.py --verify    # 階段 2：候選點複驗
    $PY task7/inference/diag/x_split_sweep.py --verify --candidates -0.08,0.0 -0.11,0.03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "task7" / "inference"))

import cpg_max                      # noqa: E402
import cpg_walk_max as cw           # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402

# 現行點（2026-09-02 選定）
CUR_XC, CUR_XD, CUR_KD = -0.110, 0.0, 3.0

# ★ 門檻**不寫死文件裡的數字**，而是在同一支腳本、同一版程式下先跑一次現行點再算。
#   文件記的後膝是 48.2，本腳本跑出 48.5 —— 用 48.2 當門檻會讓現行點自己不及格，
#   然後所有候選點都被判「比現行差」。這條線已經因為「比錯條件」更正過四次
#   （速度高估 68%、kp 120 vs 250 的落後換算…），門檻本身也適用同一條規矩。
LIM = {"speed": 0.234, "exec_front": 0.70, "knee_rear": 48.2}    # 先放文件值，開跑時覆寫

# ★ 步態參數（ω/duty/d_step/g_c）由 CLI 覆寫，**現行點與候選點共用同一份**。
#   分成兩份的話門檻就失去意義 —— 那是這條線更正過四次的同一種錯。
GAIT_KW: dict = {}

XC_GRID = (0.0, -0.030, -0.055, -0.080, -0.110)
XD_GRID = (-0.060, -0.030, 0.0, 0.030, 0.060)


def calibrate(secs: float, kd_wheel: float) -> dict:
    """跑現行點（含皮米擾動 ×4），把三項門檻校到「不比它差」。

    ★ 為什麼要跑四次而不是一次：後膝峰值在 1e-12 擾動下自己就有 0.0–0.3 N·m 的變動。
      用單次值當門檻的話，**現行點會因為擾動而不通過自己的門檻** ——
      那不是判準，那是把量測雜訊當成判準。所以取四次的上緣再加 1% 容差。
    ⚠️ 這 1%（≈0.5 N·m）必須跟另一個數字一起看才誠實：模擬對觸地衝擊系統性低估
      **1.46–1.52 倍**。也就是說容差遠小於已知的系統性偏差 —— 它處理的是雜訊，
      不是拿來替候選點開後門。真正的實機門檻是 70 N·m，那才是硬的。
    """
    rs = [run(CUR_XC, CUR_XD, kd_wheel, secs, jitter=j * 1e-12) for j in range(4)]
    knee = max(r["knee_peak_rear"] for r in rs)
    spd = min(r["speed_travel"] for r in rs)
    LIM["speed"] = 0.90 * spd                     # 容許掉 10%
    LIM["exec_front"] = 0.70                      # 絕對值：0.7 是「前腳有在跨步」的線
    LIM["knee_rear"] = knee * 1.01                # 不升（+1% 雜訊容差，見 docstring）
    yaws = [r["yaw"] for r in rs]
    print(f"# 現行點實測（同一版程式、擾動 ×4）：速度 {spd:.3f}"
          f"｜前腳執行率 {np.mean([r['exec_front'] for r in rs]):.2f}｜後膝 {knee:.1f}"
          f"｜偏航 {rs[0]['yaw']:+.1f}°（全距 {max(yaws) - min(yaws):.1f}°）"
          f"｜側偏 {rs[0]['lateral']:+.2f} m｜Δhip {stance_asym(CUR_XC, CUR_XD)[2]:+.1f}°")
    print(f"# → 門檻：速度 ≥{LIM['speed']:.3f}、前腳執行率 ≥{LIM['exec_front']:.2f}、"
          f"後膝 ≤{LIM['knee_rear']:.1f}（現行 {knee:.1f} +1% 雜訊容差）")
    return rs[0]


def stance_asym(x_c: float, x_d: float) -> tuple[float, float, float]:
    """站姿的前後 hip 角與不對稱度（度）。純運動學，不跑模擬。

    回傳 `(hip_front, hip_rear, |hip_front| − |hip_rear|)`。
    這是使用者目視看到的那個量 —— 前腳大腿與機身的夾角是 90° − |hip|。
    """
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    q = cpg_max.stand_targets(ks, f0, cpg_max.x_off_split(x_c, x_d)).reshape(4, 3)
    hf, hr = np.degrees(q[0, 1]), np.degrees(q[2, 1])
    return float(hf), float(hr), float(abs(hf) - abs(hr))


def run(x_c: float, x_d: float, kd_wheel: float, secs: float,
        jitter: float = 0.0, **gait_kw) -> dict:
    """跑一格。`jitter` 是加在 `x_c` 上的皮米級擾動，用來複驗混沌。

    `gait_kw`（omega / duty / d_step / g_c…）直接轉給 `rollout`。
    ⚠️ 改了任何一項就必須讓 `calibrate` 吃同一組 —— 否則現行點與候選點
    跑的是兩種步態，門檻就失去意義（這條線「比錯條件」已經更正過四次）。
    """
    x_off = cpg_max.x_off_split(x_c + jitter, x_d)
    return cw.rollout(gait="walk", secs=secs, x_off=x_off, kd_wheel=kd_wheel,
                      quiet=True, **{**GAIT_KW, **gait_kw})


def passes(r: dict) -> tuple[bool, str]:
    """三項門檻＋診斷。回傳 (是否全過, 未過項目的簡短標記)。"""
    bad = []
    if r["fell"] is not None:
        bad.append("跌倒")
    if r["speed_travel"] < LIM["speed"]:
        bad.append("速")
    if r["exec_front"] < LIM["exec_front"]:
        bad.append("執")
    if r["knee_peak_rear"] > LIM["knee_rear"]:
        bad.append("膝")
    # 診斷不是使用者選的三項，但沉默的 IK 縮限會表現成「步態突然變鈍」而找不到原因，
    # 而 x_d 是在改足端幾何 —— 正是會把 IK 推到邊界的那種改動。
    if r["reach_pct"] > 0 or r["lim_pct"] > 0 or r["tau_pct"] > 0:
        bad.append("診斷")
    return (not bad), ("" if not bad else "✗" + "".join(bad))


HDR = (f"{'x_c':>6}{'x_d':>6} | {'Δhip°':>7}{'速度':>7}{'執行前':>7}{'執行後':>7}"
       f"{'後膝':>7}{'前膝':>7}{'偏航°':>7}{'俯仰':>7}{'彈跳':>7}{'離地':>7}"
       f"{'縮限%':>7} | 判定")


def fmt(x_c: float, x_d: float, r: dict) -> str:
    ok, why = passes(r)
    _, _, dh = stance_asym(x_c, x_d)
    return (f"{x_c * 1000:>6.0f}{x_d * 1000:>6.0f} | {dh:>7.1f}"
            f"{r['speed_travel']:>7.3f}{r['exec_front']:>7.2f}{r['exec_rear']:>7.2f}"
            f"{r['knee_peak_rear']:>7.1f}{r['knee_peak_front']:>7.1f}"
            f"{r['yaw']:>7.1f}{r['pitch_cycle']:>7.2f}{r['bounce'] * 1000:>7.1f}"
            f"{r['min_lift'] * 1000:>7.1f}{r['reach_pct']:>7.2f} | "
            + ("✅" if ok else why))


def stage1(secs: float, kd_wheel: float, xc_grid=XC_GRID,
           xd_grid=XD_GRID) -> list[tuple[float, float, dict]]:
    print(f"# 階段 1 粗掃：kd_wheel={kd_wheel}、{secs:.0f} s、walk"
          f"（{len(xc_grid)}×{len(xd_grid)} = {len(xc_grid) * len(xd_grid)} 格）")
    calibrate(secs, kd_wheel)
    print(f"# ★ 對稱 ⟺ x_c=0；Δhip 是站姿 |hip前|−|hip後|")
    print(HDR)
    out = []
    for x_c in xc_grid:
        for x_d in xd_grid:
            r = run(x_c, x_d, kd_wheel, secs)
            print(fmt(x_c, x_d, r), flush=True)
            out.append((x_c, x_d, r))
        print("-" * len(HDR))
    return out


def stage2(cands: list[tuple[float, float]], secs: float, kds: tuple,
           kd_cal: float = CUR_KD) -> None:
    """候選點複驗：重掃 kd_wheel（耦合）＋ 皮米擾動 ×4（混沌）。"""
    print(f"\n# 階段 2 複驗：重掃 kd_wheel {kds} ＋ 每格 1e-12 m 擾動 ×4")
    # ★ 門檻同樣要校在現行點上 —— 階段 1 校過而階段 2 沒校的話，
    #   同一組候選點在兩張表會得到不同判定，那比沒有判定更糟。
    calibrate(secs, kd_cal)
    print(f"{'x_c':>6}{'x_d':>6}{'kd':>5} | {'Δhip°':>7}{'速度(4次)':>28}{'執行前':>8}"
          f"{'後膝(全距)':>14}{'偏航全距°':>10} | 判定")
    for x_c, x_d in cands:
        dh = stance_asym(x_c, x_d)[2]
        for kd in kds:
            rs = [run(x_c, x_d, kd, secs, jitter=j * 1e-12) for j in range(4)]
            sp = [r["speed_travel"] for r in rs]
            knee = [r["knee_peak_rear"] for r in rs]
            yaws = [r["yaw"] for r in rs]
            ok = all(passes(r)[0] for r in rs)
            # 混沌判準沿用這條線的做法：偏航全距是最敏感的量
            span = max(yaws) - min(yaws)
            print(f"{x_c * 1000:>6.0f}{x_d * 1000:>6.0f}{kd:>5.1f} | {dh:>7.1f}"
                  f"{'/'.join(f'{v:.3f}' for v in sp):>28}"
                  f"{np.mean([r['exec_front'] for r in rs]):>8.2f}"
                  f"{max(knee):>9.1f}({max(knee) - min(knee):>3.1f})"
                  f"{span:>10.1f} | "
                  + ("✅" if ok else "✗") + (" ⚠️混沌" if span > 6.4 else ""), flush=True)


def chaos_source(secs: float, kd_wheel: float) -> None:
    """混沌是「`x_d` 很負」造成的，還是「速度高了」造成的？

    高對稱區（`x_d=−90`）跑 0.32 m/s，比現行點快 23%。這兩件事在該區是綁在一起的，
    必須拆開才知道要調什麼：

    A 組：現行 `x_off`（`x_d=0`）用 ω 把速度推到 0.32 —— 若也混沌 → 是**速度**
    B 組：高對稱點（`x_d=−90`）降 ω 把速度壓回 0.26 —— 若就穩了 → 也是**速度**
    兩組都指向速度的話，`x_d` 是無辜的，該調的是 ω／duty 而不是放棄對稱。
    """
    print(f"\n# 混沌來源診斷：kd_wheel={kd_wheel}、{secs:.0f} s、每格 1e-12 擾動 ×4")
    print(f"{'組':>3}{'x_c':>6}{'x_d':>6}{'ω':>6}{'duty':>6} | {'速度(平均)':>10}"
          f"{'速度全距':>10}{'偏航全距°':>10}{'彈跳mm':>8}{'支撐腳':>8}{'後膝':>7}")
    cases = ([("A", -0.110, 0.0, om, 0.80) for om in (1.4, 1.6, 1.8, 2.0)]
             + [("B", -0.045, -0.090, om, 0.80) for om in (1.0, 1.2, 1.4)]
             + [("C", -0.045, -0.090, om, 0.85) for om in (1.2, 1.4, 1.6)])
    for tag, x_c, x_d, om, duty in cases:
        rs = [run(x_c, x_d, kd_wheel, secs, jitter=j * 1e-12, omega=om, duty=duty)
              for j in range(4)]
        sp = [r["speed_travel"] for r in rs]
        yaws = [r["yaw"] for r in rs]
        print(f"{tag:>3}{x_c * 1000:>6.0f}{x_d * 1000:>6.0f}{om:>6.1f}{duty:>6.2f} | "
              f"{np.mean(sp):>10.3f}{max(sp) - min(sp):>10.3f}"
              f"{max(yaws) - min(yaws):>10.1f}"
              f"{max(r['bounce'] for r in rs) * 1000:>8.1f}"
              f"{np.mean([r['support'] for r in rs]):>8.2f}"
              f"{max(r['knee_peak_rear'] for r in rs):>7.1f}", flush=True)


def rescue(secs: float, kd_wheel: float, x_c: float = -0.045,
           x_d: float = -0.090) -> None:
    """能不能救回高對稱區：`duty`/`ω` 解掉混沌之後，用 `d_step`/`g_c` 把後膝壓下來。

    混沌來源診斷（`--chaos`）的結論是：`x_d=−90` 要穩必須 `duty 0.85 + ω 1.6`，
    但那組的後膝 87.1（現行 48.5）。觸地衝擊大致隨**足端下落速度**走，
    而它由 `d_step`（水平掃幅）與 `g_c`（垂直落差）× ω 決定 —— 這是僅剩的兩個旋鈕。

    ⚠️ `g_c` 不能亂降：`g_c=0.04` 實際離地只剩 4.5 mm，腳會被地面拖著走，
    而四個診斷指標全乾淨（cpg_max.foot_targets 的警語）。所以一律印離地量。
    ⚠️ 對照組（現行 x_off）用**同一組 duty/ω/d_step/g_c** 重跑，不是拿舊數字比。
    """
    print(f"\n# 救援高對稱區 (x_c={x_c * 1000:.0f}, x_d={x_d * 1000:.0f})："
          f"duty 0.85、ω 1.6、kd_wheel={kd_wheel}、{secs:.0f} s、擾動 ×4")
    print(f"{'組':>4}{'d_step':>8}{'g_c':>7} | {'速度':>8}{'偏航全距°':>10}"
          f"{'後膝':>7}{'前膝':>7}{'離地mm':>8}{'彈跳':>7}{'支撐腳':>7}{'執行前':>7}")
    for tag, xc_, xd_ in (("對稱", x_c, x_d), ("現行", CUR_XC, CUR_XD)):
        for d_step in (0.06, 0.08, 0.10):
            for g_c in (0.06, 0.08):
                rs = [run(xc_, xd_, kd_wheel, secs, jitter=j * 1e-12,
                          omega=1.6, duty=0.85, d_step=d_step, g_c=g_c)
                      for j in range(4)]
                yaws = [r["yaw"] for r in rs]
                print(f"{tag:>4}{d_step:>8.2f}{g_c:>7.2f} | "
                      f"{np.mean([r['speed_travel'] for r in rs]):>8.3f}"
                      f"{max(yaws) - min(yaws):>10.1f}"
                      f"{max(r['knee_peak_rear'] for r in rs):>7.1f}"
                      f"{max(r['knee_peak_front'] for r in rs):>7.1f}"
                      f"{min(r['min_lift'] for r in rs) * 1000:>8.1f}"
                      f"{max(r['bounce'] for r in rs) * 1000:>7.1f}"
                      f"{np.mean([r['support'] for r in rs]):>7.2f}"
                      f"{np.mean([r['exec_front'] for r in rs]):>7.2f}", flush=True)
        print("-" * 88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--kd-wheel", type=float, default=CUR_KD, dest="kd_wheel")
    ap.add_argument("--verify", action="store_true", help="跑階段 2（候選點複驗）")
    ap.add_argument("--rescue", action="store_true",
                    help="救援高對稱區：解掉混沌後用 d_step/g_c 壓後膝")
    ap.add_argument("--chaos", action="store_true",
                    help="混沌來源診斷：是 x_d 造成的還是速度造成的")
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="階段 2 的候選點，格式 x_c,x_d（公尺），例如 -0.08,0.0")
    ap.add_argument("--kds", type=str, default="2.0,3.0,4.0",
                    help="階段 2 重掃的 kd_wheel")
    ap.add_argument("--xc", type=str, default=None, help="覆寫 x_c 網格（逗號分隔，公尺）")
    ap.add_argument("--xd", type=str, default=None, help="覆寫 x_d 網格")
    for k in ("omega", "duty", "d-step", "g-c"):
        ap.add_argument(f"--{k}", type=float, default=None, dest=k.replace("-", "_"),
                        help=f"覆寫步態的 {k}（現行點與候選點會一起吃到）")
    a = ap.parse_args()
    GAIT_KW.update({k: v for k in ("omega", "duty", "d_step", "g_c")
                    if (v := getattr(a, k)) is not None})
    if GAIT_KW:
        print(f"# 步態覆寫（現行點與候選點共用）：{GAIT_KW}")

    if a.rescue:
        rescue(a.secs, a.kd_wheel)
    elif a.chaos:
        chaos_source(a.secs, a.kd_wheel)
    elif not a.verify:
        xc = tuple(float(v) for v in a.xc.split(",")) if a.xc else XC_GRID
        xd = tuple(float(v) for v in a.xd.split(",")) if a.xd else XD_GRID
        rows = stage1(a.secs, a.kd_wheel, xc, xd)
        good = [(c, d) for c, d, r in rows if passes(r)[0]]
        print(f"\n# 過門檻的格子（{len(good)} / {len(rows)}）："
              + ("、".join(f"({c * 1000:.0f},{d * 1000:.0f})" for c, d in good) or "無"))
        # ★ 明講被丟掉的東西：只印「有幾格過」會讓人以為其餘都差不多
        print("# ⚠️ 上表每格只跑一次，未做混沌複驗 —— 過門檻的格子必須再跑 --verify")
    else:
        if a.candidates:
            cands = [tuple(float(v) for v in c.split(",")) for c in a.candidates]
        else:
            cands = [(CUR_XC, CUR_XD)]
        stage2(cands, a.secs, tuple(float(v) for v in a.kds.split(",")))
