#!/usr/bin/env python3
"""鎖輪 vs 阻尼：把輪足當點足看待，在模擬裡到底會怎樣。

2026-09-02 的問題：trip16 顯示 95% 的輪子轉動發生在站立相、前輪 +330 / 後輪 −470
反向對抗 —— 也就是「腳沒踩住」。那把輪子鎖死，是不是就退化成 task3 的點足了？

三個對照組（唯一變因是 `wheel_mode`）：

    damp   輪 kp=0、只有阻尼           ← 現行基準，也是原廠動作時的做法
    hold   輪 kp=60 鎖在固定角          ← 「當點足看」
    free   輪完全不出力                 ← 對照下界，隔離阻尼本身的貢獻

⚠️ 兩個已知陷阱，本腳本都處理：

1. **配平點 `x_off` 會隨輪摩擦移動**（`gait_baseline` 的警語）。鎖輪是比改摩擦
   更劇烈的變動，沿用 `x_off=-0.040` 對 hold 不公平 —— 所以 `--trim` 會替
   每個模式各自重掃配平點（平均俯仰過零），再用各自的配平點比。
2. **實機證據已經反對過鎖輪**：原廠設定檔的 `FSM_RL_Wheel_Kp=60` 套在開迴路上
   實測 **+39° / 12 s 偏航失控**（見 `gait_baseline` 的表）。那個 Kp=60 是搭配
   「每步重給目標角」的 RL 用的。所以本腳本一定要印偏航，而且偏航是主要判準。

用法：
    /home/huang/miniforge3/envs/rbtdog/bin/python \\
        task7/inference/diag/wheel_lock_compare.py --secs 20
    ... --trim          # 每個模式各自重掃 x_off 再比（慢很多，但這才是公平比較）
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
import gait_baseline as gb          # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402

MODES = ("damp", "hold", "free")
IS_FRONT = np.array([True, True, False, False])     # 腿序 FR, FL, RR, RL
KNEE = np.array([2, 5, 8, 11])                      # 12 維裡膝的位置


def run(wheel_mode: str, x_off: float, secs: float, kp3, kd3,
        z_sag: float, d_step: float, kd_wheel: float = None) -> dict:
    """跑一趟，回傳前進/偏航/逐輪滾動/膝力矩峰值。"""
    B = gb.BASELINE
    r = cw.Robot(kp3=kp3, kd3=kd3, kd_wheel=kd_wheel)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    q_stand = cpg_max.stand_targets(ks, f0, x_off)

    r.reset_standing(q_stand, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q_stand, wheel_mode)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()          # hold 模式鎖的就是這一刻的輪角
    r.tau_peak[:] = 0.0              # 站立段不算進步態的力矩峰值

    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    n = int(secs / mm.CTRL_DT)
    d = r.d
    p0 = d.qpos[:3].copy()
    w_prev = d.qpos[mm.WHEEL_QPOS_IDX].copy()
    roll = np.zeros(4)               # 累積滾動（折回 ±π —— 輪角讀數是包裹的）
    TH = np.zeros((n, 4))
    pitch = np.zeros(n)
    heights = np.zeros(n)
    yaw_t = np.zeros(n)

    for i in range(n):
        TH[i] = c["theta"]
        q, _ = cpg_max.joint_targets(c, f0, x_off, B["g_c"], d_step,
                                     B["d_step_y"], B["duty"], ks, z_sag)
        r.step(q, wheel_mode)
        c = step(c, np.full(4, B["mu_x"]), np.full(4, B["mu_y"]),
                 np.full(4, B["omega"]), mm.CTRL_DT)
        w = d.qpos[mm.WHEEL_QPOS_IDX]
        roll += (w - w_prev + np.pi) % (2 * np.pi) - np.pi
        w_prev = w.copy()
        qt = d.qpos[3:7]
        pitch[i] = np.degrees(np.arcsin(np.clip(
            2 * (qt[0] * qt[2] - qt[3] * qt[1]), -1, 1)))
        yaw_t[i] = cpg_max.yaw_deg(qt)
        heights[i] = d.qpos[2]

    fell = bool(d.qpos[2] < 0.25)
    disp = d.qpos[:3] - p0
    st = np.sin(cpg_max.duty_remap(TH, B["duty"])) <= 0
    return {
        "mode": wheel_mode, "x_off": x_off, "fell": fell,
        "dx": float(disp[0]), "dy": float(disp[1]),
        "speed": float(disp[0] / secs),
        "yaw": float(yaw_t[-1]), "yaw_abs_max": float(np.abs(yaw_t).max()),
        "pitch_mean": float(pitch.mean()),
        "height": float(heights.mean()),
        "roll_front": float(roll[:2].mean() * mm.WHEEL_RADIUS),
        "roll_rear": float(roll[2:].mean() * mm.WHEEL_RADIUS),
        "knee_front": float(r.tau_peak[KNEE[:2]].max()),
        "knee_rear": float(r.tau_peak[KNEE[2:]].max()),
        "tau_pct": 100.0 * r.n_tau / max(r.n_tau_tot, 1),
        "lim_pct": 100.0 * r.n_lim / max(r.n_cmd, 1),
        "stance_frac": float(st.mean()),
        "kd_wheel": r.kd_wheel,
    }


def trim_x_off(wheel_mode: str, secs: float, **kw) -> float:
    """替一個模式重掃配平點：平均俯仰過零的 x_off。

    判準沿用 `gait_baseline` 的做法（平均俯仰單調過零），不是自己另發明一個。
    """
    grid = np.arange(-0.070, 0.031, 0.010)
    pts = []
    for x in grid:
        r = run(wheel_mode, float(x), secs, **kw)
        pts.append((float(x), r["pitch_mean"], r["fell"]))
        print(f"    x_off {x * 1000:+5.0f} mm → 平均俯仰 {r['pitch_mean']:+6.2f}°"
              f"{'  ❌跌倒' if r['fell'] else ''}")
    ok = [(x, p) for x, p, f in pts if not f]
    for (x1, p1), (x2, p2) in zip(ok, ok[1:]):
        if p1 == 0 or p1 * p2 < 0:                     # 線性內插過零點
            return x1 + (x2 - x1) * (-p1) / (p2 - p1)
    print("    ⚠️ 沒有過零點 —— 這個模式在掃描範圍內配平不了")
    return min(ok, key=lambda t: abs(t[1]))[0] if ok else 0.0


def sweep_xoff(secs: float, grid: np.ndarray = None, modes=MODES, **kw) -> None:
    """三個模式各自掃 `x_off`，直接看前進距離與偏航。

    ★ 為什麼不用配平判準：`--trim` 實測發現 **鎖輪之後俯仰對 `x_off` 幾乎不敏感**
    （全距只有 0.5°，damp 是 1.7°），過零點落在哪裡幾乎是雜訊決定的。
    用一個已經失去鑑別力的判準去挑參數，會挑出一個看起來有依據、其實是隨機的點 ——
    然後拿它去代表「鎖輪的best case」，等於用錯誤的方式否定掉一個做法。
    所以這裡改成直接掃目標量（前進距離）本身。
    """
    grid = np.arange(-0.070, 0.041, 0.010) if grid is None else grid
    print(f"\n{'x_off mm':>9}", end="")
    for m in modes:
        print(f"│{m + ' 前進m':>11}{'偏航°':>7}{'後膝τ':>7}", end="")
    print()
    best = {m: None for m in modes}
    for x in grid:
        print(f"{x * 1000:>+9.0f}", end="")
        for m in modes:
            r = run(m, float(x), secs, **kw)
            flag = "✗" if r["fell"] else " "
            print(f"│{r['dx']:>10.2f}{flag}{r['yaw']:>7.1f}{r['knee_rear']:>7.1f}", end="")
            if not r["fell"] and (best[m] is None or r["dx"] > best[m]["dx"]):
                best[m] = r
        print()
    print(f"\n{'模式':>6}{'最佳 x_off':>11}{'前進 m':>9}{'速度 m/s':>10}"
          f"{'偏航°':>7}{'側偏 mm':>9}{'後膝 τ':>8}")
    for m in modes:
        r = best[m]
        if r is None:
            print(f"{m:>6}   （全部跌倒）")
            continue
        print(f"{m:>6}{r['x_off'] * 1000:>+11.0f}{r['dx']:>9.2f}{r['speed']:>10.3f}"
              f"{r['yaw']:>7.1f}{r['dy'] * 1000:>9.0f}{r['knee_rear']:>8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    # ⚠️ 預設 = `max_model.KP3` = 原廠分關節增益 [abad 60, hip 120, knee 120]，
    #    也是 BASELINE 配平時用的那組。給單一 kp 會把 ABAD 也拉到 120 ——
    #    ABAD 剛度直接決定側偏與偏航，配平點會整個跑掉（實測偏航從 −7° 變 −40°）。
    #    實機 M9 目前送的就是單一 kp，所以 `--kp 120` 是「模擬實機現況」而不是基準。
    ap.add_argument("--kp", type=float, default=None,
                    help="所有腿關節共用一個 kp（模擬實機 M9 現況）；不給則用原廠分關節 KP3")
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--z-sag", type=float, default=None, dest="z_sag",
                    help="預設 = max_model.STATIC_SAG（模擬值）")
    ap.add_argument("--d-step", type=float, default=None, dest="d_step")
    ap.add_argument("--trim", action="store_true",
                    help="每個模式各自重掃配平點 x_off（⚠️ 鎖輪下這個判準會失效，見 sweep_xoff）")
    ap.add_argument("--grid", type=str, default=None,
                    help="x_off 掃描格點 mm，例如 --grid -120,-110,-100")
    ap.add_argument("--modes", type=str, default=None,
                    help="只跑指定模式，例如 --modes hold,damp")
    ap.add_argument("--sweep-xoff", action="store_true", dest="sweep_xoff",
                    help="★ 三個模式各自掃 x_off，直接比前進距離（公平比較用這個）")
    a = ap.parse_args()
    kw = {"kp3": None if a.kp is None else np.full(3, a.kp),
          "kd3": None if a.kd is None else np.full(3, a.kd),
          "z_sag": mm.STATIC_SAG if a.z_sag is None else a.z_sag,
          "d_step": gb.BASELINE["d_step"] if a.d_step is None else a.d_step}

    print("=" * 96)
    print(f"鎖輪對照　kp={list(mm.KP3 if a.kp is None else kw['kp3'])} "
          f"kd={list(mm.KD3 if a.kd is None else kw['kd3'])} "
          f"z_sag={kw['z_sag'] * 1000:.1f}mm "
          f"d_step={kw['d_step']}　ω={gb.BASELINE['omega']} duty={gb.BASELINE['duty']}"
          f"　{a.secs:.0f} s")
    print("=" * 96)

    if a.sweep_xoff:
        g = (None if a.grid is None
             else np.array([float(v) / 1000.0 for v in a.grid.split(",")]))
        sweep_xoff(a.secs, grid=g,
                   modes=tuple(a.modes.split(",")) if a.modes else MODES, **kw)
        return 0

    x_offs = {}
    for m in MODES:
        if a.trim:
            print(f"\n[配平] wheel_mode={m}")
            x_offs[m] = trim_x_off(m, min(a.secs, 10.0), **kw)
            print(f"    → 配平點 x_off = {x_offs[m] * 1000:+.1f} mm")
        else:
            x_offs[m] = gb.BASELINE["x_off"]

    print(f"\n{'模式':>6}{'x_off':>8}{'跌倒':>5}{'前進 m':>8}{'速度 m/s':>9}"
          f"{'側偏 mm':>9}{'偏航°':>7}{'|偏航|峰':>9}{'俯仰°':>7}{'機身高 mm':>10}"
          f"{'前輪滾 mm':>10}{'後輪滾 mm':>10}{'前膝 τ':>8}{'後膝 τ':>8}")
    res = {}
    for m in MODES:
        r = res[m] = run(m, x_offs[m], a.secs, **kw)
        print(f"{m:>6}{r['x_off'] * 1000:>+8.0f}{('是' if r['fell'] else '否'):>5}"
              f"{r['dx']:>8.2f}{r['speed']:>9.3f}{r['dy'] * 1000:>9.0f}"
              f"{r['yaw']:>7.1f}{r['yaw_abs_max']:>9.1f}{r['pitch_mean']:>7.2f}"
              f"{r['height'] * 1000:>10.0f}{r['roll_front'] * 1000:>10.0f}"
              f"{r['roll_rear'] * 1000:>10.0f}"
              f"{r['knee_front']:>8.1f}{r['knee_rear']:>8.1f}")

    print("\n" + "-" * 96)
    print("判讀")
    print("-" * 96)
    b = res["damp"]
    for m in ("hold", "free"):
        r = res[m]
        print(f"\n{m} vs damp：")
        print(f"  前進　{r['dx']:.2f} m vs {b['dx']:.2f} m　"
              f"（{r['dx'] / b['dx']:+.2f}× 若同號）")
        print(f"  偏航　{r['yaw']:+.1f}° vs {b['yaw']:+.1f}°　"
              f"峰 {r['yaw_abs_max']:.1f}° vs {b['yaw_abs_max']:.1f}°")
        print(f"  後膝　{r['knee_rear']:.1f} vs {b['knee_rear']:.1f} N·m　"
              f"（門檻 70）")
        print(f"  前後輪滾動抵銷　{m}: 前{r['roll_front'] * 1000:+.0f} "
              f"後{r['roll_rear'] * 1000:+.0f} mm　|　"
              f"damp: 前{b['roll_front'] * 1000:+.0f} 後{b['roll_rear'] * 1000:+.0f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
