#!/usr/bin/env python3
"""這台狗前後對稱，那「前後完美對稱的步態」到底存不存在？

2026-09-03 使用者的質疑，前提完全成立：

    整機 38.821 kg，質心前後偏移 −0.59 mm（軸距的 0.1%）
    ABAD 前後位置 ±0.2698 m，鏡像到 0.0 µm
    連桿前後同長（大腿 260 / 小腿 280 mm）

**機構與配重確實前後對稱。** 那不對稱是哪裡來的？三個測試：

T1 **相位鏡像**：把步態序列前後對調、`x_c` 反號，跑出來應該是原步態的鏡像
   （前腳的數字跑到後腳身上）。若吻合 → **系統本身前後對稱，不對稱來自步態的方向性**。
   `PHASE_WALK` 的前後腿相位差是 **+0.5π（前腿比後腿晚 1/4 週期）** ——
   這是步態內建的、與機構無關的前後不對稱。

T2 **`x_c=0` 重掃其餘旋鈕**：對稱姿態走不好，會不會只是因為其餘參數是在
   `x_c=−110` 附近調出來的？（這條線的核心教訓就是「旋鈕耦合、單獨調看不到東西」，
   所以這個懷疑必須用掃描回答，不能用推論。）

T3 **承重與角色**：站立相的前後法向力、前後腿的執行率，看「對稱姿態」下
   前後腿是不是仍在扮演不同角色。

用法：
    PY=/home/huang/miniforge3/envs/rbtdog/bin/python
    $PY task7/inference/diag/fore_aft_symmetry.py            # T1
    $PY task7/inference/diag/fore_aft_symmetry.py --t2       # T2（較慢）
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

# 腿序 FR, FL, RR, RL。前後交換 = 索引 (0,1,2,3) → (2,3,0,1)。
SWAP = [2, 3, 0, 1]


def mirror_phase(phase) -> np.ndarray:
    """把相位序列前後對調（前腿吃後腿的相位，反之亦然）。"""
    return np.asarray(phase, dtype=float)[SWAP]


def t1(secs: float, kd_wheel: float, x_c: float) -> None:
    """相位鏡像測試。"""
    print(f"# T1 相位鏡像：{secs:.0f} s、kd_wheel={kd_wheel}、|x_c|={abs(x_c) * 1000:.0f} mm")
    p0 = np.asarray(cpg_max.PHASE_WALK)
    p1 = mirror_phase(p0)
    print(f"#   原相位 (FR,FL,RR,RL) = {p0 / np.pi} π    前腿−後腿 = {(p0[0] - p0[2]) / np.pi:+.2f}π")
    print(f"#   鏡像相位             = {p1 / np.pi} π    前腿−後腿 = {(p1[0] - p1[2]) / np.pi:+.2f}π")
    print(f"#   ★ 若系統前後對稱，鏡像組的『後腳』數字應該等於原組的『前腳』數字")

    a = cw.rollout(gait="walk", secs=secs, kd_wheel=kd_wheel, quiet=True,
                   x_off=cpg_max.x_off_split(x_c, 0.0))
    # ★ 完整的前後鏡像（對 x 軸做鏡射）有**三項**，少一項就不是對稱操作：
    #     ① 相位序列前後對調      ② 配平點 x_c 反號     ③ **行進方向反轉**（d_step 反號）
    #   第一版漏了 ③，量出「鏡像不成立、執行率變負」—— 那是我操作錯，不是系統不對稱。
    #   反轉之後狗往 −x 走，所以速度要取負號才能跟原組比。
    d_step0 = cw.GAITS["walk"]["d_step"]
    b = cw.rollout(gait="walk", secs=secs, kd_wheel=kd_wheel, quiet=True,
                   x_off=cpg_max.x_off_split(-x_c, 0.0), phase=p1, d_step=-d_step0)
    # ⚠️ `speed_travel` 是不帶方向的量（搖擺抵銷後的行進量／時間），倒著走一樣是正的。
    #   第一版在這裡多取了一次負號，於是「速度差 200.8%」—— 一個純粹由報告造成的假差異。
    #   帳面的 `speed`（qpos[0] 差）才帶方向，倒退走時它是負的。

    print(f"\n{'':>14}{'原步態':>12}{'鏡像步態':>12}{'鏡像的對應項':>16}")
    rows = [("行進速度 m/s", a["speed_travel"], b["speed_travel"], None),
            ("帳面速度（帶向）", a["speed"], b["speed"], "反向"),
            ("前腳執行率", a["exec_front"], b["exec_front"], "後腳"),
            ("後腳執行率", a["exec_rear"], b["exec_rear"], "前腳"),
            ("前膝峰值 N·m", a["knee_peak_front"], b["knee_peak_front"], "後膝"),
            ("後膝峰值 N·m", a["knee_peak_rear"], b["knee_peak_rear"], "前膝"),
            ("週期俯仰 °", a["pitch_cycle"], b["pitch_cycle"], None),
            ("彈跳 mm", a["bounce"] * 1000, b["bounce"] * 1000, None)]
    for name, va, vb, pair in rows:
        print(f"{name:>14}{va:>12.3f}{vb:>12.3f}{(pair or ''):>16}")

    print(f"\n# ★ 交叉比對（原前腳 ↔ 鏡像後腳）")
    for lab, va, vb in (("執行率", a["exec_front"], b["exec_rear"]),
                        ("執行率", a["exec_rear"], b["exec_front"]),
                        ("膝峰值", a["knee_peak_front"], b["knee_peak_rear"]),
                        ("膝峰值", a["knee_peak_rear"], b["knee_peak_front"])):
        rel = abs(va - vb) / max(abs(va), 1e-9) * 100
        print(f"#   {lab} {va:7.3f} vs {vb:7.3f}   差 {rel:5.1f}%"
              + ("  ✅ 鏡像成立" if rel < 10 else "  ⚠️ 不成立"))
    sp = abs(a["speed_travel"] - b["speed_travel"]) / max(a["speed_travel"], 1e-9) * 100
    print(f"#   速度 {a['speed_travel']:.3f} vs {b['speed_travel']:.3f}   差 {sp:.1f}%")


def t2(secs: float) -> None:
    """`x_c=0` 下重掃 ω / duty / d_step —— 對稱姿態是不是只差沒調對其餘旋鈕。"""
    print(f"# T2 `x_c=0`（完美對稱姿態）重掃其餘旋鈕，{secs:.0f} s")
    print(f"#   對照：現行點 (x_c=−110, x_d=0, kd3, ω1.4, duty.80) "
          f"→ 速度 0.257、前腳執行率 0.79")
    print(f"{'x_d':>6}{'kd':>5}{'ω':>6}{'duty':>6}{'d_step':>8} | {'速度':>8}"
          f"{'執行前':>8}{'執行後':>8}{'後膝':>7}{'偏航°':>8}{'彈跳':>7}{'離地':>7}")
    best = None
    for x_d in (-0.060, -0.090):
        for kd in (3.0, 5.0):
            for om in (1.4, 1.8):
                for duty in (0.80, 0.85):
                    for d_step in (0.10, 0.13):
                        r = cw.rollout(gait="walk", secs=secs, kd_wheel=kd, quiet=True,
                                       x_off=cpg_max.x_off_split(0.0, x_d),
                                       omega=om, duty=duty, d_step=d_step)
                        print(f"{x_d * 1000:>6.0f}{kd:>5.1f}{om:>6.1f}{duty:>6.2f}"
                              f"{d_step:>8.2f} | {r['speed_travel']:>8.3f}"
                              f"{r['exec_front']:>8.2f}{r['exec_rear']:>8.2f}"
                              f"{r['knee_peak_rear']:>7.1f}{r['yaw']:>8.1f}"
                              f"{r['bounce'] * 1000:>7.1f}"
                              f"{r['min_lift'] * 1000:>7.1f}", flush=True)
                        if best is None or r["exec_front"] > best[0]:
                            best = (r["exec_front"], x_d, kd, om, duty, d_step)
    print(f"\n# ★ `x_c=0` 能達到的最高前腳執行率：{best[0]:.2f}"
          f"（x_d={best[1] * 1000:.0f} kd={best[2]} ω={best[3]} duty={best[4]} "
          f"d_step={best[5]}）｜現行點是 0.79")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--kd-wheel", type=float, default=3.0, dest="kd_wheel")
    ap.add_argument("--x-c", type=float, default=-0.110, dest="x_c")
    ap.add_argument("--t2", action="store_true")
    a = ap.parse_args()
    if a.t2:
        t2(a.secs)
    else:
        t1(a.secs, a.kd_wheel, a.x_c)
