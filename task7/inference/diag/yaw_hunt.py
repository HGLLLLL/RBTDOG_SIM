#!/usr/bin/env python3
"""`x_c=0`（前後姿態完美對稱）那組的系統性偏航：來源與補償。

2026-09-03。`fore_aft_symmetry.py --t2` 掃出 `x_c=0` 能走 **0.402 m/s**
（比現行 0.257 快 56%）且前後大腿角完全相同，但 **20 秒偏航 +38~+45°**，
而且 32 組參數裡幾乎全是正的 —— **同號＝系統性，不是混沌**。

三個測試：

Y1 **偏航隨時間的形狀**：一次性偏掉、還是等速累積？
   （C 文件更正過一次：基準的偏航是「先 +8.6 再掃到 −7」的擺盪，
   帳面小是符號抵銷。所以看終點值不夠，要看形狀。）

   ⚠️⚠️ **一律用 `yaw_total`（逐步累積、不包裹），不要用 `yaw`（atan2 首尾相減）。**
   第一版用了 `yaw`，量出「35→46→38 先增後減」而據此推論「偏航是起步瞬態、
   之後就停住」—— 錯的：那是**繞圈時 ±180° 包裹**造成的假象。
   這是 `cpg_max.yaw_deg` 的 docstring 早就寫過的坑，這條線第二次踩。

Y2 **左右鏡像**：把相位序列左右對調，偏航應該**反號、等大**。
   若成立 → 偏航來自 `PHASE_WALK` 的側序（左後→左前→右後→右前，**左腿先動**），
   與模型無關；那它就是可以用左右差動步幅補掉的確定量。
   若不成立 → 是模型的左右不對稱（實測質心 y = **+1.53 mm**，比前後的 0.59 大 2.6 倍）。

Y3 **左右差動 `d_step` 補償**：找偏航過零點。

用法：
    PY=/home/huang/miniforge3/envs/rbtdog/bin/python
    $PY task7/inference/diag/yaw_hunt.py            # Y1 + Y2
    $PY task7/inference/diag/yaw_hunt.py --sweep    # Y3
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

# `x_c=0` 的最快組（fore_aft_symmetry.py --t2）
SYM = dict(x_c=0.0, x_d=-0.060, kd_wheel=3.0, omega=1.4, duty=0.80, d_step=0.13)
LR_SWAP = [1, 0, 3, 2]              # 腿序 FR,FL,RR,RL → 左右交換


def run(secs, d_step=None, phase=None, **kw):
    p = dict(SYM, **kw)
    return cw.rollout(gait="walk", secs=secs, quiet=True,
                      x_off=cpg_max.x_off_split(p["x_c"], p["x_d"]),
                      kd_wheel=p["kd_wheel"], omega=p["omega"], duty=p["duty"],
                      d_step=p["d_step"] if d_step is None else d_step,
                      phase=phase)


def y1(secs_list) -> None:
    print("# Y1 偏航隨時間的形狀（同一組參數、不同 secs；模擬是確定性的）")
    print(f"{'t (s)':>7}{'前進 m':>9}{'偏航°':>9}{'°/s':>8}{'°/m':>8}{'側偏 m':>9}")
    prev = None
    for t in secs_list:
        r = run(t)
        rate_t = r["yaw_total"] / t
        rate_d = r["yaw_total"] / max(r["dist"], 1e-9)
        print(f"{t:>7.0f}{r['dist']:>9.2f}{r['yaw_total']:>9.1f}{rate_t:>8.2f}"
              f"{rate_d:>8.2f}{r['lateral']:>9.2f}")
        prev = r
    print("#   ★ 若 °/s 大致固定 → 等速累積（可用固定差動補）；"
          "若前段大後段小 → 一次性瞬態（補起步就好）")


def y2(secs: float) -> None:
    print(f"\n# Y2 左右鏡像：相位序列左右對調，{secs:.0f} s")
    p0 = np.asarray(cpg_max.PHASE_WALK)
    p1 = p0[LR_SWAP]
    print(f"#   原相位 (FR,FL,RR,RL) = {p0 / np.pi} π   序列：左後→左前→右後→右前")
    print(f"#   左右鏡像             = {p1 / np.pi} π   序列：右後→右前→左後→左前")
    a, b = run(secs), run(secs, phase=p1)
    print(f"\n{'':>14}{'原步態':>10}{'左右鏡像':>12}")
    for name, key, scale in (("偏航 ° (unwrap)", "yaw_total", 1),
                             ("偏航 ° (wrap)", "yaw", 1), ("側偏 m", "lateral", 1),
                             ("行進速度", "speed_travel", 1),
                             ("前腳執行率", "exec_front", 1),
                             ("後腳執行率", "exec_rear", 1),
                             ("後膝 N·m", "knee_peak_rear", 1),
                             ("平均側傾 °", "roll_mean", 1)):
        print(f"{name:>14}{a[key] * scale:>10.3f}{b[key] * scale:>12.3f}")
    s = a["yaw_total"] + b["yaw_total"]
    print(f"\n#   偏航和 = {s:+.1f}°（完美反號時為 0）"
          f"｜相對量級 {abs(s) / max(abs(a['yaw_total']), 1e-9) * 100:.1f}%")
    print("#   ★ 反號等大 → 偏航來自步態序列的左右不對稱（可補）；"
          "同號 → 來自模型的左右不對稱")


def y3(secs: float, diffs, bases=(0.10, 0.115, 0.13)) -> None:
    """左右差動 `d_step`：左腿步幅大 → 往右偏，用來抵銷系統性偏航。

    ⚠️ 必須跟基準 `d_step` 一起掃：δ 解偏航，但基準 `d_step` 決定觸地衝擊，
    而**後膝峰值是實機的硬門檻**（模擬 48.5 已經對應實機 71–74，門檻 70）。
    只掃 δ 會選到「偏航歸零但後膝 61」的點 —— 那在實機上是跑不了的。
    """
    print(f"\n# Y3 左右差動 d_step 補償，{secs:.0f} s"
          f"（腿序 FR,FL,RR,RL；左腿 = FL,RL = 索引 1,3）")
    print(f"#   δ 表示左腿 +δ、右腿 −δ｜現行點對照：速度 0.257、執行前 0.79、"
          f"後膝 48.5、偏航 −4.3°")
    print(f"{'d_step':>8}{'δ mm':>7}{'前進 m':>9}{'偏航°':>9}{'側偏 m':>9}{'速度':>8}"
          f"{'執行前':>8}{'後膝':>7}{'彈跳':>7}{'離地':>7}")
    for base in bases:
        for dl in diffs:
            d_step = np.array([base - dl, base + dl, base - dl, base + dl])
            r = run(secs, d_step=d_step)
            flag = ""
            if r["knee_peak_rear"] <= 48.5 and abs(r["yaw_total"]) < 5 and \
                    r["exec_front"] >= 0.70 and r["fell"] is None:
                flag = "  ✅"
            print(f"{base:>8.3f}{dl * 1000:>7.1f}{r['dist']:>9.2f}{r['yaw_total']:>9.1f}"
                  f"{r['lateral']:>9.2f}{r['speed_travel']:>8.3f}"
                  f"{r['exec_front']:>8.2f}{r['knee_peak_rear']:>7.1f}"
                  f"{r['bounce'] * 1000:>7.1f}{r['min_lift'] * 1000:>7.1f}{flag}",
                  flush=True)
        print("-" * 82)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--sweep", action="store_true", help="跑 Y3（左右差動掃描）")
    ap.add_argument("--diffs", type=str, default="0,0.002,0.004,0.006,0.008,0.012")
    a = ap.parse_args()
    if a.sweep:
        y3(a.secs, [float(v) for v in a.diffs.split(",")])
    else:
        y1([5, 10, 15, 20, 30, 45, 60])
        y2(a.secs)
