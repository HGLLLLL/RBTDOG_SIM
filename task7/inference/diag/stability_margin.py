#!/usr/bin/env python3
"""靜態穩定裕度：這個 walk 到底有沒有滿足「主流 static walk」的基本條件。

2026-09-03。使用者指出：機構與配重已驗證前後對稱（質心偏移 0.59 mm、ABAD 鏡像到
0.0 µm），所以問題在 walk 步態本身，該去看主流怎麼做，而不是靠 IMU 回授遮掉。

文獻上 statically stable walk / crawl gait 的核心是兩個量，**我們一個都沒量過**：

  **穩定裕度 S** ＝ 質心（COG）在水平面的投影，到支撐多邊形邊界的最短距離。
                  在內為正、在外為負。S > 0 是「不需要任何回授就不會倒」的定義。
  **body sway（COG adjustment）** ＝ 為了讓 S > 0 而刻意做的機身橫向擺動。
                  這是 crawl gait 的標準組成，不是額外的花招。

而我們的 CPG：`MU_Y = 1.5` → `fy = 0` → **橫向足端運動恆為 0**
（`test_mu_y_1p5_gives_zero_lateral` 就是在釘這件事），
也沒有任何機身側擺項。→ 也就是說**我們從來沒有做過 COG adjustment**。

假說：S 在單腳擺動期間是負的（質心跑到支撐三角形外），狗全程靠動態勉強撐住，
於是「往哪邊倒」由起步的微小差異決定 —— 這正好解釋量到的現象：
偏航對起始相位 1e-16 的差異敏感（+3.0° vs +35.9°），皮米擾動 60 秒全距 96°。

本腳本只**量**，不改步態。輸出：
  - S 的時間序列統計（最小值、負的時間比例）
  - 每個支撐狀態（哪三隻腳著地）各自的 S
  - 質心投影相對支撐多邊形中心的橫向偏移

用法：
    PY=/home/huang/miniforge3/envs/rbtdog/bin/python
    $PY task7/inference/diag/stability_margin.py
    $PY task7/inference/diag/stability_margin.py --x-c 0 --x-d -0.060 --d-step 0.13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "task7" / "inference"))

import mujoco                       # noqa: E402
import cpg_max                      # noqa: E402
import cpg_walk_max as cw           # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402


def com_xy(m, d) -> np.ndarray:
    """整機質心在世界座標的水平投影。"""
    tot = m.body_mass.sum()
    return (m.body_mass[:, None] * d.xipos).sum(0)[:2] / tot


def margin(com: np.ndarray, pts: np.ndarray) -> float:
    """質心投影到支撐多邊形邊界的最短距離；在內為正、在外為負。

    `pts` 是著地足端的 xy（>=3 點）。少於 3 點時靜態穩定無從談起，回 nan。
    ⚠️ 點要先排成凸包順序，否則「邊」會連錯，算出來的距離沒有意義。
    """
    if len(pts) < 3:
        return float("nan")
    c = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    p = pts[order]
    inside = True
    best = np.inf
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        e = b - a
        n = np.array([e[1], -e[0]])          # 外法線（順時針時朝外）
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        s = float(np.dot(com - a, n))
        # 多邊形是逆時針（按極角排序），所以「內側」是 s < 0
        if s > 0:
            inside = False
        # 點到線段的距離
        t = float(np.clip(np.dot(com - a, e) / max(np.dot(e, e), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(com - (a + t * e))))
    return best if inside else -best


def analyse(secs, x_c, x_d, kd_wheel, d_step, g_c, duty, omega,
            phase=None, sway_x=0.0, sway_y=0.0, sway_lead=0.0,
            dt_sample=0.01):
    """跑一次 rollout，逐步量 S。直接複製 rollout 的迴圈以便取樣內部狀態。"""
    cfg = cw.GAITS["walk"]
    x_off = cpg_max.x_off_split(x_c, x_d)
    r = cw.Robot(kd_wheel=kd_wheel)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    ph = cfg["phase"] if phase is None else phase
    step = cpg_max.make_cpg_step(ph)
    q0 = cpg_max.stand_targets(ks, f0, x_off)
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(q0, "damp")
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    c = cpg_max.cpg_init(ph)
    foot_ids = mm.foot_body_ids(r.m)
    n = int(secs / mm.CTRL_DT)
    every = max(1, int(dt_sample / mm.CTRL_DT))
    rec = []       # (S, n_support, lateral_offset, support_key)
    for i in range(n):
        c = step(c, np.full(4, cfg["mu_x"]), np.full(4, 1.5),
                 np.full(4, omega), mm.CTRL_DT)
        sway = None
        if sway_x or sway_y:
            sway = cpg_max.body_sway(cpg_max.gait_phase(c["theta"], ph),
                                     sway_x, sway_y, sway_lead)
        q_des, _ = cpg_max.joint_targets(c, f0, x_off, g_c, d_step,
                                         cw.D_STEP_Y, duty, ks, mm.STATIC_SAG,
                                         sway)
        r.step(q_des, "damp")
        if i % every or i < int(1.0 / mm.CTRL_DT):     # 跳過起步第一秒
            continue
        f = r.foot_forces()
        on = f > 1.0
        pts = np.array([r.d.xpos[fid][:2] for fid, o in zip(foot_ids, on) if o])
        com = com_xy(r.m, r.d)
        S = margin(com, pts)
        lat = float("nan")
        if len(pts) >= 3:
            # 質心相對支撐多邊形形心的橫向（機身 y）偏移
            gvec = com - pts.mean(0)
            yaw = np.radians(cpg_max.yaw_deg(r.d.qpos[3:7]))
            lat = float(-gvec[0] * np.sin(yaw) + gvec[1] * np.cos(yaw))
        rec.append((S, int(on.sum()), lat,
                    "".join(c_ for c_, o in zip("FfRr", on) if o)))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--x-c", type=float, default=-0.110, dest="x_c")
    ap.add_argument("--x-d", type=float, default=0.0, dest="x_d")
    ap.add_argument("--kd-wheel", type=float, default=3.0, dest="kd_wheel")
    ap.add_argument("--d-step", type=float, default=0.10, dest="d_step")
    ap.add_argument("--g-c", type=float, default=0.08, dest="g_c")
    ap.add_argument("--duty", type=float, default=0.80)
    ap.add_argument("--omega", type=float, default=1.4)
    ap.add_argument("--sway-y", type=float, default=0.0, dest="sway_y")
    ap.add_argument("--sway-x", type=float, default=0.0, dest="sway_x")
    ap.add_argument("--sway-lead", type=float, default=0.0, dest="sway_lead")
    ap.add_argument("--ls", action="store_true",
                    help="用 lateral sequence（PHASE_WALK_LS）而不是現行的 DS")
    a = ap.parse_args()

    rec = analyse(a.secs, a.x_c, a.x_d, a.kd_wheel, a.d_step, a.g_c,
                  a.duty, a.omega,
                  phase=cpg_max.PHASE_WALK_LS if a.ls else None,
                  sway_x=a.sway_x, sway_y=a.sway_y, sway_lead=a.sway_lead)
    S = np.array([x[0] for x in rec], dtype=float)
    nsup = np.array([x[1] for x in rec])
    lat = np.array([x[2] for x in rec], dtype=float)
    ok = ~np.isnan(S)

    print(f"# 靜態穩定裕度 S：x_c={a.x_c * 1000:.0f} x_d={a.x_d * 1000:.0f} "
          f"kd={a.kd_wheel} d_step={a.d_step} g_c={a.g_c} duty={a.duty} ω={a.omega} "
          f"序列={'LS' if a.ls else 'DS'} sway=({a.sway_x*1000:.0f},{a.sway_y*1000:.0f})mm"
          f"｜{a.secs:.0f} s，取樣 {len(rec)} 點（跳過起步 1 s）")
    print(f"#   S = 質心投影到支撐多邊形邊界的最短距離，**在內為正**。"
          f"文獻的 static walk 要求 S > 0 全程成立。")
    print(f"  支撐腳數分佈：" + "  ".join(
        f"{k}腳 {100 * np.mean(nsup == k):.1f}%" for k in (2, 3, 4)))
    print(f"  S 可算的比例（≥3 腳）：{100 * ok.mean():.1f}%")
    if ok.any():
        print(f"  S  最小 {np.nanmin(S) * 1000:+.1f} mm   中位 "
              f"{np.nanmedian(S) * 1000:+.1f} mm   最大 {np.nanmax(S) * 1000:+.1f} mm")
        print(f"  ★ S < 0（質心在支撐多邊形外）的時間比例："
              f"**{100 * np.mean(S[ok] < 0):.1f}%**")
        print(f"  ★ S < 20 mm（裕度過小）的時間比例：{100 * np.mean(S[ok] < 0.020):.1f}%")
        print(f"  質心相對支撐形心的橫向偏移：中位 {np.nanmedian(lat) * 1000:+.1f} mm"
              f"   全距 {(np.nanmax(lat) - np.nanmin(lat)) * 1000:.1f} mm")
    # 分支撐狀態
    print(f"\n  {'支撐腳(F=FR f=FL R=RR r=RL)':<28}{'佔比%':>8}{'S 中位 mm':>12}{'S 最小 mm':>12}")
    keys = {}
    for (s, k, l, key) in rec:
        keys.setdefault(key, []).append(s)
    for key, vals in sorted(keys.items(), key=lambda kv: -len(kv[1])):
        v = np.array(vals, dtype=float)
        if np.isnan(v).all():
            print(f"  {key:<28}{100 * len(v) / len(rec):>8.1f}{'—':>12}{'—':>12}")
        else:
            print(f"  {key:<28}{100 * len(v) / len(rec):>8.1f}"
                  f"{np.nanmedian(v) * 1000:>12.1f}{np.nanmin(v) * 1000:>12.1f}")
