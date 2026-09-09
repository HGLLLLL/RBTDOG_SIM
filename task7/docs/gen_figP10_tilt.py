#!/usr/bin/env python3
"""圖 P10-b：模擬 ↔ 實機的機身傾角疊圖（trip10 原地踏步，2026-09-02）。

★ 這張圖是本專案「模擬可以外插到實機」的第一個定量證據。

  模擬端：把 `outputs/gait/march_run1.json` 在 MuJoCo 裡照
          `inference/play_gait_traj.py` 同一套 PD 迴圈重播，
          每個 50 Hz 幀從 `qpos[3:7]` 的四元數算 roll / pitch。
  實機端：`logs/m_logs_trip10/M9_20260902_100202.json` 的 IMU 讀值（200 Hz）。

⚠️ **時間窗的坑**：實機日誌裡 `phase == "GAIT"` 的樣本涵蓋**整段 16 秒**
   （ramp_up 3 s ＋ 步態 10 s ＋ ramp_down 3 s），不是只有中間的步態段。
   所以兩邊都必須再切 `[ramp, ramp+secs]` = 3–13 s。
   只切模擬那一邊的話會差整整 3 秒，相關係數從 0.989 掉到 0.828。

用法：
    ~/miniforge3/envs/rbtdog/bin/python task7/docs/gen_figP10_tilt.py

⚠️ 用 rbtdog 環境跑（要 mujoco + matplotlib）。中文字型要 Noto Sans CJK TC
   （`sudo apt install fonts-noto-cjk`）。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import mujoco                       # noqa: E402
import numpy as np                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "realbot"))

import coord                        # noqa: E402
import max_model as mm              # noqa: E402

MM2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}

TRAJ = ROOT / "outputs" / "gait" / "march_run1.json"
LOG = ROOT / "logs" / "m_logs_trip10" / "M9_20260902_100202.json"
OUT = ROOT / "docs" / "圖P10_傾角疊圖.png"

C_SIM, C_REAL = "#1F3864", "#D9542B"
LAG_MAX = 0.200          # 延遲搜尋範圍 ±200 ms
LAG_STEP = 0.005         # 5 ms（＝實機取樣週期）


# =============================================================================
# 模擬端
# =============================================================================
def sim_rollpitch(D: dict) -> tuple[np.ndarray, np.ndarray]:
    """重播軌跡檔，回傳每個 50 Hz 幀的 (roll, pitch)，單位度。"""
    # ★ 按名稱對應到 MJCF 腿序，不按索引；qpos 位址用 LEG_QPOS_IDX（12 個腿關節不連續）
    mj_names = [MM2SHM[l] + k for l in mm.LEGS for k in coord.LEG_KINDS]
    perm = [D["joints"].index(n) for n in mj_names]
    Q = np.array(D["q"])[:, perm]

    m = mujoco.MjModel.from_xml_path(mm.SCENE)
    d = mujoco.MjData(m)
    lo = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 0]
    hi = m.jnt_range[m.dof_jntid[mm.LEG_QVEL_IDX], 1]
    mujoco.mj_resetData(m, d)
    d.qpos[mm.LEG_QPOS_IDX] = np.clip(Q[0], lo + 1e-4, hi - 1e-4)
    d.qpos[2] = 0.55
    for _ in range(int(1.0 / m.opt.timestep)):          # 先落穩
        e = Q[0] - d.qpos[mm.LEG_QPOS_IDX]
        d.ctrl[mm.LEG_ACT_IDX] = np.clip(
            D["kp"] * e - D["kd"] * d.qvel[mm.LEG_QVEL_IDX], -150, 150)
        d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(
            -D["wheel_kd"] * d.qvel[mm.WHEEL_QVEL_IDX], -40, 40)
        mujoco.mj_step(m, d)
    print(f"模擬落穩後機身高 {d.qpos[2]*1000:.0f} mm")

    nsub = max(1, int(round(D["dt"] / m.opt.timestep)))
    roll, pitch = [], []
    for i in range(D["n"]):
        for _ in range(nsub):
            e = Q[i] - d.qpos[mm.LEG_QPOS_IDX]
            d.ctrl[mm.LEG_ACT_IDX] = np.clip(
                D["kp"] * e - D["kd"] * d.qvel[mm.LEG_QVEL_IDX], -150, 150)
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(
                -D["wheel_kd"] * d.qvel[mm.WHEEL_QVEL_IDX], -40, 40)
            mujoco.mj_step(m, d)
        w, x, y, z = d.qpos[3:7]
        roll.append(math.degrees(math.atan2(2 * (w * x + y * z),
                                            1 - 2 * (x * x + y * y))))
        pitch.append(math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))))
    return np.array(roll), np.array(pitch)


# =============================================================================
# 對齊
# =============================================================================
def best_lag(t: np.ndarray, sim: dict, tr: np.ndarray, real: dict) -> float:
    """在 ±200 ms 內找讓 roll、pitch 相關係數總和最大的延遲（秒，正 = 實機落後）。"""
    best = (-9.9, 0.0)
    for lag in np.arange(-LAG_MAX, LAG_MAX + 1e-9, LAG_STEP):
        s = sum(np.corrcoef(sim[k], np.interp(t + lag, tr, real[k]))[0, 1]
                for k in ("roll", "pitch"))
        if s > best[0]:
            best = (s, float(lag))
    return best[1]


def main() -> int:
    D = json.loads(TRAJ.read_text(encoding="utf-8"))
    p = D["params"]
    dt = D["dt"]
    n_ramp = int(round(p["ramp"] / dt))
    n_gait = int(round(p["secs"] / dt))
    print(f"軌跡 {TRAJ.name}：{D['n']} 幀 @ {1/dt:.0f} Hz，"
          f"ramp {p['ramp']} s ×2 ＋ 步態 {p['secs']} s")

    sr, sp = sim_rollpitch(D)
    sim = {"roll": sr[n_ramp:n_ramp + n_gait], "pitch": sp[n_ramp:n_ramp + n_gait]}
    t = np.arange(n_gait) * dt

    # ---- 實機：phase == GAIT，t 以第一筆 GAIT 為零，再切出同一個 3–13 s 窗
    L = json.loads(LOG.read_text(encoding="utf-8"))
    g = [x for x in L["samples"] if x["phase"] == "GAIT"]
    t_all = np.array([x["t"] for x in g]) - g[0]["t"]
    print(f"實機 {LOG.name}：GAIT 段 {len(g)} 筆，涵蓋 {t_all[-1]:.2f} s"
          f"（＝整段軌跡，含兩段 ramp）→ 切出 {p['ramp']:.0f}–{p['ramp']+p['secs']:.0f} s")
    sel = (t_all >= p["ramp"] - 1e-9) & (t_all <= p["ramp"] + p["secs"] + LAG_MAX)
    tr = t_all[sel] - p["ramp"]
    real = {"roll": np.array([x["roll"] for x in g])[sel],
            "pitch": np.array([x["pitch"] for x in g])[sel]}

    # ---- 有沒有哪一條上下顛倒？先用零延遲的相關係數判，負的就翻號
    for k in ("roll", "pitch"):
        if np.corrcoef(sim[k], np.interp(t, tr, real[k]))[0, 1] < 0:
            sim[k] = -sim[k]
            print(f"⚠️ 模擬的 {k} 與實機反向 → 已翻號")

    lag = best_lag(t, sim, tr, real)
    r = {k: np.corrcoef(sim[k], np.interp(t + lag, tr, real[k]))[0, 1]
         for k in ("roll", "pitch")}
    pk_s = {k: float(np.abs(sim[k]).max()) for k in sim}
    pk_r = {k: float(np.abs(real[k]).max()) for k in real}

    def tilt(a, b):     # 合成傾角
        return np.degrees(np.arccos(np.cos(np.radians(a)) * np.cos(np.radians(b))))
    ts = float(tilt(sim["roll"], sim["pitch"]).max())
    trl = float(tilt(real["roll"], real["pitch"]).max())

    print(f"\n★ 最佳延遲 {lag*1000:+.0f} ms（正 = 實機落後模擬）")
    print(f"  相關係數　roll {r['roll']:.3f}　pitch {r['pitch']:.3f}")
    print(f"  roll  峰值　模擬 {pk_s['roll']:.1f}°　實機 {pk_r['roll']:.1f}°"
          f"　（實/模 {pk_r['roll']/pk_s['roll']*100:.0f}%）")
    print(f"  pitch 峰值　模擬 {pk_s['pitch']:.1f}°　實機 {pk_r['pitch']:.1f}°"
          f"　（實/模 {pk_r['pitch']/pk_s['pitch']*100:.0f}%）")
    print(f"  合成傾角峰值　模擬 {ts:.1f}°　實機 {trl:.1f}°　（實/模 {trl/ts*100:.0f}%）")

    # =========================================================================
    # 畫圖
    # =========================================================================
    plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.3), dpi=200, sharex=True)
    fig.subplots_adjust(top=0.845, bottom=0.095, left=0.072, right=0.985, hspace=0.16)

    m_gait = tr <= p["secs"] + 1e-9      # 畫圖只畫 0–10 s，不畫為了延遲搜尋多留的那段
    for ax, k, label in zip(axes, ("roll", "pitch"), ("Roll　橫滾", "Pitch　俯仰")):
        ax.axhline(0, color="#CCCCCC", lw=0.8, zorder=1)
        ax.plot(t, sim[k], color=C_SIM, lw=2.2, label="模擬 MuJoCo", zorder=3)
        ax.plot(tr[m_gait], real[k][m_gait], color=C_REAL, lw=2.0,
                label="實機 D1 Max IMU", zorder=2)
        ax.set_ylim(-16, 16)
        ax.set_xlim(0, p["secs"])
        ax.set_ylabel(f"{label}（°）", fontsize=11.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.text(0.995, 0.045,
                f"r = {r[k]:.3f}　峰值 模擬 {pk_s[k]:.1f}° / 實機 {pk_r[k]:.1f}°"
                f"（{pk_r[k]/pk_s[k]*100:.0f}%）",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, color="#555555",
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
    axes[0].legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    axes[1].set_xlabel("步態段時間（秒）", fontsize=11.5)

    fig.text(0.072, 0.955,
             "模擬與實機機身姿態幾乎重合 —— D1 Max 原地踏步（2026-09-02 實機）",
             fontsize=16.5, fontweight="bold", color=C_SIM, ha="left", va="center")
    fig.text(0.072, 0.898,
             f"相關係數 roll {r['roll']:.3f}／pitch {r['pitch']:.3f}　·　"
             f"合成傾角峰值 模擬 {ts:.1f}° vs 實機 {trl:.1f}°（{trl/ts*100:.0f}%）　·　"
             f"實機落後 {lag*1000:.0f} ms",
             fontsize=11, color="#555555", ha="left", va="center")

    fig.savefig(OUT)
    print(f"\n🖼  {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
