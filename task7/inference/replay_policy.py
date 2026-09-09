"""離線回放：拿實機 M9 log（trip17）逐幀餵**狗端堆疊**（rl_obs + policy_np）與**本機堆疊**
（real_obs → obs_max + brax），比 obs 各欄與動作；另印走路段 obs 各欄範圍與 policy 動作分布，
上機乾跑時對照。

    conda run -n rbtdog python task7/inference/replay_policy.py \
        task7/logs/m_logs_trip17/M9_20260903_170013.json task7/logs/m_logs_trip17/M9_20260903_170130.json \
        --npz task7/weights/cpg_rl_max_v2_3_np.npz --pkl task7/weights/cpg_rl_max_v2_3_params.pkl \
        --out task7/outputs/replay_policy_trip17.md

限制（老實寫）：M9 舊 log 只記 roll/pitch，`m9_rec` 的 quat 由 roll/pitch 重建、gyro 由其微分，
**gyro_z ≡ 0**。這裡驗的是正負號、腿序、單位、正規化；不是閉迴路（那在 test_policy_closed_loop）。
CPG 狀態從 log 的 GAIT_IN 起點以開迴路 A 重建（A 是決定性的），last_action 用基準動作。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realbot"))
import cpg  # noqa: E402
import local_infer_max as li  # noqa: E402
import m9_rec  # noqa: E402
import max_model as mm  # noqa: E402
import obs_max  # noqa: E402
import policy_np as pn  # noqa: E402
import real_obs  # noqa: E402
import rl_obs  # noqa: E402

COLS = [("gravity", 0, 3), ("gyro", 3, 6), ("joint_pos", 6, 18), ("joint_vel", 18, 30),
        ("cmd", 30, 32), ("last_action", 32, 42), ("cpg", 42, 66)]
FAKE_C = {k: {l: v for l, v in zip(cpg.LEGS, vals)} for k, vals in (
    ("rx", (1.2, 1.4, 1.6, 1.8)), ("rx_d", (0.1, -0.1, 0.2, -0.2)),
    ("ry", (1.3, 1.5, 1.7, 1.9)), ("ry_d", (0.05, 0.15, -0.05, -0.15)),
    ("theta", (0.3, 1.9, 3.4, 5.0)))}


def cpg_states_for(rec, idx, base_p: dict):
    """重建每一幀的 CPG 狀態：開迴路 A 是決定性的，從 log 的 GAIT_IN 起點以 50 Hz 推進
    （與 M9 `GaitStream.sample` 同：t ≥ t_next 就推進；policy 的 obs 用推進**前**的狀態）。"""
    t0 = float(rec.t[idx[0]])
    step = cpg.make_step(cpg.PHASES[base_p["seq"]])
    c = cpg.init(cpg.PHASES[base_p["seq"]])
    mux = {l: base_p["mu_x"] for l in cpg.LEGS}
    muy = {l: base_p["mu_y"] for l in cpg.LEGS}
    om = {l: base_p["omega"] for l in cpg.LEGS}
    k_done, out = 0, []
    for i in idx:
        k = int((float(rec.t[i]) - t0) / 0.02 + 1e-9)     # 1e-9：邊界浮點（10.02−10.0）/0.02 = 0.99999
        while k_done < k:
            c = step(c, mux, muy, om, 0.02)
            k_done += 1
        out.append(c)
    return out


def replay(path: str, pol, infer, cmd, real_cpg: bool = True) -> dict:
    rec = m9_rec.load(path)
    base = pn.baseline_action(pol.layout)
    gait = np.array([p in ("GAIT_IN", "GAIT", "GAIT_OUT") for p in rec.phase])
    idx = np.nonzero(gait)[0]
    if idx.size == 0:
        idx = np.arange(rec.n)
    cs = cpg_states_for(rec, idx, pol.baseline) if real_cpg else [FAKE_C] * idx.size
    d_obs = np.zeros((idx.size, 66), np.float32)
    d_act = np.zeros((idx.size, 10))
    j_act = np.zeros((idx.size, 10))
    worst_obs = np.zeros(66)
    f = real_obs.RealFrame()
    for n, i in enumerate(idx):
        c_d = cs[n]
        c_arr = {k: np.array([c_d[k][real_obs.LEG_SHM[l]] for l in mm.LEGS]) for k in c_d}
        pos = {nm: float(rec.j[nm]["q"][i]) for nm in rl_obs.LEG_NAMES}
        vel = {nm: float(rec.j[nm]["v"][i]) for nm in rl_obs.LEG_NAMES}
        fr = rl_obs.Frame(pos, vel, rec.quat_raw[i], rec.gyro[i], i)
        od = rl_obs.build(fr, c_d, cmd, base)
        f.qpos[3:7] = real_obs.quat_wxyz(rec.quat_raw[i], "xyzw")
        f.qpos[mm.LEG_QPOS_IDX] = real_obs.ctrl_pos(rec)[i]
        f.qvel[3:6] = rec.gyro[i]
        f.qvel[mm.LEG_QVEL_IDX] = real_obs.ctrl_vel(rec)[i]
        oj = obs_max.build_obs(f, c_arr, cmd, base)
        worst_obs = np.maximum(worst_obs, np.abs(od - oj))
        d_obs[n] = od
        d_act[n] = pol.infer(od)
        j_act[n] = infer(oj)
    return {"path": path, "n": int(idx.size), "n_total": rec.n, "hz": rec.hz,
            "worst_obs": worst_obs, "worst_act": float(np.max(np.abs(d_act - j_act))),
            "obs": d_obs, "act": d_act, "derived_imu": bool(getattr(rec, "derived_imu", False))}


def report(results: list, cmd) -> str:
    L = ["# policy 離線回放（trip17 M9 log）", "",
         f"指令 vx {cmd[0]:g} wz {cmd[1]:g}；CPG 狀態由 log 的 GAIT_IN 起點以開迴路 A 重建（決定性）、last_action 基準。",
         "⚠️ M9 舊 log 只有 roll/pitch：quat 由其重建、gyro 由其微分、**gyro_z ≡ 0** —— "
         "驗的是正負號／腿序／單位／正規化，不是閉迴路。", ""]
    L += ["## 1. 狗端堆疊 vs 本機堆疊", "", "| log | 幀數（走路段/全部） | obs 各欄 max\\|Δ\\| | 動作 max\\|Δ\\| | 判定 |",
          "|---|---|---|---|---|"]
    ok_all = True
    for r in results:
        per = "、".join(f"{nm} {r['worst_obs'][a:b].max():.1e}" for nm, a, b in COLS)
        ok = r["worst_obs"].max() < 1e-5 and r["worst_act"] < 1e-4
        ok_all &= ok
        L.append(f"| {Path(r['path']).stem} | {r['n']}/{r['n_total']} | {per} | {r['worst_act']:.1e} | "
                 f"{'✅' if ok else '❌'} |")
    L += ["", f"**{'全部通過' if ok_all else '有不一致'}**（門檻 obs 1e-5、動作 1e-4；float32 位階）。", ""]
    L += ["## 2. 走路段實機 obs 範圍（乾跑時對照這張）", "",
          "| log | 欄 | min | max | mean | std |", "|---|---|---|---|---|---|"]
    for r in results:
        O = r["obs"]
        for nm, a, b in COLS[:4]:
            seg = O[:, a:b]
            L.append(f"| {Path(r['path']).stem[-6:]} | {nm} | {seg.min():+.3f} | {seg.max():+.3f} | "
                     f"{seg.mean():+.3f} | {seg.std():.3f} |")
    L += ["", "應該看到：gravity ≈ (0, 0, −1) 附近、joint_pos 在 ±0.5 內（相對 HOME）、"
          "joint_vel 走路段 ±10 以內。", ""]
    L += ["## 3. policy 在實機 obs 上的動作（開迴路餵，僅看有沒有飽和）", "",
          "| log | \\|a\\| 中位 | \\|a\\| 最大 | sway_x 目標 mm（中位/最大） | sway_y 目標 mm | ω 範圍 |",
          "|---|---|---|---|---|---|"]
    for r in results:
        A = r["act"]
        sw = np.tanh(A[:, 8:10]) * pn.SWAY_MAX * 1000
        om = np.array([pn.act_to_cmd(a, "nomux")[2] for a in A])
        L.append(f"| {Path(r['path']).stem[-6:]} | {np.median(np.abs(A)):.2f} | {np.abs(A).max():.2f} | "
                 f"{np.median(np.abs(sw[:, 0])):.0f}/{np.abs(sw[:, 0]).max():.0f} | "
                 f"{np.median(np.abs(sw[:, 1])):.0f}/{np.abs(sw[:, 1]).max():.0f} | "
                 f"{om.min():.2f}–{om.max():.2f} |")
    L += ["", "對照 —— 模擬驗收 v2.3（`local_infer_max --preset v2.3 --secs 20`，2026-09-09）："
          "ω 0.51–1.63、sway |x| 23 / |y| 17 mm。實機 obs 上同量級即正常；|a| 長期貼 1 = 飽和。", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--npz", default=str(HERE.parent / "weights" / "cpg_rl_max_v2_3_np.npz"))
    ap.add_argument("--pkl", default=str(HERE.parent / "weights" / "cpg_rl_max_v2_3_params.pkl"))
    ap.add_argument("--vx", type=float, default=0.30)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pol = pn.load(a.npz)
    infer = li.load_policy(a.pkl, act_dim=pol.act_dim, head=False)
    cmd = np.array([a.vx, a.wz])
    res = [replay(p, pol, infer, cmd) for p in a.logs]
    txt = report(res, cmd)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")
        print(f"\n📄 {a.out}")
    return 0 if all(r["worst_obs"].max() < 1e-5 and r["worst_act"] < 1e-4 for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
