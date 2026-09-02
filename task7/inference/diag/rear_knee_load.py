#!/usr/bin/env python3
"""C：後膝為什麼比前膝吃力 1.5 倍（86–87% vs 58–60%）。

2026-09-02 trip16 開的題。手上的線索是「前輪淨滾 +330 / 後輪 −470」的前後對抗，
但那只說明**輪子**在對抗，沒說明**為什麼是膝在扛、而且是後膝**。

把後膝力矩拆成三個可以各自證偽的來源：

    τ_knee = −Jᵀ·F  =  a_z·Fz  +  a_x·Fx        （a = −J[:, 2]，即膝那一行）
             └ 幾何 ┘  └承重┘   └推進/對抗┘

  H-A 幾何：後腿的膝對足端力的力臂本來就比前腿大 → 同樣的力、力矩就是比較大
  H-B 承重：重心偏後，後腳法向力比較大
  H-C 推進：站立相的縱向力（前後對抗的那個）只有後腿在扛

三者不互斥，本腳本量的是**各佔多少**。

⚠️ 只讀資料 + 純運動學 + 一次模擬 rollout。不碰實機。

用法：
    /home/huang/miniforge3/envs/rbtdog/bin/python task7/inference/diag/rear_knee_load.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "task7" / "inference"))
sys.path.insert(0, str(ROOT / "task7" / "realbot"))

import cpg_max                      # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402

TRAJ = ROOT / "task7/outputs/gait/walk_kp120_first.json"
TRIPS = [ROOT / "task7/logs/m_logs_trip16/M9_20260902_170254.json",
         ROOT / "task7/logs/m_logs_trip16/M9_20260902_170754.json"]

# max_model 腿序 (FR, FL, RR, RL) → SHM 腿名。與 gen_gait_traj.MM2SHM 同一份。
MM2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
IS_FRONT = np.array([True, True, False, False])   # FR, FL, RR, RL


def leg_jacobian(k: int, q3: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(3, 3) 數值 Jacobian ∂(足端 xyz)/∂(q1, q2, q3)，機身座標系。"""
    J = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = eps
        J[:, j] = (leg_kin.fk(k, q3 + dq) - leg_kin.fk(k, q3 - dq)) / (2 * eps)
    return J


def replay_cpg(params: dict, n_tot: int) -> np.ndarray:
    """重放產生 trip16 軌跡的那個 CPG，回傳 (n_tot, 4) 的 theta。

    ⚠️ 必須與 `gen_gait_traj` 逐行一致：那裡是「先用 c 算 q、再 step」，
    所以第 i 幀對應的是 step 之前的 theta。
    """
    B = params["baseline_ref"]
    p = params["params"]
    step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    mux, muy = np.full(4, B["mu_x"]), np.full(4, B["mu_y"])
    om = np.full(4, p["omega"])
    th = np.zeros((n_tot, 4))
    for i in range(n_tot):
        th[i] = c["theta"]
        c = step(c, mux, muy, om, mm.CTRL_DT)
    return th


# =============================================================================
def part1_geometry(traj: dict, th: np.ndarray, stance: np.ndarray) -> dict:
    """H-A：純幾何。單位足端力 → 膝力矩的係數，逐腿逐幀。"""
    Q = np.asarray(traj["q"]).reshape(-1, 4, 3)     # (n, 4腿, 3關節)，腿序 = mm.LEGS
    n = Q.shape[0]
    a_z = np.zeros((n, 4))      # 垂直力 1 N → 膝力矩 (N·m)
    a_x = np.zeros((n, 4))      # 前後力 1 N → 膝力矩 (N·m)
    for i in range(n):
        for k in range(4):
            J = leg_jacobian(k, Q[i, k])
            a_z[i, k] = -J[2, 2]
            a_x[i, k] = -J[0, 2]

    print("\n" + "=" * 74)
    print("Part 1 — H-A 幾何：單位足端力造成多少膝力矩（站立相平均，N·m per N）")
    print("=" * 74)
    print(f"{'腿':>4} {'|a_z| 垂直':>12} {'|a_x| 前後':>12} {'a_x/a_z':>10}")
    out = {}
    for k, leg in enumerate(mm.LEGS):
        m = stance[:, k]
        az, ax = float(np.abs(a_z[m, k]).mean()), float(np.abs(a_x[m, k]).mean())
        out[leg] = (az, ax)
        print(f"{leg:>4} {az:12.4f} {ax:12.4f} {ax / az:10.2f}")
    fz = np.mean([out[l][0] for l in ("FR", "FL")])
    rz = np.mean([out[l][0] for l in ("RR", "RL")])
    fx = np.mean([out[l][1] for l in ("FR", "FL")])
    rx = np.mean([out[l][1] for l in ("RR", "RL")])
    print(f"\n  前腿平均 a_z={fz:.4f}  a_x={fx:.4f}")
    print(f"  後腿平均 a_z={rz:.4f}  a_x={rx:.4f}")
    print(f"  ★ 後/前 力臂比：垂直 {rz / fz:.2f}×　前後 {rx / fx:.2f}×")
    return {"a_z": a_z, "a_x": a_x, "ratio_z": rz / fz, "ratio_x": rx / fx,
            "front_z": fz, "rear_z": rz, "front_x": fx, "rear_x": rx}


# =============================================================================
def part2_real(th: np.ndarray, stance: np.ndarray, traj: dict) -> dict:
    """實機膝力矩按 CPG 相位拆解：後膝的峰值落在站立相還是擺動相？"""
    print("\n" + "=" * 74)
    print("Part 2 — 實機 trip16：膝力矩按相位拆（|τ| 平均 / 峰值，N·m）")
    print("=" * 74)
    res = {}
    for tp in TRIPS:
        d = json.loads(tp.read_text())
        t0, gdt, ng = d["t_gait0"], d["gait_dt"], d["n_gait"]
        rows = []
        for s in d["samples"]:
            if s["phase"] != "GAIT":
                continue
            i = int(round((s["t"] - t0) / gdt))
            if 0 <= i < ng:
                rows.append((i, s))
        print(f"\n{tp.name}　GAIT 取樣 {len(rows)} 筆　幀 {rows[0][0]}–{rows[-1][0]}")
        print(f"{'關節':>16} {'站立|τ|':>9} {'站立峰':>8} {'擺動|τ|':>9} {'擺動峰':>8}"
              f" {'峰在':>6}")
        for k, leg in enumerate(mm.LEGS):
            name = MM2SHM[leg] + "3_knee_pitch"
            st, sw = [], []
            for i, s in rows:
                tau = abs(s["j"][name][2])
                (st if stance[i, k] else sw).append(tau)
            st, sw = np.array(st), np.array(sw)
            res.setdefault(name, []).append((st.mean(), st.max(), sw.mean(), sw.max()))
            where = "站立" if st.max() >= sw.max() else "擺動"
            print(f"{name:>16} {st.mean():9.2f} {st.max():8.2f} "
                  f"{sw.mean():9.2f} {sw.max():8.2f} {where:>6}")
    return res


def part2c_profile(th: np.ndarray, traj: dict, nbin: int = 12) -> None:
    """膝力矩沿「一個步態週期」的分佈 —— 峰值到底落在週期的哪裡？

    Part 2 顯示站立相**平均**力矩前後幾乎一樣（16 vs 16），只有**峰值**差 1.5 倍。
    平均一樣代表不是穩態承重的差異；那峰值就一定是某個瞬態，要定位它在哪一格。
    """
    duty = traj["params"]["duty"]
    ph = (cpg_max.duty_remap(th, duty) % (2 * np.pi)) / (2 * np.pi)   # 0~1，前半擺動
    sw = 1.0 - duty
    print("\n" + "=" * 74)
    print(f"Part 2c — 膝 |τ| 沿週期的分佈（{nbin} 格，兩趟合併，N·m）")
    print(f"           格 0~{int(sw * nbin) - 1} = 擺動相，之後 = 站立相；"
          f"觸地發生在格 {int(sw * nbin)} 附近")
    print("=" * 74)
    acc = {}
    for tp in TRIPS:
        d = json.loads(tp.read_text())
        t0, gdt, ng = d["t_gait0"], d["gait_dt"], d["n_gait"]
        for s in d["samples"]:
            if s["phase"] != "GAIT":
                continue
            i = int(round((s["t"] - t0) / gdt))
            if not (0 <= i < ng):
                continue
            for k, leg in enumerate(mm.LEGS):
                name = MM2SHM[leg] + "3_knee_pitch"
                b = min(int(ph[i, k] * nbin), nbin - 1)
                acc.setdefault(name, [[] for _ in range(nbin)])[b].append(
                    abs(s["j"][name][2]))
    hdr = "".join(f"{b:>6}" for b in range(nbin))
    print(f"{'關節':>16}{hdr}")
    for k, leg in enumerate(mm.LEGS):
        name = MM2SHM[leg] + "3_knee_pitch"
        row = "".join(f"{max(v) if v else 0:6.1f}" for v in acc[name])
        print(f"{name:>16}{row}   ← 峰值")
    print()
    for k, leg in enumerate(mm.LEGS):
        name = MM2SHM[leg] + "3_knee_pitch"
        row = "".join(f"{np.mean(v) if v else 0:6.1f}" for v in acc[name])
        print(f"{name:>16}{row}   ← 平均")


def part2d_control_law(th: np.ndarray, traj: dict) -> None:
    """把實機膝力矩拆成 kp·(q_des−q) 與 −kd·v 兩項。

    ★ 這是決定性的一問：若峰值由 `kp·誤差` 主導，那它是**追蹤誤差**，
    也就是「命令要腿去的地方腿去不了」，而不是外部負載把腿壓彎。
    兩者的解法完全相反（前者改軌跡，後者改增益／配重）。
    見記憶 `diagnostic-tools-lie`：kp·|err| + kd·|v| 是控制律的力矩上限。
    """
    kp, kd = traj["params"]["kp"], traj["params"]["kd"]
    print("\n" + "=" * 74)
    print(f"Part 2d — 膝力矩的控制律拆解（kp={kp} kd={kd}）：τ = kp·err − kd·v")
    print("=" * 74)
    print(f"{'關節':>16} {'峰值τ':>8} {'該刻 err':>9} {'kp·err':>8} "
          f"{'該刻 v':>8} {'kd·v':>7} {'合計':>8} {'相位':>6}")
    duty = traj["params"]["duty"]
    ph = (cpg_max.duty_remap(th, duty) % (2 * np.pi)) / (2 * np.pi)
    sw = 1.0 - duty
    for tp in TRIPS[:1]:
        d = json.loads(tp.read_text())
        t0, gdt, ng = d["t_gait0"], d["gait_dt"], d["n_gait"]
        for k, leg in enumerate(mm.LEGS):
            name = MM2SHM[leg] + "3_knee_pitch"
            best = None
            for s in d["samples"]:
                if s["phase"] != "GAIT":
                    continue
                i = int(round((s["t"] - t0) / gdt))
                if not (0 <= i < ng):
                    continue
                q, qd, tau, v = s["j"][name]
                if best is None or abs(tau) > abs(best[2]):
                    best = (q, qd, tau, v, i, k)
            q, qd, tau, v, i, k = best
            err = qd - q
            where = "擺動" if ph[i, k] < sw else "站立"
            print(f"{name:>16} {tau:8.2f} {err:9.4f} {kp * err:8.2f} "
                  f"{v:8.3f} {-kd * v:7.2f} {kp * err - kd * v:8.2f} {where:>6}")


def part2e_sink(th: np.ndarray, traj: dict, geo: dict, nbin: int = 12) -> None:
    """把追蹤誤差換算成**足端下沉量**，並看機身俯仰。

    Part 2d 只取了峰值那一刻。這裡看整個站立相的持續量 ——
    如果後膝是「持續」比前膝彎得多，那是靜態順從性（幾何）；
    如果只有觸地那幾格，那是衝擊。兩者的處置不同。

    足端下沉 δp = a_z · δq（a_z 是 Part 1 的垂直力臂）。
    ⚠️ 前後膝的關節正負號相反（X 型站姿，knee_sign ∓1），
       所以要各自乘 knee_sign 才能比較「彎得多還是少」。
    """
    duty = traj["params"]["duty"]
    ph = (cpg_max.duty_remap(th, duty) % (2 * np.pi)) / (2 * np.pi)
    sw = 1.0 - duty
    ks = leg_kin.knee_sign_of(mm.HOME)
    print("\n" + "=" * 74)
    print("Part 2e — 站立相的膝追蹤誤差 → 足端下沉（mm）＋機身俯仰")
    print("=" * 74)
    acc = {l: [] for l in mm.LEGS}
    pitch_g = []
    for tp in TRIPS:
        d = json.loads(tp.read_text())
        t0, gdt, ng = d["t_gait0"], d["gait_dt"], d["n_gait"]
        for s in d["samples"]:
            if s["phase"] != "GAIT":
                continue
            i = int(round((s["t"] - t0) / gdt))
            if not (0 <= i < ng):
                continue
            pitch_g.append(s["pitch"])
            for k, leg in enumerate(mm.LEGS):
                if ph[i, k] < sw:            # 只看站立相
                    continue
                q, qd, tau, v = s["j"][MM2SHM[leg] + "3_knee_pitch"]
                # 前腿膝角是負的（更彎 = 更負 → qd−q > 0），後腿膝角是正的
                # （更彎 = 更正 → qd−q < 0）。乘 −knee_sign 統一成「>0 = 比命令更彎」。
                acc[leg].append(-ks[k] * (qd - q))

    print(f"{'腿':>4} {'膝誤差 rad':>11} {'→ 足端下沉 mm':>15} {'相對前腿':>9}")
    sink = {}
    for k, leg in enumerate(mm.LEGS):
        e = float(np.mean(acc[leg]))
        sink[leg] = e * abs(geo["a_z"][:, k]).mean() * 1000
        print(f"{leg:>4} {e:11.4f} {sink[leg]:15.1f}", end="")
        print(f" {'—':>9}" if k < 2 else
              f" {sink[leg] / np.mean([sink['FR'], sink['FL']]):9.2f}×")
    fs = np.mean([sink["FR"], sink["FL"]])
    rs = np.mean([sink["RR"], sink["RL"]])
    print(f"\n  前腳平均下沉 {fs:.1f} mm　後腳 {rs:.1f} mm　→ 後/前 {rs / fs:.2f}×")
    print(f"  兩腳下沉差 {rs - fs:+.1f} mm，前後輪距 {2 * mm.HIP_X * 1000:.0f} mm"
          f" → 幾何上應有俯仰 {np.degrees(np.arctan2(rs - fs, 2 * mm.HIP_X * 1000)):+.2f}°")
    print(f"  實測 GAIT 段機身俯仰：平均 {np.mean(pitch_g):+.2f}°　"
          f"範圍 {np.min(pitch_g):+.2f} ~ {np.max(pitch_g):+.2f}°")


def part2b_wheels(th: np.ndarray, stance: np.ndarray) -> None:
    """輪子力矩按相位拆 —— 站立相輪子在出什麼力？（阻尼 kd=0.5，kp=0）"""
    print("\n" + "=" * 74)
    print("Part 2b — 實機 trip16：站立相的輪速與輪力矩（前後對抗的直接證據）")
    print("=" * 74)
    import coord
    for tp in TRIPS:
        d = json.loads(tp.read_text())
        t0, gdt, ng = d["t_gait0"], d["gait_dt"], d["n_gait"]
        print(f"\n{tp.name}")
        print(f"{'輪':>10} {'站立平均轉速':>13} {'站立|τ|':>9} {'擺動平均轉速':>13}")
        for k, leg in enumerate(mm.LEGS):
            w = MM2SHM[leg] + "4_foot"
            sg = coord.SIGN.get(w, 1.0) if hasattr(coord, "SIGN") else 1.0
            sv_st, sv_sw, tt = [], [], []
            for s in d["samples"]:
                if s["phase"] != "GAIT":
                    continue
                i = int(round((s["t"] - t0) / gdt))
                if not (0 <= i < ng):
                    continue
                v, tau = s["w"][w][1] * sg, abs(s["w"][w][2])
                if stance[i, k]:
                    sv_st.append(v)
                    tt.append(tau)
                else:
                    sv_sw.append(v)
            print(f"{w:>10} {np.mean(sv_st):13.3f} {np.mean(tt):9.2f} "
                  f"{np.mean(sv_sw):13.3f}")


# =============================================================================
def part3_forces(traj: dict) -> dict:
    """H-B / H-C：模擬裡走同一組參數，量四腳的法向力與縱向力。"""
    import cpg_walk_max as cw
    print("\n" + "=" * 74)
    print("Part 3 — 模擬：站立相的足端接觸力（法向 Fz / 縱向 Fx，N）")
    print("=" * 74)
    p, B = traj["params"], traj["baseline_ref"]
    r = cw.Robot(kp3=np.full(3, p["kp"]), kd3=np.full(3, p["kd"]))
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    r.reset_standing(cpg_max.stand_targets(ks, f0, p["x_off"]),
                     mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, p["x_off"]), "damp")
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    n = int(round(p["secs"] / mm.CTRL_DT))
    Fz = np.zeros((n, 4))
    Fx = np.zeros((n, 4))
    TH = np.zeros((n, 4))
    VF = np.zeros((n, 4, 3))        # 足端在世界系的速度
    for i in range(n):
        TH[i] = c["theta"]
        q, _ = cpg_max.joint_targets(c, f0, p["x_off"], p["g_c"], p["d_step"],
                                     B["d_step_y"], p["duty"], ks, p["z_sag"])
        r.step(q, "damp")
        Fz[i], Fx[i] = foot_force_xz(r)
        VF[i] = foot_vel_world(r)
        c = step(c, np.full(4, B["mu_x"]), np.full(4, B["mu_y"]),
                 np.full(4, p["omega"]), mm.CTRL_DT)

    st = np.sin(cpg_max.duty_remap(TH, p["duty"])) <= 0

    # 觸地瞬間（擺動→站立那一幀）足端相對地面的速度。★ 垂直速度四腿由構造相同
    # （同一個 g_c、同一個 ω），所以差別只可能出在水平方向 —— 那就是「對抗」。
    print(f"{'腿':>4} {'觸地次數':>9} {'觸地 vx m/s':>12} {'觸地 vz m/s':>12}")
    for k, leg in enumerate(mm.LEGS):
        idx = np.where(st[1:, k] & ~st[:-1, k])[0] + 1
        if len(idx) == 0:
            print(f"{leg:>4} {'—':>9}")
            continue
        print(f"{leg:>4} {len(idx):9d} {VF[idx, k, 0].mean():12.3f} "
              f"{VF[idx, k, 2].mean():12.3f}")

    print()
    print(f"{'腿':>4} {'站立 Fz 平均':>13} {'Fz 峰':>8} {'站立 Fx 平均':>13} {'|Fx| 峰':>8}")
    out = {}
    for k, leg in enumerate(mm.LEGS):
        m = st[:, k]
        out[leg] = (float(Fz[m, k].mean()), float(Fz[m, k].max()),
                    float(Fx[m, k].mean()), float(np.abs(Fx[m, k]).max()))
        print(f"{leg:>4} {out[leg][0]:13.1f} {out[leg][1]:8.1f} "
              f"{out[leg][2]:13.1f} {out[leg][3]:8.1f}")
    fz = np.mean([out[l][0] for l in ("FR", "FL")])
    rz = np.mean([out[l][0] for l in ("RR", "RL")])
    print(f"\n  ★ 承重前後比（站立相平均 Fz）：前 {fz:.1f} N / 後 {rz:.1f} N "
          f"= 後/前 {rz / fz:.2f}×")
    return out


def foot_vel_world(r) -> np.ndarray:
    """(4, 3) 四個輪心在世界座標系的線速度。"""
    import mujoco
    v = np.zeros((4, 3))
    buf = np.zeros(6)
    for k, b in enumerate(r.foot_bid):
        mujoco.mj_objectVelocity(r.m, r.d, mujoco.mjtObj.mjOBJ_BODY, b, buf, 0)
        v[k] = buf[3:6]          # flg_local=0 → 世界系；前三個是角速度
    return v


def foot_force_xz(r) -> tuple[np.ndarray, np.ndarray]:
    """四輪的地面接觸力（世界系 x 與 z 分量），順序同 LEGS。

    `Robot.foot_forces()` 只回法向；這裡要縱向分量才能把「承重」與「推進」拆開。
    """
    import mujoco
    d, m = r.d, r.m
    body_of = {b: k for k, b in enumerate(r.foot_bid)}
    fz, fx = np.zeros(4), np.zeros(4)
    buf = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        b1 = m.geom_bodyid[con.geom1]
        b2 = m.geom_bodyid[con.geom2]
        k = body_of.get(b1, body_of.get(b2))
        if k is None:
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        # 接觸座標系 → 世界系。frame 是 row-major 的 3×3，第一列是法向。
        fw = con.frame.reshape(3, 3).T @ buf[:3]
        sgn = 1.0 if body_of.get(b2) == k else -1.0   # 力的方向依 geom 順序
        fz[k] += sgn * fw[2]
        fx[k] += sgn * fw[0]
    return np.abs(fz), fx


# =============================================================================
def main() -> int:
    traj = json.loads(TRAJ.read_text())
    p = traj["params"]
    n_tot = traj["n"]
    th = replay_cpg(traj, n_tot)
    stance = np.sin(cpg_max.duty_remap(th, p["duty"])) <= 0     # True = 站立相

    print("=" * 74)
    print("C — 後膝負擔來源拆解　trip16 (kp120 / ω1.4 / duty0.8 / d_step0.10)")
    print("=" * 74)
    print(f"軌跡 {TRAJ.name}　{n_tot} 幀 @ {1/mm.CTRL_DT:.0f} Hz")
    print(f"站立相佔比（逐腿）：" +
          "  ".join(f"{l} {stance[:, k].mean():.2f}"
                    for k, l in enumerate(mm.LEGS)))

    g = part1_geometry(traj, th, stance)
    part2_real(th, stance, traj)
    part2c_profile(th, traj)
    part2d_control_law(th, traj)
    part2e_sink(th, traj, g)
    part2b_wheels(th, stance)
    f = part3_forces(traj)

    # ------------------------------------------------------------- 合成
    print("\n" + "=" * 74)
    print("Part 4 — 合成：預測 vs 實測")
    print("=" * 74)
    fz = np.mean([f[l][0] for l in ("FR", "FL")])
    rz = np.mean([f[l][0] for l in ("RR", "RL")])
    fx = np.mean([abs(f[l][2]) for l in ("FR", "FL")])
    rx = np.mean([abs(f[l][2]) for l in ("RR", "RL")])
    tau_f = g["front_z"] * fz + g["front_x"] * fx
    tau_r = g["rear_z"] * rz + g["rear_x"] * rx
    print(f"  前膝預測 |τ| = {g['front_z']:.4f}×{fz:.1f} + {g['front_x']:.4f}×{fx:.1f}"
          f" = {tau_f:.1f} N·m")
    print(f"  後膝預測 |τ| = {g['rear_z']:.4f}×{rz:.1f} + {g['rear_x']:.4f}×{rx:.1f}"
          f" = {tau_r:.1f} N·m")
    print(f"  預測後/前 = {tau_r / tau_f:.2f}×　實機後/前 = {60.7 / 40.1:.2f}×"
          f"（60.7 / 40.1）")
    print(f"\n  三個假說各自的貢獻比（後/前）：")
    print(f"    H-A 幾何力臂　垂直 {g['ratio_z']:.2f}×　前後 {g['ratio_x']:.2f}×")
    print(f"    H-B 承重　　　{rz / fz:.2f}×")
    print(f"    H-C 縱向力　　{rx / max(fx, 1e-9):.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
