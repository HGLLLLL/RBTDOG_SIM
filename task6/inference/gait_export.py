"""gait_export.py —— 把 walk_stable 步態導出成實機可播放的關節軌跡，並做上機前的離線檢驗。

角色：本檔在【開發機】執行（要用 MuJoCo）。狗上執行的是 task6/realbot/L7_gait_shm.py。

管線：
    cpg_walk_d1.make_cpg_step + joint_targets     ← import，不複製
      → 每 20 ms 的 12 軸 MJCF 角
      → calib_map.mjcf12_to_shm
      → 五項離線檢驗
      → gait_walk_stable.npz

⚠️ 本檔【不修改】cpg_walk_d1 的任何步態參數 —— 那支腳本產生了對照影片。
   部署用的 G_C 是本檔自己的 DEPLOY_G_C（見 §為什麼不是 0.12）。

================================================================================
§ 為什麼部署用 G_C=0.110 而不是影片的 0.12
================================================================================
影片版 G_C=0.12 的膝關節在擺動相會折到距 ctrlrange 只剩 0.0114 rad（0.65°）。
而 calib_map 的 offset 誤差量級恰好就在這個尺度上（以 POSE_LIE 對照推導限位，
leg0 abad 超出 0.0089、leg2 超出 0.0138 rad）。也就是說實機的膝很可能會頂到
機構限位，而模擬顯示 clip 0.000%，離線完全看不出來。

頂限位時馬達會持續對機構硬推，靠事後力矩中止來處理是把可預防的問題留到現場，
所以改成事前用餘裕門檻擋掉。掃描結果（--sweep 可重現）：

    G_C     膝餘裕(rad)   (度)    最大角速度    離地量 FL/FR/RL/RR (mm)
    0.100     0.1266     7.25      12.27       50.6  49.9  46.2  45.9  ✓
    0.105     0.0978     5.60      12.88       54.7  54.2  49.4  49.4  ✓
    0.110     0.0690     3.95      13.49       59.5  59.0  53.1  53.4  ✓  ← 採用
    0.115     0.0402     2.30      14.10       66.3  64.7  56.3  56.0  ✗
    0.120     0.0114     0.65      14.72       76.7  77.0  56.6  55.6  ✗  ← 影片版

代價：前腳離地量從 77 掉到 59 mm（−23%）。後腳幾乎不變（56.6 → 53.1）。
副作用是前後更均勻了（比值 1.37 → 1.12）。
"""
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "realbot"))

import calib_map
import cpg_d1
import cpg_walk_d1 as W
import d1_model

JN = ("abad", "hip", "knee")

GAIT = "walk_stable"
DEPLOY_G_C = 0.110          # 見檔頭 §。影片版是 cpg_walk_d1.GAIT_G_C = 0.12
MARGIN_MIN = 0.05           # 限位餘裕門檻(rad) = 2.9°

# 跨幀跳變門檻(rad)。步態本身單 tick 最大約 0.27 rad（13.5 rad/s × 0.02s），
# 取 0.40 留餘裕：正常軌跡不會碰到，注入式的階躍會被抓到。
JUMP_MAX = 0.40


def shm_limits(m):
    """MJCF ctrlrange → SHM 慣例限位。回傳 {(shm_leg, joint_name): (lo, hi)}。

    ⚠️ 轉換是 shm = sign * mjcf + offset，所以 **sign = -1 時上下界會對調**。
       calib_map 裡 FR/RR 的 hip 與 knee 都是 -1。忘了 sorted() 的話限位檢驗
       會恆為「超限」或恆為「通過」，兩種都是靜默失效。
    """
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = {}
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(JN):
            col = mjcf_leg * 3 + j
            s, o = calib_map.CALIB[shm_leg][jn]
            out[(shm_leg, jn)] = tuple(sorted((s * lo[col] + o, s * hi[col] + o)))
    return out


def build_trajectory(m, g_c, secs=20.0):
    """產生步態的關節軌跡。回傳 (q_mjcf (N,12), q_shm (N,4,3))。

    軌跡邏輯全部來自 cpg_walk_d1（import，不複製）。本函式只負責：
    指定 g_c、跑滿 secs、把結果轉成 SHM 慣例。
    """
    cfg = W.GAITS[GAIT]
    step = W.make_cpg_step(cfg["phase"])
    f0s, jinvs = cpg_d1.leg_ik_consts(m)

    c = cpg_d1.cpg_init()
    c["theta"] = cfg["phase"].copy()      # cpg_init 用的是 d1_model 的 trot 相位
    n = int(secs / d1_model.CTRL_DT)
    q_mjcf = np.zeros((n, 12))
    for i in range(n):
        c = step(c, np.full(4, cfg["mu_x"]), np.full(4, W.MU_Y),
                 np.full(4, cfg["omega"]), d1_model.CTRL_DT)
        q_mjcf[i] = W.joint_targets(c, f0s, jinvs, cfg["x_off"], g_c, cfg["duty"])

    q_shm = np.zeros((n, 4, 3))
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(JN):
            s, o = calib_map.CALIB[shm_leg][jn]
            q_shm[:, shm_leg, j] = s * q_mjcf[:, mjcf_leg * 3 + j] + o
    return q_mjcf, q_shm


def worst_margin(q_shm, lim):
    """最小限位餘裕(rad) 與發生在哪一軸。負值代表已經超限。"""
    from shm_common import LEGNAME
    worst, where = np.inf, ""
    for shm_leg in range(4):
        for j, jn in enumerate(JN):
            lo, hi = lim[(shm_leg, jn)]
            col = q_shm[:, shm_leg, j]
            for margin, side in ((col.min() - lo, "下界"), (hi - col.max(), "上界")):
                if margin < worst:
                    worst, where = margin, f"leg{shm_leg}({LEGNAME[shm_leg]}).{jn} {side}"
    return float(worst), where


def max_joint_vel(q_shm):
    """逐幀差分的最大關節角速度(rad/s)。用來設 L7 的 VEL_ABORT。"""
    return float(np.abs(np.diff(q_shm, axis=0) / d1_model.CTRL_DT).max())


def sweep(m, g_c_values):
    """印出 G_C 掃描表。用來重現檔頭 § 的那張表。"""
    lim = shm_limits(m)
    print(f"{'G_C':>6} {'膝餘裕(rad)':>12} {'(度)':>7} {'最大角速度':>11}  ")
    for g_c in g_c_values:
        _, q_shm = build_trajectory(m, g_c)
        margin, _ = worst_margin(q_shm, lim)
        mark = "✓" if margin >= MARGIN_MIN else "✗"
        print(f"{g_c:>6.3f} {margin:>12.4f} {np.degrees(margin):>7.2f} "
              f"{max_joint_vel(q_shm):>11.2f}  {mark}")


def calib_hash():
    """calib_map 內容的雜湊。npz 帶著它，L7 上機前比對，不符就拒跑。"""
    payload = json.dumps(
        {"legs": calib_map.LEG_MJCF2SHM,
         "calib": {str(k): {jn: list(v[jn]) for jn in JN}
                   for k, v in sorted(calib_map.CALIB.items())}},
        sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def run_checks(m, q_mjcf, q_shm):
    """五項離線檢驗。回傳 (ok, 問題清單, 統計)。"""
    from shm_common import POSE_STAND
    lim = shm_limits(m)
    problems = []

    # 1) 限位餘裕
    margin, where = worst_margin(q_shm, lim)
    if margin < MARGIN_MIN:
        problems.append(f"限位餘裕不足：{where} 只剩 {margin:.4f} rad "
                        f"（{np.degrees(margin):.2f}°），門檻 {MARGIN_MIN}")

    # 2) 角速度（不擋，只記錄——L7 拿它設 VEL_ABORT）
    vmax = max_joint_vel(q_shm)

    # 3) 跨幀跳變
    jump = np.abs(np.diff(q_shm, axis=0))
    if jump.max() > JUMP_MAX:
        idx = np.unravel_index(jump.argmax(), jump.shape)
        problems.append(f"跨幀跳變過大：{jump.max():.4f} rad @ 幀{idx[0]} "
                        f"leg{idx[1]}.{JN[idx[2]]}，門檻 {JUMP_MAX}")

    # 4) 起步位移（不擋，只記錄——L7 拿它算 ramp 時間）
    start_off = max(abs(q_shm[0, leg, j] - POSE_STAND[leg][jn])
                    for leg in range(4) for j, jn in enumerate(JN))

    # 5) MJCF 側 ctrlrange clip 率必須為 0
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    clipped = int(np.sum((q_mjcf < lo - 1e-9) | (q_mjcf > hi + 1e-9)))
    clip_pct = 100.0 * clipped / q_mjcf.size
    if clipped:
        problems.append(f"MJCF ctrlrange clip {clip_pct:.3f}%（{clipped} 個指令），必須為 0")

    stats = {"worst_margin": float(margin), "worst_margin_at": where,
             "max_joint_vel": float(vmax), "max_frame_jump": float(jump.max()),
             "start_offset_from_stand": float(start_off), "clip_pct": clip_pct}
    return not problems, problems, stats


AIR_SIM_GRID = ((20.0, 0.7), (40.0, 1.0))     # (kp, kd)。20/0.7 是原廠站立實測值
AIR_SIM_SCALES = (0.25, 0.5, 1.0)


def air_servo_sim(m, q_mjcf, kp, kd, time_scale, settle_sec=1.0):
    """吊掛空跑的直接模擬：機身固定、無接觸、位置伺服 kp/kd。

    量的是【實際會發生的事】——致動器力矩、追蹤誤差、關節速度峰值。

    ⚠️ 不要改用 mj_inverse。逆動力學會得到 45.8 N·m（超過 URDF 的 28 N·m
       effort 上限），因為 duty_remap 在擺動→站立交界處讓 dθ/dt 差 4 倍，
       指令軌跡有速度折點，二階差分的加速度會爆掉。那描述的是「完美追蹤
       所需的力矩」，而完美追蹤本來就不會發生——位置伺服必然落後。
    """
    m2 = mujoco.MjModel.from_xml_path(d1_model.SCENE)
    m2.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for a in range(m2.nu):                     # 覆寫位置伺服增益
        m2.actuator_gainprm[a][0] = kp
        m2.actuator_biasprm[a][1] = -kp
        m2.actuator_biasprm[a][2] = -kd

    d = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d, 0)
    base_qpos = d.qpos[:7].copy()              # 吊具：機身固定在初始位姿
    lo, hi = m2.actuator_ctrlrange[:, 0], m2.actuator_ctrlrange[:, 1]
    idx, vidx = d1_model.LEG_QPOS_IDX, d1_model.LEG_QVEL_IDX
    dt = d1_model.CTRL_DT

    def sample(u):
        x = np.clip(u / dt, 0.0, len(q_mjcf) - 1)
        i0 = int(np.floor(x))
        i1 = min(i0 + 1, len(q_mjcf) - 1)
        w = x - i0
        return q_mjcf[i0] * (1 - w) + q_mjcf[i1] * w

    def pin():
        d.qpos[:7] = base_qpos
        d.qvel[:6] = 0.0

    for _ in range(int(settle_sec / m2.opt.timestep)):    # 先靜置到第 0 幀
        d.ctrl[:] = np.clip(q_mjcf[0], lo, hi)
        pin()
        mujoco.mj_step(m2, d)

    wall = (len(q_mjcf) - 1) * dt / time_scale
    tau_pk = vel_pk = 0.0
    errs = []
    for k in range(int(wall / m2.opt.timestep)):
        tgt = sample(k * m2.opt.timestep * time_scale)
        d.ctrl[:] = np.clip(tgt, lo, hi)
        pin()
        mujoco.mj_step(m2, d)
        tau_pk = max(tau_pk, float(np.abs(d.actuator_force).max()))
        vel_pk = max(vel_pk, float(np.abs(d.qvel[vidx]).max()))
        errs.append(np.abs(d.qpos[idx] - tgt))
    errs = np.asarray(errs)
    return {"tau_peak": tau_pk,
            "err_peak_deg": float(np.degrees(errs.max())),
            "err_rms_deg": float(np.degrees(np.sqrt((errs ** 2).mean()))),
            "vel_peak": vel_pk}


def air_sim_table(m, q_mjcf):
    """對 AIR_SIM_GRID × AIR_SIM_SCALES 全部算一遍。寫進 npz 的 meta。"""
    table = {}
    for kp, kd in AIR_SIM_GRID:
        key = f"{kp}/{kd}"
        table[key] = {}
        for s in AIR_SIM_SCALES:
            r = air_servo_sim(m, q_mjcf, kp, kd, s)
            table[key][str(s)] = r
            print(f"  kp={kp} kd={kd} {s:>5.2f}×  力矩 {r['tau_peak']:6.2f} N·m  "
                  f"誤差峰值 {r['err_peak_deg']:6.2f}°  RMS {r['err_rms_deg']:5.2f}°  "
                  f"速度 {r['vel_peak']:5.2f} rad/s")
    return table


def export(m, out_path, g_c=DEPLOY_G_C, secs=20.0):
    """產生軌跡、跑檢驗、寫 npz。檢驗沒過就 sys.exit(1) 且不留檔案。"""
    cfg = W.GAITS[GAIT]
    q_mjcf, q_shm = build_trajectory(m, g_c, secs)
    ok, problems, stats = run_checks(m, q_mjcf, q_shm)

    print(f"[檢驗] G_C={g_c}  最小餘裕 {stats['worst_margin']:.4f} rad "
          f"（{np.degrees(stats['worst_margin']):.2f}°）@ {stats['worst_margin_at']}")
    print(f"       最大角速度 {stats['max_joint_vel']:.2f} rad/s  "
          f"最大跨幀跳變 {stats['max_frame_jump']:.4f} rad  "
          f"起步位移 {stats['start_offset_from_stand']:.4f} rad  "
          f"clip {stats['clip_pct']:.3f}%")
    if not ok:
        for p in problems:
            print(f"  ✗ {p}")
        print("→ 拒絕匯出。修正參數後重跑。")
        sys.exit(1)

    print("[空中模擬] 機身固定、無接觸，量實際力矩與追蹤誤差：")
    air = air_sim_table(m, q_mjcf)

    meta = {"gait": GAIT, "g_c": float(g_c), "omega": cfg["omega"],
            "mu_x": cfg["mu_x"], "mu_y": W.MU_Y, "x_off": cfg["x_off"],
            "duty": cfg["duty"], "ctrl_dt": d1_model.CTRL_DT, "secs": float(secs),
            "calib_hash": calib_hash(), "air_sim": air, **stats}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    np.savez(out_path,
             t=np.arange(len(q_mjcf)) * d1_model.CTRL_DT,
             q_mjcf=q_mjcf, q_shm=q_shm, f0s=f0s, jinvs=jinvs,
             meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    print(f"[匯出] {out_path}  {len(q_mjcf)} 幀  calib_hash={meta['calib_hash']}")
    return out_path


def analyze(log_path):
    """讀實機 log，算逐軸追蹤誤差。回傳 {'axes': {...}, 'overrun_pct': float, 'meta': {...}}。

    相位延遲用互相關求：把實際角相對指令角平移，找誤差平方和最小的位移量。
    只分析被驅動的腿——跳過的腿全程零增益，它的「誤差」沒有意義。
    """
    from shm_common import LEGNAME
    z = np.load(log_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    cmd, p, t = z["cmd"], z["p"], z["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 500
    active = meta.get("active_legs", [0, 1, 2, 3])

    axes, rows = {}, []
    for leg in active:
        for j, jn in enumerate(JN):
            c, a = cmd[:, leg, j], p[:, leg, j]
            err = a - c
            rms = float(np.degrees(np.sqrt((err ** 2).mean())))
            mx = float(np.degrees(np.abs(err).max()))
            # 相位延遲：平移實際角，找誤差最小的位移
            best, best_k = np.inf, 0
            for k in range(0, min(200, len(c) // 4)):
                d_ = (a[k:] - c[:len(c) - k]) if k else (a - c)
                s = float((d_ ** 2).mean())
                if s < best:
                    best, best_k = s, k
            axes[(leg, jn)] = {"rms_deg": rms, "max_deg": mx,
                               "lag_ms": best_k * dt * 1000.0}
            rows.append((leg, jn, rms, mx, best_k * dt * 1000.0))

    over = float(100.0 * np.sum(z["overrun"]) / max(1, len(z["overrun"])))
    print(f"\n[分析] {log_path}  模式={meta.get('mode')}  "
          f"time_scale={meta.get('time_scale')}  kp={meta.get('kp')}")
    print(f"{'軸':<12} {'RMS(°)':>9} {'最大(°)':>9} {'延遲(ms)':>10}")
    for leg, jn, rms, mx, lag in rows:
        print(f"leg{leg}({LEGNAME[leg]}).{jn:<5} {rms:>9.2f} {mx:>9.2f} {lag:>10.1f}")
    print(f"\n500 Hz 週期超時 {over:.2f}%"
          + ("  ⚠️ 超過 1% 時追蹤誤差的解讀要打折" if over > 1.0 else ""))
    return {"axes": axes, "overrun_pct": over, "meta": meta}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="walk_stable 軌跡導出與離線檢驗")
    ap.add_argument("--sweep", action="store_true", help="印出 G_C 掃描表")
    ap.add_argument("--export", metavar="PATH",
                    default=None, help="匯出 npz 到指定路徑")
    ap.add_argument("--g-c", type=float, default=DEPLOY_G_C, dest="g_c")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--analyze", metavar="LOGPATH", default=None,
                    help="分析實機 log，輸出逐軸追蹤誤差")
    args = ap.parse_args()

    if not any((args.sweep, args.export, args.analyze)):
        ap.error("要 --sweep / --export / --analyze 其中之一")

    if args.sweep or args.export:
        model = d1_model.make_model()
        if args.sweep:
            sweep(model, (0.080, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120))
        if args.export:
            export(model, args.export, args.g_c, args.secs)
    if args.analyze:
        analyze(args.analyze)
