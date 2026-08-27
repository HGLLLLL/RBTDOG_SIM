#!/usr/bin/env python3
"""M9 —— 步態（★★★ 狗第一次在地面上移動）。

════════════════════════════════════════════════════════════════════
這一步引入的三件新事情
════════════════════════════════════════════════════════════════════

  1. **連續執行** —— M7/M8 是走一條固定路徑；這裡是幾十秒的週期動作
  2. **★ 動態觸地** —— 今天所有實機資料都是準靜態（足端最快 0.57 m/s），
     基準步態的腳是以 **2.0 m/s** 砸向地面的
  3. **狗會在地面上移動** —— 到 2026-08-27 為止牠從來沒有過
     （所有輪子測試都是墊高離地做的）

★ 所以第一趟是**原地踏步**（`--march` / `d_step=0`）：
  引入 (1)(2) 但不引入 (3)。

════════════════════════════════════════════════════════════════════
兩種模式
════════════════════════════════════════════════════════════════════

  --traj FILE   播放 `inference/gen_gait_traj.py` 產生的軌跡檔。
                ★ **第一趟用這個** —— 每個關節角上機前都檢查過、
                  而且用 `play_gait_traj.py` 在 MuJoCo 完整播過。
  --live        狗上即時算 CPG（`realbot/cpg.py`）。
                ★ 之後要接**偏航閉迴路**只能走這條路 ——
                  文件記載唯一擋住「能直線走遠」的是偏航慢漂 −0.5~−0.8°/s，
                  播放固定檔案永遠解不了。
                實測純 Python 的 CPG+IK 是 0.043 ms/週期（預算 20 ms 的 0.2%）。

兩種模式的輸出必須一致 —— `tests/test_cpg_port.py` 逐幀比到 1e-11 rad。

════════════════════════════════════════════════════════════════════
⚠️ 保護門檻和 M7/M8 **不一樣**，不要照搬
════════════════════════════════════════════════════════════════════

步態的擺動相本來就需要 **6~8 rad/s** 的關節命令速度（M7/M8 的護欄是 2.0）。
所以 `--vcmd-max` / `--vmax` 都必須放寬 —— 但**放寬的是速度，不是力矩**。
力矩門檻維持 M8 的量級（實機承重峰值只用掉上限的 19%）。

★★ **中止語意與 M7/M8 相同**（承重）：凍結目標角、維持增益、原地撐住。
   放手＝狗塌下去。**不要按 Ctrl-C。**

════════════════════════════════════════════════════════════════════
前置條件
════════════════════════════════════════════════════════════════════
  - 狗趴在地上、16 顆洩力
  - **吊帶掛在 crouch（292 mm）以下**、鬆弛
  - ★ **地面淨空，而且要留出走動空間**（原地踏步在模擬裡仍會漂 ~10-400 mm）
  - 第二個終端機備著 `sudo ~/estop_max.sh`

用法：
    python3 M9_gait.py --traj march.json                    # 乾跑
    sudo python3 M9_gait.py --traj march.json --confirm
    sudo python3 M9_gait.py --live --march --secs 10 --confirm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import coord
import cpg
import kin
import shm_io
from M5_leg_pose import Keepalive, mc_ctrl_pid, proc_state, smoothstep
from M7_standup import TAU_HARD, read_imu_rp

LEGS12 = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]

# 力矩門檻。與 M8 同量級 —— 實機承重峰值只用掉馬達上限（150）的 19%，
# 模擬的步態峰值是 38 N·m。
TMAX = {"1_hip_roll": 50.0, "2_hip_pitch": 50.0, "3_knee_pitch": 70.0}

# 站起來用的增益與路徑（照抄 M7 已實機驗證的）
STANDUP_KP, STANDUP_KD = 250.0, 5.0


def build_standup(a, q_lie: dict, q_gait0: dict):
    """趴 → crouch → 步態起點。回傳 (名稱, 秒數, 起點, 終點)。"""
    crouch = dict(coord.POSES["crouch"])
    return [("RAMP_UP", a.ramp_kp, q_lie, q_lie),
            ("GO_crouch", a.t1, q_lie, crouch),
            ("HOLD_crouch", a.hold_mid, crouch, crouch),
            ("GO_stand", a.t2, crouch, q_gait0),
            ("HOLD_stand", a.hold, q_gait0, q_gait0)]


def build_sitdown(a, q_gait_end: dict, q_lie: dict):
    crouch = dict(coord.POSES["crouch"])
    return [("BACK_crouch", a.t2, q_gait_end, crouch),
            ("HOLDB_crouch", a.hold_mid, crouch, crouch),
            ("BACK_LIE", a.t1, crouch, q_lie),
            ("RAMP_DOWN", a.ramp_kp, q_lie, q_lie)]


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="M9 —— 步態（承重、連續、動態觸地）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traj", help="播放軌跡檔（gen_gait_traj.py 產生）")
    src.add_argument("--live", action="store_true", help="狗上即時算 CPG")
    ap.add_argument("--confirm", action="store_true", help="不帶就是乾跑")

    # ---- live 模式的步態參數
    ap.add_argument("--march", action="store_true", help="★ 原地踏步：d_step=0")
    ap.add_argument("--secs", type=float, default=10.0, help="步態本體秒數")
    ap.add_argument("--omega", type=float, default=0.5,
                    help="★ 從 0.5 開始。基準是 1.4，但那是 2.0 m/s 的觸地速度")
    ap.add_argument("--duty", type=float, default=0.80)
    ap.add_argument("--d-step", type=float, default=0.0, dest="d_step")
    ap.add_argument("--x-off", type=float, default=0.04, dest="x_off",
                    help="★ 原地踏步在 kp=250 的配平點約 +0.04（基準的 −0.04 是 "
                         "kp=120 + 前進步態掃出來的，符號相反）")
    ap.add_argument("--g-c", type=float, default=0.12, dest="g_c",
                    help="★ 不要調小！g_c=0.04 時實際離地只有 4.5 mm，"
                         "腳被地面拖著走 —— 那是文件裡的災難情境")
    ap.add_argument("--d-step-y", type=float, default=0.12, dest="d_step_y")
    ap.add_argument("--mu-x", type=float, default=1.80, dest="mu_x")
    ap.add_argument("--mu-y", type=float, default=1.50, dest="mu_y")
    ap.add_argument("--z-sag", type=float, default=None, dest="z_sag",
                    help="★ 預設 **0.036×250/kp**（實機錨點）。不是模擬的 STATIC_SAG")
    ap.add_argument("--ramp", type=float, default=3.0, help="站姿↔步態的淡入淡出秒數")

    # ---- 增益
    ap.add_argument("--kp", type=float, default=250.0)
    ap.add_argument("--kd", type=float, default=5.0)
    ap.add_argument("--wheel-kd", type=float, default=0.5, dest="wheel_kd",
                    help="★ 輪子純阻尼，kp 恆為 0。設定檔的 FSM_RL_Wheel_Kp=60 "
                         "是配「每步重給目標角」的 RL，開迴路套上去偏航失控 +39°/12s")

    # ---- 時序
    ap.add_argument("--ramp-kp", type=float, default=2.0, dest="ramp_kp")
    ap.add_argument("--t1", type=float, default=1.5)
    ap.add_argument("--t2", type=float, default=1.5)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--hold-mid", type=float, default=1.5, dest="hold_mid")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--max-secs", type=float, default=60.0, dest="max_secs",
                    help="★ 總承重時間上限。mc_ctrl 凍結已驗到 200 秒，"
                         "所以這管的是承重 —— 腿吃 41 kg 的發熱沒量過")

    # ---- 保護（⚠️ 和 M7/M8 不同）
    ap.add_argument("--emax", type=float, default=0.60,
                    help="追蹤誤差 rad。步態擺動相本來就會落後（實機 52 mm ≈ 0.3 rad）")
    ap.add_argument("--vmax", type=float, default=16.0,
                    help="★ 量測關節速度 rad/s。**基準步態（ω=1.4）的命令峰值實測 "
                         "13.1 rad/s** —— M7/M8 的 4.0 會在第一個擺動相就誤中止。"
                         "馬達規格 190 RPM = 19.9 rad/s，16 是它的 80%")
    ap.add_argument("--vcmd-max", type=float, default=14.0, dest="vcmd_max",
                    help="★ 命令速度 rad/s（乾跑檢查）。基準步態要 13.1")
    ap.add_argument("--tilt-max", type=float, default=20.0, dest="tilt_max")
    ap.add_argument("--temp-max", type=float, default=70.0, dest="temp_max")
    ap.add_argument("--tau-hits", type=int, default=3, dest="tau_hits")
    ap.add_argument("--gap-max", type=float, default=0.25, dest="gap_max",
                    help="單次迴圈間隔上限。controller 逾時 500 ms → 超過就清零")
    a = ap.parse_args()

    if a.march:
        a.d_step = 0.0
    if a.z_sag is None:
        # ★★ 實機錨點（M8 S3）：kp250 → 36 mm、kp120 → 72 mm，正比於 1/kp。
        #   **不要用 max_model.STATIC_SAG（0.0325）** —— 那是模擬值，
        #   而模擬在 z 方向高估順從性 1.8–2.1 倍，用它會補太少。
        a.z_sag = 0.036 * 250.0 / a.kp

    logp = shm_io.start_log("M9")
    mode = f"播放 {os.path.basename(a.traj)}" if a.traj else "狗上即時 CPG"
    print(f"M9 —— 步態（★★★ 承重、連續、動態觸地）　模式：{mode}\n")
    print("⚠️⚠️ 確認：狗趴在地上、16 顆洩力、**吊帶掛在 292 mm 以下且鬆弛**、")
    print("      **地面淨空且留出走動空間**、第二個終端機備著 estop。\n")

    # ---------------------------------------------------------------- 前置
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節目前帶著非零增益，先處理再跑：{live}")
        return 1
    print("✅ 16 顆全部洩力中")

    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")
    roll0, pitch0 = read_imu_rp()
    print(f"✅ 機身姿態 roll {roll0:+.1f}° pitch {pitch0:+.1f}°")

    st0 = shm_io.read_joint_state()
    by = {r["name"]: r for r in st0}
    q_lie = {j: coord.to_ctrl(j, by[j]["position"]) for j in LEGS12}

    front = [q_lie[lg + coord.KIND_KNEE] for lg in coord.FRONT_LEGS]
    rear = [q_lie[lg + coord.KIND_KNEE] for lg in coord.REAR_LEGS]
    if all(math.copysign(1, f) == math.copysign(1, r) for f in front for r in rear):
        print("\n❌ 前後膝同號 = knee_back 模式，先用 M5 喬回來")
        return 1
    print("✅ 前後膝反號（後腿往前彎，原廠預設）")

    crouch = coord.POSES["crouch"]
    too_high = [j for j in LEGS12 if j.endswith(coord.KIND_KNEE)
                and abs(q_lie[j]) < abs(crouch[j]) - 0.02]
    if too_high:
        print(f"\n❌ 起點比 crouch 還高（{too_high}）—— 吊帶把狗撐起來了。")
        print("   吊帶高度 H 必須低於 crouch 的 292 mm，否則量到的力矩是假的。")
        return 1

    # ---------------------------------------------------------------- 步態來源
    if a.traj:
        D = json.loads(open(a.traj, encoding="utf-8").read())
        if D.get("schema") != "gait_traj/1":
            print(f"❌ 不是軌跡檔（schema={D.get('schema')!r}）")
            return 1
        gait_dt = D["dt"]
        gait_names = D["joints"]
        gait_q = D["q"]
        n_gait = D["n"]
        if abs(D["kp"] - a.kp) > 1e-9 or abs(D["kd"] - a.kd) > 1e-9:
            print(f"❌ 軌跡檔的增益（kp {D['kp']} kd {D['kd']}）與 --kp/--kd "
                  f"（{a.kp}/{a.kd}）不一致。")
            print("   ⚠️ z_sag 與 kp 綁定，混用等於補償值錯的 —— 拒跑。")
            return 1
        p = D["params"]
        print(f"\n軌跡檔：{'原地踏步' if p['march'] else '前進'}　"
              f"duty {p['duty']} ω {p['omega']} d_step {p['d_step']} "
              f"x_off {p['x_off']} g_c {p['g_c']} z_sag {p['z_sag']:.4f}")
    else:
        gait_dt = 0.02          # ★ 50 Hz —— 與模擬驗證時的 CTRL_DT 相同
        gait_names = LEGS12
        # ★★ 基準姿勢是 **home**（hip 0.8 / 膝 −1.5，機身 491 mm），
        #    **不是 stand**（0.6 / −1.2，542 mm）。CPG 的 f0/knee_sign 都是
        #    照 `max_model.HOME` 建的 —— 用錯會差 1.0 rad，
        #    2026-08-27 就是 `test_live_cpg_matches_the_generated_trajectory_file`
        #    抓出來的（我第一版寫成 stand）。
        f0 = cpg.home_foot(coord.POSES["home"])
        ks = cpg.knee_signs(coord.POSES["home"])
        stp = cpg.make_step(cpg.PHASE_WALK)
        cst = cpg.init(cpg.PHASE_WALK)
        mux = {l: a.mu_x for l in cpg.LEGS}
        muy = {l: a.mu_y for l in cpg.LEGS}
        om = {l: a.omega for l in cpg.LEGS}
        q_stand_g = cpg.stand_targets(f0, ks, a.x_off)
        n_ramp = int(round(a.ramp / gait_dt))
        n_body = int(round(a.secs / gait_dt))
        n_gait = n_ramp + n_body + n_ramp
        gait_q = []
        n_clamp = 0
        for i in range(n_gait):
            qg, ncl = cpg.joint_targets(cst, f0, ks, a.x_off, a.g_c, a.d_step,
                                        a.d_step_y, a.duty, a.z_sag)
            n_clamp += ncl
            if i < n_ramp:
                s = smoothstep(i / max(n_ramp, 1))
            elif i < n_ramp + n_body:
                s = 1.0
            else:
                s = smoothstep(1.0 - (i - n_ramp - n_body) / max(n_ramp, 1))
            gait_q.append([(1 - s) * q_stand_g[j] + s * qg[j] for j in LEGS12])
            cst = stp(cst, mux, muy, om, gait_dt)
        if n_clamp:
            print(f"\n❌ IK 縮限 {n_clamp} 次 —— 靜默的縮限會讓步態突然變鈍。")
            return 1
        print(f"\n即時 CPG：{'原地踏步' if a.march else '前進'}　duty {a.duty} "
              f"ω {a.omega} d_step {a.d_step} x_off {a.x_off} g_c {a.g_c} "
              f"z_sag {a.z_sag:.4f}")
        print("✅ 全程無 IK 縮限")

    idxmap = [gait_names.index(j) for j in LEGS12]
    G = [[row[k] for k in idxmap] for row in gait_q]     # 一律轉成 LEGS12 序
    q_gait0 = {j: G[0][i] for i, j in enumerate(LEGS12)}
    q_gaitN = {j: G[-1][i] for i, j in enumerate(LEGS12)}

    # ---------------------------------------------------------------- 時序
    pre = build_standup(a, q_lie, q_gait0)
    post = build_sitdown(a, q_gaitN, q_lie)
    T_pre = sum(s[1] for s in pre)
    T_gait = n_gait * gait_dt
    T_post = sum(s[1] for s in post)
    T_END = T_pre + T_gait + T_post
    print(f"\n總時長 {T_END:.1f} 秒 = 站起來 {T_pre:.1f} + 步態 {T_gait:.1f} "
          f"+ 坐回去 {T_post:.1f}")
    if T_END > a.max_secs:
        print(f"❌ 超過 --max-secs {a.max_secs:.0f}（承重時間，非 mc_ctrl 限制）")
        return 1
    print(f"✅ 在承重上限 {a.max_secs:.0f} 秒之內")

    # ---- 命令速度檢查（★ 步態本來就快，門檻和 M7/M8 不同）
    vmax_seen, vj = 0.0, ""
    for i in range(1, n_gait):
        for k, j in enumerate(LEGS12):
            v = abs(G[i][k] - G[i - 1][k]) / gait_dt
            if v > vmax_seen:
                vmax_seen, vj = v, j
    print(f"\n步態最大命令速度 {vmax_seen:.2f} rad/s（{vj}）　上限 {a.vcmd_max}")
    if vmax_seen > a.vcmd_max:
        print("❌ 超過 --vcmd-max。降 --omega 或加大 --duty。")
        return 1
    print(f"✅ 在上限內（⚠️ 保護門檻 --vmax {a.vmax} 必須高於這個，否則會誤中止）")
    if a.vmax < vmax_seen * 1.2:
        print(f"❌ --vmax {a.vmax} 太接近命令速度 {vmax_seen:.2f} —— 會誤中止。")
        return 1

    bad = [f"{j}" for row in G for k, j in enumerate(LEGS12)
           if coord.check_limit(j, row[k], 0.03)]
    if bad:
        print(f"\n❌ {len(set(bad))} 個關節在步態中超出機構限位：{sorted(set(bad))}")
        return 1
    print("✅ 步態全程在機構限位內")

    print(f"\n增益：腿 kp {a.kp} kd {a.kd}　輪 **kp 0** kd {a.wheel_kd}（純阻尼，全程不鎖）")
    print(f"保護：力矩 ABAD {TMAX['1_hip_roll']:.0f} / HIP {TMAX['2_hip_pitch']:.0f}"
          f" / KNEE {TMAX['3_knee_pitch']:.0f}、硬上限 {TAU_HARD:.0f}")
    print(f"      誤差 {a.emax} rad、速度 {a.vmax} rad/s、傾角 ±{a.tilt_max:.0f}°、"
          f"迴圈間隔 {a.gap_max*1000:.0f} ms")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結、沒有寫入。")
        print(f"\n📄 {logp}")
        return 0
    if os.geteuid() != 0:
        print("❌ 需要 root：請加 sudo")
        return 1

    # ---------------------------------------------------------------- 執行
    idx = {j: shm_io.idx_of(j) for j in LEGS12}
    widx = {w: shm_io.idx_of(w) for w in shm_io.WHEELS}
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    abort = ""
    peak = {j: 0.0 for j in LEGS12}
    tau_hot = {j: 0 for j in LEGS12}
    samples: list = []
    recent: list = []
    kp_now = 0.0
    des_now = dict(q_lie)
    worst_gap = 0.0
    worst_gap_t = 0.0
    n_tick = 0
    t_prev = None

    def write_frame(des, kp):
        for j in LEGS12:
            shm.write_cmd(idx[j], position=coord.to_motor(j, des[j]),
                          velocity=0.0, effort=0.0, kp=kp, kd=a.kd)
        st_w = state_ro.states()
        for w, wi in widx.items():
            # ★ 輪子全程純阻尼（kp=0）。步態需要輪子能自由滾。
            shm.write_cmd(wi, position=st_w[wi]["position"],
                          velocity=0.0, effort=0.0, kp=0.0, kd=a.wheel_kd)

    bounds, tt = [], 0.0
    for nm, dur, p0, p1 in pre:
        bounds.append((tt, tt + dur, nm, p0, p1))
        tt += dur
    t_gait0 = tt
    tt += T_gait
    for nm, dur, p0, p1 in post:
        bounds.append((tt, tt + dur, nm, p0, p1))
        tt += dur

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.15)
        print(f"\n✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")
        print(f"{'t':>6s} {'階段':>12s} {'kp':>6s} {'最大|誤差|':>10s} {'最大|τ|':>8s}"
              f" {'關節':>16s} {'roll':>6s} {'pitch':>6s} {'輪速':>7s}")
        t0 = time.monotonic()
        nxt = t0
        last = -1.0
        while True:
            t = time.monotonic() - t0
            if t >= T_END:
                break
            if t_prev is not None:
                gap = t - t_prev
                if gap > worst_gap:
                    worst_gap, worst_gap_t = gap, t
                if gap > a.gap_max:
                    abort = (f"迴圈間隔 {gap*1000:.0f} ms 超過 {a.gap_max*1000:.0f}"
                             f"（controller 逾時 500 ms → 會清零）")
            t_prev = t
            n_tick += 1
            if abort:
                break

            if t < t_gait0 or t >= t_gait0 + T_gait:
                s0, s1, nm, p0, p1 = next(b for b in bounds if b[0] <= t < b[1])
                u = smoothstep((t - s0) / max(s1 - s0, 1e-6))
                if nm == "RAMP_UP":
                    kp_now = a.kp * ((t - s0) / max(s1 - s0, 1e-6))
                elif nm == "RAMP_DOWN":
                    kp_now = a.kp * max(0.0, 1 - (t - s0) / max(s1 - s0, 1e-6))
                else:
                    kp_now = a.kp
                des_now = {j: p0[j] + u * (p1[j] - p0[j]) for j in LEGS12}
            else:
                # ★ 步態：50 Hz 的目標**線性內插**到寫入頻率。
                #   模擬是零階保持（每 nsub 個物理步換一次），內插只會更平順；
                #   差異很小但要知道兩邊不完全相同。
                nm = "GAIT"
                kp_now = a.kp
                x = (t - t_gait0) / gait_dt
                i0 = int(x)
                if i0 >= n_gait - 1:
                    des_now = {j: G[-1][k] for k, j in enumerate(LEGS12)}
                else:
                    f = x - i0
                    des_now = {j: G[i0][k] + f * (G[i0 + 1][k] - G[i0][k])
                               for k, j in enumerate(LEGS12)}

            stt = state_ro.states()
            we = (0.0, "")
            wt = (0.0, "")
            tick = {}
            for j in LEGS12:
                sg = coord.SIGN[j[2:]][j[:2]]
                r = stt[idx[j]]
                q = coord.to_ctrl(j, r["position"])
                v = sg * r["velocity"]
                tau = sg * r["effort"]
                err = q - des_now[j]
                tick[j] = (round(q, 4), round(des_now[j], 4), round(tau, 2), round(v, 3))
                cap = kp_now * abs(err) + a.kd * abs(v)
                if abs(tau) <= 1.5 * cap + 1.0 and abs(tau) > abs(peak[j]):
                    peak[j] = tau
                if abs(err) > we[0]:
                    we = (abs(err), j)
                if abs(tau) > wt[0]:
                    wt = (abs(tau), j)
                lim = TMAX[j[2:]]
                if abs(tau) > TAU_HARD:
                    tau_hot[j] += 1
                    if tau_hot[j] >= 2:
                        abort = f"{j} 力矩連續 2 筆超過硬上限 {TAU_HARD}（{tau:+.1f}）"
                elif abs(tau) > lim:
                    tau_hot[j] += 1
                    if tau_hot[j] >= a.tau_hits:
                        abort = f"{j} 力矩連續 {tau_hot[j]} 筆超過 {lim}（{tau:+.1f}）"
                else:
                    tau_hot[j] = 0
                if abs(err) > a.emax:
                    abort = f"{j} 追蹤誤差 {err:+.3f} 超過 {a.emax}"
                if abs(v) > a.vmax:
                    abort = f"{j} 速度 {v:+.2f} 超過 {a.vmax}"
                if r["temp_C"] > a.temp_max:
                    abort = f"{j} 溫度 {r['temp_C']:.1f}°C 超過 {a.temp_max}"
                if abort:
                    break

            roll, pitch = read_imu_rp()
            if not abort and max(abs(roll), abs(pitch)) > a.tilt_max:
                abort = f"機身傾角 roll {roll:+.1f}° pitch {pitch:+.1f}° 超過 ±{a.tilt_max}°"

            # ★★ 輪子的 position/velocity 一定要記 ——
            #    2026-08-27 發現 M8 只記了 effort，導致實機資料無法做
            #    「觸地滾動 vs 懸空空轉」的拆解，而那是判斷
            #    「前後腿有沒有在互相對抗」的唯一方法。
            wrec = {w: (round(stt[wi]["position"], 4),
                        round(stt[wi]["velocity"], 3),
                        round(stt[wi]["effort"], 2)) for w, wi in widx.items()}
            rec = {"t": round(t, 3), "phase": nm, "kp": round(kp_now, 1),
                   "roll": round(roll, 2), "pitch": round(pitch, 2),
                   "j": tick, "w": wrec}
            recent.append(rec)
            if len(recent) > 60:
                recent.pop(0)
            samples.append(rec)
            if abort:
                break

            write_frame(des_now, kp_now)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

            if t - last >= 0.25:
                wv = max(abs(x[1]) for x in wrec.values())
                print(f"{t:6.2f} {nm:>12s} {kp_now:6.0f} {we[0]:10.4f} {wt[0]:8.2f}"
                      f" {wt[1]:>16s} {roll:+6.1f} {pitch:+6.1f} {wv:7.2f}")
                last = t
            nxt += 1.0 / a.hz
            dly = nxt - time.monotonic()
            if dly > 0:
                time.sleep(dly)
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    except Exception as e:
        abort = f"未預期的例外：{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- 收尾
    held_des, held_kp = dict(des_now), (kp_now if abort else 0.0)
    keeper = Keepalive(shm, state_ro, (lambda: write_frame(held_des, held_kp)), a.hz,
                       "凍結目標角、維持增益" if abort else "零增益保持")
    keeper.start()

    print("\n" + "=" * 76)
    if abort:
        print(f"⛔ 中止：{abort}")
        if held_kp >= 0.3 * a.kp:
            print(f"\n★★ **已凍結目標角並維持 kp={held_kp:.0f} —— 狗還撐著，沒有放手。**")
            print("   ⚠️ 步態中止會凍在某個相位，可能有腿懸空。先看狗再決定。")
        hurt = abort.split()[0]
        if hurt in LEGS12 and recent:
            print(f"\n中止前最後 12 筆 —— {hurt}")
            print(f"  {'t':>7s} {'階段':>12s} {'q':>9s} {'des':>9s} {'τ':>8s}"
                  f" {'v':>7s} {'kp|e|+kd|v|':>11s}")
            for rr in recent[-12:]:
                q_, d_, tau_, v_ = rr["j"][hurt]
                cap = rr["kp"] * abs(q_ - d_) + a.kd * abs(v_)
                print(f"  {rr['t']:7.3f} {rr['phase']:>12s} {q_:9.4f} {d_:9.4f}"
                      f" {tau_:8.2f} {v_:7.3f} {cap:11.1f}")
    else:
        print("✅ 序列完整跑完")

    print(f"\n{'關節':16s} {'峰值τ':>9s} {'門檻':>7s} {'用掉':>7s}")
    for j in LEGS12:
        lim = TMAX[j[2:]]
        print(f"{j:16s} {peak[j]:+9.2f} {lim:7.0f} {100*abs(peak[j])/lim:6.0f}%")

    el = min(t, T_END) if n_tick else 0.0
    hz = n_tick / el if el > 0 else 0.0
    print(f"\n迴圈：{n_tick} 次 / {el:.2f}s = {hz:.0f} Hz（目標 {a.hz:.0f}）"
          f"　最長單次間隔 {worst_gap*1000:.0f} ms @ t={worst_gap_t:.2f}s")
    if worst_gap > 0.5:
        print("  ❌ **有一次間隔超過 500 ms —— 那一刻指令區很可能被清成 0。**")
    elif worst_gap > a.gap_max:
        print("  ⚠️ 超過警戒線。先 uptime／top 看狗上有沒有別的東西在跑。")

    # ★ 輪子的淨滾動（分觸地/懸空是事後分析的事，這裡先給總量）
    if samples:
        w0 = samples[0]["w"]
        w1 = samples[-1]["w"]
        print(f"\n{'輪':10s} {'淨轉動 rad':>12s} {'≈ 滾動 mm':>11s}")
        for w in shm_io.WHEELS:
            d_ = shm_io.wrap_pi(w1[w][0] - w0[w][0])
            print(f"{w:10s} {d_:12.4f} {d_*kin.WHEEL_RADIUS*1000:11.0f}")
        print("  ⚠️ 這是總量，**包含懸空空轉** —— 模擬顯示空轉可佔八成。")
        print("     要判讀必須拆成觸地/懸空，事後用 `w` 欄的 position 做。")

    print("\n" + "=" * 76)
    print("★ 現在腿還在承重。[Enter] 依原路徑坐回趴姿；[Ctrl-C] 立刻放手（會塌）")
    try:
        if sys.stdin.isatty():
            input("\n   > ")
        else:
            print("   非互動模式 → 直接執行坐回去")
        keeper.stop()
        cur = dict(held_des)
        for nm, dur, tgt in (("SIT_crouch", a.t2, dict(coord.POSES["crouch"])),
                             ("SIT_LIE", a.t1, q_lie)):
            s = time.monotonic()
            while (e := time.monotonic() - s) < dur:
                u = smoothstep(e / dur)
                write_frame({j: cur[j] + u * (tgt[j] - cur[j]) for j in LEGS12},
                            max(held_kp, a.kp))
                shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
                time.sleep(1.0 / a.hz)
            cur = dict(tgt)
        s = time.monotonic()
        while (e := time.monotonic() - s) < a.ramp_kp:
            write_frame(cur, a.kp * max(0.0, 1 - e / a.ramp_kp))
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
            time.sleep(1.0 / a.hz)
        for i in range(len(shm_io.JOINTS)):
            shm.zero_gains(i)
        shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
        print("✅ 已坐回趴姿並降到零增益")
        keeper = Keepalive(shm, state_ro,
                           (lambda: [shm.zero_gains(i)
                                     for i in range(len(shm_io.JOINTS))]),
                           a.hz, "零增益保持")
        keeper.start()
    except KeyboardInterrupt:
        print("\n   （Ctrl-C：放手）")
    keeper.stop()

    out = {"schema": "m9_gait/1", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "args": vars(a), "aborted": bool(abort), "abort_reason": abort or None,
           "q_lie": q_lie, "gait_dt": gait_dt, "n_gait": n_gait,
           "t_gait0": t_gait0, "peak": peak,
           "loop": {"ticks": n_tick, "hz": round(hz, 1),
                    "worst_gap_s": round(worst_gap, 4)},
           "samples": samples[:60000]}
    jp = (logp[:-4] if logp.endswith(".log") else logp) + ".json"
    try:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        if os.geteuid() == 0 and os.getenv("SUDO_USER"):
            import pwd
            pw = pwd.getpwnam(os.environ["SUDO_USER"])
            try:
                os.chown(jp, pw.pw_uid, pw.pw_gid)
            except OSError:
                pass
        print(f"\n📊 {jp}")
    except Exception as e:
        print(f"\n⚠️ 結果檔寫入失敗：{e}")

    try:
        shm.close()
        state_ro.close()
    except Exception:
        pass
    if frozen:
        print(f"\n⏸ mc_ctrl 仍在凍結中（PID {pid}）。確認狗安全後：")
        print(f"      sudo kill -CONT {pid}")
    print(f"\n📄 {logp}")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
