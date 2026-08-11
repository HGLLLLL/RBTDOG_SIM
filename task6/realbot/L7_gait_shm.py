#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L7_gait_shm.py —— 把 walk_stable 步態以 500 Hz 串流寫進 /dev/shm/spline_shm。

⚠️ 這是【吊掛空跑】用的。狗必須吊起來、四腳離地。不是落地行走工具。
   右後腿(leg2)整條已從 CAN 失聯，一律用 --skip-legs 2 排除。

三種模式：
  jog    單關節 ±0.10 rad 三角波。用來確認 calib_map 的正負號。
         calib_map 自己標注 hip 的號是「暫定」——號反了腿會往反方向甩到限位，
         而離線檢驗完全看不出來，因為數字本身很正常。所以這關必須先過。
  leg    只驅動一條腿跑完整步態，其餘零增益。把風險限制在單腿。
  gait   三條腿同步跑。

依賴限制：本檔在【車載電腦】上執行，只能用 python3 標準庫 + numpy。
不得 import mujoco —— 狗上沒有。IK 常數由 npz 帶上來。

用法：
  python3 L7_gait_shm.py --mode gait --traj gait_walk_stable.npz --time-scale 0.25
  sudo python3 L7_gait_shm.py --mode gait --traj gait_walk_stable.npz \
       --time-scale 0.25 --skip-legs 2 --confirm
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib_map
import shm_common as SC

# --- live 模式重列的常數 ---
# 必須與 d1_model / cpg_walk_d1 一致。狗上不能 import d1_model（它 import mujoco），
# 所以在此重列。提到模組層級是為了讓 test_live_constants_match_d1_model 讀得到值
# 逐項比對 —— 比對數值，不是比對原始碼字串。
MU_MIN, MU_MAX = 1.0, 2.0
A_CONV, W_COUP, N_CPG_SUB = 50.0, 8.0, 4
D_STEP, D_STEP_Y, G_P = 0.12, 0.09, 0.01
HOME3 = np.array([0.0, 1.05, -2.00])
PHASE_WALK = np.array([0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi])

LEG_KP, LEG_KD = 20.0, 0.7     # 原廠站立實測值
STOP_KD = 3.0
CATCH_SEC = 0.5
RAMP_MIN_SEC = 2.0

# 保護門檻的安全係數。門檻 = 模擬峰值 × 係數。
# 模擬是「一切正常時會發生什麼」，超過它兩倍就不是正常了。
VEL_SAFETY = 1.5
TORQUE_SAFETY = 2.0


def expected_calib_hash():
    """本機 calib_map 的雜湊。與 gait_export.calib_hash() 必須用同一套算法。

    不 import gait_export——那支要 mujoco，狗上沒有。所以這裡重算一份。
    ⚠️ 兩邊的算法必須逐字相同，有測試 test_calib_hash_agrees_across_modules 釘住。
    """
    import hashlib
    payload = json.dumps(
        {"legs": calib_map.LEG_MJCF2SHM,
         "calib": {str(k): {jn: list(v[jn]) for jn in ("abad", "hip", "knee")}
                   for k, v in sorted(calib_map.CALIB.items())}},
        sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_trajectory(npz_path):
    """讀 npz。回傳 (q_shm (N,4,3), meta)。校正雜湊不符就拒跑。"""
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    here = expected_calib_hash()
    if meta["calib_hash"] != here:
        print(f"✗ 校正雜湊不符：軌跡檔是 {meta['calib_hash']}、本機是 {here}")
        print("  代表 calib_map 改過但軌跡沒重產。每個關節都會下錯指令，拒絕執行。")
        print("  重新產生：python gait_export.py --export <path>")
        sys.exit(1)
    return z["q_shm"], meta


def sample_at(q_shm, ctrl_dt, u):
    """在軌跡時間 u（秒）線性內插。u 可為標量或陣列。超過尾端就夾住。

    500 Hz 上取樣與 --time-scale 是同一個操作：都只是「在什麼時間點取樣」。
    """
    u = np.asarray(u, dtype=float)
    n = len(q_shm)
    x = np.clip(u / ctrl_dt, 0.0, n - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (x - i0)[..., None, None]
    out = q_shm[i0] * (1.0 - w) + q_shm[i1] * w
    return out


def playback_times(n_frames, ctrl_dt, time_scale, hz=SC.CTRL_HZ):
    """播放時每個控制週期對應的軌跡時間(秒)。

    time_scale 是播放倍率：0.25 = 四分之一速，走完同一條軌跡要花四倍時間。
    """
    total = (n_frames - 1) * ctrl_dt / time_scale      # 實際牆鐘秒數
    n_steps = int(round(total * hz)) + 1
    return np.arange(n_steps) / hz * time_scale


def live_trajectory(npz_path, secs):
    """狗上純 numpy 自己算軌跡，不讀 npz 的 q_shm。**離線檔是黃金標準**，
    不逐幀吻合就不准用。

    ★ CPG 一律在【未縮放的 50 Hz 網格】上積分，與 --time-scale 無關。
      時間縮放純粹是播放層的事（sample_at + playback_times），兩個模式共用。
      這樣 live 與 file 在任何倍速下都產生完全相同的路點序列。

      不要改成「用縮放後的 dt 即時積分」：那會在慢速下產生更密的路點，
      在擺動→站立交界處（duty_remap 讓 dθ/dt 跳 4 倍的那個折點）與 file
      差到 7.5°。兩者都不算錯，但就不再是同一條指令串流，黃金標準也就沒了。
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return _cpg_rollout(meta, z["jinvs"], secs)   # f0s 只給診斷用，這裡不需要


def _cpg_rollout(meta, jinvs, secs):
    """CPG 積分 + IK + calib_map，純 numpy。逐行對應 cpg_walk_d1 的實作。

    ⚠️ 這是 cpg_walk_d1 的第二份實作。這是【使用者決策的例外】而非疏忽 ——
       需求是「一個模式播寫死的軌跡、一個模式在狗上自己算」，而狗上沒有
       mujoco（leg_ik_consts 要跑 forward kinematics），無法 import 那份。
       折衷是把 jinvs 預先算好由 npz 帶上來，讓這裡只剩純算術，並用
       test_live_trajectory_matches_the_file_frame_by_frame 逐幀釘住
       （容忍 1e-9）。那個測試轉紅就代表兩份漂移了。

    ★ 一律用 meta["ctrl_dt"]（50 Hz）積分，不吃 time_scale —— 見 live_trajectory。
    """
    dt = meta["ctrl_dt"]
    mux, muy = np.full(4, meta["mu_x"]), np.full(4, meta["mu_y"])
    omega = np.full(4, meta["omega"])
    duty, x_off, g_c = meta["duty"], meta["x_off"], meta["g_c"]

    PHI = PHASE_WALK[None, :] - PHASE_WALK[:, None]
    rx, rxd = np.full(4, 1.5), np.zeros(4)
    ry, ryd = np.full(4, 1.5), np.zeros(4)
    th = PHASE_WALK.copy()

    n = int(secs / dt)
    out = np.zeros((n, 4, 3))
    for i in range(n):
        h = dt / N_CPG_SUB
        for _ in range(N_CPG_SUB):
            rxd += (A_CONV * (A_CONV / 4 * (mux - rx) - rxd)) * h
            rx += rxd * h
            ryd += (A_CONV * (A_CONV / 4 * (muy - ry) - ryd)) * h
            ry += ryd * h
            rbar = 0.5 * (rx + ry)
            diff = th[None, :] - th[:, None] - PHI
            th = th + (2 * np.pi * omega + W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
        th %= 2 * np.pi

        ph = (th % (2 * np.pi)) / (2 * np.pi)
        sw = 1.0 - duty
        thr = np.where(ph < sw, np.pi * ph / sw, np.pi + np.pi * (ph - sw) / duty)

        fx = 2 * (rx - MU_MIN) / (MU_MAX - MU_MIN) - 1
        fy = 2 * (ry - MU_MIN) / (MU_MAX - MU_MIN) - 1
        dx = -D_STEP * fx * np.cos(thr) + x_off
        dy = D_STEP_Y * fy * np.cos(thr)
        dz = np.where(np.sin(thr) > 0, g_c * np.sin(thr), G_P * np.sin(thr))
        off = np.stack([dx, dy, dz], -1)

        for mjcf_leg in range(4):
            q3 = HOME3 + jinvs[mjcf_leg] @ off[mjcf_leg]
            shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
            for j, jn in enumerate(("abad", "hip", "knee")):
                s, o = calib_map.CALIB[shm_leg][jn]
                out[i, shm_leg, j] = s * q3[j] + o
    return out


def guard_thresholds(meta, kp, kd, time_scale):
    """回傳 (torque_abort, vel_abort)，查 npz 內建的空中模擬表。

    ⚠️ 沒有 fallback。查不到就拒跑——門檻不能用猜的。
       L4 的 8.0 N·m 更不能照搬：原廠增益 1.0× 的模擬力矩峰值就有 10.18 N·m。
    """
    table = meta.get("air_sim", {})
    key, skey = f"{kp}/{kd}", str(time_scale)
    if key not in table or skey not in table[key]:
        print(f"✗ 軌跡檔沒有 kp={kp}/kd={kd} @ {time_scale}× 的模擬結果，拒絕執行。")
        print(f"  檔內有的組合：{ {k: sorted(v) for k, v in table.items()} }")
        print("  要用別的組合，先把它加進 gait_export.AIR_SIM_GRID/AIR_SIM_SCALES 再重產。")
        sys.exit(1)
    e = table[key][skey]
    return float(e["tau_peak"] * TORQUE_SAFETY), float(e["vel_peak"] * VEL_SAFETY)


def jog_targets(start_q, joint_idx, amp, secs, hz=SC.CTRL_HZ):
    """單關節三角波，兩個來回，起點與終點都回到 start_q。

    用來確認 calib_map 的正負號：人眼看腿往哪邊動，對照 MJCF 的正向定義
    （+knee→伸直、+abad→外張、+hip→後擺）。號反了就停在這關修映射。

    amp 刻意小（0.10 rad = 5.7°）：驗號只需要看得出方向，不需要大動作。

    ⚠️ 相位網格用 np.linspace(0.0, 2.0, n)（含右端點），不是 np.arange(n)/n*2.0。
       後者在 n 個樣本時最後一點停在 2 - 2/n，量出來的收尾殘差是 amp * 4/n
       （n=2000, amp=0.10 時約 2e-4 rad）——遠超過 test_...starting_and_ending_at_rest
       要求的 abs=1e-6，也就是「起點終點都回到 start_q」這個安全承諾在最後一刻
       沒兌現。含右端點才能讓相位在最後一個樣本精確落在 2.0（tri=0）。
    """
    n = int(secs * hz)
    ph = np.linspace(0.0, 2.0, n)                     # 0..2（含端點），兩個來回
    tri = np.where(ph < 0.5, ph * 2,
                   np.where(ph < 1.5, 2 - ph * 2, ph * 2 - 4))
    q = np.tile(np.asarray(start_q, dtype=float), (n, 1))
    q[:, joint_idx] += amp * tri
    return q


def _stream(d, targets_iter, active_legs, kp, kd, torque_abort, vel_abort,
            log, dry, label):
    """核心串流迴圈：每個 500 Hz 週期寫一組目標、檢查保護、記錄 state。

    回傳 (ok, 原因)。ok=False 時呼叫端負責卸力。

    ⚠️ log 的時間戳用 len(log["t"]) 而非本次呼叫的迴圈索引 —— 一次執行會分成
       接住／到起始姿／播放／回站姿四段，每段各呼叫一次本函式。用迴圈索引的話
       時間戳會每段從 0 重來，事後分析對齊指令時會整個錯位。
    """
    for k, tgt in enumerate(targets_iter):
        t_start = time.monotonic()
        if not dry:
            SC.zero_all(d)
            for i in active_legs:
                SC.set_leg_position(d, i, tgt[i][0], tgt[i][1], tgt[i][2], kp, kd)
            SC.publish(d)
            ok, why = SC.check_guards(d, active_legs, torque_abort, vel_abort)
            if not ok:
                return False, why
            log["t"].append(len(log["t"]) * SC.DT)
            log["cmd"].append(np.asarray(tgt, dtype=float).copy())
            log["p"].append(np.array([[getattr(d.state.legs[i], jn).p
                                       for jn in ("abad", "hip", "knee")]
                                      for i in range(4)]))
            log["v"].append(np.array([[getattr(d.state.legs[i], jn).v
                                       for jn in ("abad", "hip", "knee")]
                                      for i in range(4)]))
            log["tau"].append(np.array([[getattr(d.state.legs[i], jn).t
                                         for jn in ("abad", "hip", "knee")]
                                        for i in range(4)]))
            # 超時要記錄：Python 在 RK3588 上被 GC 打斷是正常的，但不記的話
            # 超時造成的軌跡失真會被誤讀成追蹤誤差。
            spent = time.monotonic() - t_start
            log["overrun"].append(spent > SC.DT)
            if spent < SC.DT:
                time.sleep(SC.DT - spent)
        elif k % (SC.CTRL_HZ // 2) == 0:
            print(f"  [{label}] t={k * SC.DT:5.2f}s  "
                  f"leg{active_legs[0]} 目標 " +
                  " ".join(f"{v:+.3f}" for v in tgt[active_legs[0]]))
    return True, ""


def _ramp_frames(a, b, secs, hz=SC.CTRL_HZ):
    """從姿勢 a 線性內插到姿勢 b，兩者都是 (4,3)。回傳 (N,4,3)。"""
    n = int(secs * hz)
    w = np.linspace(0.0, 1.0, n)[:, None, None]
    return np.asarray(a)[None] * (1 - w) + np.asarray(b)[None] * w


def run_gait(d, traj, meta, active_legs, time_scale, kp, kd, dry, log_path):
    """catch → ramp 到第 0 幀 → 播放 → ramp 回站姿 → 卸力。"""
    torque_abort, vel_abort = guard_thresholds(meta, kp, kd, time_scale)
    pred = meta["air_sim"][f"{kp}/{kd}"][str(time_scale)]
    print(f"\n[*] 保護門檻：力矩 {torque_abort:.2f} N·m、速度 {vel_abort:.2f} rad/s")
    print(f"[*] 模擬預測：誤差峰值 {pred['err_peak_deg']:.2f}°、"
          f"RMS {pred['err_rms_deg']:.2f}° —— 實機量到的值拿來跟這個比")

    stand = np.array([[SC.POSE_STAND[i][jn] for jn in ("abad", "hip", "knee")]
                      for i in range(4)])
    if dry:
        init = stand.copy()
        print("[dry-run] 假設起點為站姿（真機會讀 state.legs[*]）")
    else:
        ok, trans = SC.preflight_mc_stopped(d)
        if not ok:
            print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。先 SIGSTOP mc_ctrl。")
            return False
        SC.report_legs(d, active_legs)
        ok, problems = SC.preflight_motors_healthy(d, active_legs)
        if not ok:
            print(f"\n✗ 中止：被驅動的腿有 {len(problems)} 個馬達問題，拒絕寫入 ——")
            for p in problems:
                print(f"    • {p}")
            return False
        init = np.array([SC.read_leg_q(d, i) for i in range(4)])

    log = {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun")}
    frame0 = traj[0]
    ramp_sec = max(RAMP_MIN_SEC, meta["start_offset_from_stand"] / 0.25)
    u = playback_times(len(traj), meta["ctrl_dt"], time_scale)
    print(f"[*] 播放 {len(traj)} 幀 @ {time_scale}×  "
          f"→ {u[-1] / time_scale:.1f}s 牆鐘、{len(u)} 個控制週期")

    def stage(label, frames, kp_seq=None):
        """跑一段。kp_seq 給定時逐幀套用不同增益（接住段用）。回傳 True/False。"""
        print(f"\n[*] {label}")
        if kp_seq is None:
            ok, why = _stream(d, frames, active_legs, kp, kd,
                              torque_abort, vel_abort, log, dry, label)
        else:
            ok, why = True, ""
            for r in kp_seq:
                ok, why = _stream(d, frames[:1], active_legs, kp * r, kd * r,
                                  torque_abort, vel_abort, log, dry, label)
                if not ok:
                    break
        if not ok:
            print(f"⚠️ 保護觸發：{why} → 卸力中止")
            SC.passive_stop(d, active_legs, 300, STOP_KD)
        return ok

    # 接住：p_des 固定在當前實際角度，kp/kd 由 0 平滑升到設定值。
    # 凍結 mc_ctrl 後腿會因重力垂下，先用漸入增益接住，避免力矩突跳。
    n_catch = int(CATCH_SEC * SC.CTRL_HZ)
    if not stage("接住", init[None], np.linspace(0.0, 1.0, n_catch + 1)):
        return False
    if not stage("到起始姿", _ramp_frames(init, frame0, ramp_sec)):
        return False
    if not stage("播放步態", sample_at(traj, meta["ctrl_dt"], u)):
        return False
    if not stage("回站姿", _ramp_frames(traj[-1], stand, RAMP_MIN_SEC)):
        return False

    if not dry:
        SC.passive_stop(d, active_legs, 800, STOP_KD)
        n_over = int(np.sum(log["overrun"]))
        print(f"[*] 完成。500 Hz 週期超時 {n_over} / {len(log['t'])} "
              f"（{100.0 * n_over / max(1, len(log['t'])):.2f}%）")
        SC.write_log(log_path, {k: np.asarray(v) for k, v in log.items()},
                     meta={"mode": "gait", "time_scale": time_scale,
                           "kp": kp, "kd": kd,
                           "active_legs": list(active_legs), **meta})
    return True


def run_jog(d, leg, joint_name, kp, kd, log_path):
    """單關節微動驗號。只驅動一條腿的一個關節。"""
    ji = ("abad", "hip", "knee").index(joint_name)
    ok, trans = SC.preflight_mc_stopped(d)
    if not ok:
        print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。")
        return False
    SC.report_legs(d, (leg,))
    ok, problems = SC.preflight_motors_healthy(d, (leg,))
    if not ok:
        for p in problems:
            print(f"    • {p}")
        return False

    start = np.array(SC.read_leg_q(d, leg))
    print(f"\n[*] jog：leg{leg}({SC.LEGNAME[leg]}).{joint_name} "
          f"起點 {start[ji]:+.4f} rad，±0.10 rad 兩個來回")
    print("    ⚠️ 盯著腿看。對照 MJCF 正向定義：+knee→伸直、+abad→外張、+hip→後擺。")
    print("    方向不符就停下來修 calib_map，不要往下走。")

    frames = jog_targets(start, ji, amp=0.10, secs=8.0)
    full = np.tile(start, (len(frames), 4, 1))
    full[:, leg, :] = frames
    log = {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun")}
    # jog 不查空中模擬表：±0.10 rad 的慢速微動，用 L4 的保守值就對。
    ok, why = _stream(d, full, (leg,), kp, kd,
                      torque_abort=8.0, vel_abort=1.0,
                      log=log, dry=False, label="jog")
    if not ok:
        print(f"⚠️ 保護觸發：{why} → 卸力中止")
    SC.passive_stop(d, (leg,), 800, STOP_KD)
    SC.write_log(log_path, {k: np.asarray(v) for k, v in log.items()},
                 meta={"mode": "jog", "leg": leg, "joint": joint_name,
                       "kp": kp, "kd": kd, "start": start.tolist()})
    return ok


def main():
    ap = argparse.ArgumentParser(description="D1 EDU 輪足：步態串流（吊掛空跑用）")
    ap.add_argument("--mode", choices=("jog", "leg", "gait"), required=True)
    ap.add_argument("--traj", default=None, help="gait_export 產生的 npz（leg/gait 模式必填）")
    ap.add_argument("--source", choices=("file", "live"), default="file")
    ap.add_argument("--time-scale", type=float, default=0.25, dest="time_scale",
                    help="播放倍率。0.25=四分之一速（預設，先慢再快）。file/live 皆適用")
    ap.add_argument("--secs", type=float, default=5.0, help="播放秒數（從軌跡頭開始）")
    ap.add_argument("--kp", type=float, default=LEG_KP)
    ap.add_argument("--kd", type=float, default=LEG_KD)
    ap.add_argument("--skip-legs", default="2",
                    help="不驅動的腿（0=FR 1=FL 2=RR 3=RL）。預設 2 —— RR 整條已失聯")
    ap.add_argument("--only-leg", type=int, default=None,
                    help="leg 模式：只驅動這一條（SHM 腿序）")
    ap.add_argument("--jog-leg", type=int, default=0)
    ap.add_argument("--jog-joint", choices=("abad", "hip", "knee"), default="hip")
    ap.add_argument("--log", default="l7_log.npz")
    ap.add_argument("--confirm", action="store_true", help="真的驅動硬體")
    args = ap.parse_args()

    if args.mode in ("leg", "gait") and not args.traj:
        ap.error("--mode leg/gait 需要 --traj")

    try:
        skip = {int(x) for x in args.skip_legs.split(",") if x.strip() != ""}
    except ValueError:
        print(f"✗ --skip-legs 格式錯誤：{args.skip_legs!r}")
        sys.exit(1)
    if not skip <= {0, 1, 2, 3}:
        print(f"✗ --skip-legs 只能是 0~3，收到 {sorted(skip)}")
        sys.exit(1)
    active = tuple(i for i in range(4) if i not in skip)
    if args.mode == "leg":
        if args.only_leg is None:
            ap.error("--mode leg 需要 --only-leg")
        if args.only_leg in skip:
            ap.error(f"--only-leg {args.only_leg} 同時被 --skip-legs 排除了")
        active = (args.only_leg,)
    if args.mode == "jog":
        if args.jog_leg in skip:
            ap.error(f"--jog-leg {args.jog_leg} 同時被 --skip-legs 排除了")
        active = (args.jog_leg,)
    if not active:
        print("✗ 四條腿都被跳過了，沒事可做。")
        sys.exit(1)

    SC.check_struct_size()

    traj = meta = None
    if args.traj:
        if args.source == "file":
            traj, meta = load_trajectory(args.traj)
        else:
            _, meta = load_trajectory(args.traj)
            traj = live_trajectory(args.traj, meta["secs"])
        n = int(args.secs / meta["ctrl_dt"])
        traj = traj[:n]

    print(f"\n[*] 驅動的腿：{', '.join(f'leg{i}({SC.LEGNAME[i]})' for i in active)}")
    skipped = [i for i in range(4) if i not in active]
    if skipped:
        print(f"[*] ⚠️ 跳過的腿：{', '.join(f'leg{i}({SC.LEGNAME[i]})' for i in skipped)}"
              f" —— 全程零增益，完全不出力")

    if not args.confirm:
        print("=" * 66)
        print("DRY-RUN：不開啟、不寫入共享記憶體，只印出動作計畫。")
        print("要真的驅動硬體請加 --confirm（且需 sudo）。")
        print("=" * 66)
        if args.mode == "gait":
            run_gait(None, traj, meta, active, args.time_scale,
                     args.kp, args.kd, dry=True, log_path=args.log)
        print("\n⚠️ 跑真機前必讀：狗要吊掛、四腳離地、mc_ctrl 已 SIGSTOP、estop 隨手可按。")
        return

    print("=" * 66)
    print("⚠️ 真機模式：即將驅動【腿關節】。確認：狗已吊掛四腳離地、mc_ctrl 已停。")
    print("=" * 66)
    if __import__("os").geteuid() != 0:
        print("✗ 需要 root：請用 sudo 執行。")
        sys.exit(1)
    try:
        d, _buf = SC.open_shm()
    except FileNotFoundError:
        print(f"✗ 找不到 {SC.SHM_PATH}（機器人運控沒起來？）")
        sys.exit(1)
    except PermissionError:
        print("✗ 權限不足：請用 sudo。")
        sys.exit(1)

    try:
        if args.mode == "gait" or args.mode == "leg":
            run_gait(d, traj, meta, active, args.time_scale,
                     args.kp, args.kd, dry=False, log_path=args.log)
        else:
            run_jog(d, active[0], args.jog_joint, args.kp, args.kd, args.log)
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C → 卸力收尾")
        SC.passive_stop(d, active, 800, STOP_KD)
    finally:
        SC.zero_all(d)
        SC.publish(d)
        print("[*] 已歸零收尾，watchdog 兜底。測完 SIGCONT 解凍 mc_ctrl 還原。")


if __name__ == "__main__":
    main()
