#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L7_gait_shm.py —— 把 walk_stable 步態以 500 Hz 串流寫進 /dev/shm/spline_shm。

⚠️ 這是【吊掛空跑】用的。狗必須吊起來、四腳離地。不是落地行走工具。
   右後腿(leg2)整條已從 CAN 失聯，一律用 --skip-legs 2 排除。

三種模式：
  jog    單關節在 MJCF 空間下 +0.10 rad 三角波（一個來回），逐幀轉回 SHM。
         用來確認 calib_map 的正負號——calib_map 自己標注 hip 的號是「暫
         定」，號反了腿會往反方向甩到限位，而離線檢驗完全看不出來，因為
         數字本身很正常。指令必須下在 MJCF 空間（手冊判準表格用的座標），
         不是 SHM 空間，否則校正正確時操作者看到的方向會跟判準相反，反而
         把對的校正改壞。所以這關必須先過。
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

# --- jog 驗號的動作方向 ---
# 每個關節【兩個方向】的物理描述。方向不寫死——腿吊著停在哪裡是隨機的，
# 寫死就會遇到「那一邊已經頂到限位、推不動」而要一直改程式（實機當天發生三次）。
# 改成執行時自動選行程比較充裕的那一邊，並把該看到什麼印出來。
#
# abad 逐腿不同：MJCF +abad 是四條腿都往【左】，所以右腿(FR=0,RR=2)的 + 是內收、
# 左腿(FL=1,RL=3)的 + 是外張。hip/knee 四條腿一致。
_AB_R = {+1: "內收", -1: "外張"}
_AB_L = {+1: "外張", -1: "內收"}
JOG_PHRASE = {
    "abad": {0: _AB_R, 1: _AB_L, 2: _AB_R, 3: _AB_L},
    "hip":  {+1: "後擺", -1: "前擺"},
    "knee": {+1: "伸直", -1: "內彎"},
}

# MJCF 機構範圍，用來算 jog 的行程餘裕（joint range，不是 ctrlrange）
JOINT_RANGE_MJCF = {"abad": (-0.4887, +0.4887),
                    "hip": (-1.1520, +2.9670),
                    "knee": (-2.7230, -0.6020)}


def jog_phrase(leg, joint, direction):
    """該關節往 MJCF direction 動時，操作者physically 會看到什麼。"""
    tbl = JOG_PHRASE[joint]
    if joint == "abad":
        tbl = tbl[leg]
    return tbl[direction]


def pick_jog_dir(leg, joint, mjcf0, amp, forced=None):
    """選 jog 的方向。回傳 (direction, 兩向的行程 dict)。

    預設選行程比較充裕的一邊——腿停在限位附近時才不會往裡壓。
    forced 給 +1/-1 可以指定方向（--jog-dir），但仍受行程檢查約束。
    """
    lo, hi = JOINT_RANGE_MJCF[joint]
    room = {+1: hi - mjcf0, -1: mjcf0 - lo}
    if forced is not None:
        return forced, room
    return (+1 if room[+1] >= room[-1] else -1), room

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
    """讀 npz。回傳 (q_shm (N,4,3), meta)。校正雜湊不符就拒跑。

    ⚠️ 自己做 expanduser 並在檔案不存在時給明確訊息：操作者是在 3 分鐘的
       開機窗口裡打指令，讓他去讀 numpy 的 FileNotFoundError traceback、
       自己想是不是 cwd 不對，是在浪費那個窗口。
    """
    p = Path(npz_path).expanduser()
    if not p.is_file():
        print(f"✗ 找不到軌跡檔：{npz_path}")
        if str(p) != str(npz_path):
            print(f"  （展開後：{p}）")
        if not p.is_absolute():
            print(f"  目前工作目錄是 {Path.cwd()}——相對路徑是相對於這裡。")
        print("  狗上的預設位置是 ~/gait_walk_stable.npz；")
        print("  沒有的話在開發機跑 bash task6/realbot/deploy_to_dog.sh 傳過去。")
        sys.exit(1)
    npz_path = p
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


def _aborted_path(log_path):
    """保護觸發／例外／Ctrl-C 中止時的 log 檔名，加 _ABORTED 後綴。

    3 分鐘操作窗口內，觸發前那幾個週期的 cmd/p/v/tau 往往是診斷價值最高的
    一次，不能因為沒跑完全部段落就沒有檔案。後綴標明「這份是中止的」，
    不要跟正常收尾的 log 撞檔名、事後分不清楚。
    """
    p = Path(log_path)
    return p.with_name(p.stem + "_ABORTED" + p.suffix)


def _log_arrays(log):
    """log dict（list of rows）轉成 np.savez 要的 dict of ndarray。

    stage 欄位強制轉 int8——契約是 0=接住/1=到起始姿/2=播放步態/3=回站姿，
    寫成別的 dtype 的話，讀取端（gait_export.analyze）逐段過濾會對不上。
    """
    out = {k: np.asarray(v) for k, v in log.items()}
    if "stage" in out:
        out["stage"] = out["stage"].astype(np.int8)
    return out


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


STAGE_CODE = {"接住": 0, "到起始姿": 1, "播放步態": 2, "回站姿": 3}


def run_gait(d, traj, meta, active_legs, time_scale, kp, kd, dry, log_path,
             mode="gait", source="file"):
    """catch → ramp 到第 0 幀 → 播放 → ramp 回站姿 → 卸力。

    mode：寫進 log meta 的模式名稱（"gait" 或 "leg"）。由呼叫端傳入實際
    模式 —— 不然 --mode leg 跑出來的 log 也會被記成 "gait"，事後分析
    分不清這次到底驅動了幾條腿是刻意的還是步態模式本來就這樣。

    source：寫進 log meta（"file" 或 "live"）。G3-live 的通過條件是「追蹤
    誤差與同一倍速的 --source file 一致」，log 裡沒記就事後分不出哪份是哪份。

    ⚠️ 保護觸發、例外、KeyboardInterrupt 都要留下 log（見 abort_log）——
       3 分鐘操作窗口、倍速階梯本來就預期會撞到門檻，觸發前那幾個週期的
       cmd/p/v/tau 是診斷價值最高的一次，不能因為沒跑完四段就被丟掉。
    """
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

    log = {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun", "stage")}
    frame0 = traj[0]
    ramp_sec = max(RAMP_MIN_SEC, meta["start_offset_from_stand"] / 0.25)
    u = playback_times(len(traj), meta["ctrl_dt"], time_scale)
    print(f"[*] 播放 {len(traj)} 幀 @ {time_scale}×  "
          f"→ {u[-1] / time_scale:.1f}s 牆鐘、{len(u)} 個控制週期")

    current_stage = {"label": None}

    def abort_log(reason):
        """保護觸發／例外／Ctrl-C 都要寫這份 log，檔名加 _ABORTED 後綴。

        meta["secs"] 記的是「到中止為止實際跑了幾秒」，不是軌跡全長——
        中止當下的長度才是接下來要拿去看的東西。
        """
        if dry or log_path is None:
            return
        n = len(log["t"])
        if len(log["stage"]) < n:
            # 例外/Ctrl-C 打斷在某段中途，該段還沒來得及記 stage 碼，補上。
            code = STAGE_CODE.get(current_stage["label"], -1)
            log["stage"].extend([code] * (n - len(log["stage"])))
        out_path = _aborted_path(log_path)
        SC.write_log(out_path, _log_arrays(log),
                     meta={"mode": mode, "time_scale": time_scale,
                           "kp": kp, "kd": kd, "active_legs": list(active_legs),
                           "source": source, **meta, "secs": n * SC.DT,
                           "aborted": True, "abort_reason": reason,
                           "aborted_stage": current_stage["label"]})
        print(f"[記錄] 中止 log 已寫入：{out_path}  {n} 筆")

    def stage(label, frames, kp_seq=None):
        """跑一段。kp_seq 給定時逐幀套用不同增益（接住段用）。回傳 True/False。

        dry-run 下的 kp_seq 分支不逐比例呼叫 _stream：frames[:1] 每次都只有
        一個元素，_stream 內部的迴圈索引每次都從 0 開始，dry-run 的列印條件
        （k % (CTRL_HZ//2) == 0）因此每次都成立，會洗出 251 行重複訊息。
        改成印一行摘要——dry-run 本來就只是預覽，不需要真的跑漸入迴圈。
        """
        current_stage["label"] = label
        print(f"\n[*] {label}")
        n_before = len(log["t"])
        if kp_seq is None:
            ok, why = _stream(d, frames, active_legs, kp, kd,
                              torque_abort, vel_abort, log, dry, label)
        elif dry:
            print(f"  [{label}] {CATCH_SEC}s，kp 0 → {kp:.1f}，"
                  f"p_des 保持在當前角度不動")
            ok, why = True, ""
        else:
            ok, why = True, ""
            for r in kp_seq:
                ok, why = _stream(d, frames[:1], active_legs, kp * r, kd * r,
                                  torque_abort, vel_abort, log, dry, label)
                if not ok:
                    break
        if not dry:
            log["stage"].extend([STAGE_CODE[label]] * (len(log["t"]) - n_before))
        if not ok:
            print(f"⚠️ 保護觸發：{why} → 卸力中止")
            SC.passive_stop(d, active_legs, 300, STOP_KD)
            abort_log(f"保護觸發：{why}")
        return ok

    try:
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
    except (KeyboardInterrupt, Exception) as exc:
        reason = ("使用者中止(Ctrl+C)" if isinstance(exc, KeyboardInterrupt)
                  else f"例外：{exc!r}")
        print(f"\n⚠️ {reason} → 卸力中止")
        if not dry:
            SC.passive_stop(d, active_legs, 300, STOP_KD)
        abort_log(reason)
        raise

    if not dry:
        SC.passive_stop(d, active_legs, 800, STOP_KD)
        n_over = int(np.sum(log["overrun"]))
        print(f"[*] 完成。500 Hz 週期超時 {n_over} / {len(log['t'])} "
              f"（{100.0 * n_over / max(1, len(log['t'])):.2f}%）")
        if log_path is not None:
            SC.write_log(log_path, _log_arrays(log),
                         meta={"mode": mode, "time_scale": time_scale,
                               "kp": kp, "kd": kd,
                               "active_legs": list(active_legs), "source": source,
                               **meta, "secs": float(u[-1] / time_scale)})
    return True


def run_jog(d, leg, joint_name, kp, kd, log_path=None, jog_dir=None):
    """單關節微動驗號。只驅動一條腿的一個關節。

    ⚠️ 這是本檔最重要的安全關卡：唯一能抓到 calib_map 正負號錯誤的地方
       （calib_map.py 自己標注 hip 的號是「暫定」）。指令必須下在 MJCF
       空間（手冊判準表格用的座標：+knee→伸直、+abad→外張、+hip→後擺），
       不是 SHM 空間 —— 否則 sign 錯的時候，校正「正確」時操作者看到的
       方向會跟手冊判準完全相反，反而會把對的校正改壞成錯的
       （2026-08-11 總審報告記錄的實際事故路徑：leg0.hip 因此被改壞，
        離線檢驗五項全過，直到上機才會在 9 秒內把 FR 髖拉向機構限位）。

    做法：
      1. 讀當前 SHM 角度 shm0
      2. 轉成 MJCF：mjcf0 = (shm0 - offset) / sign
      3. 在 MJCF 空間做單方向 +0.10 rad 三角波（0 → +amp → 0，一個來回）
      4. 每一幀轉回 SHM：shm = sign * mjcf + offset
    這樣操作者看到的方向就直接對應 MJCF 正向定義，手冊判準表格不用改就是對
    的。順帶驗到的是「sign 與 offset 合起來的複合映射」——真正在乎的東西，
    而不是只驗 sign 單獨一項。

    log_path=None 時不寫檔（供離線測試用，不落地任何檔案）。
    """
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

    start_shm = np.array(SC.read_leg_q(d, leg))
    sign, offset = calib_map.CALIB[leg][joint_name]
    mjcf0 = (start_shm[ji] - offset) / sign

    amp, secs = 0.10, 8.0
    # 方向自動選行程比較充裕的一邊。腿吊著停在哪裡是隨機的，寫死方向就會
    # 遇到「那一邊已經頂到限位」而推不動（實機當天連續遇到三次：abad 內收
    # 頂底、hip 停在上限 0.011 rad 處、knee 伸直頂底）。
    d_mjcf, room = pick_jog_dir(leg, joint_name, mjcf0, amp, forced=jog_dir)
    move_desc = jog_phrase(leg, joint_name, d_mjcf)
    other_desc = jog_phrase(leg, joint_name, -d_mjcf)
    headroom = room[d_mjcf]
    lo_mjcf, hi_mjcf = JOINT_RANGE_MJCF[joint_name]

    if headroom < amp + 0.02:
        print(f"\n✗ 拒跑：leg{leg}({SC.LEGNAME[leg]}).{joint_name} 目前在 "
              f"MJCF {mjcf0:+.4f}，往「{move_desc}」只剩 {headroom:.4f} rad 行程，"
              f"不足 {amp:.2f}。")
        if room[-d_mjcf] >= amp + 0.02:
            print(f"  （往「{other_desc}」還有 {room[-d_mjcf]:.3f} rad——"
                  f"這是 --jog-dir 指定方向才會發生的情況，拿掉它就會自動選那邊）")
        else:
            print("  兩個方向都沒有行程。請先用手把這一軸移到中間位置再跑。")
        print(f"  （該軸 MJCF 機構範圍 [{lo_mjcf:+.4f}, {hi_mjcf:+.4f}]）")
        return False

    n = int(secs * SC.CTRL_HZ)
    ph = np.linspace(0.0, 1.0, n)                       # 0..1（含端點），一個來回
    tri = np.where(ph < 0.5, ph * 2.0, 2.0 - ph * 2.0)   # 0 → +1 → 0
    mjcf_delta = d_mjcf * amp * tri                      # MJCF 位移

    shm_dir = "正" if sign * d_mjcf > 0 else "負"
    print(f"\n[*] jog：leg{leg}({SC.LEGNAME[leg]}).{joint_name} "
          f"SHM 起點 {start_shm[ji]:+.4f} rad（換算 MJCF {mjcf0:+.4f} rad）")
    print(f"    行程餘裕：{move_desc} {room[d_mjcf]:.3f} rad / "
          f"{other_desc} {room[-d_mjcf]:.3f} rad"
          + ("（--jog-dir 指定）" if jog_dir is not None else "（自動選較充裕的一邊）"))
    print(f"\n    ★★ 這次應該看到：【{move_desc}】，幅度約 {amp:.2f} rad 一個來回 ★★\n")
    print(f"    calib_map sign={sign:+d} → SHM 指令會往{shm_dir}方向偏離起點。")
    print("    看到相反方向就停下來，跑 "
          f"sudo python3 L8_sign_probe.py --leg {leg} --stops 重量，")
    print("    不要直接改 calib_map。")

    full = np.tile(start_shm, (n, 4, 1))
    full[:, leg, ji] = sign * (mjcf0 + mjcf_delta) + offset

    log = {k: [] for k in ("t", "cmd", "p", "v", "tau", "overrun")}
    try:
        # jog 不查空中模擬表：±0.10 rad 的慢速微動，用 L4 的保守值就對。
        # ⚠️ 全部位置參數，不要改成關鍵字——測試用 spy 包一層 _stream，
        #    參數名跟 run_gait 那邊共用同一組（ta/va/...），關鍵字呼叫會炸。
        ok, why = _stream(d, full, (leg,), kp, kd, 8.0, 1.0, log, False, "jog")
    except (KeyboardInterrupt, Exception) as exc:
        reason = ("使用者中止(Ctrl+C)" if isinstance(exc, KeyboardInterrupt)
                  else f"例外：{exc!r}")
        print(f"\n⚠️ {reason} → 卸力中止")
        SC.passive_stop(d, (leg,), 300, STOP_KD)
        if log_path is not None:
            SC.write_log(_aborted_path(log_path), _log_arrays(log),
                         meta={"mode": "jog", "leg": leg, "joint": joint_name,
                               "kp": kp, "kd": kd, "start": start_shm.tolist(),
                               "active_legs": [leg], "time_scale": 1.0,
                               "source": "jog", "aborted": True,
                               "abort_reason": reason})
        raise

    if not ok:
        print(f"⚠️ 保護觸發：{why} → 卸力中止")
    SC.passive_stop(d, (leg,), 800, STOP_KD)

    # ⚠️ 這裡【不能】拿「指令 SHM 方向 vs 實測 SHM 方向」當驗號證據。
    #    兩邊都在 SHM 空間，馬達跟著指令走，所以不管 sign 對錯都會「一致」——
    #    那是套套邏輯。舊版就是這樣印「✓ 一致」，而當時 calib_map 有 11 項是錯的
    #    （2026-08-12 實機發現）。
    #
    #    sign 描述的是「編碼器座標 ↔ 物理現實」的關係，讀編碼器讀不出來。
    #    唯一的一手證據是【人眼看腿往哪動】，或用 L8_sign_probe.py 手扳量測。
    #    這段只報告伺服追得好不好，那是它真正能證明的事。
    if log["p"]:
        p_series = np.asarray(log["p"])[:, leg, ji]
        i_peak = int(np.argmax(np.abs(mjcf_delta[:len(p_series)])))
        shm_delta = float(p_series[i_peak] - start_shm[ji])
        cmd_delta = float(sign * mjcf_delta[i_peak])
        ratio = shm_delta / cmd_delta if abs(cmd_delta) > 1e-9 else 0.0
        print("\n[*] 伺服追蹤（這【不是】驗號證據，見下）：")
        print(f"    指令位移 {cmd_delta:+.4f} rad → 實測 {shm_delta:+.4f} rad"
              f"（追到 {ratio * 100:.0f}%）")
        if ratio < 0.5:
            print("    ⚠️ 追蹤量偏低，馬達可能沒出力或卡住——先確認再往下走。")
        print(f"\n[*] 驗號只能靠眼睛：剛才那條腿有沒有【{move_desc}】？")
        print("    有 → 這一軸的 calib_map 正確。沒有／反方向 → 停下來，")
        print(f"    跑 sudo python3 L8_sign_probe.py --leg {leg} --stops 重新量測。")
        print("    （不要用上面那個追蹤百分比判斷方向——指令與實測同在 SHM 空間，")
        print("      馬達跟著指令走，號對號錯都會「一致」。）")

    if log_path is not None:
        out_path = log_path if ok else _aborted_path(log_path)
        extra = {} if ok else {"aborted": True, "abort_reason": f"保護觸發：{why}"}
        SC.write_log(out_path, _log_arrays(log),
                     meta={"mode": "jog", "leg": leg, "joint": joint_name,
                           "kp": kp, "kd": kd, "start": start_shm.tolist(),
                           "active_legs": [leg], "time_scale": 1.0,
                           "source": "jog", **extra})
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
    ap.add_argument("--jog-dir", choices=("auto", "+", "-"), default="auto",
                    help="jog 往 MJCF 的哪一向動。預設 auto＝選行程比較充裕的一邊"
                         "（腿停在限位附近時才不會往裡壓）")
    ap.add_argument("--log", default=None,
                    help="log 檔路徑。預設依 mode/source/time_scale/時間戳自動命名"
                         "（避免多次執行互相覆蓋，見 G3-live 的比對需求）")
    ap.add_argument("--confirm", action="store_true", help="真的驅動硬體")
    args = ap.parse_args()

    if args.mode in ("leg", "gait") and not args.traj:
        ap.error("--mode leg/gait 需要 --traj")

    if args.log is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.log = f"l7_log_{args.mode}_{args.source}_{args.time_scale:g}x_{stamp}.npz"

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
        if not (0 <= args.only_leg <= 3):
            ap.error(f"--only-leg 只能是 0~3，收到 {args.only_leg}")
        if args.only_leg in skip:
            ap.error(f"--only-leg {args.only_leg} 同時被 --skip-legs 排除了")
        active = (args.only_leg,)
    if args.mode == "jog":
        if not (0 <= args.jog_leg <= 3):
            ap.error(f"--jog-leg 只能是 0~3，收到 {args.jog_leg}")
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
        if args.mode in ("gait", "leg"):
            run_gait(None, traj, meta, active, args.time_scale,
                     args.kp, args.kd, dry=True, log_path=args.log,
                     mode=args.mode, source=args.source)
        else:
            # jog 模式沒開 SHM，讀不到當前角度，所以只能預覽方向邏輯，不能
            # 預覽實際數值——理由跟當初補 leg 模式 dry-run 一樣：跑真機前
            # 至少要能先看到這條路徑不會馬上炸掉。
            leg = active[0]
            sign, offset = calib_map.CALIB[leg][args.jog_joint]
            print(f"\n[*] jog 計畫：leg{leg}({SC.LEGNAME[leg]}).{args.jog_joint}，"
                  f"MJCF 空間 +0.10 rad 一個來回（0 → +amp → 0）")
            print(f"    calib_map：sign={sign:+d}，offset={offset:+.4f}")
            print(f"    校正正確時，SHM 指令應往{'正' if sign > 0 else '負'}"
                  f"方向偏離目前角度；操作者應看到符合 MJCF 正向定義的動作"
                  f"（+knee→伸直、+abad→外張、+hip→後擺）。")
            print("    （dry-run 未開 SHM，讀不到目前角度，無法預覽實際數值；"
                  "--confirm 執行時 run_jog 會印出數值佐證。）")
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
                     args.kp, args.kd, dry=False, log_path=args.log,
                     mode=args.mode, source=args.source)
        else:
            run_jog(d, active[0], args.jog_joint, args.kp, args.kd, args.log,
                    jog_dir=None if args.jog_dir == 'auto' else
                    (+1 if args.jog_dir == '+' else -1))
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C → 卸力收尾")
        SC.passive_stop(d, active, 800, STOP_KD)
    finally:
        SC.zero_all(d)
        SC.publish(d)
        print("[*] 已歸零收尾，watchdog 兜底。測完 SIGCONT 解凍 mc_ctrl 還原。")


if __name__ == "__main__":
    main()
