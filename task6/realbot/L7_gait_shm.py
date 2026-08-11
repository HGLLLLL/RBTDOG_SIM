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
