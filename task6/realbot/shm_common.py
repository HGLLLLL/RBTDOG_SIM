#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shm_common.py —— D1 EDU 輪足實機 SHM 讀寫的共用底層。

由 L4_standup_shm.py 於 2026-08-11 抽出，供 L4（姿勢序列）與 L7（步態串流）共用。

★ 為什麼要抽出：SHM 結構定義絕對不能有兩份。欄位順序漂掉是靜默的 ——
  不會報錯，只會把 kp 寫進 v_des、把指令寫到錯的關節。

★ 依賴限制：本檔在【車載電腦】上執行，只能用 python3 標準庫。
  不得 import mujoco / torch / scipy。
"""

import ctypes
import os
import sys
import time
from pathlib import Path

# ============================================================
# 1) 共享記憶體結構定義（對應 include/lowlevel/lowlevel.h，packed）
# ============================================================
class JointControl(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("p_des", ctypes.c_float), ("v_des", ctypes.c_float),
                ("kp", ctypes.c_float), ("kd", ctypes.c_float), ("t_ff", ctypes.c_float)]

class LegControl(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("abad", JointControl), ("hip", JointControl),
                ("knee", JointControl), ("foot", JointControl), ("flags", ctypes.c_int32)]

class SplineCmd(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("legs", LegControl * 4), ("consumer_flags", ctypes.c_uint32 * 2)]

class JointState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("flags", ctypes.c_int32), ("p", ctypes.c_float),
                ("v", ctypes.c_float), ("t", ctypes.c_float)]

class LegState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("abad", JointState), ("hip", JointState),
                ("knee", JointState), ("foot", JointState)]

class SplineState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("legs", LegState * 4), ("consumer_flags", ctypes.c_uint32 * 2)]

class SplineData(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("cmd", SplineCmd), ("state", SplineState)]

CONSUMER_CONTROL = 0
CONSUMER_OTHER   = 1
SHM_PATH = "/dev/shm/spline_shm"
SHM_SIZE = 1024 * 10
EXPECT_SIZE = 608          # cmd 344 + state 264；對不上就不准跑（見 check_struct_size）

# ============================================================
# 2) 參數（對應範例；⚠️ POSE 需校正、gains 可調）
# ============================================================
CTRL_HZ   = 500
DT        = 1.0 / CTRL_HZ
STOP_KD   = 3.0           # 卸力收尾的阻尼（同範例；純阻尼不會灌能量，維持原值）

# ✅ 由 calib_capture.py 於 2026-08-10 從實機站姿擷取（每腿各自，已是輪足編碼器慣例）。
#    左右腿鏡像（號相反），所以每腿一組，不是 4 腿共用。來源：calib_stand.json
POSE_STAND = {
    0: {"abad": +0.5061, "hip": -2.2156, "knee": +1.2557},  # leg0 FR 右前
    1: {"abad": -0.5110, "hip": +2.2073, "knee": -1.2825},  # leg1 FL 左前
    2: {"abad": -0.5039, "hip": -2.1493, "knee": +1.2194},  # leg2 RR 右後
    3: {"abad": +0.5132, "hip": +2.1528, "knee": -1.2570},  # leg3 RL 左後
}

# ✅ 同樣由 calib_capture.py 於 2026-08-10 從【這台實機】的趴姿擷取。來源：calib_lie.json
#    趴→站的位移：knee 約 ±72°（主要動作）、abad 約 ±27°、hip 約 ±14°
POSE_LIE = {
    0: {"abad": +0.9836, "hip": -1.9568, "knee": +0.0027},  # leg0 FR 右前
    1: {"abad": -0.9756, "hip": +1.9507, "knee": -0.0044},  # leg1 FL 左前
    2: {"abad": -0.9865, "hip": -1.9396, "knee": +0.0067},  # leg2 RR 右後
    3: {"abad": +0.9783, "hip": +1.9251, "knee": -0.0068},  # leg3 RL 左後
}

# 可用於 --sequence 的姿勢名稱
POSES = {"lie": POSE_LIE, "stand": POSE_STAND}
POSE_LABEL = {"lie": "趴下", "stand": "站立"}

# leg0=FR 與 leg2=RR 均已實機確認；leg1=FL / leg3=RL 由對稱性推論
LEGNAME = {0: "FR", 1: "FL", 2: "RR", 3: "RL"}
# state flags 的故障位（bit0 是 ready，不在此表）
FAULT_BITS = ((1, "過壓"), (2, "過流"), (3, "過溫"), (4, "超速"), (5, "雙編碼器故障"))


# ============================================================
# 3) 共享記憶體存取
# ============================================================
def open_shm():
    fd = os.open(SHM_PATH, os.O_RDWR)          # 需 root
    import mmap
    buf = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    return SplineData.from_buffer(buf), buf

def zero_all(d):
    """16 關節全部零增益（安全底色），每條腿 flags=1。"""
    for i in range(4):
        leg = d.cmd.legs[i]
        for jn in ("abad", "hip", "knee", "foot"):
            j = getattr(leg, jn)
            j.p_des = j.v_des = j.kp = j.kd = j.t_ff = 0.0
        leg.flags = 1

def set_leg_position(d, i, abad, hip, knee, kp, kd):
    """對第 i 條腿的 abad/hip/knee 下位置指令；foot(輪) 維持零增益。"""
    leg = d.cmd.legs[i]
    for jn, val in (("abad", abad), ("hip", hip), ("knee", knee)):
        j = getattr(leg, jn)
        j.p_des = val; j.kp = kp; j.kd = kd; j.v_des = 0.0; j.t_ff = 0.0

def publish(d):
    d.cmd.consumer_flags[CONSUMER_CONTROL] = 1

def read_leg_q(d, i):
    s = d.state.legs[i]
    return s.abad.p, s.hip.p, s.knee.p

def preflight_mc_stopped(d):
    """0.4 秒觀察 cmd 旗標；仍在跳動代表 mc_ctrl 沒停。"""
    prev = d.cmd.consumer_flags[CONSUMER_CONTROL]; trans = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.4:
        c = d.cmd.consumer_flags[CONSUMER_CONTROL]
        if c != prev:
            trans += 1; prev = c
    return trans <= 4, trans

def preflight_motors_healthy(d, active_legs):
    """被驅動的腿其馬達必須全部 ready 且無故障位，否則拒絕執行。回傳 (ok, 問題清單)。

    為什麼需要這道關卡（2026-08-11 實機教訓）：
      當天實機出現「右後輪馬達 ready=0」的硬體故障，原廠 mc_ctrl 因此把全部 16 顆
      disable 並閃紅燈。而我們的做法是 SIGSTOP 凍結 mc_ctrl 再直接寫 SHM ——
      等於繞過了當下唯一在動作的安全機制。舊版預檢只檢查 mc_ctrl 有沒有停，
      在那種狀態下會照樣往下寫。這道關卡就是補這個缺口。

    判定依據：
      - flags bit0 = ready。0 = 該馬達未啟用/故障
      - 完全失聯的節點會被 daemon 歸零 → 溫度與電壓欄位讀到 0
      - bit1~5 = 過壓 / 過流 / 過溫 / 超速 / 雙編碼器故障

    只檢查 active_legs（會被驅動的腿）。被 --skip-legs 排除的腿不檢查，
    但 report_legs() 會把它們的狀態原樣印出來，讓操作者親眼確認跳過的是什麼。
    """
    problems = []
    for i in active_legs:
        s = d.state.legs[i]
        for jn in ("abad", "hip", "knee", "foot"):
            f = getattr(s, jn).flags
            temp = (f >> 8) & 0xFF          # 原始值直接是攝氏度（無 -40 偏移）
            volt = (f >> 16) & 0xFF
            tag = f"{LEGNAME[i]}.{jn}"
            if not (f & 1):
                problems.append(f"{tag} ready=0（未啟用/故障）")
            if temp == 0 and volt == 0:
                problems.append(f"{tag} 溫度與電壓都讀到 0 → 已從 CAN 失聯")
            for bit, name in FAULT_BITS:
                if (f >> bit) & 1:
                    problems.append(f"{tag} {name}")
    return not problems, problems

def report_legs(d, active_legs):
    """把四條腿的馬達狀態原樣印出，標明哪些會被驅動、哪些被跳過。"""
    print("\n[*] 16 顆馬達現況（★ = 這次會被驅動）：")
    for i in range(4):
        mark = "★" if i in active_legs else " "
        cells = []
        for jn in ("abad", "hip", "knee", "foot"):
            f = getattr(d.state.legs[i], jn).flags
            cells.append(f"{jn}:{'ok' if f & 1 else 'XX'}/{(f >> 8) & 0xFF}C")
        print(f"    {mark} leg{i} {LEGNAME[i]}  " + "  ".join(f"{c:<14}" for c in cells))

def check_guards(d, active_legs, torque_abort, vel_abort):
    """回傳 (ok, 說明)。被驅動的腿任一關節力矩/速度超限 → not ok。

    門檻改成參數傳入（原本是模組常數）：L4 的站姿 ramp 與 L7 的步態串流
    需求差一個數量級（步態膝關節要 ~13.5 rad/s，L4 的 ramp 只要 0.63），
    共用常數會讓其中一邊不是誤中止就是形同虛設。

    只檢查 active_legs：被跳過的腿若是故障中，其 t/v 欄位可能是損壞資料
    （實測 2026-08-11 的 RR.foot 會在 0/28/44/95 之間亂跳），拿來當保護判準
    會造成無意義的誤中止。被跳過的腿全程零增益，本來就不會出力。
    """
    for i in active_legs:
        s = d.state.legs[i]
        for jn in ("abad", "hip", "knee", "foot"):
            js = getattr(s, jn)
            if abs(js.t) > torque_abort:
                return False, f"{LEGNAME[i]}.{jn} 力矩 {js.t:.2f} > {torque_abort}"
            if abs(js.v) > vel_abort:
                return False, f"{LEGNAME[i]}.{jn} 速度 {js.v:.2f} > {vel_abort}"
    return True, ""

def passive_stop(d, active_legs, cycles=1500, stop_kd=3.0):
    """卸力收尾：被驅動的腿 kp=0、kd=stop_kd，軟軟停住；被跳過的腿維持全零。"""
    for _ in range(cycles):
        zero_all(d)
        for i in active_legs:
            for jn in ("abad", "hip", "knee"):
                getattr(d.cmd.legs[i], jn).kd = stop_kd
        publish(d)
        time.sleep(DT)
    zero_all(d)
    publish(d)


def check_struct_size():
    """結構大小對不上 = 這支程式對 SHM 的理解跟 daemon 不一致，寫下去會寫到錯的欄位。

    用 sys.exit 而非 assert：assert 在 python -O 下會被拿掉，這道關卡不能被關掉。
    """
    size = ctypes.sizeof(SplineData)
    print(f"[*] SplineData 結構大小 = {size} bytes（應為 {EXPECT_SIZE}）")
    if size != EXPECT_SIZE:
        print(f"✗ 結構大小不符（{size} != {EXPECT_SIZE}）→ 拒絕執行，避免寫壞共享記憶體。")
        sys.exit(1)


def write_log(path, log, meta):
    """把一次執行的 cmd/state 記錄寫成 npz。由 gait_export --analyze 讀。

    欄位契約（改動要同步改 gait_export 的讀取端與 test_write_and_read_log_roundtrip）：
      t (N,)  cmd (N,4,3)  p (N,4,3)  v (N,4,3)  tau (N,4,3)  overrun (N,) bool
    索引是 SHM 腿序 (0=FR 1=FL 2=RR 3=RL)、關節序 (abad, hip, knee)。
    """
    import json
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
             **{k: np.asarray(v) for k, v in log.items()})
    print(f"[記錄] {path}  {len(log['t'])} 筆")
