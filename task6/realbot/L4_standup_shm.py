#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4_standup_shm.py —— 把官方點足範例 lowlevel_demo.py 改成「共享記憶體」寫法，讓輪足也能用。

移植來源：
  demo/zsl-1/python/examples/lowlevel_demo.py（點足 zsl-1，走 UDP 的 mc_sdk_zsl_1_py.LowLevel）
本檔改為：
  直接讀寫 /dev/shm/spline_shm（不透過 SDK、不透過 UDP），因此在【輪足】上也能用。

保留範例的三個範式：
  1) 讀「當前角度」當起點           → 避免 p_des 一開始就跟實際差很多造成力矩突跳
  2) 用 ratio 把 p_des 從起點內插到目標 → 兩段式平滑站姿
  3) 結束用 kp=0、kd 大的「卸力收尾」   → 軟軟停住

新增（來自我們 L1~L3 的實機經驗）：
  - 預檢：偵測 cmd 旗標仍在跳動（mc_ctrl 沒停）就中止，不寫
  - 每關節力矩/速度超限即中止並卸力
  - 輪子(foot) 全程零增益（站姿不控制輪子）
  - 預設 dry-run（不碰硬體，只印出動作計畫）；要真的驅動硬體必須加 --confirm

======================= ⚠️ 跑真機前必讀 =======================
(1) 這會驅動【腿關節】→ 狗會站起來、身體會動 → 【必須把狗吊起來】，不能平躺/墊肚子。
(2) POSE_STAND 已由 calib_capture.py 從【這台實機】的站姿擷取（每腿各自、輪足編碼器慣例）。
    但它只對「當時擺的那個站姿」有效；換了狗或重新校正就要重抓。真機前建議再擷取一次確認。
(3) 需要 root（spline_shm 是 root 擁有）：sudo。
(4) 需先凍結/停 mc_ctrl（本程式會預檢；建議用 SIGSTOP）。
==============================================================

用法：
  python3 L4_standup_shm.py                 # dry-run：離線印出動作計畫，不碰硬體（預設）
  sudo python3 L4_standup_shm.py --confirm   # 真的驅動硬體（前提：狗吊掛、POSE 已校正、mc_ctrl 已停）
"""

import argparse
import ctypes
import os
import sys
import time

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

# ============================================================
# 2) 參數（對應範例；⚠️ POSE 需校正、gains 可調）
# ============================================================
CTRL_HZ   = 500
DT        = 1.0 / CTRL_HZ
RAMP_SEC  = 2.0            # 每一段內插時間（同範例 duration）

LEG_KP    = 80.0          # 範例值（撐體重的腿關節）
LEG_KD    = 1.0
STOP_KD   = 3.0           # 卸力收尾的阻尼（同範例）

# ✅ 由 calib_capture.py 於 2026-08-10 從實機站姿擷取（每腿各自，已是輪足編碼器慣例）。
#    左右腿鏡像（號相反），所以每腿一組，不是 4 腿共用。來源：calib_stand.json
POSE_STAND = {
    0: {"abad": +0.5061, "hip": -2.2156, "knee": +1.2557},  # leg0 FR 右前
    1: {"abad": -0.5110, "hip": +2.2073, "knee": -1.2825},  # leg1 FL 左前
    2: {"abad": -0.5039, "hip": -2.1493, "knee": +1.2194},  # leg2 RR 右後
    3: {"abad": +0.5132, "hip": +2.1528, "knee": -1.2570},  # leg3 RL 左後
}

# 安全保護（真機模式才作用）
TORQUE_ABORT = 20.0       # 任一腿關節力矩超過 → 中止（腿關節較強，門檻高於輪子測試）
VEL_ABORT    = 8.0        # 任一關節速度超過 rad/s → 中止


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

def check_guards(d):
    """回傳 (ok, 說明)。任一關節力矩/速度超限 → not ok。"""
    for i in range(4):
        s = d.state.legs[i]
        for jn in ("abad", "hip", "knee", "foot"):
            js = getattr(s, jn)
            if abs(js.t) > TORQUE_ABORT:
                return False, f"leg{i}.{jn} 力矩 {js.t:.2f} > {TORQUE_ABORT}"
            if abs(js.v) > VEL_ABORT:
                return False, f"leg{i}.{jn} 速度 {js.v:.2f} > {VEL_ABORT}"
    return True, ""

def passive_stop(d, cycles=1500):
    """卸力收尾：腿關節 kp=0、kd=STOP_KD，軟軟停住（同範例）。"""
    for _ in range(cycles):
        zero_all(d)
        for i in range(4):
            for jn in ("abad", "hip", "knee"):
                getattr(d.cmd.legs[i], jn).kd = STOP_KD
        publish(d)
        time.sleep(DT)
    zero_all(d)
    publish(d)


# ============================================================
# 4) 站姿主邏輯（移植 run()：讀起點 → 兩段內插）
# ============================================================
def interpolate(a, b, ratio):
    return a * (1.0 - ratio) + b * ratio

def run_standup(d, dry=True):
    # --- 起點 ---
    if dry:
        init = [(0.0, 0.0, 0.0)] * 4          # 離線用 0 當起點示意
        print("[dry-run] 假設起點角度全為 0（真機會讀 state.legs[*]）\n")
    else:
        ok, trans = preflight_mc_stopped(d)
        if not ok:
            print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。先 SIGSTOP mc_ctrl。")
            return False
        init = [read_leg_q(d, i) for i in range(4)]
        print("[*] 讀到起點角度：")
        for i, (a, h, k) in enumerate(init):
            print(f"    leg{i}: abad={a:+.3f} hip={h:+.3f} knee={k:+.3f}")

    # 單段內插：從當前姿勢平滑走到 POSE_STAND（每腿各自的目標）
    n = int(RAMP_SEC / DT)
    for step in range(n + 1):
        ratio = min(step / n, 1.0)
        for i in range(4):
            a0, h0, k0 = init[i]
            tgt = POSE_STAND[i]
            a = interpolate(a0, tgt["abad"], ratio)
            h = interpolate(h0, tgt["hip"],  ratio)
            k = interpolate(k0, tgt["knee"], ratio)
            if not dry:
                zero_all(d)                       # 先壓全零（輪子也歸零）
                set_leg_position(d, i, a, h, k, LEG_KP, LEG_KD)
        if not dry:
            publish(d)
            ok, why = check_guards(d)
            if not ok:
                print(f"⚠️ 保護觸發：{why} → 卸力中止")
                passive_stop(d, 300)
                return False
            time.sleep(DT)
        elif step % int(0.5 / DT) == 0:           # dry-run：每 0.5 秒印一次 leg0 目標
            a0, h0, k0 = init[0]
            tgt = POSE_STAND[0]
            a = interpolate(a0, tgt["abad"], ratio)
            h = interpolate(h0, tgt["hip"],  ratio)
            k = interpolate(k0, tgt["knee"], ratio)
            print(f"  t={step*DT:4.1f}s ratio={ratio:.2f}  "
                  f"leg0 目標 abad={a:+.3f} hip={h:+.3f} knee={k:+.3f}  (kp={LEG_KP} kd={LEG_KD})")

    print("\n[*] 站姿序列完成。")
    if not dry:
        print("[*] 進行卸力收尾 ...")
        passive_stop(d, 800)
    return True


# ============================================================
# 5) 進入點
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="輪足 D1 EDU：SHM 版站姿（移植自點足 lowlevel_demo.py）")
    ap.add_argument("--confirm", action="store_true",
                    help="真的驅動硬體（否則只做 dry-run，不碰硬體）")
    args = ap.parse_args()

    print(f"[*] SplineData 結構大小 = {ctypes.sizeof(SplineData)} bytes（應為 608）")

    if not args.confirm:
        print("="*66)
        print("DRY-RUN 模式：不開啟、不寫入共享記憶體，只印出動作計畫。")
        print("要真的驅動硬體請加 --confirm（且需 sudo）。")
        print("="*66 + "\n")
        run_standup(None, dry=True)
        print("\n⚠️ 提醒：跑真機前必讀檔頭四點 —— 尤其【狗要吊掛】與【POSE 需校正】。")
        return

    # ---- 真機模式 ----
    print("="*66)
    print("⚠️ 真機模式：即將驅動【腿關節】。確認：狗已吊掛、POSE 已校正、mc_ctrl 已停。")
    print("="*66)
    if os.geteuid() != 0:
        print("✗ 需要 root：請用 sudo 執行。")
        sys.exit(1)
    try:
        d, buf = open_shm()
    except FileNotFoundError:
        print(f"✗ 找不到 {SHM_PATH}（機器人運控沒起來？）")
        sys.exit(1)
    except PermissionError:
        print("✗ 權限不足：請用 sudo。")
        sys.exit(1)

    print(f"[*] 已映射 {SHM_PATH}")
    try:
        run_standup(d, dry=False)
    except KeyboardInterrupt:
        print("\n[*] 收到 Ctrl+C → 卸力收尾")
        passive_stop(d, 800)
    finally:
        zero_all(d); publish(d)
        print("[*] 已歸零收尾，watchdog 兜底。測完 SIGCONT 解凍 mc_ctrl 還原。")


if __name__ == "__main__":
    main()
