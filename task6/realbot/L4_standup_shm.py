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
  2) 用 ratio 把 p_des 從起點內插到目標 → 平滑站姿
  3) 結束用 kp=0、kd 大的「卸力收尾」   → 軟軟停住

新增（來自我們 L1~L3 的實機經驗）：
  - 預檢 1：偵測 cmd 旗標仍在跳動（mc_ctrl 沒停）就中止，不寫
  - 預檢 2：16 顆馬達任一 ready=0 / 已失聯 / 有故障位就中止，不寫（2026-08-11 硬體故障後補上）
  - 每關節力矩/速度超限即中止並卸力
  - 輪子(foot) 全程零增益（站姿不控制輪子）
  - 預設 dry-run（不碰硬體，只印出動作計畫）；要真的驅動硬體必須加 --confirm

動作分兩階段：
  階段 1「接住」(CATCH_SEC)：p_des 固定在當前實際角度，kp/kd 由 0 平滑升到設定值。
      凍結 mc_ctrl 後腿會因重力垂下（模型估：hip −60°、knee +79° 直到頂住機構限位），
      先用漸入增益把腿接住，避免撞限位，也避免一上來就有力矩突跳。
  階段 2「站起」(RAMP_SEC)：在滿增益下把 p_des 從接住點內插到 POSE_STAND。

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
EXPECT_SIZE = 608          # cmd 344 + state 264；對不上就不准跑（見 main()）

# ============================================================
# 2) 參數（對應範例；⚠️ POSE 需校正、gains 可調）
# ============================================================
CTRL_HZ   = 500
DT        = 1.0 / CTRL_HZ
CATCH_SEC = 0.5            # 階段 1：增益漸入「接住」腿的時間
RAMP_SEC  = 2.0            # 階段 2：內插到站姿的時間（同範例 duration）

# ✅ 2026-08-11 改用【原廠實測值】：L0_cmd_probe 量到輪足 D1 EDU 原廠 mc_ctrl 站立時，
#    16 個腿關節恆定 kp=20.000 / kd=0.700（輪關節 kp=0/kd=0.1，本程式輪子直接零增益）。
#    先前的 kp=80/kd=1 是從【點足版】lowlevel_demo.py 借來的，對輪足沒有依據。
#    代價：重力造成的穩態下垂由 1.9° 變成 7.6°（MJCF 算得 knee 重力 2.67 N·m ÷ kp），
#    這是可預測的系統性偏差，不是校正誤差。要更緊的追蹤再往上調。
LEG_KP    = 20.0          # 原廠站立值
LEG_KD    = 0.7           # 原廠站立值
STOP_KD   = 3.0           # 卸力收尾的阻尼（同範例；純阻尼不會灌能量，維持原值）

# ✅ 由 calib_capture.py 於 2026-08-10 從實機站姿擷取（每腿各自，已是輪足編碼器慣例）。
#    左右腿鏡像（號相反），所以每腿一組，不是 4 腿共用。來源：calib_stand.json
POSE_STAND = {
    0: {"abad": +0.5061, "hip": -2.2156, "knee": +1.2557},  # leg0 FR 右前
    1: {"abad": -0.5110, "hip": +2.2073, "knee": -1.2825},  # leg1 FL 左前
    2: {"abad": -0.5039, "hip": -2.1493, "knee": +1.2194},  # leg2 RR 右後
    3: {"abad": +0.5132, "hip": +2.1528, "knee": -1.2570},  # leg3 RL 左後
}

# 安全保護（真機模式才作用）
# 依 MJCF (task6/model/d1_edu_w) 計算：吊掛站姿下重力力矩最大僅 knee 2.67 / abad 2.46 N·m，
# 2 秒 ramp 的最大角速度約 0.63 rad/s。門檻取「實際需求的 3 倍上下」，出事才攔得住。
TORQUE_ABORT = 8.0        # 任一關節力矩超過 → 中止（重力只需 ~2.7，超過 8 代表卡住或打架）
VEL_ABORT    = 2.0        # 任一關節速度超過 rad/s → 中止（ramp 只需 ~0.63）

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

def preflight_motors_healthy(d):
    """16 顆馬達必須全部 ready 且無故障位，否則拒絕執行。回傳 (ok, 問題清單)。

    為什麼需要這道關卡（2026-08-11 實機教訓）：
      當天實機出現「右後輪馬達 ready=0」的硬體故障，原廠 mc_ctrl 因此把全部 16 顆
      disable 並閃紅燈。而我們的做法是 SIGSTOP 凍結 mc_ctrl 再直接寫 SHM ——
      等於繞過了當下唯一在動作的安全機制。舊版預檢只檢查 mc_ctrl 有沒有停，
      在那種狀態下會照樣往下寫。這道關卡就是補這個缺口。

    判定依據：
      - flags bit0 = ready。0 = 該馬達未啟用/故障
      - 完全失聯的節點會被 daemon 歸零 → 溫度與電壓欄位讀到 0
      - bit1~5 = 過壓 / 過流 / 過溫 / 超速 / 雙編碼器故障
    """
    problems = []
    for i in range(4):
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
        # 預檢 1：mc_ctrl 必須已凍結
        ok, trans = preflight_mc_stopped(d)
        if not ok:
            print(f"✗ 中止：cmd 旗標仍在跳動({trans}) → mc_ctrl 沒停。先 SIGSTOP mc_ctrl。")
            return False

        # 預檢 2：16 顆馬達必須全部健康（見 preflight_motors_healthy 的說明）
        ok, problems = preflight_motors_healthy(d)
        if not ok:
            print(f"✗ 中止：偵測到 {len(problems)} 個馬達問題，拒絕寫入 ——")
            for p in problems:
                print(f"    • {p}")
            print("  在馬達故障狀態下寫入 = 繞過原廠唯一在動作的安全機制。")
            print("  先排除硬體問題（可用 L5_faultwatch.py --once 檢視）再重試。")
            return False
        print("[*] 預檢通過：mc_ctrl 已凍結、16 顆馬達全部 ready 且無故障位。")

        init = [read_leg_q(d, i) for i in range(4)]
        print("[*] 讀到起點角度：")
        for i, (a, h, k) in enumerate(init):
            print(f"    leg{i}: abad={a:+.3f} hip={h:+.3f} knee={k:+.3f}")

    def drive(targets, kp, kd):
        """把四條腿的目標一次寫進 cmd 並送出。回傳 (ok, 原因)。

        ⚠️ zero_all() 一定要在「四條腿的迴圈之外」——它會清掉全部四條腿，
           放進迴圈裡會把前一條腿剛設好的指令抹掉，最後只剩 leg3 有指令。
        """
        zero_all(d)                                   # 先壓全零（輪子也歸零）
        for i, (a, h, k) in enumerate(targets):
            set_leg_position(d, i, a, h, k, kp, kd)
        publish(d)
        return check_guards(d)

    def abort(why):
        print(f"⚠️ 保護觸發：{why} → 卸力中止")
        passive_stop(d, 300)

    # --- 階段 1：增益漸入「接住」腿（p_des 固定在當前角度，kp/kd 由 0 升上來）---
    n_catch = int(CATCH_SEC / DT)
    print(f"\n[*] 階段 1：接住（{CATCH_SEC}s，p_des 保持不動，kp 0→{LEG_KP}）")
    for step in range(n_catch + 1):
        ratio = min(step / n_catch, 1.0)
        kp, kd = LEG_KP * ratio, LEG_KD * ratio
        if not dry:
            ok, why = drive(init, kp, kd)
            if not ok:
                abort(why)
                return False
            time.sleep(DT)
        elif step % max(1, int(0.25 / DT)) == 0:
            print(f"  t={step*DT:4.2f}s  kp={kp:5.1f} kd={kd:4.2f}  "
                  f"leg0 目標 abad={init[0][0]:+.3f} hip={init[0][1]:+.3f} knee={init[0][2]:+.3f}（不動）")

    # --- 階段 2：滿增益下，從接住點平滑內插到 POSE_STAND（每腿各自的目標）---
    n = int(RAMP_SEC / DT)
    print(f"\n[*] 階段 2：站起（{RAMP_SEC}s，kp={LEG_KP} kd={LEG_KD}）")
    for step in range(n + 1):
        ratio = min(step / n, 1.0)
        targets = []
        for i in range(4):
            a0, h0, k0 = init[i]
            tgt = POSE_STAND[i]
            targets.append((interpolate(a0, tgt["abad"], ratio),
                            interpolate(h0, tgt["hip"],  ratio),
                            interpolate(k0, tgt["knee"], ratio)))
        if not dry:
            ok, why = drive(targets, LEG_KP, LEG_KD)
            if not ok:
                abort(why)
                return False
            time.sleep(DT)
        elif step % int(0.5 / DT) == 0:           # dry-run：每 0.5 秒印一次 leg0 目標
            a, h, k = targets[0]
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

    # 結構大小對不上 = 這支程式對 SHM 的理解跟 daemon 不一致，寫下去會寫到錯的欄位。
    # 用 sys.exit 而非 assert：assert 在 python -O 下會被拿掉，這道關卡不能被關掉。
    size = ctypes.sizeof(SplineData)
    print(f"[*] SplineData 結構大小 = {size} bytes（應為 {EXPECT_SIZE}）")
    if size != EXPECT_SIZE:
        print(f"✗ 結構大小不符（{size} != {EXPECT_SIZE}）→ 拒絕執行，避免寫壞共享記憶體。")
        sys.exit(1)

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
