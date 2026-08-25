#!/usr/bin/env python3
"""M0 —— 寫入前的唯讀前提檢查（不需 root、不寫任何東西、不碰行程）。

回答四件在動手寫 joint_cmd 之前必須先知道的事：

  1. 結構還對不對（關節名稱順序是否與 shm_io.JOINTS 一致）
  2. joint_cmd / joint_state 的**實際更新頻率**（決定我們要用多快的迴圈去蓋寫）
  3. 目前馬達是不是洩力狀態（kp / kd / effort 是否全為 0）
  4. sudo 能不能用、mc_ctrl 的 PID 是多少（M1 要凍結它）

在狗上執行：
    python3 M0_probe.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import shm_io


def hr(t=""):
    print("\n" + "=" * 60)
    if t:
        print(t)
        print("=" * 60)


def sample_rate(name: str, stride: int, secs: float = 2.0) -> tuple[int, float]:
    """在 secs 秒內數這塊 shm 的內容變了幾次，換算更新頻率。

    ⚠️ 這是「我們觀察到的變化次數」，不是 daemon 真正的寫入頻率 ——
    連續兩次寫入若數值相同，我們數不到。所以這是**下限**。
    """
    with shm_io.Shm(name) as s:
        lo = shm_io.BASE
        hi = shm_io.BASE + len(shm_io.JOINTS) * stride
        prev = bytes(s.mm[lo:hi])
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            cur = bytes(s.mm[lo:hi])
            if cur != prev:
                n += 1
                prev = cur
        dt = time.monotonic() - t0
    return n, n / dt


def main() -> int:
    print("M0 —— D1 Max 寫入前唯讀前提檢查")
    print(f"時間 {time.strftime('%Y-%m-%d %H:%M:%S')}   使用者 {os.getenv('USER', '?')}   uid={os.geteuid()}")

    # ---------------------------------------------------------------- 1 結構
    hr("1. 結構檢查")
    for nm, stride in (("joint_cmd", shm_io.CMD_STRIDE),
                       ("joint_state", shm_io.STATE_STRIDE)):
        path = os.path.join(shm_io.SHM_DIR, nm)
        if not os.path.exists(path):
            print(f"❌ {path} 不存在")
            return 1
        st = os.stat(path)
        print(f"   {nm:12s} 大小 {st.st_size}  權限 {oct(st.st_mode)[-3:]}  "
              f"uid={st.st_uid} gid={st.st_gid}")
        with shm_io.Shm(nm) as s:
            s.verify_layout(stride)
        print(f"   {nm:12s} ✅ 16 顆關節名稱與順序正確")

    # ---------------------------------------------------------------- 2 更新頻率
    hr("2. 更新頻率（觀察到的變化次數／秒，為下限）")
    for nm, stride in (("joint_cmd", shm_io.CMD_STRIDE),
                       ("joint_state", shm_io.STATE_STRIDE)):
        n, hz = sample_rate(nm, stride)
        verdict = "活的" if n > 0 else "★ 靜止（沒有人在寫）"
        print(f"   {nm:12s} 2 秒內變化 {n:5d} 次 → ≥ {hz:7.1f} Hz   {verdict}")
    print("\n   參考：controller_manager update_rate = 1000 Hz；")
    print("        joint_shm_controller.joint_cmd_timeout = 500 ms")
    print("        → 我們接管後必須以「遠快於 2 Hz」的頻率持續寫，否則會觸發逾時。")

    # ---------------------------------------------------------------- 3 目前指令
    hr("3. 目前的 joint_cmd（確認馬達是不是洩力）")
    cmd = shm_io.read_joint_cmd()
    st = shm_io.read_joint_state()
    print(f"   {'關節':16s} {'p_des':>9s} {'kp':>7s} {'kd':>7s} {'tau_ff':>8s} "
          f"| {'實測角':>9s} {'實測速':>8s} {'力矩':>7s} {'溫度':>6s}")
    live = 0
    for c, s_ in zip(cmd, st):
        if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9:
            live += 1
        print(f"   {c['name']:16s} {c['position']:9.4f} {c['kp']:7.2f} {c['kd']:7.2f} "
              f"{c['effort']:8.3f} | {s_['position']:9.4f} {s_['velocity']:8.4f} "
              f"{s_['effort']:7.3f} {s_['temp_C']:6.1f}")
    print()
    if live == 0:
        print("   ✅ 16 顆全部 kp=kd=tau_ff=0 → 馬達洩力中，凍結 mc_ctrl 不會有動作")
    else:
        print(f"   ⚠️ 有 {live} 顆關節帶著非零增益 → 馬達正在出力。")
        print("      凍結 mc_ctrl 前請先用遙控器讓狗趴下／洩力，否則腿會掉。")

    # ---------------------------------------------------------------- 4 權限與行程
    hr("4. 權限與 mc_ctrl")
    pid = None
    try:
        pid = subprocess.run(["pgrep", "-x", "mc_ctrl"], capture_output=True,
                             text=True, timeout=5).stdout.strip().split("\n")[0]
    except Exception:
        pass
    if pid:
        print(f"   mc_ctrl PID = {pid}")
        try:
            with open(f"/proc/{pid}/stat") as f:
                print(f"   狀態 = {f.read().split(')')[-1].split()[0]}  （R/S=執行中, T=已凍結）")
        except Exception:
            pass
    else:
        print("   ⚠️ 找不到 mc_ctrl —— 運控可能沒起來，先確認狗的狀態")

    if os.geteuid() == 0:
        print("   ✅ 目前已是 root，可直接寫入")
    elif shutil.which("sudo"):
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
        if r.returncode == 0:
            print("   ✅ sudo 免密碼可用")
        else:
            print("   ⚠️ sudo 需要密碼（M1 執行時會問，正常）")
    else:
        print("   ❌ 沒有 sudo → 寫不了 joint_cmd，這條路走不通")

    hr("結論")
    print("   全部 ✅ 才進 M1_zero_write.py。任何一項 ❌ 先停下來回報。")
    print("   M0 全程唯讀，沒有對狗做任何改動。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
