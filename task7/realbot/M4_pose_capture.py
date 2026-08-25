#!/usr/bin/env python3
"""M4 —— 姿勢擷取與座標換算驗證（唯讀、不需 sudo、不碰任何行程）。

要解決的問題：**`joint_state` 讀到的馬達角，跟控制器／MJCF 的關節角是什麼關係？**

我們有一條從實機設定檔 `zg_wheels-user-parameters.yaml` 推出來的假說：

    馬達角 = side_sign × 控制器角 + offset

佐證是：馬達洩力（控制器角 = 0）時，`joint_cmd` 的 `p_des` 讀出來剛好等於各關節的
`offset`（四組全中）。但這只驗到「控制器角 = 0」這一個點 ——
**斜率（side_sign）完全沒被驗證過。** 要驗就得讓狗擺出一個已知的非零姿勢。

這支做的事：擷取一個姿勢的 16 關節平均角度，然後
  1. 用四種可能的換算式各自反推控制器角
  2. 跟設定檔記載的姿勢比對，看哪一種對得上
  3. 兩個以上的姿勢還可以**直接解出** sign 與 offset，完全不靠假說

⚠️ **這一步做不出來，之後的步態部署就無法把模擬的關節角換算成實機指令。**

用法（在狗上，遙控器先把狗擺到該姿勢並停穩）：
    python3 M4_pose_capture.py stand      # 站好之後跑
    python3 M4_pose_capture.py lie        # 趴下之後跑
    python3 M4_pose_capture.py --analyze  # 有兩個以上姿勢後，直接解 sign/offset
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import shm_io

# ---------------------------------------------------------------- 設定檔的常數
# ⚠️ 設定檔的腿序是 FR, FL, RR, RL；SHM 的腿序是 fl, fr, bl, br。**不同**。
#    這裡先轉成以 SHM 腿名為 key，避免索引對錯。
_CFG_ORDER = ["FR", "FL", "RR", "RL"]
_CFG2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}   # RR=rear-right, RL=rear-left


def _by_leg(vals):
    return {_CFG2SHM[c]: v for c, v in zip(_CFG_ORDER, vals)}


# 取自實機 /opt/export/config/zg_wheels-user-parameters.yaml（與 MATRiX 發布包逐字元相同）
SIGN = {
    "1_hip_roll":   _by_leg([-1., -1., 1., 1.]),
    "2_hip_pitch":  _by_leg([1., -1., 1., -1.]),
    "3_knee_pitch": _by_leg([-1., 1., -1., 1.]),
    "4_foot":       _by_leg([1., -1., 1., -1.]),
}
OFFSET = {
    "1_hip_roll":   _by_leg([0.523, -0.523, -0.523, 0.523]),
    "2_hip_pitch":  _by_leg([-2.443, 2.443, 2.443, -2.443]),
    "3_knee_pitch": _by_leg([-2.803, 2.803, 2.803, -2.803]),
    "4_foot":       _by_leg([0., 0., 0., 0.]),
}

# ---------------------------------------------------------------- 候選姿勢
# 設定檔記載的姿勢（控制器座標系，**設定檔腿序 FR, FL, RR, RL**）。
# ⚠️ 只有 rl_default 存在於**實機**設定檔；stand / liedown 那兩組只出現在
#    MATRiX 模擬版那份，實機檔案裡被刪掉了。
_BASE = {
    "rl_default（實機檔有）": ([0., 0., 0., 0.], [0.8, 0.8, -0.8, -0.8], [-1.5, -1.5, 1.5, 1.5]),
    "stand（僅模擬版有）":    ([0., 0., 0., 0.], [0.6, 0.6, -0.6, -0.6], [-1.2, -1.2, 1.2, 1.2]),
    "liedown（僅模擬版有）":  ([0., 0., 0., 0.], [1.4, 1.4, -1.4, -1.4], [-2.4, -2.4, 2.4, 2.4]),
}

# ★★ 這台狗有**兩種站姿**：後腿往前彎（預設）／後腿往後彎（一般機器狗的樣子）。
#    官方規格書運動性能欄明載「支持膝關節姿態變換」，機上也有 `/robot_remote/knee_mode` topic。
#
#    設定檔記載的姿勢（hip 前後反號）對應的是**後腿往前彎**——這點有兩個佐證：
#      1. hip_default_pos = [0.8, 0.8, −0.8, −0.8] 本身就是前後反號
#      2. 官方 MJCF 正向運動學：前後反號那組四輪 x = ±0.3398 對稱；
#         四腿同號那組後腿整條往後翹（膝 −0.475 vs 輪 −0.317），不對稱
#
#    另一種模式（後腿往後彎）等於把後兩腿翻成與前腿同號，所以這裡自動生出對應版本。
#    ⚠️ 這是推論 —— 實機切到另一模式時到底是不是這組值，**沒驗過**。


def _flip_rear(vals):
    """把設定檔腿序 [FR, FL, RR, RL] 的後兩腿翻成與前腿同號。"""
    return [vals[0], vals[1], -vals[2], -vals[3]]


POSES = {}
for _nm, (_ab, _hp, _kn) in _BASE.items():
    POSES[f"{_nm}｜後腿往前彎(預設)"] = {
        "1_hip_roll": _by_leg(_ab), "2_hip_pitch": _by_leg(_hp), "3_knee_pitch": _by_leg(_kn)}
    POSES[f"{_nm}｜後腿往後彎"] = {
        "1_hip_roll": _by_leg(_ab), "2_hip_pitch": _by_leg(_flip_rear(_hp)),
        "3_knee_pitch": _by_leg(_flip_rear(_kn))}

# 四種可能的換算式：由馬達角反推控制器角
VARIANTS = {
    "V1  馬達 = sign×控制 + off": lambda m, s, o: (m - o) / s,
    "V2  馬達 = sign×(控制 + off)": lambda m, s, o: m / s - o,
    "V3  馬達 = sign×控制 − off": lambda m, s, o: (m + o) / s,
    "V4  馬達 = 控制 + sign×off": lambda m, s, o: m - s * o,
}

LEGS = ["fl", "fr", "bl", "br"]
KINDS = ["1_hip_roll", "2_hip_pitch", "3_knee_pitch"]     # 輪子不做姿勢比對


def out_dir() -> str:
    d = os.path.expanduser("~/m_logs")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- 擷取
def capture(label: str, secs: float, hz: float) -> dict:
    print(f"擷取姿勢「{label}」—— {secs:.1f} 秒 @ {hz:.0f} Hz")
    print("⚠️ 這段期間請不要碰狗，也不要動遙控器。\n")

    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)

    acc = {n: [] for n in shm_io.JOINTS}
    imu_acc: list[list[float]] = []
    period, t0 = 1.0 / hz, time.monotonic()
    with shm_io.Shm("joint_state") as ss, shm_io.Shm("imu_central") as si:
        while time.monotonic() - t0 < secs:
            for r in ss.states():
                acc[r["name"]].append(r["position"])
            imu_acc.append([shm_io._F8.unpack_from(si.mm, 824 + 8 * k)[0] for k in range(10)])
            time.sleep(period)

    n = len(imu_acc)
    mean = {k: sum(v) / len(v) for k, v in acc.items()}
    # ⚠️ 標準差要看 —— 太大代表狗還在晃或伺服在抖，這筆資料不可信
    std = {k: (sum((x - mean[k]) ** 2 for x in v) / len(v)) ** 0.5 for k, v in acc.items()}
    imu = [sum(c) / n for c in zip(*imu_acc)]

    rec = {"label": label, "n": n, "secs": secs,
           "mean": mean, "std": std, "imu": imu,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = os.path.join(out_dir(), f"pose_{label}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"已存 {p}（{n} 筆取樣）\n")
    return rec


def show_capture(rec: dict) -> None:
    mean, std = rec["mean"], rec["std"]
    worst = max(std.values())
    print(f"{'關節':16s} {'實測馬達角':>11s} {'標準差':>9s}")
    for nm in shm_io.JOINTS:
        flag = "  ⚠️晃動" if std[nm] > 0.01 else ""
        print(f"{nm:16s} {mean[nm]:11.4f} {std[nm]:9.5f}{flag}")
    print(f"\n最大標準差 {worst:.5f} rad", end="")
    print("　✅ 夠穩，資料可信" if worst < 0.01
          else "　⚠️ 偏大，狗可能還在晃 —— 等它停穩再擷取一次")

    q = rec["imu"][6:10]                      # quat_x, quat_y, quat_z, quat_w
    a = rec["imu"][0:3]
    print(f"\nIMU 加速度 {a[0]:+.4f} {a[1]:+.4f} {a[2]:+.4f}　"
          f"|a| = {math.sqrt(sum(x*x for x in a)):.4f} m/s²")
    for name, (x, y, z, w) in (("xyzw", q), ("wxyz", (q[1], q[2], q[3], q[0]))):
        roll = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2*(w*y - z*x)))))
        print(f"  以 {name} 解：roll {roll:+7.2f}°　pitch {pitch:+7.2f}°")
    print("  → 狗站在平地上時，正確的那個解 roll/pitch 應該都接近 0")


# ---------------------------------------------------------------- 換算式比對
def controller_angles(mean: dict, inv) -> dict:
    out = {}
    for kind in KINDS:
        for leg in LEGS:
            nm = leg + kind
            out[nm] = inv(mean[nm], SIGN[kind][leg], OFFSET[kind][leg])
    return out


def compare(rec: dict) -> None:
    print("\n" + "=" * 70)
    print("換算式比對：把實測馬達角反推成控制器角，看哪一種對得上文件姿勢")
    print("=" * 70)

    best = []
    for vname, inv in VARIANTS.items():
        ctrl = controller_angles(rec["mean"], inv)
        for pname, pose in POSES.items():
            res = [ctrl[leg + k] - pose[k][leg] for k in KINDS for leg in LEGS]
            rms = (sum(r * r for r in res) / len(res)) ** 0.5
            best.append((rms, vname, pname, ctrl))
    best.sort(key=lambda x: x[0])

    print(f"\n{'換算式':28s} {'候選姿勢':32s} {'RMS 殘差(rad)':>13s}")
    for rms, v, p, _ in best[:8]:
        mark = "  ★" if rms < 0.15 else ""
        print(f"{v:28s} {p:32s} {rms:13.4f}{mark}")

    rms, vname, pname, ctrl = best[0]
    print(f"\n最接近：{vname}　＋　{pname}　RMS {rms:.4f} rad（{math.degrees(rms):.2f}°）")
    if rms < 0.15:
        print("★ 換算式與姿勢對上了。逐項對照：")
    else:
        print("⚠️ 沒有任何組合對得上（RMS 都太大）。")
        print("   可能是：遙控器的站姿不是文件裡任何一組、或換算式根本不是這四種。")
        print("   → 請再擷取第二個姿勢，用 --analyze 直接解 sign/offset，不要靠假說。")
    print(f"\n{'關節':16s} {'反推控制器角':>13s} {'文件值':>9s} {'差':>9s}")
    for kind in KINDS:
        for leg in LEGS:
            nm = leg + kind
            d = ctrl[nm] - POSES[pname][kind][leg]
            print(f"{nm:16s} {ctrl[nm]:13.4f} {POSES[pname][kind][leg]:9.2f} {d:+9.4f}")
        print()


# ---------------------------------------------------------------- 兩姿勢直接解
def analyze() -> None:
    files = sorted(glob.glob(os.path.join(out_dir(), "pose_*.json")))
    if len(files) < 2:
        print(f"❌ 只找到 {len(files)} 個姿勢檔，至少要兩個才能解。")
        print("   例如：先站好跑 `M4_pose_capture.py stand`，再趴下跑 `M4_pose_capture.py lie`。")
        return
    recs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            recs.append(json.load(fh))
    print(f"找到 {len(recs)} 個姿勢：{', '.join(r['label'] for r in recs)}\n")
    print("=" * 70)
    print("不靠假說，直接由兩個姿勢解每個關節的 sign 與 offset")
    print("=" * 70)
    print("解法：若 馬達 = s×控制 + off，兩姿勢相減得 s = Δ馬達 / Δ控制")
    print("⚠️ 這需要**知道兩個姿勢的控制器角**。以下用文件姿勢當假定值，")
    print("   若遙控器的實際姿勢不是那組，解出來的數字就不對 —— 看 s 是不是接近 ±1 來判斷。\n")

    a, b = recs[0], recs[1]
    print(f"姿勢 A = {a['label']}　姿勢 B = {b['label']}")
    print(f"\n{'關節':16s} {'A 馬達角':>10s} {'B 馬達角':>10s} {'Δ馬達':>9s}"
          f" {'解出的 s':>10s} {'文件 s':>8s}")
    for kind in KINDS:
        for leg in LEGS:
            nm = leg + kind
            dm = a["mean"][nm] - b["mean"][nm]
            print(f"{nm:16s} {a['mean'][nm]:10.4f} {b['mean'][nm]:10.4f} {dm:+9.4f}"
                  f" {'需知道Δ控制':>10s} {SIGN[kind][leg]:+8.0f}")
        print()
    print("⚠️ 要把「解出的 s」算出來，得先確定這兩個姿勢在控制器座標系的角度差。")
    print("   最乾淨的做法是**手動擺兩個差異明確的姿勢**（例如站立 vs 趴下），")
    print("   並記下遙控器上顯示／文件記載的目標值，再回報給我一起算。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", help="姿勢名稱，例如 stand / lie")
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--analyze", action="store_true", help="用已擷取的多個姿勢直接解 sign/offset")
    a = ap.parse_args()

    logp = shm_io.start_log("M4")
    print("M4 —— 姿勢擷取與座標換算驗證（唯讀）\n")

    if a.analyze:
        analyze()
    elif a.label:
        rec = capture(a.label, a.secs, a.hz)
        show_capture(rec)
        compare(rec)
    else:
        ap.print_help()
        return 1

    print(f"\n📄 完整輸出已存到 {logp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
