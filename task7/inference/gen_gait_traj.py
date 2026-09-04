#!/usr/bin/env python3
"""產生實機用的步態軌跡檔（M9 播放用）。

════════════════════════════════════════════════════════════════════
為什麼是「離線產生 + 狗上播放」而不是「狗上跑 CPG」
════════════════════════════════════════════════════════════════════

狗上沒有 numpy，而 `cpg_max` / `leg_kin` 都是 numpy 寫的。兩條路：

  A. 把 CPG + IK 移植成純標準函式庫 → 移植出錯的話「步態行為不對」很難查
  B. **本機用已驗證的程式算好，狗上只負責播放**

選 B。★ 關鍵理由：**基準步態本來就是開迴路的** —— 它不看回授、
只播放固定軌跡，所以「不能閉迴路」不是損失。
換來的是**上機前可以逐幀檢查要送出去的每一個關節角**，
還可以把同一個檔案先在 MuJoCo 播一遍確認行為。

（之後做 RL 閉迴路時再處理狗上推論。記憶：狗上無 torch/onnx，建議純 numpy。）

════════════════════════════════════════════════════════════════════
★ 腿序陷阱
════════════════════════════════════════════════════════════════════

`max_model.LEGS = (FR, FL, RR, RL)`，SHM 是 `(fl, fr, bl, br)`。
**輸出檔一律用 SHM 名稱，而且把名稱列表寫進檔案裡**，
讓播放器按名稱對應而不是按索引 —— 這是本專案反覆出事的地方。

用法：
    # 原地踏步（d_step=0），20 秒，kp=250
    python3 task7/inference/gen_gait_traj.py --march --secs 20 --out march_w1.0.json

    # 直線走（要等 CPG 線重掃 kp=250 的基準）
    python3 task7/inference/gen_gait_traj.py --d-step 0.116 --secs 20 --out walk.json

⚠️ 用 rbtdog 環境跑（要 numpy）：
    /home/huang/miniforge3/envs/rbtdog/bin/python task7/inference/gen_gait_traj.py ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import cpg_max                      # noqa: E402
import coord                        # noqa: E402
import gait_baseline                # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402

# max_model 的腿序 → SHM 腿名
MM2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
# 輸出的關節順序（SHM 名），與 max_model 的 12 維展開一一對應
JOINTS = [MM2SHM[l] + k for l in mm.LEGS for k in coord.LEG_KINDS]

SMOOTHSTEP_VPEAK = np.pi / 2        # 與 M7/M8 同一個係數（餘弦插值）


def blend(u: float) -> float:
    """餘弦淡入淡出，與 M5/M7/M8 的 smoothstep 同一條曲線。"""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def main() -> int:
    B = gait_baseline.BASELINE
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="產生 M9 播放用的步態軌跡檔")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--secs", type=float, default=20.0, help="步態本體的秒數")
    ap.add_argument("--march", action="store_true",
                    help="★ 原地踏步：把 d_step 設為 0（不前進）")
    ap.add_argument("--kp", type=float, default=250.0,
                    help="★ 腿關節增益。250 = 原廠站立、我們 2026-08-27 實機驗證 13 趟")
    ap.add_argument("--kd", type=float, default=5.0)
    ap.add_argument("--kp-abad", type=float, default=60.0, dest="kp_abad",
                    help="★ 步態段 ABAD 的 kp。原廠與模擬都是 [60,120,120]；"
                         "會寫進檔案頂層由 M9 比對")
    ap.add_argument("--wheel-kd", type=float, default=0.5, dest="wheel_kd",
                    help="輪子純阻尼。⚠️ kp 必須是 0 —— 設定檔的 FSM_RL_Wheel_Kp=60 "
                         "是配「每步重給目標角」的 RL，開迴路套上去會偏航失控 +39°/12s")
    ap.add_argument("--omega", type=float, default=B["omega"])
    ap.add_argument("--duty", type=float, default=B["duty"])
    ap.add_argument("--d-step", type=float, default=B["d_step"], dest="d_step")
    ap.add_argument("--x-off", type=float, default=B["x_off"], dest="x_off")
    ap.add_argument("--seq", choices=("ds", "ls", "trot"), default="ds",
                    help="★★ 相位序列。ds = 舊的 diagonal sequence（2026-09-03 前"
                         "註解一直誤標成 lateral）；ls = 文獻上靜態穩定裕度最好的"
                         " lateral sequence。⚠️ 會寫進軌跡檔並由 M9 比對")
    ap.add_argument("--x-d", type=float, default=0.0, dest="x_d",
                    help="前後差動足端偏移（前腿 +x_d、後腿 −x_d）＝軸距量")
    ap.add_argument("--sway-x", type=float, default=0.0, dest="sway_x",
                    help="★ body sway 縱向幅度 m。⚠️ 必須配 --sway-lead-x，留 0 會變擾動")
    ap.add_argument("--sway-y", type=float, default=0.0, dest="sway_y",
                    help="★ body sway 橫向幅度 m")
    ap.add_argument("--sway-lead-x", type=float, default=0.0, dest="sway_lead_x",
                    help="★ 縱向相位提前（週期比例）。縱向是二倍頻，與橫向**不能共用**")
    ap.add_argument("--sway-lead-y", type=float, default=0.0, dest="sway_lead_y",
                    help="★ 橫向相位提前")
    ap.add_argument("--g-c", type=float, default=B["g_c"], dest="g_c")
    ap.add_argument("--z-sag", type=float, default=None, dest="z_sag",
                    help="★ 預設 = **0.036 × 250/kp**（實機錨點，不是模擬的 STATIC_SAG）。"
                         "M8 S3 實測 kp250→36mm、kp120→72mm，兩點都對得上")
    ap.add_argument("--ramp", type=float, default=3.0,
                    help="從站姿淡入步態的秒數（也用於淡出）")
    ap.add_argument("--vcmd-max", type=float, default=14.0, dest="vcmd_max",
                    help="★ 命令速度上限 rad/s。**基準步態（ω=1.4）實測 10.9** ——"
                         "所以這裡比 M7/M8 的 2.0 寬很多。馬達 190 RPM = 19.9 rad/s。"
                         "⚠️ M9 的 --vmax（量測速度保護）必須高於這個，否則會誤中止")
    a = ap.parse_args()

    if a.march:
        a.d_step = 0.0

    # ★★ z_sag 的錨點換成**實機量測**，不是模擬值。
    #   2026-08-27 M8 S3 實測擺動離地損失：kp=250 → 36 mm、kp=120 → 72 mm，
    #   正比於 1/kp。`max_model.STATIC_SAG = 0.0325` 是**模擬**值，
    #   而模擬在 z 方向系統性高估順從性 1.8–2.1 倍 —— 用它會補太少。
    #   驗算：0.036 × 250/120 = 0.075，對上實機的 72 mm ✅（兩點都錨住）
    if a.z_sag is None:
        a.z_sag = 0.036 * 250.0 / a.kp

    knee_sign = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    phase = {"ds": cpg_max.PHASE_WALK, "ls": cpg_max.PHASE_WALK_LS,
             "trot": cpg_max.PHASE_TROT}[a.seq]
    step = cpg_max.make_cpg_step(phase)
    c = cpg_max.cpg_init(phase)
    # 逐腿 x_off（--x-d 0 時等於四腿共用）。站姿與步態必須用同一個。
    x_off_legs = cpg_max.x_off_split(a.x_off, a.x_d)

    mux = np.full(4, B["mu_x"])
    muy = np.full(4, B["mu_y"])
    om = np.full(4, a.omega)
    dt = mm.CTRL_DT

    q_stand = cpg_max.stand_targets(knee_sign, f0, x_off_legs)

    n_gait = int(round(a.secs / dt))
    n_ramp = int(round(a.ramp / dt))
    n_tot = n_ramp + n_gait + n_ramp

    Q = np.zeros((n_tot, 12))
    n_clamp_tot = 0
    for i in range(n_tot):
        sway = None
        if a.sway_x or a.sway_y:
            sway = cpg_max.body_sway(cpg_max.gait_phase(c["theta"], phase),
                                     a.sway_x, a.sway_y,
                                     a.sway_lead_x, a.sway_lead_y)
        q_g, ncl = cpg_max.joint_targets(c, f0, x_off_legs, a.g_c, a.d_step,
                                         B["d_step_y"], a.duty, knee_sign,
                                         a.z_sag, sway)
        n_clamp_tot += ncl
        if i < n_ramp:
            s = blend(i / max(n_ramp, 1))
        elif i < n_ramp + n_gait:
            s = 1.0
        else:
            s = blend(1.0 - (i - n_ramp - n_gait) / max(n_ramp, 1))
        Q[i] = (1.0 - s) * q_stand + s * q_g
        c = step(c, mux, muy, om, dt)

    # ------------------------------------------------------------- 檢查
    order = [mm.LEGS[i] for i in cpg_max.swing_order(phase)]
    print(f"步態：{'原地踏步' if a.march else '前進'}　序列 {a.seq.upper()}"
          f"（擺動順序 {' → '.join(order)}）")
    print(f"　　　duty {a.duty} ω {a.omega} d_step {a.d_step} "
          f"x_off {a.x_off} x_d {a.x_d} g_c {a.g_c} z_sag {a.z_sag:.4f}")
    print(f"　　　sway ({a.sway_x * 1000:.0f}, {a.sway_y * 1000:.0f}) mm  "
          f"lead ({a.sway_lead_x:.2f}, {a.sway_lead_y:.2f})")
    print(f"增益：腿 kp {a.kp} kd {a.kd}　輪 kp 0 kd {a.wheel_kd}（純阻尼）")
    print(f"長度：淡入 {a.ramp}s + 步態 {a.secs}s + 淡出 {a.ramp}s "
          f"= {n_tot * dt:.1f}s，{n_tot} 幀 @ {1/dt:.0f} Hz")

    if n_clamp_tot:
        print(f"\n❌ IK 縮限 {n_clamp_tot} 次 —— 靜默的縮限會讓步態「突然變鈍」而查不出原因。")
        return 1
    print("✅ 全程無 IK 縮限")

    bad = []
    for i in range(n_tot):
        for j, name in enumerate(JOINTS):
            msg = coord.check_limit(name, Q[i, j], 0.03)
            if msg:
                bad.append(f"幀 {i} / {name}: {msg}")
    if bad:
        print(f"\n❌ {len(bad)} 個關節角超出機構限位，前三個：")
        for b in bad[:3]:
            print("   " + b)
        return 1
    print("✅ 全程在機構限位內")

    # 命令速度（相鄰幀差，換算成峰值）
    dq = np.abs(np.diff(Q, axis=0)) / dt
    v_peak = dq.max()
    j_peak = JOINTS[int(np.unravel_index(dq.argmax(), dq.shape)[1])]
    print(f"\n最大命令速度 {v_peak:.2f} rad/s（{j_peak}）"
          f"　上限 {a.vcmd_max}")
    if v_peak > a.vcmd_max:
        # ⚠️ 舊訊息寫「或加大 --duty」是**反的**：duty 越大擺動相時間越短
        #    （(1−duty)/ω），同樣的抬腿量要更快走完 → 命令速度**更高**。
        print(f"❌ 超過 --vcmd-max。命令速度 ∝ (g_c + z_sag)·ω/(1−duty)：")
        print(f"   降 --omega（現在 {a.omega}）→ 線性下降")
        print(f"   降 --g-c（現在 {a.g_c}）→ 但 g_c 就是實際離地量，別降到失去離地")
        print(f"   ⚠️ **不要加大 --duty**，那會讓擺動相更短、命令速度更高")
        print(f"   ⚠️ z_sag 現在是 {a.z_sag:.4f}（= 0.036×250/kp，kp={a.kp:g}）——"
              f"kp 越低補償越大、命令幅度越大")
        return 1
    print(f"✅ 在上限內")
    # ★ M9 另外要求 `--vmax >= 命令速度 × 1.2`（量測速度保護不能太貼命令速度，
    #   否則正常擺動就會誤中止）。這個數字在本機就算得出來，別讓它到現場才擋人。
    need_vmax = v_peak * 1.2
    print(f"\n★ 上機時 M9 需要 --vmax ≥ {need_vmax:.2f}"
          f"（= 命令速度 × 1.2）。M9 預設 16.0 —— "
          f"{'夠' if need_vmax <= 16.0 else f'**不夠，要帶 --vmax {math.ceil(need_vmax)}**'}")
    print(f"  參考：馬達 190 RPM = 19.9 rad/s，"
          f"{math.ceil(need_vmax) if need_vmax > 16 else 16} 是它的"
          f"{(math.ceil(need_vmax) if need_vmax > 16 else 16) / 19.9 * 100:.0f}%")

    # 每個關節的行程
    print(f"\n{'關節':16s} {'最小':>9s} {'最大':>9s} {'行程':>9s} {'最大速度':>10s}")
    for j, name in enumerate(JOINTS):
        print(f"{name:16s} {Q[:, j].min():+9.4f} {Q[:, j].max():+9.4f} "
              f"{Q[:, j].max() - Q[:, j].min():9.4f} {dq[:, j].max():8.2f} rad/s")

    out = {
        "schema": "gait_traj/1",
        "generated": __file__,
        "joints": JOINTS,          # ★ 播放器按名稱對應，不按索引
        "dt": dt,
        "n": n_tot,
        "kp": a.kp, "kd": a.kd,
        "wheel_kp": 0.0, "wheel_kd": a.wheel_kd,
        # ★ 這些**必須放頂層** —— M9 的「說兩次」比對讀的是頂層鍵。
        #   放進 params 巢狀裡的話比對抓不到，等於防呆有缺口。
        "seq": a.seq, "kp_abad": a.kp_abad,
        "x_off": a.x_off, "x_d": a.x_d, "g_c": a.g_c,
        "z_sag": a.z_sag,
        "sway_x": a.sway_x, "sway_y": a.sway_y,
        "sway_lead_x": a.sway_lead_x, "sway_lead_y": a.sway_lead_y,
        "params": {k: (v if not isinstance(v, Path) else str(v))
                   for k, v in vars(a).items()},
        "baseline_ref": dict(gait_baseline.BASELINE),
        "q_stand": [round(float(x), 6) for x in q_stand],
        "q": [[round(float(x), 6) for x in row] for row in Q],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                     encoding="utf-8")
    print(f"\n📦 {a.out}　{a.out.stat().st_size / 1024:.0f} KB")
    print("\n下一步：先在 MuJoCo 播一遍驗證")
    print(f"    python3 task7/inference/play_gait_traj.py {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
