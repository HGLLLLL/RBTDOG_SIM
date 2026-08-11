"""gait_export.py —— 把 walk_stable 步態導出成實機可播放的關節軌跡，並做上機前的離線檢驗。

角色：本檔在【開發機】執行（要用 MuJoCo）。狗上執行的是 task6/realbot/L7_gait_shm.py。

管線：
    cpg_walk_d1.make_cpg_step + joint_targets     ← import，不複製
      → 每 20 ms 的 12 軸 MJCF 角
      → calib_map.mjcf12_to_shm
      → 五項離線檢驗
      → gait_walk_stable.npz

⚠️ 本檔【不修改】cpg_walk_d1 的任何步態參數 —— 那支腳本產生了對照影片。
   部署用的 G_C 是本檔自己的 DEPLOY_G_C（見 §為什麼不是 0.12）。

================================================================================
§ 為什麼部署用 G_C=0.110 而不是影片的 0.12
================================================================================
影片版 G_C=0.12 的膝關節在擺動相會折到距 ctrlrange 只剩 0.0114 rad（0.65°）。
而 calib_map 的 offset 誤差量級恰好就在這個尺度上（以 POSE_LIE 對照推導限位，
leg0 abad 超出 0.0089、leg2 超出 0.0138 rad）。也就是說實機的膝很可能會頂到
機構限位，而模擬顯示 clip 0.000%，離線完全看不出來。

頂限位時馬達會持續對機構硬推，靠事後力矩中止來處理是把可預防的問題留到現場，
所以改成事前用餘裕門檻擋掉。掃描結果（--sweep 可重現）：

    G_C     膝餘裕(rad)   (度)    最大角速度    離地量 FL/FR/RL/RR (mm)
    0.100     0.1266     7.25      12.27       50.6  49.9  46.2  45.9  ✓
    0.105     0.0978     5.60      12.88       54.7  54.2  49.4  49.4  ✓
    0.110     0.0690     3.95      13.49       59.5  59.0  53.1  53.4  ✓  ← 採用
    0.115     0.0402     2.30      14.10       66.3  64.7  56.3  56.0  ✗
    0.120     0.0114     0.65      14.72       76.7  77.0  56.6  55.6  ✗  ← 影片版

代價：前腳離地量從 77 掉到 59 mm（−23%）。後腳幾乎不變（56.6 → 53.1）。
副作用是前後更均勻了（比值 1.37 → 1.12）。
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "realbot"))

import calib_map
import cpg_d1
import cpg_walk_d1 as W
import d1_model

JN = ("abad", "hip", "knee")

GAIT = "walk_stable"
DEPLOY_G_C = 0.110          # 見檔頭 §。影片版是 cpg_walk_d1.GAIT_G_C = 0.12
MARGIN_MIN = 0.05           # 限位餘裕門檻(rad) = 2.9°


def shm_limits(m):
    """MJCF ctrlrange → SHM 慣例限位。回傳 {(shm_leg, joint_name): (lo, hi)}。

    ⚠️ 轉換是 shm = sign * mjcf + offset，所以 **sign = -1 時上下界會對調**。
       calib_map 裡 FR/RR 的 hip 與 knee 都是 -1。忘了 sorted() 的話限位檢驗
       會恆為「超限」或恆為「通過」，兩種都是靜默失效。
    """
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = {}
    for mjcf_leg in range(4):
        shm_leg = calib_map.LEG_MJCF2SHM[mjcf_leg]
        for j, jn in enumerate(JN):
            col = mjcf_leg * 3 + j
            s, o = calib_map.CALIB[shm_leg][jn]
            out[(shm_leg, jn)] = tuple(sorted((s * lo[col] + o, s * hi[col] + o)))
    return out
