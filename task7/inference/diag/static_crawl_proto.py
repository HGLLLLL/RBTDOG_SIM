#!/usr/bin/env python3
"""靜態間歇 crawl 原型（2026-09-03）—— 回應「model-based 應該要很穩」。

═══════════════════════════════════════════════════════════════════
為什麼現在的 CPG 步態會搖 ±7~11°（實機資料已釘死機制）
═══════════════════════════════════════════════════════════════════
15:53 那趟：機身傾斜與「哪條腿在擺動」完全同步（fr 擺 → roll +7.3°、
br 擺 → pitch −5.9°），量級 = 三腳承重 × kp120 柔度(72mm) 的預測值。
**是承重重分配 × 位置伺服柔度的確定性效應，不是動力學混亂。**
CPG 把「移質心、抬腿、推進」全部同時連續做，柔度效應沒人補償。

═══════════════════════════════════════════════════════════════════
本原型：文獻的標準靜態 crawl 結構 ＋ 兩個模型前饋
═══════════════════════════════════════════════════════════════════
1. **分段**（intermittent crawl）：
       SHIFT(四腳著地,質心移到對側) → SWING 後腿 → SWING 前腿 → 換邊…
   swing 時機身**完全不動** —— 傾覆力矩最小化。
2. **逐腿承重補償**（本原型的核心新東西）：
   每一瞬間按支撐幾何算各腿承重比 w_i（質心在支撐三角形的重心座標），
   命令 z 加深 `sag_unit × (4·w_i − 1)` —— 讓每條腿**沉完剛好等高** → 機身水平。
   sag_unit = 0.036×250/kp（M8 實機錨點）。純模型、零感測。
3. **對稱站姿**：x_off=0。質心位置由 SHIFT 管，不再需要不對稱 trim ——
   這正好回到使用者要的「前後對稱」。

用法：
    PY=... ; $PY task7/inference/diag/static_crawl_proto.py --secs 30 --video
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "task7" / "inference"))

import cpg_max                      # noqa: E402
import cpg_walk_max as cw           # noqa: E402
import leg_kin                      # noqa: E402
import max_model as mm              # noqa: E402

LEGS = list(mm.LEGS)                # FR, FL, RR, RL


def smoothstep(u: float) -> float:
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return u * u * (3 - 2 * u)


def load_shares(feet_xy: dict, com_xy) -> dict:
    """各支撐腿的承重比。3 腿＝重心座標（唯一解）；4 腿＝兩個三角形的平均。

    ⚠️ 支撐腿不足 3 或質心在支撐形外時，重心座標會出負值 —— 夾到 [0,1] 再歸一。
    那已經是「快翻了」的狀態，補償只求不發散。
    """
    names = list(feet_xy)
    P = np.array([feet_xy[n] for n in names], dtype=float)
    c = np.asarray(com_xy, dtype=float)

    def bary(tri):
        a, b, d = P[tri[0]], P[tri[1]], P[tri[2]]
        M = np.array([[a[0], b[0], d[0]], [a[1], b[1], d[1]], [1, 1, 1]])
        try:
            w = np.linalg.solve(M, np.array([c[0], c[1], 1.0]))
        except np.linalg.LinAlgError:
            return None
        return w

    if len(names) == 3:
        w = bary((0, 1, 2))
        if w is None:
            w = np.full(3, 1 / 3)
        w = np.clip(w, 0.0, 1.0)
        w = w / max(w.sum(), 1e-9)
        return {n: float(v) for n, v in zip(names, w)}
    # 4 腿：對角拆兩個三角形取平均（超靜定的簡單一致解）
    out = {n: 0.0 for n in names}
    for tri in ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)):
        w = bary(tri)
        if w is None:
            continue
        w = np.clip(w, 0.0, 1.0)
        w = w / max(w.sum(), 1e-9)
        for k, i in enumerate(tri):
            out[names[i]] += float(w[k]) / 4.0
    tot = max(sum(out.values()), 1e-9)
    return {n: v / tot for n, v in out.items()}


class StaticCrawl:
    """分段間歇 crawl 的目標產生器（純運動學，不做 I/O）。

    每個週期八段（LS 順序 RL→FL→RR→FR）：
        SHIFT_R(質心右移+前進) → SW_RL → SW_FL → SHIFT_L(左移+前進) → SW_RR → SW_FR
    """

    def __init__(self, kp_knee=120.0, stride=0.16, lift=0.06,
                 t_shift=0.5, t_swing=0.4, shift_y=0.055,
                 comp_gain=1.0, com_xy=(0.0, 0.0)):
        self.sag_unit = 0.036 * 250.0 / kp_knee     # 每 1/4 體重的撓度（M8 錨點）
        self.stride, self.lift = stride, lift
        self.t_shift, self.t_swing = t_shift, t_swing
        self.shift_y = shift_y
        self.comp_gain = comp_gain
        self.com = np.asarray(com_xy, dtype=float)
        f0 = leg_kin.home_foot(mm.HOME)
        self.ks = leg_kin.knee_sign_of(mm.HOME)
        # 足端狀態（機身座標）。x 對稱站姿：x_off = 0。
        self.feet = {L: f0[k].copy() for k, L in enumerate(LEGS)}
        self.z0 = {L: self.feet[L][2] for L in LEGS}
        # 段落表：(kind, leg_or_dir, duration)
        self.segs = [("shift", (+self.stride / 2, -self.shift_y), t_shift),
                     ("swing", "RL", t_swing), ("swing", "FL", t_swing),
                     ("shift", (+self.stride / 2, +2 * self.shift_y), t_shift),
                     ("swing", "RR", t_swing), ("swing", "FR", t_swing),
                     ("rebal", (0.0, -self.shift_y), t_shift * 0.6)]
        self.T = sum(s[2] for s in self.segs)
        self.i = 0
        self.t_seg = 0.0
        self.seg_start = {L: self.feet[L].copy() for L in LEGS}
        self._snap()

    def _snap(self):
        self.seg_start = {L: self.feet[L].copy() for L in LEGS}

    def targets(self, dt: float):
        """前進 dt，回傳 12 關節角。內部推進段落狀態機。"""
        kind, arg, dur = self.segs[self.i]
        self.t_seg += dt
        u = smoothstep(min(self.t_seg / dur, 1.0))

        cur = {L: self.seg_start[L].copy() for L in LEGS}
        swing_leg = None
        if kind in ("shift", "rebal"):
            dx, dy = arg
            for L in LEGS:                 # 機身動 = 全部足端反向動
                cur[L][0] = self.seg_start[L][0] - dx * u
                cur[L][1] = self.seg_start[L][1] - dy * u
        else:
            swing_leg = arg
            s = self.seg_start[swing_leg]
            cur[swing_leg] = s.copy()
            cur[swing_leg][0] = s[0] + self.stride * u
            cur[swing_leg][2] = s[2] + self.lift * math.sin(math.pi * min(self.t_seg / dur, 1.0))

        # ---- 逐腿承重補償（本原型的核心）----
        # ★ 承重轉移必須**連續**：腿是在抬起過程中逐漸卸載的，不是瞬間。
        #   第一版在段落邊界瞬切支撐集 → 補償 z 跳階 20~30mm → 機身被踹 →
        #   comp=1 反而比 comp=0 更搖。這裡用 λ（擺動腿的卸載程度）在
        #   「4 腿解」與「3 腿解」之間插值，起飛前 25% 卸載、落地前 25% 回載。
        lam = 0.0
        if swing_leg is not None:
            uu = min(self.t_seg / dur, 1.0)
            if uu < 0.25:
                lam = uu / 0.25
            elif uu > 0.75:
                lam = (1.0 - uu) / 0.25
            else:
                lam = 1.0
        feet4 = {L: cur[L][:2] for L in LEGS}
        w4 = load_shares(feet4, self.com)
        if swing_leg is not None:
            w3 = load_shares({L: cur[L][:2] for L in LEGS if L != swing_leg}, self.com)
            w = {L: (1 - lam) * w4[L] + lam * w3.get(L, 0.0) for L in LEGS}
        else:
            w = w4
        q = np.zeros((4, 3))
        nc = 0
        for k, L in enumerate(LEGS):
            p = cur[L].copy()
            if L == swing_leg:
                # 擺動腿：抬升補償也跟著 λ 連續進場（卸載多少補多少）
                p[2] += self.sag_unit * lam
                extra = self.comp_gain * self.sag_unit * (4.0 * w.get(L, 0.25) - 1.0)
                p[2] -= max(-self.sag_unit, min(self.sag_unit, extra)) * (1 - lam)
            else:
                # 站姿腿：命令加深「超額承重 × 單位撓度」→ 沉完剛好等高
                extra = self.comp_gain * self.sag_unit * (4.0 * w.get(L, 0.25) - 1.0)
                # 夾住：超過 ±sag_unit 就是 w 模型出界（快翻了），再深也沒意義且會打 IK 邊界
                extra = max(-self.sag_unit, min(self.sag_unit, extra))
                p[2] -= extra
            qk, cl = leg_kin.ik_ex(k, p, self.ks[k])
            q[k] = qk
            nc += int(cl)

        # 段落推進
        if self.t_seg >= dur:
            # 把「執行完的名目位置」寫回 feet（不含補償）
            for L in LEGS:
                self.feet[L] = cur[L]
            self.i = (self.i + 1) % len(self.segs)
            self.t_seg = 0.0
            self._snap()
        return q.reshape(12), nc, (kind if swing_leg is None else f"SW_{swing_leg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--kp", type=float, default=250.0,
                    help="膝/髖 kp。★ 靜態 crawl 是準靜態情境＝M8 驗證 kp250 的地盤"
                         "（sag 36mm、觸地速度低衝擊小）；kp120 的 sag 75mm 會讓補償打 IK 邊界")
    ap.add_argument("--stride", type=float, default=0.16)
    ap.add_argument("--lift", type=float, default=0.06)
    ap.add_argument("--t-shift", type=float, default=0.5, dest="t_shift")
    ap.add_argument("--t-swing", type=float, default=0.4, dest="t_swing")
    ap.add_argument("--shift-y", type=float, default=0.055, dest="shift_y")
    ap.add_argument("--comp", type=float, default=0.0,
                    help="逐腿承重補償增益。★ 預設 0 —— 靜態實驗證實這台會自我調平"
                         "（55mm 橫移只自然傾 1.6°，M7 的輪子卸載機制），"
                         "補償反而推到 5.1°。留著參數是為了記錄這個否定結果")
    ap.add_argument("--kd", type=float, default=5.0, dest="kd",
                    help="腿 kd。★ 5.0 = M7 在 kp250 驗證過的值；kd=1 阻尼比太低，"
                         "每個段落切換都會振鈴")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    kp3 = [min(a.kp * 0.5, 125.0), a.kp, a.kp]   # ABAD 照原廠比例但不超過 125
    r = cw.Robot(kd_wheel=0.5, kp3=kp3, kd3=[a.kd] * 3)
    g = StaticCrawl(kp_knee=a.kp, stride=a.stride, lift=a.lift,
                    t_shift=a.t_shift, t_swing=a.t_swing, shift_y=a.shift_y,
                    comp_gain=a.comp)
    q0, _, _ = g.targets(0.0)
    r.reset_standing(q0, mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(1.5 / mm.CTRL_DT)):
        r.step(q0, "damp")
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    ren = cam = None
    frames = []
    if a.video:
        import mujoco
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 900, 540
        ren = mujoco.Renderer(r.m, 540, 900)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)

    d = r.d
    n = int(a.secs / mm.CTRL_DT)
    rolls, pitchs, hs = [], [], []
    n_clamp = 0
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw0 = cpg_max.yaw_deg(d.qpos[3:7])
    tau_peak = 0.0
    for i in range(n):
        q, nc, ph = g.targets(mm.CTRL_DT)
        n_clamp += nc
        r.step(q, "damp")
        grav = cpg_max.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        pitchs.append(math.degrees(math.asin(max(-1, min(1, -grav[0])))))
        rolls.append(math.degrees(math.asin(max(-1, min(1, grav[1])))))
        hs.append(float(d.qpos[2]))
        tau_peak = max(tau_peak, float(np.abs(d.qfrc_actuator[mm.LEG_ACT_IDX] if hasattr(mm, 'LEG_ACT_IDX') else 0).max()) if i % 10 == 0 else tau_peak)
        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.30]
            cam.distance, cam.elevation, cam.azimuth = 2.0, -10, 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    rolls, pitchs, hs = map(np.array, (rolls, pitchs, hs))
    dist = float(d.qpos[0]) - x0
    print(f"[靜態crawl] kp={a.kp:g} stride={a.stride} shift_y={a.shift_y * 1000:.0f}mm "
          f"comp={a.comp:g}  週期 {g.T:.1f}s")
    print(f"  前進 {dist:+.2f} m（{dist / a.secs:.3f} m/s）  側偏 {float(d.qpos[1]) - y0:+.2f} m  "
          f"偏航 {cpg_max.yaw_deg(d.qpos[3:7]) - yaw0:+.1f}°")
    print(f"  ★ roll 全距 {rolls.max() - rolls.min():.2f}°   pitch 全距 {pitchs.max() - pitchs.min():.2f}°   "
          f"機身高變動 {(hs.max() - hs.min()) * 1000:.1f} mm")
    print(f"  IK 縮限 {n_clamp}  跌倒={'是' if float(d.qpos[2]) < 0.25 else '否'}")

    if frames:
        import imageio.v2 as iio
        out = a.out or "task7/outputs/static_crawl_proto.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264")
        print(f"  [影片] {out}")


if __name__ == "__main__":
    main()
