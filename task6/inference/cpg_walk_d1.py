"""D1 EDU 輪足版（ZSL-1w）的**可用開迴路步態** —— 不含 RL，純 CPG。

提供兩種步態：

  trot         兩拍小跑。對角腿同時動，任一時刻兩腳著地。快、俐落。
  walk         四拍走路，佔空比 0.75。四腿依序動 RL→FL→RR→FR，任一時刻約三腳著地。
               慢、從容，低步頻長步幅時唯一能用的步態（見下方 §2）。四腿離地最均勻。
  walk_stable  同上但佔空比 0.80。**垂直彈跳最小的一組**（18.4 vs walk 的 30.1 mm）、
               支撐腳最多（3.15）。代價是前後離地量差到 1.37 倍（walk 只有 1.07 倍）。
               要平穩選這個，要四腿均勻選 walk。

用法：
  conda run --no-capture-output -n rbtdog python task6/inference/cpg_walk_d1.py --video
  conda run --no-capture-output -n rbtdog python task6/inference/cpg_walk_d1.py --gait walk --video
  conda run --no-capture-output -n rbtdog python task6/inference/cpg_walk_d1.py --gait trot --omega 2.0 --mu-x 1.65

與 `cpg_openloop_d1.py` 的差別：那支是關卡 3 的驗收腳本，只驗證 MJCF/IK/伺服接得起來，
步態本身不能看（後腳實測只抬 6.3~16.9 mm，指令是 80 mm）。本檔是調過參數、四條腿都真的
抬起來的步態。兩支都保留：前者驗模型正確性，後者驗步態品質。

================================================================================
§1 三個共通的關鍵修正（2026-08-11，每項都有實測依據）
================================================================================

1. `x_off` —— 足端前後基準偏移。

   靜態量測：全機質心比四腳支撐中心**偏後 17 mm**，後腳承重 110.8 N、前腳 90.9 N
   （差 22%），機身站著就抬頭 +1.21°。走起來這個抬頭會墊高後腳的有效離地需求。
   把足端整體往後移即可配平。**這個值隨步態與步頻而變，不是常數**——
   trot 高頻用 −40 mm、walk 低頻要 −55 mm。

2. `mu_y = 1.5` —— 橫向擺動歸零。

   `dy = D_STEP_Y * fy * cos θ`，fy 由 mu_y 決定，1.5 → fy = 0。
   原本與前後共用 1.8（fy=0.6）會產生 ±54 mm 橫擺——直線走路不需要，而且是側偏與
   左右不對稱的主因。實測：側偏 −2.45 → +0.73 m，左右腿離地差距 2.0 倍 → 1.03 倍。

3. `g_c = 0.12` —— 抬腳指令。前兩項修好後才有本錢加大而不破壞推進。
   ⚠️ 不要調到 0.14：實測會讓 IK 解出的角度超出 ctrlrange，5.4% 的指令被靜默 clip。

⚠️ 本檔**刻意不修改 `d1_model` 的任何常數**。`d1_model.G_C = 0.08`、`D_STEP`、`MU_*`、
   `PHASE_OFFSET` 是 RL 訓練與推論的共用契約，改了會讓既有權重全部作廢。
   walk 步態需要非 trot 的相位，因此本檔自帶 `cpg_step`（見 §3）。

================================================================================
§2 為什麼低步頻一定要配 walk 步態
================================================================================

想「降步頻、拉大步幅」走得從容一點，用 trot 做不到。實測（trot、μx=1.8、x_off=−40）：

    ω          0.8    1.2    1.6    2.0    2.4
    前腳離地    23     18     28     71     80    mm
    垂直彈跳   43.9   39.4   30.0   15.7   18.9  mm

**彈跳在 ω≈2.0 最小，往下降反而一路變大**——兩腳支撐的空檔身體會下沉，步頻越低、
下沉時間越長。在這台機器上是步頻在把身體撐住的。

拉大步幅也不行：ω=1.4 時 μx 從 1.85 → 2.00，週期俯仰 5.41° → 16.60°，前後離地 128/28。
重配 `x_off` 也救不回來——低步頻下它是**雙穩態**不是連續配平：
ω=1.4、μx=1.9 時 x_off=−40 給均俯仰 −6.39°（後腳過抬 184 mm），−30 直接翻成 +12.38°
（前腳過抬 120 mm），中間沒有平衡點。

walk 步態解決這件事的方式是把佔空比拉到 0.75：**任一時刻三腳著地**，身體沒有下沉的空檔。
實測支撐腳數 trot 2.0 → walk 2.82，同步頻下週期俯仰 15~17° → 5.6°。

================================================================================
§3 實測結果（20 秒、名目摩擦 0.6、kp=80）
================================================================================

                        FL     FR     RL     RR   最小   週期俯仰  彈跳   支撐  速度
  walk (預設)          71.0   69.9   75.1   74.0  69.9    5.61°  30.1mm  2.80  0.43
  walk_stable          76.6   76.2   56.5   56.1  56.1    5.76°  18.4mm  3.15  0.50
  trot 快 (ω2.4)       79.0   81.9   96.7   97.1  79.0    7.37°  18.8mm  2.00  0.79
  trot 慢 (ω2.0/μ1.65) 88.4   77.7  117.3  116.1  77.7    3.46°  16.1mm  2.00  0.25
  舊 cpg_openloop      35.5   66.5   16.9    6.3   6.3   10.91°  26.6mm  2.00  0.60
  RL 權重 v3           41.4   74.8   87.5   87.4  41.4    6.75°  89.9mm  2.00  0.41

trot 穩健性（ω=2.6，20 秒，皆不跌倒、0% clip）：
  摩擦 0.3/0.5/0.6/0.8/1.0 → 最小離地 40.6/78.1/82.4/83.3/81.0 mm
  kp   60/70/80/90/100     → 最小離地 37.7/61.6/82.9/93.3/99.0 mm
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭環境錄影用；須在建立 Renderer 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import cpg_d1
import d1_model

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# ---- 相位偏移（腿序 FL, FR, RL, RR）----
# trot：對角同相，兩組差 pi。與 d1_model.PHASE_OFFSET 同值，但本檔自己留一份，
#       因為 walk 需要非 trot 的相位，而 d1_model 那份不能動（RL 契約 + 有測試釘死）。
PHASE_TROT = np.array([0.0, np.pi, np.pi, 0.0])
# walk：側序走 RL → FL → RR → FR，四腿均分一圈。
PHASE_WALK = np.array([0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi])

# ---- 步態預設（每組都是掃描出來的實測最佳點，見 §3）----
# walk 與 walk_stable 只差 duty（0.75 / 0.80），但 x_off 不能共用——
# duty 改變會改變動態配平，實測 duty=0.80 用 walk 的 −55 mm 會讓前後離地變 1.37 倍。
GAITS = {
    "walk":        dict(phase=PHASE_WALK, duty=0.75, omega=1.4, mu_x=1.90, x_off=-0.055),
    "walk_stable": dict(phase=PHASE_WALK, duty=0.80, omega=1.4, mu_x=1.90, x_off=-0.055),
    "trot":        dict(phase=PHASE_TROT, duty=0.50, omega=2.4, mu_x=1.80, x_off=-0.040),
}
MU_Y = 1.5          # 橫向擺幅 → fy = 0，直線走路不需要橫擺，見 §1-2
GAIT_G_C = 0.12     # 抬腳指令(m)。注意這是本檔的值，d1_model.G_C 是 RL 用的 0.08
SETTLE_S = 0.8      # 開走前先用帶 x_off 的站姿站穩，避免第一步從錯誤基準起跳


def make_cpg_step(phase: np.ndarray):
    """回傳一個綁定指定相位偏移的 cpg_step。

    與 `cpg_d1.cpg_step` 逐行相同，唯一差別是耦合矩陣 PHI 取自參數而非
    `d1_model.PHASE_OFFSET`——後者被測試釘死為 trot，而 walk 需要別的相位。
    （耦合項 W_COUP·Σ sin(θj−θi−Φ) 會把相位拉回 Φ 定義的關係，所以只改初始相位無效，
      必須連耦合矩陣一起換。）
    """
    PHI = phase[None, :] - phase[:, None]

    def step(c: dict, mux, muy, omega, dt: float) -> dict:
        rx, rxd, ry, ryd, th = (c["rx"].copy(), c["rx_d"].copy(),
                                c["ry"].copy(), c["ry_d"].copy(), c["theta"].copy())
        h = dt / d1_model.N_CPG_SUB
        for _ in range(d1_model.N_CPG_SUB):
            rxd += (d1_model.A_CONV * (d1_model.A_CONV / 4 * (mux - rx) - rxd)) * h
            rx += rxd * h
            ryd += (d1_model.A_CONV * (d1_model.A_CONV / 4 * (muy - ry) - ryd)) * h
            ry += ryd * h
            rbar = 0.5 * (rx + ry)
            diff = th[None, :] - th[:, None] - PHI
            th = th + (2 * np.pi * omega
                       + d1_model.W_COUP * np.sum(rbar[None, :] * np.sin(diff), 1)) * h
        return {"rx": rx, "rx_d": rxd, "ry": ry, "ry_d": ryd, "theta": th % (2 * np.pi)}

    return step


def duty_remap(th: np.ndarray, duty: float) -> np.ndarray:
    """把相位重映射成「擺動相佔一圈的 (1−duty)、站立相佔 duty」。

    原本的軌跡公式用 `sin θ > 0` 判擺動 = 佔空比固定 0.5（永遠只有兩腳著地）。
    重映射之後軌跡形狀不變、只有時間分配變，duty=0.5 時本函式恆等於原樣（可驗證：
    ph<0.5 → π·ph/0.5 = 2π·ph = θ；ph≥0.5 → π + π(ph−0.5)/0.5 = 2π·ph = θ）。
    """
    ph = (th % (2 * np.pi)) / (2 * np.pi)          # 0~1
    sw = 1.0 - duty                                 # 擺動相佔比
    return np.where(ph < sw,
                    np.pi * ph / sw,                       # 擺動 → 0~π
                    np.pi + np.pi * (ph - sw) / duty)      # 站立 → π~2π


def joint_targets(c: dict, f0s, jinvs, x_off: float, g_c: float,
                  duty: float = 0.5) -> np.ndarray:
    """CPG 狀態 → 12 個關節目標角。

    與 `cpg_d1.joint_targets` 的差別：足端 x 加上 `x_off` 基準偏移、抬腳高度用本檔的
    `g_c`、相位先經 `duty_remap`。其餘（fx/fy 換算、擺動/站立的 dz 分支）逐項相同。
    """
    th = duty_remap(c["theta"], duty)
    fx = 2 * (c["rx"] - d1_model.MU_MIN) / (d1_model.MU_MAX - d1_model.MU_MIN) - 1
    fy = 2 * (c["ry"] - d1_model.MU_MIN) / (d1_model.MU_MAX - d1_model.MU_MIN) - 1
    dx = -d1_model.D_STEP * fx * np.cos(th) + x_off
    dy = d1_model.D_STEP_Y * fy * np.cos(th)
    dz = np.where(np.sin(th) > 0, g_c * np.sin(th), d1_model.G_P * np.sin(th))
    off = np.stack([dx, dy, dz], -1)
    q = np.zeros((4, 3))
    for k in range(4):
        q[k] = d1_model.HOME3 + jinvs[k] @ off[k]
    return q.reshape(12)


def rollout(gait: str = "walk", secs: float = 20.0, omega: float = None,
            mu_x: float = None, mu_y: float = MU_Y, x_off: float = None,
            g_c: float = GAIT_G_C, duty: float = None,
            video: bool = False, quiet: bool = False) -> dict:
    """開迴路步態 rollout。未指定的參數取 `GAITS[gait]` 的預設值。"""
    cfg = GAITS[gait]
    phase = cfg["phase"]
    omega = cfg["omega"] if omega is None else omega
    mu_x = cfg["mu_x"] if mu_x is None else mu_x
    x_off = cfg["x_off"] if x_off is None else x_off
    duty = cfg["duty"] if duty is None else duty
    step = make_cpg_step(phase)

    m = d1_model.make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    f0s, jinvs = cpg_d1.leg_ik_consts(m)
    foot_gid = d1_model.foot_geom_ids(m)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    n_sub = int(round(d1_model.CTRL_DT / m.opt.timestep))
    clipped = 0

    def apply(q_des):
        nonlocal clipped
        # 位置伺服在 MJCF 內，不得有軟體 PD。這裡只統計有沒有撞到 ctrlrange。
        clipped += int(np.sum((q_des < lo - 1e-9) | (q_des > hi + 1e-9)))
        d.ctrl[:] = np.clip(q_des, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)

    def fresh_state():
        c = cpg_d1.cpg_init()
        c["theta"] = phase.copy()      # cpg_init 用的是 d1_model 的 trot 相位，這裡覆寫
        return c

    # 先用「帶 x_off 的站姿」站穩，而不是 HOME12——否則第一步要從 55 mm 外的
    # 錯誤基準跳過來，會踉蹌一下。
    c_stand = fresh_state()
    for _ in range(int(SETTLE_S / d1_model.CTRL_DT)):
        apply(joint_targets(c_stand, f0s, jinvs, x_off, 0.0, duty))

    ren = cam = None
    frames = []
    if video:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)

    c = fresh_state()
    n = int(secs / d1_model.CTRL_DT)
    half = n // 2                       # 前半段當暖身，統計只取後半段
    lift = [[] for _ in range(4)]
    pitch, height, support = [], [], []
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    fell = None

    for i in range(n):
        c = step(c, np.full(4, mu_x), np.full(4, mu_y), np.full(4, omega), d1_model.CTRL_DT)
        apply(joint_targets(c, f0s, jinvs, x_off, g_c, duty))

        grav = cpg_d1.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        if grav[2] > d1_model.FALL_GRAV_Z and fell is None:
            fell = i * d1_model.CTRL_DT
        if i >= half:
            hs = [float(d.geom_xpos[foot_gid[k]][2]) - d1_model.WHEEL_RADIUS for k in range(4)]
            for k in range(4):
                lift[k].append(hs[k])
            support.append(sum(1 for h in hs if h < 0.005))
            pitch.append(np.degrees(np.arcsin(np.clip(-grav[0], -1.0, 1.0))))
            height.append(float(d.qpos[2]))

        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.20]
            cam.distance, cam.elevation, cam.azimuth = 1.75, -6, 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    pit = np.asarray(pitch)
    hgt = np.asarray(height)
    per = max(1, int(round((1.0 / omega) / d1_model.CTRL_DT)))
    # secs 太短時取樣不足一個步態週期，這裡退回整段 p2p，避免對空清單取平均
    # （np.mean([]) 會回 nan 而且只發 RuntimeWarning，不會擋下來）。
    cyc = [np.max(pit[s:s + per]) - np.min(pit[s:s + per])
           for s in range(0, len(pit) - per, per)] or [float(pit.max() - pit.min())]

    res = {
        "gait": gait,
        "peak_lift": [float(np.percentile(l, 99)) for l in lift],   # m
        "min_lift": float(min(np.percentile(l, 99) for l in lift)),
        "pitch_cycle": float(np.mean(cyc)),
        "pitch_mean": float(pit.mean()),
        "bounce": float(hgt.max() - hgt.min()),
        "height": float(hgt.mean()),
        "support": float(np.mean(support)),
        "dist": float(d.qpos[0]) - x0,
        "lateral": float(d.qpos[1]) - y0,
        "speed": (float(d.qpos[0]) - x0) / secs,
        "fell": fell,
        "clip_pct": 100.0 * clipped / (n * 12),
    }

    if not quiet:
        pk = [v * 1000 for v in res["peak_lift"]]
        print(f"[步態] {gait}  ω={omega} μx={mu_x} μy={mu_y} "
              f"duty={duty} x_off={x_off * 1000:+.0f}mm G_C={g_c}")
        print("[離地] " + "  ".join(f"{L}={v:.1f}" for L, v in zip(d1_model.LEGS, pk))
              + f"  mm（最小 {res['min_lift'] * 1000:.1f}）")
        print(f"[姿態] 週期內俯仰 {res['pitch_cycle']:.2f}°  平均俯仰 {res['pitch_mean']:+.2f}°  "
              f"垂直彈跳 {res['bounce'] * 1000:.1f} mm  機身高 {res['height'] * 1000:.1f} mm  "
              f"平均支撐腳 {res['support']:.2f}")
        print(f"[位移] 前進 {res['dist']:+.2f} m（{res['speed']:.2f} m/s）  "
              f"側偏 {res['lateral']:+.2f} m  "
              f"跌倒={'是 @%.1fs' % fell if fell is not None else '否'}  "
              f"ctrlrange clip {res['clip_pct']:.1f}%")

    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"cpg_{gait}_d1.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", choices=sorted(GAITS), default="walk")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--mu-x", type=float, default=None, dest="mu_x")
    ap.add_argument("--mu-y", type=float, default=MU_Y, dest="mu_y")
    ap.add_argument("--x-off", type=float, default=None, dest="x_off")
    ap.add_argument("--g-c", type=float, default=GAIT_G_C, dest="g_c")
    ap.add_argument("--duty", type=float, default=None)
    ap.add_argument("--video", action="store_true")
    rollout(**vars(ap.parse_args()))
