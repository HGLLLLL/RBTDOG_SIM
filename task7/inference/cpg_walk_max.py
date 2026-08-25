"""D1 Max（ZSM-1w / zgws）的開迴路 CPG 步態 —— 不含 RL，純 CPG。

用法：
  # 靜態站立驗證（先跑這個，確認落地站得住、量到真實機身高度與四輪受力）
  conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --stand

  # 步態
  conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --gait trot --video
  conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --gait walk --video

================================================================================
§1 控制架構：PD 在迴圈裡，不在 MJCF 裡
================================================================================
官方 MJCF 的致動器是**純力矩 `<motor>`**，沒有位置伺服也沒有 ctrlrange
（原廠的 PD 由外部 `mc_ctrl` 提供）。所以這裡自己算 PD：

    tau = KP·(q_des − q) − KD·q̇        再 clip 到 TAU_MAX

內迴圈跑在 MJCF 的 timestep 0.002 s = **500 Hz，剛好等於原廠 `controller_dt`**；
CPG 指令 50 Hz。增益取原廠 RL 那組（ABAD 60 / HIP 120 / KNEE 120、Kd 1.0）。

⚠️ 這改變了診斷方式。task6 的經典症狀是「指令超出 ctrlrange 被靜默 clip」，
   這台**沒有 ctrlrange，那個症狀不會發生**。取而代之要盯三個：

     lim%    IK 解出的角度超出 `jnt_range` 的比例（超限會被關節限位硬擋）
     tau%    PD 力矩打到 TAU_MAX 飽和的比例（飽和 = 實際增益比你以為的低）
     reach%  足端目標超出腿的可達球殼、被沿徑向縮限的比例

   三個都印在每次 rollout 的輸出裡。任何一個不是 0 就不能只看步態指標。

================================================================================
§2 輪子怎麼處理 —— ★ 預設 `damp`，這是實測選出來的，不是照抄原廠
================================================================================
原廠 RL 走路時輪子是 Kp=60 / Kd=0.5（真的有位置增益）。**但那個 Kp 是搭配
「每個控制步都重新給輪子目標角」的 RL 策略**；開迴路 CPG 沒有那一層，
把目標角凍結在起步當下就會愈拖愈遠、PD 愈拉愈用力。

實測（walk_stable、ω=1.4、d_step=0.10、12 秒）：

    模式              前進    淨滾動   輪總行程    yaw     側偏
    hold（凍結目標）   935      1.0     1052    +39.1°   +616     ← 偏航失控
    hold+每步重鎖     2457     10.6     1482    +24.7°  +1124     ← 還是偏
    damp             2447     -6.2     2368     -2.4°   +364     ← ★ 採用
    free            -1550   2070.0     6947    -10.0°   +145     ← 變成往後滾

**「淨滾動」是四輪淨轉角換算的距離，用來回答「牠是在走還是在滾」。**
damp 的淨滾動只有 −6 mm 而前進 2447 mm → **確實是用腿走的**；
free 的淨滾動 2070 mm 且整台往後跑 → 那是輪子在空轉，不是步態。

偏航的來源就是 `hold`：凍結的目標角讓四顆輪子各自累積不同的角度誤差，
PD 把它換成接觸點上的縱向力，四個力不等就成了偏航力矩。
**證據是原地踏步（d_step=0）也照樣偏 −28°，而且把相位左右鏡像偏航不會換邊**——
所以不是步態的左右不對稱造成的，是輪子的控制律造成的。

⚠️ 輪關節在 MJCF 裡**沒有 damping / frictionloss**，Kd=0.5 是唯一的被動阻力。
   實機輪馬達有靜摩擦，task6 是用實測掙脫門檻填的，這台**還沒量過**。
   `--wheel-friction` 可以加，預設 0 —— 也就是目前的模擬對輪子的阻力是**樂觀的**，
   實機的靜摩擦只會讓腿式步態更好走，不會更差。

================================================================================
§3 實測結果
================================================================================
見 `task7/docs/` 的結果文件（由 `--sweep` 產生的表格）。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭錄影用；必須在建立 Renderer 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import cpg_max
import leg_kin
import max_model as mm

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# ---- 步態預設 ----
# 起點值的出處：G_C 取原廠 `leg_height` 0.10；x_off 取 0（靜態質心只偏 −0.6 mm）；
# D_STEP / OMEGA / MU_X 是掃出來的，見結果文件。
GAITS = {
    "trot": dict(phase=cpg_max.PHASE_TROT, duty=0.50, omega=2.0, mu_x=1.80,
                 x_off=0.000, d_step=0.15, g_c=0.10),
    "walk": dict(phase=cpg_max.PHASE_WALK, duty=0.75, omega=1.4, mu_x=1.80,
                 x_off=0.000, d_step=0.15, g_c=0.10),
    "walk_stable": dict(phase=cpg_max.PHASE_WALK, duty=0.80, omega=1.4, mu_x=1.80,
                        x_off=0.000, d_step=0.15, g_c=0.10),
}
MU_Y = 1.5        # → fy = 0，直線走路不需要橫擺（task6 §1-2；不歸零是側偏的主因）
D_STEP_Y = 0.12   # 橫擺尺度。ABAD 力臂約 0.41 m（D1 EDU 只有 0.22），比 task6 寬鬆
SETTLE_S = 1.5    # 開走前先站穩。這台 41 kg，比 task6 的 0.8 s 需要更久


class Robot:
    """MuJoCo 模型 + 迴圈內 PD。所有 rollout 共用，確保控制律只有一份。"""

    def __init__(self, friction: float = None, wheel_friction: float = 0.0):
        self.m = mm.make_model()
        if friction is not None:
            self.m.geom_friction[:, 0] = friction
        if wheel_friction > 0:
            self.m.dof_frictionloss[mm.WHEEL_QVEL_IDX] = wheel_friction
        self.d = mujoco.MjData(self.m)
        self.foot_bid = mm.foot_body_ids(self.m)
        self.jnt_rng = mm.leg_joint_ranges(self.m)
        self.n_sub = int(round(mm.CTRL_DT / self.m.opt.timestep))
        self.kp = np.tile(mm.KP3, 4)
        self.kd = np.tile(mm.KD3, 4)
        self.tau_max = np.tile(mm.TAU_MAX3, 4)
        # 診斷計數器
        self.n_cmd = 0          # 送出的關節指令數（= 控制步數 × 12）
        self.n_lim = 0          # 其中超出 jnt_range 的個數
        self.n_tau = 0          # PD 力矩飽和的個數（以 500 Hz 內迴圈計）
        self.n_tau_tot = 0
        self.wheel_hold = None  # None = 只做阻尼

    def reset_standing(self, q12: np.ndarray, height: float):
        """把機器人放在指定高度、指定關節角，機身水平、速度歸零。"""
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[:3] = [0.0, 0.0, height]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[mm.LEG_QPOS_IDX] = q12
        mujoco.mj_forward(self.m, d)

    def step(self, q_des: np.ndarray, wheel_mode: str = "damp"):
        """套用一個 50 Hz 控制週期：內部跑 n_sub 次 500 Hz 的 PD + mj_step。"""
        d, m = self.d, self.m
        lo, hi = self.jnt_rng[:, 0], self.jnt_rng[:, 1]
        # 診斷：指令有沒有超出關節限位。超限不會被靜默吃掉（沒有 ctrlrange），
        # 但會被關節限位約束硬擋住 —— 症狀一樣是「命令了卻沒動到」。
        self.n_lim += int(np.sum((q_des < lo - 1e-9) | (q_des > hi + 1e-9)))
        self.n_cmd += 12
        q_des = np.clip(q_des, lo, hi)

        for _ in range(self.n_sub):
            q = d.qpos[mm.LEG_QPOS_IDX]
            qd = d.qvel[mm.LEG_QVEL_IDX]
            tau = self.kp * (q_des - q) - self.kd * qd
            self.n_tau += int(np.sum(np.abs(tau) > self.tau_max))
            self.n_tau_tot += 12
            d.ctrl[mm.LEG_ACT_IDX] = np.clip(tau, -self.tau_max, self.tau_max)

            wv = d.qvel[mm.WHEEL_QVEL_IDX]
            if wheel_mode == "hold" and self.wheel_hold is not None:
                wq = d.qpos[mm.WHEEL_QPOS_IDX]
                wtau = mm.KP_WHEEL * (self.wheel_hold - wq) - mm.KD_WHEEL * wv
            elif wheel_mode == "free":
                wtau = np.zeros(4)
            else:                                    # damp
                wtau = -mm.KD_WHEEL * wv
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(wtau, -mm.TAU_MAX_WHEEL, mm.TAU_MAX_WHEEL)

            mujoco.mj_step(m, d)

    def lock_wheels(self):
        self.wheel_hold = self.d.qpos[mm.WHEEL_QPOS_IDX].copy()

    def foot_heights(self) -> list[float]:
        """四輪的離地高度（輪底相對地面），順序同 LEGS。"""
        return [float(self.d.xpos[b][2]) - mm.WHEEL_RADIUS for b in self.foot_bid]

    def foot_forces(self) -> np.ndarray:
        """四輪的地面法向接觸力（N），順序同 LEGS。"""
        f = np.zeros(4)
        body_of = {b: k for k, b in enumerate(self.foot_bid)}
        buf = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            k = body_of.get(b1, body_of.get(b2))
            if k is None:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, buf)
            # contact frame 的 x 軸是法向；力的方向對 geom1/geom2 相反，取絕對值即可
            f[k] += abs(buf[0])
        return f

    @property
    def lim_pct(self) -> float:
        return 100.0 * self.n_lim / max(1, self.n_cmd)

    @property
    def tau_pct(self) -> float:
        return 100.0 * self.n_tau / max(1, self.n_tau_tot)


def stand(secs: float = 4.0, x_off: float = 0.0, friction: float = None,
          wheel_mode: str = "damp", quiet: bool = False) -> dict:
    """靜態站立驗證（動態落地版，不是正向運動學）。

    交接文件 §8-3 要的就是這個：**讓它在平地上真的站起來並收斂**，
    量位置伺服的靜態撓度、四輪受力分佈、以及走起來之前的實際機身高度。
    """
    r = Robot(friction=friction)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    q_des = cpg_max.stand_targets(ks, f0, x_off)

    # 從純運動學高度多放 5 mm 落下，避免初始就穿模
    r.reset_standing(q_des, mm.NOMINAL_HEIGHT_KIN + 0.005)
    r.lock_wheels()

    n = int(secs / mm.CTRL_DT)
    for i in range(n):
        r.step(q_des, wheel_mode)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()          # 落地穩定後才鎖輪角

    d = r.d
    h = float(d.qpos[2])
    grav = cpg_max.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
    pitch = float(np.degrees(np.arcsin(np.clip(-grav[0], -1.0, 1.0))))
    q = d.qpos[mm.LEG_QPOS_IDX].copy()
    forces = r.foot_forces()
    # 靜態質心相對四輪支撐中心的前後偏移
    com = float(d.subtree_com[0][0]) - float(d.qpos[0])
    wheel_x = np.array([float(d.xpos[b][0]) for b in r.foot_bid]) - float(d.qpos[0])

    res = {"height": h, "pitch": pitch, "forces": forces.tolist(),
           "front_rear_ratio": float(forces[:2].sum() / max(1e-9, forces[2:].sum())),
           "sag": float(np.abs(q - q_des).max()),
           "com_x": com, "support_center_x": float(wheel_x.mean()),
           "com_offset": com - float(wheel_x.mean()),
           "tau_pct": r.tau_pct, "lim_pct": r.lim_pct,
           "settled": abs(h - mm.NOMINAL_HEIGHT_KIN) < 0.10}

    if not quiet:
        print(f"[站立] x_off={x_off * 1000:+.0f} mm  摩擦={friction or '預設1.0'}  輪={wheel_mode}")
        print(f"  機身高度  {h * 1000:.1f} mm（純運動學 {mm.NOMINAL_HEIGHT_KIN * 1000:.1f} mm，"
              f"位置伺服靜態撓度 {(mm.NOMINAL_HEIGHT_KIN - h) * 1000:+.1f} mm）")
        print(f"  俯仰      {pitch:+.2f}°   最大關節追蹤誤差 {np.degrees(res['sag']):.2f}°")
        print("  四輪受力  " + "  ".join(f"{L}={v:6.1f}" for L, v in zip(mm.LEGS, forces))
              + f"  N（總 {forces.sum():.1f}，前/後 = {res['front_rear_ratio']:.3f}）")
        print(f"  質心前後偏移 {res['com_offset'] * 1000:+.2f} mm（相對四輪支撐中心）")
        print(f"  力矩飽和 {r.tau_pct:.2f}%   指令超限 {r.lim_pct:.2f}%")
    return res


def rollout(gait: str = "trot", secs: float = 20.0, omega: float = None,
            mu_x: float = None, mu_y: float = MU_Y, x_off: float = None,
            g_c: float = None, d_step: float = None, d_step_y: float = D_STEP_Y,
            duty: float = None, friction: float = None, wheel_mode: str = "damp",
            wheel_friction: float = 0.0, z_sag: float = None, video: bool = False,
            quiet: bool = False) -> dict:
    """開迴路步態 rollout。未指定的參數取 `GAITS[gait]` 的預設值。"""
    cfg = GAITS[gait]
    phase = cfg["phase"]
    omega = cfg["omega"] if omega is None else omega
    mu_x = cfg["mu_x"] if mu_x is None else mu_x
    x_off = cfg["x_off"] if x_off is None else x_off
    duty = cfg["duty"] if duty is None else duty
    g_c = cfg["g_c"] if g_c is None else g_c
    d_step = cfg["d_step"] if d_step is None else d_step
    z_sag = mm.STATIC_SAG if z_sag is None else z_sag

    r = Robot(friction=friction, wheel_friction=wheel_friction)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(phase)
    n_reach = 0

    # 先用「帶 x_off 的站姿」站穩，而不是 HOME —— 否則第一步要從偏移過的基準跳過來。
    r.reset_standing(cpg_max.stand_targets(ks, f0, x_off), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, x_off), wheel_mode)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    ren = cam = None
    frames = []
    if video:
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(r.m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)

    c = cpg_max.cpg_init(phase)
    n = int(secs / mm.CTRL_DT)
    half = n // 2                     # 前半段當暖身，統計只取後半段
    lift = [[] for _ in range(4)]
    pitch, roll, height, support, phases = [], [], [], [], []
    d = r.d
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw0 = cpg_max.yaw_deg(d.qpos[3:7])
    w0 = d.qpos[mm.WHEEL_QPOS_IDX].copy()
    fell = None

    for i in range(n):
        c = step(c, np.full(4, mu_x), np.full(4, mu_y), np.full(4, omega), mm.CTRL_DT)
        q_des, nc = cpg_max.joint_targets(c, f0, x_off, g_c, d_step, d_step_y, duty,
                                          ks, z_sag)
        n_reach += nc
        r.step(q_des, wheel_mode)

        grav = cpg_max.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        if grav[2] > mm.FALL_GRAV_Z and fell is None:
            fell = i * mm.CTRL_DT
        if i >= half:
            hs = r.foot_heights()
            for k in range(4):
                lift[k].append(hs[k])
            # 支撐腳數用**實際接觸力**判定，不用「離地高度 < 5 mm」。
            # 高度門檻在會彈跳的步態上會騙人：機身整體騰空時腳離地面很近卻沒受力，
            # 一樣被算成支撐腳。改用接觸力就沒有這個模糊地帶。
            support.append(int(np.sum(r.foot_forces() > 1.0)))
            pitch.append(np.degrees(np.arcsin(np.clip(-grav[0], -1.0, 1.0))))
            roll.append(np.degrees(np.arcsin(np.clip(grav[1], -1.0, 1.0))))
            height.append(float(d.qpos[2]))
            phases.append(c["theta"].copy())

        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.30]
            cam.distance, cam.elevation, cam.azimuth = 2.8, -8, 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    pit = np.asarray(pitch)
    hgt = np.asarray(height)
    per = max(1, int(round((1.0 / omega) / mm.CTRL_DT)))
    # secs 太短時取樣不足一個週期，退回整段 p2p —— np.mean([]) 會回 nan 而且只發
    # RuntimeWarning，不會擋下來。
    cyc = [np.max(pit[s:s + per]) - np.min(pit[s:s + per])
           for s in range(0, len(pit) - per, per)] or [float(pit.max() - pit.min())]

    # 相位鎖定：實際相位差 vs 目標相位差，用圓形統計（±180° 包裹會讓一般標準差虛胖）
    ph = np.asarray(phases)
    lock = [float(np.degrees(cpg_max.circ_std(
        ph[:, k] - ph[:, 0] - (phase[k] - phase[0])))) for k in range(4)]

    res = {
        "gait": gait, "omega": omega, "mu_x": mu_x, "x_off": x_off,
        "g_c": g_c, "d_step": d_step, "duty": duty, "z_sag": z_sag,
        "peak_lift": [float(np.percentile(l, 99)) for l in lift],
        "min_lift": float(min(np.percentile(l, 99) for l in lift)),
        "pitch_cycle": float(np.mean(cyc)),
        "pitch_mean": float(pit.mean()),
        "roll_mean": float(np.mean(roll)),
        "bounce": float(hgt.max() - hgt.min()),
        "height": float(hgt.mean()),
        "support": float(np.mean(support)),
        "dist": float(d.qpos[0]) - x0,
        "lateral": float(d.qpos[1]) - y0,
        "speed": (float(d.qpos[0]) - x0) / secs,
        "yaw": cpg_max.yaw_deg(d.qpos[3:7]) - yaw0,
        # 淨滾動距離：回答「牠是在走還是在滾」。輪軸 +y，前進對應輪角減少。
        "net_roll": float(-np.mean(d.qpos[mm.WHEEL_QPOS_IDX] - w0) * mm.WHEEL_RADIUS),
        "fell": fell,
        "lim_pct": r.lim_pct,
        "tau_pct": r.tau_pct,
        "reach_pct": 100.0 * n_reach / max(1, n * 4),
        "phase_lock": lock,
    }

    if not quiet:
        pk = [v * 1000 for v in res["peak_lift"]]
        print(f"[步態] {gait}  ω={omega} μx={mu_x} μy={mu_y} duty={duty} "
              f"x_off={x_off * 1000:+.0f}mm D_STEP={d_step} G_C={g_c} "
              f"撓度補償={z_sag * 1000:.1f}mm 輪={wheel_mode}")
        print("[離地] " + "  ".join(f"{L}={v:.1f}" for L, v in zip(mm.LEGS, pk))
              + f"  mm（最小 {res['min_lift'] * 1000:.1f}）")
        print(f"[姿態] 週期俯仰 {res['pitch_cycle']:.2f}°  平均俯仰 {res['pitch_mean']:+.2f}°  "
              f"平均側傾 {res['roll_mean']:+.2f}°  彈跳 {res['bounce'] * 1000:.1f} mm  "
              f"機身高 {res['height'] * 1000:.1f} mm  支撐腳 {res['support']:.2f}")
        print(f"[位移] 前進 {res['dist']:+.2f} m（{res['speed']:.2f} m/s）  "
              f"側偏 {res['lateral']:+.2f} m  偏航 {res['yaw']:+.1f}°  "
              f"淨滾動 {res['net_roll'] * 1000:+.0f} mm  "
              f"跌倒={'是 @%.1fs' % fell if fell is not None else '否'}")
        print(f"[診斷] 超限 {res['lim_pct']:.2f}%  力矩飽和 {res['tau_pct']:.2f}%  "
              f"IK縮限 {res['reach_pct']:.2f}%  "
              f"相位鎖定σ " + "/".join(f"{v:.1f}" for v in lock) + "°")

    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"cpg_{gait}_max.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", action="store_true", help="只做靜態站立驗證")
    ap.add_argument("--gait", choices=sorted(GAITS), default="trot")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--mu-x", type=float, default=None, dest="mu_x")
    ap.add_argument("--mu-y", type=float, default=MU_Y, dest="mu_y")
    ap.add_argument("--x-off", type=float, default=None, dest="x_off")
    ap.add_argument("--g-c", type=float, default=None, dest="g_c")
    ap.add_argument("--d-step", type=float, default=None, dest="d_step")
    ap.add_argument("--d-step-y", type=float, default=D_STEP_Y, dest="d_step_y")
    ap.add_argument("--duty", type=float, default=None)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--wheel-mode", choices=("hold", "damp", "free"), default="damp",
                    dest="wheel_mode")
    ap.add_argument("--wheel-friction", type=float, default=0.0, dest="wheel_friction")
    ap.add_argument("--z-sag", type=float, default=None, dest="z_sag")
    ap.add_argument("--video", action="store_true")
    a = vars(ap.parse_args())
    if a.pop("stand"):
        stand(secs=min(a["secs"], 5.0), x_off=a["x_off"] or 0.0,
              friction=a["friction"], wheel_mode=a["wheel_mode"])
    else:
        rollout(**a)
