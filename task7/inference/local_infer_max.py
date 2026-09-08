"""本機 CPU 推論（RL v2）：載入 Colab 訓練的權重，在**原始網格模型**上回放並量指標、判 G3–G7。

用法：

    # 沒權重時先測管線（固定動作 = 開迴路 A 基準步態，G0 有標準答案）
    conda run --no-capture-output -n rbtdog \
        python task7/inference/local_infer_max.py --dummy --secs 20 --perturb 3 --compare

    # 有權重：12 擾動 + 同擾動開迴路對照 + 影片 → 一次印 G3–G7
    conda run --no-capture-output -n rbtdog \
        python task7/inference/local_infer_max.py \
        --params task7/weights/cpg_rl_max_v2_params.pkl --secs 60 --perturb 12 --compare --video

================================================================================
★ RL v2（2026-09-08）與 v1 的差別
================================================================================
- 動作 14 維：每腿 (mux, muy, ω) ＋ body sway (x, y)；sway 有 ±0.06 m 與 0.004 m/步斜率限制
- 基準 `gait_baseline.BASELINE_A`（LS 序列、kp250/abad60/kd2、x_off −30、g_c 0.048、z_sag 0.036）
- 增益走 `Robot(kp3=, kd3=, kd_wheel=)`，不是模型預設
- `--perturb N`：x_off 加 1e-12·seed 的皮米擾動跑 N 次取中位數（同 cpg_sweep_max）
- `--compare`：同擾動跑開迴路 A（sway=0）當對照 → G4 的相對值
- G7：峰值力矩 ×1.2 < 70 N·m、追蹤誤差 ×1.14 < 0.6 rad，**任一不過不上機**

================================================================================
★ 為什麼預設跑**原始網格模型**而不是訓練用的 zgws_mjx_kp250.xml
================================================================================
訓練模型是為了 MJX 才把碰撞網格換成原始形狀的。驗收如果也跑那個簡化模型，
等於**用同一個近似去驗證那個近似** —— 落差永遠量不到。
所以這裡預設 `max_model.SCENE`（原始網格 + 純力矩致動器 + 迴圈內 PD）。

================================================================================
★ `--dummy` 不是隨便給個固定動作 —— 它就是開迴路 A 基準步態
================================================================================
`mux=1.80 / muy=1.50 / ω=1.4 / sway=0` 用 `atanh` 反推即得。所以 `--dummy` 跑出來的
數字必須與 `cpg_walk_max.rollout(gait="walk_a", kp3=…, kd3=…)` 逐位相同（test_local_infer_max）。

================================================================================
⚠️ 網路結構必須與 Colab 訓練時逐項相同
================================================================================
policy (256,256,128)、value (256,256,256)、`normalize_observations=True`。
隱藏層大小對不上 → flax 丟 `ScopeParamShapeError`，安全；
activation / 動作分布對不上 → brax **不會報錯**，只是 policy 行為錯亂 —— 由 brax **版本**決定，
notebook 鎖死 `brax==0.14.2` 是唯一防線。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭環境錄影用；須在 import mujoco 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cpg_max
import cpg_walk_max as cw
import gait_baseline as gb
import leg_kin
import max_model as mm
import obs_max

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
DEFAULT_SCENE = mm.SCENE      # ★ 原始網格模型，不是訓練用的 scene_flat_mjx.xml

POLICY_HIDDEN = (256, 256, 128)
VALUE_HIDDEN = (256, 256, 256)


# ⚠️ 必須與 rl_env_max 同值（test_obs_max / test_local_infer_max 釘住）。
#    不同值的話 policy 輸出的同一個數字會被解成不同的頻率／位移，而且不會報錯。
OMEGA_MIN, OMEGA_MAX = 0.0, 2.0
SWAY_MAX, SWAY_SLEW = 0.060, 0.004
A = gb.BASELINE_A
PHASE = cpg_max.PHASE_WALK_LS
GATE_TAU, GATE_ERR = 70.0, 0.6                # 實機 M9 的中止門檻
SIM2REAL_TAU, SIM2REAL_ERR = 1.2, 1.14        # kp250 實測比值 ×1.14（兩次）留餘裕；誤差 ×1.14


LAYOUT_DIMS = {"full": 14, "nomux": 10}
PRESET_LAYOUT = {"v2": "full", "v2.1": "full", "v2.2": "nomux"}
PRESET_G4_DROP = {"v2": 0.60, "v2.1": 0.60, "v2.2": 0.30}      # G4：roll 峰值要降的比例
PRESET_EVAL_VX = {"v2": 0.15, "v2.1": 0.15, "v2.2": 0.30}      # 驗收指令（v2.2 指令範圍 0.15–0.40）


def baseline_action(layout: str = "full") -> np.ndarray:
    """開迴路 A 基準步態對應的固定動作（`act_to_cmd` 的反函式；sway=0）。"""
    def inv(u, lo, hi):
        return float(np.arctanh(np.clip(2 * (u - lo) / (hi - lo) - 1, -0.999, 0.999)))

    if layout == "full":
        return np.array([inv(A["mu_x"], mm.MU_MIN, mm.MU_MAX),
                         inv(A["mu_y"], mm.MU_MIN, mm.MU_MAX),
                         inv(A["omega"], OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])
    if layout == "nomux":
        return np.array([inv(A["mu_y"], mm.MU_MIN, mm.MU_MAX),
                         inv(A["omega"], OMEGA_MIN, OMEGA_MAX)] * 4 + [0.0, 0.0])
    raise ValueError(layout)


def act_to_cmd(a: np.ndarray, layout: str = "full"):
    """動作 → (mux(4), muy(4), omega(4), sway_target(2))。與 rl_env_max.act_to_cmd 逐行相同。
    "full" 14 維：(mux,muy,ω)×4 + sway；"nomux" 10 維：(muy,ω)×4 + sway，mux 固定＝基準。"""
    a = np.tanh(np.asarray(a, dtype=float))
    if layout == "full":
        leg = a[:12].reshape(4, 3)
        mux = (leg[:, 0] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
        muy = (leg[:, 1] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
        om = (leg[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
        return mux, muy, om, a[12:14] * SWAY_MAX
    if layout == "nomux":
        leg = a[:8].reshape(4, 2)
        mux = np.full(4, A["mu_x"])
        muy = (leg[:, 0] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
        om = (leg[:, 1] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
        return mux, muy, om, a[8:10] * SWAY_MAX
    raise ValueError(layout)


def slew_sway(prev, tgt):
    return prev + np.clip(np.asarray(tgt, dtype=float) - prev, -SWAY_SLEW, SWAY_SLEW)


def load_policy(path: str, act_dim: int = obs_max.ACT_DIM):
    """載入 brax 權重，回傳 `infer(obs) -> action`（deterministic）。"""
    import functools

    import jax
    from brax.io import model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    factory = functools.partial(ppo_networks.make_ppo_networks,
                                policy_hidden_layer_sizes=POLICY_HIDDEN,
                                value_hidden_layer_sizes=VALUE_HIDDEN)
    net = factory(obs_max.obs_dim(act_dim), act_dim,
                  preprocess_observations_fn=running_statistics.normalize)
    pol = ppo_networks.make_inference_fn(net)(model.load_params(path), deterministic=True)
    jpol = jax.jit(pol)
    key = jax.random.PRNGKey(0)

    def infer(obs):
        a, _ = jpol(obs, key)
        return np.asarray(a, dtype=float)

    return infer


def run_once(args, infer, seed: int = 0) -> dict:
    """一次 rollout。`seed` 只用來給 x_off 加 1e-12·seed 的皮米擾動（同 cpg_sweep_max）。"""
    import mujoco

    layout = PRESET_LAYOUT[getattr(args, "preset", "v2")]
    act_dim = LAYOUT_DIMS[layout]

    scene = args.scene or DEFAULT_SCENE
    mode = "position" if scene != mm.SCENE else "torque_pd"
    r = cw.Robot(scene=scene, actuator_mode=mode, kp3=A["kp3"], kd3=A["kd3"],
                 kd_wheel=A["wheel_kd"], solver_iters=(6, 6) if mode == "position" else None)
    ks, f0 = leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(PHASE)
    x_off = A["x_off"] + seed * 1e-12
    r.reset_standing(cpg_max.stand_targets(ks, f0, x_off), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, x_off))
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()
    ren = cam = None
    frames = []
    if args.video and seed == 0:
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(r.m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)
    c = cpg_max.cpg_init(PHASE)
    n = int(args.secs / mm.CTRL_DT)
    cmd = np.array([args.vx, args.wz])
    last_a = np.zeros(act_dim)
    sway = np.zeros(2)
    n_reach, om_hist, sway_hist = 0, [], []
    tr = cw.Trace(r, n, args.secs, A["omega"], PHASE, A["duty"])
    for i in range(n):
        obs = obs_max.build_obs(r.d, c, cmd, last_a)
        a = infer(obs)
        mux, muy, om, sw_t = act_to_cmd(a, layout)
        sway = slew_sway(sway, sw_t)
        om_hist.append(om.copy())
        sway_hist.append(sway.copy())
        c = step(c, mux, muy, om, mm.CTRL_DT)
        # sway 全零時傳 None，走與開迴路 rollout **逐位元相同**的路徑（G0 靠這個）
        sw_arg = None if not np.any(sway) else (float(sway[0]), float(sway[1]))
        tgt = cpg_max.foot_targets(c, f0, x_off, A["g_c"], A["d_step"], A["d_step_y"],
                                   A["duty"], A["z_sag"], sw_arg)
        q_des, nc = cpg_max.joint_targets(c, f0, x_off, A["g_c"], A["d_step"], A["d_step_y"],
                                          A["duty"], ks, A["z_sag"], sw_arg)
        n_reach += nc
        r.step(q_des)
        tr.record(c["theta"], tgt[:, 0])
        last_a = a
        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.30]
            cam.distance, cam.elevation, cam.azimuth = 2.0, -10, 90
            ren.update_scene(r.d, cam)
            frames.append(ren.render())
    om_arr, sw_arr = np.asarray(om_hist), np.asarray(sway_hist)
    res = tr.summarize(n_reach, extra={
        "scene": scene, "actuator_mode": mode, "seed": seed,
        "cmd_vx": args.vx, "cmd_wz": args.wz, "dummy": bool(args.dummy),
        "omega_mean": float(om_arr.mean()), "omega_min": float(om_arr.min()),
        "omega_max": float(om_arr.max()),
        "sway_x_abs": float(np.abs(sw_arr[:, 0]).mean() * 1000),
        "sway_y_abs": float(np.abs(sw_arr[:, 1]).mean() * 1000),
    })
    # ⚠️ 一定要用 `yaw_total`（逐步累積、不包裹）而不是 `yaw`（首尾相減）——轉彎 20 秒就繞過一圈。
    res["yaw_rate"] = res["yaw_total"] / args.secs
    res["_frames"] = frames
    return res


def _med(rs, k):
    return float(np.median([r[k] for r in rs]))


def run(args) -> dict:
    """跑 `perturb` 個擾動（policy），可選同擾動的開迴路對照；回傳中位數彙總與 G3–G7 判定。"""
    preset = getattr(args, "preset", "v2")
    layout = PRESET_LAYOUT[preset]
    fixed = baseline_action(layout)
    infer = (lambda _o: fixed) if args.dummy else load_policy(args.params, LAYOUT_DIMS[layout])
    n_pert = max(1, int(getattr(args, "perturb", 1)))
    rs = [run_once(args, infer, s) for s in range(n_pert)]
    res = dict(rs[0])
    res.pop("_frames", None)
    for k in ("speed_travel", "bounce", "support", "min_lift", "roll_pk", "roll_std",
              "exec_front", "exec_rear", "yaw_total", "yaw_rate"):
        res[k] = _med(rs, k)
    res["n_perturb"] = len(rs)
    res["fell_n"] = sum(r["fell"] is not None for r in rs)
    res["tau_peak_max"] = float(max(max(r["tau_peak"]) for r in rs))
    res["err_peak_max"] = float(max(r["err_peak_max"] for r in rs))
    base = None
    if getattr(args, "compare", False):
        bargs = argparse.Namespace(**{**vars(args), "dummy": True, "video": False})
        brs = [run_once(bargs, lambda _o: fixed, s) for s in range(len(rs))]
        base = {k: _med(brs, k) for k in ("roll_pk", "roll_std", "exec_front", "exec_rear",
                                           "yaw_total", "speed_travel")}
        res["baseline"] = base
    g = {}
    g["G3"] = res["fell_n"] == 0
    drop = PRESET_G4_DROP[preset]
    g["G4"] = (base is not None and res["roll_pk"] <= (1 - drop) * base["roll_pk"]
               and (res["roll_std"] <= (1 - drop) * base["roll_std"] if drop >= 0.6
                    else res["roll_std"] <= base["roll_std"]))
    g["G5"] = res["exec_front"] >= 0.9 and abs(res["exec_front"] - res["exec_rear"]) < 0.15
    g["G6"] = (abs(res["yaw_total"]) * (60.0 / args.secs) < 5.0) if args.wz == 0 else None
    g["G7"] = ((res["tau_peak_max"] * SIM2REAL_TAU < GATE_TAU)
               and (res["err_peak_max"] * SIM2REAL_ERR < GATE_ERR))
    res["gates"] = g

    src = "基準固定動作" if args.dummy else Path(args.params).name
    cw.report(res, f"[推論 {preset}] {src}  cmd=(vx {args.vx:.2f}, wz {args.wz:+.2f})  擾動 {len(rs)}"
                   f"  ω {res['omega_min']:.2f}~{res['omega_max']:.2f}"
                   f"  sway |x| {res['sway_x_abs']:.0f} |y| {res['sway_y_abs']:.0f} mm")
    print(f"[G3] 跌倒 {res['fell_n']}/{len(rs)} → {'✅' if g['G3'] else '❌'}")
    if base:
        print(f"[G4] roll 峰值 {res['roll_pk']:.2f}°（基準 {base['roll_pk']:.2f}）"
              f" std {res['roll_std']:.2f}（基準 {base['roll_std']:.2f}）"
              f" → {'✅' if g['G4'] else '❌'}（峰值降 ≥{drop:.0%}"
              f"{'、std 也降 ≥60%' if drop >= 0.6 else '、std 不升'}）")
    print(f"[G5] 執行率 前 {res['exec_front']:.2f} 後 {res['exec_rear']:.2f}"
          f" → {'✅' if g['G5'] else '❌'}（前 ≥0.9、|前−後| <0.15）")
    if g["G6"] is not None:
        print(f"[G6] 總偏航 {res['yaw_total']:+.1f}° / {args.secs:.0f}s"
              f"（換算 60 s {res['yaw_total'] * 60 / args.secs:+.1f}°）→ {'✅' if g['G6'] else '❌'}")
    print(f"[G7] 峰值力矩 {res['tau_peak_max']:.1f}×{SIM2REAL_TAU}={res['tau_peak_max'] * SIM2REAL_TAU:.1f}"
          f"（<{GATE_TAU}）  誤差 {res['err_peak_max']:.3f}×{SIM2REAL_ERR}="
          f"{res['err_peak_max'] * SIM2REAL_ERR:.3f}（<{GATE_ERR}）→ {'✅' if g['G7'] else '❌ 不上機'}")
    frames = rs[0].get("_frames") or []
    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"cpg_rl_max_{preset.replace('.', '_')}.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--dummy", action="store_true",
                    help="不載權重，用開迴路基準步態對應的固定動作測管線")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=None,
                    help="前進速度指令 m/s（預設依 preset：v2/v2.1 0.15、v2.2 0.30）")
    ap.add_argument("--wz", type=float, default=0.0, help="偏航率指令 rad/s")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--scene", type=str, default=None,
                    help="覆寫場景；預設是**原始網格模型**。"
                         "給 scene_flat_mjx_kp250.xml 可與訓練條件對照")
    ap.add_argument("--preset", default="v2", choices=sorted(PRESET_LAYOUT),
                    help="權重是哪個 preset 訓的（決定動作維度／G4 門檻／預設 vx）")
    ap.add_argument("--perturb", type=int, default=1, help="皮米擾動次數（12 = 正式驗收）")
    ap.add_argument("--compare", action="store_true",
                    help="同擾動跑開迴路 A 當對照（G4 需要）")
    a = ap.parse_args()
    if not a.dummy and not a.params:
        ap.error("要嘛給 --params，要嘛用 --dummy")
    if a.vx is None:
        a.vx = PRESET_EVAL_VX[a.preset]
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
