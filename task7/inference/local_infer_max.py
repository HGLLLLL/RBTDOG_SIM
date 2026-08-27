"""本機 CPU 推論：載入 Colab 訓練的權重，在**原始網格模型**上回放並量指標。

用法：

    # 沒權重時先測管線（固定動作 = 開迴路基準步態）
    conda run --no-capture-output -n rbtdog \
        python task7/inference/local_infer_max.py --dummy --secs 20

    # 有權重
    conda run --no-capture-output -n rbtdog \
        python task7/inference/local_infer_max.py \
        --params task7/weights/cpg_rl_max_params.pkl --secs 20 --video

================================================================================
★ 為什麼預設跑**原始網格模型**而不是訓練用的 zgws_mjx.xml
================================================================================
訓練模型是為了 MJX 才把碰撞網格換成原始形狀的。驗收如果也跑那個簡化模型，
等於**用同一個近似去驗證那個近似** —— 落差永遠量不到。
所以這裡預設 `max_model.SCENE`（原始網格 + 純力矩致動器 + 迴圈內 PD）。
想跟訓練條件對照時再用 `--scene`。

兩個模型的落差已量過（`docs/MJX模型對照_2026-08-27.md`）：
行進速度 −0.7%、彈跳 −1.7%、支撐腳 0%、離地 0%。

================================================================================
★ `--dummy` 不是隨便給個固定動作 —— 它就是開迴路基準步態
================================================================================
基準步態（`gait_baseline.BASELINE`）在這個動作空間裡是一個**固定動作**：
`mux=1.80 / muy=1.50 / ω=1.4` 用 `atanh` 反推即得。所以 `--dummy` 跑出來的
數字應該要對得上 `cpg_walk_max.rollout(gait="walk")`。這讓「管線有沒有接對」
變成一個**有標準答案**的檢查，而不是「跑起來沒炸就算過」。

================================================================================
⚠️ 網路結構必須與 Colab 訓練時逐項相同
================================================================================
policy (256,256,128)、value (256,256,256)、`normalize_observations=True`。
不匹配的後果分兩種，差別很大：

  - **隱藏層大小**對不上 → flax 丟 `ScopeParamShapeError`，當場停住，安全。
  - **activation 與動作分布類型**對不上 → 參數形狀完全相同，brax
    **不會報錯**，權重照樣載入，只是 policy 行為錯亂而毫無訊息。

而 activation 與分布是由 `make_ppo_networks` 的**預設值**決定的，也就是說
它們由 **brax 的版本**決定，不由本檔的參數決定。所以 Colab notebook 的安裝格
鎖死 `brax==0.14.2` 並在裝完後斷言，是這個靜默失敗的唯一防線。
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


def baseline_action() -> np.ndarray:
    """開迴路基準步態對應的 12 維動作（`act_to_cmd` 的反函式）。

    `--dummy` 用它，所以 dummy 跑出來的數字有標準答案可對。
    """
    def inv(u, lo, hi):
        return float(np.arctanh(np.clip(2 * (u - lo) / (hi - lo) - 1, -0.999, 0.999)))

    return np.array([inv(gb.BASELINE["mu_x"], mm.MU_MIN, mm.MU_MAX),
                     inv(gb.BASELINE["mu_y"], mm.MU_MIN, mm.MU_MAX),
                     inv(gb.BASELINE["omega"], OMEGA_MIN, OMEGA_MAX)] * 4)


# ⚠️ 必須與 notebook 第 4 格的 OMEGA_MIN / OMEGA_MAX 同值。
#    不同值的話 policy 輸出的同一個數字會被解成不同的頻率，而且不會報錯。
OMEGA_MIN, OMEGA_MAX = 0.0, 2.0


def act_to_cmd(a: np.ndarray):
    """12 維動作 → 每腿 (mux, muy, omega)。與 notebook 的 `act_to_cmd` 逐行相同。"""
    a = np.tanh(np.asarray(a, dtype=float)).reshape(4, 3)
    mux = (a[:, 0] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
    muy = (a[:, 1] + 1) / 2 * (mm.MU_MAX - mm.MU_MIN) + mm.MU_MIN
    om = (a[:, 2] + 1) / 2 * (OMEGA_MAX - OMEGA_MIN) + OMEGA_MIN
    return mux, muy, om


def load_policy(path: str):
    """載入 brax 權重，回傳 `infer(obs) -> action`（deterministic）。"""
    import functools

    import jax
    from brax.io import model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    factory = functools.partial(ppo_networks.make_ppo_networks,
                                policy_hidden_layer_sizes=POLICY_HIDDEN,
                                value_hidden_layer_sizes=VALUE_HIDDEN)
    net = factory(obs_max.OBS_DIM, obs_max.ACT_DIM,
                  preprocess_observations_fn=running_statistics.normalize)
    pol = ppo_networks.make_inference_fn(net)(model.load_params(path), deterministic=True)
    jpol = jax.jit(pol)
    key = jax.random.PRNGKey(0)

    def infer(obs):
        a, _ = jpol(obs, key)
        return np.asarray(a, dtype=float)

    return infer


def run(args) -> dict:
    """跑一段推論並回傳指標（欄位與 `cpg_walk_max.rollout` 完全相同）。"""
    import mujoco

    scene = args.scene or DEFAULT_SCENE
    # 原始網格模型的致動器是純力矩，用迴圈內 PD；MJX 模型是位置伺服。
    # 兩者不是可以自由組合的旋鈕，模式必須跟著模型檔走（Robot 會斷言擋下錯配）。
    mode = "position" if scene == mm.SCENE_MJX else "torque_pd"
    r = cw.Robot(scene=scene, actuator_mode=mode)

    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(cpg_max.PHASE_WALK)
    x_off, g_c = gb.BASELINE["x_off"], gb.BASELINE["g_c"]
    d_step, d_step_y = gb.BASELINE["d_step"], gb.BASELINE["d_step_y"]
    duty, z_sag = gb.BASELINE["duty"], gb.BASELINE["z_sag"]

    fixed = baseline_action()
    infer = (lambda _o: fixed) if args.dummy else load_policy(args.params)

    # 先站穩，與 rollout 用同一段流程（否則第一步要從偏移過的基準跳過來）
    r.reset_standing(cpg_max.stand_targets(ks, f0, x_off), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(cw.SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, x_off))
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    ren = cam = None
    frames = []
    if args.video:
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(r.m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)

    c = cpg_max.cpg_init(cpg_max.PHASE_WALK)
    n = int(args.secs / mm.CTRL_DT)
    cmd = np.array([args.vx, args.wz])
    last_a = np.zeros(obs_max.ACT_DIM)
    n_reach = 0
    om_hist = []
    # ω 是逐步變動的，`Trace` 的週期長度要有一個代表值 —— 用基準值。
    # （只影響 speed_travel 的步長與週期俯仰的分段，不影響其他欄位。）
    tr = cw.Trace(r, n, args.secs, gb.BASELINE["omega"], cpg_max.PHASE_WALK)

    for i in range(n):
        obs = obs_max.build_obs(r.d, c, cmd, last_a)
        a = infer(obs)
        mux, muy, om = act_to_cmd(a)
        om_hist.append(om.copy())
        c = step(c, mux, muy, om, mm.CTRL_DT)
        q_des, nc = cpg_max.joint_targets(c, f0, x_off, g_c, d_step, d_step_y, duty,
                                          ks, z_sag)
        n_reach += nc
        r.step(q_des)
        tr.record(c["theta"])
        last_a = a

        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.30]
            cam.distance, cam.elevation, cam.azimuth = 2.0, -10, 90
            ren.update_scene(r.d, cam)
            frames.append(ren.render())

    om_arr = np.asarray(om_hist)
    res = tr.summarize(n_reach, extra={
        "scene": scene, "actuator_mode": mode,
        "cmd_vx": args.vx, "cmd_wz": args.wz, "dummy": bool(args.dummy),
        # policy 實際用到的頻率範圍。開迴路是恆定 1.4，RL 會變 ——
        # 這一欄是判斷「policy 到底有沒有在動 ω」最直接的證據。
        "omega_mean": float(om_arr.mean()), "omega_min": float(om_arr.min()),
        "omega_max": float(om_arr.max()),
        # 偏航率（°/s）。★ G4 的判準：要顯著優於開迴路的 −0.5 ~ −0.9 °/s。
        "yaw_rate": None,
    })
    res["yaw_rate"] = res["yaw"] / args.secs

    src = "基準固定動作" if args.dummy else Path(args.params).name
    cw.report(res, f"[推論] {src}  cmd=(vx {args.vx:.2f}, wz {args.wz:+.2f})  "
                   f"場景={Path(scene).name}  ω 用到 "
                   f"{res['omega_min']:.2f}~{res['omega_max']:.2f}（平均 "
                   f"{res['omega_mean']:.2f}）")
    print(f"[G4  ] 偏航率 {res['yaw_rate']:+.3f} °/s"
          f"（開迴路基準 −0.5 ~ −0.9 °/s，要顯著優於它）")

    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "cpg_rl_max.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="")
    ap.add_argument("--dummy", action="store_true",
                    help="不載權重，用開迴路基準步態對應的固定動作測管線")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--vx", type=float, default=0.15, help="前進速度指令 m/s")
    ap.add_argument("--wz", type=float, default=0.0, help="偏航率指令 rad/s")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--scene", type=str, default=None,
                    help="覆寫場景；預設是**原始網格模型**。"
                         "給 scene_flat_mjx.xml 可與訓練條件對照")
    a = ap.parse_args()
    if not a.dummy and not a.params:
        ap.error("要嘛給 --params，要嘛用 --dummy")
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
