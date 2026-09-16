"""v3.x 權重的本機驗收：五類指令各 10 s（多種子），與零動作基準、原廠對標表並列。
    conda run -n rbtdog python task7/inference/local_infer_v3.py --weights task7/weights/cpg_rl_v3_params.pkl --seeds 3
    ... --video task7/outputs/eval_v3.mp4     # 另外渲染策略跑的影片（每指令 10 s）
輸出 outputs/eval_<權重名>.md。對標規則見 spec §5.2。
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import rl_env_v3 as v3  # noqa: E402

POLICY_HIDDEN, VALUE_HIDDEN = (256, 256, 128), (256, 256, 256)
CASES = (("直走 0.5", (0.5, 0.0, 0.0), "fwd"), ("弧線 0.5+0.5", (0.5, 0.0, 0.5), "arc"), ("原地左轉 1.3", (0.0, 0.0, 1.3), "turn"),
         ("原地右轉 1.3", (0.0, 0.0, -1.3), "turn"), ("左平移 0.20", (0.0, 0.20, 0.0), "lat"), ("右平移 0.20", (0.0, -0.20, 0.0), "lat"),
         ("左平移 0.12", (0.0, 0.12, 0.0), "lat"), ("斜走 0.3+0.15", (0.3, 0.15, 0.0), "diag"), ("站立", (0.0, 0.0, 0.0), "stand"))
# 平移指令 2026-09-16 起改 0.20／0.12（v3.3 訓練範圍 0.04–0.30，原廠實測 0.3–0.5）；0.08／0.04 是 v3.2b 時代的數字，舊表不可比。
ABAD, HIP, KNEE = [0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]


def load_policy(path: str, obs_dim: int = 88):
    from brax.io import model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    factory = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=POLICY_HIDDEN, value_hidden_layer_sizes=VALUE_HIDDEN)
    net = factory(obs_dim, v3.ACT_DIM, preprocess_observations_fn=running_statistics.normalize)
    pol = jax.jit(ppo_networks.make_inference_fn(net)(model.load_params(path), deterministic=True))
    key = jax.random.PRNGKey(0)
    return lambda obs: pol(obs, key)[0]


def rollout(env, jit_reset, jit_step, cmd, steps, policy=None, seed=0):
    s = jit_reset(jax.random.PRNGKey(seed))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    Q, M, TAU, A = [], [], [], []
    for i in range(steps):
        a = policy(s.obs) if policy is not None else jnp.zeros(v3.ACT_DIM)
        s = jit_step(s, a)
        d = s.pipeline_state
        Q.append(np.asarray(d.qpos)); A.append(np.asarray(a))
        TAU.append(np.asarray(d.actuator_force[v3.LEG_ACT_IDX]))
        M.append({k: float(s.metrics[k]) for k in ("vx", "vy", "wz", "roll", "pitch", "tau_pk", "height")}
                 | {"clr": np.asarray(env._wheel_clearance(d)) * 1000.0, "done": float(s.done)})
        if float(s.done) > 0:
            break
    n = len(M); h = max(1, int(0.2 * steps)) if n > int(0.4 * steps) else 0       # 前 2 s 淡入不算
    g = lambda k: np.array([m[k] for m in M[h:]])   # noqa: E731
    tau = np.abs(np.array(TAU[h:])) if n > h else np.zeros((1, 12))
    return dict(
        n=n, fell=M[-1]["done"] > 0, t_fall=(n * v3.CTRL_DT if M[-1]["done"] > 0 else None),
        vx=float(g("vx").mean()), vy=float(g("vy").mean()), yaw=float(np.degrees(g("wz").mean())), yaw_std=float(np.degrees(g("wz").std())),
        roll_std=float(g("roll").std()), roll_max=float(g("roll").max()), pitch_std=float(g("pitch").std()),
        tau_pk=float(g("tau_pk").max()), tau_knee_pk=float(tau[:, KNEE].max()), tau_knee_rms=float(np.sqrt((tau[:, KNEE] ** 2).mean())),
        tau_hip_pk=float(tau[:, HIP].max()), tau_hip_rms=float(np.sqrt((tau[:, HIP] ** 2).mean())),
        tau_abad_pk=float(tau[:, ABAD].max()), tau_abad_rms=float(np.sqrt((tau[:, ABAD] ** 2).mean())),
        lift=np.max(np.array([m["clr"] for m in M[h:]]), 0).round(0).tolist() if n > h else [0, 0, 0, 0],
        act_abs=float(np.abs(np.tanh(np.array(A))).mean()), Q=Q)


def factory_ref():
    ds = json.loads((HERE.parent / "outputs" / "ref_gait_dataset.json").read_text(encoding="utf-8"))["factory_benchmark"]
    pick = {"fwd": ds["fwd"][2], "arc": ds["arc_left"][1], "turn": ds["turn_left"][0], "lat": ds["lat_left"][0]}
    return {k: dict(v=b["v_body"], yaw=b["yaw_rate_deg_s"], roll_std=b["roll_std_deg"], knee_pk=b["tau_pk"]["3_knee_pitch"], knee_rms=b["tau_rms"]["3_knee_pitch"],
                    hip_pk=b["tau_pk"]["2_hip_pitch"], abad_pk=b["tau_pk"]["1_hip_roll"], lift=b["lift_apex_mm"]) for k, b in pick.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "weights" / "cpg_rl_v3_params.pkl"))
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--no-baseline", action="store_true", dest="no_baseline"); ap.add_argument("--video", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--mesh", action="store_true", help="影片用官方 STL 網格模型渲染（scene_flat.xml；物理仍是訓練模型，只換外觀）")
    ap.add_argument("--title", default="", help="影片標題前綴")
    a = ap.parse_args()
    env = v3.DualModeEnv(ref=dict(cyc_amp_rand=False)); jr, js = jax.jit(env.reset), jax.jit(env.step)   # eval：原地轉幅度固定 REF cyc_amp_turn
    pol = load_policy(a.weights, env.obs_dim); steps = int(a.secs / v3.CTRL_DT); F = factory_ref()
    rows, traj = [], {}
    for name, cmd, fam in CASES:
        if a.only and a.only not in name:
            continue
        t0 = time.time()
        R = [rollout(env, jr, js, cmd, steps, pol, seed=k) for k in range(a.seeds)]
        B = None if a.no_baseline else rollout(env, jr, js, cmd, steps, None, seed=0)
        ok = [r for r in R if not r["fell"]] or R
        agg = {k: float(np.mean([r[k] for r in ok])) for k in ("vx", "vy", "yaw", "yaw_std", "roll_std", "roll_max", "tau_pk", "tau_knee_pk", "tau_knee_rms", "tau_hip_pk", "tau_hip_rms", "tau_abad_pk", "tau_abad_rms", "act_abs")}
        agg["lift"] = np.mean([r["lift"] for r in ok], 0).round(0).tolist()
        t_falls = ",".join("%.1fs" % r["t_fall"] for r in R if r["fell"])
        agg["falls"] = "%d/%d" % (sum(r["fell"] for r in R), len(R)) + ("（%s）" % t_falls if t_falls else "")
        rows.append((name, cmd, fam, agg, B)); traj[name] = ok[0]["Q"]
        print(f"{name:12s} 摔 {agg['falls']:14s} vx {agg['vx']:+.2f} vy {agg['vy']:+.3f} yaw {agg['yaw']:+5.1f}±{agg['yaw_std']:.0f} | roll std {agg['roll_std']:.2f} max {agg['roll_max']:.1f} | "
              f"膝 τ 峰/RMS {agg['tau_knee_pk']:.0f}/{agg['tau_knee_rms']:.0f} 髖 {agg['tau_hip_pk']:.0f}/{agg['tau_hip_rms']:.0f} ABAD {agg['tau_abad_pk']:.0f}/{agg['tau_abad_rms']:.0f} | lift {agg['lift']} | |a| {agg['act_abs']:.2f}"
              + (f" ‖ 零動作: vx {B['vx']:+.2f} vy {B['vy']:+.3f} yaw {B['yaw']:+5.1f} roll std {B['roll_std']:.2f} 膝峰 {B['tau_knee_pk']:.0f} 摔 {B['fell']}" if B else "") + f" ({time.time()-t0:.0f}s)", flush=True)
    # ---- md
    wname = Path(a.weights).stem
    L = [f"# 驗收 {wname}（{a.seeds} 種子 × {a.secs:.0f} s，前 2 s 不計；對標 = 原廠 trip21 動作段）", "",
         "| 指令 | 摔 | vx | vy | 偏航 °/s（±std） | roll std/峰 ° | 膝 τ 峰/RMS | 髖 τ 峰/RMS | ABAD τ 峰/RMS | 抬腳 mm | \\|a\\| | 零動作 vx/vy/yaw/roll std/膝峰 | 原廠 v/yaw/roll std/膝峰/膝RMS/抬腳 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, cmd, fam, g, B in rows:
        f = F.get(fam)
        fs = f"{f['v']:+.2f}/{f['yaw']:+.0f}/{f['roll_std']:.1f}/{f['knee_pk']:.0f}/{f['knee_rms']:.0f}/{f['lift']:.0f}" if f else "—"
        bs = f"{B['vx']:+.2f}/{B['vy']:+.3f}/{B['yaw']:+.0f}/{B['roll_std']:.2f}/{B['tau_knee_pk']:.0f}" + ("（摔）" if B and B["fell"] else "") if B else "—"
        L.append(f"| {name} | {g['falls']} | {g['vx']:+.2f} | {g['vy']:+.3f} | {g['yaw']:+.1f}（±{g['yaw_std']:.0f}） | {g['roll_std']:.2f}/{g['roll_max']:.1f} | {g['tau_knee_pk']:.0f}/{g['tau_knee_rms']:.0f} | "
                 f"{g['tau_hip_pk']:.0f}/{g['tau_hip_rms']:.0f} | {g['tau_abad_pk']:.0f}/{g['tau_abad_rms']:.0f} | {g['lift']} | {g['act_abs']:.2f} | {bs} | {fs} |")
    L += ["", "判讀規則（spec §5.2）：速度／偏航率到原廠 80%；roll std ≤ 原廠 1.5 倍；膝／髖力矩峰 ≤ 原廠 1.3 倍（RMS 才可比）；抬腳 15–40 mm。原廠 ABAD 峰 40–60 是它 60° 命令差造成的，我們目標 ≤ 原廠。"]
    out = HERE.parent / "outputs" / f"eval_{wname}.md"; out.write_text("\n".join(L) + "\n", encoding="utf-8"); print("→", out)
    if a.video:
        import os; os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco, imageio
        from PIL import Image, ImageDraw, ImageFont
        import max_model as mm
        m = mujoco.MjModel.from_xml_path(mm.SCENE if a.mesh else v3.SCENE_V3); d = mujoco.MjData(m)
        m.vis.global_.offwidth, m.vis.global_.offheight = 960, 540
        r = mujoco.Renderer(m, 540, 960); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 3.0, 145.0, -20.0
        base = mm._id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link"); font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 28)
        wr = imageio.get_writer(a.video, fps=25, codec="libx264", quality=8, macro_block_size=None)
        for k, (name, cmd, fam, g, B) in enumerate(rows):
            Q = traj[name]
            for i in range(0, len(Q), 2):
                d.qpos[:] = Q[i]; mujoco.mj_forward(m, d); cam.lookat[:] = [d.xpos[base][0], d.xpos[base][1], 0.35]
                r.update_scene(d, camera=cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im, "RGBA")
                dr.rectangle([0, 0, 960, 74], fill=(0, 0, 0, 150)); dr.text((14, 6), f"{k+1}/{len(rows)}  {a.title or 'RL ' + wname}  {name}", font=font, fill=(255, 255, 255))
                dr.text((14, 42), f"t {i*v3.CTRL_DT:4.1f} s   10 s 平均：vx {g['vx']:+.2f}  vy {g['vy']:+.3f}  偏航 {g['yaw']:+.0f}°/s  roll std {g['roll_std']:.2f}°  膝峰 {g['tau_knee_pk']:.0f} N·m  摔 {g['falls']}", font=ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 20), fill=(255, 235, 120))
                wr.append_data(np.asarray(im))
        wr.close(); print("→", a.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
