#!/usr/bin/env python3
"""開訓前關卡（spec §6）：
1. 零訓練對齊：他的 10k ONNX 放進 WheelEnv（平地、無雜訊、無 DR、無推力）跑八指令，與 CPU rl_only_infer（圓柱平地、real）同指令並排。
2. reward 攤帳：同一批 rollout 的十三項每步平均，寫 outputs/reward_audit_wheel_v1.md。
   std 只影響 reward、不影響動力學與 obs，所以 std 0.5 的 lin／ang 由同一條 rollout 的 vx_err/vy_err/wz_err 解析算出
   （lin = 2.5·exp(−(vx_err²+vy_err²)/0.25)、ang = 1.25·exp(−wz_err²/0.25)），與重跑一次 std 0.5 逐步相等，省一半時間。
用法：conda run -n rbtdog --no-capture-output python task7/inference/diag/wheel_env_align.py [--onnx …] [--secs 8]
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rl_only_infer as ri  # noqa: E402
import wheel_terrain as wt  # noqa: E402

ROOT = HERE.parent.parent
OUT = ROOT / "outputs" / "rl_wheel"
CMDS = [(0, 0, 0), (0.3, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (0, 0.3, 0), (0, 0.5, 0), (0, 0, 0.5), (0, 0, 1.0)]
EXTRA = ("vx_err", "vy_err", "wz_err", "lin_small", "ww_force", "gap_min", "touch_n")


def run_env(onnx, cmds, secs, settle, std):
    import jax, jax.numpy as jnp, onnxruntime as ort
    import rl_env_wheel as rw
    env = rw.WheelEnv(flat=True, noise=False, push=False, fixed_cmd=True, std=std)
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]); in_name = sess.get_inputs()[0].name
    n = len(cmds)
    reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
    obs_f = jax.jit(jax.vmap(env._obs)); vb_f = jax.jit(jax.vmap(rw.quat_rot_inv))
    s = reset(jax.random.split(jax.random.PRNGKey(0), n))
    C = jnp.array(cmds, dtype=jnp.float32); Z = jnp.zeros((n, 3), dtype=jnp.float32)
    n_settle, n_run = int(settle / rw.CTRL_DT), int(secs / rw.CTRL_DT)
    rec = {k: [] for k in ("vb", "w", "fell", "gap", "clr", "T", "a")}
    t0 = time.time()
    for i in range(n_settle + n_run):
        c = Z if i < n_settle else C
        s = s.replace(info={**s.info, "cmd_fix": c, "cmd": c})        # 指令固定；reward 也對這個 cmd 算
        obs = obs_f(s.pipeline_state, c, s.info["last_a"])   # state.obs（字典）是用上一步 cmd 算的 → 以本步 cmd 重算乾淨 53 維直接餵 ONNX，不寫回 state
        a = np.stack([sess.run(None, {in_name: np.asarray(o, np.float32)[None]})[0][0] for o in np.asarray(obs)]).astype(np.float32)
        s = step(s, jnp.array(a) / rw.ACT_GAIN)
        d = s.pipeline_state
        foot = np.asarray(d.xpos[:, env.FOOT_BIDS])                                   # (n,4,3)
        rec["gap"].append(np.linalg.norm(foot[:, 2] - foot[:, 3], axis=1) - rw.WHEEL_INNER_W)   # 後輪實體間隙近似：中心距 − 0.034（與 ri.rear_gap 同式）
        rec["clr"].append(foot[:, :, 2] - foot[:, :, 2].min(1, keepdims=True))
        rec["fell"].append(np.asarray(s.metrics["fell"])); rec["a"].append(a)
        rec["T"].append({k: np.asarray(s.metrics[k]) for k in rw.TERM_KEYS + EXTRA})
        rec["vb"].append(np.asarray(vb_f(d.qpos[:, 3:7], d.qvel[:, 0:3])))
        rec["w"].append(np.asarray(d.qvel[:, 3:6]))
        if i % 50 == 0:
            print(f"  env step {i}/{n_settle + n_run}  {time.time() - t0:.0f}s", flush=True)
    R = {k: (np.array(v) if k != "T" else {kk: np.array([t[kk] for t in v]) for kk in v[0]}) for k, v in rec.items()}
    R["n_settle"] = n_settle
    return R


def env_metrics(R, j, dt=0.02):
    w = slice(R["n_settle"] + int(1.0 / dt), None)
    vb, ww, clr, a = R["vb"][w, j], R["w"][w, j], R["clr"][w, j] * 1000, R["a"][w, j]
    lifts = []
    for k in range(4):
        sg = ri.segments(clr[:, k], 8.0, dt); lifts.append(float(np.median(sg)) if sg else 0.0)
    return dict(vx=vb[:, 0].mean(), vy=vb[:, 1].mean(), yaw=math.degrees(ww[:, 2].mean()), lift=[round(x, 1) for x in lifts],
                gap=float(R["gap"][w, j].min() * 1000), a_max=float(np.abs(a).max()), fell=bool(R["fell"][:, j].any()))


def run_cpu(onnx, cmds, secs, settle, env_scene=False):
    """env_scene=False：rl_only_infer 預設（v3f 圓柱輪、平面地板）。True：WheelEnv 同一份場景（16 邊形網格輪、平地 hfield），
    CPU MuJoCo 跑——用來把「MJX vs CPU MuJoCo」和「場景（輪子幾何／hfield）」兩種差異拆開。"""
    import mujoco
    io = json.load(open(HERE.parent / "rl_wheel_io.json"))
    pol = ri.load_policy(onnx, None)
    if env_scene:
        sim = ri.Sim(io, dt=0.005, ideal=False, wheel_kv=1.0, scene=wt.make_scene_xml())
        H = wt.build_map(0, flat=True); wt.load_hfield(sim.m, H); sim.d = mujoco.MjData(sim.m)
        reset0 = sim.reset

        def reset_on_hfield(seed=0):                     # hfield 平地頂面在 H.max()（不是 z=0）
            reset0(seed); sim.d.qpos[2] += float(H.max()); mujoco.mj_forward(sim.m, sim.d)
        sim.reset = reset_on_hfield
    else:
        sim = ri.Sim(io, dt=0.005, ideal=False, wheel_kv=1.0)
    out = []
    for cmd in cmds:
        R = ri.run(sim, pol, list(cmd), secs, settle); g = ri.metrics(R, cmd)
        out.append(dict(vx=g["vx"], vy=g["vy"], yaw=g["yaw"], lift=g["lift"], a_max=g["a_max"], gap=g["gap_rr"], fell=g["fell"] is not None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT.parent / "RL_only" / "2026-10-01_17-30-20" / "2026-10-01_17-30-20_10000.onnx"))
    ap.add_argument("--secs", type=float, default=8.0); ap.add_argument("--settle", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); RC = run_cpu(a.onnx, CMDS, a.secs, a.settle); t_cpu = time.time() - t0; print(f"CPU done {t_cpu:.0f}s", flush=True)
    RS = run_cpu(a.onnx, CMDS, a.secs, a.settle, env_scene=True)
    t0 = time.time(); RE = run_env(a.onnx, CMDS, a.secs, a.settle, std=0.25); t_env = time.time() - t0; print(f"env done {t_env:.0f}s", flush=True)
    L = [f"# 零訓練對齊：他的 10k 在 WheelEnv（MJX，平地）vs CPU rl_only_infer（real）  —  {Path(a.onnx).name}", "",
         f"MJX {t_env:.0f} s（8 環境 vmap）、CPU {t_cpu:.0f} s；指令 {a.secs} s（先 {a.settle} s 零指令）；統計取指令後 1 s 起。"
         "後輪 gap ＝ 後輪實體間隙近似 ＝ 輪心距 − 0.034（官方網格內側面 +0.017 ×2，V 文件口徑；傾斜時高估 ~7 mm）（mm，取最小；env 與 cpu `gap_rr` 同式）。追蹤差 ＝ |env − cpu| / |指令|，只算有下指令的軸。",
         "cpu 欄＝rl_only_infer 預設（v3f 圓柱輪、平面地板）；cpu′ ＝ 同一支 CPU 工具改載 WheelEnv 的場景（16 邊形網格輪、平地 hfield），"
         "env vs cpu′ 只剩 MJX 與 CPU MuJoCo 的差，cpu′ vs cpu 是場景（輪子幾何）的差。", "",
         "| 指令 | 摔 env/cpu/cpu′ | vx env/cpu/cpu′ | vy env/cpu/cpu′ | 偏航 °/s env/cpu/cpu′ | 抬腳 mm env | 抬腳 mm cpu | 抬腳 mm cpu′ | 後輪 gap mm env/cpu/cpu′ | \\|a\\| max env/cpu | 追蹤差 env↔cpu | 追蹤差 env↔cpu′ |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    worst, worst_s = 0.0, 0.0

    def tdiff(e, c, cmd):
        ref = {0: ("vx", cmd[0]), 1: ("vy", cmd[1]), 2: ("yaw", math.degrees(cmd[2]))}
        diffs = [abs(e[k] - c[k]) / abs(target) * 100 for k, target in ref.values() if abs(target) > 1e-9]
        return max(diffs) if diffs else 0.0
    for j, cmd in enumerate(CMDS):
        e, c, cs = env_metrics(RE, j), RC[j], RS[j]
        dpct, dpct_s = tdiff(e, c, cmd), tdiff(e, cs, cmd); worst, worst_s = max(worst, dpct), max(worst_s, dpct_s)
        L.append(f"| vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f} | {int(e['fell'])}/{int(c['fell'])}/{int(cs['fell'])} | {e['vx']:+.2f}/{c['vx']:+.2f}/{cs['vx']:+.2f} | "
                 f"{e['vy']:+.2f}/{c['vy']:+.2f}/{cs['vy']:+.2f} | {e['yaw']:+.0f}/{c['yaw']:+.0f}/{cs['yaw']:+.0f} | {e['lift']} | {c['lift']} | {cs['lift']} | "
                 f"{e['gap']:.0f}/{c['gap']:.0f}/{cs['gap']:.0f} | {e['a_max']:.2f}/{c['a_max']:.2f} | {dpct:.1f} 個百分點 | {dpct_s:.1f} 個百分點 |")
    L += ["", f"**最大追蹤差 env↔cpu {worst:.1f} 個百分點、env↔cpu′ {worst_s:.1f} 個百分點（百分比僅供參考；關卡＝絕對差：vx 列 ≤ 0.03 m/s、wz 列 ≤ 3 °/s，vy 0.3 是他的死區不計，spec §11.8）；"
          f"摔 env {int(RE['fell'].any(0).sum())}／cpu {sum(c['fell'] for c in RC)}／cpu′ {sum(c['fell'] for c in RS)}（門檻 0）。**"]
    (OUT / "align_10k.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("→", OUT / "align_10k.md")
    # 攤帳
    import rl_env_wheel as rw
    w = slice(RE["n_settle"] + 50, None)
    T = RE["T"]
    lin5 = 2.5 * np.exp(-(T["vx_err"] ** 2 + T["vy_err"] ** 2) / 0.25); ang5 = 1.25 * np.exp(-T["wz_err"] ** 2 / 0.25)
    keys = list(T.keys())
    A = ["# reward 攤帳：他的 10k 在 WheelEnv 平地八指令（每步平均；std 0.25 為主，括號內 std 0.5）", "",
         f"取指令後 1 s 起（{a.secs - 1:.0f} s × 50 步）；前 13 欄是 reward 項（已乘權重），合計＝十三項和；`lin_small` 在 fixed_cmd 下恆 0（small 旗標固定 False）；`ww_force` 是輪輪法向力 N、`gap_min` 是前後兩對左右輪實體間隙的較小者 m（輪心距 − 0.034）、`touch_n` 是觸地輪數（診斷量，不進 reward）。"
         "std 0.5 的 lin／ang 由同一條 rollout 的 vx/vy/wz 誤差解析算出（std 不影響動力學）。", "",
         "| 指令 | " + " | ".join(keys) + " | 合計 |", "|---|" + "---|" * (len(keys) + 1)]
    for j, cmd in enumerate(CMDS):
        vals = {k: T[k][w, j].mean() for k in keys}; vals5 = {"lin": lin5[w, j].mean(), "ang": ang5[w, j].mean()}
        tot = sum(vals[k] for k in rw.TERM_KEYS)
        A.append(f"| vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f} | " + " | ".join(f"{vals[k]:+.3f}" + (f" ({vals5[k]:+.3f})" if k in vals5 else "") for k in keys) + f" | {tot:+.3f} |")
    A += ["", "讀法：`wgap` 在 vy ±0.5 要非零（他的 policy 實體間隙 ~33 mm < 80 mm；`ww` 是 pair 法向力，只有腿交叉才碰，平地恆 0 屬正常）；`lin` 在 vy 0.3／wz 0.5 這兩列 std 0.25 要明顯低於括號內的 0.5（改動二有梯度）；",
          "其餘各項量級要和他的相稱（任一項絕對值 > 2 就是淹沒追蹤項，回 spec §3 調）。"]
    (ROOT / "outputs" / "reward_audit_wheel_v1.md").write_text("\n".join(A) + "\n", encoding="utf-8"); print("→ outputs/reward_audit_wheel_v1.md")
    np.savez_compressed(OUT / "align_10k_env_raw.npz", **{k: v for k, v in RE.items() if k != "T"}, **{"T_" + k: v for k, v in T.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
