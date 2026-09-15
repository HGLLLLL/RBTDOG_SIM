"""v3.1 G0：零動作（純開迴路產生器）在六種指令下的行為，對 spec §5.1 門檻判 PASS/FAIL。CPU MJX。
    conda run -n rbtdog python task7/inference/diag/g0_v3.py --steps 200
    conda run -n rbtdog python task7/inference/diag/g0_v3.py --sweep --only TURN     # rot_frac × duty × hz
    conda run -n rbtdog python task7/inference/diag/g0_v3.py --posture 0 --only ARC
"""
import argparse, itertools, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_env_v3 as v3

CASES = (("WHEEL vx0.5", (0.5, 0.0, 0.0)), ("ARC vx0.5 wz0.5", (0.5, 0.0, 0.5)), ("TURN wz1.3", (0.0, 0.0, 1.3)),
         ("LAT vy0.08", (0.0, 0.08, 0.0)), ("DIAG vx0.3 vy0.06", (0.3, 0.06, 0.0)), ("STAND", (0.0, 0.0, 0.0)))
KEYS = ("vx", "vy", "wz", "roll", "pitch", "clr_step", "clr_stance", "tau_pk", "err_pk", "knee_v", "height", "s4", "s_arc", "reward")


def run(env, cmd, steps, jit_reset, jit_step, seed=0):
    s = jit_reset(jax.random.PRNGKey(seed))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    a = jnp.zeros(v3.ACT_DIM)
    M = {k: [] for k in KEYS}; WV, WC, LIFT = [], [], []
    done_at = None
    for i in range(steps):
        s = jit_step(s, a)
        for k in M:
            M[k].append(float(s.metrics[k]))
        WV.append(np.asarray(s.pipeline_state.qvel[v3.WHEEL_QVEL_IDX])); WC.append(np.asarray(s.info["c"]["wheel"]))
        LIFT.append(np.asarray(env._wheel_clearance(s.pipeline_state)) * 1000.0)
        if float(s.done) > 0 and done_at is None:
            done_at = i
    half = steps // 2
    out = {k: float(np.mean(v[half:])) for k, v in M.items()}
    out["tau_pk"] = float(np.max(M["tau_pk"])); out["err_pk"] = float(np.max(M["err_pk"])); out["knee_v"] = float(np.max(M["knee_v"]))
    out["roll_max"] = float(np.max(M["roll"][half:]))
    out["wz_std"] = float(np.degrees(np.std(M["wz"][half:])))
    out["roll_std"] = float(np.std(M["roll"][half:]))
    out["done_at"] = done_at
    out["yaw_deg_s"] = np.degrees(out["wz"])
    out["wheel_act"] = np.mean(WV[half:], 0).round(2); out["wheel_cmd"] = np.mean(WC[half:], 0).round(2)
    out["lift_max"] = np.max(LIFT[half:], 0).round(0)          # 每腿 (FR,FL,RR,RL) 最大離地 mm
    return out


def verdict(name, r, yaw_scale=1.0):
    """spec §5.1；yaw_scale 是步驟 0 回放決定的打折（8.1：不打折）。"""
    ok = r["done_at"] is None and r["tau_pk"] < v3.TAU_KILL
    lm = r["lift_max"]
    if name.startswith("WHEEL"):
        ok &= r["vx"] >= 0.45 and lm.max() < 10
    elif name.startswith("ARC"):
        ok &= r["yaw_deg_s"] >= 20 * yaw_scale and lm[1] > 12            # 內側前腿 FL
    elif name.startswith("TURN"):
        ok &= r["yaw_deg_s"] >= 27 * yaw_scale and (lm > 12).sum() == 4 and r["tau_pk"] <= 75
    elif name.startswith("LAT"):
        # spec §8.2 定案：開迴路 |vy| ≥ 指令 60%、力矩 ≤ 85、側傾峰 ≤ 8°（原廠實測 2–3° 是閉迴路結果，交給 RL）
        vy_cmd = float(name.split("vy")[1])
        ok &= abs(r["vy"]) >= 0.6 * vy_cmd and r["tau_pk"] <= 85 and r["roll_max"] <= 8.0 and r["roll_std"] <= 1.5
    elif name.startswith("DIAG"):
        ok &= r["vx"] >= 0.15 and r["vy"] >= 0.03
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--wheel-pos", action="store_true", dest="wheel_pos", help="用 kp 60 位置環模型（v3.1 預設不用）")
    ap.add_argument("--rot-frac", type=float, default=None, dest="rot_frac"); ap.add_argument("--lift", type=float, default=None)
    ap.add_argument("--duty", type=float, default=None); ap.add_argument("--hz", type=float, default=None)
    ap.add_argument("--posture", type=int, default=None, help="1 開 0 關（弧線姿態偏移）"); ap.add_argument("--posture-lat", type=float, default=None, dest="posture_lat"); ap.add_argument("--lat-gain", type=float, default=None, dest="lat_gain"); ap.add_argument("--lift-lat", type=float, default=None, dest="lift_lat")
    ap.add_argument("--cyc-amp-turn", type=float, default=None, dest="cyc_amp_turn"); ap.add_argument("--cyc-amp-lat", type=float, default=None, dest="cyc_amp_lat")
    ap.add_argument("--gen-turn", default=None, dest="gen_turn", help="cycle | kin"); ap.add_argument("--gen-lat", default=None, dest="gen_lat")
    ap.add_argument("--cyc-abs", action="store_true", dest="cyc_abs", help="週期用原廠絕對 des（不對中到我們站姿）")
    ap.add_argument("--cyc-wheel", type=float, default=None, dest="cyc_wheel"); ap.add_argument("--cyc-hz", type=float, default=None, dest="cyc_hz"); ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--yaw-scale", type=float, default=1.0, dest="yaw_scale"); ap.add_argument("--only", default="")
    ap.add_argument("--sweep", action="store_true", help="rot_frac {0.2,0.35,0.5} × duty {0.5,0.6,0.7} × hz {2.0,1.5}")
    ap.add_argument("--sweep-rot", default="0.2,0.35,0.5", dest="sweep_rot"); ap.add_argument("--sweep-duty", default="0.5,0.6,0.7", dest="sweep_duty")
    ap.add_argument("--sweep-hz", default="2.0,1.5", dest="sweep_hz")
    ap.add_argument("--phase", default=None, help="factory | trot（兩族一起）")
    ap.add_argument("--phase-turn", default=None, dest="phase_turn"); ap.add_argument("--phase-lat", default=None, dest="phase_lat")
    ap.add_argument("--hz-turn", type=float, default=None, dest="hz_turn"); ap.add_argument("--hz-lat", type=float, default=None, dest="hz_lat")
    ap.add_argument("--duty-turn", type=float, default=None, dest="duty_turn"); ap.add_argument("--duty-lat", type=float, default=None, dest="duty_lat")
    ap.add_argument("--cases", default="", help="逗號分隔的 case 名前綴（WHEEL,ARC,TURN,LAT,DIAG,STAND）；空＝全部")
    ap.add_argument("--lat-vy", default="", dest="lat_vy", help="額外的 LAT 速度，逗號分隔，例 0.04,0.06")
    a = ap.parse_args()
    base = {k: v for k, v in dict(rot_step_frac=a.rot_frac, lift=a.lift, floor_mu=a.mu,
                                  step_hz_turn=a.hz or a.hz_turn, step_hz_lat=a.hz or a.hz_lat, duty_turn=a.duty or a.duty_turn, duty_lat=a.duty or a.duty_lat,
                                  phase_set_turn=a.phase or a.phase_turn, phase_set_lat=a.phase or a.phase_lat).items() if v is not None}
    for k, v in dict(cyc_amp_turn=a.cyc_amp_turn, cyc_amp_lat=a.cyc_amp_lat, step_gen_turn=a.gen_turn, step_gen_lat=a.gen_lat, cyc_wheel=a.cyc_wheel, cyc_hz_scale=a.cyc_hz).items():
        if v is not None:
            base[k] = v
    if a.cyc_abs:
        base["cyc_recenter"] = False
    if a.lift_lat is not None:
        base["lift_lat"] = a.lift_lat
    if a.lat_gain is not None:
        base["lat_cmd_gain"] = a.lat_gain
    if a.posture_lat is not None:
        base["posture_lat"] = a.posture_lat
    if a.posture is not None:
        base["posture_arc"] = dict(front=0.06, rear=-0.03) if a.posture else dict(front=0.0, rear=0.0)
    cases = list(CASES) + [(f"LAT vy{v}", (0.0, float(v), 0.0)) for v in a.lat_vy.split(",") if v]
    if a.cases:
        want = a.cases.split(","); cases = [c for c in cases if any(c[0].startswith(w) for w in want)]
    grid = [base]
    if a.sweep:
        f = lambda s: [float(x) for x in s.split(",")]   # noqa: E731
        grid = [dict(base, rot_step_frac=rf, duty_turn=d, duty_lat=d, step_hz_turn=hz, step_hz_lat=hz) for hz, d, rf in itertools.product(f(a.sweep_hz), f(a.sweep_duty), f(a.sweep_rot))]
    for ref in grid:
        env = v3.DualModeEnv(wheel_pos=a.wheel_pos, ref=ref)
        jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
        print(f"\n== ref {ref}  wheel_pos {env.wheel_pos}  obs {env.obs_dim} act {env.action_size}", flush=True)
        for name, cmd in cases:
            if a.only and a.only not in name:
                continue
            if a.sweep and not a.only and not a.cases and not (name.startswith("TURN") or name.startswith("LAT")):
                continue
            t0 = time.time(); r = run(env, cmd, a.steps, jit_reset, jit_step)
            print(f"{verdict(name, r, a.yaw_scale)} {name:18s} vx {r['vx']:+.2f} vy {r['vy']:+.3f} yaw {r['yaw_deg_s']:+.0f}±{r['wz_std']:.0f}°/s | roll {r['roll']:.1f}/{r['roll_max']:.1f}(std {r['roll_std']:.1f}) pitch {r['pitch']:.1f} h {r['height']:.3f} | "
                  f"lift {r['lift_max']} clr_step {r['clr_step']:.0f} | 輪 cmd {r['wheel_cmd']} act {r['wheel_act']} | tau_pk {r['tau_pk']:.0f} err {r['err_pk']:.2f} knee_v {r['knee_v']:.1f} | "
                  f"s4 {r['s4']:.2f} arc {r['s_arc']:.2f} R {r['reward']:+.2f} | done@{r['done_at']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
