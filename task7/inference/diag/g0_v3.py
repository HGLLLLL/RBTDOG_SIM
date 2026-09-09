"""v3 G0：零動作（純原廠模式表）在三種指令下的行為。CPU MJX，每種 N 步。
    conda run -n rbtdog python task7/inference/diag/g0_v3.py --steps 150
"""
import argparse, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_env_v3 as v3

def run(env, cmd, steps, jit_reset, jit_step, seed=0):
    s = jit_reset(jax.random.PRNGKey(seed))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6,
                        "u_mode": v3.mode_of(jnp.array(cmd))[0]})
    a = jnp.zeros(v3.ACT_DIM)
    M = {k: [] for k in ("vx", "vy", "wz", "roll", "pitch", "clr_step", "clr_stance", "tau_pk", "err_pk", "knee_v", "height", "mode", "reward")}
    WV, WC = [], []
    done_at = None
    for i in range(steps):
        s = jit_step(s, a)
        for k in M:
            M[k].append(float(s.metrics[k]))
        WV.append(np.asarray(s.pipeline_state.qvel[v3.WHEEL_QVEL_IDX])); WC.append(np.asarray(s.info["c"]["wheel"]))
        if float(s.done) > 0 and done_at is None:
            done_at = i
    half = steps // 2
    out = {k: float(np.mean(v[half:])) for k, v in M.items()}
    out["tau_pk"] = float(np.max(M["tau_pk"])); out["err_pk"] = float(np.max(M["err_pk"])); out["knee_v"] = float(np.max(M["knee_v"]))
    out["done_at"] = done_at
    out["yaw_deg_s"] = np.degrees(out["wz"])
    out["wheel_act"] = np.mean(WV[half:], 0).round(2); out["wheel_cmd"] = np.mean(WC[half:], 0).round(2)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--vel-only", action="store_true", dest="vel_only", help="輪子純速度伺服（M11 實機已驗）；預設位置環 kp 60")
    ap.add_argument("--duty", type=float, default=None); ap.add_argument("--lift", type=float, default=None)
    ap.add_argument("--hz-turn", type=float, default=None, dest="hz_turn"); ap.add_argument("--hz-lat", type=float, default=None, dest="hz_lat"); ap.add_argument("--mu", type=float, default=None); ap.add_argument("--only", default="")
    a = ap.parse_args()
    ref = {k: v for k, v in dict(duty=a.duty, duty_lat=a.duty, lift=a.lift, lift_lat=a.lift, step_hz_turn=a.hz_turn, step_hz_lat=a.hz_lat, floor_mu=a.mu).items() if v is not None}
    env = v3.DualModeEnv(wheel_pos=not a.vel_only, ref=ref)
    print("wheel_pos", env.wheel_pos, "kp", env.kp_wheel, "kv", env.kv_wheel, "ref", ref)
    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    print("obs", env.obs_dim, "act", env.action_size)
    cases = (("WHEEL vx0.5", (0.5, 0.0, 0.0)), ("WHEEL vx0.5 wz0.3", (0.5, 0.0, 0.3)),
             ("TURN wz1.3", (0.0, 0.0, 1.3)), ("LAT vy0.06", (0.0, 0.06, 0.0)), ("STAND", (0.0, 0.0, 0.0)))
    for name, cmd in cases:
        if a.only and a.only not in name:
            continue
        t0 = time.time(); r = run(env, cmd, a.steps, jit_reset, jit_step)
        print(f"{name:18s} vx {r['vx']:+.2f} vy {r['vy']:+.3f} yaw {r['yaw_deg_s']:+.0f}°/s | roll {r['roll']:.1f} pitch {r['pitch']:.1f} h {r['height']:.3f} | "
              f"clr step {r['clr_step']:.0f} stance {r['clr_stance']:.0f} mm | 輪 cmd {r['wheel_cmd']} act {r['wheel_act']} | tau_pk {r['tau_pk']:.0f} err {r['err_pk']:.2f} knee_v {r['knee_v']:.1f} | mode {r['mode']:.1f} R {r['reward']:+.2f} | done@{r['done_at']} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
