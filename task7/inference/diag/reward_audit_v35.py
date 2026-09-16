"""v3.5 攤帳（spec §3.3）：拿 v3.3 的 params_2 在 W35 權重下跑三個指令各 10 s（seed 0），看新兩項是不是「咬在對的地方」。
門檻：原地轉 t_abadbias ≥ 0.5（BL 漂 8–15°）、平移 0.20 t_drift ≥ 0.10（機身系 vx 漂 0.02–0.04 ＋ vy 少追 0.03–0.05；世界系的 0.08 有七成是航向轉掉的投影）、直走兩項都 ≤ 0.05。
    conda run -n rbtdog --no-capture-output python task7/inference/diag/reward_audit_v35.py
輸出 outputs/reward_audit_v35.md；退出碼 1 = 沒過，不要上 Colab。
"""
import argparse, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np

INF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INF))
import rl_env_v3 as v3          # noqa: E402
import local_infer_v3 as L      # noqa: E402

CASES = (("原地左轉 1.3", (0.0, 0.0, 1.3)), ("左平移 0.20", (0.0, 0.20, 0.0)), ("直走 0.5", (0.5, 0.0, 0.0)))
# (下限, 上限)；None = 不管
GATE = {"原地左轉 1.3": {"t_abadbias": (0.5, None)},
        "左平移 0.20": {"t_drift": (0.10, None), "t_headlin": (None, 0.8)},     # 航向平均誤差 > 5.7° 才算有東西可咬
        "直走 0.5": {"t_abadbias": (None, 0.05), "t_drift": (None, 0.05)}}
KEYS = ("t_abadbias", "t_drift", "t_headlin", "abad_bias", "vx_drift", "head_deg", "reward")


def run_case(env, jr, js, pol, cmd, steps):
    s = jr(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    M, fell = [], False
    for _ in range(steps):
        s = js(s, pol(s.obs))
        M.append({k: float(s.metrics[k]) for k in KEYS})
        if float(s.done) > 0:
            fell = True; break
    seg = M[steps // 2:] or M                       # 後 5 s 平均
    return {k: float(np.mean([m[k] for m in seg])) for k in KEYS} | {"fell": fell, "n": len(M)}


def judge(name, r):
    bad = []
    for k, (lo, hi) in GATE[name].items():
        if lo is not None and r[k] < lo: bad.append(f"{k} {r[k]:.3f} < {lo}")
        if hi is not None and r[k] > hi: bad.append(f"{k} {r[k]:.3f} > {hi}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(INF.parent / "weights" / "cpg_rl_v3_params_2.pkl"))
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--gains", default="kp250", choices=("kp250", "factory"), help="factory＝拿 v3.4f 權重在原廠增益線上攤帳（v3.5f）")
    a = ap.parse_args()
    env = v3.DualModeEnv(gains=a.gains, ref=dict(cyc_amp_rand=False), weights=v3.W35)
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    pol = L.load_policy(a.weights, env.obs_dim); steps = int(a.secs / v3.CTRL_DT)
    rows, all_ok = [], True
    for name, cmd in CASES:
        r = run_case(env, jr, js, pol, cmd, steps); bad = judge(name, r); all_ok &= not bad
        rows.append((name, r, bad))
        print(f"{'PASS' if not bad else 'FAIL'} {name:10s} t_abadbias {r['t_abadbias']:.3f} t_drift {r['t_drift']:.3f} t_headlin {r['t_headlin']:.3f} | ABAD 漂 {r['abad_bias']:.1f}° vx 漂 {r['vx_drift']:+.3f} 航向 {r['head_deg']:+.1f}° | R/步 {r['reward']:+.2f} 摔 {r['fell']}"
              + (f"  ← {'; '.join(bad)}" if bad else ""), flush=True)
    wname = Path(a.weights).stem
    out = [f"# v3.5 攤帳（spec §3.3）：權重 `{wname}`（gains={a.gains}）在 W35 下，後 5 s 每步平均", "",
           "| 指令 | t_abadbias | t_drift | t_headlin | ABAD 漂 ° | vx 漂 m/s | 航向 ° | reward/步 | 摔 | 門檻 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r, bad in rows:
        out.append(f"| {name} | {r['t_abadbias']:.3f} | {r['t_drift']:.3f} | {r['t_headlin']:.3f} | {r['abad_bias']:.1f} | {r['vx_drift']:+.3f} | {r['head_deg']:+.1f} | {r['reward']:+.2f} | {r['fell']} | {'PASS' if not bad else 'FAIL：' + '; '.join(bad)} |")
    out += ["", "門檻：原地轉 t_abadbias ≥ 0.5、平移 t_drift ≥ 0.10 且 t_headlin ≤ 0.8、直走兩項 ≤ 0.05。全過 → 新 reward 咬在對的地方、可上 Colab。"]
    p = INF.parent / "outputs" / ("reward_audit_v35.md" if a.gains == "kp250" else "reward_audit_v35f.md"); p.write_text("\n".join(out) + "\n", encoding="utf-8"); print("→", p)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
