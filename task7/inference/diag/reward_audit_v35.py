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
KEYS = ("t_abadbias", "t_drift", "t_headlin", "t_step", "abad_bias", "vx_drift", "head_deg", "reward")

# v3.6（spec 2026-09-22 §3.3）：(name, cmd, 用 policy？)；零動作列＝名目要明確贏過現況（v3.2 教訓）
CASES36 = (("原地左轉 1.3", (0.0, 0.0, 1.3), True), ("左平移 0.20", (0.0, 0.20, 0.0), True),
           ("左平移 0.20 零動作", (0.0, 0.20, 0.0), False), ("直走 0.5", (0.5, 0.0, 0.0), True))
# v3.7（解耦產生器，名目本身就抬 25–40 mm）：零動作 t_step 要滿、policy 不設上限、不看 gap；直走／原地轉 t_step 必為 0
GATE37 = {"原地左轉 1.3": {"t_step": (None, 1e-6)},
          "左平移 0.20": {"t_drift": (None, 1.0)},
          "左平移 0.20 零動作": {"t_step": (1.2, None)},
          "直走 0.5": {"t_step": (None, 1e-6), "t_abadbias": (None, 0.05), "t_drift": (None, 0.10)}}
GATE36 = {"原地左轉 1.3": {"t_abadbias": (0.3, 0.8), "t_step": (None, 1e-6)},
          "左平移 0.20": {"t_step": (None, 1.0), "t_drift": (0.10, 0.35)},   # 無推力時 v3.5f vx 漂只剩 0.022（舊表 0.038 含推力）→ hinge 給 0.14
          "左平移 0.20 零動作": {"t_step": (0.7, None)},                 # 零動作每週期頂點實測 14／11 mm（不是 8 s 內的最大 21）→ r_step ≈ 0.6；另在 main 裡檢查 ≥ policy + 0.4
          "直走 0.5": {"t_step": (None, 0.05), "t_abadbias": (None, 0.05), "t_drift": (None, 0.05)}}


def run_case(env, jr, js, pol, cmd, steps):
    s = jr(jax.random.PRNGKey(0))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd), "t_switch": 10 ** 6})
    M, fell = [], False
    for _ in range(steps):
        s = js(s, pol(s.obs) if pol is not None else jnp.zeros(v3.ACT_DIM))
        M.append({k: float(s.metrics[k]) for k in KEYS})
        if float(s.done) > 0:
            fell = True; break
    seg = M[steps // 2:] or M                       # 後 5 s 平均
    return {k: float(np.mean([m[k] for m in seg])) for k in KEYS} | {"fell": fell, "n": len(M)}


def judge(name, r, gate=GATE):
    bad = []
    for k, (lo, hi) in gate[name].items():
        if lo is not None and r[k] < lo: bad.append(f"{k} {r[k]:.3f} < {lo}")
        if hi is not None and r[k] > hi: bad.append(f"{k} {r[k]:.3f} > {hi}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(INF.parent / "weights" / "cpg_rl_v3_params_2.pkl"))
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--gains", default="kp250", choices=("kp250", "factory"), help="factory＝拿 v3.4f 權重在原廠增益線上攤帳（v3.5f）")
    ap.add_argument("--weights-set", default="v35", choices=("v35", "v36", "v37", "v37b"), dest="wset", help="v36＝W36 攤帳（t_step／hinge；多一列零動作；無推力）；v37／v37b＝解耦產生器，門檻 GATE37")
    a = ap.parse_args()
    V36 = a.wset in ("v36", "v37", "v37b")
    W_ = {"v35": v3.W35, "v36": v3.W36, "v37": v3.W37, "v37b": v3.W37B}[a.wset]
    W_ = dict(W_, MIRROR_AUG=False)                  # 攤帳不抽鏡像回合
    env = v3.DualModeEnv(gains=a.gains, ref=dict(cyc_amp_rand=False), weights=W_, push=not V36)   # W35 含 CYC_TURN_SYM → 右轉鏡像；攤帳看的是 reward 項，圖案差異對 kp250/v3.4f 舊權重只影響右轉；v36 不要推力
    cases = CASES36 if V36 else tuple((n, c, True) for n, c in CASES)
    gate = {"v35": GATE, "v36": GATE36, "v37": GATE37, "v37b": GATE37}[a.wset]
    jr, js = jax.jit(env.reset), jax.jit(env.step)
    pol = L.load_policy(a.weights, env.obs_dim); steps = int(a.secs / v3.CTRL_DT)
    rows, all_ok = [], True
    for name, cmd, use_pol in cases:
        r = run_case(env, jr, js, pol if use_pol else None, cmd, steps); bad = judge(name, r, gate); all_ok &= not bad
        rows.append((name, r, bad))
        print(f"{'PASS' if not bad else 'FAIL'} {name:10s} t_abadbias {r['t_abadbias']:.3f} t_drift {r['t_drift']:.3f} t_headlin {r['t_headlin']:.3f} t_step {r['t_step']:.3f} | ABAD 漂 {r['abad_bias']:.1f}° vx 漂 {r['vx_drift']:+.3f} 航向 {r['head_deg']:+.1f}° | R/步 {r['reward']:+.2f} 摔 {r['fell']}"
              + (f"  ← {'; '.join(bad)}" if bad else ""), flush=True)
    if a.wset == "v36":                              # 名目（零動作）要明確贏過現況（policy）；v37 名目與 policy 都滿分，不看
        R_ = {n: r for n, r, _ in rows}
        gap = R_["左平移 0.20 零動作"]["t_step"] - R_["左平移 0.20"]["t_step"]
        bad = [f"gap {gap:.2f} < 0.4：名目沒有明確贏過現況"] if gap < 0.4 else []
        all_ok &= not bad
        rows.append(("零動作 − policy 的 t_step", {k: 0.0 for k in KEYS} | {"t_step": gap, "fell": False}, bad))
        print(f"{'PASS' if not bad else 'FAIL'} 零動作 − policy 的 t_step gap {gap:+.3f}（門檻 ≥ 0.4）")
    wname = Path(a.weights).stem
    out = [f"# {'v3.6' if V36 else 'v3.5'} 攤帳（spec §3.3）：權重 `{wname}`（gains={a.gains}）在 {a.wset.upper()} 下，後 5 s 每步平均{'，無推力' if V36 else ''}", "",
           "| 指令 | t_abadbias | t_drift | t_headlin | t_step | ABAD 漂 ° | vx 漂 m/s | 航向 ° | reward/步 | 摔 | 門檻 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r, bad in rows:
        out.append(f"| {name} | {r['t_abadbias']:.3f} | {r['t_drift']:.3f} | {r['t_headlin']:.3f} | {r['t_step']:.3f} | {r['abad_bias']:.1f} | {r['vx_drift']:+.3f} | {r['head_deg']:+.1f} | {r['reward']:+.2f} | {r['fell']} | {'PASS' if not bad else 'FAIL：' + '; '.join(bad)} |")
    if a.wset in ("v37", "v37b"):
        out += ["", "門檻（v3.7）：零動作 t_step ≥ 1.2（產生器本身抬 25–40 mm）、平移 t_drift ≤ 1.0、直走／原地轉 t_step = 0、直走 t_abadbias ≤ 0.05 且 t_drift ≤ 0.10。"]
        p = INF.parent / "outputs" / f"reward_audit_{a.wset}.md"
    elif V36:
        out += ["", "門檻（spec 2026-09-22 §3.3）：左平移 policy t_step ≤ 1.0、零動作 t_step ≥ 0.7 且比 policy 高 ≥ 0.4、t_drift 0.10–0.35；原地轉 t_abadbias 0.3–0.8、t_step = 0；直走三項 ≤ 0.05。全過 → 可上 Colab。"]
        p = INF.parent / "outputs" / "reward_audit_v36.md"
    else:
        out += ["", "門檻：原地轉 t_abadbias ≥ 0.5、平移 t_drift ≥ 0.10 且 t_headlin ≤ 0.8、直走兩項 ≤ 0.05。全過 → 新 reward 咬在對的地方、可上 Colab。"]
        p = INF.parent / "outputs" / ("reward_audit_v35.md" if a.gains == "kp250" else "reward_audit_v35f.md")
    p.write_text("\n".join(out) + "\n", encoding="utf-8"); print("→", p)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
