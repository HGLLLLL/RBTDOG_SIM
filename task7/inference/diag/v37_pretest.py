"""v3.7 前的完整實驗（2026-09-22，使用者要求「確定會進步再改」）。全部本機 CPU MJX、零動作或現有權重。
    conda run -n rbtdog --no-capture-output python task7/inference/diag/v37_pretest.py --exp E1 E2 E3 E4 E5
E1 分關節縮放掃描（零動作）：scale × vy × 左右 × 種子 → 每週期抬腳、vy、roll std、τ峰、摔
E2 候選比例強健性（零動作）：摩擦 0.4／1.4、ABAD kp ×0.7、開推力、20 s
E3 斜走與模式切換（零動作）：DIAG、LAT→STAND→反向 LAT
E4 新 REF 攤帳：名目 t_step；v3.6f 權重零樣本套新 REF（t_step、vy、摔）
E5 右轉弱 × 質心：v3.5f／v3.6f 在質心 y = +1.5（現況）／0／−1.5 mm 下的左右轉偏航
輸出 outputs/v37_pretest.md（附加模式，每個 exp 一節）。"""
import argparse, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

INF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INF))
import rl_env_v3 as v3          # noqa: E402
import local_infer_v3 as L      # noqa: E402

OUT = INF.parent / "outputs" / "v37_pretest.md"
CAND = (0.7, 1.4, 1.4)
ABAD, HIP, KNEE = [0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]


def make_env(scale=None, push=False, floor_mu=None, abad_kp=1.0, com_y_mm=None):
    ref = dict(cyc_amp_rand=False)
    if scale is not None: ref["cyc_joint_scale_lat"] = tuple(scale)
    if floor_mu is not None: ref["floor_mu"] = floor_mu
    env = v3.DualModeEnv(gains="factory", weights=v3.W36, ref=ref, push=push)
    if abad_kp != 1.0:
        idx = jnp.array(v3.LEG_ACT_IDX)[jnp.array(ABAD)]
        g = env.sys.actuator_gainprm.at[idx, 0].multiply(abad_kp); b = env.sys.actuator_biasprm.at[idx, 1].multiply(abad_kp)
        env.sys = env.sys.replace(actuator_gainprm=g, actuator_biasprm=b)
    if com_y_mm is not None:                      # 整機質心 y 設到指定值（現況 +1.5）
        base = int(v3._BASE_ID); dy = (com_y_mm - 1.5) / 1000.0 * v3.COM_Y_SCALE
        env.sys = env.sys.replace(body_ipos=env.sys.body_ipos.at[base, 1].add(dy))
    return env, jax.jit(env.reset), jax.jit(env.step)


def roll(env, jr, js, cmd, secs=10.0, pol=None, seed=0, cmd2=None, t_switch=None):
    s = jr(jax.random.PRNGKey(seed))
    s = s.replace(info={**s.info, "cmd": jnp.array(cmd), "cmd2": jnp.array(cmd2 or cmd), "t_switch": (t_switch if t_switch is not None else 10 ** 6)})
    AP, PHI, CLR, M, TAU = [], [], [], [], []; fell = None
    for i in range(int(secs / v3.CTRL_DT)):
        s = js(s, pol(s.obs) if pol else jnp.zeros(v3.ACT_DIM)); d = s.pipeline_state
        AP.append(np.asarray(s.info["apex_last"]) * 1000); PHI.append(float(s.info["phi_cyc"]))
        CLR.append(np.asarray(env._wheel_clearance(d)) * 1000); TAU.append(np.abs(np.asarray(d.actuator_force[v3.LEG_ACT_IDX])))
        M.append({k: float(s.metrics[k]) for k in ("vx", "vy", "wz", "roll", "t_step", "abad_bias", "vx_drift", "head_deg")})
        if float(s.done) > 0: fell = round(i * v3.CTRL_DT, 1); break
    h = min(100, len(M) - 1)
    AP, PHI = np.array(AP), np.array(PHI); w = np.where(np.diff(PHI) < 0)[0]; w = w[w >= h]
    pc = AP[np.clip(w + 1, 0, len(AP) - 1)] if len(w) else np.zeros((1, 4))
    tau = np.array(TAU[h:]) if len(TAU) > h else np.zeros((1, 12))
    g = lambda k: np.array([m[k] for m in M[h:]]) if len(M) > h else np.zeros(1)   # noqa: E731
    return dict(apex=[round(float(x), 1) for x in pc.mean(0)], vx=float(g("vx").mean()), vy=float(g("vy").mean()), yaw=float(np.degrees(g("wz").mean())),
                roll_std=float(g("roll").std()), knee=float(tau[:, KNEE].max()), hip=float(tau[:, HIP].max()), abad=float(tau[:, ABAD].max()),
                steps=L._steps_per_sec(np.array(CLR[h:]), max(1, len(CLR) - h) * v3.CTRL_DT) if len(CLR) > h else [0] * 4,
                t_step=float(g("t_step")[-250:].mean()), abad_bias=float(g("abad_bias")[-250:].mean()), head=float(M[-1]["head_deg"]), fell=fell)


def fmt(r):
    return f"頂點 {r['apex']} | vy {r['vy']:+.3f} | roll std {r['roll_std']:.2f} | τ 膝/髖/ABAD {r['knee']:.0f}/{r['hip']:.0f}/{r['abad']:.0f} | 步/秒 {r['steps']} | 摔 {r['fell']}"


def primary(apex, vy):          # 主動側兩腿（左移 FL/RL = idx 1,3；右移 FR/RR = idx 0,2）
    return (apex[1], apex[3]) if vy > 0 else (apex[0], apex[2])


def E1(md):
    md += ["## E1 分關節縮放掃描（零動作、無推力、10 s、2 種子平均；主動側兩腿頂點 mm）", "",
           "| (ABAD,髖,膝) | vy 指令 | 左：頂點 / vy / roll std / τ膝/髖/ABAD / 摔 | 右：同 |", "|---|---|---|---|"]
    for sc in [(1.0, 1.0, 1.0), (0.8, 1.3, 1.3), CAND, (0.6, 1.4, 1.4), (0.7, 1.6, 1.6)]:
        env, jr, js = make_env(sc)
        for vy in (0.04, 0.08, 0.12, 0.20, 0.30):
            cells = []
            for sgn in (1, -1):
                R = [roll(env, jr, js, (0.0, sgn * vy, 0.0), seed=k) for k in range(2)]
                ap = np.mean([primary(r["apex"], sgn * vy) for r in R], 0).round(0)
                cells.append(f"{ap[0]:.0f}/{ap[1]:.0f} / {np.mean([abs(r['vy']) for r in R]):.3f} / {np.mean([r['roll_std'] for r in R]):.2f} / "
                             f"{max(r['knee'] for r in R):.0f}/{max(r['hip'] for r in R):.0f}/{max(r['abad'] for r in R):.0f} / {sum(r['fell'] is not None for r in R)}")
            md.append(f"| {sc} | {vy:.2f} | {cells[0]} | {cells[1]} |"); print(md[-1], flush=True)
    return md


def E2(md):
    md += ["", f"## E2 候選 {CAND} 的強健性（零動作、vy 0.20 左／右、10 s 除註明；訓練 DR 會抽到的條件）", "",
           "| 條件 | 左 | 右 |", "|---|---|---|"]
    conds = [("基準", dict()), ("摩擦 0.4", dict(floor_mu=0.4)), ("摩擦 1.4", dict(floor_mu=1.4)), ("ABAD kp ×0.7", dict(abad_kp=0.7)),
             ("開推力（每 2 s 最大 0.6 m/s）", dict(push=True)), ("20 s", dict())]
    for name, kw in conds:
        secs = 20.0 if name == "20 s" else 10.0
        env, jr, js = make_env(CAND, **kw)
        cells = [fmt(roll(env, jr, js, (0.0, sgn * 0.20, 0.0), secs=secs, seed=0)) for sgn in (1, -1)]
        md.append(f"| {name} | {cells[0]} | {cells[1]} |"); print(md[-1], flush=True)
    return md


def E3(md):
    md += ["", f"## E3 斜走與模式切換（零動作、候選 {CAND} vs 現況）", "", "| 比例 | 情境 | 結果 |", "|---|---|---|"]
    for sc in [(1.0, 1.0, 1.0), CAND]:
        env, jr, js = make_env(sc)
        r = roll(env, jr, js, (0.3, 0.15, 0.0)); md.append(f"| {sc} | 斜走 0.3+0.15 | vx {r['vx']:+.2f} {fmt(r)} |"); print(md[-1], flush=True)
        r = roll(env, jr, js, (0.0, 0.20, 0.0), secs=12.0, cmd2=(0.0, 0.0, 0.0), t_switch=250); md.append(f"| {sc} | 左平移 5 s → 站立 7 s | {fmt(r)} |"); print(md[-1], flush=True)
        r = roll(env, jr, js, (0.0, 0.20, 0.0), secs=12.0, cmd2=(0.0, -0.20, 0.0), t_switch=250); md.append(f"| {sc} | 左平移 5 s → 右平移 7 s | {fmt(r)} |"); print(md[-1], flush=True)
        r = roll(env, jr, js, (0.0, 0.20, 0.0), secs=12.0, cmd2=(0.5, 0.0, 0.0), t_switch=250); md.append(f"| {sc} | 左平移 5 s → 直走 7 s | vx {r['vx']:+.2f} {fmt(r)} |"); print(md[-1], flush=True)
    return md


def E4(md):
    md += ["", f"## E4 新 REF（候選 {CAND}）攤帳：名目 t_step 與 v3.6f 權重零樣本", "", "| 比例 | 動作 | 指令 | t_step | vy | 頂點 | 摔 |", "|---|---|---|---|---|---|---|"]
    for sc in [(1.0, 1.0, 1.0), CAND]:
        env, jr, js = make_env(sc); pol = L.load_policy(str(INF.parent / "weights" / "cpg_rl_v3_6f_params.pkl"), env.obs_dim)
        for name, p in (("零動作", None), ("v3.6f", pol)):
            for cmd in ((0.0, 0.20, 0.0), (0.0, -0.20, 0.0), (0.0, 0.12, 0.0), (0.3, 0.15, 0.0)):
                r = roll(env, jr, js, cmd, pol=p); md.append(f"| {sc} | {name} | {cmd} | {r['t_step']:.2f} | {r['vy']:+.3f} | {r['apex']} | {r['fell']} |"); print(md[-1], flush=True)
    return md


def E5(md):
    md += ["", "## E5 右轉弱 × 質心：質心 y 設 +1.5（現況）／0／−1.5 mm，左右原地轉 1.3（2 種子平均，無推力）", "",
           "| 權重 | 質心 y | 左轉 偏航 / roll std / 摔 | 右轉 同 | 左右差 |", "|---|---|---|---|---|"]
    for wname in ("cpg_rl_v3_5f_params", "cpg_rl_v3_6f_params"):
        W_ = v3.W35 if "5f" in wname else v3.W36
        for com in (1.5, 0.0, -1.5):
            env, jr, js = make_env(com_y_mm=com); env.w = dict(W_); pol = L.load_policy(str(INF.parent / "weights" / f"{wname}.pkl"), env.obs_dim)
            cells, yaws = [], []
            for sgn in (1, -1):
                R = [roll(env, jr, js, (0.0, 0.0, sgn * 1.3), pol=pol, seed=k) for k in range(2)]
                y = np.mean([r["yaw"] for r in R]); yaws.append(abs(y))
                cells.append(f"{y:+.1f} / {np.mean([r['roll_std'] for r in R]):.2f} / {sum(r['fell'] is not None for r in R)}")
            md.append(f"| {wname[7:12]} | {com:+.1f} | {cells[0]} | {cells[1]} | {abs(yaws[0] - yaws[1]) / max(yaws) * 100:.0f}% |"); print(md[-1], flush=True)
    return md


def E6(md):
    """解耦：把 amp_l 釘成 1（cyc_lat_amp_clip=(1,1)、cyc_amp_lat=1），髖膝固定 1.87× 表（＝候選在 0.20 的等效值），只掃 ABAD 絕對倍率。"""
    md += ["", "## E6 解耦掃描（零動作、無推力、10 s）：髖膝固定 1.87× 表，ABAD 絕對倍率 × 指令 → 速度曲線", "",
           "| ABAD× | vy 指令 | 左：頂點 / vy / roll std / τ膝/髖/ABAD / 摔 | 右：同 |", "|---|---|---|---|"]
    for k in (0.3, 0.45, 0.6, 0.75, 0.94, 1.2):
        ref = dict(cyc_amp_rand=False, cyc_lat_amp_clip=(1.0, 1.0), cyc_amp_lat=1.0, cyc_joint_scale_lat=(k, 1.87, 1.87))
        env = v3.DualModeEnv(gains="factory", weights=v3.W36, ref=ref, push=False); jr, js = jax.jit(env.reset), jax.jit(env.step)
        for vy in (0.04, 0.12, 0.20, 0.30):
            cells = []
            for sgn in (1, -1):
                r = roll(env, jr, js, (0.0, sgn * vy, 0.0)); ap = primary(r["apex"], sgn * vy)
                cells.append(f"{ap[0]:.0f}/{ap[1]:.0f} / {abs(r['vy']):.3f} / {r['roll_std']:.2f} / {r['knee']:.0f}/{r['hip']:.0f}/{r['abad']:.0f} / {r['fell']}")
            md.append(f"| {k} | {vy:.2f} | {cells[0]} | {cells[1]} |"); print(md[-1], flush=True)
    return md


def E7(md):
    md += ["", "## E7 零動作原地轉 × 質心（無推力、10 s）：倒地時間與倒前偏航，看模型除了質心還有沒有別的左右不對稱", "",
           "| 質心 y | 左轉：倒 s / 偏航 / roll std | 右轉：同 |", "|---|---|---|"]
    for com in (1.5, 0.0, -1.5):
        env, jr, js = make_env(com_y_mm=com)
        cells = []
        for sgn in (1, -1):
            r = roll(env, jr, js, (0.0, 0.0, sgn * 1.3)); cells.append(f"{r['fell']} / {r['yaw']:+.1f} / {r['roll_std']:.2f}")
        md.append(f"| {com:+.1f} | {cells[0]} | {cells[1]} |"); print(md[-1], flush=True)
    return md


def E9(md):
    """解耦產生器（REF cyc_lat_decouple=True）零動作全掃：速度 × 左右 × 2 種子；強健性；斜走／切換；名目 t_step。"""
    D = dict(cyc_lat_decouple=True)
    md += ["", "## E9 解耦產生器 `cyc_lat_decouple=True`（髖膝 1.87×、ABAD k=clip(0.3+6.4(|vy|−0.10), 0.3, 1.0)）零動作、無推力", "",
           "| vy 指令 | 左：頂點 / vy / roll std / τ膝/髖/ABAD / 摔 | 右：同 | t_step |", "|---|---|---|---|"]
    def mk(**kw):
        ref = dict(cyc_amp_rand=False, **D)
        if "floor_mu" in kw: ref["floor_mu"] = kw.pop("floor_mu")
        env = v3.DualModeEnv(gains="factory", weights=v3.W36, ref=ref, push=kw.pop("push", False))
        if kw.get("abad_kp", 1.0) != 1.0:
            idx = jnp.array(v3.LEG_ACT_IDX)[jnp.array(ABAD)]
            env.sys = env.sys.replace(actuator_gainprm=env.sys.actuator_gainprm.at[idx, 0].multiply(kw["abad_kp"]), actuator_biasprm=env.sys.actuator_biasprm.at[idx, 1].multiply(kw["abad_kp"]))
        return env, jax.jit(env.reset), jax.jit(env.step)
    env, jr, js = mk()
    for vy in (0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30):
        cells, ts = [], []
        for sgn in (1, -1):
            R = [roll(env, jr, js, (0.0, sgn * vy, 0.0), seed=k) for k in range(2)]
            ap = np.mean([primary(r["apex"], sgn * vy) for r in R], 0).round(0); ts.append(np.mean([r["t_step"] for r in R]))
            cells.append(f"{ap[0]:.0f}/{ap[1]:.0f} / {np.mean([abs(r['vy']) for r in R]):.3f} / {np.mean([r['roll_std'] for r in R]):.2f} / "
                         f"{max(r['knee'] for r in R):.0f}/{max(r['hip'] for r in R):.0f}/{max(r['abad'] for r in R):.0f} / {sum(r['fell'] is not None for r in R)}")
        md.append(f"| {vy:.2f} | {cells[0]} | {cells[1]} | {np.mean(ts):.2f} |"); print(md[-1], flush=True)
    md += ["", "強健性與情境（vy 0.20 左／右除註明）：", "", "| 條件 | 左 | 右 |", "|---|---|---|"]
    for name, kw in (("摩擦 0.4", dict(floor_mu=0.4)), ("摩擦 1.4", dict(floor_mu=1.4)), ("ABAD kp ×0.7", dict(abad_kp=0.7)), ("開推力", dict(push=True)), ("20 s", dict())):
        e2, jr2, js2 = mk(**kw); secs = 20.0 if name == "20 s" else 10.0
        cells = [fmt(roll(e2, jr2, js2, (0.0, sgn * 0.20, 0.0), secs=secs)) for sgn in (1, -1)]
        md.append(f"| {name} | {cells[0]} | {cells[1]} |"); print(md[-1], flush=True)
    for name, cmd, kw in (("斜走 0.3+0.15", (0.3, 0.15, 0.0), {}), ("斜走 0.3−0.15", (0.3, -0.15, 0.0), {}),
                          ("左平移 5 s → 站立 7 s", (0.0, 0.20, 0.0), dict(secs=12.0, cmd2=(0.0, 0.0, 0.0), t_switch=250)),
                          ("左平移 5 s → 右平移 7 s", (0.0, 0.20, 0.0), dict(secs=12.0, cmd2=(0.0, -0.20, 0.0), t_switch=250)),
                          ("左平移 5 s → 直走 7 s", (0.0, 0.20, 0.0), dict(secs=12.0, cmd2=(0.5, 0.0, 0.0), t_switch=250)),
                          ("原地左轉（不該受影響）", (0.0, 0.0, 1.3), {}), ("弧線（不該受影響）", (0.5, 0.0, 0.5), {})):
        r = roll(env, jr, js, cmd, **kw); md.append(f"| {name} | vx {r['vx']:+.2f} yaw {r['yaw']:+.0f} {fmt(r)} | |"); print(md[-1], flush=True)
    return md


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", nargs="+", default=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E9"]); a = ap.parse_args()
    md = [] if OUT.exists() and a.exp != ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E9"] else ["# v3.7 前實驗（2026-09-22）—— 分關節縮放 (0.7, 1.4, 1.4) 是否值得訓練", ""]
    for e in a.exp:
        t0 = time.time(); md = globals()[e](md); md.append(f"（{e} {time.time() - t0:.0f} s）")
    with OUT.open("a" if md and not md[0].startswith("# v3.7") else "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("→", OUT)


if __name__ == "__main__":
    main()
