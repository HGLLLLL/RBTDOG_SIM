#!/usr/bin/env python3
"""同事純 RL 步態在崎嶇地形上的 MuJoCo 重現（2026-10-06，對應 10-01 兩顆 wheel_rough policy 的訓練地形）。

訓練地形（`RL_only/2026-10-01_*/…_env.yaml` terrain_generator）：
  random_rough  random_uniform_terrain  每 0.1 m 一格、高度均勻 [0.01, 0.06] m、步 0.01
  boxes         random_grid_terrain     0.45 m 方格、高度 [0.025, 0.1] m（依課程難度）、中央 2 m 平台
這裡用 MuJoCo hfield 重做兩種（可指定振幅），地板幾何仍叫 `floor`（rl_only_infer.Sim 的摔倒判斷靠它）。

抬腳口徑：足端（輪心 z − 輪半徑 0.0961）− 該點地形高 ＝ **對地淨空**；淨空 > 10 mm（遲滯 7 mm）且 ≥ 60 ms 算一步，取段內峰值中位數。
平地表用的是「足端 − 四足最低」，兩者在平地上一致，在崎嶇地上以淨空為準。

用法：
  conda run -n rbtdog --no-capture-output python task7/inference/diag/rl_only_rough_terrain.py \
      --onnx RL_only/2026-10-01_17-30-20/2026-10-01_17-30-20_10000.onnx --tag 10k \
      [--terrains "rough:0.02,rough:0.04,rough:0.06,boxes:0.025,boxes:0.05,boxes:0.10"] [--cmds "0.5,0,0;0,0.5,0;0,0,1.0"] [--seeds 3] [--video out.mp4]
  自訓 policy：--npz policy_rl16.npz [--self-collision]（輸出 outputs/rl_wheel/eval_rl_wheel_rough_{mode}_{tag}.md）；stairs:a ＝ 金字塔樓梯（seed 偶下樓、奇上樓）
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import rl_only_infer as ri  # noqa: E402
import wheel_terrain as wt  # noqa: E402

ROOT = HERE.parent.parent            # task7/
MODEL_DIR = ROOT / "model" / "zgws"
OUT = ROOT / "outputs" / "rl_only"
WHEEL_R = 0.0961
HALF = 10.0                           # 地形半寬 m（20 m × 20 m）
DX = 0.1                              # hfield 格距 m（＝訓練 horizontal_scale）
ZMAX = 1.0                            # hfield 高度量程 m（資料 0..1 對應 0..ZMAX）；stairs 8 cm×10 階＝0.8 m，超過量程的部分 mj_ray 打得到但碰撞會穿（2026-10-13 實測 0.30 時掉穿 0.8 m 平台）


def make_scene_xml(pairs: bool = False) -> str:
    """從 scene_flat_mjx_v3f.xml 產生 hfield 版場景（放同目錄，include 路徑才對）。資料在載入後用 hfield_data 填。
    pairs=True：再加輪輪 pair（`--self-collision`），輸出 scene_rough_pairs_mjx_v3f.xml。"""
    src = (MODEL_DIR / "scene_flat_mjx_v3f.xml").read_text(encoding="utf-8")
    n = int(round(2 * HALF / DX)) + 1
    hf = f'    <hfield name="terrain" nrow="{n}" ncol="{n}" size="{HALF} {HALF} {ZMAX} 0.5"/>\n'
    assert "<asset>" in src and 'name="floor"' in src
    src = src.replace("<asset>\n", "<asset>\n" + hf, 1)
    import re
    src = re.sub(r'<geom name="floor"[^>]*/>', '<geom name="floor" type="hfield" hfield="terrain" pos="0 0 0" material="groundplane"/>', src)
    src = src.replace("<!-- ⚠️ 產生檔，不要手改。產生器：make_mjx_model.py -->",
                      "<!-- ⚠️ 產生檔，不要手改。產生器：inference/diag/rl_only_rough_terrain.py（scene_flat_mjx_v3f.xml 的 hfield 版） -->", 1)
    if pairs:
        src = wt.add_wheel_pairs(src)
    out = MODEL_DIR / ("scene_rough_pairs_mjx_v3f.xml" if pairs else "scene_rough_mjx_v3f.xml")
    if not out.exists() or out.read_text(encoding="utf-8") != src:
        out.write_text(src, encoding="utf-8")
    return str(out)


def terrain_grid(kind: str, amp: float, seed: int, n: int) -> np.ndarray:
    """回傳 n×n 地形高度 m（row ↔ y、col ↔ x，MuJoCo hfield 慣例，見 _check_hfield_convention）。"""
    rng = np.random.default_rng(seed)
    xs = (np.arange(n) - (n - 1) / 2) * DX
    X, Y = np.meshgrid(xs, xs)                    # X[row, col] = x(col)
    if kind == "rough":
        # 訓練：每 0.1 m 格均勻 [0.01, 0.06]、步 0.01 → 這裡 [0, amp]、步 0.01（基準面抬 0 不影響）
        h = rng.uniform(0.0, amp, (n, n)); h = np.round(h / 0.01) * 0.01
    elif kind == "boxes":
        # 訓練：0.45 m 方格、高度 ±amp（課程），中央 2 m 平台
        cell = 0.45
        ci = np.floor((X + HALF) / cell).astype(int); cj = np.floor((Y + HALF) / cell).astype(int)
        nc = int(np.ceil(2 * HALF / cell)) + 1
        hc = rng.uniform(-amp, amp, (nc, nc))
        h = hc[cj, ci]
        h[(np.abs(X) < 1.0) & (np.abs(Y) < 1.0)] = 0.0
        h = h - h.min()
    elif kind == "stairs":
        # 金字塔樓梯（spec §4）：踏面 0.3 m、10 階、中央 ±1 m 平台；amp ＝ 階高；seed 奇數＝反向（中央是坑），整體平移到 ≥ 0
        ring = np.ceil(np.maximum(np.maximum(np.abs(X), np.abs(Y)) - 1.0, 0.0) / 0.3 - 1e-9)
        h = amp * np.clip(10 - ring, 0, 10)
        if seed % 2 == 1:
            h = -h; h = h - h.min()
    elif kind == "flat":
        h = np.zeros((n, n))
    else:
        raise ValueError(kind)
    # 邊緣 1 m 壓平到 0，避免 hfield 邊界掉落
    edge = (np.abs(X) > HALF - 1.0) | (np.abs(Y) > HALF - 1.0); h[edge] = 0.0
    return h


class Terrain:
    def __init__(self, h: np.ndarray):
        self.h = h; self.n = h.shape[0]

    def height(self, x: float, y: float) -> float:
        """雙線性內插（MuJoCo hfield 三角化是線性的，差別在三角形對角線，< 格內高差）。"""
        fx = (x + HALF) / DX; fy = (y + HALF) / DX
        i = int(np.clip(np.floor(fx), 0, self.n - 2)); j = int(np.clip(np.floor(fy), 0, self.n - 2))
        tx = float(np.clip(fx - i, 0, 1)); ty = float(np.clip(fy - j, 0, 1))
        h = self.h
        return float((1 - tx) * (1 - ty) * h[j, i] + tx * (1 - ty) * h[j, i + 1] + (1 - tx) * ty * h[j + 1, i] + tx * ty * h[j + 1, i + 1])

    def apply(self, m):
        hid = 0
        assert m.nhfield == 1 and m.hfield_nrow[hid] == self.n and m.hfield_ncol[hid] == self.n
        assert self.h.min() >= 0.0 and self.h.max() <= ZMAX, (self.h.min(), self.h.max(), ZMAX)
        m.hfield_data[:] = (self.h / ZMAX).ravel()


def _check_hfield_convention(sim, ter):
    """用 mj_ray 從上往下打，驗 row/col ↔ y/x 與高度尺度沒搞反。"""
    import mujoco
    worst = 0.0
    for x, y in [(3.3, -2.1), (-4.7, 1.9), (0.6, 5.2), (-6.1, -6.8)]:
        pnt = np.array([x, y, 5.0]); vec = np.array([0.0, 0.0, -1.0]); geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(sim.m, sim.d, pnt, vec, None, 1, -1, geomid)
        if dist < 0:
            continue
        z_ray = 5.0 - dist; worst = max(worst, abs(z_ray - ter.height(x, y)))
    assert worst < 0.02, f"hfield 慣例對不上：ray vs lookup 差 {worst:.3f} m"
    return worst


def place(sim, ter):
    """站到地形上：腳底最低點落在腳下地形最高處上方 3 mm。"""
    d = sim.d
    zs = []
    for b in sim.feet:
        p = d.xpos[b]; zs.append(ter.height(p[0], p[1]) - (p[2] - WHEEL_R))
    d.qpos[2] += max(zs) + 0.003
    sim.mj.mj_forward(sim.m, d)


def run(sim, ter, pol, cmd, secs, settle=1.0, dt_pol=0.02, seed=0):
    sim.reset(); place(sim, ter)
    n_settle, n = int(settle / dt_pol), int(secs / dt_pol)
    rec = {k: [] for k in ("vb", "ang", "rp", "clr", "tau", "a", "q", "v", "sat", "tgt", "err", "pos", "touch", "gap")}
    fell = None
    wheel_geom = [sim.mj.mj_name2id(sim.m, sim.mj.mjtObj.mjOBJ_GEOM, f"{p}_FOOT_LINK_COLL") for p in ("FAR", "FBL", "RAR", "RBL")]
    for i in range(n_settle + n):
        c = np.zeros(3) if i < n_settle else np.asarray(cmd, float)
        o, R = sim.obs(c)
        a = np.asarray(pol(o), float)
        # 接觸判定（2026-10-06 加）：四個子步任一子步輪子碰到地板就算觸地。「淨空」在方格邊緣會把跨在邊上的輪子算成抬腳，接觸才是真離地。
        tgt = a * sim.scale + sim.offset; sim.d.ctrl[sim.aid] = tgt; sim.last_a = a.copy()
        touch = np.zeros(4, bool)
        for _ in range(sim.sub):
            sim.mj.mj_step(sim.m, sim.d)
            for ci in range(sim.d.ncon):
                cc = sim.d.contact[ci]
                for k, g in enumerate(wheel_geom):
                    if (cc.geom1 == g and cc.geom2 == sim.floor) or (cc.geom2 == g and cc.geom1 == sim.floor):
                        touch[k] = True
        rec["touch"].append(touch)
        d = sim.d
        vb = R.T @ d.qvel[0:3]
        rec["vb"].append(vb); rec["ang"].append(d.qvel[3:6].copy()); rec["rp"].append(ri.rp_from_R(R))
        rec["clr"].append([d.xpos[b][2] - WHEEL_R - ter.height(d.xpos[b][0], d.xpos[b][1]) for b in sim.feet])
        rec["tau"].append(d.actuator_force[sim.aid].copy()); rec["a"].append(a.copy())
        rec["q"].append(d.qpos.copy()); rec["v"].append(d.qvel[sim.vadr].copy()); rec["pos"].append(d.qpos[0:3].copy())
        rec["sat"].append(int(np.sum((tgt[:12] < sim.lo12) | (tgt[:12] > sim.hi12))))
        rec["tgt"].append(tgt.copy()); rec["err"].append(np.abs(tgt[:12] - d.qpos[sim.qadr[:12]]))
        rec["gap"].append(ri.rear_gap(sim))
        r, p = rec["rp"][-1]
        zb = d.qpos[2] - ter.height(d.qpos[0], d.qpos[1])
        if fell is None and (sim.base_contact() or abs(r) > 60 or abs(p) > 60 or zb < 0.15):
            fell = i * dt_pol
    R = {k: np.array(v) for k, v in rec.items()}; R["fell"] = fell; R["n_settle"] = n_settle
    return R


def metrics(R, cmd, dt_pol=0.02):
    s = R["n_settle"]; w = slice(s + int(1.0 / dt_pol), None)
    vb, ang, rp, clr, tau, a = R["vb"][w], R["ang"][w], R["rp"][w], R["clr"][w] * 1000.0, R["tau"][w], R["a"][w]
    g = dict(vx=vb[:, 0].mean(), vy=vb[:, 1].mean(), yaw=math.degrees(ang[:, 2].mean()),
             roll_std=rp[:, 0].std(), roll_max=np.abs(rp[:, 0]).max(), pitch_std=rp[:, 1].std(), pitch_max=np.abs(rp[:, 1]).max())
    p = R["pos"]; g["dist"] = float(np.linalg.norm(p[-1, :2] - p[s, :2]))
    lifts, steps, air = [], [], []
    for k in range(4):
        c = clr[:, k]
        sg = ri.segments(c, 10.0, dt_pol, lo=7.0)
        lifts.append(float(np.median(sg)) if sg else 0.0); steps.append(len(sg) / (len(c) * dt_pol)); air.append(100.0 * np.mean(c > 10.0))
    g["lift"] = [round(x, 1) for x in lifts]; g["steps_s"] = [round(x, 2) for x in steps]; g["air"] = [round(x) for x in air]
    T = R["touch"][w]; tsteps, tair = [], []
    for k in range(4):
        segs, cur = 0, 0
        for f in T[:, k]:
            if not f: cur += 1
            else:
                segs += cur * dt_pol >= 0.06; cur = 0
        segs += cur * dt_pol >= 0.06
        tsteps.append(segs / (len(T) * dt_pol)); tair.append(100.0 * np.mean(~T[:, k]))
    g["tsteps_s"] = [round(x, 2) for x in tsteps]; g["tair"] = [round(x) for x in tair]
    g["clr_max"] = float(clr.max())
    g["tau_abad"] = np.abs(tau[:, 0:4]).max(); g["tau_hip"] = np.abs(tau[:, 4:8]).max(); g["tau_knee"] = np.abs(tau[:, 8:12]).max()
    g["a_max"] = np.abs(a).max(); g["wheel_v"] = np.abs(R["v"][w][:, 12:16]).max(); g["jv_max"] = np.abs(R["v"][w][:, :12]).max()
    g["fell"] = R["fell"]; g["sat"] = 100.0 * np.mean(R["sat"][w] > 0)
    err = R["err"][w]; g["err_abad"] = err[:, 0:4].max(); g["err_hk"] = err[:, 4:12].max()
    g["vcmd"] = (np.abs(np.diff(R["tgt"][w][:, :12], axis=0)) / dt_pol).max()
    g["land_vz"], g["land_n"] = ri.landing_vz(R["clr"][w], dt_pol); g["land_legs"] = ri.landing_counts(R["clr"][w], dt_pol)
    g["gap_rr"] = float(R["gap"][w].min() * 1000.0)
    return g


def fmt_row(label, cmd, g):
    name = f"vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f}"
    fell = "—" if g["fell"] is None else f"**{g['fell']:.1f} s**"
    return (f"| {label} | {name} | {fell} | {g['vx']:+.2f} | {g['vy']:+.2f} | {g['yaw']:+.0f} | {g['dist']:.1f} | {g['roll_std']:.1f}/{g['roll_max']:.0f} | {g['pitch_std']:.1f}/{g['pitch_max']:.0f} | "
            f"{g['lift']} | {g['steps_s']} | {g['air']} | {g['tsteps_s']} | {g['tair']} | {g['tau_abad']:.0f}/{g['tau_hip']:.0f}/{g['tau_knee']:.0f} | {g['err_abad']:.2f}/{g['err_hk']:.2f} | {g['vcmd']:.0f} | {g['jv_max']:.0f} | {g['sat']:.0f} | {g['gap_rr']:.0f} | {g['land_vz']:.2f}（{g['land_n']}＝{'/'.join(str(x) for x in g['land_legs'])}） |")


HDR = ("| 地形 | 指令 | 摔 | vx | vy | 偏航°/s | 位移 m | roll std/峰 | pitch std/峰 | 淨空中位 mm（FR,FL,RR,RL） | 淨空步/秒 | 淨空離地% | **真離地步/秒（接觸）** | **真離地%** | τ ABAD/髖/膝 | 誤差峰 ABAD/髖膝 | 命令速峰 | 關節速峰 | 飽和% | 後輪 gap mm | 落地 vz m/s（次＝FR/FL/RR/RL） |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT.parent / "RL_only" / "2026-09-24_13-23-11.onnx"))
    ap.add_argument("--npz", default="", help="自訓 policy 的狗端 npz（policy_np，含 act_gain）；給了就不用 --onnx，輸出到 outputs/rl_wheel/")
    ap.add_argument("--self-collision", action="store_true", dest="self_collision", help="載入含輪輪 pair 的場景（後輪互碰才看得到）")
    ap.add_argument("--io", default="", help="policy_io.json；空＝--npz 時用 inference/rl_wheel_io.json（RL_only/ 不在 git），否則 RL_only/policy_io.json")
    ap.add_argument("--terrains", default="flat:0,rough:0.02,rough:0.04,rough:0.06,boxes:0.025,boxes:0.05,boxes:0.10,stairs:0.03,stairs:0.08")
    ap.add_argument("--cmds", default="0.3,0,0;0.5,0,0;1.0,0,0;0,0.5,0;0,0,1.0")
    ap.add_argument("--secs", type=float, default=8.0); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ideal", action="store_true"); ap.add_argument("--tag", default="")
    ap.add_argument("--video", default=""); ap.add_argument("--video-secs", type=float, default=5.0, dest="video_secs")
    a = ap.parse_args()
    a.io = a.io or str(ROOT / "inference" / "rl_wheel_io.json" if a.npz else ROOT.parent / "RL_only" / "policy_io.json")
    print("io:", a.io)
    io = json.load(open(a.io)); pol = ri.load_policy(a.onnx if not a.npz else None, a.npz)
    if a.npz:
        a.tag = a.tag or Path(a.npz).stem
    scene = make_scene_xml(pairs=a.self_collision)
    sim = ri.Sim(io, dt=0.005, ideal=a.ideal, wheel_kv=1.0, scene=scene)
    n = sim.m.hfield_nrow[0]
    out_dir = OUT.parent / "rl_wheel" if a.npz else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, agg, trajs = [], {}, {}
    checked = False
    for tspec in a.terrains.split(","):
        kind, amp = tspec.split(":"); amp = float(amp)
        for seed in range(a.seeds):
            ter = Terrain(terrain_grid(kind, amp, seed, n)); ter.apply(sim.m)
            sim.reset(); sim.mj.mj_forward(sim.m, sim.d)
            if not checked and kind != "flat":
                print(f"hfield 慣例檢查：ray vs lookup 最大差 {_check_hfield_convention(sim, ter)*1000:.1f} mm"); checked = True
            label = f"{kind} {amp*100:.1f} cm" if kind != "flat" else "flat"
            if kind == "stairs":
                label += "（下）" if seed % 2 == 0 else "（上）"     # seed 偶＝中央高台往外下樓、奇＝中央坑往外上樓
            for s in a.cmds.split(";"):
                cmd = [float(x) for x in s.split(",")]
                t0 = time.time(); R = run(sim, ter, pol, cmd, a.secs, seed=seed); g = metrics(R, cmd)
                rows.append((label, seed, cmd, g)); agg.setdefault((label, tuple(cmd)), []).append(g)
                if seed == 0:
                    trajs[(label, tuple(cmd))] = (R, ter.h.copy())
                print(f"{label} s{seed} {fmt_row(label, cmd, g)}  ({time.time()-t0:.1f}s)")
            if kind == "flat":
                break
    mode = "ideal" if a.ideal else "real"
    L = [f"# RL_only 純 RL 步態在崎嶇地形（MuJoCo hfield，{mode}{'，' + a.tag if a.tag else ''}）", "",
         f"{'npz' if a.npz else 'ONNX'} `{Path(a.npz or a.onnx).name}`；模型 `{Path(scene).name}`（輪輪 pair {'開' if a.self_collision else '關'}）（＝ v3f 平地模型換 hfield 地板，格距 {DX} m）；每指令先 1 s 零指令站定、再 {a.secs} s；統計取指令後 1 s 起；每地形 {a.seeds} 個 seed。",
         "地形：`rough a` ＝ 每 0.1 m 格高度均勻 [0, a]（訓練 [0.01, 0.06]）；`boxes a` ＝ 0.45 m 方格高度 ±a、中央 2 m 平台（訓練 ±[0.025, 0.1] 課程）；`stairs a` ＝ 金字塔樓梯，階高 a、踏面 0.3 m、10 階、中央 ±1 m 平台，seed 偶＝中央高台（下樓）、奇＝中央坑（上樓）。",
         "淨空 ＝ 輪底 − 該點地形高（雙線性）；> 10 mm（遲滯 7）且 ≥ 60 ms 算一步；離地% ＝ 淨空 > 10 mm 的時間比例。**⚠️ 淨空在方格邊緣會把跨在邊上的輪子算成抬腳（2026-10-06 發現），以接觸判定的「真離地」欄為準**：四個子步都沒碰到地板才算離地、≥ 60 ms 算一步。", "",
         "## 每個 seed", "", HDR]
    for label, seed, cmd, g in rows:
        L.append(fmt_row(f"{label} s{seed}", cmd, g))
    L += ["", "## 彙總（seed 平均；摔＝幾個 seed 摔）", "",
          "| 地形 | 指令 | 摔 | vx | vy | 偏航°/s | 位移 m | roll 峰 | pitch 峰 | 淨空中位 mm（四足平均） | 淨空步/秒 | 淨空離地% | **真離地步/秒（接觸）** | **真離地%** | 膝 τ 峰 | 誤差峰 ABAD/髖膝 | 命令速峰 | 落地 vz |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (label, cmd), gs in agg.items():
        nf = sum(g["fell"] is not None for g in gs)
        m = lambda k: np.mean([g[k] for g in gs]); mx = lambda k: np.max([g[k] for g in gs])
        L.append(f"| {label} | vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f} | {nf}/{len(gs)} | {m('vx'):+.2f} | {m('vy'):+.2f} | {m('yaw'):+.0f} | {m('dist'):.1f} | {mx('roll_max'):.0f} | {mx('pitch_max'):.0f} | "
                 f"{np.mean([np.mean(g['lift']) for g in gs]):.0f} | {np.mean([np.mean(g['steps_s']) for g in gs]):.1f} | {np.mean([np.mean(g['air']) for g in gs]):.0f} | **{np.mean([np.mean(g['tsteps_s']) for g in gs]):.1f}** | **{np.mean([np.mean(g['tair']) for g in gs]):.0f}** | {mx('tau_knee'):.0f} | {mx('err_abad'):.2f}/{mx('err_hk'):.2f} | {mx('vcmd'):.0f} | {m('land_vz'):.2f} |")
    out = (out_dir / f"eval_rl_wheel_rough_{mode}_{a.tag}.md") if a.npz else (out_dir / f"eval_rl_only_rough_{mode}{('_' + a.tag) if a.tag else ''}.md"); out.write_text("\n".join(L) + "\n", encoding="utf-8"); print("→", out)
    if a.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco, imageio
        from PIL import Image, ImageDraw, ImageFont
        m = sim.m; d = mujoco.MjData(m)
        m.vis.global_.offwidth, m.vis.global_.offheight = 960, 540
        font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 26)
        font2 = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc", 20)
        wr = imageio.get_writer(a.video, fps=25, codec="libx264", quality=8, macro_block_size=None)
        base = sim.base
        for k, ((label, cmd), (R, h)) in enumerate(trajs.items()):
            m.hfield_data[:] = (h / ZMAX).ravel()
            r = mujoco.Renderer(m, 540, 960); mujoco.mjr_uploadHField(m, r._mjr_context, 0)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance, cam.azimuth, cam.elevation = 2.6, 145.0, -18.0
            g = metrics(R, cmd); Q = R["q"]
            i0 = R["n_settle"] + 25; i1 = min(i0 + int(a.video_secs / 0.02), len(Q))
            for i in range(i0, i1, 2):
                d.qpos[:] = Q[i]; mujoco.mj_forward(m, d); cam.lookat[:] = [d.xpos[base][0], d.xpos[base][1], d.xpos[base][2] - 0.15]
                r.update_scene(d, camera=cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im, "RGBA")
                dr.rectangle([0, 0, 960, 74], fill=(0, 0, 0, 150))
                dr.text((14, 6), f"{k+1}/{len(trajs)}  {a.tag or Path(a.npz or a.onnx).stem}  {label}  vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f}", font=font, fill=(255, 255, 255))
                lift_txt = f"{min(g['lift']):.0f}–{max(g['lift']):.0f} mm" if max(g["lift"]) > 0 else "0（純輪行）"
                dr.text((14, 42), f"t {(i-i0)*0.02:4.1f} s   vx {g['vx']:+.2f}  vy {g['vy']:+.2f}  偏航 {g['yaw']:+.0f}°/s  pitch 峰 {g['pitch_max']:.0f}°  淨空中位 {lift_txt}  真離地步/秒 {np.mean(g['tsteps_s']):.1f}  摔 {'無' if g['fell'] is None else g['fell']}", font=font2, fill=(255, 235, 120))
                wr.append_data(np.asarray(im))
            r.close()
        wr.close(); print("→", a.video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
