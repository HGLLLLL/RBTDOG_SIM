"""CPG-RL 論文版：地形零樣本測試（不重訓）。

拿已訓練權重 cpg_rl_paper_params.pkl，在不同地形（斜坡、凹凸 heightfield）上直走，
輸出每種地形一支影片＋量測，判斷是否需要重新訓練。

obs/CPG/IK/軟體PD 全部沿用 local_infer_paper.py（只換地形、指令固定直走）→ 公平的零樣本測試。

用法:
  conda run -n rbtdog python task4/inference/terrain_test.py --probe          # 只建地形、檢查 spawn 與幾何
  conda run -n rbtdog python task4/inference/terrain_test.py                  # 跑全部地形＋出影片
  conda run -n rbtdog python task4/inference/terrain_test.py --only flat       # 只跑某地形
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import numpy as np
import mujoco
import local_infer_paper as L          # 重用 CPG/IK/obs/load_policy
from types import SimpleNamespace

SCENE = "/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene.xml"
PARAMS = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl"
OUTDIR = "/home/huang/rbtdog_sim/task4/outputs"

# 地形清單：(name, kind, kwargs)
TERRAINS = [
    ("flat",         dict(kind="flat")),
    ("slope_up_3",   dict(kind="slope", deg=3.0)),
    ("slope_up_5",   dict(kind="slope", deg=5.0)),
    ("slope_up_7",   dict(kind="slope", deg=7.0)),
    ("slope_up_10",  dict(kind="slope", deg=10.0)),
    ("slope_up_15",  dict(kind="slope", deg=15.0)),
    ("slope_up_20",  dict(kind="slope", deg=20.0)),
    ("rough_03",     dict(kind="rough", amp=0.03)),
    ("rough_05",     dict(kind="rough", amp=0.05)),
    ("rough_08",     dict(kind="rough", amp=0.08)),
]


def _rough_field(N, extent):
    """多頻正弦疊加的 value noise，正規化到 [0,1]，中心留平台好 spawn。"""
    xs = np.linspace(-extent, extent, N)
    X, Y = np.meshgrid(xs, xs)
    f = (np.sin(1.3 * X) * np.cos(1.7 * Y)
         + 0.5 * np.sin(2.9 * X + 1.0) * np.cos(3.1 * Y + 2.0)
         + 0.3 * np.sin(5.0 * X) * np.cos(4.3 * Y + 0.5))
    f = (f - f.min()) / (f.max() - f.min())        # → [0,1]
    r = np.sqrt(X ** 2 + Y ** 2)
    f = np.where(r < 0.7, 0.5, f)                   # 中心平台 = 0.5（對應 baseline）
    return f


def make_terrain_model(kind="flat", deg=10.0, amp=0.05, N=120, extent=6.0):
    spec = mujoco.MjSpec.from_file(SCENE)
    floor = next(g for g in spec.geoms if g.name == "floor")
    meta = {"kind": kind, "deg": deg, "amp": amp}
    if kind == "flat":
        pass
    elif kind == "slope":
        a = np.radians(deg)
        # 繞 y 軸旋轉 -a：使 +x 方向為「上坡」(Z 隨 +X 增加)。符號於 probe 實測確認。
        floor.quat = [np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0]
        meta["tan"] = np.tan(a)
    elif kind == "rough":
        f = _rough_field(N, extent)
        hf = spec.add_hfield()
        hf.name = "rough"
        hf.nrow = N
        hf.ncol = N
        hf.size = [extent, extent, 2 * amp, 0.5]     # [rx, ry, z_top, z_base]
        hf.userdata = f.flatten().tolist()
        floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
        floor.hfieldname = "rough"
        floor.pos = [0.0, 0.0, -amp]                 # baseline: 中心(0.5)→ -amp + 0.5*2amp = 0
    else:
        raise ValueError(kind)
    return spec.compile(), meta


def ground_height(meta, x, y):
    """該地形在 (x,y) 的地面 z（用來算相對高度/上坡量）。rough 以 baseline 0 近似。"""
    if meta["kind"] == "slope":
        return meta["tan"] * x          # +x 上坡
    return 0.0


def reset_state(m, d, meta):
    """重置到 home；斜坡時讓機身貼合斜面姿態+高度 spawn（避免水平 spawn 穿透被踢飛）。"""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if meta["kind"] == "slope":
        a = np.radians(meta["deg"])
        # 機身繞 y 軸 pitch -a（與斜面平行，+x 朝上坡）；沿斜面法線抬高 0.27
        d.qpos[3:7] = [np.cos(a / 2), 0.0, -np.sin(a / 2), 0.0]
        n = np.array([-np.sin(a), 0.0, np.cos(a)])      # 斜面法線
        d.qpos[0:3] = 0.27 * n                          # 站在 x≈0 斜面上、離面 0.27
    mujoco.mj_forward(m, d)


def probe():
    print("=== PROBE：建各地形、檢查幾何與 spawn 穿透 ===")
    for name, kw in TERRAINS:
        m, meta = make_terrain_model(**kw)
        d = mujoco.MjData(m)
        reset_state(m, d, meta)
        base_z = float(d.qpos[2])
        foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in L.LEGS]
        # 逐腳穿透：腳 z 減去該腳 x 位置的地面高度（斜坡才正確）
        pens = [float(d.geom_xpos[g][2]) - ground_height(meta, float(d.geom_xpos[g][0]), 0.0)
                for g in foot_gid]
        extra = ""
        if meta["kind"] == "rough" and m.nhfield > 0:
            hz = m.hfield_data.reshape(m.hfield_nrow[0], m.hfield_ncol[0])
            extra = (f" hfield {m.hfield_nrow[0]}x{m.hfield_ncol[0]}"
                     f" size={np.round(m.hfield_size[0],3)} data[{hz.min():.2f},{hz.max():.2f}]")
        print(f"[{name:12s}] base_z={base_z:.3f} 逐腳(腳-地面)={[round(p,3) for p in pens]} "
              f"最深={min(pens):+.3f}{extra}")
    print("（斜坡符號檢查：正式跑時看『上坡量』是否為正）")


def rollout(name, kw, secs=8.0, video=True, verbose=True):
    m, meta = make_terrain_model(**kw)
    d = mujoco.MjData(m)
    reset_state(m, d, meta)
    f0s, jinvs = L.leg_ik_consts(m)
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in L.LEGS]
    fl_gid = foot_gid[0]
    n_sub = int(round(L.CTRL_DT / m.opt.timestep))
    kp, kd = 90.0, 3.0
    flimit = m.actuator_ctrlrange[:, 1]
    G = SimpleNamespace(d=d)
    infer = L.load_policy(PARAMS)

    def apply(q_des):
        for _ in range(n_sub):
            tau = kp * (q_des - d.qpos[7:19]) - kd * d.qvel[6:18]
            d.ctrl[:] = np.clip(tau, -flimit, flimit)
            mujoco.mj_step(m, d)

    for _ in range(int(0.5 / L.CTRL_DT)):      # settle 0.5s
        apply(L.HOME12.copy())

    ren = cam = None
    frames = []
    if video:
        ren = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)

    c = L.cpg_init()
    last_a = np.zeros(12)
    cmd = np.array([0.6, 0.0, 0.0], np.float32)
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    z0_rel = float(d.qpos[2]) - ground_height(meta, x0, y0)
    fzmin = fzmax = None
    fell = None
    for i in range(int(secs / L.CTRL_DT)):
        obs = L.build_obs(G, c, cmd, last_a, foot_gid)
        act = infer(obs)
        mux, muy, om = L.act_to_cmd(act)
        c = L.cpg_step(c, mux, muy, om, L.CTRL_DT)
        apply(L.joint_targets(c, f0s, jinvs))
        last_a = act
        grav = L.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        # 跌倒：機身翻倒(grav_z>-0.4) 或 相對地面高度過低
        rel_h = float(d.qpos[2]) - ground_height(meta, float(d.qpos[0]), float(d.qpos[1]))
        if (grav[2] > -0.4 or rel_h < 0.15) and fell is None:
            fell = i * L.CTRL_DT
        fz = float(d.geom_xpos[fl_gid][2]) - ground_height(meta, float(d.geom_xpos[fl_gid][0]), 0.0)
        fzmin = fz if fzmin is None else min(fzmin, fz)
        fzmax = fz if fzmax is None else max(fzmax, fz)
        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
            cam.distance = 2.6
            cam.elevation = -18
            cam.azimuth = 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    xf, yf = float(d.qpos[0]), float(d.qpos[1])
    fwd = xf - x0                                     # 沿指令方向(+x)前進
    lateral = yf - y0
    climb = ground_height(meta, xf, yf) - ground_height(meta, x0, y0)
    rel_h_end = float(d.qpos[2]) - ground_height(meta, xf, yf)
    res = dict(name=name, fwd=fwd, lateral=lateral, climb=climb,
               fell=fell, lift=(fzmax - fzmin), rel_h_end=rel_h_end)
    if verbose:
        print(f"[{name:12s}] 前進={fwd:+.2f}m 側偏={lateral:+.2f}m 爬升={climb:+.2f}m "
              f"抬腳={res['lift']:.3f}m 末端相對高={rel_h_end:.2f} "
              f"跌倒={'是@%.1fs' % fell if fell else '否'}")
    if video and frames:
        import imageio.v2 as iio
        os.makedirs(OUTDIR, exist_ok=True)
        out = f"{OUTDIR}/terrain_{name}.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264")
        print(f"               影片: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--novideo", action="store_true")
    args = ap.parse_args()
    if args.probe:
        probe()
        return
    terrains = TERRAINS if not args.only else [t for t in TERRAINS if t[0] == args.only]
    results = []
    for name, kw in terrains:
        results.append(rollout(name, kw, secs=args.secs, video=not args.novideo))
    print("\n=== 彙整 ===")
    print(f"{'地形':12s} {'前進(m)':>8s} {'爬升(m)':>8s} {'抬腳(m)':>8s} {'末端高':>7s} {'跌倒':>8s}")
    for r in results:
        print(f"{r['name']:12s} {r['fwd']:>8.2f} {r['climb']:>8.2f} {r['lift']:>8.3f} "
              f"{r['rel_h_end']:>7.2f} {('是@%.1fs'%r['fell']) if r['fell'] else '否':>8s}")


if __name__ == "__main__":
    main()
