"""RC 外圈直線控制器 MuJoCo 整合驗證（spec §7）。

腳本化桿量時間軸驅動 RCLineController；狗 = task4 論文版 CPG-RL + 完美 odom。
時間軸：推前進(立即latch) → 前進中右轉(手動弧線) → 放桿(等穩重latch) →
        odom 掉訊 2s(退化直通) → 恢復(跳變防護→重latch) → 倒退沿線。
注意：策略訓練指令 vx∈[0,1]，倒退段超出訓練分佈，只驗證「不跌倒、e_ct 不發散」。

用法：
  MUJOCO_GL=egl conda run -n rbtdog python task5/rc_line/sim_demo.py \
      --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
from rc_line_controller import RCLineController, Sticks, Odom, line_frame, TRACKING
import odom_missions as OM
import local_infer_paper as P

CTRL_DT = OM.CTRL_DT
OUT = os.path.join(HERE, "outputs")

# (t0, t1, fwd, lat, turn, odom_ok)
TIMELINE = [
    (0.0,  8.0,  1.0, 0.0,  0.0, True),    # 推前進：立即 latch、走直線
    (8.0,  10.0, 1.0, 0.0, -0.5, True),    # 前進中右轉：手動接管走弧線
    (10.0, 18.0, 1.0, 0.0,  0.0, True),    # 放轉向：等穩 → latch 新線
    (18.0, 20.0, 1.0, 0.0,  0.0, False),   # odom 掉訊：退化直通
    (20.0, 26.0, 1.0, 0.0,  0.0, True),    # 恢復：跳變防護 → 重 latch
    (26.0, 30.0, -0.5, 0.0, 0.0, True),    # 倒退沿線（超出訓練分佈，寬鬆驗證）
]


def timeline_at(t):
    for t0, t1, f, l, tr, ok in TIMELINE:
        if t0 <= t < t1:
            return Sticks(f, l, tr), ok
    return Sticks(0.0, 0.0, 0.0), True


def cur_waypoints(latch):
    """目前 latch 線的頭尾兩點（給 OM.frame 畫地板目標線）。"""
    if latch is None:
        return [(0.0, 0.0), (0.0, 0.0)]
    p0, psi = latch
    d, _ = line_frame(psi)
    return [tuple(p0), tuple(p0 + 8.0 * d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import imageio.v2 as iio
    print("[info] 載入策略", args.params)
    policy = P.load_policy(args.params)
    r = OM.Runner("odom", policy)
    for _ in range(int(0.5 / CTRL_DT)):                          # 站立 warmup
        r.apply(P.HOME12.copy())

    writer = iio.get_writer(f"{OUT}/rc_demo.mp4", fps=OM.FPS, codec="libx264", quality=7)
    frame_iv = int((1.0 / OM.FPS) / CTRL_DT)
    ctrl = RCLineController()
    last_odom = None
    latches = []                                                 # 每次 latch 的 (p0,psi)
    log = {"t": [], "x": [], "y": [], "e_ct": [], "state": []}
    for i in range(int(30.0 / CTRL_DT)):
        t = i * CTRL_DT
        sticks, ok = timeline_at(t)
        if ok or last_odom is None:
            x, y, yaw = r.g.odom()
            last_odom = Odom(float(x), float(y), float(yaw), t)  # 掉訊時沿用舊樣本(stamp不動)
        prev_latch = ctrl.latch
        cmd = ctrl.update(sticks, last_odom, t)
        if ctrl.latch is not None and ctrl.latch is not prev_latch:
            latches.append(ctrl.latch)
        r.drive(np.asarray(cmd, np.float32))
        e_ct = np.nan
        if ctrl.latch is not None:                               # 真值位置量 e_ct（量測用）
            _, n = line_frame(ctrl.latch[1])
            e_ct = float(n @ (np.asarray(r.xy) - ctrl.latch[0]))
        log["t"].append(t); log["x"].append(float(r.xy[0])); log["y"].append(float(r.xy[1]))
        log["e_ct"].append(e_ct); log["state"].append(ctrl.state)
        if i % frame_iv == 0:
            sub = f"state={ctrl.state}" + ("  ODOM LOST" if not ok else "")
            writer.append_data(OM.frame(r, "RC line controller", (40, 90, 220),
                                        cur_waypoints(ctrl.latch), sub))
        if r.fallen:
            print("[fail] 狗跌倒於 t=%.1fs" % t); break
    writer.close(); print("[video]", f"{OUT}/rc_demo.mp4")
    report(log, latches)
    chart(log, latches)


def report(log, latches):
    e = np.array(log["e_ct"], float)
    st = np.array(log["state"])
    runs, i = [], 0
    while i < len(st):                                           # 切出連續 TRACKING 段
        if st[i] == TRACKING:
            j = i
            while j < len(st) and st[j] == TRACKING:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    print(f"[result] latch 次數={len(latches)}  tracking 段數={len(runs)}")
    ok = True
    for k, (i, j) in enumerate(runs):
        seg, dur = e[i:j], (j - i) * CTRL_DT
        tail = seg[len(seg) // 2:]
        tmax = float(np.nanmax(np.abs(tail)))
        print(f"  段{k+1}: t={log['t'][i]:.1f}s 起  長{dur:.1f}s  "
              f"max|e_ct|={np.nanmax(np.abs(seg)):.3f}m  後半 max|e_ct|={tmax:.3f}m")
        if dur >= 2.0 and log["t"][i] < 26.0 and tmax > 0.05:    # 倒退段(26s後)寬鬆
            ok = False
    print("[result]", "PASS：各段後半 |e_ct|<0.05m" if ok else "FAIL：有段落未收斂 <0.05m")


def chart(log, latches):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(log["x"], log["y"], lw=1.5, color="#3060c0", label="trajectory")
    for k, (p0, psi) in enumerate(latches):
        d, _ = line_frame(psi)
        seg = np.array([p0 - 1.0 * d, p0 + 8.0 * d])
        a1.plot(seg[:, 0], seg[:, 1], "k--", lw=1.0,
                label="latched line" if k == 0 else None)
        a1.plot([p0[0]], [p0[1]], "k^", ms=6)
    a1.set_xlabel("x (m)"); a1.set_ylabel("y (m)"); a1.set_aspect("equal", "box")
    a1.set_title("RC line controller trajectory"); a1.legend(fontsize=9); a1.grid(alpha=0.3)
    a2.plot(log["t"], log["e_ct"], lw=1.2, color="#c04030")
    a2.axhline(0, color="gray", ls=":", lw=1)
    a2.set_xlabel("t (s)"); a2.set_ylabel("e_ct (m)")
    a2.set_title("cross-track error (NaN = not tracking)"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/rc_demo.png", dpi=120)
    print("[chart]", f"{OUT}/rc_demo.png")


if __name__ == "__main__":
    main()
