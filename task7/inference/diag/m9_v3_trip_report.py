#!/usr/bin/env python3
"""trip23（2026-09-29）段 ⑦ 四趟分析：偏航率、側傾、FK 足端離地、ABAD、輪速、時序。"""
import json, math, sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task7/realbot")
from kin import fk  # noqa

HIP_X, HIP_Y = 0.2698, 0.065
LEGS = ["fl", "fr", "bl", "br"]
SIDE = {"fl": (+1, +1), "fr": (+1, -1), "bl": (-1, +1), "br": (-1, -1)}
RUNS = [("20260929_113225", "原地左轉 +1.3"), ("20260929_113411", "原地右轉 −1.3（鏡像）"),
        ("20260929_113818", "左平移 +0.15"), ("20260929_113953", "右平移 −0.15（鏡像）")]
if len(sys.argv) > 1:
    RUNS = [tuple(a.split(":", 1)) for a in sys.argv[1:]]
D = "/home/huang/rbtdog_sim/task7/logs/m_logs_trip23/M9_%s.json"


def foot_body(leg, q1, q2, q3, roll=0.0, pitch=0.0):
    sx, sy = SIDE[leg]
    x, y, z = fk(leg, q1, q2, q3)
    x += sx * HIP_X; y += sy * HIP_Y
    if roll or pitch:
        r, p = math.radians(roll), math.radians(pitch)
        z = -math.sin(p) * x + math.sin(r) * math.cos(p) * y + math.cos(r) * math.cos(p) * z
    return x, y, z


def segments(c, thr, dt, lo=None, min_s=0.06):
    """遲滯：進 > thr、出 < lo（預設 thr−3）；段長 < min_s 丟掉 → (峰值, 長度 s) 清單。"""
    lo = thr - 3.0 if lo is None else lo
    out, cur = [], []
    for v in c:
        if (not cur and v > thr) or (cur and v > lo):
            cur.append(v)
        elif cur:
            if len(cur) * dt >= min_s:
                out.append((max(cur), len(cur) * dt))
            cur = []
    if cur and len(cur) * dt >= min_s:
        out.append((max(cur), len(cur) * dt))
    return out


def stats(xs):
    n = len(xs)
    if not n:
        return (0, 0, 0)
    m = sum(xs) / n
    return (m, math.sqrt(sum((x - m) ** 2 for x in xs) / n), max(xs, key=abs))


for stamp, name in RUNS:
    d = json.load(open(D % stamp))
    S = d["samples"]; P = d["policy"]; L = P["log"]
    print("=" * 100)
    print(f"{name}   {d['time']}   aborted={d['aborted']}  fallback={P['fallback_total']}  n_sat={P['n_sat']}  "
          f"n_clamp={P['n_clamp']}  worst_ms={P['worst_ms']}  mirror right={L[len(L)//2]['right']}")
    # ---- 指令有效步（policy log）
    act = [l for l in L if any(abs(c) > 1e-9 for c in l["cmd"]) and not l["ol"] and l["fb"] == ""]
    gz = [l["imu"][1][2] for l in act]
    gx = [l["imu"][1][0] for l in act]
    gy = [l["imu"][1][1] for l in act]
    m, s, pk = stats(gz)
    cmd = act[0]["cmd"] if act else [0, 0, 0]
    print(f"指令有效步 {len(act)}（{len(act)*0.02:.1f} s）  cmd={cmd}")
    print(f"  gyro_z 平均 {math.degrees(m):+.1f}°/s（std {math.degrees(s):.1f}）  指令 {math.degrees(cmd[2]):+.1f}°/s"
          f"  → {100*m/cmd[2] if abs(cmd[2])>0 else float('nan'):.0f}%" if abs(cmd[2]) > 0 else
          f"  gyro_z 平均 {math.degrees(m):+.2f}°/s（std {math.degrees(s):.1f}）＝ 平移時的自轉偏移")
    hz = [l["head"] for l in act]
    print(f"  head_err 末值 {hz[-1]:+.3f}  極值 {max(hz, key=abs):+.3f}")
    ms = [l["ms"] for l in L]
    print(f"  推論+產生器 ms：中位 {sorted(ms)[len(ms)//2]:.2f}  >10 ms 步數 {sum(1 for x in ms if x>10)}  最慢 {max(ms):.2f}"
          f"  fb 非空步數 {sum(1 for l in L if l['fb'])}  sat 步數(cmd 有效) {sum(1 for l in act if l['sat'])}")
    a_abs = [max(abs(x) for x in l["a"]) for l in act]
    print(f"  |a| 平均 {sum(sum(abs(x) for x in l['a'])/24 for l in act)/len(act):.3f}  峰 {max(a_abs):.2f}")

    # ---- 200 Hz 樣本：GAIT 段
    G = [r for r in S if r["phase"] == "GAIT"]
    R = [r for r in S if r["phase"] == "READY"]
    dt = 0.005
    roll = [r["roll"] for r in G]; pitch = [r["pitch"] for r in G]
    rm, rs, rp = stats(roll); pm, ps, pp = stats(pitch)
    print(f"GAIT 樣本 {len(G)}（{len(G)*dt:.1f} s）  roll 平均 {rm:+.2f} std {rs:.2f} 峰 {rp:+.2f}°   pitch 平均 {pm:+.2f} std {ps:.2f} 峰 {pp:+.2f}°")

    # 足端 z（機身座標，相對機身中心）
    def feet(r, lev=False):
        z = {}
        for lg in LEGS:
            q = [r["j"][f"{lg}{k}_{n}"][0] for k, n in ((1, "hip_roll"), (2, "hip_pitch"), (3, "knee_pitch"))]
            z[lg] = foot_body(lg, *q, r["roll"] if lev else 0, r["pitch"] if lev else 0)[2]
        return z

    def feet_sgn(r, sr, sp):
        z = {}
        for lg in LEGS:
            q = [r["j"][f"{lg}{k}_{n}"][0] for k, n in ((1, "hip_roll"), (2, "hip_pitch"), (3, "knee_pitch"))]
            z[lg] = foot_body(lg, *q, sr * r["roll"], sp * r["pitch"])[2]
        return z

    def feet_des(r):
        z = {}
        for lg in LEGS:
            q = [r["j"][f"{lg}{k}_{n}"][1] for k, n in ((1, "hip_roll"), (2, "hip_pitch"), (3, "knee_pitch"))]
            z[lg] = foot_body(lg, *q)[2]
        return z

    zR = [feet(r) for r in R]
    stand = {lg: sum(z[lg] for z in zR) / len(zR) for lg in LEGS}
    print(f"READY 站姿足端 z（機身座標，m）: " + "  ".join(f"{lg} {stand[lg]:+.3f}" for lg in LEGS))
    # roll/pitch 正負號慣例的資料判定：正確的水平校正應讓兩條支撐腿（最低兩隻）高度最接近
    best = None
    for sr in (+1, -1):
        for sp in (+1, -1):
            gap = 0.0
            for r in G:
                zz = sorted(feet_sgn(r, sr, sp).values())
                gap += zz[1] - zz[0]
            gap /= len(G)
            print(f"  水平校正 roll×{sr:+d} pitch×{sp:+d}：最低兩足高度差平均 {1000*gap:.1f} mm")
            if best is None or gap < best[0]:
                best = (gap, sr, sp)
    zz = [sorted(feet(r).values()) for r in G]
    print(f"  不校正：最低兩足高度差平均 {1000*sum(z[1]-z[0] for z in zz)/len(zz):.1f} mm  → 採 roll×{best[1]:+d} pitch×{best[2]:+d}")
    SR, SP = best[1], best[2]
    z_nom = foot_body("fl", 0.0, 0.52, -1.20)[2]
    print(f"  名目站姿 z {z_nom:+.3f}（hip +0.52/knee −1.20）；READY 實際比名目短 {1000*(sum(stand.values())/4 - z_nom):+.0f} mm（kp250 撓度）")
    zG = [feet(r) for r in G]; zGl = [feet_sgn(r, SR, SP) for r in G]; zD = [feet_des(r) for r in G]
    stD = [min(z.values()) for z in zD]
    print(f"  GAIT 命令最低足端 z 平均 {sum(stD)/len(stD):+.3f}（比名目 {1000*(sum(stD)/len(stD)-z_nom):+.0f} mm）；實際最低足端 z 平均 {sum(min(z.values()) for z in zG)/len(zG):+.3f}（比名目 {1000*(sum(min(z.values()) for z in zG)/len(zG)-z_nom):+.0f} mm）")
    # 動作各維飽和
    sat = [100*sum(1 for l in act if abs(l["a"][k]) > 0.95)/len(act) for k in range(24)]
    mean_a = [sum(l["a"][k] for l in act)/len(act) for k in range(24)]
    print("  動作 |a|>0.95 的步數 %（24 維）: " + " ".join(f"{x:.0f}" for x in sat))
    print("  動作平均（24 維）:            " + " ".join(f"{x:+.2f}" for x in mean_a))
    ground = [min(z.values()) for z in zG]; groundl = [min(z.values()) for z in zGl]; groundD = [min(z.values()) for z in zD]
    sink = sum(-g for g in ground) / len(ground) - sum(-min(z.values()) for z in zR) / len(zR)
    print(f"  機身高度變化（最低足端，GAIT 平均 vs READY）: {1000*sink:+.0f} mm（負＝下沉）")
    print(f"  {'腿':4s} {'實際離地(機身座標) 步數/頂點中位/最大 mm':44s} {'水平校正後 中位/最大':22s} {'命令抬高 中位/最大':20s} {'支撐相對站姿 mm':14s}")
    for lg in LEGS:
        c = [(z[lg] - g) * 1000 for z, g in zip(zG, ground)]
        cl = [(z[lg] - g) * 1000 for z, g in zip(zGl, groundl)]
        cd = [(z[lg] - g) * 1000 for z, g in zip(zD, groundD)]
        sg = segments(c, 8.0, dt); sgl = segments(cl, 8.0, dt); sgd = segments(cd, 8.0, dt)
        pk = sorted(p for p, _ in sg); pkl = sorted(p for p, _ in sgl); pkd = sorted(p for p, _ in sgd)
        med = lambda a: a[len(a) // 2] if a else 0.0
        mx = lambda a: a[-1] if a else 0.0
        # 支撐時（離地 < 8 mm）足端相對站姿
        st = [(z[lg] - stand[lg]) * 1000 for z, cc in zip(zG, c) if cc < 8]
        print(f"  {lg:4s} {len(sg):3d} 步  {med(pk):5.1f} / {mx(pk):5.1f}  （{len(sg)/(len(G)*dt):.2f} 步/秒）"
              f"          {len(sgl):3d} 步 {med(pkl):5.1f} / {mx(pkl):5.1f}        {len(sgd):3d} 步 {med(pkd):5.1f} / {mx(pkd):5.1f}"
              f"        {sum(st)/len(st) if st else 0:+.0f}")

    # ---- 關節：GAIT 段 τ 峰／RMS、誤差峰、ABAD 貼止點
    print(f"  {'關節':16s} {'τ峰':>7s} {'τRMS':>6s} {'|τ|>40 %':>9s} {'|err|峰':>8s} {'|q_des|≥0.52 %':>15s} {'|v|峰':>6s}")
    for lg in LEGS:
        for k, n in ((1, "hip_roll"), (2, "hip_pitch"), (3, "knee_pitch")):
            j = f"{lg}{k}_{n}"
            rows = [r["j"][j] for r in G]
            tau = [x[2] for x in rows]; err = [abs(x[0] - x[1]) for x in rows]; v = [abs(x[3]) for x in rows]
            rms = math.sqrt(sum(t * t for t in tau) / len(tau))
            over = 100 * sum(1 for t in tau if abs(t) > 40) / len(tau)
            lim = 100 * sum(1 for x in rows if abs(x[1]) >= 0.52) / len(rows) if k == 1 else float("nan")
            print(f"  {j:16s} {max(tau, key=abs):+7.1f} {rms:6.1f} {over:8.1f}% {max(err):8.3f} "
                  f"{('%.0f%%' % lim) if k == 1 else '—':>15s} {max(v):6.1f}")
    # ---- 輪
    wv = {w: max(abs(r["w"][w][1]) for r in G) for w in G[0]["w"]}
    wm = {w: sum(r["w"][w][1] for r in G) / len(G) for w in G[0]["w"]}
    print("  輪速 |v| 峰 rad/s: " + "  ".join(f"{w[:2]} {v:.1f}" for w, v in wv.items())
          + "   平均: " + "  ".join(f"{w[:2]} {v:+.2f}" for w, v in wm.items()))
    print(f"  loop {d['loop']}   超標筆數(濾後) {sum(d['over'].values())}  未濾 {sum(d['over_raw'].values())}")
