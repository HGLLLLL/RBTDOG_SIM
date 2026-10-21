"""原廠 trip21 平移 vs 模擬 v1b/v1i：同一套 FK（realbot/kin.fk，控制器座標＝MJCF 座標）量
機身高（ABAD 平面離地＝−min 足端 z ＋ 輪半徑；原廠平移平均傾角 < 1°，FK 準）、hip/knee/ABAD 角、腿伸長（ABAD→輪軸距，上限 0.54）。
    conda run -n rbtdog python task7/inference/diag/posture_vs_oem.py
結果：docs/results/AA_v1b後各版比較_2026-10-21.md §7（姿態）、§8（ABAD 共模＝四腿平均，v1j 的 abad_cm 項量的就是這個）。"""
import sys, json, numpy as np
sys.path.insert(0, "task7/inference"); sys.path.insert(0, "task7/realbot")
import kin, coord, m6_rec
import ref_gait_analysis as rga

R = kin.WHEEL_RADIUS
LEGS = ("fl", "fr", "bl", "br")

def stats(name, q1, q2, q3, extra=""):
    # q*: (N,4) in LEGS order, controller coords (rad)
    N = q1.shape[0]
    xyz = np.array([[kin.fk(l, q1[i, k], q2[i, k], q3[i, k]) for k, l in enumerate(LEGS)] for i in range(N)])
    z = xyz[:, :, 2]; ext = np.hypot(xyz[:, :, 0], xyz[:, :, 2])
    h = -z.min(1) + R
    hip = np.degrees(np.abs(q2)); knee = np.degrees(np.abs(q3)); ab = np.degrees(q1)
    print(f"{name:<22} 高 {h.mean():.3f}±{h.std():.3f} | 膝° fl/fr/bl/br " + "/".join(f"{knee[:, k].mean():.0f}" for k in range(4))
          + f"（5–95% {np.percentile(knee, 5):.0f}–{np.percentile(knee, 95):.0f}）| 髖° " + "/".join(f"{hip[:, k].mean():.0f}" for k in range(4))
          + f" | ABAD° " + "/".join(f"{ab[:, k].mean():+.0f}" for k in range(4)) + f" 擺幅 {np.ptp(ab, axis=0).mean():.0f}"
          + f" | 伸長 m 均 {ext.mean():.3f} 最大 {np.percentile(ext, 95):.3f}"
          + f" | ABAD 共模° 均 {np.degrees(q1.mean(1)).mean():+.1f}（5–95% {np.percentile(np.degrees(q1.mean(1)), 5):+.1f}～{np.percentile(np.degrees(q1.mean(1)), 95):+.1f}）" + extra)

print("=== 原廠 trip21（控制器座標；動作區間）===")
files = {"ref_idle": "160456", "ref_fwd_mid": "160825", "ref_left": "161123", "ref_right_a": "161150", "ref_right_b": "161223", "ref_turn_left": "161316"}
for lab, t in files.items():
    rec, q, tau, wv = rga.load(f"task7/logs/m_logs_trip21/M6_20260909_{t}.json")
    m = rga.moving_mask(rec, q, wv) if lab != "ref_idle" else np.ones(len(wv), bool)
    q1 = np.stack([q[l]["1_hip_roll"] for l in LEGS], 1)[m][::5]
    q2 = np.stack([q[l]["2_hip_pitch"] for l in LEGS], 1)[m][::5]
    q3 = np.stack([q[l]["3_knee_pitch"] for l in LEGS], 1)[m][::5]
    stats(lab, q1, q2, q3)

print("=== 模擬（同 FK；另附 qpos[2] 機身原點高）===")
import rl_only_infer as ri, wheel_terrain as wt
io = json.load(open("task7/inference/rl_wheel_io.json"))
sim = ri.Sim(io, dt=0.005, ideal=False, wheel_kv=1.0, scene=wt.make_flat_pairs_scene_xml())
# sim qadr order: ABAD FR FL RR RL, HIP ..., KNEE ...  → kin legs fl fr bl br = idx 1,0,3,2
PERM = [1, 0, 3, 2]
for name, npz in (("v1b", "task7/weights/rl_wheel_v1b_np.npz"), ("v1i", "task7/weights/rl_wheel_v1i_s212_np.npz")):
    pol = ri.load_policy(None, npz)
    for cmd in ((0, 0, 0), (0, 0.3, 0), (0, 0.5, 0), (0, -0.5, 0)):
        Rr = ri.run(sim, pol, list(cmd), 8.0, 1.0); w = slice(Rr["n_settle"] + 50, None)
        q = Rr["q"][w][:, sim.qadr][::2]
        q1 = q[:, 0:4][:, PERM]; q2 = q[:, 4:8][:, PERM]; q3 = q[:, 8:12][:, PERM]
        stats(f"{name} cmd{cmd}", q1, q2, q3, extra=f" | qpos z {Rr['z'][w].mean():.3f}")
