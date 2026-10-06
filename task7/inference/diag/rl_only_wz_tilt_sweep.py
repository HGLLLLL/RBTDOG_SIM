import sys, json, numpy as np
sys.path.insert(0, "task7/inference"); sys.path.insert(0, "task7")
import rl_only_infer as ri, onnxruntime as ort
io = json.load(open("RL_only/policy_io.json"))
ONNX = sys.argv[1] if len(sys.argv) > 1 else "RL_only/2026-09-24_13-23-11.onnx"   # 2026-10-06：可指定別顆 policy
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]); print("ONNX", ONNX)
print("條件 | wz | 偏航°/s | roll 平均/峰 | ABAD q 末 (FAR FBL RAR RBL) | 步/秒 | ABAD誤差峰 | 命令速峰 | fell")
for cond, tilt, fk in [("側傾-1°",-1,1.0),("側傾-2°",-2,1.0),("摩擦×1.5",0,1.5),("側傾-1°+摩擦×1.5",-1,1.5)]:
    for wz in [-0.5, -0.8, -1.0, 0.8, 1.0]:
        sim = ri.Sim(io, ideal=False)
        th = np.radians(tilt); sim.m.opt.gravity[:] = [0.0, 9.81*np.sin(th), -9.81*np.cos(th)]; sim.m.geom_friction[:, 0] *= fk
        R = ri.run(sim, sess, (0.0, 0.0, wz), secs=6.0, settle=1.0)
        s = R["n_settle"]; w = slice(s+50, None)
        ang = R["ang"][w]; rp = R["rp"][w]; q = R["q"][-25:][:, sim.qadr[:4]]; C = R["clr"][w]*1000; E=R["err"][w]; T=R["tgt"]
        steps = [len(ri.segments(C[:, j], 8.0, 0.02)) / (len(C)*0.02) for j in range(4)]
        cs = np.abs(np.diff(T[s:, :12], axis=0)).max()/0.02
        print(f"  {cond} | {wz:+.1f} | {np.degrees(ang[:,2].mean()):+6.1f} | {rp[:,0].mean():+.1f}/{rp[np.abs(rp[:,0]).argmax(),0]:+.1f} | {np.round(q.mean(0),2)} | {np.round(steps,1)} | {E[:,:4].max():.2f} | {cs:.1f} | {R['fell']}")
