import sys, json, numpy as np
sys.path.insert(0, "task7/inference"); sys.path.insert(0, "task7")
import rl_only_infer as ri, onnxruntime as ort
io = json.load(open("RL_only/policy_io.json"))
ONNX = sys.argv[1] if len(sys.argv) > 1 else "RL_only/2026-09-24_13-23-11.onnx"   # 2026-10-06：可指定別顆 policy
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]); print("ONNX", ONNX)
cases = [("vx", v) for v in [0.1,0.2,0.3,0.5,0.8,1.0,-0.1,-0.2,-0.3,-0.5,-1.0]] + \
        [("vy", v) for v in [0.3,0.35,0.4,0.5,0.6,0.8,1.0,-0.3,-0.35,-0.4,-0.5,-0.6,-0.8,-1.0]] + \
        [("wz", v) for v in [0.2,0.3,0.4,0.5,0.6,0.8,1.0,-0.2,-0.3,-0.4,-0.5,-0.6,-0.8,-1.0]]
print("軸 | 指令 | 實測 | 比例 | 步/秒max | 抬腳峰 mm | 命令速峰 | 輪速峰 | ABAD誤差峰 | roll峰 | fell")
for ax, v in cases:
    cmd = {"vx": (v,0,0), "vy": (0,v,0), "wz": (0,0,v)}[ax]
    sim = ri.Sim(io, ideal=False); R = ri.run(sim, sess, cmd, secs=6.0, settle=1.0)
    s = R["n_settle"]; w = slice(s+50, None)
    vb = R["vb"][w]; ang = R["ang"][w]; C = R["clr"][w]*1000; T = R["tgt"]; E = R["err"][w]; V = R["v"][w]; rp = R["rp"][w]
    meas = {"vx": vb[:,0].mean(), "vy": vb[:,1].mean(), "wz": ang[:,2].mean()}[ax]
    steps = max(len(ri.segments(C[:, j], 8.0, 0.02)) / (len(C)*0.02) for j in range(4))
    cs = np.abs(np.diff(T[s:, :12], axis=0)).max()/0.02
    print(f"{ax} | {v:+.2f} | {meas:+.3f} | {meas/v*100:5.0f}% | {steps:.1f} | {C.max():.0f} | {cs:.1f} | {np.abs(V[:,12:16]).max():.1f} | {E[:,:4].max():.2f} | {np.abs(rp[:,0]).max():.1f} | {R['fell']}")
