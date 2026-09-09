"""sim2real 分布層（J 文件 §2.3）：obs 各組與動作的 1-D Wasserstein（實機 std 正規化，逐維平均）與 SMD。
方法依 arXiv:2604.11090（proprioceptive distribution matching）。用法：... sim2real_gap_dist.py out.json"""
import sys, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
sys.path.insert(0,'task7/inference'); sys.path.insert(0,'task7/realbot')
from scipy.stats import wasserstein_distance
import sim2real_metrics as sm, local_infer_max as li, max_model as mm, obs_max, cpg_max, cpg_walk_max as cw, leg_kin, gait_baseline as gb
A=gb.BASELINE_A; PH=cpg_max.PHASE_WALK_LS
def rollout_obs(infer, secs=8.0, seed=0, ramp=50, lat=1, t0=3.5):
    r=cw.Robot(scene=mm.SCENE, actuator_mode="torque_pd", kp3=A["kp3"], kd3=A["kd3"], kd_wheel=A["wheel_kd"])
    ks,f0=leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME); step=cpg_max.make_cpg_step(PH); x_off=A["x_off"]+seed*1e-12
    r.reset_standing(cpg_max.stand_targets(ks,f0,x_off), mm.NOMINAL_HEIGHT_KIN+0.005)
    for i in range(int(cw.SETTLE_S/mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks,f0,x_off))
        if i==int(0.5/mm.CTRL_DT): r.lock_wheels()
    c=cpg_max.cpg_init(PH); n=int(secs/mm.CTRL_DT); cmd=np.array([0.30,0.0]); base=li.baseline_action("nomux")
    last=np.zeros(10); sway=np.zeros(2); ah=[base.copy()]*3; qvh=[r.d.qvel[mm.LEG_QVEL_IDX].copy()]*3
    OBS=[];ACT=[];TAU=[];ERR=[]
    for i in range(n):
        class D: pass
        d_=D(); d_.qpos=r.d.qpos; qv=r.d.qvel.copy(); qv[mm.LEG_QVEL_IDX]=qvh[lat]; d_.qvel=qv
        obs=obs_max.build_obs(d_,c,cmd,last); a=infer(obs); ah=[a]+ah[:2]; qvh=[r.d.qvel[mm.LEG_QVEL_IDX].copy()]+qvh[:2]
        act=ah[lat]; u=min(1.0,i/ramp); act=base+u*(act-base)
        mux,muy,om,swt=li.act_to_cmd(act,"nomux"); sway=li.slew_sway(sway,swt); c=step(c,mux,muy,om,mm.CTRL_DT)
        sw=None if not np.any(sway) else (float(sway[0]),float(sway[1]))
        q_des,_=cpg_max.joint_targets(c,f0,x_off,A["g_c"],A["d_step"],A["d_step_y"],A["duty"],ks,A["z_sag"],sw)
        r.step(q_des)
        if i*mm.CTRL_DT>=t0:
            OBS.append(obs); ACT.append(a); TAU.append(np.abs(r.d.actuator_force[mm.LEG_ACT_IDX]).copy()); ERR.append(np.abs(q_des-r.d.qpos[mm.LEG_QPOS_IDX]))
        last=a
    return np.array(OBS),np.array(ACT),np.array(TAU),np.array(ERR)
def real_data(paths):
    OBS=[];ACT=[];TAU=[];ERR=[]
    for p in paths:
        d=json.load(open(p)); S=d["samples"]; g=[s for s in S if s["phase"]=="GAIT"]; gi=[s for s in S if s["phase"] in("GAIT_IN","GAIT")]
        t0=g[0]["t"]+0.5; t1=g[-1]["t"]
        L=[e for e in d["policy"]["log"] if not e["ol"]]; tg=gi[0]["t"]; ts=tg+0.02*np.arange(len(L)); sel=(ts>=t0)&(ts<=t1)
        OBS+= [e["obs"] for e,s in zip(L,sel) if s]; ACT+=[e["a"] for e,s in zip(L,sel) if s]
        for s in g:
            if s["t"]>=t0:
                TAU.append([abs(s["j"][j][2]) for j in JN]); ERR.append([abs(s["j"][j][0]-s["j"][j][1]) for j in JN])
    return np.array(OBS,float),np.array(ACT,float),np.array(TAU,float),np.array(ERR,float)
import rl_obs; JN=rl_obs.LEG_NAMES
GROUPS={"gravity":(0,3),"gyro":(3,6),"joint_pos":(6,18),"joint_vel":(18,30),"cpg":(42,66)}
def w1(sim,real,cols,norm):
    return float(np.mean([wasserstein_distance(sim[:,k],real[:,k])/norm[k] for k in cols]))
def compare(name, sim_obs, sim_act, sim_tau, sim_err, real_obs, real_act, real_tau, real_err):
    out={}
    normo=real_obs.std(0)+1e-6; norma=real_act.std(0)+1e-6
    for g,(a,b) in GROUPS.items(): out[f"W1_{g}"]=w1(sim_obs,real_obs,range(a,b),normo)
    out["W1_action"]=w1(sim_act,real_act,range(10),norma) if real_act.std()>1e-9 else float("nan")
    # 標準化平均差（SMD）逐群
    for g,(a,b) in GROUPS.items(): out[f"SMD_{g}"]=float(np.mean(np.abs(sim_obs[:,a:b].mean(0)-real_obs[:,a:b].mean(0))/normo[a:b]))
    out["SMD_action"]=float(np.mean(np.abs(sim_act.mean(0)-real_act.mean(0))/norma)) if real_act.std()>1e-9 else float("nan")
    # 力矩與誤差：RMS 與 W1（絕對值，不正規化）
    out["tau_rms_sim"]=float(np.sqrt((sim_tau**2).mean())); out["tau_rms_real"]=float(np.sqrt((real_tau**2).mean()))
    out["tau_W1"]=float(np.mean([wasserstein_distance(sim_tau[:,k],real_tau[:,k]) for k in range(12)]))
    out["err_rms_sim"]=float(np.sqrt((sim_err**2).mean())); out["err_rms_real"]=float(np.sqrt((real_err**2).mean()))
    out["n_sim"]=len(sim_obs); out["n_real"]=len(real_obs)
    print(name, {k:round(v,3) for k,v in out.items()}); return out
if __name__=="__main__":
    infer_rl=li.load_policy('task7/weights/cpg_rl_max_v2_3_params.pkl',act_dim=10,head=False); base=li.baseline_action("nomux"); infer_a=lambda o: base
    res={}
    for name,inf,paths in (("A",infer_a,["task7/logs/m_logs_trip19/M9_20260909_102900.json"]),("RL",infer_rl,["task7/logs/m_logs_trip19/M9_20260909_104304.json","task7/logs/m_logs_trip19/M9_20260909_110247.json","task7/logs/m_logs_trip19/M9_20260909_112653.json"])):
        so=[];sa=[];st=[];se=[]
        for seed in range(3):
            o,a,t,e=rollout_obs(inf,8.0,seed); so.append(o);sa.append(a);st.append(t);se.append(e)
        so,sa,st,se=map(np.concatenate,(so,sa,st,se))
        ro,ra,rt,re=real_data(paths)
        res[name]=compare(name,so,sa,st,se,ro,ra,rt,re)
        # 也算：sim 自己跨 seed 的 W1（雜訊底）
        o1,a1,_,_=rollout_obs(inf,8.0,7); normo=ro.std(0)+1e-6
        res[name]["W1_floor_joint_pos"]=w1(so,o1,range(6,18),normo); res[name]["W1_floor_joint_vel"]=w1(so,o1,range(18,30),normo)
        if name=="RL": res[name]["W1_floor_action"]=w1(sa,a1,range(10),ra.std(0)+1e-6)
        print(" floor", {k:round(v,3) for k,v in res[name].items() if "floor" in k})
    json.dump(res,open(sys.argv[1],"w"),indent=1)
