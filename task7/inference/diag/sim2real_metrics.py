"""sim2real 指標層（J 文件 §2.2）：A 與 RL 在模擬的 roll/pitch/力矩/誤差/速度統計，口徑同實機穩態窗。
用法：conda run -n rbtdog python task7/inference/diag/sim2real_metrics.py out.json"""
import sys, json, numpy as np
sys.path.insert(0,'task7/inference'); sys.path.insert(0,'task7/realbot')
import cpg_max, cpg_walk_max as cw, gait_baseline as gb, leg_kin, local_infer_max as li, max_model as mm, obs_max
A=gb.BASELINE_A; PH=cpg_max.PHASE_WALK_LS
KNEE=[2,5,8,11]; HIP=[1,4,7,10]
def rollout(infer, secs=20.0, seed=0, ramp=50, lat=1):
    r=cw.Robot(scene=mm.SCENE, actuator_mode="torque_pd", kp3=A["kp3"], kd3=A["kd3"], kd_wheel=A["wheel_kd"])
    ks,f0=leg_kin.knee_sign_of(mm.HOME), leg_kin.home_foot(mm.HOME)
    step=cpg_max.make_cpg_step(PH); x_off=A["x_off"]+seed*1e-12
    r.reset_standing(cpg_max.stand_targets(ks,f0,x_off), mm.NOMINAL_HEIGHT_KIN+0.005)
    for i in range(int(cw.SETTLE_S/mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks,f0,x_off))
        if i==int(0.5/mm.CTRL_DT): r.lock_wheels()
    c=cpg_max.cpg_init(PH); n=int(secs/mm.CTRL_DT); cmd=np.array([0.30,0.0]); base=li.baseline_action("nomux")
    last=np.zeros(10); sway=np.zeros(2); ah=[base.copy()]*3; qvh=[r.d.qvel[mm.LEG_QVEL_IDX].copy()]*3
    R=dict(roll=[],pitch=[],tau=[],vel=[],err=[],yaw=[],om=[])
    yaw_prev=cpg_max.yaw_deg(r.d.qpos[3:7]); yaw_tot=0.0
    for i in range(n):
        class D: pass
        d_=D(); d_.qpos=r.d.qpos; qv=r.d.qvel.copy(); qv[mm.LEG_QVEL_IDX]=qvh[lat]; d_.qvel=qv
        a=infer(obs_max.build_obs(d_,c,cmd,last)); ah=[a]+ah[:2]; qvh=[r.d.qvel[mm.LEG_QVEL_IDX].copy()]+qvh[:2]
        act=ah[lat]; u=min(1.0,i/ramp); act=base+u*(act-base)
        mux,muy,om,swt=li.act_to_cmd(act,"nomux"); sway=li.slew_sway(sway,swt)
        c=step(c,mux,muy,om,mm.CTRL_DT); sw=None if not np.any(sway) else (float(sway[0]),float(sway[1]))
        q_des,_=cpg_max.joint_targets(c,f0,x_off,A["g_c"],A["d_step"],A["d_step_y"],A["duty"],ks,A["z_sag"],sw)
        # substep torques: record max over the control step
        r.step(q_des)
        grav=cpg_max.w2b(r.d.qpos[3:7],np.array([0,0,-1.0]))
        R["roll"].append(np.degrees(np.arcsin(np.clip(grav[1],-1,1)))); R["pitch"].append(np.degrees(np.arcsin(np.clip(-grav[0],-1,1))))
        R["tau"].append(np.abs(r.d.actuator_force[mm.LEG_ACT_IDX]).copy()); R["vel"].append(np.abs(r.d.qvel[mm.LEG_QVEL_IDX]).copy())
        R["err"].append(np.abs(q_des-r.d.qpos[mm.LEG_QPOS_IDX]).max())
        y=cpg_max.yaw_deg(r.d.qpos[3:7]); yaw_tot+=(y-yaw_prev+180)%360-180; yaw_prev=y; R["yaw"].append(yaw_tot); R["om"].append(om.copy())
        last=a
    R={k:np.array(v) for k,v in R.items()}; R["x"]=float(r.d.qpos[0]); return R
def metrics(R, t0=3.5, t1=None):
    dt=mm.CTRL_DT; i0=int(t0/dt); i1=len(R["roll"]) if t1 is None else int(t1/dt)
    roll=R["roll"][i0:i1]; pitch=R["pitch"][i0:i1]; tau=R["tau"][i0:i1]; vel=R["vel"][i0:i1]
    return dict(roll_std=roll.std(), roll_pk_dm=np.abs(roll-roll.mean()).max(), roll_pk99=np.percentile(np.abs(roll),99), roll_mean=roll.mean(),
                pitch_std=pitch.std(), knee_pk=tau[:,KNEE].max(), hip_pk=tau[:,HIP].max(), abad_pk=tau[:,[0,3,6,9]].max(),
                knee_v=vel[:,KNEE].max(), err_pk=R["err"][i0:i1].max(), yaw_rate=(R["yaw"][i1-1]-R["yaw"][i0])/((i1-i0)*dt),
                om_mean=R["om"][i0:i1].mean(), om_min=R["om"][i0:i1].min(), om_max=R["om"][i0:i1].max())
if __name__=="__main__":
    infer_rl=li.load_policy('task7/weights/cpg_rl_max_v2_3_params.pkl',act_dim=10,head=False)
    base=li.baseline_action("nomux"); infer_a=lambda o: base
    out={}
    for name,inf in (("A",infer_a),("RL",infer_rl)):
        rows_short=[]; rows_long=[]
        for seed in range(3):
            R=rollout(inf,20.0,seed)
            rows_short.append(metrics(R,3.5,7.5)); rows_long.append(metrics(R,3.5,20.0))
        out[name]={"short":{k:float(np.mean([r[k] for r in rows_short])) for k in rows_short[0]},
                   "long":{k:float(np.mean([r[k] for r in rows_long])) for k in rows_long[0]}}
    json.dump(out,open(sys.argv[1],"w"),indent=1)
    for nm,v in out.items():
        for w,m in v.items(): print(nm,w,{k:round(x,3) for k,x in m.items()})
