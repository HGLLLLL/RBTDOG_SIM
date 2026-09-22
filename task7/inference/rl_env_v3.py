"""CPG-RL v3.1：統一運動學產生器，模仿原廠步態（spec `docs/superpowers/specs/2026-09-15-cpg-rl-v3.1-unified-kinematic-generator-design.md`）。

  - 指令 (vx, vy, wz) → 活動度 (a_lat, a_turn, a_wheel, s4, s_arc) → 每腿活動度 s_k（誰抬、抬多高），沒有離散模式
  - 每腳每週期位移 vec_k = T·(vy·ŷ + rot_step_frac·wz × r_k)：純 vx 位移全在 x、輪子滾掉、不抬腳；
    平移／原地轉／弧線有 y 分量才抬腳。旋轉只用踏步做掉一部分（原廠約 0.2，其餘刮地）
  - 相位：原廠兩組四腿偏移（原地轉／平移），依 a_lat:a_turn 圓周插值，左右鏡像
  - 輪子：速度伺服 kd 1.0 ＋ 前饋 0.13（M11 實機已驗）；wheel = 差速×gate ＋ 平移圖案 ＋ RL 殘差（±2 rad/s ≈ ±2 N·m）
  - 動作 24 維（v3.3）：[ω_scale, amp×4, lift, sway×2, wheel_res×4, 關節目標殘差×12（±0.12 rad、限速）]；零動作＝純開迴路產生器
  - 觀測 88 維：gravity 3 | gyro 3 | jpos 12 | jvel 12 | wheel_vel 4 | cmd 3 | [s4, s_arc] 2 | head 1 | last_a 24 | cpg 24
  - v3.3：原地轉名目改原廠命令週期（幅度每回合隨機 0.3–0.9，RL 靠關節殘差學即時平衡）；ABAD 剛度 DR ×0.7–1.0
原廠參考數字來自 `outputs/ref_gait_dataset.json`（常數釘在 REF／PH_*；tests 比對 json）。
本機 CPU 可跑 reset/step 做測試（慢）；訓練在 Colab GPU。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import max_model as mm  # noqa: E402
import rl_env_max as v2  # noqa: E402
from rl_env_max import (CTRL_DT, DELAY_BASE, FALL_GRAV_Z, GYRO_BIAS, IMU_TILT_DEG, KILL_STEPS,  # noqa: E402
                        MIN_HEIGHT, N_FRAMES, NOISE_GRAV, NOISE_GYRO, NOISE_QPOS, NOISE_QVEL, PUSH_EVERY,
                        PUSH_VEL, SIM_DT, TAU_KILL, _qrot, _quat_rp, err_barrier, fk_j, ik_j,
                        tau_barrier, w2b, yaw_reward)

LEG_QPOS_IDX, LEG_QVEL_IDX = jnp.array(mm.LEG_QPOS_IDX), jnp.array(mm.LEG_QVEL_IDX)
WHEEL_QVEL_IDX = jnp.array(mm.WHEEL_QVEL_IDX)
LEG_ACT_IDX, WHEEL_ACT_IDX = jnp.array(mm.LEG_ACT_IDX), jnp.array(mm.WHEEL_ACT_IDX)
KNEE_SIGN_j = v2.KNEE_SIGN_j
SCENE_V3 = str(Path(mm.SCENE_MJX_KP250).with_name("scene_flat_mjx_v3.xml"))
SCENE_V3P = str(Path(mm.SCENE_MJX_KP250).with_name("scene_flat_mjx_v3p.xml"))

# ---------------------------------------------------------------- 原廠參考（results/L、outputs/ref_gait_dataset.json）
# 腿序 MJCF (FR, FL, RR, RL)。資料集是 shm 序 (fl, fr, bl, br)，這裡已轉。
REF = dict(
    # 輪行站姿（hip/knee 取資料集平均、ABAD 0）：前腿 hip +0.52 / knee −1.20，後腿鏡像
    stance_q12=np.array([0.0, 0.52, -1.20, 0.0, 0.52, -1.20, 0.0, -0.56, 1.20, 0.0, -0.56, 1.20]),
    L_eff=0.375, r_wheel=0.096, v_max=0.92,
    # 踏步名目，旋轉族（原地轉／弧線）與平移族（平移／斜走）各一組，依 a_lat:a_turn 混合；
    # 原廠 2.5／2.1 Hz、duty 0.8、抬 22 mm 是 kp250 做不到的，留作對照
    # G0 掃描定案（spec §8.2）：旋轉族 1.8 Hz／duty 0.5／小跑相位；平移族 2.1 Hz（＝原廠）／duty 0.7／原廠相位；抬高 35 mm
    step_hz_turn=1.8, duty_turn=0.50, step_hz_lat=2.1, duty_lat=0.70, lift=0.035, lift_lat=0.025,   # 平移抬高貼原廠 22 mm；35 mm 時 vy 0.08 力矩峰 92 > 終止線
    factory_hz_turn=2.5, factory_hz_lat=2.1, factory_duty=0.80, factory_lift=0.022,
    # 旋轉需要的地面位移裡用踏步做掉的比例（原廠 ≈ 0.2、其餘輪子刮地；G0 掃 0.2/0.5/1.0）
    rot_step_frac=0.4,
    # 側向命令增益：原廠平移的 ABAD 命令擺幅 60°（腳側向命令 ±220 mm）對實際每步 40 mm ≈ 5×；位置伺服要靠命令放大才推得動側向（同 Z_SAG 的道理）
    lat_cmd_gain=1.5,          # G0：1.5 → vy 0.08 指令得 0.090、0.04 得 0.035；2.0 過衝到 0.125；1.3 Hz 配增益會倒，2.1 Hz 才穩
    # 活動度分母
    a_ref=dict(vy=0.15, wz=1.3, vx=0.25),      # 原廠平移實測約 0.3–0.5 m/s（IMU 積分＋現場目視，2026-09-15 晚修正；之前 0.08 是腳重放距離推的、低估 4 倍）
    # 平移輪速圖案（左移 FL +Ω / RL −0.67Ω；運動學推不出，照錄檔）
    lat_wheel=dict(FL=1.0, RL=-0.67), lat_omega=3.3, lat_vy_ref=0.06,
    # 弧線姿態：內側前腳往中線 60 mm、內側後腳往外 30 mm（正 = 往中線）
    posture_arc=dict(front=0.06, rear=-0.03),
    # 平移姿態：四腳外張（原廠平移時 ABAD 平均 +5/−7/+4/−5°）。G0：開 30 mm 讓 vy 掉 25%、斜走 vy 掉到 0.013，預設關
    posture_lat=0.0,
    # v3.2：四腿踏步族的名目改用原廠命令週期（spec §9）。"cycle"＝原廠週期、"kin"＝v3.1 運動學
    # G0（spec §9.1）：平移週期 10 s 站住、roll std 0.5°；原地轉週期各幅度 3–7 s 內倒 → 退回 v3.1 對角小跑
    step_gen_lat="cycle", step_gen_turn="cycle",      # v3.3：原地轉也走原廠週期（開迴路 3–7 s 會倒，靠 RL 關節殘差平衡）
    # 平移幅度↔側速在 kp250 下很陡（滑步區）：有效幅度 0.64→0.04、0.80→0.17、0.96→0.30 m/s（0.86 在 10 s 內倒）
    #   → amp = 0.64 + 1.23·(|vy| − 0.04)，夾 [0.55, 0.78]；u4 只當開關（s4 ≥ 0.3 全開），幅度不再乘活動度
    cyc_lat_amp0=0.64, cyc_lat_vy0=0.04, cyc_lat_slope=1.23, cyc_lat_amp_clip=(0.55, 1.0), cyc_amp_lat=1.0,   # 上限放到原廠全幅（0.96 ≈ 0.30 m/s）
    cyc_amp_turn=0.7, cyc_amp_turn_range=(0.3, 0.9), cyc_amp_rand=True,   # 訓練時每回合抽幅度（小幅度站得住、大幅度轉得快＝用隨機化代替課程）；eval 用 cyc_amp_turn
    cyc_wheel=1.0, cyc_recenter=True, cyc_hz_scale=1.0,
    cyc_turn_sym=False,                 # v3.5：True＝右轉週期＝左轉鏡像、兩段錄檔相位對齊後平均（右轉錄檔幅度小兩成 → v3.4f 右轉只 −56°/s）
    # 關節偏移的分關節縮放（ABAD, HIP, KNEE）。試過等力矩換算 (1, 0.48, 0.48)：力矩峰降但旋轉也掉（幅度 0.9 倒前偏航 55→23°/s），
    # 多活的 1–2 s 是靠不動換的；平移更是砍半就不滑。兩族都維持統一縮放，力矩峰交給 RL 的 58 護欄（spec §10.2）
    cyc_joint_scale_turn=(1.0, 1.0, 1.0), cyc_joint_scale_lat=(1.0, 1.0, 1.0),
    # v3.7 候選（2026-09-22 E6）：平移族「抬高與跨距解耦」—— 髖膝固定 cyc_lat_lift× 表（抬 25–40 mm 不隨速度縮），
    #   ABAD 倍率隨指令：k = clip(abad0 + slope·(|vy| − vy0), clip)；policy 的 amp 只管 ABAD、lift 只管髖膝。預設關（golden）。
    #   E9 校準（零動作 2.1 Hz 踏步的名目側速上限 ≈ 0.21 m/s）：k 0.3 到 vy 0.12、0.75 在 0.20、1.0 在 ≥ 0.245
    #   E11／E12／E13：往後漂與航向偏轉隨膝倍率線性增加（1.87 → −0.08 m/s／−60°），但速度上限也跟著（1.6 → 0.16、1.75 → 0.18、1.87 → 0.21）；
    #   髖膝分開沒有更好。取 1.75：抬 24–30 mm、0.20 指令跑 0.18、膝峰 46–51、漂 −0.065
    #   E14／E15（v3.7b）：解耦名目往後漂（−0.11 @0.12、−0.065 @0.20）與航向偏轉（−50°/10 s）只作用在輪命令的前饋補償，不改族別混合
    #   E16：航向偏轉七成來自踏步圖案本身（輪圖案歸零只減三成），wz 前饋 0.4 砍四成（−54° → −31°），剩下交給 policy 的每腿 ABAD 幅度
    cyc_lat_vx_ff=(0.15, -0.43, 0.05, 0.10), cyc_lat_wz_ff=0.4,   # vx_ff = clip(a + b·|vy|, lo, hi)；wz_ff·sign(vy) rad/s；只在 cyc_lat_decouple 時生效
    cyc_lat_decouple=False, cyc_lat_lift=1.75, cyc_lat_abad0=0.3, cyc_lat_abad_vy0=0.12, cyc_lat_abad_slope=5.6, cyc_lat_abad_clip=(0.3, 1.0),
    # 相位組："factory"＝原廠四腿相位（配 duty ≥ 0.7 才有三腳著地）；"trot"＝對角對交替（duty 0.5 用，任何時刻兩對角腳著地）
    phase_set_turn="trot", phase_set_lat="factory",
    # ---- 隨增益變的四項（v3.4f 用 REF_FACTORY 整組覆寫；spec §4／§5）
    z_sag=0.036,                        # kp250 承重撓度（M8 實機 36 mm）；kp120 是 72 mm
    nominal_height=0.54, settle_h_range=(0.42, 0.60),
    err_bar3=(0.60, 0.45, 0.45),        # 分關節誤差護欄（ABAD, HIP, KNEE）
    wheel_space="vel",                  # "vel" = 速度伺服 kd 1.0（M11 實機已驗）；"tau" = 力矩空間（原廠 kd 0.1）
    wheel_outer_gain=1.0,               # 力矩空間下輪行族的外環增益 τ = G_w·(v_des − v)；1.0 ＝ kd 1.0 的等效行為
    trans_s=1.0,
)
SCENE_V3F = str(Path(mm.SCENE_MJX_KP250).with_name("scene_flat_mjx_v3f.xml"))

# ---- v3.4f：馬達增益完全照原廠錄檔動作段（spec 2026-09-15-cpg-rl-v3.4f-factory-gains-design.md）
#   腿 kp 250→120 讓承重撓度 36→72 mm（M8 實機），命令相關的常數要跟著重掃（spec §4）；
#   輪 kv 1.0→0.1 讓速度殘差沒有力矩權限，改走力矩空間（spec §3）。
#   z_sag／lat_cmd_gain／wheel_outer_gain 的值由 G0 掃定（Task 8），這裡是起點。
REF_FACTORY = dict(
    z_sag=0.072,                        # M8 實機：kp120 撓度 72 mm（kp250 是 36）
    err_bar3=(1.05, 0.45, 0.45),        # 原廠 ABAD 命令差實測 max 1.02 rad；髖膝 max 0.32 < 0.45 不動（spec §0.3）
    wheel_space="tau",
    wheel_outer_gain=1.0,               # G0 掃過 0／0.3／1.0：1.0 最穩（＝kd 1.0 的等效行為），低增益等於把側向輪圖案一起打折
    # ---- 以下三項是 G0 掃出來的（spec §11）。根因：撓度 36→72 mm 把腳壓在地上，
    #      輪子卸不了重就拖不動 41 kg —— 平移不是「力矩不夠」而是「腳沒離地」。
    cyc_amp_lat=1.6,                    # 命令幅度補撓度：amp 1.0 腳只離地 0.4 mm、vy 剩 17%；1.6 時腳 22–47 mm（原廠實測 16–22）、vy 達指令 64–94%
    lat_omega=2.2,                      # 3.3 是 kp250 下擬的；腿軟了同一組輪圖案讓側速衝到 4 倍（0.34 對指令 0.08）然後翻車
    posture_lat=0.06,                   # 四腳外張加寬支撐面。kp250 時因為吃掉 vy 所以關著（=0），kp120 不開就翻
)

GAIN_SETS = {
    "kp250": dict(scene=SCENE_V3, kp3=mm.KP3_A, kd3=mm.KD3_A, ref={}),          # v3.3（預設）
    "factory": dict(scene=SCENE_V3F, kp3=mm.KP3, kd3=mm.KD3, ref=REF_FACTORY),  # v3.4f
}
KV_WHEEL, TAU_FF, V_DEAD = 1.0, 0.13, 0.3          # M11：kd 1.0、前饋 0.13、|v_des|<0.3 rad/s 視為 0
Z_SAG = 0.036                                      # ★ kp250 承重撓度（M8 實測 36 mm）：命令抬高 = 目標抬高 + Z_SAG，否則腳離不了地
ACT_DIM = 24
QRES_MAX, QRES_SLEW = 0.12, 0.01     # 關節目標殘差 ±0.12 rad、每步最多改 0.01 rad（0.5 rad/s）
OMEGA_NOM_SCALE, AMP_SCALE, LIFT_SCALE, SWAY_MAX, SWAY_SLEW, WHEEL_RES = 0.28, 0.5, 0.4, 0.040, 0.004, 2.0   # WHEEL_RES：rad/s 偏移（kd 1.0 ≈ N·m）
STEP_GAIN_S = 0.2              # s_k ≥ 0.2 的腿吃完整步向量（四腿踏步時 stance 腳要同速，不能按 s_k 打折）
PH_SLEW = 0.02                 # 相位偏移每步最多改 0.02 週期
FOOT_OFF_MAX = 0.020                    # WHEEL 模式腿的順從空間（±20 mm）
NOISE_WHEEL = 0.10
TAU_BAR, ERR_BAR = 58.0, 0.45
KNEE_V_BAR = 14.0
NOMINAL_HEIGHT = 0.54                   # 原廠 stand 站姿高度（M7 實測 533 mm）
LEG_IDX = {"FR": 0, "FL": 1, "RR": 2, "RL": 3}
LEFT_LEGS, RIGHT_LEGS = jnp.array([1, 3]), jnp.array([0, 2])
KNEE12 = jnp.array([2, 5, 8, 11])
ABAD12 = jnp.array([0, 3, 6, 9])

W = dict(W_VX=2.0, W_VY=3.0, W_YAW=2.0, W_YAWI=0.5, W_YAWLIN=1.0, YAW_LIN_E=1.5, W_YAWREL=8.0, W_VYREL=3.0, W_HEAD=0.5, WZ_EMA=0.04,
         # POST_STEP_SCALE：四腿踏步時姿態類懲罰（roll/pitch/角速度/偏置）乘 (1 − 0.5·u_mode)；v3.2 第一輪策略靠停止轉動避罰（spec §9.2）
         POST_STEP_SCALE=0.6, W_H=0.3, W_LIFT=0.5, W_STANCE=0.5,
         W_ROLL=150.0, W_PITCH=100.0, W_ROLLRATE=0.5, W_PITCHRATE=0.3, W_BIAS=100.0, W_SWAYBIAS=30.0,
         W_ACT=0.05, W_OMDOT=0.5, W_QRES=0.5, W_TAU=1e-5, W_TAUBAR=0.05, W_ERRBAR=1.0, W_KNEEV=0.02, W_MODE=2.0, W_VZ=0.05,
         VX_SIG2=0.02, VY_SIG2=0.005, YAW_SIG2=0.0005, YAW_SIG2_WIDE=0.02, YAW_INST_SIG2=0.05, HEAD_SIG=0.15,
         CMD_VX=(-0.4, 0.9), CMD_VY=(0.04, 0.30), CMD_WZ=(0.2, 1.3), P_VX=0.65, P_VY=0.45, P_WZ=0.50,
         P_TURN_ONLY=0.35,        # 有 wz 時有 35% 把 vx、vy 歸零 → 純原地轉由 12% 提到約 25%（v3.3 最難的任務練最少）
         P_SWITCH=0.4, RAMP_STEPS=50, BIAS_EMA=0.02,
         W_ABADBIAS=0.0, W_DRIFT=0.0, W_HEADLIN=0.0, HEAD_LIN_E=0.5,     # v3.5 三項；預設權重 0 → v3.3 reward 逐位元不變（golden）
         CYC_TURN_SYM=False,      # v3.5：True → env 把 ref["cyc_turn_sym"] 打開（放 W 裡是為了讓「權重預設」一個開關就帶齊訓練設定）
         W_STEP=0.0, STEP_APEX=0.021)   # v3.6：平移每週期抬腳頂點獎勵；預設 0 → 逐位元不變（golden）
W35 = dict(W, W_ABADBIAS=30.0, W_DRIFT=40.0, W_HEADLIN=1.0, W_VYREL=6.0, P_VY=0.60, CYC_TURN_SYM=True)   # v3.5（spec 2026-09-16 §2）：DualModeEnv(weights=v3.W35)
# v3.6（spec 2026-09-22 §2）：t_step ＋ t_drift／t_abadbias 改死區線性（PEN_SHAPE="hinge"；二次式接近目標時梯度消失，v3.5 spec §9.1）
W36 = dict(W35, W_STEP=1.5, STEP_APEX=0.021, PEN_SHAPE="hinge", W_DRIFT_L=8.0, DRIFT_DZ=0.010, W_ABADBIAS_L=6.0, ABAD_DZ=0.0436)
# v3.7（2026-09-22 E6–E13）：平移抬高／跨距解耦（REF cyc_lat_decouple）＋ 側向指令上限 0.22（2.1 Hz 踏步、抬 1.75× 的名目上限 ≈ 0.18，再高只能教它側滑）
W37 = dict(W36, CYC_LAT_DECOUPLE=True, CMD_VY=(0.04, 0.22))
# v3.7b（2026-09-22 夜，v3.7f 訓到 64M `step` 0.41→0.25 停損）：名目的漂移／航向由輪前饋補（REF cyc_lat_*_ff，見 E14／E15）、lift 通道只能加不能減、W_STEP 2.5
#   MIRROR_AUG：每回合 50% 鏡像（obs 鏡像給 policy、動作鏡像回物理）→ 同一組權重服務左右兩種情境，訓出來的 policy 本身對稱（右轉弱＝policy 不對稱，E5–E10）
W37B = dict(W37, CYC_LAT_FF=True, LIFT_NONNEG=True, W_STEP=2.5, MIRROR_AUG=True)
# v3.7f 權重的部署設定（不重訓）：訓練時的產生器 ＋ 輪前饋 ＋ lift 夾在 ≥ 名目（policy 靠縮 lift 避漂移懲罰，部署時把這條路關掉、漂移用前饋補）
W37D = dict(W37, CYC_LAT_FF=True, LIFT_NONNEG=True)
T_KEYS = ("t_vx", "t_vy", "t_yaw", "t_yawi", "t_yawlin", "t_yawrel", "t_vyrel", "t_head", "t_h", "t_lift", "t_stance", "t_roll", "t_pitch", "t_rollrate",
          "t_pitchrate", "t_bias", "t_act", "t_omdot", "t_qres", "t_tau", "t_taubar", "t_errbar", "t_kneev", "t_mode", "t_vz", "t_abadbias", "t_drift", "t_headlin", "t_step")
METRIC_KEYS = ("height", "vx", "vy", "wz", "reward", "pitch", "roll", "mode", "s4", "s_arc", "cyc", "clr_step", "clr_stance",
               "yawerr", "vxerr", "vyerr", "tau_pk", "err_pk", "knee_v", "omega", "sway_y", "roll_bias", "abad_bias", "vx_drift", "head_deg", "head_abs") + T_KEYS
# ⚠️ reset 與 step 的 metrics 鍵集合必須相同（brax EpisodeWrapper 用 lax.scan，結構不同會炸）；tests 有 wrapper 檢查


# ---------------------------------------------------------------- 幾何
def _stance_feet():
    q = REF["stance_q12"].reshape(4, 3)
    return jnp.stack([fk_j(k, jnp.array(q[k])) for k in range(4)])


STANCE_Q12_j = jnp.array(REF["stance_q12"])
STANCE_FEET_j = _stance_feet()          # (4,3) 相對各腿 ABAD


def _vec4(d: dict, default=0.0):
    out = np.full(4, default, dtype=float)
    for k, v in d.items():
        out[LEG_IDX[k]] = v
    return jnp.array(out)


# 序 FR, FL, RR, RL。錄檔（shm 序 fl, fr, bl, br；bl=RL、br=RR）：左轉 fl 0／fr .25／bl .59／br .06；左移 fl 0／fr .19／bl .51／br .72
PH_TURN_L = jnp.array([0.25, 0.00, 0.06, 0.59])
PH_LAT_L = jnp.array([0.19, 0.00, 0.72, 0.51])
PH_TROT = jnp.array([0.5, 0.0, 0.0, 0.5])        # 對角小跑：FL+RR 0、FR+RL 0.5（左右對稱，不用鏡像）
W_LAT_L = jnp.array([0.7, 1.0, 0.7, 1.0])        # 左移：主導 FL+RL，另一側 0.7（抬高比 16/23）
W_TURN_L = jnp.array([0.7, 1.0, 1.0, 0.7])       # 左轉：主導對角 FL+RR
W_ARC_L = jnp.array([0.0, 1.0, 0.5, 0.0])        # 左弧線：內側前 FL 1.0、對角外側後 RR 0.5
DOM_TURN_L = jnp.array([0.0, 1.0, 1.0, 0.0])     # 左轉主導對（輪 gate 用）
LAT_WHEEL_L = _vec4(REF["lat_wheel"])
MIRROR_LR = jnp.array([1, 0, 3, 2])              # FR↔FL、RR↔RL
SIDE_Y_j = jnp.array(mm.SIDE_Y)                  # FR −1, FL +1, RR −1, RL +1
HIP_XY = jnp.array([[mm.HIP_X, -mm.HIP_Y], [mm.HIP_X, mm.HIP_Y], [-mm.HIP_X, -mm.HIP_Y], [-mm.HIP_X, mm.HIP_Y]])
FOOT_XY_BODY = HIP_XY + STANCE_FEET_j[:, :2]     # 站姿足端在機身座標 (4,2)：前 ≈ (+0.38, ±0.17)、後 ≈ (−0.36, ±0.17)
FRONT_j = jnp.array([1.0, 1.0, 0.0, 0.0])

# ---- v3.2 原廠命令週期（outputs/ref_cmd_cycles.json；ref_extract.cmd_cycle）：序 lat_left, lat_right, turn_left, turn_right
_CYC_PATH = Path(__file__).resolve().parents[1] / "outputs" / "ref_cmd_cycles.json"
_CYC = json.loads(_CYC_PATH.read_text(encoding="utf-8"))["cycles"]
CYC_KEYS = ("lat_left", "lat_right", "turn_left", "turn_right")
CYC_OFF = jnp.array([np.array(_CYC[k]["des"]) - np.array(_CYC[k]["q_mean"]) for k in CYC_KEYS])   # (4,100,12) 命令 − 實際平均
CYC_QMEAN = jnp.array([np.array(_CYC[k]["q_mean"]) for k in CYC_KEYS])                            # (4,12)
CYC_TAU = jnp.array([np.array(_CYC[k]["tau_w"]) for k in CYC_KEYS])                               # (4,100,4) 輪 τ
CYC_HZ = jnp.array([_CYC[k]["freq_hz"] for k in CYC_KEYS])


def _mirror_cycle(off, tau):
    """左向週期 → 右向：腿 FR↔FL、RR↔RL 對調，ABAD 反號（+ABAD 對四腿都是 +y），髖膝、輪 τ 不變號。"""
    o = off.reshape(off.shape[0], 4, 3)[:, jnp.array([1, 0, 3, 2])]
    o = o * jnp.array([-1.0, 1.0, 1.0])[None, None, :]
    return o.reshape(off.shape[0], 12), tau[:, jnp.array([1, 0, 3, 2])]


_lat_r_off, _lat_r_tau = _mirror_cycle(CYC_OFF[0], CYC_TAU[0])
CYC_OFF = CYC_OFF.at[1].set(_lat_r_off)
CYC_TAU = CYC_TAU.at[1].set(_lat_r_tau)
CYC_HZ = CYC_HZ.at[1].set(CYC_HZ[0])
CYC_QMEAN = CYC_QMEAN.at[1].set(_mirror_cycle(CYC_QMEAN[0][None], CYC_TAU[0][:1])[0][0])
CYC_N = CYC_OFF.shape[1]


def _sym_turn_cycles(off, tau, hz):
    """v3.5：turn_left 與 mirror(turn_right) 相位對齊（循環相關最大處，實測差 86°）後平均 → 左右一致、對角不對稱保留。
    回傳 (off_sym, tau_sym, hz_sym)：索引 2／3 換成對稱版，0／1（平移）不動。"""
    o, t = np.asarray(off), np.asarray(tau)
    L, TL = o[2], t[2]
    Rm, TRm = (np.asarray(x) for x in _mirror_cycle(jnp.asarray(o[3]), jnp.asarray(t[3])))
    n = L.shape[0]
    shift = int(np.argmax([np.sum(L * np.roll(Rm, k, axis=0)) for k in range(n)]))
    SL, STL = 0.5 * (L + np.roll(Rm, shift, axis=0)), 0.5 * (TL + np.roll(TRm, shift, axis=0))
    SR, STR = (np.asarray(x) for x in _mirror_cycle(jnp.asarray(SL), jnp.asarray(STL)))
    o2, t2 = o.copy(), t.copy(); o2[2], o2[3], t2[2], t2[3] = SL, SR, STL, STR
    h = np.asarray(hz).copy(); h[2] = h[3] = 0.5 * (h[2] + h[3])
    return jnp.asarray(o2), jnp.asarray(t2), jnp.asarray(h)


CYC_OFF_SYM, CYC_TAU_SYM, CYC_HZ_SYM = _sym_turn_cycles(CYC_OFF, CYC_TAU, CYC_HZ)
ERR_BAR12 = jnp.tile(jnp.array([0.60, 0.45, 0.45]), 4)      # 分關節誤差護欄：ABAD 0.6（原廠命令差本就 30°）、髖膝 0.45


def _cyc_interp(table, phi):
    """table (4,N,D)、phi 弧度 → (4,D) 線性內插。"""
    x = jnp.mod(phi / (2 * jnp.pi), 1.0) * CYC_N
    i0 = jnp.floor(x).astype(jnp.int32) % CYC_N
    i1 = (i0 + 1) % CYC_N
    f = x - jnp.floor(x)
    return table[:, i0] * (1.0 - f) + table[:, i1] * f


def cycle_offsets(phi, cmd, A, ref=None, amp_turn=None):
    """v3.2：原廠命令週期 → dict(delta(12) 關節目標偏移、wheel_tau(4) 輪力矩偏移 N·m、hz、on ∈[0,1] 週期族佔比)。
    方向用平滑符號選左右檔；幅度隨指令縮放；平移／旋轉依 a_lat:a_turn 混合。"""
    ref = ref or REF
    sym = bool(ref.get("cyc_turn_sym", False))                 # Python 層選表，預設路徑逐位元不變
    T_OFF, T_TAU, T_HZ = (CYC_OFF_SYM, CYC_TAU_SYM, CYC_HZ_SYM) if sym else (CYC_OFF, CYC_TAU, CYC_HZ)
    off = _cyc_interp(T_OFF, phi)
    tau = _cyc_interp(T_TAU, phi)
    w_ll = 0.5 * (1.0 + jnp.clip(cmd[1] / 0.02, -1.0, 1.0))
    w_tl = 0.5 * (1.0 + jnp.clip(cmd[2] / 0.3, -1.0, 1.0))
    amp_l = jnp.clip(ref["cyc_lat_amp0"] + ref["cyc_lat_slope"] * (jnp.abs(cmd[1]) - ref["cyc_lat_vy0"]), *ref["cyc_lat_amp_clip"]) * ref["cyc_amp_lat"]
    amp_t = jnp.clip(jnp.abs(cmd[2]) / ref["a_ref"]["wz"], 0.3, 1.2) * (ref["cyc_amp_turn"] if amp_turn is None else amp_turn)
    on_l, on_t = float(ref["step_gen_lat"] == "cycle"), float(ref["step_gen_turn"] == "cycle")
    wl = _lat_weight(A)
    def blend(T, sc_l=1.0, sc_t=1.0):
        lat = w_ll * T[0] + (1.0 - w_ll) * T[1]
        turn = w_tl * T[2] + (1.0 - w_tl) * T[3]
        return wl * amp_l * on_l * lat * sc_l + (1.0 - wl) * amp_t * on_t * turn * sc_t
    if ref.get("cyc_lat_decouple", False):                        # v3.7：ABAD 倍率隨指令、髖膝固定（除以 amp_l 抵掉共同幅度）
        k_abad = jnp.clip(ref["cyc_lat_abad0"] + ref["cyc_lat_abad_slope"] * (jnp.abs(cmd[1]) - ref["cyc_lat_abad_vy0"]), *ref["cyc_lat_abad_clip"])
        lift = ref["cyc_lat_lift"]; lh, lk = (lift, lift) if np.isscalar(lift) else lift      # 純量＝髖膝同倍率；(髖, 膝) 可分開
        sc_lat = jnp.tile(jnp.array([1.0, 0.0, 0.0]) * k_abad + jnp.array([0.0, lh, lk]), 4) / amp_l
    else:
        sc_lat = jnp.tile(jnp.array(ref["cyc_joint_scale_lat"]), 4)
    delta = blend(off, sc_lat, jnp.tile(jnp.array(ref["cyc_joint_scale_turn"]), 4))
    if not ref["cyc_recenter"]:                                  # 絕對模式：中心用原廠實際 q 平均，不是我們的站姿
        qm = _cyc_interp(CYC_QMEAN[:, None, :].repeat(2, 1), 0.0)
        delta = delta + (wl * on_l * (w_ll * qm[0] + (1.0 - w_ll) * qm[1]) + (1.0 - wl) * on_t * (w_tl * qm[2] + (1.0 - w_tl) * qm[3])
                         - (wl * on_l + (1.0 - wl) * on_t) * STANCE_Q12_j)
    wheel_tau = blend(tau) * ref["cyc_wheel"]          # 錄檔原始 τ（N·m）；速度空間由呼叫端除 kv
    hz = wl * (w_ll * T_HZ[0] + (1.0 - w_ll) * T_HZ[1]) + (1.0 - wl) * (w_tl * T_HZ[2] + (1.0 - w_tl) * T_HZ[3])
    return dict(delta=delta, wheel_tau=wheel_tau, hz=hz * ref["cyc_hz_scale"], on=wl * on_l + (1.0 - wl) * on_t)


def ema(prev, x, k):
    """一階低通 prev + k·(x − prev)。k=0.02 @ 50 Hz ≈ 1 s 時間常數；2.5 Hz 週期擺剩 6%，慢漂全留。"""
    return prev + k * (x - prev)


def drift_terms(abad_ema, drift_ema, w):
    """v3.5：ABAD 慢漂（4 腿偏離站姿的低通）與機身速度低頻誤差（vx, vy）的懲罰 → (t_abadbias, t_drift)。
    預設二次式（權重 0 時精確為 0.0）；v3.6 `PEN_SHAPE="hinge"`：死區＋線性 —— 二次式在接近目標時梯度自己消失
    （0.038 m/s 時只剩 0.058／步），線性到目標仍有斜率；死區留給原廠圖案本來就有的 ±3° 不對稱（spec 2026-09-22 §2.2）。"""
    if w.get("PEN_SHAPE") == "hinge":
        ta = w["W_ABADBIAS_L"] * jnp.sum(jnp.maximum(jnp.abs(abad_ema) - w["ABAD_DZ"], 0.0))
        td = w["W_DRIFT_L"] * jnp.sum(jnp.maximum(jnp.abs(drift_ema) - w["DRIFT_DZ"], 0.0))
        return ta, td
    return w["W_ABADBIAS"] * jnp.sum(abad_ema ** 2), w["W_DRIFT"] * jnp.sum(drift_ema ** 2)


def step_apex_update(phi_new, phi_old, clr, apex_run, apex_last):
    """v3.6：每週期抬腳頂點。phi_cyc 回繞（新 < 舊）＝週期結束：上一週期的 running max 交給 apex_last、apex_run 從當步 clr 重新起算。
    phi 不前進（u4 ≤ 0.01）時不會回繞，apex_last 保持。"""
    wrap = phi_new < phi_old
    return (jnp.where(wrap, clr, jnp.maximum(apex_run, clr)), jnp.where(wrap, apex_run, apex_last))


def step_apex_reward(apex_last, s_leg, cmd_vy, w):
    """v3.6：主動腿（正規化活動度 ≥ 0.9 → 1，0.7 → 0）上一週期頂點對目標的線性分數，到目標飽和。
    目標隨指令縮：週期表的抬高跟幅度一起縮，零動作 vy 0.12 只抬 7 mm，低速逼 21 mm 會拿平衡去換（spec §2.1）。"""
    s_n = s_leg / jnp.maximum(jnp.max(s_leg), 1e-6)
    w_leg = jnp.clip((s_n - 0.9) / 0.1, 0.0, 1.0)
    tgt = w["STEP_APEX"] * jnp.clip(jnp.abs(cmd_vy) / 0.15, 0.5, 1.0)
    return jnp.sum(w_leg * jnp.clip(apex_last / tgt, 0.0, 1.0)) / jnp.maximum(jnp.sum(w_leg), 1e-3)


def err_barrier_j(err12, bar12=ERR_BAR12):
    return jnp.sum(jnp.maximum(jnp.abs(err12) - bar12, 0.0) ** 2)


def _mirror_maps():
    """obs 88／動作 24 的左右鏡像（y → −y）：腿 FR↔FL、RR↔RL；ABAD、grav_y、gyro_x/z、cmd vy/wz、head_err、sway_y 變號。
    obs 排列見 DualModeEnv._obs：grav3 gyro3 q12 qvel12 wheel4 cmd3 [s4,s_arc] head1 last_a24 amp4 sinθ4 cosθ4 cyc_abad4 cyc_knee4 wheel4。"""
    swap = [1, 0, 3, 2]
    oi, os_ = np.arange(88), np.ones(88)
    ai, as_ = np.arange(ACT_DIM), np.ones(ACT_DIM)
    def swap12(idx, sg, base):
        for k in range(4):
            for j in range(3):
                idx[base + 3 * k + j] = base + 3 * swap[k] + j
            sg[base + 3 * k] = -1.0
    def swap4(idx, sg, base, neg=False):
        for k in range(4):
            idx[base + k] = base + swap[k]
            if neg: sg[base + k] = -1.0
    def act_map(idx, sg, base):
        swap4(idx, sg, base + 1); sg[base + 7] = -1.0; swap4(idx, sg, base + 8); swap12(idx, sg, base + 12)
    os_[1] = -1.0; os_[3] = -1.0; os_[5] = -1.0
    swap12(oi, os_, 6); swap12(oi, os_, 18); swap4(oi, os_, 30)
    os_[35] = -1.0; os_[36] = -1.0; os_[39] = -1.0
    act_map(oi, os_, 40)
    for b in (64, 68, 72): swap4(oi, os_, b)
    swap4(oi, os_, 76, neg=True); swap4(oi, os_, 80); swap4(oi, os_, 84)
    act_map(ai, as_, 0)
    return jnp.array(oi), jnp.array(os_), jnp.array(ai), jnp.array(as_)


OBS_MIRROR_IDX, OBS_MIRROR_SGN, ACT_MIRROR_IDX, ACT_MIRROR_SGN = _mirror_maps()


def mirror_obs(obs):
    """θ 的相位錨在 0 號腿（step 裡 phi0 = theta_free[0] − ph[0]），鏡像後四腿一起差一個常數相位 θ[0] − θ[1]：
    交換腿之後把 sin／cos 轉回去，這樣平移（差 0.19 週期）與原地轉（差半週期）都精確。"""
    o = obs[OBS_MIRROR_IDX] * OBS_MIRROR_SGN
    th = jnp.arctan2(obs[68:72], obs[72:76])
    d = th[0] - th[1]
    th_m = th[MIRROR_LR] + d
    return o.at[68:72].set(jnp.sin(th_m)).at[72:76].set(jnp.cos(th_m))


def mirror_act(a):
    return a[ACT_MIRROR_IDX] * ACT_MIRROR_SGN


def mirror_policy(pol, turn_only=True):
    """鏡像推論（v3.7，不用重訓；E10）：右轉（wz < −0.1）時把 obs 鏡像餵 policy、動作鏡像回來 → 右轉用左轉的能力
    （v3.6f 右轉 −62±10 → −84±1，與左轉 +83.5 對稱）。平移鏡像互有好壞（航向好、側傾略差），預設只套原地轉；turn_only=False 也套 vy < 0。"""
    def run(obs):
        cmd = obs[34:37]
        right = jnp.where(jnp.abs(cmd[2]) > 0.1, cmd[2] < 0, (cmd[1] < -0.02) & (not turn_only))
        a_m = mirror_act(pol(mirror_obs(obs)))
        return jnp.where(right, a_m, pol(obs))
    return run


def _mirror(v, right):
    """right 為 True（右向指令）→ 左右鏡像。"""
    return jnp.where(right, v[MIRROR_LR], v)


def activity(cmd, ref=None):
    """→ dict(lat, turn, wheel, s4, arc)，全是 0–1 純量。"""
    ar = (ref or REF)["a_ref"]
    a_lat = jnp.clip(jnp.abs(cmd[1]) / ar["vy"], 0.0, 1.0)
    a_turn = jnp.clip(jnp.abs(cmd[2]) / ar["wz"], 0.0, 1.0)
    a_wheel = jnp.clip(jnp.abs(cmd[0]) / ar["vx"], 0.0, 1.0)
    return dict(lat=a_lat, turn=a_turn, wheel=a_wheel,
                s4=jnp.maximum(a_lat, a_turn * (1.0 - a_wheel)), arc=a_turn * a_wheel)


def _lat_weight(A):
    return A["lat"] / (A["lat"] + A["turn"] + 1e-6)


def leg_activity(cmd, A):
    """每腿活動度 s_k (4,)：四腿踏步（平移／原地轉，主導 1.0 次要 0.7）與弧線（內前 1.0、對角後 0.5）取大。"""
    w_lat = _mirror(W_LAT_L, cmd[1] < 0)
    w_turn = _mirror(W_TURN_L, cmd[2] < 0)
    w_arc = _mirror(W_ARC_L, cmd[2] < 0)
    wl = _lat_weight(A)
    return jnp.maximum(A["s4"] * (wl * w_lat + (1.0 - wl) * w_turn), A["arc"] * w_arc)


def step_gain(s):
    """吃步向量的比例：s_k ≥ STEP_GAIN_S 就是 1（stance 腳要同速）。"""
    return jnp.clip(s / STEP_GAIN_S, 0.0, 1.0)


def phase_offsets(cmd, A, ref=None):
    """四腿相位偏移目標 (4,)：旋轉族與平移族各自的相位組依 a_lat:a_turn 圓周插值；方向鏡像。"""
    ref = ref or REF
    ph_t = PH_TROT if ref["phase_set_turn"] == "trot" else _mirror(PH_TURN_L, cmd[2] < 0)
    ph_l = PH_TROT if ref["phase_set_lat"] == "trot" else _mirror(PH_LAT_L, cmd[1] < 0)
    wl = _lat_weight(A)
    z = wl * jnp.exp(2j * jnp.pi * ph_l) + (1.0 - wl) * jnp.exp(2j * jnp.pi * ph_t)
    return jnp.mod(jnp.angle(z) / (2 * jnp.pi), 1.0)


def slew_phase(ph, ph_tgt, max_step=PH_SLEW):
    d = jnp.mod(ph_tgt - ph + 0.5, 1.0) - 0.5
    return jnp.mod(ph + jnp.clip(d, -max_step, max_step), 1.0)


def kin_step_vec(cmd, T, ref=None):
    """每腳每週期踏步位移 (4,2) m：vy 全給踏步；旋轉只給 rot_step_frac；vx 不踏步（輪子滾）。"""
    ref = ref or REF
    rot = ref["rot_step_frac"] * cmd[2]
    dx = T * (-rot * FOOT_XY_BODY[:, 1])
    dy = T * (ref["lat_cmd_gain"] * cmd[1] + rot * FOOT_XY_BODY[:, 0])
    return jnp.stack([dx, dy], 1)


def posture_offset(cmd, A, ref=None):
    """姿態 y 偏移 (4,)：弧線＝內側腿（REF posture_arc 正值 = 往中線）；平移＝四腳外張 posture_lat × a_lat。"""
    ref = ref or REF
    p = ref["posture_arc"]
    inner = jnp.where(cmd[2] > 0, jnp.array([0.0, 1.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 1.0, 0.0]))
    amount = FRONT_j * p["front"] + (1.0 - FRONT_j) * p["rear"]
    arc = A["arc"] * inner * amount * (-SIDE_Y_j)
    lat = A["lat"] * ref.get("posture_lat", 0.0) * SIDE_Y_j
    return arc + lat


def wheel_cmd(cmd, A, res, ref=None):
    """每輪 rad/s (4,)：差速 × gate ＋ 平移圖案 ＋ 殘差。gate：原地轉主導對 0、另兩輪 2；vx 大或沒轉時 1。"""
    ref = ref or REF
    diff = (cmd[0] - cmd[2] * SIDE_Y_j * ref["L_eff"] / 2) / ref["r_wheel"]
    dom = _mirror(DOM_TURN_L, cmd[2] < 0)
    gate = 1.0 + A["turn"] * (1.0 - A["wheel"]) * (1.0 - 2.0 * dom)
    lw = _mirror(LAT_WHEEL_L, cmd[1] < 0) * ref["lat_omega"] * jnp.clip(jnp.abs(cmd[1]) / ref["lat_vy_ref"], 0.0, 1.5)
    return diff * gate + lw + res


def wheel_ctrl(v_des):
    """實機控制律 τ = kv·(v_des − v) + τ_ff·sign 的 MJX 等價：ctrl = v_des + sign·τ_ff/kv；死區 0.3。"""
    dead = jnp.abs(v_des) < V_DEAD
    return jnp.where(dead, 0.0, v_des + jnp.sign(v_des) * TAU_FF / KV_WHEEL)


def wheel_ctrl_tau(tau_cmd, v_meas, kv):
    """v3.4f 力矩空間：回傳送給速度伺服的 ctrl，使實際力矩 = tau_cmd ＋ τ_f·sign（克服輪摩擦）。

    原廠輪子 kp 0／kd 0.1 ＋ 大幅速度目標（反推 v_des 20–50 rad/s 對實際 12–17，spec §0.2），
    等於把 PD 當力矩源。kv 0.1 下速度殘差沒有力矩權限，所以殘差與週期偏移都改成直接指定力矩。
    |tau_cmd| < τ_f 視為推不動 → 輸出零力矩（ctrl = 實測速度）。
    """
    dead = jnp.abs(tau_cmd) < TAU_FF
    tau = jnp.where(dead, 0.0, tau_cmd + jnp.sign(tau_cmd) * TAU_FF)
    return v_meas + tau / kv


def step_pattern(cmd, ref=None):
    """指令 → 產生器設定 dict：A、s(4)、g(4)、vec(4,2)、ph(4) 目標、wheel0(4) 無殘差、hz、duty、lift(4)、post_y(4)。"""
    ref = ref or REF
    A = activity(cmd, ref)
    s = leg_activity(cmd, A)                                     # 完整活動度（reward 的抬腿紀律用）
    wl = _lat_weight(A)
    # v3.2：四腿踏步族若走原廠週期，運動學踏步只留弧線那份
    cyc_on = wl * float(ref["step_gen_lat"] == "cycle") + (1.0 - wl) * float(ref["step_gen_turn"] == "cycle")
    A_kin = dict(A, s4=A["s4"] * (1.0 - cyc_on))
    s_kin = leg_activity(cmd, A_kin)
    hz = wl * ref["step_hz_lat"] + (1.0 - wl) * ref["step_hz_turn"]
    duty = wl * ref["duty_lat"] + (1.0 - wl) * ref["duty_turn"]
    lift = wl * ref["lift_lat"] + (1.0 - wl) * ref["lift"]
    return dict(A=A, s=s, s_kin=s_kin, g=step_gain(s_kin), vec=kin_step_vec(cmd, 1.0 / hz, ref), ph=phase_offsets(cmd, A, ref),
                wheel0=wheel_cmd(cmd, A, jnp.zeros(4), ref), hz=hz, duty=duty, cyc_on=cyc_on,
                lift=lift * (s_kin / jnp.maximum(jnp.max(s_kin), 1e-6)) * step_gain(s_kin), post_y=posture_offset(cmd, A, ref))


def duty_remap(th, duty):
    return v2.duty_remap(th, duty)


def act_split(a):
    """tanh 後的 12 維動作 → 各尺度。wres 是 rad/s 偏移（kd 1.0 下 ≈ N·m）。"""
    a = jnp.tanh(a)
    return dict(om=1.0 + OMEGA_NOM_SCALE * a[0], amp=1.0 + AMP_SCALE * a[1:5], lift=1.0 + LIFT_SCALE * a[5],
                sway=SWAY_MAX * a[6:8], wres=WHEEL_RES * a[8:12], qres=QRES_MAX * a[12:24],
                foot_x=FOOT_OFF_MAX * a[1:5], foot_z=FOOT_OFF_MAX * a[5])


def foot_targets(theta, amp, g, vec, lift4, sway, u, foot_x, foot_z, duty, post_y, z_sag=Z_SAG):
    """(4,3) 足端目標（相對各腿 ABAD）。
    踏步：站姿 ＋ 姿態偏移 ＋ g·amp·vec 的往返（swing 往 +vec、stance 往 −vec）＋ 抬高（只在 swing，加 g·Z_SAG）；
    輪行：站姿 ＋ RL 小偏移。u = max_k s_k（斜率限制）做兩者混合。"""
    th = duty_remap(theta, duty)
    s = jnp.sin(th)
    prog = -0.5 * jnp.cos(th)                                  # −0.5 → +0.5 across swing, back across stance
    ramp = jnp.clip(amp, 0.0, 1.0)                             # amp 從 0 斜升（reset／切指令時抬高也跟著斜升）
    dxy = (vec * (amp * g)[:, None]) * prog[:, None]
    dz = jnp.where(s > 0, (lift4 + z_sag * g) * s, 0.0) * ramp
    # 踏步位移本身已由 s_k／g_k 決定大小（s=0 時全零），不再乘 u —— 乘 u 會把 s<1 的抬高再打一次折（vy 0.03 時 51→19 mm，腳離不了地）
    step_off = jnp.concatenate([dxy, dz[:, None]], 1) + u * jnp.array([sway[0], sway[1], 0.0])
    post = jnp.stack([jnp.zeros(4), post_y, jnp.zeros(4)], 1)
    wheel_off = jnp.stack([foot_x, jnp.zeros(4), jnp.full(4, foot_z)], 1)
    return STANCE_FEET_j + post + step_off + (1.0 - u) * wheel_off


def joint_targets(feet):
    return jnp.stack([ik_j(k, feet[k], KNEE_SIGN_j[k]) for k in range(4)]).reshape(12)


def foot_actual(q12):
    q = q12.reshape(4, 3)
    return jnp.stack([fk_j(k, q[k]) for k in range(4)])


# ---------------------------------------------------------------- env
class DualModeEnv(Env):
    def __init__(self, scene: str = None, wheel_pos: bool = False, ref: dict = None, weights: dict = None, gains: str = "kp250", push: bool = True):
        """`wheel_pos=False`（預設）：純速度伺服（M11 實機已驗的 kd 1.0）；
        True：位置＋速度伺服（scene_flat_mjx_v3p，kp 60，ctrl=累加目標角；實機未驗，v3.1 不用）。
        `gains="kp250"`＝v3.3；`"factory"`＝馬達增益完全照原廠錄檔動作段（v3.4f）。`ref`／`weights` 覆寫 REF／W（掃參數用）。"""
        assert gains in GAIN_SETS, f"gains 只能是 {set(GAIN_SETS)}"
        self.gains = gains
        self.push_vel = PUSH_VEL if push else 0.0        # v3.6：push=False（驗收用）關掉每 PUSH_EVERY 步的隨機推力；訓練預設開
        G = GAIN_SETS[gains]
        self.w = dict(W, **(weights or {}))
        self.ref = {**REF, **G["ref"], **(ref or {})}      # 用 dict(REF, **a, **b) 會在 key 重複時炸（掃描時覆寫 REF_FACTORY 的鍵就會踩到）
        if self.w["CYC_TURN_SYM"]:
            self.ref = dict(self.ref, cyc_turn_sym=True)     # v3.5：右轉＝左轉鏡像平均（§2.6）
        if self.w.get("CYC_LAT_DECOUPLE", False):
            self.ref = dict(self.ref, cyc_lat_decouple=True)  # v3.7：平移抬高／跨距解耦
        self.wheel_pos = wheel_pos
        self.wheel_space = self.ref["wheel_space"]
        if scene is None:
            scene = SCENE_V3P if wheel_pos else G["scene"]
        m = mujoco.MjModel.from_xml_path(scene)
        if self.ref.get("floor_mu") is not None:
            m.geom_friction[:, 0] = self.ref["floor_mu"]
        wheel_gain_kp = float(m.actuator_gainprm[mm.WHEEL_ACT_IDX[0], 0])
        wheel_bias_q = float(m.actuator_biasprm[mm.WHEEL_ACT_IDX[0], 1])
        assert (wheel_bias_q < -1e-6) == wheel_pos, "wheel_pos 與模型的輪致動器型式不符"
        self.kv_wheel = float(-m.actuator_biasprm[mm.WHEEL_ACT_IDX[0], 2])
        self.kp_wheel = -wheel_bias_q
        assert m.opt.timestep == SIM_DT
        assert m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
        self._mj = m
        self.sys = mjx.put_model(m)
        self._lo = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 0])
        self._hi = jnp.array(m.jnt_range[mm.leg_joint_ids(m), 1])
        gids = [mm._id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{mm.PREFIX[l]}_FOOT_LINK_COLL") for l in mm.LEGS]
        self._wheel_gids = jnp.array(gids)
        self._wheel_r = float(m.geom_size[gids[0]][0])
        self.obs_dim = 3 + 3 + 12 + 12 + 4 + 3 + 2 + 1 + ACT_DIM + 24
        self.err_bar12 = jnp.tile(jnp.array(self.ref["err_bar3"]), 4)
        self._init_q = self._settled_qpos(m)

    def _settled_qpos(self, m):
        d = mujoco.MjData(m)
        q = np.asarray(STANCE_Q12_j)
        d.qpos[:3] = [0.0, 0.0, self.ref["nominal_height"] + 0.01]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[mm.LEG_QPOS_IDX] = q
        mujoco.mj_forward(m, d)
        for _ in range(int(1.5 / SIM_DT)):
            d.ctrl[mm.LEG_ACT_IDX] = q
            d.ctrl[mm.WHEEL_ACT_IDX] = 0.0
            mujoco.mj_step(m, d)
        lo, hi = self.ref["settle_h_range"]
        assert lo < d.qpos[2] < hi, f"站定高度 {d.qpos[2]:.3f} m 不在 ({lo}, {hi})"
        return jnp.array(d.qpos)

    def _wheel_ctrl(self, wheel_q, v_meas, wheel_theta=None):
        """wheel_q：速度空間＝v_des(rad/s)、力矩空間＝tau_cmd(N·m)。→ 輪致動器 ctrl。"""
        if self.wheel_pos:
            # 位置環：ctrl = 累加目標角；前饋等價 = 加 τ_ff/kp 的角度偏移
            dead = jnp.abs(wheel_q) < V_DEAD
            return jnp.where(dead, wheel_theta, wheel_theta + jnp.sign(wheel_q) * TAU_FF / self.kp_wheel)
        if self.wheel_space == "tau":
            return wheel_ctrl_tau(wheel_q, v_meas, self.kv_wheel)
        return wheel_ctrl(wheel_q)

    def _ctrl(self, q_des, w_ctrl):
        return jnp.zeros(16).at[LEG_ACT_IDX].set(q_des).at[WHEEL_ACT_IDX].set(w_ctrl)

    def _wheel_clearance(self, data):
        return data.geom_xpos[self._wheel_gids, 2] - self._wheel_r

    def _gyro_obs(self, data, info):
        return _qrot(info["imu_q"], data.qvel[3:6]) + info["gyro_bias"]

    def _obs(self, data, info, last_a):
        grav = _qrot(info["imu_q"], w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0])))
        gyro = self._gyro_obs(data, info)
        c = info["c"]
        return jnp.concatenate([
            grav, gyro,
            data.qpos[LEG_QPOS_IDX] - STANCE_Q12_j,
            info["qvel_prev"],
            data.qvel[WHEEL_QVEL_IDX],
            info["cmd"], jnp.array([info["s4"], info["s_arc"]]),
            jnp.clip(info["head_err"], -1.0, 1.0)[None],
            last_a,
            c["amp"], jnp.sin(c["theta"]), jnp.cos(c["theta"]), c["cyc_abad"], c["cyc_knee"], c["wheel"] / 5.0,
        ])

    def _sample_cmd(self, rng):
        """三軸獨立抽：vx 65%、vy 45%、wz 50% 非零；有 wz 時 35% 機率只留 wz（純原地轉）；全零（站立）自然約 10%。"""
        k = jax.random.split(rng, 9)
        w = self.w
        sgn = lambda kk: jnp.sign(jax.random.uniform(kk) - 0.5)   # noqa: E731
        vx = jnp.where(jax.random.uniform(k[0]) < w["P_VX"],
                       jax.random.uniform(k[1], minval=w["CMD_VX"][0], maxval=w["CMD_VX"][1]), 0.0)
        vy = jnp.where(jax.random.uniform(k[2]) < w["P_VY"],
                       jax.random.uniform(k[3], minval=w["CMD_VY"][0], maxval=w["CMD_VY"][1]) * sgn(k[4]), 0.0)
        wz = jnp.where(jax.random.uniform(k[5]) < w["P_WZ"],
                       jax.random.uniform(k[6], minval=w["CMD_WZ"][0], maxval=w["CMD_WZ"][1]) * sgn(k[7]), 0.0)
        turn_only = (jnp.abs(wz) > 0) & (jax.random.uniform(k[8]) < w["P_TURN_ONLY"])
        vx = jnp.where(turn_only, 0.0, vx)
        vy = jnp.where(turn_only, 0.0, vy)
        return jnp.array([vx, vy, wz])

    def reset(self, rng):
        ks = jax.random.split(rng, 8)
        data = mjx.make_data(self.sys).replace(qpos=self._init_q)
        wheel_theta = self._init_q[jnp.array(mm.WHEEL_QPOS_IDX)]
        data = data.replace(ctrl=self._ctrl(self._init_q[LEG_QPOS_IDX],
                                            self._wheel_ctrl(jnp.zeros(4), jnp.zeros(4), wheel_theta)))
        data = mjx.forward(self.sys, data)
        cmd = self._sample_cmd(ks[0])
        cmd2 = self._sample_cmd(ks[1])
        k_amp = jax.random.fold_in(ks[2], 7)
        lo_a, hi_a = self.ref["cyc_amp_turn_range"]
        cyc_amp = jnp.where(self.ref["cyc_amp_rand"], jax.random.uniform(k_amp, minval=lo_a, maxval=hi_a), self.ref["cyc_amp_turn"])
        do_switch = jax.random.uniform(ks[2]) < self.w["P_SWITCH"]
        t_switch = jnp.where(do_switch, jax.random.randint(ks[3], (), 150, 400), 10 ** 6)
        tilt = jax.random.uniform(ks[4], (2,), minval=-IMU_TILT_DEG, maxval=IMU_TILT_DEG) * jnp.pi / 180
        P0 = step_pattern(cmd, self.ref)
        c = dict(theta=jnp.zeros(4), amp=jnp.zeros(4), vec=jnp.zeros((4, 2)), wheel=jnp.zeros(4), cyc_abad=jnp.zeros(4), cyc_knee=jnp.zeros(4))
        z = jnp.zeros(ACT_DIM)
        info = {"rng": ks[5], "c": c, "cmd": cmd, "cmd2": cmd2, "t_switch": t_switch,
                "u_mode": jnp.max(P0["s"]), "ph": P0["ph"], "s4": P0["A"]["s4"], "s_arc": P0["A"]["arc"],
                "phi_cyc": jnp.zeros(()), "u4": jnp.zeros(()), "wz_ema": jnp.zeros(()), "vy_ema": jnp.zeros(()),
                "cyc_amp": cyc_amp, "qres": jnp.zeros(12), "gyro_bias": jax.random.uniform(ks[6], (3,), minval=-1.0, maxval=1.0) * GYRO_BIAS,
                "head_err": jnp.zeros(()), "imu_q": _quat_rp(tilt[0], tilt[1]),
                "delay": DELAY_BASE + jax.random.bernoulli(ks[7], 0.5).astype(jnp.int32),
                "a_hist": jnp.zeros((3, ACT_DIM)), "last_a": z, "sway": jnp.zeros(2),
                "qvel_prev": data.qvel[LEG_QVEL_IDX], "om_prev": jnp.zeros(()), "wheel_theta": wheel_theta,
                "roll_ema": jnp.zeros(()), "sway_ema": jnp.zeros(()),
                "abad_ema": jnp.zeros(4), "drift_ema": jnp.zeros(2),
                "apex_run": jnp.zeros(4), "apex_last": jnp.zeros(4),
                "kill": jnp.zeros((), jnp.int32), "step": 0,
                "mir": (jax.random.bernoulli(jax.random.fold_in(ks[7], 13)) if self.w.get("MIRROR_AUG", False) else jnp.zeros((), bool))}   # v3.7b：鏡像增強回合
        obs = self._obs(data, info, z)
        if self.w.get("MIRROR_AUG", False):
            obs = jnp.where(info["mir"], mirror_obs(obs), obs)
        zz = jnp.zeros(())
        metrics = {k: zz for k in METRIC_KEYS}
        return State(data, obs, zz, zz, metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        w = self.w
        if w.get("MIRROR_AUG", False):                                    # v3.7b：鏡像回合 —— policy 看到的是鏡像 obs，它的動作鏡像回來給物理
            action = jnp.where(info["mir"], mirror_act(action), action)
        step_i = info["step"]
        cmd = jnp.where(step_i >= info["t_switch"], info["cmd2"], info["cmd"])
        # ---- 產生器設定（連續，無模式）
        P = step_pattern(cmd, self.ref)
        u_tgt = jnp.max(P["s"])
        du = CTRL_DT / self.ref["trans_s"]
        u_mode = jnp.clip(info["u_mode"] + jnp.clip(u_tgt - info["u_mode"], -du, du), 0.0, 1.0)
        # ---- 動作（延遲、淡入）
        a_hist = jnp.concatenate([action[None], info["a_hist"][:2]], 0)
        act = a_hist[info["delay"]]
        u_ramp = jnp.clip(step_i / w["RAMP_STEPS"], 0.0, 1.0)
        act = u_ramp * act
        A = act_split(act)
        if w.get("LIFT_NONNEG", False):                                   # v3.7b：lift ∈ [1, 1+LIFT_SCALE]（a ≤ 0 死區＝名目），policy 不能把名目抬高縮掉
            A = dict(A, lift=1.0 + jnp.maximum(A["lift"] - 1.0, 0.0))
        # ---- 相位：共同相位推進 ＋ 每腿偏移（偏移目標變了就限速追）
        om = P["hz"] * A["om"]
        ph = slew_phase(info["ph"], P["ph"])
        theta_free = info["c"]["theta"] + 2 * jnp.pi * om * CTRL_DT
        ph_rad = 2 * jnp.pi * ph                                        # ph 是週期分數，θ 是弧度（v3.0 漏乘 2π）
        phi0 = theta_free[0] - ph_rad[0]
        theta = jnp.where(u_mode > 0.01, phi0 + ph_rad, info["c"]["theta"]) % (2 * jnp.pi)
        amp = info["c"]["amp"] + jnp.clip(P["g"] * A["amp"] - info["c"]["amp"], -0.05, 0.05)   # 振幅斜率
        sway = info["sway"] + jnp.clip(A["sway"] * u_mode - info["sway"], -SWAY_SLEW, SWAY_SLEW)
        lift4 = P["lift"] * A["lift"]
        feet = foot_targets(theta, amp, P["g"], P["vec"], lift4, sway, u_mode, A["foot_x"], A["foot_z"], P["duty"], P["post_y"], self.ref["z_sag"])
        # ---- v3.2：原廠命令週期（四腿踏步族），疊在運動學目標上；u4 = 四腿活動度的斜率版
        u4_tgt = jnp.clip(P["A"]["s4"] / 0.3, 0.0, 1.0) * P["cyc_on"]      # 開關，不是幅度
        u4 = jnp.clip(info["u4"] + jnp.clip(u4_tgt - info["u4"], -du, du), 0.0, 1.0)
        CY = cycle_offsets(info["phi_cyc"], cmd, P["A"], self.ref, info["cyc_amp"])
        phi_cyc = jnp.mod(info["phi_cyc"] + 2 * jnp.pi * CY["hz"] * A["om"] * CTRL_DT * (u4 > 0.01), 2 * jnp.pi)
        if self.ref["cyc_lat_decouple"]:                                 # v3.7：amp（每腿）只縮 ABAD 跨距、lift 只縮髖膝抬高
            delta = (CY["delta"].reshape(4, 3) * jnp.stack([A["amp"], jnp.full(4, A["lift"]), jnp.full(4, A["lift"])], 1)).reshape(12) * u4
        else:
            delta = (CY["delta"].reshape(4, 3) * A["amp"][:, None] * jnp.array([1.0, 1.0, 1.0])[None, :]
                     * jnp.array([1.0, A["lift"], A["lift"]])[None, :]).reshape(12) * u4
        qres = info["qres"] + jnp.clip(A["qres"] - info["qres"], -QRES_SLEW, QRES_SLEW)   # 關節殘差（限速）
        q_des = jnp.clip(joint_targets(feet) + delta + qres, self._lo, self._hi)
        # ---- 輪：差速×gate ＋ 平移圖案 ＋ 原廠 τ 週期偏移 ＋ 殘差
        #   速度空間（v3.3）：全部在 rad/s，錄檔 τ 除 kv 換算
        #   力矩空間（v3.4f，原廠 kd 0.1）：差速走外環 τ = G_w·(v_des − v)，錄檔 τ 與殘差直接是 N·m
        data = state.pipeline_state                                       # 外環回授用的是這一控制步開始時的實測輪速（在 kick 之前）
        cmd_w = cmd
        if self.ref["cyc_lat_decouple"] and w.get("CYC_LAT_FF", False):   # v3.7b：平移名目的漂移／航向前饋（只進輪命令）；v3.7f 訓練時沒有 → W 旗標控制
            a_, b_, lo_, hi_ = self.ref["cyc_lat_vx_ff"]
            has_vy = (jnp.abs(cmd[1]) > 0.02).astype(jnp.float32) * _lat_weight(P["A"]) * u4
            cmd_w = cmd + has_vy * jnp.array([1.0, 0.0, 0.0]) * jnp.clip(a_ + b_ * jnp.abs(cmd[1]), lo_, hi_) \
                        + has_vy * jnp.array([0.0, 0.0, 1.0]) * self.ref["cyc_lat_wz_ff"] * jnp.sign(cmd[1])
        A_w = activity(cmd_w, self.ref) if (self.ref["cyc_lat_decouple"] and w.get("CYC_LAT_FF", False)) else P["A"]   # 輪命令的活動度 gate 要看前饋後的指令
        v_nom = wheel_cmd(cmd_w, A_w, jnp.zeros(4), self.ref)
        v_wheel_meas = data.qvel[WHEEL_QVEL_IDX]
        if self.wheel_space == "tau":
            wheel_q = self.ref["wheel_outer_gain"] * (v_nom - v_wheel_meas) + A["wres"] + u4 * CY["wheel_tau"]
        else:
            wheel_q = v_nom + A["wres"] + u4 * CY["wheel_tau"] / self.kv_wheel
        wheel_theta = info["wheel_theta"] + wheel_q * CTRL_DT             # 位置環變體用的累加目標角
        d3 = delta.reshape(4, 3)
        c = dict(theta=theta, amp=amp, vec=P["vec"] * amp[:, None], wheel=wheel_q, cyc_abad=d3[:, 0], cyc_knee=d3[:, 2])
        s_leg, g_leg, duty = P["s"], P["g"], P["duty"]

        rng, k_push, k_dir, k_obs = jax.random.split(info["rng"], 4)
        do_push = (step_i % PUSH_EVERY) == (PUSH_EVERY - 1)
        ang = jax.random.uniform(k_dir, minval=0.0, maxval=2 * jnp.pi)
        mag = jax.random.uniform(k_push, minval=0.0, maxval=self.push_vel)
        kick = jnp.where(do_push, jnp.array([mag * jnp.cos(ang), mag * jnp.sin(ang), 0.0]), jnp.zeros(3))
        data = data.replace(qvel=data.qvel.at[0:3].add(kick))
        ctrl = self._ctrl(q_des, self._wheel_ctrl(wheel_q, v_wheel_meas, wheel_theta))

        def one(carry, _):
            d, tpk = carry
            d = mjx.step(self.sys, d.replace(ctrl=ctrl))
            return (d, jnp.maximum(tpk, jnp.abs(d.actuator_force[LEG_ACT_IDX]))), None
        (data, tau_pk12), _ = jax.lax.scan(one, (data, jnp.zeros(12)), None, length=N_FRAMES)

        grav = w2b(data.qpos[3:7], jnp.array([0.0, 0.0, -1.0]))
        vb = w2b(data.qpos[3:7], data.qvel[0:3])
        wz = data.qvel[5]
        head_err = info["head_err"] + (self._gyro_obs(data, info)[2] - cmd[2]) * CTRL_DT
        q12 = data.qpos[LEG_QPOS_IDX]
        err12 = q_des - q12
        clr = self._wheel_clearance(data)
        apex_run, apex_last = step_apex_update(phi_cyc, info["phi_cyc"], clr, info["apex_run"], info["apex_last"])   # v3.6
        knee_v = jnp.max(jnp.abs(data.qvel[LEG_QVEL_IDX][KNEE12]))
        roll_ema = info["roll_ema"] + w["BIAS_EMA"] * (grav[1] - info["roll_ema"])
        sway_ema = info["sway_ema"] + w["BIAS_EMA"] * (sway[1] - info["sway_ema"])
        abad_ema = ema(info["abad_ema"], q12[ABAD12] - STANCE_Q12_j[ABAD12], w["BIAS_EMA"])   # v3.5：ABAD 偏離站姿的慢漂（cyc_recenter=True 下站姿＝名目中心）
        drift_ema = ema(info["drift_ema"], vb[0:2] - cmd[0:2], w["BIAS_EMA"])                # v3.5：機身 vx/vy 低頻追蹤誤差（濾掉踏步 ±0.3 的高頻擺）
        t_abadbias, t_drift = drift_terms(abad_ema, drift_ema, w)

        # ---- reward
        r_vx = jnp.exp(-(vb[0] - cmd[0]) ** 2 / w["VX_SIG2"])
        r_vy = jnp.exp(-(vb[1] - cmd[1]) ** 2 / w["VY_SIG2"])
        r_yaw = yaw_reward(wz, cmd[2], w["YAW_SIG2"], w["YAW_SIG2_WIDE"])
        r_yawi = yaw_reward(wz, cmd[2], w["YAW_INST_SIG2"])
        # 線性偏航追蹤：高斯核在誤差 ≥ 0.5 rad/s 全為 0、沒有梯度（原地轉名目 0.25 對指令 1.3），這一項在整個範圍給斜率
        r_yawlin = 1.0 - jnp.clip(jnp.abs(wz - cmd[2]) / w["YAW_LIN_E"], 0.0, 1.0)
        # 相對進度（只在有該軸指令時作用；偏航用 0.5 s 低通，否則小跑的 ±60°/s 來回擺也能拿分）：
        #   進度 = 沿指令方向的分量 / 指令大小，開根號讓 25% 的追蹤已值一半分 —— 零動作名目要明確贏過「站著不動」（spec §9.2）
        wz_ema = info["wz_ema"] + w["WZ_EMA"] * (wz - info["wz_ema"])
        vy_ema = info["vy_ema"] + w["WZ_EMA"] * (vb[1] - info["vy_ema"])          # 滑步的瞬時 vy ±0.3 來回擺，低通後才是真的側移
        has_wz = (jnp.abs(cmd[2]) > 0.1).astype(jnp.float32)
        has_vy = (jnp.abs(cmd[1]) > 0.02).astype(jnp.float32)
        prog_yaw = jnp.clip(wz_ema * jnp.sign(cmd[2]) / jnp.maximum(jnp.abs(cmd[2]), 0.1), 0.0, 1.0)
        prog_vy = jnp.clip(vy_ema * jnp.sign(cmd[1]) / jnp.maximum(jnp.abs(cmd[1]), 0.02), 0.0, 1.0)
        r_yawrel = has_wz * jnp.sqrt(prog_yaw + 1e-6)
        r_vyrel = has_vy * jnp.sqrt(prog_vy + 1e-6)
        k_post = 1.0 - w["POST_STEP_SCALE"] * u_mode
        r_head = jnp.exp(-(head_err / w["HEAD_SIG"]) ** 2)
        r_headlin = 1.0 - jnp.clip(jnp.abs(head_err) / w["HEAD_LIN_E"], 0.0, 1.0)   # v3.5：高斯核 8.6° 外沒梯度；線性項到 29° 都有（平移時航向轉掉 15–23° 的主因）
        r_h = jnp.exp(-400.0 * (data.qpos[2] - self.ref["nominal_height"]) ** 2)
        sw = (jnp.sin(duty_remap(theta, duty)) > 0).astype(jnp.float32) * g_leg
        wsw = sw * s_leg
        clr_step = jnp.sum(wsw * jnp.exp(-((clr - lift4) / 0.012) ** 2)) / jnp.maximum(jnp.sum(wsw), 1e-3)
        r_lift = u_mode * clr_step
        # 模式紀律（連續）：腿 k 抬 > 10 mm 的罰乘 (1 − s_k)
        lift_pen = jnp.sum(jnp.maximum(clr - 0.010, 0.0) ** 2 * (1.0 - s_leg))
        post = jnp.stack([jnp.zeros(4), P["post_y"], jnp.zeros(4)], 1)
        foot_dev = jnp.sum((foot_actual(q12) - STANCE_FEET_j - post) ** 2, 1)
        r_stance = (1 - u_mode) * jnp.mean(jnp.exp(-foot_dev / (0.02 ** 2)))
        r_step = step_apex_reward(apex_last, s_leg, cmd[1], w)                            # v3.6：每週期抬腳頂點（只在平移族）
        t_step = w["W_STEP"] * u4 * _lat_weight(P["A"]) * r_step
        c_act = jnp.sum((action - info["last_a"]) ** 2)
        c_omdot = (om - info["om_prev"]) ** 2 * u_mode
        c_tau = jnp.sum(data.actuator_force[LEG_ACT_IDX] ** 2)
        T = {
            "t_vx": w["W_VX"] * r_vx, "t_vy": w["W_VY"] * r_vy, "t_yaw": w["W_YAW"] * r_yaw, "t_yawi": w["W_YAWI"] * r_yawi,
            "t_yawlin": w["W_YAWLIN"] * r_yawlin, "t_yawrel": w["W_YAWREL"] * r_yawrel, "t_vyrel": w["W_VYREL"] * r_vyrel,
            "t_head": w["W_HEAD"] * r_head, "t_h": w["W_H"] * r_h, "t_lift": w["W_LIFT"] * r_lift, "t_stance": w["W_STANCE"] * r_stance,
            "t_roll": k_post * w["W_ROLL"] * grav[1] ** 2, "t_pitch": k_post * w["W_PITCH"] * grav[0] ** 2,
            "t_rollrate": k_post * w["W_ROLLRATE"] * data.qvel[3] ** 2, "t_pitchrate": k_post * w["W_PITCHRATE"] * data.qvel[4] ** 2,
            "t_bias": k_post * (w["W_BIAS"] * roll_ema ** 2 + w["W_SWAYBIAS"] * sway_ema ** 2),
            "t_act": w["W_ACT"] * c_act, "t_omdot": w["W_OMDOT"] * c_omdot, "t_qres": w["W_QRES"] * jnp.sum(qres ** 2), "t_tau": w["W_TAU"] * c_tau,
            "t_taubar": w["W_TAUBAR"] * tau_barrier(tau_pk12), "t_errbar": w["W_ERRBAR"] * err_barrier_j(err12, self.err_bar12),
            "t_kneev": w["W_KNEEV"] * jnp.maximum(knee_v - KNEE_V_BAR, 0.0) ** 2,
            "t_mode": w["W_MODE"] * lift_pen, "t_vz": w["W_VZ"] * data.qvel[2] ** 2,
            "t_abadbias": t_abadbias * u_mode, "t_drift": t_drift,   # v3.5；ABAD 項只在踏步時開（輪行的站姿由 r_stance 管；攤帳：直走本來就有 5° 常態外張，不該罰）
            "t_headlin": w["W_HEADLIN"] * r_headlin,
            "t_step": t_step,                                                     # v3.6
        }
        pos = (T["t_vx"] + T["t_vy"] + T["t_yaw"] + T["t_yawi"] + T["t_yawlin"] + T["t_yawrel"] + T["t_vyrel"] + T["t_head"] + T["t_h"]
               + T["t_lift"] + T["t_stance"] + T["t_headlin"] + T["t_step"])
        neg = (T["t_roll"] + T["t_pitch"] + T["t_rollrate"] + T["t_pitchrate"] + T["t_bias"] + T["t_act"] + T["t_omdot"]
               + T["t_qres"] + T["t_tau"] + T["t_taubar"] + T["t_errbar"] + T["t_kneev"] + T["t_mode"] + T["t_vz"] + T["t_abadbias"] + T["t_drift"])
        reward = pos - neg
        kill = jnp.where(jnp.max(tau_pk12) > TAU_KILL, info["kill"] + 1, 0)
        done = jnp.where((grav[2] > FALL_GRAV_Z) | (data.qpos[2] < MIN_HEIGHT) | (kill >= KILL_STEPS), 1.0, 0.0)

        info.update({"rng": rng, "c": c, "cmd": info["cmd"], "u_mode": u_mode, "head_err": head_err,
                     "ph": ph, "s4": P["A"]["s4"], "s_arc": P["A"]["arc"], "phi_cyc": phi_cyc, "u4": u4, "wz_ema": wz_ema, "vy_ema": vy_ema, "qres": qres,
                     "a_hist": a_hist, "last_a": action, "sway": sway, "qvel_prev": data.qvel[LEG_QVEL_IDX],
                     "om_prev": om, "roll_ema": roll_ema, "sway_ema": sway_ema, "abad_ema": abad_ema, "drift_ema": drift_ema, "apex_run": apex_run, "apex_last": apex_last, "kill": kill, "step": step_i + 1,
                     "wheel_theta": wheel_theta})
        info_obs = dict(info, cmd=cmd)
        obs = self._obs(data, info_obs, action)
        n = jax.random.normal(k_obs, (self.obs_dim,))
        obs = (obs.at[0:3].add(NOISE_GRAV * n[0:3]).at[3:6].add(NOISE_GYRO * n[3:6])
               .at[6:18].add(NOISE_QPOS * n[6:18]).at[18:30].add(NOISE_QVEL * n[18:30])
               .at[30:34].add(NOISE_WHEEL * n[30:34]))
        if w.get("MIRROR_AUG", False):
            obs = jnp.where(info["mir"], mirror_obs(obs), obs)
        metrics = {"height": data.qpos[2], "vx": vb[0], "vy": vb[1], "wz": wz, "reward": reward,
                   "pitch": jnp.abs(grav[0]) * 57.29578, "roll": jnp.abs(grav[1]) * 57.29578, "mode": u_mode,
                   "clr_step": jnp.sum(sw * clr) / jnp.maximum(jnp.sum(sw), 1.0) * 1000.0,
                   "clr_stance": jnp.max(clr * (1 - g_leg)) * 1000.0,
                   "s4": P["A"]["s4"], "s_arc": P["A"]["arc"], "cyc": u4,
                   "yawerr": jnp.abs(wz - cmd[2]), "vxerr": jnp.abs(vb[0] - cmd[0]), "vyerr": jnp.abs(vb[1] - cmd[1]),
                   "tau_pk": jnp.max(tau_pk12), "err_pk": jnp.max(jnp.abs(err12)), "knee_v": knee_v, "omega": om,
                   "sway_y": sway[1] * 1000.0, "roll_bias": roll_ema * 57.29578,
                   "abad_bias": jnp.max(jnp.abs(abad_ema)) * 57.29578, "vx_drift": drift_ema[0], "head_deg": head_err * 57.29578,
                   "head_abs": jnp.abs(head_err) * 57.29578, **T}   # 訓練曲線要看絕對值：head_deg 的正負隨左右指令隨機，平均會互相抵消
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"


# ---------------------------------------------------------------- domain randomization
_M_BASE = mujoco.MjModel.from_xml_path(SCENE_V3)
_BASE_ID = mm._id(_M_BASE, mujoco.mjtObj.mjOBJ_BODY, "base_link")
COM_Y_SCALE = float(_M_BASE.body_subtreemass[_BASE_ID] / _M_BASE.body_mass[_BASE_ID])   # 整機 38.82 / base 17.03 ≈ 2.28：base 質心要移這麼多倍整機才移 1 倍
_LEG_DOF, _WHEEL_DOF = jnp.array(mm.LEG_QVEL_IDX), jnp.array(mm.WHEEL_QVEL_IDX)
_ABAD_DOF = jnp.array(mm.LEG_QVEL_IDX[::3])
ABAD12 = jnp.array([0, 3, 6, 9])


def make_domain_randomize(gains: str = "kp250", com_y_mm: float = 0.0):
    """回傳對應增益組的 domain_randomize。標稱 kp/kd 必須跟著模型走，
    否則 factory 線會被隨機化拉回 kp250 附近（DR 是乘在標稱值上的）。
    `com_y_mm > 0`（v3.6）：base_link 質心 y 每回合抽 U(−1,1)·com_y_mm·COM_Y_SCALE，整機質心 ±com_y_mm、零均值，
    洗掉「模型質心偏左 1.5 mm → policy 學方向專屬補償」（v3.5 spec §9.4）。預設 0 走原路徑。"""
    G = GAIN_SETS[gains]
    _KP_NOM_j = jnp.array(np.tile(np.asarray(G["kp3"]), 4))
    _KD_NOM = float(np.asarray(G["kd3"])[1])
    dy_max = com_y_mm / 1000.0 * COM_Y_SCALE

    def domain_randomize(sys, rng):
        """v2 那組 ＋ 輪 frictionloss 0.10–0.18、damping 0.005–0.03、ABAD kp ×0.7–1.0（v3.3 由 0.4 縮起）。"""
        @jax.vmap
        def per_env(rng):
            k = jax.random.split(rng, 10)            # ⚠️ 個數不能改：改了現有 notebook 的抽樣就變了；新鍵用 fold_in
            gf = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.4, maxval=1.4))
            s_kp = jax.random.uniform(k[1], minval=0.8, maxval=1.2)
            s_ab = jax.random.uniform(k[2], minval=0.7, maxval=1.0)      # v3.3：×0.4 太寬，平移滑步速度變異蓋掉 vy 訊號
            kv = jax.random.uniform(k[3], minval=0.5, maxval=2.0) * _KD_NOM
            kp_leg = _KP_NOM_j * s_kp
            kp_leg = kp_leg.at[ABAD12].set(_KP_NOM_j[ABAD12] * s_ab)
            gain = sys.actuator_gainprm.at[LEG_ACT_IDX, 0].set(kp_leg)
            bias = sys.actuator_biasprm.at[LEG_ACT_IDX, 1].set(-kp_leg).at[LEG_ACT_IDX, 2].set(-kv)
            bm = sys.body_mass * jax.random.uniform(k[4], (sys.nbody,), minval=0.9, maxval=1.1)
            bm = bm.at[_BASE_ID].add(jax.random.uniform(k[5], minval=0.0, maxval=5.0))
            fl = sys.dof_frictionloss
            fl = fl.at[_LEG_DOF].multiply(jax.random.uniform(k[6], minval=0.5, maxval=1.5))
            fl = fl.at[_ABAD_DOF].set(sys.dof_frictionloss[_ABAD_DOF] * jax.random.uniform(k[7], minval=0.6, maxval=1.4))
            fl = fl.at[_WHEEL_DOF].set(jax.random.uniform(k[8], minval=0.10, maxval=0.18))
            dmp = sys.dof_damping.at[_WHEEL_DOF].set(jax.random.uniform(k[9], minval=0.005, maxval=0.03))
            if dy_max > 0:
                dy = jax.random.uniform(jax.random.fold_in(rng, 99), minval=-1.0, maxval=1.0) * dy_max
                return gf, gain, bias, bm, fl, dmp, sys.body_ipos.at[_BASE_ID, 1].add(dy)
            return gf, gain, bias, bm, fl, dmp

        out = per_env(rng)
        gf, gain, bias, bm, fl, dmp = out[:6]
        in_axes = jax.tree_util.tree_map(lambda x: None, sys)
        in_axes = in_axes.replace(geom_friction=0, actuator_gainprm=0, actuator_biasprm=0, body_mass=0,
                                  dof_frictionloss=0, dof_damping=0)
        sys = sys.replace(geom_friction=gf, actuator_gainprm=gain, actuator_biasprm=bias, body_mass=bm,
                          dof_frictionloss=fl, dof_damping=dmp)
        if dy_max > 0:
            in_axes = in_axes.replace(body_ipos=0)
            sys = sys.replace(body_ipos=out[6])
        return sys, in_axes

    return domain_randomize


domain_randomize = make_domain_randomize("kp250")     # 舊名保留：v3.3 的 notebook 不用改
