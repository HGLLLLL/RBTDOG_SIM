#!/usr/bin/env python3
"""同事的純 RL policy（Isaac Lab／rsl_rl 匯出的 ONNX）→ 狗上 numpy 推論用的 .npz（M9 --v3 讀，layout `rl16`）。

用法：
    conda run -n rbtdog --no-capture-output python task7/inference/export_policy_rl.py \
        --onnx RL_only/2026-09-24_13-23-11.onnx --io RL_only/policy_io.json
    → task7/weights/rl_only_2026-09-24_np.npz

npz 內容：MLP（W0..3/b0..3，已轉置成 x @ W；mean=0、std=1）＋ `gen_json`（default pose、動作縮放、限速、增益、輪力矩上限、
CTRL_DT…）＋ `ref_json`（M9 會讀的幾個 ref 鍵）＋ LO12/HI12（MJCF 序，讀 MJCF 機構限位）。
**狗端不自己算任何常數**，全部從這裡讀；`tests/test_policy_rl_np.py` 釘住 npz 推論 == onnxruntime。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realbot"))
import max_model as mm  # noqa: E402

ROOT = HERE.parent.parent
Q_SLEW = 0.5        # 每步 q_des 最大變化量 rad（模擬 vy 1.0 命令速峰 20 rad/s × 0.02 = 0.4；這是異常擋不是正常路徑）
A_CLIP = 4.0        # policy 輸出無界；|a| 夾在 ±4（0.25×4 = 1 rad 偏離；模擬擾動下最大 3.5）
RAMP_STEPS = 50     # 淡入 1 s（與 v3 相同）
WHEEL_OUTER_GAIN = 1.0   # 輪速環 kd（他的 IdealPD damping 1.0；實機 kd1 模式交驅動器做）
TAU_FF = 0.13       # gen_v3.wheel_dead 用的死區；gen_direct 的前饋 ff 恆 0 → 死區後 effort 恆 0（訓練沒有摩擦補償）


def _mj_limits(names_mjcf_order):
    """MJCF 機構限位（controller 座標 = MJCF），按 LEG_NAMES（MJCF 序）。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(mm.SCENE_MJX_V3F)
    lo, hi, tmax = [], [], None
    for n in names_mjcf_order:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert j >= 0, n
        lo.append(float(m.jnt_range[j, 0])); hi.append(float(m.jnt_range[j, 1]))
    i = int(mm.WHEEL_ACT_IDX[0])
    tmax = float(abs(m.actuator_forcerange[i][1])) if bool(m.actuator_forcelimited[i]) else float(mm.TAU_MAX_WHEEL)
    return np.array(lo), np.array(hi), tmax


def export(onnx_path: str, io_path: str, out: str) -> dict:
    import onnx
    from onnx import numpy_helper
    io = json.load(open(io_path))
    m = onnx.load(onnx_path)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    ops = [n.op_type for n in m.graph.node]
    assert ops == ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"], ops
    for n in m.graph.node:
        if n.op_type == "Elu":
            alpha = [a.f for a in n.attribute if a.name == "alpha"]
            assert not alpha or abs(alpha[0] - 1.0) < 1e-9, alpha
    gemms = [n for n in m.graph.node if n.op_type == "Gemm"]
    Ws, bs = [], []
    for n in gemms:
        transB = {a.name: a.i for a in n.attribute}.get("transB", 0)
        W = init[n.input[1]].astype(np.float32); b = init[n.input[2]].astype(np.float32)
        Ws.append(W.T.copy() if transB else W.copy()); bs.append(b)   # torch Linear：y = x Wᵀ + b（transB=1）→ 存成 x @ W
    obs_dim, act_dim = int(io["observation"]["policy"]["dim"]), int(io["action"]["dim"])
    assert Ws[0].shape == (obs_dim, 512) and Ws[3].shape == (128, act_dim), (Ws[0].shape, Ws[3].shape)
    assert io["joints"]["names"][:12] == io["action"]["labels"][:12]
    # PhysX 序（按種類分組、腿序 FAR FBL RAR RBL）→ MJCF 序（每腿 abad hip knee、腿序 FR FL RR RL）
    physx = io["joints"]["names"]
    mjcf12 = [f"{mm.PREFIX[l]}_{k}_JOINT" for l in mm.LEGS for k in ("ABAD", "HIP", "KNEE")]
    physx_to_mjcf = [physx.index(n) for n in mjcf12]          # mjcf[k] = physx[physx_to_mjcf[k]]
    wheels_physx = physx[12:16]; wheels_mjcf = [f"{mm.PREFIX[l]}_FOOT_JOINT" for l in mm.LEGS]
    assert wheels_physx == wheels_mjcf, (wheels_physx, wheels_mjcf)
    default = np.array(io["joints"]["default_pos"], float)
    scale = np.array(io["action"]["scale"], float); offset = np.array(io["action"]["offset"], float)
    assert np.allclose(offset[:12], default[:12]) and np.allclose(offset[12:], 0.0)
    import rl_obs
    lo12, hi12, tmax = _mj_limits([f"{mm.PREFIX[l]}_{k}_JOINT" for l in mm.LEGS for k in ("ABAD", "HIP", "KNEE")])
    assert len(rl_obs.LEG_NAMES) == 12
    K = dict(obs_dim=obs_dim, act_dim=act_dim, CTRL_DT=float(io["control"]["policy_dt"]),
             DEFAULT_POS_PHYSX=default.tolist(), ACT_SCALE_PHYSX=scale.tolist(), PHYSX_TO_MJCF=physx_to_mjcf,
             JOINT_NAMES_PHYSX=physx, Q_SLEW=Q_SLEW, A_CLIP=A_CLIP, RAMP_STEPS=RAMP_STEPS,
             TAU_FF=TAU_FF, WHEEL_TAU_MAX=tmax,
             gains=dict(kp3=[60.0, 120.0, 120.0], kd3=[1.0, 1.0, 1.0], wheel_kd=WHEEL_OUTER_GAIN, wheel_kd_safe=0.5),
             obs_slices=io["observation"]["policy"]["slices"], noise=io["observation"]["policy"]["noise"],
             source=dict(task=io.get("task"), env_cfg=io.get("env_cfg"), onnx=Path(onnx_path).name))
    # 他的致動器設定（env.yaml）與 K.gains 必須一致；這裡只記錄，M9 拿 gains 與命令列「說兩次」。
    ref = dict(wheel_outer_gain=WHEEL_OUTER_GAIN, wheel_space="tau", nominal_height=0.485,
               cmd_range=dict(vx=[-1.0, 1.0], vy=[-1.0, 1.0], wz=[-1.0, 1.0]), family="rl_only")
    z = dict(mean=np.zeros(obs_dim, np.float32), std=np.ones(obs_dim, np.float32),
             obs_dim=np.int64(obs_dim), act_dim=np.int64(act_dim),
             preset=np.str_("rl_only"), layout=np.str_("rl16"), activation=np.str_("elu"), output=np.str_("linear"),
             ref_json=np.str_(json.dumps(ref, sort_keys=True)), gen_json=np.str_(json.dumps(K, sort_keys=True)),
             src_sha256=np.str_(hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()),
             LO12=lo12.astype(np.float64), HI12=hi12.astype(np.float64), WHEEL_TAU_MAX=np.float64(tmax),
             STANCE_Q12=np.array([default[physx_to_mjcf[k]] for k in range(12)], np.float64))
    for i in range(4):
        z[f"W{i}"], z[f"b{i}"] = Ws[i], bs[i]
    np.savez(out, **z)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "RL_only" / "2026-09-24_13-23-11.onnx"))
    ap.add_argument("--io", default=str(ROOT / "RL_only" / "policy_io.json"))
    ap.add_argument("--out", default=str(HERE.parent / "weights" / "rl_only_2026-09-24_np.npz"))
    a = ap.parse_args()
    z = export(a.onnx, a.io, a.out)
    print(f"→ {a.out}  obs {int(z['obs_dim'])} act {int(z['act_dim'])} layout {z['layout']} sha {str(z['src_sha256'])[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
