"""brax PPO v3 權重（.pkl）→ 狗上 numpy 推論用的 .npz（M9 --v3 讀）。

用法：
    conda run -n rbtdog python task7/inference/export_policy_v3.py \
        --params task7/weights/cpg_rl_v3_6f_params.pkl
    → task7/weights/cpg_rl_v3_6f_np.npz

npz 內容：MLP（mean/std/W0..3/b0..3）＋ 產生器參數快照（ref_json／gen_json）＋ 相位表／幾何／限位／
原廠週期表（CYC_TURN_SYM 版）＋ 鏡像索引表。**狗端不自己算任何常數**，全部從這裡讀；
tests/test_export_policy_v3.py 釘住 npz == rl_env_v3 當下值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_env_v3 as v3  # noqa: E402

HIDDEN = ("hidden_0", "hidden_1", "hidden_2", "hidden_3")
# 模組名 → npz 鍵
TABLE_KEYS = {
    "PH_TURN_L": "PH_TURN_L", "PH_LAT_L": "PH_LAT_L", "PH_TROT": "PH_TROT",
    "W_LAT_L": "W_LAT_L", "W_TURN_L": "W_TURN_L", "W_ARC_L": "W_ARC_L", "DOM_TURN_L": "DOM_TURN_L", "LAT_WHEEL_L": "LAT_WHEEL_L",
    "MIRROR_LR": "MIRROR_LR", "SIDE_Y_j": "SIDE_Y", "FRONT_j": "FRONT", "FOOT_XY_BODY": "FOOT_XY_BODY",
    "STANCE_Q12_j": "STANCE_Q12", "STANCE_FEET_j": "STANCE_FEET", "KNEE_SIGN_j": "KNEE_SIGN4",
    "CYC_OFF_SYM": "CYC_OFF", "CYC_TAU_SYM": "CYC_TAU", "CYC_HZ_SYM": "CYC_HZ", "CYC_QMEAN": "CYC_QMEAN",
    "OBS_MIRROR_IDX": "OBS_MIRROR_IDX", "OBS_MIRROR_SGN": "OBS_MIRROR_SGN",
    "ACT_MIRROR_IDX": "ACT_MIRROR_IDX", "ACT_MIRROR_SGN": "ACT_MIRROR_SGN",
}
# 起身／坐下時夾住輪 kd 的上機安全值（＝ M9_gait.WHEEL_KD_SAFE，trip16 實機驗證）。
# 這是「狗上的選擇」不是 env 的值，所以只有這一項是字面值；其餘增益一律從 rl_env_v3 讀。
WHEEL_KD_SAFE = 0.5


def _gains(env) -> dict:
    """增益快照：kp3／kd3 取 GAIN_SETS["factory"]，wheel_kd 取 env 從場景 actuator_biasprm 讀到的值。"""
    G = v3.GAIN_SETS["factory"]
    return {"kp3": [float(x) for x in np.asarray(G["kp3"]).ravel()],
            "kd3": [float(x) for x in np.asarray(G["kd3"]).ravel()],
            "wheel_kd": float(env.kv_wheel),
            "wheel_kd_safe": float(WHEEL_KD_SAFE)}


def _jsonable(v):
    if isinstance(v, (np.ndarray,)) or hasattr(v, "tolist"):
        return np.asarray(v).tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def make_env():
    return v3.DualModeEnv(gains="factory", weights=v3.W36, ref=dict(cyc_amp_rand=False), push=False)


def export(pkl: str, out: str) -> dict:
    from brax.io import model
    params = model.load_params(pkl)
    norm, pol = params[0], params[1]["params"]
    mean, std = np.asarray(norm.mean, dtype=np.float32), np.asarray(norm.std, dtype=np.float32)
    assert mean.shape == (88,), mean.shape
    assert tuple(pol.keys()) == HIDDEN, tuple(pol.keys())
    Ws = [np.asarray(pol[h]["kernel"], dtype=np.float32) for h in HIDDEN]
    bs = [np.asarray(pol[h]["bias"], dtype=np.float32) for h in HIDDEN]
    assert Ws[0].shape == (88, 256) and Ws[3].shape == (128, 2 * v3.ACT_DIM), (Ws[0].shape, Ws[3].shape)
    env = make_env()
    K = dict(ACT_DIM=v3.ACT_DIM, QRES_MAX=v3.QRES_MAX, QRES_SLEW=v3.QRES_SLEW, OMEGA_NOM_SCALE=v3.OMEGA_NOM_SCALE,
             AMP_SCALE=v3.AMP_SCALE, LIFT_SCALE=v3.LIFT_SCALE, SWAY_MAX=v3.SWAY_MAX, SWAY_SLEW=v3.SWAY_SLEW,
             WHEEL_RES=v3.WHEEL_RES, STEP_GAIN_S=v3.STEP_GAIN_S, PH_SLEW=v3.PH_SLEW, FOOT_OFF_MAX=v3.FOOT_OFF_MAX,
             TAU_FF=v3.TAU_FF, RAMP_STEPS=v3.W36["RAMP_STEPS"], CTRL_DT=float(v3.CTRL_DT), CYC_N=int(v3.CYC_N),
             gains=_gains(env), weights_preset="v36", cyc_turn_sym=True, obs_dim=88)
    z = dict(mean=mean, std=std, obs_dim=np.int64(88), act_dim=np.int64(v3.ACT_DIM),
             preset=np.str_("v36"), layout=np.str_("v3"),
             ref_json=np.str_(json.dumps(_jsonable(env.ref), sort_keys=True)),
             gen_json=np.str_(json.dumps(K, sort_keys=True)),
             src_sha256=np.str_(hashlib.sha256(Path(pkl).read_bytes()).hexdigest()),
             LO12=np.asarray(env._lo, dtype=np.float64), HI12=np.asarray(env._hi, dtype=np.float64))
    for i in range(4):
        z[f"W{i}"], z[f"b{i}"] = Ws[i], bs[i]
    for name, key in TABLE_KEYS.items():
        arr = np.asarray(getattr(v3, name))
        z[key] = arr.astype(np.int64) if arr.dtype.kind in "iu" else arr.astype(np.float64)
    np.savez(out, **z)
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "weights" / "cpg_rl_v3_6f_params.pkl"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or str(Path(a.params).with_name(Path(a.params).stem.replace("_params", "_np") + ".npz"))
    z = export(a.params, out)
    print(f"✅ {out}  obs {int(z['obs_dim'])} act {int(z['act_dim'])} sha256 {str(z['src_sha256'])[:12]}…  鍵 {len(z)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
