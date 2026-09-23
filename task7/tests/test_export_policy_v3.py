"""export_policy_v3：npz 快照必須等於 rl_env_v3 當下的值（改了 env 忘了重匯出就紅）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
jax = pytest.importorskip("jax")
import rl_env_v3 as v3  # noqa: E402
import export_policy_v3 as ex  # noqa: E402

PKL = ROOT / "weights" / "cpg_rl_v3_6f_params.pkl"
NPZ = ROOT / "weights" / "cpg_rl_v3_6f_np.npz"


@pytest.fixture(scope="module")
def env():
    return ex.make_env()


@pytest.fixture(scope="module", params=["fresh", "committed"])
def z(request, tmp_path_factory):
    """兩個來源都要過：fresh＝當下程式碼重匯出的；committed＝repo 裡那份。

    只測 fresh 的話「改了 env 忘了重匯出」永遠不會紅（重匯出的當然一致），所以 committed 也逐鍵比對。
    """
    if request.param == "fresh":
        p = tmp_path_factory.mktemp("w") / "p.npz"
        ex.export(str(PKL), str(p))
    else:
        p = NPZ
        assert p.exists(), "先跑 export_policy_v3.py"
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def test_mlp_shapes_and_meta(z):
    assert int(z["obs_dim"]) == 88 and int(z["act_dim"]) == 24
    assert str(z["preset"]) == "v36" and str(z["layout"]) == "v3"
    assert z["W0"].shape == (88, 256) and z["W3"].shape == (128, 48) and z["mean"].shape == (88,)


def test_ref_snapshot_equals_env(z, env):
    ref = json.loads(str(z["ref_json"]))
    assert ref["cyc_turn_sym"] is True and ref["cyc_amp_rand"] is False and ref["wheel_space"] == "tau"
    assert ref["z_sag"] == env.ref["z_sag"] == 0.072 and ref["lat_omega"] == 2.2 and ref["cyc_amp_lat"] == 1.6
    assert ref["cyc_lat_decouple"] is False
    np.testing.assert_allclose(ref["stance_q12"], np.asarray(env.ref["stance_q12"]))
    for k, v in env.ref.items():
        if isinstance(v, (int, float, str, bool)):
            assert ref[k] == v, k


def test_constants_and_tables_equal_module(z, env):
    K = json.loads(str(z["gen_json"]))
    assert (K["ACT_DIM"], K["QRES_MAX"], K["QRES_SLEW"]) == (v3.ACT_DIM, v3.QRES_MAX, v3.QRES_SLEW)
    assert (K["OMEGA_NOM_SCALE"], K["AMP_SCALE"], K["LIFT_SCALE"]) == (v3.OMEGA_NOM_SCALE, v3.AMP_SCALE, v3.LIFT_SCALE)
    assert (K["SWAY_MAX"], K["SWAY_SLEW"], K["WHEEL_RES"]) == (v3.SWAY_MAX, v3.SWAY_SLEW, v3.WHEEL_RES)
    assert (K["STEP_GAIN_S"], K["PH_SLEW"], K["FOOT_OFF_MAX"], K["TAU_FF"]) == (v3.STEP_GAIN_S, v3.PH_SLEW, v3.FOOT_OFF_MAX, v3.TAU_FF)
    assert K["RAMP_STEPS"] == v3.W36["RAMP_STEPS"] == 50 and K["CTRL_DT"] == v3.CTRL_DT and K["CYC_N"] == v3.CYC_N
    # gains 跟著 env 當下值走（GAIN_SETS["factory"] ＋ 場景讀到的 kv_wheel），不是抄一份字面值
    G = v3.GAIN_SETS["factory"]
    assert K["gains"]["kp3"] == [float(x) for x in np.asarray(G["kp3"]).ravel()]
    assert K["gains"]["kd3"] == [float(x) for x in np.asarray(G["kd3"]).ravel()]
    assert K["gains"]["wheel_kd"] == float(env.kv_wheel)
    assert K["gains"]["wheel_kd_safe"] == float(ex.WHEEL_KD_SAFE)
    # 同時釘住上機合約：操作卡寫死 --kp 120 --kp-abad 60 --kd 1.0 --wheel-kd 0.1，改了要先改這裡
    assert K["gains"] == {"kp3": [60.0, 120.0, 120.0], "kd3": [1.0, 1.0, 1.0], "wheel_kd": 0.1, "wheel_kd_safe": 0.5}
    for name in ("PH_TURN_L", "PH_LAT_L", "PH_TROT", "W_LAT_L", "W_TURN_L", "W_ARC_L", "DOM_TURN_L", "LAT_WHEEL_L", "MIRROR_LR",
                 "SIDE_Y_j", "FRONT_j", "FOOT_XY_BODY", "STANCE_Q12_j", "STANCE_FEET_j", "KNEE_SIGN_j",
                 "CYC_OFF_SYM", "CYC_TAU_SYM", "CYC_HZ_SYM", "CYC_QMEAN",
                 "OBS_MIRROR_IDX", "OBS_MIRROR_SGN", "ACT_MIRROR_IDX", "ACT_MIRROR_SGN"):
        key = ex.TABLE_KEYS[name]
        np.testing.assert_array_equal(z[key], np.asarray(getattr(v3, name)), err_msg=name)
    np.testing.assert_array_equal(z["LO12"], np.asarray(env._lo))
    np.testing.assert_array_equal(z["HI12"], np.asarray(env._hi))


def test_wheel_tau_max_pinned_to_model(z, env):
    """★ 狗端 `v3_wheel_rows` 夾的那條線 = **模擬裡 MuJoCo 真的在夾的那條**（輪致動器 forcerange）。

    policy 是在「τ 被夾在 ±33」的世界學的；抄一份字面值的話，場景改了不會有任何地方紅，
    而上機會送出模擬沒送過的力矩。npz 標量與 gen_json 兩邊都釘（狗端讀的是 gen_json 那份）。
    """
    import max_model as mm
    i = int(mm.WHEEL_ACT_IDX[0])
    assert bool(env._mj.actuator_forcelimited[i]), "場景沒限輪力矩 → _wheel_tau_max 會退回規格書值，這個測試要改"
    want = float(abs(env._mj.actuator_forcerange[i][1]))
    assert want == 33.0 == float(mm.TAU_MAX_WHEEL)      # 上機合約：操作卡與護欄都照 33 寫
    assert z["WHEEL_TAU_MAX"].dtype == np.float64 and float(z["WHEEL_TAU_MAX"]) == want
    assert json.loads(str(z["gen_json"]))["WHEEL_TAU_MAX"] == want
    assert ex._wheel_tau_max(env) == want


def test_committed_npz_is_current():
    """weights/cpg_rl_v3_6f_np.npz 必須是當下的 pkl 匯出的（env 當下值由上面的 committed 參數化把關）。"""
    import hashlib
    assert NPZ.exists(), "先跑 export_policy_v3.py"
    with np.load(NPZ, allow_pickle=False) as f:
        assert str(f["src_sha256"]) == hashlib.sha256(PKL.read_bytes()).hexdigest()
