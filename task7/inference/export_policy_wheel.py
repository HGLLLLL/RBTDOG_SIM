#!/usr/bin/env python3
"""我們自訓的純 RL（Brax PPO，rl_env_wheel）→ 狗上 numpy 推論用 .npz（M9 --v3 讀，layout `rl16`，gen_direct 產生器）。

用法：conda run -n rbtdog --no-capture-output python task7/inference/export_policy_wheel.py --params task7/weights/rl_wheel_v1_params.pkl
  → task7/weights/rl_wheel_v1_np.npz
npz：brax 版面（mean/std、W0..3/b0..3、swish、tanh_loc）＋ **act_gain=4.0**（狗端 policy_np 乘上去＝他的尺度，gen_direct 再 0.25·a／5·a）
＋ gen_json／ref_json／LO12/HI12/STANCE_Q12（與 export_policy_rl 同欄位，gen_tables 共用）。
區分記號：npz 的 preset="rl_wheel"、gen_json.source.family="rl_wheel"。ref_json.family **維持 "rl_only"**——
它是 rl16 介面族名，狗端 gen_direct.Params 只認 "rl_only"（狗端堆疊不改）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "realbot"))
import export_policy_rl as er  # noqa: E402

HIDDEN = ("hidden_0", "hidden_1", "hidden_2", "hidden_3")
POLICY_LAYERS, VALUE_LAYERS = (512, 256, 128), (512, 256, 128)
OBS_DIM, ACT_DIM, ACT_GAIN = 53, 16, 4.0
IO = json.loads((HERE / "rl_wheel_io.json").read_text(encoding="utf-8"))


def _networks():
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    return ppo_networks.make_ppo_networks(OBS_DIM, ACT_DIM, preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=POLICY_LAYERS, value_hidden_layer_sizes=VALUE_LAYERS)


def make_params_for_test(key):
    """隨機初始化的 (normalizer, policy, value)，normalizer 帶非平凡 mean/std（測試用）。"""
    import jax
    from brax.training.acme import running_statistics, specs
    nets = _networks()
    k1, k2, k3, k4 = jax.random.split(key, 4)
    norm = running_statistics.init_state(specs.Array((OBS_DIM,), np.float32))
    norm = norm.replace(mean=jax.random.normal(k3, (OBS_DIM,)), std=jax.random.uniform(k4, (OBS_DIM,), minval=0.5, maxval=2.0))
    return (norm, nets.policy_network.init(k1), nets.value_network.init(k2))


def brax_inference_fn(pkl: str):
    """deterministic policy：obs(batch,53) → tanh(loc)(batch,16)。"""
    from brax.io import model
    from brax.training.agents.ppo import networks as ppo_networks
    import jax
    params = model.load_params(pkl)
    nets = _networks()
    make_policy = ppo_networks.make_inference_fn(nets)
    pol = make_policy((params[0], params[1]), deterministic=True)
    return lambda obs: pol(obs, jax.random.PRNGKey(0))[0]


def export(pkl: str, out: str) -> dict:
    from brax.io import model
    params = model.load_params(pkl)
    norm, pol = params[0], params[1]["params"]
    mean, std = np.asarray(norm.mean, dtype=np.float32), np.asarray(norm.std, dtype=np.float32)
    assert mean.shape == (OBS_DIM,) and tuple(pol.keys()) == HIDDEN, (mean.shape, tuple(pol.keys()))
    Ws = [np.asarray(pol[h]["kernel"], dtype=np.float32) for h in HIDDEN]
    bs = [np.asarray(pol[h]["bias"], dtype=np.float32) for h in HIDDEN]
    assert Ws[0].shape == (OBS_DIM, POLICY_LAYERS[0]) and Ws[3].shape == (POLICY_LAYERS[2], 2 * ACT_DIM), (Ws[0].shape, Ws[3].shape)
    T = er.gen_tables(IO, source=dict(task="rl_wheel_v1", env_cfg="rl_env_wheel.WheelEnv", params=Path(pkl).name, family="rl_wheel"))
    z = dict(mean=mean, std=std, obs_dim=np.int64(OBS_DIM), act_dim=np.int64(ACT_DIM),
             preset=np.str_("rl_wheel"), layout=np.str_("rl16"), activation=np.str_("swish"), output=np.str_("tanh_loc"),
             act_gain=np.float64(ACT_GAIN),
             ref_json=np.str_(json.dumps(T["ref"], sort_keys=True)), gen_json=np.str_(json.dumps(T["K"], sort_keys=True)),
             src_sha256=np.str_(hashlib.sha256(Path(pkl).read_bytes()).hexdigest()),
             LO12=T["lo12"].astype(np.float64), HI12=T["hi12"].astype(np.float64), WHEEL_TAU_MAX=np.float64(T["tmax"]),
             STANCE_Q12=np.array([T["default"][T["physx_to_mjcf"][k]] for k in range(12)], np.float64))
    for i in range(4):
        z[f"W{i}"], z[f"b{i}"] = Ws[i], bs[i]
    np.savez(out, **z)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(HERE.parent / "weights" / "rl_wheel_v1_params.pkl"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or str(Path(a.params).with_name(Path(a.params).stem.replace("_params", "_np") + ".npz"))
    z = export(a.params, out)
    print(f"→ {out}  obs {int(z['obs_dim'])} act {int(z['act_dim'])} layout {z['layout']} act_gain {float(z['act_gain'])} sha {str(z['src_sha256'])[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
