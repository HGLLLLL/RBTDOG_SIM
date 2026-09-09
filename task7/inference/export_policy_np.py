"""brax PPO 權重（.pkl）→ 狗上 numpy 推論用的 .npz（`realbot/policy_np.py` 讀）。

用法：
    conda run -n rbtdog python task7/inference/export_policy_np.py \
        --params task7/weights/cpg_rl_max_v2_3_params.pkl --preset v2.3
    → task7/weights/cpg_rl_max_v2_3_np.npz

npz 內容：mean/std（obs 正規化）、W0..W3/b0..b3（policy MLP，含輸出層 2·act）、
obs_dim/act_dim/preset/layout、baseline_json（BASELINE_A，M9 說兩次用）、src_sha256（pkl 指紋）。
正確性由 tests/test_policy_np.py 對 brax 前向逐點比對。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gait_baseline as gb  # noqa: E402
import local_infer_max as li  # noqa: E402
import obs_max  # noqa: E402

HIDDEN = ("hidden_0", "hidden_1", "hidden_2", "hidden_3")


def export(pkl: str, out: str, preset: str) -> dict:
    from brax.io import model

    layout = li.PRESET_LAYOUT[preset]
    act_dim = li.LAYOUT_DIMS[layout]
    obs_dim = obs_max.obs_dim(act_dim, li.PRESET_HEAD[preset])
    params = model.load_params(pkl)
    norm, pol = params[0], params[1]["params"]
    mean, std = np.asarray(norm.mean, dtype=np.float32), np.asarray(norm.std, dtype=np.float32)
    assert mean.shape == (obs_dim,), f"pkl 的 obs 維度 {mean.shape} ≠ preset {preset} 的 {obs_dim}"
    assert tuple(pol.keys()) == HIDDEN, tuple(pol.keys())
    Ws = [np.asarray(pol[h]["kernel"], dtype=np.float32) for h in HIDDEN]
    bs = [np.asarray(pol[h]["bias"], dtype=np.float32) for h in HIDDEN]
    assert Ws[0].shape == (obs_dim, li.POLICY_HIDDEN[0])
    assert Ws[3].shape == (li.POLICY_HIDDEN[2], 2 * act_dim), Ws[3].shape
    base = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in gb.BASELINE_A.items()}
    z = dict(mean=mean, std=std, obs_dim=np.int64(obs_dim), act_dim=np.int64(act_dim),
             preset=np.str_(preset), layout=np.str_(layout),
             baseline_json=np.str_(json.dumps(base, sort_keys=True)),
             src_sha256=np.str_(hashlib.sha256(Path(pkl).read_bytes()).hexdigest()))
    for i in range(4):
        z[f"W{i}"], z[f"b{i}"] = Ws[i], bs[i]
    np.savez(out, **z)
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--preset", default="v2.3", choices=sorted(li.PRESET_LAYOUT))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or str(Path(a.params).with_name(Path(a.params).stem.replace("_params", "_np") + ".npz"))
    z = export(a.params, out, a.preset)
    print(f"✅ {out}  obs {int(z['obs_dim'])} act {int(z['act_dim'])} preset {z['preset']} "
          f"layout {z['layout']}  sha256 {str(z['src_sha256'])[:12]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
