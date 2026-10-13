"""rl_wheel_v1 notebook：版本釘死、throughput 格、wrap_env_fn、DR、Drive 存檔、RESUME、停損說明。"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "rl_wheel_v1_colab.ipynb"


def _src(kind):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind)


def test_notebook_cells():
    assert NB.exists(), "先跑 python3 task7/notebooks/build_nb_wheel.py"
    code = _src("code"); md = _src("markdown")
    assert 'brax==0.14.2' in code and 'assert brax.__version__ == "0.14.2"' in code and 'assert mujoco.__version__ == "3.10.0"' in code
    assert "import rl_env_wheel as rw" in code and "rw.WheelEnv()" in code and "(53, 16)" in code
    assert "for NUM_ENVS in (1024, 2048, 4096):" in code and "steps/s" in code
    assert "wrap_env_fn=rw.wrap_env" in code and "randomization_fn=rw.make_domain_randomize(env)" in code
    assert "policy_hidden_layer_sizes=(512, 256, 128)" in code and "value_hidden_layer_sizes=(512, 256, 128)" in code
    assert "num_envs=NUM_ENVS" in code and "batch_size=256 * NUM_ENVS // 2048" in code and "episode_length=rw.EPISODE_LEN" in code
    assert "drive.mount" in code and "policy_params_fn=_save_ckpt" in code and "RESUME = None" in code and "restore_params=_restore" in code
    assert 'model.save_params("rl_wheel_v1_params.pkl", params)' in code
    for k in ("ps('lin')", "ps('ang')", "ps('ww')", "ps('slip')", "ps('level')", "ps('lin_small')", "ps('fell')"):
        assert k in code, k
    assert "停損" in md and "len < 600" in md and "lin < 0.5" in md and "2 億步" in md and "2026-10-13" in md
