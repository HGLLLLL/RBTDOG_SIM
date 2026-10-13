"""rl_wheel_v1 notebook：版本釘死、throughput 格、wrap_env_fn、DR、本機 /content 存檔（不用雲端硬碟）、RESUME、停損說明。"""
import ast
import importlib.util
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "rl_wheel_v1_colab.ipynb"
GEN = NB.parent / "build_nb_wheel.py"


def _src(kind):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind)


def test_notebook_cells():
    assert NB.exists(), "先跑 python3 task7/notebooks/build_nb_wheel.py"
    code = _src("code"); md = _src("markdown")
    assert 'brax==0.14.2' in code and 'assert brax.__version__ == "0.14.2"' in code and 'assert mujoco.__version__ == "3.10.0"' in code
    assert "import rl_env_wheel as rw" in code and "rw.WheelEnv()" in code
    assert 'env.observation_size == {"state": (53,), "privileged_state": (56,)}' in code and 's.obs["state"]' in code
    assert 'policy_obs_key="state", value_obs_key="privileged_state"' in code and "discounting=0.99" in code   # 特權 critic、他的 discount
    assert "for NUM_ENVS in (1024, 2048, 4096):" in code and "steps/s" in code
    assert "wrap_env_fn=rw.wrap_env" in code and "randomization_fn=rw.make_domain_randomize(env)" in code
    assert "policy_hidden_layer_sizes=(512, 256, 128)" in code and "value_hidden_layer_sizes=(512, 256, 128)" in code
    assert "num_envs=NUM_ENVS" in code and "batch_size=256 * NUM_ENVS // 2048" in code and "episode_length=rw.EPISODE_LEN" in code
    assert "drive.mount" not in code and 'DRIVE = "/content/rl_wheel_v1"' in code and "policy_params_fn=_save_ckpt" in code and 'RESUME = "auto"' in code and "glob.glob" in code and "TIMESTEPS = 500_000_000" in code and "1.3 * bench" in code and "restore_params=_restore" in code
    assert 'model.save_params("rl_wheel_v1_params.pkl", params)' in code
    # 續訓累計步數：brax restore_params 不還原步數，存檔名要加上從 RESUME 檔名解析的 offset
    assert "STEP_OFFSET = 0 if RESUME is None else _step_offset(RESUME)" in code and 're.search(r"step(\\d+)", path)' in code
    assert 'raise ValueError("RESUME 要指到 rl_wheel_v1_step<N>.pkl，不是 rl_wheel_v1_params.pkl")' in code
    assert "rl_wheel_v1_step{STEP_OFFSET + step:012d}.pkl" in code and "total = STEP_OFFSET + step" in code
    assert "else:\n        break" in code                                     # throughput 遇第一檔不合格就停
    for k in ("ps('lin')", "ps('ang')", "ps('ww')", "ps('wgap')", "ps('lim')", "ps('sat_n')", "ps('slip')", "ps('level')", "ps('lin_small')", "ps('fell')"):
        assert k in code, k
    assert "停損" in md and "len < 600" in md and "lin < 0.5" in md and "2 億步" in md and "2026-10-13" in md and "特權" in md
    assert "`wgap` 沒比 step 0 好" in md and "`gap` 中位 < 40 mm" in md and "`ww` 沒比 step 0 低" not in md   # ww 平地恆 0，停損看 wgap


def test_code_cells_parse():
    """clone／import／bench／train／plot／save 都要是合法 Python（安裝格有 !pip，跳過）。"""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 8
    for i, c in enumerate(code_cells[2:], start=2):
        ast.parse("".join(c["source"]), filename=f"code_cell_{i}")


def test_notebook_matches_generator():
    """已 commit 的 ipynb 必須等於 build_nb_wheel.build() 的產物（改產生器忘了重產會被抓到）。"""
    spec = importlib.util.spec_from_file_location("build_nb_wheel", GEN)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    norm = lambda cells: [(c["cell_type"], "".join(c["source"])) for c in cells]
    committed = json.loads(NB.read_text(encoding="utf-8"))
    assert norm(mod.build()["cells"]) == norm(committed["cells"]), "先跑 python3 task7/notebooks/build_nb_wheel.py"
