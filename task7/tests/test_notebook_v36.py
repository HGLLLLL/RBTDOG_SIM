"""v3.6f notebook：build_nb_v3.py --gains factory --weights v36；env 帶 W36、DR 帶 com_y_mm=5.0、progress 印 step。"""
import json
import subprocess
from pathlib import Path

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"
NB = NBDIR / "cpg_rl_v3_6f_colab.ipynb"


def _code_src(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _md_src(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_v36f_notebook_env_dr_progress_save():
    assert NB.exists(), "先跑 python3 task7/notebooks/build_nb_v3.py --gains factory --weights v36"
    src = _code_src(NB)
    assert src.count('v3.DualModeEnv(gains="factory", weights=v3.W36)') == 2
    assert 'v3.DualModeEnv(gains="factory", weights=v3.W36, ref=dict(cyc_amp_rand=False))' in src
    assert 'randomization_fn=v3.make_domain_randomize("factory", com_y_mm=5.0)' in src
    assert "step {ps('t_step'):.2f}" in src and "abad {ps('abad_bias'):.1f}" in src and "head {ps('head_abs'):.1f}" in src
    assert "vy {ps('t_vyrel'):.2f}/6" in src
    assert 'model.save_params("cpg_rl_v3_6f_params.pkl", params)' in src
    assert "RESUME = None" in src
    md = _md_src(NB)
    assert "v3.6f" in md and "t_step" in md and "2026-09-22" in md and "< 0.3" in md


def test_older_notebooks_unchanged_by_builder():
    """重新產生 v3.3／v3.4f／v3.5／v3.5f 四本，cell 內容要與 HEAD 逐字同（只比 cell，不比 JSON 排版：v3.3 那本是 Colab 存過的格式）。"""
    def cells(text):
        nb = json.loads(text)
        return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
    for args, name in ((["--weights", "v33"], "cpg_rl_v3_colab.ipynb"), (["--gains", "factory", "--weights", "v33"], "cpg_rl_v3_4f_colab.ipynb"),
                       (["--weights", "v35"], "cpg_rl_v3_5_colab.ipynb"), (["--gains", "factory", "--weights", "v35"], "cpg_rl_v3_5f_colab.ipynb")):
        path = NBDIR / name
        head = subprocess.run(["git", "show", f"HEAD:task7/notebooks/{name}"], capture_output=True, text=True, cwd=NBDIR.parents[1], check=True).stdout
        subprocess.run(["python3", str(NBDIR / "build_nb_v3.py"), *args], check=True, capture_output=True)
        try:
            assert cells(path.read_text(encoding="utf-8")) == cells(head), name
        finally:
            subprocess.run(["git", "checkout", "--", str(path)], check=True, capture_output=True)


def test_v37f_notebook_env_and_stoploss():
    nb = NBDIR / "cpg_rl_v3_7f_colab.ipynb"
    assert nb.exists(), "先跑 python3 task7/notebooks/build_nb_v3.py --gains factory --weights v37"
    src = _code_src(nb)
    assert src.count('v3.DualModeEnv(gains="factory", weights=v3.W37)') == 2
    assert 'v3.DualModeEnv(gains="factory", weights=v3.W37, ref=dict(cyc_amp_rand=False))' in src
    assert 'randomization_fn=v3.make_domain_randomize("factory", com_y_mm=5.0)' in src and "step {ps('t_step'):.2f}" in src
    assert 'model.save_params("cpg_rl_v3_7f_params.pkl", params)' in src
    md = _md_src(nb)
    assert "v3.7f" in md and "cyc_lat_decouple" in md and "< 0.45" in md
    v36 = _code_src(NB)
    assert "W37" not in v36                                                          # v3.6f 那本不受影響


def test_v37bf_notebook():
    nb = NBDIR / "cpg_rl_v3_7bf_colab.ipynb"
    assert nb.exists(), "先跑 python3 task7/notebooks/build_nb_v3.py --gains factory --weights v37b"
    src = _code_src(nb)
    assert src.count('v3.DualModeEnv(gains="factory", weights=v3.W37B)') == 2 and 'model.save_params("cpg_rl_v3_7bf_params.pkl", params)' in src
    assert 'randomization_fn=v3.make_domain_randomize("factory", com_y_mm=5.0)' in src and "step {ps('t_step'):.2f}" in src
    md = _md_src(nb); assert "v3.7b" in md and "輪前饋" in md and "< 0.45" in md
    assert "W37B" not in _code_src(NBDIR / "cpg_rl_v3_7f_colab.ipynb")
