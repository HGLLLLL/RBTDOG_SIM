"""v3.5 notebook：由 build_nb_v3.py --weights v35 產生，env 要帶 W35、progress 要印 abad/drift/head。"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "cpg_rl_v3_5_colab.ipynb"


def _code_src(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_v35_notebook_uses_w35_and_prints_drift():
    assert NB.exists(), "先跑 python3 task7/notebooks/build_nb_v3.py --weights v35"
    src = _code_src(NB)
    assert src.count("v3.DualModeEnv(weights=v3.W35)") == 2                       # import 格 + train 格
    assert "v3.DualModeEnv(weights=v3.W35, ref=dict(cyc_amp_rand=False))" in src  # eval_env
    assert "abad {ps('abad_bias'):.1f}" in src and "drift {ps('vx_drift'):+.3f}" in src and "head {ps('head_deg'):+.1f}" in src
    assert "vy {ps('t_vyrel'):.2f}/6" in src
    assert 'model.save_params("cpg_rl_v3_5_params.pkl", params)' in src


def test_v33_notebook_untouched_by_builder_change():
    """v3.3 那本的 env 行與 progress 行不能帶 v3.5 的東西。"""
    src = _code_src(NB.with_name("cpg_rl_v3_colab.ipynb"))
    assert "W35" not in src and "abad_bias" not in src and "vy {ps('t_vyrel'):.2f}/3" in src
