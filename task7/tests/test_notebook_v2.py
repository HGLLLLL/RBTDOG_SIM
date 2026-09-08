"""Colab notebook v2 的契約：clone main、env 只 import 不內嵌、版本鎖死、v2 權重檔名。"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "cpg_rl_max_colab.ipynb"


def test_notebook_v2_contract():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'BRANCH = "main"' in src
    assert "import rl_env_max" in src and "MaxCpgEnv" in src
    assert "brax==0.14.2" in src and "mujoco==3.10.0" in src
    assert "cpg_rl_max_v2_params.pkl" in src
    assert "class MaxCpgEnv" not in src          # env 不再住在 notebook 裡
    assert "gb.BASELINE[" not in src             # 只能經 rl_env_max 用 BASELINE_A
    assert "policy_hidden_layer_sizes=(256, 256, 128)" in src
    assert "randomization_fn=re.domain_randomize" in src
