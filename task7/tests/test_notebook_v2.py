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


def test_notebook_v2_1_contract():
    """v2.1 notebook：preset 明寫、權重檔名不同、校準格有佔比斷言；v2 notebook 不受影響。"""
    nb = json.loads((NB.parent / "cpg_rl_max_v2_1_colab.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'PRESET = "v2.1"' in src and "MaxCpgEnv(preset=PRESET)" in src
    assert "cpg_rl_max_v2_1_params.pkl" in src and "cpg_rl_max_v2_params.pkl" not in src
    assert "0.10 <= roll_sh <= 0.25" in src           # 佔比斷言
    assert 'assert np.allclose(kp_xml, np.tile(mm.KP3_A, 4))' in src   # 增益斷言（ABAD 60）
    assert 'BRANCH = "main"' in src
    old = json.loads(NB.read_text(encoding="utf-8"))
    old_src = "\n".join("".join(c["source"]) for c in old["cells"] if c["cell_type"] == "code")
    assert "PRESET" not in old_src and "cpg_rl_max_v2_params.pkl" in old_src


def test_notebook_v2_2_contract():
    nb = json.loads((NB.parent / "cpg_rl_max_v2_2_colab.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'PRESET = "v2.2"' in src and "MaxCpgEnv(preset=PRESET)" in src
    assert "cpg_rl_max_v2_2_params.pkl" in src and "--preset v2.2" in src
    assert "re.CAL_BANDS[PRESET]" in src and "baseline_action(LAYOUT)" in src
    assert "(env.action_size, env.observation_size) == (ACT_DIM, OBS_DIM)" in src
    # 舊的兩個 notebook 不受影響
    old = json.loads((NB.parent / "cpg_rl_max_v2_1_colab.ipynb").read_text(encoding="utf-8"))
    old_src = "\n".join("".join(c["source"]) for c in old["cells"] if c["cell_type"] == "code")
    assert 'PRESET = "v2.1"' in old_src and "cpg_rl_max_v2_1_params.pkl" in old_src
