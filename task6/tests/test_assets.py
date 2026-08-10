"""資產完整性測試：17 個 STL、授權、來源記錄都要在。"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "d1_edu"

EXPECTED_MESHES = [
    "BASE_LINK.STL",
    *[f"{leg}_{part}_LINK.STL"
      for leg in ("FL", "FR", "RL", "RR")
      for part in ("ABAD", "HIP", "KNEE", "FOOT")],
]


def test_all_17_meshes_present():
    missing = [n for n in EXPECTED_MESHES if not (MODEL_DIR / "meshes" / n).is_file()]
    assert missing == [], f"缺少網格: {missing}"
    assert len(EXPECTED_MESHES) == 17


def test_license_and_source_recorded():
    assert (MODEL_DIR / "LICENSE").is_file()
    src = (MODEL_DIR / "SOURCE.md").read_text(encoding="utf-8")
    assert "zsibot/genisom_model" in src
    assert "BSD-3" in src
