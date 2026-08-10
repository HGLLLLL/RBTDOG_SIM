"""資產完整性測試：17 個 STL、授權、來源記錄都要在。"""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "d1_edu_w"

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


def test_only_referenced_meshes_copied():
    """zsl-1w 上游有 18 個 STL，BASE_LINK_ori.STL 未被 URDF 引用，不該複製進來。"""
    got = sorted(p.name for p in (MODEL_DIR / "meshes").glob("*.STL"))
    assert got == sorted(EXPECTED_MESHES), f"網格清單不符，多餘或缺少: {set(got) ^ set(EXPECTED_MESHES)}"


def test_license_and_source_recorded():
    assert (MODEL_DIR / "LICENSE").is_file()
    src = (MODEL_DIR / "SOURCE.md").read_text(encoding="utf-8")
    assert "zsibot/genisom_model" in src
    assert "BSD-3" in src
    assert "zsl-1w" in src, "SOURCE.md 必須寫明取用的是輪足版 zsl-1w 目錄"


def test_point_foot_assets_removed():
    """點足版資產必須已刪除，避免兩份模型並存造成誤用。"""
    assert not (MODEL_DIR.parent / "d1_edu").exists(), "舊的點足版目錄 task6/model/d1_edu 尚未刪除"
