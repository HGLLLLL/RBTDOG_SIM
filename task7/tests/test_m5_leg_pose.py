"""M5（腿關節驅動）與 coord.py（座標換算／限位／姿勢）的離線驗證。

★ 這份測試的存在理由，是 task7 那條反覆出現七次的教訓：
  **出錯的通常是量測／換算工具本身，不是被量的東西。**
  所以這裡刻意不只驗「程式自我一致」，而是每一項都拉一個**外部對照量**：

  | 測什麼 | 外部對照 |
  |---|---|
  | SIGN / OFFSET | `M4_pose_capture.py` 裡那份獨立副本 |
  | 換算式 | `logs/m_logs_trip6/pose_*.json` 的**實機**擷取資料 |
  | 機構限位 | 官方 MJCF `model/zgws/zgws.xml` 的 `<joint range>` |
  | 寫入語意 | 重算一次欄位位移，並記錄寫入**順序** |

  只有自我一致的測試會一起錯，抓不到「自洽但錯誤」。

⚠️ 純標準函式庫（不需要 numpy / mujoco）—— 因為被測的 `coord.py` / `shm_io.py`
   要跑在狗上，狗上沒有這些套件；測試也就不該引入它們。
⚠️ 匯入 `M5_leg_pose` 不會碰 /dev/shm（模組層只有常數與函式定義）。
"""
from __future__ import annotations

import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realbot"))

import coord                      # noqa: E402
import shm_io                     # noqa: E402
import M4_pose_capture as m4      # noqa: E402
import M5_leg_pose as m5          # noqa: E402

LOG_DIR = ROOT / "logs" / "m_logs_trip6"
MJCF = ROOT / "model" / "zgws" / "zgws.xml"

# MJCF 的關節名 = <腿前綴>_<種類>_JOINT
KIND2MJCF = {
    coord.KIND_HIP_ROLL: "ABAD",
    coord.KIND_HIP_PITCH: "HIP",
    coord.KIND_KNEE: "KNEE",
}

LEG_JOINTS = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]


# ════════════════════════════════════════════════════════ 1. 兩份 SIGN/OFFSET 不可漂移
# coord.py 是單一事實來源，但 M4 先於它存在且自帶一份副本（M4 是「驗證換算式」
# 的工具，刻意不依賴待驗的東西）。兩份都還活著，就必須逐項釘住。

def test_sign_offset_tables_have_same_shape():
    """先比結構再比值 —— 少一個 key 造成的 KeyError 會偽裝成別的錯。"""
    assert set(coord.SIGN) == set(m4.SIGN)
    assert set(coord.OFFSET) == set(m4.OFFSET)
    assert set(coord.SIGN) == set(coord.OFFSET) == set(coord.ALL_KINDS)
    for kind in coord.ALL_KINDS:
        assert set(coord.SIGN[kind]) == set(m4.SIGN[kind]) == set(coord.LEGS)
        assert set(coord.OFFSET[kind]) == set(m4.OFFSET[kind]) == set(coord.LEGS)


@pytest.mark.parametrize("kind", coord.ALL_KINDS)
@pytest.mark.parametrize("leg", coord.LEGS)
def test_sign_matches_m4(kind, leg):
    assert coord.SIGN[kind][leg] == m4.SIGN[kind][leg], \
        f"{leg}{kind} 的 side_sign 在 coord.py 與 M4_pose_capture.py 之間漂移了"


@pytest.mark.parametrize("kind", coord.ALL_KINDS)
@pytest.mark.parametrize("leg", coord.LEGS)
def test_offset_matches_m4(kind, leg):
    assert coord.OFFSET[kind][leg] == pytest.approx(m4.OFFSET[kind][leg], abs=0.0), \
        f"{leg}{kind} 的 offset 在 coord.py 與 M4_pose_capture.py 之間漂移了"


def test_sign_is_plus_or_minus_one():
    """side_sign 只可能是 ±1。若哪天變成別的值，`to_ctrl` 的除法與
    M5 裡「力矩換算乘以 sign」的等價性都會失效。"""
    for kind in coord.ALL_KINDS:
        for leg in coord.LEGS:
            assert abs(coord.SIGN[kind][leg]) == 1.0


def test_leg_name_mapping_is_consistent():
    """SHM 腿序 fl/fr/bl/br 與設定檔腿序 FR/FL/RR/RL 的對應，兩邊也必須一致。"""
    assert coord._CFG2SHM == m4._CFG2SHM
    assert tuple(coord._CFG_ORDER) == tuple(m4._CFG_ORDER)
    assert set(coord.SHM2MJCF_LEG) == set(coord.LEGS)


# ════════════════════════════════════════════════════════ 2. 換算往返

_ANGLES = (-2.5, -1.2, -0.523, -0.05, 0.0, 0.05, 0.6, 1.4, 2.79)


@pytest.mark.parametrize("angle", _ANGLES)
@pytest.mark.parametrize("joint", shm_io.JOINTS)
def test_to_ctrl_inverts_to_motor(joint, angle):
    assert coord.to_ctrl(joint, coord.to_motor(joint, angle)) == pytest.approx(angle, abs=1e-12)


@pytest.mark.parametrize("angle", _ANGLES)
@pytest.mark.parametrize("joint", shm_io.JOINTS)
def test_to_motor_inverts_to_ctrl(joint, angle):
    """反方向也要驗 —— 只驗單向的話，一個「兩邊都用同一個錯誤常數」的 bug 抓不到。"""
    assert coord.to_motor(joint, coord.to_ctrl(joint, angle)) == pytest.approx(angle, abs=1e-12)


def test_to_motor_is_affine_with_declared_sign():
    """釘住換算式的形狀：馬達角 = sign × 控制器角 + offset。
    斜率由兩點差分求得（offset 在差分中消掉），與宣告的 SIGN 逐項比對。"""
    for joint in shm_io.JOINTS:
        leg, kind = joint[:2], joint[2:]
        slope = coord.to_motor(joint, 1.0) - coord.to_motor(joint, 0.0)
        assert slope == pytest.approx(coord.SIGN[kind][leg], abs=1e-12)
        assert coord.to_motor(joint, 0.0) == pytest.approx(coord.OFFSET[kind][leg], abs=1e-12)


def test_unknown_joint_name_raises():
    with pytest.raises(ValueError):
        coord.to_motor("zz1_hip_roll", 0.0)
    with pytest.raises(ValueError):
        coord.to_ctrl("fl9_nope", 0.0)


# ════════════════════════════════════════════════════════ 3. ★ 實機資料驗證（最重要）
# 2026-08-25 用 M4 在實機擷取的四個姿勢。把 `mean`（馬達角）用 coord.py 換回
# 控制器角，跟設定檔記載的姿勢比 RMS 殘差。
# 這三個數字是**實測值**，不是理論值 —— 換算式被改壞，這裡立刻爆。
# 佐證文件：docs/座標換算式驗證結果_2026-08-25.md

def _load_pose_json(name: str) -> dict:
    with open(LOG_DIR / name, encoding="utf-8") as f:
        return json.load(f)["mean"]


def _rms_vs(motor_mean: dict, pose_ctrl: dict) -> float:
    res = [coord.to_ctrl(j, motor_mean[j]) - pose_ctrl[j] for j in pose_ctrl]
    return math.sqrt(sum(r * r for r in res) / len(res))


def _stand_knee_back() -> dict:
    return coord.flip_rear_knee_mode(coord.POSES["stand"])


REAL_CASES = [
    ("pose_stand_knee_front.json", lambda: coord.POSES["stand"], 0.0353),
    ("pose_stand_knee_back.json", _stand_knee_back, 0.0417),
    ("pose_crawl.json", lambda: coord.POSES["crouch"], 0.0862),
]


@pytest.mark.parametrize("fname,pose_fn,want_rms", REAL_CASES)
def test_real_capture_rms(fname, pose_fn, want_rms):
    """★ 實機四姿勢的 RMS 殘差 —— 座標換算鏈的端到端驗證。"""
    rms = _rms_vs(_load_pose_json(fname), pose_fn())
    assert rms == pytest.approx(want_rms, abs=1e-3), (
        f"{fname} 的 RMS 殘差變成 {rms:.4f}（原本 {want_rms}）—— "
        "座標換算式或姿勢常數被改動了，先回頭核對 coord.py")


@pytest.mark.parametrize("fname,pose_fn,want_rms", REAL_CASES)
def test_real_capture_is_discriminative(fname, pose_fn, want_rms):
    """★ 對照量：正確的姿勢必須明顯比**錯誤的姿勢**貼合。

    只看「RMS 很小」不足以定案 —— 座標驗證那次就是因為「四輪共面」對 V1/V3
    沒有鑑別力，差點選錯。這裡確認正解至少比最接近的錯誤解好 3 倍。
    """
    mean = _load_pose_json(fname)
    good = _rms_vs(mean, pose_fn())
    wrongs = {
        "stand": coord.POSES["stand"],
        "stand_knee_back": _stand_knee_back(),
        "home": coord.POSES["home"],
        "crouch": coord.POSES["crouch"],
        "crouch_knee_back": coord.flip_rear_knee_mode(coord.POSES["crouch"]),
    }
    others = [(n, _rms_vs(mean, p)) for n, p in wrongs.items()
              if abs(_rms_vs(mean, p) - good) > 1e-9]
    assert others, "候選姿勢表沒有任何『錯誤解』可以對照"
    nm, second = min(others, key=lambda x: x[1])
    assert second > 3.0 * good, (
        f"{fname}: 正解 RMS {good:.4f}，但次佳 {nm} 只有 {second:.4f} —— "
        "鑑別力不足，這筆比對不能當證據")


def test_real_capture_uses_a_flipped_rear_knee_mode():
    """兩種膝模式必須真的是不同的目標，否則上面那組比對等於沒比。"""
    front = coord.POSES["stand"]
    back = _stand_knee_back()
    for leg in coord.FRONT_LEGS:
        for k in coord.LEG_KINDS:
            assert front[leg + k] == back[leg + k], "前腿不該被膝模式切換動到"
    for leg in coord.REAR_LEGS:
        assert back[leg + coord.KIND_HIP_PITCH] == -front[leg + coord.KIND_HIP_PITCH]
        assert back[leg + coord.KIND_KNEE] == -front[leg + coord.KIND_KNEE]
        assert back[leg + coord.KIND_HIP_ROLL] == front[leg + coord.KIND_HIP_ROLL]


def test_flip_rear_knee_mode_is_involution_and_pure():
    """翻兩次要回到原狀，而且不可就地改動輸入（M5 直接把 POSES 拿去翻）。"""
    orig = dict(coord.POSES["stand"])
    once = coord.flip_rear_knee_mode(coord.POSES["stand"])
    twice = coord.flip_rear_knee_mode(once)
    assert twice == orig
    assert coord.POSES["stand"] == orig, "flip_rear_knee_mode() 汙染了 coord.POSES"


def test_capture_files_are_steady_enough_to_trust():
    """對照量：擷取當下的標準差要夠小，否則上面的 RMS 只是在量晃動。"""
    for fname, _, _ in REAL_CASES:
        with open(LOG_DIR / fname, encoding="utf-8") as f:
            rec = json.load(f)
        worst = max(rec["std"][j] for j in LEG_JOINTS)
        assert worst < 0.01, f"{fname} 最大標準差 {worst:.5f} rad —— 狗當時在晃"


# ════════════════════════════════════════════════════════ 4. 限位 vs 官方 MJCF
# ★ 這一項會自動抓到「MJCF 改版而 coord.py 沒跟上」。
#   coord.LIMITS 是手抄的，手抄就會過期。

def _mjcf_ranges() -> dict:
    """解析官方 MJCF，回傳 {MJCF 關節名: (lo, hi)}。"""
    root = ET.parse(MJCF).getroot()
    out = {}
    for j in root.iter("joint"):
        nm, rg = j.get("name"), j.get("range")
        if nm and rg:
            lo, hi = (float(x) for x in rg.split())
            out[nm] = (lo, hi)
    return out


def test_mjcf_parse_found_all_joints():
    """先確認解析器真的抓到東西 —— 空 dict 會讓下面每一項都『通過』。"""
    rg = _mjcf_ranges()
    for leg in coord.LEGS:
        for kind in coord.LEG_KINDS:
            assert f"{coord.SHM2MJCF_LEG[leg]}_{KIND2MJCF[kind]}_JOINT" in rg
    assert len(rg) >= 16


@pytest.mark.parametrize("joint", LEG_JOINTS)
def test_limits_match_mjcf(joint):
    leg, kind = joint[:2], joint[2:]
    mj_name = f"{coord.SHM2MJCF_LEG[leg]}_{KIND2MJCF[kind]}_JOINT"
    want = _mjcf_ranges()[mj_name]
    assert coord.LIMITS[joint] == pytest.approx(want, abs=1e-9), (
        f"{joint}（MJCF {mj_name}）的限位與官方 MJCF 不符：\n"
        f"  coord.LIMITS = {coord.LIMITS[joint]}\n  MJCF range   = {want}\n"
        "→ MJCF 可能改版了，coord.py 要跟上（不要反過來改 MJCF）")


def test_limits_table_covers_exactly_the_twelve_leg_joints():
    assert set(coord.LIMITS) == set(LEG_JOINTS)


def test_limits_of_wheel_is_unbounded():
    """輪關節沒有機構限位（MJCF 是 ±99999）。M5 靠這個讓輪子不被限位擋。"""
    for w in shm_io.WHEELS:
        lo, hi = coord.limits_of(w)
        assert lo == float("-inf") and hi == float("inf")
    assert coord.check_limit("fl4_foot", 1e6, margin=0.5) == ""


def test_abad_limits_are_asymmetric_left_right():
    """釘住 coord.py 註解裡特別警告的性質：ABAD 行程窄（1.22 rad）且左右不對稱。
    這是「微動測試不要挑 ABAD」的依據，性質變了就要重新評估。"""
    for leg in ("fl", "bl"):
        assert coord.LIMITS[leg + coord.KIND_HIP_ROLL] == (-0.523, 0.697)
    for leg in ("fr", "br"):
        assert coord.LIMITS[leg + coord.KIND_HIP_ROLL] == (-0.697, 0.523)
    for leg in coord.LEGS:
        lo, hi = coord.LIMITS[leg + coord.KIND_HIP_ROLL]
        assert hi - lo == pytest.approx(1.22, abs=1e-9)


def test_hip_pitch_limits_are_mirrored_front_rear():
    """前後 HIP 限位互為鏡像 —— 姿勢表也是前後反號，兩者必須配套。"""
    for leg in coord.FRONT_LEGS:
        assert coord.LIMITS[leg + coord.KIND_HIP_PITCH] == (-2.442, 2.791)
    for leg in coord.REAR_LEGS:
        assert coord.LIMITS[leg + coord.KIND_HIP_PITCH] == (-2.791, 2.442)


def test_check_limit_reports_both_directions():
    j = "fl" + coord.KIND_HIP_PITCH
    lo, hi = coord.LIMITS[j]
    assert coord.check_limit(j, 0.0) == ""
    assert "低於下限" in coord.check_limit(j, lo - 0.01)
    assert "高於上限" in coord.check_limit(j, hi + 0.01)
    # 餘裕把邊界往內縮
    assert coord.check_limit(j, hi - 0.01) == ""
    assert coord.check_limit(j, hi - 0.01, margin=0.05) != ""


# ════════════════════════════════════════════════════════ 5. 姿勢全部落在限位內

_POSE_CASES = [(nm, False) for nm in sorted(coord.POSES)] + \
              [(nm, True) for nm in sorted(coord.POSES)]

MARGIN = 0.05


@pytest.mark.parametrize("pose_name,knee_back", _POSE_CASES)
def test_poses_within_limits_with_margin(pose_name, knee_back):
    """三組姿勢（含 knee_back 變體）都要有 ≥0.05 rad 的餘裕。

    M5 啟動時用 --margin(預設 0.05) 查目標角，餘裕不足的姿勢會讓 M5 直接拒跑。
    """
    pose = coord.POSES[pose_name]
    if knee_back:
        pose = coord.flip_rear_knee_mode(pose)
    assert set(pose) == set(LEG_JOINTS)
    for j, ang in pose.items():
        msg = coord.check_limit(j, ang, MARGIN)
        assert msg == "", f"{pose_name}{'_knee_back' if knee_back else ''} 的 {j} {msg}"


@pytest.mark.parametrize("pose_name,knee_back", _POSE_CASES)
def test_pose_to_motor_covers_all_joints(pose_name, knee_back):
    pose = coord.POSES[pose_name]
    if knee_back:
        pose = coord.flip_rear_knee_mode(pose)
    mot = coord.pose_to_motor(pose)
    assert set(mot) == set(pose)
    for j in pose:
        assert coord.to_ctrl(j, mot[j]) == pytest.approx(pose[j], abs=1e-12)


def test_poses_are_front_rear_mirrored():
    """姿勢是 X 型：前後 hip/knee 反號。四腿同號會做出很怪的東西（HANDOFF 警告過）。"""
    for nm, pose in coord.POSES.items():
        for kind in (coord.KIND_HIP_PITCH, coord.KIND_KNEE):
            assert pose["fl" + kind] == pose["fr" + kind], nm
            assert pose["bl" + kind] == pose["br" + kind], nm
            assert pose["bl" + kind] == -pose["fl" + kind], nm


def test_stand_home_crouch_are_distinct_and_ordered():
    """三組姿勢必須真的不同，而且 crouch 蹲得比 stand 低（hip 角更大）。"""
    hips = {nm: p["fl" + coord.KIND_HIP_PITCH] for nm, p in coord.POSES.items()}
    assert hips["stand"] < hips["home"] < hips["crouch"]


# ════════════════════════════════════════════════════════ 6. expand_joints()

def test_expand_all_is_twelve_leg_joints_without_wheels():
    js = m5.expand_joints("all")
    assert len(js) == 12
    assert js == LEG_JOINTS
    assert not any(j.endswith(coord.KIND_WHEEL) for j in js)
    assert not set(js) & set(shm_io.WHEELS)


def test_expand_front_and_rear():
    assert m5.expand_joints("front") == [lg + k for lg in coord.FRONT_LEGS
                                         for k in coord.LEG_KINDS]
    assert m5.expand_joints("rear") == [lg + k for lg in coord.REAR_LEGS
                                        for k in coord.LEG_KINDS]
    assert m5.expand_joints("front") + m5.expand_joints("rear") == m5.expand_joints("all")


@pytest.mark.parametrize("leg", coord.LEGS)
def test_expand_single_leg(leg):
    assert m5.expand_joints(leg) == [leg + k for k in coord.LEG_KINDS]


def test_expand_full_joint_name():
    assert m5.expand_joints("fl2_hip_pitch") == ["fl2_hip_pitch"]


def test_expand_combination_dedups_and_keeps_order():
    got = m5.expand_joints("fl2_hip_pitch, fl , fr3_knee_pitch,fl")
    assert got == ["fl2_hip_pitch", "fl1_hip_roll", "fl3_knee_pitch", "fr3_knee_pitch"]
    assert len(got) == len(set(got))


def test_expand_ignores_blank_tokens():
    assert m5.expand_joints("fl, ,,fl") == [f"fl{k}" for k in coord.LEG_KINDS]
    assert m5.expand_joints("") == []
    assert m5.expand_joints("  ,  ") == []


@pytest.mark.parametrize("wheel", shm_io.WHEELS)
def test_expand_rejects_wheel_joints(wheel):
    """★ 把輪關節放進 --joints 必須擋掉：位置控制餵給輪子＝叫它轉到某個絕對角度，會全速甩。"""
    with pytest.raises(SystemExit) as e:
        m5.expand_joints(wheel)
    assert "wheel-vel" in str(e.value) or "輪關節" in str(e.value)


def test_expand_rejects_wheel_even_inside_a_valid_combination():
    """混在合法群組裡也要擋 —— 只驗單獨傳入的話，`all,fl4_foot` 會漏網。"""
    with pytest.raises(SystemExit):
        m5.expand_joints("all,fl4_foot")


@pytest.mark.parametrize("bad", ["xx", "FL", "fl5_nope", "left", "all_legs", "fl2-hip-pitch"])
def test_expand_rejects_garbage(bad):
    with pytest.raises(SystemExit):
        m5.expand_joints(bad)


# ════════════════════════════════════════════════════════ 7. smoothstep()

def test_smoothstep_endpoints_and_midpoint():
    assert m5.smoothstep(0.0) == pytest.approx(0.0, abs=1e-12)
    assert m5.smoothstep(1.0) == pytest.approx(1.0, abs=1e-12)
    assert m5.smoothstep(0.5) == pytest.approx(0.5, abs=1e-12)


def test_smoothstep_is_monotonic():
    prev = -1.0
    for i in range(201):
        v = m5.smoothstep(i / 200.0)
        assert v >= prev - 1e-15, f"u={i/200.0} 時不再遞增"
        prev = v


def test_smoothstep_clamps_out_of_range():
    for u in (-5.0, -1e-9, -0.3):
        assert m5.smoothstep(u) == pytest.approx(0.0, abs=1e-12)
    for u in (1.0 + 1e-9, 1.5, 100.0):
        assert m5.smoothstep(u) == pytest.approx(1.0, abs=1e-12)


def test_smoothstep_zero_slope_at_ends():
    """兩端速度為 0 才不會對吊著的 41 kg 機身造成衝擊 —— 這是選餘弦而非線性的理由。"""
    h = 1e-4
    assert abs(m5.smoothstep(h) - m5.smoothstep(0.0)) / h < 1e-3
    assert abs(m5.smoothstep(1.0) - m5.smoothstep(1.0 - h)) / h < 1e-3


# ════════════════════════════════════════════════════════ 8. load_ref() / ref_torque()

def test_load_ref_returns_empty_dict_when_missing(monkeypatch, tmp_path, capsys):
    """對照表不在時程式仍要能跑（少一個判準，不是崩潰）。"""
    monkeypatch.setattr(m5, "REF_JSON_PATHS", (str(tmp_path / "nope.json"),))
    assert m5.load_ref() == {}
    assert "找不到預演對照表" in capsys.readouterr().out


def test_load_ref_survives_corrupt_json(monkeypatch, tmp_path, capsys):
    bad = tmp_path / "hang_torque_ref.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(m5, "REF_JSON_PATHS", (str(bad),))
    assert m5.load_ref() == {}
    assert "讀取失敗" in capsys.readouterr().out


def test_load_ref_reads_first_existing_path(monkeypatch, tmp_path):
    """兩個候選路徑時取第一個存在的（狗上放 realbot/、本機放 reference/）。"""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    b.write_text(json.dumps({"who": "b"}), encoding="utf-8")
    monkeypatch.setattr(m5, "REF_JSON_PATHS", (str(a), str(b)))
    assert m5.load_ref() == {"who": "b"}
    a.write_text(json.dumps({"who": "a"}), encoding="utf-8")
    assert m5.load_ref() == {"who": "a"}


@pytest.mark.parametrize("ref", [
    {},
    {"poses": {}},
    {"poses": {"stand": {}}},
    {"poses": {"stand": {"tau_gravity": {}}}},
    {"poses": "不是 dict"},
    {"poses": {"stand": {"tau_gravity": {"fl2_hip_pitch": "壞掉的字串"}}}},
    {"poses": {"stand": {"tau_gravity": None}}},
])
def test_ref_torque_returns_none_instead_of_raising(ref):
    assert m5.ref_torque(ref, "stand", "fl2_hip_pitch") is None


def test_ref_torque_returns_none_for_empty_pose_name():
    """--delta 模式沒有姿勢名，M5 會傳空字串進來。"""
    ref = {"poses": {"": {"tau_gravity": {"fl2_hip_pitch": 9.9}}}}
    assert m5.ref_torque(ref, "", "fl2_hip_pitch") is None


@pytest.mark.parametrize("tau_key", m5._TAU_KEYS)
@pytest.mark.parametrize("pose_key", ["stand", "STAND"])
def test_ref_torque_tolerates_key_spellings(pose_key, tau_key):
    """產生端與消費端是兩支獨立演進的程式，key 拼法容忍是刻意的設計。"""
    ref = {"poses": {pose_key: {tau_key: {"fl2_hip_pitch": 2.5}}}}
    got = m5.ref_torque(ref, "stand", "fl2_hip_pitch")
    assert isinstance(got, float) and got == pytest.approx(2.5)


def test_pose_entry_matches_only_exact_upper_or_lower():
    """釘住 _pose_entry 的**實際**契約：只試「原樣 / 全大寫 / 全小寫」三種。
    `_pose_entry` 的 docstring 寫「大小寫不敏感」，但混合大小寫（`Stand`）不會命中 ——
    真的要不敏感就得逐一 casefold 比對。這裡照實記錄，免得誰照 docstring 去產檔案。"""
    assert m5._pose_entry({"poses": {"Stand": {"tau_gravity": {}}}}, "stand") == {}
    assert m5._pose_entry({"poses": {"STAND": {"x": 1}}}, "stand") == {"x": 1}
    assert m5._pose_entry({"poses": {"stand": {"x": 1}}}, "STAND") == {"x": 1}


def test_ref_torque_never_returns_the_motor_frame_variant():
    """★ 對照表同時存著馬達座標系的版本（key 帶 `_motor`）。拿錯座標系比拿不到還糟 ——
    一半的關節會憑空反號，正是 task7 第七號「診斷輸出騙人」。"""
    ref = {"poses": {"STAND": {
        "tau_gravity": {"fl2_hip_pitch": 2.5},
        "tau_gravity_motor": {"fl2_hip_pitch": -2.5},
        "tau_gravity_mjcf": {"FBL_HIP_JOINT": 2.5},
    }}}
    assert m5.ref_torque(ref, "stand", "fl2_hip_pitch") == pytest.approx(2.5)
    # 只有 _motor 版本時要回 None，不可退而求其次拿錯座標系的值
    only_motor = {"poses": {"STAND": {"tau_gravity_motor": {"fl2_hip_pitch": -2.5}}}}
    assert m5.ref_torque(only_motor, "stand", "fl2_hip_pitch") is None


def test_ref_hang_angle_accepts_both_layouts():
    """自然下垂那段可能在 poses['HANG_FREE'] 也可能在頂層 hang_free。"""
    a = {"poses": {"HANG_FREE": {"q_ctrl": {"fl2_hip_pitch": 0.01}}}}
    b = {"hang_free": {"qpos_ctrl": {"fl2_hip_pitch": 0.02}}}
    assert m5.ref_hang_angle(a, "fl2_hip_pitch") == pytest.approx(0.01)
    assert m5.ref_hang_angle(b, "fl2_hip_pitch") == pytest.approx(0.02)
    assert m5.ref_hang_angle({}, "fl2_hip_pitch") is None


def test_ref_stiffness_accepts_both_layouts():
    a = {"poses": {"HANG_FREE": {"stiffness_at_hang": {"fl2_hip_pitch": 30.0}}}}
    b = {"stiffness": {"fl2_hip_pitch": 31.0}}
    assert m5.ref_stiffness(a, "fl2_hip_pitch") == pytest.approx(30.0)
    assert m5.ref_stiffness(b, "fl2_hip_pitch") == pytest.approx(31.0)
    assert m5.ref_stiffness({}, "fl2_hip_pitch") is None


# ---- 對著**實際會帶上狗的那份檔案**驗，不是只驗合成資料。
#      「產生端與消費端各自演進」的風險只有這樣才抓得到。

REF_PATH = ROOT / "reference" / "hang_torque_ref.json"
_need_ref = pytest.mark.skipif(not REF_PATH.exists(),
                               reason="尚未產生 reference/hang_torque_ref.json")


@pytest.fixture(scope="module")
def shipped_ref():
    with open(REF_PATH, encoding="utf-8") as f:
        return json.load(f)


@_need_ref
@pytest.mark.parametrize("pose_name", ["stand", "home", "crouch"])
def test_m5_can_read_every_joint_from_shipped_ref(shipped_ref, pose_name):
    """★ 端到端：M5 的取值函式 × 實際的對照表檔案。

    這一項失敗的症狀在現場是**沉默的**：S0 仍會印「✅ 已載入預演對照表」，
    但每一列的「預演τ」都是「—」，反號判讀整段失效。
    """
    missing = [j for j in LEG_JOINTS if m5.ref_torque(shipped_ref, pose_name, j) is None]
    assert not missing, f"取不到 {pose_name} 預演力矩的關節：{missing}"


@_need_ref
def test_shipped_ref_hang_angles_are_readable(shipped_ref):
    missing = [j for j in LEG_JOINTS if m5.ref_hang_angle(shipped_ref, j) is None]
    assert not missing, f"取不到自然下垂角的關節：{missing}"


@_need_ref
def test_shipped_ref_torques_are_in_controller_frame(shipped_ref):
    """★ 對照量：對照表宣稱的 sign/offset 必須與 coord.py 逐項相同。

    兩邊都對了，「拿預演力矩去比實測力矩」才成立；
    只要有一項不同，現場那張「✅相符 / ❌反號」的判讀表就是騙人的。
    """
    ct = shipped_ref.get("coord_transform", {})
    sign, offset = ct.get("sign", {}), ct.get("offset", {})
    assert sign and offset, "對照表沒有記錄它自己用的換算常數 —— 無從交叉核對"
    for j in LEG_JOINTS:
        leg, kind = j[:2], j[2:]
        assert sign[j] == coord.SIGN[kind][leg], f"{j} 的 sign 與 coord.py 不符"
        assert offset[j] == pytest.approx(coord.OFFSET[kind][leg], abs=1e-9), \
            f"{j} 的 offset 與 coord.py 不符"


@_need_ref
def test_shipped_ref_limits_match_coord(shipped_ref):
    lim = shipped_ref.get("joint_limits_rad") or {}
    if not lim:
        pytest.skip("對照表沒有 joint_limits_rad")
    for j in LEG_JOINTS:
        if j in lim:
            assert tuple(lim[j]) == pytest.approx(coord.LIMITS[j], abs=1e-9), j


@_need_ref
def test_knee_back_poses_have_reference_torques(shipped_ref):
    """★ 曾經是 xfail：對照表原本缺 knee_back 那三組，導致 `--knee-back`（S5）
    **沉默地**沒有力矩對照。2026-08-26 補齊，這裡釘住不許再退回去。"""
    for pose_name in ("stand_knee_back", "crouch_knee_back"):
        missing = [j for j in LEG_JOINTS
                   if m5.ref_torque(shipped_ref, pose_name, j) is None]
        assert not missing, f"{pose_name} 取不到預演力矩：{missing}"


@_need_ref
def test_shipped_ref_has_hang_stiffness(shipped_ref):
    """★ 曾經是 xfail：缺了剛度，S1 微動就只剩「方向對不對」一個判準。
    2026-08-26 補齊。"""
    missing = [j for j in LEG_JOINTS if m5.ref_stiffness(shipped_ref, j) is None]
    assert not missing, f"取不到下垂點重力剛度的關節：{missing}"


# ════════════════════════════════════════════════════════ 9. shm_io 的寫入語意
# ⚠️ 不開真的 /dev/shm。用一個只有 `mm`（bytearray）的假物件，
#    並把 shm_io._F8 換成會記錄 (位移, 值) 的代理，這樣**寫入順序**也看得見。
#
# 為什麼一定要驗順序（不是只驗最終值）：
#   joint_shm_controller 以 1 kHz 讀取，而我們的 5 個 8-byte 寫入不是原子的。
#   撕裂讀取可能拿到「一部分新、一部分舊」的中間態。
#   damp_only 的安全論證是：kp/effort/velocity 先歸零、kd 最後寫 →
#   最壞的中間態是「舊 kd + 新的零 kp」＝仍然無位置控制。
#   若順序反過來（先寫 kd 再歸零 kp），中間態會是「新 kd + 舊 kp」——
#   那還在做位置控制，正是中止時最不該出現的狀態。
#   最終值一樣、順序不同 → 只驗最終值的測試完全抓不到這件事。


class _RecordingStruct:
    """代理 struct.Struct，記錄每一次 pack_into 的 (位移, 值)。"""

    def __init__(self, real, log):
        self._real, self._log = real, log

    def pack_into(self, buf, off, val):
        self._log.append((off, val))
        self._real.pack_into(buf, off, val)

    def unpack_from(self, buf, off):
        return self._real.unpack_from(buf, off)


class _FakeShm:
    """只有 mmap 緩衝的假 Shm。借用真的 _cmd_field_off，位移邏輯才是被測的那份。"""

    _cmd_field_off = shm_io.Shm._cmd_field_off

    def __init__(self):
        self.mm = bytearray(shm_io.SIZE)

    damp_only = shm_io.Shm.damp_only
    zero_gains = shm_io.Shm.zero_gains
    write_cmd = shm_io.Shm.write_cmd


@pytest.fixture
def fake_shm(monkeypatch):
    log: list[tuple[int, float]] = []
    monkeypatch.setattr(shm_io, "_F8", _RecordingStruct(struct.Struct("<d"), log))
    return _FakeShm(), log


def _field_off(idx: int, field: str) -> int:
    """獨立重算一次欄位位移（不呼叫被測的 _cmd_field_off）—— 對照量。"""
    return (shm_io.BASE + idx * shm_io.CMD_STRIDE + shm_io.DATA_OFF
            + shm_io.CMD_FIELDS.index(field) * 8)


@pytest.mark.parametrize("idx", [0, 1, 5, 15])
@pytest.mark.parametrize("field", shm_io.CMD_FIELDS)
def test_cmd_field_offset_formula(idx, field):
    """先釘住位移計算本身。位移算錯的話，下面所有『寫了什麼』都是寫到別的關節去。"""
    assert shm_io.Shm._cmd_field_off(None, idx, field) == _field_off(idx, field)


def test_cmd_field_offsets_do_not_overlap_records():
    """相鄰兩筆記錄的欄位不可重疊，且 tick(u64) 不會被 payload 蓋掉。"""
    last = max(_field_off(0, f) for f in shm_io.CMD_FIELDS) + 8
    assert last <= shm_io.BASE + shm_io.CMD_STRIDE
    assert shm_io.DATA_OFF >= shm_io.NAME_OFF + shm_io.NAME_LEN
    assert shm_io.TICK_OFF + 8 <= shm_io.NAME_OFF


@pytest.mark.parametrize("idx", [0, 7, 15])
def test_damp_only_writes_pure_damping(idx, fake_shm):
    shm, log = fake_shm
    shm.damp_only(idx, 3.0)

    def read(field):
        return struct.unpack_from("<d", shm.mm, _field_off(idx, field))[0]

    assert read("kp") == 0.0
    assert read("effort") == 0.0
    assert read("velocity") == 0.0
    assert read("kd") == 3.0


def test_damp_only_writes_kd_last(fake_shm):
    """★ 順序有安全意義（理由見本節開頭的長註解）。"""
    shm, log = fake_shm
    shm.damp_only(4, 3.0)
    off2field = {_field_off(4, f): f for f in shm_io.CMD_FIELDS}
    order = [off2field[off] for off, _ in log]
    assert order[-1] == "kd", f"kd 不是最後寫的：{order}"
    assert set(order[:-1]) == {"kp", "effort", "velocity"}
    assert order.index("kp") < order.index("kd")


def test_damp_only_never_touches_position(fake_shm):
    """position 不能碰：中止時腿的目標角應該維持在最後一次規劃值，
    而且 kp=0 之後它本來就不影響出力，重寫只會多一次撕裂機會。"""
    shm, log = fake_shm
    struct.pack_into("<d", shm.mm, _field_off(2, "position"), 1.2345)
    log.clear()
    shm.damp_only(2, 3.0)
    assert struct.unpack_from("<d", shm.mm, _field_off(2, "position"))[0] == 1.2345
    assert _field_off(2, "position") not in {off for off, _ in log}


def test_damp_only_only_touches_its_own_record(fake_shm):
    """只能動到指定關節那一筆 —— 位移算錯會把阻尼寫到別條腿上。"""
    shm, log = fake_shm
    shm.damp_only(3, 3.0)
    lo = shm_io.BASE + 3 * shm_io.CMD_STRIDE
    assert all(lo <= off < lo + shm_io.CMD_STRIDE for off, _ in log)


def test_zero_gains_writes_gains_first(fake_shm):
    """對照：zero_gains 的順序**故意與 damp_only / write_cmd 相反**（先增益後目標），
    因為它的目的就是讓出力盡快變 0。兩者都驗，順序的意圖才釘得住。"""
    shm, log = fake_shm
    shm.zero_gains(6)
    off2field = {_field_off(6, f): f for f in shm_io.CMD_FIELDS}
    order = [off2field[off] for off, _ in log]
    assert order[0] == "kp"
    assert set(order) == {"kp", "kd", "effort", "velocity"}
    assert all(v == 0.0 for _, v in log)


def test_write_cmd_writes_targets_before_gains(fake_shm):
    """write_cmd 的順序是「先目標值、後增益」：撕裂讀取最壞拿到
    『舊增益 + 新目標』，而舊增益是上一輪確認過的值。"""
    shm, log = fake_shm
    shm.writable = True
    shm.write_cmd(1, position=0.5, velocity=0.0, effort=0.0, kp=10.0, kd=1.0)
    off2field = {_field_off(1, f): f for f in shm_io.CMD_FIELDS}
    order = [off2field[off] for off, _ in log]
    assert order == ["position", "velocity", "effort", "kp", "kd"]


def test_write_cmd_refuses_readonly_handle(fake_shm):
    shm, _ = fake_shm
    shm.writable = False
    with pytest.raises(RuntimeError):
        shm.write_cmd(0, position=0.0)


def test_write_cmd_skips_none_fields(fake_shm):
    """None 表示「這個欄位不要動」。若哪天變成寫 0，中止路徑會意外歸零增益。"""
    shm, log = fake_shm
    shm.writable = True
    shm.write_cmd(0, position=0.25)
    assert [off for off, _ in log] == [_field_off(0, "position")]


# ════════════════════════════════════════════════════════ 附：關節表的健全性

def test_joint_table_shapes():
    assert len(shm_io.JOINTS) == 16
    assert len(set(shm_io.JOINTS)) == 16
    assert set(shm_io.WHEELS) == {lg + coord.KIND_WHEEL for lg in coord.LEGS}
    assert set(shm_io.JOINTS) == set(LEG_JOINTS) | set(shm_io.WHEELS)


@pytest.mark.parametrize("joint", shm_io.JOINTS)
def test_idx_of_matches_table_order(joint):
    assert shm_io.JOINTS[shm_io.idx_of(joint)] == joint


def test_idx_of_rejects_unknown():
    with pytest.raises(SystemExit):
        shm_io.idx_of("fl9_nope")
