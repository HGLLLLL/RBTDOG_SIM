#!/usr/bin/env python3
"""控制器座標系 ↔ 馬達座標系的換算，以及各關節的機構限位。

★ 這裡是換算式與限位的**單一事實來源**。M5 以後的腳本都從這裡取，不得各自寫死。
   （M4_pose_capture.py 有一份自己的副本 —— 它是「驗證換算式」的工具，先於本檔存在。
     `tests/test_m5_leg_pose.py` 會逐項比對兩邊，任何一邊改動而另一邊沒跟上，測試會失敗。
     `inference/hang_rehearsal.py` 也有一份，理由是狗上不能 import numpy、
     而預演跑在本機需要 numpy —— 同一個測試檔一起比對三份。）

換算式（2026-08-25 實機四姿勢驗證通過，見 `docs/座標換算式驗證結果_2026-08-25.md`）：

    馬達角 = side_sign × 控制器角 + offset
    控制器角 = (馬達角 − offset) / side_sign

而且 **控制器座標系 == MJCF 座標系**（同一份文件 §7 驗證），
所以 MJCF 的 `range` 可以直接當控制器角的限位用，不需要再換算。

⚠️ 純標準函式庫。這個檔案要跑在狗上，狗上沒有 numpy。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 腿名對應
# SHM 的腿序：fl(左前) fr(右前) bl(左後) br(右後)
# 設定檔的腿序：FR FL RR RL      ← **不一樣**，這是 task7 反覆踩到的坑
# MJCF 的機構代號：FAR(右前) FBL(左前) RAR(右後) RBL(左後)
#
# MJCF 代號的判讀依據是 body pos（不是猜的）：
#   FAR_ABAD_LINK pos="0.2698 -0.065 0"  → x>0 前、y<0 右 → 右前
#   FBL_ABAD_LINK pos="0.2698  0.065 0"  → x>0 前、y>0 左 → 左前
#   RAR_ABAD_LINK pos="-0.2698 -0.065 0" → 右後
#   RBL_ABAD_LINK pos="-0.2698  0.065 0" → 左後
SHM2MJCF_LEG = {"fl": "FBL", "fr": "FAR", "bl": "RBL", "br": "RAR"}
FRONT_LEGS = ("fl", "fr")
REAR_LEGS = ("bl", "br")
LEGS = ("fl", "fr", "bl", "br")

# 關節種類後綴（SHM 名 = 腿名 + 這個）
KIND_HIP_ROLL = "1_hip_roll"
KIND_HIP_PITCH = "2_hip_pitch"
KIND_KNEE = "3_knee_pitch"
KIND_WHEEL = "4_foot"
LEG_KINDS = (KIND_HIP_ROLL, KIND_HIP_PITCH, KIND_KNEE)   # 不含輪
ALL_KINDS = LEG_KINDS + (KIND_WHEEL,)

# ---------------------------------------------------------------- sign / offset
# 取自實機 /opt/export/config/zg_wheels-user-parameters.yaml
# （與 MATRiX 發布包逐字元相同）。原檔腿序 FR, FL, RR, RL —— 這裡已轉成 SHM 腿名為 key。
_CFG_ORDER = ("FR", "FL", "RR", "RL")
_CFG2SHM = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}


def _by_leg(vals):
    return {_CFG2SHM[c]: v for c, v in zip(_CFG_ORDER, vals)}


SIGN = {
    KIND_HIP_ROLL:  _by_leg([-1.0, -1.0, 1.0, 1.0]),
    KIND_HIP_PITCH: _by_leg([1.0, -1.0, 1.0, -1.0]),
    KIND_KNEE:      _by_leg([-1.0, 1.0, -1.0, 1.0]),
    KIND_WHEEL:     _by_leg([1.0, -1.0, 1.0, -1.0]),
}
OFFSET = {
    KIND_HIP_ROLL:  _by_leg([0.523, -0.523, -0.523, 0.523]),
    KIND_HIP_PITCH: _by_leg([-2.443, 2.443, 2.443, -2.443]),
    KIND_KNEE:      _by_leg([-2.803, 2.803, 2.803, -2.803]),
    KIND_WHEEL:     _by_leg([0.0, 0.0, 0.0, 0.0]),
}

# ---------------------------------------------------------------- 機構限位
# 直接抄自官方 MJCF `model/zgws/zgws.xml` 的 <joint range>，控制器座標系。
#
# ★★ 注意 ABAD（hip_roll）：行程只有 1.22 rad 而且**左右不對稱**，
#    而它的 offset 正好是 ±0.523 —— 也就是「控制器角 0」就緊貼在行程的一端附近。
#    微動測試不要挑這個關節，方向也不能亂選。
#
# ⚠️ knee 用 **±2.801**（URDF 值），不是 MJCF 原本的 ±2.791。
#    2026-08-25 實機癱平時膝頂在 ±2.80 的機械停點，與 URDF 吻合 → MJCF 原本緊了 0.01 rad。
#    2026-08-26 這 0.01 開始擋住實際操作：狗趴著時膝就在 ±2.80，M7 的起點限位檢查
#    永遠不過。已同步把 `model/zgws/zgws.xml` 的四個 KNEE range 改成 ±2.801。
#    ⚠️ 實測最遠到過 ±2.8021（編碼器誤差或停點本身有彈性），所以**起點**的檢查
#    不能當硬性阻擋 —— 那是狗實際所在的位置，不是我們命令的值。
LIMITS = {
    "fr" + KIND_HIP_ROLL:  (-0.697, 0.523),    # FAR_ABAD
    "fl" + KIND_HIP_ROLL:  (-0.523, 0.697),    # FBL_ABAD
    "br" + KIND_HIP_ROLL:  (-0.697, 0.523),    # RAR_ABAD
    "bl" + KIND_HIP_ROLL:  (-0.523, 0.697),    # RBL_ABAD
    "fr" + KIND_HIP_PITCH: (-2.442, 2.791),    # FAR_HIP
    "fl" + KIND_HIP_PITCH: (-2.442, 2.791),    # FBL_HIP
    "br" + KIND_HIP_PITCH: (-2.791, 2.442),    # RAR_HIP  ★ 後腿是反過來的
    "bl" + KIND_HIP_PITCH: (-2.791, 2.442),    # RBL_HIP
    "fr" + KIND_KNEE:      (-2.801, 2.801),
    "fl" + KIND_KNEE:      (-2.801, 2.801),
    "br" + KIND_KNEE:      (-2.801, 2.801),
    "bl" + KIND_KNEE:      (-2.801, 2.801),
}

# ---------------------------------------------------------------- 姿勢
# 控制器座標系。★ 前後腿 hip/knee 反號 —— 站姿是 X 型，四腿共用一組會做出怪東西。
# 數值出處：原廠運控設定檔；`stand` / `liedown` 這兩組只在 MATRiX 模擬版那份裡，
# 但 2026-08-25 實機擷取確認遙控器用的就是它們。
#
# ⚠️ 這是「後腿往前彎」的預設膝模式。切到「後腿往後彎」等於把後兩腿的
#    hip/knee 翻號（見 flip_rear_knee_mode()）。
def _pose(hip: float, knee: float, roll: float = 0.0) -> dict:
    """由「前腿的 hip/knee」生成四腿姿勢；後腿自動反號。"""
    out = {}
    for leg in FRONT_LEGS:
        out[leg + KIND_HIP_ROLL] = roll
        out[leg + KIND_HIP_PITCH] = hip
        out[leg + KIND_KNEE] = knee
    for leg in REAR_LEGS:
        out[leg + KIND_HIP_ROLL] = roll
        out[leg + KIND_HIP_PITCH] = -hip
        out[leg + KIND_KNEE] = -knee
    return out


POSES = {
    "stand":  _pose(0.6, -1.2),    # 遙控器的站立。機身離地約 0.535 m
    "home":   _pose(0.8, -1.5),    # RL 名目站姿，對上原廠 body_height 0.48
    "crouch": _pose(1.4, -2.4),    # 匍匐 / liedown。機身離地約 0.29 m
}


def flip_rear_knee_mode(pose: dict) -> dict:
    """切換膝模式：只翻後兩腿的 hip/knee 號，前腿不動。

    2026-08-25 實機驗證：`stand_knee_back` 的後腿 hip/knee 與前腿**同號**，
    前腿兩種模式差 < 0.03 rad。
    """
    out = dict(pose)
    for leg in REAR_LEGS:
        for kind in (KIND_HIP_PITCH, KIND_KNEE):
            out[leg + kind] = -out[leg + kind]
    return out


# ---------------------------------------------------------------- 換算
def _split(joint: str) -> tuple[str, str]:
    """把 SHM 關節名拆成 (腿名, 種類)。例：'fl2_hip_pitch' → ('fl', '2_hip_pitch')"""
    leg, kind = joint[:2], joint[2:]
    if leg not in LEGS or kind not in ALL_KINDS:
        raise ValueError(f"不認得的關節名 {joint!r}")
    return leg, kind


def to_motor(joint: str, ctrl_angle: float) -> float:
    """控制器角 → 馬達角（寫進 joint_cmd 用的那個座標系）。"""
    leg, kind = _split(joint)
    return SIGN[kind][leg] * ctrl_angle + OFFSET[kind][leg]


def to_ctrl(joint: str, motor_angle: float) -> float:
    """馬達角（joint_state 讀到的） → 控制器角（== MJCF 角）。"""
    leg, kind = _split(joint)
    return (motor_angle - OFFSET[kind][leg]) / SIGN[kind][leg]


def limits_of(joint: str) -> tuple[float, float]:
    """控制器座標系的 (下限, 上限)。輪關節沒有限位 → 回傳無限大。"""
    if joint.endswith(KIND_WHEEL):
        return (float("-inf"), float("inf"))
    return LIMITS[joint]


def check_limit(joint: str, ctrl_angle: float, margin: float = 0.0) -> str:
    """回傳空字串表示 OK，否則回傳說明字串。margin 是往內縮的安全餘裕（rad）。"""
    lo, hi = limits_of(joint)
    if lo == float("-inf"):
        return ""
    if ctrl_angle < lo + margin:
        return f"{ctrl_angle:+.4f} 低於下限 {lo:+.4f}（餘裕 {margin}）"
    if ctrl_angle > hi - margin:
        return f"{ctrl_angle:+.4f} 高於上限 {hi:+.4f}（餘裕 {margin}）"
    return ""


def pose_to_motor(pose: dict) -> dict:
    """整組姿勢：控制器角 → 馬達角。"""
    return {j: to_motor(j, a) for j, a in pose.items()}
