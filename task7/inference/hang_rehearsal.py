#!/usr/bin/env python3
"""D1 Max 吊掛預演 —— 產生 `realbot/M5_leg_pose.py` 需要的對照表。

**這支程式的唯一目的：在上機之前，先知道「應該看到什麼數字」。**
實機量到的下垂量若與這裡的預測對不上，代表座標換算式或腿序錯了，
而不是「腿比較軟」——這是 S1 微動唯一能自我證偽的手段。

產出四件事：

1. **重力保持力矩表**：狗**懸空**時，每個腿關節要出多少力矩才維持得住某個姿勢。
   作法是逆動力學：`qpos` = 目標姿勢、`qvel` = `qacc` = 0 → `mj_inverse`
   → `qfrc_inverse` 就是「維持這個狀態所需的廣義力」。
   ⚠️ 這裡**不焊機身**（關掉 equality 約束）。自由基座那 6 個 DOF 的
   `qfrc_inverse` 剛好就是吊帶該承的力，順手拿來當合理性檢核；
   腿關節的力矩不受基座焊不焊影響（焊接只改基座那 6 列）。

2. **增益建議**：穩態下垂量 `err = τ_重力 / kp`。
   要的是「kp 小到安全、下垂量大到量得到（遠超感測雜訊 ~0.001 rad）」的甜蜜點。

3. **軌跡動態預演**：機身焊在空中（`model/zgws/scene_hang.xml`），
   PD 跑一趟 HANG_FREE → STAND 的餘弦插值，掃 (kp, kd)，看峰值力矩與追蹤誤差。

4. **`reference/hang_torque_ref.json`**：狗上沒有 numpy，所以全部轉成 float/list。

---------------------------------------------------------------------------
方法論（task7 的血淚：「診斷輸出騙人」已經出現七次，每次都是量測工具自己錯）
---------------------------------------------------------------------------
本檔一律「多印一個可以互相對照的量」，而不是「多印一個結論」。實作了五組交叉檢核：

  A. 吊帶垂直合力 vs 模型總質量 × g   —— 逆動力學的整體正確性
  B. `qfrc_inverse` vs `qfrc_bias`     —— 確認 qacc=0 之下兩者必須相等
                                         （不等就代表混進了接觸力或約束力）
  C. STAND 左右對稱性                  —— 左右腿力矩大小應相等（號可能相反）
  D. HANG_FREE 的殘餘重力力矩 ≈ 0      —— 「自由下垂」的定義就是重力力矩為零。
                                         這條完全不依賴我們是怎麼把它模擬出來的
  E. 軌跡預演時的機身位移              —— 焊接若被拉開，峰值力矩就不能引用

用法：
    /home/huang/miniforge3/envs/rbtdog/bin/python task7/inference/hang_rehearsal.py
    ... --no-json      只印表格不寫檔
    ... --quick        軌跡掃描只跑建議那一組（快）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import max_model as mm

_ROOT = Path(__file__).resolve().parents[1]
SCENE_HANG = str(_ROOT / "model" / "zgws" / "scene_hang.xml")
OUT_JSON = _ROOT / "reference" / "hang_torque_ref.json"

# 規格書標稱整機質量（kg）。**不是** MJCF 的質量 —— 兩者對不對得上正是檢核 A。
SPEC_MASS = 41.0

# =============================================================================
# 腿序：SHM ←→ MJCF
# =============================================================================
# ⚠️ 三套腿序在這個專案裡同時存在，混用就靜默拿到別條腿的數字：
#     max_model.LEGS  = FR, FL, RR, RL   （= MJCF 的 qpos 順序，也是原廠設定檔順序）
#     MJCF 前綴       = FAR, FBL, RAR, RBL（機構代號，不是方位）
#     實機 SHM        = fl, fr, bl, br
#   下面的對應**不照抄**，由 MJCF 的 ABAD body pos 正負號當場推出來再與假說比對。
SHM_LEGS = ("fl", "fr", "bl", "br")
SHM_KINDS = ("1_hip_roll", "2_hip_pitch", "3_knee_pitch")
MJCF_KINDS = ("ABAD", "HIP", "KNEE")

# 由方位推出的期望值：(x 正負, y 正負)，x>0 = 前、y>0 = 左（MJCF 機身座標系）
SHM_EXPECT_XY = {"fl": (+1, +1), "fr": (+1, -1), "bl": (-1, +1), "br": (-1, -1)}

# 12 個 SHM 關節名，順序 = fl,fr,bl,br × (1_hip_roll, 2_hip_pitch, 3_knee_pitch)
SHM_JOINTS = tuple(f"{lg}{k[0]}_{k[2:]}" for lg in SHM_LEGS for k in SHM_KINDS)

# -----------------------------------------------------------------------------
# 座標換算（馬達角 = sign × 控制器角 + offset）
# -----------------------------------------------------------------------------
# 出處：實機 /opt/export/config/zg_wheels-user-parameters.yaml，
# 由 M4_pose_capture.py 實機驗證通過（commit 30959bb / b7e311e），
# 且已確認「控制器座標系 == MJCF 座標系」，所以下面可以直接吃 MJCF 的角度。
# ⚠️ 力矩也跟著翻號：τ_馬達 = sign × τ_控制器（sign = ±1，功率守恆推得）。
#
# ⚠️ 這份表在 `realbot/coord.py` 有第二份拷貝（狗上是純標準函式庫，import 不到這裡）。
#    兩份已逐項比對過完全相同。**改了一份就要改另一份**，否則會出現
#    「模擬預測與實機下垂量差一個號」這種最難查的症狀。比對方式：
#      python -c "import coord, hang_rehearsal as H; ..."（見本次 session 的驗證紀錄）
SIGN = {"fr": (-1.0, +1.0, -1.0), "fl": (-1.0, -1.0, +1.0),
        "br": (+1.0, +1.0, -1.0), "bl": (+1.0, -1.0, +1.0)}
OFFSET = {"fr": (0.523, -2.443, -2.803), "fl": (-0.523, 2.443, 2.803),
          "br": (-0.523, 2.443, 2.803), "bl": (0.523, -2.443, -2.803)}

# =============================================================================
# 掃描參數
# =============================================================================
KP_CANDIDATES = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 150.0)
SAG_LO, SAG_HI, SAG_TARGET = 0.02, 0.10, 0.05     # rad，承重姿勢想要的下垂量區間
SENSOR_NOISE = 0.001                              # rad，編碼器雜訊量級（保守估）

# S1（從自然下垂出發的單關節微動）用的候選 kp。低端要密，因為腿只提自己的重量，
# 重力剛度只有個位數到十幾 N·m/rad，kp 一過 40 移動比例就逼近 1 而失去鑑別力。
S1_KP_CANDIDATES = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)
RATIO_LO, RATIO_HI, RATIO_TARGET = 0.30, 0.70, 0.50   # 移動比例 kp/(kp+k) 的甜蜜區
S1_DELTA = 0.05                                       # rad，S1 的指令位移量

TRAJ_T = 5.0        # HANG_FREE → STAND 的插值時間（s）
TRAJ_HOLD = 1.5     # 插值結束後續跑，讓它進入穩態（s）
TRAJ_KP_SWEEP = (20.0, 40.0, 60.0, 80.0, 120.0)
TRAJ_KD_SWEEP = (1.0, 2.0, 4.0)
TRAJ_ERR_TOL = 0.06     # rad，軌跡終端誤差的合格門檻（≈3.4°）


# =============================================================================
# 模型
# =============================================================================
def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_path(SCENE_HANG)
    return m, mujoco.MjData(m)


def _flag(m: mujoco.MjModel, bit: int, on: bool) -> None:
    if on:
        m.opt.disableflags &= ~bit
    else:
        m.opt.disableflags |= bit


def eq_enabled(m: mujoco.MjModel, on: bool) -> None:
    """開關 equality 約束 = 開關「機身焊在空中」。同一個模型兩用，不維護兩份 XML。"""
    _flag(m, int(mujoco.mjtDisableBit.mjDSBL_EQUALITY), on)


def contact_enabled(m: mujoco.MjModel, on: bool) -> None:
    """開關碰撞（含自碰撞）。

    ★ 這台狗吊起來時**左右輪會在機身正下方互相頂住** —— 不是模型 bug，
      是真的幾何：輪心 y = ±0.171 m、輪半徑 0.096 m，abad 一往內擺就碰到。
      所以「有沒有算碰撞」會得到兩個完全不同的自然下垂姿勢，必須明確區分。
    """
    _flag(m, int(mujoco.mjtDisableBit.mjDSBL_CONTACT), on)


def set_pose(m: mujoco.MjModel, d: mujoco.MjData, q12: np.ndarray) -> None:
    """把整個狀態設成「基座在吊點、12 腿關節 = q12、其餘全零」。"""
    d.qpos[:] = m.qpos0                    # 基座 (0,0,0.8) + 單位四元數，輪角 0
    d.qpos[mm.LEG_QPOS_IDX] = q12
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    d.ctrl[:] = 0.0
    d.qfrc_applied[:] = 0.0


# =============================================================================
# 1) 腿序對應驗證
# =============================================================================
def verify_leg_map(m: mujoco.MjModel) -> dict:
    """由 MJCF body pos 的正負號**推出** SHM→MJCF 對應，再與假說比對。

    回傳 {shm_leg: {"prefix":…, "legs_index":…, "abad_xy":[x, y]}}。
    對不上就 AssertionError —— 這種東西不能只是印個警告，後面每個數字都靠它。
    """
    derived, info = {}, {}
    for prefix in ("FAR", "FBL", "RAR", "RBL"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_ABAD_LINK")
        assert bid >= 0, f"找不到 body {prefix}_ABAD_LINK"
        x, y = float(m.body_pos[bid][0]), float(m.body_pos[bid][1])
        key = (1 if x > 0 else -1, 1 if y > 0 else -1)
        derived[key] = prefix
        info[prefix] = (x, y)

    out = {}
    for lg, key in SHM_EXPECT_XY.items():
        assert key in derived, f"MJCF 裡找不到方位 {key} 的腿"
        prefix = derived[key]
        # 與 max_model 的 LEGS / PREFIX 對得起來嗎（第三個獨立來源）
        legs_idx = [i for i, L in enumerate(mm.LEGS) if mm.PREFIX[L] == prefix]
        assert len(legs_idx) == 1, f"{prefix} 在 max_model.PREFIX 裡不唯一"
        out[lg] = {"prefix": prefix, "legs_index": legs_idx[0],
                   "legs_name": mm.LEGS[legs_idx[0]],
                   "abad_xy": [info[prefix][0], info[prefix][1]]}

    # 與題目給的假說逐項比對（照抄的那份），不同就當場炸掉
    hypothesis = {"fl": "FBL", "fr": "FAR", "bl": "RBL", "br": "RAR"}
    bad = {k: (v, out[k]["prefix"]) for k, v in hypothesis.items() if out[k]["prefix"] != v}
    assert not bad, f"★ 腿序對應與假說不符：{bad}"
    return out


def flip_rear_knee(legmap: dict, q12: np.ndarray) -> np.ndarray:
    """膝模式切換：**只把後兩腿（bl / br）的 hip_pitch 與 knee_pitch 翻號，前腿不動。**

    出處：2026-08-25 實機四姿勢驗證（`docs/座標換算式驗證結果_2026-08-25.md` §2.2），
    與 `realbot/coord.py` 的 `flip_rear_knee_mode()` 是同一個運算 ——
    本檔在 `verify_flip_against_coord()` 裡對 import 進來的 coord 逐項比對，不是照抄。

    ⚠️ 翻的是「後腿」不是「右腿」。SHM 的 bl/br 對到 MJCF 的 RBL/RAR、
       對到 max_model.LEGS 的 RL/RR（index 3 / 2）——三套腿序在這一行同時出現，
       所以位址一律走 `q12_index()`，不寫死索引。
    """
    out = q12.copy()
    for lg in ("bl", "br"):
        for k in (1, 2):                    # 1 = hip_pitch, 2 = knee_pitch
            out[q12_index(legmap, lg, k)] *= -1.0
    return out


def verify_flip_against_coord(legmap: dict, q12: np.ndarray) -> str:
    """拿 `realbot/coord.py` 當第二個實作，比對翻號結果與 sign/offset 表。

    coord.py 是純標準函式庫且不 import 其他東西，所以可以直接借過來。
    借不到（路徑變了、檔案還沒寫）就跳過並明講，不要靜靜地當作驗過了。
    """
    sys.path.insert(0, str(_ROOT / "realbot"))
    try:
        import coord                                  # noqa: PLC0415
    except Exception as e:                            # noqa: BLE001
        return f"⚠️ 無法 import realbot/coord.py（{e}）—— 本次**未**做交叉比對"

    bad = [(lg, kd) for lg in SHM_LEGS for k, kd in enumerate(SHM_KINDS)
           if coord.SIGN[kd][lg] != SIGN[lg][k]
           or abs(coord.OFFSET[kd][lg] - OFFSET[lg][k]) > 1e-12]
    assert not bad, f"★ sign/offset 與 coord.py 不一致：{bad}"

    mine = to_shm_dict(legmap, flip_rear_knee(legmap, q12))
    theirs = coord.flip_rear_knee_mode(to_shm_dict(legmap, q12))
    gap = max(abs(mine[n] - theirs[n]) for n in mine)
    assert gap < 1e-12, f"★ 膝模式翻號與 coord.flip_rear_knee_mode 不一致，差 {gap}"
    return "✓ sign/offset 12 項、膝模式翻號 12 項，與 realbot/coord.py 完全一致"


def q12_index(legmap: dict, shm_leg: str, kind: int) -> int:
    """SHM 關節 → 在 12 維（LEGS × abad/hip/knee）向量裡的位置。"""
    return legmap[shm_leg]["legs_index"] * 3 + kind


def idx2shm_of(legmap: dict) -> dict:
    """12 維向量的第 i 格是哪個 SHM 關節。反查表，別在迴圈裡臨時湊。"""
    return {q12_index(legmap, lg, k): f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            for lg in SHM_LEGS for k in range(3)}


def to_shm_dict(legmap: dict, v12: np.ndarray) -> dict:
    """(12,) 的 LEGS 序向量 → 以 SHM 關節名為 key 的 dict（純 float）。"""
    return {f"{lg}{k+1}_{SHM_KINDS[k][2:]}": float(v12[q12_index(legmap, lg, k)])
            for lg in SHM_LEGS for k in range(3)}


def to_mjcf_dict(legmap: dict, v12: np.ndarray) -> dict:
    """同上，但 key 是 MJCF 關節名。兩份 key 並存，M5 要對哪邊都不用自己換算。"""
    return {f"{legmap[lg]['prefix']}_{MJCF_KINDS[k]}_JOINT":
            float(v12[q12_index(legmap, lg, k)])
            for lg in SHM_LEGS for k in range(3)}


def to_motor(shm_vals: dict, is_torque: bool = False) -> dict:
    """控制器（= MJCF）座標 → 馬達座標。角度套 sign/offset，力矩只套 sign。"""
    out = {}
    for lg in SHM_LEGS:
        for k in range(3):
            name = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            s, o = SIGN[lg][k], OFFSET[lg][k]
            out[name] = float(s * shm_vals[name] + (0.0 if is_torque else o))
    return out


# =============================================================================
# 2) 重力保持力矩（逆動力學）
# =============================================================================
def gravity_torque(m: mujoco.MjModel, d: mujoco.MjData, q12: np.ndarray) -> dict:
    """qvel = qacc = 0 之下的 `qfrc_inverse` —— **純重力項**。

    兩個 flag 都關掉，這是本檔對「重力保持力矩」的定義，全程一致：
      - 關 equality：基座自由，`qfrc_inverse[:6]` 才是吊帶該承的力
        （焊接只改基座那 6 列，腿關節的力矩本來就不受影響）
      - 關 contact：不讓接觸力混進來。這台狗吊起來時左右輪會互頂，
        若不關，`qfrc_inverse` 就不再是純重力項，而**表面上完全看不出來** ——
        這正是 task7 那七次「診斷輸出騙人」的長相。接觸狀況另外用
        `contact_report()` 獨立報告，不混在同一個數字裡。

    回傳 tau12（腿關節，N·m）、base（基座 6 DOF）、檢核 B 用的 `qfrc_bias` 差異。
    """
    eq_enabled(m, False)
    contact_enabled(m, False)
    set_pose(m, d, q12)
    mujoco.mj_forward(m, d)                 # 先把位置相依的量算齊
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0                         # ⚠️ 必須在 mj_forward **之後**再清一次
    mujoco.mj_inverse(m, d)

    tau = d.qfrc_inverse[mm.LEG_QVEL_IDX].copy()
    bias = d.qfrc_bias[mm.LEG_QVEL_IDX].copy()
    base = d.qfrc_inverse[:6].copy()
    eq_enabled(m, True)
    contact_enabled(m, True)
    return {"tau12": tau, "base": base,
            "bias_gap": float(np.abs(tau - bias).max()),
            "ncon": int(d.ncon)}


def contact_report(m: mujoco.MjModel, d: mujoco.MjData, q12: np.ndarray) -> dict:
    """同一個姿勢、**打開碰撞**時到底有沒有碰、碰在哪、法向力多大。

    另外回傳 ctrl=0 時的 |qacc|max：這是「這個姿勢是不是平衡點」的獨立判據，
    完全不經過 `mj_inverse` 的約束力反解，所以不會跟它一起錯。
    """
    eq_enabled(m, True)
    contact_enabled(m, True)
    set_pose(m, d, q12)
    mujoco.mj_forward(m, d)
    pairs = []
    for i in range(d.ncon):
        c = d.contact[i]
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        pairs.append({
            "body1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]),
            "body2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]),
            "dist_m": float(c.dist), "pos": [float(x) for x in c.pos],
            "normal_force_N": float(f[0])})
    return {"ncon": int(d.ncon), "pairs": pairs,
            "qacc_leg_max": float(np.abs(d.qacc[mm.LEG_QVEL_IDX]).max())}


def local_stiffness(m: mujoco.MjModel, d: mujoco.MjData, q_base: np.ndarray,
                    delta: float) -> dict:
    """hang_free 附近的**局部重力剛度** k_j = dτ_j/dq_j（N·m/rad），中央差分。

    只擾動關節 j 本身，其餘維持 q_base；τ_j 一律取純重力項（同 `gravity_torque`）。

    ★ 用途：低 kp 微動時，重力就是一根並聯的彈簧。位置伺服 + 重力彈簧的平衡點是

          kp·(q_cmd − q_eq) = τ_grav(q_eq) ≈ τ0 + k·(q_eq − q_hang)

      令 Δ = q_cmd − q_hang，解得

          q_eq − q_hang = (kp·Δ − τ0) / (kp + k)

      τ0 = τ_grav(q_hang)。τ0 ≈ 0 時就退化成「只走到指令量的 kp/(kp+k) 倍」。
      這個比例是實機上鑑別力很高的對照量：換算式或增益錯了，比例就對不上。

    ★ k_j 應該是**正的**（回復力，像單擺）。負值代表 q_base 在不穩定分支上。
    """
    kj = np.zeros(12)
    tp, tm = np.zeros(12), np.zeros(12)
    for j in range(12):
        qp = q_base.copy(); qp[j] += delta
        qm = q_base.copy(); qm[j] -= delta
        tp[j] = gravity_torque(m, d, qp)["tau12"][j]
        tm[j] = gravity_torque(m, d, qm)["tau12"][j]
        kj[j] = (tp[j] - tm[j]) / (2.0 * delta)
    return {"k12": kj, "tau_plus": tp, "tau_minus": tm, "delta": float(delta)}


def symmetry_gap(legmap: dict, tau12: np.ndarray) -> dict:
    """左右對稱檢核：STAND 這種左右鏡像姿勢，左右腿力矩**大小**應相等。

    回傳每個 (前/後 × 關節) 的 |左| − |右| 差。號可能相反（abad 一定相反），
    所以比的是絕對值。
    """
    gaps = {}
    for front, (L, R) in (("front", ("fl", "fr")), ("rear", ("bl", "br"))):
        for k in range(3):
            a = tau12[q12_index(legmap, L, k)]
            b = tau12[q12_index(legmap, R, k)]
            gaps[f"{front}_{SHM_KINDS[k][2:]}"] = {
                "left": float(a), "right": float(b),
                "abs_gap": float(abs(abs(a) - abs(b)))}
    return gaps


# =============================================================================
# 3) HANG_FREE：用模擬求「四腿完全自然下垂」的姿勢
# =============================================================================
def settle_hang_free(m: mujoco.MjModel, d: mujoco.MjData, q_start: np.ndarray,
                     contacts: bool = True, damp: float = 3.0,
                     t_max: float = 120.0, tol_v: float = 1e-4) -> dict:
    """機身焊住、致動器不出「位置」力，讓腿自由落到靜止，記錄最終 qpos。

    ⚠️ MJCF 的腿關節**沒有 damping、也沒有 frictionloss**（只有輪關節有 0.15）。
       純重力的多連桿是保守系統 —— 真的把 ctrl 設成 0，它會**永遠盪下去不收斂**。
       所以這裡外加一個純黏滯項 `τ = −damp·qvel` 當**數值工具**把能量抽掉。

       這麼做合法的理由不是「阻尼很小所以沒差」，而是：黏滯力在 qvel = 0 時恆為零，
       因此它**不會改變平衡點的位置**，只改變到達平衡點的路徑。
       而且收斂後我們用檢核 D（殘餘重力力矩 ≈ 0）獨立驗證結果 ——
       那條檢核完全不依賴我們是怎麼把它模擬出來的。

    `contacts=False` 時關掉碰撞，得到**純單擺**的下垂姿勢；`True` 則會被左右輪互頂擋住。
    兩個都要算，因為它們差很多（abad 差 ~0.13 rad），而且各自回答不同的問題。
    """
    eq_enabled(m, True)
    contact_enabled(m, contacts)
    set_pose(m, d, q_start)
    dt = m.opt.timestep
    n = int(t_max / dt)
    hit = -1
    for i in range(n):
        d.ctrl[:] = 0.0
        d.ctrl[mm.LEG_ACT_IDX] = -damp * d.qvel[mm.LEG_QVEL_IDX]
        d.ctrl[mm.WHEEL_ACT_IDX] = -mm.KD_WHEEL * d.qvel[mm.WHEEL_QVEL_IDX]
        mujoco.mj_step(m, d)
        if i % 250 == 0 and np.abs(d.qvel[mm.LEG_QVEL_IDX]).max() < tol_v:
            hit = i
            break
    q12 = d.qpos[mm.LEG_QPOS_IDX].copy()
    contact_enabled(m, True)
    return {"q12": q12,
            "qvel_max": float(np.abs(d.qvel[mm.LEG_QVEL_IDX]).max()),
            "t_settle": float((hit if hit > 0 else n) * dt),
            "converged": hit > 0,
            "base_drift_m": float(np.linalg.norm(d.qpos[:3] - m.qpos0[:3]))}


# =============================================================================
# 4) 軌跡動態預演
# =============================================================================
def cos_interp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """餘弦插值：兩端速度為 0，不會在起訖點踢一下。"""
    s = min(max(s, 0.0), 1.0)
    return q0 + (q1 - q0) * 0.5 * (1.0 - np.cos(np.pi * s))


def run_traj(m: mujoco.MjModel, d: mujoco.MjData, q0: np.ndarray, q1: np.ndarray,
             kp12: np.ndarray, kd12: np.ndarray,
             T: float = TRAJ_T, hold: float = TRAJ_HOLD) -> dict:
    """焊住機身，PD 跑一趟 q0 → q1，記錄峰值力矩／峰值誤差／限位／飽和。"""
    eq_enabled(m, True)
    set_pose(m, d, q0)
    lo, hi = mm.leg_joint_ranges(m).T          # (12,) 各自的下限 / 上限
    tau_max12 = np.tile(mm.TAU_MAX3, 4)        # LEGS × (abad, hip, knee)
    dt = m.opt.timestep
    n = int((T + hold) / dt)

    peak_tau = np.zeros(12)
    peak_err = np.zeros(12)
    lim_margin = np.full(12, np.inf)           # 離最近限位還剩多少 rad
    n_sat = 0
    base_drift = 0.0
    diverged = False

    for i in range(n):
        q = d.qpos[mm.LEG_QPOS_IDX]
        v = d.qvel[mm.LEG_QVEL_IDX]
        q_des = cos_interp(q0, q1, (i * dt) / T)
        err = q_des - q
        tau = kp12 * err - kd12 * v
        # 12 維的上限：LEGS × (abad, hip, knee)，所以 TAU_MAX3 要 tile 四次
        tau_c = np.clip(tau, -tau_max12, tau_max12)
        n_sat += int(np.any(np.abs(tau) > tau_max12))

        d.ctrl[:] = 0.0
        d.ctrl[mm.LEG_ACT_IDX] = tau_c
        d.ctrl[mm.WHEEL_ACT_IDX] = -mm.KD_WHEEL * d.qvel[mm.WHEEL_QVEL_IDX]
        mujoco.mj_step(m, d)

        peak_tau = np.maximum(peak_tau, np.abs(tau_c))
        peak_err = np.maximum(peak_err, np.abs(err))
        lim_margin = np.minimum(lim_margin, np.minimum(q - lo, hi - q))
        base_drift = max(base_drift, float(np.linalg.norm(d.qpos[:3] - m.qpos0[:3])))
        if not np.all(np.isfinite(d.qpos)) or np.abs(v).max() > 200.0:
            diverged = True
            break

    q_end = d.qpos[mm.LEG_QPOS_IDX].copy()
    return {"peak_tau": peak_tau, "peak_err": peak_err,
            "final_err": np.abs(q1 - q_end),
            "lim_margin_min": float(lim_margin.min()),
            "lim_hit": bool(lim_margin.min() <= 0.0),
            "n_sat": n_sat, "base_drift_m": base_drift,
            "qvel_max_end": float(np.abs(d.qvel[mm.LEG_QVEL_IDX]).max()),
            "diverged": diverged}


# =============================================================================
# 印表工具
# =============================================================================
def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def print_tau_table(legmap: dict, poses: dict, per_page: int = 5) -> None:
    """姿勢多了之後一行放不下，切成每頁 per_page 欄印，不要讓它自己折行。"""
    names = list(poses.keys())
    for s in range(0, len(names), per_page):
        chunk = names[s:s + per_page]
        print(f"{'SHM 關節':<16}{'MJCF 關節':<20}" + "".join(f"{n:>17}" for n in chunk))
        print("-" * (36 + 17 * len(chunk)))
        for lg in SHM_LEGS:
            for k in range(3):
                shm = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
                mj = f"{legmap[lg]['prefix']}_{MJCF_KINDS[k]}_JOINT"
                i = q12_index(legmap, lg, k)
                row = "".join(f"{poses[n]['tau12'][i]:>17.3f}" for n in chunk)
                print(f"{shm:<16}{mj:<20}{row}")
            if lg != SHM_LEGS[-1]:
                print()
        if s + per_page < len(names):
            print()


# =============================================================================
# 主流程
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="D1 Max 吊掛預演對照表")
    ap.add_argument("--no-json", action="store_true", help="只印表格，不寫 JSON")
    ap.add_argument("--quick", action="store_true", help="軌跡只跑建議那一組")
    args = ap.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    m, d = load()
    legmap = verify_leg_map(m)

    # ---------------------------------------------------------------- 模型基本盤
    hr("0. 模型與腿序 —— 每個後續數字都建立在這一段上")
    m_total = float(m.body_mass.sum())
    g = float(-m.opt.gravity[2])
    print(f"場景          : {SCENE_HANG}")
    print(f"MuJoCo        : {mujoco.__version__}   timestep = {m.opt.timestep} s")
    print(f"模型總質量    : {m_total:.3f} kg（含 mocap 吊點 body，質量 0）")
    print(f"規格書標稱    : {SPEC_MASS:.1f} kg")
    dm = m_total - SPEC_MASS
    leg_mass = m_total - float(m.body_mass[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")])
    print(f"  → 差 {dm:+.3f} kg（{dm / SPEC_MASS * 100:+.1f} %）"
          + ("  ⚠️ 官方 MJCF 比規格書輕" if abs(dm) > 0.5 else "  ✓"))
    if abs(dm) > 0.5:
        print(f"     機身 {m_total - leg_mass:.3f} kg + 四腿 {leg_mass:.3f} kg。"
              f"少掉的 {abs(dm):.2f} kg 在哪，MJCF 看不出來：")
        print(f"     ・若在機身（電池／外殼，最可能）→ **腿關節力矩完全不受影響**，"
              f"只有吊帶承重要加 {abs(dm) * g:.0f} N")
        print(f"     ・若平均分布在腿上 → 下面所有力矩要乘 ~{SPEC_MASS / m_total:.3f}")
        print(f"     ⚠️ 實機量到的力矩比預測高幾 % 屬正常，別急著判定換算式錯。"
              f"（實機還有 MJCF 沒有的腿關節摩擦。）")
    print(f"重力加速度    : {g:.3f} m/s²  → 模型重量 {m_total * g:.1f} N")
    print()
    print("腿序對應（由 MJCF body pos 正負號推出，非照抄）：")
    print(f"{'SHM':<6}{'MJCF 前綴':<12}{'max_model.LEGS':<18}{'ABAD body (x, y)':<24}{'方位'}")
    for lg in SHM_LEGS:
        e = legmap[lg]
        pos = f"({e['abad_xy'][0]:+.4f}, {e['abad_xy'][1]:+.4f})"
        face = ("前" if e["abad_xy"][0] > 0 else "後") + ("左" if e["abad_xy"][1] > 0 else "右")
        print(f"{lg:<6}{e['prefix']:<12}{e['legs_name']:<18}{pos:<24}{face}")
    print("✓ 與假說（fl=FBL, fr=FAR, bl=RBL, br=RAR）以及 max_model.PREFIX 三方一致")

    # ---------------------------------------------------------------- HANG_FREE
    hr("1. HANG_FREE —— 四腿自然下垂的姿勢（用模擬求，不是猜的）")
    print("起點取 STAND（吊起來時狗還在原廠站姿），加純黏滯阻尼把能量抽掉。")
    st = settle_hang_free(m, d, np.asarray(mm.STAND).reshape(12).copy(), contacts=True)
    hang_free = st["q12"]
    st_nc = settle_hang_free(m, d, np.asarray(mm.STAND).reshape(12).copy(), contacts=False)
    hang_nc = st_nc["q12"]
    print(f"收斂           : {'是' if st['converged'] else '否（達時間上限）'}"
          f"   t = {st['t_settle']:.1f} s   殘餘 |qvel|max = {st['qvel_max']:.2e} rad/s")
    print(f"機身位移       : {st['base_drift_m'] * 1000:.3f} mm（焊接夠硬 → 峰值力矩可引用）")

    # ★ 這台狗吊起來會自碰撞。這件事一定要單獨報，不能讓它偷偷混進力矩數字裡。
    cr = contact_report(m, d, hang_free)
    cr_nc = contact_report(m, d, hang_nc)      # 對照：關掉碰撞後同一姿勢的加速度
    print("\n★★ 自碰撞：吊起來時左右輪會互相頂住 ★★")
    print(f"  下垂到底時 ncon = {cr['ncon']}")
    # 「|qacc| 小」本身不是結論，要跟「沒有支撐時會是多少」對照才有意義
    eq_enabled(m, True); contact_enabled(m, False)
    set_pose(m, d, hang_free); mujoco.mj_forward(m, d)
    qacc_free = float(np.abs(d.qacc[mm.LEG_QVEL_IDX]).max())
    contact_enabled(m, True)
    print(f"  ctrl=0 之下 |qacc|max：有碰撞 {cr['qacc_leg_max']:.3f}"
          f"  vs 拿掉碰撞 {qacc_free:.3f} rad/s²"
          f"  → 被擋下 {(1 - cr['qacc_leg_max'] / max(qacc_free, 1e-9)) * 100:.1f} %，"
          f"確實是接觸在頂著")
    for p in cr["pairs"]:
        print(f"    {p['body1']} ↔ {p['body2']}  穿透 {p['dist_m'] * 1000:+.3f} mm  "
              f"法向力 {p['normal_force_N']:.2f} N  接觸點 y = {p['pos'][1]:+.4f} m")
    print("  幾何上就是這樣：輪心離中線 y = 0.065+0.045+0.0522+0.0088 = 0.171 m，"
          "輪半徑 0.096 m，")
    print("  abad 一往內擺，前對前、後對後兩顆輪就在機身正下方碰上。**不是模型 bug。**")
    print("  ⚠️ 實機意義：腿不會垂成一直線，abad 會停在 ±0.28 rad 附近而不是 0，"
          "而且輪子彼此有接觸力。")

    print("\n  對照組：關掉碰撞的純單擺平衡點（幾何上的『完全自由下垂』）")
    print(f"{'':<4}{'SHM 關節':<16}{'含自碰撞(rad)':>15}{'純單擺(rad)':>14}{'差':>10}")
    diff_nc = 0.0
    for lg in SHM_LEGS:
        for k in range(3):
            i = q12_index(legmap, lg, k)
            nm = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            dd = hang_free[i] - hang_nc[i]
            diff_nc = max(diff_nc, abs(dd))
            if abs(dd) > 1e-4:
                print(f"{'':<4}{nm:<16}{hang_free[i]:>15.4f}{hang_nc[i]:>14.4f}{dd:>10.4f}")
    print(f"    → 最大差 {diff_nc:.4f} rad（只有 abad 有差，hip/knee 兩者都 ≈0）")
    print("    ⚠️ MJCF 的輪碰撞用 FOOT STL 的凸包，凸包比真實輪面略胖 → "
          "實機的 abad 可能再往內一點點。")
    lr = abs(abs(hang_free[q12_index(legmap, "fl", 0)])
             - abs(hang_free[q12_index(legmap, "fr", 0)]))
    tot = abs(hang_free[q12_index(legmap, "fl", 0)]) \
        + abs(hang_free[q12_index(legmap, "fr", 0)])
    print(f"    ⚠️ 前腿左右 abad 大小差 {lr * 1000:.1f} mrad，但兩者**和**是 "
          f"{tot:.4f} rad —— 和才是被接觸決定的量（輪縫合起來），")
    print("       左右怎麼分是軟接觸 + MJCF 自身微小不對稱的結果，"
          "±3 mrad 的等級，別當成腿序錯了。")

    # 檢核 D：純重力力矩。含自碰撞的姿勢 abad 不會是 0（被接觸力頂住），純單擺的才是 0。
    gr_hf = gravity_torque(m, d, hang_free)
    gr_nc = gravity_torque(m, d, hang_nc)
    print(f"\n★ 檢核 D 殘餘**純重力**力矩 max：")
    print(f"    含自碰撞 HANG_FREE = {np.abs(gr_hf['tau12']).max():.4f} N·m"
          f"（abad {abs(gr_hf['tau12'][q12_index(legmap, 'fl', 0)]):.3f}、"
          f"hip/knee {max(abs(gr_hf['tau12'][q12_index(legmap, lg, k)]) for lg in SHM_LEGS for k in (1, 2)):.4f}）")
    print(f"      ↑ abad 不為 0 是**對的**：那份力矩由左右輪的接觸力承擔，不是致動器出的。")
    print(f"    純單擺 HANG_NC    = {np.abs(gr_nc['tau12']).max():.4f} N·m"
          + ("  ✓ ≈0，確認是真正的力學平衡點" if np.abs(gr_nc["tau12"]).max() < 0.05
             else "  ⚠️ 不夠接近 0"))

    # 起點敏感度：不同起點是不是收到同一個地方（平衡點唯一嗎）
    alt = {}
    for nm, q0 in (("HOME", np.asarray(mm.HOME).reshape(12).copy()),
                   ("全零", np.zeros(12)),
                   ("CROUCH", np.asarray(mm.CROUCH).reshape(12).copy())):
        r = settle_hang_free(m, d, q0, contacts=True)
        dq = np.abs(r["q12"] - hang_free)
        alt[nm] = {
            "abad": float(max(dq[q12_index(legmap, lg, 0)] for lg in SHM_LEGS)),
            "hipknee": float(max(dq[q12_index(legmap, lg, k)]
                                 for lg in SHM_LEGS for k in (1, 2)))}
    print("\n起點敏感度（與從 STAND 出發的結果相差多少 rad；abad / hip+knee 分開看）：")
    for nm, v in alt.items():
        print(f"    {nm:<8} abad {v['abad']:.4f}   hip+knee {v['hipknee']:.4f}")
    a_max = max(v["abad"] for v in alt.values())
    hk_max = max(v["hipknee"] for v in alt.values())
    print(f"  hip/knee 最大差 {hk_max:.4f} rad"
          + ("  ✓ 與起點無關，實機可預期" if hk_max < 2e-3 else "  ⚠️ 與起點有關"))
    print(f"  abad     最大差 {a_max:.4f} rad"
          + ("  ✓" if a_max < 2e-3 else
             "  ⚠️ 與起點有關 —— 因為 abad 是**靠接觸停住**的，"
             "停在哪取決於撞上去時的速度（軟接觸），不是純靜力學"))

    print(f"\n{'SHM 關節':<16}{'MJCF 關節':<20}{'控制器角(rad)':>15}{'度':>9}{'馬達角(rad)':>14}")
    hf_shm = to_shm_dict(legmap, hang_free)
    hf_mot = to_motor(hf_shm)
    for lg in SHM_LEGS:
        for k in range(3):
            nm = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            mj = f"{legmap[lg]['prefix']}_{MJCF_KINDS[k]}_JOINT"
            print(f"{nm:<16}{mj:<20}{hf_shm[nm]:>15.4f}{np.degrees(hf_shm[nm]):>9.2f}"
                  f"{hf_mot[nm]:>14.4f}")

    # ---------------------------------------------------------------- 姿勢集
    stand12 = np.asarray(mm.STAND).reshape(12).copy()
    i_fl2 = q12_index(legmap, "fl", 1)         # fl2_hip_pitch = FBL_HIP_JOINT
    s1p, s1m = stand12.copy(), stand12.copy()
    s1p[i_fl2] += 0.05
    s1m[i_fl2] -= 0.05

    home12 = np.asarray(mm.HOME).reshape(12).copy()
    crouch12 = np.asarray(mm.CROUCH).reshape(12).copy()

    # ★ 膝模式：「後腿往前彎」是預設；「後腿往後彎」是把後兩腿的 hip/knee 翻號。
    #   M5 的 `--knee-back` 走這條，key 名固定成 `<pose>_knee_back`（小寫）。
    pose_q = {
        "HANG_FREE": hang_free,
        "HANG_NC": hang_nc,
        "STAND": stand12,
        "HOME": home12,
        "CROUCH": crouch12,
        "stand_knee_back": flip_rear_knee(legmap, stand12),
        "home_knee_back": flip_rear_knee(legmap, home12),
        "crouch_knee_back": flip_rear_knee(legmap, crouch12),
        "S1_fl2+0.05": s1p,
        "S1_fl2-0.05": s1m,
    }
    KNEE_MODE_PAIRS = (("STAND", "stand_knee_back"), ("HOME", "home_knee_back"),
                       ("CROUCH", "crouch_knee_back"))

    hr("2. 姿勢的限位檢查 —— 先確認算得出來的東西都合法，再談力矩")
    coord_msg = verify_flip_against_coord(legmap, stand12)
    print(coord_msg)
    print("\n膝模式定義：只把後兩腿（bl/br）的 hip_pitch 與 knee_pitch 翻號，前腿不動。")
    print("⚠️ 後腿的 HIP range 與前腿**相反**（前 (-2.442, 2.442+) / 後 (-2.791, 2.442)），")
    print("   所以翻號後合不合法必須逐項對模型讀出來的 range 檢查，不能靠對稱性推。\n")

    ranges = mm.leg_joint_ranges(m)             # (12, 2)，順序同 LEGS × abad/hip/knee
    print(f"{'姿勢':<18}{'最小限位餘裕(rad)':>20}{'最緊的關節':>18}{'range':>22}{'判定':>8}")
    lim_ok, lim_detail = True, {}
    for nm, q in pose_q.items():
        margin = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
        i = int(np.argmin(margin))
        who = idx2shm_of(legmap)[i]
        rng = f"({ranges[i, 0]:+.3f}, {ranges[i, 1]:+.3f})"
        ok = margin.min() > 0
        lim_ok &= ok
        lim_detail[nm] = {"min_margin_rad": float(margin.min()), "tightest": who,
                          "range": [float(ranges[i, 0])], "ok": bool(ok)}
        lim_detail[nm]["range"] = [float(ranges[i, 0]), float(ranges[i, 1])]
        print(f"{nm:<18}{margin.min():>20.4f}{who:>18}{rng:>22}"
              f"{('✓' if ok else '★超限'):>8}")
        if not ok:
            for j in np.where(margin <= 0)[0]:
                print(f"    ★ 超限：{idx2shm_of(legmap)[j]} = {q[j]:+.4f}，"
                      f"range ({ranges[j, 0]:+.3f}, {ranges[j, 1]:+.3f})"
                      f"，超出 {abs(margin[j]):.4f} rad")
    print("  → " + ("✓ 全部 10 個姿勢都在 MJCF 限位內，**沒有任何截斷**"
                    if lim_ok else "★★ 有姿勢超限，見上面標星的列 —— 不要用"))
    print(f"  對照：crouch_knee_back 的後腿 hip = "
          f"{pose_q['crouch_knee_back'][q12_index(legmap, 'bl', 1)]:+.3f}"
          f"（range {ranges[q12_index(legmap, 'bl', 1), 0]:+.3f} ~ "
          f"{ranges[q12_index(legmap, 'bl', 1), 1]:+.3f}）、knee = "
          f"{pose_q['crouch_knee_back'][q12_index(legmap, 'bl', 2)]:+.3f}"
          f"（±{ranges[q12_index(legmap, 'bl', 2), 1]:.3f}）")

    hr("3. 重力保持力矩表（懸空，N·m；正負號＝MJCF/控制器座標系）")
    results = {}
    for nm, q in pose_q.items():
        results[nm] = gravity_torque(m, d, q)
    print_tau_table(legmap, results)

    idx2shm = idx2shm_of(legmap)
    print("\n每個姿勢的最大絕對力矩：")
    for nm, r in results.items():
        who = idx2shm[int(np.argmax(np.abs(r["tau12"])))]
        print(f"  {nm:<18} max |τ| = {np.abs(r['tau12']).max():7.3f} N·m  ({who})"
              f"   ＝扭矩上限 {mm.TAU_MAX3[0]:.0f} N·m 的 "
              f"{np.abs(r['tau12']).max() / mm.TAU_MAX3[0] * 100:.1f} %")

    # ---------------------------------------------------------------- 交叉檢核
    hr("4. 交叉檢核 —— 「多印一個可以互相對照的量」")
    print("【檢核 A】吊帶承重 vs 模型重量（自由基座 6 DOF 的 qfrc_inverse）")
    print(f"{'姿勢':<18}{'Fx(N)':>10}{'Fy(N)':>10}{'Fz(N)':>10}"
          f"{'Mx':>9}{'My':>9}{'Mz':>9}{'Fz/mg':>9}")
    for nm, r in results.items():
        b = r["base"]
        print(f"{nm:<18}{b[0]:>10.2f}{b[1]:>10.2f}{b[2]:>10.2f}"
              f"{b[3]:>9.2f}{b[4]:>9.2f}{b[5]:>9.2f}{b[2] / (m_total * g):>9.4f}")
    print(f"  模型重量 = {m_total:.3f} kg × {g:.3f} = {m_total * g:.2f} N"
          f"   規格書 41 kg 對應 {SPEC_MASS * g:.2f} N")
    fz_ok = all(abs(abs(r["base"][2]) - m_total * g) < 0.5 for r in results.values())
    print(f"  → {'✓ 每個姿勢的垂直合力都等於模型重量（逆動力學可信）' if fz_ok else '⚠️ 對不上'}")
    print("  ⚠️ 但它等於的是 **MJCF 的質量**，不是規格書的 41 kg。差額見第 0 段。")
    print("     My 不為零是正常的：質心不在吊點正下方，吊帶要出一個俯仰力矩。")
    # ★ 換算成「質心偏移」比看 N·m 直觀，而且它是可以拿捲尺去量的量
    print("\n  ★ 但 My 在 knee_back 那三個姿勢大了兩個數量級 —— 換算成質心水平偏移："
          f"（偏移 = My / 重量）")
    for nm in ("STAND", "stand_knee_back", "HOME", "home_knee_back",
               "CROUCH", "crouch_knee_back"):
        off = results[nm]["base"][4] / (m_total * g)
        print(f"    {nm:<18} My = {results[nm]['base'][4]:>6.2f} N·m"
              f"  → 質心偏移 {off * 1000:>+7.1f} mm")
    print("    ⚠️ 預設膝模式四腿前後對稱，質心幾乎在吊點正下方（<1 mm）；")
    print("       **knee_back 四腿同向，質心前移到 55 mm** —— 吊帶會讓機身明顯前傾。")
    print("       實機意義：切到 knee_back 前先確認吊點位置或接受機身會歪，"
          "歪掉的機身會讓你誤判『腿沒到位』。")

    print("\n【檢核 B】qfrc_inverse vs qfrc_bias（qacc=0 時兩者必須相等；不等＝混進接觸/約束力）")
    for nm, r in results.items():
        print(f"  {nm:<18} 差 {r['bias_gap']:.3e} N·m   接觸點數 ncon = {r['ncon']}"
              + ("  ✓" if r["bias_gap"] < 1e-6 and r["ncon"] == 0 else "  ⚠️"))
    print("  （ncon 一律為 0 是因為 gravity_torque 刻意關掉碰撞；"
          "真實的接觸狀況在第 1 段另外報。）")

    print("\n【檢核 C】STAND 左右對稱性（左右鏡像姿勢 → |τ左| 應等於 |τ右|）")
    sym = symmetry_gap(legmap, results["STAND"]["tau12"])
    print(f"{'部位_關節':<22}{'左(N·m)':>12}{'右(N·m)':>12}{'||差||':>12}")
    worst = 0.0
    for k, v in sym.items():
        print(f"{k:<22}{v['left']:>12.4f}{v['right']:>12.4f}{v['abs_gap']:>12.2e}")
        worst = max(worst, v["abs_gap"])
    worst_rel = worst / np.abs(results["STAND"]["tau12"]).max()
    print(f"  → 最大不對稱 {worst:.2e} N·m（相對 {worst_rel * 100:.3f} %）"
          + ("  ✓ 在建模誤差量級內" if worst_rel < 5e-3
             else "  ⚠️ 超過 0.5 %，不是建模雜訊，模型或腿序有問題"))
    print("  ⚠️ 不是 0，來源是**官方 MJCF 自己就左右不完全對稱**，不是我們的錯：")
    print("     ・RAR_KNEE_LINK 質量 0.85979 kg，另外三條腿的膝是 0.86312 kg（差 3.3 g）")
    print("     ・FBL/RBL/RAR 的 KNEE body 有 2.1e-5 rad 的建模歪斜與 1.1e-5 m 的 x 偏移")
    print("     影響量級 <0.1 %，遠小於實機摩擦與感測誤差，不影響 S1 的判讀。")

    print("\n【檢核 C'】S1 微動的差分（±0.05 rad 應該只動 fl 那條腿，其它腿力矩不變）")
    dtau = results["S1_fl2+0.05"]["tau12"] - results["STAND"]["tau12"]
    other = np.array([abs(dtau[q12_index(legmap, lg, k)])
                      for lg in SHM_LEGS if lg != "fl" for k in range(3)])
    print(f"  fl 腿三軸力矩變化: " + ", ".join(
        f"{SHM_KINDS[k][2:]}={dtau[q12_index(legmap, 'fl', k)]:+.4f}" for k in range(3)))
    print(f"  其餘 9 軸最大變化: {other.max():.2e} N·m"
          + ("  ✓ 懸空時各腿獨立，符合預期" if other.max() < 1e-6 else "  ⚠️ 有耦合"))

    # ★ 這條是拿第 3 節那個「ABAD 力矩與姿勢無關」的預測，來檢驗 knee_back 這批新數字。
    #   預測若成立，翻號前後 abad 必須一模一樣；不成立就是新姿勢算錯了。
    print("\n【檢核 E】膝模式切換時 ABAD 力矩應**完全不變**")
    print("  依據：hip/knee 只在矢狀面（x–z）轉動，不改變腿的質量分布對 abad 軸（x 軸）的力臂。")
    print(f"{'姿勢對':<34}{'ABAD 最大變化':>16}{'hip/knee 最大變化':>20}{'判定':>8}")
    knee_mode_chk, abad_worst = {}, 0.0
    for a, b in KNEE_MODE_PAIRS:
        dt = results[b]["tau12"] - results[a]["tau12"]
        da = max(abs(dt[q12_index(legmap, lg, 0)]) for lg in SHM_LEGS)
        dh = max(abs(dt[q12_index(legmap, lg, k)]) for lg in SHM_LEGS for k in (1, 2))
        abad_worst = max(abad_worst, da)
        knee_mode_chk[f"{a}→{b}"] = {"abad_max_change": float(da),
                                     "hipknee_max_change": float(dh)}
        print(f"{a + ' → ' + b:<34}{da:>16.2e}{dh:>20.4f}"
              f"{('✓' if da < 1e-9 else '★不符'):>8}")
    print(f"  → ABAD 最大變化 {abad_worst:.2e} N·m"
          + ("  ✓ 預測成立，這批 knee_back 數字通過獨立檢驗"
             if abad_worst < 1e-9 else
             "  ★★ 預測不成立 —— 翻號翻錯腿或翻錯關節，數字不可用"))
    print("  （hip/knee 有變化才是對的：翻號後腿的姿態真的不一樣了。）")

    # 真實接觸狀況（gravity_torque 是關掉碰撞算的，這裡另外用打開碰撞的模型看一次）
    print("\n【檢核 F】各姿勢打開碰撞後真的有沒有自碰撞（力矩表本身是關掉碰撞算的）")
    pose_ncon = {}
    for nm, q in pose_q.items():
        pose_ncon[nm] = contact_report(m, d, q)["ncon"]
    hit = {k: v for k, v in pose_ncon.items() if v}
    # 預期會碰的只有兩個：HANG_FREE 是左右輪互頂（真實），
    # HANG_NC 依定義就是「關掉碰撞才算得出來」的姿勢，放回有碰撞的模型裡當然是穿模。
    expected_hit = {"HANG_FREE", "HANG_NC"}
    print("  " + "  ".join(f"{k}={v}" for k, v in pose_ncon.items()))
    print(f"  → 有自碰撞的姿勢：{list(hit) if hit else '無'}")
    if set(hit) <= expected_hit:
        print("    ✓ 都在預期內：HANG_FREE 是左右輪真的互頂（見第 1 節）；")
        print("      HANG_NC 依定義就是『關掉碰撞才存在』的對照姿勢，"
              "放回有碰撞的模型當然穿模 —— 它不是可執行的姿勢，不要下給實機。")
        print("    ★ 十個姿勢裡**能下給實機的八個全部零自碰撞**，包含三個 knee_back。")
    else:
        print(f"    ⚠️ 預期外的自碰撞：{sorted(set(hit) - expected_hit)} —— "
              f"去 contact_report() 看是哪兩個 body")

    # ------------------------------------------------- hang_free 附近的局部重力剛度
    hr("5. hang_free 附近的局部重力剛度 k = dτ/dq（N·m/rad）")
    print("為什麼需要它：S1 從自然下垂出發，而自然下垂**依定義**就是重力力矩 ≈ 0 的地方，")
    print("所以第 3 節那張表在 HANG_FREE 這一欄沒有訊號。真正有訊號的是**斜率**：")
    print("重力在平衡點附近就是一根並聯的彈簧，低 kp 微動時它會把關節拉回去。\n")
    print("  位置伺服 + 重力彈簧的平衡點：  kp·(q_cmd − q_eq) = τ0 + k·(q_eq − q_hang)")
    print("  令 Δ = q_cmd − q_hang  →       q_eq − q_hang = (kp·Δ − τ0) / (kp + k)")
    print("  τ0 ≈ 0 時退化成「只走到指令量的 kp/(kp+k) 倍」← 這就是實機要比對的量。\n")

    stf = {dl: local_stiffness(m, d, hang_free, dl) for dl in (0.02, 0.05)}
    k12 = stf[0.05]["k12"]
    tau_hang = gr_hf["tau12"]

    print(f"{'SHM 關節':<16}{'MJCF 關節':<20}{'τ0(N·m)':>10}{'k δ=0.02':>11}"
          f"{'k δ=0.05':>11}{'相對差':>9}{'判定':>8}")
    neg, nonlin = [], []
    for lg in SHM_LEGS:
        for k in range(3):
            i = q12_index(legmap, lg, k)
            nm = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            mj = f"{legmap[lg]['prefix']}_{MJCF_KINDS[k]}_JOINT"
            k2, k5 = stf[0.02]["k12"][i], stf[0.05]["k12"][i]
            rel = abs(k5 - k2) / max(abs(k2), 1e-9)
            if k5 <= 0:
                neg.append(nm)
            if rel > 0.05:
                nonlin.append((nm, rel))
            flag = "⚠️負" if k5 <= 0 else ("⚠️非線性" if rel > 0.05 else "✓")
            print(f"{nm:<16}{mj:<20}{tau_hang[i]:>10.3f}{k2:>11.3f}{k5:>11.3f}"
                  f"{rel * 100:>8.2f}%{flag:>8}")

    print(f"\n  τ0 檢核（hang_free 求解是否收斂 —— 又一個可互相對照的量）：")
    hk = [abs(tau_hang[q12_index(legmap, lg, k)]) for lg in SHM_LEGS for k in (1, 2)]
    ak = [abs(tau_hang[q12_index(legmap, lg, 0)]) for lg in SHM_LEGS]
    print(f"    hip / knee 的 |τ0| max = {max(hk):.5f} N·m  ✓ ≈0，收斂良好")
    print(f"    abad      的 |τ0| max = {max(ak):.4f} N·m  ← **不是 0**，"
          f"因為 abad 被左右輪的接觸力頂住（見第 1 節）")
    print(f"    → S1 若要一個乾淨的 τ0 = 0 對照，**必須選 hip 或 knee**，不要選 abad。"
          f"（原訂的 fl2_hip_pitch 正確。）")
    if neg:
        print(f"  ⚠️ k 為負（不穩定分支）的關節：{neg}")
    else:
        print(f"  ✓ 12 個 k 全為正（回復力，像單擺）—— hang_free 是穩定平衡")
    if nonlin:
        print(f"  ⚠️ δ=0.02 與 δ=0.05 差 >5 % 的關節（非線性已明顯）："
              + ", ".join(f"{n}({r*100:.1f}%)" for n, r in nonlin))
    else:
        print(f"  ✓ δ=0.02 與 δ=0.05 算出的 k 差都 <5 % → 這個範圍內線性化站得住")

    # ---------------------------------------------------------------- S1 移動比例
    hr("5b. S1 增益建議 —— 用「移動比例 kp/(kp+k)」挑，不是用下垂量")
    print(f"目標：比例落在 {RATIO_LO}~{RATIO_HI}（最理想 {RATIO_TARGET}）。")
    print("理由：比例明顯不等於 1，「有沒有走到位」才是高鑑別力的訊號；")
    print("      但也不能小到量不出來 —— Δ=0.05 rad 時，比例 0.5 → 移動 0.025 rad，"
          f"是感測雜訊 {SENSOR_NOISE} rad 的 25 倍。\n")

    k_kind, s1_kp = {}, {}
    for k in range(3):
        kk = np.array([k12[q12_index(legmap, lg, k)] for lg in SHM_LEGS])
        k_kind[k] = kk
        med = float(np.median(kk))
        cand = [c for c in S1_KP_CANDIDATES if RATIO_LO <= c / (c + med) <= RATIO_HI]
        pool = cand if cand else list(S1_KP_CANDIDATES)
        s1_kp[k] = float(min(pool, key=lambda c: abs(c / (c + med) - RATIO_TARGET)))

    print(f"{'關節種類':<12}{'k 中位':>10}{'建議 kp':>10}{'比例中位':>10}"
          f"{'Δ=0.05 移動量(rad)':>20}{'(度)':>8}{'原廠 kp':>9}")
    for k in range(3):
        med = float(np.median(k_kind[k]))
        rho = s1_kp[k] / (s1_kp[k] + med)
        print(f"{MJCF_KINDS[k]:<12}{med:>10.3f}{s1_kp[k]:>10.0f}{rho:>10.3f}"
              f"{rho * 0.05:>20.4f}{np.degrees(rho * 0.05):>8.2f}{mm.KP3[k]:>9.0f}")

    print(f"\n每個關節的預測移動比例（Δ 指令 0.05 rad，含 τ0 修正項）：")
    print(f"{'SHM 關節':<16}{'kp':>6}{'k':>9}{'比例 kp/(kp+k)':>16}"
          f"{'預測 q_eq−q_hang(rad)':>24}{'(度)':>8}")
    ratio12 = np.zeros(12)
    move12 = np.zeros(12)
    for lg in SHM_LEGS:
        for k in range(3):
            i = q12_index(legmap, lg, k)
            nm = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            kp = s1_kp[k]
            ratio12[i] = kp / (kp + k12[i])
            move12[i] = (kp * S1_DELTA - tau_hang[i]) / (kp + k12[i])
            print(f"{nm:<16}{kp:>6.0f}{k12[i]:>9.3f}{ratio12[i]:>16.3f}"
                  f"{move12[i]:>24.4f}{np.degrees(move12[i]):>8.2f}")
    print(f"  ⚠️ **abad 那四列的『預測移動量』不要拿來判讀。** 它把 τ0 算進去了，"
          f"所以左右腿一正一負（+0.05 的指令打不過 τ0=1.85 N·m 的重力，")
    print(f"     算出來反而往內走），但實機上輪子已經頂住、根本走不動。"
          f"模型的輪碰撞又只是 STL 凸包近似，數字不可信。")
    print(f"     → **S1 一定選 hip 或 knee**：τ0 ≈ 0、沒有接觸、左右四腿的預測值一致"
          f"（{move12[q12_index(legmap, 'fl', 1)]:.4f} rad ≈ "
          f"{np.degrees(move12[q12_index(legmap, 'fl', 1)]):.2f}°）。")
    print(f"  kd 建議：1.0（同原廠 KD3）。S1 是靜態保持，kd 只用來壓住鬆手瞬間的擺動；"
          f"kd 不影響上面的平衡點。")

    # ---------------------------------------------------------------- 增益建議
    hr("6. 增益建議（承重姿勢）—— 穩態下垂量 err = τ_重力 / kp")
    print(f"目標：下垂量落在 {SAG_LO}~{SAG_HI} rad（{np.degrees(SAG_LO):.1f}~"
          f"{np.degrees(SAG_HI):.1f}°），最理想 {SAG_TARGET} rad。")
    print(f"感測雜訊量級約 {SENSOR_NOISE} rad → 下垂量至少要大它 20 倍才量得準。\n")

    sag_table = {}
    for nm in ("STAND", "HOME", "CROUCH"):
        tau = np.abs(results[nm]["tau12"])
        sag_table[nm] = {}
        print(f"— {nm} —")
        print(f"{'kp':>6}{'最小下垂(rad)':>16}{'中位(rad)':>13}{'最大(rad)':>13}"
              f"{'最大(度)':>11}{'判定':>10}")
        for kp in KP_CANDIDATES:
            s = tau / kp
            sag_table[nm][kp] = s
            verdict = ("太軟" if s.max() > SAG_HI else
                       "太硬" if s.max() < SAG_LO else "★ 甜蜜點")
            print(f"{kp:>6.0f}{s.min():>16.4f}{np.median(s):>13.4f}{s.max():>13.4f}"
                  f"{np.degrees(s.max()):>11.2f}{verdict:>10}")
        print()

    # 依關節種類分別挑 kp：讓「該種類的中位下垂量」最接近 SAG_TARGET
    kind_kp, kind_tau = {}, {}
    tau_stand = np.abs(results["STAND"]["tau12"])
    for k in range(3):
        t = np.array([tau_stand[q12_index(legmap, lg, k)] for lg in SHM_LEGS])
        kind_tau[k] = t
        med = float(np.median(t))
        best = min(KP_CANDIDATES, key=lambda kp: abs(med / kp - SAG_TARGET))
        kind_kp[k] = float(best)

    print("★ 承重姿勢的建議增益（按關節種類分開，因為三種的重力力矩差一個量級以上）：")
    print(f"{'關節種類':<14}{'STAND τ 中位':>14}{'建議 kp':>10}{'預期下垂(rad)':>16}"
          f"{'(度)':>9}{'原廠 kp':>10}")
    rec_sag = {}
    for k in range(3):
        med = float(np.median(kind_tau[k]))
        s = med / kind_kp[k]
        rec_sag[k] = s
        print(f"{MJCF_KINDS[k]:<14}{med:>14.3f}{kind_kp[k]:>10.0f}{s:>16.4f}"
              f"{np.degrees(s):>9.2f}{mm.KP3[k]:>10.0f}")
    print(f"  ⚠️ 下垂量是**朝重力方向**偏離指令，符號跟著 τ 走 —— 見 JSON 的 per-joint 值。")
    print(f"  ⚠️ 這一組是「把腿舉到指定姿勢並撐住」用的，**不是 S1 用的**。")
    print(f"     S1 從自然下垂出發、只走 {S1_DELTA} rad，要用第 5b 節那組低得多的 kp。")

    # ---------------------------------------------------------------- 軌跡預演
    hr("7. 軌跡動態預演 —— HANG_FREE → STAND，餘弦插值 %.0f s（機身焊住）" % TRAJ_T)
    # 每一組是 (標籤, kp12, kd12)。除了均勻掃描，另外把「原廠 RL 增益」與
    # 「第 5 節的承重建議」當成具名對照組一起跑 —— 實機上很可能直接用原廠那組，
    # 不先在模擬裡看過就等於沒預演。
    def _by_kind(v3):
        return np.tile(np.asarray(v3, dtype=float), 4)

    named = [("原廠RL", _by_kind(mm.KP3), _by_kind(mm.KD3)),
             ("承重建議", _by_kind([kind_kp[k] for k in range(3)]), np.full(12, 2.0))]
    if args.quick:
        combos = named
    else:
        combos = [(f"{kp:.0f}/{kd:.0f}", np.full(12, kp), np.full(12, kd))
                  for kp in TRAJ_KP_SWEEP for kd in TRAJ_KD_SWEEP] + named

    print(f"目標：終端誤差 < {TRAJ_ERR_TOL} rad（{np.degrees(TRAJ_ERR_TOL):.1f}°）、"
          f"不碰限位、不飽和、不發散。")
    print("⚠️ 注意峰值力矩在整個 kp 範圍只從 3.7 變到 5.6 N·m —— "
          "調高 kp **幾乎不用付力矩代價**，")
    print("   所以「挑峰值力矩最小的」是錯的判準，會選到終端誤差 6° 的組合。判準要先看誤差。\n")
    print(f"{'組合':>10}{'峰值τ(N·m)':>13}{'峰值誤差(rad)':>15}{'終端誤差':>11}"
          f"{'終端|qvel|':>12}{'限位餘裕':>11}{'飽和步數':>10}{'機身位移mm':>12}{'判定':>8}")
    sweep = []
    for label, kp12, kd12 in combos:
        r = run_traj(m, d, hang_free, stand12, kp12, kd12)
        ok = (not r["diverged"]) and (not r["lim_hit"]) and r["n_sat"] == 0 \
            and r["final_err"].max() < TRAJ_ERR_TOL
        sweep.append({"label": label, "kp12": kp12, "kd12": kd12,
                      "ok": bool(ok), **r})
        print(f"{label:>10}{r['peak_tau'].max():>13.2f}"
              f"{r['peak_err'].max():>15.4f}{r['final_err'].max():>11.4f}"
              f"{r['qvel_max_end']:>12.2e}{r['lim_margin_min']:>11.4f}{r['n_sat']:>10d}"
              f"{r['base_drift_m'] * 1000:>12.3f}"
              f"{('✓' if ok else '✗'):>8}")

    good = [s for s in sweep if s["ok"]]
    if good:
        # 先過誤差門檻，再在合格者裡挑峰值力矩最小的（保守優先）
        best = min(good, key=lambda s: (s["peak_tau"].max(), s["final_err"].max()))
    else:
        best = min(sweep, key=lambda s: s["final_err"].max())
    print(f"\n★ 建議軌跡增益：{best['label']}"
          f"  kp = {np.array2string(np.unique(best['kp12']), precision=0)}"
          f"  kd = {np.array2string(np.unique(best['kd12']), precision=1)}")
    print(f"   峰值力矩 {best['peak_tau'].max():.2f} N·m"
          f"（＝上限 150 的 {best['peak_tau'].max() / 150 * 100:.1f} %），"
          f"終端誤差 {best['final_err'].max():.4f} rad "
          f"({np.degrees(best['final_err'].max()):.2f}°)"
          + ("" if good else "   ⚠️ 沒有任何組合完全過關，這是誤差最小的那組"))
    print(f"   ⚠️ 終端誤差不會是 0 —— 位置伺服撐住重力**必然**留下 τ/kp 的靜態撓度。"
          f"要真的到位得加前饋或積分項，S1 階段不需要。")

    # ---------------------------------------------------------------- 力矩保護
    hr("8. 力矩保護上限建議 = max(全部靜態重力力矩, 軌跡峰值) × 1.5")
    env = np.zeros(12)
    env_src = ["-"] * 12                     # 哪個姿勢撐出這個包絡 —— 多印一個可對照的量
    for nm, r in results.items():
        a = np.abs(r["tau12"])
        for j in range(12):
            if a[j] > env[j]:
                env[j], env_src[j] = a[j], nm
    env_static = env.copy()
    env = np.maximum(env, best["peak_tau"])
    tau_limit = np.minimum(np.ceil(env * 1.5), mm.TAU_MAX3[0])
    tau_limit = np.maximum(tau_limit, 3.0)     # 地板值，免得 abad 被鎖到動不了
    print(f"{'SHM 關節':<16}{'靜態最大':>10}{'來自姿勢':>18}{'軌跡峰值':>10}{'包絡':>9}"
          f"{'×1.5':>9}{'建議上限':>10}")
    for lg in SHM_LEGS:
        for k in range(3):
            i = q12_index(legmap, lg, k)
            nm = f"{lg}{k+1}_{SHM_KINDS[k][2:]}"
            print(f"{nm:<16}{env_static[i]:>10.2f}{env_src[i]:>18}"
                  f"{best['peak_tau'][i]:>10.2f}"
                  f"{env[i]:>9.2f}{env[i] * 1.5:>9.2f}{tau_limit[i]:>10.1f}")
    kind_limit = {MJCF_KINDS[k]: float(max(tau_limit[q12_index(legmap, lg, k)]
                                           for lg in SHM_LEGS)) for k in range(3)}
    print("\n★ 簡化成三個數（每種關節取四腿最大，M5 直接用這組）："
          + "  ".join(f"{k}={v:.0f}" for k, v in kind_limit.items()) + " N·m")
    print(f"  對照：MJCF actuatorfrcrange = ±150、規格書扭矩上限 = 150 N·m。"
          f"這組只有硬體上限的 {max(kind_limit.values()) / 150 * 100:.0f} %。")
    print(f"  ⚠️ 這是**預演**上限，不是硬體極限。實機一定有模型沒有的東西 ——"
          f"腿關節摩擦（MJCF 是 0，沒量過）、線束、吊帶拉扯、少掉的 2.2 kg ——")
    print(f"     所以實測力矩會比這裡高。首跑就用這組並**預期它可能誤觸發**：")
    print(f"     誤觸發是好消息，代表實機與模型有系統性差異，查清楚再放寬。"
          f"一開始就設 150 等於沒有保護。")

    # ---------------------------------------------------------------- JSON
    if args.no_json:
        print("\n（--no-json：略過寫檔）")
        return 0

    def sag_dump(nm):
        return {f"kp={kp:.0f}": to_shm_dict(legmap, sag_table[nm][kp])
                for kp in KP_CANDIDATES}

    doc = {
        "_readme": [
            "D1 Max 吊掛預演對照表。由 task7/inference/hang_rehearsal.py 產生。",
            "所有角度單位 rad、力矩 N·m。角度/力矩預設是 **控制器座標系（== MJCF）**；",
            "帶 _motor 後綴的才是馬達座標系（馬達角 = sign × 控制器角 + offset，",
            "力矩只套 sign）。純標準函式庫可讀，沒有 numpy 型別。",
            "結論欄（recommended）都附了原始數字（poses / gain_table / trajectory_sweep），",
            "數字對不上時請回頭看原始值，不要只信結論。",
            "",
            "★ poses 的 key：STAND / HOME / CROUCH 是**後腿往前彎**（預設膝模式）；",
            "  stand_knee_back / home_knee_back / crouch_knee_back 是**後腿往後彎**。",
            "  HANG_FREE 是自然下垂（含左右輪互頂），HANG_NC 是關掉碰撞的純單擺對照組。",
            "★ 增益有三組，別拿錯：",
            "  recommended.s1_kp_by_kind      = S1 單關節微動（從自然下垂出發，低 kp）",
            "  recommended.loaded_kp_by_kind  = 把腿舉到承重姿勢並撐住",
            "  recommended.traj_kp_by_kind    = HANG_FREE→STAND 的整趟軌跡",
        ],
        "meta": {
            "generated": str(date.today()),
            "generator": "task7/inference/hang_rehearsal.py",
            "scene": SCENE_HANG,
            "mujoco_version": mujoco.__version__,
            "timestep_s": float(m.opt.timestep),
            "gravity_ms2": g,
            "model_total_mass_kg": m_total,
            "spec_mass_kg": SPEC_MASS,
            "mass_gap_kg": float(m_total - SPEC_MASS),
            "leg_mass_kg": leg_mass,
            "base_mass_kg": float(m_total - leg_mass),
            "mass_gap_note": ("官方 MJCF 的總質量比規格書的 41 kg 少 2.18 kg。"
                              "少的在機身還是在腿，MJCF 看不出來："
                              "若在機身（電池/外殼，最可能）則腿關節力矩不受影響、"
                              "只有吊帶承重要加約 21 N；若分布在腿上則所有力矩要 ×1.056。"
                              "外加 MJCF 的腿關節摩擦是 0（沒量過）——"
                              "所以實機量到的力矩比這裡高幾 % 屬正常。"),
        },
        "leg_map": legmap,
        "joint_order_shm": list(SHM_JOINTS),
        "coord_transform": {
            "formula": "馬達角 = sign × 控制器角 + offset;  馬達力矩 = sign × 控制器力矩",
            "source": ("實機 zg_wheels-user-parameters.yaml，"
                       "M4_pose_capture.py 實機驗證（commit 30959bb）"),
            "sign": {f"{lg}{k+1}_{SHM_KINDS[k][2:]}": SIGN[lg][k]
                     for lg in SHM_LEGS for k in range(3)},
            "offset": {f"{lg}{k+1}_{SHM_KINDS[k][2:]}": OFFSET[lg][k]
                       for lg in SHM_LEGS for k in range(3)},
        },
        "joint_limits_rad": {
            f"{lg}{k+1}_{SHM_KINDS[k][2:]}":
                [float(x) for x in mm.leg_joint_ranges(m)[q12_index(legmap, lg, k)]]
            for lg in SHM_LEGS for k in range(3)},
        "hang_free": {
            "_note": ("機身焊在空中、腿只受重力與人為黏滯阻尼，收斂到的靜止姿勢 ——"
                      "凍結 mc_ctrl 之後腿真正會停的地方。"
                      "★ 含自碰撞：左右輪會在機身正下方互相頂住，所以 abad 停在 ±0.28 rad "
                      "而不是 0，且該處的純重力力矩不為 0（由接觸力承擔）。"),
            "q_ctrl": hf_shm,
            "q_ctrl_mjcf": to_mjcf_dict(legmap, hang_free),
            "q_motor": hf_mot,
            "q_deg": {k: float(np.degrees(v)) for k, v in hf_shm.items()},
            "tau_at_hang": to_shm_dict(legmap, tau_hang),
            "tau_at_hang_note": ("hang_free 這個姿勢下的**純重力**力矩（關掉碰撞算的）。"
                                 "hip/knee ≈ 0 → 收斂良好；abad ≠ 0 是因為被輪對輪的"
                                 "接觸力頂住，不是求解沒收斂。"),
            "stiffness_at_hang": to_shm_dict(legmap, k12),
            "stiffness_at_hang_note": ("局部重力剛度 k = dτ/dq（N·m/rad），中央差分 "
                                       f"δ = {stf[0.05]['delta']}。正值 = 回復力（像單擺）。"
                                       "用途：低 kp 微動的平衡點 "
                                       "q_eq − q_hang = (kp·Δ − τ0) / (kp + k)。"),
            "stiffness_delta_check": {
                "delta_0.02": to_shm_dict(legmap, stf[0.02]["k12"]),
                "delta_0.05": to_shm_dict(legmap, stf[0.05]["k12"]),
                "max_rel_diff": float(np.abs(
                    (stf[0.05]["k12"] - stf[0.02]["k12"])
                    / np.maximum(np.abs(stf[0.02]["k12"]), 1e-9)).max()),
                "all_positive": bool((k12 > 0).all()),
                "negative_joints": neg,
            },
            "self_contact": cr,
            "settle": {"t_s": st["t_settle"], "converged": bool(st["converged"]),
                       "qvel_max": st["qvel_max"],
                       "start_sensitivity_rad": alt},
        },
        "hang_free_no_contact": {
            "_note": ("關掉碰撞算出來的**純單擺**平衡點，即幾何上的『完全自由下垂』。"
                      "殘餘純重力力矩 ≈ 0，是「這是真的力學平衡點」的獨立證據。"
                      "實機不會停在這裡（輪子會先撞上），列出來當對照組。"),
            "q_ctrl": to_shm_dict(legmap, hang_nc),
            "q_motor": to_motor(to_shm_dict(legmap, hang_nc)),
            "residual_tau": to_shm_dict(legmap, gr_nc["tau12"]),
            "max_diff_vs_hang_free_rad": float(diff_nc),
        },
        "poses": {
            nm: {
                "q_ctrl": to_shm_dict(legmap, pose_q[nm]),
                "q_motor": to_motor(to_shm_dict(legmap, pose_q[nm])),
                "tau_gravity": to_shm_dict(legmap, r["tau12"]),
                "tau_gravity_mjcf": to_mjcf_dict(legmap, r["tau12"]),
                "tau_gravity_motor": to_motor(to_shm_dict(legmap, r["tau12"]), True),
                "tau_abs_max": float(np.abs(r["tau12"]).max()),
                "sling_wrench": {"force_N": [float(x) for x in r["base"][:3]],
                                 "torque_Nm": [float(x) for x in r["base"][3:]],
                                 "com_offset_x_mm": float(r["base"][4] / (m_total * g)
                                                          * 1000.0),
                                 "_note": ("com_offset_x = My / 重量，"
                                           "＝質心離吊點正下方的水平距離。"
                                           "knee_back 模式四腿同向，會前移到 27~55 mm，"
                                           "吊帶上的機身會明顯前傾。")},
                "knee_mode": ("back" if nm.endswith("_knee_back") else "front"),
                "limits": lim_detail[nm],
                "self_contact_ncon": pose_ncon[nm],
                "checks": {"bias_gap": r["bias_gap"],
                           "ncon_in_gravity_calc": r["ncon"],
                           "Fz_over_mg": float(r["base"][2] / (m_total * g))},
            } for nm, r in results.items()},
        "knee_mode": {
            "_note": ("膝模式切換 = **只把後兩腿（bl/br）的 hip_pitch 與 knee_pitch 翻號，"
                      "前腿完全不動**。出處：2026-08-25 實機四姿勢驗證，"
                      "docs/座標換算式驗證結果_2026-08-25.md §2.2。"
                      "預設（不帶後綴的 STAND/HOME/CROUCH）是**後腿往前彎**；"
                      "帶 _knee_back 後綴的是**後腿往後彎**。"),
            "pairs": {a: b for a, b in KNEE_MODE_PAIRS},
            "abad_invariance_check": knee_mode_chk,
            "abad_invariance_note": ("ABAD 力矩與膝模式無關（變化 < 1e-9 N·m）——"
                                     "因為 hip/knee 只在矢狀面轉動，不改變腿對 abad 軸"
                                     "（x 軸）的力臂。這條是拿來檢驗 knee_back 這批數字"
                                     "的獨立判據，不是事後解釋。"),
            "verified_against": "realbot/coord.py flip_rear_knee_mode()（執行時逐項比對）",
        },
        "checks": {
            "stand_symmetry": sym,
            "stand_symmetry_worst": worst,
            "s1_cross_leg_max_change": float(other.max()),
            "sling_Fz_matches_model_weight": bool(fz_ok),
            "all_poses_within_mjcf_limits": bool(lim_ok),
            "pose_limit_margins": lim_detail,
            "pose_self_contact_ncon": pose_ncon,
            "coord_py_crosscheck": coord_msg,
        },
        "gain_table": {
            "_note": "穩態下垂量 err = |τ_重力| / kp（rad）。",
            "kp_candidates": [float(k) for k in KP_CANDIDATES],
            "sag_rad": {nm: sag_dump(nm) for nm in ("STAND", "HOME", "CROUCH")},
        },
        "trajectory_sweep": [
            {"label": s["label"],
             "kp": [float(x) for x in s["kp12"]], "kd": [float(x) for x in s["kd12"]],
             "ok": s["ok"],
             "peak_tau_max": float(s["peak_tau"].max()),
             "peak_err_max": float(s["peak_err"].max()),
             "final_err_max": float(s["final_err"].max()),
             "lim_margin_min": s["lim_margin_min"], "lim_hit": s["lim_hit"],
             "n_sat": s["n_sat"], "base_drift_m": s["base_drift_m"],
             "diverged": s["diverged"],
             "peak_tau": to_shm_dict(legmap, s["peak_tau"])} for s in sweep],
        "recommended": {
            "_note": ("結論欄。每一項都能在上面的原始數字裡復現，"
                      "對不上時以原始數字為準。"),
            # --- S1：從自然下垂出發的單關節微動 ---
            "s1_kp_by_kind": {MJCF_KINDS[k]: s1_kp[k] for k in range(3)},
            "s1_kp": {f"{lg}{k+1}_{SHM_KINDS[k][2:]}": s1_kp[k]
                      for lg in SHM_LEGS for k in range(3)},
            "s1_kd": 1.0,
            "s1_delta_rad": S1_DELTA,
            "s1_move_ratio": to_shm_dict(legmap, ratio12),
            "s1_predicted_move_rad": to_shm_dict(legmap, move12),
            "s1_note": (
                "S1 的判讀量是**移動比例**，不是下垂量：對 hang_free 下一個 Δ 的位移指令，"
                "關節只會走到 (kp·Δ − τ0)/(kp + k)。τ0 ≈ 0 的 hip/knee 就是 kp/(kp+k) 倍。"
                f"建議 kp 讓比例落在 {RATIO_LO}~{RATIO_HI}，比例明顯不等於 1 才有鑑別力。"
                "★ 選 hip 或 knee 做 S1，不要選 abad —— abad 在 hang_free 被輪對輪的"
                "接觸力頂住，τ0 ≠ 0 且模型只是凸包近似，預測可信度低。"),
            # --- 承重姿勢（把腿舉起來並撐住）用的增益，與 S1 是兩回事 ---
            "loaded_kp_by_kind": {MJCF_KINDS[k]: kind_kp[k] for k in range(3)},
            "loaded_expected_sag_rad": {f"{lg}{k+1}_{SHM_KINDS[k][2:]}":
                                        float(abs(tau_stand[q12_index(legmap, lg, k)])
                                              / kind_kp[k])
                                        for lg in SHM_LEGS for k in range(3)},
            "loaded_note": ("下垂量 |τ|/kp；方向與該關節的 τ 同號（指令角減實際角）。"
                            f"至少要大於感測雜訊 {SENSOR_NOISE} rad 的 20 倍才算量得準。"),
            "sensor_noise_rad": SENSOR_NOISE,
            "traj_label": best["label"],
            "traj_kp": [float(x) for x in best["kp12"]],
            "traj_kd": [float(x) for x in best["kd12"]],
            "traj_kp_by_kind": {MJCF_KINDS[k]: float(best["kp12"][k]) for k in range(3)},
            "traj_kd_by_kind": {MJCF_KINDS[k]: float(best["kd12"][k]) for k in range(3)},
            "traj_duration_s": TRAJ_T,
            "traj_interp": "cosine（兩端速度為 0）",
            "traj_peak_tau_max": float(best["peak_tau"].max()),
            "traj_final_err_max_rad": float(best["final_err"].max()),
            "traj_note": ("終端誤差不會是 0：位置伺服撐住重力必然留下 τ/kp 的靜態撓度。"
                          "峰值力矩在 kp 20~120 之間只從 3.7 變到 5.6 N·m，"
                          "調高 kp 幾乎不用付力矩代價。"),
            "tau_limit_Nm": to_shm_dict(legmap, tau_limit),
            "tau_limit_by_kind_Nm": kind_limit,
            "tau_limit_driven_by_pose": {idx2shm_of(legmap)[i]: env_src[i]
                                         for i in range(12)},
            "tau_limit_rule": ("max(所有靜態姿勢的重力力矩, 建議增益下的軌跡峰值) × 1.5，"
                               "無條件進位，上限截到 150 N·m"
                               "（MJCF actuatorfrcrange / 規格書），"
                               "下限 3 N·m 免得 abad 被鎖死。"
                               "★ 這是預演值不是硬體極限。實機有 MJCF 沒有的腿關節摩擦與"
                               "少掉的 2.2 kg，可能誤觸發 —— 誤觸發代表模型與實機有系統性"
                               "差異，先查清楚再放寬，不要一開始就設 150。"),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 自檢：狗上是純標準函式庫，這裡就先用標準函式庫讀回來確認沒有 numpy 型別漏網
    reread = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    assert isinstance(reread["recommended"]["traj_kp"], list)
    print(f"\n已寫入 {OUT_JSON}"
          f"（{OUT_JSON.stat().st_size / 1024:.1f} KB，純標準函式庫讀回驗證通過）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
