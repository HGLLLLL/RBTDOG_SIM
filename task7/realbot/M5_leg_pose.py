#!/usr/bin/env python3
"""M5 —— 腿關節驅動與姿勢控制（★ 第一次驅動 D1 Max 的腿關節，**必須吊掛**）。

★★ 前置條件（少一項就不要跑）
  - **狗用吊帶吊起來，機身完全懸空，四腿自由下垂不觸地、不碰吊帶**
  - 狗處於**洩力**狀態（16 顆全部零增益）—— M0 會擋住不符的情形
  - M0 ✅、M1 ✅（寫入會被消費）、M2/M3 ✅（馬達真的會動）
  - 手邊有急停或電源開關，且至少一個人沒有在碰狗

為什麼腿關節跟輪子完全不同：輪子離地空轉不承載任何東西，控制律理解錯了最壞
只是輪子亂轉。腿關節是 **150 N·m**，掛著 41 kg 的機身，錯誤的符號會把腿甩到
機械停點上。所以本檔的設計核心是**低增益 + 重力對照**，不是「先給原廠增益看它動不動」。

════════════════════════════════════════════════════════════════════
核心手法：低增益下的偏差量本身就是一次量測
════════════════════════════════════════════════════════════════════

kp 給小的時候，關節撐不住負載，會穩定地停在一個可以事先算出來的偏差上。
對照值由 `inference/hang_rehearsal.py` 用官方 MJCF 產生，存在
`reference/hang_torque_ref.json`。

【S2 以後：命令一個遠離平衡的姿勢】  τ_重力 = kp × 追蹤誤差

  對得上（同數量級、**同號**） → 換算式、腿序、增益符號全部正確
  差一個負號                   → side_sign 用反了，**立刻停，不要加大 kp**
  差好幾倍                     → 腿序對錯，或 kp 的單位不是我們以為的那個

【S1 微動：不適用上面那條】
  自然下垂處依定義就是「零致動器力矩的平衡點」，所以那裡沒有力矩訊號。
  微動改看**位移**，平衡點是

      q − q_下垂 = (kp·Δ − τ₀) / (kp + k)

  k 是該處的局部重力剛度、τ₀ 是該處的純重力力矩。
  ⚠️ τ₀ 不能當成 0：吊起來時**左右輪會在機身正下方互相頂住**，
     abad 被接觸力擋在 ±0.28 rad，該處 τ₀ ≈ ±1.85 N·m。
     所以對 abad 下 +0.05 的指令，它其實會往 −0.036 走 —— **那不是符號錯**。

低 kp 既安全（錯了只是走偏，重力回復力矩會抵消錯誤的伺服）又有鑑別力。

════════════════════════════════════════════════════════════════════
接管時序（中間沒有空窗）
════════════════════════════════════════════════════════════════════

    讀當前實測角  →  SIGSTOP  →  第一幀就寫 p_des = 當前角、kp = 0、kd = 小阻尼
      →  RAMP_UP   kp 0 → 目標值
      →  MOVE      p_des 由當前角餘弦插值到目標
      →  HOLD      維持並記錄
      →  RETURN    插值回起始角
      →  RAMP_DOWN kp 降回 0
      →  歸零 + 補心跳  →  SIGCONT

════════════════════════════════════════════════════════════════════
中止行為：阻尼停止，而且**保持凍結**
════════════════════════════════════════════════════════════════════

任何保護觸發 → kp=0、effort=0、kd=--abort-kd（純阻尼），
**由背景執行緒持續維持**，且**不解凍**，等人工確認。

  - 為什麼不歸零：腿帶著載荷時全歸零 = 自由落體撞機械停點
  - 為什麼不直接 SIGCONT：mc_ctrl 恢復後會繼續下指令去它凍結前的姿勢。
    若我們已經把腿移遠了，那就是一次突跳 —— **中止時最不該做的事**
  - 為什麼要背景執行緒：只寫一次的話，controller 500 ms 後會把指令區清成 0
    （實測，見 `Keepalive` 的 docstring），阻尼就沒了。而印統計 + 等人按 Enter
    遠超過 0.5 秒。**這是 2026-08-26 抓到的一個真 bug。**

⚠️ 程式若被 kill -9 之類整個殺掉，心跳停 → controller 500 ms 後清零 → 腿失力下垂。
   吊掛狀態下這是可承受的（腿只是垂下去），但**不要當成安全網**。
   M0 要求的「16 顆洩力」前置條件保證了 mc_ctrl 凍結前也是洩力狀態，
   所以事後補一個 SIGCONT 不會造成突跳。

════════════════════════════════════════════════════════════════════
用法（在狗上，需 root）
════════════════════════════════════════════════════════════════════

    # S0 乾跑：不凍結、不寫入，只印目標/換算/限位/預演力矩
    python3 M5_leg_pose.py --joints fl2_hip_pitch --delta 0.05

    # S1 單顆微動
    sudo python3 M5_leg_pose.py --joints fl2_hip_pitch --delta 0.05 --kp 10 --confirm

    # S2 單腿 → STAND 的該腿目標
    sudo python3 M5_leg_pose.py --joints fl --pose stand --kp 40 --confirm

    # S3 前兩腿  /  S4 全部 12 個腿關節
    sudo python3 M5_leg_pose.py --joints front --pose stand --kp 40 --confirm
    sudo python3 M5_leg_pose.py --joints all --pose stand --kp 40 --confirm

    # S5 姿勢切換（匍匐）
    sudo python3 M5_leg_pose.py --joints all --pose crouch --kp 40 --confirm

    # S6 腿維持姿勢 + 四輪同時轉 → 16 顆全部在我們手上
    sudo python3 M5_leg_pose.py --joints all --pose stand --kp 40 --wheel-vel 0.3 --confirm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

import coord
import shm_io


class Keepalive(threading.Thread):
    """背景持續重寫指令與心跳，讓一個「靜止的指令」真的維持得住。

    ★★ 為什麼非有不可（這是 2026-08-26 抓到的一個真 bug）：

    實測證據顯示，`joint_shm_controller` 判定指令過期後會**把指令區清成 0**
    —— 2026-08-25 M1 第一次寫入失敗讀回全 0 就是這樣，見
    `docs/實機寫入結果_第三趟_2026-08-25.md` §2。
    （HANDOFF 另有一說是套 `estop_kd=35`，但那是讀設定檔推的，沒有觀測佐證。
      兩說的安全後果相反：清零 = 失力，estop_kd = 阻尼。**以實測為準。**）

    後果：中止時若只寫**一次**阻尼指令就去印統計、等人工確認，
    0.5 秒後那個阻尼會被清掉，腿變成完全失力 —— 正是整段中止設計要避免的事。
    而印統計 + 等人按 Enter 遠遠超過 0.5 秒。

    所以任何「要維持一段時間」的指令都必須有人持續餵心跳。
    不管上面兩說哪個才對，持續維持自己的指令都是嚴格更安全的做法。
    """

    def __init__(self, shm, state_ro, payload, hz: float = 200.0, label: str = ""):
        super().__init__(daemon=True)     # daemon：主程式無論如何結束都不會被卡住
        self._shm, self._state, self._payload = shm, state_ro, payload
        self._period = 1.0 / hz
        # ⚠️ 絕對不要叫 self._stop —— threading.Thread 內部有一個 _stop() 方法，
        #    join() 會呼叫它。用同名屬性蓋掉會讓 join() 炸 TypeError，
        #    而它炸的位置在收尾階段：mc_ctrl 已凍結、SIGCONT 還沒送出去。
        self._stop_evt = threading.Event()
        self.ticks = 0
        self.errors = 0
        self.label = label

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self._payload()
                self._shm.write_tick(self._state.read_tick(shm_io.STATE_STRIDE))
                self.ticks += 1
            except Exception:
                self.errors += 1
                if self.errors > 50:      # 一直失敗就別再刷了，留給主執行緒報告
                    break
            self._stop_evt.wait(self._period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)

_HERE = os.path.dirname(os.path.abspath(__file__))
# 上機時只會把 realbot/ 的檔案傳到狗上，所以對照表要跟腳本放在一起；
# 在本機開發時它其實住在 ../reference/。兩邊都找。
REF_JSON_PATHS = (
    os.path.join(_HERE, "hang_torque_ref.json"),
    os.path.join(_HERE, "..", "reference", "hang_torque_ref.json"),
)


# ---------------------------------------------------------------- 行程管理
def mc_ctrl_pid() -> int | None:
    # pgrep -x 精確比對執行檔名；-f 會匹配到自己的命令列（task6 中過兩次，殺掉自己的 SSH）
    r = subprocess.run(["pgrep", "-x", "mc_ctrl"], capture_output=True, text=True)
    out = r.stdout.strip()
    return int(out.split("\n")[0]) if out else None


def proc_state(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0]
    except Exception:
        return "?"


# ---------------------------------------------------------------- 關節選擇
def expand_joints(spec: str) -> list[str]:
    """把 --joints 的字串展開成 SHM 關節名清單。

    接受：完整關節名、腿名(fl/fr/bl/br)、群組(front/rear/all)，逗號分隔。
    ⚠️ **輪關節永遠不會被腿名或 all 展開進來** —— 輪子只能經由 --wheel-vel 驅動。
       理由：輪子是速度控制、腿是位置控制，混在同一個白名單裡遲早會把
       位置指令寫進輪子（那等於叫輪子轉到某個絕對角度，會全速甩）。
    """
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok == "all":
            legs = coord.LEGS
        elif tok == "front":
            legs = coord.FRONT_LEGS
        elif tok == "rear":
            legs = coord.REAR_LEGS
        elif tok in coord.LEGS:
            legs = (tok,)
        elif tok in shm_io.JOINTS:
            if tok.endswith(coord.KIND_WHEEL):
                raise SystemExit(
                    f"❌ {tok} 是輪關節，不能放進 --joints（位置控制會讓輪子全速甩）。\n"
                    "   要轉輪子請用 --wheel-vel。")
            out.append(tok)
            continue
        else:
            raise SystemExit(
                f"❌ 不認得 --joints 的 {tok!r}\n"
                f"   可用：all / front / rear / 腿名 {'/'.join(coord.LEGS)} / 完整關節名")
        for lg in legs:
            out.extend(lg + k for k in coord.LEG_KINDS)
    # 去重但保序
    seen, uniq = set(), []
    for j in out:
        if j not in seen:
            seen.add(j)
            uniq.append(j)
    return uniq


# ---------------------------------------------------------------- 軌跡
def smoothstep(u: float) -> float:
    """餘弦插值：兩端速度為 0，中間最快。u 夾在 [0,1]。

    用它而不是線性插值：線性插值在起點與終點有速度不連續，
    對 41 kg 吊在吊帶上的機身來說那是兩次衝擊，會讓吊帶盪起來。
    """
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return 0.5 * (1.0 - math.cos(math.pi * u))


def load_ref() -> dict:
    """讀 MJCF 預演的重力力矩對照表。沒有就回傳空字典（程式仍可跑，只是少了對照）。"""
    for p in REF_JSON_PATHS:
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            print(f"✅ 已載入預演對照表 {p}")
            return d
        except Exception as e:
            print(f"⚠️ 預演對照表 {p} 讀取失敗（{e}）")
    print("⚠️ 找不到預演對照表 → 本次沒有力矩對照，只剩追蹤誤差一個判準。")
    print("   （對照表是低 kp 驗證的核心，強烈建議先把它帶上狗）")
    return {}


# ⚠️ 對照表由 `inference/hang_rehearsal.py` 產生。這裡刻意寫成**容忍多種 key 拼法**：
#    產生端與消費端是兩支獨立演進的程式，硬編一組 key 就是在等它們哪天對不上，
#    而且對不上的症狀會是「沒有對照值」——**一個沉默的降級**，現場很難察覺。
#    所以寧可多試幾種拼法，並在完全找不到時大聲說出來。
_TAU_KEYS = ("tau_gravity", "gravity_torque", "tau")
_QCTRL_KEYS = ("q_ctrl", "qpos_ctrl", "q")
_STIFF_KEYS = ("stiffness_at_hang", "stiffness", "k")


def _pose_entry(ref: dict, pose_name: str) -> dict:
    """依姿勢名取出對照表的那一段。大小寫不敏感。"""
    poses = ref.get("poses") or {}
    for c in (pose_name, pose_name.upper(), pose_name.lower()):
        if c in poses:
            return poses[c] or {}
    return {}


def _dig(entry: dict, keys, joint: str):
    for k in keys:
        d = entry.get(k)
        if isinstance(d, dict) and joint in d:
            try:
                return float(d[joint])
            except (TypeError, ValueError):
                return None
    return None


def ref_torque(ref: dict, pose_name: str, joint: str):
    """某姿勢下某關節的重力保持力矩（**控制器座標系**）。取不到回 None。

    ⚠️ 對照表裡可能同時有馬達座標系的版本（key 帶 `_motor`）—— 絕對不要用那個，
       這裡只認不帶 `_motor` 的。拿錯座標系比拿不到還糟。
    """
    if not pose_name:
        return None
    return _dig(_pose_entry(ref, pose_name), _TAU_KEYS, joint)


def _hang_entries(ref: dict) -> list:
    """自然下垂的相關資料**散在兩個地方**，兩個都要找。

    ⚠️ 這裡踩過一次坑：原本寫成「先找 poses['hang_free']，找到就回傳」。
       但 `poses['HANG_FREE']` 只有角度與重力力矩，**沒有** τ₀ 與剛度 ——
       那兩個在頂層的 `hang_free` 區段。於是查詢永遠停在第一個字典，
       τ₀ 靜靜地退化成 0，而 τ₀=0 正是微動判讀會誤報反號的原因。

       症狀會是「一切正常，只是判讀怪怪的」—— 又一次「沉默的降級」。
       所以這裡回傳**候選清單**，由 _dig 逐個試到取得為止。
    """
    return [_pose_entry(ref, "hang_free"), ref.get("hang_free") or {}]


def _dig_any(entries, keys, joint: str):
    for e in entries:
        v = _dig(e, keys, joint)
        if v is not None:
            return v
    return None


def ref_hang_angle(ref: dict, joint: str):
    return _dig_any(_hang_entries(ref), _QCTRL_KEYS, joint)


def ref_stiffness(ref: dict, joint: str):
    """下垂點附近的局部重力剛度 k = dτ/dq（N·m/rad）。取不到回 None。"""
    return _dig_any(_hang_entries(ref) + [ref], _STIFF_KEYS, joint)


def ref_tau_at_hang(ref: dict, joint: str):
    """自然下垂處的**純重力**力矩 τ₀。取不到回 None。

    ★★ 這個量不能省略，理由是實機的物理事實：
       吊起來時**左右輪會在機身正下方互相頂住**（預演的自碰撞偵測到 3 個接觸點，
       法向力約 3 N）。所以 abad 不是停在自由單擺的平衡點，而是被接觸力擋住 ——
       那裡的純重力力矩 τ₀ ≈ ±1.85 N·m，**不是 0**。

       後果：對 abad 下 +0.05 rad 的指令，它實際會往 **−0.036** 走，
       因為 τ₀ 比 kp·Δ 還大。這**不是符號錯**，但天真的「方向反了就是符號錯」
       判斷會在這裡誤報，把一次正常的測試判成必須中止的嚴重錯誤。
    """
    return _dig_any(_hang_entries(ref), ("tau_at_hang", "tau0"), joint)


def predict_move(ref: dict, joint: str, kp: float, delta: float):
    """預測微動指令下關節實際會走多遠（rad）。取不到資料回 None。

    位置伺服與重力並聯，平衡點解 kp·(q_des − q) = τ₀ + k·(q − q_下垂)：

        q − q_下垂 = (kp·Δ − τ₀) / (kp + k)
    """
    k = ref_stiffness(ref, joint)
    if k is None or kp + k <= 0:
        return None
    t0 = ref_tau_at_hang(ref, joint) or 0.0
    return (kp * delta - t0) / (kp + k)


def ref_tau_limit(ref: dict, joint: str):
    """預演建議的力矩保護上限（N·m）。取不到回 None。"""
    rec = ref.get("recommended") or {}
    return _dig(rec, ("tau_limit_Nm", "tau_limit"), joint)


# ---------------------------------------------------------------- 主程式
def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="M5 —— 腿關節驅動與姿勢控制（必須吊掛）")
    ap.add_argument("--joints", default="fl2_hip_pitch",
                    help="要控制的關節：all / front / rear / 腿名 / 完整關節名，逗號分隔")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pose", choices=sorted(coord.POSES),
                   help="目標姿勢（只套用到 --joints 選中的關節）")
    g.add_argument("--delta", type=float,
                   help="相對當前角度的增量（rad），微動測試用")
    ap.add_argument("--knee-back", action="store_true",
                    help="用「後腿往後彎」的膝模式（預設是往前彎）")
    ap.add_argument("--confirm", action="store_true",
                    help="不帶這個就是乾跑：不凍結行程、不寫入任何東西")

    ap.add_argument("--kp", type=float, default=10.0, help="位置增益（原廠站立用 60~120）")
    ap.add_argument("--kd", type=float, default=1.0, help="阻尼增益（原廠 1.0）")
    ap.add_argument("--abort-kd", type=float, default=3.0, help="中止時的純阻尼值")
    ap.add_argument("--abort-hold", type=float, default=20.0,
                    help="非互動模式下，中止後維持阻尼的秒數（互動模式改為等 Enter）")

    ap.add_argument("--ramp", type=float, default=2.0, help="kp 斜坡升/降的秒數")
    ap.add_argument("--move", type=float, default=4.0, help="移動到目標的秒數")
    ap.add_argument("--hold", type=float, default=3.0, help="到位後維持的秒數")
    ap.add_argument("--no-return", action="store_true",
                    help="結束時不回起始姿勢，直接降增益（★ 腿會垂下，僅在確定安全時用）")
    ap.add_argument("--hz", type=float, default=200.0)

    ap.add_argument("--tmax", type=float, default=None,
                    help="力矩保護 N·m。不給的話**逐關節**採用預演的建議值"
                         "（ABAD/HIP/KNEE 不同），都取不到才退回 25。硬體上限是 150")
    ap.add_argument("--emax", type=float, default=0.40, help="追蹤誤差保護 rad")
    ap.add_argument("--vmax", type=float, default=2.0, help="關節速度保護 rad/s")
    ap.add_argument("--temp-max", type=float, default=70.0, help="馬達溫度保護 °C")
    ap.add_argument("--tau-hits", type=int, default=3,
                    help="力矩要連續幾筆超標才中止（防單筆雜訊尖峰誤中止）")
    ap.add_argument("--tau-hard", type=float, default=3.0,
                    help="硬上限倍率：|τ| > tmax×這個值即中止（只需 --tau-hard-hits 筆）")
    ap.add_argument("--trace", action="store_true",
                    help="逐筆記錄整段軌跡進 JSON。★ 量靜摩擦掙脫門檻用："
                         "MOVE 期間 kp·誤差 會從 0 爬上去，關節在門檻處才開始動，"
                         "『開始動的那一刻的力矩』就是門檻。只留最後 0.2 秒是量不到的")
    ap.add_argument("--tau-hard-hits", type=int, default=2,
                    help="硬上限要連續幾筆。★ 不能是 1 —— 見 §effort 單筆垃圾")
    ap.add_argument("--margin", type=float, default=0.05, help="機構限位的安全餘裕 rad")

    ap.add_argument("--wheel-vel", type=float, default=0.0,
                    help="HOLD 期間同時以此角速度轉四輪（rad/s），0 = 不轉")
    ap.add_argument("--wheel-kd", type=float, default=1.5)
    ap.add_argument("--wheel-tff", type=float, default=0.0)

    a = ap.parse_args()

    # ⚠️ 這兩個組合會被靜靜忽略 —— 靜靜地做錯事比報錯危險，所以在最前面擋掉。
    if a.knee_back and not a.pose:
        ap.error("--knee-back 只對 --pose 有意義（它是把後兩腿的 hip/knee 翻號），"
                 "跟 --delta 一起給會被忽略")
    if a.kp <= 0:
        ap.error("--kp 必須大於 0（kp=0 等於沒有位置控制，而且會讓下垂量預測除以零）")

    logp = shm_io.start_log("M5")
    print("M5 —— 腿關節驅動與姿勢控制\n")
    print("⚠️⚠️ 這支會驅動腿關節。確認：狗已吊掛、機身懸空、四腿自由不觸地、手邊有急停。\n")

    joints = expand_joints(a.joints)
    if not joints:
        print("❌ --joints 沒有選到任何關節")
        return 1
    ref = load_ref()

    # 角度感測雜訊的量級。用途：判斷一個位移「有沒有大到值得相信」。
    # 2026-08-25 M4 實測四個姿勢的最大標準差是 1.2e-4 rad，這裡取保守的 1e-3。
    _rec = ref.get("recommended") or {}
    try:
        NOISE_RAD = float(_rec.get("sensor_noise_rad") or 1e-3)
    except (TypeError, ValueError):
        NOISE_RAD = 1e-3

    # ---------------------------------------------------------------- 前置檢查
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過（16 個關節名與順序相符）")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節目前帶著非零增益：{', '.join(live)}")
        # ★ 現場一定會遇到的兩種情況，分開講，否則會誤以為是硬體怪怪的
        only_kd = all(abs(c["kp"]) + abs(c["effort"]) < 1e-9
                      for c in cmd0 if c["name"] in live)
        if only_kd:
            print("\n   看起來只有 kd 非零 → **這多半是上一次 M5 中止後留下的阻尼**。")
            print("   中止時我們刻意保持 mc_ctrl 凍結，所以沒有人來清這些欄位。")
            print("   確認狗的狀態安全之後解凍，mc_ctrl 會接手把指令區清乾淨：")
            print(f"       sudo kill -CONT {mc_ctrl_pid()}")
            print("   然後重跑本程式。")
        else:
            print("   請先讓狗洩力（遙控器關閉運控 / 趴下）再跑。")
        print("\n   ★ 這個前置條件同時保證了『事後補 SIGCONT 不會突跳』，不要跳過。")
        return 1
    print("✅ 16 顆全部洩力中")

    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")

    # ---------------------------------------------------------------- 起始角與目標角
    st0 = shm_io.read_joint_state()
    by_name = {r["name"]: r for r in st0}
    start_ctrl = {j: coord.to_ctrl(j, by_name[j]["position"]) for j in joints}

    if a.pose:
        pose = coord.POSES[a.pose]
        if a.knee_back:
            pose = coord.flip_rear_knee_mode(pose)
        target_ctrl = {j: pose[j] for j in joints}
        pose_name = a.pose + ("_knee_back" if a.knee_back else "")
    elif a.delta is not None:
        target_ctrl = {j: start_ctrl[j] + a.delta for j in joints}
        pose_name = f"delta{a.delta:+g}"
    else:
        print("❌ 要嘛給 --pose，要嘛給 --delta")
        return 1

    print(f"\n目標：{pose_name}　控制關節 {len(joints)} 個　"
          f"kp={a.kp} kd={a.kd}　移動 {a.move}s　維持 {a.hold}s")

    # ★ 力矩上限逐關節取：ABAD 的重力力矩是 KNEE 的兩倍多，用同一個門檻等於
    #   對 ABAD 太鬆、對 KNEE 太緊。預演已經按關節種類給了建議值。
    DEFAULT_TMAX = 25.0
    if a.tmax is not None:
        tmax_of = {j: a.tmax for j in joints}
        tmax_src = f"命令列指定 {a.tmax}"
    else:
        tmax_of = {j: (ref_tau_limit(ref, j) or DEFAULT_TMAX) for j in joints}
        n_ref = sum(1 for j in joints if ref_tau_limit(ref, j) is not None)
        tmax_src = (f"預演建議值（{n_ref}/{len(joints)} 個關節取到，"
                    f"其餘退回 {DEFAULT_TMAX}）")
    print(f"力矩保護：{tmax_src}　"
          + "　".join(f"{j}={tmax_of[j]:.0f}" for j in joints[:3])
          + ("…" if len(joints) > 3 else ""))

    # 限位檢查（★ 起點、終點、以及插值路徑上都要查 —— 餘弦插值是單調的，
    #   所以查兩端就夠；若哪天改成非單調的軌跡，這裡要跟著改成逐點查）
    print(f"\n{'關節':16s} {'起始(控制器)':>13s} {'目標(控制器)':>13s} {'Δ':>8s}"
          f" {'目標(馬達)':>11s} {'限位':>18s} {'預演τ':>8s}")
    bad = []
    for j in joints:
        lo, hi = coord.limits_of(j)
        # 目標套 --margin（規劃值該留餘裕）；起點套硬限位（它是既成事實，
        # 狗現在就停在那，套餘裕只會無謂地擋住整趟測試）
        m_t = coord.check_limit(j, target_ctrl[j], a.margin)
        m_s = coord.check_limit(j, start_ctrl[j], 0.0)
        msg = (f"目標 {m_t}" if m_t else "") or (f"起點 {m_s}" if m_s else "")
        tau_ref = ref_torque(ref, pose_name if a.pose else "", j)
        print(f"{j:16s} {start_ctrl[j]:13.4f} {target_ctrl[j]:13.4f}"
              f" {target_ctrl[j]-start_ctrl[j]:+8.4f}"
              f" {coord.to_motor(j, target_ctrl[j]):11.4f}"
              f" {f'[{lo:+.3f},{hi:+.3f}]':>18s}"
              f" {'—' if tau_ref is None else f'{tau_ref:8.2f}'}")
        if msg:
            bad.append(f"{j}: {msg}")
    if bad:
        print("\n❌ 限位檢查不過：")
        for b in bad:
            print("   " + b)
        print("   → 目標或起始角超出機構行程。不會有任何寫入。")
        return 1
    print("\n✅ 起點與目標都在機構限位內（含餘裕）")

    # 起始角 vs 預演的自然下垂角 —— 又一個互相對照的量
    dev = [(j, start_ctrl[j] - h) for j in joints
           for h in [ref_hang_angle(ref, j)] if h is not None]
    if dev:
        worst = max(dev, key=lambda x: abs(x[1]))
        print(f"   起始角 vs 預演自然下垂：最大差 {worst[0]} {worst[1]:+.4f} rad"
              + ("　⚠️ 偏大，狗可能不是自由下垂（吊帶纏到腿？腳踩到東西？）"
                 if abs(worst[1]) > 0.25 else "　✅ 相符"))

    # ★ 預測下垂量，並檢查 --emax 是否留得夠。
    #   低 kp 是**故意**讓關節垂下來當量測手段，若 emax 比預測下垂還小，
    #   第一個 tick 就會誤中止 —— 那不是故障，是門檻設錯。
    if a.pose:
        sags = [(j, t / a.kp) for j in joints
                for t in [ref_torque(ref, pose_name, j)] if t is not None]
        if sags:
            wj, ws = max(sags, key=lambda x: abs(x[1]))
            print(f"   預測最大下垂 {wj} {ws:+.4f} rad（{math.degrees(ws):+.1f}°）"
                  f"　= 預演τ / kp({a.kp})")
            if abs(ws) > 0.8 * a.emax:
                print(f"   ⚠️ 已達 --emax({a.emax}) 的 80%，會誤中止。"
                      f"請加大 kp 或放寬 --emax（建議 --emax {abs(ws)*2:.2f}）")
        else:
            print("   ⚠️ 對照表裡沒有這個姿勢的力矩 → 本次沒有大小的對照，只剩追蹤誤差")
    else:
        # 微動：預測「實際會走多遠」。★ 必須含 τ₀，不能只用 kp/(kp+k)——
        #   abad 被輪對輪接觸頂住，τ₀ ≈ ±1.85 N·m，會讓它往指令的**反方向**走。
        pred = [(j, m) for j in joints
                for m in [predict_move(ref, j, a.kp, a.delta)] if m is not None]
        if pred:
            print("   預測位移 = (kp·Δ − τ₀)/(kp + k)　"
                  "k=局部重力剛度、τ₀=下垂處的純重力力矩：")
            for j, m in pred:
                r = m / a.delta if a.delta else 0.0
                flag = ""
                if r < 0:
                    flag = "  ← 會往指令的反方向走（τ₀ > kp·Δ），**這不是錯誤**"
                elif r > 0.9:
                    flag = "  ⚠️ 太接近 1，鑑別力低 → 調小 kp"
                elif r < 0.15:
                    flag = "  ⚠️ 太小，接近感測雜訊 → 調大 kp 或加大 Δ"
                print(f"     {j:16s} 預期實走 {m:+.4f} rad"
                      f"（指令 {a.delta:+.4f}，比例 {r:+5.2f}）{flag}")
        else:
            print("   ⚠️ 對照表裡沒有剛度/τ₀ → 微動只能看方向，且**無法排除 τ₀ 造成的反向**")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結行程、沒有寫入。")
        print(f"\n📄 完整輸出已存到 {logp}")
        return 0

    if os.geteuid() != 0:
        print("❌ 需要 root 才能寫入：請加 sudo")
        return 1

    # ---------------------------------------------------------------- 執行
    idx = {j: shm_io.idx_of(j) for j in joints}
    widx = {w: shm_io.idx_of(w) for w in shm_io.WHEELS}
    ctrl_set = set(joints)

    period = 1.0 / a.hz
    t_ramp, t_move, t_hold = a.ramp, a.move, a.hold
    t_ret = 0.0 if a.no_return else t_move
    T_MOVE_END = t_ramp + t_move
    T_HOLD_END = T_MOVE_END + t_hold
    T_RET_END = T_HOLD_END + t_ret
    T_END = T_RET_END + t_ramp

    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    abort = ""
    aborted_soft = False
    samples: list[dict] = []       # HOLD 期間的取樣，事後統計用
    peak = {j: {"tau": 0.0, "tau_raw": 0.0, "err": 0.0, "v": 0.0} for j in joints}
    tau_hot = {j: 0 for j in joints}          # 力矩連續超標的計數
    tau_hard_hot = {j: 0 for j in joints}     # 硬上限連續超標的計數
    # ★ 「不可能來自我們」的力矩取樣：計數 + 留樣本，但**不靜靜丟掉**
    bogus = {j: [] for j in joints}
    trace: list = []                          # --trace 用的完整逐筆軌跡
    # ---- 輪子的儀表化
    # ★ 2026-08-26 S6 的教訓：「左前輪沒轉起來」只能靠眼睛看，因為 M5 對輪子
    #   一個數字都沒記 —— 而 S6 的整個目的就是輪子。工具沒量到要測的東西，
    #   等於這一階沒有證據，只有印象。
    #
    # ⚠️ 輪角讀數包裹在 [−π, π]（M3 第 5 號坑：轉超過半圈時天真相減連方向都算反），
    #   所以逐筆 wrap_pi 解纏再累加。
    wheel_total = {w: 0.0 for w in shm_io.WHEELS}      # 解纏後的累積轉角（馬達座標系）
    wheel_prev = {w: st0[widx0]["position"] for w, widx0 in
                  ((w, shm_io.idx_of(w)) for w in shm_io.WHEELS)}
    wheel_hold0 = {w: 0.0 for w in shm_io.WHEELS}      # HOLD 起點的累積值
    wheel_tau: dict = {w: [] for w in shm_io.WHEELS}   # HOLD 期間的力矩
    wheel_hold_t = [None, None]                        # HOLD 的起訖時刻
    # ★ 最近幾筆的原始取樣，中止時整段印出來。
    #   「多印一個可以互相對照的量」—— 只報一個峰值數字，事後分不出
    #   那是真的過載還是單筆尖峰。有這段就分得出來。
    recent: list[dict] = []
    RECENT_N = 40                            # 200 Hz 下約 0.2 秒
    tick_start = tick_end = None
    loop_elapsed = 0.0

    def damp_payload():
        """純阻尼的一幀（不含心跳 —— 心跳由 Keepalive 統一在最後寫）。"""
        for j in joints:
            shm.damp_only(idx[j], a.abort_kd)
        for w, wi in widx.items():
            shm.zero_gains(wi)              # 輪子空轉無載荷，歸零即可
        for i in range(len(shm_io.JOINTS)):
            if shm_io.JOINTS[i] not in ctrl_set and shm_io.JOINTS[i] not in widx:
                shm.zero_gains(i)

    def zero_payload():
        for i in range(len(shm_io.JOINTS)):
            shm.zero_gains(i)

    def write_once(payload):
        try:
            payload()
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
        except Exception as e:
            print(f"⚠️ 收尾寫入失敗：{e}")

    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.15)
        print(f"\n✅ 已凍結 mc_ctrl（狀態={proc_state(pid)}）")
        print("每輪 payload 寫完後同步心跳；第一幀就帶指令，中間沒有空窗\n")
        print(f"{'t(s)':>6s} {'階段':>9s} {'kp':>6s} {'最大|誤差|':>10s}"
              f" {'最大|τ|':>8s} {'關節':>16s}")

        tick_start = state_ro.read_tick(shm_io.STATE_STRIDE)
        tick_prev = tick_start
        stale = 0
        # 實機 joint_state 是 1 kHz，我們 200 Hz 取樣 → 每輪都該前進約 5。
        # 容忍 0.1 秒的停滯（避免排程抖動誤殺），再多就是真的凍住了。
        stale_limit = max(5, int(0.1 * a.hz))
        t0 = time.monotonic()
        nxt = t0
        last_print = -1.0

        while True:
            t = time.monotonic() - t0
            if t >= T_END:
                break

            # ---- 這一刻的階段、kp、目標角
            if t < t_ramp:
                phase, kp_now, u = "RAMP_UP", a.kp * (t / t_ramp), 0.0
            elif t < T_MOVE_END:
                phase, kp_now, u = "MOVE", a.kp, smoothstep((t - t_ramp) / t_move)
            elif t < T_HOLD_END:
                phase, kp_now, u = "HOLD", a.kp, 1.0
            elif t < T_RET_END:
                phase, kp_now = "RETURN", a.kp
                u = 1.0 - smoothstep((t - T_HOLD_END) / t_move)
            else:
                phase, u = "RAMP_DOWN", (0.0 if a.no_return else 0.0)
                kp_now = a.kp * max(0.0, (T_END - t) / t_ramp)
                if a.no_return:
                    u = 1.0        # 不回程就維持在目標角，只降增益

            des_ctrl = {j: start_ctrl[j] + u * (target_ctrl[j] - start_ctrl[j])
                        for j in joints}

            # ---- 讀狀態並檢查保護
            st = state_ro.states()
            tick_rec: dict = {}

            # ★ joint_state 的心跳有沒有在動？
            #   驅動層若掛掉，讀回的角度會凍住不變。那個故障會**偽裝成**「追蹤誤差
            #   越來越大」，現場會往控制參數的方向查 —— 查錯方向。
            #   多讀一個可以互相對照的量（tick），就能一句話分辨。
            tk = state_ro.read_tick(shm_io.STATE_STRIDE)
            if tk == tick_prev:
                stale += 1
                if stale >= stale_limit:
                    abort = (f"joint_state 的心跳連續 {stale} 次沒有前進（停在 {tk}）"
                             f"—— 狀態回報已凍結，讀到的角度不可信")
                    break
            else:
                stale = 0
                tick_prev = tk
            worst_e = (0.0, "")
            worst_t = (0.0, "")
            for j in joints:
                r = st[idx[j]]
                sgn = coord.SIGN[j[2:]][j[:2]]
                q = coord.to_ctrl(j, r["position"])
                # ⚠️⚠️ 速度與力矩也必須換座標系，不是只有角度。
                #   角度：θ_m = s·θ_c + off
                #   速度：ω_m = s·ω_c          （offset 是常數，微分掉了）
                #   力矩：由功率守恆 τ_c·ω_c = τ_m·ω_m = τ_m·s·ω_c → τ_c = s·τ_m
                #   s = ±1，所以乘或除等價，這裡一律用乘。
                #
                #   ★ 這是本專案「診斷輸出騙人」的第七號樣態（拿不同座標系的量互比）。
                #     raw effort 拿去跟 MJCF 預演的力矩比，一半的關節會憑空反號。
                v = sgn * r["velocity"]
                tau = sgn * r["effort"]
                err = q - des_ctrl[j]

                # ★ 我們的控制律在這一刻所能產生的力矩上限。
                #   實測遠超過它 → 那力矩不可能來自我們的指令。
                cap = kp_now * abs(err) + a.kd * abs(v)
                is_bogus = abs(tau) > 3.0 * cap + 1.0
                if is_bogus:
                    # ⚠️ **不靜靜丟掉** —— 計數並留樣本，最後大聲報告。
                    #   靜靜過濾掉的話，真的外力事件也會被吃掉而沒人知道。
                    bogus[j].append((round(t, 3), phase, round(tau, 2), round(cap, 2)))

                if abs(err) > abs(peak[j]["err"]):
                    peak[j]["err"] = err
                if abs(tau) > abs(peak[j]["tau_raw"]):
                    peak[j]["tau_raw"] = tau
                # 峰值力矩排除「不可能來自我們」的取樣，否則整張表被一筆垃圾污染
                # （2026-08-26 fr3_knee 的峰值就被一筆 −33.2 蓋掉了真實的 −4.1）
                if not is_bogus and abs(tau) > abs(peak[j]["tau"]):
                    peak[j]["tau"] = tau
                if abs(v) > abs(peak[j]["v"]):
                    peak[j]["v"] = v
                if abs(err) > worst_e[0]:
                    worst_e = (abs(err), j)
                if abs(tau) > worst_t[0]:
                    worst_t = (abs(tau), j)
                # ★ 先記錄再檢查 —— 觸發中止的那一筆本身必須留在證據裡
                tick_rec[j] = (round(q, 4), round(des_ctrl[j], 4),
                               round(tau, 3), round(v, 3))

                # ★★ 力矩保護：要連續 --tau-hits 筆超標才中止。
                #
                #   2026-08-26 實機 S4 的教訓：bl3_knee 跳出 9.96 N·m 觸發中止，
                #   但同一刻 kp·|err| + kd·|v| 只有 4.87 —— **我們的 PD 律根本產不出
                #   那個值**，而峰值誤差與峰值速度都正常。單筆雜訊尖峰的可能性很高。
                #
                #   這與第 6 號坑同源：M3 被 `velocity` 欄位的假尖峰誤中止，
                #   當時的修法就是「連續 N 筆才採信」。我寫 M5 時沒把它套到力矩上。
                #
                #   ⚠️ 但不能無限寬容：另外留一道**單筆就跳**的硬上限（tmax 的 hard 倍），
                #   真正的失控不該被連續筆數延遲。200 Hz 下 3 筆只有 15 ms，
                #   對 150 N·m 規格的關節而言完全安全。
                # ★★ 硬上限也**不能單筆就跳**。
                #   2026-08-26 實機：fr3_knee 在 RAMP_DOWN 吐出一筆 −33.2 N·m，
                #   而同一刻 q 完全沒變（小數第四位都一樣）、v≈0、我們的控制律上限只有 0.55。
                #   33 N·m 作用在關節上而位置一動不動 —— 物理上不可能，是 effort 欄位的
                #   單筆垃圾。我原本加硬上限是為了「真失控不該被連續筆數延遲」，
                #   結果它自己被要防的雜訊打中。
                #   → 2 筆（10 ms）就夠濾掉孤立垃圾，對 150 N·m 的關節毫無風險。
                if abs(tau) > tmax_of[j] * a.tau_hard:
                    tau_hard_hot[j] += 1
                    if tau_hard_hot[j] >= a.tau_hard_hits:
                        abort = (f"{j} 力矩連續 {tau_hard_hot[j]} 筆超過硬上限 "
                                 f"{tmax_of[j] * a.tau_hard:.1f} N·m（最後 {tau:+.2f}）")
                        break
                else:
                    tau_hard_hot[j] = 0
                if abs(tau) > tmax_of[j]:
                    tau_hot[j] += 1
                    if tau_hot[j] >= a.tau_hits:
                        abort = (f"{j} 力矩連續 {tau_hot[j]} 筆超過 {tmax_of[j]:.1f} N·m"
                                 f"（最後 {tau:+.2f}）")
                        break
                else:
                    tau_hot[j] = 0
                if abs(err) > a.emax:
                    abort = (f"{j} 追蹤誤差 {err:+.3f} 超過 {a.emax} rad"
                             f"（實測 {q:+.3f} / 目標 {des_ctrl[j]:+.3f}，τ={tau:+.2f}）")
                    break
                if abs(v) > a.vmax:
                    abort = f"{j} 速度 {v:+.2f} 超過 {a.vmax} rad/s"
                    break
                # ⚠️ 實測角用 **margin=0**（硬限位），不是 a.margin。
                #   a.margin 是給「規劃的軌跡」留的餘裕，已在啟動時查過。
                #   實測角本來就可能因追蹤誤差落在餘裕帶內（尤其低 kp 故意讓它垂），
                #   若這裡也套餘裕，第一個 tick 就會誤中止。
                m = coord.check_limit(j, q, 0.0)
                if m:
                    abort = f"{j} 實測角超出機構限位：{m}"
                    break
                if r["temp_C"] > a.temp_max:
                    abort = f"{j} 溫度 {r['temp_C']:.1f}°C 超過 {a.temp_max}"
                    break

                if phase == "HOLD":
                    samples.append({"j": j, "q": q, "des": des_ctrl[j],
                                    "err": err, "tau": tau, "v": v,
                                    "temp": r["temp_C"]})
            _tick = {"t": round(t, 3), "phase": phase,
                     "kp": round(kp_now, 2), "j": tick_rec}
            recent.append(_tick)
            if a.trace:
                trace.append(_tick)
            if len(recent) > RECENT_N:
                recent.pop(0)
            if abort:
                break

            # ---- 輪子：無論有沒有在驅動都持續解纏累加
            for w in shm_io.WHEELS:
                wi = widx[w]
                praw = st[wi]["position"]
                wheel_total[w] += shm_io.wrap_pi(praw - wheel_prev[w])
                wheel_prev[w] = praw
                if phase == "HOLD":
                    wheel_tau[w].append(coord.SIGN[coord.KIND_WHEEL][w[:2]]
                                        * st[wi]["effort"])
            if phase == "HOLD":
                if wheel_hold_t[0] is None:
                    wheel_hold_t[0] = t
                    wheel_hold0 = dict(wheel_total)
                wheel_hold_t[1] = t

            # 溫度掃**全部 16 顆**，不只白名單。
            # 白名單以外的關節雖然被我們壓成零增益，但它們仍然是同一批硬體 ——
            # 某顆沒在動的馬達過熱，代表的是驅動器或電源出事，那跟我們的測試同樣相關。
            hot = [r for r in st if r["temp_C"] > a.temp_max]
            if hot:
                abort = ("；".join(f"{r['name']} {r['temp_C']:.1f}°C" for r in hot)
                         + f" 超過 {a.temp_max}°C")
                break

            # ---- 寫指令
            # ① 沒被選中的關節每輪壓零，確保只有白名單會動
            for i in range(len(shm_io.JOINTS)):
                nm = shm_io.JOINTS[i]
                if nm in ctrl_set:
                    continue
                if nm in widx and a.wheel_vel and phase == "HOLD":
                    continue
                shm.zero_gains(i)
            # ② 白名單：先目標值、後增益（見 shm_io.write_cmd 的理由）
            for j in joints:
                shm.write_cmd(idx[j],
                              position=coord.to_motor(j, des_ctrl[j]),
                              velocity=0.0, effort=0.0,
                              kp=kp_now, kd=a.kd)
            # ③ 輪子只在 HOLD 期間、且有指定速度時才驅動（純速度控制，kp=0）
            if a.wheel_vel and phase == "HOLD":
                for w, wi in widx.items():
                    shm.write_cmd(wi, position=st[wi]["position"],
                                  velocity=coord.SIGN[coord.KIND_WHEEL][w[:2]] * a.wheel_vel,
                                  effort=a.wheel_tff, kp=0.0, kd=a.wheel_kd)
            # ④ 整幀寫完才寫心跳 —— 它是「這幀備妥了」的旗標
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))

            if t - last_print >= 0.25:
                print(f"{t:6.2f} {phase:>9s} {kp_now:6.2f} {worst_e[0]:10.4f}"
                      f" {worst_t[0]:8.3f} {worst_t[1]:>16s}")
                last_print = t

            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)

        # ⚠️ tick_end 必須在 restore 之前取，否則會把解凍等待算進去（這個坑中過一次）
        tick_end = state_ro.read_tick(shm_io.STATE_STRIDE)
        loop_elapsed = time.monotonic() - t0

    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    except Exception as e:
        abort = f"未預期的例外：{type(e).__name__}: {e}"
    finally:
        # ★ 先立刻寫一幀，再交給 Keepalive 持續維持。
        #   只寫一幀是不夠的 —— controller 500 ms 後會把指令區清成 0（實測），
        #   而底下印統計 / 等人工確認遠超過 0.5 秒。見 Keepalive 的 docstring。
        if abort:
            write_once(damp_payload)
            aborted_soft = True
            keeper = Keepalive(shm, state_ro, damp_payload, a.hz, "阻尼保持")
        else:
            write_once(zero_payload)
            keeper = Keepalive(shm, state_ro, zero_payload, a.hz, "零增益保持")
        keeper.start()

    # ---------------------------------------------------------------- 收尾
    st1 = state_ro.states()
    print("\n" + "=" * 74)
    if abort:
        print(f"⛔ 中止：{abort}")
        print(f"\n★ 已切成純阻尼 kd={a.abort_kd}，並由背景執行緒**持續維持**")
        print("  （只寫一次會在 500 ms 後被 controller 清成 0 —— 那等於腿失力）")
        print("★ mc_ctrl **仍在凍結中**，故意不自動解凍：")
        print("  它恢復後會繼續下指令去凍結前的姿勢，若腿已移遠就是一次突跳。")

        # ---- 中止前最後幾筆原始取樣（只印涉事關節）
        # ★ 只報一個峰值數字，事後分不出「真的過載」還是「單筆雜訊尖峰」。
        #   把原始序列印出來，那個判斷就變成看一眼的事。
        hurt = abort.split()[0]
        if recent and hurt in joints:
            print(f"\n中止前最後 {min(15, len(recent))} 筆 —— {hurt}"
                  f"（力矩上限 {tmax_of[hurt]:.1f}；JSON 裡存了 {len(recent)} 筆）")
            print(f"  {'t(s)':>7s} {'階段':>9s} {'實測q':>9s} {'目標q':>9s}"
                  f" {'τ':>8s} {'v':>8s} {'kp|e|+kd|v|':>11s}")
            for rr in recent[-15:]:
                d4 = rr["j"].get(hurt)
                if not d4:
                    continue
                q_, des_, tau_, v_ = d4
                cap = rr["kp"] * abs(q_ - des_) + a.kd * abs(v_)
                mark = "  ←超標" if abs(tau_) > tmax_of[hurt] else ""
                print(f"  {rr['t']:7.3f} {rr['phase']:>9s} {q_:9.4f} {des_:9.4f}"
                      f" {tau_:8.3f} {v_:8.3f} {cap:11.2f}{mark}")
            print("\n  ★ 怎麼讀最後一欄：`kp·|誤差| + kd·|速度|` 是**我們的 PD 律在那一刻"
                  "所能產生的上限**。")
            print("    實測 τ 遠大於它 → 那個力矩**不是我們下的**")
            print("      → 外力撞擊、或 effort 欄位的單筆雜訊尖峰（先看誤差與速度正不正常）")
            print("    實測 τ 貼著它    → 是我們的控制器在推，代表真的推不動（機構受阻）")
    else:
        print("✅ 序列完整跑完")

    if tick_end is not None and loop_elapsed > 0:
        rate = (tick_end - tick_start) / loop_elapsed
        print(f"\n心跳 {tick_start} → {tick_end}（+{tick_end - tick_start} / "
              f"{loop_elapsed:.2f}s = {rate:.0f}/s，實機應接近 1000/s）")
        if rate < 500:
            print("⚠️ 心跳速率偏低 —— 指令可能有部分幀被判定過期，結果要打折看")

    # ---- HOLD 期間的統計：這才是主要證據
    if samples:
        print("\n" + "=" * 74)
        print("HOLD 期間統計（★ 主要證據：低 kp 下的下垂量 = 重力力矩 / kp）")
        print("=" * 74)
        print("（角度、力矩全部已換算到**控制器座標系**，與 MJCF 同框，可以直接互比）")
        if a.delta is not None:
            print("\n★ 微動測試驗的是**方向**，不是大小。")
            print("  自然下垂姿勢依定義就是「零力矩平衡點」，那裡的重力力矩 ≈ 0，")
            print("  所以下垂量沒有訊號 —— 大小的對照要等 S2 以後（命令一個遠離平衡的姿勢）。")
            print("  這裡看兩件事：① 有沒有往對的方向走 ② 走了指令量的幾成。")
            print("  ② 的預期值是 kp/(kp+k)，k 是該關節在下垂點附近的重力剛度（見預演表）。\n")
        print(f"{'關節':16s} {'目標':>8s} {'實測':>8s} {'誤差':>9s} {'實測τ':>8s}"
              f" {'kp×誤差':>9s} {'預演τ':>8s} {'判讀':>10s}")
        by_j: dict[str, list[dict]] = {}
        for s in samples:
            by_j.setdefault(s["j"], []).append(s)
        n_sign_bad = n_ok = 0
        for j in joints:
            ss = by_j.get(j, [])
            if not ss:
                continue
            n = len(ss)
            err = sum(x["err"] for x in ss) / n
            tau = sum(x["tau"] for x in ss) / n
            des = sum(x["des"] for x in ss) / n
            q = sum(x["q"] for x in ss) / n
            # ★ 靜態下 kp·(q_des − q) 應該等於關節出力，也就是 −kp·err
            kperr = -a.kp * err
            tref = ref_torque(ref, pose_name if a.pose else "", j)
            verdict = "—"
            if tref is not None and abs(tref) > 0.3:
                ratio = kperr / tref
                if ratio < -0.3:
                    verdict = "❌反號"
                    n_sign_bad += 1
                elif 0.5 <= ratio <= 2.0:
                    verdict = "✅相符"
                    n_ok += 1
                else:
                    verdict = f"⚠️{ratio:.2f}x"
            print(f"{j:16s} {des:8.4f} {q:8.4f} {err:+9.4f} {tau:+8.3f}"
                  f" {kperr:+9.3f} {'—' if tref is None else f'{tref:+8.3f}'}"
                  f" {verdict:>10s}")

        # ---- 微動模式：方向與移動比例（S1 的主要判準）
        if a.delta is not None:
            print(f"\n{'關節':16s} {'指令Δ':>9s} {'實走Δ':>9s} {'比例':>8s}"
                  f" {'預測實走':>9s} {'判讀':>12s}")
            n_dir_bad = 0
            for j in joints:
                ss = by_j.get(j, [])
                if not ss:
                    continue
                q = sum(x["q"] for x in ss) / len(ss)
                moved = q - start_ctrl[j]
                ratio = moved / a.delta if a.delta else 0.0
                # ★ 跟**預測位移**比，不是跟指令的號比。
                #   abad 的 τ₀ 會讓它正當地往指令反方向走 —— 拿指令的號當基準會誤報。
                pm = predict_move(ref, j, a.kp, a.delta)
                if pm is None:
                    # 沒有對照值時只能退回看指令方向，但要講清楚這個判斷不可靠
                    verdict = "❌走反方向?" if ratio < -0.15 else (
                        "⚠️幾乎沒動" if abs(ratio) < 0.05 else "✅有動(無對照)")
                    if ratio < -0.15:
                        n_dir_bad += 1
                elif abs(pm) < 2 * NOISE_RAD:
                    verdict = "—預測量太小"
                elif moved * pm < 0:
                    verdict, n_dir_bad = "❌與預測反號", n_dir_bad + 1
                elif abs(moved - pm) <= max(0.3 * abs(pm), 2 * NOISE_RAD):
                    verdict = "✅方向與量都符"
                else:
                    verdict = "✅方向符,量偏差"
                print(f"{j:16s} {a.delta:+9.4f} {moved:+9.4f} {ratio:8.2f}"
                      f" {'—' if pm is None else f'{pm:+9.4f}'} {verdict:>12s}")
            if n_dir_bad:
                print(f"\n❌❌ {n_dir_bad} 個關節與**預測反號**。")
                print("   預測已經把 τ₀ 算進去了，所以這不是「abad 本來就會反著走」那回事 ——")
                print("   最可能是 side_sign 用反了。反號的位置控制是正回授，")
                print("   **不要加大 kp**，越大跑得越遠。停下來核對 coord.py 的 SIGN 表。")
                print("   （低 kp 在這裡救了你一次：重力回復力矩與錯誤的伺服互相抵消，")
                print("     所以它只是走偏，沒有衝到限位。）")

        if n_sign_bad:
            print(f"\n❌❌ {n_sign_bad} 個關節的保持力矩與預演**反號**。")
            print("   最可能是 side_sign 用反了。**不要加大 kp 再試** —— ")
            print("   反號的位置控制是正回授，kp 越大跑得越遠。先回頭核對 coord.py。")
        elif n_ok:
            print(f"\n★ {n_ok} 個關節的保持力矩與 MJCF 預演相符 —— "
                  "換算式、腿序、增益符號都通過交叉驗證。")

        # ---- 吊帶卡住 vs 增益不足：兩者症狀像，成因完全不同
        stuck = [j for j in joints
                 if abs(peak[j]["err"]) > 0.5 * a.emax
                 and abs(peak[j]["tau"]) > 0.7 * tmax_of[j]]
        weak = [j for j in joints
                if abs(peak[j]["err"]) > 0.5 * a.emax
                and abs(peak[j]["tau"]) < 0.3 * tmax_of[j]]
        if stuck:
            print(f"\n⚠️ 力矩接近上限但誤差不收斂：{', '.join(stuck)}")
            print("   → 疑似**機構受阻**（吊帶纏到腿？碰到機身？）。加大 kp 不會解決，")
            print("     反而會把力矩推到更高。請目視檢查再決定。")
        if weak:
            print(f"\nℹ️ 誤差大但力矩小：{', '.join(weak)}")
            print("   → 這是**增益不足**，低 kp 階段的預期行為，不是故障。")

    # ---- 峰值
    print("\n" + "=" * 74)
    print(f"{'關節':16s} {'峰值誤差':>10s} {'峰值τ':>9s} {'峰值v':>9s} {'溫度':>7s} {'異常筆':>6s}")
    for j in joints:
        nb = len(bogus[j])
        print(f"{j:16s} {peak[j]['err']:+10.4f} {peak[j]['tau']:+9.3f}"
              f" {peak[j]['v']:+9.3f} {st1[idx[j]]['temp_C']:7.1f} {nb:6d}")

    nb_all = sum(len(v) for v in bogus.values())
    if nb_all:
        print(f"\n⚠️ 偵測到 {nb_all} 筆**不可能來自我們指令**的力矩取樣")
        print("   判準：|τ| > 3×(kp·|誤差| + kd·|速度|) + 1.0")
        print("   已排除在上面的『峰值τ』之外，但原始值列在這裡 ——")
        print("   **不靜靜丟掉**：若這些是真的外力事件，你要看得到。")
        print(f"   {'關節':16s} {'t(s)':>8s} {'階段':>9s} {'實測τ':>9s} {'我們的上限':>10s}")
        for j in joints:
            for (tb, ph, tv, cp) in bogus[j][:6]:
                print(f"   {j:16s} {tb:8.3f} {ph:>9s} {tv:9.2f} {cp:10.2f}")
        print("\n   ⚠️ 判別：同一刻的**位置有沒有變、速度是不是 0**。")
        print("     位置完全不動 + 速度 0 + 力矩爆表 = 感測單筆垃圾（物理上不可能）")
        print("     位置或速度也異常       = 可能是真的外力，要查")

    # ---- 靜摩擦掙脫門檻（--trace + --delta 才算）
    # ★ 原理：MOVE 期間目標角平滑離開實測角，`kp·誤差` 從 0 慢慢爬。
    #   關節被靜摩擦黏住不動，直到力矩爬過門檻才鬆動。
    #   **「開始動的那一刻的力矩」就是掙脫門檻。**
    #   ⚠️ 只掃 kp、看終點位移是量不準的：門檻附近位移很小，
    #     跟「沒掙脫但有彈性變形」分不出來。要看的是**動起來的時刻**，不是終點。
    if a.trace and a.delta is not None and trace:
        print("\n" + "=" * 74)
        print("靜摩擦掙脫門檻（MOVE 期間力矩爬升，記錄『開始持續滑動』的時刻）")
        print("=" * 74)
        # ★★ 不給單一數字，給**三個持續時間門檻**下的值。
        #   2026-08-26 實機教訓：HIP 在 25/100/300 ms 三種門檻下數字完全不變（可信）；
        #   ABAD 卻是 −0.229 / −0.558 / 沒有 —— 那是**瞬間微滑**被當成掙脫。
        #   只印一個數字的話，這兩種情況長得一模一樣。
        #   → 三個門檻一致 = 真的持續滑動；隨門檻變動或消失 = 微滑，不是掙脫。
        WINDOWS = [(int(0.025 * a.hz), "25ms"), (int(0.10 * a.hz), "100ms"),
                   (int(0.30 * a.hz), "300ms")]
        VTH = 0.03      # rad/s，算「在滑」的速度門檻

        def _onset(j, n_need):
            seq = [rr for rr in trace if j in rr["j"]]
            for i in range(len(seq) - n_need):
                if all(abs(seq[i + k]["j"][j][3]) > VTH for k in range(n_need)):
                    return seq[i]["t"], seq[i]["j"][j][2]
            return None

        print(f"判準：|速度| > {VTH} rad/s 持續多久才算「開始滑動」\n")
        print(f"{'關節':16s} " + " ".join(f"{lbl:>12s}" for _, lbl in WINDOWS)
              + f" {'整段最大|τ|':>11s}")
        for j in joints:
            cells, vals = [], []
            for n_need, _ in WINDOWS:
                o = _onset(j, n_need)
                cells.append(f"{o[1]:+12.3f}" if o else f"{'沒有':>12s}")
                vals.append(o[1] if o else None)
            mx = max((abs(rr["j"][j][2]) for rr in trace if j in rr["j"]), default=0.0)
            print(f"{j:16s} " + " ".join(cells) + f" {mx:11.3f}")
            if all(v is not None for v in vals) and \
                    max(abs(v) for v in vals) - min(abs(v) for v in vals) < 0.15:
                print(f"{'':16s} → ✅ 三個門檻一致，這是**真的持續滑動**。"
                      f"掙脫門檻 ≈ {abs(vals[0]):.2f} N·m")
            elif vals[0] is None:
                print(f"{'':16s} → ⚠️ **全程沒有掙脫**。掙脫門檻 **> {mx:.2f} N·m**（下界）"
                      f"，要加大 --kp 或 --delta")
            else:
                print(f"{'':16s} → ⚠️ **隨門檻變動 → 是瞬間微滑，不是掙脫。**"
                      f"這裡的數字不可當成門檻，要加大 --kp 或 --delta 重測")
        print("\n★ 正反兩個方向都跑，就能不靠模型分離摩擦與重力：")
        print("   |τ₊| = τ_重力 + f 、 |τ₋| = f − τ_重力")
        print("   → f = (|τ₊|+|τ₋|)/2 、 τ_重力 = (|τ₊|−|τ₋|)/2")
        print("   ⚠️ 兩個方向都要是「✅ 三門檻一致」，這個相減才有意義。")

    # ---- 輪子（只有 --wheel-vel 時才有意義）
    wheel_rep: dict = {}
    if a.wheel_vel and wheel_hold_t[0] is not None:
        dt = (wheel_hold_t[1] or 0) - wheel_hold_t[0]
        print("\n" + "=" * 74)
        print(f"輪子（HOLD {dt:.2f} 秒，v_des={a.wheel_vel} kd={a.wheel_kd} "
              f"tau_ff={a.wheel_tff}）　角度已解纏、已換算到控制器座標系")
        print(f"{'輪':10s} {'累積轉角':>10s} {'角速度':>9s} {'追蹤率':>7s}"
              f" {'平均τ':>8s} {'摩擦推估':>9s}  判讀")
        for w in shm_io.WHEELS:
            sgnw = coord.SIGN[coord.KIND_WHEEL][w[:2]]
            # ★ 轉角在馬達座標系解纏，再乘 sign 換到控制器座標系
            dq = sgnw * (wheel_total[w] - wheel_hold0[w])
            v = dq / dt if dt > 0 else 0.0
            tl = wheel_tau[w]
            tau_m = sum(tl) / len(tl) if tl else 0.0
            # ★ 速度用角度差分，不用 velocity 欄位：2026-08-25 實測後者雜訊 47%，
            #   前者 9.4%，兩者平均幾乎相同 → velocity 無偏但雜訊高
            trk = 100 * v / a.wheel_vel if a.wheel_vel else 0.0
            fric = a.wheel_kd * (a.wheel_vel - v)
            if abs(dq) > 0.05:
                verdict = "✅ 轉了"
            elif abs(tau_m) > 0.05:
                verdict = "⚠️ 有出力但沒轉 → 卡靜摩擦，加 --wheel-tff"
            else:
                verdict = "❌ 力矩幾乎為零 → 指令沒被接受"
            print(f"{w:10s} {dq:+10.4f} {v:+9.4f} {trk:6.0f}% {tau_m:+8.3f}"
                  f" {fric:+9.3f}  {verdict}")
            wheel_rep[w] = {"dq_hold": dq, "v_mean": v, "tau_mean": tau_m,
                            "track_pct": trk, "friction_est": fric,
                            "hold_secs": dt}
        stuck_w = [w for w in shm_io.WHEELS
                   if abs(wheel_rep[w]["dq_hold"]) <= 0.05
                   and abs(wheel_rep[w]["tau_mean"]) > 0.05]
        if stuck_w:
            print(f"\n⚠️ {', '.join(stuck_w)} 有出力但沒轉起來。")
            print("   ★ 這是**靜摩擦掙脫門檻**問題，不是馬達壞掉 ——")
            print("     M2 在 2026-08-25 用同一組參數轉動過全部四顆輪，")
            print("     而輪子的靜摩擦門檻**從來沒被量到過**（只量過動摩擦 0.15）。")
            print("   → 加一點前饋力矩再試，一次加一點：")
            print(f"     --wheel-tff 0.2  （目前 {a.wheel_tff}）")
            print("   ⚠️ tau_ff 掙脫靜摩擦的瞬間會造成速度過衝，不要一次加很大。")

    # ---- 機器可讀的結果檔
    # ★ 為什麼要有：現場的判讀靠終端輸出就夠，但**帶回去複核**不該去 parse
    #   人類格式的表格 —— 那種 parser 遲早會靜靜地讀錯一欄。
    #   分段把這個 JSON scp 回來，本機才做得了 M5 在狗上做不到的檢查
    #   （左右／前後對稱性、同一顆關節在 S2 單腿 vs S4 全身的差異、畫圖）。
    try:
        res = {
            "schema": "m5_run/1",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": logp,
            "args": vars(a),
            "joints": joints,
            "pose_name": pose_name,
            "aborted": bool(abort),
            "abort_reason": abort or None,
            "heartbeat_per_s": (
                (tick_end - tick_start) / loop_elapsed
                if tick_end is not None and loop_elapsed > 0 else None),
            "tmax_used": {j: tmax_of[j] for j in joints},
            "start_ctrl": start_ctrl,
            "target_ctrl": target_ctrl,
            "hold": {}, "peak": {}, "ref": {},
            # ★ 中止前最後 ~0.2 秒的原始取樣。帶回本機才判得出
            #   「真過載」還是「單筆雜訊尖峰」——只有峰值數字是分不出來的。
            "recent": recent,
            "wheels": wheel_rep,
            # 「不可能來自我們指令」的力矩取樣（已排除在 peak.tau 外，peak.tau_raw 含之）
            "bogus_tau": {j: bogus[j] for j in joints if bogus[j]},
            "trace": trace,        # --trace 才有；逐筆 (t, phase, kp, {關節: (q,des,τ,v)})
        }
        by_j2: dict[str, list[dict]] = {}
        for smp in samples:
            by_j2.setdefault(smp["j"], []).append(smp)
        for j in joints:
            ss = by_j2.get(j, [])
            n = len(ss)
            res["hold"][j] = {
                "n": n,
                "q_mean": (sum(x["q"] for x in ss) / n) if n else None,
                "des_mean": (sum(x["des"] for x in ss) / n) if n else None,
                "err_mean": (sum(x["err"] for x in ss) / n) if n else None,
                "tau_mean": (sum(x["tau"] for x in ss) / n) if n else None,
            }
            res["peak"][j] = {**peak[j], "temp_C": st1[idx[j]]["temp_C"]}
            res["ref"][j] = {
                "tau_gravity": ref_torque(ref, pose_name if a.pose else "", j),
                "stiffness": ref_stiffness(ref, j),
                "tau_at_hang": ref_tau_at_hang(ref, j),
                "q_hang": ref_hang_angle(ref, j),
                "predicted_move": (predict_move(ref, j, a.kp, a.delta)
                                   if a.delta is not None else None),
            }
        jp = (logp[:-4] if logp.endswith(".log") else logp) + ".json"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        # sudo 跑的話要 chown 回去，否則 robot 帳號 scp 不走（start_log 同款問題）
        if os.geteuid() == 0 and os.getenv("SUDO_USER"):
            import pwd
            pw = pwd.getpwnam(os.environ["SUDO_USER"])
            try:
                os.chown(jp, pw.pw_uid, pw.pw_gid)
            except OSError:
                pass
        print(f"\n📊 機器可讀結果已存到 {jp}")
    except Exception as e:
        # 結果檔寫不出來不該讓整趟白跑 —— 但一定要說出來，不能靜靜跳過
        print(f"\n⚠️ 機器可讀結果寫入失敗：{type(e).__name__}: {e}")
        print("   終端輸出與 .log 仍然完整，只是回去要人工判讀。")

    # ---------------------------------------------------------------- 阻尼保持與解凍
    print("\n" + "=" * 74)
    print(f"背景保持中：{keeper.label}（持續重寫指令與心跳）")
    _keep_t0 = time.monotonic()

    if aborted_soft:
        print("\n⏸ 現在腿是**被阻尼撐著**的。結束這個保持之後，腿會失力慢慢垂下。")
        print("   請先確認狗的狀態安全（吊帶有沒有纏到腿、有沒有人在下方）。")
        if sys.stdin.isatty():
            try:
                input("\n   確認完畢後按 Enter 結束阻尼保持 ⏎ ")
            except (EOFError, KeyboardInterrupt):
                print("\n   （輸入中斷，直接結束保持）")
        else:
            print(f"\n   非互動模式 → 自動維持 {a.abort_hold:.0f} 秒後結束")
            time.sleep(a.abort_hold)

    keeper.stop()
    # ★ 保持結束後才報告，這個數字才有意義：它證明保持**真的持續在跑**，
    #   而不是只寫了一幀就被 controller 在 500 ms 後清掉。
    _keep_dt = time.monotonic() - _keep_t0
    print(f"\n{keeper.label}結束：{_keep_dt:.1f} 秒內送出 {keeper.ticks} 幀"
          f"（{keeper.ticks / _keep_dt if _keep_dt > 0 else 0:.0f}/s，目標 {a.hz:.0f}/s）"
          + (f"　⚠️ 失敗 {keeper.errors} 次" if keeper.errors else ""))
    if _keep_dt > 1.0 and keeper.ticks < 0.5 * a.hz * _keep_dt:
        print("⚠️ 保持的送出速率遠低於預期 —— 指令可能有段時間處在過期狀態。")

    try:
        shm.close()
    except Exception:
        pass
    try:
        state_ro.close()
    except Exception:
        pass

    if frozen and not aborted_soft:
        os.kill(pid, signal.SIGCONT)
        time.sleep(0.3)
        print(f"\n✅ 已 SIGCONT 解凍 mc_ctrl，狀態={proc_state(pid)}")
    elif frozen:
        print(f"\n⏸ mc_ctrl 仍在凍結中（PID {pid}）。要交還控制權時執行：")
        print(f"      sudo kill -CONT {pid}")
        print("   （前置條件已保證 mc_ctrl 凍結前是洩力狀態，所以解凍本身不會出力）")

    print(f"\n📄 完整輸出已存到 {logp}")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
