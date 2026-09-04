#!/usr/bin/env python3
"""M9 —— 步態（★★★ 狗第一次在地面上移動）。

════════════════════════════════════════════════════════════════════
這一步引入的三件新事情
════════════════════════════════════════════════════════════════════

  1. **連續執行** —— M7/M8 是走一條固定路徑；這裡是幾十秒的週期動作
  2. **★ 動態觸地** —— 今天所有實機資料都是準靜態（足端最快 0.57 m/s），
     基準步態的腳是以 **2.0 m/s** 砸向地面的
  3. **狗會在地面上移動** —— 到 2026-08-27 為止牠從來沒有過
     （所有輪子測試都是墊高離地做的）

★ 所以第一趟是**原地踏步**（`--march` / `d_step=0`）：
  引入 (1)(2) 但不引入 (3)。

════════════════════════════════════════════════════════════════════
兩種模式
════════════════════════════════════════════════════════════════════

  --traj FILE   播放 `inference/gen_gait_traj.py` 產生的軌跡檔。
                ★ **第一趟用這個** —— 每個關節角上機前都檢查過、
                  而且用 `play_gait_traj.py` 在 MuJoCo 完整播過。
  --live        狗上即時算 CPG（`realbot/cpg.py`）。
  --interactive ★★ 三段互動（2026-09-03）：
                  站起來 →〔Enter〕→ 往前走 →〔Enter〕→ 原地停 →〔Enter〕→ crouch 後趴下
                步態改成**在狗上即時算**（走多久由現場決定），乾跑的所有檢查照舊。
                ⚠️ 等待期間**不能用 input()** —— 主迴圈一旦阻塞超過 500 ms，
                   controller 會判定指令過期把指令區清成 0，承重中的狗就塌了。
                   所以用 `KeyWatch`（select 輪詢，不阻塞）。
                ⚠️ 三個等待都有逾時（--hold-max），逾時是**自動往下走**不是中止 ——
                   人一分心狗就一直承重站著，而腿的發熱從沒量過。
                ★ 之後要接**偏航閉迴路**只能走這條路 ——
                  文件記載唯一擋住「能直線走遠」的是偏航慢漂 −0.5~−0.8°/s，
                  播放固定檔案永遠解不了。
                實測純 Python 的 CPG+IK 是 0.043 ms/週期（預算 20 ms 的 0.2%）。

兩種模式的輸出必須一致 —— `tests/test_cpg_port.py` 逐幀比到 1e-11 rad。

════════════════════════════════════════════════════════════════════
⚠️ 保護門檻和 M7/M8 **不一樣**，不要照搬
════════════════════════════════════════════════════════════════════

步態的擺動相本來就需要 **6~8 rad/s** 的關節命令速度（M7/M8 的護欄是 2.0）。
所以 `--vcmd-max` / `--vmax` 都必須放寬 —— 但**放寬的是速度，不是力矩**。
力矩門檻維持 M8 的量級（實機承重峰值只用掉上限的 19%）。

★★ **中止語意與 M7/M8 相同**（承重）：凍結目標角、維持增益、原地撐住。
   放手＝狗塌下去。**不要按 Ctrl-C。**

════════════════════════════════════════════════════════════════════
前置條件
════════════════════════════════════════════════════════════════════
  - 狗趴在地上、16 顆洩力
  - **吊帶掛在 crouch（292 mm）以下**、鬆弛
  - ★ **地面淨空，而且要留出走動空間**（原地踏步在模擬裡仍會漂 ~10-400 mm）
  - 第二個終端機備著 `sudo ~/estop_max.sh`

用法：
    python3 M9_gait.py --traj march.json                    # 乾跑
    sudo python3 M9_gait.py --traj march.json --confirm
    sudo python3 M9_gait.py --live --march --secs 10 --confirm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import coord
import cpg
import kin
import shm_io
from M5_leg_pose import Keepalive, mc_ctrl_pid, proc_state, smoothstep
from M7_standup import TAU_HARD, read_imu_rp

# 起身／坐下的輪阻尼。0.5 = trip16 實機驗證值；M7 只掃過 0.1–1.0。
WHEEL_KD_SAFE = 0.5


class ChatterWatch:
    """輪子抖振偵測（2026-09-03 trip17 事故後加）。

    事故簽名：輪速 ~64 Hz 正負翻轉、|v| 峰 14 rad/s（正常滾動是單向、< 2 rad/s）。
    判準：**高幅度的符號翻轉**連續累積 —— 每次相鄰兩筆 v 反號且兩者 |v| 都 > V_MIN
    記一分，同方向或低速就衰減。分數到門檻＝抖振。
    正常走路輪子單向滾，不會累積；雜訊（47%）幅度小，過不了 V_MIN。
    """

    V_MIN = 4.0        # rad/s。trip16 正常峰值 1.33、事故時 10–14
    LIMIT = 12         # 約 0.1 秒的持續抖振（200 Hz 下 ~24 筆、翻轉一半）

    def __init__(self):
        self.prev: dict = {}
        self.score: dict = {}

    def feed(self, name: str, v: float) -> bool:
        """回傳 True = 抖振確立。"""
        pv = self.prev.get(name, 0.0)
        self.prev[name] = v
        sc = self.score.get(name, 0)
        if v * pv < 0 and abs(v) > self.V_MIN and abs(pv) > self.V_MIN:
            sc += 2
        else:
            sc = max(0, sc - 1)
        self.score[name] = sc
        return sc >= self.LIMIT

LEGS12 = [lg + k for lg in coord.LEGS for k in coord.LEG_KINDS]

# 力矩門檻。與 M8 同量級 —— 實機承重峰值只用掉馬達上限（150）的 19%，
# 模擬的步態峰值是 38 N·m。
TMAX = {"1_hip_roll": 50.0, "2_hip_pitch": 50.0, "3_knee_pitch": 70.0}

# 站起來用的增益與路徑（照抄 M7 已實機驗證的）
STANDUP_KP, STANDUP_KD = 250.0, 5.0

# 判定「孤立反號跳點」時忽略的小訊號。低於這個值的力矩本來就會在 0 附近抖，
# 反號沒有意義，也不可能是峰值。
SPIKE_FLOOR = 5.0


def commit_peak(win: list, peak: dict, spikes: dict, j: str,
                over: dict | None = None) -> None:
    """把三點視窗的**中間**那筆提交進 `peak[j]`，先擋掉孤立反號跳點。

    `over` 有給的話，順便累計「濾掉跳點後仍然超過門檻」的筆數
    —— 那是 `--tau-total` 這條慢速中止路徑的輸入，見 main() 的說明。

    ════════════════════════════════════════════════════════════════
    為什麼要延遲一筆
    ════════════════════════════════════════════════════════════════

    2026-09-02 第 1 趟（trip10）收工印出 `fl3_knee_pitch 峰值 -51.52`，
    嚇了一跳 —— 那是**假的**。去 JSON 看它前後兩筆：

        t=16.110  tau +27.77  v -0.908
        t=16.115  tau -51.52  v -1.838   ← 只有這一筆反號，v 也一起跳
        t=16.120  tau +28.01  v -1.048

    整趟 74,400 個樣本裡只有 2 筆這種東西（0.003%）。剔掉之後 12 顆關節
    全部落在模擬的 81–114%，`fl3` 的真值是 +37.5（模擬 36.4）。

    ★★ **原本的 `kp·|err| + kd·|v|` 判別式擋不住它** —— 因為判別式用的就是
    **同一筆被汙染的 `v`**：`v` 被灌大 → 上限被灌大（34.3，×1.5+1 = 52.5）
    → 51.52 剛好通過。**髒資料自己給自己開了門。**
    這是「多印一個可互相對照的量」那套方法的一個盲區：兩個量來自同一筆讀值時，
    它們不獨立，對照不出東西來。**鄰居才是獨立的。**

    所以改成看前後鄰居。而鄰居要等下一筆才有 —— 這就是延遲一筆的原因。

    ════════════════════════════════════════════════════════════════
    ⚠️ 這個過濾器可以用在哪條中止路徑（2026-09-02 追加，讀完再改）
    ════════════════════════════════════════════════════════════════

    **快速路徑（`--tau-hits` 連續 N 筆、`TAU_HARD` 連續 2 筆）：不可以用。**
    那條路要在「持續超載」發生的當下就跳，延遲一筆就是延遲一拍；
    而且真實的連續超標會被「第一筆還沒判完」吃掉。
    它本來就對單筆免疫（要連續），不需要這個過濾器。

    **慢速路徑（`--tau-total` 整趟累計）：可以，而且必須用。**
    那條路本來就要累積好幾秒才觸發，慢一拍毫無影響；
    反過來說，不過濾的話兩三筆感測跳點就會把累計值灌上去（trip13 的 `fl1`
    就有 2 筆假的 84.5 / 80.9 都在 50 門檻之上）。

    ★ **同一個過濾器，在兩條路徑上的取捨相反 —— 不要一體適用。**
    """
    if len(win) < 3:
        return
    (t0, _), (t1, cap1), (t2, _) = win[-3:]
    if (abs(t1) > SPIKE_FLOOR and t1 * t0 < 0 and t1 * t2 < 0
            and abs(t1) > max(abs(t0), abs(t2))):
        spikes[j] += 1
        return
    if over is not None and abs(t1) > TMAX[j[2:]]:
        over[j] += 1
    if abs(t1) <= 1.5 * cap1 + 1.0 and abs(t1) > abs(peak[j]):
        peak[j] = t1


def report_peaks(tau_win: dict, peak: dict, spikes: dict,
                 over: dict | None = None,
                 over_raw: dict | None = None) -> None:
    """收尾補提交最後一筆，然後印峰值表。

    ★★ **這段跑在 `keeper.start()` 之後。** `Keepalive` 是 daemon 執行緒 ——
    主執行緒一拋例外，process 就結束、心跳跟著停，指令區約 0.5 秒後被清零，
    **狗會在站姿失力**。所以這裡的任何一行都不允許炸。

    抽成函式的唯一理由就是**讓它可以被測試**（`test_report_peaks_never_raises`）——
    原本內嵌在 `main()` 裡，只有真跑到最後才會第一次執行，
    等於把「第一次執行」排在狗承重的時候。
    """
    # 最後一筆永遠當不成「中間點」，補提交（沒有下一筆，只能做上限檢查）。
    # ⚠️ 中止時最後一筆正是最可疑的那筆，所以照樣不能無條件收 —— 但也不能丟，
    #    丟了會把「真的撞上去」的峰值藏起來。用上限檢查是兩害相權。
    for j in LEGS12:
        w = tau_win.get(j) or []
        if w:
            t_, cap_ = w[-1]
            if abs(t_) <= 1.5 * cap_ + 1.0 and abs(t_) > abs(peak[j]):
                peak[j] = t_

    over = over or {}
    over_raw = over_raw or {}
    print(f"\n{'關節':16s} {'峰值τ':>9s} {'門檻':>7s} {'用掉':>7s}"
          f" {'★超標筆數':>10s} {'(未濾)':>7s} {'剔除跳點':>9s}")
    for j in LEGS12:
        lim = TMAX[j[2:]]
        o = over.get(j, 0)
        print(f"{j:16s} {peak[j]:+9.2f} {lim:7.0f} {100*abs(peak[j])/lim:6.0f}%"
              f" {o:10d} {over_raw.get(j, 0):7d} {spikes.get(j, 0):9d}"
              + ("  ⚠️" if o else ""))
    n_over = sum(over.values())
    if n_over:
        print(f"  ⚠️⚠️ 有 {n_over} 筆超過門檻。**這一欄以前沒有，trip13 就是因此漏看的** ——")
        print(f"     當時 bl3 整趟 13 筆超標（峰值 114%），但收工表只印峰值，")
        print(f"     而最長連續只有 2 筆、沒到中止門檻，所以畫面上完全看不出來。")
    n_spike = sum(spikes.values())
    if n_spike:
        print(f"  ★ 共剔除 {n_spike} 筆孤立反號跳點（單筆、與前後兩筆都反號且更大）。"
              f"\n    正常量級是每趟 0~3 筆；**上到十位數就不是感測雜訊，要查 SHM 解碼**。"
              f"\n    原始值都還在 JSON 的 `samples` 裡，沒有被改掉。")


def phase_gains(nm: str, r: float, gait_kp: float, gait_kd: float) -> tuple:
    """回傳 (kp, kd)。kd 用**和 kp 相同的插值比例**，
    這樣切換過程中阻尼比 kd/kp 不會亂跑（先降 kp 再降 kd 會短暫過阻尼，反之欠阻尼）。"""
    kp = phase_kp(nm, r, gait_kp)
    if gait_kp <= 0 or abs(gait_kp - STANDUP_KP) < 1e-9:
        return kp, (gait_kd if nm in ("GAIT", "GAIT_IN", "GAIT_OUT")
                    else STANDUP_KD)
    f = (kp - STANDUP_KP) / (gait_kp - STANDUP_KP)          # 0=站立值 1=步態值
    f = 0.0 if f < 0 else (1.0 if f > 1 else f)
    return kp, STANDUP_KD + (gait_kd - STANDUP_KD) * f


def phase_kp(nm: str, r: float, gait_kp: float) -> float:
    """各階段用哪個 kp。`r` = 這一段走了幾成（0~1）。

    ════════════════════════════════════════════════════════════════
    ★★ 為什麼站起來和步態要用不同的 kp（2026-09-02 修）
    ════════════════════════════════════════════════════════════════

    原本整支程式（含站起來、承重、坐回去）全部跑 `a.kp` ——
    `STANDUP_KP = 250` 定義了卻**從來沒被用過**，是死碼。

    在 kp 一直是 250 的時候這沒差。但 2026-09-02 的 M6 錄製顯示
    **原廠是按模式切換增益的**：

        靜止站立      ABAD/HIP/KNEE 全部 250 / kd 5.0
        任何動作      ABAD 60 / HIP 120 / KNEE 120 / kd 1.0

    我們要學原廠、把步態降到 kp=120，於是這個死碼就變成安全問題：
    **`--kp 120` 會讓狗用沒測過的增益從趴姿站起來**，
    而 M7 只在 kp=250 驗證過站立（四趟）。承重站起來是整個序列風險最高的一段，
    不該拿它來試新增益。

    所以：**站起來／坐回去用 `STANDUP_KP`（M7 驗證過的 250），
    只有 `GAIT` 那段用 `--kp`。**

    ⚠️ 切換不能是階躍 —— 承重中 kp 突然砍半，撐住機身的力矩也砍半，
    狗會掉下去（靜態撓度 8.9 mm @250 → 約 18 mm @120）。
    所以在 `HOLD_stand`（進步態前的靜止段）把 kp 從 250 平滑降到步態值，
    在 `BACK_crouch` 的前半把它升回來。**降的時候狗是靜止的，升的時候是往上收，
    兩個方向都是安全的那一邊。**
    """
    if nm == "RAMP_UP":
        return STANDUP_KP * r
    if nm == "RAMP_DOWN":
        return STANDUP_KP * max(0.0, 1.0 - r)
    if nm == "HOLD_stand":                 # 進步態前：250 → 步態值
        return STANDUP_KP + (gait_kp - STANDUP_KP) * min(1.0, r)
    if nm == "BACK_crouch":                # 出步態後：前半段升回 250
        return gait_kp + (STANDUP_KP - gait_kp) * min(1.0, r / 0.5)
    if nm == "GAIT":
        return gait_kp
    # ── 互動模式（--interactive）的階段。
    #   ★ 這裡的關鍵是「降 kp 的時候狗必須是靜止的」——固定時間表版本靠
    #     `HOLD_stand` 那一段做，但互動版的 READY 是**不定長**的（等人按鍵），
    #     不能拿它做增益過渡，否則人等多久 kp 就停在中間值多久。
    #     所以另外切出 KP_DOWN / KP_UP 兩段固定長度、狗靜止的過渡。
    if nm == "KP_DOWN":                    # READY 之後：250 → 步態值
        return STANDUP_KP + (gait_kp - STANDUP_KP) * min(1.0, r)
    if nm == "KP_UP":                      # 停下之後：步態值 → 250
        return gait_kp + (STANDUP_KP - gait_kp) * min(1.0, r)
    if nm in ("GAIT_IN", "GAIT_OUT"):
        return gait_kp
    return STANDUP_KP


class KeyWatch:
    """非阻塞地等一次 Enter。

    ⚠️⚠️ **這就是互動模式最危險的地方。** 主迴圈必須以 `--hz`（200 Hz）持續寫
    `joint_cmd` 並把 `joint_state` 的時戳抄過去；一旦阻塞超過
    `joint_cmd_timeout`（500 ms），controller 判定指令過期會**把指令區清成 0**
    —— 承重狀態下那就是狗直接塌下去。所以**絕對不能用 `input()`**。

    這裡用 `select` 以 0 逾時輪詢 stdin，每次主迴圈 tick 呼叫一次，不阻塞。
    用 Enter 而不是單鍵，是為了不動 termios —— 改 raw mode 之後若程式異常結束，
    使用者的終端機會壞掉，而現場正需要那個終端機下急停指令。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.stdin.isatty()

    def pressed(self) -> bool:
        """有沒有按下 Enter（消費掉一整行）。非阻塞。"""
        if not self.enabled:
            return False
        try:
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                return False
            sys.stdin.readline()
            return True
        except Exception:
            return False


class GaitStream:
    """即時算 CPG 的步態流 —— 互動模式用，長度不定。

    ★ 為什麼不能直接播軌跡檔：檔案是固定長度的，而「走到我喊停」長度不定。
    ★ 為什麼不能每個 200 Hz tick 都推進 CPG：**那會改變積分步長**
      （CPG 是以 50 Hz 的 `CTRL_DT` 驗證的），跑出來就不是離線驗過的那個步態了。
      所以這裡維持 **50 Hz 推進 + 線性內插到寫入頻率**，與 `--traj` 路徑相同。

    `tests/test_m9_gait.py::test_gait_stream_matches_offline_trajectory`
    拿離線軌跡當外部對照逐幀比對。
    """

    GAIT_DT = 0.02          # 50 Hz，與 max_model.CTRL_DT 相同

    def __init__(self, p: dict, f0: dict, ks: dict):
        self.p = p
        self.f0, self.ks = f0, ks
        self.phase = cpg.PHASES[p.get("seq", "ds")]
        self.x_off = cpg.x_off_split(p["x_off"], p.get("x_d", 0.0))
        self.sway_p = (p.get("sway_x", 0.0), p.get("sway_y", 0.0),
                       p.get("sway_lead_x", 0.0), p.get("sway_lead_y", 0.0))
        self.c = cpg.init(self.phase)
        self.mux = {l: p["mu_x"] for l in cpg.LEGS}
        self.muy = {l: p["mu_y"] for l in cpg.LEGS}
        self.om = {l: p["omega"] for l in cpg.LEGS}
        self.step = cpg.make_step(self.phase)
        self.n_clamp = 0
        self.t_next = 0.0
        self.q_prev = self.q_next = self._compute()

    def _compute(self) -> dict:
        sway = None
        if self.sway_p[0] or self.sway_p[1]:
            sway = cpg.body_sway(cpg.gait_phase(self.c["theta"], self.phase),
                                 *self.sway_p)
        q, ncl = cpg.joint_targets(self.c, self.f0, self.ks, self.x_off,
                                   self.p["g_c"], self.p["d_step"],
                                   self.p["d_step_y"], self.p["duty"],
                                   self.p["z_sag"], sway)
        self.n_clamp += ncl
        return q

    def sample(self, t: float) -> dict:
        """步態開始後 t 秒的 12 個關節目標。t 必須單調不減。"""
        while t >= self.t_next:
            self.q_prev = self.q_next
            self.c = self.step(self.c, self.mux, self.muy, self.om, self.GAIT_DT)
            self.q_next = self._compute()
            self.t_next += self.GAIT_DT
        f = 1.0 - (self.t_next - t) / self.GAIT_DT
        f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
        return {j: self.q_prev[j] + f * (self.q_next[j] - self.q_prev[j])
                for j in self.q_prev}


class InteractivePlan:
    """互動式三段流程的狀態機：站起來 →〔Enter〕→ 走 →〔Enter〕→ 停 →〔Enter〕→ 趴下。

    與固定時間表（`bounds`）的差別只在「階段怎麼推進」，**保護邏輯完全共用** ——
    這條線最痛的教訓就是同一件事兩份實作，所以主迴圈只有一個。

    階段（`dur=None` 代表等按鈕）：

        RAMP_UP → GO_crouch → HOLD_crouch → GO_stand
        ★ READY      站著等 Enter ①
        GAIT_IN → GAIT（等 Enter ②）→ GAIT_OUT
        ★ STOPPED    站著等 Enter ③
        BACK_crouch → HOLDB_crouch → BACK_LIE → RAMP_DOWN

    ⚠️ 三個等待階段都有各自的逾時（`hold_max`），步態有 `walk_max`。
      沒有逾時的話，人一分心狗就一直承重站著 —— 而**腿吃 41 kg 的發熱從沒量過**
      （M7/M8 最長只撐 34.6 秒）。逾時不是中止，是自動走到下一階段。
    """

    WAIT = {"READY", "GAIT", "STOPPED"}

    def __init__(self, a, q_lie: dict, q_stand: dict, gs: "GaitStream"):
        self.a, self.gs = a, gs
        self.q_lie, self.q_stand = q_lie, q_stand
        crouch = dict(coord.POSES["crouch"])
        # (名稱, 時長 or None, 起點, 終點)
        # ⚠️ 坐下段刻意**不叫** BACK_crouch —— 那個名字在 `phase_kp` 有
        #   「從步態 kp 升回 250」的專屬分支，而互動版在 KP_UP 已經升完了，
        #   沿用會讓 kp 先階躍掉回 120 再升回去。
        self.segs = [
            ("RAMP_UP", a.ramp_kp, q_lie, q_lie),
            ("GO_crouch", a.t1, q_lie, crouch),
            ("HOLD_crouch", a.hold_mid, crouch, crouch),
            ("GO_stand", a.t2, crouch, q_stand),
            ("READY", None, q_stand, q_stand),          # ① 等 Enter（kp 250）
            ("KP_DOWN", a.kp_shift, q_stand, q_stand),  # 靜止中 250 → 步態 kp
            ("GAIT_IN", a.ramp, q_stand, None),
            ("GAIT", None, None, None),                 # ② 等 Enter
            ("GAIT_OUT", a.ramp, None, q_stand),
            ("KP_UP", a.kp_shift, q_stand, q_stand),    # 靜止中 步態 kp → 250
            ("STOPPED", None, q_stand, q_stand),        # ③ 等 Enter
            ("SIT_crouch", a.t2, q_stand, crouch),
            ("SIT_hold", a.hold_mid, crouch, crouch),
            ("SIT_LIE", a.t1, crouch, q_lie),
            ("RAMP_DOWN", a.ramp_kp, q_lie, q_lie),
        ]
        self.i = 0
        self.t_seg = 0.0          # 目前階段的起始時刻
        self.t_gait0 = None       # 步態（含淡入）的起始時刻
        self.u_out0 = 1.0         # 進 GAIT_OUT 時的步態混合比例（見 _des）
        self.notes: list = []     # 給 log 的階段轉換紀錄

    @property
    def name(self) -> str:
        return self.segs[self.i][0]

    def _limit(self) -> float:
        """目前等待階段的逾時秒數。"""
        return self.a.walk_max if self.name == "GAIT" else self.a.hold_max

    def _advance(self, t: float, why: str, dur=None) -> None:
        """推進到下一階段。

        ⚠️ `dur` 有值（＝定長階段自然結束）時，新階段的起點要用
        **`t_seg + dur`** 而不是當下的 `t` —— 用 `t` 的話每次推進都把誤差歸零，
        單一 tick 就跨不過第二個階段，狀態機會愈落愈後。
        等待階段沒有預定長度，才用當下的 `t`。
        """
        self.notes.append((round(t, 2), self.name, why))
        print(f"\n  ▸ {self.name} 結束（{why}），進入 {self.segs[self.i + 1][0]}\n")
        self.i += 1
        self.t_seg = t if dur is None else self.t_seg + dur
        if self.segs[self.i][0] == "GAIT_IN":
            self.t_gait0 = t

    def update(self, t: float, key: bool):
        """回傳 (階段名, 12 關節目標, kp, kd, 是否結束)。

        ⚠️ 用**迴圈**推進，不是一次只推一階 —— 主迴圈若卡頓一下（實測最長間隔
        13–31 ms，但 controller 逾時是 500 ms），單一 tick 可能跨過整個短階段。
        一次只推一階的話狀態機會愈落愈後，而畫面上完全看不出來。
        """
        for _ in range(len(self.segs) + 1):        # 上限：最多跨完整條流程
            nm = self.name
            el = t - self.t_seg
            dur = self.segs[self.i][1]
            if nm in self.WAIT:
                if key:
                    key = False                    # 一次按鍵只推一個等待階段
                    self._advance(t, "按下 Enter")
                    continue
                if el >= self._limit():
                    self._advance(t, f"逾時 {self._limit():.0f} s（自動繼續）")
                    continue
                break
            # ★ 起步淡入期間也要能喊停 —— 使用者的語意是「隨時」，而
            #   `GAIT_IN` 不在等待集合裡，原本會被當成非等待階段忽略掉按鍵，
            #   等於**起步後有 --ramp 秒按不了停**。
            #   直接跳到 GAIT_OUT，並從**當下的混合比例**開始降（不是從 1.0），
            #   否則會先跳回全步態再淡出 —— 那是反方向的突變。
            if key and nm == "GAIT_IN":
                self.u_out0 = smoothstep(min(el / dur, 1.0))
                i_out = next(k for k, sg in enumerate(self.segs)
                             if sg[0] == "GAIT_OUT")
                self.notes.append((round(t, 2), nm, "淡入中按停"))
                print(f"\n  ▸ {nm} 中途按停（混合比例 {self.u_out0:.2f}），"
                      f"直接進入 GAIT_OUT\n")
                self.i = i_out
                self.t_seg = t
                # 淡出時間按比例縮短：只淡入到 0.4 就沒必要花整個 ramp 淡出
                self.segs[i_out] = ("GAIT_OUT",
                                    max(0.5, self.a.ramp * self.u_out0),
                                    None, self.q_stand)
                key = False
                continue
            if el >= dur:
                if self.i + 1 >= len(self.segs):   # 最後一段跑完 = 收工
                    nm, dur = self.segs[self.i][0], self.segs[self.i][1]
                    return (nm, self._des(t, 1.0), *self._gains(nm, 1.0), True)
                self._advance(t, "時間到", dur)
                continue
            break
        else:
            print("  ⚠️ 階段推進超過上限 —— 檢查是不是有 0 秒的階段")

        nm = self.name
        dur = self.segs[self.i][1]
        r = 1.0 if dur in (None, 0) else min((t - self.t_seg) / dur, 1.0)
        return (nm, self._des(t, r), *self._gains(nm, r), False)

    def _des(self, t: float, r: float) -> dict:
        nm, dur, p0, p1 = self.segs[self.i]
        if nm in ("GAIT_IN", "GAIT", "GAIT_OUT"):
            qg = self.gs.sample(t - self.t_gait0)
            if nm == "GAIT":
                return qg
            u = (smoothstep(r) if nm == "GAIT_IN"
                 else self.u_out0 * smoothstep(1.0 - r))
            return {j: (1 - u) * self.q_stand[j] + u * qg[j] for j in qg}
        u = smoothstep(r)
        return {j: p0[j] + u * (p1[j] - p0[j]) for j in p0}

    def _gains(self, nm: str, r: float):
        kp, kd = phase_gains(nm, r, self.a.kp, self.a.kd)
        return kp, kd, phase_kp(nm, r, self.a.kp_abad)


def build_standup(a, q_lie: dict, q_gait0: dict):
    """趴 → crouch → 步態起點。回傳 (名稱, 秒數, 起點, 終點)。"""
    crouch = dict(coord.POSES["crouch"])
    return [("RAMP_UP", a.ramp_kp, q_lie, q_lie),
            ("GO_crouch", a.t1, q_lie, crouch),
            ("HOLD_crouch", a.hold_mid, crouch, crouch),
            ("GO_stand", a.t2, crouch, q_gait0),
            ("HOLD_stand", a.hold, q_gait0, q_gait0)]


def build_sitdown(a, q_gait_end: dict, q_lie: dict):
    crouch = dict(coord.POSES["crouch"])
    return [("BACK_crouch", a.t2, q_gait_end, crouch),
            ("HOLDB_crouch", a.hold_mid, crouch, crouch),
            ("BACK_LIE", a.t1, crouch, q_lie),
            ("RAMP_DOWN", a.ramp_kp, q_lie, q_lie)]


def gain_mismatches(D: dict, kp: float, kd: float, wheel_kd: float) -> list:
    """（相容用）只比三個增益。新程式請用 `param_mismatches`。"""
    return param_mismatches(D, {"kp": kp, "kd": kd, "wheel_kd": wheel_kd})


# 「說兩次」要比對的欄位。★ 只要它**會改變送給馬達的角度或力矩**，就必須在這裡。
#   數值欄位比值，字串欄位（seq）比字面。
_NUM_FIELDS = ("kp", "kp_abad", "kd", "wheel_kd", "x_off", "x_d", "g_c", "z_sag",
               "sway_x", "sway_y", "sway_lead_x", "sway_lead_y")
_STR_FIELDS = ("seq",)


def param_mismatches(D: dict, want: dict) -> list:
    """軌跡檔的參數 vs 命令列的參數，回傳 `[(欄位, 檔案值 or None, 命令列值)]`。

    「說兩次」防呆的比對本體。**抽成函式是為了可測** —— `main()` 在讀軌跡檔之前
    就會先開 `/dev/shm`，所以這道防呆在狗以外的機器上跑不到，只能單獨測它。

    ⚠️ **每一個會改變輸出角度的參數都要比。** 歷史教訓：
    最早只比 `kp`/`kd`，漏了 `wheel_kd` —— 用 `--wheel-kd 3.0` 產的檔、跑的時候
    忘了帶旗標就**靜默用回 0.5**，而 0.5 正是「前腳不跨步」那組；症狀是
    「模擬明明好了、實機還是老樣子」，**而所有診斷指標都乾淨**。
    2026-09-03 新增的 `seq`（DS/LS 相位序列）與 `sway_*` 是同一類：
    忘了帶 `--seq ls` 會靜默跑回 diagonal sequence，外觀幾乎一樣但那是另一個步態。

    `want` 裡沒有的欄位不比（例如純播放模式不需要比 live 專屬的參數）。
    """
    bad = []
    for k, v in want.items():
        if v is None:
            continue
        got = D.get(k)
        if k in _STR_FIELDS:
            if got is None or str(got) != str(v):
                bad.append((k, got, v))
        elif got is None or abs(float(got) - float(v)) > 1e-9:
            bad.append((k, got, v))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="M9 —— 步態（承重、連續、動態觸地）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traj", help="播放軌跡檔（gen_gait_traj.py 產生）")
    src.add_argument("--live", action="store_true", help="狗上即時算 CPG")
    ap.add_argument("--confirm", action="store_true", help="不帶就是乾跑")

    # ---- live 模式的步態參數
    ap.add_argument("--march", action="store_true", help="★ 原地踏步：d_step=0")
    ap.add_argument("--secs", type=float, default=10.0, help="步態本體秒數")
    ap.add_argument("--omega", type=float, default=0.5,
                    help="★ 從 0.5 開始。基準是 1.4，但那是 2.0 m/s 的觸地速度")
    ap.add_argument("--duty", type=float, default=0.80)
    ap.add_argument("--d-step", type=float, default=0.0, dest="d_step")
    ap.add_argument("--x-off", type=float, default=0.04, dest="x_off",
                    help="★ 原地踏步在 kp=250 的配平點約 +0.04（基準的 −0.04 是 "
                         "kp=120 + 前進步態掃出來的，符號相反）")
    ap.add_argument("--g-c", type=float, default=0.12, dest="g_c",
                    help="★ 不要調小！g_c=0.04 時實際離地只有 4.5 mm，"
                         "腳被地面拖著走 —— 那是文件裡的災難情境")
    ap.add_argument("--seq", choices=("ds", "ls", "trot"), default="ds",
                    help="★★ 相位序列。**ds 是舊的（diagonal sequence）**，"
                         "ls 才是文獻上靜態穩定裕度最好的 lateral sequence。"
                         "2026-09-03 前註解一直把 ds 誤標成 ls，見 docs/E_*。"
                         "⚠️ 忘了帶會靜默跑回 ds —— 已納入「說兩次」比對")
    ap.add_argument("--x-d", type=float, default=0.0, dest="x_d",
                    help="前後差動的足端偏移（前腿 +x_d、後腿 −x_d）＝軸距量。"
                         "0 就是四腿共用 --x-off。姿態前後對稱 ⟺ --x-off 為 0")
    ap.add_argument("--sway-x", type=float, default=0.0, dest="sway_x",
                    help="★ body sway 縱向幅度 m（後腿擺動時機身往前）。"
                         "建議組 0.015。⚠️ 必須配 --sway-lead-x")
    ap.add_argument("--sway-y", type=float, default=0.0, dest="sway_y",
                    help="★ body sway 橫向幅度 m（左腿擺動時機身往右）。建議組 0.010")
    ap.add_argument("--sway-lead-x", type=float, default=0.0, dest="sway_lead_x",
                    help="★ 縱向相位提前（週期比例）。建議組 0.90。"
                         "⚠️ 留 0 會讓 sway 變成純擾動 —— 機身要在腿抬起**之前**移好")
    ap.add_argument("--sway-lead-y", type=float, default=0.0, dest="sway_lead_y",
                    help="★ 橫向相位提前。建議組 0.20。縱向是二倍頻，"
                         "所以兩軸的 lead **不能共用一個值**")
    ap.add_argument("--d-step-y", type=float, default=0.12, dest="d_step_y")
    ap.add_argument("--mu-x", type=float, default=1.80, dest="mu_x")
    ap.add_argument("--mu-y", type=float, default=1.50, dest="mu_y")
    ap.add_argument("--z-sag", type=float, default=None, dest="z_sag",
                    help="★ 預設 **0.036×250/kp**（實機錨點）。不是模擬的 STATIC_SAG")
    ap.add_argument("--ramp", type=float, default=3.0, help="站姿↔步態的淡入淡出秒數")

    # ---- ★ 互動模式（2026-09-03）
    ap.add_argument("--interactive", action="store_true",
                    help="★★ 三段互動：站起來→〔Enter〕→往前走→〔Enter〕→原地停→"
                         "〔Enter〕→crouch 後趴下。步態改成**狗上即時算 CPG**"
                         "（走多久由現場決定），乾跑的所有檢查照舊")
    ap.add_argument("--hold-max", type=float, default=25.0, dest="hold_max",
                    help="★ 每個等待階段的逾時（秒）。逾時不是中止，是自動往下走。"
                         "⚠️ 等待中狗是承重站著的，而**腿吃 41 kg 的發熱從沒量過**")
    ap.add_argument("--walk-max", type=float, default=20.0, dest="walk_max",
                    help="★ 步態段的逾時（秒）。忘了按 Enter 時的保險")
    ap.add_argument("--kp-shift", type=float, default=1.5, dest="kp_shift",
                    help="增益過渡秒數（250↔步態 kp）。★ 這段狗是靜止的 —— "
                         "承重中 kp 階躍砍半會讓機身掉下去")

    # ---- 增益
    ap.add_argument("--kp", type=float, default=250.0)
    ap.add_argument("--kp-abad", type=float, default=60.0, dest="kp_abad",
                    help="★★ 步態段 ABAD 的 kp（hip/knee 用 --kp）。"
                         "原廠 RL 與**所有模擬驗證**都是 [60,120,120]，"
                         "但 2026-09-03 之前 M9 把 --kp 套到全部 12 關節 —— "
                         "實機 ABAD 因此比模擬硬一倍，重現實驗證實這正是"
                         "「前腳不跨＋劇烈搖擺」兩趟失敗的根因（俯仰 4.7→11.4°、"
                         "執行率 0.40→0.00）。站起來／坐下不受影響（原廠站立 ABAD 也是 250）")
    ap.add_argument("--kd", type=float, default=5.0)
    ap.add_argument("--wheel-kd", type=float, default=0.5, dest="wheel_kd",
                    help="★ 輪子純阻尼，kp 恆為 0。設定檔的 FSM_RL_Wheel_Kp=60 "
                         "是配「每步重給目標角」的 RL，開迴路套上去偏航失控 +39°/12s")
    ap.add_argument("--tau-total", type=int, default=5, dest="tau_total",
                    help="★★ 慢速力矩保護：整趟**累計**超過門檻的筆數（已濾掉孤立"
                         "跳點）達到這個數就中止。0 = 關閉。"
                         "預設 5 —— trip10/11/12 全部 ≤1，trip13（ω1.0）的 "
                         "bl3 是 13、fr3 是 5。⚠️ 這條和 --tau-hits（連續筆數）"
                         "是兩條獨立的路徑：連續的防持續超載，累計的防衝擊尖峰")

    # ---- 時序
    ap.add_argument("--ramp-kp", type=float, default=2.0, dest="ramp_kp")
    ap.add_argument("--t1", type=float, default=1.5)
    ap.add_argument("--t2", type=float, default=1.5)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--hold-mid", type=float, default=1.5, dest="hold_mid")
    ap.add_argument("--hz", type=float, default=200.0)
    ap.add_argument("--max-secs", type=float, default=60.0, dest="max_secs",
                    help="★ 總承重時間上限。mc_ctrl 凍結已驗到 200 秒，"
                         "所以這管的是承重 —— 腿吃 41 kg 的發熱沒量過")

    # ---- 保護（⚠️ 和 M7/M8 不同）
    ap.add_argument("--emax", type=float, default=0.60,
                    help="追蹤誤差 rad。⚠️ 預設 0.6 是照 **kp=250** 的擺動落後訂的"
                         "（M8 實測 52 mm ≈ 0.3 rad）；**kp=120 的設計落後就是 0.6**"
                         "（M8：108 mm）—— 2026-09-03 三趟步態全在 0.60~0.62 被它誤殺，"
                         "每一步的膝都規律地衝 0.45~0.55。kp=120 的步態請帶 --emax 0.8。"
                         "真正的安全靠力矩三條路（連續/累計/硬上限），不靠這條")
    ap.add_argument("--vmax", type=float, default=16.0,
                    help="★ 量測關節速度 rad/s。**基準步態（ω=1.4）的命令峰值實測 "
                         "13.1 rad/s** —— M7/M8 的 4.0 會在第一個擺動相就誤中止。"
                         "馬達規格 190 RPM = 19.9 rad/s，16 是它的 80%")
    ap.add_argument("--vcmd-max", type=float, default=14.0, dest="vcmd_max",
                    help="★ 命令速度 rad/s（乾跑檢查）。基準步態要 13.1")
    ap.add_argument("--tilt-max", type=float, default=20.0, dest="tilt_max")
    ap.add_argument("--temp-max", type=float, default=70.0, dest="temp_max")
    ap.add_argument("--tau-hits", type=int, default=3, dest="tau_hits")
    ap.add_argument("--gap-max", type=float, default=0.25, dest="gap_max",
                    help="單次迴圈間隔上限。controller 逾時 500 ms → 超過就清零")
    a = ap.parse_args()

    if a.march:
        a.d_step = 0.0
    if a.z_sag is None:
        # ★★ 實機錨點（M8 S3）：kp250 → 36 mm、kp120 → 72 mm，正比於 1/kp。
        #   **不要用 max_model.STATIC_SAG（0.0325）** —— 那是模擬值，
        #   而模擬在 z 方向高估順從性 1.8–2.1 倍，用它會補太少。
        a.z_sag = 0.036 * 250.0 / a.kp

    logp = shm_io.start_log("M9")
    mode = f"播放 {os.path.basename(a.traj)}" if a.traj else "狗上即時 CPG"
    print(f"M9 —— 步態（★★★ 承重、連續、動態觸地）　模式：{mode}\n")
    print("⚠️⚠️ 確認：狗趴在地上、16 顆洩力、**吊帶掛在 292 mm 以下且鬆弛**、")
    print("      **地面淨空且留出走動空間**、第二個終端機備著 estop。\n")

    # ---------------------------------------------------------------- 前置
    with shm_io.Shm("joint_cmd") as s:
        s.verify_layout(shm_io.CMD_STRIDE)
    with shm_io.Shm("joint_state") as s:
        s.verify_layout(shm_io.STATE_STRIDE)
    print("✅ 結構檢查通過")

    cmd0 = shm_io.read_joint_cmd()
    live = [c["name"] for c in cmd0
            if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
    if live:
        print(f"❌ 這些關節目前帶著非零增益，先處理再跑：{live}")
        return 1
    print("✅ 16 顆全部洩力中")

    pid = mc_ctrl_pid()
    if pid is None:
        print("❌ 找不到 mc_ctrl")
        return 1
    print(f"✅ mc_ctrl PID={pid} 狀態={proc_state(pid)}")
    roll0, pitch0 = read_imu_rp()
    print(f"✅ 機身姿態 roll {roll0:+.1f}° pitch {pitch0:+.1f}°")

    st0 = shm_io.read_joint_state()
    by = {r["name"]: r for r in st0}
    q_lie = {j: coord.to_ctrl(j, by[j]["position"]) for j in LEGS12}

    front = [q_lie[lg + coord.KIND_KNEE] for lg in coord.FRONT_LEGS]
    rear = [q_lie[lg + coord.KIND_KNEE] for lg in coord.REAR_LEGS]
    if all(math.copysign(1, f) == math.copysign(1, r) for f in front for r in rear):
        print("\n❌ 前後膝同號 = knee_back 模式，先用 M5 喬回來")
        return 1
    print("✅ 前後膝反號（後腿往前彎，原廠預設）")

    crouch = coord.POSES["crouch"]
    too_high = [j for j in LEGS12 if j.endswith(coord.KIND_KNEE)
                and abs(q_lie[j]) < abs(crouch[j]) - 0.02]
    if too_high:
        print(f"\n❌ 起點比 crouch 還高（{too_high}）—— 吊帶把狗撐起來了。")
        print("   吊帶高度 H 必須低於 crouch 的 292 mm，否則量到的力矩是假的。")
        return 1

    # ---------------------------------------------------------------- 步態來源
    if a.traj:
        D = json.loads(open(a.traj, encoding="utf-8").read())
        if D.get("schema") != "gait_traj/1":
            print(f"❌ 不是軌跡檔（schema={D.get('schema')!r}）")
            return 1
        gait_dt = D["dt"]
        gait_names = D["joints"]
        gait_q = D["q"]
        n_gait = D["n"]
        # ★ 這是刻意的「說兩次」設計：軌跡檔說一次、操作者在命令列說一次，
        #   兩邊要對得上，這樣「拿錯檔案」會被擋下來。**不要改成自動採用檔案值**
        #   —— 那等於把唯一一道防呆拿掉。
        #   但錯誤訊息要直接給出可以貼上去的指令，不要讓人在狗前面猜。
        #
        # ⚠️⚠️ **三個增益都要比，不能只比 kp/kd**（2026-09-03 補）。
        #   輪阻尼送給狗的是**命令列的值**（預設 0.5），不是檔案裡的值。
        #   漏比它的後果：用 `--wheel-kd 3.0` 產的檔、跑的時候忘了帶那個旗標，
        #   狗會**靜默用回 0.5** —— 而 0.5 正是「前腳不跨步」那組
        #   （模擬 0.03 vs 3.0 的 0.79）。症狀是「模擬明明好了、實機還是老樣子」，
        #   **而所有診斷指標都乾淨**。見 `docs/C_後膝負擔與鎖輪實驗_2026-09-02.md`。
        # ★ 只比「檔案裡有記」的欄位 —— 舊檔沒有 seq/sway，不該因此拒跑；
        #   但只要檔案記了，命令列就必須對上（否則等於跑到另一組步態）。
        want = {"kp": a.kp, "kd": a.kd, "wheel_kd": a.wheel_kd}
        for k in ("seq", "kp_abad", "x_off", "x_d", "g_c", "z_sag",
                  "sway_x", "sway_y", "sway_lead_x", "sway_lead_y"):
            if k in D:
                want[k] = getattr(a, k)
        bad = param_mismatches(D, want)
        # 輪子的位置增益必須是 0。檔案若寫了別的值，M9 也不會照做（第 544 行硬寫 0），
        # 那種「檔案說一套、實際做一套」比不一致更危險 —— 直接擋。
        if abs(D.get("wheel_kp", 0.0)) > 1e-9:
            print(f"❌ 軌跡檔的 wheel_kp = {D['wheel_kp']}，但 M9 一律送 0。")
            print("   設定檔的 FSM_RL_Wheel_Kp=60 是配「每步重給目標角」的 RL，")
            print("   開迴路套上去實測偏航失控 +39°/12s —— 拒跑。")
            return 1
        if bad:
            for k, got, want in bad:
                shown = "（檔案裡沒有這一項）" if got is None else f"{got:g}"
                print(f"❌ 軌跡檔的 {k} = {shown}，與 --{k.replace('_', '-')} "
                      f"（{want:g}）不一致。")
            print("   ⚠️ z_sag 與 kp 綁定、wheel_kd 與 x_off 耦合 —— 混用等於跑到"
                  "另一組步態，拒跑。")
            cmd = (f"--traj {os.path.basename(a.traj)} --kp {D['kp']:g}"
                   f" --kd {D['kd']:g} --wheel-kd {D.get('wheel_kd', 0.5):g}")
            print(f"\n   → 這個檔案要這樣跑：")
            print(f"     python3 M9_gait.py {cmd}")
            print(f"     sudo python3 M9_gait.py {cmd} --confirm")
            print("   （確認過那組增益就是你要的再跑）")
            return 1
        p = D["params"]
        print(f"\n軌跡檔：{'原地踏步' if p['march'] else '前進'}　"
              f"duty {p['duty']} ω {p['omega']} d_step {p['d_step']} "
              f"x_off {p['x_off']} g_c {p['g_c']} z_sag {p['z_sag']:.4f}")
    else:
        gait_dt = 0.02          # ★ 50 Hz —— 與模擬驗證時的 CTRL_DT 相同
        gait_names = LEGS12
        # ★★ 基準姿勢是 **home**（hip 0.8 / 膝 −1.5，機身 491 mm），
        #    **不是 stand**（0.6 / −1.2，542 mm）。CPG 的 f0/knee_sign 都是
        #    照 `max_model.HOME` 建的 —— 用錯會差 1.0 rad，
        #    2026-08-27 就是 `test_live_cpg_matches_the_generated_trajectory_file`
        #    抓出來的（我第一版寫成 stand）。
        f0 = cpg.home_foot(coord.POSES["home"])
        ks = cpg.knee_signs(coord.POSES["home"])
        phase = cpg.PHASES[a.seq]
        stp = cpg.make_step(phase)
        cst = cpg.init(phase)
        mux = {l: a.mu_x for l in cpg.LEGS}
        muy = {l: a.mu_y for l in cpg.LEGS}
        om = {l: a.omega for l in cpg.LEGS}
        # 逐腿 x_off（--x-d 0 時等於四腿共用 --x-off）。站姿與步態必須用同一個。
        x_off_legs = cpg.x_off_split(a.x_off, a.x_d)
        q_stand_g = cpg.stand_targets(f0, ks, x_off_legs)
        n_ramp = int(round(a.ramp / gait_dt))
        n_body = int(round(a.secs / gait_dt))
        n_gait = n_ramp + n_body + n_ramp
        gait_q = []
        n_clamp = 0
        for i in range(n_gait):
            sway = None
            if a.sway_x or a.sway_y:
                sway = cpg.body_sway(cpg.gait_phase(cst["theta"], phase),
                                     a.sway_x, a.sway_y,
                                     a.sway_lead_x, a.sway_lead_y)
            qg, ncl = cpg.joint_targets(cst, f0, ks, x_off_legs, a.g_c, a.d_step,
                                        a.d_step_y, a.duty, a.z_sag, sway)
            n_clamp += ncl
            if i < n_ramp:
                s = smoothstep(i / max(n_ramp, 1))
            elif i < n_ramp + n_body:
                s = 1.0
            else:
                s = smoothstep(1.0 - (i - n_ramp - n_body) / max(n_ramp, 1))
            gait_q.append([(1 - s) * q_stand_g[j] + s * qg[j] for j in LEGS12])
            cst = stp(cst, mux, muy, om, gait_dt)
        if n_clamp:
            print(f"\n❌ IK 縮限 {n_clamp} 次 —— 靜默的縮限會讓步態突然變鈍。")
            return 1
        print(f"\n即時 CPG：{'原地踏步' if a.march else '前進'}　序列 {a.seq.upper()}"
              f"（擺動順序 {' → '.join(cpg.swing_order(phase))}）")
        print(f"  duty {a.duty} ω {a.omega} d_step {a.d_step} "
              f"x_off {a.x_off} x_d {a.x_d} g_c {a.g_c} z_sag {a.z_sag:.4f}")
        print(f"  sway ({a.sway_x * 1000:.0f}, {a.sway_y * 1000:.0f}) mm  "
              f"lead ({a.sway_lead_x:.2f}, {a.sway_lead_y:.2f})")
        print("✅ 全程無 IK 縮限")

    # ★ 互動模式要的步態參數來源。**與上面所有乾跑檢查用的是同一組** ——
    #   分成兩份就會出現「檢查的是 A、跑的是 B」，那正是這條線最痛的一類問題。
    if a.traj:
        _B = D["baseline_ref"]
        p_src = dict(D["params"])
        mu_x_src, mu_y_src = _B["mu_x"], _B["mu_y"]
        d_step_y_src = _B["d_step_y"]
    else:
        p_src = dict(seq=a.seq, x_off=a.x_off, x_d=a.x_d, g_c=a.g_c,
                     d_step=a.d_step, duty=a.duty, omega=a.omega, z_sag=a.z_sag,
                     sway_x=a.sway_x, sway_y=a.sway_y,
                     sway_lead_x=a.sway_lead_x, sway_lead_y=a.sway_lead_y)
        mu_x_src, mu_y_src, d_step_y_src = a.mu_x, a.mu_y, a.d_step_y

    idxmap = [gait_names.index(j) for j in LEGS12]
    G = [[row[k] for k in idxmap] for row in gait_q]     # 一律轉成 LEGS12 序
    q_gait0 = {j: G[0][i] for i, j in enumerate(LEGS12)}
    q_gaitN = {j: G[-1][i] for i, j in enumerate(LEGS12)}

    # ---------------------------------------------------------------- 時序
    pre = build_standup(a, q_lie, q_gait0)
    post = build_sitdown(a, q_gaitN, q_lie)
    T_pre = sum(s[1] for s in pre)
    T_gait = n_gait * gait_dt
    T_post = sum(s[1] for s in post)
    if a.interactive:
        # ★ 互動模式沒有固定長度，能算的是**最壞情況**（三個等待都吃滿逾時）。
        #   承重上限管的是腿的發熱，所以必須用最壞情況去比，不能用「預計」。
        # ⚠️ 互動版的階段序列**沒有 HOLD_stand**（READY 取代了它的位置），
        #   所以不能直接用 T_pre —— 那會多算 a.hold 秒。預算和實際跑的必須是同一件事。
        T_pre_i = T_pre - a.hold
        T_END = (T_pre_i + a.hold_max + a.kp_shift + a.ramp + a.walk_max
                 + a.ramp + a.kp_shift + a.hold_max + T_post)
        print(f"\n互動模式　最壞情況 {T_END:.1f} 秒：")
        print(f"  站起來 {T_pre_i:.1f} ＋ 等①{a.hold_max:.0f} ＋ 降增益 {a.kp_shift:.1f}"
              f" ＋ 淡入 {a.ramp:.1f} ＋ 走 {a.walk_max:.0f} ＋ 淡出 {a.ramp:.1f}"
              f" ＋ 升增益 {a.kp_shift:.1f} ＋ 等③{a.hold_max:.0f} ＋ 趴下 {T_post:.1f}")
        print(f"  （等②就是「走」那段；三個等待逾時都是自動往下走，不是中止）")
    else:
        T_END = T_pre + T_gait + T_post
        print(f"\n總時長 {T_END:.1f} 秒 = 站起來 {T_pre:.1f} + 步態 {T_gait:.1f} "
              f"+ 坐回去 {T_post:.1f}")
    if T_END > a.max_secs:
        print(f"❌ 超過 --max-secs {a.max_secs:.0f}（承重時間，非 mc_ctrl 限制）")
        if a.interactive:
            print(f"   → 調小 --hold-max / --walk-max，或明確放寬 --max-secs。")
            print(f"   ⚠️ 放寬前先想清楚：M7/M8 最長只撐 34.6 秒，發熱沒量過。")
        return 1
    print(f"✅ 在承重上限 {a.max_secs:.0f} 秒之內")

    # ---- 命令速度檢查（★ 步態本來就快，門檻和 M7/M8 不同）
    vmax_seen, vj = 0.0, ""
    for i in range(1, n_gait):
        for k, j in enumerate(LEGS12):
            v = abs(G[i][k] - G[i - 1][k]) / gait_dt
            if v > vmax_seen:
                vmax_seen, vj = v, j
    print(f"\n步態最大命令速度 {vmax_seen:.2f} rad/s（{vj}）　上限 {a.vcmd_max}")
    if vmax_seen > a.vcmd_max:
        print("❌ 超過 --vcmd-max。降 --omega 或加大 --duty。")
        return 1
    print(f"✅ 在上限內（⚠️ 保護門檻 --vmax {a.vmax} 必須高於這個，否則會誤中止）")
    if a.vmax < vmax_seen * 1.2:
        print(f"❌ --vmax {a.vmax} 太接近命令速度 {vmax_seen:.2f} —— 會誤中止。")
        return 1

    bad = [f"{j}" for row in G for k, j in enumerate(LEGS12)
           if coord.check_limit(j, row[k], 0.03)]
    if bad:
        print(f"\n❌ {len(set(bad))} 個關節在步態中超出機構限位：{sorted(set(bad))}")
        return 1
    print("✅ 步態全程在機構限位內")

    print(f"\n增益：步態 ABAD {a.kp_abad:g} / HIP·KNEE {a.kp:g}　kd {a.kd}"
          f"（站立段一律 {STANDUP_KP:g}）　輪 **kp 0** kd {a.wheel_kd}（純阻尼）")
    print(f"保護：力矩 ABAD {TMAX['1_hip_roll']:.0f} / HIP {TMAX['2_hip_pitch']:.0f}"
          f" / KNEE {TMAX['3_knee_pitch']:.0f}、硬上限 {TAU_HARD:.0f}")
    print(f"      誤差 {a.emax} rad、速度 {a.vmax} rad/s、傾角 ±{a.tilt_max:.0f}°、"
          f"迴圈間隔 {a.gap_max*1000:.0f} ms")

    if not a.confirm:
        print("\n[乾跑] 沒有帶 --confirm，到此為止。沒有凍結、沒有寫入。")
        print(f"\n📄 {logp}")
        return 0
    if os.geteuid() != 0:
        print("❌ 需要 root：請加 sudo")
        return 1

    # ---------------------------------------------------------------- 執行
    idx = {j: shm_io.idx_of(j) for j in LEGS12}
    widx = {w: shm_io.idx_of(w) for w in shm_io.WHEELS}
    shm = shm_io.Shm("joint_cmd", write=True)
    state_ro = shm_io.Shm("joint_state")
    frozen = False
    abort = ""
    peak = {j: 0.0 for j in LEGS12}
    tau_hot = {j: 0 for j in LEGS12}
    # ★★ `peak` 專用的三點視窗，見 commit_peak()。**只影響回報，不影響中止。**
    tau_win: dict = {j: [] for j in LEGS12}
    spikes = {j: 0 for j in LEGS12}
    over = {j: 0 for j in LEGS12}       # 濾掉跳點後仍超標的筆數 → --tau-total
    over_raw = {j: 0 for j in LEGS12}   # 未濾的，只用於回報（好和 over 對照）
    samples: list = []
    recent: list = []
    kp_now, kd_now = 0.0, STANDUP_KD
    kp_abad_now = 0.0
    chatter = ChatterWatch()
    dead_hits = 0
    des_now = dict(q_lie)
    worst_gap = 0.0
    worst_gap_t = 0.0
    plan_done = False
    last_state_tick = -1
    n_stale = 0
    n_tick = 0
    t_prev = None

    def write_frame(des, kp, kd, wheel_kd, kp_abad):
        for j in LEGS12:
            # ★ ABAD 與 hip/knee 分開的 kp —— 模擬與原廠都是 [60,120,120]
            kj = kp_abad if j.endswith("1_hip_roll") else kp
            shm.write_cmd(idx[j], position=coord.to_motor(j, des[j]),
                          velocity=0.0, effort=0.0, kp=kj, kd=kd)
        st_w = state_ro.states()
        for w, wi in widx.items():
            # ★ 輪子全程純阻尼（kp=0）。步態需要輪子能自由滾。
            shm.write_cmd(wi, position=st_w[wi]["position"],
                          velocity=0.0, effort=0.0, kp=0.0, kd=wheel_kd)

    def wheel_kd_of(nm: str) -> float:
        """★★ 輪阻尼按階段排程（2026-09-03 trip17 事故後加）。

        `--wheel-kd 3.0` 在起身時讓四顆輪 ~64 Hz 抖振（v 峰 14 rad/s、
        力矩 ±15 N·m 高頻交變、刺耳聲）—— 對照 trip16 同階段 kd=0.5 是
        1.33 rad/s / 幾乎不翻轉。三個只在實機存在的因素疊加：
        velocity 欄位雜訊 47%、輪慣量小 + driver 1 kHz 離散阻尼、
        起身時輪必須**承重滾動**（原廠起身後輪各滾 ~100 mm）。

        → 起身／坐下**一律用 WHEEL_KD_SAFE（0.5，trip16 實機驗證）**，
          `--wheel-kd` 只作用在步態段（GAIT_IN/GAIT/GAIT_OUT）。
        ⚠️ kd=3.0 連步態段行不行都還是未知 —— 走路的支撐腳同樣是承重+滾動。
          上機前先做吊掛抖振測試分辨（driver 迴路不穩 vs 地面黏滑）。
        """
        if nm in ("GAIT", "GAIT_IN", "GAIT_OUT"):
            return a.wheel_kd
        return min(a.wheel_kd, WHEEL_KD_SAFE)

    bounds, tt = [], 0.0
    for nm, dur, p0, p1 in pre:
        bounds.append((tt, tt + dur, nm, p0, p1))
        tt += dur
    t_gait0 = tt
    tt += T_gait
    for nm, dur, p0, p1 in post:
        bounds.append((tt, tt + dur, nm, p0, p1))
        tt += dur

    plan = kw = None
    if a.interactive:
        # ★ 步態改成即時算：走多久由現場決定，固定長度的 G 播不完也停不下來。
        #   參數從既有來源取，**與上面所有乾跑檢查用的是同一組**。
        gp = dict(p_src, mu_x=mu_x_src, mu_y=mu_y_src, d_step_y=d_step_y_src)
        gs = GaitStream(gp, cpg.home_foot(coord.POSES["home"]),
                        cpg.knee_signs(coord.POSES["home"]))
        plan = InteractivePlan(a, q_lie, q_gait0, gs)
        kw = KeyWatch()
        if not kw.enabled:
            print("❌ 互動模式需要 tty（不要用 nohup／背景執行）—— 沒有 tty 就按不了 Enter。")
            return 1

    # ★ RT 迴圈期間關掉 Python циклic GC —— 兩趟實跑都在 t=19.37 出現 45 ms 停頓
    #   （同一時刻＝同一分配數＝gen2 回收，不是負載）。迴圈裡只有 append、無循環引用，
    #   關掉不會漏記憶體；finally 會重新打開並補收一次。
    import gc
    gc.disable()
    try:
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        time.sleep(0.15)
        print(f"\n✅ 已凍結 mc_ctrl（{proc_state(pid)}）\n")
        print(f"{'t':>6s} {'階段':>12s} {'kp':>6s} {'最大|誤差|':>10s} {'最大|τ|':>8s}"
              f" {'關節':>16s} {'roll':>6s} {'pitch':>6s} {'輪速':>7s}")
        t0 = time.monotonic()
        nxt = t0
        last = -1.0
        while True:
            t = time.monotonic() - t0
            if t >= T_END:
                break
            if t_prev is not None:
                gap = t - t_prev
                if gap > worst_gap:
                    worst_gap, worst_gap_t = gap, t
                if gap > a.gap_max:
                    abort = (f"迴圈間隔 {gap*1000:.0f} ms 超過 {a.gap_max*1000:.0f}"
                             f"（controller 逾時 500 ms → 會清零）")
            t_prev = t
            n_tick += 1
            if abort:
                break

            if a.interactive:
                nm, des_now, kp_now, kd_now, kp_abad_now, plan_done = plan.update(
                    t, kw.pressed())
            elif t < t_gait0 or t >= t_gait0 + T_gait:
                s0, s1, nm, p0, p1 = next(b for b in bounds if b[0] <= t < b[1])
                r = (t - s0) / max(s1 - s0, 1e-6)      # 這一段走了幾成
                u = smoothstep(r)
                kp_now, kd_now = phase_gains(nm, r, a.kp, a.kd)
                kp_abad_now = phase_kp(nm, r, a.kp_abad)
                des_now = {j: p0[j] + u * (p1[j] - p0[j]) for j in LEGS12}
            else:
                # ★ 步態：50 Hz 的目標**線性內插**到寫入頻率。
                #   模擬是零階保持（每 nsub 個物理步換一次），內插只會更平順；
                #   差異很小但要知道兩邊不完全相同。
                nm = "GAIT"
                kp_now, kd_now = a.kp, a.kd
                kp_abad_now = a.kp_abad
                x = (t - t_gait0) / gait_dt
                i0 = int(x)
                if i0 >= n_gait - 1:
                    des_now = {j: G[-1][k] for k, j in enumerate(LEGS12)}
                else:
                    f = x - i0
                    des_now = {j: G[i0][k] + f * (G[i0 + 1][k] - G[i0][k])
                               for k, j in enumerate(LEGS12)}

            stt = state_ro.states()
            # ★★ 資料新鮮度（2026-09-03 16:47 假中止的根因）：
            #   停頓後的追趕 tick 會重複讀到**同一幀** joint_state，
            #   「連續 N 筆」保護把一筆實體樣本數成 N 筆 → 假中止。
            #   driver 是 1 kHz、我們 200 Hz，正常時每 tick 必有新幀；
            #   幀沒前進就跳過保護計數（指令與心跳照寫）。
            tick_state = state_ro.read_tick(shm_io.STATE_STRIDE)
            stale = (tick_state == last_state_tick)
            last_state_tick = tick_state
            we = (0.0, "")
            wt = (0.0, "")
            tick = {}
            for j in LEGS12:
                if stale:
                    break
                sg = coord.SIGN[j[2:]][j[:2]]
                r = stt[idx[j]]
                q = coord.to_ctrl(j, r["position"])
                v = sg * r["velocity"]
                tau = sg * r["effort"]
                err = q - des_now[j]
                tick[j] = (round(q, 4), round(des_now[j], 4), round(tau, 2), round(v, 3))
                kj = kp_abad_now if j.endswith("1_hip_roll") else kp_now
                cap = kj * abs(err) + a.kd * abs(v)
                w = tau_win[j]
                w.append((tau, cap))
                if len(w) > 3:
                    w.pop(0)
                commit_peak(w, peak, spikes, j, over)
                if abs(err) > we[0]:
                    we = (abs(err), j)
                if abs(tau) > wt[0]:
                    wt = (abs(tau), j)
                lim = TMAX[j[2:]]
                if abs(tau) > lim:
                    over_raw[j] += 1
                # ── 快速路徑：連續 N 筆。**一個字都沒改。**
                #    它防的是「持續超載」，而且刻意不做任何過濾 ——
                #    過濾要等下一筆，那會讓保護晚一拍。
                if abs(tau) > TAU_HARD:
                    tau_hot[j] += 1
                    if tau_hot[j] >= 2:
                        abort = f"{j} 力矩連續 2 筆超過硬上限 {TAU_HARD}（{tau:+.1f}）"
                elif abs(tau) > lim:
                    tau_hot[j] += 1
                    if tau_hot[j] >= a.tau_hits:
                        abort = f"{j} 力矩連續 {tau_hot[j]} 筆超過 {lim}（{tau:+.1f}）"
                else:
                    tau_hot[j] = 0
                # ── ★★ 慢速路徑（2026-09-02 新增）：整趟累計筆數。
                #    trip13（ω1.0）暴露的缺陷：觸地衝擊尖峰只有 5~25 ms 寬，
                #    `bl3_knee_pitch` 整趟 13 筆超過 70（峰值 80.0 = 114%），
                #    但**最長連續只有 2 筆** —— 差一筆就被快速路徑攔下。
                #    而且那 13 筆是每個步態週期一兩筆、散在 16 秒裡，
                #    任何 20 筆視窗內最多也只有 2 筆 ——
                #    ★ 滑動視窗規則同樣分不出來，只有**整趟累計**分得出來：
                #      trip10/11/12 全部 ≤1（而且那 1 筆還是跳點），trip13 是 13。
                #    ⚠️ 這條用**濾掉跳點後**的計數（`over`，由 commit_peak 累計）。
                #      過濾器要等下一筆，但這條本來就是累積好幾秒才觸發的，
                #      慢一拍無所謂 —— 和快速路徑的取捨不同，不要混為一談。
                if a.tau_total and over[j] >= a.tau_total:
                    abort = (f"{j} 整趟累計 {over[j]} 筆超過 {lim}"
                             f"（峰值 {peak[j]:+.1f}）—— 衝擊尖峰型超載")
                if abs(err) > a.emax:
                    abort = f"{j} 追蹤誤差 {err:+.3f} 超過 {a.emax}"
                if abs(v) > a.vmax:
                    abort = f"{j} 速度 {v:+.2f} 超過 {a.vmax}"
                if r["temp_C"] > a.temp_max:
                    abort = f"{j} 溫度 {r['temp_C']:.1f}°C 超過 {a.temp_max}"
                if abort:
                    break

            # ★ 馬達失力偵測（trip17：急停後 M9 還印「狗還撐著」——那是錯的）。
            #   簽名：控制律上限 kp|e|+kd|v| 很大、實測 |τ| 卻趨近 0，全腿一致。
            #   單腿單筆可能是 effort 垃圾（已知 0.11%），所以要求多數腿連續多筆。
            if kp_now > 50 and not abort and len(tick) == 12:
                n_dead = sum(1 for j in LEGS12
                             if (lambda q_, d_, t_, v_, k_:
                                 k_ * abs(q_ - d_) > 20 and
                                 abs(t_) < 0.1 * k_ * abs(q_ - d_))(
                                 *tick[j],
                                 kp_abad_now if j.endswith("1_hip_roll") else kp_now))
                # ★ 門檻 4 腿：實測急停後重力讓多數關節停在目標附近，
                #   持續顯示失力簽名的只有四個膝（4–7 腿擺盪）。6 腿永遠湊不滿。
                dead_hits = dead_hits + 1 if n_dead >= 4 else 0
                if dead_hits >= 40:      # 200 Hz 下 0.2 秒
                    abort = ("馬達沒有在執行指令（誤差大但力矩≈0，多腿一致）"
                             "—— 可能已按硬體急停或馬達進保護。"
                             "★ 凍結目標角此時**沒有支撐作用**，狗是靠機構卡著")
            roll, pitch = read_imu_rp()
            if not stale and not abort and max(abs(roll), abs(pitch)) > a.tilt_max:
                abort = f"機身傾角 roll {roll:+.1f}° pitch {pitch:+.1f}° 超過 ±{a.tilt_max}°"

            # ★★ 輪子的 position/velocity 一定要記 ——
            #    2026-08-27 發現 M8 只記了 effort，導致實機資料無法做
            #    「觸地滾動 vs 懸空空轉」的拆解，而那是判斷
            #    「前後腿有沒有在互相對抗」的唯一方法。
            for w, wi in widx.items():
                if stale:
                    break
                if not abort and chatter.feed(w, stt[wi]["velocity"]):
                    abort = (f"{w} 輪抖振（高頻正負翻轉、|v|>{ChatterWatch.V_MIN}）"
                             f"—— wheel_kd={wheel_kd_of(nm):g} 在此階段不穩定")
            wrec = None if stale else {w: (round(stt[wi]["position"], 4),
                        round(stt[wi]["velocity"], 3),
                        round(stt[wi]["effort"], 2)) for w, wi in widx.items()}
            if not stale:
                rec = {"t": round(t, 3), "phase": nm, "kp": round(kp_now, 1),
                       "roll": round(roll, 2), "pitch": round(pitch, 2),
                       "j": tick, "w": wrec}
                recent.append(rec)
                if len(recent) > 60:
                    recent.pop(0)
                samples.append(rec)
            if abort:
                break

            if stale:
                n_stale += 1
            write_frame(des_now, kp_now, kd_now, wheel_kd_of(nm), kp_abad_now)
            shm.write_tick(tick_state)
            # ★ 收工判斷放在寫完這一幀之後 —— 直接 break 會少寫最後一幀，
            #   而最後一幀正是「kp 已降到 0」的那一幀。
            if plan_done:
                print(f"\n✅ 互動流程走完（{t:.1f} s）")
                break

            if t - last >= 0.25:
                wv = max(abs(x[1]) for x in wrec.values())
                print(f"{t:6.2f} {nm:>12s} {kp_now:6.0f} {we[0]:10.4f} {wt[0]:8.2f}"
                      f" {wt[1]:>16s} {roll:+6.1f} {pitch:+6.1f} {wv:7.2f}")
                last = t
            nxt += 1.0 / a.hz
            dly = nxt - time.monotonic()
            # ★ 停頓後重新對時 —— 否則會以 <1ms 間隔爆發追趕（實測 12 連發），
            #   每一發都讀同一幀資料。掉拍就掉拍，不補。
            if dly < -3.0 / a.hz:
                nxt = time.monotonic() + 1.0 / a.hz
                dly = -1.0
            if dly > 0:
                time.sleep(dly)
    except KeyboardInterrupt:
        abort = "使用者 Ctrl-C"
    except Exception as e:
        abort = f"未預期的例外：{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- 收尾
    gc.enable()
    gc.collect()
    held_des, held_kp = dict(des_now), (kp_now if abort else 0.0)
    held_kd = kd_now if abort else STANDUP_KD
    held_kp_abad = kp_abad_now if abort else 0.0
    keeper = Keepalive(shm, state_ro,
                       (lambda: write_frame(held_des, held_kp, held_kd,
                                            WHEEL_KD_SAFE, held_kp_abad)), a.hz,
                       "凍結目標角、維持增益" if abort else "零增益保持")
    keeper.start()

    print("\n" + "=" * 76)
    if abort:
        print(f"⛔ 中止：{abort}")
        if held_kp >= 0.3 * a.kp:
            print(f"\n★★ **已凍結目標角並維持 kp={held_kp:.0f} —— 狗還撐著，沒有放手。**")
            print("   ⚠️ 步態中止會凍在某個相位，可能有腿懸空。先看狗再決定。")
        hurt = abort.split()[0]
        if hurt in LEGS12 and recent:
            print(f"\n中止前最後 12 筆 —— {hurt}")
            print(f"  {'t':>7s} {'階段':>12s} {'q':>9s} {'des':>9s} {'τ':>8s}"
                  f" {'v':>7s} {'kp|e|+kd|v|':>11s}")
            for rr in recent[-12:]:
                q_, d_, tau_, v_ = rr["j"][hurt]
                cap = rr["kp"] * abs(q_ - d_) + a.kd * abs(v_)
                print(f"  {rr['t']:7.3f} {rr['phase']:>12s} {q_:9.4f} {d_:9.4f}"
                      f" {tau_:8.2f} {v_:7.3f} {cap:11.1f}")
    else:
        print("✅ 序列完整跑完")

    report_peaks(tau_win, peak, spikes, over, over_raw)

    el = min(t, T_END) if n_tick else 0.0
    hz = n_tick / el if el > 0 else 0.0
    if n_stale:
        print(f"\n舊幀跳過 {n_stale} 次（停頓後的追趕 tick，保護與紀錄未受污染）")
    print(f"\n迴圈：{n_tick} 次 / {el:.2f}s = {hz:.0f} Hz（目標 {a.hz:.0f}）"
          f"　最長單次間隔 {worst_gap*1000:.0f} ms @ t={worst_gap_t:.2f}s")
    if worst_gap > 0.5:
        print("  ❌ **有一次間隔超過 500 ms —— 那一刻指令區很可能被清成 0。**")
    elif worst_gap > a.gap_max:
        print("  ⚠️ 超過警戒線。先 uptime／top 看狗上有沒有別的東西在跑。")

    # ★ 輪子的淨滾動（分觸地/懸空是事後分析的事，這裡先給總量）
    if samples:
        w0 = samples[0]["w"]
        w1 = samples[-1]["w"]
        print(f"\n{'輪':10s} {'淨轉動 rad':>12s} {'≈ 滾動 mm':>11s}")
        for w in shm_io.WHEELS:
            d_ = shm_io.wrap_pi(w1[w][0] - w0[w][0])
            print(f"{w:10s} {d_:12.4f} {d_*kin.WHEEL_RADIUS*1000:11.0f}")
        print("  ⚠️ 這是總量，**包含懸空空轉** —— 模擬顯示空轉可佔八成。")
        print("     要判讀必須拆成觸地/懸空，事後用 `w` 欄的 position 做。")

    print("\n" + "=" * 76)
    print("★ 現在腿還在承重。[Enter] 依原路徑坐回趴姿；[Ctrl-C] 立刻放手（會塌）")
    try:
        if sys.stdin.isatty():
            input("\n   > ")
        else:
            print("   非互動模式 → 直接執行坐回去")
        keeper.stop()
        cur = dict(held_des)
        for nm, dur, tgt in (("SIT_crouch", a.t2, dict(coord.POSES["crouch"])),
                             ("SIT_LIE", a.t1, q_lie)):
            s = time.monotonic()
            while (e := time.monotonic() - s) < dur:
                u = smoothstep(e / dur)
                # ★ 坐回趴姿是承重動作 —— 用 M7 驗證過的站立增益，
                #   不是步態那組（步態可能是 kp=120 的軟增益，撐不住）。
                write_frame({j: cur[j] + u * (tgt[j] - cur[j]) for j in LEGS12},
                            max(held_kp, STANDUP_KP), STANDUP_KD,
                            WHEEL_KD_SAFE, max(held_kp, STANDUP_KP))
                shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
                time.sleep(1.0 / a.hz)
            cur = dict(tgt)
        s = time.monotonic()
        while (e := time.monotonic() - s) < a.ramp_kp:
            _k = STANDUP_KP * max(0.0, 1 - e / a.ramp_kp)
            write_frame(cur, _k, STANDUP_KD, WHEEL_KD_SAFE, _k)
            shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
            time.sleep(1.0 / a.hz)
        for i in range(len(shm_io.JOINTS)):
            shm.zero_gains(i)
        shm.write_tick(state_ro.read_tick(shm_io.STATE_STRIDE))
        print("✅ 已坐回趴姿並降到零增益")
        keeper = Keepalive(shm, state_ro,
                           (lambda: [shm.zero_gains(i)
                                     for i in range(len(shm_io.JOINTS))]),
                           a.hz, "零增益保持")
        keeper.start()
    except KeyboardInterrupt:
        print("\n   （Ctrl-C：放手）")
    keeper.stop()

    out = {"schema": "m9_gait/1", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "args": vars(a), "aborted": bool(abort), "abort_reason": abort or None,
           "q_lie": q_lie, "gait_dt": gait_dt, "n_gait": n_gait,
           "t_gait0": t_gait0, "peak": peak, "spikes": spikes, "over": over, "over_raw": over_raw,
           "loop": {"ticks": n_tick, "hz": round(hz, 1),
                    "worst_gap_s": round(worst_gap, 4)},
           "samples": samples[:60000]}
    jp = (logp[:-4] if logp.endswith(".log") else logp) + ".json"
    try:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        if os.geteuid() == 0 and os.getenv("SUDO_USER"):
            import pwd
            pw = pwd.getpwnam(os.environ["SUDO_USER"])
            try:
                os.chown(jp, pw.pw_uid, pw.pw_gid)
            except OSError:
                pass
        print(f"\n📊 {jp}")
    except Exception as e:
        print(f"\n⚠️ 結果檔寫入失敗：{e}")

    try:
        shm.close()
        state_ro.close()
    except Exception:
        pass
    if frozen:
        print(f"\n⏸ mc_ctrl 仍在凍結中（PID {pid}）。確認狗安全後：")
        print(f"      sudo kill -CONT {pid}")
    print(f"\n📄 {logp}")
    return 1 if abort else 0


if __name__ == "__main__":
    sys.exit(main())
