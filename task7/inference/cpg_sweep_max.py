"""D1 Max 開迴路 CPG 的**多擾動**參數掃描器。

    conda run --no-capture-output -n rbtdog python task7/inference/cpg_sweep_max.py --plan full

================================================================================
§1 為什麼要另外寫一支 —— `cpg_walk_max.py --sweep` 不夠用
================================================================================
那支每個參數值只跑**一次**。但這台的步態指標有一部分是**混沌**的：
trot 把 x_off 擾動 1e-12 m 就能讓速度差 4 倍、偏航全距 128°。
在那種指標上，「單次數字」不是量測，是抽籤。

所以這支的每一格都是**一組 rollout**（預設 6 個，彼此只差 1e-12 m 的 x_off），
表格印的是 **中位數 + 全距**。判讀規則只有一條：

    ★ 全距比你要主張的差異還大時，那個差異不存在。

================================================================================
§2 記憶體 —— 這支會擋著不讓機器當掉
================================================================================
⚠️ 2026-08-26 一次平行掃描把 16 GB 的開發機 OOM 弄當機。根因有兩層：

  1. `Robot()` 每次都重建整個 MJCF（**約 1.15 GB**，網格吃掉大部分）。
     已在 `cpg_walk_max._model()` 修掉：模型只建一次，每次還原被改寫的欄位。
     修完單一進程的 RSS 從「每跑一次漲 200 MB」變成完全不漲。
  2. 平行度沒有跟可用記憶體掛勾。修好第 1 點之後每個 worker 仍是**固定 1.3 GB**，
     16 核全開就是 21 GB —— 核數再多也沒用，這是記憶體綁死的工作。

這支的做法：worker 數由 `MemAvailable` 反推（見 `safe_workers`），
而且每收到一筆結果就重驗一次；掉到 `FLOOR_GB` 以下**立刻收掉整池並回報**，
不是繼續硬跑。★ 寧可掃到一半有結果，不要掃完整組但機器當掉。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ★ 必須在 import mujoco / numpy 之前。否則每個 worker 會各自開一票 BLAS 執行緒，
#   16 個 worker × 16 執行緒互相搶核，掃描反而變慢。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cpg_walk_max as cw
import max_model as mm

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# ---- 記憶體常數（實測，見 §2）----
PER_WORKER_GB = 1.5     # 實測穩態 1.29 GB，進位留餘裕
RESERVE_GB = 2.5        # 留給作業系統 / 編輯器 / 另一個 session
FLOOR_GB = 1.2          # 低於此值立刻收掉整池

# ---- 擾動 ----
# 1e-12 m = 1 皮米。物理上等於沒動，所以指標一旦跟著變，變的原因只可能是**混沌**，
# 不是「這個參數真的比較好」。這是 cpg_walk_max §GAITS 用過的同一個手法。
JITTER = 1e-12


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("讀不到 MemAvailable")


def safe_workers(requested: int | None) -> int:
    """由可用記憶體反推 worker 數。★ 不是由核數反推 —— 這是記憶體綁死的工作。"""
    avail = mem_available_gb()
    by_mem = int((avail - RESERVE_GB) // PER_WORKER_GB)
    by_cpu = os.cpu_count() or 1
    n = max(1, min(by_mem, by_cpu, requested or by_cpu))
    print(f"[記憶體] 可用 {avail:.1f} GB → 記憶體容得下 {by_mem} 個 worker"
          f"（每個 {PER_WORKER_GB} GB，保留 {RESERVE_GB} GB）；"
          f"核數 {by_cpu}；要求 {requested or '未指定'} → **採用 {n} 個**")
    if by_mem < 1:
        raise SystemExit(f"✖ 可用記憶體只有 {avail:.1f} GB，連一個 worker 都放不下。"
                         f"關掉一些程式再跑。")
    return n


# =============================================================================
# worker
# =============================================================================
def _run(job: dict) -> dict:
    """跑一個 rollout。模型由 `cpg_walk_max._model()` 在本 worker 內快取。"""
    kw = dict(job["kw"])
    kw["x_off"] = (kw.get("x_off") if kw.get("x_off") is not None
                   else cw.GAITS[kw["gait"]]["x_off"]) + job["seed"] * JITTER
    r = cw.rollout(quiet=True, **kw)
    r["_cell"] = job["cell"]
    r["_seed"] = job["seed"]
    # 各格的 secs 可以不一樣（長時程那組），所以偏航要能換算成**每秒**才比得了
    r["_secs"] = kw["secs"]
    return r


# =============================================================================
# 聚合與輸出
# =============================================================================
def _agg(rs: list[dict]) -> dict:
    """一格（同參數、不同擾動）的統計。中位數看趨勢，全距看能不能引用。"""
    def med(k, s=1.0):
        return float(np.median([r[k] for r in rs])) * s

    def rng(k, s=1.0):
        v = [r[k] * s for r in rs]
        return float(max(v) - min(v))

    # ⚠️ 只有全距**不足以**判斷兩格的差異是不是真的 —— 全距不告訴你分布落在哪。
    #    兩格的中位數差 12°、全距各 13° 與 8°，可以是「完全不重疊」也可以是「疊一半」，
    #    這兩種情況的結論完全相反。所以偏航另外把 min/max 印出來。
    def span(k):
        v = [r[k] for r in rs]
        return [float(min(v)), float(max(v))]

    return {
        "n": len(rs),
        "n_fell": sum(1 for r in rs if r["fell"] is not None),
        "yaw_span": span("yaw"),
        "speed_path_span": span("speed_path"),
        # ★ 表格印的是 speed_travel（以步態週期為步長），不是 speed_path。
        #   speed_path 把機身左右搖擺算成前進，實測高估約 68%。
        "speed_travel": med("speed_travel"), "speed_travel_rng": rng("speed_travel"),
        "speed_path": med("speed_path"), "speed_path_rng": rng("speed_path"),
        "speed_net": med("speed_net"),
        "speed": med("speed"),
        "yaw": med("yaw"), "yaw_rng": rng("yaw"),
        # 每秒偏航率：不同 secs 的格唯一能互比的量。慢漂是不是真的，看這個。
        "yaw_rate": float(np.median([r["yaw"] / r.get("_secs", 20.0) for r in rs])),
        "yaw_rate_rng": float(np.ptp([r["yaw"] / r.get("_secs", 20.0) for r in rs])),
        "bounce": med("bounce", 1000), "bounce_rng": rng("bounce", 1000),
        "min_lift": med("min_lift", 1000),
        "support": med("support"), "support_rng": rng("support"),
        "pitch_mean": med("pitch_mean"), "pitch_mean_rng": rng("pitch_mean"),
        "pitch_cycle": med("pitch_cycle"),
        "height": med("height", 1000),
        "lateral": med("lateral", 1000),
        "net_roll": med("net_roll", 1000),
        # ★★ 執行率：2026-08-27 補上。前腳「抬起來原地放下」時其餘指標全是乾淨的，
        #    只有這一項會露餡。前/後分開看——缺陷是**前後不對稱**，平均會把它抹掉。
        "exec_front": med("exec_front"), "exec_front_rng": rng("exec_front"),
        "exec_rear": med("exec_rear"),
        "step_self_front": float(np.median([np.mean(r["step_self"][:2]) for r in rs])),
        "lim_pct": max(r["lim_pct"] for r in rs),
        "tau_pct": max(r["tau_pct"] for r in rs),
        "reach_pct": max(r["reach_pct"] for r in rs),
    }


HDR = (f"{'值':>10} |{'跌倒':>6}{'★行進速度m/s':>15}{'★前腳執行':>13}{'後腳':>7}"
       f"{'前腿自走mm':>12}{'彈跳mm':>13}{'支撐腳':>12}"
       f"{'離地mm':>8}{'平均俯仰°':>15}{'偏航°/s':>16}{'超限%':>7}{'飽和%':>7}")


def _row(label: str, a: dict) -> str:
    fell = f"{a['n_fell']}/{a['n']}"
    return (f"{label:>10} |{fell:>6}"
            f"{a['speed_travel']:>9.3f}±{a['speed_travel_rng']:>4.3f}"
            f"{a['exec_front']:>8.2f}±{a['exec_front_rng']:>3.2f}"
            f"{a['exec_rear']:>7.2f}{a['step_self_front']:>12.1f}"
            f"{a['bounce']:>7.1f}±{a['bounce_rng']:>4.1f}"
            f"{a['support']:>7.2f}±{a['support_rng']:>3.2f}"
            f"{a['min_lift']:>8.1f}"
            f"{a['pitch_mean']:>+9.2f}±{a['pitch_mean_rng']:>4.2f}"
            f"{a['yaw_rate']:>+9.3f}±{a['yaw_rate_rng']:>5.3f}"
            f"{a['lim_pct']:>7.2f}{a['tau_pct']:>7.2f}")


# =============================================================================
# 掃描計畫
# =============================================================================
def build_plan(name: str, secs: float, nseed: int) -> list[dict]:
    """(cell, kw) 的清單。cell = (掃描名, 參數值標籤)。"""
    S = dict(secs=secs)
    P: list[tuple[str, str, dict]] = []

    def add(sweep, vals, key, gait="walk", fmt="{}", **extra):
        for v in vals:
            P.append((sweep, fmt.format(v), dict(S, gait=gait, **{key: v}, **extra)))

    if name in ("full", "trim"):
        # x_off 配平點。★ 判準是**平均俯仰過零**，不是偏航（偏航在這裡是混沌的）。
        for g in ("walk", "walk_fast"):
            add(f"x_off/{g}", [-0.050, -0.045, -0.040, -0.035, -0.030, -0.025, -0.020],
                "x_off", gait=g, fmt="{:.3f}")
    if name in ("full", "params"):
        add("duty", [0.70, 0.75, 0.80, 0.85, 0.90], "duty", fmt="{:.2f}")
        add("mu_y", [1.0, 1.25, 1.5, 1.75, 2.0], "mu_y", fmt="{:.2f}")
        add("地面摩擦", [0.3, 0.4, 0.5, 0.7, 1.0], "friction", fmt="{:.1f}")
        add("omega", [1.0, 1.2, 1.4, 1.6, 1.8], "omega", fmt="{:.1f}")
        add("d_step", [0.06, 0.08, 0.10, 0.13, 0.16], "d_step", fmt="{:.2f}")
        add("g_c", [0.05, 0.065, 0.08, 0.10, 0.12], "g_c", fmt="{:.3f}")
    if name in ("full", "abad"):
        # ABAD 的 1.85 是**下界不是量測值**（正向沒掙脫，見結果文件）。
        # 掃它是為了回答「步態到底吃不吃這個數字」—— 若不敏感，量不準就不擋路。
        add("ABAD摩擦", [0.0, 1.5, 1.85, 2.5, 3.5, 5.0], "abad_friction", fmt="{:.2f}")
        add("腿關節摩擦", [0.0, 0.75, 1.5, 2.25, 3.0], "leg_friction", fmt="{:.2f}")
    if name in ("full", "base"):
        for g in ("walk", "walk_fast", "trot"):
            P.append(("預設值", g, dict(S, gait=g)))
    if name == "yaw":
        # ① 直接複驗交接文件 §2 的那組 A/B（兩邊都用舊的 x_off −30 隔離變因）。
        #    文件量到 −12.4° → −0.1°，並且自己標注「不要據此宣布偏航解決了」。
        #    這裡把同一組跑成分布，讓那個 12° 的差異接受全距檢驗。
        for lf in (0.0, 1.5):
            P.append(("§2複驗 x_off−30", f"腿摩擦{lf:.1f}",
                      dict(S, gait="walk", x_off=-0.030, leg_friction=lf)))
        # ② 長時程。HANDOFF 說偏航慢漂 −0.5~−0.8°/s、60 秒累積 −50°，
        #    而那是「唯一擋住能直線走遠的東西」。★ 20 秒看不出慢漂，必須拉長。
        for g in ("walk", "walk_fast"):
            for t in (20.0, 60.0):
                P.append((f"長時程/{g}", f"{t:.0f}s", dict(gait=g, secs=t)))
        # ③ walk_fast 的慢漂能不能靠配平消掉？
        #    它的**俯仰**配平點在 −46 mm（walk 是 −41），但預設兩組都寫 −40。
        #    若慢漂隨 x_off 單調變號，那它是配平沒配好，不是非得上閉迴路。
        for v in (-0.055, -0.050, -0.046, -0.040, -0.035):
            P.append(("慢漂vs配平/walk_fast 60s", f"{v:.3f}",
                      dict(gait="walk_fast", secs=60.0, x_off=v)))

    if name == "omega_trim":
        # 全掃描顯示 **ω 才是速度的主槓桿**（1.0→1.8 是 0.076→0.293 m/s，3.9 倍），
        # 而舊的 speed_path 度量把它壓成只有 +23% —— 度量錯，槓桿就看不見。
        # ω=1.8 在 60 s 給到 0.316 m/s（全部設定裡最快）但**還沒配平**（俯仰 +0.23°）。
        # 配平點一定要用它自己的 ω 重掃 —— 這是同一條教訓的第三次。
        # ⚠️ 第一次掃 −0.075~−0.040 全是正俯仰，方向掃反了：ω 變大時配平點
        #    是往**靠近 0** 的方向移動（1.4 → −41 mm，1.8 → −25 mm），不是往更負。
        for v in (-0.040, -0.035, -0.030, -0.025, -0.020, -0.015, -0.010):
            P.append(("x_off @ ω1.8 60s", f"{v:.3f}",
                      dict(gait="walk", secs=60.0, omega=1.8, x_off=v)))

    if name == "g1":
        # ★ MJX 訓練模型的落差量測。**每一段只與前一段差一項**：
        #
        #   A  網格   + 外部PD    + solver 預設   ← 基準（既有全部結論的來源）
        #   C1 網格   + 位置伺服  + solver 預設   ← 只換「PD 由誰算」
        #   C2 圓柱   + 位置伺服  + solver 預設   ← 再換碰撞形狀（含關掉自碰撞）
        #   C3 圓柱   + 位置伺服  + solver 6/6    ← 訓練實際要用的模型
        #
        # ⚠️ 第一版把「換模型檔」與「換 actuator_mode」綁在同一步，結果 B1 是
        #    **把力矩寫進位置伺服的 ctrl**（等於命令「目標角 = 42 rad」），
        #    量出 5/12 跌倒、彈跳 185 mm，差點得到「換形狀害步態垮掉」的假結論。
        #    模型檔與致動器模式**不是獨立的兩個旋鈕** —— 模式必須跟著模型檔走。
        #    `Robot.__init__` 現在兩個方向都有 biastype 斷言，這種配置會當場擋下。
        #
        # ⚠️ 四段共用 gait="walk"，所以 `_run` 加的 x_off 擾動序列一致、可逐 seed 對照。
        _MD = Path(mm.SCENE).parent
        P.append(("G1/模型", "A 網格+外部PD", dict(S, gait="walk")))
        P.append(("G1/模型", "C1 網格+位置伺服",
                  dict(S, gait="walk", scene=str(_MD / "scene_diag_mesh_position.xml"),
                       actuator_mode="position")))
        P.append(("G1/模型", "C2a 圓柱輪",
                  dict(S, gait="walk", scene=str(_MD / "scene_diag_cyl_position.xml"),
                       actuator_mode="position")))
        P.append(("G1/模型", "C2b 球輪",
                  dict(S, gait="walk", scene=str(_MD / "scene_diag_sph_position.xml"),
                       actuator_mode="position")))
        P.append(("G1/模型", "C2c 圓盤5mm",
                  dict(S, gait="walk", scene=str(_MD / "scene_diag_disc5_position.xml"),
                       actuator_mode="position")))
        P.append(("G1/模型", "C3 訓練模型",
                  dict(S, gait="walk", scene=mm.SCENE_MJX, actuator_mode="position")))

    if name == "duty_kp":
        # ★ duty × kp 二維掃描（2026-08-27）。
        #
        # 為什麼要二維：前腳擺動相只執行指令的 2–5%，兩個候選解釋是
        #   (a) 位置伺服對擺動腿慣量太軟（kp=120 是原廠**RL 設定檔**的值）
        #   (b) 擺動相時間太短（duty 0.80 + ω 1.4 → 只有 143 ms）
        # 分開掃很可能各自都「成立」，而不知道哪個是主因 ——
        # 那正是這個專案 2026-08-27 已經踩過一次的坑（第一版 G1 對照）。
        #
        # ⚠️ `duty ≤ 0.70 必跌` 這條結論**是在 kp=120 下量的**，高增益下未必成立，
        #    所以 duty 要往下掃到 0.60。
        # ⚠️ `z_sag` 不是獨立旋鈕，是**為某個 kp 量出來的撓度補償**。
        #    實測靜態撓度：kp=120 → 32.5 mm、kp=250 → **16.7 mm**（幾乎正比於 1/kp）。
        #    所以這裡讓它**隨 kp 等比縮放**。固定 z_sag 反而是引入混淆 ——
        #    高增益格的補償會按定義就是錯的（補過頭 → 離地量被吃掉）。
        # ⚠️ `x_off`（配平點）同樣與增益綁死，但它沒有這種簡單的比例關係，
        #    這裡維持基準值，配平留到選定 (duty, kp) 之後用 `--plan trim` 重掃。
        for duty in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            for kp in (120.0, 180.0, 250.0, 360.0, 480.0):
                # kd 隨 kp 等比例縮放，維持阻尼比。原廠兩組都符合這個關係
                # （120/1.0 與 250/5.0 不完全等比，但 kd 對執行率的敏感度低，
                #  見 gain_compare 的 250/kd2 vs 250/kd5）。
                kd = 1.0 * (kp / 120.0)
                P.append((f"duty {duty:.2f}", f"kp{kp:.0f}",
                          dict(S, gait="walk", duty=duty,
                               kp3=[kp * 0.5, kp, kp], kd3=[kd, kd, kd],
                               z_sag=mm.STATIC_SAG * 120.0 / kp)))

    if name == "duty_kp2":
        # duty_kp 的決選：只留有希望的格，改用 12 擾動確認。
        # duty 0.60–0.70 全部淘汰（彈跳 50–100 mm、支撐腳 2.0–2.4），不再花時間。
        for duty in (0.80, 0.85):
            for kp in (250.0, 360.0, 480.0):
                kd = 1.0 * (kp / 120.0)
                P.append((f"決選 duty {duty:.2f}", f"kp{kp:.0f}",
                          dict(S, gait="walk", duty=duty,
                               kp3=[kp * 0.5, kp, kp], kd3=[kd, kd, kd],
                               z_sag=mm.STATIC_SAG * 120.0 / kp)))
        # 對照組：現況
        P.append(("決選 duty 0.80", "kp120(現況)", dict(S, gait="walk")))

    if name == "gc_winner":
        # 決選格（duty 0.85 / kp 480）的離地量只有 55 mm。掃 g_c 確認它調得回來。
        for g in (0.08, 0.10, 0.12, 0.14, 0.16):
            P.append(("g_c @ d0.85 kp480", f"{g:.2f}",
                      dict(S, gait="walk", duty=0.85, g_c=g,
                           kp3=[240.0, 480.0, 480.0], kd3=[4.0, 4.0, 4.0],
                           z_sag=mm.STATIC_SAG * 120.0 / 480.0)))
        # 同一組 g_c 在較保守的 kp360 也掃一次 —— kp=480 是原廠站立值的 1.9 倍，
        # 拿它跟「沒調過 g_c 的 kp360」比是不公平的比較。
        for g in (0.08, 0.10, 0.12, 0.14):
            P.append(("g_c @ d0.85 kp360", f"{g:.2f}",
                      dict(S, gait="walk", duty=0.85, g_c=g,
                           kp3=[180.0, 360.0, 360.0], kd3=[3.0, 3.0, 3.0],
                           z_sag=mm.STATIC_SAG * 120.0 / 360.0)))
        # 順便確認配平：kp480/d0.85 的平均俯仰是 +0.10，x_off 幾乎不用動，掃三點看趨勢
        for v in (-0.050, -0.040, -0.030):
            P.append(("x_off @ d0.85 kp480", f"{v:.3f}",
                      dict(S, gait="walk", duty=0.85, x_off=v, g_c=0.12,
                           kp3=[240.0, 480.0, 480.0], kd3=[4.0, 4.0, 4.0],
                           z_sag=mm.STATIC_SAG * 120.0 / 480.0)))

    if name == "drift":
        # `walk_fast` 的慢漂不隨 x_off 變（見 plan=yaw），但**隨 d_step 變號**：
        # 20 s 掃描裡 0.08→+8.6°、0.10→+8.2°、0.13→−19.4°、0.16→−35.6°。
        # 若真的有過零點，那就能同時要「快」與「直」，不必先上偏航閉迴路。
        # ⚠️ 必須用 60 s —— 20 秒的偏航被起步暫態蓋過，看不出慢漂率。
        for v in (0.100, 0.110, 0.115, 0.120, 0.130):
            P.append(("d_step vs 慢漂 60s", f"{v:.3f}",
                      dict(gait="walk_fast", secs=60.0, d_step=v)))
        # 加速的**另一條路**：不加步幅，加頻率。20 s 掃描裡 ω=1.8 給到 0.335 m/s
        # 而彈跳比 ω=1.4 還小。若慢漂是跟著 d_step 而不是跟著速度，
        # 這條路就能同時要快與直 —— 那比先做偏航閉迴路便宜得多。
        for v in (1.4, 1.6, 1.8, 2.0):
            P.append(("omega vs 慢漂 60s", f"{v:.1f}",
                      dict(gait="walk", secs=60.0, omega=v)))
        # ★ 主張「walk 沒有系統性慢漂」之前，必須排除「只是漂得更慢」。
        #    60 s 的偏航率跨零，但 60 s 也只有 15 公尺。拉到 120 s（約 30 m）再驗一次：
        #    若真的沒漂，偏航總量不該隨時間長大。
        #    ★ 這一格救回一個錯誤結論：60 s 的偏航率跨零，看起來像「walk 不漂」，
        #      120 s 卻是 12 次全部 −46.9° ~ −25.4°、沒有一次靠近零。
        #      walk 只是**前 60 秒被起步暫態抵銷**。所以偏航率一定要看**區間斜率**，
        #      不能用「總偏航 ÷ 總秒數」—— 後者會被起步那一段永久稀釋。
        for t in (20.0, 60.0, 120.0, 180.0):
            P.append(("超長時程/walk", f"{t:.0f}s", dict(gait="walk", secs=t)))
        for t in (60.0, 120.0):
            P.append(("超長時程/walk_fast", f"{t:.0f}s", dict(gait="walk_fast", secs=t)))

    jobs = []
    for sweep, label, kw in P:
        for s in range(nseed):
            jobs.append({"cell": [sweep, label], "kw": kw, "seed": s})
    return jobs


# =============================================================================
# main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="full",
                    choices=("full", "trim", "params", "abad", "base", "yaw", "drift",
                             "omega_trim", "g1", "duty_kp", "duty_kp2", "gc_winner"))
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=6, help="每格的擾動數")
    ap.add_argument("--procs", type=int, default=None,
                    help="worker 上限；實際值仍會被可用記憶體壓下來")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    jobs = build_plan(a.plan, a.secs, a.seeds)
    n = safe_workers(a.procs)
    est = len(jobs) * a.secs * 0.17 / n
    print(f"[計畫] {a.plan}：{len(jobs) // a.seeds} 格 × {a.seeds} 擾動 = {len(jobs)} 次 "
          f"rollout，每次 {a.secs:.0f} s 模擬 → 估計 {est / 60:.1f} 分鐘")

    t0 = time.time()
    res: dict[tuple, list] = {}
    stopped = None
    ctx = mp.get_context("spawn")     # fork 會把父進程的記憶體一起帶走
    with ctx.Pool(n) as pool:
        try:
            for i, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=1), 1):
                res.setdefault(tuple(r["_cell"]), []).append(r)
                avail = mem_available_gb()
                if avail < FLOOR_GB:
                    stopped = (f"可用記憶體掉到 {avail:.2f} GB（< {FLOOR_GB} GB），"
                               f"在第 {i}/{len(jobs)} 筆收掉整池")
                    pool.terminate()
                    break
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  已花 {time.time() - t0:.0f}s  "
                          f"可用記憶體 {avail:.1f} GB", flush=True)
        except KeyboardInterrupt:
            pool.terminate()
            stopped = "使用者中斷"

    if stopped:
        print(f"\n⚠️ **掃描沒跑完**：{stopped}\n"
              f"   下面的表格只含已完成的格，**不完整的格不要引用**。\n")

    # ---- 輸出 ----
    print(f"\n{'=' * 118}")
    print(f"多擾動掃描  plan={a.plan}  每格 {a.seeds} 擾動（x_off ±{JITTER:g} m）  "
          f"{a.secs:.0f} s/次  共 {time.time() - t0:.0f} s")
    print("★ 判讀：全距（±後面那個數）比你要主張的差異還大時，那個差異不存在。")
    print("=" * 118)
    out = {"plan": a.plan, "secs": a.secs, "seeds": a.seeds, "jitter": JITTER,
           "stopped": stopped, "when": datetime.now().isoformat(timespec="seconds"),
           "cells": {}}
    # ⚠️ `res` 的插入順序是**完成順序**（imap_unordered），不是計畫順序。
    #    直接迭代它，表格的列會亂跳、掃描的單調趨勢會看不出來。照 jobs 的順序重排。
    order = list(dict.fromkeys(tuple(j["cell"]) for j in jobs))
    last = None
    for key in order:
        if key not in res:
            continue
        sweep, label = key
        rs = res[key]
        if sweep != last:
            print(f"\n── {sweep} ──")
            print(HDR)
            last = sweep
        ag = _agg(rs)
        if len(rs) < a.seeds:
            label += "*"          # 不完整的格，標記出來
        print(_row(label, ag))
        out["cells"][f"{sweep}|{label}"] = ag

    p = Path(a.out) if a.out else OUT_DIR / f"sweep_{a.plan}_{a.secs:.0f}s.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"\n[存檔] {p}")
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
