"""D1 Max（ZSM-1w / zgws）的開迴路 CPG 步態 —— 不含 RL，純 CPG。

用法：
  # 靜態站立驗證（先跑這個，確認落地站得住、量到真實機身高度與四輪受力）
  conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --stand

  # 步態
  conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --gait trot --video
  conda run --no-capture-output -n rbtdog python task7/inference/cpg_walk_max.py --gait walk --video

================================================================================
§1 控制架構：PD 在迴圈裡，不在 MJCF 裡
================================================================================
官方 MJCF 的致動器是**純力矩 `<motor>`**，沒有位置伺服也沒有 ctrlrange
（原廠的 PD 由外部 `mc_ctrl` 提供）。所以這裡自己算 PD：

    tau = KP·(q_des − q) − KD·q̇        再 clip 到 TAU_MAX

內迴圈跑在 MJCF 的 timestep 0.002 s = **500 Hz，剛好等於原廠 `controller_dt`**；
CPG 指令 50 Hz。增益取原廠 RL 那組（ABAD 60 / HIP 120 / KNEE 120、Kd 1.0）。

⚠️ 這改變了診斷方式。task6 的經典症狀是「指令超出 ctrlrange 被靜默 clip」，
   這台**沒有 ctrlrange，那個症狀不會發生**。取而代之要盯三個：

     lim%    IK 解出的角度超出 `jnt_range` 的比例（超限會被關節限位硬擋）
     tau%    PD 力矩打到 TAU_MAX 飽和的比例（飽和 = 實際增益比你以為的低）
     reach%  足端目標超出腿的可達球殼、被沿徑向縮限的比例

   三個都印在每次 rollout 的輸出裡。任何一個不是 0 就不能只看步態指標。

================================================================================
§2 輪子怎麼處理 —— ★ 預設 `damp`，這是實測選出來的，不是照抄原廠
================================================================================
原廠 RL 走路時輪子是 Kp=60 / Kd=0.5（真的有位置增益）。**但那個 Kp 是搭配
「每個控制步都重新給輪子目標角」的 RL 策略**；開迴路 CPG 沒有那一層，
把目標角凍結在起步當下就會愈拖愈遠、PD 愈拉愈用力。

實測（duty=0.80 的 walk 相位、ω=1.4、d_step=0.10、12 秒）：

    模式              前進    淨滾動   輪總行程    yaw     側偏
    hold（凍結目標）   935      1.0     1052    +39.1°   +616     ← 偏航失控
    hold+每步重鎖     2457     10.6     1482    +24.7°  +1124     ← 還是偏
    damp             2447     -6.2     2368     -2.4°   +364     ← ★ 採用
    free            -1550   2070.0     6947    -10.0°   +145     ← 變成往後滾

**「淨滾動」是四輪淨轉角換算的距離，用來回答「牠是在走還是在滾」。**
damp 的淨滾動只有 −6 mm 而前進 2447 mm → **確實是用腿走的**；
free 的淨滾動 2070 mm 且整台往後跑 → 那是輪子在空轉，不是步態。

偏航的來源就是 `hold`：凍結的目標角讓四顆輪子各自累積不同的角度誤差，
PD 把它換成接觸點上的縱向力，四個力不等就成了偏航力矩。
**證據是原地踏步（d_step=0）也照樣偏 −28°，而且把相位左右鏡像偏航不會換邊**——
所以不是步態的左右不對稱造成的，是輪子的控制律造成的。

★ 更新（2026-08-25，commit 9141393）：**輪關節的靜摩擦已經是實機實測值了**。
   MJCF 現在自帶 `frictionloss="0.15"`（四輪各 0.15 N·m，純庫倫、無黏滯，
   由實機四輪驅動四組獨立量測平均而來）。上面那張表是**在加摩擦之前**量的。
   加了之後本檔調好的參數**全部仍適用**：walk 0.19 → 0.21 m/s（+8%），
   彈跳／俯仰／支撐腳幾乎不變，跌倒仍是 0，診斷仍全 0%。

⚠️ `--wheel-friction` 現在可以**加也可以減**（守衛是 `is not None`，不是 `> 0`）。
   要拿「無摩擦」的對照組就 `--wheel-friction 0`。

================================================================================
§3 實測結果
================================================================================
見 `task7/docs/` 的結果文件（由 `--sweep` 產生的表格）。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # 無頭錄影用；必須在建立 Renderer 前設定

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco

import cpg_max
import gait_baseline as gb
import leg_kin
import max_model as mm

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# ---- 步態預設（每組都是掃出來的實測點，見結果文件的表）----
#
#   walk        ★ 建議用這組。最直、彈跳最小。**行進 0.148 m/s**
#   walk_fast   同樣的相位與佔空比，加大步幅。**行進 0.263 m/s**
#   trot        ❌ **不要用。** 見下。
#
# ⚠️⚠️ 2026-08-26 速度數字全部下修 —— **不是步態變慢了，是舊的度量是錯的**。
#      舊文件引用的 `speed_path` 是**逐控制步**累加的路徑長，把機身每一步的
#      左右搖擺一起算成前進。實測 walk 在幾乎不偏航時（20 s、偏航 +7.9°）：
#          帳面 x 0.143 ｜ ★行進 0.148 ｜ 路徑 0.249   ← 路徑高估 68%
#      cos(7.9°)=0.99，那 68% 不可能來自轉彎。改用 `speed_travel`
#      （以一個步態週期為步長，週期內搖擺自己抵銷）。它與帳面 x 在直線時一致，
#      而且**與時間長度無關**（walk 20 s 0.148、180 s 0.150，即使已轉了 −51°）。
#      → 之前所有引用「路徑速度 0.24~0.32 m/s」的地方，實際行進速度是 0.15~0.27。
#
# ⚠️ 佔空比是這台的硬條件：實測 duty ≤ 0.70 **一定跌倒**（24 秒內）。
#    0.75 勉強能走但彈跳 52 mm，0.80 才收斂到 28 mm。task6 的 D1 EDU 連 duty=0.50
#    的 trot 都走得動 —— 這台重一倍，低佔空比的空檔身體就撐不住了。
#
# ⚠️⚠️ **trot 對擾動是混沌的。** 把 x_off 擾動 1e-12 m（一皮米）跑 6 次：
#      walk      偏航全距   6.2°   速度 0.194–0.195  ← 穩
#      walk_fast 偏航全距  11.2°   速度 0.294–0.297  ← 穩
#      trot      偏航全距 128.3°   速度 0.110–0.464  ← **速度差 4 倍**
#    所以 trot 的任何單次數字都不能引用，也不能拿來調參。它留在這裡只是為了
#    佐證「這台不適合低佔空比」。**walk 系列的速度可以引用，偏航要當 ±3° 看。**
# ⚠️ x_off 隨輪摩擦而變。無摩擦時「平均俯仰過零」落在 −42 mm，
#    加上實測 frictionloss=0.15 之後移到 **−30 mm**。改摩擦就要重掃這一項。
#    （其餘三項 —— duty、mu_y、地面摩擦門檻 —— 加摩擦後結論完全沒變，已重驗。）
# ⚠️ `x_off` 每次改摩擦都要重掃 —— 它是**配平點**，不是只讓速度整體平移。
#    判準是**平均俯仰過零**（偏航太吵，在 −0.035↔−0.025 之間會從 −1.1° 跳到 −18.3°，
#    那是混沌不是趨勢）。歷次：無摩擦 −42 → 併入輪摩擦 0.15 後 −30
#    → 2026-08-26 併入實測腿關節摩擦 1.5 後 **−40**。
#
# ★ 12 擾動重掃（2026-08-26，`cpg_sweep_max.py --plan trim`）把這條判準**量化**了：
#     平均俯仰：全距 0.01–0.02°，而整個掃描範圍變化 0.84° → 雜訊只佔 2%。**可用。**
#     偏航    ：全距 3.6–11.7°，而整個掃描範圍變化約 18°   → 雜訊佔 20–60%。**不可用。**
#   這就是為什麼配平只能看俯仰。現在有數字可以引用，不必再靠印象。
GAITS = {
    # ★ walk 是**基準步態**，參數凍結在 `gait_baseline.py`，判準來源也寫在那裡。
    #   要改它必須連同 `docs/基準步態凍結_D1Max_walk_2026-08-27.md` 一起改。
    "walk": gb.walk_gait(),
    # ★ walk_fast 的配平點與 walk **不一樣**（2026-08-26 用 12 擾動重掃才看到）。
    #   平均俯仰過零：walk −41 mm、walk_fast **−46 mm**。之前兩組共用 −40 是有偏差的
    #   （walk_fast 在 −40 的平均俯仰是 −0.12°±0.01，−46 是 +0.05°±0.00）。
    #   配平點會隨 d_step 移動，不是只隨摩擦移動。
    "walk_fast": dict(phase=cpg_max.PHASE_WALK, duty=0.80, omega=1.4, mu_x=1.80,
                      x_off=-0.046, d_step=0.13, g_c=0.08),
    # trot 的 x_off 沒有重掃 —— 它的指標是混沌的，掃了也選不出東西。
    "trot": dict(phase=cpg_max.PHASE_TROT, duty=0.50, omega=3.0, mu_x=1.80,
                 x_off=-0.050, d_step=0.10, g_c=0.08),
}
MU_Y = gb.MU_Y        # → fy = 0，直線走路不需要橫擺（task6 §1-2；不歸零是側偏的主因）
D_STEP_Y = gb.D_STEP_Y  # 橫擺尺度。ABAD 力臂約 0.41 m（D1 EDU 只有 0.22），比 task6 寬鬆
SETTLE_S = 1.5    # 開走前先站穩。這台 41 kg，比 task6 的 0.8 s 需要更久


# ---- 模型快取 ----------------------------------------------------------------
# ⚠️ 這不是微優化，是**擋當機的**。一次 `make_model()` 要吃掉約 0.9 GB（網格），
#    而 `Robot()` 原本每個 rollout 都重建一次。單一進程掃 30 組會讓 RSS 一路衝到
#    1.6 GB 以上（glibc arena 不還給 OS），平行掃就直接把 16 GB 的機器 OOM 掉
#    —— 2026-08-26 就是這樣把開發機弄當的。
#
# 模型本身在整場掃描裡是**不變的**，會被改寫的只有下面幾個欄位。所以快取一份，
# 每次建 Robot 時**先還原成原始值再套用覆寫** —— 少了「還原」這步，上一次
# `--friction 0.3` 的設定會靜默滲進下一次跑，而且四個診斷指標全是乾淨的。
#
# ★ 快取是**按場景路徑分開**的。MJX 對照要同時用兩個場景（原始網格 / 圓柱），
#   共用單一快取會讓第二個場景**靜默拿到第一個場景的模型** ——
#   那樣 G1 對照就變成「同一個模型跟自己比」，看起來完美通過。
_CACHE: dict = {}


def _model(scene: str = None):
    """取得共用模型，並把可被改寫的欄位還原成 MJCF 的原值。"""
    path = scene or mm.SCENE
    if path not in _CACHE:
        m = mujoco.MjModel.from_xml_path(path)
        _CACHE[path] = (m, m.geom_friction.copy(), m.dof_frictionloss.copy(),
                        int(m.opt.iterations), int(m.opt.ls_iterations))
    m, gf, fl, it, ls = _CACHE[path]
    m.geom_friction[:] = gf
    m.dof_frictionloss[:] = fl
    m.opt.iterations, m.opt.ls_iterations = it, ls
    return m


class Robot:
    """MuJoCo 模型 + 迴圈內 PD。所有 rollout 共用，確保控制律只有一份。"""

    def __init__(self, friction: float = None, wheel_friction: float = None,
                 leg_friction: float = None, abad_friction: float = None,
                 scene: str = None, actuator_mode: str = "torque_pd",
                 solver_iters: tuple = None):
        assert actuator_mode in ("torque_pd", "position"), \
            f"actuator_mode 只能是 torque_pd / position，收到 {actuator_mode!r}"
        self.m = _model(scene)
        self.actuator_mode = actuator_mode
        if solver_iters is not None:
            # MuJoCo 預設 100/50。MJX 沒有提早收斂，迭代數是**固定成本**，
            # 訓練模型必須調低。已實測對 walk 的影響可忽略
            # （travel 0.1471→0.1467、bounce 16.7→17.1、support 3.204→3.206）。
            self.m.opt.iterations, self.m.opt.ls_iterations = solver_iters
        if friction is not None:
            self.m.geom_friction[:, 0] = friction
        # ⚠️ 守衛是 `is not None` 不是 `> 0`。MJCF 現在自帶 frictionloss=0.15（實機實測），
        #    若寫成 `if wheel_friction > 0` 就**只能加不能減**，
        #    `--wheel-friction 0` 會被靜默忽略、拿不回無摩擦的對照組。
        if wheel_friction is not None:
            self.m.dof_frictionloss[mm.WHEEL_QVEL_IDX] = wheel_friction
        # 腿關節同理。MJCF 自帶 1.5（2026-08-26 實機量到的靜摩擦掙脫門檻），
        # `--leg-friction 0` 可取回舊的無摩擦對照組。
        if leg_friction is not None:
            self.m.dof_frictionloss[mm.LEG_QVEL_IDX] = leg_friction
        # ABAD 單獨覆寫。★ 必須排在 leg_friction 之後 —— 它是 leg_friction 的子集，
        #   順序反過來會被整組覆蓋掉而**靜默失效**。
        #   會需要這個是因為 ABAD 的 1.85 是**下界不是量測值**（見結果文件），
        #   要能問「步態到底吃不吃這個數字」就得單獨掃它。
        if abad_friction is not None:
            self.m.dof_frictionloss[mm.LEG_QVEL_IDX[::3]] = abad_friction
        # position 模式要求模型的致動器本身是 affine 的位置伺服。
        # ⚠️ biastype 不是 affine 時 `ctrl` 會被當**力矩**直接施加，機器人當場塌掉，
        #    而且不會有任何錯誤訊息（task4 地形版踩過）。
        bt = self.m.actuator_biastype[mm.LEG_ACT_IDX]
        if actuator_mode == "position":
            assert np.all(bt == mujoco.mjtBias.mjBIAS_AFFINE), \
                f"position 模式需要 affine 致動器，實得 biastype={bt}"
        else:
            # ★ 反向也要擋。把力矩寫進「位置伺服」的 ctrl，等於在命令
            #   「目標角 = 42 rad」—— 機器人會瘋掉，但**不會有任何錯誤訊息**，
            #   而且超限／飽和／IK縮限三個診斷指標仍然是 0.00%
            #   （它們量的是我們算出來的東西，不是模型怎麼解讀它）。
            #   2026-08-27 的 G1 第一版就是這樣得到「換形狀害 5/12 跌倒」的假結論。
            assert np.all(bt == mujoco.mjtBias.mjBIAS_NONE), (
                f"torque_pd 模式需要純力矩致動器（biastype=none），實得 {bt}。"
                "這個模型的致動器是位置伺服，力矩會被當成目標角。")
        self.d = mujoco.MjData(self.m)
        self.foot_bid = mm.foot_body_ids(self.m)
        self.jnt_rng = mm.leg_joint_ranges(self.m)
        self.n_sub = int(round(mm.CTRL_DT / self.m.opt.timestep))
        self.kp = np.tile(mm.KP3, 4)
        self.kd = np.tile(mm.KD3, 4)
        self.tau_max = np.tile(mm.TAU_MAX3, 4)
        # 診斷計數器
        self.n_cmd = 0          # 送出的關節指令數（= 控制步數 × 12）
        self.n_lim = 0          # 其中超出 jnt_range 的個數
        self.n_tau = 0          # PD 力矩飽和的個數（以 500 Hz 內迴圈計）
        self.n_tau_tot = 0
        self.wheel_hold = None  # None = 只做阻尼

    def reset_standing(self, q12: np.ndarray, height: float):
        """把機器人放在指定高度、指定關節角，機身水平、速度歸零。"""
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[:3] = [0.0, 0.0, height]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[mm.LEG_QPOS_IDX] = q12
        mujoco.mj_forward(self.m, d)

    def step(self, q_des: np.ndarray, wheel_mode: str = "damp"):
        """套用一個 50 Hz 控制週期：內部跑 n_sub 次 500 Hz 的 PD + mj_step。"""
        d, m = self.d, self.m
        if self.actuator_mode == "position" and wheel_mode != "damp":
            raise ValueError("position 模式的輪子由模型的 velocity 致動器固定成阻尼，"
                             f"不支援 wheel_mode={wheel_mode!r}")
        lo, hi = self.jnt_rng[:, 0], self.jnt_rng[:, 1]
        # 診斷：指令有沒有超出關節限位。超限不會被靜默吃掉（沒有 ctrlrange），
        # 但會被關節限位約束硬擋住 —— 症狀一樣是「命令了卻沒動到」。
        self.n_lim += int(np.sum((q_des < lo - 1e-9) | (q_des > hi + 1e-9)))
        self.n_cmd += 12
        q_des = np.clip(q_des, lo, hi)

        if self.actuator_mode == "position":
            # 模型內建的 position 致動器**每個物理步**自己算 PD，也就是 500 Hz ——
            # 剛好等於原廠 controller_dt。我們只要把目標角放進 ctrl。
            # 輪子用 velocity 致動器、目標速度恆 0 ＝ 純阻尼，與 torque_pd 的 damp 等價。
            d.ctrl[mm.LEG_ACT_IDX] = q_des
            d.ctrl[mm.WHEEL_ACT_IDX] = 0.0
            for _ in range(self.n_sub):
                mujoco.mj_step(m, d)
                # 飽和診斷改用致動器**實際出力**（position 模式下我們算不到 tau）。
                self.n_tau += int(np.sum(
                    np.abs(d.actuator_force[mm.LEG_ACT_IDX]) > self.tau_max - 1e-6))
                self.n_tau_tot += 12
            return

        for _ in range(self.n_sub):
            q = d.qpos[mm.LEG_QPOS_IDX]
            qd = d.qvel[mm.LEG_QVEL_IDX]
            tau = self.kp * (q_des - q) - self.kd * qd
            self.n_tau += int(np.sum(np.abs(tau) > self.tau_max))
            self.n_tau_tot += 12
            d.ctrl[mm.LEG_ACT_IDX] = np.clip(tau, -self.tau_max, self.tau_max)

            wv = d.qvel[mm.WHEEL_QVEL_IDX]
            if wheel_mode == "hold" and self.wheel_hold is not None:
                wq = d.qpos[mm.WHEEL_QPOS_IDX]
                wtau = mm.KP_WHEEL * (self.wheel_hold - wq) - mm.KD_WHEEL * wv
            elif wheel_mode == "free":
                wtau = np.zeros(4)
            else:                                    # damp
                wtau = -mm.KD_WHEEL * wv
            d.ctrl[mm.WHEEL_ACT_IDX] = np.clip(wtau, -mm.TAU_MAX_WHEEL, mm.TAU_MAX_WHEEL)

            mujoco.mj_step(m, d)

    def lock_wheels(self):
        self.wheel_hold = self.d.qpos[mm.WHEEL_QPOS_IDX].copy()

    def foot_heights(self) -> list[float]:
        """四輪的離地高度（輪底相對地面），順序同 LEGS。"""
        return [float(self.d.xpos[b][2]) - mm.WHEEL_RADIUS for b in self.foot_bid]

    def foot_forces(self) -> np.ndarray:
        """四輪的地面法向接觸力（N），順序同 LEGS。"""
        f = np.zeros(4)
        body_of = {b: k for k, b in enumerate(self.foot_bid)}
        buf = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            k = body_of.get(b1, body_of.get(b2))
            if k is None:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, buf)
            # contact frame 的 x 軸是法向；力的方向對 geom1/geom2 相反，取絕對值即可
            f[k] += abs(buf[0])
        return f

    @property
    def lim_pct(self) -> float:
        return 100.0 * self.n_lim / max(1, self.n_cmd)

    @property
    def tau_pct(self) -> float:
        return 100.0 * self.n_tau / max(1, self.n_tau_tot)


class Trace:
    """逐步取樣 + 統計 —— **開迴路 rollout 與 RL 推論共用同一份定義**。

    ⚠️ 不要在別處複製第二份統計。兩份遲早分岔，而分岔之後
       「RL 比開迴路快 x%」這種結論就不可信了：那個百分比會同時包含
       真實差異與兩份統計程式的差異，事後分不開。

    統計只取**後半段**（前半當暖身）。判定規則寫在各欄位旁邊。
    """

    def __init__(self, robot: "Robot", n_steps: int, secs: float, omega: float,
                 phase: np.ndarray):
        self.r, self.n, self.secs = robot, n_steps, secs
        self.omega = omega
        self.phase = np.asarray(phase)
        self.half = n_steps // 2
        self.i = 0
        d = robot.d
        self.x0, self.y0 = float(d.qpos[0]), float(d.qpos[1])
        self.yaw0 = cpg_max.yaw_deg(d.qpos[3:7])
        self.w0 = d.qpos[mm.WHEEL_QPOS_IDX].copy()
        # 路徑長：步態一偏航就走弧線，「x 位移 / 時間」會**嚴重低估**實際走多快。
        self.path_len = 0.0
        self.prev_xy = np.array([self.x0, self.y0])
        # ★ 逐步的 xy 全留著。`path_len` 是**逐控制步**累加的，會把機身每一步的
        #   左右搖擺一起算成「走過的距離」。要分開就得能用**一個步態週期**當步長重算。
        self.xy_hist = np.empty((n_steps, 2))
        self.lift = [[] for _ in range(4)]
        self.pitch, self.roll, self.height, self.support, self.phases = [], [], [], [], []
        self.fell = None
        # ★ 累積偏航（不包裹）。`yaw_deg` 用 atan2，值域 (−180, 180]，
        #   首尾相減在**轉彎**時會包裹而給出完全錯誤的值 ——
        #   實測：指令 wz=+0.3 rad/s 跑 20 秒（真實轉了 +224.6°），
        #   首尾相減得到 **−135.4°**，符號還是反的，看起來像「偏航指令接反了」。
        #   開迴路直走時兩者相同（總偏航 < 60°，不會包裹），所以這個坑一直沒露出來。
        #   逐步累加折回 ±180 的增量就不會包裹。
        self.yaw_total = 0.0
        self._yaw_prev = self.yaw0

    def record(self, theta: np.ndarray):
        """在 `robot.step()` 之後呼叫一次。`theta` 是當下的 CPG 相位 (4,)。"""
        d, r, i = self.r.d, self.r, self.i
        xy = np.array([float(d.qpos[0]), float(d.qpos[1])])
        self.path_len += float(np.linalg.norm(xy - self.prev_xy))
        self.prev_xy = xy
        self.xy_hist[i] = xy

        yaw_now = cpg_max.yaw_deg(d.qpos[3:7])
        self.yaw_total += (yaw_now - self._yaw_prev + 180.0) % 360.0 - 180.0
        self._yaw_prev = yaw_now

        grav = cpg_max.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        if grav[2] > mm.FALL_GRAV_Z and self.fell is None:
            self.fell = i * mm.CTRL_DT
        if i >= self.half:
            hs = r.foot_heights()
            for k in range(4):
                self.lift[k].append(hs[k])
            # 支撐腳數用**實際接觸力**判定，不用「離地高度 < 5 mm」。
            # 高度門檻在會彈跳的步態上會騙人：機身整體騰空時腳離地面很近卻沒受力，
            # 一樣被算成支撐腳。改用接觸力就沒有這個模糊地帶。
            self.support.append(int(np.sum(r.foot_forces() > 1.0)))
            self.pitch.append(np.degrees(np.arcsin(np.clip(-grav[0], -1.0, 1.0))))
            self.roll.append(np.degrees(np.arcsin(np.clip(grav[1], -1.0, 1.0))))
            self.height.append(float(d.qpos[2]))
            self.phases.append(np.asarray(theta).copy())
        self.i += 1

    def summarize(self, n_reach: int = 0, extra: dict = None) -> dict:
        d, r = self.r.d, self.r
        pit = np.asarray(self.pitch)
        hgt = np.asarray(self.height)
        per = max(1, int(round((1.0 / self.omega) / mm.CTRL_DT)))
        # secs 太短時取樣不足一個週期，退回整段 p2p —— np.mean([]) 會回 nan
        # 而且只發 RuntimeWarning，不會擋下來。
        cyc = [np.max(pit[s:s + per]) - np.min(pit[s:s + per])
               for s in range(0, len(pit) - per, per)] or [float(pit.max() - pit.min())]

        # 相位鎖定：實際相位差 vs 目標相位差，用圓形統計
        # （±180° 的包裹會讓一般標準差虛胖，看起來像沒鎖定其實鎖得很好）。
        #
        # ⚠️ 這是一個**在開迴路下必然漂亮的指標**，不要拿它當步態品質的證據。
        #    CPG 是純前饋的，機身狀態不回授進振盪器，所以相位差恆等於設定值、σ 恆為 0，
        #    就算機器人已經翻倒在地也一樣是 0。
        #    ★ 接上 RL 之後它才**開始有資訊量** —— policy 會逐步改 ω，
        #      相位差就不再恆定，這時 σ 才真的在量「步態有沒有散掉」。
        ph = np.asarray(self.phases)
        lock = [float(np.degrees(cpg_max.circ_std(
            ph[:, k] - ph[:, 0] - (self.phase[k] - self.phase[0])))) for k in range(4)]

        # ---- 行進速度：三個都留，因為三個回答的是不同問題 ----
        #   speed        帳面。x 位移 / 時間。**一偏航就嚴重低估。**
        #   speed_path   逐控制步的路徑長 / 時間。**把機身搖擺算成前進，嚴重高估**（+68%）。
        #   speed_travel ★ 以**一個步態週期**為步長重算的路徑長 / 時間。
        #                週期內的搖擺自己抵銷掉，弧線仍然算得到 —— 這個才是「走多快」。
        xy = self.xy_hist
        stride = (np.linalg.norm(xy[per:] - xy[:-per], axis=1)
                  if self.n > per else None)
        speed_travel = (float(stride.sum()) / (len(stride) * per) / mm.CTRL_DT
                        if stride is not None and len(stride) else float("nan"))
        net = float(np.linalg.norm(xy[-1] - xy[0]))

        res = {
            "peak_lift": [float(np.percentile(l, 99)) for l in self.lift],
            "min_lift": float(min(np.percentile(l, 99) for l in self.lift)),
            "pitch_cycle": float(np.mean(cyc)),
            "pitch_mean": float(pit.mean()),
            "roll_mean": float(np.mean(self.roll)),
            "bounce": float(hgt.max() - hgt.min()),
            "height": float(hgt.mean()),
            "support": float(np.mean(self.support)),
            "dist": float(d.qpos[0]) - self.x0,
            "lateral": float(d.qpos[1]) - self.y0,
            "speed": (float(d.qpos[0]) - self.x0) / self.secs,
            "path_len": self.path_len,
            "speed_path": self.path_len / self.secs,  # ⚠️ 含機身搖擺，別當行進速度引用
            "speed_travel": speed_travel,             # ★ 搖擺抵銷後的行進速度
            "net_disp": net,                          # 起點到終點的直線距離
            "speed_net": net / self.secs,             # 走弧線時會低於 speed_travel
            # ⚠️ `yaw` 是首尾相減，**只在總偏航 < 180° 時可用**（開迴路直走的情況）。
            #    轉彎時一定要看 `yaw_total`，否則會包裹成錯誤的值甚至錯誤的符號。
            "yaw": cpg_max.yaw_deg(d.qpos[3:7]) - self.yaw0,
            "yaw_total": self.yaw_total,        # ★ 逐步累積、不包裹。轉彎時看這個
            # 淨滾動距離：回答「牠是在走還是在滾」。輪軸 +y，前進對應輪角減少。
            "net_roll": float(-np.mean(d.qpos[mm.WHEEL_QPOS_IDX] - self.w0)
                              * mm.WHEEL_RADIUS),
            "fell": self.fell,
            "lim_pct": r.lim_pct,
            "tau_pct": r.tau_pct,
            "reach_pct": 100.0 * n_reach / max(1, self.n * 4),
            "phase_lock": lock,
        }
        res.update(extra or {})
        return res


def report(res: dict, header: str) -> None:
    """把 `Trace.summarize` 的結果印成四行 —— 開迴路與 RL 推論用同一份格式。"""
    pk = [v * 1000 for v in res["peak_lift"]]
    print(header)
    print("[離地] " + "  ".join(f"{L}={v:.1f}" for L, v in zip(mm.LEGS, pk))
          + f"  mm（最小 {res['min_lift'] * 1000:.1f}）")
    print(f"[姿態] 週期俯仰 {res['pitch_cycle']:.2f}°  平均俯仰 {res['pitch_mean']:+.2f}°  "
          f"平均側傾 {res['roll_mean']:+.2f}°  彈跳 {res['bounce'] * 1000:.1f} mm  "
          f"機身高 {res['height'] * 1000:.1f} mm  支撐腳 {res['support']:.2f}")
    print(f"[位移] 前進 {res['dist']:+.2f} m（帳面 {res['speed']:.2f}，"
          f"★行進 {res['speed_travel']:.2f}，路徑 {res['speed_path']:.2f} m/s"
          f"｜路徑含機身搖擺，勿當行進速度引用）  "
          f"側偏 {res['lateral']:+.2f} m  偏航 {res['yaw']:+.1f}°  "
          f"淨滾動 {res['net_roll'] * 1000:+.0f} mm  "
          f"跌倒={'是 @%.1fs' % res['fell'] if res['fell'] is not None else '否'}")
    print(f"[診斷] 超限 {res['lim_pct']:.2f}%  力矩飽和 {res['tau_pct']:.2f}%  "
          f"IK縮限 {res['reach_pct']:.2f}%  "
          f"相位鎖定σ " + "/".join(f"{v:.1f}" for v in res["phase_lock"]) + "°")


def stand(secs: float = 4.0, x_off: float = 0.0, friction: float = None,
          wheel_mode: str = "damp", quiet: bool = False) -> dict:
    """靜態站立驗證（動態落地版，不是正向運動學）。

    交接文件 §8-3 要的就是這個：**讓它在平地上真的站起來並收斂**，
    量位置伺服的靜態撓度、四輪受力分佈、以及走起來之前的實際機身高度。
    """
    r = Robot(friction=friction)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    q_des = cpg_max.stand_targets(ks, f0, x_off)

    # 從純運動學高度多放 5 mm 落下，避免初始就穿模
    r.reset_standing(q_des, mm.NOMINAL_HEIGHT_KIN + 0.005)
    r.lock_wheels()

    n = int(secs / mm.CTRL_DT)
    for i in range(n):
        r.step(q_des, wheel_mode)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()          # 落地穩定後才鎖輪角

    d = r.d
    h = float(d.qpos[2])
    grav = cpg_max.w2b(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
    pitch = float(np.degrees(np.arcsin(np.clip(-grav[0], -1.0, 1.0))))
    q = d.qpos[mm.LEG_QPOS_IDX].copy()
    forces = r.foot_forces()
    # 靜態質心相對四輪支撐中心的前後偏移
    com = float(d.subtree_com[0][0]) - float(d.qpos[0])
    wheel_x = np.array([float(d.xpos[b][0]) for b in r.foot_bid]) - float(d.qpos[0])

    res = {"height": h, "pitch": pitch, "forces": forces.tolist(),
           "front_rear_ratio": float(forces[:2].sum() / max(1e-9, forces[2:].sum())),
           "sag": float(np.abs(q - q_des).max()),
           "com_x": com, "support_center_x": float(wheel_x.mean()),
           "com_offset": com - float(wheel_x.mean()),
           "tau_pct": r.tau_pct, "lim_pct": r.lim_pct,
           "settled": abs(h - mm.NOMINAL_HEIGHT_KIN) < 0.10}

    if not quiet:
        print(f"[站立] x_off={x_off * 1000:+.0f} mm  摩擦={friction or '預設1.0'}  輪={wheel_mode}")
        print(f"  機身高度  {h * 1000:.1f} mm（純運動學 {mm.NOMINAL_HEIGHT_KIN * 1000:.1f} mm，"
              f"位置伺服靜態撓度 {(mm.NOMINAL_HEIGHT_KIN - h) * 1000:+.1f} mm）")
        print(f"  俯仰      {pitch:+.2f}°   最大關節追蹤誤差 {np.degrees(res['sag']):.2f}°")
        print("  四輪受力  " + "  ".join(f"{L}={v:6.1f}" for L, v in zip(mm.LEGS, forces))
              + f"  N（總 {forces.sum():.1f}，前/後 = {res['front_rear_ratio']:.3f}）")
        print(f"  質心前後偏移 {res['com_offset'] * 1000:+.2f} mm（相對四輪支撐中心）")
        print(f"  力矩飽和 {r.tau_pct:.2f}%   指令超限 {r.lim_pct:.2f}%")
    return res



def rollout(gait: str = "trot", secs: float = 20.0, omega: float = None,
            mu_x: float = None, mu_y: float = MU_Y, x_off: float = None,
            g_c: float = None, d_step: float = None, d_step_y: float = D_STEP_Y,
            duty: float = None, friction: float = None, wheel_mode: str = "damp",
            wheel_friction: float = None, leg_friction: float = None,
            abad_friction: float = None, z_sag: float = None, video: bool = False,
            scene: str = None, actuator_mode: str = "torque_pd",
            solver_iters: tuple = None, quiet: bool = False) -> dict:
    """開迴路步態 rollout。未指定的參數取 `GAITS[gait]` 的預設值。"""
    cfg = GAITS[gait]
    phase = cfg["phase"]
    omega = cfg["omega"] if omega is None else omega
    mu_x = cfg["mu_x"] if mu_x is None else mu_x
    x_off = cfg["x_off"] if x_off is None else x_off
    duty = cfg["duty"] if duty is None else duty
    g_c = cfg["g_c"] if g_c is None else g_c
    d_step = cfg["d_step"] if d_step is None else d_step
    z_sag = mm.STATIC_SAG if z_sag is None else z_sag

    r = Robot(friction=friction, wheel_friction=wheel_friction,
              leg_friction=leg_friction, abad_friction=abad_friction,
              scene=scene, actuator_mode=actuator_mode, solver_iters=solver_iters)
    ks = leg_kin.knee_sign_of(mm.HOME)
    f0 = leg_kin.home_foot(mm.HOME)
    step = cpg_max.make_cpg_step(phase)
    n_reach = 0

    # 先用「帶 x_off 的站姿」站穩，而不是 HOME —— 否則第一步要從偏移過的基準跳過來。
    r.reset_standing(cpg_max.stand_targets(ks, f0, x_off), mm.NOMINAL_HEIGHT_KIN + 0.005)
    for i in range(int(SETTLE_S / mm.CTRL_DT)):
        r.step(cpg_max.stand_targets(ks, f0, x_off), wheel_mode)
        if i == int(0.5 / mm.CTRL_DT):
            r.lock_wheels()

    ren = cam = None
    frames = []
    if video:
        r.m.vis.global_.offwidth, r.m.vis.global_.offheight = 1000, 600
        ren = mujoco.Renderer(r.m, 600, 1000)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(r.m, cam)

    c = cpg_max.cpg_init(phase)
    n = int(secs / mm.CTRL_DT)
    d = r.d
    # ★ 取樣與統計都在 Trace 裡，與 RL 推論端 (local_infer_max.py) 共用同一份定義。
    #   不要在推論端複製第二份：兩份遲早分岔，分岔之後
    #   「RL 比開迴路快 x%」會同時包含真實差異與兩份程式的差異，事後分不開。
    tr = Trace(r, n, secs, omega, phase)

    for i in range(n):
        c = step(c, np.full(4, mu_x), np.full(4, mu_y), np.full(4, omega), mm.CTRL_DT)
        q_des, nc = cpg_max.joint_targets(c, f0, x_off, g_c, d_step, d_step_y, duty,
                                          ks, z_sag)
        n_reach += nc
        r.step(q_des, wheel_mode)
        tr.record(c["theta"])

        if ren is not None and i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.30]
            cam.distance, cam.elevation, cam.azimuth = 2.0, -10, 90
            ren.update_scene(d, cam)
            frames.append(ren.render())

    res = tr.summarize(n_reach, extra={
        "actuator_mode": actuator_mode,
        "scene": scene or mm.SCENE,
        "gait": gait, "omega": omega, "mu_x": mu_x, "x_off": x_off,
        "g_c": g_c, "d_step": d_step, "duty": duty, "z_sag": z_sag,
    })

    if not quiet:
        report(res, f"[步態] {gait}  ω={omega} μx={mu_x} μy={mu_y} duty={duty} "
                    f"x_off={x_off * 1000:+.0f}mm D_STEP={d_step} G_C={g_c} "
                    f"撓度補償={z_sag * 1000:.1f}mm 輪={wheel_mode}")

    if frames:
        import imageio.v2 as iio
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"cpg_{gait}_max.mp4"
        iio.mimsave(str(out), frames, fps=25, codec="libx264")
        print("[影片]", out)
    return res



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stand", action="store_true", help="只做靜態站立驗證")
    ap.add_argument("--gait", choices=sorted(GAITS), default="trot")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--mu-x", type=float, default=None, dest="mu_x")
    ap.add_argument("--mu-y", type=float, default=MU_Y, dest="mu_y")
    ap.add_argument("--x-off", type=float, default=None, dest="x_off")
    ap.add_argument("--g-c", type=float, default=None, dest="g_c")
    ap.add_argument("--d-step", type=float, default=None, dest="d_step")
    ap.add_argument("--d-step-y", type=float, default=D_STEP_Y, dest="d_step_y")
    ap.add_argument("--duty", type=float, default=None)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--wheel-mode", choices=("hold", "damp", "free"), default="damp",
                    dest="wheel_mode")
    ap.add_argument("--wheel-friction", type=float, default=None, dest="wheel_friction",
                    help="覆寫輪關節 frictionloss；MJCF 預設已是實測的 0.15，給 0 可拿無摩擦對照")
    ap.add_argument("--leg-friction", type=float, default=None, dest="leg_friction",
                    help="覆寫腿關節 frictionloss；MJCF 預設已是實測的 1.5，"
                         "給 0 可拿回無摩擦對照組（2026-08-26 之前的行為）")
    ap.add_argument("--abad-friction", type=float, default=None, dest="abad_friction",
                    help="只覆寫四個 ABAD 的 frictionloss（套在 --leg-friction 之後）；"
                         "MJCF 預設 1.85 是下界不是量測值，用這個掃它的敏感度")
    ap.add_argument("--z-sag", type=float, default=None, dest="z_sag")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--sweep", type=str, default=None,
                    help="掃描一個參數，例如 --sweep omega=1.2,1.4,1.6")
    a = vars(ap.parse_args())
    sweep = a.pop("sweep")
    if sweep:
        key, vals = sweep.split("=")
        a.pop("stand")
        print(f"{key:>18} | {'跌倒':>5}{'速度':>7}{'偏航°':>7}{'側偏mm':>8}{'離地mm':>8}"
              f"{'彈跳':>7}{'週期俯仰':>9}{'支撐':>6}{'高':>7}{'超限%':>7}{'飽和%':>7}")
        for v in vals.split(","):
            a[key] = float(v)
            r = rollout(**dict(a, quiet=True))
            print(f"{v:>18} | {('是' if r['fell'] else '否'):>5}{r['speed']:>7.2f}"
                  f"{r['yaw']:>7.1f}{r['lateral'] * 1000:>8.0f}"
                  f"{r['min_lift'] * 1000:>8.1f}{r['bounce'] * 1000:>7.1f}"
                  f"{r['pitch_cycle']:>9.2f}{r['support']:>6.2f}"
                  f"{r['height'] * 1000:>7.1f}{r['lim_pct']:>7.2f}{r['tau_pct']:>7.2f}")
    elif a.pop("stand"):
        stand(secs=min(a["secs"], 5.0), x_off=a["x_off"] or 0.0,
              friction=a["friction"], wheel_mode=a["wheel_mode"])
    else:
        rollout(**a)
