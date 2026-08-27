"""D1 Max 輪足版（ZSM-1w / zgws）模型與常數的**單一事實來源**。

task7 所有腳本的常數都必須從這裡取，不得各自寫死。

數值出處（每一項都標明是「官方檔案」「規格書」還是「我們量的」）：
  - 幾何 / 質量 / 限位：官方 MJCF `task7/model/zgws/zgws.xml`（MATRiX v0.1.2 原檔）
  - 增益 / 站姿 / 抬腿高度：官方運控設定檔 `task7/reference/matrix_zgws/`
    （解讀見 `task7/docs/D1Max_原廠運控參數_MATRiX解包_2026-08-25.md`）
  - 扭矩上限：規格書（比 MJCF 的 actuatorfrcrange 保守，見 TAU_MAX）
  - CPG 結構常數：沿用 task4 論文標準版，經 task6 在 D1 EDU 上驗證過的那組

⚠️ 與 task6 的 `d1_model` 最重要的結構差異：**站姿是前後鏡像的 X 型**。
   task6 的 `HOME3` 是四腿共用一個三元組，這台不行——`HOME` 在這裡是 (4,3)。
"""
from pathlib import Path

import mujoco
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "zgws"
SCENE = str(_MODEL_DIR / "scene_flat.xml")

# MJX / Colab 訓練用的場景（產生器 `model/zgws/make_mjx_model.py`）。
# 碰撞網格換成原始形狀、致動器換成位置伺服、關掉自碰撞、無 STL 相依。
# ⚠️ 它與 SCENE **不是同一個物理模型**，差異的量化對照見
#    `task7/docs/MJX模型對照_2026-08-27.md`。引用數字時要標明是哪一個。
SCENE_MJX = str(_MODEL_DIR / "scene_flat_mjx.xml")

# =============================================================================
# 機構
# =============================================================================
# 腿序取原廠 `zg_wheels-motion_config.yaml` 的 FR/FL/RR/RL，剛好也是 MJCF 的 qpos 順序，
# 省掉一層重排。用 tuple：腿序一旦被就地改寫，PHASE_* 的對角關係會靜默失效。
LEGS = ("FR", "FL", "RR", "RL")

# MJCF 用的是機構代號，SDK / 感測器用的是方位名。對應關係由 MJCF 自己的 <sensor> 段確認
# （`FR_hip_pos` 綁 `FAR_ABAD_JOINT`，餘類推），不是我們猜的。
PREFIX = {"FR": "FAR", "FL": "FBL", "RR": "RAR", "RL": "RBL"}

# 每腿的左右號（y 軸）與前後號（x 軸）。ABAD 的 y 偏移、abad→hip 的 x 偏移都靠這兩個決定。
SIDE_Y = np.array([-1.0, +1.0, -1.0, +1.0])   # FR, FL, RR, RL：右腿 −1
SIDE_X = np.array([+1.0, +1.0, -1.0, -1.0])   # 前腿 +1

# 連桿尺寸（m），逐項對過官方 MJCF 的 body pos
HIP_X = 0.2698          # base → ABAD 關節，x
HIP_Y = 0.065           # base → ABAD 關節，y
ABAD_TO_HIP_X = 0.0587  # ABAD → HIP 關節，x（★ 後腿是 −0.0587，用 SIDE_X）
ABAD_TO_FOOT_Y = 0.045 + 0.0522 + 0.0088   # = 0.1060，ABAD 之後的 y 偏移總和
L_THIGH = 0.26          # HIP → KNEE
L_SHANK = 0.28          # KNEE → FOOT（輪心）

# 輪半徑：從官方 FOOT 碰撞網格實算 0.0961 m。
# 規格書寫 0.09、原廠設定檔 `feet_z_offset` 寫 0.095 —— 三個數字都不一樣。
# 這裡取實算值，因為觸地判定要跟模擬裡真的碰到地板的那個面一致。
WHEEL_RADIUS = 0.0961

# =============================================================================
# 關節在 qpos / qvel 裡的位址
# =============================================================================
# ⚠️ 12 個腿關節在 qpos 裡**不連續**（每 3 個之後夾一個輪關節）。
#    `qpos[7:19]` 這種寫法會把輪角當關節角讀進去，IK 變奇異矩陣而不報錯。
#    task6 踩過這個坑，這裡明列位址，並由 test_max_model 對名稱查詢逐項驗證。
LEG_QPOS_IDX = np.array([7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21])
LEG_QVEL_IDX = np.array([6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])
WHEEL_QPOS_IDX = np.array([10, 14, 18, 22])
WHEEL_QVEL_IDX = np.array([9, 13, 17, 21])

# 致動器順序與 MJCF 的 <actuator> 段一致：每腿 abad/hip/knee/foot 連續四個。
LEG_ACT_IDX = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
WHEEL_ACT_IDX = np.array([3, 7, 11, 15])

# =============================================================================
# 姿態（原廠運控設定檔，腿序 FR/FL/RR/RL）
# =============================================================================
# ★ 前兩腿與後兩腿 hip/knee 反號 —— 站姿是 X 型。四腿共用一組 HOME 會做出怪東西。
# 三組姿態都用 MJCF 正向運動學驗證過（見 task7/model/zgws/SOURCE.md）。
STAND = np.array([[0.0, 0.6, -1.2],      # FR   機身離地 0.5418 m
                  [0.0, 0.6, -1.2],      # FL
                  [0.0, -0.6, 1.2],      # RR
                  [0.0, -0.6, 1.2]])     # RL
HOME = np.array([[0.0, 0.8, -1.5],       # FR   機身離地 0.4914 m ← 對上原廠 body_height 0.48
                 [0.0, 0.8, -1.5],       # FL   走路的名目站姿用這組，不是 STAND
                 [0.0, -0.8, 1.5],       # RR
                 [0.0, -0.8, 1.5]])      # RL
CROUCH = np.array([[0.0, 1.4, -2.4], [0.0, 1.4, -2.4],
                   [0.0, -1.4, 2.4], [0.0, -1.4, 2.4]])   # 趴姿，0.2916 m

HOME12 = HOME.reshape(12)

# 名目機身高度（m）。
# `_KIN` 是 HOME 的**純運動學**值；下面兩個是落地站定後量到的。
NOMINAL_HEIGHT_KIN = 0.4914
NOMINAL_HEIGHT = 0.4493      # `--stand`（輪 damp，= 步態實際用的模式）
NOMINAL_HEIGHT_WALK = 0.4825  # 走路時的平均機身高（walk 預設，20 s）
                              # ⚠️ 比靜態站立**高** 33 mm —— 走起來身體會被撐高。

# ★ 擺動相的抬腿補償量（m）。名字叫 SAG 是因為它源自位置伺服的靜態撓度：
#   kp 有限，要撐住 41 kg 就必須留追蹤誤差，所以**指令的足端基準比實際站立的足端低**。
#
#   ⚠️ 但它不是一個乾淨的物理量，是一個**調參槓桿**，三個數字都不一樣：
#       靜態站立（輪 hold）  撓度 32.5 mm   ← 這個常數的值，當初是這樣量到的
#       靜態站立（輪 damp）  撓度 42.1 mm   ← 現在 `--stand` 印的
#       **走路時**           有效只剩 ~10 mm ← 因為機身被撐高 33 mm（見上）
#     所以不要期待「補了 z_sag 之後實際離地就等於 g_c」。
#     實測 g_c=0.08 → 指令 112.5 mm → **實際離地 102.7 mm**（差 ~10 mm，就是上面那個）。
#
#   ⚠️ 沒有這個補償的後果很惡劣：實際離地 = g_c − 撓度，g_c 小於撓度時**腿根本不離地**，
#      而且超限／飽和／IK 縮限／相位鎖定**四個診斷指標全是乾淨的**，完全看不出問題。
#      實測未補償時整台倒退走（12 秒 −987 mm）。詳見 `cpg_max.foot_targets`。
#   ⚠️ 它與 KP3 綁在一起。改增益就必須重掃這個值（不是重跑 `--stand` 就好，見上）。
STATIC_SAG = 0.0325

FALL_GRAV_Z = -0.4   # 機身座標系重力 z 分量高於此值視為翻倒（約傾倒 66°），同 task6

# =============================================================================
# 控制
# =============================================================================
CTRL_DT = 0.02    # 50 Hz 出 CPG 指令
SIM_DT = 0.002    # MJCF 預設 timestep，剛好等於原廠 controller_dt（500 Hz）
                  # → PD 內迴圈跑在 500 Hz，與實機運控同頻率，不是我們挑的巧合值

# ★ 增益取原廠 RL 策略那組（`zg_wheels-user-parameters.yaml` 的 FSM_RL_*）。
#   ABAD 與 HIP/KNEE **不共用一個 Kp**（60 vs 120），這點與 task6 的 D1 EDU 不同。
KP3 = np.array([60.0, 120.0, 120.0])   # abad, hip, knee
KD3 = np.array([1.0, 1.0, 1.0])
KP_WHEEL, KD_WHEEL = 60.0, 0.5         # 原廠輪子是真的有位置增益的（D1 EDU 幾乎只做阻尼）

# 扭矩上限。MJCF 的 actuatorfrcrange 寫 150（腿）/ 40（輪），規格書寫 150 / 33。
# 輪取規格書的保守值；腿兩者一致。
TAU_MAX3 = np.array([150.0, 150.0, 150.0])
TAU_MAX_WHEEL = 33.0

# 關節速度上限（rad/s）。設定檔寫 24.0、規格書 190 RPM ≈ 19.9 —— 不一致，保守取 19.9。
# 只用於診斷（有沒有超速），不拿來 clip。
QVEL_MAX_LEG = 19.9
QVEL_MAX_WHEEL = 125.7   # 1200 RPM

# =============================================================================
# CPG（結構常數沿用 task4 / task6；步態相關的數值在 cpg_walk_max.py 裡掃）
# =============================================================================
MU_MIN, MU_MAX = 1.0, 2.0
A_CONV = 50.0
G_P = 0.01        # 站立相的下壓量
W_COUP = 8.0      # 相位耦合強度
N_CPG_SUB = 4     # 每個控制週期的 CPG 次步數

# =============================================================================
# 唯讀鎖
# =============================================================================
# 這些是 ndarray，預設可就地改寫（HOME[0, 1] = 99 會成功且不報錯），
# 任一處被汙染就會全域擴散。鎖成唯讀，改寫直接丟 ValueError。
for _const in (SIDE_Y, SIDE_X, STAND, HOME, CROUCH, HOME12,
               LEG_QPOS_IDX, LEG_QVEL_IDX, WHEEL_QPOS_IDX, WHEEL_QVEL_IDX,
               LEG_ACT_IDX, WHEEL_ACT_IDX, KP3, KD3, TAU_MAX3):
    _const.flags.writeable = False
del _const


# =============================================================================
# 模型與名稱查詢
# =============================================================================
def make_model() -> mujoco.MjModel:
    """載入平地場景（官方 zgws.xml + 我們寫的地板）。"""
    return mujoco.MjModel.from_xml_path(SCENE)


def _id(m: mujoco.MjModel, objtype, name: str) -> int:
    """名稱查 id，查不到當場擋下來。

    `mj_name2id` 找不到會回 −1，而 Python 的 −1 索引會**靜默拿到最後一個元素**
    （拿到 RL 的資料卻以為是 FR）。task6 踩過，這裡一律 assert。
    """
    i = mujoco.mj_name2id(m, objtype, name)
    assert i >= 0, f"名稱契約破裂：找不到 {objtype} `{name}`"
    return i


def foot_body_ids(m: mujoco.MjModel) -> list[int]:
    """四個輪 body 的 id，順序同 LEGS。body 原點 = 輪軸心，是 IK 與觸地判定的參考點。"""
    return [_id(m, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX[l]}_FOOT_LINK") for l in LEGS]


def abad_body_ids(m: mujoco.MjModel) -> list[int]:
    """四個 ABAD body 的 id，順序同 LEGS。body 原點 = ABAD 關節原點，是 IK 的基準。"""
    return [_id(m, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX[l]}_ABAD_LINK") for l in LEGS]


def leg_joint_ids(m: mujoco.MjModel) -> np.ndarray:
    """12 個腿關節的 joint id，順序 = LEGS × (abad, hip, knee)。"""
    return np.array([_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX[l]}_{j}_JOINT")
                     for l in LEGS for j in ("ABAD", "HIP", "KNEE")])


def leg_joint_ranges(m: mujoco.MjModel) -> np.ndarray:
    """(12, 2) 的關節限位，順序同 leg_joint_ids。

    ⚠️ 左右 ABAD 限位是鏡像的、前後 HIP 限位也是鏡像的，四腿**不能共用一組**。
       所以這裡從模型讀，不寫死。
    """
    return m.jnt_range[leg_joint_ids(m)].copy()
