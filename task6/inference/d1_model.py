"""D1 EDU 輪足版（ZSL-1w）模型與常數的單一事實來源。

所有 task6 腳本與 Colab notebook 的常數都必須與本檔一致，否則訓練與推論會對不起來。
數值出處：
  - 幾何/質量/慣量/限位：zsibot/genisom_model 的 zsl-1w/urdf/ZSL-1W.urdf
  - kp/kd：AgibotTech/agibot_D1_Edu-Ultra 的 demo/zsl-1/python/examples/lowlevel_demo.py:68-73
  - CPG 常數：沿用 task4 論文標準版（cpg_rl_paper_colab.ipynb）
"""
from pathlib import Path

import mujoco
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "d1_edu_w"
SCENE = str(_MODEL_DIR / "scene.xml")
SCENE_MJX = str(_MODEL_DIR / "scene_mjx.xml")

# ---- 機構 ----
LEGS = ("FL", "FR", "RL", "RR")        # 見 task6/docs 對照表（官方文件腿序自相矛盾，尚未定案）
                                       # 用 tuple：腿序一旦被就地改寫，PHASE_OFFSET 的 trot 對角
                                       # 關係會靜默失效（見 test_cpg_d1 的相位釘住測試）
HOME3 = np.array([0.0, 1.05, -2.00])   # abad, hip, knee；knee 軸為 +y，故 hip 為正（與點足版相反）
HOME12 = np.tile(HOME3, 4)
WHEEL_RADIUS = 0.0710                  # 輪半徑(m)；輪子在模擬中熔接鎖死，當成圓腳
NOMINAL_HEIGHT = 0.2695  # 實際站定後的機身高度(m)。keyframe 的 0.2948 是純運動學值，
                         # 實機/模擬因 kp=80 位置伺服的靜態撓度會沉降約 2.5cm。
                         # 訓練的高度獎勵要用這個值，不是 key_qpos[2]。
                         # 實測（輪碰撞 geom 修正為 y=±0.0475 之後）：
                         # t=1s 0.26981 / 2s 0.27002 / 3s 0.26932 / 5s 0.26907

# ---- 控制 ----
CTRL_DT = 0.02      # 50 Hz，落在 SDK 建議的 20~50 Hz 內
SIM_DT = 0.004
KP, KD = 80.0, 1.0  # 原廠 demo 值（取自點足版 lowlevel_demo；輪足版無 LowLevel demo）不得擅改
TAU_MAX = 28.0      # URDF effort；官網 48 N·m 為峰值，取保守值

# ---- CPG（與 task4 論文標準版逐項相同）----
MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 4.5
A_CONV = 50.0
D_STEP = 0.12      # 前後步幅尺度
D_STEP_Y = 0.09    # 側向擺幅尺度：本機 abad 行程僅 ±28°(Go2 ±60°)，沿用 0.12 會超限 14%
G_C = 0.08
G_P = 0.01
W_COUP = 8.0
N_CPG_SUB = 4
PHASE_OFFSET = np.array([0.0, np.pi, np.pi, 0.0])   # trot：FL, FR, RL, RR

# ---- 觀測 ----
OBS_DIM = 69        # 76 −3(機身線速度，實機拿不到) −4(觸地布林，無可用訊號，見關卡3)
ACT_DIM = 12

# ---- 唯讀鎖 ----
# 這三個是 ndarray，預設可就地改寫（HOME3[0] = 99 會成功且不報錯），
# 任一處被汙染就會全域擴散到訓練與推論。鎖成唯讀，改寫會直接丟 ValueError。
for _const in (HOME3, HOME12, PHASE_OFFSET):
    _const.flags.writeable = False
del _const


def make_model(mjx: bool = False) -> mujoco.MjModel:
    """建立 D1 EDU 模型。mjx=True 回傳訓練用場景，False 回傳本機渲染場景。"""
    return mujoco.MjModel.from_xml_path(SCENE_MJX if mjx else SCENE)


def foot_geom_ids(m: mujoco.MjModel) -> list[int]:
    """四顆輪子碰撞 geom 的 id，順序同 LEGS。

    輪子鎖死當圓腳，功能上等同足端，故沿用 foot_* 命名供 IK 與觸地判定引用。
    名稱查不到時 mj_name2id 回傳 -1，而 Python 的 -1 索引會靜默拿到最後一個元素
    （拿到 RR 的資料卻以為是 FL），所以這裡必須當場擋下來。
    """
    ids = []
    for leg in LEGS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg)
        assert gid >= 0, f"名稱契約破裂：找不到 geom {leg}"
        ids.append(gid)
    return ids


def knee_actuator_ids(m: mujoco.MjModel) -> list[int]:
    """四個膝致動器的 id，順序同 LEGS。供診斷用（obs 不再使用力矩判觸地，見關卡 3）。

    同 foot_geom_ids：-1 會被 Python 索引靜默吃掉，必須 assert。
    """
    ids = []
    for leg in LEGS:
        name = f"{leg}_knee"
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"名稱契約破裂：找不到 actuator {name}"
        ids.append(aid)
    return ids
