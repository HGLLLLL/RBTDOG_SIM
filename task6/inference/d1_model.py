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
WHEEL_RADIUS = 0.0710                  # 輪半徑(m)

# ---- 關節在 qpos / qvel 裡的位址 ----
# 輪子有自己的鉸鏈（2026-08-11 起不再熔死），所以 12 個腿關節在 qpos 裡**不連續**：
#   qpos: 0-6 基座 | 7,8,9 FL abad/hip/knee | 10 FL 輪 | 11,12,13 FR | 14 FR 輪 | ...
# 「腿關節 = qpos[7:19]」這個舊假設會靜默把輪角當成關節角讀進去（IK 會變奇異矩陣，
# obs 會餵錯 12 個數字而不報錯），所以改用明列位址，並由 test_model 對名稱查詢驗證。
LEG_QPOS_IDX = np.array([7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21])
LEG_QVEL_IDX = np.array([6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])
WHEEL_QPOS_IDX = np.array([10, 14, 18, 22])
WHEEL_QVEL_IDX = np.array([9, 13, 17, 21])
NOMINAL_HEIGHT = 0.2695  # 實際站定後的機身高度(m)。keyframe 的 0.2948 是純運動學值，
                         # 實機/模擬因 kp=80 位置伺服的靜態撓度會沉降約 2.5cm。
                         # 訓練的高度獎勵要用這個值，不是 key_qpos[2]。
                         # 實測（輪碰撞 geom 修正為 y=±0.0475 之後）：
                         # t=1s 0.26981 / 2s 0.27002 / 3s 0.26932 / 5s 0.26907
FALL_GRAV_Z = -0.4       # 機身座標系中重力 z 分量高於此值視為翻倒（約傾倒 66°）。
                         # 訓練的 done 條件與推論的跌倒判定必須用同一個值，否則
                         # 「訓練認為還沒倒」的姿態在推論會被記成跌倒（反之亦然）。

# ---- 控制 ----
CTRL_DT = 0.02      # 50 Hz，落在 SDK 建議的 20~50 Hz 內
SIM_DT = 0.004
KP, KD = 80.0, 1.0  # 取自點足版 lowlevel_demo:68-73（輪足版官方沒有 LowLevel demo）。
                    # 2026-08-11 實機從 /spline_shm 指令區量到原廠**站立**用的是
                    # 腿 kp=20/kd=0.7、輪 kp=0/kd=0.1，比這裡軟 4 倍；但原廠是把 p_des
                    # 當力矩合成把手（站立時刻意留 22° 追蹤誤差、t_ff=0），與本專案
                    # 「p_des 是真實目標」的架構不同，不能直接搬。走路時的原廠增益尚未量到。
                    # 決策：走路沿用點足版的 80/1（後腳實測能正常抬腿）。不得擅改。
TAU_MAX = 28.0      # URDF effort；官網 48 N·m 為峰值，取保守值

# ---- CPG（與 task4 論文標準版逐項相同）----
MU_MIN, MU_MAX = 1.0, 2.0
OMEGA_MIN, OMEGA_MAX = 0.0, 2.5   # 4.5 → 2.5（2026-08-11）：kp=80 時 ω≈2.4–3.4 是倒退走帶
                                  # （抬不夠的腿在擺動相被往前拖，把機身往後推），舊上限讓
                                  # 動作空間有一大半落在裡面。訓好的 policy 常用 1.3–1.75。
                                  # ★ 與 notebook 的 OMEGA_MAX 是兩份，改一邊必須改另一邊。
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

    輪子有自己的被動鉸鏈，但輪心仍是 IK 與觸地判定的參考點，故沿用 foot_* 命名。
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
