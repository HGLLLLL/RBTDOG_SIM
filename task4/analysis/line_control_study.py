"""odom 校正走直線 —— 核心算法研讀版（從原始碼複製，逐行加註，原檔未動）。

來源：
  1. line_frame / line_control  ← task4/inference/local_infer_paper.py:100-117
  2. Go2Gait.odom()             ← task3/go2_gait.py:135-142（此處以註解說明，不依賴 MuJoCo）
  3. run_line 的呼叫方式         ← task4/inference/odom_missions.py:169-196

本檔可獨立執行（只需 numpy），用一台簡化的「單車模型」小車代替機器狗，
驗證控制律確實能把一台有初始偏移+初始歪頭的車拉回目標直線上：

  python line_control_study.py
"""
import numpy as np


# =====================================================================
# (0) 工具函式 —— 複製自 local_infer_paper.py:97
# =====================================================================
def wrap(a):
    """把任意角度正規化到 (-pi, pi]。
    原理：sin/cos 對 2*pi 週期不敏感，atan2(sin(a), cos(a)) 會吐回等價的主值角。
    為什麼必要：航向誤差 yaw - psi 直接相減可能得到 350°，
    但實際上只差 -10°；不 wrap 的話控制器會朝「遠路」轉一大圈。"""
    return np.arctan2(np.sin(a), np.cos(a))


# =====================================================================
# (1) 目標線的座標架 —— 複製自 local_infer_paper.py:100-104
# =====================================================================
def line_frame(psi):
    """目標線的方向 d 與左法向 n（世界系單位向量）。

    psi = 目標線在世界系的航向角（rad）。走直線實驗中 psi=0（沿 +x 走）。
    d = (cos psi, sin psi)   → 沿著線「往前」的單位向量
    n = (-sin psi, cos psi)  → 把 d 逆時針轉 90° 得到的「左側」法向量
    之後 e_ct = n · (p - p0)：位置差在 n 上的投影，
    正值 = 車在線的左邊，負值 = 在右邊。"""
    d = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-np.sin(psi), np.cos(psi)])
    return d, n


# =====================================================================
# (2) 核心控制律 —— 複製自 local_infer_paper.py:107-117
# =====================================================================
def line_control(p, yaw, p0, psi_target, vx, k_yaw, k_ct, no_lateral=False):
    """方案 A 解耦控制：wz 用航向誤差鎖航向、vy 用 cross-track 誤差滑回線上。
    p, p0 為世界系 (x,y)；回傳 (cmd[vx,vy,wz] float32, e_ct, e_yaw)。

    參數：
      p          = odom 量到的目前位置 (x, y)，世界系
      yaw        = odom 量到的目前航向（rad），世界系
      p0         = 目標線上的一個定錨點（直線實驗 = 起步時 latch 的 odom 位置）
      psi_target = 目標線航向（直線實驗 = 0，course 實驗 = 各段的絕對航向）
      vx         = 前進速度指令（常數 0.6 m/s）
      k_yaw      = 航向 P 增益（3.0）
      k_ct       = 橫向 P 增益（1.5）
      no_lateral = True 時關閉 vy 修正（做消融實驗用，只剩鎖航向）
    """
    _, n = line_frame(psi_target)          # 只需要法向量 n；方向向量 d 用 _ 丟棄

    # --- 橫向誤差（cross-track error）---
    # (p - p0) 是「從線上的錨點指向車」的向量；點積 n 取它在左法向上的分量。
    # 幾何意義：車偏離目標線的『垂直距離』（帶正負號：左正右負）。
    # 注意錨點 p0 沿線移動不影響 e_ct（沿線分量被 n 投影消掉），所以 p0 只要
    # 是線上任一點都行 —— 這就是起步時 latch 一次即可、之後不必更新的原因。
    e_ct = float(n @ (np.asarray(p, float) - np.asarray(p0, float)))

    # --- 航向誤差 ---
    # wrap 保證取最短角差；正值 = 頭偏向左。
    e_yaw = float(wrap(yaw - psi_target))

    # --- 兩個解耦的 P 控制器 ---
    # wz（轉向角速度）：誤差偏左(+) → 給負 wz 往右轉回來。clip 到 ±1.0 rad/s
    # 避免大誤差時猛甩（RL 策略的訓練分佈也只到這個範圍）。
    wz = float(np.clip(-k_yaw * e_yaw, -1.0, 1.0))

    # vy（body 系橫移速度）：偏左(+e_ct) → 給負 vy 往右橫移。clip 到 ±0.3 m/s
    # （四足橫移比前進慢得多，太大步態會亂）。
    # ★ 隱含假設：e_ct 是「世界系」誤差，vy 卻是「機身系」指令 ——
    #   只有在 yaw ≈ psi_target（頭已對準線）時兩者方向才一致。
    #   這個前提由上面的 wz 迴路主動維持：wz 收斂快（k_yaw 大、誤差角通常小），
    #   所以絕大部分時間 vy 的修正方向是對的；大航向誤差時會短暫失準但無害。
    vy = 0.0 if no_lateral else float(np.clip(-k_ct * e_ct, -0.3, 0.3))

    # 回傳給 RL 策略的三維速度指令 [前進, 橫移, 轉向]，外加兩個誤差供記錄。
    return np.array([vx, vy, wz], np.float32), e_ct, e_yaw


# =====================================================================
# (3) odom 量測來源 —— 原始碼摘錄（go2_gait.py:135-142），依賴 MuJoCo 故不執行
# =====================================================================
# def odom(self):
#     """完美里程計（取代羅盤）：回傳世界系 (x, y, yaw)。"""
#     x, y, _ = self.sensor("odom_pos")        # imu site 的 framepos → 真值位置
#     w, xx, yy, zz = self.sensor("imu_quat")  # framequat → 真值姿態四元數
#     yaw = np.arctan2(2*(w*zz + xx*yy), 1 - 2*(yy*yy + zz*zz))  # 四元數→yaw
#     bx, by = self._odom_xy_bias              # 預設 0；可注入偏差做退化實驗
#     return float(x+bx), float(y+by), wrap(yaw + self._odom_yaw_bias)
#
# 重點：這是「作弊級」完美 odom（直接讀模擬真值），沒有累積漂移。
# 對照組 compass_yaw() 則是真航向 + Gauss-Markov 慢零偏 + 白噪（RMS≈0.5°），
# 且完全沒有位置資訊 —— 這正是兩組實驗的差異來源。


# =====================================================================
# (4) 走直線實驗的呼叫方式 —— 原始碼摘錄（odom_missions.py:173,187-188）
# =====================================================================
# warmup 之後：
#   p0 = r.odom_xy()          # ← latch：把「起步瞬間的 odom 位置」當作目標線錨點
# 每個控制週期（0.02s）：
#   cmd, _, _ = P.line_control(r.odom_xy(),   # 目前 odom 位置
#                              r.yaw_meas(),  # odom 航向（odom 模式 = 真值）
#                              p0[nm],        # 起步 latch 的錨點
#                              0.0,           # psi_target=0：沿世界 +x 直走
#                              VX, K_YAW, K_CT)   # 0.6, 3.0, 1.5
#   r.drive(cmd)              # cmd 餵給 RL 策略 → CPG → 關節目標 → 力矩
#
# 對照組 compass 只有：
#   yaw_rate = clip(-K_YAW * wrap(yaw_meas - 0), -1, 1); cmd = [VX, 0, yaw_rate]
# 即「只鎖航向、無橫向修正」——一旦被噪聲推離線，就再也回不來，只會平行漂移。


# =====================================================================
# (5) 獨立驗證：用簡化運動學模型跑一遍控制律（本檔新增，非複製）
# =====================================================================
def simulate(y0=0.8, yaw0_deg=25.0, secs=15.0, dt=0.02,
             vx=0.6, k_yaw=3.0, k_ct=1.5, no_lateral=False):
    """單車運動學：p' = R(yaw) @ [vx, vy]，yaw' = wz。
    機器狗的 RL 策略近似把 cmd 忠實執行，所以這個模型足以看控制律的收斂行為。
    起始條件故意給很差：偏離線 0.8 m、歪頭 25°。"""
    p = np.array([0.0, y0]); yaw = np.radians(yaw0_deg)
    p0 = np.array([0.0, 0.0])                 # 目標線 y=0 上的錨點
    traj = []
    for _ in range(int(secs / dt)):
        cmd, e_ct, e_yaw = line_control(p, yaw, p0, 0.0, vx, k_yaw, k_ct, no_lateral)
        vx_c, vy_c, wz_c = cmd
        c, s = np.cos(yaw), np.sin(yaw)       # body 系速度轉世界系
        p = p + dt * np.array([c * vx_c - s * vy_c, s * vx_c + c * vy_c])
        yaw = wrap(yaw + dt * wz_c)
        traj.append([p[0], p[1], np.degrees(yaw), e_ct, np.degrees(e_yaw)])
    return np.array(traj)


if __name__ == "__main__":
    for tag, nolat in [("完整(vy+wz)", False), ("消融(只鎖航向)", True)]:
        t = simulate(no_lateral=nolat)
        # 收斂判定：|y| 最後一次超過 2cm 的時間點
        bad = np.where(np.abs(t[:, 1]) > 0.02)[0]
        t_conv = (bad[-1] + 1) * 0.02 if len(bad) else 0.0
        print(f"{tag:14s} 末端 y={t[-1,1]:+.4f} m  yaw={t[-1,2]:+.2f}°  "
              f"|y|收斂到2cm內耗時 {t_conv:.1f}s")
    print("\n完整版逐秒軌跡（x, y, yaw°, e_ct, e_yaw°）：")
    t = simulate()
    for i in range(0, len(t), 50):            # 每 1 秒印一列
        print(f"  t={i*0.02:4.1f}s  x={t[i,0]:5.2f}  y={t[i,1]:+.3f}  "
              f"yaw={t[i,2]:+6.2f}°  e_ct={t[i,3]:+.3f}  e_yaw={t[i,4]:+6.2f}°")
