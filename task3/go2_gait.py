"""Go2 CPG trot 控制器（含轉彎）。指令介面: v_cmd(前進 0~1), turn_cmd(-1~1, 右轉為正)。
力矩致動器 → 軟體 PD；站立相貼地後推、擺動相抬起前移；轉彎用左右腿差動步幅。
航向來源 compass_yaw() 模擬 Xsens MTi-680G 的『融合航向』輸出（見下方誤差模型）。"""
import numpy as np, mujoco
from go2_model import make_model

LEGS = ["FL", "FR", "RL", "RR"]
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}   # trot 對角同相
SIDE = {"FL": +1.0, "RL": +1.0, "FR": -1.0, "RR": -1.0}    # 左腿+1 右腿-1
HOME = np.array([0.0, 0.9, -1.8])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class Go2Gait:
    def __init__(self, freq=3.0, duty=0.55, lift=0.12, stride=0.22,
                 turn_gain=0.10, z0=-0.28, kp=70.0, kd=1.5,
                 imu_heading_rms_deg=0.5, imu_heading_tau=20.0, imu_seed=0,
                 odom_xy_bias=(0.0, 0.0), odom_yaw_bias=0.0):
        # imu_*：Xsens MTi-680G 融合航向誤差模型參數。
        #   規格書 heading 精度 0.5° RMS；拆成「慢速零偏(Gauss-Markov, 時間常數 tau)」
        #   + 「白噪」兩部分合成，零偏占約 0.9、白噪補足其餘，使總 RMS≈規格值。
        self.m = make_model()
        self.d = mujoco.MjData(self.m)
        self.freq, self.duty, self.lift = freq, duty, lift
        self.stride, self.turn_gain, self.z0 = stride, turn_gain, z0
        self.kp, self.kd = kp, kd
        self.imu_heading_rms_deg = imu_heading_rms_deg
        self.imu_heading_tau = imu_heading_tau
        self.imu_seed = imu_seed
        self._odom_xy_bias = np.asarray(odom_xy_bias, dtype=float)
        self._odom_yaw_bias = float(odom_yaw_bias)
        self.flimit = self.m.actuator_ctrlrange[:, 1]
        self.f0, self.Jinv = self._calc_ik()
        self.center = np.array([self.f0[0], z0])
        self._sadr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, i):
                      self.m.sensor_adr[i] for i in range(self.m.nsensor)}
        self.reset()

    def _init_imu(self):
        """初始化 MTi-680G 融合航向誤差狀態（可重現，依 imu_seed）。"""
        self._rng = np.random.default_rng(self.imu_seed)
        total = np.radians(self.imu_heading_rms_deg)
        self._head_bias_sd = 0.90 * total
        self._head_white_sd = np.sqrt(max(total ** 2 - self._head_bias_sd ** 2, 0.0))
        self._he_time = float(self.d.time)
        self._he_bias = self._head_bias_sd * self._rng.standard_normal()
        self._he_white = self._head_white_sd * self._rng.standard_normal()

    # ---------- 幾何 / IK ----------
    def _foot_xz(self, th, ca):
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[8] = th; d.qpos[9] = ca
        mujoco.mj_forward(m, d)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
        piv = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FL_thigh")]
        p = d.geom_xpos[gid] - piv
        return np.array([p[0], p[2]])

    def _calc_ik(self):
        f0 = self._foot_xz(0.9, -1.8); e = 1e-3
        J = np.zeros((2, 2))
        J[:, 0] = (self._foot_xz(0.9 + e, -1.8) - self._foot_xz(0.9 - e, -1.8)) / (2 * e)
        J[:, 1] = (self._foot_xz(0.9, -1.8 + e) - self._foot_xz(0.9, -1.8 - e)) / (2 * e)
        return f0, np.linalg.inv(J)

    def _foot_traj(self, phase, stride_leg):
        ph = phase % 1.0
        if ph < self.duty:                      # 站立相：貼地由前往後
            s = ph / self.duty
            x = stride_leg * (0.5 - s); z = 0.0
        else:                                   # 擺動相：抬起由後往前
            s = (ph - self.duty) / (1 - self.duty)
            x = stride_leg * (-0.5 + s); z = self.lift * np.sin(np.pi * s)
        return x, z

    # ---------- 控制 ----------
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self._init_imu()

    def targets(self, t, v_cmd, turn_cmd):
        q = np.zeros(12)
        for k, leg in enumerate(LEGS):
            stride_leg = self.stride * v_cmd + SIDE[leg] * self.turn_gain * turn_cmd
            ph = self.freq * t + LEG_PHASE[leg]
            dx, dz = self._foot_traj(ph, stride_leg)
            dth, dca = self.Jinv @ (self.center + np.array([dx, dz]) - self.f0)
            q[3 * k:3 * k + 3] = [0.0, HOME[1] + dth, HOME[2] + dca]
        return q

    def step(self, t, v_cmd, turn_cmd):
        qdes = self.targets(t, v_cmd, turn_cmd)
        tau = self.kp * (qdes - self.d.qpos[7:19]) - self.kd * self.d.qvel[6:18]
        self.d.ctrl[:] = np.clip(tau, -self.flimit, self.flimit)
        mujoco.mj_step(self.m, self.d)

    # ---------- 感測 ----------
    def sensor(self, name):
        a = self._sadr[name]
        dim = 3 if name != "imu_quat" else 4
        return self.d.sensordata[a:a + dim].copy()

    def compass_yaw(self):
        """模擬 Xsens MTi-680G 的『融合航向』輸出 (rad)。
        真實 Go2 上，MTi-680G 內部已融合陀螺/加速度/磁力計(+GNSS)才吐航向，
        故不直接用裸磁力計，而是在真實航向上疊裝置級誤差：慢速零偏 + 白噪，
        合成 RMS≈規格書的 0.5°。每個模擬時刻只更新一次量測（多次讀取回同值）。"""
        tnow = float(self.d.time)
        if tnow != self._he_time:
            dt = tnow - self._he_time
            self._he_time = tnow
            if dt > 0:                                   # 零偏走 Gauss-Markov（一階馬可夫）
                a = np.exp(-dt / self.imu_heading_tau)
                self._he_bias = (a * self._he_bias +
                                 np.sqrt(1.0 - a * a) * self._head_bias_sd
                                 * self._rng.standard_normal())
            self._he_white = self._head_white_sd * self._rng.standard_normal()
        return wrap(self.true_yaw() + self._he_bias + self._he_white)

    def mag_yaw(self):
        """（保留）由裸磁力計直接解算航向：yaw = atan2(-mag_x, -mag_y)。"""
        mx, my, mz = self.sensor("imu_mag")
        return np.arctan2(-mx, -my)

    def true_yaw(self):
        """真實 yaw（由 framequat，僅供對照/驗證）。"""
        w, x, y, z = self.sensor("imu_quat")
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def odom(self):
        """完美里程計（取代羅盤）：回傳世界系 (x, y, yaw)。
        位置取自 imu site 的 framepos，航向由 framequat 解算；bias 預設 0，可注入偏移做實驗。"""
        x, y, _ = self.sensor("odom_pos")
        w, xx, yy, zz = self.sensor("imu_quat")
        yaw = np.arctan2(2 * (w * zz + xx * yy), 1 - 2 * (yy * yy + zz * zz))
        bx, by = self._odom_xy_bias
        return float(x + bx), float(y + by), wrap(yaw + self._odom_yaw_bias)

    @property
    def xy(self):
        return self.d.qpos[0:2].copy()

    @property
    def height(self):
        return float(self.d.qpos[2])
