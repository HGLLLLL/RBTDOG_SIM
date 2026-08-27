"""`shm_io` 的 IMU 讀取 —— 欄位位移釘死。

為什麼要有這支：IMU 是 CPG-RL 觀測層前 6 維（重力向量 + 角速度）的唯一來源，
而那塊 SHM **只有 2026-08-25 一次離線快照解碼過**。位移讀錯不會報錯，
只會讓 policy 拿到亂數 —— 症狀是「狗一走就倒」，會被誤判成 RL 沒訓練好。
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))

import shm_io  # noqa: E402


class _FakeShm(shm_io.Shm):
    """不開真的 /dev/shm，用一塊 bytearray 假裝。"""

    def __init__(self, buf):
        self.mm = buf
        self.writable = False

    def close(self):
        pass


def test_imu_offsets_match_documented_layout():
    """acc[3] @824、gyro[3] @848、quat[4] @872（全部 f64）。"""
    buf = bytearray(shm_io.SIZE)
    vals = [1.0, 2.0, 3.0,          # acc
            0.11, 0.22, 0.33,       # gyro
            0.5, 0.5, 0.5, 0.5]     # quat (xyzw)
    for i, v in enumerate(vals):
        struct.pack_into("<d", buf, shm_io.IMU_BASE + i * 8, v)

    got = _FakeShm(buf).imu()
    assert got["acc"] == [1.0, 2.0, 3.0]
    assert got["gyro"] == [0.11, 0.22, 0.33]
    assert got["quat"] == [0.5, 0.5, 0.5, 0.5]
    # 文件寫的三個位移，直接釘住，改動必須是刻意的
    assert shm_io.IMU_BASE == 824
    assert shm_io.IMU_BASE + 3 * 8 == 848      # gyro
    assert shm_io.IMU_BASE + 6 * 8 == 872      # quat


def test_imu_reproduces_the_2026_08_25_snapshot():
    """用實機那次快照的數值：acc=(0.046, −0.121, 9.886)、|q|=1。

    這一組是「機身剛好水平」時量到的，也是目前 xyzw 判定的唯一證據 ——
    把它固定成測試，之後任何人重讀這塊 SHM 都有一個對照點。
    ⚠️ 但**它不能證明 xyzw 是對的**，證明要靠
       `docs/現場操作卡_IMU平放複核.md` 的 T2 / T3（刻意前傾與側傾）。
    """
    import math
    buf = bytearray(shm_io.SIZE)
    # 由 roll=−0.12°、pitch=−1.61° 反算的四元數（xyzw 排列）
    r, p, y = math.radians(-0.12), math.radians(-1.61), 0.0
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    z = cr * cp * sy - sr * sp * cy
    yq = cr * sp * cy + sr * cp * sy
    for i, v in enumerate([0.046, -0.121, 9.886, 0.0, 0.0, 0.0, x, yq, z, w]):
        struct.pack_into("<d", buf, shm_io.IMU_BASE + i * 8, v)

    got = _FakeShm(buf).imu()
    a = got["acc"]
    assert abs(math.sqrt(sum(v * v for v in a)) - 9.887) < 0.002
    q = got["quat"]
    assert abs(math.sqrt(sum(v * v for v in q)) - 1.0) < 1e-9
    # 用 xyzw 解回去要拿回原本的角度；用 wxyz 解會得到別的東西
    qx, qy, qz, qw = q
    roll = math.degrees(math.atan2(2 * (qw * qx + qy * qz),
                                   1 - 2 * (qx * qx + qy * qy)))
    assert abs(roll - (-0.12)) < 0.01
