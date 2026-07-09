"""odom 感測器與 Go2Gait.odom() 單元測試（無 pytest，直接 assert）。
run: conda run -n rbtdog python task3/tests/test_odom.py"""
import sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
from go2_gait import Go2Gait, wrap
from walk_line import GAIT


def main():
    g = Go2Gait(**GAIT); g.reset()
    assert "odom_pos" in g._sadr, "odom_pos 感測器不存在"
    assert "imu_mag" not in g._sadr, "magnetometer 應已移除"

    x, y, yaw = g.odom()
    # 航向與 true_yaw（同一顆四元數）應完全一致
    assert abs(wrap(yaw - g.true_yaw())) < 1e-9, (yaw, g.true_yaw())
    # 零偏移下位置貼近 base（imu site 僅約 2.6cm 偏移）
    assert abs(x - g.d.qpos[0]) < 0.1 and abs(y - g.d.qpos[1]) < 0.1, (x, y, g.d.qpos[:2])

    # 偏移注入：差值等於 bias
    g2 = Go2Gait(**GAIT, odom_xy_bias=(1.0, -2.0), odom_yaw_bias=0.1); g2.reset()
    x2, y2, yaw2 = g2.odom()
    assert abs((x2 - x) - 1.0) < 1e-6 and abs((y2 - y) + 2.0) < 1e-6, (x2, y2)
    assert abs(wrap(yaw2 - (yaw + 0.1))) < 1e-6, (yaw2, yaw)
    print("PASS test_odom")


if __name__ == "__main__":
    main()
