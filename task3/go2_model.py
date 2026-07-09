"""建立含 9 軸 IMU 的 Go2 模型（用 MjSpec 以正確解析網格路徑）。"""
import mujoco

SCENE = "/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene.xml"


def make_model():
    spec = mujoco.MjSpec.from_file(SCENE)
    # 在機身內建的 imu site 上加感測器
    def add(name, stype):
        s = spec.add_sensor()
        s.name = name
        s.type = stype
        s.objtype = mujoco.mjtObj.mjOBJ_SITE
        s.objname = "imu"
        return s
    add("imu_acc",  mujoco.mjtSensor.mjSENS_ACCELEROMETER)
    add("imu_gyro", mujoco.mjtSensor.mjSENS_GYRO)
    add("odom_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS)     # 完美里程計位置（取代羅盤）
    add("imu_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT)   # 真實姿態，僅供對照
    return spec.compile()


if __name__ == "__main__":
    m = make_model()
    print("model 建立成功, nsensor =", m.nsensor, " magnetic =", m.opt.magnetic)
    for i in range(m.nsensor):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
        print(f"  {nm:10s} adr={m.sensor_adr[i]} dim={m.sensor_dim[i]}")
