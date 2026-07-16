"""terrain2 幾何/gz 測試。run: conda run -n rbtdog python task4/inference/tests/test_terrain2.py"""
import sys, numpy as np, mujoco
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import terrain2 as T


def main():
    # 平台平滑：|x|<1 高度=0、振幅=0
    assert abs(T.slope_z(0.0)) < 1e-9 and abs(T.slope_z(0.5)) < 1e-9
    assert T.amp_at(0.0) == 0.0 and T.amp_at(0.5) == 0.0
    # 粗糙度漸變：中段~一半、遠端=AMP_MAX(0.08)
    assert abs(T.amp_at(2.0) - 0.04) < 1e-6, T.amp_at(2.0)
    assert abs(T.amp_at(3.0) - 0.08) < 1e-6 and abs(T.amp_at(6.0) - 0.08) < 1e-6
    # 斜坡最陡 15°：相鄰折點最大斜率 tan(15°)
    dz = np.diff(T.KNOTS_Z); dx = np.diff(T.KNOTS_X)
    assert abs(np.max(np.abs(dz/dx)) - np.tan(np.radians(15.0))) < 1e-6
    # gz 在平台≈0
    assert abs(float(T.gz_np(0.0, 0.0))) < 1e-6
    # ★ 幾何 == gz：mj_ray 打表面對照雙線性 gz
    m = T.build_terrain2_model("mujoco_menagerie/unitree_go2/scene_mjx.xml")
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    bad = 0
    # 只打地形（group 0 = floor hfield + safety_net）；排除機器人身體(group 2/3)避免
    # 在原點被機器狗擋住。geomgroup mask index=group。
    geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    for x in np.arange(-5.5, 5.6, 0.5):
        for y in [-2.0, 0.0, 2.0]:
            gid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(m, d, np.array([x, y, 5.0]), np.array([0, 0, -1.0]),
                                 geomgroup, 1, -1, gid)
            surf = 5.0 - dist
            if abs(surf - float(T.gz_np(x, y))) > 0.02:
                bad += 1
    assert bad == 0, f"幾何/gz 不一致點數={bad}"
    print("PASS test_terrain2")


if __name__ == "__main__":
    main()
