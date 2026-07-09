"""折線任務幾何純函式測試（無 sim）。
run: conda run -n rbtdog python task4/inference/tests/test_course_geom.py"""
import sys
import numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
from odom_missions import ideal_waypoints, point_to_polyline, SEGS


def main():
    pts = ideal_waypoints(SEGS)
    # 航向序列 0,-45,45,-45,0 → 終點應回到初始線 y=0，且 x≈24.14
    end = pts[-1]
    assert abs(end[1]) < 1e-9, f"終點應回到 y=0，得 {end}"
    assert abs(end[0] - 24.1421) < 1e-3, f"終點 x 應≈24.14，得 {end[0]}"
    # 起點在原點、共 6 個航點
    assert pts.shape == (6, 2) and np.allclose(pts[0], [0, 0])
    # 第一段沿 +x 走 5m
    assert np.allclose(pts[1], [5.0, 0.0])
    # 點到折線距離：折線上的點距離為 0；偏離 1m 應得 1m
    assert point_to_polyline([5.0, 0.0], pts) < 1e-9
    assert abs(point_to_polyline([2.5, 1.0], pts) - 1.0) < 1e-9   # 第一段正上方 1m
    print("PASS test_course_geom")


if __name__ == "__main__":
    main()
