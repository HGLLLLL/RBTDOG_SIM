"""local_infer_terrain2：舊模型(12維)零樣本跑 v2 地形不崩、能前進。
run: conda run -n rbtdog python task4/inference/tests/test_infer2.py"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import local_infer_terrain2 as L

OLD = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl"   # 12維 fixed


def main():
    infer, act_dim = L.load_policy_any(OLD)
    assert act_dim == 12, act_dim
    r = L.rollout(OLD, terrain="rough2", secs=4.0, video=False)
    assert r["mode"] == "fixed"
    assert r["fell"] is None or r["fell"] > 1.0, f"不應一開始就跌 {r}"
    # 平台起步應能往前一點（零樣本、地形難，門檻放寬）
    assert r["dist"] > 0.1, f"前進距離過小 {r['dist']}"
    print("PASS test_infer2", r)


if __name__ == "__main__":
    main()
