"""terrain3b vs terrain2_1 並排對照（同今早 terrain_compare_terrain3 的課程/odom/鏡頭）。

只把今早的 terrain3 換成修正版 terrain3b（slip懲罰0.1 + 爬坡獎勵重訓）。
重用 terrain_compare_terrain3.render_run，不改它。terrain3b 推論仍配 cpg3（GC_MAX=0.25，
與訓練一致；只有 env reward 改過、不影響推論映射）。

用法：
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_terrain3b.py --check
  MUJOCO_GL=egl conda run -n rbtdog python task4/analysis/terrain_compare_terrain3b.py --exp both
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/analysis")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import terrain_compare as TC
import terrain_compare_terrain3 as T3
import cpg2, cpg3

OUT = "/home/huang/rbtdog_sim/task4/outputs/2026-07-21-terrain3b-vs-terrain2_1"
W_NEW = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain3b_params.pkl"
W_OLD = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_terrain2_1_params.pkl"


def exp1(secs=45.0):
    print("=== 實驗1 爬坡：terrain3b(新) ‖ terrain2_1(舊)，皆 odom ===")
    fl, sl = T3.render_run(W_NEW, cpg3, TC.build_course_slopes, "side", secs, "terrain3b (new)")
    fr, sr = T3.render_run(W_OLD, cpg2, TC.build_course_slopes, "side", secs, "terrain2_1 (old)")
    T3._combine(fl, fr, f"{OUT}/exp1_slope_compare.mp4"); return sl, sr


def exp2(secs=35.0):
    print("=== 實驗2 凹凸：terrain3b(新) ‖ terrain2_1(舊)，皆 odom ===")
    fl, sl = T3.render_run(W_NEW, cpg3, TC.build_course_rough, "rear45", secs, "terrain3b (new)")
    fr, sr = T3.render_run(W_OLD, cpg2, TC.build_course_rough, "rear45", secs, "terrain2_1 (old)")
    T3._combine(fl, fr, f"{OUT}/exp2_rough_compare.mp4"); return sl, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["1", "2", "both"], default="both")
    ap.add_argument("--check", action="store_true", help="3s smoke，不出影片")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check:
        print("=== 3s smoke ===")
        T3.render_run(W_NEW, cpg3, TC.build_course_slopes, "side", 3.0, "terrain3b slope")
        T3.render_run(W_OLD, cpg2, TC.build_course_slopes, "side", 3.0, "terrain2_1 slope")
        print("CHECK DONE"); return
    if args.exp in ("1", "both"): exp1()
    if args.exp in ("2", "both"): exp2()


if __name__ == "__main__":
    main()
