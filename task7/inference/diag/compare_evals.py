"""把多份 outputs/eval_*.md 的關鍵欄並排（給主管／交接用）。
    python3 task7/inference/diag/compare_evals.py eval_cpg_rl_v3_5f_params.md:v3.5f eval_cpg_rl_v3_6f_params_mirror.md:v3.6f+鏡像 ...
"""
import re, sys
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "outputs"
COLS = {"摔": 1, "vx": 2, "vy": 3, "偏航": 4, "roll": 5, "膝": 6, "抬腳": 9, "步/秒": 10, "ABAD漂": 11, "vx漂": 12, "航向": 13}
ROWS = ("原地左轉 1.3", "原地右轉 1.3", "左平移 0.20", "右平移 0.20", "左平移 0.12", "斜走 0.3+0.15", "直走 0.5")

def load(fn):
    T = {}
    for line in (OUT / fn).read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("| 指令") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) < 14: continue
            T[c[0]] = c
    return T

def main():
    files = [a.split(":") for a in sys.argv[1:]]
    tabs = {name: load(fn) for fn, name in files}
    for row in ROWS:
        print(f"\n### {row}")
        print("| 版本 | 摔 | 偏航 °/s | vy | 抬腳 mm | 步/秒 @8mm | roll std/峰 | 膝峰/RMS | ABAD 漂 | vx 漂 | 航向 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, T in tabs.items():
            c = T.get(row)
            if not c: print(f"| {name} | — |"); continue
            has_steps = len(c) >= 17
            g = lambda k: c[COLS[k] if has_steps or COLS[k] < 10 else COLS[k] - 1]  # 舊表沒有步/秒欄
            print(f"| {name} | {g('摔')} | {g('偏航')} | {g('vy')} | {g('抬腳')} | {g('步/秒') if has_steps else '—'} | {g('roll')} | {g('膝')} | {g('ABAD漂')} | {g('vx漂')} | {g('航向')} |")

if __name__ == "__main__":
    main()
