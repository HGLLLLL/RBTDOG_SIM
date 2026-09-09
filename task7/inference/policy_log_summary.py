"""M9 `--policy` 趟的 log 摘要：obs 各欄範圍（對照 replay_policy_trip17.md §2）、退回次數、
推論耗時、動作／sway／ω 分布、與 M9 既有的力矩峰值。上機每趟結束拉回本機就跑這個。

    conda run -n rbtdog python task7/inference/policy_log_summary.py task7/logs/m_logs_trip19/M9_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import policy_np as pn  # noqa: E402

COLS = [("gravity", 0, 3), ("gyro", 3, 6), ("joint_pos", 6, 18), ("joint_vel", 18, 30),
        ("cpg", 42, 66)]


def summarize(path: str) -> str:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    P = d.get("policy")
    L = [f"# {Path(path).name}", f"中止：{d.get('abort_reason') or '無'}", ""]
    if not P:
        return "\n".join(L + ["（這趟沒有 --policy）"])
    steps = P["log"]
    L += [f"policy {Path(P['path']).name}  preset {P['preset']}  vx {P['vx']} wz {P['wz']}  gain {P['gain']}",
          f"步數 {P['steps']}　退回 {P['fallback_total']}　最慢推論 {P['worst_ms']} ms"
          f"{'　★ 退回開迴路：' + P['open_loop_why'] if P['open_loop'] else ''}", ""]
    fbs = [s["fb"] for s in steps if s["fb"]]
    if fbs:
        from collections import Counter
        L += ["退回原因：" + "、".join(f"{k}×{v}" for k, v in Counter(fbs).most_common()), ""]
    O = np.array([s["obs"] for s in steps if s["obs"] is not None], dtype=float)
    if O.size:
        L += ["## obs 範圍（對照 replay_policy_trip17.md §2）", "",
              "| 欄 | min | max | mean | std |", "|---|---|---|---|---|"]
        for nm, a, b in COLS:
            seg = O[:, a:b]
            L.append(f"| {nm} | {seg.min():+.3f} | {seg.max():+.3f} | {seg.mean():+.3f} | {seg.std():.3f} |")
        g = O[:, 0:3]
        L += ["", f"gravity 平均 ({g[:, 0].mean():+.3f}, {g[:, 1].mean():+.3f}, {g[:, 2].mean():+.3f})"
              f"　gyro_z 範圍 {O[:, 5].min():+.2f}~{O[:, 5].max():+.2f} rad/s（★ 第一次有走路時的原始 gyro_z）", ""]
    A = np.array([s["a"] for s in steps], dtype=float)
    ACT = np.array([s["act"] for s in steps], dtype=float)
    SW = np.array([s["sway"] for s in steps], dtype=float) * 1000
    om = np.array([pn.act_to_cmd(a, "nomux")[2] for a in ACT])
    ms = np.array([s["ms"] for s in steps], dtype=float)
    L += ["## 動作", "",
          f"|a| 中位 {np.median(np.abs(A)):.2f}　最大 {np.abs(A).max():.2f}（貼 1 = 飽和）",
          f"實際 sway |x| 平均 {np.abs(SW[:, 0]).mean():.0f} mm（最大 {np.abs(SW[:, 0]).max():.0f}）"
          f"　|y| 平均 {np.abs(SW[:, 1]).mean():.0f}（最大 {np.abs(SW[:, 1]).max():.0f}）",
          f"ω 範圍 {om.min():.2f}–{om.max():.2f}（基準 1.4；模擬驗收 0.51–1.63）",
          f"推論 ms：中位 {np.median(ms):.2f}　p99 {np.percentile(ms, 99):.2f}　最大 {ms.max():.2f}", ""]
    pk = d.get("peak", {})
    if pk:
        top = sorted(pk.items(), key=lambda kv: -abs(kv[1]))[:4]
        L += ["## 力矩峰值（M9）", "", "、".join(f"{j} {v:.1f}" for j, v in top),
              "（門檻 KNEE 70；trip17 開迴路 A 膝 59.5）", ""]
    return "\n".join(L)


def main() -> int:
    for p in sys.argv[1:]:
        print(summarize(p))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
