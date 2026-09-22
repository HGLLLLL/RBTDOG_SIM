#!/usr/bin/env python3
"""recon5_graph —— 把 `ros2 node info` 的原始輸出重建成 node／topic 圖。

為什麼不用 rqt_graph：**狗上沒有 GUI**，`rqt_graph` 起不來，而且它也不會輸出檔案。
`recon5_nav_ai.sh` 把每個節點的 `ros2 node info` 全撈回本機，這支在本機重建同一張圖。

產出（都寫進輸出目錄）：
  nodes_topics.md   完整表格：節點→發布／訂閱、topic→發布者／訂閱者（報告直接用這份）
  graph.dot         完整圖（Graphviz）→ `dot -Tsvg graph.dot -o graph.svg`
  graph_core.dot    主幹圖（只留有發也有收、非視覺化的 topic；同一對節點的多條 topic 併成一條邊）
  graph_core_nx.dot / graph_core_rk.dot   同上，每塊板一張（報告用這兩張）
  graph_core.mmd    Mermaid（只留感測→SLAM／定位→導航→cmd_vel 的主幹，看得懂的那張）
  graph.json        機器可讀

用法：  python3 recon5_graph.py <recon5 輸出目錄>
"""
from __future__ import annotations

import json
import os
import re
import sys

# 這些 topic 每個節點都接，畫進圖只會變成毛球 → 主幹圖排除（完整表格照樣保留）
NOISE = re.compile(r"^/(parameter_events|rosout)$|/transition_event$|"
                   r"^/tf(_static)?$|^/clock$|^/diagnostics|"
                   # nav2 生命週期的 bond 心跳：每個 lifecycle 節點都跟管理者連一條，
                   # 不濾掉整張圖會被它淹成毛球（2026-09-22 第一版就是這樣）
                   r"/bond$")
VIS = re.compile(r"/vis/|visualization|/debug|/dbg/|marker", re.I)

BOARD_FILES = {"nx": "nav_ai_nx.log", "rk": "nav_ai_rk.log"}
BOARD_NAME = {"nx": "Orin NX（應用板）", "rk": "RK3588（運控板）"}

# 主幹圖的分組（依名稱歸類，只影響畫圖的顏色與階層）
# ⚠️ 順序就是優先序（第一個命中的贏）。運控／HAL 一定要排在導航前面，
#    否則 `battery_controller`、`joint_shm_controller` 這些會被「controller」誤抓進導航
#    （2026-09-22 第一版畫出來就是這樣錯的）。導航那條也不能用裸的 `controller`。
GROUPS = [
    ("感測驅動", r"rslidar|imu_driver|uss_driver|uwb_driver|gps_driver|sixents|robot_camera|"
                r"livox|camera_driver"),
    ("SLAM／定位", r"slam|lvio|mapping|localization|aritag|apriltag|robot_tf|^/MEB$"),
    ("感知", r"perception|stereo|detect|track|segment"),
    ("運控／HAL", r"robot_hal|joint_shm|imu_shm|mc_ctrl|switch_controller|estop|battery|"
                  r"led_controller|fill_light|robot_remote|robot_monitor|robot_manager|"
                  r"robot_diagnostic"),
    ("導航", r"planner_server|controller_server|bt_navigator|behavior_server|costmap|"
             r"collision_monitor|waypoint|velocity_optimizer|nav2|navigo|map_server|"
             r"charging|charge|kanon|lifecycle_manager|roamerx|alg_interface|remoix"),
]


def group_of(name: str) -> str:
    for g, pat in GROUPS:
        if re.search(pat, name, re.I):
            return g
    return "其他"


def parse_node_info(text: str):
    """切出每個 `ros2 node info` 區塊，回 {node: {pubs:[(topic,type)], subs:[...], ...}}"""
    out = {}
    cur, section = None, None
    for line in text.splitlines():
        m = re.match(r"^={10,}\s*NODEINFO\s+(\S+)\s*={10,}$", line)
        if m:
            cur = m.group(1)
            out[cur] = {"pubs": [], "subs": [], "services": 0, "actions": 0}
            section = None
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.endswith(":") and not s.startswith("/"):
            low = s.lower()
            if "publishers" in low:
                section = "pubs"
            elif "subscribers" in low:
                section = "subs"
            elif "service servers" in low or "service clients" in low:
                section = "svc"
            elif "action" in low:
                section = "act"
            else:
                section = None
            continue
        # 形如 "/front_lidar: sensor_msgs/msg/PointCloud2"
        m = re.match(r"^(/\S+):\s*(\S+)$", s)
        if m and section in ("pubs", "subs"):
            out[cur][section].append((m.group(1), m.group(2)))
        elif m and section == "svc":
            out[cur]["services"] += 1
        elif m and section == "act":
            out[cur]["actions"] += 1
    return out


def build(d):
    boards = {}
    for b, fn in BOARD_FILES.items():
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as f:
            txt = f.read()
        nodes = parse_node_info(txt)
        if nodes:
            boards[b] = nodes
    return boards


def md_tables(boards):
    L = ["# D1 Max 的 ROS 2 節點與 topic 全圖（由 `ros2 node info` 重建）", "",
         "狗上沒有 GUI，`rqt_graph` 起不來。這份是把每個節點的 `ros2 node info` "
         "撈回本機重建的同一張圖。兩塊板是**兩個獨立的 ROS DOMAIN**"
         "（RK=66、NX=24），只有影像 topic 由 `bridge_image_topics_*` 橋接。", ""]
    for b, nodes in boards.items():
        topics = {}
        for n, info in nodes.items():
            for t, ty in info["pubs"]:
                topics.setdefault(t, {"type": ty, "pub": [], "sub": []})["pub"].append(n)
            for t, ty in info["subs"]:
                topics.setdefault(t, {"type": ty, "pub": [], "sub": []})["sub"].append(n)
        L += ["---", "", "## %s：%d 個節點、%d 個 topic" % (BOARD_NAME[b], len(nodes), len(topics)), "",
              "### 節點 → 發布 / 訂閱", "",
              "| 節點 | 分類 | 發布 | 訂閱 | 服務 | 動作 |", "|---|---|---|---|---|---|"]
        for n in sorted(nodes):
            i = nodes[n]
            L.append("| `%s` | %s | %d | %d | %d | %d |"
                     % (n, group_of(n), len(i["pubs"]), len(i["subs"]),
                        i["services"], i["actions"]))
        L += ["", "### Topic → 發布者 / 訂閱者", "",
              "| Topic | 型別 | 發布者 | 訂閱者 |", "|---|---|---|---|"]
        for t in sorted(topics):
            x = topics[t]
            L.append("| `%s` | `%s` | %s | %s |"
                     % (t, x["type"],
                        "、".join("`%s`" % p for p in sorted(set(x["pub"]))) or "—",
                        "、".join("`%s`" % p for p in sorted(set(x["sub"]))) or "—"))
        L += ["", "### 每個節點的完整接線", ""]
        for n in sorted(nodes):
            i = nodes[n]
            L.append("**`%s`**（%s）" % (n, group_of(n)))
            L.append("- 發布：" + ("、".join("`%s`" % t for t, _ in i["pubs"]) or "—"))
            L.append("- 訂閱：" + ("、".join("`%s`" % t for t, _ in i["subs"]) or "—"))
            L.append("")
    return "\n".join(L)


def dot(boards):
    """完整圖（含雜訊 topic 以外的全部）。用 `dot -Tsvg graph.dot -o graph.svg` 畫。"""
    L = ["digraph ros {", '  rankdir=LR;', '  node [fontname="Sans"];',
         '  edge [color="#888888"];']
    colors = {"感測驅動": "#cde8d0", "SLAM／定位": "#cfe2f3", "感知": "#fce5cd",
              "導航": "#d9d2e9", "運控／HAL": "#f4cccc", "其他": "#eeeeee"}
    for b, nodes in boards.items():
        L.append('  subgraph cluster_%s {' % b)
        L.append('    label="%s"; style=dashed;' % BOARD_NAME[b])
        for n in sorted(nodes):
            g = group_of(n)
            L.append('    "%s:%s" [label="%s", shape=box, style=filled, fillcolor="%s"];'
                     % (b, n, n, colors[g]))
        L.append("  }")
    seen_t = set()
    for b, nodes in boards.items():
        for n, info in nodes.items():
            for t, _ in info["pubs"] + info["subs"]:
                if NOISE.search(t):
                    continue
                key = (b, t)
                if key not in seen_t:
                    seen_t.add(key)
                    L.append('  "T%s:%s" [label="%s", shape=ellipse, '
                             'style=filled, fillcolor="#ffffff"];' % (b, t, t))
        for n, info in nodes.items():
            for t, _ in info["pubs"]:
                if not NOISE.search(t):
                    L.append('  "%s:%s" -> "T%s:%s";' % (b, n, b, t))
            for t, _ in info["subs"]:
                if not NOISE.search(t):
                    L.append('  "T%s:%s" -> "%s:%s";' % (b, t, b, n))
    L.append("}")
    return "\n".join(L)


def dot_core(boards, only_board=None):
    """主幹圖（DOT）：只留**真的有人發也有人收**的 topic，去掉視覺化與雜訊。

    完整圖 42＋21 個節點、280 個 topic，畫出來 1777×11390 pt 一條細長帶子，沒人看得懂。
    這張為了能放進報告做了四件事：
      - 去掉 /parameter_events、/rosout、/tf、transition_event（每個節點都接）
      - 去掉 /vis/、marker、debug（給 RViz 看的，不是資料流）
      - 去掉只有發布者沒有訂閱者的 topic（沒人用的輸出）
      - **把同一對節點之間的多條 topic 併成一條邊**（不併的話光邊標籤就 5611 pt 寬）
    `only_board` 給值就只畫那塊板，兩塊板各一張才塞得進一頁。
    """
    colors = {"感測驅動": "#cde8d0", "SLAM／定位": "#cfe2f3", "感知": "#fce5cd",
              "導航": "#d9d2e9", "運控／HAL": "#f4cccc", "其他": "#eeeeee"}
    L = ["digraph ros_core {",
         '  rankdir=LR; splines=spline; concentrate=true;',
         '  graph [fontname="Sans", fontsize=11, nodesep=0.3, ranksep=1.4];',
         '  node [fontname="Sans", fontsize=11];',
         '  edge [fontname="Sans", fontsize=8, color="#8a8a8a", fontcolor="#444444"];']
    for b, nodes in boards.items():
        if only_board and b != only_board:
            continue
        pub, sub = {}, {}
        for n, info in nodes.items():
            for t, _ in info["pubs"]:
                pub.setdefault(t, []).append(n)
            for t, _ in info["subs"]:
                sub.setdefault(t, []).append(n)
        keep = {t for t in pub if t in sub
                and not NOISE.search(t) and not VIS.search(t)}
        # 併邊：同一對節點之間的所有 topic 收進一條
        edges = {}
        for t in sorted(keep):
            for p_ in sorted(set(pub[t])):
                for q_ in sorted(set(sub[t])):
                    if p_ != q_:
                        edges.setdefault((p_, q_), []).append(t.lstrip("/"))
        used = {n for e in edges for n in e}
        L.append('  subgraph cluster_%s {' % b)
        L.append('    label="%s"; style=dashed; fontsize=15; color="#bbbbbb";' % BOARD_NAME[b])
        for g, _ in GROUPS + [("其他", "")]:
            members = [n for n in sorted(used) if group_of(n) == g]
            if not members:
                continue
            L.append('    subgraph cluster_%s_%s {' % (b, re.sub(r"\W", "", g)))
            L.append('      label="%s"; style="filled"; color="#f2f2f2"; fontsize=12;' % g)
            for n in members:
                L.append('      "%s:%s" [label="%s", shape=box, style="rounded,filled", '
                         'fillcolor="%s"];' % (b, n, n, colors[g]))
            L.append("    }")
        L.append("  }")
        for (p_, q_), ts in sorted(edges.items()):
            if len(ts) <= 2:
                lab = "\\n".join(ts)
            else:
                lab = "\\n".join(ts[:2]) + "\\n+%d 個" % (len(ts) - 2)
            L.append('  "%s:%s" -> "%s:%s" [label="%s"];' % (b, p_, b, q_, lab))
    L.append("}")
    return "\n".join(L)


def mermaid_core(boards):
    """主幹圖：去掉雜訊與視覺化 topic，只留真正的資料流，人看得懂的那張。"""
    L = ["flowchart LR"]
    ids, nid = {}, [0]

    def idof(kind, b, name):
        k = (kind, b, name)
        if k not in ids:
            nid[0] += 1
            ids[k] = "%s%d" % ("n" if kind == "node" else "t", nid[0])
        return ids[k]

    for b, nodes in boards.items():
        L.append('  subgraph %s["%s"]' % (b.upper(), BOARD_NAME[b]))
        for g, _ in GROUPS + [("其他", "")]:
            members = [n for n in sorted(nodes) if group_of(n) == g]
            if not members:
                continue
            L.append('    subgraph %s_%s["%s"]' % (b, re.sub(r"\W", "", g), g))
            for n in members:
                L.append('      %s["%s"]' % (idof("node", b, n), n))
            L.append("    end")
        L.append("  end")

    for b, nodes in boards.items():
        for n, info in nodes.items():
            for t, _ in info["pubs"]:
                if NOISE.search(t) or VIS.search(t):
                    continue
                subs = [m for m, i2 in nodes.items()
                        if any(tt == t for tt, _ in i2["subs"])]
                for m in subs:
                    L.append("  %s -- %s --> %s"
                             % (idof("node", b, n), t.lstrip("/"), idof("node", b, m)))
    # 去重
    out, seen = [], set()
    for line in L:
        if line.startswith("  ") and "-->" in line:
            if line in seen:
                continue
            seen.add(line)
        out.append(line)
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 recon5_graph.py <recon5 輸出目錄>", file=sys.stderr)
        return 2
    d = sys.argv[1]
    boards = build(d)
    if not boards:
        print("❌ 在 %s 找不到 NODEINFO 區塊（recon5 跑過了嗎？）" % d, file=sys.stderr)
        return 1

    with open(os.path.join(d, "nodes_topics.md"), "w") as f:
        f.write(md_tables(boards) + "\n")
    with open(os.path.join(d, "graph.dot"), "w") as f:
        f.write(dot(boards) + "\n")
    with open(os.path.join(d, "graph_core.dot"), "w") as f:
        f.write(dot_core(boards) + "\n")
    for b in boards:
        with open(os.path.join(d, "graph_core_%s.dot" % b), "w") as f:
            f.write(dot_core(boards, only_board=b) + "\n")
    with open(os.path.join(d, "graph_core.mmd"), "w") as f:
        f.write(mermaid_core(boards) + "\n")
    with open(os.path.join(d, "graph.json"), "w") as f:
        json.dump(boards, f, ensure_ascii=False, indent=1)

    for b, nodes in boards.items():
        ts = {t for i in nodes.values() for t, _ in i["pubs"] + i["subs"]}
        print("%s：%d 個節點、%d 個 topic" % (BOARD_NAME[b], len(nodes), len(ts)))
    print("寫出 nodes_topics.md / graph.dot / graph_core.mmd / graph.json 於 %s" % d)
    print("畫圖：dot -Tsvg %s/graph.dot -o %s/graph.svg   （沒有 graphviz 就看 .mmd 或表格）"
          % (d, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
