#!/usr/bin/env python3
"""recon5_graph —— 把 `ros2 node info` 的原始輸出重建成 node／topic 圖。

為什麼不用 rqt_graph：**狗上沒有 GUI**，`rqt_graph` 起不來，而且它也不會輸出檔案。
`recon5_nav_ai.sh` 把每個節點的 `ros2 node info` 全撈回本機，這支在本機重建同一張圖。

產出（都寫進輸出目錄）：
  nodes_topics.md   完整表格：節點→發布／訂閱、topic→發布者／訂閱者（報告直接用這份）
  graph.dot         Graphviz 原始檔 → `dot -Tsvg graph.dot -o graph.svg`
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
                   r"^/tf(_static)?$|^/clock$|^/diagnostics")
VIS = re.compile(r"/vis/|visualization|/debug|marker", re.I)

BOARD_FILES = {"nx": "nav_ai_nx.log", "rk": "nav_ai_rk.log"}
BOARD_NAME = {"nx": "Orin NX（應用板）", "rk": "RK3588（運控板）"}

# 主幹圖的分組（依名稱歸類，只影響畫圖的顏色與階層）
GROUPS = [
    ("感測驅動", r"rslidar|imu_driver|uss_driver|uwb_driver|gps_driver|sixents|robot_camera"),
    ("SLAM／定位", r"slam|lvio|mapping|localization|aritag|apriltag|robot_tf"),
    ("感知", r"perception"),
    ("導航", r"planner|controller|bt_navigator|behavior|costmap|collision|waypoint|"
             r"velocity_optimizer|nav2|navigo|map_server|charging|MEB|kanon|lifecycle"),
    ("運控／HAL", r"robot_hal|joint_shm|imu_shm|mc_ctrl|switch_controller|estop|battery|"
                  r"led_controller|fill_light|robot_remote|robot_monitor|robot_manager"),
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
