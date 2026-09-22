#!/usr/bin/env python3
"""recon3_report —— 把第三趟偵察的原始輸出整理成報告填空表。

吃 `recon3_sensors_d1max.sh` 的輸出目錄，產出 `報告填空表.md`：
報告項目 **1.3 / 7.1 / 7.2 / 7.4** 四張表。

★ 規則：**沒量到的一律寫「未取到」**，絕不填手冊宣稱值。
  報告要的是實機證據；混進宣稱值，整張表就沒人能分辨哪個是真的量過的。

用法：  python3 recon3_report.py <輸出目錄>
        （可以事後重跑，不必再連狗）
"""
from __future__ import annotations

import json
import os
import re
import sys

NA = "未取到"

PHASES = [("rk", "idle"), ("rk", "walk"), ("nx", "idle"), ("nx", "walk")]
BOARD_NAME = {"rk": "RK3588（運控板）", "nx": "Orin NX（應用板）"}
STATIC_FILE = {"rk": "rk3588_static.log", "nx": "orinnx_static.log"}


# ----------------------------------------------------------------- 讀檔
def load_json(path):
    try:
        with open(path, "r", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def load_text(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def parse_markers(text):
    """撈出靜態盤點刻意印的 @@ 標記。"""
    kv, topics, topics2, camdevs, lidarcfg, files = {}, [], [], [], [], []
    for line in text.splitlines():
        if line.startswith("@@KV "):
            k, _, v = line[5:].partition("=")
            kv[k.strip()] = v.strip()
        elif line.startswith("@@TOPIC "):
            parts = line[8:].split("|")
            while len(parts) < 4:
                parts.append(NA)
            topics.append({"topic": parts[0].strip(), "type": parts[1].strip(),
                           "hz": parts[2].strip() or NA, "pubs": parts[3].strip() or NA})
        elif line.startswith("@@TOPIC2 "):
            rest = line[9:].strip()
            name = rest.split()[0] if rest else ""
            ty = rest[rest.find("[") + 1:rest.find("]")] if "[" in rest else NA
            if name:
                topics2.append({"topic": name, "type": ty})
        elif line.startswith("@@CAMDEV "):
            camdevs.append([line[9:].strip(), None])
        elif camdevs and camdevs[-1][1] is None and line.strip().startswith("Card:"):
            camdevs[-1][1] = line.split(":", 1)[1].strip()
        elif line.startswith("@@LIDARCFG "):
            lidarcfg.append(line[11:].strip())
        elif line.startswith("@@FILE "):
            files.append(line[7:].strip())
    return {"kv": kv, "topics": topics, "topics2": topics2, "camdevs": camdevs,
            "lidarcfg": lidarcfg, "files": files}


def topic_blocks(text):
    """把靜態 log 裡每個感測 topic 的區塊切出來 → {topic: 區塊文字}。"""
    blocks, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^-{10,}\s+(\S+)\s+-{10,}$", line)
        if m:
            if cur:
                blocks[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
            continue
        if cur:
            buf.append(line)
    if cur:
        blocks[cur] = "\n".join(buf)
    return blocks


def grab(block, key):
    """從 `ros2 topic echo` 的 YAML 區塊裡抓一個純量欄位。"""
    m = re.search(r"^\s*%s:\s*(.+)$" % re.escape(key), block, re.M)
    return m.group(1).strip() if m else None


# ----------------------------------------------------------------- 表格工具
def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(v, suffix="", nd=1):
    if v is None:
        return NA
    if isinstance(v, float):
        return ("%%.%df%%s" % nd) % (v, suffix)
    return "%s%s" % (v, suffix)


def tegrastats_stats(s):
    """把 tegrastats 的原始行解析成統計值（GPU 使用率、RAM、整機功耗）。

    Orin 的 GPU 使用率在 devfreq 只有千分比的瞬時值，
    `GR3D_FREQ` 是第二個獨立來源 —— 兩個對得上才敢寫進報告。
    """
    out = {}
    lines = s.get("tegrastats") or []
    for key, pat, scale in (("gr3d", r"GR3D_FREQ (\d+)%", 1.0),
                            ("ram_mb", r"RAM (\d+)/", 1.0),
                            ("vdd_in_w", r"VDD_IN (\d+)mW", 0.001)):
        vals = [float(m.group(1)) * scale
                for L in lines for m in [re.search(pat, L)] if m]
        if vals:
            out[key] = {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}
    return out


# ----------------------------------------------------------------- 1.3
def sec_13(d, samples, statics):
    L = ["## 1.3　CPU / GPU / Memory 資源使用情形", "",
         "### 1.3-a　兩塊板的靜態規格（實機讀取）", ""]

    rows = []
    for b in ("rk", "nx"):
        kv = statics[b]["kv"]
        s = samples.get((b, "idle")) or samples.get((b, "walk")) or {}
        mem = (s.get("mem_end") or {}).get("total_mb")
        rows.append([
            BOARD_NAME[b],
            kv.get("board", NA),
            kv.get("cpu_model", NA),
            kv.get("ncpu", str(s.get("ncpu", NA))),
            kv.get("cpu_maxmhz", NA),
            fmt(mem, " MB", 0) if mem else (
                "%.0f MB" % (int(kv["mem_total_kb"]) / 1024) if kv.get("mem_total_kb") else NA),
            kv.get("isolcpus") or (s.get("isolated_cpus") or NA),
            kv.get("os", NA),
            kv.get("firmware", NA),
        ])
    L += [table(["板", "板卡型號", "CPU", "核數", "最高 MHz", "RAM", "核心隔離",
                 "OS", "韌體"], rows), ""]

    L += ["### 1.3-b　資源使用情形：待機 vs 走路", "",
          "採樣方式：每秒讀 `/proc/stat`、`/proc/meminfo`、`devfreq`、`thermal_zone`，"
          "窗口起訖各讀一次 `/proc/<pid>/stat` 算行程 CPU%。走路那段由人用遙控器操作。", ""]
    rows = []
    for b, ph in PHASES:
        s = samples.get((b, ph))
        if not s:
            rows.append([BOARD_NAME[b], "待機" if ph == "idle" else "走路",
                         NA, NA, NA, NA, NA, NA])
            continue
        per = [v for v in (s.get("per_cpu_pct") or []) if v is not None]
        mem = s.get("mem_end") or {}
        gpu = s.get("gpu_npu") or {}
        gpu_txt, npu_note = [], []
        for tag, st in sorted(gpu.items()):
            if "cur_freq" in tag:
                continue
            # RK3588 的 devfreq load 是 "<百分比>@<頻率>Hz" 字串 → 取前面的數字
            if "mean" not in st:
                first, last = str(st.get("raw_first", "")), str(st.get("raw_last", ""))
                try:
                    v0 = float(first.split("@")[0]); v1 = float(last.split("@")[0])
                except Exception:
                    continue
                gpu_txt.append("%s %.0f%% → %.0f%%" % (tag.split(":")[-2], v0, v1))
                if "npu" in tag and v0 == v1 == 100.0:
                    npu_note.append(tag)
                continue
            if not st.get("max"):
                continue
            mean, mx = st["mean"], st["max"]
            # Jetson 的 gpu load 是千分比（0–1000）→ 除 10 才是 %
            if "gpu" in tag and mx > 100:
                mean, mx = mean / 10.0, mx / 10.0
            gpu_txt.append("%s 平均 %.1f%% 峰 %.1f%%" % (tag.split(":")[-2] if ":" in tag else tag, mean, mx))
        # tegrastats 的 GR3D_FREQ 是 Orin GPU 使用率的第二個獨立來源
        teg = tegrastats_stats(s)
        if teg.get("gr3d"):
            g = teg["gr3d"]
            gpu_txt.append("tegrastats GR3D 平均 %.0f%% 峰 %.0f%%" % (g["mean"], g["max"]))
        therm = s.get("thermal_c") or {}
        hot = max(((v["max_c"], k) for k, v in therm.items()), default=(None, None))
        rows.append([
            BOARD_NAME[b],
            "待機" if ph == "idle" else "走路",
            "%.0f s" % s.get("secs", 0),
            fmt(s.get("cpu_overall_pct"), "%"),
            "%.1f%%（cpu%d）" % (max(per), per.index(max(per))) if per else NA,
            "%s / %s（%s）" % (fmt(mem.get("used_mb"), " MB", 0),
                              fmt(mem.get("total_mb"), " MB", 0),
                              fmt(mem.get("used_pct"), "%")),
            "；".join(gpu_txt) if gpu_txt else NA,
            "%.1f°C（%s）" % (hot[0], hot[1]) if hot[0] is not None else NA,
        ])
    L += [table(["板", "情境", "窗口", "整體 CPU", "最忙的核", "記憶體 已用/總量",
                 "GPU / NPU 負載", "最高溫"], rows), ""]
    if any("npu" in t for s2 in samples.values() for t in (s2.get("gpu_npu") or {})):
        L += ["> ⚠️ **RK3588 的 NPU 讀值要小心**：`devfreq` 的 `load` 節點兩個情境都固定 100，"
              "而 Mali GPU 的 `utilisation` 是 0。固定不動的讀值通常不是真實使用率。"
              "下一趟用 `sudo cat /sys/kernel/debug/rknpu/load` 對照才能定案 —— "
              "這件事有意義，因為原廠的運控策略是 RKNN 模型（`librknn_model.so`），"
              "NPU 若真的被佔著，我們自己要上 NPU 推論就得先確認排擠。", ""]

    # 每核對照（運控在 RK，cpu7 被核心隔離 —— 這張表就是在驗它）
    for b in ("rk", "nx"):
        si, sw = samples.get((b, "idle")), samples.get((b, "walk"))
        if not si:
            continue
        n = si.get("ncpu", 0)
        rows = []
        for i in range(n):
            gi = (si.get("per_cpu_pct") or [None] * n)[i]
            gim = (si.get("per_cpu_max_pct") or [None] * n)[i]
            gw = (sw.get("per_cpu_pct") or [None] * n)[i] if sw else None
            gwm = (sw.get("per_cpu_max_pct") or [None] * n)[i] if sw else None
            rows.append(["cpu%d" % i, fmt(gi, "%"), fmt(gim, "%"),
                         fmt(gw, "%"), fmt(gwm, "%")])
        L += ["### 1.3-c%d　%s 每核使用率（待機 vs 走路）"
              % (1 if b == "rk" else 2, BOARD_NAME[b]), "",
              table(["核", "待機 平均", "待機 峰", "走路 平均", "走路 峰"], rows), ""]

        iso = si.get("isolated_cpus")
        if iso:
            L += ["核心隔離 `isolcpus = %s`。那顆核上跑什麼，看下面的執行緒表 —— "
                  "行程層級的 `Cpus_allowed_list` 看不出來（`mc_ctrl` 主執行緒是 0-6）。" % iso, ""]

    # 行程層級
    for b, ph in PHASES:
        s = samples.get((b, ph))
        if not s or not s.get("top_proc"):
            continue
        rows = [[r["pid"], r["comm"], fmt(r["cpu_pct"], "%"),
                 fmt(r["rss_mb"], " MB"), r.get("cpus_allowed") or NA,
                 (r.get("cmd") or "")[:60]]
                for r in s["top_proc"][:10]]
        L += ["### 1.3-d　%s / %s　吃資源的前 10 個行程" %
              (BOARD_NAME[b], "待機" if ph == "idle" else "走路"), "",
              table(["PID", "行程", "CPU", "RSS", "可用核", "指令"], rows), ""]

    # 執行緒層級（主執行緒的 mask 騙人：mc_ctrl 主執行緒是 0-6，但 cpu7 是隔離核）
    for b, ph in PHASES:
        s = samples.get((b, ph))
        if not s or not s.get("top_thread"):
            continue
        iso = (s.get("isolated_cpus") or "").strip()
        rows = [[r["proc"], r["thread"], r["pid"], r["tid"], fmt(r["cpu_pct"], "%"),
                 "cpu%s" % r["last_cpu"], r.get("cpus_allowed") or NA]
                for r in s["top_thread"][:12]]
        L += ["### 1.3-e　%s / %s　關鍵行程的執行緒" %
              (BOARD_NAME[b], "待機" if ph == "idle" else "走路"), "",
              table(["行程", "執行緒", "PID", "TID", "CPU", "上次在哪顆核",
                     "可用核"], rows), ""]
        if iso:
            on_iso = [r for r in s["top_thread"]
                      if str(r["last_cpu"]) in iso.replace("-", ",").split(",")]
            L += ["核心隔離 `isolcpus = %s`。取樣瞬間落在隔離核上的執行緒：%s" % (
                iso,
                "、".join("`%s/%s`" % (r["proc"], r["thread"]) for r in on_iso)
                or "這次沒抓到（只代表取樣瞬間，不能斷定沒有）"), ""]

    rows = []
    for b, ph in PHASES:
        s = samples.get((b, ph))
        if not s:
            continue
        t = tegrastats_stats(s)
        if not t:
            continue
        rows.append([BOARD_NAME[b], "待機" if ph == "idle" else "走路",
                     "%.0f%% / %.0f%%" % (t["gr3d"]["mean"], t["gr3d"]["max"])
                     if t.get("gr3d") else NA,
                     "%.0f MB" % t["ram_mb"]["mean"] if t.get("ram_mb") else NA,
                     "%.1f W / %.1f W" % (t["vdd_in_w"]["mean"], t["vdd_in_w"]["max"])
                     if t.get("vdd_in_w") else NA])
    if rows:
        L += ["### 1.3-f　Orin NX 的 GPU 與整機功耗（`tegrastats`，每秒一筆）", "",
              table(["板", "情境", "GPU 平均/峰", "RAM 平均", "VDD_IN 平均/峰"], rows), "",
              "`VDD_IN` 是模組輸入功耗，不是整台狗的耗電。", ""]

    # tegrastats 原始行（Orin 的 GPU 使用率只有這裡看得到）
    for b in ("nx", "rk"):
        for ph in ("idle", "walk"):
            s = samples.get((b, ph))
            if s and s.get("tegrastats"):
                L += ["### 1.3-g　%s / %s　tegrastats 原始輸出（前 3 行）" %
                      (BOARD_NAME[b], ph), "", "```",
                      "\n".join(s["tegrastats"][:3]), "```", ""]
            elif s and s.get("tegrastats_note") and b == "nx":
                L += ["> %s / %s tegrastats：%s" %
                      (BOARD_NAME[b], ph, s["tegrastats_note"]), ""]
    return "\n".join(L)


# ----------------------------------------------------------------- 7.1
SENSOR_GROUPS = [
    ("光達 LiDAR", r"lidar|scan|points"),
    ("相機 Camera", r"camera|image|rgb|depth"),
    ("IMU", r"imu"),
    ("超音波 Ultrasonic", r"uss|ultra|range"),
    ("RTK / GNSS", r"rtk|gps|gnss|navsat"),
    ("UWB", r"uwb"),
]


def sec_71(statics):
    L = ["## 7.1　感測器種類與數量", "",
         "來源：兩塊板的 ROS2 topic 實表（RK `ROS_DOMAIN_ID=66`、NX `=24`）。"
         "`Hz` 是現場實測（`ros2 topic hz`，best_effort 會自動退一步重試），不是設定值。", ""]

    rows, all_topics = [], []
    for b in ("rk", "nx"):
        for t in statics[b]["topics"]:
            rows.append([BOARD_NAME[b], t["topic"], t["type"], t["hz"], t["pubs"]])
            all_topics.append((b, t))
    if rows:
        L += [table(["板", "Topic", "型別", "實測 Hz", "發布者數"], rows), ""]
    else:
        L += ["⚠️ **%s** —— 兩塊板都沒撈到感測 topic。先看 log 裡的 `ROS_DOMAIN_ID` 與 "
              "`RMW_IMPLEMENTATION` 兩行，那是最常見的原因。" % NA, ""]

    L += ["### 7.1-b　按種類清點", "",
          "這裡只數 **topic**（同名的已去重）。一顆感測器可能發多個 topic"
          "（光達另發自己的 IMU），也有 topic 根本不是感測器而是算出來的"
          "（`/laser_scan` 是點雲壓成的 2D 掃描）。"
          "所以「幾顆」要配合裝置列舉與安裝位置一起判，不能直接拿 topic 數當顆數。", ""]
    rows = []
    for name, pat in SENSOR_GROUPS:
        # 同一個 topic 可能在兩塊板上都看得到（影像由 bridge_image_topics_66 橋過去）
        # → 去重，否則相機會被數成兩倍
        hits = sorted({t["topic"] for b, t in all_topics
                       if re.search(pat, t["topic"], re.I)})
        rows.append([name, len(hits), "、".join(hits) if hits else NA])
    L += [table(["種類", "topic 數", "topic"], rows), ""]

    rows2 = []
    for b in ("rk", "nx"):
        for t in statics[b].get("topics2", []):
            rows2.append([BOARD_NAME[b], t["topic"], t["type"]])
    if rows2:
        L += ["### 7.1-b2　其他名稱像感測器的 topic（存在，但這趟沒量頻率）", "",
              table(["板", "Topic", "型別"], rows2), "",
              "這些多半是導航／SLAM 的中間產物（點雲重投影、視覺化 marker），"
              "不是獨立的感測器。列出來是為了證明「感測 topic 沒有被漏看」。", ""]

    L += ["### 7.1-c　裝置列舉（交叉佐證）", ""]
    rows = []
    for b in ("rk", "nx"):
        kv = statics[b]["kv"]
        cams = statics[b]["camdevs"]
        rows.append([BOARD_NAME[b],
                     "%d 個節點" % len(cams) if cams else NA,
                     kv.get("ros_domain_id", NA),
                     kv.get("topic_count", NA)])
    L += [table(["板", "/dev/video*", "ROS_DOMAIN_ID", "topic 總數"], rows), "",
          "USB 裝置、序列埠、CAN、光達獨立網段的原始輸出在 "
          "`rk3588_static.log` / `orinnx_static.log` 的「7.1」各段。", ""]
    return "\n".join(L)


# ----------------------------------------------------------------- 7.2
def ffprobe_row(path, label):
    d = load_json(path)
    if not d or not d.get("streams"):
        return [label, NA, NA, NA, NA, NA]
    v = None
    for s in d["streams"]:
        if s.get("codec_type") == "video":
            v = s
            break
    v = v or d["streams"][0]
    fps = v.get("avg_frame_rate") or v.get("r_frame_rate") or ""
    if "/" in fps:
        a, _, b = fps.partition("/")
        try:
            fps = "%.2f" % (float(a) / float(b)) if float(b) else NA
        except Exception:
            fps = NA
    br = v.get("bit_rate") or (d.get("format") or {}).get("bit_rate")
    return [label,
            v.get("codec_name", NA),
            "%s × %s" % (v.get("width", "?"), v.get("height", "?")),
            fps or NA,
            v.get("pix_fmt", NA),
            "%.2f Mbps" % (int(br) / 1e6) if br and str(br).isdigit() else NA]


def sec_72(d, statics):
    L = ["## 7.2　Camera：解析度 / Frame Rate", "",
         "來源：從 PC 用 `ffprobe` 實拉 RTSP（`rtsp://<RK>:8554/{front,back}`），"
         "**這是實際編碼出來的畫面**，不是手冊值。", ""]
    rows = [ffprobe_row(os.path.join(d, "rtsp_front.json"), "前相機 RTSP `/front`"),
            ffprobe_row(os.path.join(d, "rtsp_back.json"), "後相機 RTSP `/back`")]
    L += [table(["來源", "編碼", "解析度", "Frame Rate", "像素格式", "碼率"], rows), ""]

    cam_topics = []
    for b in ("rk", "nx"):
        for t in statics[b]["topics"]:
            if re.search(r"camera|image", t["topic"], re.I):
                cam_topics.append([BOARD_NAME[b], t["topic"], t["type"], t["hz"]])
    if cam_topics:
        L += ["### 7.2-b　相機的 ROS2 topic", "",
              table(["板", "Topic", "型別", "實測 Hz"], cam_topics), ""]
    else:
        L += ["### 7.2-b　相機的 ROS2 topic", "",
              "%s —— 沒有相機 topic。D1 Max 的影像是走 RTSP，"
              "不一定會發成 ROS2 image topic；以 RTSP 實測為準。" % NA, ""]

    L += ["### 7.2-c　`/dev/video*`（節點數 ≠ 相機顆數）", ""]
    for b in ("rk", "nx"):
        devs = statics[b]["camdevs"]
        if not devs:
            continue
        groups = {}
        for dev, card in devs:
            groups.setdefault(card or NA, []).append(dev)
        rows = [[card, len(v), "、".join(sorted(v)[:4]) + ("…" if len(v) > 4 else "")]
                for card, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
        L += ["**%s**：共 %d 個節點" % (BOARD_NAME[b], len(devs)), "",
              table(["Card type", "節點數", "節點"], rows), ""]
    L += ["`rkcif` 是 MIPI 擷取節點，`rkisp_*` 是 ISP 的統計／參數／raw 讀回，"
          "`video-dec0` / `video-enc0` 是硬體編解碼 —— **這些都不是獨立的相機**。"
          "擷取到的原生格式與解析度見靜態 log 的「7.2」段。", ""]
    return "\n".join(L)


# ----------------------------------------------------------------- 7.4
def sec_74(statics):
    L = ["## 7.4　LiDAR 等其他感測器規格", ""]

    # 光達：topic 實測 + PointCloud2 欄位
    rows = []
    for b in ("rk", "nx"):
        blocks = statics[b]["blocks"]
        for t in statics[b]["topics"]:
            if not re.search(r"lidar|scan|points", t["topic"], re.I):
                continue
            if "PointCloud2" not in t["type"] and "LaserScan" not in t["type"]:
                continue
            blk = blocks.get(t["topic"], "")
            rows.append([BOARD_NAME[b], t["topic"], t["type"], t["hz"],
                         grab(blk, "width") or NA,
                         grab(blk, "height") or NA,
                         grab(blk, "point_step") or NA,
                         grab(blk, "frame_id") or NA])
    L += ["### 7.4-a　光達：實測頻率與每掃資料量", ""]
    if rows:
        L += [table(["板", "Topic", "型別", "實測 Hz", "width", "height",
                     "point_step", "frame_id"], rows), "",
              "`width × height` 是一筆訊息裡的點數；`point_step` 是每點位元組數。"
              "線數要配合下面的設定檔判讀（`height=1` 的無序點雲看不出線數）。", ""]
    else:
        L += ["%s —— 沒有量到光達點雲。可能是 NX 連不上，或 topic 用 best_effort "
              "而訂閱沒對上（腳本已重試一次）。" % NA, ""]

    L += ["### 7.4-b　光達設定檔的關鍵欄位", ""]
    cfg = statics["rk"]["lidarcfg"] + statics["nx"]["lidarcfg"]
    if cfg:
        L += ["```"] + cfg[:60] + ["```", "",
              "型號、線數、掃描頻率、FOV、echo mode 通常就在這幾行裡；"
              "設定檔全文在靜態 log 的 `@@FILE` 段。", ""]
    else:
        L += ["%s —— 找不到光達設定檔。" % NA, ""]

    # 超音波 / IMU / RTK / UWB
    rows = []
    for b in ("rk", "nx"):
        blocks = statics[b]["blocks"]
        for t in statics[b]["topics"]:
            if re.search(r"lidar|scan|points|camera|image", t["topic"], re.I) \
               and "Range" not in t["type"] and "Imu" not in t["type"]:
                continue
            blk = blocks.get(t["topic"], "")
            extra = []
            for k in ("min_range", "max_range", "field_of_view", "radiation_type"):
                v = grab(blk, k)
                if v:
                    extra.append("%s=%s" % (k, v))
            rows.append([BOARD_NAME[b], t["topic"], t["type"], t["hz"],
                         "；".join(extra) if extra else "—"])
    if rows:
        L += ["### 7.4-c　其他感測器（超音波 / IMU / RTK / UWB）", "",
              table(["板", "Topic", "型別", "實測 Hz", "訊息自帶的規格欄位"], rows), "",
              "`sensor_msgs/Range` 自帶 `min_range` / `max_range` / `field_of_view`，"
              "那就是超音波的量程與視角規格，不必查手冊。", ""]

    L += ["### 7.4-d　安裝位置", "",
          "這趟沒有重新量安裝位置 —— URDF 解析的結果已在 "
          "`task7/docs/D1Max_機器狗資訊.md` §1.5，"
          "含「URDF 的 `IMU_LUA300C` 少一個零、以手冊為準」那個坑。", ""]
    return "\n".join(L)


# ----------------------------------------------------------------- 主流程
def main():
    if len(sys.argv) < 2:
        print("用法: python3 recon3_report.py <輸出目錄>", file=sys.stderr)
        return 2
    d = sys.argv[1]
    if not os.path.isdir(d):
        print("❌ 不是目錄: %s" % d, file=sys.stderr)
        return 1

    samples = {}
    for b, ph in PHASES:
        s = load_json(os.path.join(d, "%s_%s.json" % (b, ph)))
        if s:
            samples[(b, ph)] = s

    statics = {}
    for b in ("rk", "nx"):
        txt = load_text(os.path.join(d, STATIC_FILE[b]))
        st = parse_markers(txt)
        st["blocks"] = topic_blocks(txt)
        st["raw_len"] = len(txt)
        statics[b] = st

    head = [
        "# D1 Max 報告填空表（第三趟偵察）",
        "",
        "- 產生：`recon3_report.py`　來源目錄：`%s`" % os.path.abspath(d),
        "- 對應報告項目：**1.3**、**7.1**、**7.2**、**7.4**",
        "- 規則：**沒量到的一律寫「%s」，不填手冊宣稱值。**"
        "要對照手冊請看 `task7/docs/D1Max_機器狗資訊.md`。" % NA,
        "",
        "| 資料來源 | 狀態 |",
        "|---|---|",
    ]
    for b in ("rk", "nx"):
        head.append("| %s 靜態盤點 | %s |" % (
            BOARD_NAME[b],
            "✅ %d 字元、%d 個感測 topic" % (statics[b]["raw_len"], len(statics[b]["topics"]))
            if statics[b]["raw_len"] else "❌ 沒有輸出"))
    for b, ph in PHASES:
        head.append("| %s / %s 採樣 | %s |" % (
            BOARD_NAME[b], "待機" if ph == "idle" else "走路",
            "✅ %s 秒" % samples[(b, ph)].get("secs") if (b, ph) in samples else "❌ 未取到"))
    head.append("")

    body = "\n\n---\n\n".join([
        sec_13(d, samples, statics),
        sec_71(statics),
        sec_72(d, statics),
        sec_74(statics),
    ])

    out = os.path.join(d, "報告填空表.md")
    with open(out, "w") as f:
        f.write("\n".join(head) + "\n" + body + "\n")
    print("寫出 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
