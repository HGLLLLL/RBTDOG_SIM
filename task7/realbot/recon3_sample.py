#!/usr/bin/env python3
"""recon3_sample —— 狗上資源採樣器（報告項目 1.3）。

**唯讀、不留檔、不碰馬達。** 只讀 /proc 與 /sys，外加（若存在）tegrastats。

★ 這支不會被 scp 到狗上。`recon3_sensors_d1max.sh` 是這樣跑的：

    ssh robot@<ip> "python3 - 30 idle" < recon3_sample.py

檔案內容從 stdin 進去、參數照樣在 argv，所以**狗上不產生任何檔案**，
維持前兩趟偵察「不在狗上留東西」的原則。

用法：  python3 recon3_sample.py <秒數> <標籤>
        python3 recon3_sample.py 30 idle

輸出：  stdout = JSON（給 recon3_report.py 產表）
        stderr = 人讀摘要（現場直接看）

為什麼要用 Python 而不是在 bash 裡 top/vmstat：
報告要回答的是「哪個行程吃掉哪幾顆核」。那需要 /proc/stat 與 /proc/<pid>/stat
兩次取樣相減，還要對上 Cpus_allowed_list（cpu7 被運控核心隔離，要證明它真的只給運控）。
top 的輸出格式跨版本會變，解析它比自己讀 /proc 脆弱。
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_KB = (os.sysconf("SC_PAGE_SIZE") // 1024) if hasattr(os, "sysconf") else 4


# ----------------------------------------------------------------- 小工具
def read_text(path, limit=4096):
    """讀檔，失敗回 None（權限不足、節點不存在都算失敗，呼叫端自己判）。"""
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(limit).strip()
    except Exception:
        return None


def read_int(path):
    t = read_text(path, 64)
    if t is None:
        return None
    try:
        return int(t.split()[0])
    except Exception:
        return None


# ----------------------------------------------------------------- CPU
def cpu_snapshot():
    """回 {cpu 名: (busy_jiffies, total_jiffies)}。busy 不含 idle 與 iowait。"""
    out = {}
    txt = read_text("/proc/stat", 65536) or ""
    for line in txt.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        name = parts[0]
        vals = [int(v) for v in parts[1:]]
        # user nice system idle iowait irq softirq steal guest guest_nice
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals[:8]) if len(vals) >= 8 else sum(vals)
        out[name] = (total - idle, total)
    return out


def cpu_pct(a, b):
    """兩個 snapshot 相減 → {cpu 名: 使用率 %}。"""
    out = {}
    for name, (busy_b, total_b) in b.items():
        if name not in a:
            continue
        busy_a, total_a = a[name]
        dt = total_b - total_a
        if dt <= 0:
            continue
        out[name] = round(100.0 * (busy_b - busy_a) / dt, 1)
    return out


# ----------------------------------------------------------------- 記憶體
def meminfo():
    txt = read_text("/proc/meminfo", 8192) or ""
    d = {}
    for line in txt.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().split()
        if v and v[0].isdigit():
            d[k.strip()] = int(v[0])          # kB
    tot = d.get("MemTotal")
    avail = d.get("MemAvailable")
    return {
        "total_mb": round(tot / 1024, 1) if tot else None,
        "available_mb": round(avail / 1024, 1) if avail else None,
        "used_mb": round((tot - avail) / 1024, 1) if (tot and avail) else None,
        "used_pct": round(100.0 * (tot - avail) / tot, 1) if (tot and avail) else None,
        "cached_mb": round(d.get("Cached", 0) / 1024, 1),
        "swap_total_mb": round(d.get("SwapTotal", 0) / 1024, 1),
        "swap_used_mb": round((d.get("SwapTotal", 0) - d.get("SwapFree", 0)) / 1024, 1),
    }


# ----------------------------------------------------------------- 行程
def proc_snapshot():
    """{pid: {ticks, comm, rss_mb, cpus_allowed, cmd}}"""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        stat = read_text("/proc/%s/stat" % d, 4096)
        if not stat:
            continue
        # comm 可能含空白與括號 → 以最後一個 ')' 切
        try:
            rp = stat.rindex(")")
            comm = stat[stat.index("(") + 1: rp]
            rest = stat[rp + 2:].split()
            utime, stime = int(rest[11]), int(rest[12])
        except Exception:
            continue
        statm = read_text("/proc/%s/statm" % d, 256)
        rss_mb = None
        if statm:
            try:
                rss_mb = round(int(statm.split()[1]) * PAGE_KB / 1024.0, 1)
            except Exception:
                pass
        cmd = read_text("/proc/%s/cmdline" % d, 512)
        cmd = (cmd.replace("\x00", " ").strip() if cmd else "")
        cpus = None
        status = read_text("/proc/%s/status" % d, 8192)
        if status:
            for line in status.splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    cpus = line.split(":", 1)[1].strip()
                    break
        out[int(d)] = {
            "ticks": utime + stime,
            "comm": comm,
            "rss_mb": rss_mb,
            "cpus_allowed": cpus,
            "cmd": cmd[:120],
        }
    return out


THREAD_PATTERNS = ("mc_ctrl", "robot_hal", "robot_camera", "rslidar", "nav2",
                   "arc_", "localization", "perception", "robot_slam", "uss_", "imu_")


def threads_info():
    """關鍵行程的**每個執行緒**：綁哪幾顆核、上次跑在哪顆核、用了多少 tick。

    為什麼要到執行緒層級：`/proc/<pid>/status` 給的是主執行緒的 mask。
    2026-09-22 量到 mc_ctrl 主執行緒是 0-6，但核心明明隔離了 cpu7
    —— 那顆核上跑的是誰，只有看 task/*/status 才知道。
    """
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        comm = read_text("/proc/%s/comm" % d, 64) or ""
        if not any(pat in comm for pat in THREAD_PATTERNS):
            continue
        for t in os.listdir("/proc/%s/task" % d) if os.path.isdir("/proc/%s/task" % d) else []:
            base = "/proc/%s/task/%s" % (d, t)
            stat = read_text(base + "/stat", 4096)
            if not stat:
                continue
            try:
                rest = stat[stat.rindex(")") + 2:].split()
                ticks = int(rest[11]) + int(rest[12])
                processor = int(rest[36])          # 上次執行在哪顆核
            except Exception:
                continue
            cpus = None
            status = read_text(base + "/status", 8192)
            if status:
                for line in status.splitlines():
                    if line.startswith("Cpus_allowed_list:"):
                        cpus = line.split(":", 1)[1].strip()
                        break
            out[(int(d), int(t))] = {
                "pid": int(d), "tid": int(t),
                "proc": comm, "thread": read_text(base + "/comm", 64) or "",
                "ticks": ticks, "last_cpu": processor, "cpus_allowed": cpus,
            }
    return out


def thread_delta(a, b, secs, topn=25):
    rows = []
    for key, tb in b.items():
        ta = a.get(key)
        if ta is None:
            continue
        d = tb["ticks"] - ta["ticks"]
        if d < 0:
            continue
        rows.append({"pid": tb["pid"], "tid": tb["tid"], "proc": tb["proc"],
                     "thread": tb["thread"],
                     "cpu_pct": round(100.0 * d / CLK_TCK / secs, 1),
                     "last_cpu": tb["last_cpu"], "cpus_allowed": tb["cpus_allowed"]})
    rows.sort(key=lambda r: r["cpu_pct"], reverse=True)
    return rows[:topn]


def proc_delta(a, b, secs, topn=15):
    rows = []
    for pid, pb in b.items():
        pa = a.get(pid)
        if pa is None:
            continue                        # 窗口中途才出現的行程：算不出正確的 CPU%
        dticks = pb["ticks"] - pa["ticks"]
        if dticks < 0:
            continue
        rows.append({
            "pid": pid,
            "comm": pb["comm"],
            "cpu_pct": round(100.0 * dticks / CLK_TCK / secs, 1),
            "rss_mb": pb["rss_mb"],
            "cpus_allowed": pb["cpus_allowed"],
            "cmd": pb["cmd"],
        })
    rows.sort(key=lambda r: (r["cpu_pct"], r["rss_mb"] or 0), reverse=True)
    return rows[:topn]


# ----------------------------------------------------------------- GPU / NPU
def gpu_npu_paths():
    """把可能承載 GPU/NPU 負載的節點全找出來。

    路徑因板而異（RK3588 的 Mali 走 devfreq、RKNPU 在 debugfs；
    Orin 的 GPU 是 ga10b），所以用 glob 掃 + 明列候選，不寫死單一路徑。
    """
    cands = []
    for p in sorted(glob.glob("/sys/class/devfreq/*")):
        name = os.path.basename(os.path.realpath(p))
        cands.append(("devfreq:%s:load" % name, os.path.join(p, "load")))
        cands.append(("devfreq:%s:cur_freq" % name, os.path.join(p, "cur_freq")))
    fixed = [
        ("rknpu:load", "/sys/kernel/debug/rknpu/load"),
        ("mali:utilisation", "/sys/devices/platform/fb000000.gpu/utilisation"),
        ("orin_gpu:load", "/sys/devices/platform/gpu.0/load"),
        ("orin_gpu:load2", "/sys/devices/gpu.0/load"),
    ]
    for tag, path in fixed:
        if os.path.exists(path):
            cands.append((tag, path))
    for p in sorted(glob.glob("/sys/devices/platform/*gpu*/load")):
        cands.append(("gpu:%s" % p.split("/")[-2], p))
    # 去重（同一個檔可能被兩種方式撈到）
    seen, out = set(), []
    for tag, path in cands:
        if path in seen:
            continue
        seen.add(path)
        out.append((tag, path))
    return out


def thermal_zones():
    out = []
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        out.append((read_text(os.path.join(p, "type"), 64) or os.path.basename(p),
                    os.path.join(p, "temp")))
    return out


# ----------------------------------------------------------------- 網路
def net_snapshot():
    out = {}
    txt = read_text("/proc/net/dev", 16384) or ""
    for line in txt.splitlines()[2:]:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        f = rest.split()
        if len(f) >= 9:
            out[name.strip()] = (int(f[0]), int(f[8]))     # rx_bytes, tx_bytes
    return out


# ----------------------------------------------------------------- 主流程
def main():
    if len(sys.argv) < 3:
        print("用法: python3 recon3_sample.py <秒數> <標籤>", file=sys.stderr)
        return 2
    secs = float(sys.argv[1])
    label = sys.argv[2]

    host = os.uname().nodename
    series_paths = gpu_npu_paths()
    therm_paths = thermal_zones()

    # tegrastats（Orin NX 專屬）。它是唯讀的監看工具，不改任何設定。
    teg_proc, teg_lines, teg_note = None, [], None
    teg_bin = shutil.which("tegrastats")
    if teg_bin:
        try:
            teg_proc = subprocess.Popen(
                [teg_bin, "--interval", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            teg_note = "tegrastats 啟動失敗: %r" % (e,)
    else:
        teg_note = "這塊板沒有 tegrastats（預期：只有 Orin NX 有）"

    print("[recon3] %s: 採樣 %s 秒（標籤 %s）…" % (host, secs, label), file=sys.stderr)

    t0 = time.time()
    cpu_a, proc_a, net_a = cpu_snapshot(), proc_snapshot(), net_snapshot()
    thr_a = threads_info()
    mem_a = meminfo()

    per_sec_cpu, series = [], {tag: [] for tag, _ in series_paths}
    therm_series = {tag: [] for tag, _ in therm_paths}
    prev = cpu_a
    while time.time() - t0 < secs:
        time.sleep(1.0)
        cur = cpu_snapshot()
        per_sec_cpu.append(cpu_pct(prev, cur))
        prev = cur
        for tag, path in series_paths:
            v = read_text(path, 128)
            if v is not None:
                series[tag].append(v)
        for tag, path in therm_paths:
            v = read_int(path)
            if v is not None:
                therm_series[tag].append(v / 1000.0 if abs(v) > 1000 else float(v))

    elapsed = time.time() - t0
    cpu_b, proc_b, net_b = cpu_snapshot(), proc_snapshot(), net_snapshot()
    thr_b = threads_info()
    mem_b = meminfo()

    if teg_proc is not None:
        try:
            teg_proc.terminate()
            out, _ = teg_proc.communicate(timeout=5)
            teg_lines = (out or "").splitlines()[:120]
        except Exception as e:
            teg_note = "tegrastats 收尾失敗: %r" % (e,)

    window = cpu_pct(cpu_a, cpu_b)
    ncpu = len([k for k in window if k != "cpu"])
    per_cpu = [window.get("cpu%d" % i) for i in range(ncpu)]
    per_cpu_max = []
    for i in range(ncpu):
        vals = [s.get("cpu%d" % i) for s in per_sec_cpu if s.get("cpu%d" % i) is not None]
        per_cpu_max.append(max(vals) if vals else None)

    # 數字系列（load / 頻率）算 min/mean/max；非數字（例如 rknpu 的 "Core0: 12%"）留原樣
    series_stat = {}
    for tag, vals in series.items():
        if not vals:
            continue
        # RKNPU 的 debugfs load 是三核一行：
        #   "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,"
        # → 拆成三條數字系列。這是唯一可信的 NPU 使用率來源
        #   （devfreq 的 load 節點固定 100 是假的，2026-09-22 實測）。
        if "rknpu" in tag:
            cores = {}
            for v in vals:
                for idx, pct in re.findall(r"Core(\d+):\s*(\d+)\s*%", v):
                    cores.setdefault(int(idx), []).append(float(pct))
            if cores:
                for idx in sorted(cores):
                    c = cores[idx]
                    series_stat["rknpu:core%d:load_pct" % idx] = {
                        "min": min(c), "mean": round(sum(c) / len(c), 1),
                        "max": max(c), "n": len(c)}
                continue
        nums = []
        for v in vals:
            try:
                # RK3588 的 devfreq load 格式是 "<百分比>@<頻率>Hz"（例：'0@300000000Hz'）
                nums.append(float(v.split()[0].split("@")[0]))
            except Exception:
                pass
        if len(nums) == len(vals) and nums:
            series_stat[tag] = {"min": min(nums),
                                "mean": round(sum(nums) / len(nums), 1),
                                "max": max(nums), "n": len(nums)}
        else:
            series_stat[tag] = {"raw_first": vals[0], "raw_last": vals[-1], "n": len(vals)}

    therm_stat = {t: {"mean_c": round(sum(v) / len(v), 1), "max_c": max(v)}
                  for t, v in therm_series.items() if v}

    net_delta = {}
    for name, (rx_b, tx_b) in net_b.items():
        if name not in net_a:
            continue
        rx_a, tx_a = net_a[name]
        rx_mb = (rx_b - rx_a) / 1e6 / elapsed
        tx_mb = (tx_b - tx_a) / 1e6 / elapsed
        if rx_mb > 0.01 or tx_mb > 0.01:
            net_delta[name] = {"rx_MBps": round(rx_mb, 3), "tx_MBps": round(tx_mb, 3)}

    result = {
        "label": label,
        "host": host,
        "secs": round(elapsed, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uid": os.getuid(),
        "ncpu": ncpu,
        "cpu_overall_pct": window.get("cpu"),
        "per_cpu_pct": per_cpu,
        "per_cpu_max_pct": per_cpu_max,
        "loadavg": read_text("/proc/loadavg", 128),
        "isolated_cpus": read_text("/sys/devices/system/cpu/isolated", 128),
        "mem_start": mem_a,
        "mem_end": mem_b,
        "top_proc": proc_delta(proc_a, proc_b, elapsed),
        "top_thread": thread_delta(thr_a, thr_b, elapsed),
        "gpu_npu": series_stat,
        "thermal_c": therm_stat,
        "net_MBps": net_delta,
        "tegrastats_note": teg_note,
        "tegrastats": teg_lines,
        "psi_cpu": read_text("/proc/pressure/cpu", 256),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1)
    print()

    # ---------------- 人讀摘要（stderr，現場直接看） ----------------
    e = sys.stderr
    print("", file=e)
    print("---- %s / %s：%.0f 秒 ----" % (host, label, elapsed), file=e)
    print("整體 CPU %.1f%%   %d 核   loadavg %s"
          % (result["cpu_overall_pct"] or -1, ncpu, (result["loadavg"] or "?").split(" /")[0]), file=e)
    for i in range(ncpu):
        avg, mx = per_cpu[i], per_cpu_max[i]
        bar = "#" * int(round((avg or 0) / 5.0))
        print("  cpu%-2d 平均 %5.1f%%  峰 %5.1f%%  %s"
              % (i, avg if avg is not None else -1, mx if mx is not None else -1, bar), file=e)
    if result["isolated_cpus"]:
        print("  核心隔離 isolcpus = %s" % result["isolated_cpus"], file=e)
    m = mem_b
    print("記憶體 %.0f / %.0f MB（%.0f%%）  swap 用 %.0f MB"
          % (m["used_mb"] or 0, m["total_mb"] or 0, m["used_pct"] or 0, m["swap_used_mb"] or 0), file=e)
    print("前 8 名行程（CPU% / RSS MB / 可用核）:", file=e)
    for r in result["top_proc"][:8]:
        print("  %6.1f%%  %8s MB  cpus=%-8s  %s"
              % (r["cpu_pct"], r["rss_mb"], r["cpus_allowed"] or "?", r["comm"]), file=e)
    iso = result["isolated_cpus"]
    if result["top_thread"]:
        print("關鍵行程的執行緒（CPU%% / 上次在哪顆核 / 可用核）:".replace("%%", "%"), file=e)
        for r in result["top_thread"][:10]:
            print("  %6.1f%%  cpu%-2d  cpus=%-8s  %s / %s"
                  % (r["cpu_pct"], r["last_cpu"], r["cpus_allowed"] or "?",
                     r["proc"], r["thread"]), file=e)
        if iso:
            on_iso = [r for r in result["top_thread"]
                      if str(r["last_cpu"]) in iso.replace("-", ",").split(",")]
            print("  → 隔離核 %s 上抓到的執行緒：%s"
                  % (iso, "、".join("%s/%s" % (r["proc"], r["thread"]) for r in on_iso)
                     or "這次取樣沒抓到（只代表取樣瞬間，不代表沒有）"), file=e)
    if series_stat:
        print("GPU / NPU / devfreq:", file=e)
        for tag, s in sorted(series_stat.items()):
            if "mean" in s:
                # Jetson 的 gpu load 是千分比（0–1000），不除 10 會看成 200%
                if "gpu" in tag and "cur_freq" not in tag and s["max"] > 100:
                    print("  %-34s 平均 %.1f%%  峰 %.1f%%（原始千分比 %s/%s）"
                          % (tag, s["mean"] / 10.0, s["max"] / 10.0,
                             s["mean"], s["max"]), file=e)
                else:
                    print("  %-34s 平均 %s  峰 %s" % (tag, s["mean"], s["max"]), file=e)
            else:
                print("  %-34s %s → %s" % (tag, s["raw_first"], s["raw_last"]), file=e)
    if therm_stat:
        hot = sorted(therm_stat.items(), key=lambda kv: kv[1]["max_c"], reverse=True)[:5]
        print("溫度（最高 5 區）: " + "  ".join(
            "%s %.1f°C" % (k, v["max_c"]) for k, v in hot), file=e)
    if net_delta:
        print("網路流量: " + "  ".join(
            "%s rx %.2f MB/s" % (k, v["rx_MBps"]) for k, v in net_delta.items()), file=e)
    if teg_lines:
        print("tegrastats 首行: %s" % teg_lines[0][:160], file=e)
    elif teg_note:
        print("tegrastats: %s" % teg_note, file=e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
