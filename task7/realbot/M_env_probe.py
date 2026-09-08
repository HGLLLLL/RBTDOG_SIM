#!/usr/bin/env python3
"""M_env_probe —— 狗上電腦（RK3588）探測：numpy 有沒有、policy 前向幾 ms、CPU 能用幾核。

零風險：不開任何 /dev/shm、不碰馬達、不需要 root。狗趴著即可。

為什麼要有這支：CPG-RL 訓練完的網路要在狗上每 20 ms 出一次指令。
兩份自家文件對「狗上有沒有 numpy」說法相反，從沒查過；純 Python 前向在本機要 6 ms，
RK3588 若慢 4 倍就是 24 ms —— 那要在**訓練前**知道，不是訓練完才發現要重寫推論。

用法（狗上）：python3 M_env_probe.py
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time

SIZES = (68, 256, 256, 128, 12)


def make_weights(sizes=SIZES, seed=0):
    rng = random.Random(seed)
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        s = 1.0 / math.sqrt(a)
        W = [[rng.uniform(-s, s) for _ in range(a)] for _ in range(b)]
        bias = [rng.uniform(-s, s) for _ in range(b)]
        layers.append((W, bias))
    return layers


def forward_py(layers, x):
    h = list(x)
    n = len(layers)
    for li, (W, b) in enumerate(layers):
        out = []
        last = (li == n - 1)
        for row, bb in zip(W, b):
            s = bb
            for w, v in zip(row, h):
                s += w * v
            out.append(s if last else (s if s > 0 else 0.0))   # ReLU，最後一層線性
        h = out
    return h


def forward_np(layers, x):
    import numpy as np
    h = np.asarray(x, float)
    n = len(layers)
    for li, (W, b) in enumerate(layers):
        h = np.asarray(W) @ h + np.asarray(b)
        if li != n - 1:
            h = np.maximum(h, 0.0)
    return h


def bench(fn, n=100):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts), max(ts)


def main() -> int:
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        info["affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity"] = None
    for mod in ("numpy", "onnxruntime", "torch"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    layers = make_weights()
    x = [0.0] * SIZES[0]
    forward_py(layers, x)
    info["mlp_py_ms"] = bench(lambda: forward_py(layers, x), 50)
    if info["numpy"]:
        import numpy as np
        Wn = [(np.asarray(W), np.asarray(b)) for W, b in layers]

        def f():
            h = np.asarray(x)
            for li, (W, b) in enumerate(Wn):
                h = W @ h + b
                if li != len(Wn) - 1:
                    h = np.maximum(h, 0.0)
            return h
        f()
        info["mlp_np_ms"] = bench(f, 200)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"\n純 Python 前向 {info['mlp_py_ms'][0]:.2f} ms（最大 {info['mlp_py_ms'][1]:.2f}）"
          f"；50 Hz 預算 20 ms")
    if info.get("mlp_np_ms"):
        print(f"numpy 前向 {info['mlp_np_ms'][0]:.3f} ms（最大 {info['mlp_np_ms'][1]:.3f}）")
    else:
        print("numpy：無 → 推論只能純 Python 或自帶 wheel")
    d = os.path.expanduser("~/m_logs")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, time.strftime("ENV_%Y%m%d_%H%M%S.json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"→ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
