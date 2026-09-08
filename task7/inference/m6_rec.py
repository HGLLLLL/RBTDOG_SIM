"""`M6_load_probe.py --record` 錄檔（schema `m6_record/1`）的載入器。

三支離線工具（real_obs / imu_check / obs_compare）共用。存檔時 `_fold()` 會把
整段不變的欄位存成純量（例如 kp 全程 60.0），這裡一律展開成 `(N,)` 陣列，
讓下游不用到處判斷「這是純量還是陣列」。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realbot"))
import shm_io  # noqa: E402

SCHEMA = "m6_record/1"
FIELDS = ("q", "v", "tau", "des", "ff", "kp", "kd")


@dataclass
class Rec:
    path: str
    label: str
    note: str
    t: np.ndarray            # (N,) 秒，從 0 起
    hz: float                # 實際達到的取樣率
    j: dict                  # name -> {field: (N,) float64}，馬達座標系（原樣）
    imu: np.ndarray          # (N, 10)：acc[3] gyro[3] quat[4]（quat 原始存放順序）

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def acc(self) -> np.ndarray:
        return self.imu[:, 0:3]

    @property
    def gyro(self) -> np.ndarray:
        return self.imu[:, 3:6]

    @property
    def quat_raw(self) -> np.ndarray:
        return self.imu[:, 6:10]


def from_dict(d: dict, path: str = "") -> Rec:
    if d.get("schema") != SCHEMA:
        raise ValueError(f"不是 M6 錄檔（schema={d.get('schema')!r}，需要 {SCHEMA}）")
    t = np.asarray(d["t"], dtype=float)
    n = len(t)
    missing = [nm for nm in shm_io.JOINTS if nm not in d["joints"]]
    if missing:
        raise ValueError(f"錄檔缺關節 {missing}")
    j = {}
    for nm in shm_io.JOINTS:
        j[nm] = {}
        for f in FIELDS:
            v = d["joints"][nm][f]
            arr = (np.full(n, float(v)) if not isinstance(v, list)
                   else np.asarray(v, dtype=float))
            if arr.shape != (n,):
                raise ValueError(f"{nm}.{f} 長度 {arr.shape} ≠ N={n}")
            j[nm][f] = arr
    imu = np.asarray(d["imu"], dtype=float)
    if imu.shape != (n, 10):
        raise ValueError(f"imu 形狀 {imu.shape} ≠ ({n}, 10)")
    hz = float(d.get("hz_actual") or (n / max(t[-1], 1e-9)))
    return Rec(path=path, label=d.get("label", ""), note=d.get("note", ""),
               t=t, hz=hz, j=j, imu=imu)


def load(path) -> Rec:
    p = Path(path)
    return from_dict(json.loads(p.read_text(encoding="utf-8")), str(p))
