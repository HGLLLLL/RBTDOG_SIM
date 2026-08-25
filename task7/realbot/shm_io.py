#!/usr/bin/env python3
"""D1 Max 的 /dev/shm 讀寫底層 —— M0/M1/M2 共用。

結構是 2026-08-25 從實機快照逆向出來的，完整佐證見 `shm_decode.py` 檔尾的 §驗證。
本檔只放「跑在狗上」需要的最小實作：純標準函式庫，不依賴 numpy。

三塊共享記憶體都是 Boost.Interprocess managed segment（1 MiB），內含 `SharedVector`：

    joint_cmd    base=752  stride=112   name[64] + position, velocity, effort, kp, kd   (f64)
    joint_state  base=752  stride=120   name[64] + position, velocity, effort, temp, voltage (f64) + error (u64)
    imu_central  數值自 824 起：acc[3], gyro[3], quat[4]（xyzw）

⚠️ 只覆寫已知位移的數值欄位，**絕不碰標頭與名稱**，避免破壞容器結構。
"""
from __future__ import annotations

import mmap
import os
import struct

SHM_DIR = "/dev/shm"
SIZE = 1024 * 1024

BASE = 752
NAME_OFF, NAME_LEN = 8, 64
DATA_OFF = 72

CMD_STRIDE = 112
STATE_STRIDE = 120

# joint_cmd 的 5 個欄位（順序即記憶體順序），對應 ros2_control 的 command interface
CMD_FIELDS = ("position", "velocity", "effort", "kp", "kd")

# 16 顆關節在 SharedVector 裡的順序。
# 與 /opt/robot/install/robot_hal/share/robot_hal/config/zsm/robot_hal.yaml
# 的 joint_shm_controller.joints 逐項一致。
JOINTS = [
    "fl1_hip_roll", "fl2_hip_pitch", "fl3_knee_pitch", "fl4_foot",
    "fr1_hip_roll", "fr2_hip_pitch", "fr3_knee_pitch", "fr4_foot",
    "bl1_hip_roll", "bl2_hip_pitch", "bl3_knee_pitch", "bl4_foot",
    "br1_hip_roll", "br2_hip_pitch", "br3_knee_pitch", "br4_foot",
]
WHEELS = ["fl4_foot", "fr4_foot", "bl4_foot", "br4_foot"]

_F8 = struct.Struct("<d")


class Shm:
    """一塊共享記憶體的 mmap 包裝。write=True 需要 root。"""

    def __init__(self, name: str, write: bool = False):
        self.name = name
        self.path = os.path.join(SHM_DIR, name)
        flags = os.O_RDWR if write else os.O_RDONLY
        self.fd = os.open(self.path, flags)          # 不加 O_CREAT：讀不到就是讀不到
        prot = mmap.PROT_READ | (mmap.PROT_WRITE if write else 0)
        self.mm = mmap.mmap(self.fd, SIZE, mmap.MAP_SHARED, prot)
        self.writable = write

    def close(self):
        try:
            self.mm.close()
        finally:
            os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---------------------------------------------------------------- 讀
    def _name_at(self, off: int) -> str:
        raw = self.mm[off + NAME_OFF: off + NAME_OFF + NAME_LEN]
        return raw.split(b"\0", 1)[0].decode(errors="replace")

    def read_records(self, stride: int, nfields: int) -> list[dict]:
        out = []
        for i in range(len(JOINTS)):
            o = BASE + i * stride
            rec = {"name": self._name_at(o)}
            for k in range(nfields):
                p = o + DATA_OFF + k * 8
                rec[k] = _F8.unpack_from(self.mm, p)[0]
            out.append(rec)
        return out

    def verify_layout(self, stride: int) -> None:
        """名稱順序對不上就直接拒絕往下走 —— 結構若改版，後面每個欄位都會寫錯關節。"""
        got = [self._name_at(BASE + i * stride) for i in range(len(JOINTS))]
        if got != JOINTS:
            raise SystemExit(
                "❌ 關節順序與預期不符，結構可能改版了，拒絕繼續。\n"
                f"   預期: {JOINTS}\n   實際: {got}")

    # ---------------------------------------------------------------- 寫（僅 joint_cmd）
    def _cmd_field_off(self, idx: int, field: str) -> int:
        return BASE + idx * CMD_STRIDE + DATA_OFF + CMD_FIELDS.index(field) * 8

    def write_cmd(self, idx: int, *, position=None, velocity=None,
                  effort=None, kp=None, kd=None) -> None:
        """寫單一關節的指令欄位。

        ⚠️ 寫入順序刻意固定為「先目標值、後增益」：
           joint_shm_controller 以 1 kHz 讀取，我們的 5 個 8-byte 寫入不是原子的。
           若先寫增益、後寫目標，撕裂讀取可能拿到「新增益 + 舊目標」→ 意外出力。
           反過來則最壞只會拿到「舊增益 + 新目標」，而舊增益是我們上一輪確認過的值。
        """
        if not self.writable:
            raise RuntimeError("這塊 shm 是唯讀開啟的")
        for f, v in (("position", position), ("velocity", velocity),
                     ("effort", effort), ("kp", kp), ("kd", kd)):
            if v is not None:
                _F8.pack_into(self.mm, self._cmd_field_off(idx, f), float(v))

    def zero_gains(self, idx: int) -> None:
        """把單一關節壓成完全不出力：kp = kd = effort = 0。

        歸零時**先寫增益**（與 write_cmd 相反）：這裡的目的就是讓出力盡快變 0，
        撕裂讀取拿到「新的零增益 + 舊目標」也仍然是零出力。
        """
        for f in ("kp", "kd", "effort", "velocity"):
            _F8.pack_into(self.mm, self._cmd_field_off(idx, f), 0.0)


def read_joint_state() -> list[dict]:
    """回傳 16 筆 {name, position, velocity, effort, temp, voltage}。唯讀，不需 root。"""
    with Shm("joint_state") as s:
        recs = s.read_records(STATE_STRIDE, 5)
    keys = ("position", "velocity", "effort", "temp_C", "voltage_V")
    return [{"name": r["name"], **{k: r[i] for i, k in enumerate(keys)}} for r in recs]


def read_joint_cmd() -> list[dict]:
    """回傳 16 筆 {name, position, velocity, effort, kp, kd}。唯讀，不需 root。"""
    with Shm("joint_cmd") as s:
        recs = s.read_records(CMD_STRIDE, 5)
    return [{"name": r["name"], **{k: r[i] for i, k in enumerate(CMD_FIELDS)}} for r in recs]


def idx_of(joint: str) -> int:
    if joint not in JOINTS:
        raise SystemExit(f"❌ 不認得的關節名 {joint!r}\n   可用：{', '.join(JOINTS)}")
    return JOINTS.index(joint)


# ---------------------------------------------------------------- 自動存檔
class _Tee:
    """同時寫到終端機與檔案。每行都 flush —— 程式若中途死掉，已印出的仍在檔案裡。"""

    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, data):
        self._s.write(data)
        self._s.flush()
        self._f.write(data)
        self._f.flush()

    def flush(self):
        self._s.flush()
        self._f.flush()

    def isatty(self):
        return self._s.isatty()


def start_log(prefix: str, dirname: str = "~/m_logs") -> str:
    """把 stdout/stderr 同時導向 <dirname>/<prefix>_<時間戳>.log，回傳檔案路徑。

    M0/M1/M2 的結果一定要能帶回去複核，光印在螢幕上不夠 ——
    現場沒有網路，事後要靠這些檔案判讀。
    """
    import datetime
    import sys

    # ⚠️ M1/M2 是用 sudo 跑的，直接 expanduser("~") 會變成 /root/m_logs，
    #    之後用 robot 帳號 scp 拿不到。改成解析「原本呼叫 sudo 的那個使用者」的家目錄，
    #    並把檔案 chown 回去，log 才帶得走。
    if dirname.startswith("~") and os.geteuid() == 0 and os.getenv("SUDO_USER"):
        import pwd
        pw = pwd.getpwnam(os.environ["SUDO_USER"])
        d = os.path.join(pw.pw_dir, dirname.lstrip("~/"))
        owner = (pw.pw_uid, pw.pw_gid)
    else:
        d = os.path.expanduser(dirname)
        owner = None

    os.makedirs(d, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, f"{prefix}_{ts}.log")
    fh = open(path, "w", encoding="utf-8")
    if owner:
        for p in (d, path):
            try:
                os.chown(p, *owner)
            except OSError:
                pass
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return path
