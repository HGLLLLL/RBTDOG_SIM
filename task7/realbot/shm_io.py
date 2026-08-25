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
TICK_OFF = 0        # ★ 記錄開頭的 u64 是「這一幀的時戳／心跳」，見下
NAME_OFF, NAME_LEN = 8, 64
DATA_OFF = 72

# ★★ TICK 是 2026-08-25 M1 第一次寫入失敗後才搞清楚的欄位。
#
# 證據（recon2 的兩份快照，間隔 2 秒）：
#     joint_cmd   base+0: 1184236 → 1186664   (+2428/≈2.4s ≈ 1 kHz)
#     joint_state base+0: 1186768 → 1188865   (+2097，同一時鐘、晚約 2 秒取樣)
#   且 **16 筆記錄的值完全相同** → 是整幀共用的時戳，不是每關節的欄位。
#   值 0x1211EC 大於 1 MiB 的區段大小，所以不是段內指標。
#
# 為什麼重要：`joint_shm_controller` 用 `joint_cmd_timeout: 500 ms` 判斷指令新不新。
# 我們 SIGSTOP 凍結 mc_ctrl 之後這個心跳就停了 → controller 判定過期 →
# **把整個指令區清成 0**。第一次 M1 讀回全 0（0/16 相符）就是這個原因，
# 不是我們的 5 個浮點數沒寫進去。
#
# 作法：每一輪把 joint_state 當下的 tick 抄進 joint_cmd（同一個時鐘，保證是新的），
# 而且**先寫 payload、最後寫 tick** —— tick 等同「這幀備妥了」的旗標。
#
# ✅ 2026-08-25 實機驗證通過：加上心跳後 M1 讀回 16/16 相符、最大力矩 0.058 N·m。
#    實測速率 +3000 / 3.0 秒 = **恰好 1000/s**，與 controller_manager 的
#    update_rate: 1000 Hz 一致（先前從快照估的 1.2 kHz 偏高，因為誤以為 scp 間隔正好 2 秒）。

_U8 = struct.Struct("<Q")

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

    def read_tick(self, stride: int) -> int:
        """讀整幀共用的時戳（16 筆都一樣，取第 0 筆）。"""
        return _U8.unpack_from(self.mm, BASE + TICK_OFF)[0]

    def write_tick(self, value: int) -> None:
        """把時戳寫進全部 16 筆記錄（joint_cmd 專用）。

        ⚠️ 一定要在 payload 都寫完之後才呼叫 —— 它是「這幀備妥了」的旗標。
        """
        if not self.writable:
            raise RuntimeError("這塊 shm 是唯讀開啟的")
        for i in range(len(JOINTS)):
            _U8.pack_into(self.mm, BASE + i * CMD_STRIDE + TICK_OFF, int(value))

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
