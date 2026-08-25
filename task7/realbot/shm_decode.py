#!/usr/bin/env python3
"""D1 Max 的 /dev/shm 解碼器 —— joint_state / joint_cmd / imu_central。

結構是 2026-08-25 從實機快照逆向出來的（`recon2_d1max.sh` 拉回的 .bin），
不是官方文件。已驗證的部分見下方 §驗證；沒驗到的都標了 ⚠️。

三塊共享記憶體都是 **Boost.Interprocess 的 managed segment**，各 1 MiB，
裡面一個名為 `SharedVector` 的容器，元素是「名稱 + 數值欄位」的定長記錄。
`motor_platform_type: 7 # 5 mujoco # 6 shm 7 shmContainer` 這行註解
（實機 `/opt/export/config/zg_wheels-user-parameters.yaml`）佐證了 "shmContainer" 這個形式。

用法：
    python3 shm_decode.py joint_state_t0.bin
    python3 shm_decode.py joint_cmd_t0.bin
    python3 shm_decode.py imu_central_t0.bin
    python3 shm_decode.py joint_state_t0.bin joint_state_t1.bin   # 兩份做差分

    # 直接讀實機（唯讀，非 sudo）：
    scp robot@192.168.234.1:/dev/shm/joint_state . && python3 shm_decode.py joint_state
"""
from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------- 結構常數
#
# 記錄基底 752、名稱在 +8（64 bytes，零結尾），數值欄位從 +72 起，全部 little-endian float64。
# 基底與 stride 是從「名稱字串出現的位置」反推的：
#   joint_state 名稱在 760, 880, 1000, ...  → 間隔 120，基底 = 760 − 8 = 752
#   joint_cmd   名稱在 760, 872, 984, ...   → 間隔 112，基底同上
BASE = 752
TICK_OFF = 0        # ★ 整幀共用的時戳／心跳（16 筆值相同，~1.2 kHz 遞增），2026-08-25 第三趟才確認
NAME_OFF, NAME_LEN = 8, 64
DATA_OFF = 72

LAYOUT = {
    # stride, 欄位名稱（float64，依序），最後是否有一個 uint64
    "joint_state": (120, ["position", "velocity", "effort", "temp_C", "voltage_V"], "error"),
    "joint_cmd":   (112, ["position", "velocity", "effort", "kp", "kd"], None),
}

# 16 顆關節在 SharedVector 裡的順序（實機讀出來的，**不是** 官方動作設定檔那組 FR/FL/RR/RL）
JOINT_ORDER = [
    "fl1_hip_roll", "fl2_hip_pitch", "fl3_knee_pitch", "fl4_foot",
    "fr1_hip_roll", "fr2_hip_pitch", "fr3_knee_pitch", "fr4_foot",
    "bl1_hip_roll", "bl2_hip_pitch", "bl3_knee_pitch", "bl4_foot",
    "br1_hip_roll", "br2_hip_pitch", "br3_knee_pitch", "br4_foot",
]

# imu_central：單一記錄，名稱 "imu_central" 在 760，數值從 824 起
IMU_OFF = 824
IMU_FIELDS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
              "quat_x", "quat_y", "quat_z", "quat_w"]


def _f64(buf: np.ndarray, off: int, n: int) -> np.ndarray:
    return np.frombuffer(buf[off:off + 8 * n].tobytes(), dtype="<f8")


def _name(buf: np.ndarray, off: int) -> str:
    return buf[off:off + NAME_LEN].tobytes().split(b"\0")[0].decode(errors="replace")


def decode_joints(buf: np.ndarray, kind: str) -> list[dict]:
    """解 joint_state 或 joint_cmd，回傳 16 筆 dict。"""
    stride, fields, tail_u64 = LAYOUT[kind]
    out = []
    for i in range(16):
        o = BASE + i * stride
        rec = {"name": _name(buf, o + NAME_OFF)}
        vals = _f64(buf, o + DATA_OFF, len(fields))
        rec.update(dict(zip(fields, vals)))
        if tail_u64:
            end = o + stride
            rec[tail_u64] = int(np.frombuffer(buf[end - 8:end].tobytes(), dtype="<u8")[0])
        out.append(rec)
    return out


def decode_imu(buf: np.ndarray) -> dict:
    vals = _f64(buf, IMU_OFF, len(IMU_FIELDS))
    d = dict(zip(IMU_FIELDS, vals))
    q = np.array([d["quat_x"], d["quat_y"], d["quat_z"], d["quat_w"]])
    d["quat_norm"] = float(np.linalg.norm(q))
    x, y, z, w = q
    d["roll_deg"] = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    d["pitch_deg"] = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    d["yaw_deg"] = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    d["acc_norm"] = float(np.linalg.norm([d["acc_x"], d["acc_y"], d["acc_z"]]))
    return d


def kind_of(path: str) -> str:
    for k in ("joint_state", "joint_cmd", "imu_central"):
        if k in path:
            return k
    raise SystemExit(f"檔名認不出種類（要含 joint_state / joint_cmd / imu_central）：{path}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    path = argv[0]
    kind = kind_of(path)
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size != 1024 * 1024:
        print(f"⚠️ 大小 {buf.size} 不是預期的 1 MiB，結構可能不同")

    if kind == "imu_central":
        d = decode_imu(buf)
        print(f"=== {path} ===")
        print(f"  加速度 (m/s²)  x={d['acc_x']:+9.5f}  y={d['acc_y']:+9.5f}  z={d['acc_z']:+9.5f}"
              f"   |a|={d['acc_norm']:.4f}")
        print(f"  角速度 (rad/s) x={d['gyro_x']:+9.5f}  y={d['gyro_y']:+9.5f}  z={d['gyro_z']:+9.5f}")
        print(f"  四元數 xyzw    {d['quat_x']:+.6f} {d['quat_y']:+.6f} "
              f"{d['quat_z']:+.6f} {d['quat_w']:+.6f}   |q|={d['quat_norm']:.6f}")
        print(f"  → roll={d['roll_deg']:+7.2f}°  pitch={d['pitch_deg']:+7.2f}°  yaw={d['yaw_deg']:+7.2f}°")
        print()
        print("  ⚠️ 四元數順序 xyzw 是推定的（見檔尾 §驗證）。要確認就把狗平放地面，")
        print("     roll/pitch 應該 ≈0；用 wxyz 解會得到荒謬的角度。")
    else:
        recs = decode_joints(buf, kind)
        fields = LAYOUT[kind][1]
        tail = LAYOUT[kind][2]
        hdr = f"{'關節':16s}" + "".join(f"{f:>12s}" for f in fields)
        if tail:
            hdr += f"{tail:>8s}"
        print(f"=== {path} ===")
        print(hdr)
        for r in recs:
            line = f"{r['name']:16s}" + "".join(f"{r[f]:12.5f}" for f in fields)
            if tail:
                line += f"{r[tail]:8d}"
            print(line)
        # 名稱一致性檢查：順序跑掉的話後面全部會對錯關節
        got = [r["name"] for r in recs]
        if got != JOINT_ORDER:
            print()
            print("⚠️ 關節順序與預期不符！結構可能改版了，不要直接採信上面的數值。")
            print(f"   預期: {JOINT_ORDER}")
            print(f"   實際: {got}")

    if len(argv) > 1:
        b2 = np.fromfile(argv[1], dtype=np.uint8)
        diff = int((buf != b2).sum())
        print()
        print(f"=== 與 {argv[1]} 的差分 ===")
        print(f"  變動位元組 {diff} / {buf.size} ({100*diff/buf.size:.4f}%)"
              f"  → {'活的串流' if diff else '完全相同（靜態或取樣間隔太短）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# ==============================================================================
# §驗證（2026-08-25，資料來自 recon2_20260825_112721/shm/）
# ==============================================================================
#
# ✅ 已驗證
#   - 16 筆記錄的名稱完整且順序為 fl1..4, fr1..4, bl1..4, br1..4
#   - joint_cmd 的 position 欄位讀出來**恰好等於**實機
#     /opt/export/config/zg_wheels-user-parameters.yaml 的 abad/hip/knee_offset：
#         fl1 −0.523  fr1 +0.523  bl1 +0.523  br1 −0.523   (config abad_offset)
#         fl2 +2.443  fr2 −2.443  bl2 −2.443  br2 +2.443   (config hip_offset)
#         fl3 +2.803  fr3 −2.803  bl3 −2.803  br3 +2.803   (config knee_offset)
#     四組全中 → 欄位位移正確，而且 joint_cmd 是**馬達座標系**
#   - joint_cmd 的 5 個欄位對應 ros2_control 實機列出的 command interface：
#     每個關節都是 position / velocity / effort / kp / kd（`ros2 control
#     list_hardware_interfaces` 實測，16×5 = 80 個，全被 joint_shm_controller claimed）
#   - imu_central 加速度 z ≈ 9.89 m/s²、x/y ≈ 0 → 重力在 +Z、機身水平；
#     以 xyzw 解四元數得 roll −0.12° / pitch −1.61°（與加速度自洽），
#     以 wxyz 解會得到 roll −30.6°（與加速度矛盾）→ 支持 xyzw
#   - 三塊 shm 兩次取樣的 md5 都不同 → 都是活的串流
#
# ⚠️ 未驗證 / 待確認
#   - `temp_C` / `voltage_V` 只是「25.0 / 53~54」看起來像溫度與電壓的合理推測，
#     雖然 /joint_shm_controller/joint_sensor 的 msg 定義確實有 temp 與 voltage
#   - `error` 欄位所有關節都是 1，意義不明（1 = OK 還是 1 = 某個 flag？）
#     2026-08-25 補：故障排除時讀 /joint_shm_controller/joint_sensor，健康狀態下
#     16 顆同樣是 error=1 且 error_msg 全空 → **1 = 正常**，這點已確認。
#   - 「馬達角 = side_sign × 控制器角 + offset」這個換算式仍未實證。
#     取樣當下馬達是洩力的（joint_cmd 的 kp/kd/effort 全 0），四肢應該是靠在機構限位上，
#     用該式反推得到 abad 0.571 / hip 1.146 / knee −2.738 rad，接近但不等於任何一組
#     文件姿態 → 與「洩力癱在限位上」自洽，但**不足以當作證明**。
#     要確認：讓狗站好（已知姿態）再取一次快照比對。
#   - 更新頻率沒量（只知道兩次取樣之間有變）
#   - `TICK_OFF`（base+0）的單位與語意未定。已知：16 筆值相同、~1.2 kHz 單調遞增、
#     joint_cmd 與 joint_state 用同一個時鐘。夠用來維持心跳，但不知道它是 tick 還是別的。
#
# 2026-08-25 第三趟（M1 寫入測試）的更正：
#   - base+0 **不是**容器內部欄位，是時戳。凍結 mc_ctrl 後它停住 →
#     joint_shm_controller 依 joint_cmd_timeout=500ms 判定過期 → 把指令區清成 0。
#     這就是第一次 M1 讀回全 0（0/16 相符）的原因。
#   - **寫入路徑本身是通的**：非 root 寫不進去，但 sudo 下寫入不會被拒絕；
#     問題只在於少了心跳。
#
# 資料流（由 ros2_control 與啟動腳本推得）：
#     mc_ctrl ──寫──▶ /dev/shm/joint_cmd ──讀──▶ joint_shm_controller
#                                                    │ 寫 command interfaces
#                                                    ▼
#                                          zsi_actuator_driver ──▶ 16 顆馬達
#     馬達 ──▶ /dev/shm/joint_state（含溫度、電壓、error）
#   佐證：/opt/runtime/bin/start_motion_control.sh 會先等 /dev/shm/joint_cmd 出現
#         才啟動 mc_ctrl → shm 由別人（ros2_control 端）建立，mc_ctrl 是寫入者。
# ==============================================================================
