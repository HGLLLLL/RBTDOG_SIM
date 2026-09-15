"""原廠錄檔在 MuJoCo 開迴路回放（spec v3.1 §5 步驟 0）：檢查模擬器的輪地接觸能不能重現原廠動作。

腿：位置伺服 kp 60/120/120、kd 1.0（`scene_flat_mjx.xml` 就是這組，= 原廠動作段的增益），ctrl = 錄檔 des。
輪：原廠 kd 0.1 幾乎是力矩控制、v_des 沒錄到 → 把錄檔實測 τ（正 = 向前滾）直接加到輪 DOF（qfrc_applied），輪致動器關掉。
起點：qpos 腿 = 錄檔 q、des = q 定住 1 s，再逐 tick（500 Hz = 錄檔率）餵 des／τ，回放 `--window` 秒；每檔 `--starts` 個起點。
判讀（spec §5 步驟 0）：原地轉回放偏航 ≥ 50% 原廠且不摔 → 輪地模型可信；< 25% → 先掃 --mu。

    conda run -n rbtdog python task7/inference/diag/replay_factory.py --window 1.5 --starts 3
    conda run -n rbtdog python task7/inference/diag/replay_factory.py --tags 161316 --mu 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realbot"))
import coord  # noqa: E402
import kin  # noqa: E402
import m6_rec  # noqa: E402
import max_model as mm  # noqa: E402
import ref_gait_analysis as rga  # noqa: E402

SHM_OF = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
KINDS = ("1_hip_roll", "2_hip_pitch", "3_knee_pitch")
TAGS = {"161316": "turn_left", "161342": "turn_right", "161123": "lat_left", "161223": "lat_right",
        "161544": "arc_left", "161711": "arc_right", "160911": "fwd_fast"}


def load_ctrl_frame(path: str):
    """→ (rec, q, des, tau_w, v_w, lift_rec)，關節 MJCF 序、控制器座標；輪量正 = 向前滾；lift_rec = FK 高度相對 1 s 滾動最低點。"""
    from scipy.ndimage import minimum_filter1d
    rec = m6_rec.load(path)
    q = np.stack([coord.to_ctrl(SHM_OF[L] + k, rec.j[SHM_OF[L] + k]["q"]) for L in mm.LEGS for k in KINDS], 1)
    des = np.stack([coord.to_ctrl(SHM_OF[L] + k, rec.j[SHM_OF[L] + k]["des"]) for L in mm.LEGS for k in KINDS], 1)
    tau_w = np.stack([rec.j[SHM_OF[L] + "4_foot"]["tau"] / coord.SIGN[coord.KIND_WHEEL][SHM_OF[L]] for L in mm.LEGS], 1)
    v_w = np.stack([rec.j[SHM_OF[L] + "4_foot"]["v"] / coord.SIGN[coord.KIND_WHEEL][SHM_OF[L]] for L in mm.LEGS], 1)
    z = np.stack([[kin.fk(SHM_OF[L], *row)[2] for row in q[:, 3 * i:3 * i + 3]] for i, L in enumerate(mm.LEGS)], 1)
    lift = z - minimum_filter1d(z, int(rga.HZ), axis=0)
    return rec, q, des, tau_w, v_w, lift


def make_model(mu: float | None = None) -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(mm.SCENE_MJX)
    kp = m.actuator_gainprm[mm.LEG_ACT_IDX, 0]
    assert np.allclose(kp, np.tile([60.0, 120.0, 120.0], 4)), f"scene_flat_mjx 腿增益不是原廠動作組：{kp}"
    assert np.allclose(-m.actuator_biasprm[mm.LEG_ACT_IDX, 2], 1.0)
    m.actuator_gainprm[mm.WHEEL_ACT_IDX, :] = 0.0
    m.actuator_biasprm[mm.WHEEL_ACT_IDX, :] = 0.0
    # 輪關節照 M11 實測（與 scene_flat_mjx_v3 同）：τ_f 0.13、b 0.015、armature 0.004
    m.dof_frictionloss[mm.WHEEL_QVEL_IDX] = 0.13
    m.dof_damping[mm.WHEEL_QVEL_IDX] = 0.015
    m.dof_armature[mm.WHEEL_QVEL_IDX] = 0.004
    if mu is not None:
        m.geom_friction[:, 0] = mu
    return m


def _settle(m, d, q12, secs=1.0):
    d.qpos[:] = 0.0
    d.qpos[2], d.qpos[3] = 0.56, 1.0
    d.qpos[mm.LEG_QPOS_IDX] = q12
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    d.ctrl[mm.LEG_ACT_IDX] = q12
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)


def replay(m, q, des, tau_w, i0: int, n: int, v0=None) -> dict:
    """v0 = (v_body_x, yaw_rate, wheel_v(4))：錄檔在 i0 已在動，給模擬同樣的初速（沒有外部定位，v_body 由四輪平均估）。"""
    d = mujoco.MjData(m)
    _settle(m, d, q[i0])
    if v0 is not None:
        vb, wz, wv0 = v0
        d.qvel[0], d.qvel[5] = vb, wz
        d.qvel[mm.WHEEL_QVEL_IDX] = wv0
        mujoco.mj_forward(m, d)
    base = mm._id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    gids = [mm._id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{mm.PREFIX[L]}_FOOT_LINK_COLL") for L in mm.LEGS]
    r = float(m.geom_size[gids[0]][0])
    yaw, clr, wv, fell = np.zeros(n), np.zeros((n, 4)), np.zeros((n, 4)), False
    zfk = np.zeros((n, 4))                      # 與錄檔同定義：FK 高度（相對 ABAD）
    pos0 = d.xpos[base].copy(); R0 = d.xmat[base].reshape(3, 3).copy()
    roll = np.zeros(n); disp = np.zeros((n, 2))
    j = 0
    for j in range(n):
        i = min(i0 + j, len(des) - 1)
        d.ctrl[mm.LEG_ACT_IDX] = des[i]
        d.qfrc_applied[mm.WHEEL_QVEL_IDX] = tau_w[i]
        mujoco.mj_step(m, d)
        yaw[j] = d.qvel[5]
        clr[j] = d.geom_xpos[gids, 2] - r
        wv[j] = d.qvel[mm.WHEEL_QVEL_IDX]
        qq = d.qpos[mm.LEG_QPOS_IDX]
        zfk[j] = [kin.fk(SHM_OF[L], *qq[3 * k:3 * k + 3])[2] for k, L in enumerate(mm.LEGS)]
        R = d.xmat[base].reshape(3, 3)
        roll[j] = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        disp[j] = (R0.T @ (d.xpos[base] - pos0))[:2]          # 起點機身座標的位移 (x 前, y 左)
        if d.xmat[base].reshape(3, 3)[2, 2] < 0.5 or d.qpos[2] < 0.25:
            fell = True
            break
    lift_fk = zfk[:j + 1] - zfk[:j + 1].min(0)
    return dict(yaw_rad_s=yaw[:j + 1], clr=clr[:j + 1], lift_fk=lift_fk, wheel_v=wv[:j + 1], fell=fell, height=float(d.qpos[2]),
                roll_deg=roll[:j + 1], disp=disp[:j + 1])


def _q_dict(rec):
    """ref_gait_analysis.moving_mask 要的 q dict（shm 名、控制器座標）。"""
    return {l: {k: coord.to_ctrl(l + k, rec.j[l + k]["q"]) for k in coord.LEG_KINDS} for l in rga.LEGS}


def run_file(path: str, window: float, starts: int, mu=None) -> list:
    rec, q, des, tau_w, v_w, lift = load_ctrl_frame(path)
    m = make_model(mu)
    n = int(window * rga.HZ)
    mv = rga.moving_mask(rec, _q_dict(rec), v_w)
    idx = np.nonzero(mv)[0]
    a, b = int(idx[0]) + int(rga.HZ), int(idx[-1]) - n
    if b <= a:                                   # 視窗比動作段長：從動作段起點放到底
        a, b = int(idx[0]), int(idx[0])
        n = min(n, int(idx[-1]) - a)
    rows = []
    for i0 in np.linspace(a, b, starts).astype(int):
        v0 = (float(mm.WHEEL_RADIUS * v_w[i0].mean()), float(rec.gyro[i0, 2]), v_w[i0])
        r = replay(m, q, des, tau_w, i0, n, v0)
        k = len(r["yaw_rad_s"])
        rows.append(dict(
            t0=float(rec.t[i0]), fell=r["fell"], secs=k / rga.HZ,
            yaw_sim=float(np.degrees(r["yaw_rad_s"].mean())), yaw_rec=float(np.degrees(rec.gyro[i0:i0 + k, 2].mean())),
            lift_sim=(r["clr"].max(0) * 1000).round(0).tolist(), lift_fk_sim=(r["lift_fk"].max(0) * 1000).round(0).tolist(),
            lift_rec=(lift[i0:i0 + k].max(0) * 1000).round(0).tolist(),
            wheel_sim=r["wheel_v"].mean(0).round(1).tolist(), wheel_rec=v_w[i0:i0 + k].mean(0).round(1).tolist(),
            roll_max=float(np.abs(r["roll_deg"]).max()), disp_xy=(r["disp"][-1] * 1000).round(0).tolist(),
            roll_rec_max=float(np.degrees(np.abs(np.cumsum(rec.gyro[i0:i0 + k, 0]) / rga.HZ).max()))))
    return rows


def md(results: dict, mu) -> str:
    L = [f"# 原廠錄檔 MuJoCo 開迴路回放（trip21；腿 kp 60/120/120 kd 1.0、輪 = 錄檔 τ；μ {mu if mu is not None else 'MJCF 預設'}）", "",
         "| 檔 | 起點 s | 回放 s | 偏航 模擬/原廠 °/s | 比 | 離地間隙 模擬 mm (FR,FL,RR,RL) | FK 抬高 模擬 | FK 抬高 原廠 | 輪速 模擬 | 輪速 原廠 | 摔 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, rows in results.items():
        for r in rows:
            ratio = r["yaw_sim"] / r["yaw_rec"] if abs(r["yaw_rec"]) > 3 else float("nan")
            L.append(f"| {TAGS.get(tag, tag)} | {r['t0']:.1f} | {r['secs']:.2f} | {r['yaw_sim']:+.0f} / {r['yaw_rec']:+.0f} | {ratio:.2f} | "
                     f"{r['lift_sim']} | {r['lift_fk_sim']} | {r['lift_rec']} | {r['wheel_sim']} | {r['wheel_rec']} | {'是' if r['fell'] else ''} |")
    L += ["", "判讀：原地轉比 ≥ 0.5 且不摔 → 輪地模型可信、G0 門檻照 spec；比 < 0.25 或輪被刮地鎖死 → 先掃 --mu 0.6/0.8。開迴路回放閉迴路策略會發散，只信前 1.5 s、三個起點一致才算。"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=str(HERE.parent / "logs" / "m_logs_trip21"))
    ap.add_argument("--tags", nargs="*", default=list(TAGS))
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "replay_factory_trip21.md"))
    a = ap.parse_args()
    results = {}
    for tag in a.tags:
        path = next(Path(a.logdir).glob(f"M6_*_{tag}.json"))
        results[tag] = run_file(str(path), a.window, a.starts, a.mu)
        for r in results[tag]:
            print(f"{TAGS.get(tag, tag):11s} t0 {r['t0']:5.1f} 放 {r['secs']:.1f}s yaw {r['yaw_sim']:+5.0f}/{r['yaw_rec']:+5.0f} | roll_max 模擬 {r['roll_max']:.1f}° (原廠 gyro 積分 {r['roll_rec_max']:.1f}°) | 位移 x/y {r['disp_xy']} mm | clr {r['lift_sim']} fk {r['lift_fk_sim']} / rec {r['lift_rec']} | 輪 {r['wheel_sim']} / {r['wheel_rec']} | fell {r['fell']}")
    txt = md(results, a.mu)
    out = Path(a.out) if a.mu is None else Path(a.out).with_name(f"replay_factory_trip21_mu{a.mu}.md")
    out.write_text(txt + "\n", encoding="utf-8")
    print("→", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
