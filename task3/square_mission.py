"""任務3 主程式：機器狗走正方形（10m×右轉90°×5圈）。
版本1 (open-loop)：無感測器，靠航位推算（時間）決定走多久、轉多久 → 誤差累積。
版本2 (compass) ：用 magnetometer 羅盤回授，走直修正航向、轉彎轉到精準 90° → 誤差小。
輸出：兩版軌跡對比圖 + 航向對比圖 + 終點誤差數據。
"""
import numpy as np, sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
from go2_gait import Go2Gait
from walk_line import GAIT, COMPASS_GAIN     # 與走直線 demo 共用同一組穩定步態參數

# ---- 校準得到的標稱值（對應 walk_line.GAIT 的新穩定步態，2026-07-02 重新校準）----
NOMINAL_SPEED = 0.892        # m/s（前進 v=1）
NOMINAL_YAWRATE = np.radians(58.7)  # rad/s（turn=+1 原地右轉的角速度大小）
SIDE_LEN = 10.0              # m
LAPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_SIDES = LAPS * 4
MAG_NOISE = np.radians(1.0)  # (已停用) 舊的羅盤雜訊；航向誤差改由 go2_gait 的 MTi-680G 模型提供
LOG_EVERY = 10               # 每 N 步記錄一次（降負擔）


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def run_mission(use_compass, seed=0):
    rng = np.random.default_rng(seed)
    g = Go2Gait(**GAIT)
    dt = g.m.opt.timestep
    for i in range(int(0.5 / dt)):     # 先站穩
        g.step(0, 0, 0)
    t = 0.5
    traj = [g.xy.copy()]
    head_log = []                       # (t, true_yaw, est_yaw, target)
    target = 0.0                        # 目標航向，右轉每次 -90°

    def compass():
        return g.compass_yaw()      # 誤差已由 MTi-680G 融合航向模型內建於 compass_yaw()

    def rec(est):
        traj.append(g.xy.copy())
        head_log.append((t, g.true_yaw(), est, target))

    for side in range(N_SIDES):
        # ---- 直行 10m（航位推算固定時間）----
        n_walk = int((SIDE_LEN / NOMINAL_SPEED) / dt)
        for k in range(n_walk):
            turn = np.clip(COMPASS_GAIN * wrap(compass() - target), -1, 1) if use_compass else 0.0
            g.step(t, 1.0, turn); t += dt
            if k % LOG_EVERY == 0:
                rec(compass() if use_compass else np.nan)

        # ---- 右轉 90° ----
        target = wrap(target - np.pi / 2)
        if use_compass:
            # 閉環：轉到羅盤≈target；門檻放寬 3°、需連續 0.1s；硬性超時 6s 保證結束
            settle = 0; need = int(0.1 / dt); cap = int(6.0 / dt); k = 0
            while True:
                err = wrap(compass() - target)
                g.step(t, 0.0, np.clip(3.0 * err, -1, 1)); t += dt; k += 1
                if k % LOG_EVERY == 0:
                    rec(compass())
                settle = settle + 1 if abs(err) < np.radians(3.0) else 0
                if settle >= need or k >= cap:
                    break
        else:
            # 開環：轉固定時間 = 90°/標稱角速度
            n_turn = int(((np.pi / 2) / NOMINAL_YAWRATE) / dt)
            for k in range(n_turn):
                g.step(t, 0.0, 1.0); t += dt
                if k % LOG_EVERY == 0:
                    rec(np.nan)
        print(f"    邊 {side+1}/{N_SIDES} 完成  t={t:.0f}s  pos={np.round(g.xy,2)}  真航向={np.degrees(g.true_yaw()):.0f}deg", flush=True)

    return np.array(traj), np.array(head_log), g


def main():
    print("跑 版本1（開環，無感測器）...")
    traj1, h1, _ = run_mission(use_compass=False)
    print("跑 版本2（羅盤校正）...")
    traj2, h2, _ = run_mission(use_compass=True)

    start = np.array([0.0, 0.0])
    err1 = np.linalg.norm(traj1[-1] - traj1[0])
    err2 = np.linalg.norm(traj2[-1] - traj2[0])
    print(f"\n=== 終點誤差（終點應回到起點）===")
    print(f"  版本1 (無IMU) : {err1:6.2f} m")
    print(f"  版本2 (羅盤)  : {err2:6.2f} m")
    print(f"  改善倍數      : {err1/max(err2,1e-6):.1f}x")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    # 理想正方形
    ideal = np.array([[0,0],[10,0],[10,-10],[0,-10],[0,0]])
    ax[0].plot(ideal[:,0], ideal[:,1], 'k--', lw=1.5, alpha=0.6, label='理想方形 (1圈)')
    ax[0].plot(traj1[:,0], traj1[:,1], 'r-', lw=1.0, alpha=0.8, label=f'版本1 無IMU (終點誤差 {err1:.1f}m)')
    ax[0].plot(traj2[:,0], traj2[:,1], 'b-', lw=1.0, alpha=0.8, label=f'版本2 羅盤 (終點誤差 {err2:.2f}m)')
    ax[0].plot(*start, 'go', ms=11, label='起點/終點目標', zorder=5)
    ax[0].plot(traj1[-1,0], traj1[-1,1], 'rx', ms=13, mew=3, zorder=5)
    ax[0].plot(traj2[-1,0], traj2[-1,1], 'bx', ms=13, mew=3, zorder=5)
    ax[0].set_aspect('equal'); ax[0].grid(alpha=0.3)
    ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)')
    ax[0].set_title(f'走正方形軌跡俯視（{LAPS}圈）'); ax[0].legend(fontsize=9)
    # 航向（unwrap 消除 ±180 回繞）
    ax[1].plot(h2[:,0], np.degrees(np.unwrap(h2[:,3])), 'k--', lw=1.2, alpha=0.6, label='目標航向')
    ax[1].plot(h1[:,0], np.degrees(np.unwrap(h1[:,1])), 'r-', lw=1, label='版本1 真實航向')
    ax[1].plot(h2[:,0], np.degrees(np.unwrap(h2[:,1])), 'b-', lw=1, label='版本2 真實航向')
    ax[1].set_xlabel('時間 (s)'); ax[1].set_ylabel('航向 yaw 累積 (deg)')
    ax[1].set_title('航向追蹤（每圈右轉共 -360°）'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "/home/huang/rbtdog_sim/task3/square_result.png"
    plt.savefig(out, dpi=110)
    print("結果圖已存:", out)
    np.save("/home/huang/rbtdog_sim/task3/traj1.npy", traj1)
    np.save("/home/huang/rbtdog_sim/task3/traj2.npy", traj2)


if __name__ == "__main__":
    main()
