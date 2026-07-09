"""任務3-B：走直線 40m + 外部干擾，對比「有羅盤校正 vs 無」。
目標視覺 = 一支永遠指向前方(+x)的箭頭（目標航向）。加入側推+yaw 干擾：
V2(羅盤) 會把身體轉回與箭頭對齊，V1(開環) 偏掉不校正。
輸出：real-time 並排影片（正後方45°俯視，狗歪就自動縮小畫面）+ 軌跡對比圖。
用法: python walk_line.py            # 產影片 line_video.mp4 + line_result.png
"""
import numpy as np, sys
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
import mujoco
from go2_gait import Go2Gait
from PIL import Image, ImageDraw

# 長步幅但求穩的步態：步幅 0.32m、步頻 2.6Hz、站立相佔比 0.6、低抬腳 0.12（壓垂直彈跳）。
# 相較前版 0.36/2.3/0.55/0.16：側傾 roll −46%、垂直彈跳 −24%，走直更穩，速度仍≈0.85 m/s。
GAIT = dict(freq=2.6, duty=0.60, lift=0.12, stride=0.32, kp=90, kd=3.0, turn_gain=0.18)
COMPASS_GAIN = 4.0          # 羅盤航向 P 增益（配合長步幅調高，鎖向更緊）
DIST = 40.0                 # 目標前進距離 (m)
FPS = 25
W = H = 480
MAX_T = 90.0
# 干擾排程：(開始時間, 持續, 側推力Fy, yaw力矩Tz)，同時施加到兩隻。
# 交替方向：V1 左右擺盪回不了正（無校正），V2 每次都把航向 snap 回箭頭方向。
# 以≈0.93 m/s 走 40m 約需 43s，故把 5 次干擾平均鋪在行程內。
PUSHES = [(7, 0.5, 35, 10), (14, 0.5, -35, -10), (21, 0.5, 35, 10),
          (28, 0.5, -35, -10), (35, 0.5, 35, 10)]
ARROW_MAT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float).flatten()  # z軸->+x


def dir_mat(d):
    """回傳讓箭頭長軸(local z)指向世界向量 d 的旋轉矩陣（row-major 展平）。"""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(up, d); x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return np.column_stack([x, y, d]).flatten()


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def make_dog():
    g = Go2Gait(**GAIT)
    g.bid = mujoco.mj_name2id(g.m, mujoco.mjtObj.mjOBJ_BODY, "base")
    g.rend = mujoco.Renderer(g.m, height=H, width=W)
    g.cam = mujoco.MjvCamera()
    g.cam.azimuth = 0.0
    g.cam.elevation = -45.0
    g.cam_dist = 4.0                                     # 平滑用的目前距離
    return g


def apply_push(g, t):
    """施加當前排程的外力，並回傳作用中的 (fy, tz)（無則 None）。"""
    g.d.xfrc_applied[g.bid] = 0
    for t0, dur, fy, tz in PUSHES:
        if t0 <= t < t0 + dur:
            g.d.xfrc_applied[g.bid, 1] = fy
            g.d.xfrc_applied[g.bid, 5] = tz
            return fy, tz
    return None


def add_geom(scn, gtype, size, pos, mat, rgba):
    gm = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(gm, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def render_dog(g, label, color, push=None):
    x, y = g.xy
    # 相機在狗正後方跟隨(lookat=狗)，方位角固定朝 +x(不隨狗轉)->狗歪時身體在框內轉向。
    # 隨橫偏輕微拉遠(縮小畫面)但有上限，讓狗始終清晰可見。
    target_d = min(8.0, 3.0 + 0.25 * abs(y))
    g.cam_dist += 0.06 * (target_d - g.cam_dist)
    g.cam.lookat[:] = [x, y, 0.3]
    g.cam.distance = g.cam_dist
    g.rend.update_scene(g.d, g.cam)
    scn = g.rend.scene
    add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.04, 0.04, 0.7],   # 目標航向：恆指 +x
             [x, y, 0.55], ARROW_MAT, [1, 0.1, 0.1, 1])
    if push is not None:                                  # 外力箭頭：指向側推方向、長度依大小
        fy, tz = push
        d = np.sign(fy)                                   # +1=+y, -1=-y
        L = 0.4 + 0.02 * abs(fy)                          # 力越大箭頭越長
        # 箭頭尾端在狗外側、頭指向狗身，直觀呈現「被從側邊推」。
        base = np.array([x, y - d * (0.15 + L), 0.35])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.06, 0.06, L],
                 base, dir_mat([0, d, 0]), [1.0, 0.55, 0.0, 1])   # 橘色
    img = g.rend.render()
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W, 24], fill=(255, 255, 255))
    dr.text((8, 6), label, fill=color)
    dr.text((8, H - 18), f"x={x:5.1f}m  y={y:+5.1f}m  heading={np.degrees(g.true_yaw()):+4.0f}deg",
            fill=(20, 20, 20))
    if push is not None:                                  # 外力提示文字
        fy, _ = push
        arrow = "<--" if fy > 0 else "-->"                # 相機朝+x：螢幕右=世界-y，故+y推力在畫面往左
        dr.text((W - 170, 30), f"{arrow} PUSH {abs(fy):.0f}N", fill=(255, 120, 0))
    return np.asarray(im)


def main():
    g1 = make_dog()   # V1 開環
    g2 = make_dog()   # V2 羅盤
    dt = g1.m.opt.timestep
    for _ in range(int(0.5 / dt)):
        g1.step(0, 0, 0); g2.step(0, 0, 0)
    t = 0.5
    frame_iv = int((1.0 / FPS) / dt)
    traj1, traj2, head1, head2 = [], [], [], []
    import imageio.v2 as iio
    writer = iio.get_writer("/home/huang/rbtdog_sim/task3/line_video.mp4",
                            fps=FPS, codec="libx264", quality=7)
    k = 0
    while t < MAX_T and g2.xy[0] < DIST:
        push = apply_push(g1, t); apply_push(g2, t)       # 兩狗同排程，共用 push 資訊
        turn1 = 0.0                                       # 開環：不修正
        turn2 = np.clip(COMPASS_GAIN * wrap(g2.compass_yaw() - 0.0), -1, 1)  # 羅盤保持朝向 0
        g1.step(t, 1.0, turn1); g2.step(t, 1.0, turn2)
        t += dt; k += 1
        if k % frame_iv == 0:
            traj1.append(g1.xy.copy()); traj2.append(g2.xy.copy())
            head1.append(g1.true_yaw()); head2.append(g2.true_yaw())
            a = render_dog(g1, "V1: no IMU (open-loop)", (200, 0, 0), push)
            b = render_dog(g2, "V2: IMU compass (heading hold)", (0, 0, 200), push)
            writer.append_data(np.concatenate([a, b], axis=1))
    writer.close()
    traj1 = np.array(traj1); traj2 = np.array(traj2)
    dur = t - 0.5
    print(f"影片已存 line_video.mp4  時長≈{len(traj1)/FPS:.0f}s (real-time, 模擬{dur:.0f}s)")
    print(f"V1 終點 xy={np.round(traj1[-1],2)}  V2 終點 xy={np.round(traj2[-1],2)}")

    # 軌跡對比圖
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, color='r', ls='--', lw=1.5, alpha=0.7, label='目標航向 (箭頭方向, +x)')
    ax.plot(traj1[:, 0], traj1[:, 1], 'r-', lw=1.6, label=f'V1 無IMU (終點 y={traj1[-1,1]:+.1f}m)')
    ax.plot(traj2[:, 0], traj2[:, 1], 'b-', lw=1.6, label=f'V2 羅盤 (終點 y={traj2[-1,1]:+.1f}m)')
    for t0, *_ in PUSHES:                                 # 標干擾發生的大約 x 位置
        pass
    ax.plot(0, 0, 'go', ms=10, label='起點')
    ax.set_xlabel('x 前進 (m)'); ax.set_ylabel('y 橫向偏移 (m)')
    ax.set_title('走直線 40m + 側向干擾：有無羅盤校正對比')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("/home/huang/rbtdog_sim/task3/line_result.png", dpi=110)
    print("軌跡圖已存 line_result.png")


if __name__ == "__main__":
    main()
