# 交接檔：CPG-RL 地形零樣本測試（新 session 讀這一份即可接手）

> 建立：2026-07-15　·　用途：換帳號/重登後，新 session 讀本檔即可無縫繼續。
> 一句話：**用現有已訓練的 CPG-RL 論文版權重（不重訓），在斜坡與凹凸地形上做零樣本測試，輸出影片給使用者看，判斷是否需要重新訓練。**

---

## 0. 使用者的原始需求（逐字）

1.（前一輪，已完成）研究 CPG-RL 論文有沒有做地形訓練 → 已寫報告 `task4/docs/cpg_rl_terrain_training_study.md`。
   結論：**論文只在平地訓練**，地形是續作 Visual CPG-RL 才有。
2.（**當前任務**）「你先幫我做一個測試，你拿目前的 CPG-RL(論文正式版) 先在模擬環境去跑跑看不同地形（**斜坡以及凹凸路面**），最後把**影片輸出**給我看，看看**有沒有可能不用重新訓練**。」

**互動風格要求**：繁體中文；先研究再動手；不要臆測，有問題要提出；每個結論要能對照證據。
使用者環境：無 MuJoCo 經驗、用 conda env `rbtdog`。

---

## 1. 當前進度

- ✅ 已讀完論文全文、已確認論文只在平地訓練（報告已寫）。
- ✅ 已摸清本機推論管線與模型建立方式（見下）。
- ✅ 已確認環境：**conda env `rbtdog`**，`mujoco 3.10.0`，`jax 0.10.2`（CPU）。MjSpec 可用、hfield API 可用。
- ⏳ **下一步（尚未做）：寫地形測試腳本 `task4/inference/terrain_test.py`，跑 flat/slope/rough，輸出影片。**
  - 我正要建立這個檔案時被使用者中斷來寫本交接檔。**接手後就從「第 4 節：要寫的腳本」開始做。**

---

## 2. 關鍵檔案與事實（都已查證，非臆測）

| 項目 | 值 |
|---|---|
| 已訓練權重 | `/home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl` |
| 本機推論範本 | `task4/inference/local_infer_paper.py`（**CPG/IK/obs/load_policy 直接 import 重用**） |
| 訓練 notebook | `task4/notebooks/cpg_rl_paper_colab.ipynb`（obs 76 維、動作 12 維、W_COUP=8） |
| 地形研究報告 | `task4/docs/cpg_rl_terrain_training_study.md` |
| 執行環境 | `conda run -n rbtdog python ...` |
| 場景 XML | `/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene.xml` |
| 地板 geom | `<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>` |

**模型如何建**（`task3/go2_model.py::make_model()`）：
```python
spec = mujoco.MjSpec.from_file("/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene.xml")
# 加 4 個 sensor 到機身 imu site：imu_acc, imu_gyro, odom_pos(FRAMEPOS), imu_quat(FRAMEQUAT)
return spec.compile()
```
→ 用 MjSpec，**可在 compile 前程式化改地形**（改 floor geom 或加 hfield）。

**推論管線重點**（`local_infer_paper.py`）：
- 常數（**必須與訓練一致**）：`MU_MIN=1,MU_MAX=2, OMEGA_MIN=0,OMEGA_MAX=4.5, A_CONV=50, D_STEP=0.12, G_C=0.08, G_P=0.01, W_COUP=8, N_CPG_SUB=4, CTRL_DT=0.02, FOOT_CONTACT_H=0.03, OBS_DIM=76, ACT_DIM=12`。
- 可 import 重用的函式：`cpg_init, cpg_step, act_to_cmd, joint_targets, leg_ik_consts(m), qinv/qrot/w2b, build_obs(g,c,cmd,last_a,foot_gid), load_policy(path)`。
- `build_obs(g, ...)` 只用到 `g.d`（MjData）→ 可傳一個只有 `.d` 屬性的 shim 物件。
- `load_policy(path)` 回傳 `infer(obs)->act`（policy 256/256/128、value 256³、normalize、deterministic）。
- 動作→CPG→IK→關節角→**軟體 PD**（kp=90,kd=3，力矩上限 clip）。PD apply 每控制步跑 `n_sub=round(CTRL_DT/timestep)` 次 mj_step。

**MjSpec API（已實測 3.10.0）**：
- floor geom 有 `.quat`（[w,x,y,z]）、`.pos`、`.type`、`.hfieldname`、`.alt`。
- `hf = spec.add_hfield()` 有欄位：`name, nrow, ncol, size, userdata`。
  - `size` = `[radius_x, radius_y, z_top, z_base]`。
  - 斜坡：改 floor geom `.quat` 繞 y 軸旋轉（`[cos(a/2),0,±sin(a/2),0]`），plane 在原點 z=0 → 機器人 spawn 沒問題。
  - 凹凸：`spec.add_hfield()` 設 nrow/ncol/size/userdata（高度 flatten row-major），再把 floor geom 改 `type=mjGEOM_HFIELD`、`hfieldname="rough"`。
  - ⚠️ MuJoCo 會把 hfield 高度**正規化到 [0,1]** 再乘 size[2]；別依賴絕對值。用小振幅（z_top≈0.06~0.10），並在 rollout 前讓機器人 settle（跑 HOME12 約 0.5s），rough 地形初始 base z 可稍微墊高避免穿模。

---

## 3. 設計決策（已定，接手照做即可）

- **公平性**：obs/CPG/IK/PD 全部沿用 `local_infer_paper.py`，**不改任何映射**，只換地形 → 這才是「零樣本、不重訓」的公平測試。
- **指令**：直接餵固定 `cmd=[vx=0.6, vy=0, wz=0]`（直走），**不用** local_infer 的轉向+odom line control（那是走絕對直線用的，這裡不需要）。vx=0.6 在訓練分布內。
- **地形清單（建議）**：
  - `flat`（sanity：確認 policy 在本腳本下能正常走，作為 baseline）
  - `slope_up_10`, `slope_up_15`（上坡；若時間允許加 `slope_down_10`）
  - `rough_03`（z_top≈0.06, 振幅~0.03m）, `rough_05`（z_top≈0.10, 振幅~0.05m）
- **每種地形**：跑 ~8s，輸出一支 mp4 到 `task4/outputs/terrain_<name>.mp4`，鏡頭跟著機器人。
- **量測/判定**（印出來，並可彙整成一張表）：
  - 前進距離（world x 位移）
  - 是否跌倒：用訓練一致條件 `grav[2] > -0.4`（機身翻倒）或 base 相對地面高度 < 0.15
  - FL 腳世界 z 的 min/max（看有無抬腳/拖地）
  - 末端 base 高度
- **fell 偵測跨地形**：斜坡要扣掉 plane 在 (x,y) 的地面高度；rough 以 baseline 0 近似。或直接用 `grav[2] > -0.4`（與地形無關，最穩）。

---

## 4. 要寫的腳本（接手就做這個）

檔案：`/home/huang/rbtdog_sim/task4/inference/terrain_test.py`

骨架：
```python
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/huang/rbtdog_sim/task3")
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import numpy as np, mujoco
import local_infer_paper as L   # 重用 cpg_init/cpg_step/act_to_cmd/joint_targets/leg_ik_consts/build_obs/load_policy/w2b
from types import SimpleNamespace

SCENE = "/home/huang/rbtdog_sim/mujoco_menagerie/unitree_go2/scene.xml"

def add_sensors(spec):  # 複製 go2_model.make_model 的 4 個 sensor（build_obs 其實只需 qpos/qvel/geom_xpos，可省，但保留以防 odom）
    ...

def make_terrain_model(kind, deg=10.0, amp=0.05):
    spec = mujoco.MjSpec.from_file(SCENE)
    floor = next(g for g in spec.geoms if g.name == "floor")
    if kind == "flat":
        pass
    elif kind.startswith("slope"):
        a = np.radians(deg)              # 正=上坡(+x)；實測後確認符號
        floor.quat = [np.cos(a/2), 0, -np.sin(a/2), 0]
    elif kind.startswith("rough"):
        N = 80
        # value noise / 多個正弦疊加，正規化到[0,1]，中心留平台
        xs = np.linspace(-6, 6, N); X, Y = np.meshgrid(xs, xs)
        f = (np.sin(1.3*X)*np.cos(1.7*Y) + 0.5*np.sin(2.9*X+1)*np.cos(3.1*Y+2)
             + 0.3*np.sin(5.0*X)*np.cos(4.3*Y))
        f = (f - f.min())/(f.max()-f.min())
        r = np.sqrt(X**2+Y**2); f = np.where(r < 0.6, 0.5, f)   # 中心平台好 spawn
        hf = spec.add_hfield(); hf.name="rough"; hf.nrow=N; hf.ncol=N
        hf.size = [6, 6, 2*amp, 0.5]
        hf.userdata = f.flatten().tolist()
        floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
        floor.hfieldname = "rough"
        floor.pos = [0, 0, -amp]         # 讓 baseline≈0（注意正規化，實測校正）
    add_sensors(spec)
    return spec.compile()

def rollout(kind, params, secs=8.0, deg=10.0, amp=0.05, video=True):
    m = make_terrain_model(kind, deg, amp)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if kind.startswith("rough"): d.qpos[2] += amp + 0.02   # 墊高避免穿模
    mujoco.mj_forward(m, d)
    f0s, jinvs = L.leg_ik_consts(m)
    foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, lg) for lg in L.LEGS]
    n_sub = int(round(L.CTRL_DT / m.opt.timestep))
    kp, kd = 90.0, 3.0
    flimit = m.actuator_ctrlrange[:, 1]
    G = SimpleNamespace(d=d)
    infer = L.load_policy(params)
    def apply(q_des):
        for _ in range(n_sub):
            tau = kp*(q_des - d.qpos[7:19]) - kd*d.qvel[6:18]
            d.ctrl[:] = np.clip(tau, -flimit, flimit); mujoco.mj_step(m, d)
    for _ in range(int(0.5/L.CTRL_DT)): apply(L.HOME12.copy())   # settle
    c = L.cpg_init(); last_a = np.zeros(12); frames=[]
    ren = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cmd = np.array([0.6, 0.0, 0.0], np.float32)
    x0 = float(d.qpos[0]); fzmin=fzmax=None; fell=None
    for i in range(int(secs/L.CTRL_DT)):
        obs = L.build_obs(G, c, cmd, last_a, foot_gid)
        act = infer(obs)
        mux,muy,om = L.act_to_cmd(act); c = L.cpg_step(c, mux, muy, om, L.CTRL_DT)
        apply(L.joint_targets(c, f0s, jinvs)); last_a = act
        grav = L.w2b(d.qpos[3:7], np.array([0,0,-1.0]))
        if grav[2] > -0.4 and fell is None: fell = i*L.CTRL_DT
        fz = d.geom_xpos[foot_gid[0]][2]
        fzmin = fz if fzmin is None else min(fzmin,fz); fzmax = fz if fzmax is None else max(fzmax,fz)
        if i % 2 == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]; cam.distance=2.5; cam.elevation=-20; cam.azimuth=90
            ren.update_scene(d, cam); frames.append(ren.render())
    dist = float(d.qpos[0]) - x0
    print(f"[{kind}] 前進={dist:+.2f}m 跌倒={'是@%.1fs'%fell if fell else '否'} FL抬腳={fzmax-fzmin:.3f}m 末端高={float(d.qpos[2]):.2f}")
    if video and frames:
        import imageio.v2 as iio
        out = f"/home/huang/rbtdog_sim/task4/outputs/terrain_{kind}.mp4"
        iio.mimsave(out, frames, fps=25, codec="libx264"); print("  影片:", out)

if __name__ == "__main__":
    P = "/home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl"
    rollout("flat", P)                       # 先 sanity
    rollout("slope_up_10", P, deg=10)
    rollout("slope_up_15", P, deg=15)
    rollout("rough_03", P, amp=0.03)
    rollout("rough_05", P, amp=0.05)
```

**執行**：`conda run -n rbtdog python task4/inference/terrain_test.py`

---

## 5. 已知風險 / 一定要現場校正的點

1. **hfield 正規化**：MuJoCo 把高度正規化到 [0,1]×size[2]，我上面的 `floor.pos=[0,0,-amp]` baseline 校正**要實測**（build 後印 `m.hfield_size`、檢查 spawn 時腳與地面的穿透）。若穿模：加大 settle 或墊高初始 z。
2. **斜坡符號**：`quat` 繞 y 的正負決定上坡/下坡，**跑一次看影片或看前進時 base z 有沒有升高**來確認，別假設。
3. **`add_sensors`**：`build_obs` 其實只用 qpos/qvel/geom_xpos，理論上**不需要** sensor 就能跑；若省略 sensor 可簡化。但 `leg_ik_consts` 需要 keyframe 0 存在（menagerie go2 有 `home` keyframe）。保險起見照 make_model 加 sensor 也行。
4. **flat sanity 必先過**：若 flat 都走不動，先查 obs 組法（欄位順序、HOME12、contact 布林）與 `local_infer_paper.py` 是否逐項一致，再談地形。
5. **CPU 速度**：每 rollout 約數千 mj_step，CPU 可接受；5 種地形跑完約數十秒~數分鐘。
6. 論文本身的預期（用來解讀結果）：平地訓練的 policy 對**輕微不平**（泡棉碎屑）有魯棒性 → 我們預期**小斜坡、小凹凸能過，大斜坡/大起伏會失敗或拖地**。這正好回答使用者「是否需要重訓」。

---

## 6. 交付與回報（做完要給使用者的）

- 影片：`task4/outputs/terrain_*.mp4`（每種地形一支）。
- 一張彙整表：地形 / 前進距離 / 是否跌倒 / 抬腳量 / 末端高度。
- 結論回答使用者的問題：**哪些地形零樣本可過、哪些不行、是否需要重訓**（連回 `cpg_rl_terrain_training_study.md` 的方案 A/B/C 建議）。
- 用繁體中文；先給重點結論再給細節。

---

## 7. 相關指令速查

```bash
# 跑測試
conda run -n rbtdog python /home/huang/rbtdog_sim/task4/inference/terrain_test.py
# 探 MjSpec/hfield API
conda run -n rbtdog python -c "import mujoco; print(mujoco.__version__)"
# 既有可運作的推論（平地，可用來對照 obs 正確性）
conda run -n rbtdog python /home/huang/rbtdog_sim/task4/inference/local_infer_paper.py --params /home/huang/rbtdog_sim/task4/weights/cpg_rl_paper_params.pkl --secs 8 --video
```
```

**接手第一步**：讀 `local_infer_paper.py` 與本檔第 2 節確認常數，然後建立 `task4/inference/terrain_test.py`（第 4 節骨架），先跑 `flat` sanity，再跑斜坡/凹凸，最後輸出影片與彙整表給使用者。
