# task6 — 智元 D1 EDU 輪足版（ZSL-1w）CPG-RL 走路

把 task4 在 Go2 上驗證過的 CPG-RL 論文標準版移植到智元 D1 EDU **輪足版**。
四顆輪子在模擬中熔接鎖死，當成 71 mm 圓腳走路，動作空間維持 12 維。
訓練在 Colab GPU（MJX），推論在本機 CPU。**與 task4 完全獨立，不共用程式碼。**

---

## ⚠️ 先讀這四件事

（下面所有「Cell N」與 notebook 一樣都是 **0-based**：第一格 markdown = Cell 0、
安裝格 = Cell 1、版本檢查 = Cell 2、clone = Cell 3、常數 = Cell 4、
IK = Cell 5、env = Cell 6、DR = Cell 7、smoke = Cell 8、訓練 = Cell 9。）

### 1. sim2real 目前沒有已驗證的部署路徑

- ✅ **已確認**：官方 SDK **不提供**輪足版的關節控制。
  `nm -D` 驗證 `zsl-1w` 的庫導出 **0 個** LowLevel 符號，
  文件 `deploy.md:168` 亦明載「ZSL-1w 不提供 LowLevel 接口」。
  實機透過 HighLevel 讀是全開的，寫只到 `move(vx, vy, yaw_rate)` 這一層。
- ❓ **未定案**：機上另有一個**不經過 SDK** 的共享記憶體介面 `/spline_shm`
  （`create_spline_shm()` 是標頭檔的 `static inline`，直接對機上 daemon 通訊）。
  **讀取**已在實機驗證可用；**寫入**尚未經硬體驗證。

→ 所以**不要說「完全沒有部署路徑」**，也**不要說「已經有部署路徑」**。
正確表述與完整證據見 `docs/D1_EDU_W_規格與模型對照.md` §3
與 `docs/D1EDU_輪足_lowlevel_調查與實測指南.md`。

模擬訓練與驗證的價值不受此影響。

### 2. 腿序與關節零點尚未定案

官方四份資料的腿序互相矛盾（高層 API 用 FL 開頭、SHM 底層用 FR 開頭）；
實機 `shm_probe` 實測確認這台走 **FR 開頭**那套，且 SHM 讀到的關節角度**與 URDF 限位對不上**。
模擬端採 `LEGS = ("FL", "FR", "RL", "RR")`，內部自洽。
**sim2real 前必須逐腿實測並補一層 offset / 正負校正。** 見規格文件 §5。

### 3. Colab 抓的是 origin 上的內容——本機新 commit 要先 push

notebook 的 Cell 3 用
`git clone --depth 1 --branch feat/d1-edu-cpg-rl https://github.com/HGLLLLL/RBTDOG_SIM.git`
取模型檔。分支 `feat/d1-edu-cpg-rl` **已經在 origin 上**（`main` 上仍然沒有 `task6/`），
所以不需要「第一次 push」；但 `--depth 1` 抓的是**遠端當下的內容**，
**本機剛改好還沒 push 的 MJCF/notebook 修正，Colab 看不到**。

→ 每次改完 `task6/model/` 或 notebook，開 Colab 前先 `git push`。
（或改用 Colab 左側檔案面板手動上傳整個 `task6/model/d1_edu_w/`，含 `meshes/` 的 17 個 STL。）
Cell 3 有 assert 會當場擋下，不會拖到 Cell 5 才炸。

### 4. jax 必須 `<0.10`（Colab GPU 端有不確定性）

`brax==0.14.2` 的 `ppo.train` 用 `jax.device_put_replicated`，該 API 在 jax 0.10 已移除
（本機 jax 0.10.2 實測直接 `AttributeError`）。notebook 的 Cell 1 因此安裝
`"jax[cuda12]<0.10"`，Cell 2 有 `assert hasattr(jax, "device_put_replicated")` 早死擋著。

**這一項無法在本機驗證**：Colab 的 GPU jax 是預裝的，降版未必能同時保住 CUDA 支援。
裝完務必確認 Cell 2 印出的 `devices:` 有 `cuda`。
**裝不起來或掉回 CPU 請回報，不要自行改 brax 版本**——換 brax 會動到
`make_ppo_networks` 的預設 activation，權重與本機推論端會靜默不匹配。

---

## 資料夾

```
task6/
├── model/d1_edu_w/   MJCF（d1_edu_w.xml / scene.xml / scene_mjx.xml）+ 17 個 STL
│                     + SOURCE.md（來源與授權）+ LICENSE（BSD-3-Clause）
├── inference/        d1_model.py   常數的單一事實來源
│                     cpg_d1.py     CPG 動力學 + 動作解碼 + 腿部 IK（純函式）
│                     obs_d1.py     69 維 observation
│                     cpg_openloop_d1.py  關卡 3：開迴路 CPG 驗證
│                     local_infer_d1.py   關卡 6：本機 CPU 推論
├── notebooks/        cpg_rl_d1w_colab.ipynb（MJX + brax PPO）
├── tests/            pytest（155 項；含 mjx.put_model 實跑與 base↔wheel exclude 的回歸）
├── weights/          訓練好的權重（.pkl；需自行從 Colab 下載放入）
├── outputs/          影片與圖（*.mp4 被 repo 全域 .gitignore 擋掉，不入庫）
└── docs/             D1_EDU_W_規格與模型對照.md（規格 / SDK / obs / 腿序 / 已知限制）
                      D1EDU_輪足_lowlevel_調查與實測指南.md（SDK 調查 + 三階段實機測試步驟）
```

## 與 Go2（task4）的關鍵差異

| 項目 | Go2 | D1 EDU 輪足 |
|---|---|---|
| 總質量 | 15 kg | **20.56 kg**（四顆輪各 0.901 kg）|
| home 關節角 | `[0, 0.9, -1.8]` | `[0, 1.05, -2.00]`（**膝軸 +y**，點足版是 −y）|
| 足端 | 17.5 mm 球 | **71 mm 輪（熔接鎖死）**，碰撞用 cylinder |
| PD | `apply_pd()` 覆寫 90/3 | **XML 內建 kp=80 / kd=1（原廠值），無 `apply_pd`** |
| 力矩上限 | 23.7 / knee 45.43 | 28 全關節 |
| 步幅尺度 | 前後 / 側向同為 0.12 | 前後 0.12、**側向 0.09**（abad 行程僅 ±28°，Go2 ±60°）|
| obs | 76 維 | **69 維** = 76 −3（機身線速度）−4（觸地布林）|
| 觸地判定 | 腳掌世界高度 | **移除**（實測四個候選訊號全部不可用）|
| 高度獎勵基準 | keyframe z | **0.2695 m**（實際站定高度，非 keyframe 的 0.2948）|
| `done` 低姿門檻 | 0.18（站立高 0.30）| **0.16** = 0.18/0.30×0.2695 等比例換算 |
| 負重 DR | 0~8 kg | 0~5 kg（官方額定 payload）|
| IMU DR | 無 | 重力 / 角速度加雜訊與偏差 |

## 使用流程

```bash
# 1. 跑測試（關卡 1~3 的回歸）
conda run --no-capture-output -n rbtdog python -m pytest task6/tests -v

# 2. 開迴路 CPG 驗證 + 影片（關卡 3）
conda run --no-capture-output -n rbtdog python task6/inference/cpg_openloop_d1.py --secs 8 --video

# 3. 訓練（Colab GPU）：先確認上面第 3 點——最新 commit 已 push 到 origin，
#    再上傳並開啟 notebooks/cpg_rl_d1w_colab.ipynb，
#    執行階段 → 變更類型 → GPU，先跑 Cell 8 的 Smoke test 印出 PASSED 再開訓練，
#    訓完下載 cpg_rl_d1w_params.pkl 放進 task6/weights/

# 4. 本機推論（關卡 6）
conda run --no-capture-output -n rbtdog python task6/inference/local_infer_d1.py \
    --params task6/weights/cpg_rl_d1w_params.pkl --secs 20 --video --push

# 沒有權重時可先測管線
conda run --no-capture-output -n rbtdog python task6/inference/local_infer_d1.py --dummy --secs 8 --video
```

## 已通過的驗證

| 關卡 | 內容 | 結果 |
|---|---|---|
| 1 | MJCF 與 URDF 逐項對帳（質量 / 慣量 / 限位 / 膝軸 / 致動器）| PASS |
| 2 | home 姿態靜態穩定 | 沉降 2.47 cm 後停住，之後 1 秒 z 峰對峰 1.36 mm |
| 3 | 開迴路 CPG 走路 | 前進 **4.57 m** / 理論 4.61 m = **0.99**、不跌倒、FL 抬腳 0.083 m、末端機身高 0.250 m |
| 4/5 | Colab MJX 環境 smoke test | obs (69,)、reward 有限值；`mjx.put_model` 實跑通過；本機 CPU 以極小規模（num_envs=4、num_timesteps=40、num_evals=2）實跑完整 `ppo.train`，兩次 eval 皆通過並存出權重 |
| 6 | 本機 CPU 推論管線 | 可載入 brax 權重並輸出影片與指標 |

理論值的算法（trot 每個 CPG 週期有**兩個**站立相，漏掉這個 2 會把正常步態誤判成打滑）：

```
fx     = 2*(mu - MU_MIN)/(MU_MAX - MU_MIN) - 1          # mu=1.8 → 0.6
theory = 2 * (2 * D_STEP * fx) * omega * secs           # = 2 * 0.144 * 2 * 8 = 4.608 m
```

## 站姿的正常行為

home keyframe 的機身高度 0.2948 m 是**純運動學值**（四輪剛好觸地）。
實際站定後機身會沉降到 **0.2695 m**（`d1_model.NOMINAL_HEIGHT`）——
這是 kp=80 位置伺服在 20.56 kg 下的靜態撓度，**是物理不是 bug**；
沉降後 1 秒內 z 峰對峰僅約 1.4 mm。

訓練的高度獎勵用 `NOMINAL_HEIGHT`，**不是** `key_qpos[2]`。

## 已知限制

- **摩擦超過 1.0 後開迴路步態崩潰**（實測 1.5 → −0.85 m、3.0 → +0.19 m）。
  訓練的摩擦 DR 範圍是 [0.3, 1.0]，該範圍內全部正常（0.3/0.5/0.8/1.0 → 5.28/5.20/4.57/4.12 m）。
- `armature` / `damping` = 0.01 / 0.1 為**假設值**（URDF 未提供）。
- kp/kd 取自**點足版** demo（輪足版沒有 LowLevel demo）；已用 DR kp∈[60,100]、kd∈[0.5,2.0] 涵蓋。
- **MJX 不支援 (cylinder, box) 碰撞組合**，而 base 的碰撞 box 與四顆輪的碰撞 cylinder
  會產生 4 組 pair，`mjx.put_model` 會直接拋 `NotImplementedError`
  （只做 `MjModel.from_xml_path` 編譯是驗不出來的）。
  `d1_edu_w.xml` 因此加了 `<contact><exclude body1="base" body2="*_wheel"/>`。
  對 CPU 物理零影響——全關節行程隨機掃 32008 個姿態，base 盒與輪的最短距離 0.0348 m
  （margin 只有 0.001 m）；加 exclude 前後跑同一段 5 秒開迴路走路，qpos 軌跡逐步 bitwise 相同。
- 本機 CPU 場景刻意使用 MuJoCo 預設求解器迭代（100/50）。
  改用 MJX 的 1/5 會讓同一段開迴路只走 3.82 m（接觸約束不收斂造成穿透與打滑）。

## ⚠️ 常數是刻意的重複

`inference/d1_model.py` 與 `notebooks/cpg_rl_d1w_colab.ipynb`（Cell 4）各有一份常數，
因為 Colab 不 import 本地模組。**改一邊就必須改另一邊**，
否則訓練出的權重在本機推論會靜默走樣（維度對得上、行為對不上，沒有任何錯誤訊息）。
清單見 `docs/D1_EDU_W_規格與模型對照.md` §7。
