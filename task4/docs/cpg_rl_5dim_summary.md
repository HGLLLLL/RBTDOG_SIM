# CPG-RL 小規模測試（5 維起步版）結論總整

> 日期：2026-07-08　·　狀態：**成功（proof-of-concept）**
> 目標：先用最精簡的 CPG-RL，把「model-based CPG + learning-based RL」的 hybrid 走路流程**端到端跑通**，驗證可行性與整條工具鏈，再決定是否擴到論文標準版。

---

## 1. 一句話結論

**成功。** 用 5 維動作的簡化 CPG-RL，在 Colab GPU(MJX) 訓練、下載權重、回本機 CPU 推論，Go2 能**穩定前進、追上指令速度、走直線、不跌倒**。整條「Colab 訓練 → 本機部署」的 hybrid 流程完全打通，接在既有 task3 技術棧（Go2 + 軟體 PD + 羅盤）之上。

---

## 2. 這次做了什麼（設計摘要）

| 項目 | 內容 |
|---|---|
| 方法 | CPG-RL：RL 調變 Hopf 振盪器的振幅/頻率，CPG 產生腳掌軌跡 |
| 動作（5 維）| 4 條腿振幅目標 μ + 1 個共用頻率 ω |
| 觀測（51 維）| 重力方向、身體線/角速度、關節角/速度、指令、上一動作、CPG 狀態(r, ṙ, sin/cos φ) |
| 振盪器 | 二階振幅動力學（a=50）+ 固定 trot 相位偏移（無腿間耦合）|
| 腳掌→關節 | 重用 task3 的常數線性化 Jacobian（矢狀面 x-z）|
| 關節→力矩 | 軟體 PD，**kp=90 / kd=3、力矩上限 ±23.7/膝±45.43**（與 task3 完全對齊）|
| 訓練 | MJX + Brax PPO，Colab T4 GPU，~5000 萬步、數分鐘~十幾分鐘 |
| 抗擾 | domain randomization：地面摩擦、PD 增益、body 質量 |
| 部署 | 本機 CPU（jax 0.10.2 / brax 0.14.2）推論，接回 task3 迴圈 + 羅盤走直線 |

---

## 3. 實測結果（本機推論）

| 指標 | 數值 |
|---|---|
| 前進速度 | **≈0.57 m/s**（指令 0.6 m/s，追蹤良好）|
| 橫向偏移 | +0.32 m / 走 8.6 m（軌跡很直）|
| 穩定性 | **不跌倒**，身體高度穩定在 0.26 m |
| 權重載入 | brax 直接重建網路 + `load_params` 成功，無版本問題 |
| sim-to-sim | Colab(MJX) 訓 → 本機(MuJoCo) 推論行為一致 |

輸出：`task4/cpg_rl_infer.mp4`（走路影片）、`task4/cpg_rl_infer.png`（軌跡圖）。

---

## 4. 過程中發現並修掉的關鍵問題（重要經驗）

| 問題 | 症狀 | 修正 |
|---|---|---|
| **CPG foot 映射符號反了** | dummy 開迴路 1.3s 就跌倒（其實在往後推）| `dx = -STRIDE·amp·cos(θ)`；RL 無法自行修正方向(只設 μ,ω≥0)，這是**必修** bug |
| brax State 欄位名 | `AttributeError: 'State' has no attribute 'data'` | `state.data` → `state.pipeline_state` |
| metrics 鍵不一致 | 訓練 `scan` carry 結構不符（差一個 `reward`）| reset/step 的 metrics 都加上 `"reward"` 鍵 |
| Colab 無螢幕渲染 | `GLFWError: DISPLAY missing` | `MUJOCO_GL=egl` 要在 `import mujoco` 之前設；或直接改在本機用 `local_infer.py` 渲染 |
| MJX 致動器不同 | `go2_mjx.xml` 是位置伺服、task3 是力矩 | 用 `apply_pd()` 把訓練 PD 對齊成 kp=90/kd=3（位置伺服 ≡ 軟體 PD）|

---

## 5. 已知限制（→ 這些正是升級論文版的理由）

1. **腳會拖地**：抬腳高度 `dz = LIFT·amp·sin θ` 被振幅綁住，且獎勵沒給抬腳誘因 → 平地上策略學成省力的貼地拖行。**論文版用固定離地量 g_c（不乘振幅）可從結構上解決。**
2. **只會 trot**：無腿間耦合、用固定相位鎖死，學不了步態轉換。
3. **純矢狀面**：只有 x-z，沒有側向 y（無 μ_y、不用髖外展）。
4. **近似 IK**：只在 home 姿態線性化一次，大幅擺動時精度差。
5. **只做到 sim-to-sim**：尚未上真機（但已加 domain randomization 為 sim-to-real 鋪路）。

---

## 6. 驗證了什麼、下一步

**驗證了**：hybrid（CPG + RL）在你的硬體/工具鏈上完全可行；「Colab 訓一次 → 本機永久推論」的路線成立；PD 對齊 + obs 逐項一致是 sim-to-sim 成功的關鍵。

**下一步**：做**論文標準版**（12 維動作 μx/μy/ω、腿間耦合、2D 腳掌、固定離地 g_c、含觸地布林觀測），可同時解決拖地、獲得側向/步態彈性、更貼近可 sim-to-real 的水準。→ 見 `cpg_rl_paper_colab.ipynb`。

---

## 相關檔案

- `cpg_rl_colab.ipynb` — 本次 5 維訓練 notebook
- `local_infer.py` — 本機推論（接 task3 + 羅盤）
- `cpg_rl_implementation_guide.md` — 完整 step-by-step 教學
- `hybrid_locomotion_tutorial.md` — hybrid 方法觀念白話版
