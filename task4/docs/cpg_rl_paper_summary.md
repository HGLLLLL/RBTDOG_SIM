# CPG-RL 論文標準版 結論總整

> 日期：2026-07-08　·　狀態：**成功（含抗擾、拖地已解）**
> 承接 5 維起步版（見 `cpg_rl_5dim_summary.md`），本版對齊 Bellegarda & Ijspeert (2022) 原論文，
> 目標：拿回完整表達力（側向、步態協調）、並從結構上解決 5 維版的**腳拖地**問題。

---

## 1. 一句話結論

**成功。** 論文標準版（12 維動作、2D 腳掌、腿間耦合振盪器、固定離地 g_c）在 Colab GPU(MJX) 訓練、回本機 CPU 推論，Go2 走得**又直又穩、腳確實抬離地面、且能扛住外力干擾**。相較 5 維版最大的進步是**拖地問題被結構性解決**。

---

## 2. 設計摘要（與 5 維版的差異）

| 面向 | 5 維起步版 | **論文標準版** |
|---|---|---|
| 動作 | 5 維（4 μ + 1 共用 ω）| **12 維：每腿 (μx, μy, ω)**，μ∈[1,2], ω∈[0,4.5]Hz |
| 腳掌平面 | 純矢狀面 x-z | **2D（含側向 y，用髖外展）** |
| 相位 | 固定 trot 偏移 | **腿間耦合振盪器**（Kuramoto，W_COUP=8 鎖 trot）|
| 抬腳高度 | `LIFT·amp·sin`（綁振幅→拖地）| **固定 g_c=0.08·sin（不乘振幅）** ★ |
| 觀測 | 51 維 | **76 維**（加腳觸地布林 + 每腿 rx,ṙx,ry,ṙy,sinθ,cosθ）|
| 逆運動學 | 2D 常數 Jacobian | **逐腿 3×3 數值 Jacobian**（含外展；左右腿正負自動處理）|
| 網路 | policy 128³ | policy (256,256,128) / value 256³ |

其餘沿用 5 維版：PD 對齊 task3（kp=90/kd=3、力矩上限 ±23.7/膝±45.43）、domain randomization（摩擦/PD增益/質量）、Brax PPO、Colab GPU 訓 → 本機 CPU 推論。

---

## 3. 為什麼論文版能解決「拖地」

5 維版拖地的根因：抬腳 `dz = LIFT·amp·sin θ` 被振幅縮放，且獎勵沒給抬腳誘因 → 平地上策略學成貼地滑行最省力。

論文版的腳掌 z 映射用**固定離地常數 g_c**（不乘振幅）：

```
dz = g_c·sin θ   （擺動相 sin>0）
dz = g_p·sin θ   （站立相，微量下壓）
```

於是**只要腿在擺動，CPG 結構就強制腳抬起 g_c**——策略只能決定「多快擺(ω)、跨多遠(μ)」，卻改不了離地高度，等於把「偷懶拖地」這個選項從根本拿掉。

---

## 4. 實測結果（本機推論，含側推干擾）

指令直走 0.6 m/s + 羅盤航向鎖定，並施加 5 次側推(±35N)+yaw 力矩(±10Nm)：

| 指標 | 數值 |
|---|---|
| 前進速度 | **≈0.50 m/s**（有干擾下，指令 0.6）|
| 橫向偏移 | **−0.02 m / 走 10 m**（幾乎完美直線）|
| 抗擾 | 扛住側推，航向鎖得很緊 |
| **擺動抬腳量** | **≈ 0.046 m（不拖地）** |
| 穩定性 | 不跌倒，高度穩定 0.24 m |

對照 5 維版（0.57 m/s、y 漂 0.32m/8.6m、無干擾、會拖地）：**論文版即使加了干擾，走得更直、且腳確實抬起**。

輸出：`outputs/cpg_rl_paper_infer.mp4`。

---

## 5. 對論文的忠實度與我們的取捨

**忠實處**：12 維動作 (μx, μy, ω)、二階振幅動力學(a=50)、腿間相位耦合、`f(r)=2(r-μmin)/(μmax-μmin)-1` 映射、固定 g_c 離地、觀測含 CPG 狀態與觸地布林。

**刻意的取捨/差異**：
- **模擬器**：MJX（MuJoCo）而非論文的 Isaac Gym（PhysX）。
- **機器人**：Go2 而非 A1。
- **IK**：用 task3 風格的**數值線性化 Jacobian**（home 附近）逐腿求解，而非解析 IK；近似但實測有效、且容易保證正確。
- **腳掌參考**：以「相對 home 腳位的偏移」表示（沿用 task3 慣例），而非論文的絕對 `-h` 基準。
- **Sim-to-real**：僅做到 sim-to-sim（MJX→本機 MuJoCo），尚未上真機（但 domain randomization 已鋪路）。

---

## 6. 品質保證（怎麼確認沒重蹈 5 維版的坑）

- **5 維版踩過的 5 個雷全數靜態核對通過**：CPG dx 符號、`state.pipeline_state`、metrics 含 `reward`、`MUJOCO_GL` 早於 import、`apply_pd` PD 對齊。
- **執行期驗證**：本機 `build_obs` 實測 = 76 維、與 notebook `_obs` 欄位順序逐項一致（sim-to-sim 安全）；訓練/推論網路結構一致（load_params 不失敗）。
- **CPG 數學先在本機開迴路驗證**（`paper_cpg_proto.py`）：前進 +1.84m/6s、不跌倒、FL 腳抬起 ≈0.053m——先確認方向與抬腳，才建 notebook。

---

## 7. 已知限制與後續可做

1. **僅 sim-to-sim**：尚未上真機；下一步可加大 domain randomization、加感測延遲/雜訊模型，往真機 A1/Go2 部署。
2. **步態固定 trot**：耦合目前鎖 trot；可放寬耦合/加指令讓策略學步態轉換（walk↔trot↔gallop）。
3. **平地為主**：可加地形高度觀測 + 崎嶇地形訓練（→ Visual CPG-RL 方向），提升 perceptive locomotion。
4. **能耗/自然度**：可加力矩/滑動懲罰進一步優化步態自然度與省電。

---

## 8. 相關檔案

- 訓練：`notebooks/cpg_rl_paper_colab.ipynb`
- 推論：`inference/local_infer_paper.py`
- 權重：`weights/cpg_rl_paper_params.pkl`
- 影片：`outputs/cpg_rl_paper_infer.mp4`
- 觀念/教學：`docs/hybrid_locomotion_tutorial.md`、`docs/cpg_rl_implementation_guide.md`
- 前一版：`docs/cpg_rl_5dim_summary.md`

---

### 總結

task4 從「hybrid 方法研究」一路做到「論文標準版 CPG-RL 訓練 + 本機部署」，完整驗證了 **model-based CPG + learning-based RL** 在你的硬體與既有 task3 技術棧上可行。論文版在保留 CPG 可解釋骨架的同時，靠 RL 取得抗擾與速度追蹤能力，並以固定離地的振盪器結構解決了拖地——是一個乾淨、可延伸（往真機/地形/多步態）的基礎。
