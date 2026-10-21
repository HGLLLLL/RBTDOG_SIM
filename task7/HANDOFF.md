# task7 交接：D1 Max 現況卡

- 最後更新：**2026-10-21 上午（v1f 及格線全過但使用者看影片抓到平移「腿外張、右平移步頻快」→ 量測＝兩腿踏、兩腿被拖著橫滑 0.5 m/s；不上機，改 v1g）**（分支 `main`）
- **這份只放「現在」，上限 150 行。** 收工時覆寫本檔，不是往上疊。
  被取代的段落搬到 `docs/handoff_history/`（2026-08-25～09-22 全文在 `HANDOFF_2026-08-25_2026-09-22_全文.md`；
  09-22 §1 在 `HANDOFF_2026-09-22_§1_v3.7b_Colab訓練.md`；10-06 §1 trip25 在 `HANDOFF_2026-10-06_§1_trip25.md`；10-14 §1 trip26／27 在 `HANDOFF_2026-10-14_§1_trip26-27收工.md`。查歷史用 grep）。
- 規格／SHM／座標／IMU 慣例／原廠參數 → `docs/D1Max_機器狗資訊.md`（95 KB，查章節）。目錄地圖 → `README.md`；文件索引 → `docs/README.md`；每次上機 → `docs/現場SOP_通用_2026-08-27.md`。

---

## 1. 下次第一件事：**v1g＝修掉 v1f 的「拖腿橫滑」**（spec 待寫；trip29 卡 §9 保留，換 npz 即可）

- **v1f 不上機（10-21 使用者看影片）**：平移時「腿外張、右平移步頻快」。量測（vy ±0.5）：**只有兩～三條腿踏步，右前／右後（左平移）或右前（右平移）全程觸地 99–100%、被拖著沿輪軸橫滑 0.49–0.53 m/s（＝整個機身速度）**；v1b 四腿都踏（觸地 56–80%、滑 0.13–0.33）。外張＝被拖的腿 ABAD 外張 8°（FR −8.0／RR −7.9；§5 承重外張 ±4.5° 的線）。左平移 2 腿踏、右平移 3 腿踏 → 使用者看到的步頻差。**機制**：roll＋impact 讓「四腿踏步」每一步都要付錢，而 `slip`（0.075）太便宜：拖兩輪橫滑 0.5 m/s 每步只扣 0.04，比抬腳（impact）＋重心轉移（roll）便宜 → 膝 hinge 擋掉鎖膝後，policy 走到下一條最便宜的路。實機上拖著承重輪橫滑是高摩擦、會抖、會磨胎，不能上。
- **policy 對稱度實測**：|π(鏡像 o) 翻回 − π(o)| 平均 v1b 0.41、v1f 0.39（動作尺度 0.5–1）→ **鏡像增強 5 億步沒讓網路變對稱**，roll 左右一樣小是 roll 懲罰＋拖腿步態的結果。要真對稱得用架構對稱化 π_sym＝½[π(o)＋M π(M o)]（狗端要改 policy_np 兩次前向）。
- **v1f 拖腿的量測細節（10-21）**：決定變數是 |vy| 不是有沒有 vx——vy 0.5（vx 0／0.1／0.3 都同）FR＋RR 全程觸地 0.97–1.0、橫滑 0.5、膝 52–61°；vy 0.3（vx 0／0.3／0.5 都同）只 FR 全程觸地、膝 73–81° 看起來正常；v1b 任何指令四腿 18–22 次。直腿外張＝拖腿的幾何必然（機身以 0.5 m/s 離開被釘住的腳，腿要伸直外張才搆得到）。原廠平移四腿都踏、次要腿「原地重放」抬 8–10 mm（L 文件）→ 修動機（slip）不修症狀（膝 hinge）。
- **v1g 方向（待使用者定）**：讓拖腿比踏步貴——只改 `slip` 0.075 → 0.75（v1f 拖腿 raw ≈ 0.5／步 → −0.4、v1b ≈ 0.15 → −0.1）；**不加 ABAD hinge**（原廠平移 ABAD 擺幅 11–20°，會壓掉側向跨幅；使用者定）；膝 hinge 維持 60°（70° 會扣到 v1b 自己 61–67° 的腿）。驗收加欄：平移每腿落地次數 min／max ≥ 0.5（vy 0.3 與 0.5）。**開訓前關卡**：在新 reward 下攤帳，v1b 的四腿步態每步總分必須高於 v1f 的拖腿步態，否則權重再調。

- **v1f 及格線全過**（Z 文件 `docs/results/Z_純RL自訓v1f_MuJoCo及格線_2026-10-21.md`）：對稱 vy ±0.5 roll 峰 **2.1／1.7°**（v1b 9.3／3.6）；落地 vz **0.10／0.17**（v1b 0.36／0.37）；平地 13／斜走 9／延遲雜訊 8／崎嶇地 114 全 0 摔（v1b 崎嶇 1 摔）；直走 0.33／0.54／1.05 準（v1d 的 +0.1 超速消失＝延遲 DR 造成）；wz 26／57；護欄內（延遲雜訊下 ABAD 誤差峰 0.68 貼 0.70）。
  姿態：機身高與前後晃回到 v1b 水準（0.54 m、pitch std 0.6–1.0°）；膝 55–60°（貼 hinge）比 v1b 的 64–80° 直一點、髖較垂直、抬腳 5–9 mm（v1b 13–20）。**使用者看 `outputs/rl_wheel/compare_v1b_vs_v1f_mesh.mp4` 決定**：可接受就上機；要更蹲就 `knee_min_deg` 60 → 70 ＋ 髖 hinge 再訓（50 分）。
- **v1f 是什麼**：`WheelEnv(**rw.V1F)`＝v1b 熱啟動＋鏡像 50%＋roll 10＋impact 3（門檻 0.3 m/s）＋IMU 偏差 ±2° DR＋`angxy` 0.5（罰 roll／pitch 角速度）＋`knee`（膝比 60° 更直才罰，w 0.5）；**無延遲 DR**。狗端零改動，`--grav-offset 0.7 -1.5` 照舊，npz sha `5370132e3f15`。
- **三版教訓（Z §4）**：v1c 從頭訓＋無門檻 impact → 2,650 萬步內踏步被清零、側向軸死；v1d 延遲 DR → 多給命令（直走 +0.1）、鎖膝（42°）、前後晃 3 倍；只罰 roll 會把重心轉移推到 pitch，用角速度項補。從頭訓若再做：impact 要 0→3 暖身。
- **上機（trip29，§9 卡）**：站立 → 直走 0.3／0.5 → 轉 ±0.5／±1.0 → 平移 ±0.5（**重點：左右 roll 一樣小、踏地聲比 v1b 輕**；`CGB` 旗標可當場跑 v1b 對照）→ 斜走 ±0.3 → 平移 0.6。狗端 log 先 `clean_dog_logs.sh`。
- 影子跟跑 `test_policy_rl_{np,closed_loop}.py` 是給學長 ONNX 版的（要 ONNX 對照），自訓 npz 不適用；匯出一致性由 `test_export_policy_wheel.py` 蓋。
- **全套 pytest @af165f8（knee／w_angxy 改動後）：1121 passed, 1 warning（jaxopt），35 分**（10-21 上午）。
- **工具坑**：`conda run -n rbtdog python - <<EOF` 不會把 heredoc 餵進 stdin，python 片段先寫成檔案；`--cmds` 以負號開頭要寫 `--cmds=-0.3,...`。

## 2. 權重現況

| 權重 | 狀態 | 驗收表（`outputs/`） |
|---|---|---|
| **rl_wheel_v1f**（5 億步，v1b 熱啟動）`weights/rl_wheel_v1f_params.pkl`／`rl_wheel_v1f_np.npz`（sha `5370132e3f15`） | 及格線全過但**不上機**：平移是兩腿踏、兩腿拖著橫滑 0.5 m/s（表上看不出，影片看得出）；roll 2.1／1.7°、落地 0.10／0.17、崎嶇地 0 摔、直走準 | `rl_wheel/eval_rl_only_real_{v1f,diag_v1f,diag_d1n_v1f,vxlow_v1f}.md`、`eval_rl_wheel_rough_real_v1f.md`；**Z 文件** |
| rl_wheel_v1d（2.12 億步中場）`weights/rl_wheel_v1d_s212_np.npz`（sha `652aecce2ed3`） | 對稱 roll 1.6／2.7°、落地 0.22／0.17、rough 0 摔 **但** 平移膝 42°、pitch 晃 3 倍、直走 +0.1（§1）；不上機 | `rl_wheel/eval_*_v1d_s212.md`、`compare_v1b_vs_v1d_s212_mesh.mp4`（本機） |
| rl_wheel_v1c（從頭訓 5 億，**失敗**）`weights/rl_wheel_v1c_{params.pkl,np.npz}`（sha `06d649a136c3`） | 側向軸死掉、不抬腳（§1）；留作對照 | `rl_wheel/eval_rl_only_real_{v1c_s1,diag_v1c_s1}.md`、`eval_rl_wheel_rough_real_v1c_s1.md`、`reward_audit_wheel_v1c_s1.md` |
| **rl_wheel_v1b**（自訓，MJX，10 億步）`weights/rl_wheel_v1b_params.pkl`／`rl_wheel_v1b_np.npz`（sha `a02a4631d3f3`） | ★ **已上機（trip26／27／28）**：trip26＋27 24 趟 0 摔，trip28（10-20）主管實體看步態；後輪不碰、慢轉不飄、vy 0.3 會動；`--grav-offset 0.7 -1.5` 標準旗標；弱點＝左平移 roll 7–10°、平移落地重。模擬：斜走 9 條、倒退 5 條 0 摔（`rl_wheel/eval_rl_only_real_{diag,back}_*_v1b.md`） | `rl_wheel/eval_rl_only_{real,ideal}_v1b.md`、`eval_rl_wheel_rough_real_v1b.md`、`eval_rl_only_real_landcol_v1b.md`（落地 vz 欄）；X 文件 |
| rl_wheel_v1（5 億步）`weights/rl_wheel_v1_params.pkl`／`_np.npz`（sha `83220469ef90`） | 被 v1b 取代。 | `rl_wheel/eval_rl_only_*_v1.md`；W 文件 |
| **RL_only 10k** `weights/rl_only_2026-10-01_10k_np.npz`（sha `5ec719a97175`） | ★ 已上機（trip25）11 趟 0 摔；平移後輪互碰、慢轉飄 → 被 v1b 取代；落地最輕（每次 0.10–0.19 m/s）當 `impact` 項的對照 | `rl_only/eval_rl_only_{real,ideal}_10k.md`、`eval_rl_only_rough_real_10k.md` |
| RL_only 5k／0924 | 對照組／trip24 已上機 12 趟 0 摔 | `rl_only/eval_rl_only_*_5k.md`／T 文件 |
| **v3.6f** `weights/cpg_rl_v3_6f_params.pkl`／`cpg_rl_v3_6f_np.npz`（sha `cac8679831c3`） | CPG-RL 線可部署最佳（`--mirror --mirror-lat`），trip22／23 驗過；線暫停 | `eval_cpg_rl_v3_6f_params_mirrorall.md`、Q／R 文件 |
| v3.7f／v3.5f／v3.7bf | 不用／上一代／沒訓 | — |

- 報告影片（本機，mp4 被 .gitignore）：`outputs/rl_wheel/report_v1b_mesh.mp4`、`rough_compare_10k_vs_v1b_mesh_v2.mp4`（報告用）、`diag_v1b_mesh.mp4`（斜走六段，10-20）；各有 `_small`。
- ⚠️ 2026-09-22 之前的 `outputs/eval_*.md` 都含每 2 s 的訓練推力。

## 3. 工具一行版

- **自訓環境** `inference/rl_env_wheel.py`（`WheelEnv`、`V1C` 預設字典、`make_domain_randomize`、`wrap_env`；鏡像表 `mirror_obs/act/priv/data`；`IMPACT_SETTLE` 25）；地形 `wheel_terrain.py`；對齊／攤帳 `diag/wheel_env_align.py [--preset v1c] [--npz …] --tag …` → `outputs/reward_audit_wheel_<tag>.md`（每次落地 vz ＝ 欄 `impact_vz`／`td_n`，不能只讀 `impact_vz`）；匯出 `export_policy_wheel.py --params …`；notebook 產生器 `notebooks/build_nb_wheel.py`（寫 v1 與 v1c 兩本）。
- **golden** `tests/test_rl_env_wheel_golden.py` 釘住 `WheelEnv()`（＝v1b）長回合＋短回合（重生／推力／指令切換）逐位元；失敗＝預設路徑被動到，查清楚再決定，不放寬。`tests/test_rl_env_v3_golden.py` 釘 v3.3。
- **驗收工具** `rl_only_infer.py --npz … --self-collision [--delay N] [--noise] [--push 0.5] [--grav-bias R P] [--grav-offset R P] [--video … --mesh]` → `outputs/rl_wheel/eval_rl_only_{real,ideal}_<tag>.md`（**新欄「落地 vz m/s（次）」**＝觸地前一步下落速度，`landing_vz()`）；崎嶇地 `diag/rl_only_rough_terrain.py --npz … --self-collision --tag …`（同樣多一欄）；並排影片 `diag/rl_only_rough_compare_video.py`。
- **純 RL 匯出** `inference/export_policy_rl.py`；影子跟跑 `RL_ONLY_NPZ=… pytest tests/test_policy_rl_{np,closed_loop}.py`。
- **上機** `realbot/M9_gait.py --v3 PATH.npz --interactive --cmd vx vy wz …`（rl16 強制不鏡像；`--wheel-mode kd1`、`--vmax 18`、`--grav-offset 0.7 -1.5` 必帶；READY 站好讀值差 > 1° 就改當天值）；操作卡 `docs/現場操作卡_純RL自訓v1b_trip26_2026-10-14.md`（§7 trip27、§8 trip28 展示含斜走）。
- **實機 log 分析** `inference/diag/rl16_trip_report.py`（⚠️ 抬腳欄偏置待修，X §5）；里程法只能同腿前後比（X §3／§6）。
- **CPG-RL v3 線**（暫停）`inference/local_infer_v3.py --preset … --mirror --mirror-lat`；`export_policy_v3.py`。
- **給同事的獨立 repo `~/d1max-rl-deploy`**：commit 不加 AI 署名；改過的驗收工具要同步搬。
- 全套測試 `conda run -n rbtdog python -m pytest task7/tests -q`（約 30 分）。

## 4. 未解／待決（依優先序）

0. **v1c**：Colab 開訓 → 中場檢查 → 及格線 → Z 文件 → trip29（§1）。plan 的 Minor 審查項（測試覆蓋：鏡像＋延遲同開、重生後 delay 重抽、對稱化測試的覆寫範圍、IMU 偏差 .T 方向）記在 `.superpowers/sdd/progress.md`，最終審查修正波處理；最終審查 Minor 清單在 `.superpowers/sdd/progress.md`。
0′. **模型左右不對稱（10-20 量到）**：`base_link` 質心 y +3.45 mm、RAR 膝 0.8598 vs 其他 0.8631 kg、**機身慣量主軸 iquat 有小傾斜（影響比質心偏移大）**；質心在 DR ±50 mm 內、慣量傾斜 DR 沒隨機化。訓練模型沒改。要不要把慣量姿態加進 DR → 使用者定。測試 `tests/test_rl_env_wheel_mirror.py::test_model_lr_asymmetry_is_within_dr`。
0″. `rl16_trip_report.py` 抬腳欄要修（X §5）；里程法併入。
1. **M9 硬上限誤中止**（trip25 ④）：`tau_hot` 連續 2 筆要加「相隔 ≥ 4 ms」（`M9_gait.py` 約 2075 行，`test_m9_gait.py` 要一起改）。
1′. 影子跟跑 Δact 門檻放寬到 1e-3（U §2）。
2. **CPG-RL v3 線（暫停）**：要重啟看 T／R 文件與 v3.8 草案。
3. **狗端 `~/m_logs` 沒清**（trip26／27／28 混舊檔）→ 下次上機前 `bash task7/realbot/clean_dog_logs.sh` 看一次再 `--delete`；本機 `m_logs_trip22/`、`m_logs_trip24/` 舊檔待清（09-30 指令已給）。
4. repo 根目錄 `RL_only/`（學長 onnx／pt，7.5 MB）、`1014_ref.txt`、`1014_reply.txt` 未入版控、位置不對；`RL_only/` 被工具當預設路徑引用，搬之前要改 `rl_only_infer.py`／`wheel_env_align.py`／`rl_only_rough_terrain.py` 的預設。

## 5. 硬限制與防呆（不隨版本變）

- 輪速伺服 **kd ≤ 1.0**；步態段一律 `--wheel-mode kd1`；RL 趟必帶 `--vmax 18`；ABAD 承重外張 ±4.5°。
- 誤差護欄：純 RL 10-01 版起 `--emax-abad 0.80 --emax 0.55`；v3 `0.91／0.55`。
- MJX 3.10 **無 hfield–cylinder 碰撞**（訓練輪子是膠囊；pair 圓盤在輪胎外側、碰不到實機內面互碰，靠 `wgap`）；Brax AutoReset 不重呼叫 reset（重生在 `WheelEnv.step` 內）。
- **模型左右不對稱**（§4 0′）：鏡像等變性只在對稱化副本上成立（4 步 qpos 3e-5）；hfield 接觸點數鏡像兩邊可差 1。鏡像增強把這些當雜訊。
- **診斷量口徑**：每次落地 vz ＝ Σ`impact_vz`／Σ`td_n`，除步數會被無落地的步稀釋（10-20 審查抓到）；出生後 25 步的落地不算事件。
- 「輪子的圖案」和「腿的剛度」耦合：換增益不能只換增益。停損線要信。
- 上機每次：`push_to_dog.sh` 看到「✅ 全部一致」才往下；兩個終端機，一個備 `sudo ~/estop_max.sh`；收工 `pull_from_dog.sh tripNN` 一定帶編號；吊帶 `H` < 292 mm（實務 260）；每趟跑完 `sudo kill -CONT <PID>`。
- `outputs/stale/` 不推上狗；狗上走 `/dev/shm/{joint_cmd,joint_state,imu_central}`；推論排 cpu 0–5。
- 實機 log 讀法：200 Hz `samples[].j[name] = (q, q_des, τ, v)`、`w[name] = (pos, v, effort)`；50 Hz `policy.log[].imu = [quat xyzw, gyro]`；IMU **+roll ＝ 右側低**。
- 操作卡的預期值只能來自驗過的模擬表；實機異常先問「模擬同指令＋1° 側傾會不會也這樣」。

## 6. 報告偵察線（2026-09-22 完成，不阻塞）

`docs/results/N_感測器與資源實測_2026-09-22.md`、`O_導航與AI架構_2026-09-22.md`。

## 7. 里程碑（一行版）

08-27 承重站立 → 09-02 第一次走路 → 09-09 RL v2.3 首次上機 → 09-15 v3.1 統一產生器 → 09-22 v3.6f → 09-23／29 v3.6f 上機（trip22／23）→ 09-29 主線換純 RL → 09-30 純 RL 首次上機（trip24）12 趟 0 摔 →
10-06 10k 上機（trip25）11 趟 0 摔 → 10-13 自訓 v1／v1b → 10-14 v1b 上機（trip26＋27）24 趟 0 摔、`--grav-offset` 定案 → **10-20 trip28 主管實體看步態；v1c 環境（鏡像＋roll＋impact＋延遲／IMU 偏差 DR）完成、攤帳關卡過；從頭訓 5 億步側向軸死掉 → v1d＝v1b 熱啟動＋impact 門檻，等 Colab**。
