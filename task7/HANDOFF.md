# task7 交接：D1 Max 現況卡

- 最後更新：**2026-10-21 深夜（v1i 2.12 億步中場：姿態完全回到 v1b 範圍（腳在髖下 ±4 cm、腳距 69／78、膝 62–70°）、左 roll 3.3°／右 1.0°、落地 0.14／0.18、vy 0.3 四腿踏；右前腿在右平移仍釘；等使用者看影片定上機）**（分支 `main`）
- **這份只放「現在」，上限 150 行。** 收工時覆寫本檔，不是往上疊。
  被取代的段落搬到 `docs/handoff_history/`（2026-08-25～09-22 全文在 `HANDOFF_2026-08-25_2026-09-22_全文.md`；
  09-22 §1 在 `HANDOFF_2026-09-22_§1_v3.7b_Colab訓練.md`；10-06 §1 trip25 在 `HANDOFF_2026-10-06_§1_trip25.md`；10-14 §1 trip26／27 在 `HANDOFF_2026-10-14_§1_trip26-27收工.md`。查歷史用 grep）。
- 規格／SHM／座標／IMU 慣例／原廠參數 → `docs/D1Max_機器狗資訊.md`（95 KB，查章節）。目錄地圖 → `README.md`；文件索引 → `docs/README.md`；每次上機 → `docs/現場SOP_通用_2026-08-27.md`。

---

## 1. 下次第一件事：**使用者看 `outputs/rl_wheel/compare_v1b_vs_v1i_s212_mesh.mp4` 定 v1i 上機**（trip29 卡 §9 換 `rl_wheel_v1i_s212_np.npz` sha `2bc2c3d2bc72`；Colab 5 億步跑完的 params 可再驗一次，2 億後曲線已平）

- **v1i s212 結果**（AA 文件 §6）：姿態 ✅ 腳相對髖 ±4 cm／腳距 69／78 cm／髖 30°／膝 62–70°／機身 0.55／pitch std 0.6–0.8；對稱 ✅ 左 roll 3.3°（v1b 9.3）、右 1.0°；落地 ✅ 0.14／0.18；vy 0.3 四腿 17/17/17/17；左平移四腿 16/17/17/17；斜走 9、延遲雜訊 8 全 0 摔；崎嶇地 1/125（boxes 10 側移，v1b 同格）；直走準。**仍在**：右平移右前腿釘（1 次、觸地 100%、橫滑 0.51；使用者 v1h 時說可接受）；wz 0.5 兩腿釘（v1b 也弱）；延遲雜訊下 gap 最低 130、wz −1.0 gap 187（< 200，v1b 213）。
- **原廠對照（AA §7，`inference/diag/posture_vs_oem.py`）**：原廠遙控器站立 0.534 m、平移 0.545 m／膝 64–67°／腿伸長 0.45；v1i 平移 0.55／膝 62–70°／0.45–0.46 → **不偏高、不過直**；差在 ABAD 原廠四腿對稱外張 ±5–7°、v1i 同號 −2～−6°（整排偏一側）＋後腿髖左右不一。v1b 平移 0.53 比原廠低（帶 roll）。`h_max` 0.56 不必降；要降是「比原廠低」的選擇（使用者定）。
- **上機要看的**：右前腿在實機右平移時有沒有拖地聲；左右平移側傾肉眼是否一樣小；踏地聲比 v1b 輕。wz ±1.0 後輪 gap 187 → 原地快轉前後輪會比 v1b 近一點，聽有沒有碰。

- **v1h 影片問題（使用者 10-21 傍晚）**：右前腿釘著可接受（有抬一點），但平移時**前腳往前、後腳往後伸開**。量測（平移 0.5）：前腳在髖前 14–21 cm、後腳在髖後 9–17 cm、前後腳距 95–98 cm（v1b 60–75、站立 65）、髖角 ±46° 開到 7–19°；v1d 75–88 → v1f 90–96 → v1h 95–98，**每加一道姿態約束就伸多一點**。機制：站高、鎖膝、拖腿都被封後，policy 用拉長支撐面讓踏步時機身不晃（roll／pitch 角速度都在罰）；髖開＋膝 65° 的髖到腳距離仍長＝繞過膝 hinge 的直腿，`dev` 對髖只有 0.1/rad 擋不住。
- **v1i**：`WheelEnv(**rw.V1I)`＝V1H ＋ **`splay`**（w 3）罰腳相對髖在機身 x 方向超過 **0.08 m** 的部分（v1b 平移 ±8 cm 內、站立 ±1 cm）——約束物理量不約束關節角，髖膝怎麼配都繞不過。關卡（`reward_audit_wheel_{v1b,v1h}_v1i{,_rough9,_boxes9,_stairs9}.md`）：13 列 v1b 贏 10；`splay` v1b 全 ≈ 0（最多 −0.07）、v1h 平地／rough −0.33～−0.89；輸的 3 列同 v1h 關卡（boxes9 vy 0.3 −1.12、stairs9 +0.5 −0.80、−0.5 −0.14，v1b 自己在第 9 級追蹤差／落地重），照前例接受。訓練盯 `splay`（v1b 起點 ≈ 0）、`knee`、`height`、`stuck`。訓完量：前後腳距 ≤ 75 cm、腳相對髖 ±8 cm、膝 ≥ 65、機身 ≤ 0.56、每腿落地、roll、落地 vz。

- **v1h 結果**（AA 文件 §5；表 `outputs/rl_wheel/eval_rl_only_real_{v1h,diag_v1h,diag_d1n_v1h}.md`、`eval_rl_wheel_rough_real_v1h.md`）：姿態 ✅ 膝 65–74°／機身 0.51 m／pitch std 0.3–0.7°／ABAD 0–5°；落地 ✅ 0.19／0.18；左平移 ✅ 四腿 14/14/14/15、roll 1.1°；崎嶇地 ✅ 0/125；直走準；延遲雜訊 0 摔。**未解**：右平移右前腿 5 次（觸地 96%、橫滑 0.44）、roll 4.5°（v1b 3.6；左右差 3.4° > 門檻 3）；vy 0.3 右前 0 次；wz 0.5 只兩腿踏（v1b 也差）。同一條腿（FR）在 v1f／v1g／v1h 都是弱的那條，policy 鏡像殘差 0.4 沒變 → 網路本身不對稱，stuck（每腿上限 1／步）它在那幾個情境選擇付。
- **再修的選項**：(a) `STUCK_CAP` 1 → 3（付不起）；(b) 架構對稱化 π_sym＝½[π(o)＋M π(M o)]（真對稱；狗端 policy_np 兩次前向，匯出要改）；(c) 先上機，右前腿在實機上拖不拖用耳朵和 log 判。

- **v1h（使用者定「1＋2」，10-21 下午）**：`WheelEnv(**rw.V1H)`＝V1G ＋ 膝 hinge **65°／w 2**（70° 讓 v1b 自己平地扣 −0.4、樓梯 −0.6，使用者改 65）＋ 新項 **`height`**（機身對地形高 > 0.56 m 線性罰 w 10）＋ `roll` 10 → **5**。impact 3／門檻 0.3、stuck、slip 不動。
  **關卡（`reward_audit_wheel_{v1b,v1g}_v1h{,_rough9,_boxes9,_stairs9}.md`，工具新增 `--terrain KIND:LEVEL`、`--set k=v`）**：13 列 v1b 贏 10 列——平地 4 列差 0.6–1.6、rough9 三列差 0.8–1.4、boxes9 ±0.5 差 0.6–1.1、stairs9 −0.5 差 0.1；**輸 3 列**：boxes9 vy 0.3（−0.54）、stairs9 vy +0.5（−0.39）、stairs9 vy 0.3（−0.12）。試過 impact 3 → 2 沒翻盤（−0.51／−0.24／−0.07）：輸的主因是 v1b 在第 9 級地形上本來就追蹤差（boxes9 vy 0.3 的 lin 只 1.63）、落地重（stairs9 impact −0.4～−0.7），v1g 在那裡的優勢有一部分是真的。第 9 級是訓練分布少數（平均等級 2.4），**接受這三列、impact 維持 3**（控制者判斷；使用者定 65°）。
  訓練盯：`knee`（v1b 起點約 −0.1，往 0）、`height`（≈ 0）、`stuck`（≈ 0）。中場（2.1 億步）看：膝 ≥ 65°、機身高 ≤ 0.56、每腿落地 min／max ≥ 0.5、roll 峰 ≤ 4°、落地 vz ≤ 0.25。

- **v1g s212 中場（`weights/rl_wheel_v1g_s212_np.npz` sha `d8e06a034709`；表 `outputs/rl_wheel/eval_rl_only_real_{v1g_s212,diag_v1g_s212}.md`、`eval_rl_wheel_rough_real_v1g_s212.md`、`reward_audit_wheel_v1g_s212.md`）**：vy 0.3 四腿 16／16／14／16 ✅；vy +0.5 **4／15／15／16**（FR 觸地 98%、橫滑 0.49、最長連續觸地 2.4 s）、vy −0.5 **0／18／18／19**（FR 7 s 沒抬）、wz 0.5 **1／20／0／1**（三腿 7 s 釘著）；膝 50–58°（比 hinge 60° 更直，每步付 knee −0.24）、機身 0.57 m；roll 峰 3.6／1.4°、落地 0.28／0.22；直走準；boxes 10 側移 **摔 2/3**（v1b 1、v1f 0），其餘地形 0 摔。**不是點一下歸零，是硬付錢**：MJX 攤帳 stuck 在 wz 0.5 −1.73、vy −0.5 −0.73（vy +0.5 只 −0.03，FR 在 MJX 微抬 9 mm、CPU 6 mm 貼門檻）。
  **關鍵發現**：在 V1G 下、平地固定指令，**v1b 的步態每步總分都高於 v1g 自己**（vy +0.5 3.87 vs 3.68、wz 0.5 3.72 vs 2.37、vy −0.5 3.73 vs 2.98），但 Colab 訓練 eval 的 R 從 2624 升到 2900 → v1g 在**訓練分布**（地形 10 級為主、推力、IMU 偏差、鏡像、隨機指令）上比 v1b 好，在平地上比 v1b 差。最可能：崎嶇地上踏步的 impact 是平地的 3–4 倍（最終審查量過 stairs 8 側移 −0.75／步），policy 在地形上學會少踏步、帶回平地。**平地攤帳關卡抓不到這個，關卡要加地形列（fixed_tile）**。
- **候選（待使用者定）**：(A) 加正向的每腿騰空時間獎勵（legged_gym `feet_air_time`：每次落地給 swing 時長 − 0.5 s 的分，四腿都要踏才拿得到）直接對沖 impact 的反踏步壓力，`stuck` 淨空門檻 5 → 10 mm 防貼門檻；(B) 降 impact 壓力：w 3 → 1.5、門檻 0.3 → 0.4（放棄一部分落地輕）；(C) A＋B。無論哪個：關卡加 rough／boxes／stairs 列，v1b 要在每列都贏。

- **v1g**：`WheelEnv(**rw.V1G)`＝V1F ＋ `w_slip` 0.75 ＋ **`stuck`**（w 1.0；側移 |vy|>0.1 或旋轉 |wz|>0.3 指令下，任一腿足端淨空 < 5 mm 連續超過 0.6 s 就罰超過秒數、上限 1 s；直走／站立不罰；用淨空不用接觸旗標）。原廠平移四腿都踏、次要腿每週期原地重放 8–10 mm／0.2 s、週期 0.47 s（L 文件）→ 不觸發；淨空門檻 5 mm 是為了把那種低抬腳算成「有抬」。impact 3／門檻 0.3、膝 hinge 60°、不加 ABAD hinge（使用者定）。
  **關卡（含 stuck 的 V1G，`outputs/reward_audit_wheel_{v1b,v1f}_v1g.md`）**：每步總分 v1b vs v1f ＝ vy 0.3 4.09 vs 3.26、vy +0.5 3.87 vs 2.78、vy −0.5 3.73 vs 2.70；v1b 的 `stuck` 全 0、v1f −0.93～−1.0 → **過**。進度列多印 `stuck`（應全程 ≈ 0）、`slip`。訓完新門檻：平移每腿落地次數 min／max ≥ 0.5（vy 0.3 與 0.5，驗收表「落地 vz（次＝FR/FL/RR/RL）」欄）、膝 ≥ 60°、抬腳峰 ≥ 8 mm（防「幾乎不離地的小抖動」第三條路）。

- **v1g 第一版（只改 `w_slip` 0.075 → 0.75，env／notebook `rl_wheel_v1g_colab.ipynb` 已做、測試過，10-21 中午）關卡結果**（`outputs/reward_audit_wheel_{v1b,v1f}_v1g.md`）：V1G 下每步總分 v1b 四腿 vs v1f 拖腿＝vy +0.5 3.87 vs 3.76、vy −0.5 3.73 vs 3.63（只勝 0.11）、**vy 0.3 4.09 vs 4.26 v1f 反勝** → **不開訓**。原因：v1b 四腿步態自己也滑 0.13–0.17 m/s，slip 區分力弱；MJX 裡被拖的輪子接觸斷續（touch_n 2.3 vs CPU 3.5），slip／touch 都被低估。slip ×30 也救不了 vy 0.3 列（算過）。
- **提議（待使用者）**：加 `stuck` 項——每腿用足端淨空 < 8 mm 算連續站立時間，在 |vy_cmd| > 0.1 或 |wz_cmd| > 0.3 時超過 0.6 s 就扣（每腿每步 −0.5）；原廠／v1b 每 0.4–0.5 s 抬一次不觸發，v1f 拖腿 −1.0／步。用淨空不用接觸旗標。`slip` 0.75 留著。驗收表已加「每腿落地次數」欄（`landing_counts`）。

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
- **全套 pytest @af165f8：1121 passed（10-21 上午）；@4cc7734（v1h：stuck／height 項後）：1128 passed, 1 warning**（10-21 晚；v1i 的 splay 只分檔跑過 26＋10＋31 passed）。
- **工具坑**：`conda run -n rbtdog python - <<EOF` 不會把 heredoc 餵進 stdin，python 片段先寫成檔案；`--cmds` 以負號開頭要寫 `--cmds=-0.3,...`。

## 2. 權重現況

| 權重 | 狀態 | 驗收表（`outputs/`） |
|---|---|---|
| **rl_wheel_v1i**（2.12 億步中場）`weights/rl_wheel_v1i_s212_np.npz`（sha `2bc2c3d2bc72`） | ★ **候選上機版**：姿態回 v1b（腳距 69／78、膝 62–70）、roll 3.3／1.0°、落地 0.14／0.18、vy 0.3 四腿踏、崎嶇地 1/125；右前腿右平移仍釘（AA §6） | `rl_wheel/eval_*_v1i_s212.md`、`compare_v1b_vs_v1i_s212_mesh.mp4`（本機） |
| rl_wheel_v1h（5 億步，v1b 熱啟動）`weights/rl_wheel_v1h_params.pkl`／`rl_wheel_v1h_np.npz`（sha `9de7024271b4`） | 膝 65–74°／機身 0.51、落地 0.19／0.18、左平移四腿 roll 1.1°、崎嶇地 0/125；**但平移前後腳伸開（腳距 95–98 cm）**、右前腿釘（AA §5） | `rl_wheel/eval_*_v1h.md`、`compare_v1b_vs_v1h_mesh.mp4`（本機） |
| rl_wheel_v1g（2.12 億步中場）`weights/rl_wheel_v1g_s212_np.npz` | 不上機：膝 50–58°、FR 釘、boxes 10 摔 2 | `rl_wheel/eval_*_v1g_s212.md` |
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
