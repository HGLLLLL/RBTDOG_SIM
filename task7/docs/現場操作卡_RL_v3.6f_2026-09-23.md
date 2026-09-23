# 現場操作卡：CPG-RL v3.6f 上機（鏡像右轉＋右平移）—— 2026-09-23

> **這趟的意義：狗第一次跑 v3 —— 88 維 obs、狗端 numpy 產生器、輪子走 effort 力矩欄。**
> 退路不再是「軌跡檔 A」，而是**零動作名目產生器**（`--policy-gain 0`）；所以段 ②③ 先單獨把退路本身跑過。
> 設計：`docs/superpowers/specs/2026-09-22-cpg-rl-v3.6f-on-dog-design.md`（§4 七段、§6 執行紀錄）。
> 通用連線／安全／中止語意：`docs/現場SOP_通用_2026-08-27.md`（**§2.2 承重類**那張清單一項都不能缺）。

**流程（每段一趟）**：站起來（kp 250）→ 1.5 s 靜止過渡到 60/120/120 →〔Enter〕→ 走 →〔Enter〕→ 停 →〔Enter〕→ crouch → 趴下

---

## 0. 上機前提（缺一項就不要開始）

- SOP §2.2 **承重類**全部打勾；電壓 > 53 V；場地淨空 ≥ 5 m；輪子貼膠帶。
- **吊帶 `H` < 292 mm（實務 260），段 ⓪–③ 必掛**，④ 起視情況。掛太高＝空中抬腿，每欄都漂亮但是假的。
- 本機：
  ```bash
  cd ~/rbtdog_sim && bash task7/realbot/push_to_dog.sh     # 要看到「✅ 全部一致」
  ```
  比 trip19 多三個檔：`gen_v3.py`、`rl_obs_v3.py`、`cpg_rl_v3_6f_np.npz`。任何一行不是 ✅ 就停。
- 狗上（唯讀、零風險）：
  ```bash
  ssh robot@192.168.234.1
  python3 -c "import numpy, gen_v3, rl_obs_v3, policy_np; \
  p=policy_np.load('cpg_rl_v3_6f_np.npz'); print(numpy.__version__, p.layout, p.obs_dim, p.act_dim, p.src_sha256[:12])"
  # 要看到：1.21.5 v3 88 24 cac8679831c3
  ```
- 兩個終端機：T1 跑 M9（先開 `M_faultwatch.py` 導到檔案），T2 打好 `sudo bash ~/estop_max.sh` **不按 Enter**。
- **每段跑完 `sudo kill -CONT <PID>` 解凍**（忘了下一段一定怪）。

## 1. 共同旗標（每段都貼這一串，只換 `--policy-gain` 與 `--cmd`）

```
--v3 cpg_rl_v3_6f_np.npz --interactive \
--kp 120 --kp-abad 60 --kd 1.0 --wheel-kd 0.1 \
--vmax 18 --vcmd-max 16 --emax-abad 0.91 --emax 0.55 --tau-total 5 --wheel-vmax 30
```

- 增益**說兩次**：npz 記的是訓練時的 factory 組，和上面四個不一致就拒跑。不要自己改 `--kp`。
- `--mirror` 預設**開**（原地右轉、右平移走鏡像推論）；只有要做 A/B 對照才加 `--no-mirror`。
- `--policy-gain` 只吃 **0–1**（0 ＝零動作名目、1 ＝ policy；M9 會擋範圍外的值）。
- `--wheel-kd 0.1` 只在步態三段（GAIT_IN／GAIT／GAIT_OUT）生效；起身、坐下、凍結一律回 `WHEEL_KD_SAFE` 0.5 純阻尼。
- `--wheel-vmax 30`（v3 專用）＝四輪實測 |velocity| 超過就中止。輪子走力矩欄**沒有速度回授拉它回來**，
  τ 一直同號就一直加速，而 ChatterWatch 只認得正負翻轉的抖振，單向失控它一次都不會叫。名目輪速 ≤ 6 rad/s，
  30 是「已經明顯不對」的線，不是「剛好擦到」。另外 `v3_wheel_rows` 寫出去的 effort 本身已夾在 **±33 N·m**
  （npz 從模擬場景的輪致動器 forcerange 帶上來，就是 policy 訓練時 MuJoCo 在夾的同一條）。
- 乾跑（不帶 `--confirm`）先跑一次，**每一段換指令都要重跑乾跑**——限位／命令速度檢查吃的就是 `--cmd`。

## 2. 七段順序（任何一段中止 → 停在該段、拉 log，不往下）

| 段 | 指令（共同旗標之外） | 看什麼 | 過關 |
|---|---|---|---|
| ⓪ 乾跑 | `--policy-gain 0 --cmd 0 0 0`（**不帶** `--confirm`） | 說兩次 ✅、推論＋產生器 ms、限位表 ✅、命令速度 | 兩個 ✅ 都在；推論＋產生器 **< 5 ms**（程式的門是 10 ms，5 是保守要求）；**「貼止點」那一行有印出來**（`n/N 步至少有一個關節貼在止點上`；段 ⓪ 是 `--cmd 0 0 0`，印 `0/550` 就對；平移 0.12 的乾跑才會看到約 56%，見 §3）—— 這是限位飽和有在被量的證據，段 ① 之後才有 JSON 的 `n_sat` 可看 |
| ① 軟增益站立 | `sudo … --policy-gain 0 --cmd 0 0 0 --confirm --walk-max 5` | 起身 250 →（1.5 s 靜止）→ 60/120/120 kd 1.0；機身**再下沉 ≈ 36 mm**；四膝誤差與力矩 | 站住、誤差 < 護欄一半、無中止；**收工 JSON `policy.fallback_total == 0`、`open_loop_why` 必須正好是「按停淡出：指令歸零、零動作名目」（按 Enter 停走本來就會把 `open_loop` 設成 true，那是設計；是別的原因才算退回）、且 `n_sat` 欄存在** —— 這一段是 `rl_obs_v3.read_frame`（真 shm 讀取）第一次被真的跑到，本機沒有測試，有一步退回就代表讀法不對（`cmd 0 0 0` 這趟 `n_sat` 應該是 0） |
| ② 零動作直走 | `--policy-gain 0 --cmd 0.3 0 0 --walk-max 5` | **第一次走輪子 effort 路徑**：四輪 `velocity` 與 `effort` 的**符號**（往前走四顆應同號、effort 與速度同向）、跟得上 v_nom、無抖振；腿不跨步是對的。⚠️ 讀側輪符號由影子跟跑釘住，**寫側（effort 欄）的符號實機沒驗過** —— 接反就是往後走 | 走直、無抖振、vx ≥ 0.2 |
| ③ 零動作踏步 | `--policy-gain 0 --cmd 0 0.12 0`（5 s）；`--policy-gain 0 --cmd 0 0 0.8`（5 s，吊帶） | **硬門**（見 §4）：抬腳、側傾、`fl1/bl1_hip_roll` 的 ABAD 力矩與誤差 | 不倒、不中止 |
| ④ policy 直走 | `--policy-gain 1 --cmd 0.3 0 0 --walk-max 5` | 與 ② 的差別（policy 應該只做小修正） | 同 ② |
| ⑤ policy 左平移／左轉 | `--policy-gain 1 --cmd 0 0.12 0`；`--policy-gain 1 --cmd 0 0 0.8`（各 5 s → 10 s） | 步/秒、側傾、偏航率；log 的 `right=False` | 不倒；偏航率 ≥ 模擬 50% |
| ⑥ 鏡像右轉／右平移 | `--policy-gain 1 --cmd 0 0 -0.8`；`--policy-gain 1 --cmd 0 -0.12 0` | log 的 **`right=True`**；與 ⑤ 的左右對稱程度 | 同 ⑤ |
| ⑦ 放大 | 平移 `--cmd 0 0.15 0`／`--cmd 0 -0.15 0`；原地轉 `--cmd 0 0 1.3`／`--cmd 0 0 -1.3` | 對模擬驗收表（`outputs/eval_cpg_rl_v3_6f_params_mirrorall.md`） | — |

**平移 0.20 這趟不跑**：名目最大命令速度 15.84 rad/s，M9 要求 `--vmax ≥ 1.2×` → 要開到 19，
已經是腿關節規格上限 19.9 rad/s 的 95%（`D1Max_機器狗資訊.md`：規格書 190 RPM ≈ 19.9，設定檔寫 24 是矛盾值，保守取 19.9）。
要跑 0.20 得先另外決定護欄，不是現場臨時放寬。

### 段 ⓪ 乾跑要看到的幾行

```
CPG-RL v3.6f：cpg_rl_v3_6f_np.npz　sha cac8679831c3　obs 88 act 24　指令 … 鏡像 開（這趟走鏡像/不鏡像）　混合比例 0（乾跑：零動作名目）
✅ 推論 x.xx ms ＋ 產生器 x.xx ms（預算 10 ms）
✅ 零動作名目 N 步（站姿 hip +0.52 / knee −1.20，高 0.5x m）
步態最大命令速度 xx.xx rad/s（fl1_hip_roll）　上限 16.0
✅ 限位表 12 個關節都落在機構範圍內　＋　步態全程在產生器限位 LO12/HI12 內
   （n/N 步至少有一個關節貼在止點上 …）
```

各指令的名目最大命令速度（本機量過，`--secs 10`）：
直走 0.3 → **0**（v3.6f 前進全靠輪，腿不跨步）；平移 0.12 → **13.98**；平移 0.15 → **14.67**；
原地轉 0.8 → **6.86**；原地轉 1.3 → **11.15**。全都在 `--vcmd-max 16` 內，且 `--vmax 18 ≥ 1.2×`。

**「貼止點」那行看到一大半是預期的**。⚠️ M9 印的分母是 **`--secs` 加上前面 1 s 站姿的全步數**
（`--secs 10` → 550 步），不是「指令有效的 500 步」—— 前 50 步是站姿，一顆都不貼，所以現場讀到的數字
比「除以指令有效步」小一截。**現場會看到**：平移 0.12 → **約 56%**（`fl1_hip_roll` 31%、`bl1_hip_roll` 25%），
平移 0.15 → **57%**，原地轉 1.3 → **13%**，原地轉 0.8 與直走 0.3 → **0%**。
（同樣的數字除以指令有效步是 62%／63%／14%／0% —— 這是 §3 在講週期比例時用的口徑，別拿去對現場那一行。）詳見 §3。

## 3. 為什麼是這些數字

| 旗標 | 值 | 怎麼來的 |
|---|---|---|
| `--emax-abad` | **0.91** rad | 影子跟跑（`tests/test_policy_v3_closed_loop.py`，五指令各 300 步）ABAD 命令-實際誤差峰 **0.697** rad ×1.3 = 0.906 → 0.91。設計上限 1.2，沒有用滿。 |
| `--emax` | **0.55** rad | 同來源，髖膝誤差峰 **0.394** rad（原地左轉）×1.3 = 0.512，取整到 0.55。設計上限 0.8。⚠️ **`--emax` 是通用旗標**（舊 `--traj`／`--policy` 路徑也吃它，預設 0.60）——只有 **`--emax-abad` 是 v3 專用**（`a.v3 and j.endswith("1_hip_roll")` 才生效），所以在 v3 趟裡 `--emax` 管的是髖膝、`--emax-abad` 管 ABAD。 |
| `--vmax` | **18** rad/s | RL 趟的既定值（trip19 定案）。影子跟跑膝速峰 8.2 rad/s ×1.3 = 10.7，18 已經遠高於它；18 同時滿足平移 0.15 的 `1.2 × 14.67 = 17.6`。 |
| `--wheel-vmax` | **30** rad/s | v3 專用的輪速中止。名目輪速 ≤ 6 rad/s（直走 0.3 最高），馬達上限 125.7 —— 30 離名目夠遠不會誤中止，離失控夠近能在撞牆前叫。effort 路徑的單向失控 ChatterWatch 抓不到，這是唯一一道。 |
| `--vcmd-max` | **16** rad/s | 預設 14 會剛好卡死平移 0.12 的 13.98。16 收得住 0.12／0.15／轉 1.3，同時擋掉 0.20 的 15.84 以上那一檔。 |
| `--tau-total` | 5 | 沿用預設。**不要**關掉。2026-09-23 現場改：v3 模式只在步態三段累計 —— `fl1_hip_roll` 的 effort 讀值在 kp 250 站立段每趟有 4–5 筆孤立單筆尖峰（−50…−121，同號、前後 −8…−26），⑤a／⑤b 兩趟都在走完之後的 STOPPED／SIT 被累計到第 5 筆中止；GAIT 段一筆都沒超過。站立段仍有快速路徑（連續 3 筆）與硬上限 120。 |

⚠️ **護欄峰值是「步末抽樣」量的**（50 Hz 控制步的末端），實機 200 Hz 逐拍量會再高 **≤ 3%**；
輪 τ 峰 19.56 N·m 是**死區前**的值（死區後每顆再 +0.13）。兩件事都在 0.91／0.55 的餘裕裡，但誤中止時先想到它們。

### ABAD 誤差 0.70 不是雜訊 —— 平移名目本來就騎在止點上

平移指令下，產生器給 `fl1/bl1_hip_roll`（右平移＝鏡像的 `fr1/br1`）的**命令**有很大一段直接釘在
**−0.523 rad 的機構止點**上（0.12 指令、**除以指令有效步**：`fl1` 35%、`bl1` 27% 的週期；M9 乾跑印的是除以全步數的 31%／25%），而實際關節只到 ∓0.15–0.19 rad
→ 命令與實際差 0.70 rad，四個會抬腿的指令都一樣。

這是**訓練出來的長相**，不是快撞到：v3.6f 的 ABAD 幅度是原廠 ×1.6（補 kp120 的撓度），產生器最後一步
`np.clip(q_des, LO12, HI12)` 夾的就是同一條線，訓練時就是這樣跑的。所以 v3 的乾跑限位檢查改成
「npz 限位表落在機構範圍內」＋「軌跡在 npz 限位內」兩道（margin 0），不再用 `coord.check_limit` 的 0.03 餘裕。

→ **段 ③ 現場要盯的就是這兩顆**（鏡像時看 `fr1/br1`）：ABAD 力矩有沒有持續壓在止點上發熱、誤差是不是穩在 0.7 附近而不是繼續爬。
持續貼止點 ＋ 力矩單向飽和 ＝ 停，不要用「反正護欄還沒跳」說服自己。

### 腿力矩可能碰 TMAX，這是已知的

模擬腿 τ 峰 **42–48 N·m**（平移／原地轉），M9 的 `TMAX` 是 ABAD/HIP **50**、KNEE **70**。
快速中止路徑是「**連續 `--tau-hits`（預設 3）步超過 TMAX**」→ 實機上有機會跳，而且不一定是壞事。

- **`TMAX` 這趟不動**（50/70 是承重保護，不是調參旋鈕）。
- 真的誤中止（看 log 確認是尖峰不是持續飽和）再動的是 `--tau-hits`（連續步數）與 `--tau-total`（整趟累計），**現場不要改 TMAX**。
- 持續 > 40 N·m 不跳也要停：那是熱的問題，不是門檻的問題。

## 4. 硬門：段 ③ 倒了就回頭，policy 不上

零動作名目（`--policy-gain 0`）在模擬**平移不倒**（0.12／0.15 側傾峰 2.5°），但**原地轉不是**：0.8 側傾峰 24°（會撞 20° 傾角護欄）、1.3 在 3.7 s 倒（驗收表零動作欄早就寫「摔」）——原地轉的原廠週期表開迴路本來就站不住，設計是靠 RL 殘差平衡，policy 才是在轉的那個。
2026-09-23 現場更正：**硬門只有 ③a 平移**；③b 零動作原地轉不當硬門（跑了只會被傾角護欄凍結）。退路仍成立：永久退回會把指令歸零，指令 0 的名目＝站姿，穩。實機仍未驗過平移名目——軟增益 120、原廠週期表、41 kg。

- 段 ③a 平移 0.12 **倒了、中止、或一直退回**（③b 原地轉見上，不是硬門）：
  **停在這裡**，不跑段 ④ 以後。回頭做零動作的實機掃描（週期、抬高、ABAD 幅度），policy 先不上。
- 理由：policy 的退路就是這個名目。名目自己站不住，`--policy-gain 1` 出事時沒有東西可以退。
- 段 ①②③ 三段都是零動作，等於把「退路」整條路徑（軟增益站立 → 輪力矩 → 踏步）單獨驗過一次。

## 5. 現場一定要記得的五件事

0. **`o` ＋ Enter ＝ 永久零動作名目**（走路不中止，policy 從此不混入）。空行 Enter 才是推進階段（停走）。
   v3 的退路**不是** trip17 的軌跡檔 A，是名目產生器本身——所以按 `o` 之後看到的步態會變，那是對的。
1. **指令時序**：**GAIT_IN 全程 cmd = 0**（產生器輸出站姿）；**進 GAIT 後 1 s（50 步）**才切到 `--cmd`；
   Enter 停走 → **指令歸零、同時開始 3 s 淡出**（不是先歸零 1 s 再淡出）。
   所以按 Enter 起步之後，「淡入 `--ramp` 3 s ＋ 再 1 s」大約 4 秒狗才開始照指令走 —— 這是設計，不是當掉。
2. **GAIT_IN 淡入段的輪力矩是全量的**：腿目標從站姿 smoothstep 淡入，但輪 τ 第一步就照送。看到「腿還沒開始走輪子先動」不要誤判。
3. **換到軟增益機身會再下沉 ≈ 36 mm**（z_sag 36 → 72）。段 ① 要**看到**它；沒下沉反而要查增益有沒有真的換過去。
4. M9 的其他護欄（傾角 20°、輪抖振 ChatterWatch、`--gap-max`、溫度 70）一個沒動；真正的急停在 T2。

## 6. 中止或退回之後

```bash
bash task7/realbot/pull_from_dog.sh <trip>       # 例：pull_from_dog.sh trip22
```

收工 JSON 的 `policy` 區塊每步記：`obs(88)`、`a(24)`、**`a_raw`（policy 原始輸出，未乘 gain、未被退路歸零）**、
`q_des(12)`、`tau(4)`、`v_wheel(4)`、`right`、`cmd(3)`、**`sat`（這一步有幾個關節的目標被限位夾掉）**、原始 imu、ms、退回原因；
整趟另有 `n_clamp`（產生器 IK 縮限次數，正常是 0）、**`n_sat`（至少一個關節頂住限位的步數 —— 平移趟本來就會很大，見 §3）**、`fallback_total`、`open_loop`。

先看三件事：**退回次數與原因**（`open_loop=true` 且 `open_loop_why` 是「按停淡出…」是正常的，每趟按 Enter 停走都會這樣）（`stale`／`obs_nan`／`act_nan`／`slow`／`exc:*`／`exc_gen:*`／**`gen_nan`**）、**`right` 旗標對不對**（⑥ 應該 True）、
**中止那一刻的關節**（ABAD 還是膝？誤差還是力矩還是速度？）。`n_clamp > 0` ＝ 產生器被縮限，那是另一類問題，回報。

## 7. 每段的完整指令（狗上 T1，`cd ~`；每段兩行：先乾跑、再 `sudo … --confirm`）

前置（本機一次、狗上一次）：
```bash
# 本機
cd ~/rbtdog_sim && bash task7/realbot/push_to_dog.sh                       # 要看到「✅ 全部一致」
# 狗上 T1
ssh robot@192.168.234.1
cd ~ && python3 -c "import numpy, gen_v3, rl_obs_v3, policy_np; p=policy_np.load('cpg_rl_v3_6f_np.npz'); print(numpy.__version__, p.layout, p.obs_dim, p.act_dim, p.src_sha256[:12])"
#   → 1.21.5 v3 88 24 cac8679831c3
nohup python3 M_faultwatch.py --hz 10 --secs 3600 > /dev/null 2>&1 &        # 背景記錄到 ~/m_logs/Mfault_*，整天開著
# 狗上 T2（急停備著，打好不按 Enter）
ssh robot@192.168.234.1
sudo bash ~/estop_max.sh
```

共同旗標存成變數，之後每段只換 gain 與 cmd：
```bash
C="--v3 cpg_rl_v3_6f_np.npz --interactive --kp 120 --kp-abad 60 --kd 1.0 --wheel-mode kd1 --wheel-kd 1.0 --vmax 18 --vcmd-max 16 --emax-abad 0.91 --emax 0.55 --tau-total 5 --wheel-vmax 30 --hold-max 15"
```

- `--wheel-mode kd1 --wheel-kd 1.0`（2026-09-23 現場 trip22 段 ⑤a 後改）：原本的 `tau` 模式（50 Hz 外環算好寫 effort、kd 0.1）在**腳離地時發散**
  —— 後左輪 −4→+4→−8→−13→+12→−20→+35 rad/s 每步翻號放大，被 `--wheel-vmax 30` 擋下。輪子沒負載時慣量太小，1 N·m 在 20 ms 內改 2.5 rad/s，
  50 Hz 外環迴路增益 > 2。kd1 ＝ spec §2.2 備案：velocity 給名目輪速、kd 1.0 讓驅動器 1 kHz 自己算 1.0·(v_nom − v)（M11 驗過不抖）、effort 只剩 wres＋週期表力矩。
  法則和模擬一樣、只是外環快 20 倍。段 ①②④ 是用 tau 模式跑過的（有負載時沒問題）。
- `--hold-max 15`（2026-09-23 現場加）：互動模式的承重預算把兩個「等 Enter」各算 `--hold-max` 秒，預設 25 會讓最壞情況 77 s 超過 `--max-secs 60` 被拒跑。15 → 57 s。
  **意思是 READY 站著 15 s 沒按 Enter 就自己起走、STOPPED 15 s 沒按就自己趴下**，手要放在 Enter 上。`--walk-max 10` 的趟要再改 `--hold-max 12`（56 s）。

每段先乾跑再實跑；乾跑的 `--cmd`、`--walk-max` 要和實跑一樣（承重預算是用 `--walk-max` 算的，預設 20 會超 60 被拒）。逐行複製：

```bash
# ⓪ 乾跑（不寫入、不需 sudo；段 ⓪ 沒有實跑）
python3 M9_gait.py $C --policy-gain 0 --cmd 0 0 0 --walk-max 5

# ① 軟增益站立（吊帶）
sudo python3 M9_gait.py $C --policy-gain 0 --cmd 0 0 0 --walk-max 5 --confirm

# ② 零動作直走（吊帶）—— 第一次走輪子 effort 路徑，看四輪速度同號、無抖振
python3 M9_gait.py $C --policy-gain 0 --cmd 0.3 0 0 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 0 --cmd 0.3 0 0 --walk-max 5 --confirm

# ③ 零動作踏步（吊帶）—— 硬門：倒了／中止／一直退回就停在這裡，policy 不上
python3 M9_gait.py $C --policy-gain 0 --cmd 0 0.12 0 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 0 --cmd 0 0.12 0 --walk-max 5 --confirm
python3 M9_gait.py $C --policy-gain 0 --cmd 0 0 0.8 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 0 --cmd 0 0 0.8 --walk-max 5 --confirm

# ④ policy 直走（對 ② 只應是小修正）
python3 M9_gait.py $C --policy-gain 1 --cmd 0.3 0 0 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0.3 0 0 --walk-max 5 --confirm

# ⑤ policy 左平移、左轉（log right=False；穩了再把 --walk-max 改 10）
python3 M9_gait.py $C --policy-gain 1 --cmd 0 0.12 0 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 0.12 0 --walk-max 5 --confirm
python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 0.8 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 0.8 --walk-max 5 --confirm

# ⑥ 鏡像右轉、右平移（log 要看到 right=True，與 ⑤ 左右對稱）
python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 -0.8 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 -0.8 --walk-max 5 --confirm
python3 M9_gait.py $C --policy-gain 1 --cmd 0 -0.12 0 --walk-max 5
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 -0.12 0 --walk-max 5 --confirm

# ⑦ 放大（平移 0.15、原地轉 1.3；0.20 這趟不跑）
python3 M9_gait.py $C --policy-gain 1 --cmd 0 0.15 0 --walk-max 10 --hold-max 12
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 0.15 0 --walk-max 10 --hold-max 12 --confirm
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 -0.15 0 --walk-max 10 --hold-max 12 --confirm
python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 1.3 --walk-max 10 --hold-max 12
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 1.3 --walk-max 10 --hold-max 12 --confirm
sudo python3 M9_gait.py $C --policy-gain 1 --cmd 0 0 -1.3 --walk-max 10 --hold-max 12 --confirm
```

**每一趟的鍵盤流程（T1）**：站起來後畫面停在 `READY` → **Enter ＝ 起走**（先 3 s 淡入、再 1 s 才照 `--cmd` 動）
→ 走的時候**隨時 Enter ＝ 停走**（指令歸零＋ 3 s 淡出回站姿，淡入中按也有效）→ 停在 `STOPPED` → **Enter ＝ 坐回趴姿**。
Enter 是「優雅停」不是急停：要 3 秒才站定；狗要倒、腿卡住、程式沒回應一律走 §8 的 estop。
沒按 Enter 也會自己走：`--walk-max` 到了自動淡出、`--hold-max`（25 s）到了自動往下走，人不要離開鍵盤。
`o` 再 Enter ＝ 永久零動作名目（policy 不再混入，走路不中止）；只有**空行 Enter** 才推進階段。
T1 必須是真的終端機（不能 nohup／背景），不然 Enter 讀不到。
**趴下之後**畫面還會印一次「★ 現在腿還在承重。[Enter] 依原路徑坐回趴姿；[Ctrl-C] 立刻放手」—— 那是固定時間表模式留下的提示，互動模式走到這裡狗**已經趴好、零增益**了；
按 **Ctrl-C** 跳過（按 Enter 會用 kp 250 把狗從趴姿撐到 crouch 再趴一次，沒意義）。然後照它印的 `sudo kill -CONT <PID>`。

- 每段結束程式會印 `sudo kill -CONT <PID>`，**照打**。忘了，下一段 `mc_ctrl` 狀態會不對。
- 逾時：`--walk-max` 到了自動淡出，不是中止；`--hold-max`（預設 25 s）到了自動往下走，人不要離開。

## 8. 出事了：急停 → 解凍 → 把資料搬回來

**先分兩種情況**（看 T1 畫面）：

**A. M9 自己中止了**（畫面有 `⛔ 中止：…` ＋「狗還撐著，沒有放手」＋ `[Enter] 依原路徑坐回趴姿`）
1. 看狗：腿在承重、目標角已凍結。**先看，再決定**。
2. 正常情況：T1 按 **Enter** → 依原路徑坐回趴姿、降到零增益 → 程式印 `sudo kill -CONT <PID>` → 照打。JSON 與 log 都會寫好。
3. 若狗姿勢已經不能安全坐下（腿懸空卡住、要倒）：轉去情況 B。

**B. 要立刻放手**（狗要倒、腿卡住、程式沒回應、任何你覺得不對的）
1. **T2 按 Enter**（那行已經打好的 `sudo bash ~/estop_max.sh`）。它會：殺 M9 → 確認死透（含 `/proc/*/maps` 交叉檢查）→ 解凍 `mc_ctrl` → 印 16 顆是否全零增益。
2. **腿約 0.5 s 後失力下垂**（controller 500 ms 逾時清零），**不是**阻尼停止 —— 腿下方不能有人、吊帶要能承接。
3. 看 estop 的三行：
   - `[2/3] ✗✗ 連 SIGKILL 都殺不掉` → **切電源**，不要解凍。
   - `[2/3] ⚠️ 還有非原廠行程映射著 joint_cmd` → 照它印的 pid 手動 `sudo kill -9 <pid>`，再跑一次 estop。
   - `[3/3] ✅ 原廠控制已接手` → 原廠遙控可以用了。
4. 若 T2 也沒反應／SSH 斷線：**切機器人電源**。吊帶是最後一道。

**C. 之後把資料搬回本機**（兩種情況都一樣）
```bash
# 本機
cd ~/rbtdog_sim && bash task7/realbot/pull_from_dog.sh trip22        # 收到 task7/logs/m_logs_trip22/（換成這趟編號）
# scp 權限失敗時（M9 用 sudo 跑、chown 沒成功）：
ssh robot@192.168.234.1 'sudo chown -R robot:robot ~/m_logs' && bash task7/realbot/pull_from_dog.sh trip22
```
會拿到：`M9_<時間>.log`（畫面輸出）、`M9_<時間>.json`（**只有 M9 自己走完中止／收工路徑才有**，含每步 `obs/a/a_raw/q_des/tau/v_wheel/right/sat`、`n_clamp/n_sat/fallback_total`）、`Mfault_*.log/.csv`（背景 10 Hz 故障監看）。
⚠️ **被 estop 殺掉的那趟沒有 JSON**（每步紀錄在記憶體，SIGKILL 來不及寫），只剩 `.log` 與 `Mfault_*`。所以能用 A 就不要用 B。

拉回來先看：`.log` 最後 12 筆（中止前的關節 q/des/τ/v）、JSON 的 `policy.fallback_total` 與各步 `fb` 原因、`right` 旗標、`n_sat`。

---

**段 ⓪–⑦ 的實機結果寫回**
`docs/superpowers/specs/2026-09-22-cpg-rl-v3.6f-on-dog-design.md` §6，
定案結論另開 `docs/results/<字母>_<題目>_<日期>.md`。
