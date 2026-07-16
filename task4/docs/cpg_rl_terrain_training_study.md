# CPG-RL 地形訓練研究：論文有沒有做？我們怎麼加？

> 日期：2026-07-15
> 對象論文：**CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion**,
> Guillaume Bellegarda & Auke Ijspeert, IEEE RA-L 2022（arXiv:2211.00458）。
> 承接檔案：`notebooks/cpg_rl_paper_colab.ipynb`（Go2 · MJX · Brax PPO，目前**只在平地**訓練）。
> 目標：訓練出**對地形（斜坡/崎嶇/樓梯）有一定適應性**的 CPG-RL 模型。
>
> ⚠️ 本報告只做研究與方案評估，**尚未動任何 code**。等你看完給下一步指令。

---

## 0. 一句話結論

**原 CPG-RL 論文完全沒有做地形訓練——明確只在平地訓練**，唯一與地面有關的隨機化是摩擦係數；
不平地形只在「部署後測試」出現（丟泡棉碎屑），靠平地訓練附帶學到的魯棒性硬扛。
**地形是續作《Visual CPG-RL》才加的**。
要在我們 MJX/Brax 架構加地形，技術上**可行**（MJX 已支援 heightfield 碰撞，且 MuJoCo Playground 有現成「平地→Perlin heightfield finetune」範例），
但**必須先修 env 三個「寫死平地假設」的地方**，否則地形訓練一定學歪。

---

## 1. 論文到底有沒有做地形訓練？（逐段查證，非臆測）

### 1.1 訓練：白紙黑字「永遠平地」

§III-D *Training Details* 原文：

> **"The terrain is always flat during training.** With each reset, we sample new parameters h and g_c ... New velocity commands ... are sampled every 5 seconds. Although we find that domain randomization is not strictly needed to perform a sim-to-real transfer, unless specified we randomize the following parameters during training (kept constant during an episode):"
> - **ground coefficient of friction varied in [0.3, 1]**
> - limb mass varied within 20% of nominal values
> - added base mass up to 5 kg
> - external push of up to 0.5 m/s applied in a random direction to the base every 15 seconds
>
> **"No noise is added to the observation."**

→ 訓練期跟「地面」有關的**只有摩擦**。沒有斜坡、沒有台階、沒有高度場、沒有崎嶇地。

### 1.2 觀測空間：三種版本全是本體感覺，零地形資訊

§III-B 定義 `obs_full` / `obs_med` / `obs_min`，最豐富的 `obs_full` 是：
velocity commands + body state(orientation/linear/angular vel) + joint state + foot contact booleans + last action + CPG states `{r,θ,φ,ṙ,θ̇,φ̇}`。
**沒有任何 heightmap / 地形高度 / 外感（exteroception）**。論文的一大賣點正是「只靠本體感覺、甚至只靠觸地布林（obs_min）就能走」。

### 1.3 唯一的「不平地形」= 部署後測試，不是訓練

§IV-A-4 *Uneven Terrain* 原文：

> "As discussed in Section III-D, **we train our policies only on flat terrain**, though the coefficient of friction is varied. ... We tested adding debris such as **soft foam and hard styrofoam** (see Figure 1 and video) and found the policy to be robust to such terrain. These materials are very light and easy to kick and crumple by A1 ... it did not immediately fall."

→ 這是**部署測試**（deployment robustness），驗證平地訓練「附帶」得到的抗擾能力，**不是把地形放進訓練迴圈**。
Abstract / §I 也把 "uneven terrain" 歸類為 *"disturbances not seen in training"*（訓練時沒見過的干擾）。

### 1.4 影片裡「鑽矮樑」不是地形

Fig.1 的 "crawl underneath a ledge" 是 **body height h 線上調變**（把身體壓低），屬 §IV-A-3，不是地形適應。

### 1.5 小結

| 面向 | 原 CPG-RL 論文 |
|---|---|
| 斜坡訓練 | ❌ 無 |
| 樓梯/台階訓練 | ❌ 無 |
| 崎嶇/高度場訓練 | ❌ 無 |
| 地形觀測（heightmap 等） | ❌ 無 |
| 與地面有關的隨機化 | ✅ 只有摩擦 [0.3,1] |
| 不平地形 | 只在**部署後測試**（泡棉碎屑），非訓練 |

---

## 2. 續作《Visual CPG-RL》才是加地形的那篇（可借鏡）

**Visual CPG-RL** (Bellegarda, Shafiee, Ijspeert, arXiv:2212.14400) 把地形/外感放進來，是我們要「地形適應」時真正對齊的參考：

- **模擬器/機器人**：Isaac Gym（PhysX）+ Unitree **Go1**。
- **外感觀測**：機身周圍 **17×11 = 187 點高度圖**（grid 間距 0.1 m）；真機用 depth camera（RealSense D435i/T265）建 elevation map。
- **觀測組成**：velocity commands(3) + **heightmap(187)** + proprioception(body/joint/contacts) + CPG states(12) + last action(12)。
- **地形 curriculum**：從**平地**開始 → **隨機箱體**（寬 0.4–2 m、高 0.1–1 m），逐步變難。
- **定位**：偏「**避障 + 導航**」（左右轉、繞開障礙），而非專攻斜坡/樓梯步態，但它示範了「**heightmap 外感 + 地形 curriculum**」這套機制，正是 perceptive locomotion 的骨架。

> 註：Visual CPG-RL 的「地形」主要是散佈箱體障礙，不是連續斜坡或標準樓梯。若你要的是斜坡/樓梯，機制可借（heightmap + curriculum），但地形產生器要自己設計。

---

## 3. 我們的現況：`cpg_rl_paper_colab.ipynb` 是純平地

- 地面：`scene_mjx.xml` 的單一水平地板 plane。
- 隨機化（cell-13）：摩擦 [0.3,1]、PD 增益、連桿質量 ±20%、軀幹加重 0–8kg。**無任何地形項**。
- 觀測 76 維：本體感覺 + CPG 狀態，**無地形資訊**（和原論文一致）。
- 抬腳：`G_C=0.08` **寫死**、策略改不了。

**結論：我們目前 100% 複刻了原論文「平地」的設定。要地形適應，得自己加——論文沒給範本。**

---

## 4. ⚠️ 動地形之前，env 有三個「寫死平地」的地雷（必修）

以下是我實際讀 `cell-11` 對出來的，**不先修，任何地形方案都會學歪**：

| # | 位置 | 現況（平地假設） | 非平地會怎樣 | 修法 |
|---|---|---|---|---|
| 1 | `height = data.qpos[2]`；`height_pen=(height-0.30)²`；`done = height<0.18` | 用**世界座標 z** 當身高 | 爬坡/上台階時世界 z 升高 → 高度懲罰誤觸發、誤判跌倒 | 改成**相對腳下地形高度**（世界 z − 地形高度） |
| 2 | `_foot_contact`：`foot_z < 0.03m` | 用**世界高度**近似觸地 | 台階頂端腳在 3cm 以上卻踩實 → 觸地布林（obs 一部分）全錯 | 改用**實際接觸力/接觸偵測**（MJX contact），不要用世界高度 |
| 3 | `G_C=0.08` 固定 | 抬腳高度寫死、策略動不了 | 崎嶇/台階可能抬不夠高 → 卡腳 | 比照論文**每回合隨機化 g_c**，或把 g_c 併入 action 讓策略學 |

> 地雷 1、2 對「斜面」方案也會發作（傾斜地板後世界 z 不再等於離地高度）。所以無論選哪個方案都要先處理。

---

## 5. 加地形的方案（含 MJX 可行性查證）

**關鍵查證結果（非臆測）**：
- MJX 目前支援的碰撞幾何：**PLANE、HFIELD、SPHERE、CAPSULE、BOX、MESH**。**heightfield 碰撞已被支援**（修了 GitHub issue #1491）。
- **MuJoCo Playground 有現成範例**：四足先在**平地**訓練，再在 **Perlin noise heightfield** 上 finetune——正是我們要的路徑，可直接參考其 terrain / reward 寫法。
- notebook 用 `pip install mujoco mujoco-mjx`（裝最新版），版本夠新就有 hfield 支援；**但務必在 smoke test 階段先驗證 hfield 碰撞真的有作用**（歷史上有「地形太高就偵測不到碰撞」的 issue #1164，要留意 hfield 參數）。

### 方案 A（★已定案並實作：平地→斜坡過渡版）

> 定案日期 2026-07-15。使用者決定：**上坡＋下坡、角度 0–15°、curriculum 漸進**。
> 實作於**新筆記本** `notebooks/cpg_rl_terrain_colab.ipynb`（不動原 `cpg_rl_paper_colab.ipynb`）。

**原始構想 vs 定案差異**：原構想是「傾斜整個無限地板 plane」，但那是**均勻斜面、沒有平地→斜面的過渡**，也無法同時表達上坡與下坡（無限 plane 在 z=0 會擋住下坡）。因此改為下述「平台＋分段斜坡」的地形。

**地形設計（單一共享靜態地形，用 box 幾何，非 hfield → MJX 最穩、零 per-env 批次化風險）**：
- **中央平台**：一塊 box，頂面在 z=0，機器人 spawn 在上面走一段平地。
- **上坡側（+x）**：從平台邊緣起，**分段逐漸變陡的斜坡**（例如 7.5°→15°），往上。
- **下坡側（−x）**：鏡像，往下（7.5°→15° 下降）。所以原本的無限地板 plane **降到 z=−10 當安全底網**（避免下坡被擋、接住跌落）。
- 地面高度 `gz(x)` 是已知的分段線性函數（用 `jnp.interp(x, 折點x, 折點z)` 一行實現），**幾何與 gz 由同一組折點產生 → 保證一致**。

**上坡 / 下坡怎麼涵蓋**：`reset` 時隨機讓機身**面向 +x（走上坡）或 −x（走下坡）**（設定 base yaw = 0 或 π）。指令固定 body 前進 vx=0.6，面哪邊就走哪種坡。

**curriculum 怎麼做（重點）**：Brax/MJX 的地形幾何在訓練開始就固定（domain_randomize 只跑一次），無法逐回合改陡。因此 curriculum 用 **distance-based emergent 方式**：地形本身「越走越陡」（平台→7.5°→15°），機器人靠**里程獎勵**自然先學會緩坡、走得越穩才推進到越陡的段——這是 legged locomotion 常用、且不需全域步數的漸進機制。（未來可再加 legged_gym 式的 per-env 難度 tiering，v1 先用 emergent。）

**觀測**：**維持 76 維、盲走**——斜坡靠 obs 既有的重力向量 `grav`（IMU）感知（零樣本測試已證明「斜坡失敗是沒見過、不是看不到」），不加 heightmap。

**必配的地雷修復（見 §4，訓練前一定要做，否則學歪）**：
1. **身高/跌倒判定改相對地面**：`rel_h = qpos[2] − gz(qpos[0])`；height 獎勵與 done 都用 `rel_h`（不再用世界 z）。
2. **觸地布林改用 gz**：`contact = (foot_z − gz(foot_x)) < 門檻`（不再用世界高度；地形已知故解析可算）。
3. **g_c**：斜坡不需更高抬腳（零樣本測試 rough 才需要），**v1 維持 g_c 固定**，保留為可調；崎嶇階段（方案 B）再隨機化。
4. **upright 獎勵**：≤15° 時 `grav_xy²` 懲罰很小（sin15°²≈0.07，遠小於追蹤獎勵 1.5），故 v1 維持原式、不特別做坡面對齊；若發現壓抑爬坡再改成相對坡面法線。

**防走歪的 reward 調整（2026-07-15）**：零樣本測試中機器人在斜/崎嶇地會橫向漂移。為避免「一直走歪」：
- **指令改純直走** `cmd=[vx,0,0]`（不再隨機下橫移/轉向），讓任何橫移/轉向都被追蹤項扣分。
- **世界橫向偏移懲罰（有界）** `−0.8·(1−exp(−y²/0.10))`（spawn y=0，範圍 [−0.8,0]，y≈0.3m 罰到 −0.47、再遠飽和）。
- **提高航向率權重** `r_yaw 0.8→1.2`（把偏航率壓向 0＝鎖航向）。

> ⚠️ **踩過的雷**：一開始用**無界** `−1.0·y²`，訓到 ~8M 步 reward 變 **nan**——world y 可達數公尺 → 巨大負回報 → value 爆炸 → nan 經 normalize 統計汙染整批。**教訓：所有 reward 項必須有界**。已改成上面的有界版，並加 obs `nan_to_num`＋reward/qpos 有限性防護（發散回合直接 done、reward 歸零），杜絕 nan 擴散。

**Domain randomization**：沿用原本（摩擦/PD/質量/負重/推力），地形為共享靜態、**不需 per-env 批次化**。

**為何穩**：box 是 MJX 核心碰撞幾何；單一共享地形無 per-env hfield 批次化風險；gz 解析、與幾何同源。

**限制**：只有「連續斜面」，沒有高度起伏（崎嶇/台階留給方案 B/C）。凹凸零樣本已夠好，之後再與方案 B 合併。

### 方案 B：Perlin heightfield（崎嶇地、盲走）— 對齊 Playground
- 做法：把地板換成 `<hfield>`，用 Perlin/隨機噪聲生成崎嶇高度場；配 **terrain curriculum**（平地→小起伏→大起伏）。
- 觀測：**維持 76 維（盲走）**，靠本體感覺 + 觸地反應式適應——最貼近「CPG-RL 精神」（論文強調少感測）。
- 學到什麼：對碎石/坑窪的反應式魯棒（類似論文泡棉測試，但這次放進訓練）。
- 依賴：MJX hfield 碰撞（需 smoke test 驗證）；修地雷 1、2、3（崎嶇地尤其需要 g_c 隨機化/可調）。
- 風險：hfield 碰撞在 MJX 的數值/效能行為要實測；curriculum 沒設好會直接崩訓練。

### 方案 C：heightfield + 高度圖觀測（perceptive，對齊 Visual CPG-RL）— 效果最強、工程最重
- 做法：方案 B 的地形 + 在 obs 加**機身周圍高度掃描**（如 Visual CPG-RL 的 17×11 或縮小版 grid），obs 從 76 → 76+N。
- 觀測：需在 MJX 內查詢「機身周圍格點的地形高度」。對 Perlin/hfield 可**解析取樣**（我們自己生成地形，高度函數已知，不必用射線）——避開「MJX ray 對 hfield 未實作」（issue #2155）的坑。
- 學到什麼：**預judge 地形**、可望跨較大落差/台階（真正的 perceptive locomotion）。
- 成本：obs 擴大、網路要調、本機推論端也要能提供 heightmap（真機需 depth camera 建圖）——sim-to-real 難度跳升。

### 方案 D：散佈箱體/台階（樓梯）— Visual CPG-RL 式
- 做法：場景散佈 BOX geoms 當台階/障礙（MJX 支援 box 碰撞）。
- 難點：**per-env 不同地形**較麻煩（geom 在 model 中共享，要靠批次化 sys 改 `geom_pos/geom_size`）；規則台階建議還是用 hfield 做（方案 B/C）更省事。
- 適用：你若特別要「標準樓梯」再考慮，否則 hfield 能涵蓋大部分崎嶇+台階需求。

### 方案對照表

| 方案 | 學到 | MJX 風險 | 改 obs？ | 需 curriculum？ | 工程量 | Sim-to-real 難度 |
|---|---|---|---|---|---|---|
| A 傾斜 plane | 斜坡 | 極低（plane 最穩） | 否 | 否/弱 | ★ 小 | 低 |
| B Perlin hfield（盲走） | 崎嶇反應式 | 中（需驗 hfield） | 否 | **要** | ★★ 中 | 中 |
| C hfield + heightmap | 崎嶇+台階（預判） | 中 | **要** | **要** | ★★★ 大 | 高 |
| D 散佈箱體 | 樓梯/障礙 | 中（per-env 麻煩） | 視情況 | 要 | ★★★ 大 | 中高 |

---

## 6. 針對你的目標的建議路徑（漸進式，先不做，等你決定）

目標是「**對地形有一定適應性**」，我建議**分階段、先低風險驗證再加碼**：

1. **第 0 步（必做前置）**：先修 §4 的三個地雷（世界 z→相對地形高度、觸地改用接觸力、g_c 隨機化）。這步不論選哪個方案都要做，且可先在**現有平地**上確認訓練沒退步。
2. **第 1 階段（推薦起點）= 方案 A 傾斜 plane**：改動最小、MJX 零風險、obs 不動，先拿到「斜坡適應」的成果與訓練管線驗證。
3. **第 2 階段 = 方案 B Perlin hfield（盲走）**：加 terrain curriculum，拿到「崎嶇反應式適應」。這一階段最貼近 CPG-RL「少感測」精神，性價比最高。
4. **第 3 階段（可選）= 方案 C 加 heightmap**：要真正「看得到、能跨台階」再上，但 sim-to-real 成本大、且真機需視覺建圖。

> 我的推薦：**先 A 再 B**。A 幾乎零風險先驗證管線與地雷修復，B 才是「地形適應」的主菜；C/D 視你是否真的需要台階/預判再說。

---

## 7. 需要你拍板的決策（我不臆測，等你回覆）

1. **地形種類優先序**：你要的「地形適應」主要是 **(a) 斜坡**、**(b) 崎嶇碎石**、還是 **(c) 樓梯/台階**？三者對應的方案與工程量差很多。
2. **盲走 vs 有視覺**：可以接受**盲走（不加 heightmap，維持 76 維、純本體感覺反應式適應）**嗎？還是一定要能「預判地形」（要加 heightmap，obs 變大、真機要視覺）？這決定走 B 還是 C。
3. **只求 sim-to-sim 還是要上真機**：若最終要上真機 Go2，方案 C 的 heightmap 需要機上 depth camera 建圖，難度大增；若只到 sim-to-sim（如你們目前 task4 狀態），C 相對單純。
4. **是否接受漸進式（先 A 驗證再 B）**，還是要我直接一步到位做某個方案。

---

## 8. 參考來源

- CPG-RL 原論文（PDF）：https://arxiv.org/pdf/2211.00458 ／ EPFL：https://infoscience.epfl.ch/record/298615
- Visual CPG-RL（續作，加地形/外感）：https://arxiv.org/html/2212.14400v2
- MJX 支援的碰撞幾何（含 hfield）：https://mujoco.readthedocs.io/en/stable/mjx.html
- MJX hfield 碰撞需求 issue #1491：https://github.com/google-deepmind/mujoco/issues/1491
- MJX hfield 高程過高偵測不到碰撞 issue #1164：https://github.com/google-deepmind/mujoco/issues/1164
- MJX ray 未支援 hfield issue #2155：https://github.com/google-deepmind/mujoco/issues/2155
- MuJoCo Playground（平地→Perlin heightfield finetune 範例）：https://arxiv.org/html/2502.08844v1 ／ https://playground.mujoco.org/
