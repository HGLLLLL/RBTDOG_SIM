# task6/realbot — 輪足 D1 EDU 實機底層控制工具鏈

把 CPG-RL policy 部署到**輪足 D1 EDU 實機**的底層工具。因官方 SDK 不提供輪足版 low-level
（`nm -D` 驗證 `zsl-1w` 庫 0 個 LowLevel 符號、`deploy.md:168` 明載不提供），唯一的路是本體板載
共享記憶體 `/spline_shm`（1 kHz 馬達介面）。完整調查見 `../docs/D1EDU_輪足_lowlevel_調查與實測指南.md`。

**2026-08-10 全程實機驗證通過：讀取 + 速度/力矩/位置三種控制 + 校正層初版。**

---

## ⚠️ 安全總則

- 所有「寫入」程式都需 root（`/spline_shm` 為 root 擁有）、需先停/凍結 `mc_ctrl`。
- 停 `mc_ctrl` 後遙控器與原廠保護全失效。**驅動腿關節前務必把狗吊起來、輪子離地。**
- 每支寫入程式都有預檢（偵測 cmd 旗標仍跳動＝mc_ctrl 沒停就拒跑）+ watchdog 兜底（程式停＝馬達癱軟）。
- `mc_ctrl` 建議用 `SIGSTOP` 凍結、`SIGCONT` 解凍（不必 reboot）。它由 `robot-launch.service` 管，勿殺 `spline_daemon`。

## 連線 SOP（有線經交換器，已驗證 0.4ms）

1. 狗接同一台交換器；電腦加第二 IP：`sudo ip addr add 192.168.168.100/24 dev <你的有線網卡>`（重開機消失）
2. `ssh dog`（金鑰登入；狗有線 IP 192.168.168.168、密碼 firefly）
3. 編譯：`g++ -O2 -o <name> <name>.cpp -lrt`

---

## 階梯式工具（風險由低到高）

| 檔案 | 階段 | 做什麼 | 風險 | 實測結果 |
|---|---|---|---|---|
| `L0_shm_probe.cpp` | L0 | 唯讀讀 16 關節 p/v/t + IMU（人眼/CSV） | 零 | ~880Hz，leg0=FR 確認 |
| `L0_shm_handshake.cpp` | L0 | 唯讀觀察 mc_ctrl↔daemon 握手節奏 | 零 | mc_ctrl ~500Hz 寫 cmd |
| `L1_zero_write.cpp` | L1 | 全零增益寫入（證明寫得進、daemon 認，馬達零出力） | 低（狗平躺） | ack 100%、力矩≈0 |
| `L2_wheel_spin.cpp` | L2 | 慢轉單顆輪（速度/力矩控制） | 中（輪懸空） | 需 t_ff≈0.8 掙脫靜摩擦後轉起 |
| `L3_position.cpp` | L3 | 單輪位置控制（轉到定角度撐住） | 中（輪懸空） | 誤差 0.2%，乾淨定住 |
| `L4_standup_shm.py` | L4 | 站姿（移植點足範例，per-leg），**預設 dry-run** | 高（需吊掛） | 2026-08-11 實機驗證通過：吊掛驅動 FR/FL/RL 三腿（RR 故障跳過），趴下／站立皆平順 |
| `shm_common.py` | — | SHM 結構與安全骨架，L4/L7 共用。結構定義只能有這一份。 | — | — |
| `L7_gait_shm.py` | L7 | 步態串流（吊掛空跑）。三模式 jog/leg/gait。操作見 `docs/L7_吊掛空跑操作手冊.md`。 | 高（需吊掛） | 見操作手冊 |

## 校正層

| 檔案 | 說明 |
|---|---|
| `calib_capture.py` | 唯讀擷取當前姿勢的 16 關節角度 → 每腿 POSE（遙控器擺姿勢再跑） |
| `calib_stand.json` / `calib_lie.json` | 實機站姿 / 趴下姿勢資料（編碼器慣例，左右鏡像） |
| `calib_map.py` | **MJCF→實機映射**：`shm_cmd = SIGN*mjcf + OFFSET` + 腿序重排 `[1,0,3,2]` |

映射信心：腿序重排 ✅、abad/knee 正負 ✅、**hip 絕對正負 + 所有 offset ⚠️ 待吊掛時單關節微動確認**。

---

## 三種控制模式（皆實機驗證）

馬達力矩 = `kp*(p_des−p) + kd*(v_des−v) + t_ff`

- **位置控制**：kp>0、p_des=目標、kd=阻尼（policy 部署主用；原廠 demo 值 kp=80/kd=1）
- **速度控制**：kp=0、kd>0、v_des=目標
- **力矩/前饋**：t_ff（掙脫靜摩擦、順應）

## 待辦（下一階段）

1. 吊掛實機，確認 hip 絕對正負 + 複核所有 offset（`calib_map.py`）
2. 腿關節位置控制（L3 邏輯換到 hip/knee，吊掛）
3. L4 站姿上吊掛實機
4. 接 CPG-RL policy 閉環
5. **驅動腿關節前建議先向原廠確認 spline_shm 寫入政策與保固**
