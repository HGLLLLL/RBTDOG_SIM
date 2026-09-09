# 現場操作卡：RL v2.3 policy 上機（2026-09-09）

> **這趟的意義：狗第一次跑 RL。** policy 只在 trip17 走通的開迴路 A 周圍調變，
> 退路永遠是 A（現場輸入 `o`＋Enter 即退回，走路不中止）。
> 設計 `docs/superpowers/specs/2026-09-09-rl-policy-on-dog-design.md`；
> 通用連線／安全前提見 `現場SOP_通用_2026-08-27.md`；步態本體與 trip17 卡完全相同。

**流程**：站起來 →〔Enter〕→ 走（policy）→〔Enter〕→ 停 →〔Enter〕→ crouch → 趴下
**三段上機，一段一趟，每趟結束拉 log 看摘要再決定下一段**：
① `--policy-gain 0` 乾跑 5 s（走的是純 A，policy 只看不動）→ ② `gain 1` 走 5 s → ③ `gain 1` 走 10 s

---

## 0. 這趟的參數（基準步態＝trip17 那組，一個都沒改）

```
軌跡檔  A_kp250_walk.json（LS / x_off −30 mm / g_c 0.048 / z_sag 0.036 / duty 0.8 / ω 1.4）
腿增益  ABAD 60 / HIP·KNEE 250 / kd 2.0　輪 kd 0.5
policy  cpg_rl_max_v2_3_np.npz（v2.3；obs 66 / act 10；指令 vx 0.30 wz 0）
```

本機驗過的（今天）：numpy 前向 vs brax 1000 筆 < 1e-5；狗端 obs 與本機逐位元相同；
MuJoCo 內狗端堆疊影子跟跑 200 步目標角差 2e-7 rad；trip17 兩趟 log 回放全過
（`outputs/replay_policy_trip17.md`）；模擬驗收 v2.3：G4 ✅ G7 ✅（67.6 < 70）、G6 ❌ 60 s 漂 −30°
（走 10 s ≈ 5°，與 A 的左偏同級）。

**M9 會自己擋的**：軌跡檔／命令列與 policy 訓練基準不一致（說兩次）、推論 > 10 ms、
`--policy` 沒搭 `--traj`＋`--interactive`。

---

## 1. 本機：推檔

```bash
cd ~/rbtdog_sim
bash task7/realbot/push_to_dog.sh
```

要看到**全部 ✅**（比 trip17 多三個檔）：`policy_np.py`、`rl_obs.py`、`cpg_rl_max_v2_3_np.npz`。
任何一行不是 ✅ 就停。

## 2. 狗上：import 檢查（唯讀、零風險）

```bash
ssh robot@192.168.234.1
cd ~
python3 -c "import numpy, policy_np, rl_obs; p=policy_np.load('cpg_rl_max_v2_3_np.npz'); \
print(numpy.__version__, p.preset, p.obs_dim, p.act_dim, p.src_sha256[:12])"
# 要看到：1.21.5 v2.3 66 10 5a9de707c38b
```

現場準備五項與 trip17 卡相同（淨空 ≥ 5 m、吊帶 292 mm 以下鬆弛、輪貼膠帶、T2 急停待命、電壓 > 53 V）。
T1 先開 faultwatch 導到檔案；T2 打好 `sudo bash estop_max.sh` 不按 Enter。

## 3. 乾跑（T1，不帶 `--confirm`）

```bash
python3 M9_gait.py --traj A_kp250_walk.json --interactive \
        --kp 250 --kp-abad 60 --kd 2.0 --wheel-kd 0.5 \
        --seq ls --x-off -0.030 --g-c 0.048 --hold-max 10 --walk-max 5 \
        --policy cpg_rl_max_v2_3_np.npz --policy-gain 0
```

除了 trip17 卡列的那些 ✅，**多出這四行**：

```
RL policy：cpg_rl_max_v2_3_np.npz　preset v2.3　obs 66 act 10　sha 5a9de707c38b
  指令 vx 0.3 wz 0　混合比例 0（乾跑：動作固定基準）　起步淡入 50 步
✅ 基準步態與 policy 訓練基準一致（說兩次）
✅ 推論 0.xx ms/步（預算 10 ms）        ← 狗上 M_env_probe 量過 MLP 0.14 ms；> 2 ms 就先查 cpu 親和性
```

## 4. 段 ①：`--policy-gain 0`（狗走純 A；policy 只看、只記）

同上指令加 `--confirm`，前面加 `sudo`。走 5 秒按 Enter 停。

**這段看什麼**：狗的行為應與 trip17 17:00 那趟一模一樣（roll 6–7°、膝 ≈ 60）。
畫面上若出現「policy 退回開迴路 A（連續 25 步退回…）」→ 這段照走完（本來就是 A），但**不進段 ②**，拉 log 查原因。

結束後本機：
```bash
bash task7/realbot/pull_from_dog.sh            # 或照 SOP 拉 ~/m_logs
conda run -n rbtdog python task7/inference/policy_log_summary.py task7/logs/m_logs_trip19/M9_<時間>.json
```
要看到：`退回 0`、推論 p99 < 2 ms、obs 各欄範圍落在 `outputs/replay_policy_trip17.md` §2 那張表附近
（gravity 平均 ≈ (0, 0, −1)、joint_pos ±0.5、joint_vel ±13、gyro ±1.5），
`|a| 中位 ≈ 0.4`、ω 範圍大致 0.3–1.7、sway 目標平均 20–25 mm。
**任何一欄範圍差一個量級（例如 joint_vel 到 ±100、gravity z 是 +1）→ 停，不進段 ②。**

## 5. 段 ②：`--policy-gain 1`，走 5 s

```bash
sudo python3 M9_gait.py --traj A_kp250_walk.json --interactive \
        --kp 250 --kp-abad 60 --kd 2.0 --wheel-kd 0.5 \
        --seq ls --x-off -0.030 --g-c 0.048 --hold-max 10 --walk-max 5 \
        --policy cpg_rl_max_v2_3_np.npz --policy-gain 1 --confirm
```

**這段看什麼**（畫面 200 Hz 那行）：
- 起步 1 秒是淡入，之後 sway 上來、腳步頻率會變（ω 0.5–1.6 之間跳）—— 正常。
- **roll 應該 ≤ trip17（6–7°）**：模擬說降 36%。若 roll 反而 > 10° → 輸入 `o`＋Enter 退回 A，走完再停。
- 膝峰值：模擬 ×1.2 推估 67.6，門檻 70 —— **這是最緊的一項**。M9 超過會自己中止（那是設計）。
- 偏航：5 s 內 < 3° 都算正常。

中止或退回都不是失敗：拉 log 跑 `policy_log_summary.py`，看退回原因與那一刻的 obs／動作。

## 6. 段 ③：同段 ②，`--walk-max 10`

與 trip17 17:00 對照四個數：roll 峰值、膝峰值、髖峰值（trip17 髖已到門檻 93%）、10 s 偏航。

---

## 7. 現場一定要記得的三件事

1. **`o`＋Enter = 退回開迴路 A，走路繼續**；空行 Enter 才是推進階段（停走）。
2. M9 的所有護欄（70 N·m / 0.6 rad / 傾角 20° / 輪抖振）一個沒動；真正的急停在 T2。
3. 每趟 log 多了 `policy` 區塊（每步 obs／動作／原始 IMU／推論 ms／退回原因）—— 這也是第一次記到**走路時的原始 gyro_z**，之後 v2.4 的 head_err 要靠它。
