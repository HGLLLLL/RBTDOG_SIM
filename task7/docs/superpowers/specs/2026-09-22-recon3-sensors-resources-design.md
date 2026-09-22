# 第三趟唯讀偵察：感測器盤點 ＋ 兩情境資源採樣 —— 設計

- 日期：2026-09-22
- 目標機：**D1 Max（中狗）** RK3588 + Orin NX 雙板
- 動機：報告項目 **1.3 CPU / GPU / Memory 資源使用情形**、**7.1 感測器種類與數量**、
  **7.2 Camera 解析度 / Frame Rate**、**7.4 LiDAR 等其他感測器規格** 目前只有手冊宣稱值，
  沒有實機證據。這四項要的是「量到的」而不是「文件寫的」。
- 前兩趟：`realbot/recon_d1max.sh`（路線判斷）、`realbot/recon2_d1max.sh`（運控設定與 SHM）。
  這趟補的是**感測與資源**，兩者都沒碰過。

## 0. 使用者已定的三件事（2026-09-22 早上）

| 問題 | 決定 |
|---|---|
| 1.3 在什麼情況下量 | **待機 ＋ 原廠走路兩段對照**（人用遙控器走，腳本只採樣） |
| 7.2 相機量到什麼程度 | **設定檔 ＋ ROS2 topic ＋ 從 PC 用 `ffprobe` 實拉 RTSP** |
| 權限界線 | **允許唯讀的 `sudo` 指令**；仍不寫狗上檔案、不啟停行程、不送馬達指令 |

## 1. 交付物

| 檔案 | 作用 |
|---|---|
| `realbot/recon3_sensors_d1max.sh` | PC 端驅動：本機預檢 → 靜態盤點 → 待機採樣 → 走路採樣 → RTSP 實測 → 產表 |
| `realbot/recon3_report.py` | 吃輸出目錄，產 `報告填空表.md`（1.3 / 7.1 / 7.2 / 7.4 四張表）；可事後重跑 |
| `docs/現場操作卡_recon3_感測器與資源_2026-09-22.md` | 現場一頁式操作卡 |

## 2. 為什麼資源採樣用 Python 而不是 bash

要的是「每核使用率」與「哪個行程吃掉它」，那需要 `/proc/stat` 與 `/proc/<pid>/stat`
兩次取樣相減。bash 做這件事又長又容易錯。

Python 腳本**透過 stdin 餵給狗執行**（`ssh ... 'python3 -' <<< "$SAMPLER"`），
所以**狗上不留檔**，維持前兩趟「不在狗上產生任何檔案」的原則。
RK3588 已實測有 Python 3.10.12（`docs/D1Max_機器狗資訊.md` §3.2）；NX 是 Ubuntu 22.04。
python3 不存在時該段標「未取到」並說明，不猜。

## 3. 三段流程

### Phase 0 靜態盤點（兩板並行，狗趴著即可）

- **1.3 的底**：`lscpu`、`/proc/cpuinfo`、`/proc/cmdline`（`isolcpus`）、
  `/sys/devices/system/cpu/isolated`、cpufreq governor 與最高頻、`free`、`swapon`、
  GPU/NPU 裝置節點與驅動版本、`thermal_zone*`
- **7.1**：兩板的 ROS2 topic / node 全表（RK `ROS_DOMAIN_ID=66`、NX `=24`）、
  每個感測 topic 的型別 / QoS / 發布者數、`lsusb`、`/dev/video*`、`/dev/tty*`、
  感測驅動設定檔（`rslidar_sdk` / `uss` / `uwb` / `imu` / `gps` / `camera`）、
  從狗上 ping 與 curl 光達獨立網段 `192.168.1.102` / `192.168.2.102`
- **7.2**：`robot_camera` 與 `mediamtx` 設定、`v4l2-ctl --list-formats-ext`（若有工具）
- **7.4**：光達 yaml 的型號 / 線數 / 掃描頻率 / FOV / echo mode、
  `/front_lidar` 一筆 `PointCloud2` 的 header 與 `fields` / `point_step` / `width`×`height`、
  `topic hz` 實測、超音波 `sensor_msgs/Range` 自帶的 `min_range` / `max_range` / `field_of_view`、
  IMU 與 lidar-IMU 的實測頻率

### Phase 1 待機採樣（預設 30 s，兩板同時）

每秒取樣：每核使用率、`/proc/meminfo`、`devfreq` 的 load 與 cur_freq、`thermal_zone*`、
`/proc/net/dev`（看光達是不是真的在灌流量）、NX 另跑 `tegrastats`。
窗口起訖各讀一次 `/proc/<pid>/stat` → 前 15 名行程的 CPU% 與 RSS，
並記 `mc_ctrl` 的 `Cpus_allowed_list`。

### Phase 2 走路採樣（預設 40 s）

腳本停下來提示「請用遙控器讓狗站起來走」，按 Enter 才開始，結束再按 Enter。
**腳本自己不送任何馬達指令、不碰 `/dev/shm`。**

### Phase 3 RTSP 實測（本機）

`ffprobe -show_streams -of json` 拉 `rtsp://<RK>:8554/front` 與 `/back`，
取寬高、codec、`avg_frame_rate`、bitrate。本機沒有 ffprobe 就跳過並標明。

## 4. 機器可讀標記

靜態探測對關鍵事實多印一行 `@@` 標記，讓 `recon3_report.py` 不必解析自由文字：

```
@@TOPIC <name>|<type>|<hz>|<publishers>
@@CAMDEV <devnode>|<formats>
@@LIDARCFG <key>: <value>
@@KV <key>=<value>
```

原始輸出照樣全留，標記只是額外的一行。

## 5. 安全設計（沿用前兩趟，並補這趟的新風險）

1. 不啟停任何行程、不寫狗上檔案、不寫 `/dev/shm`、不送馬達指令。
2. `sudo` 只用唯讀查詢（`lsusb -v`、`dmesg`、`/sys/kernel/debug/rknpu/load`）。
   先試 `sudo -n`；不行才用 `sudo -S` 餵密碼並快取憑證，之後一律 `sudo -n`。
   密碼預設值已在 repo 文件裡（RK `bot` / NX `1`），可用環境變數覆寫，不新增暴露。
3. **遠端段刻意不用 `set -u`** —— 第一趟就是死在 `/opt/runtime/env.bash` 的
   `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:...`（非互動 SSH 下未設）→ shell 當場結束，
   而 `2>/dev/null` 把錯誤吞了，看起來像正常結束。
4. 每個區段包 subshell，單段炸掉不拖垮整份；所有 `ros2` 指令上 `timeout`。
5. NX 連不到就跳過該板並在 `verdict.log` 標明，不讓整趟失敗。
6. 判讀只掃遠端 log、且排除區段標題行 —— 第一趟被自己印的關鍵字誤報過。

## 6. 輸出

```
recon3_<時間>/
  recon3.log            完整過程
  rk3588_static.log     RK 靜態盤點原始輸出
  orinnx_static.log     NX 靜態盤點原始輸出
  rk_idle.json  rk_walk.json  nx_idle.json  nx_walk.json
  rk_idle.log   ...      採樣器的人讀摘要（stderr）
  rtsp_front.json  rtsp_back.json
  verdict.log           自動判讀
  報告填空表.md         ★ 1.3 / 7.1 / 7.2 / 7.4 四張表
```

沒取到的欄位一律寫「未取到」加原因，**不填手冊宣稱值**——
報告要的是實機證據，混進宣稱值會讓整張表不可信。

## 7. 驗證方式

無法在本機對狗實跑，所以：

1. `bash -n` 語法檢查全部腳本。
2. **採樣器在本機實跑**（本機也是 Linux，`/proc/stat` 同結構）→ 確認 JSON 正確、
   每核百分比合理、top 行程有值。
3. `recon3_report.py` 吃本機產的 JSON 產表 → 確認四張表都長出來、缺項顯示「未取到」。
4. 靜態探測的 ROS2 段無法本機驗，但沿用 recon2 已實機跑過的 source 寫法。

## 8. 非目標

- 不做 SLAM / 導航功能驗證（那是 5.x 項目，另一趟）。
- 不動 CPG-RL 上機前置（88 維 obs、政策匯出），與這趟無關。
