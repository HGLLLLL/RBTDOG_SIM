# 現場操作卡（第二趟）：補 ROS2 與運控設定

**這張卡是給你在現場離線照著做的。** 出發前先看過一遍。

- 對象：D1 Max（`zsm-1w`）的 **RK3588 一台**（Orin NX 這趟不用連）
- 風險：**零**。全程唯讀，不 sudo、不寫檔到狗上、不啟停行程、不送馬達指令
- 預計時間：**5–10 分鐘**（比第一趟短很多）

---

## 為什麼要跑第二趟

第一趟的腳本有 bug，**ROS2 整段完全沒執行**：

```
腳本用了 set -u，而狗的 /opt/runtime/env.bash 第二行是
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/runtime/lib
非互動 SSH 下 LD_LIBRARY_PATH 未設 → unbound variable → shell 當場結束
再加上我寫了 2>/dev/null，錯誤訊息也被吞掉，看起來像正常結束
```

已在本機重現並修好（把 source 包進 `set +u` 的 subshell）。

**所以「有沒有 `rt/lowcmd`」這個核心問題，第一趟其實沒有答案。**

---

## 第一趟已經確定的事（不用再查）

| | |
|---|---|
| `/dev/shm` | 有 **`joint_cmd`**、**`joint_state`**、**`imu_central`**，各 1 MB，`root:root`、`-rw-r--r--`（非 root 可讀、寫要 root） |
| 運控 | `mc_ctrl r` 在跑，由 `/opt/runtime/bin/start_motion_control.sh` 起、`robot-launch server` 管 |
| HAL | `robot_hal_node` 吃 `controller_manager.yaml` → 用 **ros2_control** |
| 目錄 | `/opt/export/{mc,config,rknn_model_crypto}` ← 與 MATRiX 發布包同結構 |
| ROS2 | RK `DOMAIN_ID=66`、NX `=24`，中間 `domain_bridge` 橋接；兩邊都 `rmw_zenoh_cpp` |
| SDK | UDP **8082** 有 listen |
| 韌體 | RK `0.1.7`(2026-02-06) / NX `0.3.6`(2026-02-28) |
| Orin NX | 純導航感知（nav2 / arc_lvio / localization / 各種 driver），**沒有運控** → 這趟不用連 |

---

## 步驟

### 1. 連上狗的 WiFi

profile 第一趟已經建好了，直接起：

```bash
nmcli con up d1max-wifi
ping -c3 192.168.234.1
```

**如果 profile 不在了**（或第一趟沒建成），重建：

```bash
nmcli dev wifi rescan && nmcli dev wifi list | grep -i XG2WIFI
nmcli con add type wifi ifname wlan0 con-name d1max-wifi \
  ssid "XG2WIFI_xxxxxx" \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "12345678" \
  ipv4.never-default yes ipv4.ignore-auto-dns yes ipv6.method disabled \
  connection.autoconnect no
nmcli con up d1max-wifi
```

> **這趟不需要加 `192.168.168.0/24` 的路由**，因為不連 Orin NX。

### 2. 跑腳本

```bash
cd ~/rbtdog_sim
bash task7/realbot/recon2_d1max.sh
```

不帶參數就是 WiFi 模式（`192.168.234.1`）。金鑰第一趟裝過的話不用再打密碼；沒裝的話密碼是 `bot`。

**跑多久**：3–6 分鐘。ROS2 查詢有 30 秒逾時保護，卡住會自己跳過。
最後會多花約 10 秒把三個 SHM 檔各拉兩份回來（共約 6 MB）。

### 3. 收工

```bash
tar czf recon2_YYYYmmdd_HHMMSS.tar.gz recon2_YYYYmmdd_HHMMSS/
nmcli con down d1max-wifi && nmcli con up Q500_ForDog
ping -c2 8.8.8.8
```

---

## 這趟要撈什麼

| 目標 | 為什麼重要 |
|---|---|
| ★★★ `ros2 topic list` + 底層 topic 的 **msg 定義** | 回答「有沒有 `rt/lowcmd`」。有的話，部署路線當場定案 |
| ★★★ `ros2 control list_hardware_interfaces` | 既然用 ros2_control，這會**直接列出每個關節的 command / state interface**。這是比 topic 更確定的答案 |
| ★★★★ `/opt/export/config/*.yaml` | **這台機器真實在用的運控參數**。拿來驗證我們從 MATRiX 發布包解出來的那組增益（ABAD 60 / HIP 120 / KNEE 120）是不是真的 |
| ★★ `joint_cmd` / `joint_state` / `imu_central` 快照 | 各拉兩份（間隔 2 秒），回本機 diff 就知道哪些位元組是活的、結構長什麼樣 |
| 啟動腳本 + `bridge_config.yaml` + `robot_hal` 設定 | 搞清楚 `mc_ctrl` 怎麼被叫起來、關節怎麼定義的 |

跑完會印一段**自動判讀**，直接告訴你 ROS2 有沒有載入成功、`lowcmd` 有沒有命中、
`/opt/export/config` 有沒有撈到真參數。

---

## 卡住了怎麼辦

### 判讀說「❌ 還是沒載入」

看 `rk3588.log` 最開頭那個「ROS2 環境載入」區段的錯誤訊息（這次沒有 `2>/dev/null`，錯誤會印出來）。
把那幾行帶回來就好，**不要在現場 debug**。

### `ros2 topic list` 是空的但環境載入成功了

可能是 zenoh router 的 scouting 需要時間，或 topic 真的都在別的 domain。
**不用處理** —— `ros2 control list_hardware_interfaces` 和 `/opt/export/config` 那兩項才是這趟的重點，
它們不依賴 topic 探索。

### `/opt/export/config/*.yaml` 顯示「讀不到，可能需要權限」

記下來就好，**不要用 sudo 硬讀**。權限本身就是有用的資訊。

### SHM 檔拉不回來

`scp /dev/shm/xxx` 對某些系統會失敗（特殊檔案系統）。不是災難——
腳本裡的 `od` dump 已經把開頭 256 bytes 印在 log 裡了。

### 腳本跑到一半卡住

`Ctrl-C`。已經寫出來的 `rk3588.log` 仍然有效，帶回來。

### 最低限度收穫

只要能 ssh 進去，跑這三行就有價值：

```bash
ssh robot@192.168.234.1 'cat /opt/export/config/*.yaml' > export_config.txt
ssh robot@192.168.234.1 'bash -c "set +u; . /opt/runtime/env.bash; ros2 topic list"' > topics.txt
ssh robot@192.168.234.1 'bash -c "set +u; . /opt/runtime/env.bash; ros2 control list_hardware_interfaces"' > hwif.txt
```

**注意那個 `set +u`** —— 少了它就會重演第一趟的失敗。

---

## 絕對不要做的事

- ❌ 不要 `sudo` 到狗上跑任何東西
- ❌ 不要 `robot-launch stop` 任何節點（`mc_ctrl` 停掉狗會癱）
- ❌ 不要對 `/dev/shm/joint_cmd` **寫入**任何東西。這趟純讀
- ❌ 不要在 RK3588 上裝東西或跑重的程式（官方明文：勿在此板開發應用程式）
- ❌ 不要用 `pkill -f` / `pgrep -f` 搭配會匹配到自己命令列的字串

---

## 附：一頁指令速查

```bash
nmcli con up d1max-wifi
ping -c3 192.168.234.1
cd ~/rbtdog_sim && bash task7/realbot/recon2_d1max.sh
tar czf recon2_*.tar.gz recon2_*/
nmcli con down d1max-wifi && nmcli con up Q500_ForDog
```
