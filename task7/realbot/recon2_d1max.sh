#!/usr/bin/env bash
# recon2_d1max.sh — D1 Max 第二趟唯讀偵察｜補第一趟漏掉的 ROS2 + 運控設定
#
# 第一趟（recon_d1max.sh）撈到的：
#   ✅ /dev/shm 有 joint_cmd / joint_state / imu_central（各 1 MB，root:root，非 root 可讀）
#   ✅ mc_ctrl 在跑、robot_hal 用 ros2_control、/opt/export/{mc,config} 結構
#   ✅ 韌體版本 RK 0.1.7 / NX 0.3.6
#   ❌ ROS2 整段沒跑到 —— 那支腳本 `set -u` 碰上 env.bash 的
#      `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:...`（非互動 SSH 下未設）→ shell 當場結束
#
# 這一趟修掉那個 bug，並補撈：
#   1. ROS2 topic / node 全表、底層 topic 的 msg 定義
#   2. ros2_control 的 hardware interfaces（既然 robot_hal 用 controller_manager）
#   3. /opt/export/config/*.yaml —— ★ 這台機器真實在用的運控參數
#      （拿來驗證我們從 MATRiX 發布包解出來的那組 zg_wheels-user-parameters）
#   4. 啟動腳本與 domain_bridge 設定
#   5. joint_cmd / joint_state / imu_central 的實際內容（scp 回本機離線分析）
#
# ★ 安全設計：全程唯讀。
#   - 不 sudo、不寫任何檔案到狗上、不啟停任何行程、不送任何馬達指令
#   - shm 只讀不寫（scp 拉回本機分析，不在狗上產生暫存檔）
#
# ⚠️ 只連 RK3588。第一趟已確認 Orin NX 上完全沒有運控相關行程（純導航/感知）。
#
# 用法：
#   bash recon2_d1max.sh                  # WiFi 模式，RK=192.168.234.1
#   bash recon2_d1max.sh 192.168.168.168  # 有線模式
#
# 帳密：robot / bot

set -uo pipefail

RK_IP="${1:-192.168.234.1}"
SSH_OPTS=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o ControlPersist=180
          -o ControlPath="/tmp/.recon2-%r@%h:%p")

OUT_DIR="${RECON_OUT:-./recon2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR/shm"

hr()  { printf '%s\n' "------------------------------------------------------------"; }
hdr() { hr; printf '### %s\n' "$*"; hr; }

# ---------------------------------------------------------------- 遠端偵察腳本
#
# ⚠️ 這裡故意「不用 set -u」。第一趟就是栽在 set -u + source ROS 環境。
#    每個區段包在 subshell 裡，某一段炸掉也不會拖垮整份腳本。
#
read -r -d '' PROBE <<'EOS'
sec() { echo; echo "======== $* ========"; }

# ---------------------------------------------------------------- ROS2
sec "ROS2 環境載入（第一趟就是死在這裡）"
(
  set +u                                   # ★ 關鍵：ROS setup 會引用一堆未設變數
  [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash
  [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash
  echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設}"
  echo "RMW=${RMW_IMPLEMENTATION:-未設}"
  echo "ros2 路徑: $(command -v ros2 || echo 找不到)"
  echo "載入成功 ✅"
) 2>&1

# 之後每個 ROS2 區段都自己重新 source（subshell 不會把環境帶出來）
ros2run() {
  (
    set +u
    [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash      >/dev/null 2>&1
    [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash >/dev/null 2>&1
    timeout "${TMO:-30}" "$@"
  ) 2>&1
}

sec "★★★ ROS2 topic 全表"
ros2run ros2 topic list | sort

sec "ROS2 node 全表"
ros2run ros2 node list | sort

sec "★★★ 底層相關 topic 的型別與 msg 定義"
(
  set +u
  [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash      >/dev/null 2>&1
  [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash >/dev/null 2>&1
  TOPICS=$(timeout 30 ros2 topic list 2>/dev/null)
  HITS=$(echo "$TOPICS" | grep -iE "low|motor|joint|cmd|state|imu|odom" )
  if [ -z "$HITS" ]; then
    echo "(沒有名稱含 low/motor/joint/cmd/state/imu/odom 的 topic)"
  else
    for t in $HITS; do
      echo "---- $t ----"
      TYPE=$(timeout 15 ros2 topic type "$t" 2>/dev/null | head -1)
      echo "型別: ${TYPE:-取不到}"
      [ -n "$TYPE" ] && timeout 15 ros2 interface show "$TYPE" 2>/dev/null
      echo
    done
  fi
) 2>&1

sec "廠商自訂 msg 介面"
ros2run ros2 interface list \
  | grep -viE "^ *(std_msgs|sensor_msgs|geometry_msgs|nav_msgs|builtin_interfaces|std_srvs|action_msgs|rcl_interfaces|lifecycle_msgs|tf2_msgs|diagnostic_msgs|shape_msgs|trajectory_msgs|visualization_msgs|unique_identifier_msgs|statistics_msgs|composition_interfaces|test_msgs|rosgraph_msgs|example_interfaces|actionlib_msgs|stereo_msgs|map_msgs|pcl_msgs|nav2_msgs|control_msgs|controller_manager_msgs)" \
  | head -80

sec "★★★ ros2_control 硬體介面（robot_hal 用 controller_manager）"
echo "-- list_hardware_interfaces --"
ros2run ros2 control list_hardware_interfaces
echo
echo "-- list_controllers --"
ros2run ros2 control list_controllers
echo
echo "-- list_hardware_components --"
ros2run ros2 control list_hardware_components

# ---------------------------------------------------------------- 運控設定
sec "★★★ /opt/export 完整結構"
ls -laR /opt/export 2>/dev/null | head -120

sec "★★★★ /opt/export/config/*.yaml —— 這台機器真實在用的運控參數"
for f in /opt/export/config/*.yaml /opt/export/config/*.yml; do
  [ -f "$f" ] || continue
  echo
  echo "-------- $f --------"
  cat "$f" 2>/dev/null || echo "(讀不到，可能需要權限)"
done

sec "/opt/export/mc/bin 內容"
ls -l /opt/export/mc/bin 2>/dev/null

sec "啟動腳本 /opt/runtime/bin"
ls -l /opt/runtime/bin 2>/dev/null
for f in /opt/runtime/bin/start_motion_control.sh \
         /opt/runtime/bin/bridge_config.yaml \
         /opt/runtime/bin/start_zenoh_router.sh; do
  [ -f "$f" ] || continue
  echo
  echo "-------- $f --------"
  cat "$f" 2>/dev/null
done

sec "robot_hal 設定（ros2_control 的關節定義應該在這）"
find /opt/robot/install/robot_hal/share -name "*.yaml" 2>/dev/null | head -10
for f in $(find /opt/robot/install/robot_hal/share -name "*.yaml" 2>/dev/null | head -6); do
  echo
  echo "-------- $f --------"
  cat "$f" 2>/dev/null
done

sec "robot_hal 的 URDF / xacro（若有）"
find /opt/robot/install -name "*.urdf*" -o -name "*.xacro" 2>/dev/null | head -20

sec "/opt/robot/install 的套件清單"
ls /opt/robot/install 2>/dev/null

# ---------------------------------------------------------------- SHM
sec "★★ /dev/shm 詳細（含 stat）"
ls -l /dev/shm/ 2>/dev/null
for f in joint_cmd joint_state imu_central; do
  echo
  echo "-- /dev/shm/$f --"
  stat -c '大小=%s  權限=%A  擁有者=%U:%G  修改時間=%y' "/dev/shm/$f" 2>/dev/null || echo "(不存在)"
done

sec "★★ SHM 是不是活的（同一檔案間隔 1 秒取兩次 md5）"
for f in joint_cmd joint_state imu_central; do
  A=$(md5sum "/dev/shm/$f" 2>/dev/null | cut -d' ' -f1)
  sleep 1
  B=$(md5sum "/dev/shm/$f" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$A" ]; then
    echo "$f: 讀不到"
  elif [ "$A" = "$B" ]; then
    echo "$f: 兩次 md5 相同 → 靜態（或更新很慢）"
  else
    echo "$f: 兩次 md5 不同 → ★ 活的串流"
  fi
done

sec "★★ SHM 開頭 256 bytes 當 float32 看（joint_state）"
od -A d -t f4 -N 256 /dev/shm/joint_state 2>/dev/null || echo "(讀不到)"

sec "★★ SHM 開頭 256 bytes 當 float32 看（joint_cmd）"
od -A d -t f4 -N 256 /dev/shm/joint_cmd 2>/dev/null || echo "(讀不到)"

sec "★★ SHM 開頭 128 bytes 當 uint64 看（找時戳，joint_state）"
od -A d -t u8 -N 128 /dev/shm/joint_state 2>/dev/null || echo "(讀不到)"

sec "偵察結束"
EOS

# ---------------------------------------------------------------- 執行
{
  echo "D1 Max 第二趟唯讀偵察   $(date -Is)"
  echo "RK3588 = $RK_IP"
  echo

  hdr "本機預檢"
  ip -br addr
  echo
  if ping -c2 -W2 "$RK_IP" >/dev/null 2>&1; then echo "✅ ping $RK_IP 通"; else echo "❌ ping $RK_IP 不通"; fi
  echo

  hdr "RK3588 偵察  (robot@$RK_IP)"
  ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "bash -s" <<< "$PROBE" 2>&1 | tee "$OUT_DIR/rk3588.log"
} 2>&1 | tee "$OUT_DIR/recon2.log"

# ---------------------------------------------------------------- 拉回 SHM 快照
hdr "拉回 SHM 快照（純讀取，不在狗上留檔）"
for f in joint_cmd joint_state imu_central; do
  for t in t0 t1; do
    if scp "${SSH_OPTS[@]}" -q "robot@$RK_IP:/dev/shm/$f" "$OUT_DIR/shm/${f}_${t}.bin" 2>/dev/null; then
      echo "✅ $f ($t)  $(stat -c%s "$OUT_DIR/shm/${f}_${t}.bin" 2>/dev/null) bytes"
    else
      echo "❌ $f ($t) 拉不回來"
    fi
    [ "$t" = t0 ] && sleep 2      # 間隔 2 秒，之後可 diff 出哪些位元組是活的
  done
done | tee "$OUT_DIR/shm/fetch.log"

ssh -O exit -o ControlPath="/tmp/.recon2-%r@%h:%p" "robot@$RK_IP" 2>/dev/null || true

# ---------------------------------------------------------------- 自動判讀
{
  hdr "自動判讀"
  # 只掃遠端輸出，且排除區段標題行（開頭是 ========），
  # 否則會被腳本自己印的關鍵字命中而誤報 —— 第一趟就中過。
  BODY=$(grep -v "^======== " "$OUT_DIR/rk3588.log" 2>/dev/null)

  echo "-- ROS2 有沒有跑起來 --"
  # ⚠️ 用 herestring 不要用 `echo "$BODY" | grep -q`：
  #    grep -q 命中後立刻結束 → echo 收到 SIGPIPE(141) → set -o pipefail 把整條管線
  #    判成失敗 → 明明命中卻回報「沒命中」。2026-08-25 第二趟就中過這個。
  if grep -q "載入成功" <<< "$BODY"; then
    echo "  ✅ 環境載入成功（第一趟的 bug 已修）"
  else
    echo "  ❌ 還是沒載入，看 rk3588.log 開頭"
  fi
  echo
  echo "-- 路線 B：rt/lowcmd / rt/lowstate --"
  if echo "$BODY" | grep -inE "lowcmd|lowstate"; then
    echo "  ★★★ 命中 → 底層可寫，照 msg 定義下 q_des/kp/kd/tau_ff"
  else
    echo "  (沒有命中)"
  fi
  echo
  echo "-- ros2_control 硬體介面 --"
  echo "$BODY" | grep -icE "command interface|state interface" | \
    xargs -I{} echo "  找到 {} 行 interface 描述（0 表示沒撈到）"
  echo
  echo "-- SHM 活性 --"
  echo "$BODY" | grep -E "活的串流|靜態|讀不到" | sed 's/^/  /'
  echo
  echo "-- /opt/export/config 有沒有撈到真實運控參數 --"
  echo "$BODY" | grep -c "FSM_RL_" | xargs -I{} echo "  FSM_RL_* 出現 {} 次（>0 表示撈到了）"
} 2>&1 | tee "$OUT_DIR/verdict.log"

echo
echo "===================================================="
echo "完成。輸出目錄：$OUT_DIR"
echo "  recon2.log   完整過程"
echo "  rk3588.log   原始輸出"
echo "  verdict.log  自動判讀"
echo "  shm/*.bin    joint_cmd / joint_state / imu_central 各兩份快照（間隔 2 秒）"
echo
echo "打包帶回：  tar czf ${OUT_DIR}.tar.gz $OUT_DIR"
echo "===================================================="
