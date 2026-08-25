#!/usr/bin/env bash
# recon_d1max.sh — 智元 D1 Max（zsm-1w）唯讀偵察｜離線一趟收完版
#
# 目的：在「沒有網際網路、沒有人可以問」的現場，一趟把判斷後續路線需要的資料全撈回來。
#
#   B) ROS2 上有沒有 rt/lowcmd / rt/lowstate（底層馬達介面）→ 連 msg 定義一起撈
#   C) /dev/shm 有沒有 spline_shm 之類的共享記憶體馬達介面
#   D) 兩塊板的韌體版本（決定該跟原廠要哪個版本的 SDK）
#
# ★ 安全設計：全程唯讀。
#   - 不用 sudo、不寫任何檔案到狗上、不啟停任何行程、不送任何馬達指令
#   - 只跑 cat / ls / ps / ros2 topic list / ss 這類查詢指令
#   - 輸出全部收在本機
#
# 用法：
#   bash recon_d1max.sh                 # WiFi 模式（預設）：RK=192.168.234.1  NX=192.168.168.100
#   bash recon_d1max.sh --wired         # 有線模式：RK=192.168.168.168  NX=192.168.168.100
#   bash recon_d1max.sh <RK_IP> <NX_IP> # 自訂
#
# 帳密（官方文件 2.5 設備登入）：
#   RK3588  robot@...  密碼 bot
#   OrinNX  robot@...  密碼 1
#
# ⚠️ 強烈建議先裝金鑰，否則每個區段都要打一次密碼：
#   ssh-copy-id robot@192.168.234.1     # 密碼 bot
#   ssh-copy-id robot@192.168.168.100   # 密碼 1

set -uo pipefail

# ---------------------------------------------------------------- 參數
MODE="wifi"
case "${1:-}" in
  --wired) MODE="wired"; shift ;;
  --wifi)  MODE="wifi";  shift ;;
esac

if [ "$MODE" = "wired" ]; then
  RK_IP="${1:-192.168.168.168}"
else
  RK_IP="${1:-192.168.234.1}"
fi
NX_IP="${2:-192.168.168.100}"

SSH_OPTS=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o ControlPersist=120
          -o ControlPath="/tmp/.recon-%r@%h:%p")

OUT_DIR="${RECON_OUT:-./recon_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

hr()  { printf '%s\n' "------------------------------------------------------------"; }
hdr() { hr; printf '### %s\n' "$*"; hr; }

# ---------------------------------------------------------------- 本機預檢
preflight() {
  hdr "本機預檢（$MODE 模式）"
  echo "-- 介面 --";  ip -br addr
  echo
  echo "-- 路由 --";  ip route
  echo
  if [ "$MODE" = "wifi" ]; then
    echo "-- 有線網段的路由（官方文件 5.7）--"
    if ip route | grep -q "192.168.168.0/24"; then
      echo "✅ 已有 192.168.168.0/24 的路由，Orin NX 應該連得到"
    else
      echo "⚠️ 沒有 192.168.168.0/24 的路由 → 連得到 RK 但連不到 Orin NX"
      echo "   補上：sudo ip route add 192.168.168.0/24 via ${RK_IP}"
    fi
    echo
    echo "-- 預設路由是否還在對外的介面 --"
    ip route | grep '^default' || echo "⚠️ 沒有預設路由（現在是離線狀態，這是預期的）"
  fi
  echo
  echo "-- 連通性 --"
  for ip in "$RK_IP" "$NX_IP"; do
    if ping -c2 -W2 "$ip" >/dev/null 2>&1; then
      echo "✅ ping $ip 通"
    else
      echo "❌ ping $ip 不通"
    fi
  done
  echo
  echo "-- ARP（有 MAC 才代表實體層通；狗的網口可能沒有 LED，別盯燈）--"
  ip neigh | grep -E "192\.168\.(234|168)\." || echo "(ARP 表裡沒有狗)"
  echo
}

# ---------------------------------------------------------------- 遠端偵察腳本
# 全部塞在一個 here-doc 裡，讓每台板子只要輸入一次密碼。
#
# ⚠️ 這裡故意「不用 set -u」。2026-08-25 第一次上機就是栽在這：
#    set -u 碰上 /opt/runtime/env.bash 的
#        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/runtime/lib
#    非互動 SSH 下 LD_LIBRARY_PATH 未設 → unbound variable → shell 當場結束，
#    ROS2 整段完全沒跑到，而且 2>/dev/null 把錯誤也吞了，看起來像正常結束。
read -r -d '' PROBE <<'EOS'
sec() { echo; echo "======== $* ========"; }

sec "身分"
hostname 2>/dev/null; uname -a
grep -E "^(NAME|VERSION)=" /etc/os-release 2>/dev/null

sec "★ 韌體版本 /opt/release/version.yaml"
cat /opt/release/version.yaml 2>/dev/null || echo "(讀不到)"

sec "SHM 內容"
# 2026-08-25 實測：D1 Max 是 joint_cmd / joint_state / imu_central，
# 不是 D1 EDU 的 spline_shm。關鍵字別寫死成單一名稱。
ls -l /dev/shm/ 2>/dev/null || echo "(讀不到)"

sec "行程（運控相關關鍵字）"
ps -eo pid,ppid,comm,args --sort=comm 2>/dev/null \
  | grep -iE "mc_ctrl|spline|robot-launch|robot_launch|motion|locomo|ctrl|daemon|ros|zenoh|rmw" \
  | grep -v grep || echo "(無匹配)"

sec "行程總數與前 20 名 CPU"
ps -eo pcpu,pid,comm --sort=-pcpu 2>/dev/null | head -21

sec "監聽中的埠"
(ss -lntup 2>/dev/null || netstat -lntup 2>/dev/null) | head -50 || echo "(無工具)"

sec "本機網路設定（了解狗自己的網段）"
ip -br addr 2>/dev/null; echo "--"; ip route 2>/dev/null

sec "robot-launch 節點"
if command -v robot-launch >/dev/null 2>&1; then
  robot-launch help 2>&1 | head -40
else
  echo "(無 robot-launch)"
fi

sec "/opt 目錄結構（深度 2）"
ls -la /opt 2>/dev/null
find /opt -maxdepth 2 -type d 2>/dev/null | head -40

sec "檔案系統裡有沒有 lowlevel / spline 的痕跡"
find /opt /usr/local /home/robot -maxdepth 4 \
     \( -iname "*lowcmd*" -o -iname "*lowstate*" -o -iname "*spline*" -o -iname "*lowlevel*" \) \
     2>/dev/null | head -30 || true
echo "(以上為搜尋結果，空白表示沒找到)"

# ---------------- ROS2 ----------------
sec "ROS2 環境"
if [ -f /opt/runtime/env.bash ]; then
  echo "-- /opt/runtime/env.bash --"; cat /opt/runtime/env.bash
else
  echo "(無 /opt/runtime/env.bash)"
fi
# ★ source 一定要包在 set +u 的 subshell 裡，且不要吞掉 stderr（見檔頭的說明）
(
  set +u
  [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash
  [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash
  echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設}  RMW=${RMW_IMPLEMENTATION:-未設}"
  echo "ros2 路徑: $(command -v ros2 || echo 找不到)"
) 2>&1

# 之後每個 ros2 指令都自己重新 source（subshell 的環境不會帶出來）
ros2run() {
  (
    set +u
    [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash      >/dev/null 2>&1
    [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash >/dev/null 2>&1
    timeout 30 "$@"
  ) 2>&1
}

if [ -x /opt/ros/humble/bin/ros2 ] || command -v ros2 >/dev/null 2>&1; then
  sec "★★★ ROS2 topic 全表"
  TOPICS=$(ros2run ros2 topic list | sort)
  if [ -z "$TOPICS" ]; then
    echo "(topic list 是空的 —— 不代表沒有 topic，可能是 DOMAIN_ID 或 RMW 不對，見上方 env)"
  else
    echo "$TOPICS"
  fi

  sec "ROS2 node 全表"
  ros2run ros2 node list | sort

  sec "★★★ 底層相關 topic 的型別與 msg 定義"
  HITS=$(echo "$TOPICS" | grep -iE "low|motor|joint|cmd|state" || true)
  if [ -z "$HITS" ]; then
    echo "(沒有名稱含 low/motor/joint/cmd/state 的 topic)"
  else
    for t in $HITS; do
      echo "---- $t ----"
      TYPE=$(ros2run ros2 topic type "$t" | head -1)
      echo "型別: ${TYPE:-(取不到)}"
      if [ -n "$TYPE" ]; then
        echo "-- msg 定義 --"
        ros2run ros2 interface show "$TYPE"
      fi
      echo
    done
  fi

  sec "★★★ ros2_control 硬體介面"
  ros2run ros2 control list_hardware_interfaces
  ros2run ros2 control list_controllers

  sec "所有自訂 msg 介面（找廠商私有型別）"
  ros2run ros2 interface list \
    | grep -viE "^ *(std_msgs|sensor_msgs|geometry_msgs|nav_msgs|builtin_interfaces|std_srvs|action_msgs|rcl_interfaces|lifecycle_msgs|tf2_msgs|diagnostic_msgs|shape_msgs|trajectory_msgs|visualization_msgs|unique_identifier_msgs|statistics_msgs|composition_interfaces|test_msgs|rosgraph_msgs|example_interfaces|actionlib_msgs|stereo_msgs|map_msgs|pcl_msgs|nav2_msgs|control_msgs|controller_manager_msgs)" \
    | head -60
else
  sec "ROS2"
  echo "(這塊板沒有 ros2 指令)"
fi

sec "偵察結束"
EOS

# ---------------------------------------------------------------- 執行
remote() {
  local ip="$1" label="$2" tag="$3"
  hdr "$label  (robot@$ip)"
  if ssh "${SSH_OPTS[@]}" "robot@$ip" "bash -s" <<< "$PROBE" 2>&1 | tee "$OUT_DIR/$tag.log"; then
    :
  else
    echo "[WARN] 連不上或指令失敗：robot@$ip"
  fi
  echo
}

{
  echo "D1 Max 唯讀偵察   $(date -Is)"
  echo "模式=$MODE   RK3588=$RK_IP   OrinNX=$NX_IP"
  echo
  preflight
  remote "$RK_IP" "RK3588（運動控制板）" "rk3588"
  remote "$NX_IP" "Orin NX（應用板）"     "orinnx"
} 2>&1 | tee "$OUT_DIR/recon.log"

# 關掉共用連線
for ip in "$RK_IP" "$NX_IP"; do
  ssh -O exit -o ControlPath="/tmp/.recon-%r@%h:%p" "robot@$ip" 2>/dev/null || true
done

# ---------------------------------------------------------------- 自動判讀
# 只掃遠端輸出（rk3588.log / orinnx.log），不掃 recon.log，
# 否則下面印出的提示文字會被自己的 grep 命中而誤報。
{
  hdr "自動判讀"
  # ⚠️ 一定要排除區段標題行（開頭 ========），否則會被腳本自己印的關鍵字命中而誤報。
  #    2026-08-25 第一趟就中過：標題「找 spline_shm」讓判讀誤報「路線 C 命中」。
  BODY=$(grep -hv "^======== " "$OUT_DIR"/rk3588.log "$OUT_DIR"/orinnx.log 2>/dev/null)
  FOUND=0
  echo "-- 路線 B：ROS2 底層指令介面 --"
  if echo "$BODY" | grep -inE "lowcmd|lowstate"; then
    echo "  ★ 命中 → 底層可寫，照 D1 MaxPro 的 rt/lowcmd 結構走"; FOUND=1
  else
    echo "  (沒有命中)"
  fi
  echo
  echo "-- 路線 C：共享記憶體馬達介面 --"
  # 名稱依機型而異：D1 EDU 是 spline_shm，D1 Max 是 joint_cmd / joint_state
  if echo "$BODY" | grep -inE "spline_shm|joint_cmd|joint_state|motor_cmd"; then
    echo "  ★ 命中 → 可沿用 task6 的 shm_common.py 骨架（仍須先做唯讀驗證）"; FOUND=1
  else
    echo "  (沒有命中)"
  fi
  echo
  [ "$FOUND" = 0 ] && echo "→ 兩條都沒命中：走高層 SDK，並向原廠詢問底層開放條件。"
  echo
  echo "-- 韌體版本（拿去對照 SDK README 的相容表）--"
  grep -A6 "韌體版本" "$OUT_DIR"/rk3588.log "$OUT_DIR"/orinnx.log 2>/dev/null | head -30 || echo "  (沒撈到)"
  echo
  echo "⚠️ ROS2 topic list 是空的，不代表沒有 topic —— 可能是 ROS_DOMAIN_ID 或 RMW 不對。"
  echo "   回報時請連 env.bash 那一段一起帶回來判讀。"
} 2>&1 | tee "$OUT_DIR/verdict.log"

echo
echo "===================================================="
echo "完成。輸出目錄：$OUT_DIR"
echo "  recon.log    完整過程（含本機預檢）"
echo "  rk3588.log   RK3588 的原始輸出"
echo "  orinnx.log   Orin NX 的原始輸出"
echo "  verdict.log  自動判讀結果"
echo
echo "打包帶回：  tar czf ${OUT_DIR}.tar.gz $OUT_DIR"
echo "===================================================="
