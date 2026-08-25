#!/usr/bin/env bash
# recon_d1max.sh — 智元 D1 Max（zsm-1w）唯讀偵察
#
# 目的：一次回答「這台狗能怎麼控制」的三個關鍵問題
#   B) ROS2 上有沒有 rt/lowcmd / rt/lowstate（底層馬達介面）
#   C) /dev/shm 有沒有 spline_shm 之類的共享記憶體馬達介面
#   D) 韌體版本是多少（決定該用哪個版本的 SDK）
#
# ★ 安全設計：全程唯讀。
#   - 不用 sudo、不寫任何檔案到狗上、不啟停任何行程
#   - 只跑 cat / ls / ps / ros2 topic list / ss 這類查詢指令
#   - 輸出全部收在本機，不動狗的檔案系統
#
# 用法：
#   bash recon_d1max.sh                                  # 用預設 IP
#   bash recon_d1max.sh <RK_IP> <NX_IP>                  # 自訂
#   bash recon_d1max.sh 192.168.234.1 192.168.168.100    # WiFi 連線時的 RK + 有線的 NX
#
# 預設帳密（官方文件 2.5 設備登入）：
#   RK3588  robot@192.168.234.1  或 robot@192.168.168.168   密碼 bot
#   OrinNX  robot@192.168.168.100                            密碼 1
# 建議先把公鑰裝上去免密碼：ssh-copy-id robot@<ip>

set -uo pipefail

RK_IP="${1:-192.168.234.1}"
NX_IP="${2:-192.168.168.100}"
SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=no)

OUT_DIR="${RECON_OUT:-./recon_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

hr() { printf '%s\n' "------------------------------------------------------------"; }
hdr() { hr; printf '### %s\n' "$*"; hr; }

# 在遠端跑一段唯讀腳本；失敗不中斷整體流程
remote() {
  local ip="$1" label="$2" script="$3"
  hdr "$label  (robot@$ip)"
  if ! ssh "${SSH_OPTS[@]}" "robot@$ip" "bash -s" <<< "$script" 2>&1; then
    echo "[WARN] 連不上或指令失敗：robot@$ip"
  fi
  echo
}

# ---------------------------------------------------------------- 共用偵察段
read -r -d '' COMMON_PROBE <<'EOS'
set -u
echo "== uname =="
uname -a
echo
echo "== 韌體版本 /opt/release/version.yaml =="
cat /opt/release/version.yaml 2>/dev/null || echo "(讀不到)"
echo
echo "== OS =="
cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)=" || true
echo
echo "== /dev/shm 內容（★ 找 spline_shm / imu_shm）=="
ls -l /dev/shm/ 2>/dev/null || echo "(讀不到)"
echo
echo "== 行程（過濾運控相關關鍵字）=="
ps -eo pid,comm,args --sort=comm 2>/dev/null \
  | grep -iE "mc_ctrl|spline|robot-launch|robot_launch|motion|locomo|ctrl|daemon|ros|zenoh" \
  | grep -v grep || echo "(無匹配)"
echo
echo "== 監聽中的埠 =="
(ss -lntup 2>/dev/null || netstat -lntup 2>/dev/null) | head -40 || echo "(無工具)"
echo
echo "== robot-launch 節點狀態 =="
command -v robot-launch >/dev/null 2>&1 && (robot-launch help 2>&1 | head -30) || echo "(無 robot-launch)"
echo
echo "== ROS2 環境 =="
if [ -f /opt/runtime/env.bash ]; then
  echo "-- /opt/runtime/env.bash --"; cat /opt/runtime/env.bash
else
  echo "(無 /opt/runtime/env.bash)"
fi
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設}  RMW=${RMW_IMPLEMENTATION:-未設}"
echo
echo "== ★★ ROS2 topic list（找 lowcmd / lowstate）=="
if command -v ros2 >/dev/null 2>&1 || [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  [ -f /opt/runtime/env.bash ] && source /opt/runtime/env.bash 2>/dev/null
  [ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash 2>/dev/null
  timeout 20 ros2 topic list 2>&1 | sort || echo "(ros2 topic list 逾時或失敗)"
else
  echo "(這塊板沒有 ros2)"
fi
EOS

# ---------------------------------------------------------------- 執行
# 偵察輸出獨立收在 recon.log —— 判讀提示不寫進去，
# 否則下面的 grep 會被提示文字裡的關鍵字自己騙到（誤報「有命中」）。
{
  echo "D1 Max 唯讀偵察  $(date -Is)"
  echo "RK3588 = $RK_IP   OrinNX = $NX_IP"
  echo

  remote "$RK_IP" "RK3588（運動控制板）" "$COMMON_PROBE"
  remote "$NX_IP" "Orin NX（應用板）"     "$COMMON_PROBE"
} 2>&1 | tee "$OUT_DIR/recon.log"

hdr "判讀提示"
cat <<'EOT'
1. 全文搜底層指令 topic 名（見下方自動判讀）：
     有  → 走路線 B，底層可寫，照 D1 MaxPro 的訊息結構下 q_des/kp/kd/tau_ff
     沒有→ 看第 2 點
2. 全文搜共享記憶體介面名（在 /dev/shm 段）：
     有  → 走路線 C，可沿用 task6 的 shm_common.py 骨架（仍需先做唯讀驗證）
     沒有→ 走路線 A（高層 SDK），並向原廠詢問底層開放條件
3. 記下兩塊板的 version.yaml，比對 SDK README 的相容表再決定用 0.1.1 還是 0.2.1
4. 若 ros2 topic list 在 RK 上是空的，不代表沒有 topic —— 運控可能用板內 DDS 或
   不同的 ROS_DOMAIN_ID。回報結果時附上 env.bash 內容一起判讀。
EOT

echo
echo "完成。輸出：$OUT_DIR/recon.log"
echo "自動判讀（只掃偵察輸出，不含上面的提示文字）："
if grep -inE "lowcmd|lowstate|spline_shm" "$OUT_DIR/recon.log"; then
  echo "  ↑ ★ 有命中，底層可能開放"
else
  echo "  (沒有命中 lowcmd / lowstate / spline_shm)"
fi
