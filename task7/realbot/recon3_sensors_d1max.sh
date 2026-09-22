#!/usr/bin/env bash
# recon3_sensors_d1max.sh —— D1 Max（中狗）第三趟唯讀偵察
#                            感測器盤點 ＋ 兩情境資源採樣
#
# 目的：把報告的這四項從「手冊宣稱」變成「實機量到」：
#   1.3  CPU / GPU / Memory 資源使用情形   ← 待機 vs 原廠走路兩段對照，兩塊板都採
#   7.1  感測器種類與數量                  ← ROS2 topic/node 全表 ＋ 驅動設定檔 ＋ 裝置列舉
#   7.2  Camera：解析度 / Frame Rate       ← 設定檔 ＋ ROS2 topic ＋ 從 PC 用 ffprobe 實拉 RTSP
#   7.4  LiDAR 等其他感測器規格            ← 光達 yaml ＋ PointCloud2 欄位 ＋ 實測 hz ＋ Range 欄位
#
# 前兩趟：recon_d1max.sh（判斷控制路線）、recon2_d1max.sh（運控設定與 SHM）。
# 這趟補的是感測與資源，那兩趟都沒碰過。
#
# ★ 安全設計
#   - 不送任何馬達指令、不寫 /dev/shm、不啟停任何行程、**不在狗上寫任何檔案**
#     （資源採樣器是用 `ssh ... 'python3 -' < recon3_sample.py` 從 stdin 餵進去的）
#   - sudo 只用唯讀查詢（lsusb -v / dmesg / debugfs 的 load 節點）；--no-sudo 可完全關掉
#   - 走路那段由**你**用遙控器操作，腳本只負責在旁邊採樣
#
# 用法：
#   bash recon3_sensors_d1max.sh                    # WiFi：RK=192.168.234.1 NX=192.168.168.100
#   bash recon3_sensors_d1max.sh --wired            # 有線：RK=192.168.168.168
#   bash recon3_sensors_d1max.sh --no-walk          # 只做靜態盤點 ＋ 待機採樣（不必操狗）
#   bash recon3_sensors_d1max.sh --idle 60 --walk 60
#   bash recon3_sensors_d1max.sh --rk-only          # NX 連不上時
#   bash recon3_sensors_d1max.sh <RK_IP> <NX_IP>
#
# 環境變數：RK_PW（預設 bot）、NX_PW（預設 1）、RECON_OUT（輸出目錄）
#
# ⚠️ 建議先裝金鑰，否則每次連線都要打密碼：
#   ssh-copy-id robot@192.168.234.1     # 密碼 bot
#   ssh-copy-id robot@192.168.168.100   # 密碼 1

set -uo pipefail

# ================================================================ 參數
MODE="wifi"
IDLE_SECS=30
WALK_SECS=40
DO_WALK=1
DO_RTSP=1
USE_SUDO=1
RK_ONLY=0
NX_ONLY=0
SAMPLE_ONLY=0
STATIC_ONLY=0
POS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --wired)    MODE="wired"; shift ;;
    --wifi)     MODE="wifi";  shift ;;
    --idle)     IDLE_SECS="$2"; shift 2 ;;
    --walk)     WALK_SECS="$2"; shift 2 ;;
    --no-walk)  DO_WALK=0; shift ;;
    --no-rtsp)  DO_RTSP=0; shift ;;
    --no-sudo)  USE_SUDO=0; shift ;;
    --rk-only)  RK_ONLY=1; shift ;;
    --sample-only) SAMPLE_ONLY=1; shift ;;
    --static-only) STATIC_ONLY=1; shift ;;
    --nx-only)  NX_ONLY=1; shift ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *)          POS+=("$1"); shift ;;
  esac
done

if [ "$MODE" = "wired" ]; then
  RK_IP="${POS[0]:-192.168.168.168}"
else
  RK_IP="${POS[0]:-192.168.234.1}"
fi
NX_IP="${POS[1]:-192.168.168.100}"

RK_PW="${RK_PW:-bot}"
NX_PW="${NX_PW:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLER="$HERE/recon3_sample.py"
REPORTER="$HERE/recon3_report.py"

if [ ! -f "$SAMPLER" ]; then
  echo "❌ 找不到採樣器 $SAMPLER —— 這支腳本要和 recon3_sample.py 放在同一個目錄" >&2
  exit 1
fi

SSH_OPTS=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o ControlPersist=600
          -o ControlPath="/tmp/.recon3-%r@%h:%p")

OUT_DIR="${RECON_OUT:-./recon3_$(date +%Y%m%d_%H%M%S)}"
# ⚠️ 防呆：`RECON_OUT=$(ls -dt recon3_* | head -1)` 會抓到打包好的 .tar.gz（2026-09-22 踩過），
#    輸出目錄變成一個「檔案」，後面每個重導向都失敗 —— 而且是在狗走完之後才發現。
#    要選最近的目錄請用 `ls -dt recon3_*/`（尾巴的斜線才只列目錄）。
if [ -e "$OUT_DIR" ] && [ ! -d "$OUT_DIR" ]; then
  echo "❌ RECON_OUT=$OUT_DIR 存在但不是目錄（打包檔？）。" >&2
  echo "   選最近的目錄：RECON_OUT=\$(ls -dt recon3_*/ | head -1)" >&2
  exit 1
fi
mkdir -p "$OUT_DIR" || { echo "❌ 建不了輸出目錄 $OUT_DIR" >&2; exit 1; }
if [ ! -w "$OUT_DIR" ]; then
  echo "❌ 輸出目錄 $OUT_DIR 不可寫" >&2; exit 1
fi

hr()  { printf '%s\n' "------------------------------------------------------------"; }
hdr() { hr; printf '### %s\n' "$*"; hr; }

# 要跑哪幾塊板
BOARDS=()
[ "$NX_ONLY" = 0 ] && BOARDS+=("rk")
[ "$RK_ONLY" = 0 ] && BOARDS+=("nx")

ip_of()    { [ "$1" = rk ] && echo "$RK_IP" || echo "$NX_IP"; }
pw_of()    { [ "$1" = rk ] && echo "$RK_PW" || echo "$NX_PW"; }
label_of() { [ "$1" = rk ] && echo "RK3588（運控板）" || echo "Orin NX（應用板）"; }

declare -A ALIVE SUDO_OK PY_CMD

# ================================================================ 本機預檢
preflight() {
  hdr "本機預檢（$MODE 模式）"
  echo "-- 介面 --"; ip -br addr
  echo; echo "-- 路由 --"; ip route
  echo
  if [ "$MODE" = "wifi" ] && [ "$RK_ONLY" = 0 ]; then
    echo "-- 有線網段的路由（要連 Orin NX 就需要這條）--"
    # herestring 而非管線：pipefail 下 `cmd | grep -q` 會因 SIGPIPE 誤判失敗
    if grep -q "192.168.168.0/24" <<< "$(ip route)"; then
      echo "✅ 已有 192.168.168.0/24 的路由"
    else
      echo "⚠️ 沒有 192.168.168.0/24 的路由 → 連得到 RK 但連不到 Orin NX"
      echo "   補上：sudo ip route add 192.168.168.0/24 via ${RK_IP}"
      echo "   （感測器驅動與設定檔幾乎全在 NX，這條不通 7.1/7.4 會缺一大半）"
    fi
    echo
  fi
  echo "-- 本機 ffprobe（7.2 的 RTSP 實測要用）--"
  if command -v ffprobe >/dev/null 2>&1; then
    echo "✅ $(ffprobe -version 2>/dev/null | head -1)"
  else
    echo "⚠️ 沒有 ffprobe → RTSP 實測會跳過（裝：sudo pacman -S ffmpeg / apt install ffmpeg）"
    DO_RTSP=0
  fi
  echo
  echo "-- 連通性 --"
  for b in "${BOARDS[@]}"; do
    local ip; ip="$(ip_of "$b")"
    if ping -c2 -W2 "$ip" >/dev/null 2>&1; then
      echo "✅ ping $ip 通（$(label_of "$b")）"
    else
      echo "❌ ping $ip 不通（$(label_of "$b")）"
    fi
  done
  echo
  echo "-- ARP（有 MAC 才代表實體層通；狗的網口沒有 LED，別盯燈）--"
  ip neigh | grep -E "192\.168\.(234|168|1|2)\." || echo "(ARP 表裡沒有狗)"
  echo
}

# ================================================================ 連線暖機 ＋ sudo 授權
# 先一塊板一塊板地建立 ControlMaster 連線：密碼一次打完，
# 後面的並行採樣才不會兩塊板同時跳密碼提示互相打斷。
warmup() {
  hdr "連線暖機與權限確認"
  for b in "${BOARDS[@]}"; do
    local ip pw; ip="$(ip_of "$b")"; pw="$(pw_of "$b")"
    ALIVE[$b]=0; SUDO_OK[$b]=0; PY_CMD[$b]="python3 -"

    echo "-- $(label_of "$b")  robot@$ip --"
    if ! ssh "${SSH_OPTS[@]}" "robot@$ip" "true" 2>/dev/null; then
      echo "   ❌ 連不上，這塊板整段跳過"
      continue
    fi
    ALIVE[$b]=1
    echo "   ✅ SSH 通  $(ssh "${SSH_OPTS[@]}" "robot@$ip" 'hostname; python3 -V 2>&1 | head -1' 2>/dev/null | tr '\n' ' ')"

    # python3 在不在（採樣器要用）
    if ! ssh "${SSH_OPTS[@]}" "robot@$ip" "command -v python3 >/dev/null" 2>/dev/null; then
      echo "   ⚠️ 這塊板沒有 python3 → 1.3 的採樣會跳過（不猜數字）"
      PY_CMD[$b]=""
    fi

    if [ "$USE_SUDO" = 0 ]; then
      echo "   （--no-sudo：不嘗試 sudo）"
      continue
    fi
    # 先試免密碼；不行才餵一次密碼把 sudo 憑證快取起來，之後一律 sudo -n
    if ssh "${SSH_OPTS[@]}" "robot@$ip" "sudo -n true" 2>/dev/null; then
      SUDO_OK[$b]=1; echo "   ✅ sudo 免密碼可用"
    elif ssh "${SSH_OPTS[@]}" "robot@$ip" \
           "printf '%s\n' '$pw' | sudo -S -p '' -v" >/dev/null 2>&1 \
         && ssh "${SSH_OPTS[@]}" "robot@$ip" "sudo -n true" 2>/dev/null; then
      SUDO_OK[$b]=1; echo "   ✅ sudo 憑證已快取（唯讀查詢用）"
    else
      echo "   ⚠️ sudo 不可用 → NPU/GPU 的 debugfs 負載與 lsusb -v 會標「未取到」"
    fi
    # 採樣器跑在 root 下才讀得到 debugfs 的 load 節點；它本身唯讀
    if [ "${SUDO_OK[$b]}" = 1 ] && [ -n "${PY_CMD[$b]}" ]; then
      PY_CMD[$b]="sudo -n python3 -"
    fi
  done
  echo
}

# ================================================================ 靜態盤點（遠端）
#
# ⚠️ 遠端這段**故意不用 set -u**。第一趟偵察就是死在這：
#    set -u 碰上 /opt/runtime/env.bash 的
#        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/runtime/lib
#    非互動 SSH 下 LD_LIBRARY_PATH 未設 → unbound variable → shell 當場結束，
#    而 2>/dev/null 把錯誤吞了，看起來像正常結束（ROS2 整段完全沒跑到）。
#
read -r -d '' PROBE <<'EOS'
sec() { echo; echo "======== $* ========"; }
kv()  { echo "@@KV $1=$2"; }
# sudo 包裝：沒授權就明說跳過，不要讓輸出看起來像「查了但沒有」
S() {
  if [ "${SUDO_OK:-0}" = 1 ]; then sudo -n "$@" 2>&1
  else echo "(未取到：無 sudo 授權，略過 $*)"; fi
}

# ---------------------------------------------------------------- 身分
sec "身分與韌體"
kv host      "$(hostname 2>/dev/null)"
kv kernel    "$(uname -r)"
kv arch      "$(uname -m)"
kv os        "$(grep -m1 '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '\"')"
kv board     "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
kv python3   "$(python3 -V 2>&1)"
uname -a
echo "-- /opt/release/version.yaml --"
cat /opt/release/version.yaml 2>/dev/null || echo "(讀不到)"
FW="$(grep -m1 -iE 'version' /opt/release/version.yaml 2>/dev/null | tr -d ' ')"
kv firmware "${FW:-未取到}"

# ---------------------------------------------------------------- 1.3 靜態規格
sec "★ 1.3　CPU 規格"
kv ncpu "$(nproc --all 2>/dev/null)"
if command -v lscpu >/dev/null 2>&1; then
  lscpu 2>/dev/null
  kv cpu_model "$(lscpu 2>/dev/null | grep -m1 -E '^Model name' | cut -d: -f2- | xargs)"
  kv cpu_maxmhz "$(lscpu 2>/dev/null | grep -m1 -E 'CPU max MHz' | cut -d: -f2- | xargs)"
else
  echo "(無 lscpu)"; grep -E 'model name|Hardware|CPU part|processor' /proc/cpuinfo | head -30
fi
echo "-- /proc/cmdline（看核心隔離）--"
cat /proc/cmdline 2>/dev/null
kv isolcpus "$(cat /sys/devices/system/cpu/isolated 2>/dev/null)"
kv nohz_full "$(cat /sys/devices/system/cpu/nohz_full 2>/dev/null)"
echo "-- 每核最高頻與 governor --"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  [ -d "$p" ] || continue
  echo "$(basename "$p")  cpus=$(cat "$p/affected_cpus" 2>/dev/null)  max=$(cat "$p/cpuinfo_max_freq" 2>/dev/null)  gov=$(cat "$p/scaling_governor" 2>/dev/null)"
done

sec "★ 1.3　記憶體"
free -h 2>/dev/null
kv mem_total_kb "$(grep -m1 MemTotal /proc/meminfo | awk '{print $2}')"
kv swap_total_kb "$(grep -m1 SwapTotal /proc/meminfo | awk '{print $2}')"
swapon --show 2>/dev/null || echo "(無 swap 或無工具)"

sec "★ 1.3　GPU / NPU 裝置與驅動"
echo "-- devfreq 節點（GPU/NPU 的頻率與負載大多掛這裡）--"
for d in /sys/class/devfreq/*; do
  [ -e "$d" ] || continue
  echo "$d -> $(readlink -f "$d" 2>/dev/null)"
  echo "   cur_freq=$(cat "$d/cur_freq" 2>/dev/null)  max=$(cat "$d/max_freq" 2>/dev/null)  load=$(cat "$d/load" 2>/dev/null)  gov=$(cat "$d/governor" 2>/dev/null)"
done
echo "-- 裝置節點 --"
ls -l /dev/dri /dev/mali0 /dev/rknpu* /dev/nvhost* 2>/dev/null | head -30
echo "-- RKNPU（RK3588）--"
kv rknpu_version "$(cat /sys/kernel/debug/rknpu/version 2>/dev/null)"
echo "version: $(S cat /sys/kernel/debug/rknpu/version)"
echo "load:    $(S cat /sys/kernel/debug/rknpu/load)"
echo "-- Jetson（Orin NX）--"
if command -v tegrastats >/dev/null 2>&1; then
  echo "tegrastats 存在：$(command -v tegrastats)"
  timeout 4 tegrastats --interval 1000 2>&1 | head -3
else
  echo "(無 tegrastats)"
fi
command -v nvpmodel >/dev/null 2>&1 && nvpmodel -q 2>&1 | head -10 || echo "(無 nvpmodel)"
cat /etc/nv_tegra_release 2>/dev/null || echo "(無 /etc/nv_tegra_release)"
echo "-- OpenCL / CUDA 痕跡 --"
ls /usr/lib/aarch64-linux-gnu/ 2>/dev/null | grep -iE "^lib(cuda|cudnn|nvinfer|mali|rknn|OpenCL)" | head -20
ls /usr/local 2>/dev/null | head -20

sec "★ 1.3　溫度與風扇"
for z in /sys/class/thermal/thermal_zone*; do
  [ -e "$z" ] || continue
  printf '%s  %s  %s\n' "$(basename "$z")" "$(cat "$z/type" 2>/dev/null)" "$(cat "$z/temp" 2>/dev/null)"
done
ls /sys/class/hwmon/*/name 2>/dev/null | while read -r n; do echo "$n: $(cat "$n")"; done

# ---------------------------------------------------------------- 7.1 裝置列舉
sec "★ 7.1　USB / 序列埠 / 影像裝置"
echo "-- lsusb --"
lsusb 2>/dev/null || echo "(無 lsusb)"
echo "-- lsusb -t --"
lsusb -t 2>/dev/null || echo "(無)"
echo "-- lsusb -v（只留 iProduct / bInterfaceClass，需要 sudo 才完整）--"
# 不能寫成 `S lsusb -v | grep ...` —— 沒授權時 S 印的那句提示會被 grep 濾掉，
# 輸出就變成空白，看起來像「查過但沒有」。缺資料和沒查過必須分得出來。
if [ "${SUDO_OK:-0}" = 1 ]; then
  sudo -n lsusb -v 2>/dev/null | grep -iE "iProduct|iManufacturer|bInterfaceClass" | head -40
else
  echo "(未取到：無 sudo 授權，略過 lsusb -v)"
fi
echo "-- 影像裝置 --"
ls -l /dev/video* 2>/dev/null || echo "(沒有 /dev/video*)"
echo "-- 序列埠（IMU / GPS / UWB 常走 UART）--"
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS* /dev/ttyS[0-9] 2>/dev/null || echo "(沒有)"
echo "-- CAN --"
ip -d link show type can 2>/dev/null | head -30 || echo "(無 CAN 介面)"
echo "-- i2c / spi --"
ls /dev/i2c-* /dev/spidev* 2>/dev/null || echo "(沒有)"
echo "-- 網路介面（光達走獨立網段）--"
ip -br addr 2>/dev/null
ip route 2>/dev/null

sec "★ 7.1／7.4　光達的獨立網段（從狗上看）"
for lip in 192.168.1.102 192.168.2.102; do
  if ping -c2 -W2 "$lip" >/dev/null 2>&1; then
    echo "✅ ping $lip 通"
    echo "   HTTP: $(curl -s -m 4 -o /dev/null -w '%{http_code}' "http://$lip" 2>/dev/null)"
    curl -s -m 4 "http://$lip" 2>/dev/null | head -5
  else
    echo "❌ ping $lip 不通"
  fi
done
echo "-- 光達 UDP 埠有沒有在收（RoboSense msop 6699 / difop 7788 是慣用值）--"
(ss -lunp 2>/dev/null || netstat -lunp 2>/dev/null) | head -40

sec "★ 7.1　感測相關行程"
ps -eo pid,pcpu,rss,comm,args --sort=-pcpu 2>/dev/null \
  | grep -iE "lidar|rslidar|uss|uwb|imu|gps|rtk|camera|mediamtx|nav2|arc_|localization|mc_ctrl|robot_" \
  | grep -v grep | head -40
echo "-- mc_ctrl 的核親和性（cpu7 隔離是不是真的只給運控）--"
for p in $(pgrep -x mc_ctrl 2>/dev/null); do
  echo "pid $p  Cpus_allowed_list=$(grep Cpus_allowed_list "/proc/$p/status" 2>/dev/null | awk '{print $2}')"
done
echo "-- robot-launch 服務清單 --"
command -v robot-launch >/dev/null 2>&1 && timeout 10 robot-launch list 2>&1 | head -40 || echo "(無 robot-launch)"

# ---------------------------------------------------------------- 7.2 相機
sec "★ 7.2　相機設定與能力"
# ⚠️ RK3588 有 40 幾個 /dev/video*（rkisp 的 statistics / params / rawrd 全算一個節點）。
#    第一趟對每一個都跑兩次 v4l2-ctl，光這段就兩千多行。這裡只留真正會出畫面的。
if command -v v4l2-ctl >/dev/null 2>&1; then
  echo "-- v4l2-ctl --list-devices（一次看完拓樸）--"
  v4l2-ctl --list-devices 2>&1 | head -60
fi
NCAM=0
for d in /dev/video*; do
  [ -e "$d" ] || continue
  echo "@@CAMDEV $d"
  command -v v4l2-ctl >/dev/null 2>&1 || { echo "(無 v4l2-ctl)"; continue; }
  CARD=$(v4l2-ctl -d "$d" --info 2>/dev/null | grep -m1 "Card type" | cut -d: -f2- | xargs)
  echo "   Card: ${CARD:-未取到}"
  case "$CARD" in
    *statistic*|*params*|*iqtool*|*rawrd*|*selfpath*|*fbcpath*|*dec*|*enc*|"")
      echo "   （非擷取節點，略過格式列舉）"; continue ;;
  esac
  [ "$NCAM" -ge 6 ] && { echo "   （已列舉 6 個擷取節點，其餘略過）"; continue; }
  NCAM=$((NCAM+1))
  v4l2-ctl -d "$d" --list-formats-ext 2>&1 | grep -E "\[|Size: Discrete|Interval" | sort -u | head -20
done
echo "-- RTSP 伺服器（mediamtx）--"
(ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E "8554|8082|1935" || echo "(沒看到 RTSP 埠在 listen)"
MTX=$(find /opt /etc /home/robot -maxdepth 6 -type f \( -iname "mediamtx*.yml" -o -iname "mediamtx*.yaml" -o -iname "mediamtx*.conf" \) 2>/dev/null | head -5)
for f in $MTX; do
  echo "@@FILE $f"
  echo "  -- 全域設定（前 40 行）--"
  grep -vE "^\s*#" "$f" 2>/dev/null | grep -vE "^\s*$" | head -40
  echo "  -- ★ paths 段（RTSP 的路徑名稱與來源就在這，第一趟被 head 截掉了）--"
  sed -n '/^paths:/,$p' "$f" 2>/dev/null | head -60
  echo "  -- rtsp 相關設定 --"
  grep -inE "^rtsp|^rtmp|^hls|^webrtc|encryption" "$f" 2>/dev/null | head -20
done
[ -z "$MTX" ] && echo "(找不到 mediamtx 設定檔)"

# ---------------------------------------------------------------- 感測驅動設定檔
sec "★ 7.1／7.2／7.4　感測驅動設定檔"
# ⚠️ 2026-09-22 第一趟就是這裡漏掉光達設定：檔名叫 `config.yaml`，
#    只用檔名 pattern 找一定找不到。改成**按內容**找關鍵字。
CFG_NAME=$(find /opt/robot /opt/export /opt/runtime /etc/robot /home/robot -maxdepth 8 -type f \
        \( -iname "*lidar*" -o -iname "*uss*" -o -iname "*uwb*" -o -iname "*imu*" \
           -o -iname "*gps*" -o -iname "*rtk*" -o -iname "*camera*" -o -iname "*sensor*" \) \
        \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.conf" -o -name "*.ini" \) \
        2>/dev/null | head -30)
CFG_BODY=$(timeout 90 grep -rls --include="*.yaml" --include="*.yml" \
        -e lidar_type -e msop_port -e difop -e rslidar -e uss_ -e ultrasonic \
        -e camera_name -e frame_rate -e resolution \
        /opt/robot /opt/export /opt/runtime 2>/dev/null | head -30)
CFG=$(printf '%s\n%s\n' "$CFG_NAME" "$CFG_BODY" | grep -v '^$' | sort -u)
if [ -z "$CFG" ]; then
  echo "(找不到感測驅動設定檔 —— 換條路：看 /opt/robot/install/*/share/*/config)"
  ls -d /opt/robot/install/*/share/*/config 2>/dev/null | head -20
else
  for f in $CFG; do
    echo "@@FILE $f"
    sed -n '1,140p' "$f" 2>/dev/null
    echo "   … (截斷於 140 行)"
  done
fi
echo "-- 光達設定的關鍵欄位（挑出來標記，方便產表）--"
for f in $CFG; do
  grep -inE "lidar_type|device_ip|dest_ip|msop|difop|rps|echo_mode|frame_id|min_dist|max_dist|angle|split|dense|point_cloud|line|frame_rate|resolution|fps|width|height" "$f" 2>/dev/null \
    | sed "s|^|@@LIDARCFG $f:|" | head -25
done

# ---------------------------------------------------------------- ROS2
sec "ROS2 環境（第一趟就是死在這一段）"
if [ -f /opt/runtime/env.bash ]; then
  echo "-- /opt/runtime/env.bash --"; cat /opt/runtime/env.bash
else
  echo "(無 /opt/runtime/env.bash)"
fi
(
  set +u
  [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash
  [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash
  echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設}  RMW=${RMW_IMPLEMENTATION:-未設}"
  echo "ros2: $(command -v ros2 || echo 找不到)"
  echo "@@KV ros_domain_id=${ROS_DOMAIN_ID:-未設}"
) 2>&1

# 每個 ros2 指令都自己重新 source（subshell 的環境不會帶出來）。第一個參數是 timeout 秒數。
ros2run() {
  local t="$1"; shift
  (
    set +u
    [ -f /opt/runtime/env.bash ]      && . /opt/runtime/env.bash      >/dev/null 2>&1
    [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash >/dev/null 2>&1
    timeout "$t" "$@"
  ) 2>&1
}

# ★ 2026-09-22 的教訓：第一趟每個 topic 都「先試預設 QoS、空的再試 best_effort」，
#   每個沒資料的 topic 就吃滿兩次 14 秒的 timeout，25 個 topic 收了 12 分鐘。
#   而且實測發布者全是 **RELIABLE**（文件說 best_effort 是錯的）→ 第二次嘗試純浪費。
#   改法：先用 `topic info -v` 讀出發布者數與它的 Reliability，然後**只查一次**，
#   發布者數 0 的直接跳過。timeout 也收緊（10 Hz 的 topic 1 秒就有平均值）。
qos_flags_of() {   # 從 info 的 QoS 決定要用什麼訂閱，回傳要加的旗標
  case "$1" in
    *BEST_EFFORT*) echo "--qos-reliability best_effort --qos-durability volatile" ;;
    *)             echo "" ;;
  esac
}

hz_of() {
  local t="$1" flags="$2"
  ros2run 7 ros2 topic hz -w 10 $flags "$t" \
    | grep -m1 'average rate' | sed 's/.*average rate: *//' | xargs
}

echo_once() {
  local t="$1" flags="$2"
  ros2run 9 ros2 topic echo --once --truncate-length 120 $flags "$t"
}

if [ -x /opt/ros/humble/bin/ros2 ] || command -v ros2 >/dev/null 2>&1; then
  sec "★★★ ROS2 topic 全表（含型別，一次撈完）"
  # `topic list -t` 一次就把型別帶回來 → 不必對每個 topic 再呼叫一次 `topic type`
  TOPICS_T=$(ros2run 30 ros2 topic list -t | sort)
  TOPICS=$(echo "$TOPICS_T" | awk '{print $1}')
  if [ -z "$TOPICS" ]; then
    echo "(topic list 是空的 —— 不代表沒有 topic，先看上面的 DOMAIN_ID 與 RMW)"
  else
    echo "$TOPICS_T"
    kv topic_count "$(echo "$TOPICS" | grep -c . )"
  fi

  sec "ROS2 node 全表"
  ros2run 30 ros2 node list | sort

  # 要細量的 topic：報告 7.1／7.2／7.4 真正要的那些。白名單而非關鍵字比對 ——
  # 第一趟的關鍵字命中了 navigo 的視覺化 marker 與 0 發布者的 topic，每個都吃滿 timeout。
  CORE_RE='^/(front|rear)_lidar(/imu)?$|^/(front|rear)_camera/image_compressed$|imu_central$|^/uss_driver/[^/]+/range$|^/uwb$|^/uwb_point$|^/rtk_pvh$|^/uni_rtk_pvh$|^/gnss/data$|^/gps/rtk$|^/laser_scan$|^/battery_controller/battery_all$|^/joint_shm_controller/joint_sensor$'
  SENSOR_RE='lidar|imu|uss|ultra|range|camera|image|rtk|gps|gnss|uwb|scan|point|depth|battery'

  sec "★★★ 7.1／7.2／7.4　核心感測 topic：QoS、發布者、實測頻率、一筆內容"
  HITS=$(echo "$TOPICS" | grep -E "$CORE_RE" || true)
  if [ -z "$HITS" ]; then
    echo "(白名單沒有命中 —— 看上面的 topic 全表，可能名稱換了)"
  else
    for t in $HITS; do
      echo "---------------- $t ----------------"
      T0=$(date +%s)
      TYPE=$(echo "$TOPICS_T" | grep -m1 -E "^$t " | sed 's/.*\[\(.*\)\].*/\1/')
      INFO=$(ros2run 12 ros2 topic info -v "$t")
      PUBS=$(echo "$INFO" | grep -m1 -i "Publisher count" | grep -oE '[0-9]+')
      # 只看「發布者」那一段的 Reliability（訂閱者的 QoS 與我們無關）
      PUBQOS=$(echo "$INFO" | sed -n '/Endpoint type: PUBLISHER/,/^$/p' | grep -m1 "Reliability:")
      FLAGS=$(qos_flags_of "$PUBQOS")
      if [ "${PUBS:-0}" = 0 ]; then
        echo "@@TOPIC $t|${TYPE:-未取到}|無發布者|0"
        echo "（發布者數 0 → 不量頻率也不 echo，這種 topic 只會吃滿 timeout）"
      else
        HZ=$(hz_of "$t" "$FLAGS")
        echo "@@TOPIC $t|${TYPE:-未取到}|${HZ:-未取到}|${PUBS}"
        echo "-- 發布者 QoS: ${PUBQOS:-未取到}  訂閱旗標: ${FLAGS:-（預設 reliable）} --"
        echo "-- 一筆內容（陣列截斷 120）--"
        echo_once "$t" "$FLAGS"
      fi
      echo "-- topic info -v --"
      echo "$INFO"
      echo "（本 topic 花了 $(( $(date +%s) - T0 )) 秒）"
      echo
    done
  fi

  sec "7.1　其他名稱像感測器的 topic（只列型別，不量頻率）"
  echo "$TOPICS_T" | grep -iE "$SENSOR_RE" | grep -vE "^($(echo "$HITS" | paste -sd'|')) " \
    | sed 's/^/@@TOPIC2 /' | head -60

  sec "感測 msg 的欄位定義（7.4 要引用的規格欄位）"
  for ty in sensor_msgs/msg/PointCloud2 sensor_msgs/msg/Range sensor_msgs/msg/Imu; do
    echo "---- $ty ----"
    ros2run 10 ros2 interface show "$ty" | head -30
  done

  sec "廠商自訂 msg 介面"
  ros2run 30 ros2 interface list \
    | grep -viE "^ *(std_msgs|sensor_msgs|geometry_msgs|nav_msgs|builtin_interfaces|std_srvs|action_msgs|rcl_interfaces|lifecycle_msgs|tf2_msgs|diagnostic_msgs|shape_msgs|trajectory_msgs|visualization_msgs|unique_identifier_msgs|statistics_msgs|composition_interfaces|test_msgs|rosgraph_msgs|example_interfaces|actionlib_msgs|stereo_msgs|map_msgs|pcl_msgs|nav2_msgs|control_msgs|controller_manager_msgs)" \
    | head -80
else
  sec "ROS2"
  echo "(這塊板沒有 ros2 指令)"
fi

sec "dmesg 裡的感測器痕跡（型號常只出現在這）"
if [ "${SUDO_OK:-0}" = 1 ]; then
  sudo -n dmesg -T 2>/dev/null \
    | grep -iE "lidar|imu|icm42|lua300|camera|ov[0-9]{4}|imx[0-9]{3}|ttyTHS|usb .*video|rknpu|mali|nvgpu" \
    | tail -60
else
  echo "(未取到：無 sudo 授權，略過 dmesg；非 root 讀 dmesg 多半被 kernel.dmesg_restrict 擋)"
fi

sec "靜態盤點結束"
EOS

tag_of() { [ "$1" = rk ] && echo rk3588 || echo orinnx; }

# ★ 兩塊板**並行**跑。第一趟是依序跑，各約 6 分鐘 —— 一半的時間是白等的。
#   代價是過程中看不到即時輸出，所以跑完會把兩份 log 依序印出來。
static_probe_all() {
  hdr "靜態盤點（兩塊板並行；每塊板的輸出跑完才會印出來）"
  local pids=() b ip tag
  for b in "${BOARDS[@]}"; do
    [ "${ALIVE[$b]}" = 1 ] || { echo "（$(label_of "$b") 連不上，跳過）"; continue; }
    ip="$(ip_of "$b")"; tag="$(tag_of "$b")"
    echo "→ $(label_of "$b") 開始盤點…（log: ${tag}_static.log）"
    ( ssh "${SSH_OPTS[@]}" "robot@$ip" "SUDO_OK=${SUDO_OK[$b]} bash -s" <<< "$PROBE" \
        > "$OUT_DIR/${tag}_static.log" 2>&1 ) &
    pids+=($!)
  done
  [ ${#pids[@]} -eq 0 ] && return
  wait "${pids[@]}" 2>/dev/null
  for b in "${BOARDS[@]}"; do
    tag="$(tag_of "$b")"
    [ -s "$OUT_DIR/${tag}_static.log" ] || continue
    hdr "靜態盤點結果：$(label_of "$b")"
    cat "$OUT_DIR/${tag}_static.log"
  done
  echo
}

# ================================================================ 資源採樣
sample_phase() {
  local phase="$1" secs="$2"
  hdr "資源採樣：$phase（$secs 秒，兩塊板同時）"
  local pids=()
  for b in "${BOARDS[@]}"; do
    [ "${ALIVE[$b]}" = 1 ] || continue
    if [ -z "${PY_CMD[$b]}" ]; then
      echo "($(label_of "$b")：沒有 python3，跳過採樣)" ; continue
    fi
    local ip tag; ip="$(ip_of "$b")"; tag="$([ "$b" = rk ] && echo rk || echo nx)"
    echo "→ $(label_of "$b") 開始採樣…"
    ( ssh "${SSH_OPTS[@]}" "robot@$ip" "${PY_CMD[$b]} $secs $phase" \
        < "$SAMPLER" > "$OUT_DIR/${tag}_${phase}.json" 2> "$OUT_DIR/${tag}_${phase}.log" ) &
    pids+=($!)
  done
  if [ ${#pids[@]} -eq 0 ]; then
    echo "⚠️ 沒有任何板子可採樣"; return
  fi
  wait "${pids[@]}" 2>/dev/null
  echo
  for b in "${BOARDS[@]}"; do
    local tag; tag="$([ "$b" = rk ] && echo rk || echo nx)"
    [ -s "$OUT_DIR/${tag}_${phase}.log" ] || continue
    hr; echo "### $(label_of "$b") / $phase 摘要"; hr
    cat "$OUT_DIR/${tag}_${phase}.log"
  done
  echo
}

# ================================================================ RTSP 實測（本機）
rtsp_probe() {
  hdr "★ 7.2　RTSP 實測（從 PC 拉流，本機 ffprobe）"
  if [ "$DO_RTSP" != 1 ]; then echo "(已停用或本機沒有 ffprobe)"; return; fi
  for path in front back; do
    local url="rtsp://${RK_IP}:8554/${path}"
    echo "-- $url --"
    if timeout 30 ffprobe -v error -rtsp_transport tcp \
         -show_entries stream=index,codec_name,codec_type,width,height,avg_frame_rate,r_frame_rate,bit_rate,pix_fmt \
         -show_entries format=format_name,bit_rate \
         -of json "$url" > "$OUT_DIR/rtsp_${path}.json" 2> "$OUT_DIR/rtsp_${path}.err"; then
      cat "$OUT_DIR/rtsp_${path}.json"
    else
      echo "❌ 拉不到（錯誤留在 rtsp_${path}.err）"
      head -5 "$OUT_DIR/rtsp_${path}.err"
      echo "   常見原因：這條路徑名稱不對、要先在 APP 開影像、或 WiFi 沒連到狗的熱點"
    fi
    echo
  done
}

# ================================================================ 主流程
{
  echo "D1 Max 第三趟唯讀偵察（感測器 ＋ 資源）  $(date -Is)"
  echo "模式=$MODE  RK3588=$RK_IP  OrinNX=$NX_IP"
  echo "待機採樣=${IDLE_SECS}s  走路採樣=$([ "$DO_WALK" = 1 ] && echo "${WALK_SECS}s" || echo 停用)  sudo=$USE_SUDO"
  echo

  preflight
  warmup

  # ★ 順序：先做要人配合的 1.3 採樣，再做機器自己慢慢跑的靜態盤點。
  #   2026-09-22 第一趟是反過來的，結果人在等靜態盤點，12 分鐘後被停掉，1.3 一個字都沒拿到。
  if [ "$STATIC_ONLY" = 1 ]; then
    echo "（--static-only：跳過資源採樣）"
    static_probe_all
    rtsp_probe
    exit 0
  fi

  sample_phase idle "$IDLE_SECS"

  if [ "$DO_WALK" = 1 ]; then
    hr
    cat <<'MSG'
### 下一段要量「走路時」的資源使用（報告 1.3 的對照組）

現在請你：
  1. 確認場地淨空、急停在手（另開一個終端機備 `bash task7/realbot/estop_max.sh`）
  2. 用原廠遙控器讓狗站起來，走一段（直走或原地踏步都可以，維持動作不要停）
  3. 狗**正在走**的時候，回到這裡按 Enter 開始採樣

腳本自己不會送任何馬達指令，只在旁邊讀 /proc 與 /sys。
要跳過這段就按 Ctrl-C（前面的結果都已經存下來了）。
MSG
    hr
    read -r -p "狗已經在走了嗎？按 Enter 開始採樣…" _ || true
    sample_phase walk "$WALK_SECS"
    echo "（採樣結束，可以讓狗停下來／趴下了）"
    read -r -p "按 Enter 繼續做 RTSP 實測…" _ || true
  else
    echo "（--no-walk：跳過走路採樣，1.3 只會有待機數字）"
  fi

  if [ "$SAMPLE_ONLY" = 1 ]; then
    echo "（--sample-only：跳過靜態盤點與 RTSP）"
  else
    static_probe_all
    rtsp_probe
  fi
} 2>&1 | tee "$OUT_DIR/recon3.log"

# 關掉共用連線
for b in "${BOARDS[@]}"; do
  ssh -O exit -o ControlPath="/tmp/.recon3-%r@%h:%p" "robot@$(ip_of "$b")" 2>/dev/null || true
done

# ================================================================ 自動判讀
# 只掃遠端輸出與 JSON，不掃 recon3.log —— 否則腳本自己印的提示文字會被 grep 命中而誤報
# （第一趟就中過這個坑：標題「找 spline_shm」讓判讀誤報「路線 C 命中」）。
{
  hdr "自動判讀：四個報告項目各拿到什麼"
  STATIC=$(grep -hv "^======== " "$OUT_DIR"/*_static.log 2>/dev/null)

  echo "-- 1.3　CPU / GPU / Memory --"
  PHASES="idle"; [ "$DO_WALK" = 1 ] && PHASES="idle walk"
  for b in "${BOARDS[@]}"; do
    tag="$([ "$b" = rk ] && echo rk || echo nx)"
    for ph in $PHASES; do
      if [ -s "$OUT_DIR/${tag}_${ph}.json" ]; then
        echo "  ✅ ${tag}_${ph}.json  $(wc -c < "$OUT_DIR/${tag}_${ph}.json") bytes"
      else
        echo "  ❌ ${tag}_${ph}.json 沒有內容（$(label_of "$b")）"
      fi
    done
  done
  echo "  GPU/NPU 負載節點："
  grep -h "devfreq:" "$OUT_DIR"/*_idle.log 2>/dev/null | head -10 || echo "    (未取到)"

  echo
  echo "-- 7.1　感測器種類與數量 --"
  N=$(grep -h "^@@TOPIC " "$OUT_DIR"/*_static.log 2>/dev/null | wc -l)
  echo "  感測 topic 筆數：${N:-0}"
  grep -h "^@@TOPIC " "$OUT_DIR"/*_static.log 2>/dev/null | sed 's/^@@TOPIC /    /' | head -40

  echo
  echo "-- 7.2　Camera --"
  grep -h "^@@CAMDEV " "$OUT_DIR"/*_static.log 2>/dev/null | sed 's/^/  /' || echo "  (沒有 /dev/video*)"
  if [ "$DO_RTSP" != 1 ]; then
    echo "  （RTSP 實測已停用 / 本機沒有 ffprobe）"
  else
    for p in front back; do
      if [ -s "$OUT_DIR/rtsp_${p}.json" ]; then
        echo "  ✅ RTSP $p 有量到（rtsp_${p}.json）"
      else
        echo "  ❌ RTSP $p 未取到"
      fi
    done
  fi

  echo
  echo "-- 7.4　LiDAR 等其他感測器 --"
  L=$(grep -h "^@@LIDARCFG " "$OUT_DIR"/*_static.log 2>/dev/null | wc -l)
  echo "  光達設定檔關鍵欄位筆數：${L:-0}"
  grep -h "^@@LIDARCFG " "$OUT_DIR"/*_static.log 2>/dev/null | head -20 | sed 's/^/    /'

  echo
  echo "⚠️ ROS2 topic list 是空的不代表沒有 topic —— 先看 log 裡的 DOMAIN_ID 與 RMW 兩行。"
  echo "⚠️ 感測 topic 多半是 best_effort，用預設 QoS 訂閱會收不到；腳本已自動退一步重試。"
} 2>&1 | tee "$OUT_DIR/verdict.log"

# ================================================================ 產報告填空表
if [ -f "$REPORTER" ] && command -v python3 >/dev/null 2>&1; then
  python3 "$REPORTER" "$OUT_DIR" && echo "✅ 已產生 $OUT_DIR/報告填空表.md"
else
  echo "（跳過填空表：找不到 $REPORTER 或本機沒有 python3）"
fi

echo
echo "===================================================="
echo "完成。輸出目錄：$OUT_DIR"
echo "  recon3.log          完整過程"
echo "  rk3588_static.log   RK 靜態盤點"
echo "  orinnx_static.log   NX 靜態盤點"
echo "  {rk,nx}_{idle,walk}.json / .log   資源採樣"
echo "  rtsp_front.json rtsp_back.json    相機實測"
echo "  verdict.log         自動判讀"
echo "  報告填空表.md       ★ 1.3 / 7.1 / 7.2 / 7.4 四張表"
echo
echo "打包帶回：  tar czf ${OUT_DIR}.tar.gz $OUT_DIR"
echo "重產表：    python3 $REPORTER $OUT_DIR"
echo "===================================================="
