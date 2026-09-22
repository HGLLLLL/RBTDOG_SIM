#!/usr/bin/env bash
# recon4_d1max.sh —— 補 N 報告最後兩項（2026-09-22 留下的疑問 5 與 6）
#
#   5. **走路時的 NPU 與 GPU** —— 上次的 NPU 0% 是站著量的，而且那趟 RK 的採樣沒帶 sudo，
#      讀不到 debugfs。這趟 RK 的採樣器跑在 sudo 下，直接拿 `rknpu/load` 的三核使用率。
#   6. **前後光達 IMU 頻率差 15%**（206.5 vs 175.0）—— 上次是先後各 7 秒各量一次，
#      而且 `ros2 topic hz` 只看到達時間。這趟改成**同時訂閱、量 30 秒、並看 header 時戳**，
#      分辨「感測器真的慢」與「訊息在路上掉了」。
#
# 兩段是獨立的：
#   A 段（頻率）狗趴著站著都行，不用人動
#   B 段（走路）要你用遙控器讓狗走，腳本只採樣
#
# ★ 安全設計：與第三趟相同 —— 不送馬達指令、不寫 /dev/shm、不啟停行程、
#   **不在狗上寫任何檔案**（兩支 Python 都是用 `ssh ... 'python3 -' < 檔` 從 stdin 餵進去）。
#   sudo 只用來讀 debugfs 的 NPU 使用率（RK 的 sudo 實測免密碼）。
#
# 用法：
#   bash recon4_d1max.sh                     # A 段 30 s ＋ B 段 40 s
#   bash recon4_d1max.sh --rates-only        # 只做 A 段（不必操狗）
#   bash recon4_d1max.sh --walk-only         # 只做 B 段
#   bash recon4_d1max.sh --rate-secs 60 --walk-secs 60
#   bash recon4_d1max.sh --wired             # RK 走有線 192.168.168.168
#
# 環境變數：RK_PW（預設 bot）、RECON_OUT（輸出目錄）

set -uo pipefail

MODE="wifi"; RATE_SECS=30; WALK_SECS=40; DO_RATES=1; DO_WALK=1; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --wired)      MODE="wired"; shift ;;
    --wifi)       MODE="wifi";  shift ;;
    --rate-secs)  RATE_SECS="$2"; shift 2 ;;
    --walk-secs)  WALK_SECS="$2"; shift 2 ;;
    --rates-only) DO_WALK=0; shift ;;
    --walk-only)  DO_RATES=0; shift ;;
    -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
    *)            POS+=("$1"); shift ;;
  esac
done

if [ "$MODE" = "wired" ]; then RK_IP="${POS[0]:-192.168.168.168}"
else                           RK_IP="${POS[0]:-192.168.234.1}"; fi
NX_IP="${POS[1]:-192.168.168.100}"
RK_PW="${RK_PW:-bot}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLER="$HERE/recon3_sample.py"
RATER="$HERE/recon4_rate_check.py"
for f in "$SAMPLER" "$RATER"; do
  [ -f "$f" ] || { echo "❌ 找不到 $f" >&2; exit 1; }
done

SSH_OPTS=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o ControlPersist=600
          -o ControlPath="/tmp/.recon4-%r@%h:%p")

OUT_DIR="${RECON_OUT:-./recon4_$(date +%Y%m%d_%H%M%S)}"
# ⚠️ 防呆：`RECON_OUT=$(ls -dt recon4_* | head -1)` 會抓到打包好的 .tar.gz（2026-09-22 踩過），
#    輸出目錄變成一個「檔案」，後面每個重導向都失敗 —— 而且是在狗走完之後才發現。
#    要選最近的目錄請用 `ls -dt recon4_*/`（尾巴的斜線才只列目錄）。
if [ -e "$OUT_DIR" ] && [ ! -d "$OUT_DIR" ]; then
  echo "❌ RECON_OUT=$OUT_DIR 存在但不是目錄（打包檔？）。" >&2
  echo "   選最近的目錄：RECON_OUT=\$(ls -dt recon4_*/ | head -1)" >&2
  exit 1
fi
mkdir -p "$OUT_DIR" || { echo "❌ 建不了輸出目錄 $OUT_DIR" >&2; exit 1; }
if [ ! -w "$OUT_DIR" ]; then
  echo "❌ 輸出目錄 $OUT_DIR 不可寫" >&2; exit 1
fi

hr()  { printf '%s\n' "------------------------------------------------------------"; }
hdr() { hr; printf '### %s\n' "$*"; hr; }

# A 段要量的 topic。**不放點雲**：2 MB 的訊息 Python 反序列化會變成瓶頸，
# 量到的就不是感測器的頻率而是我們自己的速度（點雲頻率用 topic hz 就夠，已量過）。
NX_TOPICS="/front_lidar/imu /rear_lidar/imu /imu_driver/imu_central \
/uss_driver/uss_left/range /uss_driver/uss_right/range /rtk_pvh /uwb"
# RK 那側發同一顆機身 IMU（走 shm）→ 兩邊一起量就知道橋接有沒有掉訊息
RK_TOPICS="/imu_shm_publisher/imu_central"

# ROS 環境一定要包在 set +u 裡：env.bash 的 LD_LIBRARY_PATH 在非互動 SSH 下未設，
# set -u 會讓 shell 當場結束，而且錯誤常被 2>/dev/null 吞掉（第一趟偵察的教訓）。
ros_py() {   # $1=秒數  $2...=topic
  printf 'bash -c %s' "'set +u
. /opt/runtime/env.bash >/dev/null 2>&1
. /opt/ros/humble/setup.bash >/dev/null 2>&1
exec python3 - $*'"
}

# ================================================================ 預檢
{
  echo "D1 Max 第四趟：補 NPU/GPU（走路）與光達 IMU 頻率  $(date -Is)"
  echo "模式=$MODE  RK=$RK_IP  NX=$NX_IP  A段=${RATE_SECS}s  B段=$([ "$DO_WALK" = 1 ] && echo "${WALK_SECS}s" || echo 停用)"
  echo

  hdr "連通性與權限"
  for ip in "$RK_IP" "$NX_IP"; do
    if ping -c2 -W2 "$ip" >/dev/null 2>&1; then echo "✅ ping $ip"
    else echo "❌ ping $ip 不通"; fi
  done
  RK_OK=0; NX_OK=0
  ssh "${SSH_OPTS[@]}" "robot@$RK_IP" true 2>/dev/null && RK_OK=1 || echo "❌ SSH RK 連不上"
  ssh "${SSH_OPTS[@]}" "robot@$NX_IP" true 2>/dev/null && NX_OK=1 || echo "❌ SSH NX 連不上（路由補了嗎？sudo ip route add 192.168.168.0/24 via $RK_IP）"

  # NPU 一定要 sudo 才讀得到 /sys/kernel/debug/rknpu/load
  RK_PY="python3 -"
  if [ "$RK_OK" = 1 ]; then
    if ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "sudo -n true" 2>/dev/null; then
      RK_PY="sudo -n python3 -"; echo "✅ RK sudo 免密碼 → NPU 三核使用率讀得到"
    elif ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "printf '%s\n' '$RK_PW' | sudo -S -p '' -v" >/dev/null 2>&1 \
         && ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "sudo -n true" 2>/dev/null; then
      RK_PY="sudo -n python3 -"; echo "✅ RK sudo 憑證已快取 → NPU 讀得到"
    else
      echo "⚠️ RK sudo 不可用 → **這趟的主要目的（走路時的 NPU）會拿不到**"
    fi
  fi
  echo

  # ============================================================ A 段：頻率
  if [ "$DO_RATES" = 1 ]; then
    hdr "A 段　光達 IMU 等頻率（同時訂閱 ${RATE_SECS} 秒；狗不用動）"
    pids=()
    if [ "$NX_OK" = 1 ]; then
      echo "→ Orin NX：$NX_TOPICS"
      ( ssh "${SSH_OPTS[@]}" "robot@$NX_IP" "$(ros_py "$RATE_SECS $NX_TOPICS")" \
          < "$RATER" > "$OUT_DIR/rates_nx.json" 2> "$OUT_DIR/rates_nx.log" ) &
      pids+=($!)
    fi
    if [ "$RK_OK" = 1 ]; then
      echo "→ RK3588：$RK_TOPICS"
      ( ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "$(ros_py "$RATE_SECS $RK_TOPICS")" \
          < "$RATER" > "$OUT_DIR/rates_rk.json" 2> "$OUT_DIR/rates_rk.log" ) &
      pids+=($!)
    fi
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}" 2>/dev/null
    for f in rates_nx rates_rk; do
      [ -s "$OUT_DIR/$f.log" ] || continue
      hr; echo "### $f"; hr; cat "$OUT_DIR/$f.log"
    done

    # 備援：狗上匯不進 rclpy 的話，A 段會整段空手而回。那就退回 `ros2 topic hz`，
    # 用**長窗口**並且**兩個 topic 同時**量 —— 雖然分不出掉包，至少能確認
    # 「175 vs 206」不是短窗口的假象。這一趟不能白跑。
    if [ "$NX_OK" = 1 ] && ! grep -q '"topics"' "$OUT_DIR/rates_nx.json" 2>/dev/null; then
      hdr "A 段備援：ros2 topic hz 長窗口（rclpy 不能用）"
      pids=()
      for t in /front_lidar/imu /rear_lidar/imu; do
        tag=$(echo "$t" | tr '/' '_')
        ( ssh "${SSH_OPTS[@]}" "robot@$NX_IP" \
            "$(printf 'bash -c %s' "'set +u
. /opt/runtime/env.bash >/dev/null 2>&1
timeout $((RATE_SECS + 5)) ros2 topic hz -w 2000 $t'")" \
            > "$OUT_DIR/hz$tag.log" 2>&1 ) &
        pids+=($!)
      done
      wait "${pids[@]}" 2>/dev/null
      for t in /front_lidar/imu /rear_lidar/imu; do
        tag=$(echo "$t" | tr '/' '_')
        echo "-- $t --"; tail -12 "$OUT_DIR/hz$tag.log"
      done
    fi
    echo
  fi

  # ============================================================ B 段：走路
  if [ "$DO_WALK" = 1 ]; then
    hr
    cat <<'MSG'
### B 段　走路時的 NPU / GPU / CPU

現在請你：
  1. 場地淨空，另開終端機備好急停：bash task7/realbot/estop_max.sh
  2. 用原廠遙控器讓狗站起來走（直走或原地踏步都行，**動作不要停**）
  3. 狗**正在走**的時候回來按 Enter

腳本自己不送任何馬達指令，只讀 /proc 與 /sys。要跳過按 Ctrl-C（A 段結果已存下）。
MSG
    hr
    read -r -p "狗已經在走了嗎？按 Enter 開始採樣 ${WALK_SECS} 秒…" _ || true
    pids=()
    if [ "$RK_OK" = 1 ]; then
      echo "→ RK3588 採樣（$RK_PY）"
      ( ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "$RK_PY $WALK_SECS walk" \
          < "$SAMPLER" > "$OUT_DIR/rk_walk.json" 2> "$OUT_DIR/rk_walk.log" ) &
      pids+=($!)
    fi
    if [ "$NX_OK" = 1 ]; then
      echo "→ Orin NX 採樣"
      ( ssh "${SSH_OPTS[@]}" "robot@$NX_IP" "python3 - $WALK_SECS walk" \
          < "$SAMPLER" > "$OUT_DIR/nx_walk.json" 2> "$OUT_DIR/nx_walk.log" ) &
      pids+=($!)
    fi
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}" 2>/dev/null
    echo "（採樣結束，可以讓狗停下來／趴下了）"
    for f in rk_walk nx_walk; do
      [ -s "$OUT_DIR/$f.log" ] || continue
      hr; echo "### $f"; hr; cat "$OUT_DIR/$f.log"
    done
  fi
} 2>&1 | tee "$OUT_DIR/recon4.log"

ssh -O exit -o ControlPath="/tmp/.recon4-%r@%h:%p" "robot@$RK_IP" 2>/dev/null || true
ssh -O exit -o ControlPath="/tmp/.recon4-%r@%h:%p" "robot@$NX_IP" 2>/dev/null || true

# ================================================================ 判讀
{
  hdr "自動判讀"
  echo "-- 疑問 5：走路時的 NPU / GPU --"
  if [ -s "$OUT_DIR/rk_walk.json" ]; then
    grep -E "rknpu:core[0-9]+:load_pct|fb000000.gpu" "$OUT_DIR/rk_walk.log" \
      | sed 's/^/  RK /' || echo "  RK：沒有 NPU 讀值（sudo 沒授權？）"
  else
    echo "  ❌ 沒有 RK 的走路採樣"
  fi
  if [ -s "$OUT_DIR/nx_walk.json" ]; then
    grep -E "gpu|GR3D" "$OUT_DIR/nx_walk.log" | sed 's/^/  NX /' | head -6
  else
    echo "  ❌ 沒有 NX 的走路採樣"
  fi
  echo
  echo "-- 疑問 6：前後光達 IMU 的頻率 --"
  if grep -q '"topics"' "$OUT_DIR/rates_nx.json" 2>/dev/null; then
    sed -n '/^----/,$p' "$OUT_DIR/rates_nx.log" | sed 's/^/  /'
  elif ls "$OUT_DIR"/hz_*.log >/dev/null 2>&1; then
    echo "  （rclpy 不能用，下面是 ros2 topic hz 長窗口的備援結果；分不出掉包）"
    for f in "$OUT_DIR"/hz_*.log; do
      echo "  -- $(basename "$f" .log | sed 's/^hz//' | tr '_' '/') --"
      grep -E "average rate|min:|max:|std_dev" "$f" | tail -4 | sed 's/^/    /'
    done
  else
    echo "  ❌ 沒有頻率量測，看 rates_nx.log"
  fi
} 2>&1 | tee "$OUT_DIR/verdict.log"

echo
echo "===================================================="
echo "完成。輸出目錄：$OUT_DIR"
echo "  rates_{nx,rk}.json/.log   A 段頻率（含 header 時戳分析）"
echo "  {rk,nx}_walk.json/.log    B 段走路採樣（RK 含 NPU 三核）"
echo "  verdict.log               自動判讀"
echo "打包帶回：  tar czf ${OUT_DIR}.tar.gz $OUT_DIR"
echo "===================================================="
