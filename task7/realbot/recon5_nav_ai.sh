#!/usr/bin/env bash
# recon5_nav_ai.sh —— 報告 5.1–5.4（SLAM／建圖定位／導航避障／完成度）
#                      與 6.1–6.4（AI 功能／模型與推論平台／運算需求與資料來源／完成度）
#
# ★ 這趟**不測導航效果**（使用者明確說不用實際跑），只把「架構是什麼」撈回來：
#   節點接線、nav2 的 plugin、行為樹、SLAM 與定位的設定、AI 模型檔與推論平台。
#   結論由架構與設定檔推出來，而不是猜。
#
# ★ 安全設計：全程唯讀。不送馬達指令、不啟停行程、不寫狗上任何檔案、不改任何參數。
#   （`ros2 param dump` 只讀不寫；`ros2 node info` 只查圖。）
#
# 用法：
#   bash recon5_nav_ai.sh                 # WiFi：RK=192.168.234.1 NX=192.168.168.100
#   bash recon5_nav_ai.sh --wired
#   bash recon5_nav_ai.sh <RK_IP> <NX_IP>
#
# 估時 2–3 分鐘（兩塊板並行）。環境變數：RK_PW（預設 bot）、RECON_OUT。

set -uo pipefail

MODE="wifi"; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --wired) MODE="wired"; shift ;;
    --wifi)  MODE="wifi";  shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) POS+=("$1"); shift ;;
  esac
done
if [ "$MODE" = "wired" ]; then RK_IP="${POS[0]:-192.168.168.168}"
else                           RK_IP="${POS[0]:-192.168.234.1}"; fi
NX_IP="${POS[1]:-192.168.168.100}"
RK_PW="${RK_PW:-bot}"

# ⚠️ 這行漏了會在最後重建圖表那步炸掉（set -u → HERE: unbound variable）
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH_OPTS=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o ControlPersist=600
          -o ControlPath="/tmp/.recon5-%r@%h:%p")

OUT_DIR="${RECON_OUT:-./recon5_$(date +%Y%m%d_%H%M%S)}"
if [ -e "$OUT_DIR" ] && [ ! -d "$OUT_DIR" ]; then
  echo "❌ RECON_OUT=$OUT_DIR 存在但不是目錄（打包檔？）。選目錄請用 ls -dt recon5_*/" >&2
  exit 1
fi
mkdir -p "$OUT_DIR" || { echo "❌ 建不了 $OUT_DIR" >&2; exit 1; }

hr()  { printf '%s\n' "------------------------------------------------------------"; }
hdr() { hr; printf '### %s\n' "$*"; hr; }

# ================================================================ NX：導航／SLAM／AI
# ⚠️ 遠端段不用 set -u（env.bash 的 LD_LIBRARY_PATH 在非互動 SSH 下未設，會讓 shell 當場結束）
read -r -d '' PROBE_NX <<'EOS'
sec() { echo; echo "======== $* ========"; }

ros2run() {
  local t="$1"; shift
  ( set +u
    [ -f /opt/runtime/env.bash ] && . /opt/runtime/env.bash >/dev/null 2>&1
    [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash >/dev/null 2>&1
    timeout "$t" "$@" ) 2>&1
}

sec "★ 5.x　ROS2 node 全表"
ros2run 25 ros2 node list | sort

sec "★★★ 5.x　**全部**節點的接線（誰吃什麼、發什麼 —— 這段是畫 node/topic 圖的原始資料）"
# 狗上沒有 GUI，rqt_graph 跑不起來。所以把 `ros2 node info` 全撈回來，
# 在本機用 recon5_graph.py 重建同一張圖（DOT／Mermaid／表格）。
NODES=$(ros2run 25 ros2 node list | sort | grep -E '^/')
echo "$NODES" | sed 's/^/@@NODE /'
echo "節點數：$(echo "$NODES" | grep -c .)"
for n in $NODES; do
  echo "================ NODEINFO $n ================"
  ros2run 10 ros2 node info "$n"
done

sec "★★★ 5.3　nav2 的 plugin 與參數（planner／controller／costmap／BT）"
for n in /planner_server /controller_server /bt_navigator /collision_monitor \
         /global_costmap/global_costmap /local_costmap/local_costmap /map_server \
         /waypoint_follower /behavior_server; do
  echo "---------------- param dump $n ----------------"
  ros2run 15 ros2 param dump "$n"
done

sec "★ 5.3　行為樹 XML（nav2 的決策流程）"
find /opt/robot /opt/ros/humble/share -maxdepth 8 -type f -name "*.xml" 2>/dev/null \
  | grep -iE "bt|behavior|tree|navigate" | head -20
echo "-- 廠商自己的 BT（非 nav2 內建）--"
for f in $(find /opt/robot -maxdepth 8 -type f -name "*.xml" 2>/dev/null | grep -iE "bt|behavior|tree|navigate" | head -5); do
  echo "@@FILE $f"; sed -n '1,80p' "$f"
done

sec "★★★ 5.1／5.2　SLAM 與定位的設定檔"
CFG=$(find /opt/robot -maxdepth 9 -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) 2>/dev/null \
      | grep -iE "slam|lvio|mapping|localiz|loc_|odom|tf|nav|navigo|planner|controller|costmap|behavior|charg|apriltag|aritag" \
      | head -40)
for f in $CFG; do echo "@@FILE $f"; sed -n '1,400p' "$f"; echo "   …(截斷 400 行)"; done

sec "★★★ 5.3　nav2_params 的 plugin 段（上一趟被 head 截在 140 行，這次整份抓）"
for f in $(find /opt/robot -maxdepth 9 -type f -name "nav2_params.yaml" 2>/dev/null | head -4); do
  echo "@@FILE $f"; cat "$f"
done
echo "-- 所有設定檔裡的 plugin 宣告（一眼看完用什麼演算法）--"
grep -rhE "plugin|_plugin[s]?:" /opt/robot --include="*.yaml" 2>/dev/null \
  | grep -vE "^\s*#" | sed 's/^[[:space:]]*//' | sort -u | head -60

sec "★ 5.2　地圖檔（有地圖才代表建過圖）"
# ★ /ota 一定要搜：alg_manager 的 SAVE_MAP_PATH=/ota/alg_data/map、
#   CALIBRATION_FILE_PATH=/ota/calibration_results.yaml
find /ota /userdata /opt/robot /home/robot -maxdepth 7 -type f \
     \( -name "*.pgm" -o -name "*.pcd" -o -name "*.bin" -o -name "*.map" -o -name "*.posegraph" \
        -o -name "*map*.yaml" -o -name "*.osm" -o -name "*.json" \) 2>/dev/null | head -40 | while read -r f; do
  ls -lh "$f"
done
echo "-- /ota（地圖與校正檔的家）--"
ls -laR /ota 2>/dev/null | head -50
echo "-- 校正檔（外參，SLAM 要用）--"
sed -n '1,60p' /ota/calibration_results.yaml 2>/dev/null || echo "(讀不到 /ota/calibration_results.yaml)"
echo "-- 地圖／建圖相關目錄 --"
ls -la /userdata 2>/dev/null | head -20
find /userdata -maxdepth 3 -type d 2>/dev/null | grep -iE "map|slam|loc" | head -20

sec "★★★ 6.2　AI 模型檔（型別、大小＝推論平台的直接證據）"
for d in /opt/robot /opt/export /userdata /home/robot; do
  find "$d" -maxdepth 9 -type f \
    \( -iname "*.onnx" -o -iname "*.engine" -o -iname "*.plan" -o -iname "*.trt" \
       -o -iname "*.rknn" -o -iname "*.pt" -o -iname "*.pth" -o -iname "*.tflite" \
       -o -iname "*.wts" -o -iname "*.caffemodel" -o -iname "*.param" -o -iname "*.crypto" \) \
    2>/dev/null | head -40 | while read -r f; do ls -lh "$f"; done
done

sec "★★★ 5.4／6.4　廠商的演算法管理器（哪些模組 enabled＝實際上線的範圍）"
for f in $(find /opt/robot/zsibot /opt/robot -maxdepth 5 -type f -name "alg_manager*.y*ml" 2>/dev/null | head -4); do
  echo "@@FILE $f"; cat "$f"
done
echo "-- 只看模組名稱與 enabled --"
# ⚠️ 不能寫成 `grep ... $(find ...)`：find 沒找到東西時 grep 少了檔案參數，
#    會改去讀 stdin **當場卡死**（2026-09-22 本機試跑時卡住）。一定要先判斷空值。
AM=$(find /opt/robot -maxdepth 5 -type f -name "alg_manager*.y*ml" 2>/dev/null | head -4)
if [ -n "$AM" ]; then
  grep -hE "^  [a-z_]+:|enabled:|commands:" $AM 2>/dev/null | head -60
else
  echo "(找不到 alg_manager 設定檔)"
fi

sec "★★★ 6.1／6.3　感知的設定檔（模型輸入尺寸、類別、門檻、裝置）"
PCFG=$(find /opt/robot -maxdepth 9 -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.txt" -o -name "*.names" \) 2>/dev/null \
       | grep -iE "percept|detect|track|segment|fusion|model|class|label|obstacle|infer" | head -40)
for f in $PCFG; do echo "@@FILE $f"; sed -n '1,200p' "$f"; echo "   …(截斷 200 行)"; done
echo "-- 模型路徑與推論裝置的關鍵字（跨所有設定檔）--"
grep -rhiE "model_?(path|file|name)|\.onnx|\.engine|\.plan|\.trt|\.rknn|device|precision|fp16|int8|dla|gpu_id|batch|input_(w|h|size|shape)|conf_?thre|nms" \
  /opt/robot --include="*.yaml" --include="*.yml" --include="*.json" 2>/dev/null \
  | grep -vE "^\s*#" | sed 's/^[[:space:]]*//' | sort -u | head -60

sec "★★ 6.2　推論平台（TensorRT / CUDA / onnxruntime / DLA）"
echo "-- TensorRT / CUDA 函式庫 --"
ls -l /usr/lib/aarch64-linux-gnu/libnvinfer.so* /usr/lib/aarch64-linux-gnu/libcudnn.so* \
      /usr/local/cuda/lib64/libcudart.so* 2>/dev/null | head -12
dpkg -l 2>/dev/null | grep -iE "tensorrt|cudnn|cuda-|nvinfer|deepstream|vpi" | awk '{print $2, $3}' | head -20
echo "-- python 套件 --"
python3 -c "
import importlib
for m in ('torch','onnxruntime','tensorrt','cv2','numpy','ultralytics','rclpy'):
    try:
        mod = importlib.import_module(m)
        print('  %-14s %s' % (m, getattr(mod, '__version__', '?')))
    except Exception as ex:
        print('  %-14s ✗ %s' % (m, type(ex).__name__))
" 2>&1
echo "-- 感知行程連到哪些推論庫（看實際用的是什麼）--"
for p in $(pgrep -f "perception|arc_lvio|robot_slam" 2>/dev/null | head -4); do
  echo "pid $p $(cat /proc/$p/comm 2>/dev/null)"
  tr '\0' '\n' < "/proc/$p/maps" 2>/dev/null | awk '{print $6}' \
    | grep -iE "nvinfer|cudnn|cudart|cuda|onnx|opencv|tensorrt|nvdla|nvmedia|vpi" \
    | sort -u | sed 's/^/    /' | head -12
done

sec "★ 6.3／5.4　資料來源與紀錄（有沒有在錄資料、有沒有跑過）"
ls -la /home/robot 2>/dev/null | head -20
find /userdata -maxdepth 2 2>/dev/null | head -30
echo "-- 最近的 log 目錄 --"
ls -lt /userdata/log 2>/dev/null | head -10
echo "-- 導航相關的錯誤（只看最後幾行，判斷成熟度）--"
grep -rhiE "fail|error|abort|timeout" /home/robot/robot_launch_log 2>/dev/null \
  | grep -iE "nav|plan|control|slam|loc|percept|costmap" | tail -25

sec "★ 5.4　狀態機與生命週期（是不是真的上線在跑）"
ros2run 15 ros2 lifecycle nodes 2>&1 | head -20
for n in /planner_server /controller_server /bt_navigator /map_server /localization; do
  echo "$n: $(ros2run 10 ros2 lifecycle get "$n" 2>&1 | head -1)"
done
ros2run 15 ros2 service list | grep -iE "nav|plan|slam|map|charg|percept" | head -30

sec "NX 結束"
EOS

# ================================================================ RK：運控側的 AI（RKNN）
read -r -d '' PROBE_RK <<'EOS'
sec() { echo; echo "======== $* ========"; }
S() { if [ "${SUDO_OK:-0}" = 1 ]; then sudo -n "$@" 2>&1; else echo "(未取到：無 sudo)"; fi; }

sec "★★★ 6.2　運控側的 RKNN 模型（原廠的 AI 運控策略）"
ls -lhR /opt/export/rknn_model_crypto 2>/dev/null | head -40
echo "-- 其他模型檔 --"
find /opt/export /opt/robot /userdata -maxdepth 7 -type f \
  \( -iname "*.rknn" -o -iname "*.onnx" -o -iname "*.crypto" -o -iname "*.engine" \) \
  2>/dev/null | head -30 | while read -r f; do ls -lh "$f"; done
echo "-- RKNN 執行庫 --"
ls -lh /opt/export/mc/bin/librknn_model.so /usr/lib/librknnrt.so 2>/dev/null
strings /opt/export/mc/bin/librknn_model.so 2>/dev/null \
  | grep -iE "rknn_(init|run|query|inputs_set|outputs_get)|\.rknn|core_mask|npu" | sort -u | head -20

sec "★ 6.2　NPU 驅動與使用率（原廠的推論平台）"
cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo "version: $(S cat /sys/kernel/debug/rknpu/version)"
echo "load: $(S cat /sys/kernel/debug/rknpu/load)"

sec "★ 6.1　mc_ctrl 有沒有在跑模型（看它載了什麼庫）"
for p in $(pgrep -x mc_ctrl 2>/dev/null); do
  tr '\0' '\n' < "/proc/$p/maps" 2>/dev/null | awk '{print $6}' \
    | grep -iE "rknn|onnx|torch|openblas|mali|opencl" | sort -u | sed 's/^/  /'
done

sec "★★★ RK 這側的**全部**節點接線（畫圖用；預期沒有導航，導航全在 NX）"
ros2run() {
  local t="$1"; shift
  ( set +u
    [ -f /opt/runtime/env.bash ] && . /opt/runtime/env.bash >/dev/null 2>&1
    [ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash >/dev/null 2>&1
    timeout "$t" "$@" ) 2>&1
}
NODES=$(ros2run 25 ros2 node list | sort | grep -E '^/')
echo "$NODES" | sed 's/^/@@NODE /'
echo "節點數：$(echo "$NODES" | grep -c .)"
for n in $NODES; do
  echo "================ NODEINFO $n ================"
  ros2run 10 ros2 node info "$n"
done
echo "-- topic 全表（含型別）--"
ros2run 25 ros2 topic list -t | sort

sec "RK 結束"
EOS

# ================================================================ 執行
{
  echo "D1 Max 第五趟：導航／SLAM 架構（5.x）與 AI（6.x）  $(date -Is)"
  echo "模式=$MODE  RK=$RK_IP  NX=$NX_IP"
  echo "★ 唯讀：不送馬達指令、不啟停行程、不寫狗上檔案、不改參數。導航不實測效果。"
  echo

  hdr "連通性"
  RK_OK=0; NX_OK=0
  ping -c2 -W2 "$RK_IP" >/dev/null 2>&1 && echo "✅ ping $RK_IP" || echo "❌ ping $RK_IP"
  ping -c2 -W2 "$NX_IP" >/dev/null 2>&1 && echo "✅ ping $NX_IP" || echo "❌ ping $NX_IP（路由：sudo ip route add 192.168.168.0/24 via $RK_IP）"
  ssh "${SSH_OPTS[@]}" "robot@$RK_IP" true 2>/dev/null && RK_OK=1 || echo "❌ SSH RK"
  ssh "${SSH_OPTS[@]}" "robot@$NX_IP" true 2>/dev/null && NX_OK=1 || echo "❌ SSH NX"
  SUDO_RK=0
  if [ "$RK_OK" = 1 ]; then
    if ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "sudo -n true" 2>/dev/null; then SUDO_RK=1
    elif ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "printf '%s\n' '$RK_PW' | sudo -S -p '' -v" >/dev/null 2>&1 \
         && ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "sudo -n true" 2>/dev/null; then SUDO_RK=1; fi
    echo "RK sudo：$([ "$SUDO_RK" = 1 ] && echo 可用 || echo 不可用)"
  fi
  echo

  hdr "盤點中（兩塊板並行，跑完才印；估 2–3 分鐘）"
  pids=()
  if [ "$NX_OK" = 1 ]; then
    echo "→ Orin NX（導航／SLAM／AI 感知）"
    ( ssh "${SSH_OPTS[@]}" "robot@$NX_IP" "bash -s" <<< "$PROBE_NX" \
        > "$OUT_DIR/nav_ai_nx.log" 2>&1 ) &
    pids+=($!)
  fi
  if [ "$RK_OK" = 1 ]; then
    echo "→ RK3588（運控側 RKNN）"
    ( ssh "${SSH_OPTS[@]}" "robot@$RK_IP" "SUDO_OK=$SUDO_RK bash -s" <<< "$PROBE_RK" \
        > "$OUT_DIR/nav_ai_rk.log" 2>&1 ) &
    pids+=($!)
  fi
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}" 2>/dev/null

  for f in nav_ai_nx nav_ai_rk; do
    [ -s "$OUT_DIR/$f.log" ] || continue
    hdr "$f"
    cat "$OUT_DIR/$f.log"
  done
} 2>&1 | tee "$OUT_DIR/recon5.log"

ssh -O exit -o ControlPath="/tmp/.recon5-%r@%h:%p" "robot@$RK_IP" 2>/dev/null || true
ssh -O exit -o ControlPath="/tmp/.recon5-%r@%h:%p" "robot@$NX_IP" 2>/dev/null || true

# ================================================================ 判讀
{
  hdr "自動判讀：8 個報告項目各拿到什麼"
  NX="$OUT_DIR/nav_ai_nx.log"; RK="$OUT_DIR/nav_ai_rk.log"
  q() { printf '  %-58s %s\n' "$1" "$2"; }
  # 第一個參數是 regex，其餘是要掃的檔案（可以給多個）
  has() { local re="$1"; shift; grep -qiE "$re" "$@" 2>/dev/null && echo "✅" || echo "❌"; }

  q "5.1 SLAM 架構（arc_lvio／robot_slam 的接線）"   "$(has 'arc_lvio|robot_slam' "$NX")"
  q "5.2 建圖／定位（map_server／localization／地圖檔）" "$(has 'map_server|localization' "$NX")"
  q "5.3 規劃／控制 plugin（param dump）"              "$(has 'plugin|planner_plugin|controller_plugin' "$NX")"
  q "5.3 避障（collision_monitor／costmap layer）"      "$(has 'collision_monitor|inflation|obstacle_layer|voxel' "$NX")"
  q "5.4 生命週期狀態（是不是 active）"                 "$(has 'active|inactive|unconfigured' "$NX")"
  q "6.1／6.3 感知設定檔（類別／輸入尺寸／門檻）"        "$(has '@@FILE.*(percept|detect|track|segment|fusion)' "$NX")"
  q "6.2 AI 模型檔（onnx／engine／rknn）"               "$(has '\.(onnx|engine|plan|trt|rknn|crypto)' "$NX" "$RK")"
  q "6.2 推論平台（TensorRT／CUDA／RKNN）"              "$(has 'nvinfer|libcudart|tensorrt|rknn' "$NX")"
  q "6.2 運控側 RKNN 模型（RK）"                        "$(has 'rknn|crypto' "$RK")"
  echo
  echo "-- 找到的模型檔 --"
  grep -hE "\.(onnx|engine|plan|trt|rknn|crypto|pt|pth|tflite)$|\.(onnx|engine|plan|trt|rknn|crypto|pt|pth|tflite) " "$NX" "$RK" 2>/dev/null \
    | sed 's/^/  /' | head -25 || echo "  (沒有)"
  echo
  echo "-- nav2 的 plugin --"
  grep -hE "plugin|_plugin:" "$NX" 2>/dev/null | grep -vE "^\s*#" | sort -u | sed 's/^/  /' | head -25 || echo "  (沒有)"
  echo
  echo "-- 生命週期 --"
  grep -hE "^/(planner_server|controller_server|bt_navigator|map_server|localization):" "$NX" 2>/dev/null | sed 's/^/  /'
} 2>&1 | tee "$OUT_DIR/verdict.log"

# ================================================================ 重建 node／topic 圖
# 狗上沒有 GUI，rqt_graph 起不來 → 用撈回來的 `ros2 node info` 在本機重建同一張圖
GRAPH="$HERE/recon5_graph.py"
if [ -f "$GRAPH" ] && command -v python3 >/dev/null 2>&1; then
  echo
  python3 "$GRAPH" "$OUT_DIR" || echo "（圖表重建失敗，原始 log 還在）"
fi

echo
echo "===================================================="
echo "完成。輸出目錄：$OUT_DIR"
echo "  nodes_topics.md 節點與 topic 全表（報告直接用）"
echo "  graph.dot       Graphviz → dot -Tsvg graph.dot -o graph.svg"
echo "  graph_core.mmd  Mermaid 主幹圖（去掉雜訊與視覺化 topic）"
echo "  nav_ai_nx.log   導航／SLAM／AI 感知（Orin NX）"
echo "  nav_ai_rk.log   運控側 RKNN（RK3588）"
echo "  verdict.log     自動判讀"
echo "打包帶回：  tar czf ${OUT_DIR}.tar.gz $OUT_DIR"
echo "===================================================="
