#!/bin/bash
# deploy_to_dog.sh —— 把 L7 吊掛空跑需要的檔案傳到狗上，並在狗上做離線自檢。
#
# 在【開發機】執行：  bash task6/realbot/deploy_to_dog.sh
#
# 傳到狗的家目錄（扁平佈局，與既有的 ~/estop.sh、~/L5_faultwatch.py 一致）。
# L7 用 sys.path.insert(檔案所在目錄)，所以 calib_map.py 與 shm_common.py
# 必須跟 L7_gait_shm.py 放在同一層——扁平佈局剛好滿足。
#
# 這支腳本【不會】驅動任何關節，也不碰 mc_ctrl。純傳檔 + 唯讀自檢。

set -euo pipefail

DOG="${DOG:-dog}"                 # ~/.ssh/config 裡的主機別名；可用 DOG=firefly@192.168.168.168 覆寫
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK6="$(dirname "$HERE")"

FILES=(
    "$HERE/shm_common.py"          # SHM 結構與安全骨架（L4/L7 共用）
    "$HERE/calib_map.py"           # MJCF ↔ 實機編碼器的映射
    "$HERE/L7_gait_shm.py"         # 步態串流本體
    "$HERE/L4_standup_shm.py"      # 上機第一步要重驗它
    "$HERE/L5_faultwatch.py"       # estop.sh 會叫它印馬達現況
    "$HERE/estop.sh"               # 緊急停止
    "$TASK6/inference/gait_export.py"   # 為了能在狗上直接 --analyze
    "$TASK6/weights/gait_walk_stable.npz"
)

echo "=== 1/3 檢查本機檔案齊全 ==="
for f in "${FILES[@]}"; do
    [ -f "$f" ] || { echo "✗ 缺檔：$f"; exit 1; }
    printf "  %-28s %8d bytes\n" "$(basename "$f")" "$(stat -c%s "$f")"
done

echo
echo "=== 2/3 傳到 $DOG:~/ ==="
scp "${FILES[@]}" "$DOG:~/"
ssh "$DOG" "chmod +x ~/estop.sh"

echo
echo "=== 3/3 狗上離線自檢（唯讀，不碰 SHM、不動馬達）==="
ssh "$DOG" 'bash -s' <<'REMOTE'
set -e
cd ~
echo "--- python 與 numpy ---"
python3 -c "import sys, numpy; print('  python', sys.version.split()[0], '/ numpy', numpy.__version__)"

echo "--- 模組載入（狗上沒有 mujoco，載得起來才代表相依乾淨）---"
python3 -c "
import sys; sys.path.insert(0, '.')
import shm_common, calib_map, L7_gait_shm, gait_export
print('  shm_common / calib_map / L7_gait_shm / gait_export 全部 OK')
"

echo "--- SHM 結構大小（必須 608）---"
python3 -c "
import ctypes, sys; sys.path.insert(0, '.')
import shm_common as SC
n = ctypes.sizeof(SC.SplineData)
print(f'  SplineData = {n} bytes', '✓' if n == SC.EXPECT_SIZE else '✗ 不符！')
raise SystemExit(0 if n == SC.EXPECT_SIZE else 1)
"

echo "--- 校正雜湊：軌跡檔 vs 狗上的 calib_map ---"
python3 -c "
import json, sys; sys.path.insert(0, '.')
import numpy as np, L7_gait_shm as L7
m = json.loads(str(np.load('gait_walk_stable.npz', allow_pickle=False)['meta_json']))
here, npz = L7.expected_calib_hash(), m['calib_hash']
print(f'  軌跡檔 {npz} / 狗上 {here}', '✓ 一致' if here == npz else '✗ 不一致，L7 會拒跑')
print(f'  步態 {m[\"gait\"]}  G_C={m[\"g_c\"]}  {len(np.load(\"gait_walk_stable.npz\")[\"q_shm\"])} 幀')
raise SystemExit(0 if here == npz else 1)
"

echo "--- dry-run（不開 SHM、不寫入）---"
python3 L7_gait_shm.py --mode gait --traj ~/gait_walk_stable.npz \
        --time-scale 0.25 --secs 5 > /tmp/l7_dry.txt 2>&1 \
    && echo "  gait dry-run OK" || { echo "  ✗ gait dry-run 失敗"; cat /tmp/l7_dry.txt; exit 1; }
grep -E "保護門檻|模擬預測|播放" /tmp/l7_dry.txt | sed 's/^/    /'
REMOTE

echo
echo "=== 完成。狗上檔案就緒，且離線自檢全過。 ==="
echo "接下來照 task6/docs/L7_吊掛空跑操作手冊.md 走，注意軌跡檔路徑是 ~/gait_walk_stable.npz"
