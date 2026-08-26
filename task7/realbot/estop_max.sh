#!/usr/bin/env bash
# estop_max.sh —— 緊急停止。放在狗上 ~/，測試時**第二個終端機備著**。
#
#     sudo ~/estop_max.sh
#
# 做三件事，順序不能換：
#   1) 殺掉我們所有會寫 joint_cmd 的程式
#   2) **確認真的死了**（含 /proc/*/maps 交叉檢查）
#   3) 才解凍原廠 mc_ctrl，把控制權交回去
#
# ★ 為什麼順序不能換（task6 的教訓，直接照抄）：
#   殺不掉卻解凍 = 我們的程式與原廠運控**同時寫同一塊指令區**，
#   比什麼都不做更糟。所以第 2 步失敗時要**拒絕解凍**並叫人切電源。
#
# ★ D1 Max 與 D1 EDU 的差別（不要照搬 task6 的心智模型）：
#   task6 是「不舉旗 → spline_daemon watchdog ~20 ms 清零」。
#   這台是「心跳停 → joint_shm_controller 依 joint_cmd_timeout(500 ms) 判過期
#   → **把指令區清成 0**」（實測，見 docs/實機寫入結果_第三趟 §2）。
#   也就是殺掉程式之後，腿大約 **0.5 秒**後失力下垂，不是 20 ms，而且是失力不是阻尼。
#   吊掛狀態下可以承受，但**腿下方不要有人或東西**。
#
# ★ 為什麼是「直接殺」而不是「叫 M5 自己做阻尼停止」（想過，刻意不做）：
#   M5 的中止路徑會進入阻尼保持並**等人按 Enter**。但按 estop 的人在**另一個終端機**，
#   沒有人會去按那個 Enter。而本腳本第 2 步在 0.3 秒後就會 SIGKILL ——
#   所以「優雅停止」實際上只會拿到 0.3 秒的阻尼，之後照樣被砍。
#   複雜度加在最關鍵的安全路徑上，換來 0.3 秒。不划算。
#   急停要的是**快、簡單、一定會成功**，不是優雅。
set -uo pipefail        # ⚠️ 不用 -e：這支的每一步都要跑完並回報，不能中途靜靜退出

# ⚠️ pattern 逐一列出我們自己會寫 joint_cmd 的程式。
#    不要用 "estop" 之類會匹配到本腳本自己的字串（task6 中過兩次，一次殺掉自己的 SSH）。
#    ★ 新增任何會寫入的工具時，**必須同步加進這個清單** ——
#      task6 就是漏加 L7，按下 estop 只解凍不殺程式。
#      下面的 §maps 交叉檢查就是為了讓「漏加」會被抓到而不是靜靜失效。
WRITERS=("M5_leg_pose.py" "M1_zero_write.py" "M2_wheel_spin.py" "M3_wheel_tour.py")

# 原廠自己的行程本來就會 mmap joint_cmd，不是我們的問題，交叉檢查時要排除
FACTORY_RE='mc_ctrl|ros2|controller_manager|robot_hal|zsi_actuator|spline'

echo "════════════ ESTOP（D1 Max）════════════"

# ---------------------------------------------------------------- 1) 殺
killed=0
for pat in "${WRITERS[@]}"; do
    if pkill -f "$pat" 2>/dev/null; then
        echo "  [1/3] SIGTERM → $pat"
        killed=1
    fi
done
[ "$killed" -eq 0 ] && echo "  [1/3] 沒有我們的程式在跑（可能已自行結束）"

sleep 0.3

# ---------------------------------------------------------------- 2) 確認死透
alive=""
for pat in "${WRITERS[@]}"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
        pkill -9 -f "$pat" 2>/dev/null
        sleep 0.3
        pgrep -f "$pat" >/dev/null 2>&1 && alive="$alive $pat"
    fi
done

# ★ 交叉檢查：直接問「現在到底誰把 joint_cmd 映射進自己的位址空間」。
#   這不依賴上面那份硬編清單 —— 清單漏列的話這裡會抓到。
#   （多印一個可以互相對照的量，比多印一個結論有用。）
stray=""
for d in /proc/[0-9]*; do
    p="${d#/proc/}"
    grep -q "/dev/shm/joint_cmd" "$d/maps" 2>/dev/null || continue
    cmd="$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)"
    [ -z "$cmd" ] && continue
    echo "$cmd" | grep -Eq "$FACTORY_RE" && continue      # 原廠的，正常
    stray="$stray\n        pid=$p  $cmd"
done

if [ -n "$alive" ]; then
    echo "  [2/3] ✗✗ 連 SIGKILL 都殺不掉：$alive"
    echo "        【不要解凍 mc_ctrl】—— 會變成兩個程式同時寫指令區。"
    echo "        請立刻切斷機器人電源。"
    exit 1
fi
if [ -n "$stray" ]; then
    echo "  [2/3] ⚠️ 還有非原廠行程映射著 joint_cmd（WRITERS 清單可能漏列）："
    printf "%b\n" "$stray"
    echo "        【不要解凍 mc_ctrl】。先手動 kill 上面這些 pid，再重跑本腳本。"
    exit 1
fi
echo "  [2/3] ✅ 我們的程式都已終止，且沒有非原廠行程映射 joint_cmd"
echo "        → controller 會在 joint_cmd_timeout(500 ms) 後把指令區清零，腿失力下垂"

# ---------------------------------------------------------------- 3) 解凍
MCPID="$(pgrep -x mc_ctrl 2>/dev/null | head -1)"
if [ -n "$MCPID" ]; then
    st_before="$(ps -o stat= -p "$MCPID" 2>/dev/null | tr -d ' ')"
    kill -CONT "$MCPID" 2>/dev/null
    sleep 0.4
    st_after="$(ps -o stat= -p "$MCPID" 2>/dev/null | tr -d ' ')"
    echo "  [3/3] SIGCONT → mc_ctrl (pid $MCPID)　狀態 $st_before → $st_after"
    case "$st_after" in
        T*) echo "        ⚠️ 仍是 T（凍結）。再送一次，或直接切電源。" ;;
        "") echo "        ⚠️ 行程不見了。需要 reboot 才能恢復原廠控制。" ;;
        *)  echo "        ✅ 原廠控制已接手" ;;
    esac
else
    echo "  [3/3] ✗ 找不到 mc_ctrl！可能已被殺掉，需要 reboot 才能恢復原廠控制。"
fi

# ---------------------------------------------------------------- 現況
echo
echo "joint_cmd 現況（全部應該是 0）："
python3 - <<'PY' 2>/dev/null || echo "  （讀取失敗，請手動跑 python3 M0_probe.py）"
import sys
sys.path.insert(0, "/home/robot")
try:
    import shm_io
except Exception as e:
    raise SystemExit(f"  import shm_io 失敗：{e}")
live = [c for c in shm_io.read_joint_cmd()
        if abs(c["kp"]) + abs(c["kd"]) + abs(c["effort"]) > 1e-9]
if live:
    print(f"  ⚠️ 還有 {len(live)} 顆帶著增益：")
    for c in live:
        print(f"     {c['name']:16s} kp={c['kp']:.2f} kd={c['kd']:.2f} tau={c['effort']:.2f}")
    print("  → 若剛按下 estop，等 1 秒再看一次（controller 逾時需要 500 ms）")
else:
    print("  ✅ 16 顆全部零增益")
PY
echo "════════════════════════════════════════"
