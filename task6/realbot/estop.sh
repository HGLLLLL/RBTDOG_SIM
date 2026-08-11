#!/bin/bash
# estop.sh —— 緊急停止。放在狗上 ~/estop.sh，測試時第二個終端機備著。
#
# 做兩件事：
#   1) 殺掉我們的控制程式 → 它不再舉旗 → spline_daemon 的 watchdog 約 10 個週期
#      （~20ms）內把馬達指令清零 → 馬達癱軟
#   2) 解凍原廠 mc_ctrl → 恢復原廠控制（若有硬體故障它會自行 disable 全部馬達）
#
# 用法：  sudo ~/estop.sh
#
# 注意：pattern 逐一列出我們自己的 SHM 寫入程式，不要用 "estop" 之類的字串，
#       否則 pkill 會匹配到本腳本自己。
#
# ⚠️ 2026-08-11 修：原本只殺 L4_standup_shm.py。新增 L7_gait_shm.py 之後，
#    跑 L7 時按下 estop 只會解凍 mc_ctrl 而不殺 L7 →【原廠運控與我們的程式
#    同時寫同一塊指令區】，比什麼都不做更糟。新增工具時必須同步加進這個清單。
WRITERS=("L4_standup_shm.py" "L7_gait_shm.py")

echo "=== ESTOP ==="

killed=0
for pat in "${WRITERS[@]}"; do
    if pkill -f "$pat"; then
        echo "  [1/2] 已送出 SIGTERM → $pat"
        killed=1
    fi
done

# 確認真的死了才解凍運控。殺不掉卻解凍 = 兩個寫入者搶同一塊指令區。
sleep 0.2
alive=""
for pat in "${WRITERS[@]}"; do
    if pgrep -f "$pat" > /dev/null; then
        pkill -9 -f "$pat"
        sleep 0.2
        pgrep -f "$pat" > /dev/null && alive="$alive $pat"
    fi
done

if [ -n "$alive" ]; then
    echo "  [1/2] ✗✗ 連 SIGKILL 都殺不掉：$alive"
    echo "         【不要解凍 mc_ctrl】—— 會變成兩個程式同時寫指令區。"
    echo "         請立刻切斷機器人電源。"
    exit 1
fi

if [ "$killed" -eq 1 ]; then
    echo "  [1/2] 控制程式已終止 → watchdog 將在 ~20ms 內讓馬達癱軟"
else
    echo "  [1/2] 沒有我們的控制程式在跑（可能已自行結束）"
fi

MCPID=$(pgrep -x mc_ctrl)
if [ -n "$MCPID" ]; then
    kill -CONT "$MCPID" 2>/dev/null && echo "  [2/2] 已解凍 mc_ctrl (pid $MCPID) → 原廠控制接手"
    sleep 0.3
    echo "        mc_ctrl 狀態: $(ps -o stat= -p "$MCPID" 2>/dev/null)  （S 或 Sl = 正常執行；T = 仍凍結）"
else
    echo "  [2/2] ✗ 找不到 mc_ctrl 行程！可能已被殺掉，需要 reboot 才能恢復原廠控制。"
fi

echo
echo "16 顆馬達現況："
python3 "$HOME/L5_faultwatch.py" --once 2>/dev/null | grep -E "存活|原廠控制狀態" \
    || echo "  （L5_faultwatch.py 不在，請手動確認）"
