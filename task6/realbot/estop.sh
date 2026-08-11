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
# 注意：pattern 用 "L4_standup" 而非 "estop"，避免 pkill 匹配到本腳本自己。

echo "=== ESTOP ==="

if pkill -f "L4_standup_shm.py"; then
    echo "  [1/2] 已殺掉 L4 控制程式 → watchdog 將在 ~20ms 內讓馬達癱軟"
else
    echo "  [1/2] 沒有 L4 程式在跑（可能已自行結束）"
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
