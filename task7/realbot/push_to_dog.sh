#!/usr/bin/env bash
# 把腿關節測試要用的檔案全部推到狗上，並**逐檔比對校驗和**。
#
# ★ 為什麼要有這支：HANDOFF 的鐵則是「每次上機都重傳全部會用到的腳本，
#   版本不一致的症狀會偽裝成硬體問題」。手動 scp 很容易漏一個檔，
#   而漏檔的表現是「行為跟預期不一樣」—— 現場會往硬體方向查，查錯方向。
#
#   光是重傳還不夠：scp 可能靜默失敗、可能傳到別的目錄、狗上可能有舊的 .pyc。
#   所以傳完一定要**比對兩邊的 sha256**，那是唯一能證明「跑的就是我改的」的量。
#
# 用法：  bash task7/realbot/push_to_dog.sh
#         DOG=robot@192.168.234.1 bash task7/realbot/push_to_dog.sh
set -euo pipefail

DOG="${DOG:-robot@192.168.234.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="$HERE/../reference/hang_torque_ref.json"

# ⚠️ 這份清單就是「這趟會用到的全部東西」。加新腳本記得加進來。
FILES=(
  "$HERE/shm_io.py"
  "$HERE/coord.py"
  "$HERE/M0_probe.py"
  "$HERE/M2_wheel_spin.py"   # 單顆輪的對照實驗（2026-08-25 已知good）
  "$HERE/M5_leg_pose.py"
  "$HERE/M6_load_probe.py"   # 承重狀態唯讀擷取（零風險）
  "$HERE/M_faultwatch.py"
  "$HERE/estop_max.sh"      # ★ 急停。第二個終端機一定要備著
)

# 力矩對照表住在 ../reference/，但狗上只有一層，要跟腳本放在一起。
if [[ -f "$REF" ]]; then
  FILES+=("$REF")
else
  echo "⚠️ 找不到 $REF"
  echo "   → M5 會少掉力矩對照，只剩追蹤誤差一個判準。"
  echo "   先跑：/home/huang/miniforge3/envs/rbtdog/bin/python task7/inference/hang_rehearsal.py"
  read -r -p "   仍要繼續傳檔嗎？[y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

echo "目標：$DOG:~/"
echo "檔案 ${#FILES[@]} 個"
echo

scp "${FILES[@]}" "$DOG:~/"

echo
echo "==== 校驗和比對（唯一能證明「狗上跑的就是我改的」的量）===="

# 本機的 sha256，key 是檔名（不含路徑）
declare -A LOCAL
for f in "${FILES[@]}"; do
  LOCAL["$(basename "$f")"]="$(sha256sum "$f" | cut -d' ' -f1)"
done

# 狗上的 sha256。⚠️ 一次 ssh 拿回全部，不要每個檔開一次連線（慢，且部分失敗時難判讀）
names=("${!LOCAL[@]}")
remote_out="$(ssh "$DOG" "cd ~ && sha256sum ${names[*]} 2>&1" || true)"

fail=0
for n in "${names[@]}"; do
  # 用 awk 精確比對第二欄的檔名，避免 grep 打到子字串（例如 shm_io.py vs shm_io.py.bak）
  got="$(awk -v n="$n" '$2 == n {print $1}' <<<"$remote_out")"
  if [[ -z "$got" ]]; then
    printf '  %-26s ❌ 狗上找不到\n' "$n"
    fail=1
  elif [[ "$got" == "${LOCAL[$n]}" ]]; then
    printf '  %-26s ✅ %s\n' "$n" "${got:0:12}"
  else
    printf '  %-26s ❌ 不一致 本機 %s / 狗上 %s\n' "$n" "${LOCAL[$n]:0:12}" "${got:0:12}"
    fail=1
  fi
done

# 舊的 .pyc 會讓 import 拿到上一版的模組 —— 這正是「版本不一致偽裝成硬體問題」
echo
# ⚠️ 這些 .pyc 多半是 root 建的（M5 要 sudo 跑），robot 帳號刪不動 → 先一般權限、再 sudo。
# ★ 失敗**不能靜靜跳過**：原本寫成 `rm -rf ... && echo 已清除`，rm 失敗時就只是
#   不印成功訊息，最後仍然印「✅ 全部一致」—— 失敗被吞掉，讀的人以為都好了。
echo "清掉狗上的 __pycache__"
if ssh "$DOG" "rm -rf ~/__pycache__ 2>/dev/null || sudo rm -rf ~/__pycache__ 2>/dev/null"; then
  echo "  ✅ 已清除"
else
  echo "  ⚠️ 清不掉（多半是 root 建的 .pyc）"
  echo "     **不影響正確性**：Python 匯入時會比對 .pyc 標頭記的原始檔 mtime 與大小，"
  echo "     對不上就重新編譯；scp 過去的 mtime 是新的 → 舊 .pyc 必被判失效。"
  echo "     真的要清：ssh $DOG 'sudo rm -rf ~/__pycache__'"
fi

echo
if [[ $fail -ne 0 ]]; then
  echo "❌ 有檔案對不上 —— **先不要上機測試**。重跑本腳本或手動確認。"
  exit 1
fi
ssh "$DOG" "chmod +x ~/estop_max.sh" 2>/dev/null

echo "✅ 全部一致。"
echo
echo "★ 開兩個終端機："
echo "     [1] ssh $DOG   → 跑測試"
echo "     [2] ssh $DOG   → 急停備著，出事就打 sudo ~/estop_max.sh"
echo
echo "  然後從 S0 乾跑開始（不需 sudo、不寫入）："
echo "     python3 M0_probe.py"
echo "     python3 M5_leg_pose.py --joints fl2_hip_pitch --delta 0.05"
