#!/usr/bin/env bash
# 清理狗上的 ~/m_logs/：**只刪本機 task7/logs/ 底下已經有同 sha256 備份的檔**。
#
# ★ 原則：狗上的 log 是唯一的一手資料，刪錯一個就再也拿不回來。所以
#   (1) 預設只列出，不刪；帶 --delete 才真的刪。
#   (2) 判斷「已備份」用 sha256 比對整個 task7/logs/**，不看檔名（同名不同內容是這條線踩過的坑）。
#   (3) 30 分鐘內改過的檔一律不動（faultwatch 可能還在寫；剛跑完的趟可能還沒拉回來）。
#   (4) 本機沒有的檔列出來並印出拉回的指令，**不刪**。
#
# 用法：
#   bash task7/realbot/clean_dog_logs.sh            # 只列：可刪 / 沒備份 / 太新
#   bash task7/realbot/clean_dog_logs.sh --delete   # 刪「可刪」那一群（刪前再問一次）
#   DOG=robot@1.2.3.4 bash task7/realbot/clean_dog_logs.sh
set -uo pipefail

DOG="${DOG:-robot@192.168.234.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/../logs"
KEEP_MIN=30
DO_DELETE=0
[[ "${1:-}" == "--delete" ]] && DO_DELETE=1

echo "狗：$DOG:~/m_logs/　本機備份：$LOGS"
echo

# ---- 狗上：每個檔的 sha256 與「是否 30 分鐘內改過」（一次 ssh 拿完）
remote="$(ssh "$DOG" "cd ~/m_logs 2>/dev/null || exit 3
  for f in *; do
    [[ -f \"\$f\" ]] || continue
    recent=0; [[ -n \"\$(find . -maxdepth 1 -name \"\$f\" -mmin -$KEEP_MIN)\" ]] && recent=1
    printf '%s %s %s\n' \"\$(sha256sum \"\$f\" | cut -d' ' -f1)\" \"\$recent\" \"\$f\"
  done")" || { echo "❌ 連不上狗或 ~/m_logs 不存在"; exit 1; }
[[ -z "$remote" ]] && { echo "狗上 ~/m_logs 是空的，沒事做。"; exit 0; }

# ---- 本機：task7/logs/** 的 sha256 索引（key = sha）
declare -A LOCAL
while IFS= read -r line; do
  LOCAL["${line%% *}"]=1
done < <(find "$LOGS" -type f -exec sha256sum {} + 2>/dev/null)
echo "本機索引：${#LOCAL[@]} 個不重複檔案"
echo

deletable=(); missing=(); recent=()
while read -r sha rec name; do
  [[ -z "$name" ]] && continue
  if [[ "$rec" == "1" ]]; then
    recent+=("$name")
  elif [[ -n "${LOCAL[$sha]:-}" ]]; then
    deletable+=("$name")
  else
    missing+=("$name")
  fi
done <<< "$remote"

echo "==== 可刪（本機已有同內容備份）：${#deletable[@]} 個"
printf '     %s\n' "${deletable[@]}"
echo
echo "==== 本機沒有備份（**不刪**）：${#missing[@]} 個"
printf '     %s\n' "${missing[@]}"
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "     → 先拉回來：bash task7/realbot/pull_from_dog.sh tripNN   （之後再跑本腳本）"
fi
echo
echo "==== $KEEP_MIN 分鐘內改過（**不刪**，可能還在寫或剛跑完）：${#recent[@]} 個"
printf '     %s\n' "${recent[@]}"
echo

if [[ ${#deletable[@]} -eq 0 ]]; then
  echo "沒有可刪的檔。"
  exit 0
fi
if [[ $DO_DELETE -eq 0 ]]; then
  echo "（只列出。要刪「可刪」那 ${#deletable[@]} 個：加 --delete）"
  exit 0
fi

read -r -p "真的刪掉狗上這 ${#deletable[@]} 個檔？[y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "取消。"; exit 0; }

# 檔名逐一加引號，避免空白／特殊字元；用 rm -f 對不存在的檔不報錯，root 檔會失敗並列出
q=""
for f in "${deletable[@]}"; do q+=" $(printf '%q' "$f")"; done
out="$(ssh "$DOG" "cd ~/m_logs && rm -v -f -- $q 2>&1")"
n_ok=$(grep -c "^removed" <<< "$out" || true)
echo "$out" | grep -v "^removed" || true
echo
echo "✅ 刪了 $n_ok 個；狗上剩 $(ssh "$DOG" 'ls -1 ~/m_logs | wc -l') 個"
if [[ $n_ok -lt ${#deletable[@]} ]]; then
  echo "⚠️ 有 $(( ${#deletable[@]} - n_ok )) 個沒刪掉 —— 多半是 root 建的檔："
  echo "   ssh $DOG 'sudo chown -R robot:robot ~/m_logs'  之後重跑"
fi
