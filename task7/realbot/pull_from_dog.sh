#!/usr/bin/env bash
# 把狗上的 ~/m_logs/ 收回本機，**攤平、不套巢狀目錄**。
#
# ★ 為什麼要有這支：`scp -r host:~/m_logs ./m_logs_tripN` 在目標目錄
#   **已經存在**時，會把來源整個放進去變成 `m_logs_tripN/m_logs/` ——
#   2026-08-26 這個坑踩了兩次，兩次都要事後手動攤平。
#   這裡用 remote glob（`~/m_logs/*`）只複製檔案本身，不會有那層目錄。
#
# 用法：
#   bash task7/realbot/pull_from_dog.sh              # 收全部到 m_logs_trip7
#   bash task7/realbot/pull_from_dog.sh trip8        # 收到 m_logs_trip8
#   bash task7/realbot/pull_from_dog.sh trip7 'M6_*' # 只收 M6 的
set -uo pipefail

DOG="${DOG:-robot@192.168.234.1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIP="${1:-trip7}"
PAT="${2:-*}"
DST="$HERE/../logs/m_logs_${TRIP#m_logs_}"

mkdir -p "$DST"
before=$(ls -1 "$DST" 2>/dev/null | wc -l)

echo "來源：$DOG:~/m_logs/$PAT"
echo "目標：$DST"
echo "（收檔前已有 $before 個檔案）"
echo

# ★ 關鍵：remote glob 由狗上的 shell 展開 → 複製的是**檔案**，不是目錄，
#   所以不會在目標下多生一層。引號不能少，否則 glob 會在本機展開（本機沒有那些檔）。
if ! scp "$DOG:~/m_logs/$PAT" "$DST/"; then
  echo
  echo "❌ scp 失敗。常見原因："
  echo "   - 狗上 ~/m_logs 還沒有符合 '$PAT' 的檔案"
  echo "   - 檔案是 root 建的且權限不足（M5 用 sudo 跑時會這樣；"
  echo "     shm_io.start_log 會 chown 回 robot，但若那段失敗就會卡在這）"
  echo "     → ssh $DOG 'sudo chown -R robot:robot ~/m_logs' 之後重試"
  exit 1
fi

after=$(ls -1 "$DST" 2>/dev/null | wc -l)
echo
echo "==== 收回結果 ===="
echo "  檔案數 $before → $after（新增 $((after - before))）"
echo "  目錄大小 $(du -sh "$DST" | cut -f1)"

# 巢狀目錄檢查 —— 若還是不小心生出來了，當場講，不要等到分析時才發現
if [[ -d "$DST/m_logs" ]]; then
  echo
  # ⚠️ 反引號不能出現在雙引號字串裡 —— 那是命令替換，會真的去跑 scp -r。
  #    `bash -n` 抓不到（語法是合法的），只有實際觸發那一行才會炸。
  echo '⚠️ 目標下出現巢狀的 m_logs/ —— 有人用了 scp -r。攤平它：'
  echo "     mv -f '$DST/m_logs/'* '$DST/' && rmdir '$DST/m_logs'"
fi

echo
echo "最新的 5 個："
ls -t "$DST" | head -5 | sed 's/^/  /'
