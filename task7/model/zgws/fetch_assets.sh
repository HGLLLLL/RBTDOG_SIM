#!/usr/bin/env bash
# fetch_assets.sh — 取回 zgws MJCF 需要的 STL 網格
#
# 網格共 54 MB（BASE_LINK.STL 單檔 10 MB），沒有進版控。
# 這支腳本從官方 MATRiX 發布包重新取回，放到 ./assets/。
#
# 下載約 2.1 GB，會暫存在 ~/.cache/matrix_dl/（跑第二次會重用，不重抓）。
#
# 用法：  bash task7/model/zgws/fetch_assets.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${MATRIX_CACHE:-$HOME/.cache/matrix_dl}"
TARBALL="$CACHE/base-0.1.2.tar.gz"
URL="https://github.com/zsibot/matrix/releases/download/v0.1.2/base-0.1.2.tar.gz"
SHA256="2cbb40861e89c40735cd64b24e8b64d88d012f335bdb405b5ed52db86f8b4e38"

mkdir -p "$CACHE" "$HERE/assets"

if [ -f "$TARBALL" ] && sha256sum "$TARBALL" | grep -q "^$SHA256 "; then
  echo "[skip] 已有並通過校驗：$TARBALL"
else
  echo "[get ] 下載 base-0.1.2.tar.gz（2.1 GB）…"
  curl -L --retry 3 -o "$TARBALL" "$URL"
  echo "[check] 校驗 sha256…"
  sha256sum "$TARBALL" | grep -q "^$SHA256 " \
    || { echo "[FAIL] sha256 不符，檔案可能損毀或上游改版。"; exit 1; }
fi

echo "[tar ] 解出 Content/model/zgws/assets/ …"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar xzf "$TARBALL" -C "$TMP" ./Content/model/zgws/assets/

# 只取機器人本體的 17 個 LINK 網格；同目錄的 curb*.stl 是地圖障礙物，不需要
cp "$TMP"/Content/model/zgws/assets/*LINK.STL "$HERE/assets/"

echo "[done] $(ls "$HERE"/assets/*.STL | wc -l) 個網格已放進 $HERE/assets/"
echo
echo "驗證："
echo "  conda run --no-capture-output -n rbtdog python -c \\"
echo "    \"import mujoco;m=mujoco.MjModel.from_xml_path('$HERE/scene_flat.xml');print('OK',m.nq,m.nu)\""
