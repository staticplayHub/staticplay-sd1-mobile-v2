#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ZIP_PATH="$ROOT/_out/sd15-onnx-pack.zip"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Missing $ZIP_PATH. Create it first:"
  echo "  ./make_sd15_pack_zip.sh"
  exit 1
fi

adb devices 1>/dev/null
adb push "$ZIP_PATH" "/sdcard/Download/sd15-onnx-pack.zip"
echo "OK: pushed to phone Downloads as sd15-onnx-pack.zip"

