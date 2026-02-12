#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TOKENIZER_DIR="$ROOT/_cache/sd15/tokenizer"
if [[ ! -d "$TOKENIZER_DIR" ]]; then
  echo "ERROR: tokenizer folder not found at: $TOKENIZER_DIR"
  echo "Run: ./download_sd15.sh"
  exit 1
fi

APP_ID="com.anonymous.staticplaysd1mobile"
DEST="/sdcard/Android/data/$APP_ID/files/tokenizer"

adb shell "mkdir -p $DEST"
adb push "$TOKENIZER_DIR/vocab.json" "$DEST/vocab.json"
adb push "$TOKENIZER_DIR/merges.txt" "$DEST/merges.txt"
adb push "$TOKENIZER_DIR/tokenizer_config.json" "$DEST/tokenizer_config.json"
adb push "$TOKENIZER_DIR/special_tokens_map.json" "$DEST/special_tokens_map.json"

echo "OK: pushed tokenizer files to $DEST"
