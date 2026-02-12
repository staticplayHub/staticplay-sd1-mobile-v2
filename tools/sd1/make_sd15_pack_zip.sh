#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

IN_DIR="_out/sd15-onnx"
OUT_ZIP="_out/sd15-onnx-pack.zip"

if [[ ! -d "$IN_DIR" ]]; then
  echo "Missing $IN_DIR. Run export first:"
  echo "  ./.venv/bin/python export_sd15_to_onnx.py --in ./_cache/sd15 --out ./_out/sd15-onnx"
  exit 1
fi

mkdir -p _out
rm -f "$OUT_ZIP"

cd _out
zip -r -q "$(basename "$OUT_ZIP")" "sd15-onnx"

echo "OK: $ROOT/$OUT_ZIP"

