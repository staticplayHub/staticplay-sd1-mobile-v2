#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p _cache/sd15

# If you followed the `hfcli` setup, tokens are stored here.
if [[ -d "/home/tiny/.hf" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="/home/tiny/.hf"
fi

# SD 1.5 in diffusers format (not gated). Download once, reuse locally for export.
hf download \
  stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir ./_cache/sd15 \
  --include "model_index.json" \
  --include "scheduler/*" \
  --include "text_encoder/*" \
  --include "tokenizer/*" \
  --include "unet/*" \
  --include "vae/*" \
  --include "*.json" \
  --exclude "safety_checker/*"

if [[ ! -f "$ROOT/_cache/sd15/model_index.json" ]]; then
  echo "ERROR: model_index.json not found after download."
  echo "If this repo becomes gated or unavailable, switch to a diffusers-format SD1.5 repo"
  echo "and accept any required license on Hugging Face, then re-run this script."
  exit 1
fi

echo "OK: downloaded to $ROOT/_cache/sd15"
