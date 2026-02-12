#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Torch CPU wheels (export does not require GPU)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

python -m pip install \
  diffusers==0.35.1 \
  transformers==4.48.3 \
  accelerate==1.2.1 \
  safetensors==0.5.2 \
  onnx==1.17.0

echo "OK: venv ready at $ROOT/.venv"

