#!/usr/bin/env bash
set -euo pipefail

echo "=== NVIDIA GPU ==="
nvidia-smi

echo

echo "=== CUDA compiler ==="
nvcc --version

echo

echo "=== PyTorch CUDA ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
