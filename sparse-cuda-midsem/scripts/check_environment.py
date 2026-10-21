#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import platform
import subprocess


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable ({exc})"

print("SparseCUDA environment check")
print("============================")
print("OS:", platform.platform())
print("Python:", platform.python_version())
print("nvcc:", run(["nvcc", "--version"]))
print("nvidia-smi:", run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"]))

if importlib.util.find_spec("torch") is None:
    print("PyTorch: NOT INSTALLED")
else:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
