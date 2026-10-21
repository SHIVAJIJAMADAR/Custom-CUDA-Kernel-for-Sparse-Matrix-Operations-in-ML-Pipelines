# SparseCUDA Environment Check — Windows PowerShell
# Run: powershell -ExecutionPolicy Bypass -File scripts\check_environment.ps1

Write-Host "SparseCUDA Environment Check" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan

Write-Host "`n--- OS ---"
Write-Host "OS: $([System.Environment]::OSVersion.VersionString)"
Write-Host "Machine: $env:COMPUTERNAME"

Write-Host "`n--- Python ---"
try { python --version 2>&1 | Write-Host } catch { Write-Host "Python: NOT FOUND" }

Write-Host "`n--- NVIDIA GPU ---"
try {
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>&1 | Write-Host
} catch { Write-Host "nvidia-smi: NOT FOUND" }

Write-Host "`n--- CUDA Toolkit ---"
Write-Host "CUDA_PATH: $env:CUDA_PATH"
Write-Host "CUDA_HOME: $env:CUDA_HOME"
try { nvcc --version 2>&1 | Write-Host } catch { Write-Host "nvcc: NOT FOUND" }

Write-Host "`n--- MSVC ---"
try {
    $cl = where.exe cl 2>$null
    if ($cl) { Write-Host "cl.exe: $cl" } else { Write-Host "cl.exe: NOT FOUND" }
} catch { Write-Host "cl.exe: NOT FOUND" }

Write-Host "`n--- PyTorch ---"
python -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA build: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"@ 2>&1 | Write-Host

Write-Host "`n--- Summary ---"
Write-Host "Check complete." -ForegroundColor Green
