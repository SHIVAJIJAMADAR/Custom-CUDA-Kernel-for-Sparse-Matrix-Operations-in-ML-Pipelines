# Environment Remediation & Hardware Setup Plan

**Project:** Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines  
**Document Status:** Actionable Environment Remediation Plan (Phase 1)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (GA107, Ampere `sm_86`, 4 GB GDDR6)  
**Host Platform:** Windows 11 Home 64-bit | Python 3.13.0 (64-bit)  

---

## 1. Discovered Environment Baseline & Diagnostic Audit

An audit conducted on October 21, 2026, confirmed the following local hardware and software state:

| Component | Current State | Requirement for CUDA Development | Status / Gap |
| :--- | :--- | :--- | :--- |
| **GPU Hardware** | NVIDIA GeForce RTX 3050 Laptop GPU | Compute Capability 8.6, 4 GB VRAM | ✅ Detected & Operational |
| **NVIDIA Driver** | `566.07` | Driver $\ge 525$ (supports CUDA 12.0–12.7) | ✅ Compatible |
| **Host Python** | `Python 3.13.0` (64-bit) | Python $\ge 3.10$ | ✅ Compatible |
| **CUDA Toolkit / `nvcc`** | Not found in PATH or standard paths | CUDA Toolkit 12.4 (or 12.x) with `nvcc.exe` | 🔴 **CRITICAL BLOCKER 1** |
| **Host C++ Compiler** | `cl.exe` missing | MSVC C++ Build Tools (VS 2022 v143 toolset) | 🔴 **CRITICAL BLOCKER 2** |
| **PyTorch Distribution** | `2.10.0+cpu` (`torch.cuda.is_available() == False`) | PyTorch with CUDA 12.4 support (`torch 2.6.0+cu124`) | 🔴 **CRITICAL BLOCKER 3** |
| **Environment Variables** | `CUDA_HOME`, `CUDA_PATH` unset | System environment variables pointing to CUDA directory | 🔴 **BLOCKER 4** |
| **Profiling Tools** | `ncu` / `nsys` missing | NVIDIA Nsight Compute & Nsight Systems | 🟡 **BLOCKER 5 (Required for R9/R11)** |
| **Target ML Workload** | Demo config (uncalibrated placeholder) | Real model weight matrix or verified benchmark layer | 🟡 **BLOCKER 6 (Required for R1)** |

---

## 2. Step-by-Step Remediation Plan

> [!CAUTION]
> In accordance with project integrity standards, **system-wide software installations and global Python environment modifications are not performed automatically**. Follow the precise manual instructions below to unblock physical GPU development.

### Step 1: Install Visual Studio 2022 C++ Build Tools
The CUDA compiler `nvcc` requires Microsoft Visual C++ (`cl.exe`) to compile host code and link extensions.

1. Download the **Visual Studio 2022 Build Tools** installer from Microsoft:  
   🔗 [https://aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)
2. Run the installer and select the workload:
   - **Desktop development with C++** (`Microsoft.VisualStudio.Workload.VCTools`)
   - Ensure the following sub-components are checked in the installation details:
     - *MSVC v143 - VS 2022 C++ x64/x86 build tools*
     - *Windows 11 SDK (or Windows 10 SDK 10.0.22621.0+)*
     - *C++ CMake tools for Windows*
3. Complete installation (requires $\sim 7\text{ GB}$ disk space).

**Verification Command:**
Open PowerShell and verify that `cl.exe` can be located:
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl.exe
```
*Expected Output:* `Microsoft (R) C/C++ Optimizing Compiler Version 19.4x... for x64`.

---

### Step 2: Install NVIDIA CUDA Toolkit 12.4 Update 1
NVIDIA Driver `566.07` supports CUDA up to version 12.7. We recommend **CUDA 12.4 Update 1** because it aligns with verified PyTorch binary wheels available for Python 3.13 (`torch==2.6.0+cu124`). *(Alternatively, CUDA Toolkit 12.6 with `torch==2.6.0+cu126` is also verified compatible).*

1. Download the CUDA 12.4 Update 1 network or local installer for Windows 11:  
   🔗 [https://developer.nvidia.com/cuda-12-4-1-download-archive](https://developer.nvidia.com/cuda-12-4-1-download-archive)
2. Run the installer. Choose **Custom (Advanced)** installation:
   - Check: **CUDA** (Development, Runtime, Documentation, Samples)
   - Uncheck: Display Driver (keep existing driver `566.07`, which is newer)
   - Check: **Nsight Compute** and **Nsight Systems** (if available in installer)
3. Default installation target path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`

**Verification Command:**
```powershell
nvcc --version
```
*Expected Output:* `Cuda compilation tools, release 12.4, V12.4.xxx`.

---

### Step 3: Configure System Environment Variables
Ensure system-wide environment variables are populated:

1. **`CUDA_PATH`**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
2. **`CUDA_HOME`**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
3. Add to system **`PATH`**:
   - `%CUDA_PATH%\bin`
   - `%CUDA_PATH%\libnvvp`
   - `C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.x\` (if profiling tools installed)

**PowerShell Configuration Command (Run as Administrator):**
```powershell
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", "Machine")
[Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", "Machine")
$oldPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$newEntries = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;"
if ($oldPath -notlike "*$newEntries*") {
    [Environment]::SetEnvironmentVariable("Path", $newEntries + $oldPath, "Machine")
}
```

---

### Step 4: Create Isolated Virtual Environment & Install CUDA-Enabled PyTorch
To preserve the global Python environment without modifying machine-wide packages:

1. **Create and Activate a Dedicated Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   ```

2. **Install Pre-built CUDA 12.4 PyTorch Wheel (`cp313-win_amd64`):**
   ```powershell
   pip install torch==2.6.0+cu124 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements-dev.txt
   ```
   *(Note: If CUDA 12.6 is selected in Step 2, run `pip install torch==2.6.0+cu126 torchvision --extra-index-url https://download.pytorch.org/whl/cu126` instead).*

3. **Verify PyTorch GPU Initialization:**
   ```powershell
   python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Compute Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
   ```
   *Expected Output:*
   ```
   PyTorch Version: 2.6.0+cu124
   CUDA Available: True
   Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
   Compute Capability: (8, 6)
   ```

---

### Step 5: Verify cuBLAS & GPU Dense GEMM Acceleration
Verify that standard PyTorch GPU tensor math and dense operations execute on the RTX 3050 without error:

```powershell
python -c "import torch; a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16); b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16); c = torch.matmul(a, b); print('Dense GEMM Succeeded! Result shape:', c.shape)"
```

---

### Step 6: Build the PyTorch C++/CUDA Extension in Developer Shell
When compiling custom CUDA extensions via PyTorch's `CUDAExtension`, execute within the MSVC x64 developer environment so `cl.exe` is in PATH:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
python python/build_extension.py
```
*Verification:* Inspect generated `.pyd` module in `csrc/` or site-packages.

---

### Step 7: Execute the Complete Test Suite
Run the test suite with CUDA tests unblocked:
```powershell
python -m pytest tests/ -v
```
*Expected Output:* All 72 tests passing (including `test_cuda_kernel.py`).

---

## 3. Potential Hazards & Mitigations

1. **Long PATH Truncation:** On Windows, modifying system `PATH` via GUI can truncate strings exceeding 2048 characters. Always check length or use PowerShell `Environment::SetEnvironmentVariable`.
2. **Python 3.13 vs Older Wheels:** Older PyTorch releases (e.g., PyTorch 2.1–2.4) do **not** support Python 3.13. Do not attempt to install `torch<2.6.0` on Python 3.13.
3. **Laptop Power & Thermal Throttling:** The RTX 3050 Laptop GPU operates under variable dynamic boost TGP (35W–80W). Benchmark scripts must use consistent warmup loops and query GPU clock stability where feasible.
4. **VRAM Limitation (4 GB):** Avoid tests allocating single matrices exceeding $1\text{ GB}$ (e.g., $16384 \times 16384$ in FP32 requires $1\text{ GB}$). Keep batch dimension $N \le 512$ during parameter sweeps.
