#!/usr/bin/env python3
"""
CUDA Development Environment Readiness and GPU Smoke-Test Sequence
Project: Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines
Repository: https://github.com/SHIVAJIJAMADAR/Custom-CUDA-Kernel-for-Sparse-Matrix-Operations-in-ML-Pipelines

Executes the official Phase 2C Smoke-Test Sequence:
  - Test A: PyTorch GPU initialization (device detection, capability, allocation)
  - Test B: Dense GPU computation (cuBLAS matmul vs CPU reference)
  - Test C: Native CUDA compilation & kernel launch (nvcc + MSVC + vector_add)
  - Test D: cuSPARSE baseline readiness (PyTorch CUDA CSR SpMM vs dense reference)
  - Test E: PyTorch C++/CUDA extension build readiness (toolchain compiler integration)

Reports exact status for each readiness gate: PASSED, FAILED, or BLOCKED.
Never fabricates outputs; captures live empirical measurements.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def check_tool(name: str) -> str | None:
    path = shutil.which(name)
    return path


def test_a_pytorch_gpu_init() -> dict:
    """Test A — PyTorch GPU initialization."""
    print_header("Test A — PyTorch GPU Initialization")
    result = {
        "test": "Test A (PyTorch GPU Init)",
        "status": "BLOCKED",
        "details": {},
        "notes": "",
    }

    try:
        import torch
    except ImportError as exc:
        result["status"] = "BLOCKED"
        result["notes"] = f"PyTorch is not installed in the active environment: {exc}"
        print(f"[-] PyTorch: Not installed ({exc})")
        return result

    result["details"]["torch_version"] = torch.__version__
    result["details"]["cuda_build_version"] = torch.version.cuda
    result["details"]["cuda_available"] = torch.cuda.is_available()

    print(f"[*] PyTorch Version: {torch.__version__}")
    print(f"[*] PyTorch CUDA Build Version: {torch.version.cuda}")
    print(f"[*] torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        result["status"] = "BLOCKED"
        if torch.version.cuda is None:
            result["notes"] = (
                f"Installed PyTorch build is CPU-only ('{torch.__version__}'). "
                "A CUDA-enabled build (e.g. 'torch==2.6.0+cu124') must be installed."
            )
        else:
            result["notes"] = (
                "CUDA-enabled PyTorch installed, but torch.cuda.is_available() returned False. "
                "Check NVIDIA driver (nvidia-smi) or GPU accessibility."
            )
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    device_name = torch.cuda.get_device_name(0)
    device_cap = torch.cuda.get_device_capability(0)
    result["details"]["device_name"] = device_name
    result["details"]["compute_capability"] = f"{device_cap[0]}.{device_cap[1]}"

    print(f"[+] Active GPU: {device_name}")
    print(f"[+] Compute Capability: {device_cap[0]}.{device_cap[1]}")

    try:
        t = torch.zeros((16, 16), device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        result["details"]["tensor_allocation"] = "Success"
        print(f"[+] GPU Tensor Allocation: Succeeded (allocated {t.numel()} elements on cuda:0)")
        result["status"] = "PASSED"
    except Exception as exc:
        result["status"] = "FAILED"
        result["notes"] = f"Tensor allocation on CUDA failed: {exc}"
        print(f"[-] GPU Tensor Allocation FAILED: {exc}")

    return result


def test_b_dense_gpu_computation() -> dict:
    """Test B — Dense GPU computation."""
    print_header("Test B — Dense GPU Computation (cuBLAS / Tensor Core Math)")
    result = {
        "test": "Test B (Dense GPU Computation)",
        "status": "BLOCKED",
        "details": {},
        "notes": "",
    }

    try:
        import torch
    except ImportError:
        result["status"] = "BLOCKED"
        result["notes"] = "PyTorch not installed."
        print("[-] Blocked: PyTorch not installed.")
        return result

    if not torch.cuda.is_available():
        result["status"] = "BLOCKED"
        result["notes"] = "Blocked on Test A: CUDA GPU execution unavailable in current PyTorch."
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    try:
        torch.manual_seed(42)
        M, K, N = 128, 128, 128
        a_gpu = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b_gpu = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()

        c_cpu_ref = torch.matmul(a_gpu.cpu(), b_gpu.cpu())

        torch.testing.assert_close(c_gpu.cpu(), c_cpu_ref, rtol=1e-4, atol=1e-4)

        result["details"]["shape"] = list(c_gpu.shape)
        result["details"]["dtype"] = str(c_gpu.dtype)
        result["status"] = "PASSED"
        print(f"[+] Dense Matmul Output Shape: {tuple(c_gpu.shape)}")
        print("[+] Numerical Correctness: Verified against CPU reference (assert_close rtol=1e-4, atol=1e-4)")
        print("[+] Status: PASSED — Dense GPU computation backend functional.")
    except Exception as exc:
        result["status"] = "FAILED"
        result["notes"] = f"Dense computation error: {exc}"
        print(f"[-] Status: FAILED — {exc}")

    return result


def test_c_native_cuda_compilation() -> dict:
    """Test C — Native CUDA compilation & execution."""
    print_header("Test C — Native CUDA Compilation (nvcc + MSVC toolchain)")
    result = {
        "test": "Test C (Native CUDA Compilation)",
        "status": "BLOCKED",
        "details": {},
        "notes": "",
    }

    nvcc_path = check_tool("nvcc")
    cl_path = check_tool("cl")

    result["details"]["nvcc_path"] = nvcc_path or "NOT FOUND in PATH"
    result["details"]["cl_path"] = cl_path or "NOT FOUND in PATH"

    print(f"[*] nvcc compiler: {nvcc_path or 'NOT FOUND in PATH'}")
    print(f"[*] MSVC cl.exe compiler: {cl_path or 'NOT FOUND in PATH'}")

    if not nvcc_path:
        result["status"] = "BLOCKED"
        result["notes"] = "NVIDIA CUDA Toolkit (nvcc.exe) is not installed or not in PATH."
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    if not cl_path:
        result["status"] = "BLOCKED"
        result["notes"] = (
            "Microsoft Visual C++ compiler (cl.exe) is not available in the current shell environment. "
            "Must run inside the Visual Studio 2022 x64 Native Tools Command Prompt."
        )
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    source_path = ROOT_DIR / "scripts" / "smoke_test_native.cu"
    if not source_path.exists():
        result["status"] = "FAILED"
        result["notes"] = f"Test source file {source_path} not found."
        print(f"[-] Status: FAILED — {result['notes']}")
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        exe_path = Path(tmpdir) / "smoke_test_native.exe"
        compile_cmd = [
            "nvcc",
            "-O3",
            "-arch=sm_86",
            str(source_path),
            "-o",
            str(exe_path),
        ]
        print(f"[*] Executing compile command: {' '.join(compile_cmd)}")
        try:
            comp_proc = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[+] Native CUDA compilation SUCCEEDED.")
        except subprocess.CalledProcessError as comp_err:
            result["status"] = "FAILED"
            result["notes"] = f"nvcc compilation failed:\nSTDOUT: {comp_err.stdout}\nSTDERR: {comp_err.stderr}"
            print(f"[-] nvcc compilation FAILED:\n{comp_err.stderr}")
            return result

        try:
            run_proc = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"[+] Binary execution output:\n{run_proc.stdout.strip()}")
            result["status"] = "PASSED"
            result["notes"] = "Compiled binary executed and verified on GPU."
        except subprocess.CalledProcessError as run_err:
            result["status"] = "FAILED"
            result["notes"] = f"Binary execution failed:\nSTDOUT: {run_err.stdout}\nSTDERR: {run_err.stderr}"
            print(f"[-] Binary execution FAILED:\n{run_err.stderr}")

    return result


def test_d_cusparse_baseline() -> dict:
    """Test D — cuSPARSE baseline readiness."""
    print_header("Test D — cuSPARSE Baseline Readiness (CSR SpMM on GPU)")
    result = {
        "test": "Test D (cuSPARSE Baseline Readiness)",
        "status": "BLOCKED",
        "details": {},
        "notes": "",
    }

    try:
        import torch
    except ImportError:
        result["status"] = "BLOCKED"
        result["notes"] = "PyTorch not installed."
        print("[-] Blocked: PyTorch not installed.")
        return result

    if not torch.cuda.is_available():
        result["status"] = "BLOCKED"
        result["notes"] = "CUDA is not available in PyTorch. Blocked on Test A."
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    try:
        torch.manual_seed(101)
        M, K, N = 64, 64, 32
        dense_w = torch.randn(M, K, device="cuda", dtype=torch.float32)
        mask = torch.rand(M, K, device="cuda") > 0.7
        sparse_dense_w = dense_w * mask.float()

        csr_matrix = sparse_dense_w.to_sparse_csr()
        x = torch.randn(K, N, device="cuda", dtype=torch.float32)

        y_spmm = torch.matmul(csr_matrix, x)
        torch.cuda.synchronize()

        y_ref = torch.matmul(sparse_dense_w, x)

        torch.testing.assert_close(y_spmm, y_ref, rtol=1e-4, atol=1e-4)

        result["status"] = "PASSED"
        result["details"]["nnz"] = int(csr_matrix._nnz())
        result["details"]["shape"] = list(y_spmm.shape)
        print(f"[+] cuSPARSE CSR Matrix Nonzeros (NNZ): {csr_matrix._nnz()}")
        print(f"[+] cuSPARSE SpMM Output Shape: {tuple(y_spmm.shape)}")
        print("[+] Numerical Correctness: Verified vs dense reference (assert_close rtol=1e-4, atol=1e-4)")
        print("[+] Status: PASSED — cuSPARSE CSR SpMM execution confirmed on GPU.")
    except Exception as exc:
        result["status"] = "FAILED"
        result["notes"] = f"cuSPARSE SpMM execution error: {exc}"
        print(f"[-] Status: FAILED — {exc}")

    return result


def test_e_extension_build_readiness() -> dict:
    """Test E — PyTorch C++/CUDA extension build readiness."""
    print_header("Test E — PyTorch C++/CUDA Extension Build Readiness")
    result = {
        "test": "Test E (Extension Build Readiness)",
        "status": "BLOCKED",
        "details": {},
        "notes": "",
    }

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        result["status"] = "BLOCKED"
        result["notes"] = "PyTorch not installed."
        print("[-] Blocked: PyTorch not installed.")
        return result

    cl_path = check_tool("cl")
    nvcc_path = check_tool("nvcc")

    result["details"]["CUDA_HOME"] = str(CUDA_HOME)
    result["details"]["cl_found"] = bool(cl_path)
    result["details"]["nvcc_found"] = bool(nvcc_path)

    print(f"[*] torch.utils.cpp_extension.CUDA_HOME: {CUDA_HOME}")
    print(f"[*] MSVC cl.exe present: {bool(cl_path)} ({cl_path})")
    print(f"[*] nvcc present: {bool(nvcc_path)} ({nvcc_path})")

    if not torch.cuda.is_available():
        result["status"] = "BLOCKED"
        result["notes"] = "CUDA-enabled PyTorch required for CUDAExtension."
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    if not cl_path or not nvcc_path or CUDA_HOME is None:
        result["status"] = "BLOCKED"
        reasons = []
        if not cl_path:
            reasons.append("MSVC cl.exe missing from PATH (run inside VS 2022 Native Tools Command Prompt)")
        if not nvcc_path:
            reasons.append("nvcc.exe missing from PATH")
        if CUDA_HOME is None:
            reasons.append("CUDA_HOME environment variable unset")
        result["notes"] = "; ".join(reasons)
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    print("[*] Testing minimal C++/CUDA extension compilation...")
    try:
        from torch.utils.cpp_extension import load_inline
        cuda_src = r"""
        #include <torch/extension.h>
        __global__ void dummy_kernel(float* out) { out[0] = 42.0f; }
        torch::Tensor run_dummy() {
            auto out = torch::zeros({1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
            dummy_kernel<<<1, 1>>>(out.data_ptr<float>());
            return out;
        }
        """
        cpp_src = "torch::Tensor run_dummy();"
        module = load_inline(
            name="smoke_test_inline_ext",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["run_dummy"],
            extra_cuda_cflags=["-arch=sm_86"],
            verbose=False,
        )
        val = module.run_dummy().item()
        if abs(val - 42.0) < 1e-4:
            result["status"] = "PASSED"
            result["notes"] = "Minimal inline CUDA extension compiled and executed successfully."
            print(f"[+] Status: PASSED — Inline CUDA extension compiled and executed. Return: {val}")
        else:
            result["status"] = "FAILED"
            result["notes"] = f"Inline extension returned unexpected value: {val}"
            print(f"[-] Status: FAILED — {result['notes']}")
    except Exception as exc:
        result["status"] = "FAILED"
        result["notes"] = f"Extension compilation failed: {exc}"
        print(f"[-] Status: FAILED — {exc}")

    return result


def run_full_suite() -> list[dict]:
    print("=" * 78)
    print("  CUDA ENVIRONMENT READINESS & GPU SMOKE-TEST SUITE")
    print(f"  Host OS: {platform.platform()}")
    print(f"  Architecture: {platform.architecture()[0]}")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    print("=" * 78)

    results = []
    results.append(test_a_pytorch_gpu_init())
    results.append(test_b_dense_gpu_computation())
    results.append(test_c_native_cuda_compilation())
    results.append(test_d_cusparse_baseline())
    results.append(test_e_extension_build_readiness())

    print_header("Smoke-Test Sequence Summary")
    print(f"{'Test ID':<35} | {'Status':<10} | {'Diagnostic Summary'}")
    print("-" * 78)
    for res in results:
        notes = res.get("notes") or "Verified successfully"
        # Truncate notes if too long for tabular view
        short_notes = notes.split("\n")[0][:45]
        print(f"{res['test']:<35} | {res['status']:<10} | {short_notes}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    run_full_suite()
