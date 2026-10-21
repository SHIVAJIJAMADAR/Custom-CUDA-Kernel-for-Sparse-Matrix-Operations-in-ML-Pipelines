#!/usr/bin/env python3
"""
CUDA Development Environment Readiness and GPU Smoke-Test Sequence
Project: Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines
Repository: https://github.com/SHIVAJIJAMADAR/Custom-CUDA-Kernel-for-Sparse-Matrix-Operations-in-ML-Pipelines

Executes the official Phase 2C / 2C.1 Smoke-Test Sequence:
  - Test A: PyTorch GPU initialization (device detection, capability, allocation)
  - Test B: Dense GPU computation (FP32 & FP16 cuBLAS matmul vs CPU reference)
  - Test C: Native CUDA compilation & kernel launch (nvcc + MSVC + vector_add)
  - Test D: cuSPARSE baseline readiness (PyTorch CUDA CSR SpMM vs dense reference)
  - Test E: PyTorch C++/CUDA extension readiness & inline kernel build/execution

Reports exact status for each readiness gate: PASSED, FAILED, or BLOCKED.
Explicitly distinguishes:
  1. Prerequisite Detected
  2. Compilation Completed
  3. Code Successfully Executed on Target GPU
Never fabricates outputs; captures live empirical measurements.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def check_tool(name: str) -> str | None:
    return shutil.which(name)


def test_a_pytorch_gpu_init() -> dict:
    """Test A — PyTorch GPU initialization and device identification."""
    print_header("Test A — PyTorch GPU Initialization & Device Identification")
    result = {
        "test": "Test A (PyTorch GPU Init)",
        "prerequisite_detected": False,
        "compilation_completed": "N/A (Pre-built wheel)",
        "code_executed": False,
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

    result["prerequisite_detected"] = True

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_mb = props.total_memory / (1024 * 1024)

    result["details"]["device_count"] = device_count
    result["details"]["device_name"] = device_name
    result["details"]["compute_capability"] = f"{device_cap[0]}.{device_cap[1]}"
    result["details"]["total_memory_mb"] = f"{total_mem_mb:.1f} MB"

    print(f"[+] Detected CUDA Devices: {device_count}")
    print(f"[+] Active GPU (Device 0): {device_name}")
    print(f"[+] Compute Capability: {device_cap[0]}.{device_cap[1]} (Ampere sm_86 expected: {(device_cap == (8, 6))})")
    print(f"[+] Total Dedicated VRAM: {total_mem_mb:.1f} MB")

    try:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda:0", dtype=torch.float32)
        torch.cuda.synchronize()
        readback = t.cpu().tolist()
        assert readback == [1.0, 2.0, 3.0, 4.0], f"Readback mismatch: {readback}"
        result["details"]["tensor_allocation"] = "Success"
        result["code_executed"] = True
        result["status"] = "PASSED"
        result["notes"] = f"Initialized {device_name} (sm_{device_cap[0]}{device_cap[1]}, {total_mem_mb:.0f} MB)"
        print(f"[+] GPU Tensor Allocation: Succeeded (allocated & verified 4 float32 elements on cuda:0)")
        print(f"[+] Status: PASSED — PyTorch GPU runtime initialized.")
    except Exception as exc:
        result["status"] = "FAILED"
        result["notes"] = f"Tensor allocation on CUDA failed: {exc}"
        print(f"[-] GPU Tensor Allocation FAILED: {exc}")

    return result


def test_b_dense_gpu_computation() -> dict:
    """Test B — Dense GPU computation in FP32 and FP16."""
    print_header("Test B — Dense GPU Computation (cuBLAS GEMM / Tensor Cores)")
    result = {
        "test": "Test B (Dense GPU Computation)",
        "prerequisite_detected": False,
        "compilation_completed": "N/A (cuBLAS runtime)",
        "code_executed": False,
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

    result["prerequisite_detected"] = True

    try:
        torch.manual_seed(42)
        # 1. FP32 Dense GEMM verification
        M, K, N = 128, 128, 128
        a32_gpu = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b32_gpu = torch.randn(K, N, device="cuda", dtype=torch.float32)
        c32_gpu = torch.matmul(a32_gpu, b32_gpu)
        torch.cuda.synchronize()

        c32_cpu_ref = torch.matmul(a32_gpu.cpu(), b32_gpu.cpu())
        torch.testing.assert_close(c32_gpu.cpu(), c32_cpu_ref, rtol=1e-4, atol=1e-4)
        print(f"[+] FP32 Dense Matmul [{M}x{K}] * [{K}x{N}]: Succeeded, output verified against CPU reference.")

        # 2. FP16 Dense GEMM verification (Tensor Core path)
        a16_gpu = a32_gpu.half()
        b16_gpu = b32_gpu.half()
        c16_gpu = torch.matmul(a16_gpu, b16_gpu)
        torch.cuda.synchronize()

        c16_cpu_ref = torch.matmul(a16_gpu.cpu().float(), b16_gpu.cpu().float())
        torch.testing.assert_close(c16_gpu.cpu().float(), c16_cpu_ref, rtol=1e-2, atol=1e-2)
        print(f"[+] FP16 Dense Matmul [{M}x{K}] * [{K}x{N}]: Succeeded, output verified against reference.")

        result["details"]["fp32_shape"] = list(c32_gpu.shape)
        result["details"]["fp16_shape"] = list(c16_gpu.shape)
        result["code_executed"] = True
        result["status"] = "PASSED"
        result["notes"] = "FP32 & FP16 dense GEMM executed and numerically verified vs reference."
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
        "prerequisite_detected": False,
        "compilation_completed": False,
        "code_executed": False,
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

    if not nvcc_path or not cl_path:
        missing = []
        if not nvcc_path:
            missing.append("nvcc.exe (CUDA Toolkit 12.4)")
        if not cl_path:
            missing.append("cl.exe (MSVC v143 / Visual Studio 2022)")
        result["status"] = "BLOCKED"
        result["notes"] = f"Missing toolchain components: {', '.join(missing)}."
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    result["prerequisite_detected"] = True

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
            subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            result["compilation_completed"] = True
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
            result["code_executed"] = True
            result["status"] = "PASSED"
            result["notes"] = "Native CUDA kernel compiled with nvcc and executed on GPU."
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
        "prerequisite_detected": False,
        "compilation_completed": "N/A (cuSPARSE runtime)",
        "code_executed": False,
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

    result["prerequisite_detected"] = True

    try:
        torch.manual_seed(101)
        M, K, N = 64, 64, 32
        dense_w = torch.randn(M, K, device="cuda", dtype=torch.float32)
        # Create an 80% sparse mask
        mask = torch.rand(M, K, device="cuda") > 0.8
        sparse_dense_w = dense_w * mask.float()

        # Suppress PyTorch Sparse CSR beta warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse CSR tensor support.*")
            csr_matrix = sparse_dense_w.to_sparse_csr()

        x = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # 1. Execute via torch.matmul (dispatches to cuSPARSE SpMM)
        y_matmul = torch.matmul(csr_matrix, x)
        torch.cuda.synchronize()

        # 2. Execute via torch.sparse.mm
        y_sparse_mm = torch.sparse.mm(csr_matrix, x)
        torch.cuda.synchronize()

        # 3. Dense reference comparison
        y_ref = torch.matmul(sparse_dense_w, x)

        torch.testing.assert_close(y_matmul, y_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y_sparse_mm, y_ref, rtol=1e-4, atol=1e-4)

        result["code_executed"] = True
        result["status"] = "PASSED"
        result["details"]["nnz"] = int(csr_matrix._nnz())
        result["details"]["shape"] = list(y_matmul.shape)
        result["notes"] = f"cuSPARSE CSR SpMM verified vs dense reference (NNZ={csr_matrix._nnz()})."
        print(f"[+] cuSPARSE CSR Matrix Nonzeros (NNZ): {csr_matrix._nnz()} (Sparsity: {(1.0 - csr_matrix._nnz()/(M*K))*100:.1f}%)")
        print(f"[+] cuSPARSE SpMM Output Shape: {tuple(y_matmul.shape)}")
        print("[+] Numerical Correctness: Verified vs dense reference (assert_close rtol=1e-4, atol=1e-4)")
        print("[+] Status: PASSED — cuSPARSE CSR SpMM execution confirmed on GPU.")
    except Exception as exc:
        result["status"] = "FAILED"
        result["notes"] = f"cuSPARSE SpMM execution error: {exc}"
        print(f"[-] Status: FAILED — {exc}")

    return result


def test_e_extension_build_readiness() -> dict:
    """Test E — PyTorch C++/CUDA extension toolchain readiness & inline kernel build."""
    print_header("Test E — PyTorch C++/CUDA Extension Toolchain Readiness & Inline Kernel Build")
    result = {
        "test": "Test E (Extension Build Readiness)",
        "prerequisite_detected": False,
        "compilation_completed": False,
        "code_executed": False,
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
    print(f"[*] MSVC cl.exe present: {bool(cl_path)} ({cl_path or 'NOT IN PATH'})")
    print(f"[*] nvcc present: {bool(nvcc_path)} ({nvcc_path or 'NOT IN PATH'})")

    if not torch.cuda.is_available():
        result["status"] = "BLOCKED"
        result["notes"] = "CUDA-enabled PyTorch required for CUDAExtension."
        print(f"[-] Status: BLOCKED — {result['notes']}")
        return result

    # Check prerequisites
    if not cl_path or not nvcc_path or CUDA_HOME is None:
        result["status"] = "BLOCKED"
        reasons = []
        if not cl_path:
            reasons.append("MSVC cl.exe missing from PATH (run inside VS 2022 Native Tools Command Prompt)")
        if not nvcc_path:
            reasons.append("nvcc.exe missing from PATH (install CUDA Toolkit 12.4)")
        if CUDA_HOME is None:
            reasons.append("CUDA_HOME environment variable unset")
        result["notes"] = "; ".join(reasons)
        print(f"[-] Status: BLOCKED (Prerequisite Gate) — {result['notes']}")
        return result

    result["prerequisite_detected"] = True

    print("[*] Prerequisites satisfied. Executing minimal C++/CUDA extension compilation & import test...")
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
        result["compilation_completed"] = True
        val = module.run_dummy().item()
        if abs(val - 42.0) < 1e-4:
            result["code_executed"] = True
            result["status"] = "PASSED"
            result["notes"] = "Minimal inline CUDA extension compiled, imported, and executed on GPU."
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
    print(f"{'Test ID':<32} | {'Prereq':<8} | {'Compile':<8} | {'Execute':<8} | {'Status':<10}")
    print("-" * 78)
    for res in results:
        prereq = "YES" if res["prerequisite_detected"] else "NO"
        comp = "YES" if res["compilation_completed"] is True else ("N/A" if res["compilation_completed"] != False else "NO")
        exec_ok = "YES" if res["code_executed"] else "NO"
        print(f"{res['test']:<32} | {prereq:<8} | {comp:<8} | {exec_ok:<8} | {res['status']:<10}")
    print("=" * 78)

    print("\nActionable Diagnostics & Blocker Details:")
    for res in results:
        print(f"- {res['test']}: [{res['status']}] {res.get('notes') or 'Verified'}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    run_full_suite()
