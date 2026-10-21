# Custom-CUDA-Kernel-for-Sparse-Matrix-Operations-in-ML-Pipelines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![CUDA Architecture](https://img.shields.io/badge/CUDA%20Compute-8.6%20(Ampere)-green.svg)](https://developer.nvidia.com/cuda-gpus)
[![Tests Passing](https://img.shields.io/badge/tests-40%20passed%2C%201%20skipped-brightgreen.svg)]()

Custom CUDA kernels for Sparse Matrix-Matrix Multiplication (SpMM) optimized for structured and unstructured sparsity patterns arising in deep learning inference pipelines. Evaluated rigorously against industry-standard baselines (**cuBLAS dense GEMM** and **cuSPARSE CSR SpMM**) on the NVIDIA Ampere architecture.

---

## 1. Project Purpose & Official Problem Statement

In modern deep learning architectures (e.g., Transformers, Large Language Models, and Vision Transformers), weight pruning and activation sparsity create matrices with 50% to 95%+ zero values. While dense matrix multiplication (**cuBLAS GEMM**) achieves high arithmetic intensity, it performs unnecessary floating-point operations on zero elements. Conversely, off-the-shelf sparse libraries (**cuSPARSE**) are designed for general irregular sparsity, often suffering from memory divergence and suboptimal memory bandwidth utilization when applied to deep learning matrices with structured or row-clustered sparsity.

**Official Objective (Project 10, pages 21–22, `CUDA_GPU_Projects_1-10_Getting_Started.pdf`):**
1. Profile and characterize sparsity patterns found in deep learning weight matrices (density, block occupancy, and row-length skew).
2. Implement efficient sparse storage format converters (Compressed Sparse Row - CSR, and Block-Sparse).
3. Develop custom CUDA kernels:
   - **V1:** CSR Row-per-Warp SpMM with intra-warp shuffle reduction (`__shfl_down_sync`).
   - **V2:** Tile-based Block-Sparse SpMM using on-chip shared memory.
   - **V3:** Load-balanced SpMM mitigating warp divergence on skewed row distributions.
4. Establish rigorous mathematical correctness verification against a masked-dense oracle ($Y = (W \odot M) \times X$).
5. Measure execution latency using hardware CUDA events with L2 cache eviction, determining the exact **break-even sparsity threshold** where custom sparse kernels outperform dense cuBLAS.
6. Integrate the verified kernels into PyTorch via a custom C++/CUDA extension (`SparseLinear`).

---

## 2. Current Implementation Status

| Milestone | Component | Status | Description |
| :--- | :--- | :--- | :--- |
| **Foundation** | Git Repository & Layout | ✅ **Complete & Verified** | Synchronized with GitHub remote; target layout established. |
| **Formats** | CSR Converter & Block-Sparse | ✅ **Complete & Verified** | Bidirectional converters, row-length analysis, and 2-of-8 packing. |
| **Correctness** | Masked-Dense Oracle | ✅ **Complete & Verified** | Unit tests passing on CPU (40 passed, 1 skipped). |
| **Sparsity Analysis** | Workload Profiling Tool | ✅ **Complete & Verified** | Density, block occupancy, skew metrics in `patterns/analyze_sparsity.py`. |
| **Custom Kernel V0** | Preserved Block-Sparse Kernel | ⚠️ **Implemented, Unverified** | Preserved from mid-semester prototype; unverified on GPU. |
| **Custom Kernel V1** | CSR Row-per-Warp SpMM | 📋 **Planned (Phase 2)** | Core Project 10 requirement; scheduled for Phase 2. |
| **Baselines** | cuBLAS & cuSPARSE Harvesters | ⚠️ **Implemented, Unverified** | Benchmarking harnesses written; execution blocked on GPU runtime. |
| **GPU Execution** | Kernel Compilation & Timing | 🔴 **Blocked** | Blocked on CUDA Toolkit 12.x and MSVC C++ Build Tools. |

> [!IMPORTANT]
> **Performance Claim Guard:** No empirical GPU speedup claims are made at this stage. The preserved V0 CUDA kernel has **not yet been verified or timed on physical GPU hardware**. The project adheres strictly to empirical, reproducible measurements.

---

## 3. Repository Structure

```
Custom-CUDA-Kernel-for-Sparse-Matrix-Operations-in-ML-Pipelines/
├── README.md                      # Project overview, status, and reproduction guide
├── .gitignore                     # Git hygiene for Python, CUDA, VS, and build artifacts
├── requirements-dev.txt           # Python development dependencies
├── conftest.py                    # Pytest configuration and sys.path setup
├── configs/
│   └── demo_config.yaml           # Placeholder demo configuration (explicitly synthetic)
├── formats/
│   ├── __init__.py                # Format module exports
│   ├── csr.py                     # Dense <-> CSR conversions and row-length statistics
│   └── blocked.py                 # Preserved 2-of-8 block-sparse packer & unpacker
├── kernels/
│   └── sparse_linear_v0.cu        # Preserved mid-semester V0 block-sparse CUDA kernel
├── csrc/
│   ├── sparse_op.cpp              # PyTorch C++ extension binding
│   └── sparse_op.h                # Operator function definitions
├── python/
│   └── build_extension.py         # PyTorch C++/CUDA extension build script
├── patterns/
│   ├── __init__.py
│   └── analyze_sparsity.py        # Workload sparsity and row-skew characterization
├── bench/
│   ├── benchmark_dense.py         # Dense PyTorch / cuBLAS baseline harness
│   ├── benchmark_custom.py        # Custom kernel benchmark harness
│   └── cusparse_spmm_baseline.cu  # cuSPARSE SpMM baseline scaffold
├── tests/
│   ├── test_formats.py            # CSR round-trip and correctness oracle tests
│   ├── test_packer.py             # Preserved 2-of-8 packing logic tests
│   ├── test_cpu_reference.py      # CPU reference matmul tests (FP16 & FP32)
│   └── test_cuda_kernel.py        # GPU kernel test (skipped when CUDA unavailable)
├── scripts/
│   ├── check_environment.py       # Python environment inspector
│   ├── check_environment.ps1      # PowerShell environment checker
│   └── check_cuda_env.sh          # Linux/WSL environment checker
├── profiling/
│   └── README.md                  # Nsight Compute / Nsight Systems profiling guide
├── results/
│   ├── raw/                       # Structured JSON benchmark logs
│   └── plots/                     # Publication-quality benchmark figures
├── docs/
│   ├── REQUIREMENTS.md            # Traceable Project 10 requirement specifications
│   ├── ARCHITECTURE.md            # System architecture, data flow, and memory hierarchy
│   ├── DEVELOPMENT_CHECKLIST.md   # Component-by-component development status ledger
│   ├── ENVIRONMENT_SETUP.md       # Windows 11 CUDA / MSVC environment remediation plan
│   ├── BENCHMARK_PROTOCOL.md      # Rigorous benchmarking and cache management protocol
│   └── RESULTS.md                 # Current results and test execution evidence
└── sparse-cuda-midsem/            # Pristine archive of original mid-semester prototype
```

---

## 4. Verified CPU Capabilities & Reproduction Guide

All data structures, format conversions, mathematical references, and test oracles are 100% verified on the CPU.

### Prerequisites:
- Python 3.10+ (tested on Python 3.13.0 64-bit)
- `pytest`, `torch`, `numpy`, `scipy`

### Reproduction Command:
Run the complete test suite from the repository root:

```bash
python -m pytest tests/ -v
```

### Verified Output:
```
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[...] PASSED (7 tests)
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip[...] PASSED (8 tests)
tests/test_cuda_kernel.py::test_custom_cuda_matches_reference SKIPPED (CUDA unavailable)
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[...] PASSED (11 tests)
tests/test_formats.py::TestCSRProperties::test_sparsity_calculation PASSED
tests/test_formats.py::TestCSRProperties::test_row_ptr_shape PASSED
tests/test_formats.py::TestRowLengths::test_uniform_row_lengths PASSED
tests/test_formats.py::TestRowLengths::test_skewed_row_lengths PASSED
tests/test_formats.py::TestRowLengths::test_empty_matrix_stats PASSED
tests/test_formats.py::TestMaskedConversion::test_masked_csr_matches_dense_masked PASSED
tests/test_formats.py::TestMaskedConversion::test_shape_mismatch_raises PASSED
tests/test_formats.py::TestCorrectnessOracle::test_csr_matmul_matches_dense_oracle[...] PASSED (5 tests)
tests/test_packer.py::test_pack_unpack_has_expected_structure PASSED
tests/test_packer.py::test_top2_selection_is_by_block_norm PASSED

======================== 40 passed, 1 skipped in 3.66s ========================
```

---

## 5. Known Environment Blockers & Remediation

GPU compilation and profiling are currently blocked on the local Windows 11 host:

1. **CUDA Toolkit / `nvcc` Missing:** No CUDA compiler is present on the system.
2. **MSVC `cl.exe` Missing:** Visual Studio 2022 C++ Build Tools are not installed (`nvcc` requires `cl.exe`).
3. **PyTorch CPU Build:** Installed PyTorch is `2.10.0+cpu` (`torch.cuda.is_available() == False`).
4. **Environment Variables Unset:** `CUDA_HOME` and `CUDA_PATH` are not configured.

A comprehensive, step-by-step resolution plan is provided in:  
📄 **[`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md)**

---

## 6. Planned Implementation Milestones

- **Phase 1 (Current):** Repository foundation, mid-semester code preservation, requirements traceability, and environment remediation plan.
- **Phase 2:** Workload selection (real model layer profiling) and Custom Kernel V1 implementation (CSR Row-per-Warp SpMM with intra-warp shuffle reduction).
- **Phase 3:** Custom Kernel V2 implementation (Tile-based Block-Sparse SpMM) and cuSPARSE baseline benchmarking.
- **Phase 4:** Custom Kernel V3 implementation (Load-balanced CSR SpMM for skewed row distributions).
- **Phase 5:** Systematic empirical benchmark sweeps (sparsity, dimension, skewness) and break-even analysis.
- **Phase 6:** PyTorch `SparseLinear` layer integration, visual interactive demonstration, and final technical report.