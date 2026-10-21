# Custom-CUDA-Kernel-for-Sparse-Matrix-Operations-in-ML-Pipelines

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![CUDA Architecture](https://img.shields.io/badge/CUDA%20Compute-8.6%20(Ampere)-green.svg)](https://developer.nvidia.com/cuda-gpus)
[![Tests Passing](https://img.shields.io/badge/tests-71%20passed%2C%201%20skipped-brightgreen.svg)]()

Custom CUDA kernels for Sparse Matrix-Matrix Multiplication (SpMM) optimized for structured and unstructured sparsity patterns arising in deep learning inference pipelines. Designed for rigorous evaluation against industry-standard baselines (**cuBLAS dense GEMM** and **cuSPARSE CSR SpMM**) on the NVIDIA Ampere architecture (GPU execution and empirical benchmarking pending toolchain readiness).

---

## 1. Project Purpose & Official Problem Statement

In modern deep learning architectures (e.g., Transformers, Large Language Models, and Vision Transformers), weight pruning and activation sparsity create matrices with 50% to 99% zero values. While dense matrix multiplication (**cuBLAS GEMM**) achieves high arithmetic intensity, it performs unnecessary floating-point operations on zero elements. Conversely, off-the-shelf sparse libraries (**cuSPARSE**) are designed for general irregular sparsity, often suffering from memory divergence and suboptimal memory bandwidth utilization when applied to deep learning matrices with structured or row-clustered sparsity.

**Official Objective (Project 10, pages 21–22, `CUDA_GPU_Projects_1-10_Getting_Started.pdf`):**
1. **Workload & Sparsity Characterization:** Profile real target weight matrices/masks (density, block occupancy, and row-length skew). Synthetic matrices support controlled unit testing and parameter sweeps, but do not replace characterization of a real target workload.
2. **Format Conversions:** Implement sparse storage format converters (Compressed Sparse Row - CSR, and applicable Block-Sparse/BSR formats).
3. **Official Baseline Ladder:** Measure performance against all four required comparisons under identical inputs, dtypes, and dimensions:
   - Baseline 1: Dense cuBLAS GEMM
   - Baseline 2: cuSPARSE CSR SpMM (cuSPARSE BSR is treated as applicable additional work for block-structured workloads, not a replacement for CSR)
   - Baseline 3: Naive custom sparse kernel
   - Baseline 4: Optimized custom sparse kernel
4. **Custom Kernel Ladder:**
   - **V1 (Core Required):** CSR Row-per-Warp SpMM with reconciled memory coalescing (Column-Coalesced Mapping B).
   - **V2 (Workload-Dependent):** Tile-based Block-Sparse SpMM using on-chip shared memory.
   - **V3 (Workload-Dependent):** Load-balanced SpMM mitigating warp divergence on skewed row distributions.
   - *Preserved Mid-Sem:* V0 block-sparse prototype remains distinctly labeled and archived.
5. **Correctness Verification:** Verify all custom outputs against a mathematical masked-dense reference ($Y = (W \odot M) \times X$) across rigorous numerical tolerances.
6. **Empirical Benchmarks & Sweeps:** Measure execution latency using hardware CUDA events across the official **50%–99% sparsity sweep range**, determining the exact **break-even sparsity threshold** where custom kernels outperform dense cuBLAS. (L2 cache eviction is an explicit benchmark methodology choice to evaluate cold-cache vs warm-cache behavior, not an unconditional separate deliverable).
7. **PyTorch Integration:** Wrap verified kernels in a PyTorch C++/CUDA extension (`SparseLinear`).

---

## 2. Current Implementation Status

| Milestone | Component | Status | Description |
| :--- | :--- | :--- | :--- |
| **Foundation** | Git Repository & Layout | ✅ **Complete & Verified** | Synchronized with GitHub remote; target layout established. |
| **Target Workload** | Real Model Weight Matrix | ✅ **Acquired & Profiled (Phase 2A & 2B)** | BERT-Base [3072, 768] extracted, 0% native sparsity; 50%–99% sweep in `docs/WORKLOAD_CHARACTERIZATION.md`. |
| **Formats** | CSR Converter & Block-Sparse | ✅ **Complete & Verified** | Bidirectional converters, row-length analysis, and 2-of-8 packing. |
| **Correctness** | Masked-Dense Oracle & Unit Tests | ✅ **Complete & Verified** | 71 tests passing on CPU (71 passed, 1 skipped). |
| **Sparsity Analysis** | Workload Profiling Tool | ✅ **Complete & Verified** | Density, block occupancy, skew metrics in `patterns/analyze_sparsity.py`. |
| **Custom Kernel V0** | Preserved Block-Sparse Kernel | ⚠️ **Implemented, Unverified** | Preserved from mid-semester prototype; unverified on GPU. |
| **Custom Kernel V1** | CSR Row-per-Warp SpMM | 📐 **Reconciled Architecture (Phase 2B)** | Column-Coalesced Mapping B specified; execution gated on environment. |
| **Baselines** | cuBLAS & cuSPARSE Harvesters | ✅ **Runtime Verified (Phase 2C.1)** | cuBLAS dense GEMM and cuSPARSE CSR SpMM verified on RTX 3050 in smoke tests. |
| **GPU Execution** | Kernel Compilation & Timing | 🔴 **Readiness Gate Blocked** | PyTorch GPU runtime operational (Tests A, B, D pass); native custom kernel compilation (Tests C, E) blocked on CUDA Toolkit & MSVC. |

> [!IMPORTANT]
> **Performance Claim Guard:** No empirical GPU speedup claims are made at this stage. The preserved V0 CUDA kernel and V1 design have **not yet been executed on physical GPU hardware** because the local build toolchain is pending. The project strictly guards against fabricated performance numbers.

---

## 3. Repository Structure

```
Custom-CUDA-Kernel-for-Sparse-Matrix-Operations-in-ML-Pipelines/
├── README.md                      # Project overview, status, and reproduction guide
├── .gitignore                     # Git hygiene for Python, CUDA, VS, and build artifacts
├── requirements-dev.txt           # Core Python development & test dependencies
├── requirements-workload.txt      # Optional workload acquisition dependencies (HF Hub, safetensors)
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
│   ├── check_cuda_env.sh          # Linux/WSL environment checker
│   ├── smoke_test_cuda.py         # Automated Phase 2C CUDA readiness & smoke tests (Tests A-E)
│   └── smoke_test_native.cu       # Standalone native CUDA C++ kernel test (Test C)
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
- Dependencies: `pytest`, `torch`, `pyyaml` (install via `pip install -r requirements-dev.txt`)

### Fast Unit Tests (No Model Checkpoint Required):
Runs all unit tests (format converters, CPU reference matmul, 2-of-8 block packing, and synthetic magnitude pruning tie-breaking) in $< 3$ seconds:

```bash
python -m pytest tests/ -m "not workload" -v
```

### Complete Test Suite (Including BERT Workload):
Executes all 72 tests (including BERT layer-0 intermediate weight validation and golden masked-dense reference checks):

```bash
python -m pytest tests/ -v
```

### Verified Output:
```
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[...] PASSED (7 tests)
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip[...] PASSED (8 tests)
tests/test_cuda_kernel.py::test_custom_cuda_matches_reference SKIPPED (sparse_cuda_ext CUDA extension module is not compiled)
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[...] PASSED (11 tests)
tests/test_formats.py::TestCSRProperties::test_sparsity_calculation PASSED
tests/test_formats.py::TestCSRProperties::test_row_ptr_shape PASSED
tests/test_formats.py::TestRowLengths::test_uniform_row_lengths PASSED
tests/test_formats.py::TestRowLengths::test_skewed_row_lengths PASSED
tests/test_formats.py::TestRowLengths::test_empty_matrix_stats PASSED
tests/test_formats.py::TestMaskedConversion::test_masked_csr_matches_dense_masked PASSED
tests/test_formats.py::TestMaskedConversion::test_shape_mismatch_raises PASSED
tests/test_formats.py::TestCorrectnessOracle::test_csr_matmul_matches_dense_oracle[...] PASSED (5 tests)
tests/test_formats.py::TestCorrectnessOracle::test_threshold_based_sparsity PASSED
tests/test_packer.py::test_pack_unpack_has_expected_structure PASSED
tests/test_packer.py::test_top2_selection_is_by_block_norm PASSED
tests/test_workload.py::TestMagnitudePruningUnit::test_exact_topk_retention_count PASSED
tests/test_workload.py::TestMagnitudePruningUnit::test_deterministic_tie_breaking_by_flattened_index PASSED
tests/test_workload.py::TestMagnitudePruningUnit::test_signed_magnitude_pruning PASSED
tests/test_workload.py::TestMagnitudePruningUnit::test_boundary_and_zero_sparsity PASSED
tests/test_workload.py::TestMagnitudePruningUnit::test_input_with_existing_zeros PASSED
tests/test_workload.py::TestBERTWeightProperties::test_tensor_shape_and_dtype PASSED
tests/test_workload.py::TestBERTWeightProperties::test_tensor_finite_and_statistics PASSED
tests/test_workload.py::TestBERTWeightProperties::test_native_sparsity_is_zero PASSED
tests/test_workload.py::TestBERTDeterministicMasks::test_mask_determinism_and_reproducibility[...] PASSED (5 tests)
tests/test_workload.py::TestBERTDeterministicMasks::test_retained_and_pruned_weights[...] PASSED (4 tests)
tests/test_workload.py::TestBERTDeterministicMasks::test_empty_rows_at_high_sparsity PASSED
tests/test_workload.py::TestBERTWorkloadCorrectnessReference::test_csr_matches_dense_reference[...] PASSED (12 tests)
tests/test_workload.py::TestBERTWorkloadCorrectnessReference::test_metadata_file_validity PASSED

======================= 71 passed, 1 skipped in 13.02s =======================
```

### GPU Environment Smoke-Test Suite (Tests A–E):
Executes the official gated CUDA verification sequence:

```bash
python scripts/smoke_test_cuda.py
```
- **Test A (PyTorch GPU Init):** ✅ **PASSED** (`RTX 3050 Laptop GPU`, sm_86, 4096 MiB VRAM).
- **Test B (Dense GPU Computation):** ✅ **PASSED** (FP32 & FP16 cuBLAS matmul vs CPU reference).
- **Test C (Native CUDA Compilation):** 🔴 **BLOCKED** (Gated on `nvcc` & MSVC `cl.exe`).
- **Test D (cuSPARSE Baseline Readiness):** ✅ **PASSED** (GPU CSR SpMM vs dense reference).
- **Test E (Extension Toolchain Readiness):** 🔴 **BLOCKED** (Gated on `cl.exe`, `nvcc`, `CUDA_HOME`).

---

## 5. Current Environment Status & Remaining Blockers

- **GPU Hardware & Driver:** NVIDIA GeForce RTX 3050 Laptop GPU (GA107, sm_86, Driver 566.07) — ✅ Operational.
- **PyTorch CUDA Environment:** `torch==2.6.0+cu124` & `torchvision==0.21.0+cu124` in `.venv` — ✅ Operational.
- **cuBLAS & cuSPARSE Backends:** Verified executing on GPU in Tests B & D — ✅ Operational.
- **Native CUDA Compilation (nvcc):** 🔴 **BLOCKED** (Requires CUDA Toolkit 12.4.1 installation).
- **Host C++ Compiler (MSVC cl.exe):** 🔴 **BLOCKED** (Requires VS 2022 C++ Build Tools v143).
- **Compiler Environment Variables (`CUDA_HOME`):** 🔴 **BLOCKED** (Pending CUDA Toolkit install).

A comprehensive, step-by-step resolution plan is provided in:  
📄 **[`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md)**

---

## 6. Implementation Milestones & Roadmap

- **Phase 1 [Complete]:** Repository foundation, mid-semester code preservation, requirements traceability, and environment remediation plan.
- **Phase 2A & 2B [Complete]:** Target workload acquisition (BERT-Base layer 0), empirical sparsity characterization, deterministic pruning masks, CPU correctness oracle, and V1 architecture reconciliation.
- **Phase 2C [Current]:** CUDA development environment readiness, dependency isolation, toolchain compatibility audit, and GPU smoke-test suite (Tests A–E).
- **Phase 3:** Custom Kernel V1 implementation (CSR Row-per-Warp SpMM) and cuSPARSE CSR baseline evaluation.
- **Phase 4:** Custom Kernel V2 implementation (Tile-based Block-Sparse SpMM) and cuSPARSE BSR baseline.
- **Phase 5:** Custom Kernel V3 implementation (Load-balanced CSR SpMM for skewed row distributions).
- **Phase 6:** Systematic empirical benchmark sweeps (sparsity 50%–99%, dimension sweeps) and break-even analysis.
- **Phase 7:** PyTorch `SparseLinear` layer integration, visual interactive demonstration, and final technical report.