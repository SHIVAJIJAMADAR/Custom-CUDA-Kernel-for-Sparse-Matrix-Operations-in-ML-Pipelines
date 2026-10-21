# Project 10 Master Development Checklist

**Project:** Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines  
**Standard Status Taxonomy:**
- `[COMPLETE & VERIFIED]`: Fully implemented, automated tests passing with verified provenance.
- `[IMPLEMENTED BUT UNVERIFIED]`: Source code exists, but execution on target hardware is pending.
- `[BLOCKED]`: Cannot proceed due to missing prerequisite software, hardware, or dependencies.
- `[NOT STARTED]`: Scheduled for future implementation phases.
- `[OPTIONAL]`: Advanced exploration beyond core project scope.

---

## 1. Environment & Infrastructure Foundation

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **GPU Hardware Detection** | RTX 3050 Laptop (sm_86) | `[COMPLETE & VERIFIED]` | Detected via system diagnostic: GA107, 4 GB VRAM. |
| **NVIDIA Display Driver** | Driver $\ge 525$ | `[COMPLETE & VERIFIED]` | Driver `566.07` installed and operational. |
| **Host Python Environment** | Python $\ge 3.10$ | `[COMPLETE & VERIFIED]` | Python `3.13.0` (64-bit) active. |
| **Visual Studio C++ Build Tools** | MSVC `cl.exe` | `[BLOCKED]` | Missing MSVC v143 toolset. Blocked pending user install. |
| **NVIDIA CUDA Toolkit 12.x** | `nvcc` compiler | `[BLOCKED]` | Missing `nvcc.exe`. Blocked pending user install. |
| **CUDA-Enabled PyTorch** | `torch.cuda.is_available()` | `[BLOCKED]` | Currently `2.10.0+cpu`. Blocked pending `cu124` install. |
| **Environment Variables (`CUDA_PATH`)**| System PATH config | `[BLOCKED]` | Blocked on CUDA Toolkit installation. |
| **Nsight Profiling Suite** | `ncu` / `nsys` | `[BLOCKED]` | Blocked on CUDA Toolkit / Nsight installation. |
| **Git Repository & Remote Link** | GitHub synchronization | `[COMPLETE & VERIFIED]` | Local git initialized on `main`, tracking remote `origin/main`. |
| **Root Cleanliness & .gitignore** | Build artifact hygiene | `[COMPLETE & VERIFIED]` | Root `.gitignore` configured for Python, CUDA, VS, and OS. |

---

## 2. Workload Discovery & Sparsity Analysis (R1)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Real Target Workload Selection** | R1.1 | `[BLOCKED]` | Blocked pending model selection (LLaMA/BERT/ViT projection). |
| **Workload Placeholder Configuration** | R1.1 | `[COMPLETE & VERIFIED]` | `configs/demo_config.yaml` explicitly marked as synthetic example. |
| **Sparsity Characterization Script** | R1.2 | `[COMPLETE & VERIFIED]` | `patterns/analyze_sparsity.py` implements density, skew, & block analysis. |
| **Row-Length Statistics & Distribution**| R1.2 | `[COMPLETE & VERIFIED]` | Mean, std, skewness, min, max, median implemented and tested. |
| **Format Recommendation Logic** | R1.3 | `[COMPLETE & VERIFIED]` | Pattern classification and format advisory implemented. |

---

## 3. Sparse Formats & Converters (R2)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **CSR Conversion Utility** | R2.1 | `[COMPLETE & VERIFIED]` | `formats/csr.py` (`dense_to_csr`, `csr_to_dense`, `row_lengths`). |
| **CSR Format Unit Tests** | R2.1 | `[COMPLETE & VERIFIED]` | 23 tests passing in `tests/test_formats.py`. |
| **2-of-8 Block-Sparse Packer (Preserved)**| R2.2 | `[COMPLETE & VERIFIED]` | `formats/blocked.py` (preserved mid-semester format). |
| **2-of-8 Block-Sparse Unit Tests** | R2.2 | `[COMPLETE & VERIFIED]` | 17 tests passing in `tests/test_packer.py` and `test_cpu_reference.py`. |
| **BSR Converter** | R2.2 | `[NOT STARTED]` | Scheduled for Phase 3. |

---

## 4. Baselines (R3)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Dense PyTorch/cuBLAS Harness** | R3.1 | `[IMPLEMENTED BUT UNVERIFIED]`| `bench/benchmark_dense.py` written; blocked on CUDA PyTorch. |
| **cuSPARSE CSR SpMM Harness** | R3.2 | `[IMPLEMENTED BUT UNVERIFIED]`| `bench/cusparse_spmm_baseline.cu` scaffolded; blocked on nvcc. |
| **cuSPARSE BSR SpMM Harness** | R3.3 | `[NOT STARTED]` | Scheduled for Phase 3. |

---

## 5. Custom CUDA Kernels (R4, R5, R6)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **V0 Block-Sparse Kernel (Preserved)**| Legacy | `[IMPLEMENTED BUT UNVERIFIED]`| `kernels/sparse_linear_v0.cu` preserved; unverified on GPU. |
| **V0 Kernel PyTorch Extension Build** | R10.1 | `[BLOCKED]` | `python/build_extension.py` blocked on MSVC `cl.exe` + `nvcc`. |
| **V0 Kernel Test (`test_cuda_kernel`)**| Legacy | `[BLOCKED]` | Test skipped on CPU: `tests/test_cuda_kernel.py`. |
| **V1 CSR Row-per-Warp Kernel** | R4 | `[NOT STARTED]` | Implementation scheduled for Phase 2. |
| **V1 Intra-Warp Shuffle Reduction** | R4.2 | `[NOT STARTED]` | Scheduled for Phase 2. |
| **V2 Block-Sparse Tiled Kernel** | R5 | `[NOT STARTED]` | Implementation scheduled for Phase 3. |
| **V3 Load-Balanced Skewed-Row Kernel**| R6 | `[NOT STARTED]` | Implementation scheduled for Phase 4. |

---

## 6. Correctness Verification Oracle (R7)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Masked-Dense Mathematical Oracle** | R7.1 | `[COMPLETE & VERIFIED]` | Implemented in `tests/test_formats.py` ($Y = (W \odot M) \times X$). |
| **FP32 / FP16 Tolerance Checks** | R7.2 | `[COMPLETE & VERIFIED]` | Tested with rigorous tolerance assertions. |
| **Boundary & Edge-Case Coverage** | R7.3 | `[COMPLETE & VERIFIED]` | All-zeros, all-ones, single-element, and skew edge cases tested. |
| **Custom Kernel vs Oracle Verification**| R7.1 | `[BLOCKED]` | Blocked until CUDA kernels can execute on GPU. |

---

## 7. Systematic Benchmarking & Break-Even Analysis (R8, R9)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **CUDA Event Timing Protocol** | R9.1 | `[IMPLEMENTED BUT UNVERIFIED]`| Defined in `bench/benchmark_dense.py`; blocked on GPU. |
| **L2 Cache Flush Protocol** | R9.1 | `[IMPLEMENTED BUT UNVERIFIED]`| Protocol defined in `docs/BENCHMARK_PROTOCOL.md`. |
| **Sparsity Parameter Sweep (50%–98%)**| R8.1 | `[NOT STARTED]` | Scheduled for Phase 5. |
| **Matrix Dimension Sweep (512–4096)** | R8.1 | `[NOT STARTED]` | Scheduled for Phase 5. |
| **Row Skewness Sweep** | R8.1 | `[NOT STARTED]` | Scheduled for Phase 5. |
| **Break-Even Analysis & Crossover** | R9.2 | `[NOT STARTED]` | Scheduled for Phase 5. |
| **Automated Plotting Scripts** | R11.1 | `[NOT STARTED]` | Scheduled for Phase 5. |

---

## 8. PyTorch Integration & Deliverables (R10, R11)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **PyTorch C++ Binding (`sparse_op.cpp`)**| R10.1 | `[IMPLEMENTED BUT UNVERIFIED]`| Source file present; blocked on compilation. |
| **Drop-in `SparseLinear` Layer** | R10.2 | `[NOT STARTED]` | Scheduled for Phase 5. |
| **Comprehensive Technical Report** | R11.2 | `[NOT STARTED]` | Scheduled for Phase 6. |
| **Interactive Visual Demonstration** | R11.3 | `[NOT STARTED]` | Scheduled for Phase 6. |

---

## 9. Optional Scope (R12)

| Task / Item | Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **cuSPARSELt 2:4 Structured Sparsity** | R12.1 | `[OPTIONAL]` | Explore Ampere Sparse Tensor Cores if core scope succeeds. |
| **Triton SpMM Kernel Comparison** | R12.2 | `[OPTIONAL]` | Explore Triton kernel comparison. |
| **INT8 Quantized SpMM** | R12.3 | `[OPTIONAL]` | Mixed-precision quantization exploration. |
