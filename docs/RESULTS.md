# Project 10 Results & Validation Ledger

**Project:** Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines  
**Document Status:** Current Results Tracking (Phase 1)  
**Last Updated:** October 21, 2026  

---

## 1. Executive Summary of Current Results

| Test Category | Total Tests | Passed | Skipped | Failed | Execution Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mid-Sem 2-of-8 Block-Sparse Packer** | 2 | 2 | 0 | 0 | ✅ Verified on CPU |
| **CPU Reference Matmul (FP32 & FP16)** | 15 | 15 | 0 | 0 | ✅ Verified on CPU |
| **CSR Round-Trip & Edge Cases** | 11 | 11 | 0 | 0 | ✅ Verified on CPU |
| **CSR Properties & Row-Length Stats** | 5 | 5 | 0 | 0 | ✅ Verified on CPU |
| **Masked Conversion & Shape Checks** | 2 | 2 | 0 | 0 | ✅ Verified on CPU |
| **Masked Correctness Oracle ($A_{\text{masked}} \times X$)**| 5 | 5 | 0 | 0 | ✅ Verified on CPU |
| **CUDA Kernel Execution (`test_cuda_kernel.py`)** | 1 | 0 | 1 | 0 | ⏸️ Correctly Skipped (No CUDA GPU) |
| **TOTAL** | **41** | **40** | **1** | **0** | **40 Passed, 1 Skipped** |

> [!IMPORTANT]
> **No GPU performance claims have been recorded yet.**  
> While the mathematical logic, format converters, and CPU references are thoroughly verified (40 tests passing), no CUDA kernels have run on the physical GPU due to missing CUDA Toolkit / MSVC prerequisites. All benchmark numbers from the mid-semester presentation were preliminary targets, not measured GPU empirical data.

---

## 2. Verified CPU Test Results (Empirical Log)

The complete CPU test suite was executed against Python 3.13 on October 21, 2026:

```
platform win32 -- Python 3.13.0, pytest-8.3.4, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: C:\Users\nikit\OneDrive\Desktop\EDI_CUDA
collected 41 items

tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[1-32-256] PASSED
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[4-32-256] PASSED
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[16-64-256] PASSED
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[8-64-512] PASSED
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[4-128-256] PASSED
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_sparse_matmul_matches_dense[32-128-512] PASSED
tests/test_cpu_reference.py::TestCPUReferenceMatmul::test_fp16_reference PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip_preserves_active_blocks[32-256] PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip_preserves_active_blocks[64-256] PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip_preserves_active_blocks[128-256] PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip_preserves_active_blocks[64-512] PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip_preserves_active_blocks[128-512] PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_roundtrip_preserves_active_blocks[256-256] PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_zero_weight_handling PASSED
tests/test_cpu_reference.py::TestPackUnpackRoundtrip::test_single_strong_block PASSED
tests/test_cuda_kernel.py::test_custom_cuda_matches_reference SKIPPED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[4-4-0.0] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[4-4-0.5] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[4-4-0.9] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[8-16-0.75] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[1-1-0.0] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[32-256-0.95] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_roundtrip[64-64-0.5] PASSED
tests/test_formats.py::TestCSRRoundtrip::test_all_zeros PASSED
tests/test_formats.py::TestCSRRoundtrip::test_all_ones PASSED
tests/test_formats.py::TestCSRRoundtrip::test_single_element PASSED
tests/test_formats.py::TestCSRRoundtrip::test_fp16_roundtrip PASSED
tests/test_formats.py::TestCSRProperties::test_sparsity_calculation PASSED
tests/test_formats.py::TestCSRProperties::test_row_ptr_shape PASSED
tests/test_formats.py::TestRowLengths::test_uniform_row_lengths PASSED
tests/test_formats.py::TestRowLengths::test_skewed_row_lengths PASSED
tests/test_formats.py::TestRowLengths::test_empty_matrix_stats PASSED
tests/test_formats.py::TestMaskedConversion::test_masked_csr_matches_dense_masked PASSED
tests/test_formats.py::TestMaskedConversion::test_shape_mismatch_raises PASSED
tests/test_formats.py::TestCorrectnessOracle::test_csr_matmul_matches_dense_oracle[1-8-8-0.5] PASSED
tests/test_formats.py::TestCorrectnessOracle::test_csr_matmul_matches_dense_oracle[4-16-32-0.75] PASSED
tests/test_formats.py::TestCorrectnessOracle::test_csr_matmul_matches_dense_oracle[8-32-64-0.9] PASSED
tests/test_formats.py::TestCorrectnessOracle::test_csr_matmul_matches_dense_oracle[16-64-128-0.95] PASSED
tests/test_formats.py::TestCorrectnessOracle::test_threshold_based_sparsity PASSED
tests/test_packer.py::test_pack_unpack_has_expected_structure PASSED
tests/test_packer.py::test_top2_selection_is_by_block_norm PASSED

======================== 40 passed, 1 skipped in 3.66s ========================
```

---

## 3. Pending GPU Benchmark Experiments

The following benchmarks are scheduled for execution once the environment remediation plan is completed:

| Experiment ID | Benchmark Target | Metric | Status |
| :--- | :--- | :--- | :--- |
| **EXP-01** | cuBLAS Dense GEMM vs Matrix Dimension ($M, K \in [512, 4096]$) | Latency ($\mu s$), TFLOPS | `[BLOCKED on CUDA]` |
| **EXP-02** | cuSPARSE CSR SpMM vs Sparsity ($50\% \to 99\%$) | Latency ($\mu s$), Speedup vs cuBLAS | `[BLOCKED on CUDA]` |
| **EXP-03** | Custom V1 (CSR Row-per-Warp) vs cuSPARSE CSR | Latency ($\mu s$), Speedup | `[BLOCKED on CUDA]` |
| **EXP-04** | Custom V2 (Block-Sparse) vs cuSPARSE BSR | Latency ($\mu s$), Effective Bandwidth | `[BLOCKED on CUDA]` |
| **EXP-05** | Custom V3 (Load-Balanced) on Skewed Row Distributions | Warp Divergence %, Tail Latency | `[BLOCKED on CUDA]` |
| **EXP-06** | Break-Even Sparsity Crossover Analysis | Crossover Sparsity Threshold ($S^*$) | `[BLOCKED on CUDA]` |
| **EXP-07** | End-to-End PyTorch SparseLinear Layer Forward Latency | Batch Latency ($ms$) | `[BLOCKED on CUDA]` |
