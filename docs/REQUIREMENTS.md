# Project 10 Requirements Specification: Traceability Matrix

**Project:** Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines  
**Official Source:** Project 10, pages 21–22, `CUDA_GPU_Projects_1-10_Getting_Started.pdf`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (Ampere `sm_86`, 4 GB VRAM)  
**Document Status:** Baselined (Phase 1)

---

## 1. Executive Summary & Objective

The primary objective of Project 10 is to design, implement, optimize, and profile custom CUDA kernels for sparse matrix-matrix multiplication (SpMM) tailored to specific sparsity patterns arising in machine learning workloads (e.g., pruned transformer attention projections, feed-forward layers, or MLP weights). 

The custom kernels must be benchmarked across the **four official Project 10 comparison baselines**:
1. **Dense cuBLAS GEMM** ($Y = A \times X$ or $Y = X \times W^T$)
2. **cuSPARSE CSR SpMM** (`cusparseSpMM` using default and specialized algorithms)
3. **Naive custom sparse kernel** (baseline generic sparse implementation, e.g. basic CSR row-per-warp)
4. **Optimized custom sparse kernel** (tailored variant, e.g. shared-memory tiled, shuffle-reduced, or load-balanced)

Performance and correctness must be evaluated across systematic sweeps of sparsity levels (**official 50%–99% range**), matrix dimensions, block configurations, and row distributions.

---

## 2. Traceable Requirements Matrix

### R1. Target Workload & Sparsity Characterization
- **R1.1 Workload Selection (Mandatory Prerequisite):** Identify at least one representative deep learning weight matrix (e.g., LLaMA / Mistral / BERT / ViT linear projection) or verified pruned model layer. Synthetic matrices support controlled unit testing, debugging, and parameter sweeps, but **do not replace characterization of a real target workload**.
- **R1.2 Sparsity Profiling (Mandatory):** Implement an analysis script (`patterns/analyze_sparsity.py`) computing:
  - Total dimensions $(M, K)$, active dtype, element count, non-zero element count ($NNZ$), and overall sparsity fraction ($\% = 1 - NNZ / (M \times K)$).
  - Block occupancy / pattern analysis across candidate block tile sizes ($8\times 8, 16\times 16, 32\times 32, 64\times 64$).
  - Row-length distribution (min, max, mean, median, standard deviation, skewness coefficient).
  - Sparsity classification: Unstructured, Block-sparse, Diagonal/Band, or Skewed row-sparse.
- **R1.3 Format Recommendation:** Automatically advise format selection (CSR vs BSR vs Blocked) based on block-density metrics.

### R2. Sparse Format Conversions & Storage
- **R2.1 Compressed Sparse Row (CSR - Mandatory):**
  - Implement bidirectional converter: Dense $\leftrightarrow$ CSR (`formats/csr.py`).
  - Storage structures: `values` (1D array of nonzeros), `col_indices` (1D array of column indices, `int32`), `row_ptr` (1D array of row offsets of length $M+1$, `int32`).
  - Support masked-dense to CSR extraction with automatic zero-pruning.
- **R2.2 Block-Sparse & BSR Storage (Applicable / Workload-Dependent):**
  - Implement block-sparse representation for $B \times B$ non-zero sub-blocks (`formats/blocked.py`, `formats/bsr.py`) when block structure is present.
  - Preserve mid-semester 2-of-8 block-sparse packer/unpacker with metadata encoding.
- **R2.3 Data Integrity & Memory Footprint:**
  - Round-trip fidelity: $Dense \to Sparse \to Dense \equiv Dense$.
  - Exact memory footprint accounting comparing dense vs sparse storage.

### R3. Industry Baselines (cuBLAS & cuSPARSE)
- **R3.1 cuBLAS Dense GEMM (Mandatory Baseline 1):**
  - Implement standard dense matrix multiplication baseline ($Y = A \times X$).
  - Measure execution latency using GPU CUDA events with warmup. (L2 cache eviction is an explicit benchmark methodology choice to evaluate cold-cache vs warm-cache latency, not an unconditional separate deliverable).
- **R3.2 cuSPARSE CSR SpMM (Mandatory Baseline 2):**
  - Implement `cusparseSpMM` wrapper executing CSR SpMM ($Y = \alpha \cdot A_{\text{CSR}} \cdot X_{\text{dense}} + \beta Y$).
  - Evaluate both default and non-default algorithms (`CUSPARSE_SPMM_ALG_DEFAULT`, `CUSPARSE_SPMM_CSR_ALG2`).
- **R3.3 cuSPARSE BSR Baseline (Applicable Additional Work):**
  - Evaluate `cusparseSpMM` in BSR format as applicable additional work when block sparsity warrants it, not as a mandatory replacement for cuSPARSE CSR.

### R4. Custom Kernel V1: CSR Row-per-Warp SpMM (Mandatory Baseline 3 / Core Kernel)
- **R4.1 Thread-to-Data Mapping:**
  - Assign exactly one 32-thread CUDA warp per output row of matrix $A$ (or multiple rows per threadblock).
  - Threads within a warp cooperate to process the non-zero elements of row $i$ in parallel across output column dimension $N$.
- **R4.2 Intra-Warp Reduction:**
  - Use warp-synchronous shuffle intrinsics (`__shfl_down_sync`) to reduce partial products without shared memory bank conflicts.
- **R4.3 Memory Coalescing:**
  - Ensure coalesced global memory reads across columns of dense input matrix $X$.
  - Implement vectorized loads (`float4` / `half2`) where alignment allows.

### R5. Custom Kernel V2: Block-Sparse SpMM
- **R5.1 Tiling & Shared Memory:**
  - Implement 2D threadblock tiling utilizing on-chip Shared Memory (`__shared__`) to cache reusable blocks of the dense matrix $X$.
  - Block sizes matched to Ampere SM shared memory limits (48 KB / 100 KB).
- **R5.2 Threadblock Specialization:**
  - Preserve and refine the mid-semester V0 tile-based concept into a verified V2 kernel.
  - Mitigate uncoalesced memory traffic via cooperative shared memory loading.

### R6. Custom Kernel V3: Load-Balanced CSR SpMM (Skewed Rows)
- **R6.1 Load Imbalance Mitigation:**
  - Address severe warp divergence and tail latency caused by high row-length variance.
  - Implement row binning (sorting rows by $NNZ$) or segmented scan / work-stealing CTA assignment.
- **R6.2 Dynamic Work Allocation:**
  - Assign short rows to single threads/half-warps; assign long rows to full warps or entire threadblocks.

### R7. Masked-Dense Correctness Validation Oracle
- **R7.1 Golden Mathematical Reference:**
  - Correctness oracle: $Y_{\text{expected}} = (W \odot M_{\text{mask}}) \times X$.
  - Direct dense reference implementation running in 64-bit precision where needed.
- **R7.2 Numerical Tolerance Criteria:**
  - FP32 tolerance: Absolute error $\le 10^{-5}$, Relative error $\le 10^{-4}$.
  - FP16 tolerance: Absolute error $\le 10^{-3}$, Relative error $\le 10^{-2}$.
- **R7.3 Edge Case Test Coverage:**
  - 100% sparse (all zeros), 0% sparse (all ones), single non-zero, extreme aspect ratios ($M \ll K$ or $M \gg K$).

### R8. Comprehensive Evaluation Matrix
- **R8.1 Systematic Parameter Sweeps (Mandatory):**
  - **Sparsity Levels:** $50\%, 60\%, 70\%, 80\%, 85\%, 90\%, 95\%, 98\%, 99\%$ (official Project 10 range: 50%–99%).
  - **Matrix Dimensions:** $M, K \in \{512, 1024, 2048, 4096\}$, $N \in \{32, 64, 128, 256, 512\}$.
  - **Block Sizes:** $8\times 8, 16\times 16, 32\times 32$ for block-sparse kernels (when applicable).
  - **Row Skewness:** Uniform vs Zipfian/power-law distribution of row non-zero counts.
  - **Data Types:** FP32 and FP16.
- **R8.2 End-to-End Layer Benchmark:**
  - Evaluate linear projection layer with sparse weights in a simulated or real forward pass.

### R9. Performance Metrics & Break-Even Analysis
- **R9.1 Core Metrics:**
  - Execution Latency ($\mu s$ / $ms$) via CUDA events (excluding host-to-device transfer).
  - Effective Computational Throughput (GFLOPS): $\frac{2 \times NNZ \times N}{\text{Latency}}$.
  - Effective Memory Bandwidth Utilization (GB/s): $\frac{\text{Bytes Read} + \text{Bytes Written}}{\text{Latency}}$.
  - Speedup relative to cuBLAS Dense GEMM ($S_{\text{dense}} = T_{\text{cuBLAS}} / T_{\text{custom}}$).
  - Speedup relative to cuSPARSE CSR ($S_{\text{cuSPARSE}} = T_{\text{cuSPARSE}} / T_{\text{custom}}$).
- **R9.2 Break-Even Sparsity Analysis:**
  - Identify the exact sparsity crossover percentage where the custom sparse kernel outperforms dense cuBLAS.
  - Document sensitivity to dense batch size / matrix width $N$.
- **R9.3 Model Accuracy Retention (Pruning Context):**
  - Document accuracy retention vs sparsity trade-offs if magnitude or structured pruning is applied to a real model.

### R10. PyTorch Extension & Integration
- **R10.1 C++/CUDA Extension:**
  - Complete `csrc/sparse_op.cpp` binding using PyTorch C++ API (`torch::Tensor`).
  - Provide Python module `sparse_ops` wrapping forward SpMM calls.
- **R10.2 Drop-in Sparse Linear Layer:**
  - Implement `torch.nn.Module` subclass `SparseLinear` executing the custom kernel in forward passes.

### R11. Deliverables & Demonstration
- **R11.1 Benchmark Evidence & Provenance:**
  - Raw JSON/CSV execution logs stored in `results/raw/`.
  - Fully automated, reproducible plotting scripts generating publication-quality figures in `results/plots/`.
- **R11.2 Comprehensive Technical Report:**
  - Detailed written document explaining design choices, memory hierarchy optimizations, divergence analysis, and experimental findings.
- **R11.3 Visual Demonstration:**
  - Interactive or scriptable visual demo visualizing matrix sparsity patterns, step-by-step kernel execution / tile mappings, and real-time benchmark speedup comparisons based on real measured data.

### R12. Explicitly Optional Work
- **R12.1 cuSPARSELt 2:4 Structured Sparsity:** Optional exploration of Ampere Sparse Tensor Core 2:4 support.
- **R12.2 OpenAI Triton SpMM:** Optional Triton kernel comparison.
- **R12.3 INT8 Quantized SpMM:** Optional mixed-precision quantization.
