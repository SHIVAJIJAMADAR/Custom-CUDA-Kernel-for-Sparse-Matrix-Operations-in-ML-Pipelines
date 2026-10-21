# Rigorous GPU Benchmark & Correctness Protocol

**Project:** Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines  
**Authoritative Reference:** Project 10, pages 21–22, `CUDA_GPU_Projects_1-10_Getting_Started.pdf`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (Ampere `sm_86`, 4 GB GDDR6)  
**Document Status:** Baselined (Phase 1)

---

## 1. Benchmarking Philosophy & Integrity Standards

To ensure scientifically valid and reproducible results, the benchmarking suite must adhere to the following principles:

1. **No Unearned Claims:** A kernel cannot be reported as faster than cuBLAS or cuSPARSE without empirical, logged, reproducible timing evidence.
2. **Four Required Comparison Baselines:** All evaluations must compare against the four official baselines under identical mathematical inputs, shapes, dtypes, and device states:
   - **Baseline 1:** Dense cuBLAS GEMM ($Y = A \times X$)
   - **Baseline 2:** cuSPARSE CSR SpMM (`cusparseSpMM`)
   - **Baseline 3:** Naive custom sparse (e.g., generic CSR row-per-warp)
   - **Baseline 4:** Optimized custom sparse (e.g., shared-memory tiled / load-balanced / shuffle-reduced)
   *(cuSPARSE BSR is evaluated as applicable additional work when block sparsity warrants it, not as a replacement for CSR).*
3. **Synthetic vs Real Workloads:** Synthetic matrices (e.g. random or uniform masks) support controlled unit testing, debugging, and systematic parameter sweeps, but **do not replace empirical characterization of a real target workload**.
4. **C/C++ & CUDA Event Timing:** Host wall-clock measurements (`time.time()`) are forbidden for GPU kernel timing. All timings must use GPU hardware timer events (`cudaEventRecord`, `cudaEventElapsedTime`).
5. **Statistical Rigor:** Benchmarks must run sufficient iterations (minimum 100 runs) with explicit warmup (minimum 30 runs) and report distribution metrics (median, mean, std dev, min, max).

---

## 2. Experimental Measurement Protocol

### 2.1 Hardware Stabilization & Warmup
Laptop GPUs frequently fluctuate clock speeds due to dynamic power boost and thermal limits.
- **Warmup:** Execute $\ge 30$ full iterations prior to recording timing events.
- **Clock Stabilization:** Query GPU temperature and clock frequency before and after benchmark batches using `pynvml` or `nvidia-smi`.
- **Thermal Throttling Protection:** If GPU junction temperature exceeds $80^\circ\text{C}$, introduce a 2-second sleep between parameter sweeps.

### 2.2 Cache State Management: Warm vs. Cold Cache
By default, repeated kernel runs operate on warm L2 cache ($2\text{ MB}$ on RTX 3050). In real inference pipelines, activations and weight matrices are frequently fetched from high-bandwidth device memory (cold cache).
- **Methodology Choice:** Evaluating cold-cache vs warm-cache latency is an explicit benchmark methodology choice to study cache residency effects, not an unconditional separate deliverable.
- **Warm-Cache Benchmark:** Measure steady-state throughput where matrix tiles reside in L2 cache.
- **Cold-Cache Benchmark:** Allocate a dummy buffer equal to twice the L2 cache size ($2 \times 2\text{ MB} = 4\text{ MB}$). Between measured iterations, launch a memory-touch kernel to evict existing cache lines:
  ```python
  # Flush L2 cache
  dummy_cache_flush.zero_()
  torch.cuda.synchronize()
  ```

### 2.3 Timing Harness Code Structure
```python
# Standardized CUDA Event Timing Pattern
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 1. Warmup loop
for _ in range(num_warmup):
    kernel_fn(*args)
torch.cuda.synchronize()

# 2. Measurement loop
timings_ms = []
for _ in range(num_iterations):
    if flush_l2:
        cache_flush_buffer.fill_(0)
    
    start_event.record()
    kernel_fn(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    timings_ms.append(start_event.elapsed_time(end_event))
```

---

## 3. Metrics & Derivation Formulas

### 3.1 Effective Computational Throughput (GFLOPS)
In a sparse-dense matrix multiplication $Y = A \times X$, where $A \in \mathbb{R}^{M \times K}$ has $NNZ$ non-zero elements and $X \in \mathbb{R}^{K \times N}$:
Each non-zero element performs $1$ multiply and $1$ accumulate across all $N$ columns:

$$\text{FLOPs} = 2 \times NNZ \times N$$

$$\text{Effective GFLOPS} = \frac{2 \times NNZ \times N}{\text{Latency (seconds)} \times 10^9}$$

> [!NOTE]
> Effective GFLOPS measures useful work relative to the non-zero count, not raw dense operations.

### 3.2 Effective Memory Bandwidth (GB/s)
Memory traffic accounts for reading the sparse representation of $A$, reading the dense matrix $X$, and writing the output $Y$:

$$\text{Bytes Read}(A_{\text{CSR}}) = NNZ \times \text{sizeof(dtype)} + NNZ \times \text{sizeof(int32)} + (M + 1) \times \text{sizeof(int32)}$$

$$\text{Bytes Read}(X) = K \times N \times \text{sizeof(dtype)}$$

$$\text{Bytes Written}(Y) = M \times N \times \text{sizeof(dtype)}$$

$$\text{Effective Bandwidth (GB/s)} = \frac{\text{Bytes Read} + \text{Bytes Written}}{\text{Latency (seconds)} \times 10^9}$$

### 3.3 Speedup Ratios
$$\text{Speedup vs Dense} = \frac{\text{Latency}(\text{cuBLAS Dense GEMM})}{\text{Latency}(\text{Custom SpMM})}$$

$$\text{Speedup vs cuSPARSE} = \frac{\text{Latency}(\text{cuSPARSE CSR SpMM})}{\text{Latency}(\text{Custom SpMM})}$$

### 3.4 Break-Even Sparsity
The break-even point is the exact sparsity percentage $S^* \in (0, 1)$ where:
$$\text{Latency}(\text{Custom SpMM}(S^*)) = \text{Latency}(\text{cuBLAS Dense})$$
For any sparsity $S > S^*$, the custom kernel delivers positive speedup.

---

## 4. Parameter Sweep Space

| Parameter | Values Explored | Justification |
| :--- | :--- | :--- |
| **Sparsity Level** | $50\%, 60\%, 70\%, 80\%, 85\%, 90\%, 95\%, 98\%, 99\%$ | Characterize break-even threshold and scaling curves across official 50%–99% range. |
| **Matrix Height ($M$)** | $512, 1024, 2048, 4096$ | Representative of transformer attention / FFN projection dimensions. |
| **Matrix Width ($K$)** | $512, 1024, 2048, 4096$ | Matching inner dimension scaling. |
| **Batch / Activation ($N$)**| $32, 64, 128, 256, 512$ | Token batch size in autoregressive decoding vs prompt prefill. |
| **Data Types** | FP32, FP16 | Standard deep learning numerical representations. |
| **Row Skewness** | Uniform, Normal, Zipfian ($\alpha = 1.5$) | Simulates natural activation distributions vs uniform pruning. |

---

## 5. Artifact Storage & Provenance

1. **Raw Benchmarks:** Saved as structured JSON files in `results/raw/` containing full configuration metadata, hardware timestamps, GPU clock speeds, and individual run timings.
2. **Reproducible Plots:** Generated via Python scripts saving high-resolution PNG/SVG figures in `results/plots/`.
