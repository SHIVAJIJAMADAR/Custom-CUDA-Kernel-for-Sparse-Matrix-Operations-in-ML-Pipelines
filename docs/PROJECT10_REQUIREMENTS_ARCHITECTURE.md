# Project 10 — Requirements-Locked Architecture

## 1. Authority and scope

Primary source: *CUDA and GPU Research Track — Getting Started Guide*, Project 10, pages 21–22. The existing EDI mid-semester PPT is treated as a prior proposal, not as the scope authority.

**Project objective:** characterize sparsity in a real target model's weights or masks, build sparse-times-dense matrix multiplication (SpMM) kernel(s) suited to that pattern, and compare them fairly against cuSPARSE and dense cuBLAS.

**Research question:** at which sparsity levels and structures does a custom sparse kernel beat cuSPARSE and dense cuBLAS?

Do not claim a speedup unless correctness and repeatable measurements support it. If the custom kernel does not beat cuSPARSE on the target pattern, document why; dense cuBLAS remains a mandatory comparison and its break-even point must be reported.

## 2. Requirements traceability

| ID | Requirement from Project 10 | Implementation evidence / acceptance check |
|---|---|---|
| R1 | Characterize target sparsity, block structure, and row-length distribution | `patterns/` analysis script emits dimensions, dtype, nonzero count, sparsity percentage, block occupancy, row-length statistics, and the selected target layer/mask source |
| R2 | Convert dense inputs to CSR / BSR / blocked formats | `formats/` converters with round-trip and edge-case tests; only implement/use the formats required by the observed target pattern |
| R3 | Dense cuBLAS baseline | Reproducible dense matmul benchmark with the exact same input values, shape, dtype, and device as sparse runs |
| R4 | cuSPARSE CSR SpMM baseline | Reproducible `torch.sparse.mm` CSR baseline (or equivalent supported cuSPARSE path), checked against the masked dense reference |
| R5 | Custom kernel v1: row-per-warp CSR SpMM | Correctness tests and benchmark row in the result CSV |
| R6 | Custom kernel v2: block-sparse with shared-memory tiles | Implement after v1; only when it matches the target workload and can be compared fairly |
| R7 | Custom kernel v3: load-balanced variant for skewed rows | Implement and evaluate when row-length analysis shows skew is relevant; document results either way |
| R8 | Correctness against dense matmul using the masked matrix | All variants compared against `A_masked @ X`; explicit dtype tolerances and edge cases; no speed numbers for failing variants |
| R9 | Required baselines: dense cuBLAS, cuSPARSE CSR, naive custom sparse, optimized custom sparse | One comparison table with all applicable variants and identical workload parameters |
| R10 | Experiments: sparsity 50%–99%, block size, dense width N, format, row skew/load balance, dtype, end-to-end latency/accuracy | Config-driven benchmark sweeps; results saved as raw CSV/JSON, with unsupported configurations marked rather than fabricated |
| R11 | Metrics: latency, speedup vs cuSPARSE and dense, effective GFLOPS, memory bandwidth, accuracy retention if pruned, break-even sparsity | Calculated from measured data and included in plots/report |
| R12 | PyTorch extension and end-to-end linear-layer/small-model test | Python-callable extension and output/correctness test in a linear layer; accuracy retention only if pruning changes model weights |
| R13 | Include format-conversion cost when relevant | Report kernel-only time and conversion + execution time when conversion is part of the real workflow |
| R14 | Final deliverables | Sparse kernels/converters, benchmark data/plots, break-even analysis, technical report, and demo |

## 3. Scope lock

### Required core
- Workload/sparsity characterization based on the actual target weights or mask.
- CSR converter and CSR SpMM v1.
- Dense cuBLAS and cuSPARSE CSR SpMM baselines.
- Masked-dense correctness oracle and automated tests.
- Block-sparse v2 and load-balanced v3 along the Project 10 plan.
- Fair benchmark harness, required metrics, break-even analysis, PyTorch integration, and demo/report.

### Conditional / optional in the source
- Triton block-sparse baseline.
- cuSPARSELt 2:4 baseline, only if supported by the target GPU/workload (the guide states Ampere or newer for sparse Tensor Core 2:4).
- Other sparse formats when the measured target pattern warrants them.

### Do not silently add
- A generic sparse library intended to support every format/workload.
- Model training, a pruning research project, cloud services, authentication, data storage, multi-user roles, or unrelated dashboard features.
- A hard-coded 2-of-8 block pattern before confirming that it matches the real target workload.
- Performance claims based solely on synthetic random sparsity.

A visual demo is allowed as a presentation layer for the required benchmark evidence. It should only display required workload settings, correctness status, latency/speedup, effective GFLOPS, memory bandwidth, memory/format information when measured, break-even results, and plots. It must not expand the compute scope.

## 4. Proposed architecture

```text
Real target layer / weight matrix / mask
                 |
                 v
       Workload & sparsity analyzer
       (shape, dtype, sparsity, block pattern,
        row-length distribution, target device)
                 |
                 v
       Format selection + converters
         |            |             |
        CSR          BSR         blocked format
         |            |             |
         v            v             v
 Dense cuBLAS   cuSPARSE CSR    (only where relevant)
 reference       baseline
         \            /
          \          /
           v        v
        Correctness oracle: A_masked @ X
                 |
                 v
       Custom sparse kernel ladder
       v1 CSR row-per-warp
       v2 block-sparse + shared-memory tiles
       v3 row/load-balanced for skewed rows
                 |
                 v
       Benchmark harness (same workload)
       warmup + CUDA events + repeated runs
       correctness gate + environment metadata
                 |
                 v
       CSV/JSON -> metrics + plots + break-even analysis
                 |
                 v
       PyTorch extension -> end-to-end linear-layer test
                 |
                 v
       Demo/report showing measured evidence
```

## 5. Data and correctness contract

For a linear-layer style SpMM workload, define `A` as the original weight matrix, `M` as its chosen mask, `A_masked = A * M`, and `X` as the dense input. The reference output is `Y_ref = A_masked @ X`. Each sparse representation and kernel must produce `Y_custom` that matches `Y_ref` within an explicit tolerance for the tested dtype.

Tests must cover ordinary dimensions plus relevant edge cases (empty rows, uneven dimensions, non-multiple tile sizes, all-zero and dense masks, and skewed row lengths). Correctness must run before a variant is eligible for performance comparison.

Synthetic matrices may be used for unit tests and harness development. They are not proof that the final chosen sparse pattern represents the real model; the target model/weights/mask must be recorded for final results.

## 6. Benchmark and result contract

- Keep device, workload, shape, dtype, mask, and comparison conditions consistent across baselines.
- Warm up before timing; use CUDA events for GPU timing and repeated measurements, reporting the median and spread. The general guide recommends 100+ timed iterations.
- Record GPU model, driver, CUDA, PyTorch, and relevant library versions with every result.
- Store raw measurements as CSV/JSON. Never hand-enter or invent benchmark numbers.
- Report kernel-only latency separately from conversion + execution and end-to-end layer latency where relevant.
- Required derived comparisons: speedup vs cuSPARSE, speedup vs dense cuBLAS, effective GFLOPS, memory bandwidth, and break-even sparsity point. If pruning changes a model, report accuracy retention too.

## 7. Repository structure

```text
sparse-kernels/
  README.md
  pyproject.toml              # or requirements.txt; keep the setup appropriate to the actual environment
  CMakeLists.txt              # when the CUDA/C++ build path is established
  formats/
    __init__.py
    csr.py
    bsr.py
    blocked.py
  kernels/
    reference.py
    csr_spmm.cu
    block_sparse_spmm.cu
    load_balanced_spmm.cu
  python/
    extension_bindings.cpp
    ops.py
  patterns/
    analyze_sparsity.py
    workload_config.json
  bench/
    run_baselines.py
    run_sweeps.py
    metrics.py
  tests/
    test_formats.py
    test_correctness.py
    test_edge_cases.py
  results/
    raw/                       # generated CSV/JSON, not fabricated
    plots/                     # generated from raw results
  demo/                        # only the minimal presentation layer for required metrics
  docs/
    ARCHITECTURE.md
    BENCHMARK_PROTOCOL.md
    RESULTS.md
```

Files should be adjusted to the chosen build system, language boundary, and actual hardware. Avoid empty modules or speculative features added only to make the tree look complete.

## 8. Today’s implementation target

The first milestone should be a **reproducible vertical slice**, not a claim that half of all research/optimization is finished:

1. Confirm toolchain/GPU availability and record versions.
2. Commit this scope and repository structure.
3. Implement the masked-dense reference and CSR/BSR conversion tests.
4. Implement the sparsity analyzer for the selected target matrix/mask.
5. Establish dense cuBLAS and cuSPARSE CSR benchmark paths and export raw results.
6. Implement and test CSR row-per-warp custom kernel v1 if the available NVIDIA environment supports compiling/running it.
7. Generate the first comparison plot/report from actual measurements.

Do not substitute fake results if GPU/compiler/library access is unavailable. In that case, complete CPU/reference tests and benchmark harness interfaces, clearly label GPU performance as pending, and move CUDA execution to a supported NVIDIA environment.

## 9. Suggested Antigravity instruction

> Implement only Project 10 as specified in `docs/PROJECT10_REQUIREMENTS_ARCHITECTURE.md` (or this file if kept at repository root). Treat the requirements table and scope lock as binding. First inspect the repository and environment; report available compiler, NVIDIA GPU, CUDA, PyTorch, and cuSPARSE support before coding performance paths. Build tests and the masked-dense correctness oracle before benchmarking. Implement in small, testable increments: converters and analysis, baselines, CSR SpMM v1, then the later kernels in the defined order. Do not invent benchmark numbers, do not assume a sparsity pattern that has not been verified against the target weights/mask, and do not add features or dependencies outside the scope. Keep raw data reproducible and ensure any demo only visualizes the required measured metrics. After each increment, run relevant tests, show exact commands/results, and list known blockers instead of pretending they passed.

## 10. Decisions that must be confirmed before final performance claims

- Actual NVIDIA GPU and supported CUDA/library versions.
- Actual target layer/model and access to its weights or mask.
- Whether the observed pattern is CSR-like, block-structured, or hardware-supported 2:4.
- The target dtypes and layer input shapes to benchmark.

These are workload inputs, not permission to expand the project's scope.
