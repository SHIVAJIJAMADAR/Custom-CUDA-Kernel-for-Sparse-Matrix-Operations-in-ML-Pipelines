# Workload Characterization: BERT-Base Encoder Intermediate Projection

**Project:** Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines  
**Authoritative Reference:** Project 10, pages 21–22, `CUDA_GPU_Projects_1-10_Getting_Started.pdf`  
**Document Status:** Complete & Empirically Verified (Phase 2A)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (Ampere `sm_86`, 4 GB GDDR6)  

---

## 1. Workload Identity & Provenance

The primary real model workload selected and investigated for Project 10 is the intermediate dense Feed-Forward projection layer from **BERT-Base Uncased**:

- **Model Identifier:** `google-bert/bert-base-uncased`
- **Pinned Checkpoint Revision (Git SHA):** `86b5e0934494bd15c9632b12f734a8a67f723594`
- **Model License:** Apache-2.0
- **Source Checkpoint File:** `model.safetensors` (440,449,768 bytes, 420.04 MB)
- **Target Tensor Key:** `bert.encoder.layer.0.intermediate.dense.weight`
- **Local Extraction Location:** `data/bert_intermediate_weight.pt` (local, gitignored)
- **Machine-Readable Provenance Ledger:** `results/raw/bert_workload_metadata.json` (tracked in Git)

---

## 2. Mathematical Orientation & Matrix Layout

In the BERT Transformer encoder architecture, the intermediate dense layer expands the hidden dimension ($d_{\text{model}} = 768$) to the feed-forward intermediate dimension ($d_{\text{ffn}} = 3072$), followed by a GeLU activation function.

### Layer Dimensions:
- Output feature dimension ($M$): $3072$
- Input feature dimension ($K$): $768$
- Total element count: $3072 \times 768 = 2,359,296$ elements
- Data type: `torch.float32` (4 bytes per element)
- In-memory footprint (dense): $9,437,184\text{ bytes} \approx 9.00\text{ MB}$

### Mathematical SpMM Formulation:
When executing forward inference on a batch of token activations $X$, the sparse matrix multiplication operates as:

$$Y[M, N] = A_{\text{sparse}}[M, K] \times X_{\text{dense}}[K, N]$$

Where:
- $A_{\text{sparse}} \in \mathbb{R}^{3072 \times 768}$ represents the pruned sparse weight matrix stored in CSR format (`values`, `col_indices`, `row_ptr`).
- $X_{\text{dense}} \in \mathbb{R}^{768 \times N}$ represents the dense input activation tensor for token batch size $N \in \{16, 32, 64, 128, 256, 512\}$.
- $Y[3072, N]$ represents the output pre-activation tensor.

*Equivalence Note:* In standard PyTorch `nn.Linear(768, 3072)`, input tokens are organized as $X_{\text{tokens}} \in \mathbb{R}^{B \times 768}$, computing $Y_{\text{tokens}} = X_{\text{tokens}} \times W^T$. Transposing to column-major token vectors yields the canonical SpMM orientation $Y = A \times X$.

---

## 3. Original Weight Matrix Inspection (Before Pruning)

The original extracted tensor was audited prior to applying any sparsity masks.

### Measured Empirical Properties:
- **Exact-Zero Count:** $0$ out of $2,359,296$ elements.
- **Native Sparsity:** **0.0000%** (100% dense).
- **Minimum Value:** $-0.297842$
- **Maximum Value:** $+0.322216$
- **Mean Value:** $-3.0439 \times 10^{-5}$
- **Standard Deviation:** $0.037422$
- **Row-Length Statistics (CSR):** Every row has exactly $768$ nonzeros ($\text{min} = 768, \text{max} = 768, \text{std} = 0.0, \text{CV} = 0.0$).
- **Native Block Occupancy:** 100% full across block sizes $8\times 8, 16\times 16, 32\times 32, 64\times 64$.

> [!IMPORTANT]
> Pretrained BERT weights possess **zero native sparsity**. To study sparse matrix acceleration, sparsity must be introduced via a principled pruning method.

---

## 4. Controlled Experimental Mask Generation

To evaluate custom kernels across the Project 10 official sweep ($50\%$ to $99\%$), **deterministic global magnitude pruning** is employed as the standard baseline pruning protocol:

1. **Retention Policy:** Retain the elements with the largest absolute magnitudes $|A_{ij}|$.
2. **Threshold Computation:** For target sparsity $s \in [0.50, 0.99]$, the retention count is $k = \text{round}(N_{\text{total}} \times (1 - s))$. The threshold $\tau$ is the $k$-th largest value of $|A|$.
3. **Deterministic Tie-Breaking:** If values share the boundary threshold $\tau$, ties are broken deterministically by flattened index order to ensure bitwise reproducibility.
4. **Storage:** Masks are stored in `data/masks/mask_sp{pct}.pt` (gitignored). Full metadata and thresholds are stored in `results/raw/bert_workload_metadata.json` (tracked in Git).

---

## 5. Measured Workload Statistics Across Official Sweep (50%–99%)

The table below presents empirical statistics measured directly on the BERT intermediate projection:

| Target Sparsity | Actual Nonzeros ($NNZ$) | Magnitude Threshold ($\tau$) | Row Nonzeros (Min / Max) | Row Nonzeros (Mean ± Std) | Empty Rows | Row CV | Block Sparsity ($8\times 8$) | Block Sparsity ($32\times 32$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Original)**| 2,359,296 | 0.000000 | 768 / 768 | 768.0 ± 0.0 | 0 | 0.0000 | 0.00% | 0.00% |
| **50%** | 1,179,648 | 0.024712 | 89 / 576 | 384.0 ± 44.5 | 0 | 0.1158 | 0.00% | 0.00% |
| **60%** | 943,718 | 0.030904 | 47 / 538 | 307.2 ± 46.5 | 0 | 0.1514 | 0.00% | 0.00% |
| **70%** | 707,789 | 0.038196 | 21 / 501 | 230.4 ± 46.7 | 0 | 0.2028 | 0.00% | 0.00% |
| **80%** | 471,859 | 0.047479 | 6 / 444 | 153.6 ± 43.6 | 0 | 0.2838 | 0.01% | 0.00% |
| **85%** | 353,894 | 0.053516 | 2 / 401 | 115.3 ± 40.2 | 0 | 0.3486 | 0.16% | 0.00% |
| **90%** | 235,930 | 0.061454 | 0 / 355 | 76.8 ± 34.6 | 2 | 0.4500 | 1.83% | 0.00% |
| **95%** | 117,965 | 0.073820 | 0 / 295 | 38.4 ± 24.8 | 13 | 0.6472 | 11.23% | 0.00% |
| **98%** | 47,186 | 0.088522 | 0 / 226 | 15.4 ± 10.4 | 57 | 0.6784 | 28.46% | 0.00% |
| **99%** | 23,593 | 0.098763 | 0 / 180 | 7.7 ± 7.1 | 172 | 0.9282 | 53.31% | 0.00% |

---

## 6. Architectural Insights & Format Selection Rationale

### Why CSR is the Optimal Format for this Workload
1. **Unstructured Distribution:** Under global magnitude pruning, nonzeros are distributed irregularly across the matrix. Even at 90% element sparsity, 98.17% of $8\times 8$ blocks and 100% of $32\times 32$ blocks contain at least one nonzero element.
2. **Failure of Block Formats on Unstructured Pruning:** A $32\times 32$ block format applied at 90% sparsity would store 100% of the blocks, with each block having an average fill of only $10.2$ elements out of $1024$ ($< 1\%$ fill rate). Storing such sparse blocks in BSR would cause a massive $> 90\times$ storage and compute overhead.
3. **Row Variance & Empty Rows:** At 98% and 99% sparsity, row lengths range from 0 to 226, producing empty rows (172 rows at 99%) and moderate row skew ($\text{CV} \to 0.93$). This validates:
   - **Kernel V1 (CSR Row-per-Warp):** Essential for handling irregular row lengths without padding waste.
   - **Kernel V3 (Load-Balanced CSR):** Strongly motivated at $\ge 95\%$ sparsity to eliminate warp divergence caused by variance between empty rows and dense rows (up to 180 nonzeros).

---

## 7. Workload Correctness Verification

Mathematical reference validation was executed on CPU across the full sweep:
$$Y_{\text{ref}} = (A_{\text{original}} \odot M) \times X_{\text{dense}}$$

- **Numerical Agreement:** The dense reference $Y_{\text{ref}}$ matches CSR reconstruction dot products $Y_{\text{csr}}$ with maximum absolute error $\le 1.0 \times 10^{-6}$ across all tested dimensions ($N \in \{16, 32, 64, 128\}$) and sparsity levels.
- **Boundary Handling:** Tested and confirmed that empty rows output exact zero vectors without memory access faults.
- **Reproducibility:** Tested and confirmed that mask generation is bitwise identical across multiple runs.

---

## 8. Unresolved Questions & Next Steps

1. **Downstream Task Accuracy Retention:** Global magnitude pruning produces controlled sparsity for kernel benchmarking. Measuring actual GLUE benchmark accuracy retention if the pruned weights were evaluated end-to-end remains a separate evaluation task.
2. **Fine-Tuning / Sparse Retraining:** Wanda or SparseGPT masks could be explored as advanced candidates if post-pruning fine-tuning is investigated.
3. **Toolchain Resolution:** Physical GPU compilation of Kernel V1 remains gated on the installation of the MSVC compiler and CUDA Toolkit.
