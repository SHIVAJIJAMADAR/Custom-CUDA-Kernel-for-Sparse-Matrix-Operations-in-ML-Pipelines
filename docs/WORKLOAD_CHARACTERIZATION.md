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

## 4. Controlled Experimental Mask Generation & Metric Definitions

To evaluate custom kernels across the Project 10 official sweep ($50\%$ to $99\%$), **deterministic global magnitude pruning** is employed as the standard baseline pruning protocol:

1. **Retention Policy:** Retain the exact top-$k$ elements with the largest absolute magnitudes $|A_{ij}|$, where $k = \text{round}(N_{\text{total}} \times (1 - s))$ for target sparsity $s \in [0.50, 0.99]$.
2. **Threshold Computation:** The magnitude threshold $\tau$ is the value of the $k$-th retained element ($|A_{(k)}|$).
3. **Deterministic Tie-Breaking:** If multiple elements share identical magnitude values ($|A_i| == |A_j|$), ties are broken deterministically by prioritizing the element with the smaller flattened row-major tensor index. This is implemented via PyTorch's stable sorting (`torch.sort(flat_abs, descending=True, stable=True)`).
4. **Storage:** Masks are stored in `data/masks/mask_sp{pct}.pt` (local, gitignored). Workload metadata and thresholds are committed to `results/raw/bert_workload_metadata.json` (tracked in Git).

### Formal Definition of Block Sparsity
In this project, **Block Sparsity** for a given block dimension $B \times B$ is defined strictly as the percentage of non-overlapping blocks in the matrix grid that contain **exactly zero** nonzero entries:

$$\text{Block Sparsity} = \frac{\text{Count of Blocks with } NNZ = 0}{\text{Total Number of } (B \times B) \text{ Blocks}} \times 100\%$$

Conversely, a block is considered **active / non-empty** if it contains at least one nonzero entry ($NNZ \ge 1$). **Block Density** is defined as $1.0 - (\text{Block Sparsity} / 100\%)$.

---

## 5. Measured Workload Statistics Across Official Sweep (50%–99%)

The table below presents empirical statistics measured directly on the BERT intermediate projection ($3072 \times 768$, $2,359,296$ total elements), fully synchronized with `results/raw/bert_workload_metadata.json`:

| Target Sparsity | Actual Nonzeros ($NNZ$) | Magnitude Threshold ($\tau$) | Row Nonzeros (Min / Max) | Row Nonzeros (Mean ± Std) | Empty Rows | Row CV | Block Sparsity ($8\times 8$) | Block Sparsity ($32\times 32$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Original)**| 2,359,296 | 0.000000 | 768 / 768 | 768.0 ± 0.0 | 0 | 0.0000 | 0.00% | 0.00% |
| **50%** | 1,179,648 | 0.024712 | 89 / 576 | 384.0 ± 44.5 | 0 | 0.1158 | 0.00% | 0.00% |
| **60%** | 943,718 | 0.030904 | 47 / 538 | 307.2 ± 46.5 | 0 | 0.1514 | 0.00% | 0.00% |
| **70%** | 707,789 | 0.038196 | 30 / 492 | 230.4 ± 45.3 | 0 | 0.1965 | 0.00% | 0.00% |
| **80%** | 471,859 | 0.047479 | 7 / 433 | 153.6 ± 39.3 | 0 | 0.2556 | 0.00% | 0.00% |
| **85%** | 353,894 | 0.053516 | 4 / 382 | 115.2 ± 34.2 | 0 | 0.2970 | 0.01% | 0.00% |
| **90%** | 235,930 | 0.061454 | 2 / 338 | 76.8 ± 27.5 | 0 | 0.3581 | 0.15% | 0.00% |
| **95%** | 117,965 | 0.073820 | 0 / 276 | 38.4 ± 18.1 | 5 | 0.4709 | 4.20% | 0.00% |
| **98%** | 47,186 | 0.088522 | 0 / 226 | 15.4 ± 10.4 | 57 | 0.6784 | 28.46% | 0.00% |
| **99%** | 23,593 | 0.098763 | 0 / 180 | 7.7 ± 7.1 | 172 | 0.9282 | 53.31% | 0.00% |

---

## 6. Architectural Insights & Format Selection Rationale

### Why CSR is the Mathematically Optimal Format for this Workload
1. **Unstructured Distribution:** Under global magnitude pruning, nonzeros are distributed irregularly across the matrix. At 90% element sparsity, 99.85% of $8\times 8$ blocks and 100% of $32\times 32$ blocks contain at least one nonzero element.
2. **Corrected Block-Sparsity Analysis on $32\times 32$ Blocks:**
   For a $3072 \times 768$ matrix, there are $96 \times 24 = 2,304$ non-overlapping blocks of size $32 \times 32$ ($1,024$ elements per block):
   - At **90% element sparsity**, $NNZ = 235,930$. Every single one of the $2,304$ blocks is non-empty (0.0% block sparsity). The mean fill per block is $\frac{235,930}{2,304} = 102.4$ elements per block, representing an in-block fill rate of $\frac{102.4}{1,024} = 10.0\%$. Storing full $32 \times 32$ dense blocks in BSR would require storing all $2,304$ blocks ($2,359,296$ values), resulting in **0.0% storage savings** despite 90% element sparsity.
   - At **99% element sparsity**, $NNZ = 23,593$. Every single one of the $2,304$ blocks still contains at least one nonzero (0.0% block sparsity). The mean fill per block is $\frac{23,593}{2,304} = 10.24 \approx 10.2$ elements per block, representing an in-block fill rate of $\frac{10.24}{1,024} \approx 1.00\%$. Storing these blocks in standard BSR would incur a $\sim 100\times$ storage overhead compared to the retained nonzeros.
   - **Conclusion:** Standard Block Compressed Sparse Row (BSR) formats are fundamentally incompatible with unstructured magnitude pruning. Compressed Sparse Row (CSR) is the mathematically mandated format because it stores only actual nonzero entries and their indices without block-padding overhead.
3. **Row Length Variance & Empty Rows:** At 98% and 99% sparsity, row lengths range from 0 to 226, producing empty rows (57 at 98%, 172 at 99%) and notable row skew ($\text{CV} \to 0.93$). This validates:
   - **Kernel V1 (CSR Row-per-Warp):** Essential for handling irregular row lengths without padding waste.
   - **Kernel V3 (Load-Balanced CSR):** Strongly motivated at $\ge 95\%$ sparsity to eliminate warp divergence caused by variance between empty rows and dense rows (up to 180 nonzeros).

---

## 7. Workload Correctness Verification & Test Suite Architecture

### 7.1 Golden Mathematical Reference
Mathematical reference validation is executed on CPU across the full sweep:
$$Y_{\text{ref}} = (A_{\text{original}} \odot M) \times X_{\text{dense}}$$

- **Numerical Agreement:** The dense reference $Y_{\text{ref}}$ matches CSR reconstruction dot products $Y_{\text{csr}}$ with maximum absolute error $\le 1.0 \times 10^{-6}$ across all tested dimensions ($N \in \{16, 32, 64, 128\}$) and sparsity levels.
- **Boundary Handling:** Tested and confirmed that empty rows output exact zero vectors without memory access faults.
- **Reproducibility:** Tested and confirmed that mask generation is bitwise identical across multiple runs via stable sorting.

### 7.2 Separation of Test Paths & Dependency Architecture
To prevent unit tests from triggering unexpected external network downloads:
1. **Lightweight Unit Tests (`TestMagnitudePruningUnit`):**
   - Execute instantly on synthetic, deterministic tensors with deliberate boundary ties.
   - Run via: `pytest tests/ -m "not workload" -v`
   - Validates tie-breaking, signed sorting, exact top-$k$ counts, and edge cases in $< 3$ seconds.
2. **Workload Integration Tests (`@pytest.mark.workload`):**
   - Require the local extracted tensor `data/bert_intermediate_weight.pt`.
   - If the local tensor is absent, tests skip gracefully with clear instructions instead of initiating a 420 MB network download.
   - Run via: `pytest tests/ -v`
3. **Explicit Workload Acquisition:**
   - To acquire the model checkpoint and extract the target layer:
     ```bash
     python patterns/prepare_bert_workload.py
     ```
   - Requires optional dependencies declared in `requirements-workload.txt`:
     ```bash
     pip install -r requirements-workload.txt
     ```
     (`huggingface_hub>=0.20.0` and `safetensors>=0.4.0`).

---

## 8. Unresolved Questions & Next Steps

1. **Downstream Task Accuracy Retention:** Global magnitude pruning produces controlled sparsity for kernel benchmarking. Measuring actual GLUE benchmark accuracy retention if the pruned weights were evaluated end-to-end remains a separate evaluation task.
2. **Fine-Tuning / Sparse Retraining:** Wanda or SparseGPT masks could be explored as advanced candidates if post-pruning fine-tuning is investigated.
3. **Toolchain Resolution:** Physical GPU compilation of Kernel V1 remains gated on the installation of the MSVC compiler and CUDA Toolkit.
