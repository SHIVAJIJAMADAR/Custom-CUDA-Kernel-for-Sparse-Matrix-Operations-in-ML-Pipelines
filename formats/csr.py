"""CSR (Compressed Sparse Row) format conversion utilities.

This module implements dense-to-CSR and CSR-to-dense conversions for use
in SpMM kernels and cuSPARSE baselines.

The CSR format stores:
  - values:    1-D array of nonzero values  [nnz]
  - col_idx:   1-D array of column indices  [nnz]
  - row_ptr:   1-D array of row pointers    [M+1]

For an M×K matrix with nnz nonzeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CSRMatrix:
    """CSR representation of a sparse matrix."""
    values: torch.Tensor     # [nnz] — nonzero values
    col_idx: torch.Tensor    # [nnz] — column indices (int32)
    row_ptr: torch.Tensor    # [M+1] — row pointers (int32)
    M: int                   # number of rows
    K: int                   # number of columns
    nnz: int                 # number of nonzeros

    @property
    def sparsity(self) -> float:
        """Fraction of elements that are zero."""
        total = self.M * self.K
        return 1.0 - self.nnz / total if total > 0 else 0.0

    @property
    def density(self) -> float:
        """Fraction of elements that are nonzero."""
        return 1.0 - self.sparsity


def dense_to_csr(dense: torch.Tensor, *, threshold: float = 0.0) -> CSRMatrix:
    """Convert a 2-D dense tensor to CSR format.

    Args:
        dense: A [M, K] tensor.
        threshold: Absolute value threshold below which elements are
                   treated as zero. Default 0.0 (exact zeros only).

    Returns:
        CSRMatrix with values, col_idx, row_ptr on the same device as input.
    """
    if dense.dim() != 2:
        raise ValueError(f"Expected 2-D tensor, got {dense.dim()}-D")

    M, K = dense.shape
    # Identify nonzero positions
    if threshold > 0.0:
        mask = dense.abs() > threshold
    else:
        mask = dense != 0

    # Build row_ptr
    nnz_per_row = mask.sum(dim=1)  # [M]
    row_ptr = torch.zeros(M + 1, dtype=torch.int32, device=dense.device)
    torch.cumsum(nnz_per_row, dim=0, out=row_ptr[1:])

    nnz = int(row_ptr[-1].item())

    # Build values and col_idx
    if nnz == 0:
        values = torch.empty(0, dtype=dense.dtype, device=dense.device)
        col_idx = torch.empty(0, dtype=torch.int32, device=dense.device)
    else:
        # Get nonzero indices
        nz_rows, nz_cols = torch.where(mask)
        # Sort by row (torch.where already returns row-major order)
        values = dense[nz_rows, nz_cols].contiguous()
        col_idx = nz_cols.to(torch.int32).contiguous()

    return CSRMatrix(
        values=values,
        col_idx=col_idx,
        row_ptr=row_ptr,
        M=M,
        K=K,
        nnz=nnz,
    )


def csr_to_dense(csr: CSRMatrix, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Reconstruct a dense [M, K] tensor from CSR representation.

    Args:
        csr: A CSRMatrix.
        dtype: Optional output dtype. Defaults to csr.values.dtype.

    Returns:
        Dense [M, K] tensor.
    """
    out_dtype = dtype or csr.values.dtype
    device = csr.values.device if csr.nnz > 0 else csr.row_ptr.device
    out = torch.zeros(csr.M, csr.K, dtype=out_dtype, device=device)

    if csr.nnz == 0:
        return out

    row_ptr = csr.row_ptr
    col_idx = csr.col_idx
    values = csr.values

    for i in range(csr.M):
        start = int(row_ptr[i].item())
        end = int(row_ptr[i + 1].item())
        if start < end:
            cols = col_idx[start:end].long()
            out[i, cols] = values[start:end].to(out_dtype)

    return out


def dense_masked_to_csr(
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> CSRMatrix:
    """Apply a binary mask to a weight matrix and convert to CSR.

    This is the canonical path for the correctness oracle:
    A_masked = weight * mask, then convert to CSR.

    Args:
        weight: [M, K] weight matrix.
        mask: [M, K] binary mask (0 or 1), same shape as weight.

    Returns:
        CSRMatrix of the masked weight matrix.
    """
    if weight.shape != mask.shape:
        raise ValueError(f"Shape mismatch: weight {weight.shape} vs mask {mask.shape}")
    masked = weight * mask
    return dense_to_csr(masked)


def row_lengths(csr: CSRMatrix) -> torch.Tensor:
    """Return the number of nonzeros per row as a 1-D int tensor [M]."""
    return (csr.row_ptr[1:] - csr.row_ptr[:-1]).to(torch.int32)


def row_length_stats(csr: CSRMatrix) -> dict:
    """Compute row-length distribution statistics.

    Returns a dict with min, max, mean, median, std, and a histogram of
    row lengths. Useful for deciding if load-balancing is needed (R7).
    """
    lengths = row_lengths(csr).float()
    if lengths.numel() == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": int(lengths.min().item()),
        "max": int(lengths.max().item()),
        "mean": float(lengths.mean().item()),
        "median": float(lengths.median().item()),
        "std": float(lengths.std().item()),
        "total_nnz": csr.nnz,
        "num_empty_rows": int((lengths == 0).sum().item()),
    }
