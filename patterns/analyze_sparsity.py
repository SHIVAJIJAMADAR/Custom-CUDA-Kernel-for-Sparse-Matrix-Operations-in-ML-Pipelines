"""Sparsity analysis tool for target weight matrices (R1).

Given a weight matrix (dense or with a mask), this module computes:
  - Matrix dimensions (M, K)
  - Dtype
  - Total element count
  - Nonzero count and sparsity percentage
  - Block occupancy analysis (for block_size × block_size tiles)
  - Row-length distribution statistics (min, max, mean, median, std, skew)
  - Whether the pattern appears structured, random, or block-like
  - Suggested sparse format based on the observed pattern

Usage:
    python patterns/analyze_sparsity.py --weights path/to/weights.pt [--layer-key weight]
    python patterns/analyze_sparsity.py --random M K --sparsity 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formats.csr import dense_to_csr, row_length_stats


def analyze_sparsity(
    matrix: torch.Tensor,
    *,
    block_sizes: tuple[int, ...] = (8, 16, 32, 64),
    label: str = "unknown",
) -> dict:
    """Analyze sparsity characteristics of a 2-D weight matrix.

    Args:
        matrix: [M, K] dense tensor (zeros represent sparsity).
        block_sizes: Block sizes to evaluate for block-occupancy analysis.
        label: Human-readable label for the matrix (e.g., layer name).

    Returns:
        Dict with all analysis results.
    """
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.dim()}-D")

    M, K = matrix.shape
    total = M * K
    nonzero_mask = matrix != 0
    nnz = int(nonzero_mask.sum().item())
    sparsity = 1.0 - nnz / total if total > 0 else 0.0

    # ---- Row-length analysis (CSR perspective) ----
    csr = dense_to_csr(matrix)
    rl_stats = row_length_stats(csr)

    # Compute coefficient of variation for row-length skew
    if rl_stats["mean"] > 0:
        cv = rl_stats["std"] / rl_stats["mean"]
    else:
        cv = 0.0

    # ---- Block occupancy analysis ----
    block_analysis = {}
    for bs in block_sizes:
        if M < bs or K < bs:
            continue
        # Trim to block-aligned dimensions
        M_aligned = (M // bs) * bs
        K_aligned = (K // bs) * bs
        trimmed = nonzero_mask[:M_aligned, :K_aligned]

        # Reshape into blocks and check if each block has any nonzero
        row_blocks = M_aligned // bs
        col_blocks = K_aligned // bs
        blocks = trimmed.view(row_blocks, bs, col_blocks, bs)
        blocks = blocks.permute(0, 2, 1, 3).contiguous()  # [rb, cb, bs, bs]

        # Block-level statistics
        block_nnz = blocks.float().sum(dim=(-1, -2))  # [rb, cb]
        total_blocks = row_blocks * col_blocks
        nonempty_blocks = int((block_nnz > 0).sum().item())
        block_density = nonempty_blocks / total_blocks if total_blocks > 0 else 0.0

        # Elements per nonempty block
        if nonempty_blocks > 0:
            nonempty_nnz = block_nnz[block_nnz > 0]
            mean_fill = float(nonempty_nnz.mean().item())
            max_fill = int(nonempty_nnz.max().item())
        else:
            mean_fill = 0.0
            max_fill = 0

        block_analysis[bs] = {
            "block_size": bs,
            "row_blocks": row_blocks,
            "col_blocks": col_blocks,
            "total_blocks": total_blocks,
            "nonempty_blocks": nonempty_blocks,
            "block_density": round(block_density, 4),
            "block_sparsity": round(1.0 - block_density, 4),
            "mean_fill_per_nonempty_block": round(mean_fill, 1),
            "max_fill_per_block": max_fill,
            "max_possible_fill": bs * bs,
        }

    # ---- Pattern classification heuristic ----
    pattern_type = _classify_pattern(sparsity, block_analysis, cv)

    # ---- Format suggestion ----
    suggested_format = _suggest_format(sparsity, block_analysis, cv, pattern_type)

    return {
        "label": label,
        "shape": {"M": M, "K": K},
        "dtype": str(matrix.dtype),
        "total_elements": total,
        "nonzero_count": nnz,
        "sparsity": round(sparsity, 6),
        "density": round(1.0 - sparsity, 6),
        "row_length_stats": rl_stats,
        "row_length_cv": round(cv, 4),
        "block_analysis": block_analysis,
        "pattern_classification": pattern_type,
        "suggested_format": suggested_format,
    }


def _classify_pattern(
    sparsity: float,
    block_analysis: dict,
    cv: float,
) -> str:
    """Heuristic pattern classification based on measured data."""
    if sparsity < 0.1:
        return "dense (sparsity < 10%)"
    if sparsity < 0.3:
        return "mildly sparse"

    # Check for block structure: if block_sparsity >> element_sparsity,
    # the pattern has significant block structure.
    for bs_key in sorted(block_analysis.keys(), reverse=True):
        ba = block_analysis[bs_key]
        if ba["block_sparsity"] > sparsity + 0.15:
            return f"block-structured (detected at {bs_key}×{bs_key})"

    if cv > 1.5:
        return "skewed/unbalanced rows"
    if sparsity > 0.9:
        return "highly sparse / unstructured"
    return "moderately sparse / unstructured"


def _suggest_format(
    sparsity: float,
    block_analysis: dict,
    cv: float,
    pattern_type: str,
) -> str:
    """Suggest a sparse format based on the observed pattern."""
    if sparsity < 0.3:
        return "dense (cuBLAS) — sparsity too low for sparse kernels to benefit"

    if "block-structured" in pattern_type:
        return "BSR or blocked format — significant block structure detected"

    if cv > 1.5:
        return "CSR with load balancing — high row-length variance detected"

    if sparsity > 0.5:
        return "CSR — general-purpose sparse format suitable for unstructured sparsity"

    return "CSR — default choice for moderate sparsity"


def main():
    p = argparse.ArgumentParser(description="Analyze sparsity of a weight matrix")
    p.add_argument("--weights", type=str, help="Path to a .pt/.pth file containing a weight tensor")
    p.add_argument("--layer-key", type=str, default=None,
                   help="Key in state_dict to extract (e.g., 'fc.weight')")
    p.add_argument("--random", nargs=2, type=int, metavar=("M", "K"),
                   help="Generate a random matrix of size M×K for testing")
    p.add_argument("--sparsity", type=float, default=0.75,
                   help="Sparsity level for random matrix (default 0.75)")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save JSON results")
    args = p.parse_args()

    if args.random:
        M, K = args.random
        torch.manual_seed(42)
        w = torch.randn(M, K)
        mask = (torch.rand(M, K) > args.sparsity).float()
        matrix = w * mask
        label = f"random_{M}x{K}_sparsity{args.sparsity}"
    elif args.weights:
        data = torch.load(args.weights, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            if args.layer_key:
                matrix = data[args.layer_key]
            else:
                # List available keys
                print(f"Available keys in state_dict: {list(data.keys())}")
                # Try to find a weight tensor
                candidates = [k for k in data.keys() if "weight" in k.lower()]
                if candidates:
                    key = candidates[0]
                    print(f"Auto-selected: {key}")
                    matrix = data[key]
                else:
                    print("No 'weight' key found. Specify --layer-key.")
                    sys.exit(1)
        else:
            matrix = data
        label = args.layer_key or Path(args.weights).stem
    else:
        p.print_help()
        sys.exit(1)

    if matrix.dim() != 2:
        print(f"Warning: tensor has {matrix.dim()} dimensions, reshaping to 2-D")
        matrix = matrix.view(matrix.shape[0], -1)

    results = analyze_sparsity(matrix, label=label)

    # Pretty-print
    print(json.dumps(results, indent=2, default=str))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
