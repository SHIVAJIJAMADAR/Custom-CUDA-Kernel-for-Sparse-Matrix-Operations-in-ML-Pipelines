"""BERT-Base Workload Extraction, Pruning, and Sparsity Characterization.

Authoritative Reference: Project 10, pages 21–22, CUDA_GPU_Projects_1-10_Getting_Started.pdf
Model: google-bert/bert-base-uncased (Apache-2.0)
Pinned Revision: 86b5e0934494bd15c9632b12f734a8a67f723594
Layer: bert.encoder.layer.0.intermediate.dense.weight
Expected Shape: [3072, 768], torch.float32

This script:
1. Downloads/extracts the exact weight tensor from model.safetensors into local data/ (gitignored).
2. Analyzes original pretrained matrix characteristics (native sparsity, block occupancy, row statistics).
3. Deterministically generates experimental sparsity masks using global magnitude pruning
   for the official 50%–99% sweep (50%, 60%, 70%, 80%, 85%, 90%, 95%, 98%, 99%).
4. Characterizes each masked matrix (CSR row-length distribution, block occupancy at 8, 16, 32, 64).
5. Exports structured machine-readable metadata with full provenance to results/raw/bert_workload_metadata.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import torch

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from formats.csr import dense_to_csr, csr_to_dense, row_length_stats
from patterns.analyze_sparsity import analyze_sparsity

# Constants & Provenance
MODEL_ID = "google-bert/bert-base-uncased"
PINNED_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
LICENSE = "Apache-2.0"
TARGET_TENSOR_KEY = "bert.encoder.layer.0.intermediate.dense.weight"
EXPECTED_SHAPE = (3072, 768)
OFFICIAL_SPARSITY_SWEEP = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]


def obtain_bert_weight_tensor(
    cache_dir: Path | str = "data/cache",
    output_weight_path: Path | str = "data/bert_intermediate_weight.pt",
) -> torch.Tensor:
    """Download checkpoint if needed and extract target weight tensor."""
    import safetensors.torch
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_weight_path = Path(output_weight_path).resolve()
    output_weight_path.parent.mkdir(parents=True, exist_ok=True)

    # If already extracted locally, load directly
    if output_weight_path.exists():
        print(f"[Provenance] Loading previously extracted tensor from {output_weight_path}")
        tensor = torch.load(output_weight_path, map_location="cpu", weights_only=True)
        return tensor

    print(f"[Download] Fetching model.safetensors from {MODEL_ID} (rev: {PINNED_REVISION[:8]})...")
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=PINNED_REVISION,
        cache_dir=str(cache_dir),
    )

    print(f"[Extraction] Extracting '{TARGET_TENSOR_KEY}' from {model_path}...")
    with safetensors.safe_open(model_path, framework="pt") as f:
        available_keys = f.keys()
        if TARGET_TENSOR_KEY not in available_keys:
            raise KeyError(f"Key '{TARGET_TENSOR_KEY}' not found in safetensors checkpoint!")
        tensor = f.get_tensor(TARGET_TENSOR_KEY)

    # Validate tensor properties
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Expected shape {EXPECTED_SHAPE}, got {tuple(tensor.shape)}")
    if not tensor.isfinite().all():
        raise ValueError("Extracted tensor contains NaN or Inf values!")

    # Save to local gitignored cache
    torch.save(tensor, output_weight_path)
    print(f"[Storage] Saved target weight tensor to {output_weight_path} ({tensor.numel() * 4 / 1024**2:.2f} MB)")
    return tensor


def generate_magnitude_mask(weight: torch.Tensor, target_sparsity: float) -> tuple[torch.Tensor, float, int]:
    """Generate deterministic global magnitude pruning mask.

    Retention policy:
      Retain exact top-k elements with largest absolute magnitude |A_ij|, where:
        k = round(total_elements * (1.0 - target_sparsity))
      Ties at the retention boundary (|A_i| == |A_j|) are broken deterministically
      by prioritizing elements with the smaller flattened row-major tensor index.
      This is implemented via a stable sort (torch.sort with stable=True).

    Args:
      weight: 2D dense weight tensor.
      target_sparsity: Desired sparsity fraction in [0.0, 1.0).

    Returns:
      (binary_mask [M, K] (bool), actual_threshold (float), retained_count (int))
    """
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"Target sparsity must be in [0.0, 1.0), got {target_sparsity}")

    total_elements = weight.numel()
    k_retain = int(round(total_elements * (1.0 - target_sparsity)))

    if k_retain == 0:
        return torch.zeros_like(weight, dtype=torch.bool), float("inf"), 0
    if k_retain >= total_elements:
        return torch.ones_like(weight, dtype=torch.bool), 0.0, total_elements

    flat_abs = torch.abs(weight.flatten())
    # stable=True guarantees that elements with identical magnitude preserve
    # their original relative order (i.e. smaller flattened row-major index first).
    sorted_vals, sorted_indices = torch.sort(flat_abs, descending=True, stable=True)
    threshold = float(sorted_vals[k_retain - 1].item())

    # Build mask deterministically using exact top-k indices
    topk_indices = sorted_indices[:k_retain]
    flat_mask = torch.zeros(total_elements, dtype=torch.bool)
    flat_mask[topk_indices] = True
    mask = flat_mask.view_as(weight)

    return mask, threshold, k_retain


def characterize_workload_and_masks(
    weight: torch.Tensor,
    output_dir: Path | str = "data/masks",
    metadata_output_path: Path | str = "results/raw/bert_workload_metadata.json",
) -> Dict[str, Any]:
    """Analyze original tensor and generate experimental masks across official sweep."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_path = Path(metadata_output_path).resolve()
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    M, K = weight.shape
    total_elements = weight.numel()
    memory_footprint_bytes = total_elements * weight.element_size()

    # 1. Inspect original matrix before pruning
    exact_zeros = int((weight == 0).sum().item())
    native_sparsity = exact_zeros / total_elements
    original_analysis = analyze_sparsity(weight, label="bert_layer0_intermediate_original")

    # 2. Sweep official sparsity levels
    sweep_results = []
    for sp_target in OFFICIAL_SPARSITY_SWEEP:
        mask, thresh, nnz = generate_magnitude_mask(weight, sp_target)
        actual_sparsity = 1.0 - (nnz / total_elements)
        
        # Save mask tensor locally (gitignored)
        mask_path = output_dir / f"mask_sp{int(sp_target * 100):02d}.pt"
        torch.save(mask, mask_path)

        # Apply mask
        masked_weight = weight * mask

        # Run sparsity and block analysis
        masked_analysis = analyze_sparsity(
            masked_weight,
            label=f"bert_layer0_intermediate_sp{int(sp_target * 100):02d}",
        )

        sweep_results.append({
            "target_sparsity": sp_target,
            "actual_sparsity": round(actual_sparsity, 6),
            "retained_nonzero_count": nnz,
            "pruned_zero_count": total_elements - nnz,
            "magnitude_threshold": thresh,
            "mask_file": str(mask_path.relative_to(PROJECT_ROOT)),
            "row_length_stats": masked_analysis["row_length_stats"],
            "row_length_cv": masked_analysis["row_length_cv"],
            "block_analysis": masked_analysis["block_analysis"],
            "pattern_classification": masked_analysis["pattern_classification"],
            "suggested_format": masked_analysis["suggested_format"],
        })

    # 3. Assemble full metadata provenance ledger
    metadata = {
        "metadata_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": MODEL_ID,
            "pinned_revision": PINNED_REVISION,
            "license": LICENSE,
            "framework": "PyTorch / safetensors",
        },
        "target_layer": {
            "tensor_key": TARGET_TENSOR_KEY,
            "shape": {"M": M, "K": K},
            "dtype": str(weight.dtype),
            "element_count": total_elements,
            "memory_footprint_bytes": memory_footprint_bytes,
            "memory_footprint_mb": round(memory_footprint_bytes / (1024 * 1024), 2),
            "operation": "Y[3072, N] = A[3072, 768] @ X[768, N]",
            "mathematical_orientation": {
                "A_sparse": [3072, 768],
                "X_dense": [768, "N"],
                "Y_out": [3072, "N"],
            },
        },
        "original_unpruned_matrix": {
            "exact_zero_count": exact_zeros,
            "native_sparsity_percent": round(native_sparsity * 100, 4),
            "is_natively_sparse": exact_zeros > 0,
            "min_val": float(weight.min().item()),
            "max_val": float(weight.max().item()),
            "mean_val": float(weight.mean().item()),
            "std_val": float(weight.std().item()),
            "row_length_stats": original_analysis["row_length_stats"],
            "block_analysis": original_analysis["block_analysis"],
        },
        "pruning_methodology": {
            "method": "deterministic_global_magnitude_pruning",
            "retention_policy": "Retain exact top-k elements with largest absolute magnitude |A_ij|, breaking ties by smaller flattened row-major index via stable sort",
            "tie_breaking": "Deterministic by smaller flattened row-major index via torch.sort(stable=True)",
            "k_retention_formula": "k = round(total_elements * (1.0 - target_sparsity))",
            "official_sweep_levels": [f"{int(s * 100)}%" for s in OFFICIAL_SPARSITY_SWEEP],
        },
        "experimental_sparsity_sweep": sweep_results,
    }

    # Save metadata JSON (committed to git)
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Ledger] Saved complete workload metadata to {metadata_output_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="BERT Workload Acquisition & Sparsity Characterization")
    parser.add_argument("--cache-dir", default="data/cache", help="Hugging Face download cache directory")
    parser.add_argument("--weight-out", default="data/bert_intermediate_weight.pt", help="Extracted weight output path")
    parser.add_argument("--masks-out", default="data/masks", help="Directory to store generated mask tensors")
    parser.add_argument("--metadata-out", default="results/raw/bert_workload_metadata.json", help="Metadata output JSON")
    args = parser.parse_args()

    weight = obtain_bert_weight_tensor(args.cache_dir, args.weight_out)
    metadata = characterize_workload_and_masks(weight, args.masks_out, args.metadata_out)
    print("\n--- Workload Characterization Complete ---")
    print(f"Layer: {metadata['target_layer']['tensor_key']}")
    print(f"Shape: {metadata['target_layer']['shape']['M']} x {metadata['target_layer']['shape']['K']}")
    print(f"Native Sparsity: {metadata['original_unpruned_matrix']['native_sparsity_percent']}%")
    print(f"Sweep Levels Generated: {len(metadata['experimental_sparsity_sweep'])} masks (50% to 99%)")


if __name__ == "__main__":
    main()
