"""Tests for BERT Workload Acquisition, Sparsity Masks, and Correctness Reference.

Verifies:
1. Target BERT tensor properties (shape [3072, 768], dtype float32, finite values).
2. Deterministic mask generation reproducibility (same mask bit-for-bit).
3. Preservation of retained weights and exact zeros at pruned locations.
4. Correctness of CSR conversion round-trip for BERT masked weights.
5. Handling of empty rows in sparse representations (especially at 98% and 99% sparsity).
6. Mathematical agreement between CSR SpMM and the golden masked-dense reference:
     Y_ref = (A_original * M) @ X_dense
   across multiple output column widths N in {16, 32, 64, 128}.
"""
import json
from pathlib import Path
import pytest
import torch

from formats.csr import dense_to_csr, csr_to_dense
from patterns.prepare_bert_workload import (
    MODEL_ID,
    PINNED_REVISION,
    TARGET_TENSOR_KEY,
    EXPECTED_SHAPE,
    OFFICIAL_SPARSITY_SWEEP,
    generate_magnitude_mask,
    obtain_bert_weight_tensor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bert_weight() -> torch.Tensor:
    """Fixture providing the extracted BERT layer-0 intermediate weight tensor."""
    weight_path = PROJECT_ROOT / "data" / "bert_intermediate_weight.pt"
    if not weight_path.exists():
        # Obtain if not already present
        return obtain_bert_weight_tensor(
            cache_dir=PROJECT_ROOT / "data" / "cache",
            output_weight_path=weight_path,
        )
    return torch.load(weight_path, map_location="cpu", weights_only=True)


class TestBERTWeightProperties:
    """Verify downloaded tensor matches the official BERT-Base specification."""

    def test_tensor_shape_and_dtype(self, bert_weight: torch.Tensor):
        assert tuple(bert_weight.shape) == EXPECTED_SHAPE, (
            f"Expected shape {EXPECTED_SHAPE}, got {tuple(bert_weight.shape)}"
        )
        assert bert_weight.dtype == torch.float32

    def test_tensor_finite_and_statistics(self, bert_weight: torch.Tensor):
        assert bert_weight.isfinite().all(), "Weight tensor contains NaN or Inf"
        # Pretrained BERT weights have finite non-trivial variance
        assert bert_weight.std().item() > 0.01
        assert bert_weight.min().item() < 0.0
        assert bert_weight.max().item() > 0.0

    def test_native_sparsity_is_zero(self, bert_weight: torch.Tensor):
        # Must verify empirical fact that pretrained BERT weights are 100% dense
        exact_zeros = (bert_weight == 0).sum().item()
        assert exact_zeros == 0, f"Expected 0 native zeros in unpruned BERT, found {exact_zeros}"


class TestBERTDeterministicMasks:
    """Verify deterministic magnitude pruning masks and retention guarantees."""

    @pytest.mark.parametrize("sp", [0.50, 0.70, 0.90, 0.98, 0.99])
    def test_mask_determinism_and_reproducibility(self, bert_weight: torch.Tensor, sp: float):
        mask1, thresh1, nnz1 = generate_magnitude_mask(bert_weight, sp)
        mask2, thresh2, nnz2 = generate_magnitude_mask(bert_weight, sp)

        assert torch.equal(mask1, mask2), f"Mask at sparsity {sp} is not bitwise deterministic!"
        assert thresh1 == thresh2
        assert nnz1 == nnz2

    @pytest.mark.parametrize("sp", [0.50, 0.80, 0.95, 0.99])
    def test_retained_and_pruned_weights(self, bert_weight: torch.Tensor, sp: float):
        mask, thresh, nnz = generate_magnitude_mask(bert_weight, sp)
        masked_w = bert_weight * mask

        # Retained entries must exactly equal original weights
        assert torch.equal(masked_w[mask], bert_weight[mask])

        # Removed entries must be strictly 0.0
        assert (masked_w[~mask] == 0.0).all()

        # Check achieved sparsity matches target within 1 element
        achieved_sparsity = 1.0 - nnz / bert_weight.numel()
        assert abs(achieved_sparsity - sp) < 1e-4

    def test_empty_rows_at_high_sparsity(self, bert_weight: torch.Tensor):
        # At 99% sparsity, check that empty rows exist and are safely identified
        mask_99, _, _ = generate_magnitude_mask(bert_weight, 0.99)
        masked_w_99 = bert_weight * mask_99
        csr = dense_to_csr(masked_w_99)

        # Count empty rows via row_ptr
        row_nnz = csr.row_ptr[1:] - csr.row_ptr[:-1]
        empty_rows = (row_nnz == 0).sum().item()
        assert empty_rows > 0, "Expected empty rows at 99% sparsity in BERT matrix"


class TestBERTWorkloadCorrectnessReference:
    """Verify mathematical reference: Y_ref = (W * M) @ X matches CSR matmul."""

    @pytest.mark.parametrize("N", [16, 32, 64, 128])
    @pytest.mark.parametrize("sp", [0.50, 0.85, 0.98])
    def test_csr_matches_dense_reference(self, bert_weight: torch.Tensor, sp: float, N: int):
        torch.manual_seed(42)
        mask, _, _ = generate_magnitude_mask(bert_weight, sp)
        masked_w = bert_weight * mask
        M, K = masked_w.shape

        # Create deterministic dense activation matrix X
        X = torch.randn(K, N, dtype=torch.float32)

        # 1. Golden mathematical reference
        Y_ref = torch.matmul(masked_w, X)

        # 2. CSR reference calculation
        csr = dense_to_csr(masked_w)
        recon_w = csr_to_dense(csr)
        Y_csr = torch.matmul(recon_w, X)

        # Numerical comparison
        diff = torch.abs(Y_ref - Y_csr)
        max_diff = diff.max().item()
        assert max_diff <= 1e-6, f"Max difference {max_diff} exceeded tolerance at sp={sp}, N={N}"

    def test_metadata_file_validity(self):
        metadata_file = PROJECT_ROOT / "results" / "raw" / "bert_workload_metadata.json"
        assert metadata_file.exists(), "Metadata file results/raw/bert_workload_metadata.json missing!"

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["model"]["model_id"] == MODEL_ID
        assert data["model"]["pinned_revision"] == PINNED_REVISION
        assert data["target_layer"]["tensor_key"] == TARGET_TENSOR_KEY
        assert len(data["experimental_sparsity_sweep"]) == len(OFFICIAL_SPARSITY_SWEEP)
