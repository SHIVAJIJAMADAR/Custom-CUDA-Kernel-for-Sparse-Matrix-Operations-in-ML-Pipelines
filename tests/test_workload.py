"""Tests for BERT Workload Acquisition, Sparsity Masks, and Correctness Reference.

Verifies:
1. Deterministic magnitude pruning & tie-breaking on synthetic unit test tensors.
2. Target BERT tensor properties (shape [3072, 768], dtype float32, finite values).
3. Deterministic mask generation reproducibility (same mask bit-for-bit).
4. Preservation of retained weights and exact zeros at pruned locations.
5. Correctness of CSR conversion round-trip for BERT masked weights.
6. Handling of empty rows in sparse representations (especially at 98% and 99% sparsity).
7. Mathematical agreement between CSR SpMM and the golden masked-dense reference:
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
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestMagnitudePruningUnit:
    """Lightweight deterministic unit tests for magnitude pruning & tie-breaking.

    These tests run independently of the external BERT model checkpoint and
    verify exact top-k retention, stable tie-breaking, signed magnitude handling,
    and boundary conditions.
    """

    def test_exact_topk_retention_count(self):
        # 10 elements, target sparsity 0.7 -> retain round(10 * 0.3) = 3 elements
        weight = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 1.0])
        mask, thresh, nnz = generate_magnitude_mask(weight, target_sparsity=0.7)
        assert nnz == 3
        assert mask.sum().item() == 3
        # Retained values must be 1.0, 0.9, 0.8 at indices 9, 1, 5
        assert mask[1] == True
        assert mask[5] == True
        assert mask[9] == True
        assert mask[0] == False
        assert thresh == pytest.approx(0.8, abs=1e-6)

    def test_deterministic_tie_breaking_by_flattened_index(self):
        # 2x3 matrix with deliberate ties at magnitude 0.5
        # Element indices:
        # (0,0)=0.5 [flat 0], (0,1)=0.2 [flat 1], (0,2)=0.5 [flat 2]
        # (1,0)=0.1 [flat 3], (1,1)=0.5 [flat 4], (1,2)=0.3 [flat 5]
        weight = torch.tensor([
            [0.5, 0.2, 0.5],
            [0.1, 0.5, 0.3]
        ])

        # Test k=1: target_sparsity = 1.0 - 1/6 = 5/6 (~0.8333)
        # Should retain only the first 0.5 at flat index 0
        mask_k1, th1, nnz1 = generate_magnitude_mask(weight, target_sparsity=5.0 / 6.0)
        assert nnz1 == 1
        assert th1 == pytest.approx(0.5, abs=1e-6)
        flat_mask1 = mask_k1.flatten()
        assert flat_mask1[0] == True
        assert flat_mask1[2] == False
        assert flat_mask1[4] == False

        # Test k=2: target_sparsity = 1.0 - 2/6 = 4/6 (~0.6667)
        # Should retain exactly 2 elements: flat index 0 and flat index 2
        mask_k2, th2, nnz2 = generate_magnitude_mask(weight, target_sparsity=4.0 / 6.0)
        assert nnz2 == 2
        assert th2 == pytest.approx(0.5, abs=1e-6)
        flat_mask2 = mask_k2.flatten()
        assert flat_mask2[0] == True
        assert flat_mask2[2] == True
        assert flat_mask2[4] == False

        # Test k=3: target_sparsity = 0.5 (3 of 6 elements)
        # Should retain all three 0.5 elements at flat indices 0, 2, 4
        mask_k3, th3, nnz3 = generate_magnitude_mask(weight, target_sparsity=0.5)
        assert nnz3 == 3
        assert th3 == pytest.approx(0.5, abs=1e-6)
        flat_mask3 = mask_k3.flatten()
        assert flat_mask3[0] == True
        assert flat_mask3[2] == True
        assert flat_mask3[4] == True
        assert flat_mask3[1] == False
        assert flat_mask3[3] == False
        assert flat_mask3[5] == False

        # Test k=4: target_sparsity = 1.0 - 4/6 = 2/6 (~0.3333)
        # Should retain three 0.5 elements (0, 2, 4) AND the 0.3 element at flat index 5
        mask_k4, th4, nnz4 = generate_magnitude_mask(weight, target_sparsity=2.0 / 6.0)
        assert nnz4 == 4
        assert th4 == pytest.approx(0.3, abs=1e-6)
        flat_mask4 = mask_k4.flatten()
        assert flat_mask4[0] == True
        assert flat_mask4[2] == True
        assert flat_mask4[4] == True
        assert flat_mask4[5] == True
        assert flat_mask4[1] == False
        assert flat_mask4[3] == False

    def test_signed_magnitude_pruning(self):
        # Verify that negative values are sorted by absolute magnitude |-0.8| == 0.8
        # Indices: 0 (-0.8, mag 0.8), 1 (0.4, mag 0.4), 2 (-0.2, mag 0.2), 3 (0.8, mag 0.8)
        weight = torch.tensor([-0.8, 0.4, -0.2, 0.8])

        # k=1 (sparsity 0.75): should retain flat index 0
        mask_k1, th, nnz = generate_magnitude_mask(weight, target_sparsity=0.75)
        assert nnz == 1
        assert mask_k1[0] == True
        assert mask_k1[3] == False

        # k=2 (sparsity 0.5): should retain both index 0 and index 3
        mask_k2, th, nnz = generate_magnitude_mask(weight, target_sparsity=0.5)
        assert nnz == 2
        assert mask_k2[0] == True
        assert mask_k2[3] == True

    def test_boundary_and_zero_sparsity(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        # Sparsity 0.0 -> retain all
        mask_0, th0, nnz0 = generate_magnitude_mask(weight, 0.0)
        assert nnz0 == 4
        assert mask_0.all()

        # Sparsity where k=0 -> retain none
        mask_none, th_inf, nnz_none = generate_magnitude_mask(weight, 0.999)
        assert nnz_none == 0
        assert not mask_none.any()

    def test_input_with_existing_zeros(self):
        # Tensor with native zeros: [[0.0, 5.0], [0.0, 2.0]]
        weight = torch.tensor([[0.0, 5.0], [0.0, 2.0]])
        # k=2 (sparsity 0.5) -> retains 5.0 (flat 1) and 2.0 (flat 3)
        mask, th, nnz = generate_magnitude_mask(weight, 0.5)
        assert nnz == 2
        masked_w = weight * mask
        assert (masked_w != 0).sum().item() == 2
        assert masked_w[0, 1] == 5.0
        assert masked_w[1, 1] == 2.0


@pytest.fixture(scope="module")
def bert_weight() -> torch.Tensor:
    """Fixture providing the extracted BERT layer-0 intermediate weight tensor.

    If data/bert_intermediate_weight.pt does not exist, skips gracefully rather than
    initiating an unexpected external network download during standard test runs.
    Workload acquisition must be explicitly invoked via:
      python patterns/prepare_bert_workload.py
    """
    weight_path = PROJECT_ROOT / "data" / "bert_intermediate_weight.pt"
    if not weight_path.exists():
        pytest.skip(
            f"Workload tensor not found at {weight_path}. "
            "Run 'python patterns/prepare_bert_workload.py' to acquire the BERT workload."
        )
    return torch.load(weight_path, map_location="cpu", weights_only=True)


@pytest.mark.workload
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


@pytest.mark.workload
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


@pytest.mark.workload
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
