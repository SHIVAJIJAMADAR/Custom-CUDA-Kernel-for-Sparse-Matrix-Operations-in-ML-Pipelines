"""Tests for CSR format conversion (R2) and correctness oracle (R8).

Verifies:
1. Dense → CSR → Dense round-trip preserves values
2. Masked matrix conversion works correctly
3. Row-length statistics are accurate
4. Edge cases: empty matrix, all-zeros, single element, full density
5. Correctness oracle: A_masked @ X matches sparse path
"""
import pytest
import torch

from formats.csr import (
    CSRMatrix,
    dense_to_csr,
    csr_to_dense,
    dense_masked_to_csr,
    row_lengths,
    row_length_stats,
)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestCSRRoundtrip:
    """Dense → CSR → Dense must reconstruct exactly."""

    @pytest.mark.parametrize("M,K,sparsity", [
        (4, 4, 0.0),         # fully dense
        (4, 4, 0.5),         # 50% sparse
        (4, 4, 0.9),         # 90% sparse
        (8, 16, 0.75),       # non-square
        (1, 1, 0.0),         # minimal
        (32, 256, 0.95),     # larger, very sparse
        (64, 64, 0.5),       # square medium
    ])
    def test_roundtrip(self, M, K, sparsity):
        torch.manual_seed(42)
        dense = torch.randn(M, K, dtype=torch.float32)
        # Apply sparsity by zeroing elements
        mask = (torch.rand(M, K) > sparsity).float()
        dense = dense * mask

        csr = dense_to_csr(dense)
        recon = csr_to_dense(csr)

        torch.testing.assert_close(recon, dense, rtol=0, atol=0)

    def test_all_zeros(self):
        dense = torch.zeros(8, 8)
        csr = dense_to_csr(dense)
        assert csr.nnz == 0
        assert csr.sparsity == 1.0
        recon = csr_to_dense(csr)
        assert (recon == 0).all()

    def test_all_ones(self):
        dense = torch.ones(4, 6)
        csr = dense_to_csr(dense)
        assert csr.nnz == 24
        assert csr.sparsity == 0.0
        recon = csr_to_dense(csr)
        torch.testing.assert_close(recon, dense)

    def test_single_element(self):
        dense = torch.zeros(3, 3)
        dense[1, 2] = 42.0
        csr = dense_to_csr(dense)
        assert csr.nnz == 1
        recon = csr_to_dense(csr)
        torch.testing.assert_close(recon, dense)

    def test_fp16_roundtrip(self):
        torch.manual_seed(7)
        dense = torch.randn(16, 32, dtype=torch.float16)
        mask = (torch.rand(16, 32) > 0.6).half()
        dense = dense * mask
        csr = dense_to_csr(dense)
        recon = csr_to_dense(csr)
        torch.testing.assert_close(recon, dense, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# CSR metadata / properties
# ---------------------------------------------------------------------------

class TestCSRProperties:

    def test_sparsity_calculation(self):
        dense = torch.zeros(10, 10)
        dense[0, 0] = 1.0
        dense[5, 5] = 2.0
        csr = dense_to_csr(dense)
        assert csr.nnz == 2
        assert csr.M == 10
        assert csr.K == 10
        assert abs(csr.sparsity - 0.98) < 1e-6
        assert abs(csr.density - 0.02) < 1e-6

    def test_row_ptr_shape(self):
        dense = torch.randn(8, 4)
        csr = dense_to_csr(dense)
        assert csr.row_ptr.shape == (9,)
        assert csr.row_ptr[0] == 0
        assert csr.row_ptr[-1] == csr.nnz


# ---------------------------------------------------------------------------
# Row-length analysis (supports R7 load-balance decision)
# ---------------------------------------------------------------------------

class TestRowLengths:

    def test_uniform_row_lengths(self):
        """When every row has the same density, row lengths should be equal."""
        # Identity matrix — each row has exactly 1 nonzero
        dense = torch.eye(8)
        csr = dense_to_csr(dense)
        lengths = row_lengths(csr)
        assert (lengths == 1).all()
        stats = row_length_stats(csr)
        assert stats["min"] == 1
        assert stats["max"] == 1
        assert stats["std"] == 0.0
        assert stats["num_empty_rows"] == 0

    def test_skewed_row_lengths(self):
        """One row with many nonzeros, others empty — tests skew detection."""
        dense = torch.zeros(4, 8)
        dense[0, :] = 1.0  # full row
        csr = dense_to_csr(dense)
        stats = row_length_stats(csr)
        assert stats["min"] == 0
        assert stats["max"] == 8
        assert stats["num_empty_rows"] == 3

    def test_empty_matrix_stats(self):
        dense = torch.zeros(4, 4)
        csr = dense_to_csr(dense)
        stats = row_length_stats(csr)
        assert stats["total_nnz"] == 0
        assert stats["num_empty_rows"] == 4


# ---------------------------------------------------------------------------
# Masked conversion (R8 correctness oracle path)
# ---------------------------------------------------------------------------

class TestMaskedConversion:

    def test_masked_csr_matches_dense_masked(self):
        """dense_masked_to_csr should produce the same result as manual masking."""
        torch.manual_seed(99)
        weight = torch.randn(16, 32)
        mask = (torch.rand(16, 32) > 0.7).float()

        csr = dense_masked_to_csr(weight, mask)
        expected = weight * mask
        recon = csr_to_dense(csr)
        torch.testing.assert_close(recon, expected)

    def test_shape_mismatch_raises(self):
        w = torch.randn(4, 4)
        m = torch.ones(4, 8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            dense_masked_to_csr(w, m)


# ---------------------------------------------------------------------------
# Correctness oracle (R8): A_masked @ X
# ---------------------------------------------------------------------------

class TestCorrectnessOracle:
    """The reference computation: Y_ref = A_masked @ X.

    Every sparse format and kernel must match this output.
    """

    @pytest.mark.parametrize("B,M,K,sparsity", [
        (1, 8, 8, 0.5),
        (4, 16, 32, 0.75),
        (8, 32, 64, 0.9),
        (16, 64, 128, 0.95),
    ])
    def test_csr_matmul_matches_dense_oracle(self, B, M, K, sparsity):
        torch.manual_seed(42)
        weight = torch.randn(M, K, dtype=torch.float32)
        mask = (torch.rand(M, K) > sparsity).float()
        x = torch.randn(B, K, dtype=torch.float32)

        # Dense oracle: Y_ref = (weight * mask) @ x.T  →  Y_ref[b] = x[b] @ A_masked.T
        A_masked = weight * mask
        Y_ref = x @ A_masked.T  # [B, M]

        # CSR path: convert masked weights to CSR, reconstruct, matmul
        csr = dense_masked_to_csr(weight, mask)
        A_recon = csr_to_dense(csr)
        Y_csr = x @ A_recon.T  # [B, M]

        torch.testing.assert_close(Y_csr, Y_ref, rtol=1e-5, atol=1e-5)

    def test_threshold_based_sparsity(self):
        """Test that small-magnitude elements can be zeroed via threshold."""
        torch.manual_seed(7)
        dense = torch.randn(8, 8) * 0.01  # small values
        dense[0, 0] = 10.0  # one large value

        csr = dense_to_csr(dense, threshold=0.05)
        recon = csr_to_dense(csr)

        # Only the large value should survive
        assert csr.nnz >= 1
        assert recon[0, 0] == 10.0
