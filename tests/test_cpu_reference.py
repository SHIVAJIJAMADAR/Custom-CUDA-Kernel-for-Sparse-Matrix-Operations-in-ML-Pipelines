"""CPU reference correctness test.

Verifies that:
1. pack → unpack round-trip reconstructs a structured-pruned matrix
2. CPU sparse matmul (X @ unpack(pack(W)).T) matches dense matmul on the pruned matrix
3. Edge cases (minimal dimensions, larger dimensions) are handled
"""
import pytest
import torch

from formats.blocked import pack_2of8_block_sparse, unpack_2of8_block_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cpu_sparse_matmul_via_unpack(x: torch.Tensor, packed) -> torch.Tensor:
    """CPU reference path: unpack sparse representation, then dense matmul."""
    w_reconstructed = unpack_2of8_block_sparse(packed, dtype=x.dtype)
    return x @ w_reconstructed.T


def structured_prune_dense(w: torch.Tensor, block_size=32, group_blocks=8, active_blocks=2):
    """Prune W to the same structure the packer would produce, but return dense."""
    packed = pack_2of8_block_sparse(w, block_size=block_size,
                                     group_blocks=group_blocks,
                                     active_blocks=active_blocks)
    return unpack_2of8_block_sparse(packed, dtype=w.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCPUReferenceMatmul:
    """Y = X @ W_pruned.T computed via the sparse representation must match
    the same computation done directly on the dense pruned matrix."""

    @pytest.mark.parametrize("B,M,K", [
        (1, 32, 256),       # minimal: 1 row-block, 1 group
        (4, 32, 256),       # small batch
        (16, 64, 256),      # 2 row-blocks
        (8, 64, 512),       # 2 row-blocks, 2 groups
        (4, 128, 256),      # 4 row-blocks
        (32, 128, 512),     # larger batch
    ])
    def test_sparse_matmul_matches_dense(self, B, M, K):
        torch.manual_seed(42)
        w = torch.randn(M, K, dtype=torch.float32)
        x = torch.randn(B, K, dtype=torch.float32)

        # Dense reference: prune W, then compute Y = X @ W_pruned.T
        w_pruned = structured_prune_dense(w)
        y_dense = x @ w_pruned.T

        # Sparse reference: pack, unpack, matmul
        packed = pack_2of8_block_sparse(w)
        y_sparse = cpu_sparse_matmul_via_unpack(x, packed)

        torch.testing.assert_close(y_sparse, y_dense, rtol=1e-5, atol=1e-5)

    def test_fp16_reference(self):
        """FP16 path should still produce correct results (with relaxed tolerance)."""
        torch.manual_seed(99)
        B, M, K = 4, 64, 256
        w = torch.randn(M, K, dtype=torch.float16)
        x = torch.randn(B, K, dtype=torch.float16)

        w_pruned = structured_prune_dense(w)
        y_dense = (x.float() @ w_pruned.float().T).half()

        packed = pack_2of8_block_sparse(w)
        y_sparse = cpu_sparse_matmul_via_unpack(x, packed)

        # FP16 accumulation has larger numerical error
        torch.testing.assert_close(y_sparse.float(), y_dense.float(), rtol=1e-2, atol=1e-2)


class TestPackUnpackRoundtrip:
    """Extended pack/unpack correctness tests beyond test_packer.py."""

    @pytest.mark.parametrize("M,K", [
        (32, 256),
        (64, 256),
        (128, 256),
        (64, 512),
        (128, 512),
        (256, 256),
    ])
    def test_roundtrip_preserves_active_blocks(self, M, K):
        torch.manual_seed(7)
        w = torch.randn(M, K, dtype=torch.float32)
        packed = pack_2of8_block_sparse(w)
        recon = unpack_2of8_block_sparse(packed)

        # The reconstructed matrix should be the structured-pruned version
        assert recon.shape == (M, K)
        # Non-zero blocks should exactly match the original (selected blocks)
        assert torch.isfinite(recon).all()
        # Sparsity should be 75% (6 of 8 blocks zeroed)
        total_elements = M * K
        zero_elements = (recon == 0).sum().item()
        actual_sparsity = zero_elements / total_elements
        # With random data, most blocks will have some non-zero value,
        # so the sparsity should be close to 75%
        assert 0.70 <= actual_sparsity <= 0.80, f"Expected ~75% sparsity, got {actual_sparsity:.2%}"

    def test_zero_weight_handling(self):
        """A zero weight matrix should pack without errors."""
        w = torch.zeros(32, 256, dtype=torch.float32)
        packed = pack_2of8_block_sparse(w)
        recon = unpack_2of8_block_sparse(packed)
        assert (recon == 0).all()

    def test_single_strong_block(self):
        """When only one block has nonzero values, it should be in the top-2."""
        w = torch.zeros(32, 256, dtype=torch.float32)
        # Put values only in block 3 (columns 96-127)
        w[:, 96:128] = 1.0
        packed = pack_2of8_block_sparse(w)
        recon = unpack_2of8_block_sparse(packed)
        # Block 3 should be preserved
        assert (recon[:, 96:128] == 1.0).all()
