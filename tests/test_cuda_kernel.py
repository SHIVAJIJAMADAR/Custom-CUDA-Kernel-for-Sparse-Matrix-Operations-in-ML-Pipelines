import pytest
import torch

from formats.blocked import pack_2of8_block_sparse, unpack_2of8_block_sparse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_custom_cuda_matches_reference():
    pytest.importorskip("sparse_cuda_ext", reason="sparse_cuda_ext CUDA extension module is not compiled")

    torch.manual_seed(11)
    B, M, K = 4, 32, 256
    w = torch.randn(M, K, dtype=torch.float16)
    packed = pack_2of8_block_sparse(w)
    sparse_w = unpack_2of8_block_sparse(packed)
    x = torch.randn(B, K, dtype=torch.float16, device="cuda")

    values = packed.values.cuda()
    positions = packed.positions.cuda()
    y_custom = torch.ops.sparse_cuda.linear(x, values, positions, M, K, 32, 8, 2)
    y_ref = x @ sparse_w.cuda().T

    torch.testing.assert_close(y_custom.float(), y_ref.float(), rtol=2e-2, atol=2e-2)
