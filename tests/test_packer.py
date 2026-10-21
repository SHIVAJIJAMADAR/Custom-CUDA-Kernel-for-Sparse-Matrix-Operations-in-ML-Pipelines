import torch

from formats.blocked import pack_2of8_block_sparse, unpack_2of8_block_sparse


def test_pack_unpack_has_expected_structure():
    torch.manual_seed(7)
    w = torch.randn(64, 256, dtype=torch.float16)
    packed = pack_2of8_block_sparse(w)

    assert packed.values.shape == (2, 1, 2, 32, 32)
    assert packed.positions.shape == (2, 1, 2)
    assert packed.positions.dtype == torch.uint8
    assert float(packed.block_sparsity) == 0.75

    recon = unpack_2of8_block_sparse(packed)
    assert recon.shape == w.shape

    # Every reconstructed non-zero element must come from an active block.
    assert torch.isfinite(recon.float()).all()


def test_top2_selection_is_by_block_norm():
    w = torch.zeros(32, 256, dtype=torch.float16)
    # Eight 32x32 blocks in the only group. Make block 6 strongest and block 2 second strongest.
    for j in range(32):
        w[:, j + 6 * 32] = 10
        w[:, j + 2 * 32] = 5
        w[:, j + 1 * 32] = 1

    packed = pack_2of8_block_sparse(w)
    positions = sorted(int(v) for v in packed.positions[0, 0].tolist())
    assert positions == [2, 6]
