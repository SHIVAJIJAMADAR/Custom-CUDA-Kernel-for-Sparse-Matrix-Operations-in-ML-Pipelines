from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Packed2of8:
    values: torch.Tensor       # [row_blocks, groups, 2, BS, BS]
    positions: torch.Tensor    # [row_blocks, groups, 2], uint8
    M: int
    K: int
    block_size: int = 32
    group_blocks: int = 8
    active_blocks: int = 2

    @property
    def block_sparsity(self) -> float:
        return 1.0 - self.active_blocks / self.group_blocks


def pack_2of8_block_sparse(
    weight: torch.Tensor,
    *,
    block_size: int = 32,
    group_blocks: int = 8,
    active_blocks: int = 2,
) -> Packed2of8:
    """Pack a [M,K] weight matrix into a fixed 2-of-8 block pattern.

    Blocks are grouped along K. In each group, the active_blocks blocks with the
    largest Frobenius norm are retained. The other blocks are omitted.
    """
    if weight.dim() != 2:
        raise ValueError("weight must be 2-D [M,K]")
    if block_size <= 0 or group_blocks <= 0 or active_blocks <= 0:
        raise ValueError("block parameters must be positive")
    if active_blocks > group_blocks:
        raise ValueError("active_blocks cannot exceed group_blocks")

    M, K = map(int, weight.shape)
    span = block_size * group_blocks
    if M % block_size != 0:
        raise ValueError(f"M={M} must be divisible by block_size={block_size}")
    if K % span != 0:
        raise ValueError(f"K={K} must be divisible by block_size*group_blocks={span}")

    w = weight.detach().contiguous()
    rb = M // block_size
    groups = K // span

    # Materialize blocks in a deterministic layout.
    block_bank = w.view(rb, block_size, groups, group_blocks, block_size)
    block_bank = block_bank.permute(0, 2, 3, 1, 4).contiguous()
    # [rb, groups, group_blocks, BS, BS]

    norms = block_bank.float().pow(2).sum(dim=(-1, -2))
    top = torch.topk(norms, k=active_blocks, dim=2, largest=True, sorted=True)
    pos = top.indices.to(torch.uint8)

    gather_idx = pos.long().unsqueeze(-1).unsqueeze(-1).expand(
        rb, groups, active_blocks, block_size, block_size
    )
    values = torch.gather(block_bank, dim=2, index=gather_idx).contiguous()

    return Packed2of8(
        values=values,
        positions=pos.contiguous(),
        M=M,
        K=K,
        block_size=block_size,
        group_blocks=group_blocks,
        active_blocks=active_blocks,
    )


def unpack_2of8_block_sparse(packed: Packed2of8, *, dtype=None) -> torch.Tensor:
    """Reconstruct the structured-pruned dense matrix for correctness checking."""
    BS = packed.block_size
    GB = packed.group_blocks
    AB = packed.active_blocks
    rb, groups = packed.values.shape[:2]
    out = torch.zeros(
        (packed.M, packed.K),
        dtype=dtype or packed.values.dtype,
        device=packed.values.device,
    )

    values = packed.values
    positions = packed.positions.long()

    for rbi in range(rb):
        m0 = rbi * BS
        for gi in range(groups):
            k_group0 = gi * GB * BS
            for ai in range(AB):
                p = int(positions[rbi, gi, ai].item())
                k0 = k_group0 + p * BS
                out[m0:m0 + BS, k0:k0 + BS] = values[rbi, gi, ai]
    return out
