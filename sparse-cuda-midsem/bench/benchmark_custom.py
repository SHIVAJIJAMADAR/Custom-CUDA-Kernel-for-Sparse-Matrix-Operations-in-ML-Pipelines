#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch

from python.sparse_format import pack_2of8_block_sparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this benchmark on the NVIDIA development machine.")

    # Build sparse representation on CPU, then move to GPU once.
    w = torch.randn(args.m, args.k, dtype=torch.float16)
    packed = pack_2of8_block_sparse(w)
    x = torch.randn(args.batch, args.k, device="cuda", dtype=torch.float16)
    values = packed.values.cuda()
    positions = packed.positions.cuda()

    import sparse_cuda_ext  # noqa: F401

    # Use the custom operator registered by the extension.
    op = torch.ops.sparse_cuda.linear
    for _ in range(30):
        _ = op(x, values, positions, packed.M, packed.K, 32, 8, 2)
    torch.cuda.synchronize()

    starts = torch.cuda.Event(enable_timing=True)
    ends = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        starts.record()
        _ = op(x, values, positions, packed.M, packed.K, 32, 8, 2)
        ends.record()
        ends.synchronize()
        times.append(starts.elapsed_time(ends) * 1000.0)

    times.sort()
    n = len(times)
    print({
        "implementation": "Custom CUDA V0",
        "batch": args.batch,
        "M": args.m,
        "K": args.k,
        "dtype": "float16",
        "block": "32x32",
        "pattern": "2-of-8",
        "median_us": times[n // 2],
        "p90_us": times[min(n - 1, int(n * 0.90))],
        "warning": "timing excludes preprocessing/packing by design",
    })


if __name__ == "__main__":
    main()
