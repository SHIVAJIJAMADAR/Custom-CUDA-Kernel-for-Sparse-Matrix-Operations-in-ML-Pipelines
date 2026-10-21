#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this benchmark on the NVIDIA development machine.")

    x = torch.randn(args.batch, args.k, device="cuda", dtype=torch.float16)
    w = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16)

    for _ in range(30):
        _ = x @ w.T
    torch.cuda.synchronize()

    starts = torch.cuda.Event(enable_timing=True)
    ends = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        starts.record()
        _ = x @ w.T
        ends.record()
        ends.synchronize()
        times.append(starts.elapsed_time(ends) * 1000.0)

    times.sort()
    n = len(times)
    print({
        "implementation": "PyTorch dense matmul / cuBLAS-backed",
        "batch": args.batch,
        "M": args.m,
        "K": args.k,
        "dtype": "float16",
        "median_us": times[n // 2],
        "p90_us": times[min(n - 1, int(n * 0.90))],
    })


if __name__ == "__main__":
    main()
