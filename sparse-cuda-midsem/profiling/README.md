# Profiling Runbook

## Nsight Systems

Use this first to inspect end-to-end CPU/GPU scheduling and kernel launches.

Example:

```bash
nsys profile -o results/nsys_custom \
  python bench/benchmark_custom.py --batch 64 --m 4096 --k 4096
```

## Nsight Compute

Once the kernel is isolated, inspect memory efficiency, occupancy, warp execution and instruction-level behavior.

Example:

```bash
ncu --set full \
    --target-processes all \
    -o results/ncu_custom \
    python bench/benchmark_custom.py --batch 64 --m 4096 --k 4096
```

The exact command may need adjustment for the installed Nsight version and environment.
