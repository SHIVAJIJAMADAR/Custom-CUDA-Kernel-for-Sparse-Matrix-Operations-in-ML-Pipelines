# Mid-Sem Evidence Map

This file maps the mid-sem presentation claims to concrete implementation artifacts.

| Presentation area | Evidence in repository |
|---|---|
| CUDA environment / GPU verification | `scripts/check_cuda_env.sh`, `scripts/check_environment.py` |
| Target sparse transformation | `python/sparse_format.py` |
| Block partitioning | `pack_2of8_block_sparse()` |
| Structured pruning | top-2 Frobenius norm selection in `python/sparse_format.py` |
| Sparse block packing | `Packed2of8.values` |
| Compact metadata | `Packed2of8.positions` (`uint8`) |
| CPU reference | `unpack_2of8_block_sparse()` |
| V0 CUDA kernel | `kernels/sparse_linear_v0.cu` |
| PyTorch integration | `csrc/sparse_op.cpp`, `csrc/sparse_op.h`, `python/build_extension.py` |
| Correctness tests | `tests/test_packer.py`, `tests/test_cuda_kernel.py` |
| Dense benchmark | `bench/benchmark_dense.py` |
| Custom benchmark | `bench/benchmark_custom.py` |
| Future profiling | `profiling/README.md` |

## Honesty rule

A repository artifact is evidence that the implementation exists in source form. It is **not by itself evidence that a GPU run succeeded**. GPU-specific claims should be backed by generated benchmark/profiling output from the team's CUDA machine.
