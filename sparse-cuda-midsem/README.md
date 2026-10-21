# SparseCUDA — Mid-Sem Implementation Bundle

This repository is the implementation backbone for the project:

> **Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines**

## What is implemented

- Model/weight sparsity profiling utilities.
- 2-of-8 **block-sparsity candidate** packer with 32×32 blocks.
- Compact active-block position metadata.
- CPU reference reconstruction for correctness.
- A V0 custom CUDA kernel for sparse-dense matrix multiplication.
- PyTorch C++/CUDA custom operator registration.
- Correctness tests that can be executed on a CUDA-enabled machine.
- Benchmark harness for dense PyTorch/cuBLAS-backed matmul and the custom CUDA operator.
- Environment-check scripts and profiling command templates.

## Important scope note

The 2-of-8 + 32×32 configuration is intentionally a **primary candidate**, not a universally optimal claim. The project should select/freeze the final pattern only after profiling the target ML workload and comparing against the appropriate NVIDIA sparse baselines.

## Current runtime note

This package can be prepared and CPU-tested without an NVIDIA GPU, but actual CUDA compilation/execution requires a CUDA-capable NVIDIA system with a compatible CUDA-enabled PyTorch installation.

## Technical basis

PyTorch documents the C++/CUDA custom-operator path through `TORCH_LIBRARY` / `torch.library`, and NVIDIA documents the CUDA execution/memory model and cuSPARSE sparse-matrix formats used as the comparison basis.

- PyTorch custom C++/CUDA operators: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- PyTorch torch.library: https://docs.pytorch.org/docs/stable/library.html
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/
- cuSPARSE: https://docs.nvidia.com/cuda/cusparse/
- cuBLAS: https://docs.nvidia.com/cuda/cublas/
- Nsight Compute: https://docs.nvidia.com/nsight-compute/
- Nsight Systems: https://docs.nvidia.com/nsight-systems/
