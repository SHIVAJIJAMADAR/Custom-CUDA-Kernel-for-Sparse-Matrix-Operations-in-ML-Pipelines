// Direct cuSPARSE SpMM baseline skeleton for the CUDA development machine.
// This file is intentionally kept separate from the PyTorch benchmark so the
// baseline can be controlled and timed independently of Python overhead.
//
// Recommended next step on the CUDA machine:
// 1) create CSR/BSR/Blocked-ELL descriptors for the same target matrix
// 2) call cusparseSpMM with the same resident inputs
// 3) warm up, time with CUDA events, and report median/p90
//
// NVIDIA cuSPARSE documentation:
// https://docs.nvidia.com/cuda/cusparse/

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

int main() {
    std::cout << "cuSPARSE benchmark scaffold: descriptor construction and timing should be wired to the target workload.\n";
    std::cout << "Library: " << CUSPARSE_VERSION << "\n";
    return 0;
}
