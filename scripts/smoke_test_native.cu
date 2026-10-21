/*
 * Standalone Native CUDA Smoke Test (Test C)
 * Project: Custom CUDA Kernel for Sparse Matrix Operations in ML Pipelines
 * 
 * Verifies that:
 * 1. nvcc and host MSVC compiler work together.
 * 2. CUDA runtime API (cudaMalloc, cudaMemcpy, cudaFree) functions correctly.
 * 3. Kernel launch syntax and grid/block execution work on target GPU.
 * 4. Device synchronization and data readback match expected values.
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "[SMOKE TEST C] Starting native CUDA compilation and execution verification..." << std::endl;

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "FAILED: No CUDA devices found." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Target Device: " << prop.name << " (Compute " 
              << prop.major << "." << prop.minor << ", "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB VRAM)" << std::endl;

    const int n = 1024;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c[i] = 0.0f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": got " << h_c[i]
                      << ", expected " << expected << std::endl;
            correct = false;
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    if (correct) {
        std::cout << "[SMOKE TEST C PASSED] Native CUDA kernel compiled, executed, and verified successfully." << std::endl;
        return 0;
    } else {
        std::cerr << "[SMOKE TEST C FAILED] Numerical verification mismatch." << std::endl;
        return 1;
    }
}
