#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {

// V0 correctness kernel.
// Layout:
// values    [row_blocks, groups, active_blocks, BS, BS]
// positions [row_blocks, groups, active_blocks], each entry in [0, group_blocks)
// x         [batch, K]
// output    [batch, M]
//
// Weight equation:
// Y[b,m] = sum_active W[m,k] * X[b,k]
// where global k = (group * group_blocks + position) * BS + local_k.
//
// One CTA computes a 16x16 output tile. This is intentionally simple and correct;
// later versions can introduce shared-memory tiling, register blocking, MMA, etc.
template <typename scalar_t>
__global__ void sparse_linear_v0_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ values,
    const uint8_t* __restrict__ positions,
    scalar_t* __restrict__ out,
    int B,
    int M,
    int K,
    int row_blocks,
    int groups,
    int active_blocks,
    int BS,
    int group_blocks) {

    const int local_b = threadIdx.y;
    const int local_m = threadIdx.x;
    const int b = blockIdx.y * blockDim.y + local_b;
    const int m = blockIdx.x * blockDim.x + local_m;

    if (b >= B || m >= M) return;

    const int rb = m / BS;
    const int r = m % BS;
    if (rb >= row_blocks) return;

    float acc = 0.0f;

    for (int g = 0; g < groups; ++g) {
        for (int a = 0; a < active_blocks; ++a) {
            const size_t meta_idx = (static_cast<size_t>(rb) * groups + g) * active_blocks + a;
            const int block_pos = static_cast<int>(positions[meta_idx]);
            if (block_pos < 0 || block_pos >= group_blocks) continue;

            const int global_block_col = g * group_blocks + block_pos;
            const int k0 = global_block_col * BS;

            const size_t w_block_base =
                (((static_cast<size_t>(rb) * groups + g) * active_blocks + a) * BS * BS) +
                static_cast<size_t>(r) * BS;

            for (int kk = 0; kk < BS; ++kk) {
                const int k = k0 + kk;
                if (k >= K) continue;
                const float xv = static_cast<float>(x[static_cast<size_t>(b) * K + k]);
                const float wv = static_cast<float>(values[w_block_base + kk]);
                acc = fmaf(xv, wv, acc);
            }
        }
    }

    out[static_cast<size_t>(b) * M + m] = static_cast<scalar_t>(acc);
}

} // namespace

torch::Tensor sparse_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& values,
    const torch::Tensor& positions,
    int64_t M,
    int64_t K,
    int64_t block_size,
    int64_t group_blocks,
    int64_t active_blocks) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(positions.is_cuda(), "positions must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "V0 currently supports FP16 input");
    TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be FP16");
    TORCH_CHECK(positions.scalar_type() == torch::kUInt8, "positions must be uint8");
    TORCH_CHECK(x.dim() == 2, "x must be [batch, K]");
    TORCH_CHECK(values.dim() == 5, "values must be [row_blocks, groups, active, BS, BS]");
    TORCH_CHECK(positions.dim() == 3, "positions must be [row_blocks, groups, active]");
    TORCH_CHECK(block_size == 32, "V0 currently targets 32x32 blocks");
    TORCH_CHECK(group_blocks == 8, "V0 currently targets groups of 8 blocks");
    TORCH_CHECK(active_blocks == 2, "V0 currently targets 2 active blocks per group");
    TORCH_CHECK(M > 0 && K > 0, "M and K must be positive");
    TORCH_CHECK(K % (block_size * group_blocks) == 0,
                "K must be divisible by block_size * group_blocks (256 for V0)");
    TORCH_CHECK(M % block_size == 0, "M must be divisible by block_size (32 for V0)");
    TORCH_CHECK(x.size(1) == K, "x second dimension must equal K");

    const int B = static_cast<int>(x.size(0));
    const int Mi = static_cast<int>(M);
    const int Ki = static_cast<int>(K);
    const int BS = static_cast<int>(block_size);
    const int GB = static_cast<int>(group_blocks);
    const int AB = static_cast<int>(active_blocks);
    const int row_blocks = Mi / BS;
    const int groups = Ki / (BS * GB);

    TORCH_CHECK(values.size(0) == row_blocks, "values row_blocks mismatch");
    TORCH_CHECK(values.size(1) == groups, "values groups mismatch");
    TORCH_CHECK(values.size(2) == AB, "values active-block dimension mismatch");
    TORCH_CHECK(values.size(3) == BS && values.size(4) == BS, "values block dimensions must be 32x32");
    TORCH_CHECK(positions.size(0) == row_blocks, "positions row_blocks mismatch");
    TORCH_CHECK(positions.size(1) == groups, "positions groups mismatch");
    TORCH_CHECK(positions.size(2) == AB, "positions active-block dimension mismatch");

    auto out = torch::empty({B, Mi}, x.options());

    const dim3 block(16, 16);
    const dim3 grid((Mi + block.x - 1) / block.x,
                    (B + block.y - 1) / block.y);

    const at::cuda::CUDAGuard device_guard(x.device());
    sparse_linear_v0_kernel<at::Half><<<grid, block>>>(
        reinterpret_cast<const at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(values.data_ptr<at::Half>()),
        positions.data_ptr<uint8_t>(),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        B, Mi, Ki, row_blocks, groups, AB, BS, GB);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
