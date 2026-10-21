#pragma once

#include <torch/extension.h>

torch::Tensor sparse_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& values,
    const torch::Tensor& positions,
    int64_t M,
    int64_t K,
    int64_t block_size,
    int64_t group_blocks,
    int64_t active_blocks);
