#include "sparse_op.h"

TORCH_LIBRARY(sparse_cuda, m) {
    m.def("linear(Tensor x, Tensor values, Tensor positions, int M, int K, int block_size, int group_blocks, int active_blocks) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparse_cuda, CUDA, m) {
    m.impl("linear", &sparse_linear_cuda);
}
