# Mid-Sem Development Checklist

## Foundation
- [ ] CUDA-capable development machine verified
- [ ] PyTorch CUDA build verified
- [ ] Target ML model/layer selected
- [ ] Target matrix dimensions recorded

## Sparse pipeline
- [x] 32×32 block partitioning implementation
- [x] 2-of-8 candidate pruning logic
- [x] packed active block values
- [x] compact block-position metadata
- [x] CPU reconstruction reference

## Custom kernel
- [x] V0 kernel source
- [x] PyTorch operator registration
- [ ] CUDA build on team GPU
- [ ] CUDA correctness test passed on team GPU

## Baselines
- [x] Dense matmul benchmark harness
- [ ] Direct cuSPARSE CSR benchmark
- [ ] Direct cuSPARSE BSR benchmark
- [ ] Direct cuSPARSE Blocked-ELL benchmark

## Final optimization path
- [ ] shared-memory tiled kernel
- [ ] warp/register optimization
- [ ] Tensor Core / MMA path
- [ ] Nsight-driven tuning
- [ ] real model-layer integration
