from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parents[1]

setup(
    name="sparse_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="sparse_cuda_ext",
            sources=[
                str(ROOT / "csrc" / "sparse_op.cpp"),
                str(ROOT / "kernels" / "sparse_linear_v0.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
