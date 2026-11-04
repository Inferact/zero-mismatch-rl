# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA extensions of Megatron Lite (metadata and packages live in pyproject.toml).

Build with torch importable: ``pip wheel --no-build-isolation --no-deps .``.  The gencodes are
explicit, so the result does not depend on the builder's GPU or ``TORCH_CUDA_ARCH_LIST``
(torch adds no arch flags of its own when the nvcc flags already name one).
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

K31_GENCODES = [
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
]

setup(
    ext_modules=[
        CUDAExtension(
            name="megatron.lite.primitive.kernels._k31_mxfp4_dequant",
            sources=["megatron/lite/primitive/kernels/k31_mxfp4_dequant.cu"],
            extra_compile_args={"cxx": [], "nvcc": ["-O3", "--fmad=false", *K31_GENCODES]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
