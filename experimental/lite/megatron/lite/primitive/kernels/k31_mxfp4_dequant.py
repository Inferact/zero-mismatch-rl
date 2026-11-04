# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``k31_mxfp4_dequant.cu``: gathered checkpoint MXFP4 -> BF16 in one pass.

The kernel is the ``_k31_mxfp4_dequant`` extension built with the package (``setup.py``,
sm_100a / sm_103a).
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from megatron.lite.primitive.kernels import _k31_mxfp4_dequant as _ext
except ImportError as exc:
    raise ImportError(
        "megatron.lite.primitive.kernels._k31_mxfp4_dequant is not built: install Megatron Lite from "
        "a wheel built in the training image (cd experimental/lite && "
        "pip wheel --no-build-isolation --no-deps -w dist . && pip install --no-deps dist/*.whl), "
        "or build it in place with python setup.py build_ext --inplace"
    ) from exc


def dequant_experts(packed: Tensor, scale: Tensor, experts: Tensor) -> Tensor:
    """``dequantize_k31_mxfp4(packed[experts], scale[experts]).to(bf16)``, bitwise, one kernel."""

    return _ext.dequant(
        packed.contiguous().view(torch.uint8),
        scale.contiguous().view(torch.uint8),
        experts.to(device=packed.device, dtype=torch.int64).contiguous(),
    )


__all__ = ["dequant_experts"]
