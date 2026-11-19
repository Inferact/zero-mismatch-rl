"""Shared host-side scalar and buffer validation."""

import math
import struct


def _contract_scalars(scale, eps, lower_bound):
    bits = []
    for name, value in (("scale", scale), ("eps", eps), ("lower_bound", lower_bound)):
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite host scalar")
        try:
            encoded = struct.pack("!f", value)
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name} must be representable as finite FP32") from error
        bits.append(encoded)
    if struct.unpack("!f", bits[1])[0] <= 0 or not -5 <= lower_bound <= 0:
        raise ValueError(
            "eps must be positive in FP32 and lower_bound must lie in [-5, 0]"
        )
    return tuple(bits)


def _range(t):
    start = t.data_ptr()
    return start, start + t.numel() * t.element_size()


def _pool_size(state_in, state_indices, n, h, device):
    """Validate host-visible pool metadata, without reading device indices."""
    if state_indices is None:
        return n
    import torch

    if not isinstance(state_indices, torch.Tensor):
        raise TypeError("state_indices must be a torch.Tensor")
    if (
        state_indices.device != device
        or state_indices.dtype != torch.int32
        or state_indices.shape != (n,)
        or not state_indices.is_contiguous()
        or (state_indices.numel() and state_indices.data_ptr() % 32)
    ):
        raise ValueError(
            "state_indices must be contiguous, 32-byte aligned CUDA Int32 [N]"
        )
    if (
        state_in.ndim != 4
        or tuple(state_in.shape[1:]) != (h, 128, 128)
        or state_in.shape[0] >= 2**31
        or state_in.shape[0] < n
    ):
        raise ValueError("indexed state must have shape [P,H,128,128], N <= P < 2**31")
    return state_in.shape[0]
