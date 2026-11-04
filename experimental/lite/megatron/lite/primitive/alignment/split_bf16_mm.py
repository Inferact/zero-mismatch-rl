# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FP32-accurate backward GEMMs on BF16 tensor cores (cuBLAS, fp32 accumulate and output).

The training backward of the vLLM-visible K31 owners is an FP32 VJP.  Its weight operand is
always a BF16 tensor (frozen base weights, MXFP4 dequantized into BF16, LoRA factors), i.e. exactly
representable in BF16, so only the FP32 activation-gradient operand needs care: it is split into
BF16 terms ``a = t0 + t1 (+ t2)`` (residual ``<= 2^-16`` / ``2^-24`` relative per element) and the
product is the sum of the term products, FP32 output.  Against FP64 with the K-chunked
accumulation below: 2 terms (default) 2.5e-6, 3 terms 5e-7, FP32 SGEMM 1.5e-6 to 2e-6 (K=7168 and
40000).  Whole-model LoRA grads against an FP64 yardstick are closer with 2 terms than with the
FP32 reference backward (slice15tail EP16: global rel 2.9e-3 vs 3.2e-3; slice4 EP4: 5.3e-3 vs 5.7e-3).

Accumulation: the sm100 BF16 MMA chain loses precision linearly in K (measured against FP64 at
K=3584/6144: 3.8e-6/6.8e-6, FP32 SIMT 5.6e-7/7.9e-7).  K is therefore cut into
``TC_KCHUNK``-wide blocks (512) accumulated through the GEMM epilogue in FP32
(``addmm``/``baddbmm`` with ``out_dtype=float32``, beta=1): 4.7e-7, i.e. FP32 SIMT accuracy, at
~1/6 of the SIMT time.

No batch invariance is needed in backward, so plain cuBLAS is used (vLLM's batch-invariant mode
overrides neither ``aten::mm.dtype``/``addmm.dtype`` nor ``bmm.dtype``/``baddbmm.dtype``).

Module settings (no environment): ``SPLIT_MM`` = "split" (production), "fp32" (FP32 SIMT reference) or
"fp64" (float64 yardstick) for gradient validation; ``SPLIT_TERMS`` 2; ``TC_KCHUNK`` 512 (0: one MMA chain).
"""

from __future__ import annotations

import torch
from torch import Tensor

_F32 = torch.float32
SPLIT_MM = "split"
SPLIT_TERMS = 2
TC_KCHUNK = 512


def split_enabled() -> bool:
    """``SPLIT_MM != "split"`` selects the FP32 SIMT reference GEMMs (``"fp64"``: float64 yardstick)."""

    return SPLIT_MM == "split"


def split_terms() -> int:
    """BF16 terms per FP32 operand: 2 leave ``<= 2^-16`` relative residual per element, 3 ``<= 2^-24``."""

    return SPLIT_TERMS


def kchunk() -> int:
    return TC_KCHUNK


def split_bf16_terms(a: Tensor, n: int | None = None) -> list[Tensor]:
    """``a ~= sum(terms)`` with each term BF16; the residual of ``n`` terms is ``<= 2^(-8n) |a|``."""

    n = split_terms() if n is None else n
    r = a.float()
    out = []
    for i in range(n):
        t = r.to(torch.bfloat16)
        out.append(t)
        if i + 1 < n:
            r = r - t.float()
    return out


def _chunks(k: int) -> int:
    ck = kchunk()
    return k if ck <= 0 or ck >= k else ck


def mm_bf16(terms: Tensor | list[Tensor], b: Tensor) -> Tensor:
    """``sum(t @ b for t in terms)``: BF16 operands, FP32 out, K-chunked epilogue accumulation.

    The terms are stacked along M (one GEMM per K block for all terms, accumulated in place),
    then the per-term products are added in term order.
    """

    terms = [terms] if isinstance(terms, Tensor) else terms
    m = terms[0].shape[0]
    a = terms[0] if len(terms) == 1 else torch.cat(terms, 0)
    k = b.shape[0]
    ck = _chunks(k)
    out = torch.mm(a[:, :ck], b[:ck], out_dtype=_F32)
    for k0 in range(ck, k, ck):
        torch.ops.aten.addmm.dtype_out(out, a[:, k0 : k0 + ck], b[k0 : k0 + ck], _F32, out=out)
    if len(terms) == 1:
        return out
    res = out[:m].clone()
    for i in range(1, len(terms)):
        res += out[i * m : (i + 1) * m]
    return res


def bmm_bf16(terms: Tensor | list[Tensor], b: Tensor) -> Tensor:
    """Batched :func:`mm_bf16` (``[B, n, K] @ [B, K, m]``)."""

    terms = [terms] if isinstance(terms, Tensor) else terms
    n = terms[0].shape[1]
    a = terms[0] if len(terms) == 1 else torch.cat(terms, 1)
    k = b.shape[1]
    ck = _chunks(k)
    out = torch.ops.aten.bmm.dtype(a[:, :, :ck], b[:, :ck], _F32)
    for k0 in range(ck, k, ck):
        torch.ops.aten.baddbmm.dtype_out(out, a[:, :, k0 : k0 + ck], b[:, k0 : k0 + ck], _F32, out=out)
    if len(terms) == 1:
        return out
    res = out[:, :n].clone()
    for i in range(1, len(terms)):
        res += out[:, i * n : (i + 1) * n]
    return res


def mm_f32(a: Tensor, b: Tensor, *, exact_a: bool = False) -> Tensor:
    """FP32 ``a @ b`` for BF16-exact ``b`` (BF16 or FP32 holding BF16 values)."""

    if not split_enabled():
        if SPLIT_MM == "fp64":
            return torch.mm(a.double(), b.double()).float()
        return torch.mm(a.float(), b.float())
    b16 = b if b.dtype == torch.bfloat16 else b.to(torch.bfloat16)
    if exact_a or a.dtype == torch.bfloat16:
        return mm_bf16(a if a.dtype == torch.bfloat16 else a.to(torch.bfloat16), b16)
    return mm_bf16(split_bf16_terms(a), b16)


__all__ = ["bmm_bf16", "kchunk", "mm_bf16", "mm_f32", "split_bf16_terms", "split_enabled", "split_terms"]
