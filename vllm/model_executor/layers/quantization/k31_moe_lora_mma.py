# SPDX-License-Identifier: Apache-2.0
"""K31 expert LoRA: wrappers for csrc/k31/k31_moe_lora_mma.cu (semantics documented there).

For the K31 MoE (both arms) mxfp4.K31FiW4A16Experts routes the expert LoRA here wherever _k31_lora_cuda_ok
accepts the layout; the result equals punica's Triton kernel bit for bit (production rollout e2e, ids and log-prob bits).
"""

import torch

# cp.async ring depth of the shrink kernels (launch shape only; bits do not depend on it)
ST13 = 6
ST2 = 6
# warps per shrink CTA (8 rank columns each): launch shape only
WN13 = 2
WN2 = 2


_EXT = None


def ext():
    """csrc/k31/k31_moe_lora_mma.cu, built with vLLM (sm_100a / sm_103a, --fmad=false)."""
    global _EXT
    if _EXT is None:
        from vllm import _k31_moe_lora_mma_C

        _EXT = _k31_moe_lora_mma_C
    return _EXT


def _i32(t):
    t = t.reshape(-1)
    return (t if t.dtype == torch.int32 else t.to(torch.int32)).contiguous()


def w13(ly, x, a1, a3, b1, b3, t2e, t2l, nnt, p2e, rt, topk, tlm, enabled, lids, nlids):
    """ly[M*topk, 2N] (zero-initialised BF16) += expert-LoRA w13 delta of the live local slots.
    a1/a3 [R, H] shared, b1/b3 [E, N, R]."""
    m = x.shape[0]
    t13 = torch.empty(m, 2 * a1.shape[0], device=x.device, dtype=torch.bfloat16)
    e = ext()
    e.shrink13(x.contiguous(), a1, a3, t13, ST13, WN13)
    e.expand13(t13, b1, b3, t2e, t2l, nnt, p2e, rt, topk, _i32(tlm), enabled, _i32(lids), nlids, ly)


def w2(d2, act_s, a2, b2, ww, t2e, t2l, nnt, p2e, rt, topk, tlm, enabled, lids, nlids):
    """d2[M, topk, H] (zero-initialised BF16) += routed-weighted expert-LoRA w2 delta of the live local slots.
    act_s [S, I] permuted, a2 [E, R, I], b2 [H, R] shared, ww [M*topk] fp32."""
    t2 = torch.empty(act_s.shape[0], a2.shape[1], device=act_s.device, dtype=torch.bfloat16)
    e = ext()
    e.shrink2(act_s, a2, t2e, t2l, nnt, p2e, rt, t2, ST2, WN2)
    e.expand2(t2, b2, t2e, t2l, nnt, p2e, rt, topk, ww, _i32(tlm), enabled, _i32(lids), nlids, d2)
