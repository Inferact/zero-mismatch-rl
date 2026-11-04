# SPDX-License-Identifier: Apache-2.0
"""K31 expert-LoRA glue kernels: wrappers for csrc/k31/k31_moe_lora_fuse.cu (see that file for the exact semantics)."""

import torch



_EXT = None


def ext():
    """csrc/k31/k31_moe_lora_fuse.cu, built with vLLM (sm_100a / sm_103a, --fmad=false)."""
    global _EXT
    if _EXT is None:
        from vllm import _k31_moe_lora_fuse_C

        _EXT = _k31_moe_lora_fuse_C
    return _EXT


def i32(t: torch.Tensor) -> torch.Tensor:
    t = t.reshape(-1)
    return (t if t.dtype == torch.int32 else t.to(torch.int32)).contiguous()


def gather_add(g1, e2p32, ly, inter):
    return ext().gather_add(g1.contiguous(), e2p32, ly.contiguous(), inter)


def act_cast(y32, p2e32):
    return ext().act_cast(y32.contiguous(), p2e32)


def slot(o2, e2p32, ww, d2):
    return ext().slot(o2.contiguous(), e2p32, ww.float().contiguous(), d2.contiguous())


def slot_sum(o2, e2p32, ww, d2, out):
    ext().slot_sum(o2.contiguous(), e2p32, ww.float().contiguous(), d2.contiguous(), out)
