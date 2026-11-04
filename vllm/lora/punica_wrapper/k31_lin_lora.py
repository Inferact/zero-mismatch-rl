# SPDX-License-Identifier: Apache-2.0
"""K31 linear LoRA: wrapper for csrc/k31/k31_lin_lora.cu (semantics documented there).

Under VLLM_BATCH_INVARIANT=1 punica_gpu.py routes the linear LoRA here wherever ok() accepts the operands; the
result equals punica's Triton path bit for bit (production rollout e2e, ids and log-prob bits).
Row-parallel layers under K31 tp-ordered take apply_split when split_wins (prefill chunks: decode keeps the single
input gather), else gather the input and apply.
"""

import torch

# slice counts up to this (at most 2) run fused (x loaded once for all slices); more slices run one cluster grid per
# slice in the same launch (launch shape only)
FUSE_MAX = 2
# apply_split adds a collective and two launches (~135 us at TP 8 for any M) and spares moving M x K
# bf16: it is the faster path from M*K ~ 1.3e7..1.5e7 (K = 33792 / 12288 / 6144: M ~ 430 / 1190 / 2100). Taken
# from 2^24 (M >= 497 / 1366 / 2731), far above the largest CUDA-graph batch.
SPLIT_MK = 1 << 24
RANK = 32
MAXS = 7


_EXT = None


def ext():
    """csrc/k31/k31_lin_lora.cu, built with vLLM (sm_100a / sm_103a, --fmad=false)."""
    global _EXT
    if _EXT is None:
        from vllm import _k31_lin_lora_C

        _EXT = _k31_lin_lora_C
    return _EXT


def _slot0(w):
    """[max_loras(=1), 1, rows, cols] or [1, rows, cols] -> [rows, cols] view"""
    while w.dim() > 2:
        if w.size(0) != 1:
            return None
        w = w[0]
    return w


def ok(y, x, lora_a_stacked, lora_b_stacked, output_slices, k=None):
    """True when the call is the case this kernel reproduces bit for bit (else use punica's Triton path).
    k: the shrink width when x holds only this rank's columns of the input (apply_split)."""
    if not (1 <= len(output_slices) <= MAXS and len(lora_a_stacked) == len(output_slices)):
        return False
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.shape[1] or x.shape[1] % 8 or not x.is_cuda:
        return False
    if y.dtype != torch.bfloat16 or not y.is_contiguous() or y.shape[-1] != sum(output_slices):
        return False
    for a, b, n in zip(lora_a_stacked, lora_b_stacked, output_slices):
        if a.size(0) != 1 or b.size(0) != 1 or not a.is_contiguous() or not b.is_contiguous():
            return False
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            return False
        if a.shape[-2] != RANK or a.shape[-1] != (k or x.shape[1]) or b.shape[-1] != RANK or b.shape[-2] != n:
            return False
    return True


def apply(y, x, lora_a_stacked, lora_b_stacked, scale, meta):
    """y[m, off_s:off_s+N_s] = punica-BI add_lora_linear delta (add_inputs=False); every row of y is written.

    meta = LoRAKernelMeta.meta_args(M, ..): (token_lora_mapping, sorted idx, tokens per lora, start loc,
    active_lora_ids, no_lora_flag_cpu, num_active_loras). Same token selection as punica: no token when the
    CPU no-LoRA flag is set; else the tokens whose slot is one of active_lora_ids[:num_active_loras]. The
    other rows are +0."""
    tlm, _, _, _, ids, no_lora, nact = meta
    if no_lora.item():
        y.zero_()
        return
    y2 = y.view(-1, y.shape[-1])
    a = [_slot0(w) for w in lora_a_stacked]
    b = [_slot0(w) for w in lora_b_stacked]
    nids = min(int(nact.item()), ids.numel())
    ext().lin_lora(y2, x.contiguous(), a, b, tlm, ids, nids, float(scale), FUSE_MAX)


def split_ok(y, x, lora_a_stacked, lora_b_stacked, output_slices, w):
    """apply_split covers the call: one slice, TP w in (2, 4, 8), x [M, K / w] = this rank's input columns."""
    return len(output_slices) == 1 and w in (2, 4, 8) and ok(y, x, lora_a_stacked, lora_b_stacked, output_slices,
                                                               k=w * x.shape[1])


def split_wins(m, k):
    """apply_split is the faster path for an [m, k] row-parallel input (k = full input width)."""
    return m * k >= SPLIT_MK


def apply_split(y, x, lora_a_stacked, lora_b_stacked, scale, meta, tpo):
    """apply(y, tpo.gather_cols(x), ..) bit for bit, x [M, K / W] holding this rank's columns of a row-parallel
    input: split_cols brings each rank the input columns of its 8 / W shrink splits, lin_lora_part runs exactly those
    split chains, gather_cols (own region and epoch) collects the fp32 partials of the 8 splits, lin_lora_red
    reduces them in split order and expands. A rank receives 1/W of the gathered input bytes (plus 128 B of partials
    per split and token) and shrinks K / W columns instead of K."""
    tlm, _, _, _, ids, no_lora, nact = meta
    if no_lora.item():
        y.zero_()
        return
    a, b = _slot0(lora_a_stacked[0]), _slot0(lora_b_stacked[0])
    nl = 8 // tpo.W
    p = torch.empty((x.shape[0], nl * RANK), dtype=torch.float32, device=x.device)
    ext().lin_lora_part(p, tpo.split_cols(x), a, tpo.me * nl, a.shape[1])
    nids = min(int(nact.item()), ids.numel())
    ext().lin_lora_red(y.view(-1, y.shape[-1]), tpo.gather_cols(p, part=True), b, tlm, ids, nids, float(scale))
