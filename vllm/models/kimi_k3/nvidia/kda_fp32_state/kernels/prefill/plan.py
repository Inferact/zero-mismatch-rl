"""Bind bounded FP32 coefficient windows and ordered recurrent scans."""

from dataclasses import dataclass
import torch
from .coefficients import compiled_coeff_kernel
from .scan import (
    compiled_scan_kernel,
    KPACK,
    PAIR_ENGINE,
    SMEM_TWO_PER_SM,
    OVERLAP,
    SMEM_BYTES,
    TWO_AHEAD,
)
from .schedule import SEGMENT, MAX_REQUESTS, compiled_schedule
from .policy import _placement, _plan_windows


@dataclass(frozen=True)
class WindowPlan:
    calls: tuple
    workspace_bytes: int

    def run(self):
        for kernel, arguments in self.calls:
            kernel(*arguments)


@torch.no_grad()
def prepare(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    state_in,
    cu_seqlens,
    out,
    state_out,
    workspace_limit_bytes,
    *,
    state_indices=None,
    scale,
    eps,
    lower_bound,
):
    n, h = cu_seqlens.numel() - 1, q.shape[2]
    total = q.shape[1]
    if n > MAX_REQUESTS:
        raise ValueError(f"prefill supports at most {MAX_REQUESTS} requests")
    if n == 0:
        return WindowPlan((), 0)
    capacity = (total + SEGMENT - 1) // SEGMENT + n
    metadata_bytes = (5 * n + 1 + 4 * capacity + 8 * n * h + 32 * (capacity + 2)) * 4
    row_bytes = SEGMENT * h * 392 * 4
    slots = min(capacity, (workspace_limit_bytes - metadata_bytes) // row_bytes)
    if slots < 1:
        raise ValueError("workspace cannot hold metadata and one scheduled segment")
    capability = torch.cuda.get_device_capability(q.device)
    index = q.device.index
    state = dict(
        coeff=compiled_coeff_kernel(index, capability),
        scan=compiled_scan_kernel(index, capability, state_indices is not None),
        schedule=compiled_schedule(index, capability),
        workspace=torch.empty(
            slots * SEGMENT * h * 392, device=q.device, dtype=torch.float32
        ),
        prefix=torch.empty(n + 1, device=q.device, dtype=torch.int32),
        ranks=torch.empty(capacity, device=q.device, dtype=torch.int32),
        tasks=torch.zeros(capacity, 3, device=q.device, dtype=torch.int32),
        order=torch.empty(4 * n, device=q.device, dtype=torch.int32),
        flags=torch.zeros(
            8 * n * h + 32 * (capacity + 2), device=q.device, dtype=torch.int32
        ),
        capacity=capacity,
        max_requests=n,
        sms=torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    calls = bind(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        state_in,
        cu_seqlens,
        state,
        out,
        state_out,
        float(scale),
        float(eps),
        float(lower_bound),
        state_indices,
    )
    # Bound tensor views in calls retain all allocations throughout graph lifetime.
    return WindowPlan(calls, slots * row_bytes + metadata_bytes)


def bind(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    state_in,
    cu_seqlens,
    state,
    output,
    state_out,
    scale,
    eps,
    lower_bound,
    state_indices,
):
    calls = []
    p = state
    n, h = cu_seqlens.numel() - 1, q.shape[2]
    total = q.shape[1]
    capacity = (total + SEGMENT - 1) // SEGMENT + n
    assert capacity <= p["capacity"] and n <= p["max_requests"]
    prefix, order = (p["prefix"][: n + 1], p["order"][: 4 * n])
    slots = min(capacity, p["workspace"].numel() // (SEGMENT * h * 392))
    sms = p["sms"]
    rows, long_pairs = _placement(n, h, sms, max_rows=8)
    ctas = long_pairs * 8 + (n * h - long_pairs) * (16 // rows)
    radix4 = (3 if ctas <= 2 * sms else (2 if rows == 4 else 4)) | 64
    if rows == 2 or long_pairs:
        radix4 |= TWO_AHEAD
    if rows == 4 and not long_pairs and ctas <= sms:
        radix4 |= KPACK
    if long_pairs and ctas <= sms:
        radix4 |= PAIR_ENGINE
    smem = SMEM_TWO_PER_SM if ctas <= 2 * sms and not long_pairs else SMEM_BYTES
    plan = _plan_windows(slots, capacity, n, total, h, ctas > 3 * sms)
    # Per-CTA window chaining avoids the whole-grid tail when more than
    # one resident wave is needed. No length/head-specific cutoff.
    overlap = ctas > 2 * sms and len(plan) > 1
    if overlap:
        radix4 |= OVERLAP
    flags = p["flags"][:ctas]
    base = -(-ctas // 32) * 32
    counters = p["flags"][base : base + 32 * len(plan)]
    calls.append(
        (
            p["schedule"],
            (
                cu_seqlens,
                prefix,
                p["ranks"],
                p["tasks"],
                order,
                1,  # Length-proportional ranks for every runtime batch.
                p["flags"],
                base + 32 * len(plan) if overlap else 0,
            ),
        )
    )
    row = SEGMENT * h * 392
    for index, (start, count, offset, wait) in enumerate(plan):
        workspace = p["workspace"][offset * row : (offset + count) * row]
        if overlap and index > 0:
            wait += 2
        calls.append(
            (
                p["coeff"],
                (
                    q,
                    k,
                    g,
                    beta,
                    a_log,
                    dt_bias.view(-1),
                    workspace,
                    scale,
                    eps,
                    lower_bound,
                    cu_seqlens,
                    prefix,
                    p["tasks"],
                    start,
                    count,
                    wait,
                    counters,
                    index,
                ),
            )
        )
        calls.append(
            (
                p["scan"],
                (
                    v,
                    workspace,
                    state_out,
                    state_in,
                    output,
                    cu_seqlens,
                    prefix,
                    p["ranks"],
                    order,
                    state_indices,
                    start,
                    count,
                    rows,
                    long_pairs,
                    radix4,
                    flags,
                    counters,
                    index,
                    smem,
                ),
            )
        )
    return tuple(calls)
