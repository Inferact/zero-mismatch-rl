"""Independent mathematical recurrence: FP64 state, no per-token BF16 rounding."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def oracle(Q, K, V, G, Beta, A, Bias, S0, Cu, O, S1, H, R: tl.constexpr = 8):
    bh = tl.program_id(0)
    rv = tl.program_id(1) * R + tl.arange(0, R)
    seq = bh // H
    head = bh % H
    kk = tl.arange(0, 128)
    start = tl.load(Cu + seq)
    end = tl.load(Cu + seq + 1)
    s = tl.load(S0 + bh * 16384 + rv[:, None] * 128 + kk[None, :]).to(tl.float64)
    al = lib.exp(tl.load(A + head).to(tl.float64))
    bias = tl.load(Bias + head * 128 + kk).to(tl.float64)
    for t in range(start, end):
        off = (t * H + head) * 128 + kk
        q = tl.load(Q + off).to(tl.float64)
        k = tl.load(K + off).to(tl.float64)
        q = q / lib.sqrt(tl.sum(q * q, 0) + 1.0e-6) * 0.08838834764831845
        k = k / lib.sqrt(tl.sum(k * k, 0) + 1.0e-6)
        raw = tl.load(G + off).to(tl.float64) + bias
        gate = lib.exp(-5.0 / (1.0 + lib.exp(-al * raw)))
        beta = 1.0 / (1.0 + lib.exp(-tl.load(Beta + t * H + head).to(tl.float64)))
        s = s * gate[None, :]
        v = tl.load(V + (t * H + head) * 128 + rv).to(tl.float64)
        delta = (v - tl.sum(s * k[None, :], 1)) * beta
        s = s + delta[:, None] * k[None, :]
        y = tl.sum(s * q[None, :], 1)
        tl.store(O + (t * H + head) * 128 + rv, y)
    tl.store(S1 + bh * 16384 + rv[:, None] * 128 + kk[None, :], s)


@torch.no_grad()
def run(
    q, k, v, g, beta, a_log, dt_bias, state_in, cu_seqlens, *, full_precision=False
):
    dtype = torch.float64 if full_precision else torch.bfloat16
    output = torch.empty(q.shape, dtype=dtype, device=q.device)
    state = torch.empty(state_in.shape, dtype=dtype, device=q.device)
    oracle[(state_in.shape[0] * q.shape[2], 16)](
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        state_in,
        cu_seqlens,
        output,
        state,
        q.shape[2],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return {"output": output, "state_out": state}
