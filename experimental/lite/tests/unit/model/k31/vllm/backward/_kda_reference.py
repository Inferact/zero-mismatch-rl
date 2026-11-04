"""Sequential-recurrence reference for the K31 KDA primitive (conv -> KDA -> gated norm), any float dtype.

Math (per head, state S [K, V], token order):
  q, k = l2norm(conv q, k) (eps 1e-6 under the sqrt), q *= D**-0.5, v = conv v
  a_t = lower * sigmoid(exp(A_log) * (g1_t + dt_bias))      (log decay, per channel k)
  beta_t = sigmoid(raw beta_t)
  S <- S * exp(a_t)[:, None];  u = beta_t (v_t - S^T k_t);  S <- S + k_t u^T;  o_t = S^T q_t
  out_t = o_t * rsqrt(mean(o_t^2) + eps) * w * sigmoid(g2_t)
Heads are independent end to end, so the reference runs in head groups to bound memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _conv_silu(x: torch.Tensor, w: torch.Tensor, cu: list[int]) -> torch.Tensor:
    W = w.shape[1]
    out = []
    for s, e in zip(cu[:-1], cu[1:]):
        seq = x[s:e].T.unsqueeze(0)
        y = F.conv1d(F.pad(seq, (W - 1, 0)), w.unsqueeze(1), groups=w.shape[0])
        out.append(F.silu(y)[0].T)
    return torch.cat(out)


def kda_reference(
    mixed_qkv, g1, beta, g2, conv_weight, A_log, dt_bias, norm_weight, cu, *, num_heads, head_dim, lower, eps,
    heads: slice | None = None,
):
    """Returns [T, Hs*D] in the dtype of the (already cast) inputs for heads ``heads`` (default all)."""
    H, D = num_heads, head_dim
    HD = H * D
    hs = heads or slice(0, H)
    h0, h1 = hs.start, hs.stop
    n = h1 - h0
    T = mixed_qkv.shape[0]
    cols = slice(h0 * D, h1 * D)
    w = conv_weight
    q = _conv_silu(mixed_qkv[:, :HD][:, cols], w[:HD][cols], cu).reshape(T, n, D)
    k = _conv_silu(mixed_qkv[:, HD:2 * HD][:, cols], w[HD:2 * HD][cols], cu).reshape(T, n, D)
    v = _conv_silu(mixed_qkv[:, 2 * HD:][:, cols], w[2 * HD:][cols], cu).reshape(T, n, D)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    a = lower * torch.sigmoid(A_log[h0:h1].exp()[None, :, None] * (g1[:, cols].reshape(T, n, D)
                                                                      + dt_bias[cols].reshape(n, D)))
    b = torch.sigmoid(beta[:, h0:h1])
    decay = a.exp()
    outs = []
    for s, e in zip(cu[:-1], cu[1:]):
        S = q.new_zeros(n, D, D)
        for t in range(s, e):
            S = S * decay[t].unsqueeze(-1)
            kv = (k[t].unsqueeze(-1) * S).sum(-2)
            u = b[t].unsqueeze(-1) * (v[t] - kv)
            S = S + k[t].unsqueeze(-1) * u.unsqueeze(-2)
            outs.append((q[t].unsqueeze(-1) * S).sum(-2))
    o = torch.stack(outs)  # [T, n, D]
    y = o * torch.rsqrt(o.square().sum(-1, keepdim=True) / D + eps)
    y = y * norm_weight.to(o.dtype) * torch.sigmoid(g2[:, cols].reshape(T, n, D))
    return y.reshape(T, n * D)
