"""KDA backward recompute: the gated RMSNorm uses the weight the forward used (rounded to BF16)."""

from __future__ import annotations

import sys
import types

import torch

from megatron.lite.model.k31.vllm.primitive.kda import KDAKernelConfig, fla_functional


def _fake_chunk_kda(*, q, k, v, g, beta, **kwargs):
    gate = torch.sigmoid(q.float() + k.float() + g.float()) * beta.unsqueeze(-1)
    return (v.float() * gate).to(v.dtype), None


def _inputs(cfg: KDAKernelConfig, total: int):
    g = torch.Generator().manual_seed(3)
    hd, h, w = cfg.projection_size, cfg.num_heads, cfg.conv_width

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g).to(dtype)

    return [
        rnd(total, 3 * hd),
        rnd(total, hd),
        rnd(total, h),
        rnd(total, hd),
        rnd(3 * hd, w, dtype=torch.float32),
        rnd(h, dtype=torch.float32),
        rnd(hd, dtype=torch.float32),
    ]


def _run(cfg, cu_seqlens, inputs, norm_weight):
    leaves = [t.clone().requires_grad_(i in (0, 1, 2, 3)) for i, t in enumerate(inputs)]
    out = fla_functional(cfg, cu_seqlens)(*leaves, norm_weight)
    out.float().square().sum().backward()
    return [out.detach(), *(leaves[i].grad for i in range(4))]


def test_recompute_norm_weight_is_rounded_like_the_forward(monkeypatch):
    fake = types.ModuleType("fla.ops.kda")
    fake.chunk_kda = _fake_chunk_kda
    monkeypatch.setitem(sys.modules, "fla.ops.kda", fake)
    cfg = KDAKernelConfig(num_heads=1)
    cu_seqlens = torch.tensor([0, 17, 40], dtype=torch.int32)
    inputs = _inputs(cfg, 40)
    weight = torch.empty(cfg.head_dim).uniform_(
        0.5, 2.0, generator=torch.Generator().manual_seed(5)
    )
    rounded = weight.to(torch.bfloat16).float()
    assert not torch.equal(weight, rounded)
    got = _run(cfg, cu_seqlens, inputs, weight)
    want = _run(cfg, cu_seqlens, inputs, rounded)
    for name, a, b in zip(
        ("out", "mixed_qkv", "g1", "beta", "g2"), got, want, strict=True
    ):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
