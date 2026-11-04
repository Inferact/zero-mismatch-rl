"""AttnRes: functional training owner == the rollout's in-place native op, bit for bit."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from megatron.lite.model.k31.vllm.primitive import attn_res
from megatron.lite.primitive.kernels import vllm_k31
from megatron.lite.primitive.kernels.vllm_k31 import AttnResAdapter

H = 7168
EPS = 1e-5


def _t(*shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)


def _weights(seed):
    norm = (1.0 + 0.1 * _t(H, seed=seed).float()).to(torch.bfloat16)
    proj = _t(H, seed=seed + 1, scale=0.05)
    out_norm = (1.0 + 0.1 * _t(H, seed=seed + 2).float()).to(torch.bfloat16)
    return norm, proj, out_norm


def test_attn_res_adapter_rejects_wrong_hidden_before_lookup(monkeypatch) -> None:
    lookup = Mock()
    monkeypatch.setattr(vllm_k31, "_k31_symbol", lookup)
    prefix = torch.zeros(2, 16, dtype=torch.bfloat16)
    blocks = torch.zeros(2, 1, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="hidden"):
        AttnResAdapter()(prefix, None, blocks, torch.ones(16), torch.ones(16), None, 0, -1, EPS, EPS)
    lookup.assert_not_called()


def _vllm_call(prefix, delta, blocks, slots, norm, proj, out_norm, nb, write):
    """The rollout call on in-place buffers (as the vLLM decoder layer does)."""

    fn = vllm_k31._k31_symbol("ops.attn_res", "attn_res")
    t = prefix.shape[0]
    bank = prefix.new_zeros((t, slots, H))
    for i, b in enumerate(blocks):
        bank[:, i].copy_(b)
    p = prefix.clone()
    out = fn(p, delta, bank, norm, proj, out_norm, nb, write, EPS, EPS if out_norm is not None else 0.0)
    return out, p, bank


CASES = [
    # (num_blocks, has_delta, write, out_norm)
    (0, False, 0, True),  # layer 0 pre-attn
    (1, False, -1, True),  # layer 0 post-attn (block-write layer)
    (1, True, -1, True),  # layer 1 pre/post-attn
    (1, True, 1, True),  # layer 12 pre-attn (block write)
    (2, False, -1, True),  # layer 12 post-attn
    (4, True, -1, True),
    (7, True, 7, True),  # layer 84 pre-attn
    (8, True, -1, True),  # layer 84+ inside the last block
    (8, True, -1, False),  # head aggregation (output_norm None)
    (0, True, -1, False),
]


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("nb,has_delta,write,out_norm", CASES)
@pytest.mark.parametrize("tokens", [1, 7, 300])
def test_functional_owner_equals_inplace_rollout_op(k31_bi, nb, has_delta, write, out_norm, tokens):
    norm, proj, onorm = _weights(11)
    onorm = onorm if out_norm else None
    blocks = tuple(_t(tokens, H, seed=100 + i) for i in range(nb))
    prefix = _t(tokens, H, seed=20)
    delta = _t(tokens, H, seed=21) if has_delta else None
    slots = max(8, nb, write + 1)
    ref_out, ref_prefix, ref_bank = _vllm_call(prefix, delta, blocks, slots, norm, proj, onorm, nb, write)
    out, new_prefix = attn_res(
        AttnResAdapter(),
        prefix,
        delta,
        blocks,
        norm,
        proj,
        onorm,
        num_blocks=nb,
        block_write_idx=write,
        eps=EPS,
        output_norm_eps=EPS if out_norm else 0.0,
    )
    assert torch.equal(out, ref_out)
    assert torch.equal(new_prefix, ref_prefix)
    if write >= 0:
        assert torch.equal(ref_bank[:, write], new_prefix)
    # value semantics: no caller buffer mutated
    assert torch.equal(prefix, _t(tokens, H, seed=20))


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_attn_res_rows_are_batch_invariant(k31_bi) -> None:
    norm, proj, onorm = _weights(31)
    blocks = tuple(_t(257, H, seed=200 + i) for i in range(5))
    prefix, delta = _t(257, H, seed=40), _t(257, H, seed=41)
    big, _ = attn_res(
        AttnResAdapter(), prefix, delta, blocks, norm, proj, onorm,
        num_blocks=5, block_write_idx=-1, eps=EPS, output_norm_eps=EPS,
    )
    for m in (1, 2, 33):
        small, _ = attn_res(
            AttnResAdapter(), prefix[:m].clone(), delta[:m].clone(),
            tuple(b[:m].clone() for b in blocks), norm, proj, onorm,
            num_blocks=5, block_write_idx=-1, eps=EPS, output_norm_eps=EPS,
        )
        assert torch.equal(small, big[:m]), m


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_decoder_stream_equals_vllm_inplace_sequence(k31_bi) -> None:
    """26 layers (3 block boundaries) of vLLM's in-place pre/post sequence vs the
    functional K31 stream, with random stand-ins for attention / MLP outputs."""

    from megatron.lite.model.k31.vllm.model import _AttnResWeights, _attn_res_call
    from megatron.lite.model.k31.config import K31Config

    cfg = K31Config(num_hidden_layers=26, attn_res_block_size=12, kda_layers=list(range(26)))
    t = 19
    fn = vllm_k31._k31_symbol("ops.attn_res", "attn_res")
    sites = []
    for layer in range(cfg.num_hidden_layers):
        pre, post = _AttnResWeights(H).cuda(), _AttnResWeights(H).cuda()
        with torch.no_grad():
            for i, site in enumerate((pre, post)):
                site.norm_weight.copy_(_weights(1000 + 10 * layer + i)[0])
                site.proj_weight.copy_(_weights(1000 + 10 * layer + i)[1].unsqueeze(0))
        ln_in = (1 + 0.1 * _t(H, seed=5000 + layer).float()).to(torch.bfloat16)
        ln_post = (1 + 0.1 * _t(H, seed=6000 + layer).float()).to(torch.bfloat16)
        sites.append((pre, post, ln_in, ln_post))
    attn_outs = [_t(t, H, seed=7000 + i) for i in range(cfg.num_hidden_layers)]
    mlp_outs = [_t(t, H, seed=8000 + i) for i in range(cfg.num_hidden_layers)]
    embed = _t(t, H, seed=9000)

    # vLLM (model forward / decoder layer, in place)
    nblk = cfg.num_attn_res_blocks
    bank = embed.new_empty((t, nblk, H))
    prefix_sum, hidden = embed.clone(), None
    ref_inputs = []
    for layer in range(cfg.num_hidden_layers):
        pre, post, ln_in, ln_post = sites[layer]
        is_w = cfg.is_block_write_layer(layer)
        h = fn(prefix_sum, hidden, bank, pre.norm_weight, pre.proj_weight.squeeze(0), ln_in,
               cfg.prev_valid_blocks(layer), cfg.block_write_idx(layer) if is_w else -1, EPS, EPS)
        a = attn_outs[layer]
        if is_w:
            prefix_sum, d = a.clone(), None
        else:
            d = a
        h2 = fn(prefix_sum, d, bank, post.norm_weight, post.proj_weight.squeeze(0), ln_post,
                cfg.prev_valid_blocks(layer) + int(is_w), -1, EPS, EPS)
        ref_inputs.append((h.clone(), h2.clone()))
        hidden = mlp_outs[layer]
    head_w = _weights(77)
    ref_head = fn(prefix_sum, hidden, bank, head_w[0], head_w[1], None, nblk, -1, EPS, 0.0)

    # K31 functional stream (model.K31Layer logic)
    class _LN:
        def __init__(self, w):
            self.weight, self.eps = w, EPS

    prefix, delta, blocks = embed, None, ()
    for layer in range(cfg.num_hidden_layers):
        pre, post, ln_in, ln_post = sites[layer]
        is_w = cfg.is_block_write_layer(layer)
        h, prefix = _attn_res_call(prefix, delta, blocks, pre, _LN(ln_in),
                                   num_blocks=cfg.prev_valid_blocks(layer),
                                   block_write_idx=cfg.block_write_idx(layer) if is_w else -1, eps=EPS)
        if is_w:
            blocks = (*blocks, prefix)
        a = attn_outs[layer]
        if is_w:
            prefix, d = a, None
        else:
            d = a
        h2, prefix = _attn_res_call(prefix, d, blocks, post, _LN(ln_post),
                                    num_blocks=cfg.prev_valid_blocks(layer) + int(is_w),
                                    block_write_idx=-1, eps=EPS)
        assert torch.equal(h, ref_inputs[layer][0]), f"layer {layer} pre"
        assert torch.equal(h2, ref_inputs[layer][1]), f"layer {layer} post"
        delta = mlp_outs[layer]
    head_site = _AttnResWeights(H).cuda()
    with torch.no_grad():
        head_site.norm_weight.copy_(head_w[0])
        head_site.proj_weight.copy_(head_w[1].unsqueeze(0))
    head, _ = _attn_res_call(prefix, delta, blocks, head_site, None,
                             num_blocks=nblk, block_write_idx=-1, eps=EPS)
    assert len(blocks) == nblk
    assert torch.equal(head, ref_head)
