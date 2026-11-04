"""Line-A owner VJPs vs FP32 references (relative error and cosine are printed).

The forward values come from rollout kernels; the references are autograd of
plain FP32 torch graphs of the same math on FP32 copies of the same inputs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.k31.vllm.primitive import (
    LoRAProjection,
    attn_res,
    attn_res_reference,
    gate_linear,
    rms_norm,
    situ_and_mul,
    visible_linear,
    visible_lora_linear,
    vllm_fold_lora_b,
)
from megatron.lite.model.k31.vllm.primitive._contract import (
    check_parameter_versions,
    parameter_versions,
)
from megatron.lite.model.k31.vllm.runtime_metadata import build_k31_runtime
from megatron.lite.primitive.kernels.vllm_k31 import (
    AttnResAdapter,
    BILinearAdapter,
    BIRouterAdapter,
    LoRALeafAdapter,
    PunicaLoRADeltaAdapter,
    RMSNormAdapter,
    SiTUAndMulAdapter,
)

COS_MIN = 0.9999
REL_MAX = 1e-2


def _metric(name: str, got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    g, r = got.double().flatten(), ref.double().flatten()
    rel = float((g - r).norm() / r.norm().clamp_min(1e-30))
    cos = float(F.cosine_similarity(g, r, dim=0))
    print(f"BWD_METRIC {name} rel={rel:.3e} cos={cos:.8f}", flush=True)
    return rel, cos


def _check(name, got, ref, rel_max=REL_MAX, cos_min=COS_MIN):
    rel, cos = _metric(name, got, ref)
    assert rel <= rel_max and cos >= cos_min, (name, rel, cos)


def _t(*shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)


def test_inference_tensor_versions_use_non_mutating_sentinel() -> None:
    with torch.inference_mode():
        parameter = torch.ones(2)
        versions = parameter_versions((parameter,))
        assert versions == (-1,)
        check_parameter_versions((parameter,), versions)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("k,n", [(7168, 49376), (12288, 7168), (1536, 18432), (33792, 7168)])
def test_frozen_linear_dx_and_no_dw(k31_bi, k, n) -> None:
    x = _t(64, k, seed=1).requires_grad_(True)
    w = _t(n, k, seed=2, scale=k**-0.5)  # frozen master
    y = visible_linear(BILinearAdapter(), x, w)
    dy = _t(64, n, seed=3)
    y.backward(dy)
    x32 = x.detach().float().requires_grad_(True)
    F.linear(x32, w.float()).backward(dy.float())
    _check(f"linear_dx[{k}x{n}]", x.grad, x32.grad)
    w_train = w.clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    visible_linear(BILinearAdapter(), x2, w_train).backward(dy)
    w32 = w.float().requires_grad_(True)
    F.linear(x.detach().float(), w32).backward(dy.float())
    _check(f"linear_dw[{k}x{n}]", w_train.grad, w32.grad)


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_router_dx_fp32(k31_bi) -> None:
    x = _t(64, 7168, seed=4).requires_grad_(True)
    w = _t(896, 7168, seed=5, scale=7168**-0.5)
    logits = gate_linear(BIRouterAdapter(), x, w)
    dy = _t(64, 896, seed=6, dtype=torch.float32)
    logits.backward(dy)
    x32 = x.detach().float().requires_grad_(True)
    F.linear(x32, w.float()).backward(dy)
    _check("router_dx", x.grad, x32.grad)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize(
    "k,sizes",
    [(7168, (12288, 12288, 12288, 12288, 128, 96)), (12288, (7168,)), (7168, (1536, 576))],
    ids=["kda_in", "o_proj", "dsa_qkv_a"],
)
def test_lora_leaf_grads(k31_bi, k, sizes) -> None:
    rank, alpha = 32, 64.0
    names = [f"s{i}" for i in range(len(sizes))]
    lora = LoRAProjection(k, names, sizes, rank=rank, alpha=alpha, max_rank=rank).cuda()
    with torch.no_grad():
        for i, name in enumerate(names):
            lora.lora_A[name].copy_(_t(rank, k, seed=10 + i, scale=k**-0.5))
            lora.lora_B[name].copy_(_t(sizes[i], rank, seed=20 + i, scale=0.05))
    w = _t(sum(sizes), k, seed=7, scale=k**-0.5)
    x = _t(64, k, seed=8).requires_grad_(True)
    meta = build_k31_runtime([64], "cuda", with_lora=True).lora_meta
    y = visible_lora_linear(PunicaLoRADeltaAdapter(), LoRALeafAdapter(), x, w, lora, meta)
    dy = _t(64, sum(sizes), seed=9)
    y.backward(dy)

    x32 = x.detach().float().requires_grad_(True)
    a32 = [lora.lora_A[n].detach().float().requires_grad_(True) for n in names]
    b32 = [lora.lora_B[n].detach().float().requires_grad_(True) for n in names]
    delta = torch.cat(
        [(x32 @ a32[i].T) @ (b32[i] * lora.scaling).T for i in range(len(names))], dim=1
    )
    (F.linear(x32, w.float()) + delta).backward(dy.float())
    _check(f"lora_dx[{k}]", x.grad, x32.grad)
    for i, n in enumerate(names):
        _check(f"lora_dA[{k}:{i}]", lora.lora_A[n].grad, a32[i].grad)
        _check(f"lora_dB[{k}:{i}]", lora.lora_B[n].grad, b32[i].grad)
    # the visible B is vLLM's folded bf16 tensor
    assert torch.equal(
        vllm_fold_lora_b(lora.lora_B[names[0]], lora.scaling),
        (lora.lora_B[names[0]].detach() * lora.scaling).to(torch.bfloat16),
    )


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("width", [7168, 3584, 1536, 512])
def test_rms_norm_grads(k31_bi, width) -> None:
    x = _t(64, width, seed=30, scale=2.0).requires_grad_(True)
    w = (1 + 0.1 * _t(width, seed=31).float()).to(torch.bfloat16).requires_grad_(True)
    y = rms_norm(RMSNormAdapter(), x, w, 1e-5)
    dy = _t(64, width, seed=32)
    y.backward(dy)
    x32 = x.detach().float().requires_grad_(True)
    w32 = w.detach().float().requires_grad_(True)
    F.rms_norm(x32, (width,), w32, 1e-5).backward(dy.float())
    _check(f"rmsnorm_dx[{width}]", x.grad, x32.grad)
    _check(f"rmsnorm_dw[{width}]", w.grad, w32.grad)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("width", [2 * 33792, 2 * 6144])
def test_situ_grads(k31_bi, width) -> None:
    x = _t(64, width, seed=40, scale=6.0).requires_grad_(True)
    y = situ_and_mul(SiTUAndMulAdapter(), x, beta=4.0, linear_beta=25.0)
    dy = _t(64, width // 2, seed=41)
    y.backward(dy)
    x32 = x.detach().float().requires_grad_(True)
    d = width // 2
    gate, up = x32[:, :d], x32[:, d:]
    ref = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))
    ref.backward(dy.float())
    _check(f"situ_dx[{width}]", x.grad, x32.grad)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("nb,has_delta,out_norm", [(0, False, True), (3, True, True), (8, True, False)])
def test_attn_res_grads(k31_bi, nb, has_delta, out_norm) -> None:
    h, t = 7168, 33
    blocks = [_t(t, h, seed=50 + i).requires_grad_(True) for i in range(nb)]
    prefix = _t(t, h, seed=60).requires_grad_(True)
    delta = _t(t, h, seed=61).requires_grad_(True) if has_delta else None
    norm = (1 + 0.1 * _t(h, seed=62).float()).to(torch.bfloat16)
    proj = _t(h, seed=63, scale=0.05)
    onorm = (1 + 0.1 * _t(h, seed=64).float()).to(torch.bfloat16) if out_norm else None
    out, new_prefix = attn_res(
        AttnResAdapter(), prefix, delta, blocks, norm, proj, onorm,
        num_blocks=nb, block_write_idx=-1, eps=1e-5, output_norm_eps=1e-5 if out_norm else 0.0,
    )
    dy = _t(t, h, seed=65)
    dp = _t(t, h, seed=66) if has_delta else None
    torch.autograd.backward(
        [out, new_prefix] if has_delta else [out], [dy, dp] if has_delta else [dy]
    )
    # FP32 reference: same graph on FP32 leaves (no bf16 rounding of the prefix add).
    p32 = prefix.detach().float().requires_grad_(True)
    d32 = delta.detach().float().requires_grad_(True) if has_delta else None
    b32 = [b.detach().float().requires_grad_(True) for b in blocks]
    ref_out, ref_prefix = attn_res_reference(
        p32, d32, b32, norm, proj, onorm, num_blocks=nb, eps=1e-5,
        output_norm_eps=1e-5 if out_norm else 0.0,
    )
    torch.autograd.backward(
        [ref_out, ref_prefix] if has_delta else [ref_out],
        [dy.float(), dp.float()] if has_delta else [dy.float()],
    )
    _check(f"attn_res_dprefix[nb={nb}]", prefix.grad, p32.grad)
    if has_delta:
        _check(f"attn_res_ddelta[nb={nb}]", delta.grad, d32.grad)
    for i, b in enumerate(blocks):
        _check(f"attn_res_dblock{i}[nb={nb}]", b.grad, b32[i].grad)
