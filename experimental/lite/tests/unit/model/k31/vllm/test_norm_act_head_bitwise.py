"""RMSNorm, SiTU, embedding and selected-token logprob forwards are the rollout's."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.k31.vllm.model import _vllm_selected_log_probs
from megatron.lite.model.k31.vllm.primitive import rms_norm, situ_and_mul, situ_and_mul_reference
from megatron.lite.primitive.kernels.vllm_k31 import RMSNormAdapter, SiTUAndMulAdapter

EPS = 1e-5
NORM_WIDTHS = (7168, 3584, 1536, 512)  # final / latent / q_a / kv_a
SITU_WIDTHS = (2 * 33792, 2 * 6144)  # dense layer-0 gate_up, shared experts


def _t(*shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("width", NORM_WIDTHS)
def test_rms_norm_is_rollout_bi_kernel(k31_bi, width) -> None:
    from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant

    x = _t(513, width, seed=1, scale=3.0)
    w = (1 + 0.1 * _t(width, seed=2).float()).to(torch.bfloat16)
    ref = rms_norm_batch_invariant(x, w, EPS)
    out = rms_norm(RMSNormAdapter(), x, w, EPS)
    assert torch.equal(out, ref)
    for m in (1, 5, 64):
        assert torch.equal(rms_norm(RMSNormAdapter(), x[:m].clone(), w, EPS), ref[:m])


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("width", SITU_WIDTHS)
def test_situ_is_the_cuda_op_not_the_native_form(k31_bi, width) -> None:
    x = _t(513, width, seed=3, scale=6.0)
    ref = torch.empty(513, width // 2, dtype=torch.bfloat16, device="cuda")
    torch.ops._C.situ_and_mul(ref, x, 4.0, 25.0)
    out = situ_and_mul(SiTUAndMulAdapter(), x, beta=4.0, linear_beta=25.0)
    assert torch.equal(out, ref)
    for m in (1, 7):
        assert torch.equal(
            situ_and_mul(SiTUAndMulAdapter(), x[:m].clone(), beta=4.0, linear_beta=25.0), ref[:m]
        )
    native = situ_and_mul_reference(x, 4.0, 25.0)
    diff = (native.float() - out.float()).abs()
    # Same math; the CUDA kernel may differ from the torch form by 1 bf16 ulp.
    assert float(diff.max()) <= float(out.float().abs().max()) * 2**-7


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("temperature", [1.0, 0.7, 1.1], ids=["t1", "t0.7", "t1.1"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16-logits", "fp32-logits"])
def test_selected_logprob_value_is_rollout_and_grad_is_log_softmax(k31_bi, dtype, temperature) -> None:
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    vocab = 163840
    logits = _t(37, vocab, seed=5, scale=4.0, dtype=dtype).requires_grad_(True)
    labels = torch.randint(0, vocab, (37,), device="cuda")
    selected, _ = _vllm_selected_log_probs(logits, labels, temperature)
    processed = logits.detach()
    if temperature != 1.0:
        processed = torch.empty_like(processed, dtype=torch.float32).copy_(processed)
        rows = torch.zeros(37, dtype=torch.int32, device="cuda")
        apply_temperature(processed, rows, torch.tensor([temperature], device="cuda"))
    ref = compute_token_logprobs(processed, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(selected.detach(), ref)
    selected.sum().backward()
    ref_logits = logits.detach().float().requires_grad_(True)
    F.log_softmax(ref_logits / temperature, -1).gather(-1, labels.unsqueeze(-1)).sum().backward()
    assert torch.equal(logits.grad.float(), ref_logits.grad.to(dtype).float())


def _near_certain_logits(n: int, vocab: int, seed: int, dtype: torch.dtype):
    """Rows like an LM's p ~ 1 tokens, plus ordinary rows.

    The first ``n`` rows have one dominant logit and 16 runner-up tokens right after
    it (same 1024-wide kernel block) that carry k * 2^-23 of its mass, k in
    [0.14, 150]; the background is negligible.  The selected-token logprob is then
    a few FP32 ulps of 2^-23, and the in-block reduction order decides its last
    bits.  The last ``n // 4`` rows are ordinary random rows.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, vocab, device="cuda", generator=g) * 2.0 - 30.0
    labels = torch.randint(0, vocab, (n,), device="cuda", generator=g)
    top = torch.full((n,), 20.0, device="cuda")
    k = torch.exp(torch.linspace(-2.0, 5.0, n, device="cuda"))
    w = torch.rand(n, 16, device="cuda", generator=g) + 0.05
    w = w / w.sum(1, keepdim=True)
    runner = (labels[:, None] + torch.arange(1, 17, device="cuda")) % vocab
    x.scatter_(1, runner, top[:, None] + torch.log(w * k[:, None] * 2.0**-23))
    x.scatter_(1, labels[:, None], top[:, None])
    ordinary = torch.randn(n // 4, vocab, device="cuda", generator=g) * 4.0
    ordinary_labels = torch.randint(0, vocab, (n // 4,), device="cuda", generator=g)
    return torch.cat([x, ordinary]).to(dtype), torch.cat([labels, ordinary_labels])


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().contiguous().view(torch.int32)


class TestNearCertainSelectedLogprob:
    """Training value == rollout value for p ~ 1 tokens, whatever the rollout batch was.

    Rollout scores each token in a decode batch of <= max_num_seqs rows; training
    scores the whole packed micro-batch at once.  A full-model run failed on one
    such token: the straight-through sum ``r + (v - r)`` rounded the kernel value
    0xB3FFFFFF to 0xB4000000.  Slice gates never produce these tokens.
    """

    @pytest.mark.gpus(1, min_architecture="blackwell")
    @pytest.mark.parametrize(
        "dtype", [torch.bfloat16, torch.float32], ids=["bf16-raw-logits", "fp32-processed-logits"]
    )
    @pytest.mark.parametrize("rollout_rows", [1, 5, 16], ids=["decode-1", "decode-5", "decode-16"])
    def test_value_is_rollout_kernel_bitwise(self, k31_bi, dtype, rollout_rows) -> None:
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

        logits, labels = _near_certain_logits(4096, 163840, seed=7, dtype=dtype)
        n = logits.shape[0]
        # rollout: rows arrive in a shuffled order, a few at a time
        perm = torch.randperm(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
        rollout = torch.empty(n, dtype=torch.float32, device="cuda")
        for i in range(0, n, rollout_rows):
            rows = perm[i : i + rollout_rows]
            rollout[rows] = compute_token_logprobs(logits[rows], labels[rows].unsqueeze(-1)).squeeze(-1)
        # training: one packed call with gradient
        x = logits.clone().requires_grad_(True)
        selected, token_log_probs = _vllm_selected_log_probs(x, labels, 1.0)
        assert torch.equal(_bits(selected), _bits(rollout))

        reference = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).detach()
        confident = (rollout > -(2.0**-18)) & (rollout < 0)
        assert int(confident.sum()) >= 1500  # the regime is really exercised
        if dtype is torch.bfloat16:  # this seed: the old join r + (v - r) loses the value on 12 rows
            old_sum = reference + (rollout - reference)
            assert int((_bits(old_sum) != _bits(rollout)).sum()) > 0

        selected.sum().backward()
        ref_logits = logits.detach().float().requires_grad_(True)
        F.log_softmax(ref_logits, -1).gather(-1, labels.unsqueeze(-1)).sum().backward()
        assert torch.equal(x.grad.float(), ref_logits.grad.to(dtype).float())


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_embedding_is_plain_lookup(k31_bi) -> None:
    from megatron.lite.model.k31.vllm.model import _EmbeddingState

    emb = _EmbeddingState(1000, 7168).cuda()
    with torch.no_grad():
        emb.weight.copy_(_t(1000, 7168, seed=9))
    ids = torch.randint(0, 1000, (64,), device="cuda")
    assert torch.equal(emb(ids), emb.weight[ids])
