"""The trainer's K31 indexer runs the rollout's top-k: DeepSelect, required (no fallback)."""

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.kernels import vllm_k31 as K

pytest.importorskip("vllm")


class _Linear(torch.nn.Module):
    def forward(self, x, w):
        return x @ w.t()


class TestIndexerTopkPairing:
    @pytest.mark.parametrize("tokens", [1, 7], ids=["one-token", "packed"])
    def test_calls_the_deepselect_topk(self, monkeypatch, tokens):
        heads, dim, hidden = 32, 8, 16
        calls = []

        def op(*args, **kwargs):
            calls.append((args, kwargs))
            return args[13]

        def quant(x, group, column_major_scales, use_ue8m0):
            return x.clone(), torch.ones(x.shape[0], 1)

        symbols = {"per_token_group_quant_fp8": quant, "_encode_layer_name": lambda n: n}
        monkeypatch.setattr(K, "_op", lambda namespace, name: op)
        monkeypatch.setattr(K, "_symbol", lambda module, name: symbols[name])
        adapter = K.DSAIndexerAdapter()
        adapter.linear = _Linear()
        metadata = SimpleNamespace(
            k_cache_prefix="layers.0.indexer.k_cache",
            attention_metadata=object(),
            k_cache=torch.zeros(1),
            max_model_len=64,
            max_total_seq_len=64,
            topk_buffer=torch.zeros(tokens, 4, dtype=torch.int32),
        )
        bf16 = dict(dtype=torch.bfloat16)
        adapter(
            torch.randn(tokens, hidden, **bf16),
            torch.randn(tokens, hidden, **bf16),
            wq=torch.randn(heads * dim, hidden, **bf16),
            wk=torch.randn(dim, hidden, **bf16),
            wv=torch.randn(heads, hidden, **bf16),
            k_norm_weight=torch.ones(dim),
            k_norm_bias=torch.zeros(dim),
            k_norm_eps=1e-6,
            num_heads=heads,
            head_dim=dim,
            topk=4,
            metadata=metadata,
        )
        ((args, kwargs),) = calls
        assert kwargs == {"topk_backend": "deep_select", "require_deep_select": True}
        # no PCP / DCP: the compressed logits rows start at column zero, as DeepSelect needs
        assert (args[15], args[19], args[20]) == (False, 0, 1)


class TestIndexerHeads:
    @pytest.mark.parametrize("heads", [4, 16, 64], ids=["4", "16", "64"])
    def test_only_the_k31_head_count(self, heads):
        adapter = K.DSAIndexerAdapter()
        bf16 = dict(dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="32 heads"):
            adapter(
                torch.randn(2, 16, **bf16),
                torch.randn(2, 16, **bf16),
                wq=torch.randn(heads * 8, 16, **bf16),
                wk=torch.randn(8, 16, **bf16),
                wv=torch.randn(heads, 16, **bf16),
                k_norm_weight=torch.ones(8),
                k_norm_bias=torch.zeros(8),
                k_norm_eps=1e-6,
                num_heads=heads,
                head_dim=8,
                topk=4,
                metadata=None,
            )
