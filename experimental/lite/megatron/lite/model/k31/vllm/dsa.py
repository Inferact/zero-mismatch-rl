"""DSA (MLA + indexer) attention slot (skeleton stub).

Parameter names/shapes mirror the rollout's TP=1 layout:

* ``fused_qkv_a_proj``: ``K31Linear`` 7168 -> q_a (1536) + kv_a_proj_with_mqa (576);
* ``q_a_layernorm.weight`` [1536], ``kv_a_layernorm.weight`` [512] (bf16);
* ``q_b_proj``: 1536 -> 96 x 192;  ``kv_b_proj``: 512 -> 96 x 256 (never LoRA,
  read as absorbed W_UK / W_UV);  ``g_proj``: 7168 -> 12288;  ``o_proj``: 12288 -> 7168;
* ``indexer.wq_b`` 1536 -> 32 x 128, ``indexer.wk`` 7168 -> 128,
  ``indexer.k_norm.{weight,bias}`` [128], ``indexer.weights_proj`` 7168 -> 32
  (frozen, no LoRA).

Forward is the rollout's ``MultiHeadLatentAttentionWrapper.forward``
for a sparse, NoPE, gated layer, op for op: fused_qkv_a -> split -> q_a / kv_a
RMSNorm -> q_b -> frozen indexer top-k -> absorbed sparse MLA (fp8_ds_mla cache,
pinned FlashMLA fp8 kernel) -> ``attn * sigmoid(g_proj(h))`` -> o_proj.  The
attention core and its backward live in ``primitive/attention.py``; the
per-batch kernel metadata is built here (``K31DSAMetadataBuilder``).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.model import K31Linear, K31LoRASpec
from megatron.lite.model.k31.vllm.primitive.attention import (
    absorbed_weights,
    attention_core,
    default_dsa_adapters,
    indexer_topk,
    output_gate,
)
from megatron.lite.model.k31.vllm.primitive.norm import rms_norm
from megatron.lite.model.k31.vllm.runtime_metadata import (
    K31Runtime,
    initialize_k31_vllm_batch_invariance,
)
from megatron.lite.primitive.kernels.vllm_k31 import (
    DSA_BLOCK_SIZE,
    FP8_DS_MLA_TOKEN_BYTES,
    RMSNormAdapter,
    _symbol,
)
from megatron.lite.primitive.parallel import ParallelState


class _Norm(nn.Module):
    def __init__(self, size: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16))


class _Indexer(nn.Module):
    def __init__(self, config: K31Config) -> None:
        super().__init__()
        h = config.hidden_size
        width = config.index_num_heads * config.index_head_dim
        self.wq_b = K31Linear(config.q_lora_rank, (width,), ("wq_b",))
        self.wk = K31Linear(h, (config.index_head_dim,), ("wk",))
        self.k_norm = _Norm(config.index_head_dim, bias=True)
        self.weights_proj = K31Linear(h, (config.index_num_heads,), ("weights_proj",))


class K31DSAAttention(nn.Module):
    def __init__(
        self,
        config: K31Config,
        *,
        layer_idx: int,
        ps: ParallelState | None,
        selected_stages: frozenset[str],
        lora: K31LoRASpec | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ps = ps
        self.selected_stages = selected_stages
        h = config.hidden_size
        heads = config.num_attention_heads
        qk = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.fused_qkv_a_proj = K31Linear(
            h,
            (config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim),
            ("q_a_proj", "kv_a_proj_with_mqa"),
            lora=lora,
        )
        self.q_a_layernorm = _Norm(config.q_lora_rank)
        self.kv_a_layernorm = _Norm(config.kv_lora_rank)
        self.q_b_proj = K31Linear(config.q_lora_rank, (heads * qk,), ("q_b_proj",), lora=lora)
        self.kv_b_proj = K31Linear(
            config.kv_lora_rank,
            (heads * (config.qk_nope_head_dim + config.v_head_dim),),
            ("kv_b_proj",),
        )
        self.g_proj = K31Linear(h, (heads * config.v_head_dim,), ("g_proj",), lora=lora)
        self.o_proj = K31Linear(heads * config.v_head_dim, (h,), ("o_proj",), lora=lora)
        self.indexer = _Indexer(config)

    def forward(
        self, hidden_states: torch.Tensor, *, metadata: Any, runtime: K31Runtime
    ) -> torch.Tensor:
        if not isinstance(metadata, K31DSARuntimeMetadata):
            raise TypeError("K31 DSA layers need K31DSARuntimeMetadata (build_dsa_metadata)")
        cfg = self.config
        if not cfg.mla_use_nope or not cfg.mla_use_output_gate:
            raise NotImplementedError("K31 DSA forward is ported for NoPE + output gate only")
        if cfg.index_k_norm_type != "layernorm":
            raise NotImplementedError("K31 DSA indexer k-norm must be layernorm")
        heads = cfg.num_attention_heads
        nope, rope = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
        eps = cfg.rms_norm_eps
        adapters = self._adapters

        qkv_lora = self.fused_qkv_a_proj(hidden_states, runtime)
        q_c, kv_lora = qkv_lora.split([cfg.q_lora_rank, cfg.kv_lora_rank + rope], dim=-1)
        q_c = rms_norm(RMSNormAdapter(), q_c, self.q_a_layernorm.weight, eps)
        kv_c, k_pe = kv_lora.split([cfg.kv_lora_rank, rope], dim=-1)
        kv_c_normed = rms_norm(RMSNormAdapter(), kv_c, self.kv_a_layernorm.weight, eps)
        q = self.q_b_proj(q_c, runtime).view(-1, heads, nope + rope)

        indexer = self.indexer
        topk = indexer_topk(
            adapters.indexer,
            hidden_states,
            q_c,
            weights={
                "wq": indexer.wq_b.weight,
                "wk": indexer.wk.weight,
                "wv": indexer.weights_proj.weight,
                # The rollout stores the k-norm affine in fp32.
                "k_norm_weight": indexer.k_norm.weight.detach().float(),
                "k_norm_bias": indexer.k_norm.bias.detach().float(),
                "k_norm_eps": eps,
            },
            num_heads=cfg.index_num_heads,
            head_dim=cfg.index_head_dim,
            topk=cfg.index_topk,
            metadata=metadata.indexer,
        )
        w_uk_t, w_uv = absorbed_weights(
            self.kv_b_proj.weight,
            kv_lora_rank=cfg.kv_lora_rank,
            num_heads=heads,
            qk_nope_head_dim=nope,
            v_head_dim=cfg.v_head_dim,
        )
        attn = attention_core(
            adapters,
            q,
            kv_c_normed,
            k_pe,
            topk,
            metadata,
            w_uk_t=w_uk_t,
            w_uv=w_uv,
            softmax_scale=(nope + rope) ** -0.5,
            qk_nope_head_dim=nope,
            padded_heads=_padded_heads(heads),
        )
        gated = output_gate(attn, self.g_proj(hidden_states, runtime))
        return self.o_proj(gated, runtime)

    @property
    def _adapters(self):
        adapters = self.__dict__.get("_dsa_adapters")
        if adapters is None:
            adapters = default_dsa_adapters()
            self.__dict__["_dsa_adapters"] = adapters
        return adapters


def _padded_heads(heads: int) -> int:
    # FlashMLA SM100 fp8 sparse decode pads h_q to 64 or 128 (the rollout pads 96 -> 128).
    if heads <= 64:
        return 64
    if heads <= 128:
        return 128
    raise NotImplementedError(f"K31 DSA supports at most 128 heads at TP=1, got {heads}")


# ---------------------------------------------------------------------------
# per-batch DSA kernel metadata (packed fresh prefills, context 0)
# ---------------------------------------------------------------------------
DSA_INDEXER_HEAD_DIM = 128
DSA_INDEXER_TOKEN_BYTES = DSA_INDEXER_HEAD_DIM + 4


@dataclass
class K31IndexerRuntimeMetadata:
    k_cache_prefix: str
    k_cache: torch.Tensor
    attention_metadata: Any
    topk_buffer: torch.Tensor
    max_model_len: int
    max_total_seq_len: int


@dataclass
class K31DSARuntimeMetadata:
    token_counts: tuple[int, ...]
    block_size: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    req_id_per_token: torch.Tensor
    query_start_loc: torch.Tensor
    workspace_starts: torch.Tensor
    positions: torch.Tensor
    mla_cache: torch.Tensor
    k_scale: torch.Tensor
    cache_lens: torch.Tensor
    dummy_block_table: torch.Tensor
    indexer: K31IndexerRuntimeMetadata


class K31DSAMetadataBuilder:
    """Build per-call DSA metadata for a packed batch of fresh requests."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_model_len: int,
        topk: int = 2048,
        block_size: int = DSA_BLOCK_SIZE,
        max_logits_mb: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.max_model_len = int(max_model_len)
        self.topk = int(topk)
        self.block_size = int(block_size)
        if max_logits_mb is None:
            max_logits_mb = importlib.import_module("vllm.envs").VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
        self.max_logits_bytes = int(max_logits_mb) * 1024 * 1024
        # vLLM get_max_prefill_buffer_size: max_model_len * 40.
        self.max_total_seq_len = self.max_model_len * 40

    def _indexer_metadata(
        self,
        token_counts: list[int],
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Any:
        module = "vllm.v1.attention.backends.mla.indexer"
        metadata_cls = _symbol(module, "DeepseekV32IndexerMetadata")
        prefill_cls = _symbol(module, "DeepseekV32IndexerPrefillMetadata")
        builder_cls = _symbol(module, "DeepseekV32IndexerMetadataBuilder")
        build_chunk = _symbol(module, "build_prefill_chunk_metadata")
        seq_lens_cpu = torch.tensor(token_counts, dtype=torch.int32)
        query_start_loc_cpu = query_start_loc.cpu()
        # All requests are prefills with query_len == seq_len (context 0).
        chunk_specs = builder_cls._split_indexer_prefill_chunks(
            seq_lens_cpu, seq_lens_cpu, self.max_total_seq_len, self.max_logits_bytes
        )
        chunks = []
        for req_slice, query_slice in chunk_specs:
            chunk = build_chunk(
                req_slice.start,
                req_slice.stop,
                query_start_loc,
                query_start_loc_cpu,
                seq_lens,
                seq_lens,
                seq_lens_cpu,
                block_table,
                1,
                query_slice=query_slice,
                skip_kv_gather=query_slice.start > 0,
            )
            if chunk is not None:
                chunks.append(chunk)
        return metadata_cls(
            seq_lens=seq_lens,
            max_seq_len=max(token_counts),
            slot_mapping=slot_mapping,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=len(token_counts),
            num_prefill_tokens=sum(token_counts),
            prefill=prefill_cls(chunks, max_prefill_seq_len=max(token_counts)),
            decode=None,
        )

    def build(
        self, token_counts: list[int], *, layer_name: str = "k31.dsa"
    ) -> K31DSARuntimeMetadata:
        counts = [int(c) for c in token_counts]
        if not counts or min(counts) <= 0:
            raise ValueError("token_counts must be positive")
        if max(counts) > self.max_model_len:
            raise ValueError(
                f"request of {max(counts)} tokens exceeds max_model_len {self.max_model_len}"
            )
        dev, bs = self.device, self.block_size
        num_reqs = len(counts)
        blocks = [(c + bs - 1) // bs for c in counts]
        block_table = torch.zeros((num_reqs, max(blocks)), dtype=torch.int32)
        start = 0
        for r, nb in enumerate(blocks):
            block_table[r, :nb] = torch.arange(start, start + nb, dtype=torch.int32)
            start += nb
        num_blocks = start
        block_table = block_table.to(dev)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
        query_start_loc[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
        query_start_loc = query_start_loc.to(dev)
        req_id = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=dev), torch.tensor(counts, device=dev)
        )
        positions = torch.cat([torch.arange(c, dtype=torch.int64, device=dev) for c in counts])
        slot_mapping = (
            block_table[req_id.long(), positions // bs].long() * bs + positions % bs
        ).contiguous()
        seq_lens = torch.tensor(counts, dtype=torch.int32, device=dev)
        mla_cache = torch.zeros(
            (num_blocks, bs, FP8_DS_MLA_TOKEN_BYTES), dtype=torch.uint8, device=dev
        )
        k_cache = torch.zeros(
            (num_blocks, bs, DSA_INDEXER_TOKEN_BYTES), dtype=torch.uint8, device=dev
        )
        indexer = K31IndexerRuntimeMetadata(
            k_cache_prefix=f"{layer_name}.indexer.k_cache",
            k_cache=k_cache,
            attention_metadata=self._indexer_metadata(
                counts, query_start_loc, seq_lens, block_table, slot_mapping
            ),
            topk_buffer=torch.empty((sum(counts), self.topk), dtype=torch.int32, device=dev),
            max_model_len=self.max_model_len,
            max_total_seq_len=self.max_total_seq_len,
        )
        return K31DSARuntimeMetadata(
            token_counts=tuple(counts),
            block_size=bs,
            block_table=block_table,
            slot_mapping=slot_mapping,
            req_id_per_token=req_id,
            query_start_loc=query_start_loc,
            workspace_starts=query_start_loc[:-1].contiguous(),
            positions=positions,
            mla_cache=mla_cache,
            k_scale=torch.ones((), dtype=torch.float32, device=dev),
            cache_lens=torch.full((1,), self.max_model_len, dtype=torch.int32, device=dev),
            dummy_block_table=torch.empty((1, 1), dtype=torch.int32, device=dev),
            indexer=indexer,
        )


def initialize_k31_dsa_runtime(device: torch.device | str) -> None:
    """Process-local vLLM state the DSA calls read: strict BI init + workspace manager."""

    initialize_k31_vllm_batch_invariance()
    workspace = importlib.import_module("vllm.v1.worker.workspace")
    if not workspace.is_workspace_manager_initialized():
        workspace.init_workspace_manager(torch.device(device), num_ubatches=1)


def build_dsa_metadata(
    runtime: K31Runtime, *, max_model_len: int, topk: int = 2048, layer_name: str = "k31.dsa"
) -> K31DSARuntimeMetadata:
    """DSA metadata for one packed microbatch.

    One object may serve every DSA layer of the microbatch: each call rewrites
    the private caches before reading them, and backward only reads tensors
    saved from its own call.  ``max_model_len`` must be the rollout's value.
    """

    device = runtime.cu_seqlens.device
    initialize_k31_dsa_runtime(device)
    return K31DSAMetadataBuilder(device=device, max_model_len=max_model_len, topk=topk).build(
        list(runtime.seq_lens), layer_name=layer_name
    )


__all__ = [
    "K31DSAAttention",
    "K31DSAMetadataBuilder",
    "K31DSARuntimeMetadata",
    "K31IndexerRuntimeMetadata",
    "build_dsa_metadata",
    "initialize_k31_dsa_runtime",
]
