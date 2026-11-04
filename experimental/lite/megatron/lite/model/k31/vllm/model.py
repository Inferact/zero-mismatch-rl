"""Native, fail-closed K31 execution on the rollout's vLLM kernels.

Structure follows the DS4 vLLM implementation: BF16 masters owned here,
forward values from vLLM / BI-overlay kernels through
``primitive.kernels.vllm_k31`` adapters, backward from the autograd owners in
``primitive``.  The decoder stream is AttnRes (prefix + block bank) instead of
DS4's mHC streams.

Layer slots implemented in their own modules (their signatures are the contract):

* ``kda.K31KDAAttention``  -- 69 KDA layers;
* ``dsa.K31DSAAttention``  -- 24 DSA layers;
* ``moe.K31MoE``           -- MoE (router, FI W4A16 experts, expert LoRA,
  canonical combine, latent tail, shared experts).

Every slot is ``forward(hidden_states [T, H] bf16, *, metadata, runtime) ->
[T, H] bf16`` and builds its projections from ``K31Linear`` so the LoRA leaf,
checkpoint names and the bi-gemm plan are shared.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import _pytree

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.primitive import (
    LoRAProjection,
    attn_res,
    rms_norm,
    situ_and_mul,
    visible_linear,
    visible_lora_linear,
)
from megatron.lite.model.k31.vllm.runtime_metadata import K31Runtime
from megatron.lite.primitive.kernels.vllm_k31 import (
    AttnResAdapter,
    BILinearAdapter,
    LoRALeafAdapter,
    PunicaLoRADeltaAdapter,
    RMSNormAdapter,
    SiTUAndMulAdapter,
    TemperatureAdapter,
    TokenLogprobAdapter,
)
from megatron.lite.primitive.parallel import ParallelState, build_pipeline_chunk_layout

_ALIASES = {
    "attn": ("attn_res", "linear", "kda", "dsa"),
    "moe": ("router_moe", "deepep"),
}


def _validate_finite(stage: str, **tensors: torch.Tensor | None) -> None:
    if os.environ.get("MLITE_VALIDATE_FINITE") != "1":
        return
    for name, tensor in tensors.items():
        if tensor is None or not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            continue
        raise FloatingPointError(
            f"MLITE_NONFINITE stage={stage} tensor={name} "
            f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
            f"nonfinite={int((~finite).sum().item())}"
        )


class _ForwardValue(torch.autograd.Function):
    """Return ``value`` exactly in the forward; pass the gradient through to ``reference``."""

    @staticmethod
    def forward(ctx, reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("value must match reference in shape and dtype")
        return value.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


def _rollout_selected_log_probs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature != 1.0:
        logits = TemperatureAdapter()(logits, temperature)
    return TokenLogprobAdapter()(logits, labels.unsqueeze(-1)).squeeze(-1)


def _vllm_selected_log_probs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM V2-runner selected-token logprob value, PyTorch ``log_softmax`` gradient.

    The forward value is the rollout's ``compute_token_logprobs`` on the raw head
    logits (temperature 1) or the FP32 processed logits, bit for bit; the backward
    is the mathematically identical ``log_softmax`` graph.  The two are joined by
    ``_ForwardValue``, not by ``reference + (value - reference).detach()``: that
    straight-through sum is exact only while ``value`` and ``reference`` are within
    a factor of two (Sterbenz), which fails for near-certain tokens where both are
    a few ulps of 2^-23 (kernel 0xB3FFFFFF, torch 0xB4BFFFFE -> sum 0xB4000000).
    """

    rollout_value = None
    if logits.is_cuda:
        rollout_value = _rollout_selected_log_probs(logits, labels, temperature)
    scaled_logits = logits.float() / temperature
    token_log_probs = F.log_softmax(scaled_logits, dim=-1)
    reference = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if rollout_value is None:
        return reference, token_log_probs
    selected = _ForwardValue.apply(reference, rollout_value.detach())
    return selected, token_log_probs


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class K31LoRASpec:
    """Rollout LoRA shape: ``--max-lora-rank``, adapter r/alpha, target names."""

    rank: int
    alpha: float
    max_rank: int
    targets: frozenset[str]
    use_rslora: bool = False

    def wants(self, names: Sequence[str]) -> bool:
        hit = [name in self.targets for name in names]
        if any(hit) and not all(hit):
            raise ValueError(
                f"packed projection {tuple(names)} is only partially targeted by LoRA"
            )
        return all(hit)


class K31Linear(nn.Module):
    """One vLLM (possibly packed) bf16 projection, optionally LoRA-wrapped.

    ``slice_names`` are the checkpoint projection names in vLLM's packed order
    (they are also the peft target names); ``weight`` is the fused ``[sum(out),
    in]`` master exactly as vLLM holds it, so the GEMM has vLLM's (K, N).
    """

    def __init__(
        self,
        in_features: int,
        slice_sizes: Sequence[int],
        slice_names: Sequence[str],
        *,
        lora: K31LoRASpec | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.slice_sizes = tuple(int(s) for s in slice_sizes)
        self.slice_names = tuple(slice_names)
        if len(self.slice_sizes) != len(self.slice_names):
            raise ValueError("one name per packed slice")
        self.weight = nn.Parameter(
            torch.empty(sum(self.slice_sizes), self.in_features, dtype=torch.bfloat16)
        )
        self.lora: LoRAProjection | None = None
        if lora is not None and lora.wants(self.slice_names):
            self.lora = LoRAProjection(
                self.in_features,
                self.slice_names,
                self.slice_sizes,
                rank=lora.rank,
                alpha=lora.alpha,
                max_rank=lora.max_rank,
                use_rslora=lora.use_rslora,
            )

    def forward(self, value: torch.Tensor, runtime: K31Runtime | None = None) -> torch.Tensor:
        if self.lora is None:
            return visible_linear(BILinearAdapter(), value, self.weight)
        if runtime is None or runtime.lora_meta is None:
            raise RuntimeError("LoRA-wrapped projection requires runtime.lora_meta")
        return visible_lora_linear(
            PunicaLoRADeltaAdapter(),
            LoRALeafAdapter(),
            value,
            self.weight,
            self.lora,
            runtime.lora_meta,
        )


class K31RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return rms_norm(RMSNormAdapter(), value, self.weight, self.eps)


class K31DenseMLP(nn.Module):
    """vLLM dense MLP: gate_up -> ``_C.situ_and_mul`` -> down (layer 0, shared experts)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        beta: float,
        linear_beta: float | None,
        lora: K31LoRASpec | None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)
        self.gate_up_proj = K31Linear(
            hidden_size,
            (intermediate_size, intermediate_size),
            ("gate_proj", "up_proj"),
            lora=lora,
        )
        self.down_proj = K31Linear(intermediate_size, (hidden_size,), ("down_proj",), lora=lora)

    def forward(self, value: torch.Tensor, runtime: K31Runtime | None = None) -> torch.Tensor:
        gate_up = self.gate_up_proj(value, runtime)
        act = situ_and_mul(
            SiTUAndMulAdapter(), gate_up, beta=self.beta, linear_beta=self.linear_beta
        )
        return self.down_proj(act, runtime)


class _AttnResWeights(nn.Module):
    """``*_res_norm.weight`` [H] and ``*_res_proj.weight`` [1, H] of one AttnRes site."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.proj_weight = nn.Parameter(torch.zeros(1, hidden_size, dtype=torch.bfloat16))


@dataclass
class AttnResStream:
    """Decoder stream between layers (value semantics of vLLM's in-place buffers).

    ``prefix`` is vLLM's ``prefix_sum``; ``delta`` the not-yet-added MLP output
    (vLLM's ``hidden_states`` between layers); ``blocks`` the written bank
    slots in order.
    """

    prefix: torch.Tensor
    delta: torch.Tensor | None
    blocks: tuple[torch.Tensor, ...]


def _stream_flatten(stream: AttnResStream):
    return [stream.prefix, stream.delta, stream.blocks], None


def _stream_unflatten(children, _context) -> AttnResStream:
    prefix, delta, blocks = children
    return AttnResStream(prefix=prefix, delta=delta, blocks=tuple(blocks))


# Activation recompute (``primitive.recompute`` structured checkpoint) flattens the
# layer's argument and result; the stream is a pytree node of its three fields.
_pytree.register_pytree_node(
    AttnResStream,
    _stream_flatten,
    _stream_unflatten,
    serialized_type_name="megatron.lite.model.k31.vllm.model.AttnResStream",
)


class _LayerEntry(torch.autograd.Function):
    """Identity on a layer's incoming stream tensors (computation reads the aliases).

    Its only role is the order of floating-point gradient sums.  A bank block is
    read by both AttnRes calls of every later layer; without this node the two
    per-layer gradients would be added one at a time into the block's running sum
    (``((g_L,post + g_L,pre) + g_L-1,post) + ...``), while a checkpointed layer can
    only return their per-layer sum.  With the entry (and the layer passing on the
    bank it received, not the aliases) both modes add one per-layer sum per reader,
    so the BF16 block gradient -- and every upstream gradient -- is bit-identical
    with and without activation recompute.
    """

    @staticmethod
    def forward(ctx, *tensors: torch.Tensor):
        ctx.set_materialize_grads(False)
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grads):
        return grads


def _enter_layer(stream: AttnResStream) -> AttnResStream:
    tensors = [stream.prefix, *([] if stream.delta is None else [stream.delta]), *stream.blocks]
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in tensors):
        return stream
    out = list(_LayerEntry.apply(*tensors))
    prefix = out.pop(0)
    delta = None if stream.delta is None else out.pop(0)
    return AttnResStream(prefix=prefix, delta=delta, blocks=tuple(out))


def capture_forward_context():
    """Recompute context: re-install the vLLM forward context active at forward time.

    The layer's MoE kernels read the per-microbatch ``DPMetadata`` that
    ``k31_vllm_forward_context`` installs; backward (and with it the recompute)
    runs after that ``with`` block has exited, possibly after newer microbatches
    installed theirs.
    """

    module = importlib.import_module("vllm.forward_context")
    context = getattr(module, "_forward_context", None)
    if context is None:
        return None
    return lambda: module.override_forward_context(context)


def _attn_res_call(
    stream_prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: tuple[torch.Tensor, ...],
    site: _AttnResWeights,
    output_norm: K31RMSNorm | None,
    *,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return attn_res(
        AttnResAdapter(),
        stream_prefix,
        delta,
        blocks,
        site.norm_weight,
        site.proj_weight.squeeze(0),
        None if output_norm is None else output_norm.weight,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=0.0 if output_norm is None else output_norm.eps,
    )


# ---------------------------------------------------------------------------
# decoder layer
# ---------------------------------------------------------------------------
class K31Layer(nn.Module):
    def __init__(
        self,
        config: K31Config,
        ps: ParallelState | None = None,
        layer_idx: int = 0,
        *,
        selected_layer_ids: frozenset[int] = frozenset(),
        selected_module_names: frozenset[str] = frozenset(),
        lora: K31LoRASpec | None = None,
    ) -> None:
        super().__init__()
        from megatron.lite.model.k31.vllm.dsa import K31DSAAttention
        from megatron.lite.model.k31.vllm.kda import K31KDAAttention
        from megatron.lite.model.k31.vllm.moe import K31MoE

        selected = layer_idx in selected_layer_ids
        self.config = config
        self.layer_idx = layer_idx
        self.selected_stages = selected_module_names if selected else frozenset()
        eps = config.rms_norm_eps
        h = config.hidden_size
        self.input_layernorm = K31RMSNorm(h, eps)
        self.post_attention_layernorm = K31RMSNorm(h, eps)
        self.self_attention_res = _AttnResWeights(h)
        self.mlp_res = _AttnResWeights(h)
        attention_cls = (
            K31KDAAttention if config.is_kda_layer(layer_idx) else K31DSAAttention
        )
        self.self_attn = attention_cls(
            config,
            layer_idx=layer_idx,
            ps=ps,
            selected_stages=self.selected_stages,
            lora=lora,
        )
        if config.is_moe_layer(layer_idx):
            self.mlp = K31MoE(
                config,
                layer_idx=layer_idx,
                ps=ps,
                selected_stages=self.selected_stages,
                lora=lora,
            )
        else:
            self.mlp = K31DenseMLP(
                h,
                config.intermediate_size,
                beta=config.activation_situ_beta,
                linear_beta=config.activation_situ_linear_beta,
                lora=lora,
            )
        self.is_block_write_layer = config.is_block_write_layer(layer_idx)
        self.block_write_idx = config.block_write_idx(layer_idx)
        self.prev_valid_blocks = config.prev_valid_blocks(layer_idx)

    def _selected(self, stage: str) -> None:
        if stage not in self.selected_stages:
            raise NotImplementedError(
                f"layer {self.layer_idx} stage {stage!r} was not executed"
            )

    def forward(
        self,
        stream: AttnResStream,
        *,
        attention_metadata: Any = None,
        moe_metadata: Any = None,
        runtime: K31Runtime,
    ) -> AttnResStream:
        self._selected("attn_res")
        eps = self.config.rms_norm_eps
        incoming_blocks = stream.blocks
        stream = _enter_layer(stream)
        blocks = stream.blocks
        # vLLM decoder layer _pre_attn_norm
        hidden, prefix = _attn_res_call(
            stream.prefix,
            stream.delta,
            blocks,
            self.self_attention_res,
            self.input_layernorm,
            num_blocks=self.prev_valid_blocks,
            block_write_idx=self.block_write_idx if self.is_block_write_layer else -1,
            eps=eps,
        )
        new_block = prefix if self.is_block_write_layer else None
        if new_block is not None:
            blocks = (*blocks, new_block)
        _validate_finite(f"layer_{self.layer_idx}.attn_res_pre", hidden=hidden)
        attn_out = self.self_attn(hidden, metadata=attention_metadata, runtime=runtime)
        _validate_finite(f"layer_{self.layer_idx}.attention", attn_out=attn_out)
        # vLLM decoder layer _post_attn_norm
        if self.is_block_write_layer:
            prefix, delta = attn_out, None
        else:
            delta = attn_out
        hidden, prefix = _attn_res_call(
            prefix,
            delta,
            blocks,
            self.mlp_res,
            self.post_attention_layernorm,
            num_blocks=self.prev_valid_blocks + int(self.is_block_write_layer),
            block_write_idx=-1,
            eps=eps,
        )
        _validate_finite(f"layer_{self.layer_idx}.attn_res_post", hidden=hidden)
        if isinstance(self.mlp, K31DenseMLP):
            self._selected("linear")
            mlp_out = self.mlp(hidden, runtime)
        else:
            mlp_out = self.mlp(hidden, metadata=moe_metadata, runtime=runtime)
        _validate_finite(f"layer_{self.layer_idx}.mlp", mlp_out=mlp_out)
        # The bank travels on as received (not this layer's entry aliases) plus the
        # block written here: every later reader adds its per-layer gradient sum
        # straight to the block's producer, with and without activation recompute.
        out_blocks = incoming_blocks if new_block is None else (*incoming_blocks, new_block)
        return AttnResStream(prefix=prefix, delta=mlp_out, blocks=out_blocks)


class _LMHeadState(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # ParallelLMHead -> UnquantizedEmbeddingMethod.apply -> linear_batch_invariant
        return visible_linear(BILinearAdapter(), value, self.weight)


class _EmbeddingState(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # VocabParallelEmbedding (TP=1) -> F.embedding
        return F.embedding(input_ids, self.weight)


def fold_attn_res_for_pipeline(stream: AttnResStream) -> torch.Tensor:
    """Training PP send: the whole decoder stream ``[prefix, delta, *bank]`` as [T, 1, (2+n)H].

    The rollout runs PP=1, so the next stage must issue exactly the ``attn_res``
    call the PP=1 graph issues: ``prefix`` and the pending ``delta`` travel
    separately and the kernel performs ``prefix += delta`` itself (the same rule
    as DS4, which sends its full mHC stream state).  Pre-adding them here -- what
    vLLM's own PP send does -- would turn the receiving layer's call into the
    ``delta=None`` variant, which is not the PP=1 graph.  Every stage ends with a
    layer, so ``delta`` (that layer's MLP output) is always present.
    """

    if stream.delta is None:
        raise ValueError("pipeline send requires the pending MLP delta of the stage's last layer")
    packed = torch.stack((stream.prefix, stream.delta, *stream.blocks), dim=1)
    return packed.reshape(packed.shape[0], 1, -1)


def unfold_attn_res_from_pipeline(
    packed: torch.Tensor, *, hidden_size: int, num_blocks: int
) -> AttnResStream:
    """Inverse of :func:`fold_attn_res_for_pipeline`; ``num_blocks`` is checked, not trusted."""

    width = packed.shape[-1] * (packed.shape[1] if packed.dim() == 3 else 1)
    if width != (2 + num_blocks) * hidden_size:
        raise ValueError(
            f"pipeline stream width {width} != (2 + {num_blocks} blocks) x hidden {hidden_size}"
        )
    rows = packed.reshape(packed.shape[0], 2 + num_blocks, hidden_size)
    # Dense [T, H] operands, as the PP=1 graph hands them to the kernels.
    return AttnResStream(
        prefix=rows[:, 0].contiguous(),
        delta=rows[:, 1].contiguous(),
        blocks=tuple(rows[:, i + 2].contiguous() for i in range(num_blocks)),
    )


class K31Model(nn.Module):
    """K31 decoder on rollout kernels; every selected stage must execute explicitly."""

    def __init__(
        self,
        config: K31Config,
        train_config=None,
        ps: ParallelState | None = None,
        *,
        vpp_chunk_id: int | None = None,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
        selected_layer_ids: tuple[int, ...] = (),
        selected_module_names: tuple[str, ...] = (),
        lora: K31LoRASpec | None = None,
    ):
        super().__init__()
        del use_thd, hf_path, attention_backend_override, mtp_enable_train
        del mtp_detach_encoder
        if vpp_chunk_id is not None:
            raise NotImplementedError("The K31 vLLM implementation does not support VPP.")
        if mtp_enable:
            raise NotImplementedError("The K31 vLLM implementation does not support MTP.")
        self.config = config
        self.train_config = train_config
        self.ps = ps or ParallelState()
        layout = build_pipeline_chunk_layout(config.num_hidden_layers, self.ps, None, None)
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        selected_ids = frozenset(selected_layer_ids)
        self.selected_layer_ids = tuple(i for i in self.layer_indices if i in selected_ids)
        # A selection that stops inside this stage is a layer-prefix audit: it runs the selected
        # layers and returns the decoder stream at that boundary, never logprobs or a loss.
        self.layer_prefix_audit = len(self.selected_layer_ids) != len(self.layer_indices)
        selected_names = frozenset(
            stage for name in selected_module_names for stage in _ALIASES.get(name, (name,))
        )
        self.lora_spec = lora
        self.embed_tokens = (
            _EmbeddingState(config.vocab_size, config.hidden_size) if self.pre_process else None
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): K31Layer(
                    config,
                    self.ps,
                    layer_idx=layer_idx,
                    selected_layer_ids=selected_ids,
                    selected_module_names=selected_names,
                    lora=lora,
                )
                for layer_idx in self.layer_indices
            }
        )
        h = config.hidden_size
        self.norm = K31RMSNorm(h, config.rms_norm_eps) if self.post_process else None
        self.output_attn_res = _AttnResWeights(h) if self.post_process else None
        self.lm_head = _LMHeadState(h, config.vocab_size) if self.post_process else None
        self._input_tensor: torch.Tensor | None = None

    def set_input_tensor(self, input_tensor) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("K31Model expects one input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def _initial_stream(
        self, input_ids: torch.Tensor | None, hidden_states: torch.Tensor | None
    ) -> AttnResStream:
        if hidden_states is None and not self.pre_process:
            if self._input_tensor is None:
                raise ValueError("non-first PP stage requires its input tensor")
            first = self.layer_indices[0]
            return unfold_attn_res_from_pipeline(
                self._input_tensor,
                hidden_size=self.config.hidden_size,
                num_blocks=self.config.prev_valid_blocks(first),
            )
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("input_ids or hidden_states is required.")
            if os.getenv("MLITE_VALIDATE_INDICES") == "1" and input_ids.numel():
                low, high = torch.aminmax(input_ids)
                if int(low) < 0 or int(high) >= self.config.vocab_size:
                    raise ValueError("input token IDs are outside the embedding vocabulary")
            assert self.embed_tokens is not None
            hidden_states = self.embed_tokens(input_ids.reshape(-1))
        if hidden_states.ndim != 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        # vLLM model forward: prefix_sum = embeddings, hidden = None.
        return AttnResStream(prefix=hidden_states, delta=None, blocks=())

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_metadata: Any = None,
        moe_metadata: Any = None,
        runtime: K31Runtime | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
        **unused,
    ) -> dict[str, torch.Tensor]:
        del unused, position_ids
        if runtime is None:
            raise ValueError("K31 vLLM forward requires a K31Runtime")
        if not self.selected_layer_ids:
            raise NotImplementedError(
                "K31 vLLM forward requires a non-empty contiguous layer prefix."
            )
        if self.layer_prefix_audit and labels is not None:
            raise ValueError(
                "training logprobs and loss need every layer of this pipeline stage "
                f"({len(self.selected_layer_ids)} of {len(self.layer_indices)} selected); "
                "a partial layer selection is a prefix audit"
            )
        stream = self._initial_stream(input_ids, hidden_states)
        for layer_idx in self.selected_layer_ids:
            layer = self.layers[str(layer_idx)]
            stream = layer(
                stream,
                attention_metadata=(
                    attention_metadata.get(layer_idx)
                    if isinstance(attention_metadata, dict)
                    else attention_metadata
                ),
                moe_metadata=(
                    moe_metadata.get(layer_idx) if isinstance(moe_metadata, dict) else moe_metadata
                ),
                runtime=runtime,
            )
        if not self.post_process:
            return {"hidden_states": fold_attn_res_for_pipeline(stream)}
        if self.layer_prefix_audit:
            # Layer-prefix audit: stop at the decoder boundary (vLLM layer hook).
            return {
                "prefix": stream.prefix,
                "hidden_states": stream.delta,
                "blocks": torch.stack(stream.blocks, dim=1)
                if stream.blocks
                else stream.prefix.new_empty((stream.prefix.shape[0], 0, stream.prefix.shape[1])),
            }
        if self.norm is None or self.output_attn_res is None or self.lm_head is None:
            raise RuntimeError("final pipeline stage is missing the output head")
        # vLLM model forward tail (no output norm), then compute_logits.
        hidden, _ = _attn_res_call(
            stream.prefix,
            stream.delta,
            stream.blocks,
            self.output_attn_res,
            None,
            num_blocks=self.config.num_attn_res_blocks,
            block_write_idx=-1,
            eps=self.config.rms_norm_eps,
        )
        _validate_finite("model.attn_res_head", hidden=hidden)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        _validate_finite("model.logits", logits=logits)
        result = {"hidden_states": hidden, "logits": logits}
        if labels is not None:
            temperature_value = float(
                temperature.detach().float().item()
                if isinstance(temperature, torch.Tensor)
                else temperature
            )
            if temperature_value <= 0:
                raise ValueError("temperature must be positive")
            flat_labels = labels.reshape(-1).long()
            if flat_labels.numel() != logits.shape[0]:
                raise ValueError("labels must contain one target per visible token")
            selected, token_log_probs = _vllm_selected_log_probs(
                logits, flat_labels, temperature_value
            )
            result["log_probs"] = selected
            if calculate_entropy:
                result["entropy"] = -(token_log_probs.exp() * token_log_probs).sum(dim=-1)
            token_loss = -selected
            mask = (
                torch.ones_like(token_loss)
                if loss_mask is None
                else loss_mask.reshape(-1).to(token_loss.dtype)
            )
            denominator = mask.sum()
            if not bool(denominator > 0):
                raise ValueError("loss_mask must select at least one token")
            result["loss"] = (token_loss * mask).sum() / denominator
        return result


__all__ = [
    "AttnResStream",
    "K31DenseMLP",
    "K31Layer",
    "K31Linear",
    "K31LoRASpec",
    "K31Model",
    "K31RMSNorm",
    "capture_forward_context",
    "fold_attn_res_for_pipeline",
    "unfold_attn_res_from_pipeline",
]
