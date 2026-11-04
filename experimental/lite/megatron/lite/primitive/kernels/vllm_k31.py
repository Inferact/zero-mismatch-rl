# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict adapters for the K31 vLLM / BI-overlay inference kernels.

Conventions (shared by every K31 adapter module):

* An adapter is a parameter-free ``nn.Module`` (``_Adapter``).  It validates
  its inputs *before* resolving any vLLM symbol, calls exactly one vLLM entry
  (the function the rollout itself calls), and returns the kernel's value
  unchanged through ``_output_boundary``.  No numerical fallback, no vLLM model
  objects, no forward-context reads.
* vLLM symbols are resolved at call time with ``_symbol(module, name)`` /
  ``_op(namespace, name)``; a missing symbol raises ``NotImplementedError``.
* The K31 model package inside the image's vLLM is *discovered* (the overlay
  directory that ships ``k31_bi_gemm.py``) instead of being spelled out, see
  ``k31_vllm_package``.  Use ``_k31_symbol(relative_module, name)`` for any
  overlay module that lives in that package (``k31_bi_gemm``, ``ops.attn_res``,
  ``k31_kda_cfbi``, ...).
* Fail closed on dispatch: when a vLLM wrapper has a fast path and a fallback
  (Triton persistent matmul, Triton attn_res), the adapter asserts the
  deployment path is the one taken, otherwise it raises.
"""

from __future__ import annotations

import functools
import glob
import importlib
import importlib.util
import os
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn

from megatron.lite.primitive.autograd import inference_only


# ---------------------------------------------------------------------------
# symbol resolution
# ---------------------------------------------------------------------------
def _symbol(module: str, name: str) -> Any:
    try:
        value = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"vLLM kernel entry {module}.{name} is unavailable; "
            "run inside the K31 image."
        ) from exc
    return value


def _op(namespace: str, name: str) -> Callable[..., Any]:
    try:
        op = getattr(getattr(torch.ops, namespace), name)
    except (AttributeError, RuntimeError) as exc:
        raise NotImplementedError(
            f"required torch.ops.{namespace}.{name} is not registered by this vLLM build"
        ) from exc
    return op


@functools.lru_cache(maxsize=1)
def k31_vllm_package() -> str:
    """Dotted name of the vLLM package that hosts the K31 model + BI overlays.

    Found as the unique ``vllm/models/*/nvidia`` directory that contains the
    bi-gemm overlay module.  Raises when the overlay chain is not mounted.
    """

    spec = importlib.util.find_spec("vllm.models")
    if spec is None or not spec.submodule_search_locations:
        raise NotImplementedError("vllm.models is not importable")
    hits: list[str] = []
    for root in spec.submodule_search_locations:
        hits.extend(glob.glob(os.path.join(root, "*", "nvidia", "k31_bi_gemm.py")))
    packages = sorted(
        {
            "vllm.models." + os.path.basename(os.path.dirname(os.path.dirname(h))) + ".nvidia"
            for h in hits
        }
    )
    if len(packages) != 1:
        raise NotImplementedError(
            "expected exactly one vLLM model package carrying the K31 bi-gemm overlay, "
            f"found {packages}"
        )
    return packages[0]


def k31_model_family() -> str:
    """Directory name of the K31 vLLM model family (``vllm.models.<family>``)."""

    return k31_vllm_package().split(".")[2]


def _k31_symbol(relative_module: str, name: str) -> Any:
    return _symbol(f"{k31_vllm_package()}.{relative_module}", name)


def _k31_module(relative_module: str):
    try:
        return importlib.import_module(f"{k31_vllm_package()}.{relative_module}")
    except ImportError as exc:
        raise NotImplementedError(
            f"K31 overlay module {relative_module} is unavailable"
        ) from exc


# ---------------------------------------------------------------------------
# validation helpers (same contract as vllm_ds4)
# ---------------------------------------------------------------------------
def _tensors(values: Iterable[Any]) -> Iterable[Tensor]:
    for value in values:
        if isinstance(value, Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            yield from _tensors(value)


def _output_boundary(value: Any, *dependencies: Any) -> Any:
    deps = tuple(_tensors(dependencies))
    if isinstance(value, Tensor):
        if value.requires_grad or any(tensor.requires_grad for tensor in deps):
            return inference_only(value, *deps)
        return value
    if isinstance(value, tuple):
        return tuple(_output_boundary(item, *deps) for item in value)
    if isinstance(value, list):
        return [_output_boundary(item, *deps) for item in value]
    return value


def _tensor(
    value: Tensor,
    name: str,
    *,
    ndim: int | None = None,
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    contiguous: bool = True,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(value.shape)}")
    if dtype is not None:
        allowed = (dtype,) if isinstance(dtype, torch.dtype) else dtype
        if value.dtype not in allowed:
            raise TypeError(f"{name} dtype must be one of {allowed}, got {value.dtype}")
    if contiguous and not value.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; shape={tuple(value.shape)} stride={value.stride()}"
        )


def _same_device(named: dict[str, Tensor]) -> None:
    devices = {value.device for value in named.values()}
    if len(devices) != 1:
        detail = ", ".join(f"{name}={value.device}" for name, value in named.items())
        raise ValueError(f"all tensors must be on one device ({detail})")


class _Adapter(nn.Module):
    """Parameter-free base with a stable inference-only contract."""

    def __init__(self) -> None:
        super().__init__()


# ---------------------------------------------------------------------------
# dense linear / router / LoRA leaf  (overlay bi-gemm-r5 + lora-f32add-bi-r3)
# ---------------------------------------------------------------------------
def _bi_gemm():
    module = _k31_module("k31_bi_gemm")
    version = getattr(module, "VERSION", "")
    if version < "r4":
        raise NotImplementedError(f"K31 bi-gemm overlay >= r4 required, found {version!r}")
    return module


class BILinearAdapter(_Adapter):
    """``UnquantizedLinearMethod.apply`` under BI: ``linear_batch_invariant``.

    Fails closed unless the K31 bi-gemm takes the shape (K <= 3584: cuBLASLt
    S=1; K > 3584: tcgen05 split-K), i.e. the Triton persistent matmul fallback
    of ``linear_batch_invariant`` is never reached.
    """

    def forward(self, value: Tensor, weight: Tensor) -> Tensor:
        _tensor(value, "x", dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=2, dtype=torch.bfloat16, contiguous=False)
        if value.shape[-1] != weight.shape[1]:
            raise ValueError("x and weight inner dimensions differ")
        _same_device({"x": value, "weight": weight})
        bg = _bi_gemm()
        if not (bg.enabled() and bg.eligible(weight.shape[1], weight.shape[0])):
            raise RuntimeError(
                f"K31 bi-gemm does not take (K={weight.shape[1]}, N={weight.shape[0]}); "
                "refusing the Triton persistent-matmul fallback"
            )
        fn = _symbol(
            "vllm.model_executor.determinism.batch_invariant", "linear_batch_invariant"
        )
        return _output_boundary(fn(value, weight, None), value, weight)


class BIRouterAdapter(_Adapter):
    """``GateLinear.forward`` under BI (overlay gate_linear): fp32 ``bi_router``."""

    def forward(self, value: Tensor, weight: Tensor) -> Tensor:
        _tensor(value, "x", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=2, dtype=torch.bfloat16)
        _same_device({"x": value, "weight": weight})
        bg = _bi_gemm()
        if not (bg.enabled() and bg.eligible(weight.shape[1], weight.shape[0])):
            raise RuntimeError("K31 bi-gemm does not take the router shape")
        return _output_boundary(bg.bi_router(value, weight), value, weight)


class LoRALeafAdapter(_Adapter):
    """Overlay lora-f32add-bi-r3 leaf: ``bf16(fp32(x @ w.T) + fp32(delta))``."""

    def forward(self, value: Tensor, weight: Tensor, delta: Tensor) -> Tensor:
        _tensor(value, "x", ndim=2, dtype=torch.bfloat16)
        _tensor(weight, "weight", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(delta, "delta", ndim=2, dtype=torch.bfloat16)
        if delta.shape != (value.shape[0], weight.shape[0]):
            raise ValueError("delta must be [tokens, out_features]")
        _same_device({"x": value, "weight": weight, "delta": delta})
        bg = _bi_gemm()
        if bg.VERSION < "r3" or not bg.enabled():
            raise RuntimeError("K31 lora-f32add leaf needs bi-gemm >= r3 enabled")
        if not bg.f32add_eligible(weight.shape[1], weight.shape[0]):
            raise RuntimeError("K31 lora-f32add leaf does not take this shape")
        return _output_boundary(bg.lora_f32add(value, weight, delta), value, weight, delta)


class PunicaLoRADeltaAdapter(_Adapter):
    """``PunicaWrapperGPU.add_lora_linear(delta=0, ..., scale=1.0, add_inputs=False)``.

    Exactly the rollout call made by the lora-f32add-bi-r3 ``_apply_sync``:
    an fp32 ``[slices, tokens, r]`` shrink buffer followed by expand into a
    zero bf16 delta.  ``meta`` is a vLLM ``LoRAKernelMeta`` prepared by
    ``runtime_metadata.build_lora_kernel_meta``.
    """

    def forward(
        self,
        value: Tensor,
        lora_a_stacked: tuple[Tensor, ...],
        lora_b_stacked: tuple[Tensor, ...],
        output_slices: tuple[int, ...],
        meta: Any,
        *,
        specialize_active_lora: bool = False,
    ) -> Tensor:
        _tensor(value, "x", ndim=2, dtype=torch.bfloat16)
        if not (len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)):
            raise ValueError("LoRA slice tuples must have equal length")
        for i, (a, b) in enumerate(zip(lora_a_stacked, lora_b_stacked, strict=True)):
            _tensor(a, f"lora_a[{i}]", ndim=4, dtype=torch.bfloat16)
            _tensor(b, f"lora_b[{i}]", ndim=4, dtype=torch.bfloat16)
            if a.shape[-1] != value.shape[1] or b.shape[-2] != output_slices[i]:
                raise ValueError(f"LoRA slice {i} shape mismatch")
            if a.shape[-2] != b.shape[-1]:
                raise ValueError(f"LoRA slice {i} rank mismatch")
        shrink = _symbol("vllm.lora.ops.triton_ops", "lora_shrink")
        expand = _symbol("vllm.lora.ops.triton_ops", "lora_expand")
        tokens = value.shape[0]
        rank = lora_b_stacked[0].size(-1)
        delta = torch.zeros(
            (tokens, sum(output_slices)), device=value.device, dtype=value.dtype
        )
        buffer = torch.empty(
            (len(output_slices), tokens, rank), dtype=torch.float32, device=value.device
        )
        # vLLM caches the shrink/expand pointer tables in module dicts keyed by the weights'
        # data_ptr()s; for a single slice the cached entry is the weight tensor itself.  The
        # rollout's stacked weights are persistent, but training builds fresh bf16 stacks every
        # forward: every call then adds an entry that pins its stack forever (~9.5 GiB
        # per GPU after one step, second rollout OOM), and a multi-slice entry whose addresses are
        # later reused by a stack of another shape would hand the kernel stale strides.  Drop our
        # keys before (stale) and after (leak) the launch; the kernels and values are unchanged.
        _drop_punica_ptr_cache(lora_a_stacked, lora_b_stacked)
        try:
            shrink(
                value,
                lora_a_stacked,
                buffer,
                *meta.meta_args(tokens, specialize_active_lora),
                1.0,
            )
            expand(
                buffer,
                lora_b_stacked,
                delta,
                *meta.meta_args(tokens, specialize_active_lora),
                offset_start=0,
                add_inputs=False,
            )
        finally:
            _drop_punica_ptr_cache(lora_a_stacked, lora_b_stacked)
        return _output_boundary(delta, value, lora_a_stacked, lora_b_stacked)


def _drop_punica_ptr_cache(lora_a_stacked, lora_b_stacked) -> None:
    utils = importlib.import_module("vllm.lora.ops.triton_ops.utils")
    utils._LORA_A_PTR_DICT.pop(tuple(t.data_ptr() for t in lora_a_stacked), None)
    utils._LORA_B_PTR_DICT.pop(tuple(t.data_ptr() for t in lora_b_stacked), None)


# ---------------------------------------------------------------------------
# norms and activations
# ---------------------------------------------------------------------------
class RMSNormAdapter(_Adapter):
    """``RMSNorm.forward_cuda`` under BI: ``rms_norm_batch_invariant``."""

    def forward(self, value: Tensor, weight: Tensor, eps: float) -> Tensor:
        _tensor(value, "x", dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=1)
        if weight.shape[0] != value.shape[-1]:
            raise ValueError("RMSNorm weight must match the hidden dimension")
        if eps <= 0:
            raise ValueError("eps must be positive")
        _same_device({"x": value, "weight": weight})
        fn = _symbol(
            "vllm.model_executor.determinism.batch_invariant", "rms_norm_batch_invariant"
        )
        return _output_boundary(fn(value, weight, eps), value, weight)


class SiTUAndMulAdapter(_Adapter):
    """``SituAndMul.forward_cuda``: ``torch.ops._C.situ_and_mul(out, x, beta, lin_beta)``.

    The CUDA kernel and the torch-native form differ by 1 ulp on some inputs,
    so the training forward must call this op.
    """

    def forward(self, value: Tensor, beta: float, linear_beta: float | None) -> Tensor:
        _tensor(value, "x", dtype=torch.bfloat16)
        if value.shape[-1] % 2:
            raise ValueError("SiTU input must hold gate and up halves")
        op = _op("_C", "situ_and_mul")
        d = value.shape[-1] // 2
        out = torch.empty(value.shape[:-1] + (d,), dtype=value.dtype, device=value.device)
        op(out, value, float(beta), -1.0 if linear_beta is None else float(linear_beta))
        return _output_boundary(out, value)


# ---------------------------------------------------------------------------
# AttnRes (native SM100 op only)
# ---------------------------------------------------------------------------
ATTN_RES_HIDDEN = 7168
ATTN_RES_MAX_BLOCKS = 8


class AttnResAdapter(_Adapter):
    """The rollout ``attn_res(...)`` wrapper, restricted to its native op branch.

    Same arguments as the vLLM wrapper.  The native op mutates ``prefix``
    (``prefix += delta`` rounded to bf16, when ``delta`` is given) and writes
    ``blocks[:, block_write_idx]`` in place; callers that need value semantics
    pass scratch copies (``primitive/attn_res.py`` does).
    """

    def forward(
        self,
        prefix: Tensor,
        delta: Tensor | None,
        blocks: Tensor,
        norm_weight: Tensor,
        qk_weight: Tensor,
        output_norm_weight: Tensor | None,
        num_blocks: int,
        block_write_idx: int,
        eps: float,
        output_norm_eps: float,
    ) -> Tensor:
        _tensor(prefix, "prefix", ndim=2, dtype=torch.bfloat16)
        if delta is not None:
            _tensor(delta, "delta", ndim=2, dtype=torch.bfloat16)
            if delta.shape != prefix.shape:
                raise ValueError("delta must match prefix")
        _tensor(blocks, "blocks", ndim=3, dtype=torch.bfloat16)
        for name, w in (("norm_weight", norm_weight), ("qk_weight", qk_weight)):
            _tensor(w, name, ndim=1)
        if output_norm_weight is not None:
            _tensor(output_norm_weight, "output_norm_weight", ndim=1)
        tokens, hidden = prefix.shape
        if hidden != ATTN_RES_HIDDEN:
            raise ValueError(f"native AttnRes requires hidden={ATTN_RES_HIDDEN}")
        if blocks.shape[0] != tokens or blocks.shape[2] != hidden:
            raise ValueError("blocks must be [tokens, slots, hidden]")
        if not 0 <= num_blocks <= min(ATTN_RES_MAX_BLOCKS, blocks.shape[1]):
            raise ValueError(f"num_blocks={num_blocks} outside the native range")
        if not -1 <= block_write_idx < blocks.shape[1]:
            raise ValueError("block_write_idx outside the bank")
        if not prefix.is_cuda or torch.cuda.get_device_capability(prefix.device)[0] != 10:
            raise RuntimeError("native AttnRes requires an sm_10x GPU")
        op_name = f"{k31_model_family()}_attn_res"
        if not hasattr(torch.ops._C, op_name):
            raise NotImplementedError("native AttnRes op is not registered by this build")
        fn = _k31_symbol("ops.attn_res", "attn_res")
        return _output_boundary(
            fn(
                prefix,
                delta,
                blocks,
                norm_weight,
                qk_weight,
                output_norm_weight,
                num_blocks,
                block_write_idx,
                eps,
                output_norm_eps,
            ),
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
        )


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------
class TokenLogprobAdapter(_Adapter):
    """V2 model runner ``compute_token_logprobs`` (selected-token log-softmax)."""

    def forward(self, logits: Tensor, token_ids: Tensor) -> Tensor:
        _tensor(logits, "logits", ndim=2, dtype=(torch.bfloat16, torch.float32))
        _tensor(token_ids, "token_ids", ndim=2, contiguous=False)
        if token_ids.shape[0] != logits.shape[0]:
            raise ValueError("token_ids rows must match logits rows")
        fn = _symbol("vllm.v1.worker.gpu.sample.logprob", "compute_token_logprobs")
        return _output_boundary(fn(logits, token_ids), logits)


class TemperatureAdapter(_Adapter):
    """V2 sampler ``apply_temperature`` on an FP32 copy of the logits (one temperature, every row)."""

    def forward(self, logits: Tensor, temperature: float) -> Tensor:
        _tensor(logits, "logits", ndim=2, dtype=(torch.bfloat16, torch.float32))
        fn = _symbol("vllm.v1.worker.gpu.sample.gumbel", "apply_temperature")
        scaled = torch.empty_like(logits, dtype=torch.float32).copy_(logits.detach())
        rows = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
        fn(scaled, rows, torch.full((1,), temperature, dtype=torch.float32, device=logits.device))
        return scaled


__all__ = [
    "ATTN_RES_HIDDEN",
    "ATTN_RES_MAX_BLOCKS",
    "AttnResAdapter",
    "BILinearAdapter",
    "BIRouterAdapter",
    "LoRALeafAdapter",
    "PunicaLoRADeltaAdapter",
    "RMSNormAdapter",
    "SiTUAndMulAdapter",
    "TemperatureAdapter",
    "TokenLogprobAdapter",
    "k31_model_family",
    "k31_vllm_package",
]


# ---------------------------------------------------------------------------
# DSA: fp8_ds_mla KV, pinned FlashMLA fp8 sparse kernel, absorbed
# MLA bmm, frozen V3.2-style indexer.
# ---------------------------------------------------------------------------
# fp8_ds_mla token row: 512 float8_e4m3 NoPE values, 4 float32 scales (one per
# 128 values), 64 bfloat16 "rope" values (K31 is NoPE: these are un-rotated).
FP8_DS_MLA_TOKEN_BYTES = 656
FP8_DS_MLA_HEAD_DIM = 576
FP8_DS_MLA_NOPE_DIM = 512
FP8_DS_MLA_ROPE_DIM = 64
# FlashMLA SM100 fp8 sparse decode supports h_q in {64, 128}.
FP8_SPARSE_DECODE_HEADS = (64, 128)
DSA_BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# fp8_ds_mla cache: insert and upconvert
# ---------------------------------------------------------------------------


class DSAMLAKVInsertAdapter(_Adapter):
    """``MLACommonBaseImpl.do_kv_cache_update`` for the 656-byte fp8_ds_mla layout."""

    def forward(
        self,
        kv_c_normed: Tensor,
        k_pe: Tensor,
        cache: Tensor,
        slot_mapping: Tensor,
        k_scale: Tensor,
    ) -> Tensor:
        _tensor(kv_c_normed, "kv_c_normed", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(k_pe, "k_pe", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(cache, "mla_cache", ndim=3, dtype=torch.uint8)
        _tensor(slot_mapping, "slot_mapping", ndim=1, dtype=torch.int64)
        if cache.shape[-1] != FP8_DS_MLA_TOKEN_BYTES:
            raise ValueError(f"fp8_ds_mla cache rows must be {FP8_DS_MLA_TOKEN_BYTES} bytes")
        if kv_c_normed.shape[-1] != FP8_DS_MLA_NOPE_DIM or k_pe.shape[-1] != FP8_DS_MLA_ROPE_DIM:
            raise ValueError("fp8_ds_mla expects kv_c 512 and k_pe 64 wide")
        if not (kv_c_normed.shape[0] == k_pe.shape[0] == slot_mapping.numel()):
            raise ValueError("kv_c_normed, k_pe and slot_mapping token counts differ")
        _same_device({"kv_c": kv_c_normed, "k_pe": k_pe, "cache": cache, "slots": slot_mapping})
        _symbol("vllm._custom_ops", "concat_and_cache_mla")(
            kv_c_normed.detach(),
            k_pe.detach(),
            cache,
            slot_mapping.flatten(),
            kv_cache_dtype="fp8_ds_mla",
            scale=k_scale,
        )
        return cache


class DSAFP8KVUpconvertAdapter(_Adapter):
    """Official fp8_ds_mla -> bf16 [rows, 576] gather (the BF16 prefill workspace).

    Values equal ``bf16(fp8 * scale_fp32)`` for NoPE and the raw bf16 tail
    Used only to feed the vendor sparse-attention backward.
    """

    def forward(
        self, cache: Tensor, workspace: Tensor, block_table: Tensor, workspace_starts: Tensor
    ) -> Tensor:
        _tensor(cache, "mla_cache", ndim=3, dtype=torch.uint8)
        _tensor(workspace, "workspace", ndim=2, dtype=torch.bfloat16)
        _tensor(block_table, "block_table", ndim=2, dtype=torch.int32)
        _tensor(workspace_starts, "workspace_starts", ndim=1, dtype=torch.int32)
        if workspace.shape[-1] != FP8_DS_MLA_HEAD_DIM:
            raise ValueError("upconvert workspace must be 576 wide")
        _symbol("vllm._custom_ops", "cp_gather_and_upconvert_fp8_kv_cache")(
            cache, workspace, block_table, workspace_starts, int(block_table.shape[0])
        )
        return workspace


# ---------------------------------------------------------------------------
# Absorbed MLA batched GEMMs (W_UK^T absorb, W_UV up-projection)
# ---------------------------------------------------------------------------


class DSAAbsorbBmmAdapter(_Adapter):
    """The two ``torch.bmm`` calls of ``MLAAttention.forward_impl`` / ``_v_up_proj``.

    Under ``VLLM_BATCH_INVARIANT=1`` vLLM replaces ``torch.bmm`` (and
    ``aten::bmm``) with ``bmm_batch_invariant``; this adapter invokes the same
    replaced entry with the same ``out=`` layout so the per-row reduction is
    identical.
    """

    @staticmethod
    def _bmm() -> Callable[..., Tensor]:
        return _symbol("vllm.model_executor.determinism.batch_invariant", "bmm_batch_invariant")

    def absorb(self, q_nope: Tensor, w_uk_t: Tensor) -> Tensor:
        """(B, N, P) x (N, P, L) -> token-major (B, N, L), as the non-padded branch."""
        _tensor(q_nope, "q_nope", ndim=3, dtype=torch.bfloat16, contiguous=False)
        _tensor(w_uk_t, "W_UK_T", ndim=3, dtype=torch.bfloat16, contiguous=False)
        b, n, _p = q_nope.shape
        out = q_nope.new_empty((b, n, w_uk_t.shape[-1]))
        self._bmm()(q_nope.transpose(0, 1), w_uk_t, out=out.transpose(0, 1))
        return out

    def v_up(self, attn: Tensor, w_uv: Tensor, out: Tensor) -> Tensor:
        """(B, N, L) -> (N, B, L) x (N, L, V) -> out (B, N*V) via its transposed view."""
        _tensor(w_uv, "W_UV", ndim=3, dtype=torch.bfloat16, contiguous=False)
        n, lr, v = w_uv.shape
        x = attn.reshape(-1, n, lr).transpose(0, 1)
        view = out.view(-1, n, v)
        self._bmm()(x, w_uv, out=view.transpose(0, 1))
        return out


# ---------------------------------------------------------------------------
# Pinned FlashMLA fp8 sparse kernel (mixed-batch path, every token a row)
# ---------------------------------------------------------------------------


class DSASparseMLAAdapter(_Adapter):
    """``FlashMLASparseImpl._forward_fp8_kv_mixed_batch`` + ``_fp8_flash_mla_kernel``.

    Under BI the overlays (ds4-sparse-mla, -allow, sparse-mla-tp1) send every
    token of a step through the fp8 decode kernel as one ``(1, T)`` batch with
    a pinned whole-request plan; a row's reduction depends only on its own
    top-k list (measured: single-row, chunked and repacked calls are
    bitwise equal).  Returns ``(out, lse)`` with ``out`` (T, H, 512) and the
    natural-log ``lse`` (T, H).
    """

    def convert_indices(
        self,
        req_id_per_token: Tensor,
        block_table: Tensor,
        logical_topk: Tensor,
        *,
        block_size: int,
    ) -> Tensor:
        _tensor(req_id_per_token, "req_id_per_token", ndim=1, dtype=torch.int32)
        _tensor(block_table, "block_table", ndim=2, dtype=torch.int32)
        _tensor(logical_topk, "logical_topk", ndim=2, dtype=torch.int32)
        convert = _symbol(
            "vllm.v1.attention.backends.mla.sparse_utils",
            "triton_convert_req_index_to_global_index",
        )
        return convert(
            req_id_per_token,
            block_table,
            logical_topk,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=logical_topk.shape[1],
        )

    def forward(
        self,
        q: Tensor,
        cache: Tensor,
        global_topk: Tensor,
        *,
        softmax_scale: float,
        padded_heads: int,
        cache_lens: Tensor,
        dummy_block_table: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _tensor(q, "sparse_mla.q", ndim=3, dtype=torch.bfloat16)
        _tensor(cache, "mla_cache", ndim=3, dtype=torch.uint8)
        _tensor(global_topk, "global_topk", ndim=2, dtype=torch.int32)
        if padded_heads not in FP8_SPARSE_DECODE_HEADS or q.shape[1] > padded_heads:
            raise ValueError(f"fp8 sparse decode pads heads to 64/128, got {padded_heads}")
        if q.shape[-1] != FP8_DS_MLA_HEAD_DIM or q.shape[0] != global_topk.shape[0]:
            raise ValueError("q must be (T, H, 576) with one top-k row per token")
        flashmla = "vllm.v1.attention.ops.flashmla"
        backend = "vllm.v1.attention.backends.mla.flashmla_sparse"
        get_mla_metadata = _symbol(flashmla, "get_mla_metadata")
        flash_mla_with_kvcache = _symbol(flashmla, "flash_mla_with_kvcache")
        pin = _symbol(backend, "_k31_pin_sched")

        actual_heads = q.shape[1]
        q4 = q.unsqueeze(0)
        if actual_heads < padded_heads:
            q_padded = q4.new_zeros((1, q4.shape[1], padded_heads, q4.shape[3]))
            q_padded[:, :, :actual_heads, :] = q4
            q4 = q_padded
        indices = global_topk.unsqueeze(0)
        sched, _ = get_mla_metadata()
        pin(sched, q4, indices)
        out, lse = flash_mla_with_kvcache(
            q=q4,
            k_cache=cache.view(torch.uint8).unsqueeze(-2),
            block_table=dummy_block_table,
            head_dim_v=FP8_DS_MLA_NOPE_DIM,
            cache_seqlens=cache_lens,
            tile_scheduler_metadata=sched,
            is_fp8_kvcache=True,
            indices=indices,
            softmax_scale=softmax_scale,
        )
        if actual_heads < padded_heads:
            out = out[:, :, :actual_heads, :]
            lse = lse[:, :actual_heads, :]
        return out.squeeze(0), lse.squeeze(0).transpose(0, 1)


# ---------------------------------------------------------------------------
# Frozen V3.2-style DSA indexer
# ---------------------------------------------------------------------------


class DSAIndexerAdapter(_Adapter):
    """The K31 DSA indexer module forward, minus the module objects.

    Same op sequence as the vLLM layer: BI linear projections, fp32
    LayerNorm on the shared key, per-head weights scaled by H^-1/2 in bf16,
    ue8m0 per-token-group FP8 query with the scale folded into fp32 weights,
    then ``torch.ops.vllm.sparse_attn_indexer`` with the DeepSelect top-k the
    rollout runs under batch invariance (-1 tail).  The op writes the indexer K
    cache and fills ``topk_buffer[:T]``; the result is inference-only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = BILinearAdapter()

    def forward(
        self,
        hidden_states: Tensor,
        q_c: Tensor,
        *,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        k_norm_weight: Tensor,
        k_norm_bias: Tensor | None,
        k_norm_eps: float,
        num_heads: int,
        head_dim: int,
        topk: int,
        metadata: Any,
    ) -> Tensor:
        _tensor(
            hidden_states, "indexer.hidden_states", ndim=2, dtype=torch.bfloat16, contiguous=False
        )
        _tensor(q_c, "indexer.q_c", ndim=2, dtype=torch.bfloat16, contiguous=False)
        if num_heads != 32:
            # the rollout pads fewer heads to 32 for DeepGEMM; that padding is not ported
            raise NotImplementedError("K31 indexer has 32 heads; head padding is not ported")
        hidden = hidden_states.detach()
        index_q = self.linear(q_c.detach(), wq).view(-1, num_heads, head_dim)
        k_lin = self.linear(hidden, wk)
        if k_norm_bias is None:
            raise NotImplementedError(
                "K31 uses index_k_norm_type=layernorm; RMSNorm keys are not ported"
            )
        # Indexer k-norm (layernorm type): fp32 LayerNorm with fp32 weight/bias, cast back to bf16.
        index_k = torch.nn.functional.layer_norm(
            k_lin.float(), (head_dim,), k_norm_weight, k_norm_bias, k_norm_eps
        ).type_as(k_lin)
        index_v = self.linear(hidden, wv) * (num_heads**-0.5)
        quant = _symbol(
            "vllm.model_executor.layers.quantization.utils.fp8_utils", "per_token_group_quant_fp8"
        )
        q_flat = index_q.reshape(-1, head_dim).contiguous()
        q_fp8, q_scale = quant(q_flat, head_dim, column_major_scales=False, use_ue8m0=True)
        q_fp8 = q_fp8.view(index_q.shape)
        q_scale = q_scale.view(index_q.shape[0], num_heads, -1)
        weights = (
            (index_v.float().unsqueeze(-1) * q_scale * (head_dim**-0.5)).squeeze(-1).contiguous()
        )
        encode = _symbol("vllm.utils.torch_utils", "_encode_layer_name")
        forward_module = importlib.import_module("vllm.forward_context")
        context = forward_module.ForwardContext(
            no_compile_layers={},
            attn_metadata={metadata.k_cache_prefix: metadata.attention_metadata},
            slot_mapping={},
        )
        with forward_module.override_forward_context(context):
            result = _op("vllm", "sparse_attn_indexer")(
                hidden,
                encode(metadata.k_cache_prefix),
                metadata.k_cache,
                q_fp8,
                None,
                index_k,
                weights,
                head_dim,
                "ue8m0",
                topk,
                head_dim,
                metadata.max_model_len,
                metadata.max_total_seq_len,
                metadata.topk_buffer,
                False,
                False,
                False,
                encode(""),
                False,
                0,
                1,
                1,
                topk_backend="deep_select",
                require_deep_select=True,
            )
        # Inputs are detached and the selection is integral: nothing to differentiate.
        return result[: hidden.shape[0]]


__all__ += [
    "DSA_BLOCK_SIZE",
    "DSAAbsorbBmmAdapter",
    "DSAFP8KVUpconvertAdapter",
    "DSAIndexerAdapter",
    "DSAMLAKVInsertAdapter",
    "DSASparseMLAAdapter",
    "FP8_DS_MLA_HEAD_DIM",
    "FP8_DS_MLA_TOKEN_BYTES",
]


# ---------------------------------------------------------------------------
# KDA: the BI=1 rollout prefill branch of the KDA layer, kernel by kernel
#   short conv  -> causal_conv1d_fn            (q / k / v column slices, SiLU, zero initial state)
#   core        -> <k31 pkg>.k31_kda_cfbi.mixed (CF-BI v5 prefill engine, ctx = 0 per sequence)
#   output norm -> FusedRMSNormGated method     (sigmoid gate; the CustomOp method the engine dispatched)
# Each sequence of a packed batch owns state slot 1..R of a per-call temporary pool; slot 0 is vLLM's
# NULL_BLOCK_ID, for which causal_conv1d_fn writes no output at all.
# ---------------------------------------------------------------------------
KDA_HEAD_DIM = 128
KDA_CONV_WIDTH = 4
KDA_NORM_METHODS = ("forward_cuda", "forward_native")


def _kda_cu_seqlens(cu_seqlens: Tensor) -> Tensor:
    cu = cu_seqlens.detach().to("cpu", torch.int32)
    if cu.ndim != 1 or cu.numel() < 2 or int(cu[0]) != 0 or bool((cu.diff() <= 0).any()):
        raise ValueError(f"cu_seqlens must be 0-based strictly increasing, got {cu.tolist()}")
    return cu


class KDACausalConvAdapter(_Adapter):
    """``causal_conv1d_fn`` exactly as the rollout prefill calls it for one of q / k / v.

    ``x`` is a ``[T, HD]`` column slice of the packed q|k|v projection.  The conv state pool uses the
    rollout's mamba layout -- storage ``[slot, W-1, 3HD]`` seen as ``(slot, 3HD, W-1)`` -- because the
    Triton kernel takes the state strides as ``tl.constexpr``: the dim-first layout compiles to code that
    rounds some outputs differently (1 bf16 ulp)."""

    def forward(self, x: Tensor, weight: Tensor, conv_state: Tensor, cu_seqlens: Tensor) -> Tensor:
        _tensor(x, "x", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=2, dtype=torch.float32, contiguous=False)
        _tensor(conv_state, "conv_state", ndim=3, dtype=torch.bfloat16, contiguous=False)
        _same_device({"x": x, "weight": weight, "conv_state": conv_state})
        if x.stride(1) != 1 or weight.stride(1) != 1 or weight.shape[0] != x.shape[1]:
            raise ValueError("conv x / weight must be row-major with matching channels")
        if conv_state.stride(1) != 1 or conv_state.shape[1] != x.shape[1]:
            raise ValueError("conv_state must be the rollout layout (slot, dim, W-1) with stride(1) == 1")
        cu = _kda_cu_seqlens(cu_seqlens)
        R, T, dev = cu.numel() - 1, x.shape[0], x.device
        if int(cu[-1]) != T or conv_state.shape[0] < R + 1:
            raise ValueError("cu_seqlens / conv_state slots do not cover the batch")
        from types import SimpleNamespace

        fn = _symbol("vllm.model_executor.layers.mamba.ops.causal_conv1d", "causal_conv1d_fn")
        meta_fn = _symbol("vllm.v1.attention.backends.utils", "compute_causal_conv1d_metadata")
        nums, batch_ptr, offset_ptr = meta_fn(cu, device=dev)
        meta = SimpleNamespace(nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offset_ptr)
        out = fn(
            x.transpose(0, 1),
            weight,
            None,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=torch.zeros(R, dtype=torch.bool, device=dev),
            cache_indices=torch.arange(1, R + 1, dtype=torch.int32, device=dev),
            query_start_loc=cu.to(dev),
            metadata=meta,
        ).transpose(0, 1)
        return _output_boundary(out[:T], x, weight)


class KDACFBIPrefillAdapter(_Adapter):
    """``k31_kda_cfbi.mixed`` with ``ctx = 0`` for every sequence: all tokens take the chunk-aligned
    prefill engine, which the CF-BI contract makes bitwise equal to the decode engine."""

    def forward(self, q, k, v, g, beta, A_log, dt_bias, cu_seqlens: Tensor, lower: float) -> Tensor:
        for name, value in (("q", q), ("k", k), ("v", v), ("g", g)):
            _tensor(value, name, ndim=4, dtype=torch.bfloat16, contiguous=False)
        _tensor(beta, "beta", ndim=3, dtype=torch.bfloat16, contiguous=False)
        _tensor(A_log, "A_log", ndim=1, dtype=torch.float32)
        _tensor(dt_bias, "dt_bias", ndim=1, dtype=torch.float32)
        _same_device({"q": q, "k": k, "v": v, "g": g, "beta": beta, "A_log": A_log, "dt_bias": dt_bias})
        if q.shape[-1] != KDA_HEAD_DIM or not lower < 0:
            raise ValueError("K31 CF-BI needs head_dim 128 and a negative gate lower bound")
        cu = _kda_cu_seqlens(cu_seqlens)
        R, H, D, dev = cu.numel() - 1, q.shape[2], q.shape[3], q.device
        if int(cu[-1]) != q.shape[1]:
            raise ValueError("cu_seqlens does not cover the tokens")
        cfbi = _k31_module("k31_kda_cfbi")
        rows = getattr(cfbi, "ROWS", cfbi.BT)
        if cfbi.D != KDA_HEAD_DIM or (cfbi.BT, rows, cfbi.ROWF) not in ((64, 64, 640), (16, 17, 256)):
            raise NotImplementedError(f"unexpected CF-BI geometry D={cfbi.D} BT={cfbi.BT} rows={rows} ROWF={cfbi.ROWF}")
        # state pool: E fp32 [H,D,D], row cache fp32 [H,rows,ROWF] (v5 64x640, v6 17x256), pos int32 [H] per slot.
        # mixed() zeroes E and pos of ctx == 0 rows; the row cache is zeroed too so that it is finite (results do
        # not depend on it).
        pools = (
            torch.zeros(R + 1, H, D, D, dtype=torch.float32, device=dev),
            torch.zeros(R + 1, H, rows, cfbi.ROWF, dtype=torch.float32, device=dev),
            torch.zeros(R + 1, H, dtype=torch.int32, device=dev),
        )
        out = cfbi.mixed(
            q, k, v, g, beta, A_log, dt_bias,
            torch.arange(1, R + 1, dtype=torch.int32, device=dev),
            cu[:R].tolist(), cu.diff().tolist(), [0] * R,
            *pools, float(lower),
        )
        return _output_boundary(out, q, k, v, g, beta, A_log, dt_bias)


class KDAGatedNormAdapter(_Adapter):
    """``FusedRMSNormGated(head_dim, activation="sigmoid")`` through the method the engine's CustomOp
    dispatched to (recorded per layer in the rollout dump: ``forward_cuda``).  The Triton kernel writes its
    output into ``x``; the adapter normalises a private copy."""

    def __init__(self, method: str = "forward_cuda") -> None:
        super().__init__()
        if method not in KDA_NORM_METHODS:
            raise ValueError(f"norm method must be one of {KDA_NORM_METHODS}")
        self.method = method

    def forward(self, x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor:
        _tensor(x, "x", ndim=4, dtype=torch.bfloat16)
        _tensor(gate, "gate", ndim=3, dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=1, dtype=torch.bfloat16)
        _same_device({"x": x, "gate": gate, "weight": weight})
        from types import SimpleNamespace

        cls = _symbol("vllm.third_party.flash_linear_attention.ops.kda", "FusedRMSNormGated")
        holder = SimpleNamespace(weight=weight, bias=None, activation="sigmoid", eps=float(eps))
        scratch = x.clone()
        return _output_boundary(getattr(cls, self.method)(holder, scratch, gate), x, gate, weight)


__all__ += [
    "KDACFBIPrefillAdapter",
    "KDACausalConvAdapter",
    "KDAGatedNormAdapter",
]
