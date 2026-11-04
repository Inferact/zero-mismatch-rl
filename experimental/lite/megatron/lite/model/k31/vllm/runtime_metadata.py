"""Runtime contract of the K31 vLLM implementation.

Like DS4, this module builds the small per-batch runtime contract directly; it
does not construct a vLLM model or read vLLM's process-global forward context.

* ``initialize_k31_vllm_batch_invariance`` follows ``VLLM_BATCH_INVARIANT`` as DS4 does.  With it
  set, it turns on vLLM BI mode in the training process and fails closed unless the rollout's BI
  overlay chain and environment are in effect (same bi-gemm, KDA CF-BI, MoE tree, LoRA leaf), so the
  trainer's log-probs equal the rollout's.  Without it (a BI=0 rollout) the trainer keeps its own
  kernels, which need no process-wide mode, and its log-probs differ from the rollout's by kernel
  rounding: old_log_prob is then recomputed by the trainer, never taken from the rollout.
* ``K31Runtime`` carries the per-microbatch facts every layer slot needs
  (token count, positions, sequence boundaries, the LoRA kernel metadata).
* Per-layer kernel metadata (KDA state pools, DSA indexer/FlashMLA metadata,
  MoE metadata) is owned by the KDA / DSA / MoE slots; they add their builders
  here or in their own modules and pass them through ``attention_metadata`` /
  ``moe_metadata`` dicts keyed by global layer id, exactly as DS4 does.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist

from megatron.lite.primitive.kernels.vllm_k31 import _k31_module, _symbol, k31_vllm_package
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts.config import refuse_retired_env

# Environment of the rollout that changes numerics
# or the kernel path.  Values must match exactly.
_REQUIRED_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
}
# cuBLAS settings the BI init writes; cuBLAS reads them when its first handle
# is created, so they must already be in the process environment.
_CUBLAS_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "CUBLASLT_WORKSPACE_SIZE": "1"}
# vLLM itself rewrites CUBLAS_WORKSPACE_CONFIG later in the process
# (``override_envs_for_invariance`` -> ":4096:8"), so the check reads the values
# this module saw first, not the live environment.
_STARTUP_CUBLAS_ENV = {key: os.environ.get(key) for key in _CUBLAS_ENV}
_INITIALIZED = False

_MIN_BI_GEMM_VERSION = "r5"

# K31-only environment switches the trainer no longer reads, and what replaced each one (the
# generic MLITE_* ones are refused by every Megatron Lite config: ``mlite.config.RETIRED_ENV``).
RETIRED_ENV = {
    "K31_RECOMPUTE_NUM_LAYERS": "impl_cfg.recompute_num_layers",
    "K31_RECOMPUTE_KEEP_EXPERT_LATENT": "always on (routed latent kept on the recompute tape)",
    "K31_MOE_COMPACT_LEAVES": "always on (compact leaf-row expert path)",
    "K31_MOE_FI_MAX_ROWS": "megatron.lite.primitive.alignment.vllm_fi_w4a16_moe.FI_MAX_ROWS",
    "K31_MOE_BWD_ROWS": "megatron.lite.primitive.alignment.vllm_fi_w4a16_moe.EXPERT_BWD_ROWS",
    "K31_MOE_BWD_EXPERTS": "megatron.lite.primitive.alignment.vllm_fi_w4a16_moe.EXPERT_BWD_CHUNK",
    "K31_MOE_BWD_FEWSYNC": "always on (few-sync split expert backward)",
    "K31_MOE_FI_CACHE_W13": "always off (FI w13 rebuilt per call from the checkpoint buffers)",
    "K31_MOE_EP_TRANSPORT": "megatron.lite.model.k31.vllm.moe.default_ep_transport",
    "K31_MXFP4_DEQUANT_CUDA": "always on (k31_mxfp4_dequant CUDA kernel)",
    "K31_MOE_EXPERT_BWD": "megatron.lite.primitive.alignment.vllm_fi_w4a16_moe.EXPERT_BWD_MODE",
    "K31_BWD_SPLIT_MM": "megatron.lite.primitive.alignment.split_bf16_mm.SPLIT_MM",
    "K31_BWD_SPLIT_TERMS": "megatron.lite.primitive.alignment.split_bf16_mm.SPLIT_TERMS",
    "K31_BWD_TC_KCHUNK": "megatron.lite.primitive.alignment.split_bf16_mm.TC_KCHUNK",
    "K31_TP_TREE": "always on under VLLM_BATCH_INVARIANT=1",
    "K31_MOE_TREE": "always on under VLLM_BATCH_INVARIANT=1",
    "K31_TP_ORDERED": "always on under VLLM_BATCH_INVARIANT=1",
    "K31_BI_NCCL_UNPIN": "always on under VLLM_BATCH_INVARIANT=1",
    "K31_MOE_FI_W4A16": "on for the K31 checkpoint in both arms (model-gated in vLLM)",
    "K31_BI_GEMM": "always on under VLLM_BATCH_INVARIANT=1",
    "K31_LIN_LORA_CUDA": "always on under VLLM_BATCH_INVARIANT=1",
    "K31_MOE_LORA_CUDA": "on for the K31 checkpoint in both arms (model-gated in vLLM)",
    "K31_TREE_BUILD": "none: the K31 CUDA modules are built with the vLLM and mLite wheels",
    "K31_BI_GEMM_BUILD": "none: the K31 CUDA modules are built with the vLLM wheel",
    "K31_KDA_CFBI_BUILD": "none: the K31 CUDA modules are built with the vLLM wheel",
    "K31_BI_GEMM_CUTLASS_INCLUDE": "none: the K31 CUDA modules are built with the vLLM wheel",
}


def _check_env() -> None:
    bad = {
        key: os.environ.get(key)
        for key, value in _REQUIRED_ENV.items()
        if os.environ.get(key) != value
    }
    if bad:
        raise RuntimeError(
            "K31 vLLM training requires the rollout BI environment; offending: "
            + ", ".join(f"{k}={v!r}" for k, v in sorted(bad.items()))
        )


def _check_retired_env() -> None:
    refuse_retired_env(RETIRED_ENV, "K31 training")


def _check_cublas_env() -> None:
    bad = {k: _STARTUP_CUBLAS_ENV[k] for k, v in _CUBLAS_ENV.items() if _STARTUP_CUBLAS_ENV[k] != v}
    if bad:
        raise RuntimeError(
            "set the BI cuBLAS environment before the process starts "
            f"({', '.join(f'{k}={v}' for k, v in _CUBLAS_ENV.items())}); got {bad}"
        )


def _check_no_cublas_env() -> None:
    """BI=0: neither BI cuBLAS setting, inherited (e.g. from the shell that ran ``ray start``: a Ray
    runtime_env cannot unset a variable) or set since."""
    found = {k: _STARTUP_CUBLAS_ENV[k] or os.environ.get(k) for k in _CUBLAS_ENV}
    found = {k: v for k, v in found.items() if v is not None}
    if found:
        raise RuntimeError(
            "VLLM_BATCH_INVARIANT=0 trainer with the BI cuBLAS environment "
            f"({', '.join(f'{k}={v!r}' for k, v in found.items())}): start Ray without it"
        )


def initialize_k31_vllm_batch_invariance() -> None:
    """Mirror the rollout's process-local batch-invariant mode (``VLLM_BATCH_INVARIANT``).

    BI=1 fails closed unless the rollout's BI environment is in effect; BI=0 fails closed when the
    process carries the BI cuBLAS environment.  Idempotent: later calls in the same process return
    after the first success.
    """

    global _INITIALIZED
    if _INITIALIZED:
        return
    _check_retired_env()
    batch_invariant = bool(importlib.import_module("vllm.envs").VLLM_BATCH_INVARIANT)
    if batch_invariant:
        _check_env()
        _check_cublas_env()
        # The trainer's own reductions need no vLLM NCCL determinism pins (1 channel, NVLS off); the
        # rollout skips them too on its K31 tree / tp-ordered topologies.
        _symbol("vllm.model_executor.determinism.batch_invariant", "init_batch_invariance")(
            pin_nccl=False
        )
    else:
        _check_no_cublas_env()
    k31_vllm_package()
    bi_gemm = _k31_module("k31_bi_gemm")
    if getattr(bi_gemm, "VERSION", "") < _MIN_BI_GEMM_VERSION:
        raise RuntimeError(
            f"K31 bi-gemm overlay {_MIN_BI_GEMM_VERSION}+ required, got "
            f"{getattr(bi_gemm, 'VERSION', None)!r}"
        )
    if not bi_gemm.enabled():
        raise RuntimeError("K31 bi-gemm needs an sm_10x GPU")
    # The trainer's KDA runs the CF-BI engines in both arms (KDACFBIPrefillAdapter): load them
    # here, not at the first microbatch.  cfbi.enabled() is the rollout's CF-BI routing, which
    # follows VLLM_BATCH_INVARIANT.
    cfbi = _k31_module("k31_kda_cfbi")
    cfbi.kernels()
    if batch_invariant:
        if not cfbi.enabled():
            raise RuntimeError("K31 KDA CF-BI engine is disabled")
        if torch.backends.cuda.matmul.fp32_precision != "ieee":
            raise RuntimeError("BI init did not pin FP32 matmul precision to IEEE")
    _INITIALIZED = True


# ---------------------------------------------------------------------------
# per-microbatch runtime
# ---------------------------------------------------------------------------
@dataclass
class K31Runtime:
    """Per-microbatch facts shared by every layer slot (caller-owned)."""

    num_tokens: int
    seq_lens: tuple[int, ...]
    cu_seqlens: torch.Tensor  # int32 [B + 1]
    positions: torch.Tensor  # int64 [T]
    lora_meta: Any | None = None  # vLLM LoRAKernelMeta, all tokens on slot 0


def build_lora_kernel_meta(num_tokens: int, device: torch.device | str, *, max_loras: int = 1):
    """The punica metadata of a rollout batch whose every token uses adapter slot 0."""

    meta_cls = _symbol("vllm.lora.ops.triton_ops.lora_kernel_metadata", "LoRAKernelMeta")
    meta = meta_cls.make(max_loras, num_tokens, device=device)
    meta.prepare_tensors(torch.zeros(num_tokens, dtype=torch.int32, device=device))
    return meta


def build_k31_runtime(
    seq_lens: list[int] | tuple[int, ...],
    device: torch.device | str,
    *,
    with_lora: bool,
    max_loras: int = 1,
) -> K31Runtime:
    lengths = tuple(int(n) for n in seq_lens)
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("K31 batches need positive sequence lengths")
    total = sum(lengths)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), 0)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in lengths])
    return K31Runtime(
        num_tokens=total,
        seq_lens=lengths,
        cu_seqlens=cu.to(device),
        positions=positions.to(device),
        lora_meta=build_lora_kernel_meta(total, device, max_loras=max_loras)
        if with_lora
        else None,
    )


@contextmanager
def k31_vllm_forward_context(
    batch: Any,
    parallel_state: ParallelState,
    *,
    vllm_config: Any,
) -> Iterator[None]:
    """Install the vLLM per-microbatch token contract (DPMetadata) for MoE kernels."""

    input_ids = getattr(batch, "input_ids", None)
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("K31 vLLM batches require tensor input_ids")
    local = int(getattr(batch, "total_tokens", None) or input_ids.numel())
    if local <= 0:
        raise ValueError("K31 vLLM batches must contain at least one token")
    tokens = torch.tensor([local], dtype=torch.int32, device=input_ids.device)
    if parallel_state.ep_size > 1:
        if parallel_state.ep_group is None:
            raise RuntimeError("K31 vLLM with EP>1 requires an initialized EP group")
        gathered = [torch.empty_like(tokens) for _ in range(parallel_state.ep_size)]
        dist.all_gather(gathered, tokens, group=parallel_state.ep_group)
        tokens = torch.cat(gathered)
    dp_metadata = _symbol("vllm.forward_context", "DPMetadata")(tokens.cpu())
    forward_context = _symbol("vllm.forward_context", "create_forward_context")(
        None, vllm_config, dp_metadata=dp_metadata
    )
    with _symbol("vllm.forward_context", "override_forward_context")(forward_context):
        yield


__all__ = [
    "K31Runtime",
    "build_k31_runtime",
    "build_lora_kernel_meta",
    "initialize_k31_vllm_batch_invariance",
    "k31_vllm_forward_context",
]
