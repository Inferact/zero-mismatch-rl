"""K31 checkpoint contracts: frozen base load and LoRA adapter load/export.

* Base masters are loaded **as stored** (no dtype conversion): bf16 dense
  weights, fp32 KDA state parameters, packed MXFP4 experts (uint8 nibbles +
  uint8 E8M0 scales).  Names map onto vLLM's fused TP=1 layout so every
  projection has the rollout's (K, N): KDA ``in_proj_qkvgfab`` = [q; k; v; g;
  f_a; b], ``conv1d`` = [q; k; v], DSA ``fused_qkv_a_proj`` = [q_a;
  kv_a_proj_with_mqa], ``gate_up_proj`` = [gate; up], experts ``w13`` = [w1; w3].
* The base is frozen during K31 RL, so weight sync exports only the LoRA
  adapter, in the directory format vLLM's ``--lora-modules`` loads
  (``adapter_config.json`` + ``adapter_model.safetensors``, peft names).  The
  raw peft tensors are the training parameters; vLLM's scaling fold is applied
  by the forward (``primitive.linear.vllm_fold_lora_b``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.k31.config import K31Config
from megatron.lite.primitive.parallel import ParallelState

HF_PREFIX = "language_model.model."
HF_LM_HEAD = "language_model.lm_head.weight"
PEFT_PREFIX = "base_model.model." + HF_PREFIX

_SHARED_EXPERT_LORA = ("lora_w1_A", "lora_w3_A", "lora_w2_B")
_PER_EXPERT_LORA = ("lora_w1_B", "lora_w3_B", "lora_w2_A")
_EXPERT_BUFFERS = (
    "w13_weight_packed",
    "w13_weight_scale",
    "w2_weight_packed",
    "w2_weight_scale",
)
_TOP_LEVEL = {
    "embed_tokens.weight": HF_PREFIX + "embed_tokens.weight",
    "norm.weight": HF_PREFIX + "norm.weight",
    "output_attn_res.norm_weight": HF_PREFIX + "output_attn_res_norm.weight",
    "output_attn_res.proj_weight": HF_PREFIX + "output_attn_res_proj.weight",
    "lm_head.weight": HF_LM_HEAD,
}
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def EXPERT_CLASSIFIER(name: str) -> bool:
    """Per-expert parameters (reduced over the expert-DP group only).

    The shared outer expert-LoRA factors are replicated on every EP rank and
    therefore are *not* expert parameters (their gradients reduce over DP).
    """

    if ".mlp.experts." not in name:
        return False
    return not name.rsplit(".", 1)[-1] in _SHARED_EXPERT_LORA


def PLACEMENT_FN(param_name: str) -> list:
    """DCP placements on the runtime meshes (dense ``[PP, DP, CP, TP]``, expert ``[PP, EDP, EP, ETP]``).

    Per-expert parameters hold this EP rank's contiguous block of experts
    (``K31WeightSpec.expert_range``), i.e. ``Shard(0)`` on the EP dimension.
    Declaring them replicated would make DCP keep one EP rank's block for every
    rank.  Everything else (dense, shared outer expert-LoRA factors) is
    replicated; pipeline stages are told apart by the per-stage key prefix.
    """

    from torch.distributed.tensor import Replicate, Shard

    if EXPERT_CLASSIFIER(param_name):
        return [Replicate(), Replicate(), Shard(0), Replicate()]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


@dataclass(frozen=True)
class _LoadItem:
    hf_names: tuple[str, ...]
    kind: str = "concat"  # concat | a_log | expert


class K31WeightSpec:
    """Native (vLLM fused layout) <-> release checkpoint names for one model chunk."""

    def __init__(self, config: K31Config, ps: ParallelState | None = None) -> None:
        self.config = config
        self.ps = ps or ParallelState()

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def validate_load(self, ps: ParallelState) -> None:
        if (ps.tp_size, ps.etp_size) != (1, 1):
            raise NotImplementedError("K31 vLLM checkpoint load requires TP/ETP=1")
        if ps.ep_size <= 0 or self.config.num_experts % ps.ep_size:
            raise ValueError(f"EP={ps.ep_size} must divide {self.config.num_experts} experts")
        self.ps = ps

    # ------------------------------------------------------------------ names
    def _hf_layer_prefix(self, layer_idx: int, attr: str) -> str:
        if attr.startswith("mlp.") and self.config.is_moe_layer(layer_idx):
            attr = "block_sparse_moe." + attr[len("mlp.") :]
        return f"{HF_PREFIX}layers.{layer_idx}.{attr}"

    def plan(self, model: nn.Module) -> dict[str, _LoadItem]:
        """Native state name -> checkpoint sources, for every non-LoRA tensor of ``model``."""

        from megatron.lite.model.k31.vllm.model import K31Linear

        linears = {
            name: module for name, module in model.named_modules() if isinstance(module, K31Linear)
        }
        plan: dict[str, _LoadItem] = {}
        names = [n for n, _ in model.named_parameters()] + [n for n, _ in model.named_buffers()]
        for native in names:
            if ".lora" in native or native.startswith("lora"):
                continue
            if native in _TOP_LEVEL:
                plan[native] = _LoadItem((_TOP_LEVEL[native],))
                continue
            match = _LAYER_RE.match(native)
            if match is None:
                raise KeyError(f"no checkpoint mapping for {native}")
            layer, attr = match.groups()
            layer_idx = int(layer)
            module_path, _, leaf = native.rpartition(".")
            if module_path in linears and leaf == "weight":
                linear = linears[module_path]
                parent = attr.rpartition(".")[0].rpartition(".")[0]
                plan[native] = _LoadItem(
                    tuple(
                        self._hf_layer_prefix(layer_idx, f"{parent}.{s}.weight")
                        for s in linear.slice_names
                    )
                )
                continue
            if attr in ("self_attention_res.norm_weight", "self_attention_res.proj_weight",
                        "mlp_res.norm_weight", "mlp_res.proj_weight"):
                site, kind = attr.split(".")
                suffix = "norm.weight" if kind == "norm_weight" else "proj.weight"
                plan[native] = _LoadItem((self._hf_layer_prefix(layer_idx, f"{site}_{suffix}"),))
                continue
            if attr == "self_attn.conv1d.weight":
                plan[native] = _LoadItem(
                    tuple(
                        self._hf_layer_prefix(layer_idx, f"self_attn.{p}_conv1d.weight")
                        for p in ("q", "k", "v")
                    )
                )
                continue
            if attr == "self_attn.A_log":
                plan[native] = _LoadItem(
                    (self._hf_layer_prefix(layer_idx, "self_attn.A_log"),), kind="a_log"
                )
                continue
            if attr.startswith("mlp.experts.") and leaf in _EXPERT_BUFFERS:
                plan[native] = _LoadItem(
                    (self._hf_layer_prefix(layer_idx, "mlp.experts"),), kind="expert:" + leaf
                )
                continue
            plan[native] = _LoadItem((self._hf_layer_prefix(layer_idx, attr),))
        return plan

    # ------------------------------------------------------------------ tensors
    def expert_range(self) -> range:
        local = self.config.num_experts // self.ps.ep_size
        start = self.ps.ep_rank * local
        return range(start, start + local)

    def materialize(self, item: _LoadItem, read, target: torch.Tensor) -> torch.Tensor:
        if item.kind.startswith("expert:"):
            leaf = item.kind.split(":", 1)[1]
            base = item.hf_names[0]
            proj, field = leaf.split("_weight_")
            suffix = "weight_packed" if field == "packed" else "weight_scale"
            parts = ("w1", "w3") if proj == "w13" else ("w2",)
            rows = [
                torch.cat([read(f"{base}.{e}.{p}.{suffix}") for p in parts], dim=0)
                for e in self.expert_range()
            ]
            return torch.stack(rows, dim=0)
        tensors = [read(name) for name in item.hf_names]
        if item.kind == "a_log":
            (value,) = tensors
            if value.dim() == 4:
                value = value.view(value.shape[2])
            return value.narrow(0, 0, target.shape[0])
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


class _SafetensorsIndex:
    def __init__(self, path: str | os.PathLike, filename: str = "model.safetensors.index.json"):
        self.root = Path(path)
        index = self.root / filename
        if index.is_file():
            self.weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            single = self.root / filename.replace(".index.json", "")
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as handle:
                self.weight_map = {k: single.name for k in handle.keys()}
        self._handles: dict[str, object] = {}

    def read(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        file = self.weight_map.get(name)
        if file is None:
            raise KeyError(f"{name} is missing from {self.root}")
        handle = self._handles.get(file)
        if handle is None:
            handle = safe_open(str(self.root / file), framework="pt", device="cpu")
            self._handles[file] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


def _models(chunks: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    return [chunks] if isinstance(chunks, nn.Module) else list(chunks)


def load_hf_weights(
    chunk: nn.Module,
    hf_path: str,
    model_cfg: K31Config,
    ps: ParallelState,
) -> None:
    """Load the frozen base of one model chunk, bitwise as stored in the release."""

    if not hf_path:
        return
    spec = K31WeightSpec(model_cfg, ps)
    spec.validate_load(ps)
    index = _SafetensorsIndex(hf_path)
    state = dict(chunk.named_parameters())
    state.update(dict(chunk.named_buffers()))
    try:
        for native, item in spec.plan(chunk).items():
            target = state[native]
            value = spec.materialize(item, index.read, target)
            if value.dtype != target.dtype:
                raise TypeError(
                    f"{native}: checkpoint dtype {value.dtype} != native {target.dtype}; "
                    "K31 loads base tensors as stored"
                )
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"{native}: checkpoint shape {tuple(value.shape)} != native {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(value, non_blocking=False)
    finally:
        index.close()


# ---------------------------------------------------------------------------
# LoRA adapter (vLLM --lora-modules directory)
# ---------------------------------------------------------------------------
def _lora_entries(model: nn.Module, spec: K31WeightSpec) -> Iterator[tuple[str, nn.Parameter, str, str]]:
    """Yield ``(native_name, param, peft_name, kind)`` for every LoRA tensor of a chunk.

    ``kind`` is ``dense``, ``expert_shared`` or ``expert_local``.
    """

    from megatron.lite.model.k31.vllm.primitive.linear import LoRAProjection

    for path, module in model.named_modules():
        if isinstance(module, LoRAProjection):
            match = _LAYER_RE.match(path)
            if match is None:
                raise KeyError(f"LoRA outside a decoder layer: {path}")
            layer, attr = match.groups()
            layer_idx = int(layer)
            # attr = "<parent>.<linear>.lora"
            parent = attr.rsplit(".", 2)[0]
            for name in module.slice_names:
                peft_base = spec._hf_layer_prefix(layer_idx, f"{parent}.{name}")
                peft_base = "base_model.model." + peft_base
                yield (f"{path}.lora_A.{name}", module.lora_A[name], f"{peft_base}.lora_A.weight", "dense")
                yield (f"{path}.lora_B.{name}", module.lora_B[name], f"{peft_base}.lora_B.weight", "dense")
    for native, param in model.named_parameters():
        leaf = native.rsplit(".", 1)[-1]
        if ".mlp.experts." not in native or leaf not in _SHARED_EXPERT_LORA + _PER_EXPERT_LORA:
            continue
        match = _LAYER_RE.match(native)
        layer, attr = match.groups()
        layer_idx = int(layer)
        proj, factor = leaf[len("lora_"):].split("_")
        peft = "base_model.model." + spec._hf_layer_prefix(
            layer_idx, f"mlp.experts.{proj}.lora_{factor}.weight"
        )
        kind = "expert_shared" if leaf in _SHARED_EXPERT_LORA else "expert_local"
        yield native, param, peft, kind


def lora_adapter_config(spec_like, target_modules: Iterable[str]) -> dict:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(spec_like.rank),
        "lora_alpha": spec_like.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": bool(spec_like.use_rslora),
        "target_modules": sorted(target_modules),
    }


def load_lora_adapter(
    chunk: nn.Module,
    adapter_dir: str | os.PathLike,
    model_cfg: K31Config,
    ps: ParallelState,
    *,
    strict: bool = True,
) -> int:
    """Load a peft adapter (the rollout's ``--lora-modules`` directory) into a chunk.

    Tensors are cast like vLLM (``.to(bf16)``); the scaling fold is *not*
    applied here.  Returns the number of tensors loaded.
    """

    lora_spec = getattr(chunk, "lora_spec", None)
    if lora_spec is None:
        raise RuntimeError("model was built without a LoRA spec")
    config = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    rank, alpha = int(config["r"]), config.get("lora_alpha", config["r"])
    if rank != lora_spec.rank or float(alpha) != float(lora_spec.alpha):
        raise ValueError(
            f"adapter r/alpha {rank}/{alpha} != model {lora_spec.rank}/{lora_spec.alpha}"
        )
    if bool(config.get("use_rslora", False)) != bool(lora_spec.use_rslora):
        raise ValueError("adapter use_rslora differs from the model LoRA spec")
    spec = K31WeightSpec(model_cfg, ps)
    spec.validate_load(ps)
    index = _SafetensorsIndex(adapter_dir, "adapter_model.safetensors.index.json")
    loaded = 0
    try:
        for native, param, peft, kind in _lora_entries(chunk, spec):
            if peft not in index.weight_map:
                if strict:
                    raise KeyError(f"adapter lacks {peft}")
                continue
            value = index.read(peft)
            if kind == "expert_local":
                r = spec.expert_range()
                value = value[r.start : r.stop]
            value = value.to(torch.bfloat16)
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"{peft}: shape {tuple(value.shape)} != {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(value)
            loaded += 1
    finally:
        index.close()
    return loaded


# Name of the int64 ``[rollout_ep_rank, rollout_ep_size, num_experts]`` tensor that heads an
# expert-sliced export: vLLM's own key (``LoRAModel.from_tensor_dict``), else the K31 overlay's.
# The rollout's LoRA loader checks it against its own EP placement and refuses the payload on any
# mismatch; a loader that does not know it rejects the unknown name.
try:
    from vllm.lora.lora_model import LORA_EXPERT_SLICE_KEY as LORA_EXPERT_SLICE_TAG
except ImportError:
    LORA_EXPERT_SLICE_TAG = "__k31_lora_expert_slice__"

# The K31 adapter export can hand each rollout rank only its own experts (``lora_expert_slice``).
SUPPORTS_LORA_EXPERT_SLICE = True


@dataclass(frozen=True)
class _ExpertExchange:
    """Rows of the per-expert LoRA factors moved for one expert-sliced export.

    ``send[i]`` is ``(local row, rows)`` this rank sends to EP-group rank ``i``;
    ``recv[i]`` the ``(rows)`` it receives from ``i``.  Received pieces arrive in
    group-rank order, which is expert order, and concatenate to ``target``.
    """

    send: tuple[tuple[int, int], ...]
    recv: tuple[int, ...]
    target: tuple[int, int]


def _expert_exchange_from_rows(
    rows: list[tuple[int, int, int]],
    ep_rank: int,
    num_experts: int,
    rollout_ep_rank: int,
    rollout_ep_size: int,
) -> _ExpertExchange:
    """Exchange plan of actor EP rank ``ep_rank`` given every group rank's
    ``(actor ep_rank, rollout ep_rank, rollout ep_size)`` in group-rank order."""

    if rollout_ep_size <= 0 or num_experts % rollout_ep_size:
        raise ValueError(f"rollout EP={rollout_ep_size} must divide {num_experts} experts")
    if not 0 <= rollout_ep_rank < rollout_ep_size:
        raise ValueError(f"rollout EP rank {rollout_ep_rank} outside [0, {rollout_ep_size})")
    ep_size = len(rows)
    if num_experts % ep_size:
        raise ValueError(f"actor EP={ep_size} must divide {num_experts} experts")
    if [row[0] for row in rows] != list(range(ep_size)):
        raise RuntimeError(f"actor EP ranks are not in EP-group order: {[row[0] for row in rows]}")
    if {row[2] for row in rows} != {rollout_ep_size}:
        raise ValueError(f"rollout EP size differs across the actor EP group: {rows}")
    if tuple(rows[ep_rank][1:]) != (rollout_ep_rank, rollout_ep_size):
        raise RuntimeError(f"rank {ep_rank} row {rows[ep_rank]} is not its own slice request")
    per_rollout = num_experts // rollout_ep_size
    per_actor = num_experts // ep_size
    start = rollout_ep_rank * per_rollout
    own = ep_rank * per_actor
    send: list[tuple[int, int]] = []
    recv: list[int] = []
    for peer, (_, peer_rollout_rank, _) in enumerate(rows):
        lo = max(own, peer_rollout_rank * per_rollout)
        hi = min(own + per_actor, (peer_rollout_rank + 1) * per_rollout)
        send.append((lo - own, hi - lo) if hi > lo else (0, 0))
        peer_own = peer * per_actor
        recv.append(max(min(peer_own + per_actor, start + per_rollout) - max(peer_own, start), 0))
    if sum(recv) != per_rollout:
        raise RuntimeError(f"expert exchange covers {sum(recv)} of {per_rollout} experts")
    return _ExpertExchange(tuple(send), tuple(recv), (start, per_rollout))


def _plan_expert_exchange(
    ps: ParallelState, num_experts: int, rollout_ep_rank: int, rollout_ep_size: int
) -> _ExpertExchange:
    """Plan which per-expert rows each actor EP rank sends so every rank ends with the
    contiguous expert block ``[r * E/R, (r + 1) * E/R)`` of its colocated rollout rank ``r``.

    The actor EP group holds every expert once (expert-DP replicas are bitwise equal), so
    the exchange stays inside it whatever the actor and rollout EP sizes are.
    """

    mine = (int(ps.ep_rank), int(rollout_ep_rank), int(rollout_ep_size))
    if ps.ep_size > 1:
        device = torch.device("cuda", torch.cuda.current_device())
        local = torch.tensor(mine, dtype=torch.int64, device=device)
        gathered = [torch.empty_like(local) for _ in range(ps.ep_size)]
        dist.all_gather(gathered, local, group=ps.ep_group)
        rows = [tuple(int(v) for v in g.tolist()) for g in gathered]
    else:
        rows = [mine]
    return _expert_exchange_from_rows(
        rows, int(ps.ep_rank), num_experts, int(rollout_ep_rank), int(rollout_ep_size)
    )


def _stage_plans_from_rows(
    rows: list[tuple[int, ...]],
    ep_rank: int,
    num_experts: int,
    pp_size: int,
) -> list[_ExpertExchange]:
    """Exchange plans of actor EP rank ``ep_rank``, one per target PP stage.

    ``rows`` holds, in EP-group order, every EP peer's ``(actor ep_rank, r_0, R_0, ...,
    r_{P-1}, R_{P-1})``: the rollout EP rank/size wanted by that peer's PP peer at each
    stage.  ``plans[t]`` moves experts inside this stage's EP group so the rank at EP
    position ``e`` ends with the block its PP peer at stage ``t`` must yield.
    """

    if any(len(row) != 1 + 2 * pp_size for row in rows):
        raise RuntimeError(f"expert slice rows do not match PP={pp_size}: {rows}")
    plans = []
    for target in range(pp_size):
        lo = 1 + 2 * target
        stage_rows = [(row[0], row[lo], row[lo + 1]) for row in rows]
        plans.append(
            _expert_exchange_from_rows(
                stage_rows, ep_rank, num_experts, rows[ep_rank][lo], rows[ep_rank][lo + 1]
            )
        )
    if len({plan.target[1] for plan in plans}) != 1:
        raise ValueError(f"rollout EP size differs across PP stages: {rows}")
    return plans


def _plan_expert_exchanges(
    ps: ParallelState, num_experts: int, rollout_ep_rank: int, rollout_ep_size: int
) -> list[_ExpertExchange]:
    """``_plan_expert_exchange`` for every PP stage (one plan when PP=1).

    With PP>1 a stage's experts must reach the colocated rollout rank of every PP peer,
    so the stage's EP group runs one exchange per target stage; ``_pipeline_entries``
    then scatters the blocks over the PP group.
    """

    if ps.pp_size <= 1:
        return [_plan_expert_exchange(ps, num_experts, rollout_ep_rank, rollout_ep_size)]
    device = torch.device("cuda", torch.cuda.current_device())
    mine = torch.tensor([rollout_ep_rank, rollout_ep_size], dtype=torch.int64, device=device)
    by_stage = [torch.empty_like(mine) for _ in range(ps.pp_size)]
    dist.all_gather(by_stage, mine, group=ps.pp_group)
    local = torch.cat([torch.tensor([int(ps.ep_rank)], dtype=torch.int64, device=device), *by_stage])
    if ps.ep_size > 1:
        gathered = [torch.empty_like(local) for _ in range(ps.ep_size)]
        dist.all_gather(gathered, local, group=ps.ep_group)
    else:
        gathered = [local]
    rows = [tuple(int(v) for v in g.tolist()) for g in gathered]
    plans = _stage_plans_from_rows(rows, int(ps.ep_rank), num_experts, ps.pp_size)
    per_rollout = num_experts // rollout_ep_size
    if plans[ps.pp_rank].target != (rollout_ep_rank * per_rollout, per_rollout):
        raise RuntimeError(f"stage {ps.pp_rank} plan does not target its own rollout rank")
    return plans


def _exchange_experts(tensor: torch.Tensor, plan: _ExpertExchange, ps: ParallelState) -> torch.Tensor:
    """This rank's rollout expert block of one per-expert factor (data movement only)."""

    if ps.ep_size == 1:
        start, count = plan.target
        return tensor[start : start + count].contiguous()
    tensor = tensor.contiguous()
    inputs = [tensor[row : row + count] for row, count in plan.send]
    outputs = [tensor.new_empty((count, *tensor.shape[1:])) for count in plan.recv]
    dist.all_to_all(outputs, inputs, group=ps.ep_group)
    return torch.cat([out for out, count in zip(outputs, plan.recv) if count], dim=0)


def export_hf_weights(
    chunks: nn.Module | Iterable[nn.Module],
    model_cfg: K31Config,
    ps: ParallelState,
    **kwargs,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Weight sync: yield the LoRA adapter tensors with peft names (base is frozen).

    By default per-expert factors are all-gathered over the EP group so every rank
    yields the full ``[num_experts, ...]`` tensor.  With
    ``lora_expert_slice=(rollout_ep_rank, rollout_ep_size)`` every rank instead yields
    only the block ``[r * E/R, (r + 1) * E/R)`` its colocated rollout rank keeps
    (exchanged inside the actor EP group), headed by ``LORA_EXPERT_SLICE_TAG``; dense
    and shared outer factors are yielded whole either way.

    With PP>1 every rank still yields the whole adapter (all stages, layer order): a
    stage broadcasts its dense/shared tensors over the PP group, and for per-expert
    factors either broadcasts the EP-gathered tensor or, sliced, exchanges inside its
    EP group once per target stage and scatters each PP peer its own rollout block.
    """

    kwargs.pop("target", None)
    kwargs.pop("resync_config", None)
    # The adapter is small and yielded tensor by tensor; the bucket size is moot.
    kwargs.pop("buffer_max_size_bytes", None)
    to_cpu = bool(kwargs.pop("cpu", False))
    expert_slice = kwargs.pop("lora_expert_slice", None)
    export_dtype = kwargs.pop("export_dtype", None)
    if export_dtype is not None and str(export_dtype).removeprefix("torch.") not in (
        "bfloat16",
        "bf16",
    ):
        raise ValueError(f"K31 adapter export is BF16 (the rollout's LoRA dtype), got {export_dtype!r}")
    if kwargs:
        raise TypeError(f"unsupported export options: {sorted(kwargs)}")
    if (ps.tp_size, ps.etp_size) != (1, 1):
        raise NotImplementedError("K31 adapter export requires TP/ETP=1")
    spec = K31WeightSpec(model_cfg, ps)
    items = [entry for model in _models(chunks) for entry in _lora_entries(model, spec)]
    plans = None
    if expert_slice is not None:
        rollout_ep_rank, rollout_ep_size = (int(v) for v in expert_slice)
        plans = _plan_expert_exchanges(ps, model_cfg.num_experts, rollout_ep_rank, rollout_ep_size)
        tag = torch.tensor(
            [rollout_ep_rank, rollout_ep_size, model_cfg.num_experts], dtype=torch.int64
        )
        yield LORA_EXPERT_SLICE_TAG, tag if to_cpu else tag.cuda()
    if ps.pp_size <= 1:
        plan = plans[0] if plans is not None else None
        entries = (_export_entry(param, peft, kind, ps, plan) for _n, param, peft, kind in items)
    else:
        entries = _pipeline_entries(items, ps, plans)
    for peft, tensor in entries:
        yield peft, tensor.cpu() if to_cpu else tensor


def _export_entry(
    param: torch.Tensor, peft: str, kind: str, ps: ParallelState, plan: _ExpertExchange | None = None
):
    """One adapter tensor as vLLM loads it, BF16.

    Per-expert factors are all-gathered over the EP group (whole ``[num_experts, ...]``),
    or, with an exchange ``plan``, reduced to the rollout expert block the plan targets.
    """

    tensor = param.detach()
    full = getattr(tensor, "full_tensor", None)
    if callable(full):
        tensor = full()
    if kind == "expert_local" and plan is not None:
        tensor = _exchange_experts(tensor, plan, ps)
    elif kind == "expert_local" and ps.ep_size > 1:
        parts = [torch.empty_like(tensor) for _ in range(ps.ep_size)]
        dist.all_gather(parts, tensor.contiguous(), group=ps.ep_group)
        tensor = torch.cat(parts, dim=0)
    return peft, tensor.to(torch.bfloat16)


def _pipeline_entries(
    items, ps: ParallelState, plans: list[_ExpertExchange] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Every PP rank yields the whole adapter, stage by stage in layer order.

    Each rollout worker is fed by its colocated trainer rank, so with PP>1 every
    rank must also yield the other stages' LoRA tensors (Megatron's per-tensor
    generator does the same).  Stage ``s`` broadcasts its manifest (peft name,
    exported shape) over the PP group, then each tensor after the EP gather inside
    the stage.  Only one tensor is in flight at a time.

    Expert-sliced (``plans`` from ``_plan_expert_exchanges``): for a per-expert factor
    stage ``s`` runs one EP exchange per target stage ``t`` (``plans[t]``), so its rank at
    EP position ``e`` holds the rollout block wanted by its PP peer at stage ``t``, and
    scatters those blocks over the PP group.  Every rank then yields only its own
    rollout rank's experts; nothing whole crosses the PP group for those factors.
    """

    if ps.pp_group is None or ps.pp_global_ranks is None:
        raise RuntimeError("K31 PP adapter export requires the pipeline group")
    if plans is not None and len(plans) != ps.pp_size:
        raise RuntimeError(f"{len(plans)} expert exchange plans for PP={ps.pp_size}")
    device = torch.device("cuda", torch.cuda.current_device())
    manifest_local = []
    for _native, param, peft, kind in items:
        shape = list(param.shape)
        sliced = kind == "expert_local" and plans is not None
        if sliced:
            shape[0] = plans[ps.pp_rank].target[1]
        elif kind == "expert_local":
            shape[0] *= ps.ep_size
        manifest_local.append((peft, tuple(shape), sliced))
    for stage in range(ps.pp_size):
        src = ps.pp_global_ranks[stage]
        mine = ps.pp_rank == stage
        manifest = [manifest_local if mine else None]
        dist.broadcast_object_list(manifest, src=src, group=ps.pp_group)
        for index, (name, shape, sliced) in enumerate(manifest[0]):
            if mine and sliced:
                _native, param, peft, kind = items[index]
                blocks = [
                    _export_entry(param, peft, kind, ps, plans[target])[1].contiguous()
                    for target in range(ps.pp_size)
                ]
                for block in blocks:
                    if tuple(block.shape) != tuple(shape):
                        raise RuntimeError(f"{name}: block {tuple(block.shape)} != manifest {shape}")
                tensor = torch.empty_like(blocks[stage])
                dist.scatter(tensor, scatter_list=blocks, src=src, group=ps.pp_group)
                del blocks
            elif sliced:
                tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
                dist.scatter(tensor, scatter_list=None, src=src, group=ps.pp_group)
            else:
                if mine:
                    _native, param, peft, kind = items[index]
                    _name, tensor = _export_entry(param, peft, kind, ps)
                    tensor = tensor.contiguous()
                    if tuple(tensor.shape) != tuple(shape):
                        raise RuntimeError(f"{name}: exported {tuple(tensor.shape)} != manifest {shape}")
                else:
                    tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
                dist.broadcast(tensor, src=src, group=ps.pp_group)
            yield name, tensor


def save_hf_weights(
    chunks: nn.Module | Iterable[nn.Module],
    path: str,
    model_cfg: K31Config,
    ps: ParallelState,
    **kwargs,
) -> None:
    """Write a vLLM ``--lora-modules`` adapter directory; global rank 0 writes.

    Every rank takes part in the export collectives (EP gather, PP broadcast), so
    with PP>1 rank 0 holds the other stages' tensors as well.
    """

    models = _models(chunks)
    lora_spec = getattr(models[0], "lora_spec", None)
    if lora_spec is None:
        raise RuntimeError("model was built without a LoRA spec")
    writer = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    tensors = {}
    for name, tensor in export_hf_weights(models, model_cfg, ps, **kwargs):
        if writer:
            tensors[name] = tensor.cpu().contiguous()
    if not writer:
        return
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    _write_adapter_verified(tensors, out / "adapter_model.safetensors")
    config_tmp = out / f"adapter_config.json.tmp{os.getpid()}"
    config_tmp.write_text(
        json.dumps(lora_adapter_config(lora_spec, lora_spec.targets), indent=1) + "\n",
        encoding="utf-8",
    )
    os.replace(config_tmp, out / "adapter_config.json")


def _write_adapter_verified(tensors: dict[str, torch.Tensor], target: Path) -> None:
    """Write the adapter to a temp file, read it back and compare every tensor bitwise, then rename.

    Same failure class as the optimizer files (``primitive/ckpt/dcp.py``): a short or partly
    rewritten file on a network filesystem must never become the published adapter.
    """
    from safetensors.torch import load_file, save_file

    for attempt in range(3):
        tmp = target.with_name(f"{target.name}.tmp{attempt}")
        save_file(tensors, str(tmp))
        try:
            back = load_file(str(tmp))
            bad = next(
                (
                    name
                    for name, tensor in tensors.items()
                    if name not in back
                    or back[name].dtype != tensor.dtype
                    or tuple(back[name].shape) != tuple(tensor.shape)
                    or not torch.equal(
                        back[name].contiguous().view(torch.uint8), tensor.contiguous().view(torch.uint8)
                    )
                ),
                None,
            )
            if bad is None and len(back) != len(tensors):
                bad = f"{len(back)} tensors read back, {len(tensors)} written"
        except Exception as exc:  # noqa: BLE001
            bad = repr(exc)[:300]
        if bad is None:
            os.replace(tmp, target)
            return
        print(f"[megatron.lite] adapter {tmp} failed verification ({bad}); rewriting", flush=True)
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"adapter {target} failed verification three times")


__all__ = [
    "EXPERT_CLASSIFIER",
    "HF_PREFIX",
    "K31WeightSpec",
    "LORA_EXPERT_SLICE_TAG",
    "PEFT_PREFIX",
    "PLACEMENT_FN",
    "SUPPORTS_LORA_EXPERT_SLICE",
    "export_hf_weights",
    "load_hf_weights",
    "load_lora_adapter",
    "lora_adapter_config",
    "save_hf_weights",
]
