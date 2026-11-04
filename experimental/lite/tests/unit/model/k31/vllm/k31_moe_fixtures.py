"""Loaders and synthetic data for the K31 MoE tests.

Checkpoint and adapter paths come from the environment (never hard-coded):
``K31_MODEL_DIR`` (an HF-format K31 slice), ``K31_ADAPTER_DIR`` (PEFT adapter
in vLLM ``--lora-modules`` format), ``K31_ROLLOUT_DUMP`` (MoE rollout dump directory).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm.model import K31DenseMLP, K31LoRASpec
from megatron.lite.model.k31.vllm.moe import K31MoE
from megatron.lite.primitive.parallel import ParallelState

_TEXT = "language_model.model.layers.{layer}."
_ADAPTER = "base_model.model.language_model.model.layers.{layer}."
# --max-lora-rank of the rollout (k31 deploy flags)
ROLLOUT_MAX_LORA_RANK = 32
MOE_LORA_TARGETS = frozenset({"experts", "gate_proj", "up_proj", "down_proj"})


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def adapter_config(adapter_dir: Path) -> dict:
    return json.loads((adapter_dir / "adapter_config.json").read_text())


def lora_spec(
    adapter_dir: Path | None = None, *, rank: int = 32, alpha: float = 64.0
) -> K31LoRASpec:
    if adapter_dir is not None:
        ad = adapter_config(adapter_dir)
        rank, alpha = int(ad["r"]), float(ad["lora_alpha"])
    return K31LoRASpec(
        rank=rank, alpha=alpha, max_rank=ROLLOUT_MAX_LORA_RANK, targets=MOE_LORA_TARGETS
    )


def default_config() -> K31Config:
    return K31Config()


def ep_state(group=None) -> ParallelState:
    """A ParallelState carrying only the EP facts the MoE slot reads."""

    import torch.distributed as dist

    ps = ParallelState()
    if group is not None:
        ps.ep_group = group
        ps.ep_size = dist.get_world_size(group)
        ps.ep_rank = dist.get_rank(group)
    return ps


def make_moe(
    config: K31Config,
    *,
    layer: int = 1,
    group=None,
    lora: K31LoRASpec | None = None,
    transport: str | None = None,
    device="cuda",
) -> K31MoE:
    with torch.device(device):
        moe = K31MoE(
            config,
            layer_idx=layer,
            ps=ep_state(group),
            selected_stages=frozenset({"router_moe", "deepep"}),
            lora=lora if lora is not None else lora_spec(),
        )
    if transport is not None and group is not None:
        moe.transport = transport
    for name, p in moe.named_parameters():
        p.requires_grad_(".lora" in name or "lora_" in name)
    return moe


class _Checkpoint:
    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self.dir = model_dir
        self.map = json.loads((model_dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self._open = {}
        self._safe_open = safe_open

    def get(self, key: str) -> torch.Tensor:
        shard = self.map[key]
        handle = self._open.get(shard)
        if handle is None:
            handle = self._safe_open(str(self.dir / shard), framework="pt", device="cpu")
            self._open[shard] = handle
        return handle.get_tensor(key)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def load_moe_layer(
    model_dir: Path,
    layer: int,
    *,
    adapter_dir: Path | None,
    group=None,
    transport: str | None = None,
    device="cuda",
) -> K31MoE:
    config = K31Config.from_hf(str(model_dir))
    moe = make_moe(
        config,
        layer=layer,
        group=group,
        lora=lora_spec(adapter_dir),
        transport=transport,
        device=device,
    )
    ck = _Checkpoint(model_dir)
    p = _TEXT.format(layer=layer) + "block_sparse_moe."
    ex = moe.experts
    with torch.no_grad():
        moe.gate.weight.copy_(ck.get(p + "gate.weight"))
        moe.gate.e_score_correction_bias.copy_(ck.get(p + "gate.e_score_correction_bias").float())
        moe.routed_expert_down_proj.weight.copy_(ck.get(p + "routed_expert_down_proj.weight"))
        moe.routed_expert_norm.weight.copy_(ck.get(p + "routed_expert_norm.weight"))
        moe.routed_expert_up_proj.weight.copy_(ck.get(p + "routed_expert_up_proj.weight"))
        se = moe.shared_experts
        se.gate_up_proj.weight.copy_(
            torch.cat(
                [
                    ck.get(p + "shared_experts.gate_proj.weight"),
                    ck.get(p + "shared_experts.up_proj.weight"),
                ]
            )
        )
        se.down_proj.weight.copy_(ck.get(p + "shared_experts.down_proj.weight"))
        w13, s13, w2, s2 = [], [], [], []
        for e in range(ex.local_offset, ex.local_offset + ex.local_experts):
            q = p + f"experts.{e}."
            w13.append(
                torch.cat(
                    [_bytes(ck.get(q + "w1.weight_packed")), _bytes(ck.get(q + "w3.weight_packed"))]
                )
            )
            s13.append(
                torch.cat(
                    [_bytes(ck.get(q + "w1.weight_scale")), _bytes(ck.get(q + "w3.weight_scale"))]
                )
            )
            w2.append(_bytes(ck.get(q + "w2.weight_packed")))
            s2.append(_bytes(ck.get(q + "w2.weight_scale")))
        ex.w13_weight_packed.copy_(torch.stack(w13))
        ex.w13_weight_scale.copy_(torch.stack(s13))
        ex.w2_weight_packed.copy_(torch.stack(w2))
        ex.w2_weight_scale.copy_(torch.stack(s2))
    if adapter_dir is not None:
        load_moe_adapter(moe, adapter_dir, layer)
    return moe


def _adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(adapter_dir / "adapter_model.safetensors"))


def load_moe_adapter(moe: K31MoE, adapter_dir: Path, layer: int, tensors=None) -> None:
    t = tensors if tensors is not None else _adapter_tensors(adapter_dir)
    p = _ADAPTER.format(layer=layer) + "block_sparse_moe."
    ex = moe.experts
    sl = slice(ex.local_offset, ex.local_offset + ex.local_experts)
    with torch.no_grad():
        ex.lora_w1_A.copy_(t[p + "experts.w1.lora_A.weight"].reshape(ex.lora_w1_A.shape))
        ex.lora_w3_A.copy_(t[p + "experts.w3.lora_A.weight"].reshape(ex.lora_w3_A.shape))
        ex.lora_w2_B.copy_(t[p + "experts.w2.lora_B.weight"].reshape(ex.lora_w2_B.shape))
        ex.lora_w1_B.copy_(t[p + "experts.w1.lora_B.weight"][sl])
        ex.lora_w3_B.copy_(t[p + "experts.w3.lora_B.weight"][sl])
        ex.lora_w2_A.copy_(t[p + "experts.w2.lora_A.weight"][sl])
    load_mlp_adapter(moe.shared_experts, t, p + "shared_experts.")


def load_mlp_adapter(mlp: K31DenseMLP, t: dict, prefix: str) -> None:
    with torch.no_grad():
        for proj, name in (
            ("gate_up_proj", "gate_proj"),
            ("gate_up_proj", "up_proj"),
            ("down_proj", "down_proj"),
        ):
            lora = getattr(mlp, proj).lora
            lora.lora_A[name].copy_(t[prefix + f"{name}.lora_A.weight"])
            lora.lora_B[name].copy_(t[prefix + f"{name}.lora_B.weight"])


def load_dense_mlp(model_dir: Path, *, adapter_dir: Path | None, device="cuda") -> K31DenseMLP:
    config = K31Config.from_hf(str(model_dir))
    with torch.device(device):
        mlp = K31DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            beta=config.activation_situ_beta,
            linear_beta=config.activation_situ_linear_beta,
            lora=lora_spec(adapter_dir),
        )
    ck = _Checkpoint(model_dir)
    p = _TEXT.format(layer=0) + "mlp."
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(
            torch.cat([ck.get(p + "gate_proj.weight"), ck.get(p + "up_proj.weight")])
        )
        mlp.down_proj.weight.copy_(ck.get(p + "down_proj.weight"))
    if adapter_dir is not None:
        load_mlp_adapter(mlp, _adapter_tensors(adapter_dir), _ADAPTER.format(layer=0) + "mlp.")
    return mlp


def runtime_for(num_tokens: int, device="cuda"):
    from megatron.lite.model.k31.vllm.runtime_metadata import build_k31_runtime

    return build_k31_runtime([num_tokens], device, with_lora=True)


# --------------------------------------------------------------------------- synthetic
def random_mxfp4_experts(
    num_experts: int, hidden: int, inter: int, *, seed: int, device="cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random MXFP4 bytes with a realistic E8M0 spread (identical on every rank)."""

    g = torch.Generator(device=device).manual_seed(seed)

    def nib(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device=device)

    def scl(*shape):
        return torch.randint(117, 125, shape, generator=g, dtype=torch.uint8, device=device)

    w13 = nib(num_experts, 2 * inter, hidden // 2)
    s13 = scl(num_experts, 2 * inter, hidden // 32)
    w2 = nib(num_experts, hidden, inter // 2)
    s2 = scl(num_experts, hidden, inter // 32)
    return w13, s13, w2, s2


def load_random_experts(moe: K31MoE, full: tuple[torch.Tensor, ...]) -> None:
    """Global random bytes -> this rank's EP slice."""

    ex = moe.experts
    sl = slice(ex.local_offset, ex.local_offset + ex.local_experts)
    with torch.no_grad():
        for dst, src in zip(
            (ex.w13_weight_packed, ex.w13_weight_scale, ex.w2_weight_packed, ex.w2_weight_scale),
            full,
            strict=True,
        ):
            dst.copy_(src[sl])


def random_expert_lora(moe: K31MoE, *, seed: int, scale_b: float = 0.02) -> None:
    """Non-zero expert LoRA, identical across ranks (per-expert slices by global id)."""

    ex = moe.experts
    cfg = moe.config
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = ex.lora_w1_A.shape[1]
    h, i, e = cfg.routed_expert_hidden_size, cfg.moe_intermediate_size, cfg.num_experts
    w1_a = torch.randn(1, r, h, generator=g) / h**0.5
    w3_a = torch.randn(1, r, h, generator=g) / h**0.5
    w2_b = torch.randn(1, h, r, generator=g) * scale_b
    w1_b = torch.randn(e, i, r, generator=g) * scale_b
    w3_b = torch.randn(e, i, r, generator=g) * scale_b
    w2_a = torch.randn(e, r, i, generator=g) / i**0.5
    sl = slice(ex.local_offset, ex.local_offset + ex.local_experts)
    with torch.no_grad():
        for dst, src in (
            (ex.lora_w1_A, w1_a),
            (ex.lora_w3_A, w3_a),
            (ex.lora_w2_B, w2_b),
            (ex.lora_w1_B, w1_b[sl]),
            (ex.lora_w3_B, w3_b[sl]),
            (ex.lora_w2_A, w2_a[sl]),
        ):
            dst.copy_(src.to(dst.dtype))


def random_routing(t: int, num_experts: int, top_k: int, *, seed: int, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(t)])
    w = torch.rand(t, top_k, generator=g) + 0.05
    w = w / w.sum(-1, keepdim=True)
    return ids.to(device), w.float().to(device)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    it = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.int64: torch.int64,
        torch.int32: torch.int32,
    }
    view = it.get(a.dtype)
    if view is None:
        return torch.equal(a, b)
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape:
        return -1
    it = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
    return int((a.contiguous().view(it) != b.contiguous().view(it)).sum())
