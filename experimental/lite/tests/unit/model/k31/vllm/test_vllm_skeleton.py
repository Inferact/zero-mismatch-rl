from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model import registry
from megatron.lite.model.k31.config import K31Config
from megatron.lite.model.k31.vllm import checkpoint as ckpt
from megatron.lite.model.k31.vllm import protocol, runtime_metadata
from megatron.lite.model.k31.vllm.model import (
    AttnResStream,
    K31Linear,
    K31Model,
    fold_attn_res_for_pipeline,
    unfold_attn_res_from_pipeline,
)
from megatron.lite.primitive.modules.lora import LoraConfig
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts import ParallelConfig

from _tiny import TINY_TARGETS, tiny_config, tiny_lora


# --------------------------------------------------------------------------- registry
def test_registry_exposes_only_the_vllm_impl() -> None:
    assert registry.resolve_runtime_model_name("k31", "vllm") == "k31"
    assert registry.TRAIN_RUNTIME_MODULES["k31"] == "megatron.lite.model.k31.vllm.protocol"
    with pytest.raises(ValueError):
        registry.resolve_runtime_model_name("k31", "lite")
    assert registry.get_train_runtime_module("k31") is protocol


# --------------------------------------------------------------------------- config
def _hf_text(**extra) -> dict:
    text = {
        "hidden_size": 16,
        "vocab_size": 32,
        "num_hidden_layers": 4,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3],
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
        "dsa_config": {"dsa_layers": [4], "index_topk": 8, "index_num_heads": 2},
        "num_nextn_predict_layers": 0,
        "model_type": "ignored",
    }
    text.update(extra)
    return {"text_config": text, "tie_word_embeddings": False}


def test_config_converts_one_based_layer_lists() -> None:
    config = K31Config._from_hf_dict(_hf_text())
    assert config.kda_layers == [0, 1, 2]
    assert config.dsa_layers == [3]
    assert config.kda_in_proj_sizes == (8, 8, 8, 8, 4, 2)
    assert [config.prev_valid_blocks(i) for i in (0, 1, 12, 13)] == [0, 1, 1, 2]
    assert config.num_attn_res_blocks == 1


def test_config_rejects_uncovered_layers() -> None:
    with pytest.raises(ValueError, match="neither"):
        K31Config._from_hf_dict(_hf_text(num_hidden_layers=5))


# --------------------------------------------------------------------------- contract
def _impl(**overrides) -> protocol.ImplConfig:
    values = dict(
        parallel=ParallelConfig(),
        use_deepep=True,
        lora=LoraConfig(rank=2, alpha=4, target_modules=tuple(sorted(TINY_TARGETS))),
        selector=protocol.SelectorConfig((0, 1), ("attn", "moe")),
        rollout_max_model_len=5120,
    )
    values.update(overrides)
    return protocol.ImplConfig(**values)


def test_contract_accepts_the_deployment_shape() -> None:
    protocol._validate_contract(tiny_config(), _impl())


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"parallel": ParallelConfig(tp=2)}, NotImplementedError),
        ({"parallel": ParallelConfig(cp=2)}, NotImplementedError),
        ({"use_deepep": False}, NotImplementedError),
        ({"lora": LoraConfig()}, NotImplementedError),
        ({"mtp_enable": True}, NotImplementedError),
        ({"selector": protocol.SelectorConfig((1, 2), ("attn",))}, ValueError),
        ({"selector": protocol.SelectorConfig((0,), ("moe",))}, ValueError),
        ({"max_lora_rank": 1}, ValueError),
        ({"rollout_max_model_len": 0}, ValueError),
        ({"optimizer": "fsdp2", "parallel": ParallelConfig(ep=2)}, NotImplementedError),
    ],
    ids=[
        "tp",
        "cp",
        "no-deepep",
        "no-lora",
        "mtp",
        "non-prefix",
        "layer0-partial",
        "rank",
        "no-max-model-len",
        "fsdp2-ep",
    ],
)
def test_contract_fails_closed(overrides, error) -> None:
    with pytest.raises(error):
        protocol._validate_contract(tiny_config(), _impl(**overrides))


class TestSelectRouterLayers:
    @pytest.mark.parametrize(
        "width,expected",
        [(3, [1, 2]), (2, [0, 1])],
        ids=["per-decoder-layer", "per-moe-layer"],
    )
    def test_keeps_moe_columns(self, width, expected) -> None:
        config = tiny_config()  # layer 0 dense, layers 1-2 MoE
        routed = torch.arange(5 * width * 2).reshape(5, width, 2)
        out = protocol.select_router_layers(config, routed)
        assert torch.equal(out, routed[:, expected])

    def test_nested_rows(self) -> None:
        config = tiny_config()
        rows = [torch.arange(n * 3 * 2).reshape(n, 3, 2) for n in (4, 1)]
        nested = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
        out = protocol.select_router_layers(config, nested)
        for got, row in zip(out.unbind(0), rows, strict=True):
            assert torch.equal(got, row[:, 1:])

    def test_rejects_unknown_width(self) -> None:
        with pytest.raises(ValueError, match="layer columns"):
            protocol.select_router_layers(tiny_config(), torch.zeros(2, 5, 2, dtype=torch.long))


def test_attention_metadata_shares_one_dsa_object(monkeypatch) -> None:
    from megatron.lite.model.k31.vllm import dsa

    calls = []

    def fake(runtime, *, max_model_len):
        calls.append(max_model_len)
        return object()

    monkeypatch.setattr(dsa, "build_dsa_metadata", fake)
    config = tiny_config(num_hidden_layers=4, kda_layers=[0, 2], dsa_layers=[1, 3])
    meta = protocol.build_attention_metadata(
        config, object(), layer_ids=(0, 1, 2, 3), rollout_max_model_len=5120
    )
    assert calls == [5120]
    assert meta[0] is None and meta[2] is None
    assert meta[1] is meta[3] is not None


def test_selector_expands_aliases_and_rejects_unknown() -> None:
    selector = protocol.SelectorConfig((0,), ("attn", "moe"))
    assert selector.module_names == ("attn_res", "linear", "kda", "dsa", "router_moe", "deepep")
    with pytest.raises(ValueError):
        protocol.SelectorConfig((0,), ("mhc",))


# --------------------------------------------------------------------------- checkpoint names
def test_checkpoint_plan_maps_fused_layouts() -> None:
    config = tiny_config()
    model = K31Model(config, lora=tiny_lora())
    plan = ckpt.K31WeightSpec(config).plan(model)
    p = ckpt.HF_PREFIX
    assert plan["layers.0.self_attn.in_proj_qkvgfab.weight"].hf_names == tuple(
        f"{p}layers.0.self_attn.{n}.weight"
        for n in ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj")
    )
    assert plan["layers.0.self_attn.conv1d.weight"].hf_names == tuple(
        f"{p}layers.0.self_attn.{n}_conv1d.weight" for n in "qkv"
    )
    assert plan["layers.0.mlp.gate_up_proj.weight"].hf_names == (
        f"{p}layers.0.mlp.gate_proj.weight",
        f"{p}layers.0.mlp.up_proj.weight",
    )
    assert plan["layers.1.mlp.shared_experts.gate_up_proj.weight"].hf_names == (
        f"{p}layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
        f"{p}layers.1.block_sparse_moe.shared_experts.up_proj.weight",
    )
    assert plan["layers.2.self_attn.fused_qkv_a_proj.weight"].hf_names == (
        f"{p}layers.2.self_attn.q_a_proj.weight",
        f"{p}layers.2.self_attn.kv_a_proj_with_mqa.weight",
    )
    assert plan["layers.2.self_attn.indexer.wq_b.weight"].hf_names == (
        f"{p}layers.2.self_attn.indexer.wq_b.weight",
    )
    assert plan["layers.0.self_attention_res.proj_weight"].hf_names == (
        f"{p}layers.0.self_attention_res_proj.weight",
    )
    assert plan["lm_head.weight"].hf_names == ("language_model.lm_head.weight",)
    assert plan["layers.1.mlp.experts.w13_weight_packed"].kind == "expert:w13_weight_packed"
    assert not any("lora" in name for name in plan)


def _write_checkpoint(tmp_path, model, config) -> dict[str, torch.Tensor]:
    """Synthetic release checkpoint covering exactly the plan's sources."""

    from safetensors.torch import save_file

    spec = ckpt.K31WeightSpec(config)
    state = dict(model.named_parameters())
    state.update(dict(model.named_buffers()))
    gen = torch.Generator().manual_seed(7)
    tensors: dict[str, torch.Tensor] = {}
    slice_sizes = {
        f"{name}.weight": module.slice_sizes
        for name, module in model.named_modules()
        if isinstance(module, K31Linear)
    }

    def rand(shape, dtype):
        if dtype == torch.uint8:
            return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
        return torch.randn(shape, generator=gen).to(dtype)

    for native, item in spec.plan(model).items():
        target = state[native]
        if item.kind.startswith("expert:"):
            leaf = item.kind.split(":", 1)[1]
            proj, field = leaf.split("_weight_")
            suffix = "weight_packed" if field == "packed" else "weight_scale"
            parts = ("w1", "w3") if proj == "w13" else ("w2",)
            rows = target.shape[1] // len(parts)
            for e in range(config.num_experts):
                for part in parts:
                    tensors[f"{item.hf_names[0]}.{e}.{part}.{suffix}"] = rand(
                        (rows, target.shape[2]), torch.uint8
                    )
        elif item.kind == "a_log":
            tensors[item.hf_names[0]] = rand((target.shape[0] + 3,), target.dtype)
        else:
            sizes = slice_sizes.get(native) or (target.shape[0] // len(item.hf_names),) * len(
                item.hf_names
            )
            for name, rows in zip(item.hf_names, sizes, strict=True):
                tensors[name] = rand((rows, *target.shape[1:]), target.dtype)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    return tensors


@pytest.mark.parametrize("ep_size,ep_rank", [(1, 0), (2, 1)], ids=["ep1", "ep2-rank1"])
def test_base_load_is_bitwise_and_expert_sharded(tmp_path, ep_size, ep_rank) -> None:
    config = tiny_config()
    ps = ParallelState(ep_size=ep_size, ep_rank=ep_rank)
    model = K31Model(config, ps=ps, lora=tiny_lora())
    tensors = _write_checkpoint(tmp_path, K31Model(config, lora=tiny_lora()), config)
    ckpt.load_hf_weights(model, str(tmp_path), config, ps)
    p = ckpt.HF_PREFIX
    fused = model.layers["0"].self_attn.in_proj_qkvgfab.weight
    assert torch.equal(fused[:8], tensors[f"{p}layers.0.self_attn.q_proj.weight"])
    assert torch.equal(fused[-2:], tensors[f"{p}layers.0.self_attn.b_proj.weight"])
    a_log = model.layers["0"].self_attn.A_log
    assert torch.equal(a_log, tensors[f"{p}layers.0.self_attn.A_log"][: a_log.shape[0]])
    experts = model.layers["1"].mlp.experts
    local = config.num_experts // ep_size
    first = ep_rank * local
    base = f"{p}layers.1.block_sparse_moe.experts"
    assert experts.w13_weight_packed.shape[0] == local
    assert torch.equal(
        experts.w13_weight_packed[0],
        torch.cat(
            [tensors[f"{base}.{first}.w1.weight_packed"], tensors[f"{base}.{first}.w3.weight_packed"]]
        ),
    )
    assert torch.equal(experts.w2_weight_scale[-1], tensors[f"{base}.{first + local - 1}.w2.weight_scale"])


def test_base_load_rejects_dtype_conversion(tmp_path) -> None:
    config = tiny_config()
    model = K31Model(config, lora=tiny_lora())
    tensors = _write_checkpoint(tmp_path, model, config)
    from safetensors.torch import save_file

    name = f"{ckpt.HF_PREFIX}layers.0.self_attn.dt_bias"
    tensors[name] = tensors[name].to(torch.bfloat16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(TypeError, match="as stored"):
        ckpt.load_hf_weights(model, str(tmp_path), config, ParallelState())


def _write_adapter(tmp_path, model, rank=2, alpha=4.0) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    spec = ckpt.K31WeightSpec(model.config)
    gen = torch.Generator().manual_seed(11)
    tensors = {}
    for _native, param, peft, kind in ckpt._lora_entries(model, spec):
        shape = tuple(param.shape)
        if kind == "expert_local":
            shape = (model.config.num_experts, *shape[1:])
        tensors[peft] = torch.randn(shape, generator=gen).to(torch.bfloat16)
    tmp_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": alpha, "target_modules": sorted(TINY_TARGETS)})
    )
    return tensors


def test_lora_peft_names_match_the_rollout_adapter_layout() -> None:
    model = K31Model(tiny_config(), lora=tiny_lora())
    names = {peft for _n, _p, peft, _k in ckpt._lora_entries(model, ckpt.K31WeightSpec(model.config))}
    p = "base_model.model." + ckpt.HF_PREFIX
    assert f"{p}layers.0.self_attn.g_proj.lora_A.weight" in names
    assert f"{p}layers.0.mlp.down_proj.lora_B.weight" in names
    assert f"{p}layers.1.block_sparse_moe.shared_experts.up_proj.lora_A.weight" in names
    assert f"{p}layers.1.block_sparse_moe.experts.w1.lora_A.weight" in names
    assert f"{p}layers.1.block_sparse_moe.experts.w2.lora_A.weight" in names
    assert f"{p}layers.2.self_attn.kv_a_proj_with_mqa.lora_B.weight" in names
    assert not any("kv_b_proj" in n or "indexer" in n or "routed_expert" in n for n in names)


def test_lora_adapter_roundtrip_is_bytewise(tmp_path) -> None:
    config = tiny_config()
    model = K31Model(config, lora=tiny_lora())
    source = _write_adapter(tmp_path / "in", model)
    loaded = ckpt.load_lora_adapter(model, tmp_path / "in", config, ParallelState())
    assert loaded == len(source)
    ckpt.save_hf_weights(model, str(tmp_path / "out"), config, ParallelState())
    from safetensors.torch import load_file

    exported = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    assert exported.keys() == source.keys()
    for name, tensor in source.items():
        assert torch.equal(exported[name], tensor), name
    exported_cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert (exported_cfg["r"], exported_cfg["lora_alpha"]) == (2, 4.0)


def test_lora_adapter_rejects_mismatched_scaling(tmp_path) -> None:
    config = tiny_config()
    model = K31Model(config, lora=tiny_lora())
    _write_adapter(tmp_path, model, alpha=8.0)
    with pytest.raises(ValueError, match="alpha"):
        ckpt.load_lora_adapter(model, tmp_path, config, ParallelState())


@pytest.mark.parametrize(
    "name,expert",
    [
        ("layers.1.mlp.experts.lora_w1_B", True),
        ("layers.1.mlp.experts.lora_w2_A", True),
        ("layers.1.mlp.experts.lora_w1_A", False),
        ("layers.1.mlp.experts.lora_w2_B", False),
        ("layers.1.mlp.shared_experts.gate_up_proj.lora.lora_A.gate_proj", False),
        ("layers.0.self_attn.in_proj_qkvgfab.lora.lora_B.q_proj", False),
    ],
    ids=["w1B", "w2A", "shared-w1A", "shared-w2B", "shared-mlp", "dense"],
)
def test_expert_classifier_keeps_shared_outer_factors_dense(name, expert) -> None:
    assert ckpt.EXPERT_CLASSIFIER(name) is expert


# --------------------------------------------------------------------------- PP stream
class TestPipelineStream:
    @pytest.mark.parametrize("num_blocks", [0, 1, 2, 8], ids=["no-bank", "one", "two", "full-bank"])
    def test_roundtrip_keeps_delta_separate(self, num_blocks) -> None:
        torch.manual_seed(3)
        t, h = 5, 16
        prefix = torch.randn(t, h).bfloat16()
        delta = torch.randn(t, h).bfloat16()
        blocks = tuple(torch.randn(t, h).bfloat16() for _ in range(num_blocks))
        packed = fold_attn_res_for_pipeline(AttnResStream(prefix, delta, blocks))
        assert packed.shape == (t, 1, (2 + num_blocks) * h)
        stream = unfold_attn_res_from_pipeline(packed, hidden_size=h, num_blocks=num_blocks)
        # The receiving layer gets the PP=1 attn_res operands: prefix and delta unsummed.
        assert torch.equal(stream.prefix, prefix)
        assert torch.equal(stream.delta, delta)
        assert all(torch.equal(a, b) for a, b in zip(stream.blocks, blocks, strict=True))
        assert all(x.is_contiguous() for x in (stream.prefix, stream.delta, *stream.blocks))

    def test_roundtrip_routes_gradients(self) -> None:
        t, h = 3, 8
        leaves = [torch.randn(t, h, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        packed = fold_attn_res_for_pipeline(AttnResStream(leaves[0], leaves[1], tuple(leaves[2:])))
        stream = unfold_attn_res_from_pipeline(packed, hidden_size=h, num_blocks=2)
        weights = [float(i + 1) for i in range(4)]
        total = sum(w * x.sum() for w, x in zip(weights, (stream.prefix, stream.delta, *stream.blocks)))
        total.backward()
        for w, leaf in zip(weights, leaves, strict=True):
            assert torch.equal(leaf.grad, torch.full_like(leaf, w))

    def test_rejects_missing_delta_and_wrong_width(self) -> None:
        t, h = 2, 4
        with pytest.raises(ValueError, match="delta"):
            fold_attn_res_for_pipeline(AttnResStream(torch.zeros(t, h), None, ()))
        packed = torch.zeros(t, 1, 3 * h)
        with pytest.raises(ValueError, match="width"):
            unfold_attn_res_from_pipeline(packed, hidden_size=h, num_blocks=2)

    def test_later_stage_names_layers_by_global_index(self) -> None:
        ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_layout="Ett|tL")
        model = K31Model(tiny_config(), ps=ps)
        assert list(model.layer_indices) == [2]
        assert list(model.layers) == ["2"]
        stage_layers = {
            name.split(".")[1]
            for name, _ in model.named_parameters()
            if name.startswith("layers.")
        }
        assert stage_layers == {"2"}


class TestRecomputeWiring:
    @pytest.mark.parametrize(
        "counts,pp_rank,expected",
        [
            ((), 0, 5),
            ((2,), 1, 2),
            ((9,), 0, 5),
            ((4, 1), 0, 4),
            ((4, 1), 1, 1),
        ],
        ids=["unset-all", "one-count", "clamped", "per-stage-0", "per-stage-1"],
    )
    def test_block_recompute_layers(self, counts, pp_rank, expected) -> None:
        layers = list(range(5))
        ps = SimpleNamespace(pp_rank=pp_rank, pp_size=2)
        assert protocol._recompute_layers(layers, ps, counts) == layers[:expected]

    def test_block_recompute_counts_must_match_pp(self) -> None:
        with pytest.raises(ValueError, match="pp=2"):
            protocol._recompute_layers([0, 1], SimpleNamespace(pp_rank=0, pp_size=2), (1, 1, 1))

    @pytest.mark.parametrize(
        "value,expected",
        [(None, ()), (3, (3,)), ([4, 2], (4, 2))],
        ids=["default", "int", "list"],
    )
    def test_recompute_num_layers_config(self, value, expected) -> None:
        impl = _impl() if value is None else _impl(recompute_num_layers=value)
        assert impl.recompute_num_layers == expected

    def test_stream_is_a_pytree_node(self) -> None:
        from torch.utils import _pytree

        blocks = (torch.ones(2, 3), torch.zeros(2, 3))
        for delta in (None, torch.full((2, 3), 2.0)):
            stream = AttnResStream(torch.randn(2, 3), delta, blocks)
            leaves, spec = _pytree.tree_flatten(stream)
            tensors = [x for x in leaves if isinstance(x, torch.Tensor)]
            assert len(tensors) == 1 + (delta is not None) + len(blocks)
            back = _pytree.tree_unflatten(leaves, spec)
            assert back.prefix is stream.prefix and back.delta is delta
            assert all(a is b for a, b in zip(back.blocks, blocks, strict=True))

    def test_module_map_picks_the_layer_kind(self) -> None:
        config = replace(tiny_config(), num_hidden_layers=3, kda_layers=[0, 1])
        model = K31Model(config, selected_layer_ids=(0, 1, 2), selected_module_names=("attn", "moe"))
        rmap = protocol._RECOMPUTE_MODULE_MAP
        kinds = [
            (rmap["kda"](layer) is not None, rmap["dsa"](layer) is not None)
            for layer in model.layers.values()
        ]
        assert kinds == [(True, False), (True, False), (False, True)]
        assert all(rmap["moe"](layer) is layer.mlp for layer in model.layers.values())

    def test_layer_entry_sums_per_layer(self) -> None:
        """A block read twice by one layer reaches its producer as one per-layer sum."""
        from megatron.lite.model.k31.vllm.model import _enter_layer

        block = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        prefix = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        seen = []
        stream = _enter_layer(AttnResStream(prefix, None, (block,)))
        stream.blocks[0].register_hook(lambda g: seen.append(g.clone()))
        (stream.blocks[0] * 2.0 + stream.blocks[0] * 3.0 + stream.prefix).sum().backward()
        assert len(seen) == 1 and torch.equal(seen[0], torch.full_like(block, 5.0))
        assert torch.equal(block.grad, torch.full_like(block, 5.0))
        with torch.no_grad():
            same = AttnResStream(prefix, None, (block,))
            assert _enter_layer(same) is same


def test_layer_flags_follow_vllm_decoder_rules() -> None:
    config = replace(tiny_config(), num_hidden_layers=3, attn_res_block_size=2)
    model = K31Model(config, selected_layer_ids=(0, 1, 2), selected_module_names=("attn", "moe"))
    flags = [
        (layer.is_block_write_layer, layer.block_write_idx, layer.prev_valid_blocks)
        for layer in model.layers.values()
    ]
    assert flags == [(True, 0, 0), (False, 0, 1), (True, 1, 1)]


def test_forward_requires_runtime_and_prefix() -> None:
    model = K31Model(tiny_config())
    with pytest.raises(ValueError, match="K31Runtime"):
        model(input_ids=torch.zeros(3, dtype=torch.long))
    with pytest.raises(NotImplementedError, match="prefix"):
        model(input_ids=torch.zeros(3, dtype=torch.long), runtime=SimpleNamespace())


class _LayerRan(Exception):
    pass


@pytest.mark.parametrize(
    "selected,labels,refused",
    [((0, 1), True, True), ((0, 1, 2), True, False), ((0, 1), False, False)],
    ids=["partial-with-labels", "every-layer-with-labels", "prefix-audit"],
)
def test_training_forward_needs_every_stage_layer(monkeypatch, selected, labels, refused) -> None:
    model = K31Model(tiny_config(), selected_layer_ids=selected, selected_module_names=("attn", "moe"))

    def ran(*args, **kwargs):
        raise _LayerRan

    for layer in model.layers.values():
        monkeypatch.setattr(layer, "forward", ran)
    kwargs = {"hidden_states": torch.zeros(3, 16), "runtime": SimpleNamespace()}
    if labels:
        kwargs["labels"] = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError if refused else _LayerRan, match="every layer" if refused else None):
        model(**kwargs)


def test_checkpoint_mtp_layers_do_not_block_training_without_mtp() -> None:
    config = tiny_config(num_nextn_predict_layers=1)
    protocol._validate_contract(config, _impl())
    with pytest.raises(NotImplementedError, match="MTP"):
        protocol._validate_contract(config, _impl(mtp_enable=True))


@pytest.mark.parametrize(
    "hf",
    [
        {"text_config": {"logit_scale": 0.5}},
        {"text_config": {"logit_scale": 2.0}},
        {"logit_scale": 2.0, "text_config": {}},
        {"logit_scale": 2.0, "text_config": {"logit_scale": 1.0}},
    ],
    ids=["text-half", "text-double", "top-level", "top-level-over-text"],
)
def test_checkpoint_logit_scale_is_refused(hf) -> None:
    from megatron.lite.model.k31.config import K31Config

    with pytest.raises(ValueError, match="logit_scale"):
        K31Config._from_hf_dict(hf)


class TestRetiredEnv:
    @pytest.mark.parametrize("name", sorted(runtime_metadata.RETIRED_ENV))
    def test_retired_switch_is_refused_with_its_replacement(self, monkeypatch, name) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        for other in rm.RETIRED_ENV:
            monkeypatch.delenv(other, raising=False)
        rm._check_retired_env()
        monkeypatch.setenv(name, "0")
        with pytest.raises(RuntimeError) as err:
            rm._check_retired_env()
        assert f"{name} -> {rm.RETIRED_ENV[name]}" in str(err.value)

    @pytest.mark.parametrize(
        "name",
        ["K31_TREE_BUILD", "K31_BI_GEMM_BUILD", "K31_KDA_CFBI_BUILD", "K31_BI_GEMM_CUTLASS_INCLUDE"],
    )
    def test_retired_build_switch_is_refused(self, monkeypatch, name) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        for other in rm.RETIRED_ENV:
            monkeypatch.delenv(other, raising=False)
        monkeypatch.setenv(name, "/local/k31tree")
        with pytest.raises(RuntimeError, match=f"{name} -> none: the K31 CUDA modules are built"):
            rm._check_retired_env()

    @pytest.mark.parametrize("value", ["", "  "], ids=["empty", "blank"])
    def test_empty_value_counts_as_unset(self, monkeypatch, value) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        for other in rm.RETIRED_ENV:
            monkeypatch.setenv(other, value)
        rm._check_retired_env()

    def test_only_k31_switches_stay_k31_owned(self) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        assert all(name.startswith("K31_") for name in rm.RETIRED_ENV)

    def test_init_checks_retired_switches_first(self, monkeypatch) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        monkeypatch.setattr(rm, "_INITIALIZED", False)
        monkeypatch.setenv("K31_RECOMPUTE_NUM_LAYERS", "4")
        with pytest.raises(RuntimeError, match="K31_RECOMPUTE_NUM_LAYERS -> "):
            rm.initialize_k31_vllm_batch_invariance()


class TestBatchInvarianceInit:
    def test_cublas_check_reads_startup_values_not_vllm_rewrite(self, monkeypatch) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        monkeypatch.setattr(rm, "_STARTUP_CUBLAS_ENV", dict(rm._CUBLAS_ENV))
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # vLLM override_envs_for_invariance
        rm._check_cublas_env()
        monkeypatch.setattr(rm, "_STARTUP_CUBLAS_ENV", {**rm._CUBLAS_ENV, "CUBLASLT_WORKSPACE_SIZE": None})
        with pytest.raises(RuntimeError, match="before the process starts"):
            rm._check_cublas_env()

    @pytest.mark.parametrize(
        "bi,stream,cublas,cfbi_built,error",
        [
            ("0", None, {}, True, None),
            ("1", "1", "startup", True, None),
            ("1", None, "startup", True, (RuntimeError, "rollout BI environment")),
            ("0", None, {}, False, (ImportError, "_k31_kda_pf4_C")),
            ("0", None, "startup", True, (RuntimeError, "CUBLAS_WORKSPACE_CONFIG=':16:8'")),
            ("0", None, {"CUBLASLT_WORKSPACE_SIZE": "1"}, True, (RuntimeError, "without it")),
        ],
        ids=[
            "bi0-no-process-mode",
            "bi1-process-mode",
            "bi1-needs-rollout-env",
            "bi0-needs-cfbi-engines",
            "bi0-inherits-bi-cublas",
            "bi0-bi-cublas-set-later",
        ],
    )
    def test_init_follows_vllm_batch_invariant(
        self, monkeypatch, bi, stream, cublas, cfbi_built, error
    ) -> None:
        from types import SimpleNamespace

        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        def kernels():
            if not cfbi_built:
                raise ImportError("vllm._k31_kda_pf4_C")
            return ("pf4", "dsync8")

        calls = []
        # enabled() is vLLM's: the rollout's CF-BI routing follows VLLM_BATCH_INVARIANT; the
        # engines themselves (kernels()) are what the trainer's KDA runs in both arms.
        modules = {
            "k31_bi_gemm": SimpleNamespace(VERSION=rm._MIN_BI_GEMM_VERSION, enabled=lambda: True),
            "k31_kda_cfbi": SimpleNamespace(enabled=lambda: bi == "1", kernels=kernels),
        }
        monkeypatch.setattr(rm, "_INITIALIZED", False)
        # "startup": the process started with the BI cuBLAS environment; a dict: set after startup
        startup = dict(rm._CUBLAS_ENV) if cublas == "startup" else dict.fromkeys(rm._CUBLAS_ENV)
        monkeypatch.setattr(rm, "_STARTUP_CUBLAS_ENV", startup)
        for name in rm._CUBLAS_ENV:
            monkeypatch.delenv(name, raising=False)
        for name, value in (startup if cublas == "startup" else cublas).items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(rm, "k31_vllm_package", lambda: None)
        monkeypatch.setattr(rm, "_k31_module", modules.__getitem__)
        monkeypatch.setattr(rm, "_symbol", lambda module, name: lambda **kw: calls.append((name, kw)))
        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
        for name in rm.RETIRED_ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", bi)
        if stream is None:
            monkeypatch.delenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", raising=False)
        else:
            monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", stream)
        if error is not None:
            with pytest.raises(error[0], match=error[1]):
                rm.initialize_k31_vllm_batch_invariance()
            assert rm._INITIALIZED is False
            return
        rm.initialize_k31_vllm_batch_invariance()
        assert rm._INITIALIZED is True
        expected = [("init_batch_invariance", {"pin_nccl": False})] if bi == "1" else []
        assert calls == expected

    def test_adapters_run_without_the_bi_process_mode(self, monkeypatch) -> None:
        from megatron.lite.primitive.kernels import vllm_k31 as vk

        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        monkeypatch.setattr(vk, "_symbol", lambda module, name: lambda x, w, eps: x * w)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        w = torch.full((4,), 2.0, dtype=torch.bfloat16)
        assert torch.equal(vk.RMSNormAdapter()(x, w, 1e-6), x * w)

    def test_second_call_is_a_no_op(self, monkeypatch) -> None:
        from megatron.lite.model.k31.vllm import runtime_metadata as rm

        monkeypatch.setattr(rm, "_INITIALIZED", True)
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        rm.initialize_k31_vllm_batch_invariance()
