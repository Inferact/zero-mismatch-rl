from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import torch

from megatron.lite.model.k31.vllm import protocol
from megatron.lite.model.k31.vllm.model import K31Model
from megatron.lite.primitive.kernels import vllm_k31

from _tiny import tiny_config, tiny_lora


def _sources() -> list[Path]:
    root = Path(inspect.getfile(protocol)).parent
    return sorted(root.glob("*.py")) + sorted((root / "primitive").glob("*.py")) + [
        Path(inspect.getfile(vllm_k31))
    ]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]


def test_k31_package_has_no_lite_impl_and_no_sibling_model_imports() -> None:
    forbidden = ("megatron.lite.model.k31.lite", "megatron.lite.model.deepseek_v4")
    for path in _sources():
        for name in _imports(path):
            assert not name.startswith(forbidden), (path, name)
    assert not (Path(inspect.getfile(protocol)).parents[1] / "lite").exists()


def test_vllm_model_family_is_discovered_not_spelled() -> None:
    """vLLM's K31 package path is resolved at runtime (k31_vllm_package)."""

    pattern = re.compile(r"vllm\.models\.[a-z0-9_]+\.nvidia")
    for path in _sources():
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_model_is_native_composition_not_whole_model_wrapper() -> None:
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden_bases = {"MegatronModel"} | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm")
            for alias in node.names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {
                    base.id if isinstance(base, ast.Name) else base.attr
                    for base in node.bases
                    if isinstance(base, (ast.Name, ast.Attribute))
                }
                assert not bases & forbidden_bases, (path, node.name, bases)
    assert K31Model.__module__.endswith(".k31.vllm.model")


def test_tiny_model_state_preserves_release_dtypes() -> None:
    model = K31Model(tiny_config(), lora=tiny_lora())
    state = model.state_dict()
    fp32 = {n for n, t in state.items() if t.dtype == torch.float32}
    assert fp32 == {
        n
        for n in state
        if n.endswith(
            (
                ".self_attn.A_log",
                ".self_attn.dt_bias",
                ".self_attn.conv1d.weight",
                ".self_attn.o_norm.weight",
                ".mlp.gate.e_score_correction_bias",
            )
        )
    }
    uint8 = {n for n, t in state.items() if t.dtype == torch.uint8}
    assert uint8 and all(".mlp.experts.w" in n for n in uint8)
    others = {t.dtype for n, t in state.items() if n not in fp32 | uint8}
    assert others == {torch.bfloat16}
    assert sum(t.numel() for t in state.values()) < 200_000


def test_static_suite_never_constructs_release_dimensions() -> None:
    config = tiny_config()
    assert config.hidden_size <= 16 and config.vocab_size <= 32
    assert config.num_hidden_layers == 3 and config.num_experts <= 4
