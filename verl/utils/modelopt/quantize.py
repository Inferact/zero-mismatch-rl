# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ModelOpt NVFP4 quantization config and application for Megatron QAT."""

import contextlib
import copy
import fnmatch
import inspect

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn import TensorQuantizer

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg.append({"quantizer_name": key, "enable": False})
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode != "w4a16":
        raise ValueError(f"Only 'w4a16' is supported, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = mtq.normalize_quant_cfg_list(_NVFP4_W4A16_QUANTIZER_CFG)
    disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
    if isinstance(disabled_cfg, dict):
        disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
    quant_cfg.extend(disabled_cfg)
    quant_cfg.extend(ignore_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


# Megatron (TE spec) weight quantizers of a Nemotron-H ModelOpt MIXED_PRECISION
# deployment: W4A16 NVFP4 routed/shared experts and lm_head, FP8 Mamba in/out_proj.
_MIXED_PRECISION_WEIGHT_QUANTIZERS = {
    "*experts.linear_fc*weight_quantizer": "W4A16_NVFP4",
    "*output_layer.weight_quantizer": "W4A16_NVFP4",
    "*mixer.in_proj.weight_quantizer": "FP8",
    "*mixer.out_proj.weight_quantizer": "FP8",
}


def _dequantize(algorithm: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    from megatron.lite.model.nemotron_h.quantization import nvfp4_decode_values

    if algorithm == "FP8":
        return tensors["weight"].float() * tensors["weight_scale"].float()
    values = nvfp4_decode_values(tensors["weight"])
    rows, cols = values.shape
    unit = tensors["weight_scale"].float() * tensors["weight_scale_2"].float().reshape(())
    return (values.reshape(rows, cols // 16, 16) * unit[..., None]).reshape(rows, cols)


class RequantSTEQuantizer(TensorQuantizer):
    """Weight fake-quantizer whose forward value is the deployed weight.

    The forward pass sees dequant(requantize(w)), with requantize the rule the
    exporter uses at every sync (so trained and served weights agree); the
    gradient passes straight through to the BF16 master.
    """

    def __init__(self, algorithm: str):
        super().__init__()
        self.algorithm = algorithm

    def forward(self, inputs):
        from megatron.lite.model.nemotron_h.quantization import requantize

        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise RuntimeError(f"RequantSTEQuantizer expects a 2-D weight, got {type(inputs)}")
        with torch.no_grad():
            deployed = _dequantize(self.algorithm, requantize(self.algorithm, inputs.detach().to(torch.bfloat16)))
        return inputs + (deployed.to(inputs.dtype) - inputs).detach()


def _te_grouped_weight_only_fn(package, func_name, self, *args):
    """Weight-only quantized call of TE's ``_GroupedLinear`` functional.

    ModelOpt 0.44 reads the GEMM count from ``non_tensor_args[0]``, which is
    ``apply_bias`` in TE 2.17; the weights are located by parameter name instead.
    """
    import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear

    # Under ModelOpt's replace_function the original forward is cached at ``_forward``;
    # the ``_forward`` call carries the placeholder ctx, ``_apply`` does not.
    fn = te_grouped_linear._GroupedLinear
    params = list(inspect.signature(getattr(fn, "_forward", fn.forward)).parameters)
    weights_start = params.index("weights_and_biases") - (0 if func_name == "_forward" else 1)
    new_args = list(args)
    for i in range(self.num_gemms):
        weight = args[weights_start + i]
        if not isinstance(weight, torch.Tensor) or weight.shape != getattr(self, f"weight{i}").shape:
            raise RuntimeError(f"TE _GroupedLinear argument {weights_start + i} is not weight{i}")
        new_args[weights_start + i] = self.weight_quantizer(weight)
    return getattr(package, func_name)(*new_args)


@contextlib.contextmanager
def _weight_only_grouped_linear_quant_modules():
    """Quantize Megatron's TE grouped linears with verl's weight-only subclasses.

    Registers verl subclasses of ModelOpt's quant modules in ModelOpt's
    QuantModuleRegistry for the duration of one ``mtq.quantize`` and restores
    ModelOpt's own registrations afterwards.
    """
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
    from modelopt.torch.quantization.plugins.megatron import (
        _MegatronTEGroupedColumnParallelLinear,
        _MegatronTEGroupedRowParallelLinear,
    )

    class _WeightOnlyColumnGroupedLinear(_MegatronTEGroupedColumnParallelLinear):
        _quantized_linear_fn = staticmethod(_te_grouped_weight_only_fn)

    class _WeightOnlyRowGroupedLinear(_MegatronTEGroupedRowParallelLinear):
        _quantized_linear_fn = staticmethod(_te_grouped_weight_only_fn)

    registry = mtq.QuantModuleRegistry
    swaps = [
        (TEColumnParallelGroupedLinear, _MegatronTEGroupedColumnParallelLinear, _WeightOnlyColumnGroupedLinear),
        (TERowParallelGroupedLinear, _MegatronTEGroupedRowParallelLinear, _WeightOnlyRowGroupedLinear),
    ]
    keys = [registry.get_key(nn_cls) for nn_cls, _, _ in swaps]
    for (nn_cls, _, verl_cls), key in zip(swaps, keys, strict=True):
        registry.unregister(nn_cls)
        registry.register({nn_cls: key})(verl_cls)
    try:
        yield tuple(verl_cls for _, _, verl_cls in swaps)
    finally:
        for (nn_cls, modelopt_cls, _), key in zip(swaps, keys, strict=True):
            registry.unregister(nn_cls)
            registry.register({nn_cls: key})(modelopt_cls)


def apply_mixed_precision_qat(model: nn.Module) -> nn.Module:
    """Insert deployment-matching weight fake-quant on the quantized deployment layers."""
    from modelopt.torch.quantization.plugins.transformer_engine import _QuantTEGroupedLinear

    # Every enabled weight quantizer is replaced by RequantSTEQuantizer below; the
    # cfg only makes mtq.quantize create it.
    quant_cfg = [{"quantizer_name": "*", "enable": False}]
    for name in _MIXED_PRECISION_WEIGHT_QUANTIZERS:
        quant_cfg.append({"quantizer_name": name, "cfg": {"num_bits": (4, 3), "axis": None}, "enable": True})
    with _weight_only_grouped_linear_quant_modules() as weight_only_grouped_classes:
        mtq.quantize(model, {"quant_cfg": quant_cfg, "algorithm": None})
    replaced = 0
    for module_name, module in model.named_modules():
        quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
            continue
        if module_name.startswith("mtp.") or ".mtp." in module_name:
            # The checkpoint's MTP block is BF16; the drafter serves it unquantized.
            quantizer.disable()
            continue
        if isinstance(module, _QuantTEGroupedLinear) and not isinstance(module, weight_only_grouped_classes):
            raise RuntimeError(f"{module_name}: {type(module).__name__} has no weight-only grouped quant module")
        name = f"{module_name}.weight_quantizer"
        algorithm = next(a for p, a in _MIXED_PRECISION_WEIGHT_QUANTIZERS.items() if fnmatch.fnmatch(name, p))
        module.weight_quantizer = RequantSTEQuantizer(algorithm).to(next(module.parameters()).device)
        replaced += 1
    if replaced == 0:
        raise RuntimeError("mixed-precision QAT matched no weight quantizer")
    print(f"[QAT modelopt_mixed] deployment-matching weight fake-quant on {replaced} modules")
    return model


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    if qat_mode == "modelopt_mixed":
        return apply_mixed_precision_qat(model)
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
