# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU tests for the weight-only TE grouped-linear QAT path in
``verl.utils.modelopt.quantize``: the QuantModuleRegistry swap must not leak
past one ``mtq.quantize`` call, and the weight locator must refuse arguments
that are not the module's weights."""

import inspect
import types

import pytest
import torch

pytest.importorskip("modelopt.torch.quantization")
pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("megatron.core.extensions.transformer_engine")

import modelopt.torch.quantization as mtq  # noqa: E402
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear  # noqa: E402
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)

from verl.utils.modelopt import quantize  # noqa: E402

_GROUPED = (TEColumnParallelGroupedLinear, TERowParallelGroupedLinear)


def _registered():
    """(quant base class, key) that ModelOpt would use for each grouped linear."""
    registry = mtq.QuantModuleRegistry
    return [(registry[cls].__bases__[0], registry.get_key(cls)) for cls in _GROUPED]


def test_registry_swap_is_scoped_to_the_block():
    before = _registered()
    with quantize._weight_only_grouped_linear_quant_modules() as verl_classes:
        inside = _registered()
        assert [cls for cls, _ in inside] == list(verl_classes)
        assert [key for _, key in inside] == [key for _, key in before]
        for (verl_cls, _), (modelopt_cls, _) in zip(inside, before, strict=True):
            assert issubclass(verl_cls, modelopt_cls)
    assert _registered() == before


def test_registry_swap_is_restored_on_exception():
    before = _registered()
    with pytest.raises(ValueError, match="boom"):
        with quantize._weight_only_grouped_linear_quant_modules():
            raise ValueError("boom")
    assert _registered() == before


def _grouped():
    return types.SimpleNamespace(
        num_gemms=2,
        weight_quantizer=lambda w: w + 1,
        weight0=torch.zeros(4, 8),
        weight1=torch.zeros(4, 8),
    )


def _fixed_args():
    """Placeholders for the ``_GroupedLinear.forward`` arguments between ``ctx`` and
    ``weights_and_biases``; their number depends on the installed TE version."""
    fn = te_grouped_linear._GroupedLinear
    params = list(inspect.signature(getattr(fn, "_forward", fn.forward)).parameters)
    return tuple(object() for _ in range(params.index("weights_and_biases") - 1))


@pytest.mark.parametrize("func_name, prefix", [("_apply", ()), ("_forward", ("ctx",))])
def test_weight_locator_quantizes_only_the_weights(func_name, prefix):
    module = _grouped()
    weights = [torch.zeros(4, 8) for _ in range(2)]
    biases = [torch.zeros(4) for _ in range(2)]
    package = types.SimpleNamespace(**{func_name: lambda *args: args})
    args = (*prefix, *_fixed_args(), *weights, *biases)

    out = quantize._te_grouped_weight_only_fn(package, func_name, module, *args)

    start = len(args) - 4
    assert all(o is a for o, a in zip(out[:start], args[:start], strict=True))
    for i in range(2):
        assert torch.equal(out[start + i], weights[i] + 1)
        assert out[start + 2 + i] is biases[i]


def test_weight_locator_rejects_a_shape_mismatch():
    module = _grouped()
    package = types.SimpleNamespace(_apply=lambda *args: args)
    args = (*_fixed_args(), torch.zeros(8, 4), torch.zeros(4, 8))
    with pytest.raises(RuntimeError, match="is not weight0"):
        quantize._te_grouped_weight_only_fn(package, "_apply", module, *args)
