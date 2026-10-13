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
"""Quantize exported HF weights into a ModelOpt MIXED_PRECISION checkpoint's format.

The rollout serves a ModelOpt mixed-precision checkpoint (``hf_quant_config.json``
with ``quant_algo: MIXED_PRECISION``) while the trainer holds BF16 weights. Every
weight sync re-quantizes the current weights with the rule that produced the
Nemotron-H Lightning checkpoint from its BF16 release
(``megatron.lite.model.nemotron_h.quantization.requantize``: Transformer Engine
NVFP4 4over6/MSE with the 256 E4M3 bound and global = amax / 1536; FP8 per-tensor
amax / 448), so the theta0 deployment equals the true-on-policy recipe's.

The static activation ``input_scale`` and attention ``k_scale``/``v_scale`` are
calibration data and are forwarded from the checkpoint, so every layer receives
a complete checkpoint-format set.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import torch

_STATIC_SUFFIXES = ("input_scale", "k_scale", "v_scale")


def load_mixed_precision_config(model_path: str) -> dict | None:
    path = os.path.join(model_path, "hf_quant_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        quant = json.load(f)["quantization"]
    if quant.get("quant_algo") != "MIXED_PRECISION":
        return None
    return quant


class ModelOptMixedPrecisionExporter:
    def __init__(self, model_path: str, quant: dict):
        from safetensors import safe_open

        self.layers = {name: cfg["quant_algo"] for name, cfg in quant["quantized_layers"].items()}
        unknown = set(self.layers.values()) - {"FP8", "W4A16_NVFP4"}
        if unknown:
            raise NotImplementedError(f"unsupported ModelOpt quant_algo(s): {sorted(unknown)}")
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        # module name -> {suffix: tensor} for the static scales kept from the checkpoint
        self.static: dict[str, dict[str, torch.Tensor]] = {}
        by_file: dict[str, list[str]] = {}
        for key, file in weight_map.items():
            if key.rpartition(".")[2] in _STATIC_SUFFIXES and not key.startswith("mtp."):
                by_file.setdefault(file, []).append(key)
        for file, keys in by_file.items():
            with safe_open(os.path.join(model_path, file), framework="pt") as f:
                for key in keys:
                    module, _, suffix = key.rpartition(".")
                    self.static.setdefault(module, {})[suffix] = f.get_tensor(key)

    @staticmethod
    def _requantize(algo: str, name: str, weight: torch.Tensor):
        from megatron.lite.model.nemotron_h.quantization import requantize

        for suffix, value in requantize(algo, weight.to(torch.bfloat16)).items():
            yield name.removesuffix("weight") + suffix, value

    def __call__(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in named_tensors:
            module, _, leaf = name.rpartition(".")
            algo = self.layers.get(module)
            if leaf == "weight" and algo is not None:
                yield from self._requantize(algo, name, tensor)
            else:
                yield name, tensor
            if leaf == "weight":
                for suffix, static in self.static.get(module, {}).items():
                    yield f"{module}.{suffix}", static.to(tensor.device)
