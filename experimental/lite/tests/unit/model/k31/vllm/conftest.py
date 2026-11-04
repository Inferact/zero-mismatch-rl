from __future__ import annotations

import importlib.util

import pytest
import torch


@pytest.fixture(scope="session")
def k31_bi():
    """BI mode of the rollout, in this process (K31 image + overlay chain only)."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires an sm_10x GPU")
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("requires the K31 vLLM image")
    from megatron.lite.model.k31.vllm.runtime_metadata import (
        initialize_k31_vllm_batch_invariance,
    )

    initialize_k31_vllm_batch_invariance()
    from megatron.lite.primitive.kernels.vllm_k31 import _k31_module

    return _k31_module("k31_bi_gemm")
