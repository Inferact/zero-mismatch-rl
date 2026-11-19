"""Independent FP64 diagnostics; finite errors are reported, not new thresholds."""

import json
import os
from pathlib import Path

import pytest
import torch

from .fp64_reference import run as reference
from .helpers import case, prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("length", [8, 64, 4096, 32768])
@pytest.mark.parametrize("gate", ["normal", "open", "saturated"])
def test_fp64_diagnostic(length, gate):
    x = case([length], h=3)
    if gate != "normal":
        x["A_log"].zero_()
        x["dt_bias"].zero_()
        x["g"].fill_(-1000)
        if gate == "saturated":
            x["g"][..., ::2] = 1000
            x["beta"][:, ::2] = 1000
            x["beta"][:, 1::2] = -1000
    out, state, _ = prefill(x)
    expected = reference(
        *[x[k] for k in ("q", "k", "v", "g", "beta")],
        x["A_log"],
        x["dt_bias"],
        x["state_in"],
        x["cu_seqlens"],
        full_precision=True,
    )
    record = dict(length=length, gate=gate)
    for name, actual, wanted in [
        ("output", out, expected["output"]),
        ("state", state, expected["state_out"]),
    ]:
        assert torch.isfinite(actual).all() and torch.isfinite(wanted).all()
        delta = actual.double() - wanted
        record[name] = dict(
            nrmse=float(
                delta.square().mean().sqrt()
                / wanted.square().mean().sqrt().clamp_min(1e-300)
            ),
            max_normalized=float(
                delta.abs().max() / wanted.abs().max().clamp_min(1e-300)
            ),
        )
    if destination := os.environ.get("KDA_ACCURACY_JSONL"):
        with Path(destination).open("a") as output:
            output.write(json.dumps(record) + "\n")
