# FP32-state KDA tests

Run from this branch's root with CUDA-enabled PyTorch, the kernel's
[requirements](../../../vllm/models/kimi_k3/nvidia/kda_fp32_state/requirements.txt),
pytest and Triton installed:

```bash
python -m pytest -q tests/kernels/kda_fp32_state
KDA_ACCURACY_JSONL=accuracy.jsonl python -m pytest -q tests/kernels/kda_fp32_state/test_numerics.py
```

The suite checks effective BF16 outputs, complete FP32 state and every valid
checkpoint bytewise; NaN/Inf fails. Coverage includes prefill versus Q1 and
configurable Q (both terminal-only and per-token checkpoint modes), arbitrary splits/PD, batch subsets/reordering, varlen and empty
requests, 4K/32K sequences, nonzero FP32 initial state, extreme gates, indexed
pools, in-place/out-of-place commit with all acceptance counts, eager/Graph
metadata updates and compile-cache reuse. Two matching GPUs are needed for
complete-head sharding and cross-device PD transfer; these tests skip otherwise.

`test_deploy_shape.py` runs the K3.1 per-rank head count of 96, which the other modules do
not reach: it calls their shape-agnostic cases with `h=96` and restates the cases that
hard-code a head count or a workspace too small for H=96 (prefill windows from 24 MiB to the
default, mixed and empty lengths, pooled prefill, prefill Graph replay with dynamic metadata,
Q1/Q4/Q6/Q8 decode, acceptance commit with Graph replay). `KDA_LONG_CASES=1` adds the 20K- and
196K-token shapes. Tests skip unless the current GPU is SM100/SM103 (`conftest.py`).

`test_numerics.py` uses an independent FP64 recurrence. Optional JSONL output
appends per-case output/state NRMSE and maximum normalized errors; these are
diagnostics rather than unagreed error thresholds. `fp64_reference.py` has no
production-kernel dependencies. The suite covers the kernel API, not model
convolution, scheduler integration, distributed framework execution or logits.
