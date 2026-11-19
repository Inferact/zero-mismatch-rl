# Earlier B300 Q1/Q8 and forward measurements

These measurements predate the configurable-Q contribution. Exact standalone
source hashes are in [package-sources.json](package-sources.json); experiment
hashes, hardware/software and original summaries are in [summary.json](summary.json).
Raw summaries/accuracy retain the original BF16 reference records for provenance;
[cases.csv](cases.csv) retains the FP32, CF-BI and external-reference timing rows.
Only FP32-state kernels are contributed in this PR.

B300 SXM6 AC, two allocated GPUs connected by NV18, serial timing on one GPU.
Driver 615.71.10, Python 3.12.3, PyTorch 2.12.0+cu132, CUDA 13.2, CuTe DSL 4.8.1.

References:

- [FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/commit/c3af0c13328fd23b256c53b9b4f94d9e160d118c): Q1 and native Q8.
- [FlashKDA d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72](https://github.com/MoonshotAI/FlashKDA/commit/d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72), [CUTLASS 5c149f52a436782210263fb2f19b354443a61c6a](https://github.com/NVIDIA/cutlass/commit/5c149f52a436782210263fb2f19b354443a61c6a): packed forward.
- [vLLM CF-BI 5735778f43f7d94108d432706dc5b17765bf723a](https://github.com/Inferact/zero-mismatch-rl/commit/5735778f43f7d94108d432706dc5b17765bf723a): C16/v6 and C64/v5, unchanged CUDA for sm_103a.

231 cases: Q1 21, Q8 21, chunk prefill 84, prefill 9, 192K forward 96.
H=12/16/32, D=128. Decode B=64, KV limits 8K/16K/32K/64K/128K/256K/512K.
Chunk prefill B=8/16; prefill B=16 with maxima 8K/16K/32K. The 192K workload
uses selected CP16 rank fragments, 12288 query tokens per case; it does not
represent 16-card framework execution. Exact lengths and seeds: [cases.csv](cases.csv),
[inputs.json](inputs.json). All are forward workloads, without backward.

Q/K/V/g/beta/output are BF16; A_log/dt_bias are FP32. FP32 recurrent state,
CF-BI entry matrix/cache and BF16 external-reference states follow their own
contracts. Common history inputs prepare complete valid entry states outside timing.
The main comparison uses compact state buffers, not the indexed pools of the later Q4/Q6 run.

One complete call per fixed-address Graph, three replays per sample, rotated
implementation order, three warmup rounds and nine retained samples. Time includes
all prepared GPU API work; compilation/allocation/history preparation/capture/host
overhead are excluded. Q8 counts are request_index modulo 9. FlashInfer uses native
Q8 and a dtype-preserving commit helper. CF-BI uses eight public Q1 calls, full
E/cache/position snapshots and device selection/writeback. This is an adapter,
not a native multi-token CF-BI kernel. Acceptance decision generation is excluded.

[Phase data](phase-cases.csv) cover 64 starting positions, Q1/Q8 and three H values.
The summary retains complete-cycle results. [FP64 accuracy](accuracy.csv) and
[per-gate maxima](accuracy-by-gate.csv) cover H=3, lengths 1/8/64/256/4096/32768,
normal/open/saturated gates and nonzero BF16-representable initial state.
CF-BI state comparison covers closed chunks only; FlashInfer covers Q1/Q8 only.
Maxima across different case coverage are not like-for-like rankings or error bounds.

The [performance overview](../../README.md) reports the original paired ratios.
Do not cross-multiply ratios with later experiments or interpret these kernel data
as full vLLM end-to-end performance. Exact sources and raw samples are retained;
this archived dataset does not include the old complete external runner. The Q4/Q6
[data page](../q4-q6-b300-20261119/README.en.md) includes a standalone reproducible runner.
