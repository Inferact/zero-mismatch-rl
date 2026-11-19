# B300: Q4/Q6 FP32 state, FlashInfer and vLLM CF-BI

[中文](README.md) · [Performance overview](../../README.md)

B=64, H=12/16/32, D=128; Q4/Q6 correspond to MTP3/MTP5. All paths directly index a
state pool and include all Q checkpoints, device acceptance selection and in-place
state commit. Our state/checkpoints are **FP32**, FlashInfer's are **BF16**, and
CF-BI's entry matrix and row cache are FP32. Q/K/V/gate/beta inputs and outputs
are BF16. Different implementations are not required to match bitwise.

## Environment and sources

B300 SXM6 AC, two GPUs with NV18; measurements run serially on one GPU. Driver
615.71.10, Python 3.12.3, PyTorch 2.14.0a0+b2c75dd062.nv26.09, CUDA 13.4,
CuTe DSL 4.8.1. Default clocks without locking. [Environment](environment.json).

| Implementation | Source |
|---|---|
| FP32 state | Configurable-Q candidate implementation, based on [e652140](https://github.com/NVIDIA-JerryChen/kda-consistent/commit/e6521408691ca3b95e47a138670966f4192ab2ef); measured file SHA256 values in [sources.json](sources.json) |
| FlashInfer | [flashinfer-ai/flashinfer @ c3af0c1](https://github.com/flashinfer-ai/flashinfer/tree/c3af0c13328fd23b256c53b9b4f94d9e160d118c), native grouped Q4/Q6 |
| CF-BI C16/C64 | [Inferact/zero-mismatch-rl @ 5735778](https://github.com/Inferact/zero-mismatch-rl/tree/5735778f43f7d94108d432706dc5b17765bf723a), branch `vllm/kimi-k3.1`, C16=v6 and C64=v5 |

CF-BI uses the original [public decode wrapper](https://github.com/Inferact/zero-mismatch-rl/blob/5735778f43f7d94108d432706dc5b17765bf723a/vllm/models/kimi_k3/nvidia/k31_kda_cfbi.py)
and unchanged CUDA implementations, built with `-O3 --fmad=false -UNDEBUG -lineinfo`
for `sm_103a`. FlashInfer retains its original kernels and compiler functions;
unrelated imports/dispatch were removed. Physical commit uses the same
dtype-preserving copy helper as our implementation. See [source hashes](sources.json)
and [upstream provenance](upstream-provenance.json). FlashKDA PR #5, framework,
convolution and communication costs are outside this experiment.

## Single-call performance

42 cases: Q4/Q6 × H12/16/32 × seven KV length lists. Each row is the geometric
mean of seven per-case median latencies, in µs. Acceptance counts are
`request_index % (Q + 1)`, covering 0 through Q.

| Q | H | FP32 state | FlashInfer | CF-BI C16 adapter | CF-BI C64 adapter | FP32 / FlashInfer |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 61.406 | 51.241 | 511.581 | 772.122 | 1.198× |
| 4 | 16 | 77.842 | 71.873 | 588.161 | 978.350 | 1.083× |
| 4 | 32 | 143.419 | 135.437 | 987.657 | 1803.975 | 1.059× |
| 6 | 12 | 77.833 | 75.781 | 712.728 | 1025.625 | 1.027× |
| 6 | 16 | 100.361 | 98.020 | 814.433 | 1299.032 | 1.024× |
| 6 | 32 | 188.120 | 185.391 | 1360.947 | 2385.270 | 1.015× |

Across H, geometric means are **88.175 / 79.306 / 667.327 / 1108.673 µs** for Q4
and **113.690 / 111.255 / 924.429 / 1470.219 µs** for Q6, in table order.
Our FP32-state latency is **11.2% / 2.2% higher** than FlashInfer.
[Per-case CSV](main.csv) · [All raw samples](main.json) · [Summary](summary.json).

**This CF-BI version has no native multi-token checkpoint API.** The adapter makes
Q public Q1 calls, snapshots the full entry matrix, row cache and position after
each token, saves the original state for zero acceptance, then selects and
physically writes back the accepted state. It copies the entire cache capacity,
including unused capacity, and times wrapper layout conversions. These numbers
measure the **complete functional adapter, not native CF-BI Q4/Q6 kernel latency**;
they do not establish the same speedup for the kernels alone.

## Chunk positions and complete cycles

384 cases cover Q4/Q6, three H values and all starting positions 0–63, including
chunk boundaries. [Position CSV](positions.csv) / [raw samples](positions.json).

Complete cycles accept all Q tokens: Q4 runs 16 calls over 64 tokens; Q6 runs
32 calls over 192 tokens, covering the full reachable period of the 64-row
contract. Below is the median cycle time divided by calls, in µs/call.
Every call still emits all checkpoints and commits state.

| Q | H | FP32 state | FlashInfer | CF-BI C16 adapter | CF-BI C64 adapter |
|---|---:|---:|---:|---:|---:|
| 4 | 12 | 57.788 | 42.625 | 441.499 | 749.472 |
| 4 | 16 | 76.854 | 64.989 | 512.826 | 963.301 |
| 4 | 32 | 147.260 | 132.801 | 930.783 | 1783.552 |
| 6 | 12 | 74.526 | 65.377 | 610.192 | 997.681 |
| 6 | 16 | 98.784 | 89.293 | 704.048 | 1281.007 |
| 6 | 32 | 190.432 | 182.782 | 1273.071 | 2360.430 |

Across H, geometric mean FP32/FlashInfer ratios are **1.211× for Q4** and
**1.095× for Q6**. Acceptance counts, calls per Graph and state progression differ
from the single-call table, so results remain separate. Divide by Q for µs/token.
[Cycle CSV](cycles.csv) · [Raw samples](cycles.json).

## Numerical accuracy and consistency

Independent FP64 recurrence; B=2, H=3, D=128, normal/open/saturated gates and
nonzero BF16-representable initial state. Each sequence has 64 tokens: execute
64−Q history tokens one at a time, then Q4/Q6. Compare the last Q outputs and
final state. CF-BI ends at a closed chunk, where its entry matrix equals the
logical recurrent state. Six cases per implementation; the table lists observed
maxima, not error bounds. NRMSE=RMS(error)/RMS(reference); maximum normalized
error=max(abs(error))/max(abs(reference)).

| Implementation | Output NRMSE | Output max normalized error | State NRMSE | State max normalized error |
|---|---:|---:|---:|---:|
| FP32 state | 1.688e-3 | 3.436e-3 | 2.033e-7 | 4.899e-7 |
| FlashInfer | 1.229e-2 | 1.695e-2 | 1.228e-2 | 3.049e-2 |
| CF-BI C16 | 1.688e-3 | 3.436e-3 | 3.725e-7 | 8.054e-7 |
| CF-BI C64 | 1.688e-3 | 3.436e-3 | 4.550e-7 | 8.014e-7 |

FlashInfer rounds each historical Q1 state to BF16. Its errors include that
history rounding and **are not the intrinsic error of a single Q4/Q6 call**.
All [24 diagnostic records](accuracy.json) are finite.

Checks cover eager/Graph outputs, complete state pools and all checkpoints.
FP32-state and CF-BI checkpoints match their own independent Q1 prefixes
bitwise. FlashInfer is compared against its compact-indexed grouped path;
no BF16 Q1/multi-token bitwise consistency is claimed for FlashInfer.
Device slots and acceptance counts change under a fixed Graph; unselected slots
remain unchanged and zero acceptance preserves initial state. Production source
matches the earlier 586-check configurable-Q validation. This is not full vLLM
integration or a new multi-GPU acceptance run. [Validation](validation.json).

## Workloads and timing boundary

- [workloads.json](workloads.json) retains the seven confirmed B64 Q8 KV length
  arrays without resampling, with limits 8K/16K/32K/64K/128K/256K/512K. Q changes
  to 4/6 and entry positions become `KV length − Q`.
- History is represented by state: supply a shared nonzero anchor at a closed
  64-token boundary and actually execute the remaining `context % 64` history
  tokens for each implementation. There is no full 8K–512K history replay or
  fabricated cache created by changing positions alone.
- Pool capacity 129, 64 unique reversed odd slots, in-place writeback. Inputs
  are synthetic; there is no per-case kernel specialization.
- Hot-address CUDA Graphs with event nodes around one complete prepared GPU API.
  Reset is outside timing and warms state. Rotate implementation order, queue
  eight executions per implementation per group, then synchronize/read events.
  Single-call results discard 16 warmups and retain 80 samples, reporting medians.
- Position tests discard four warmups and retain 12 samples. Cycle tests discard
  two warmup cycles and retain ten cycles.
- Include decode, all checkpoints, selection, physical commit and CF-BI adapter
  GPU work. Exclude compilation, allocation, capture, history preparation,
  acceptance decisions, metadata generation, CPU scheduling and communication.
  This is neither cold-L2 nor end-to-end model performance.

## Reproduce

Requires the CUDA 13.x compiler toolchain and two visible B300 SXM6 AC GPUs;
only GPU0 is timed. From a checkout with CUDA-enabled PyTorch installed,
`prepare.py` downloads and hash-verifies CF-BI sources at the pinned commit.
`fi_q1.py` / `fi_q8.py` retain FlashInfer's
[Apache-2.0](../../../../vllm/models/kimi_k3/nvidia/kda_fp32_state/licenses/FlashInfer-Apache-2.0.txt) notices.

```bash
python -m pip install -r vllm/models/kimi_k3/nvidia/kda_fp32_state/requirements.txt pytest triton ninja
export KDA_BENCH_ROOT="$PWD/q46-reproduction"
python benchmarks/kda_fp32_state/data/q4-q6-b300-20261119/prepare.py "$KDA_BENCH_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$KDA_BENCH_ROOT/source"
export RESULT_DIR="$KDA_BENCH_ROOT/results"
export TORCH_EXTENSIONS_DIR="$KDA_BENCH_ROOT/cache/torch"
export TRITON_CACHE_DIR="$KDA_BENCH_ROOT/cache/triton"
export CCACHE_DIR="$KDA_BENCH_ROOT/cache/ccache"
export SOURCE_DIGEST="$(cat "$KDA_BENCH_ROOT/source-digest.txt")"
python "$KDA_BENCH_ROOT/source/verify.py"
python "$KDA_BENCH_ROOT/source/benchmark.py" smoke
python "$KDA_BENCH_ROOT/source/benchmark.py" main
python "$KDA_BENCH_ROOT/source/benchmark.py" phases
python "$KDA_BENCH_ROOT/source/verify.py"
```

Preparation copies this branch's FP32 module into an isolated snapshot named
`kda_consistent`, as imported by the frozen runner. This table was measured
before adding the module to the vLLM branch; original source hashes remain in
[sources.json](sources.json). See the [overview](../../README.md) for kernel file
mapping and separate validation of this branch. Module paths and API wrappers
differ, so the overall reproduction digest differs.
