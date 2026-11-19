# B300 native multi-query decode

B300 SXM6 AC, CuTe DSL 4.8.1. All implementations produce every query token's
BF16 output and retain their complete final continuation state. No per-token
checkpoints or acceptance selection are included. FP32 recurrent state and
CF-BI entry/cache state are FP32; the Q1 FlashInfer reference stores BF16 state.

The multi-query comparison uses the public CF-BI row engines in one launch:

| Implementation | Q1 | Q4/Q6/Q8 |
|---|---|---|
| FP32 state | `prepare_decode(..., state_out=state)` | same API, `checkpoint_states=None` |
| CF-BI C16 | runtime `decode` → `rows_slots` | `rows_fast` → `rows_v6_kernel` |
| CF-BI C64 | runtime `decode` → `rows_fast_slots` | `rows_fast` → `dsync7_kernel` at B64/H12/16/32 |
| FlashInfer | native single-token path | not included in this final-state-only comparison |

The C64 slots entry can select a Q1-only `dsync9_kernel`; passing multiple tokens
there is invalid. This benchmark instead uses its general `rows_fast` API with
compact per-request state, as also used by the other implementations. No
per-request gather/scatter or benchmark-created checkpoint copies are timed.
Each implementation retains its own arithmetic contract; cross-implementation
bitwise equality is not required.

## Results

Geometric mean paired latency ratios; lower is better. Main matrix:

| Q | FP32 state = 1× | CF-BI C16 | CF-BI C64 | FlashInfer |
|---|---:|---:|---:|---:|
| 1 | 1.00× | 3.23× | 3.88× | 0.91× |
| 4 | 1.00× | 4.41× | 7.88× | — |
| 6 | 1.00× | 4.85× | 8.79× | — |
| 8 | 1.00× | 5.05× | 9.30× | — |

Complete-cycle mean, per call:

| Q | FP32 state = 1× | CF-BI C16 | CF-BI C64 | FlashInfer |
|---|---:|---:|---:|---:|
| 1 | 1.00× | 2.79× | 4.17× | 0.77× |
| 4 | 1.00× | 4.79× | 9.24× | — |
| 6 | 1.00× | 5.35× | 10.19× | — |
| 8 | 1.00× | 5.52× | 10.41× | — |

Raw [main](main.json), [positions](positions.json), [cycles](cycles.json),
[summary](summary.json), [environment](environment.json) and [source hashes](measured-sources.json).

## Workloads and timing

B=64, H=12/16/32, D=128, Q=1/4/6/8. The seven confirmed KV-length lists span
8K–512K: [exact workloads](workloads.json). Context is KV length minus Q, without
resampling. The common nonzero BF16-representable FP32 anchor is at a closed
64-token boundary; the remaining `context % 64` history tokens are actually
executed under each implementation's contract before timing. These are state
continuation workloads, not a replay of the entire nominal KV history.

- Main matrix: 84 paired cases, one complete call per CUDA Graph, 16 warmups and
  80 event samples per implementation. Reset outside timing, rotated order,
  eight queued Graph/event instances, fixed hot addresses. Report median per case.
- Position matrix: all 64 positions for each Q/H (768 cases), four warmups and
  12 retained samples per implementation.
- Complete cycles: 12 Q/H cases, `64 / gcd(Q,64)` calls per Graph, two warmups
  and ten samples; latency normalized per call. Calls advance the full Q tokens.

Compilation, allocations during preparation/capture, history construction and
state reset are outside the GPU timing interval. The public CF-BI API allocates
its output during capture; the captured kernel writes it directly on replay.
There is no extra output-copy kernel. No convolution, model routing, framework
communication or host dispatch cost is included.

Correctness checks compare each implementation's one-call multi-Q execution
against its own Q1 chain, then eager against Graph, including the entry matrix,
all valid cache rows and position. NaN/Inf in valid data fails. Our terminal-only
mode is also tested against prefill and every-token checkpoint execution.

## Reproduce

From the checkout root, with the kernel requirements, Triton, pytest, Ninja and
a CUDA development toolchain available:

```bash
python benchmarks/kda_fp32_state/data/native-multiq-b300-20261119/prepare.py /tmp/kda-native-q
cd /tmp/kda-native-q/source
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/external"
export SOURCE_DIGEST="$(cat ../source-digest.txt)"
export RESULT_DIR="$(realpath ../results)"
export TORCH_EXTENSIONS_DIR="$(realpath ../cache/torch)"
export TRITON_CACHE_DIR="$(realpath ../cache/triton)"
export CUTE_DSL_CACHE_DIR="$(realpath ../cache/cute)"
python verify.py
python native.py smoke
python native.py main
python native.py phases
python verify.py
```

The recorded runner requires two B300 GPUs; it times serially on GPU0. Its
preparation script downloads the pinned CF-BI sources and verifies their hashes.
Raw data and full environment versions accompany the results.

References:
[CF-BI 5735778](https://github.com/Inferact/zero-mismatch-rl/tree/5735778f43f7d94108d432706dc5b17765bf723a),
[FlashInfer c3af0c1](https://github.com/flashinfer-ai/flashinfer/tree/c3af0c13328fd23b256c53b9b4f94d9e160d118c).
See [upstream source hashes](upstream-provenance.json).
