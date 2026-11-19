# FP32-state KDA performance

B300 kernel-level comparisons. FP32 state/checkpoints, BF16 Q/K/V/g/beta and output;
gate parameters and arithmetic are FP32. FlashInfer uses BF16 state/checkpoints.
CF-BI uses FP32 entry matrices and row caches. No model, convolution or communication
costs are included. No BF16-state implementation is contributed by this change.

## Decode and prefill comparison

B300 SXM6 AC, CuTe DSL 4.8.1. All tables report geometric means of paired
per-case **latency ratios**; lower is better. H=12/16/32, D=128; decode B=64.

### Native decode: outputs and final state

Both sides process Q tokens in one kernel invocation and retain complete final
continuation state. No per-token checkpoint or acceptance selection is included.
The CF-BI multi-Q comparison uses the public general row engine (`rows_fast`);
Q1 uses the existing vLLM decode entry. Our implementation uses the terminal-only
mode of `prepare_decode`.

| Q | FP32 state = 1× | CF-BI C16 | CF-BI C64 | FlashInfer |
|---|---:|---:|---:|---:|
| 1 | 1.00× | 3.23× | 3.88× | 0.91× |
| 4 | 1.00× | 4.41× | 7.88× | — |
| 6 | 1.00× | 4.85× | 8.79× | — |
| 8 | 1.00× | 5.05× | 9.30× | — |

Complete-cycle mean latency, normalized per call, with all Q tokens advancing:

| Q | FP32 state = 1× | CF-BI C16 | CF-BI C64 | FlashInfer |
|---|---:|---:|---:|---:|
| 1 | 1.00× | 2.79× | 4.17× | 0.77× |
| 4 | 1.00× | 4.79× | 9.24× | — |
| 6 | 1.00× | 5.35× | 10.19× | — |
| 8 | 1.00× | 5.52× | 10.41× | — |

All 64 chunk positions, boundary updates and full reachable cycles are covered.
FlashInfer is included only for Q1 in this final-state-only comparison.

### MTP: per-token checkpoints and accepted-state commit

Earlier complete-API measurements, relative to FlashInfer. Our state/checkpoints
are FP32; FlashInfer's are BF16. These are a separate output obligation from the
terminal-only table above.

| Workload | Reference = 1× | FP32 state | CF-BI C16/C64 |
|---|---|---:|---|
| Decode Q4 + commit | FlashInfer | 1.11× | Not natively supported |
| Decode Q6 + commit | FlashInfer | 1.02× | Not natively supported |
| Decode Q8 + commit | FlashInfer | 1.03× | Not natively supported |

CF-BI supports multiple tokens in its row engines but does not export per-token
complete checkpoints or acceptance-driven restoration. Its vLLM BI route disables
speculative decoding. The old benchmark-specific Q-times-Q1 snapshot measurements
remain in the raw data only; they are not native multi-Q performance results.

### Prefill

| Workload | Reference = 1× | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|
| Chunk prefill | FlashKDA | 2.38× | 8.40× | 6.63× |
| Prefill | FlashKDA | 2.86× | 9.09× | 5.27× |
| 192K forward | FlashKDA | 1.59× | 4.97× | 2.78× |

Prefill retains the earlier confirmed packed-varlen measurements; its kernel
code is unchanged. It remains slower than FlashKDA and faster than the measured
CF-BI implementations in these workloads.

- [Native Q1/Q4/Q6/Q8: raw measurements and reproduction](data/native-multiq-b300-20261119/README.md).
- [Earlier Q1/Q8 and prefill data](data/b300-20261116/README.md).
- [Earlier Q4/Q6 checkpoint/commit data](data/q4-q6-b300-20261119/README.en.md).

## Proposed module versus measured sources

Native multi-Q measurements use this module directly. [Kernel mapping](kernel-mapping.json)
records the source lineage; the multi-Q kernel and API now add terminal-only state
output while preserving the ordered FP32 recurrence. Prefill and Q1 arithmetic
are unchanged. Correctness and paired checkpoint-mode regression are recorded in
[validation](validation.json), with [raw regression samples](regression.json).
The full suite passes **387 tests** on two B300 GPUs, including the new terminal-only
mode. Twelve independent FP64 diagnostics cover lengths 8/64/4096/32768
([errors](accuracy.json)). The earlier checkpoint/commit and prefill comparisons
retain their own source and measurement records.

These are kernel API results, not full vLLM model measurements. The recurrent and
CF-BI numerical contracts differ; both trainer and rollout must use the same
contract for train/inference alignment.

## Run this module

From the checkout root after installing the module's requirements:

```bash
PYTHONPATH=. python benchmarks/kda_fp32_state/bench.py --mode decode --query-length 4 --state-pool --accept-counts 0,1,2,3,4
PYTHONPATH=. python benchmarks/kda_fp32_state/bench.py --mode decode --query-length 6 --state-pool --accept-counts 0,1,2,3,4,5,6
PYTHONPATH=. python benchmarks/kda_fp32_state/bench.py --mode decode --query-length 8 --state-pool --accept-counts 0,1,2,3,4,5,6,7,8
PYTHONPATH=. python benchmarks/kda_fp32_state/bench.py --mode prefill --lengths 0,128,1024,4096 --heads 16
```

If executing from an uninstalled source tree, set `PYTHONPATH=.`. This branch is a
partial vLLM source snapshot; the kernel module can also be imported from this
tree without building the unrelated vLLM extensions.

The CLI defaults to prefill, lengths 0/128/1024/4096, H=16, decode B=64/Q1,
workspace 256 MiB and nine samples. Each Graph contains one complete call;
ten replays per sample, three discarded warmup samples. Prepare, compilation,
allocation, initial-state preparation and acceptance decisions are excluded.
With acceptance counts, GPU commit is included. Fixed addresses give hot-address
timings; these CLI examples are not the full external comparison matrix.
