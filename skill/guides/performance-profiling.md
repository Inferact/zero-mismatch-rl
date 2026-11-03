# Performance and profiling

A single-file guide in four parts: **workload and metric definition** (first prove you are measuring the same task), **fair A/B** (two kinds of A/B not to be conflated, pairing, computation), **decomposition and ablation** (separate timing, collecting enough information in one profile, micro bench, ablations that detect interactions, priorities and gates), and the **lever conclusion table** (which switches moved throughput, which are dead ends). Performance gates and correctness gates are recorded independently: low throughput does not mean wrong numerics, and correct numerics do not mean performance is acceptable. Concrete numbers are in cases: R3 return path [cases/nemotron/r3-return-path.md](../../cases/nemotron/r3-return-path.md), PC overhead [cases/nemotron/reports/bi0-prefix-caching-cost.md](../../cases/nemotron/reports/bi0-prefix-caching-cost.md). For BI kernel selection and cost see the [BI kernel appendix](training-model-support/appendix-bi-kernels.md); for the statistical protocol of paired acceptance see [evaluation and paired acceptance](recipe-evaluation/02-evaluation-and-acceptance.md#pairing-and-computation).

When performance tuning is mandatory in the Day-0 flow, which parameters to tune, and what counts as passing, see the day-0 workflow's performance-tuning step (not included here); this guide is the method.

## 1. Workload and metric definition

### Three throughput metrics

- **Per-request tok/s**: the reciprocal of TPOT. Measures decode latency.
- **Engine total tok/s**: completed tokens of all concurrent requests / common timing window. With multiple engines, use the full common window and total completed tokens; do not add up per-engine throughputs; across nodes with unsynchronized clocks, do not compute the window directly from timestamps.
- **RL per-step gen time (makespan)**: from the first request of a rollout batch entering to the last request leaving. This is the quantity RL cares about; it is affected by long-tail requests and scheduling, and can move in the opposite direction from the other two.

State which metric is reported; ratios can only be compared within the same metric. Order-of-magnitude references for the three metrics on Nemotron GB200 (per-request, aggregate, RL step) are in [r3-return-path §workload](../../cases/nemotron/r3-return-path.md#workload).

### Translating rollout targets into actual kernel workload

"batch=128, n=8, input 2K, output 8K" must at least clarify:

- Is 128 the number of prompts, the number of expanded requests, or per-engine concurrency? If it is prompts with 8 siblings each, total requests = 1024.
- Are 2K/8K max lengths or actual fixed per-request lengths? Record mean/max/percentiles of input tokens, actual total output and number of completed requests.
- `max_num_seqs` is the scheduler's cap on simultaneously active sequences; `max_num_batched_tokens` is the per-schedule token budget; neither equals the total request count.
- How many requests does each DP engine receive, how large are prefill chunks, how does the active decode batch change over time? A global 128 is not the per-rank kernel M.
- Is the prefix cache cold, warm, or naturally hit? The n siblings of the same prompt affect hits and prefill workload.
- Natural EOS, ignore EOS/fixed length, sampling parameters, seed, request order, max model length, scheduling policy.

### Replay bench: equal-workload replay of one RL step

Request count and prompt×n match one RL step, output lengths are taken per request from a real rollout, replayed at fixed length with ignore EOS, and the metric is makespan. Measure repeat noise of the same config first, then compare arm differences; differences smaller than noise support no conclusion. Both arms replay the same length table, so token counts, request counts and long tails are identical, and only then can the difference be attributed to the execution path. Fixed output suits equal-workload comparisons; it is not a natural-generation quality experiment. Nemotron's workload definition, noise and per-arm numbers are in [r3-return-path](../../cases/nemotron/r3-return-path.md).

In natural generation, even with the same weights and seed, different numerical paths/scheduling do not guarantee the same token counts; record output token receipts, and add frozen-token replay to aid attribution when needed. A short 2K+64 profile can locate startup/local hotspots, but **cannot** represent long-run 2K+8K decode, cache/page pressure, scheduling tails or full rollout speed; short profiles, full load tests and training E2E use different experiment IDs.

### Metrics are cumulative

`vllm:*_total` are process-cumulative counters; when the same engine runs multiple phases, each phase needs `after - before`; for why the first phase happens to be right and how to cross-check, see [ablation method 04](debugging-mismatch/04-ablation-method.md#metrics-are-cumulative).

## 2. Fair A/B

### Do not conflate two kinds of A/B

- **Causal comparison of the BI switch**: same software/model/scheduling/cache, only BI changed; if the switch automatically picks a different kernel, record the actual branch, and do not quietly add other optimizations.
- **Each side's best available config**: BI=0 uses the platform model's strong default/verified-best path, BI=1 uses the fastest path satisfying the agreed consistency. Backends may differ; every difference is disclosed.

Do not force BI=0 to use FA2, a specific MoE, non-default attention or Graph off to narrow the gap; and do not call experiments that locally turn off switches within BI=1 "native BI=1". If a path really must be replaced, name the candidate separately and pass the full contract gate. Shared optimizations (reducing launches both sides have, enabling a supported network path) are given to both sides equally first, and the shared gain and the reduction in BI's extra overhead are reported separately. BI=1 being faster than BI=0 is not automatically fake data, but it is enough to trigger an audit of config, workload, cache and timing.

### What to fix

Weights, inputs, generated tokens/lengths, output cap, concurrency, nodes, software, warmup and timing boundaries. First confirm output receipts and workload are identical; otherwise you can only report each side's own workload throughput. Separate the fresh-process first rollout from warm steady state.

**Same fabric**: before comparing communication bottlenecks, first confirm both arms use the same transport; check steps and counterexamples in [fault localization 05](debugging-mismatch/05-comm-fault-ladder.md#same-fabric-fairness). **Same node**: small gains need same-node pairing, alternating order and multiple rounds; with different node pools, counterbalance, and do not cherry-pick the fastest cross-pool combination. **JIT pollution**: JIT kernels specialized on token count recompile for every new prefill size under BI=0; check that compile lines during the bench are 0. **BI=0 silently running the BI kernel**: a fork whose gate only checks whether the `.so` exists makes BI=0 run the BI kernel; grep `vllm::batch_invariant::` in nsys to confirm. DS4 instances of both are in the [DS4 case](../../cases/deepseek-v4-flash/README.md#serving-performance-levers-gb200-dp8ep8).

### Computation and reporting

```text
output tokens/ms = completed output tokens / (wall_seconds * 1000)
BI throughput deficit = 1 - throughput_BI1 / throughput_BI0
BI time overhead      = time_BI1 / time_BI0 - 1    # matched work only
```

Latency reduction `1-new/old` differs from throughput gain `old/new-1`; 16.35% lower throughput does not correspond to a 16.35% increase in time. Report numerator, denominator, requests/completions, raw receipts, config differences, sample count and variance; take the median over multiple pairs and report single pairs separately (Nemotron's per-pair and median numbers are in [delivered results](../../cases/nemotron/results/final/README.md)). Gains of individual items A and B cannot be added; measure the combination separately. A few observations are not enough to say a combination beats the best single item. Skipping the old-logprob forward is listed separately only when the RL algorithm allows it, and does not count as a generic gain of the BI kernel. The historical "11.8% E2E" figure, once questioned for lacking same-metric evidence, can no longer serve as a baseline; every ratio must be recomputable from explicit artifacts.

## 3. Decomposition and ablation

### Two stages: rule out structural flaws first, then fine-tune

Once proxy inference consistency passes, do a performance feasibility check: sweep token row counts on real operator shapes, count launches and host gaps, look for per-token launches, synchronous readbacks, duplicate copies and abnormal workspaces; confirm the target Graph can capture/replay. After training backward is wired up, measure forward and forward+backward (by token count) on the main topology chosen in C1, compare with the native training path of the same build, and check against the C1 budget; avoid choosing a scheme just because decode is fast when training/prefill is extremely slow. Fine-tuning after full consistency passes goes E2E profile → micro benchmark → E2E acceptance: micro benchmarks first screen representative high-time-share shapes, then sweep all target shapes and edges; all correctness must pass, and performance is assessed by real call frequency and time share; the final gain is based on multi-round paired E2E on the same workload. New/replacement kernels do not use Triton; use CuTe DSL, CUDA/CUTLASS or reuse existing implementations; historical Triton data is only a reference. Final delivery still needs full acceptance, see [validation.md](../validation.md). When a structural problem is found, fix it first and regress numerics; do not turn this stage into endless tuning.

### Decompose the cost first

Time model load/JIT, rollout prefill/decode, old-logprob, backward/update, weight sync, and offload/wake separately. Split weight sync further into quantization, reshard and transfer: in an external 310B case transfer took under 2 s and end-to-end about 4 s, so optimizing bandwidth alone saves at most half. Total kernel times can overlap and cannot be added as wall time. nsys window, rank/GPU normalization, Graph node tracing and sampling intervals must be consistent.

Profile the effective config and actual kernels, not just the BI switch. In DS4 the same quantized `.so` can load under both BI0/BI1, but different MoE layouts select different schedulers. For NVFP4 MoE models, account separately for dense linear, expert GEMM, expert routing/indexing, all-to-all and attention; one framework or vendor name does not represent the whole path. One real server run of Nemotron simultaneously used FlashInfer FP8 linear, Humming NVFP4 dense GEMM, Humming indexed experts and FlashInfer NVLink one-sided all-to-all; another BI=0 trial, where dense GEMM actually selected FlashInfer CuTe DSL, hit a guard expecting Humming and did not even complete forward ([raw evidence](../../cases/nemotron/dapo-20260927.md)). Such failures only prompt a re-audit of backends and build paths, and cannot enter the performance denominator.

### Measure at the production scale

A speedup measured on fewer nodes carries the small setup's per-rank load, collective shapes and fixed costs. When a cost is paid per
micro-batch, per layer or per rank, the gain depends on samples per rank and on rank count, so the small-scale number does not
transfer. Two Model A cases (a large MoE; 2026-10, 8×4 GB200):

- **Packing.** On 2 nodes (FSDP×8, 32 samples per rank) cu_seqlens packing cut forward+backward from 211 s to 145 s. Swapped into G6
  at FSDP×32 (8 samples per rank), the policy update went from 257 s to 340–423 s: each rank still ran 4 micro-batches, 1.6–1.8 of them
  padding added only to align micro-batch counts across ranks, and every micro-batch paid a fixed cost of about 7 s (NVFP4 expert decode
  and all-gather). It was rolled back at the next checkpoint.
- **Megatron.** From 2 nodes (PP2/EP4, 3.7–4.0 ms per token per GPU) the policy update was extrapolated to 82–138 s against FSDP's
  260 s. The first 8-node steps (EP32) took 251–271 s: per-rank tokens ranged 2.6k–35k with contiguous slicing, a 4096-token budget made
  every rank run 8 micro-batches (2.8–3.1 of them padding) that all waited for the slowest rank in each EP32 all-to-all, and per-layer
  host synchronizations stalled the GPU. A smoke step on the GB300 bring-up took 930 s, 522 s of it in the weight sync after a save; the owner's reaction to that 15-minute step (one step cannot possibly take 15 minutes, so something must be wrong) is the right
  prior: a step far above the extrapolation is a defect to find, not a measurement to accept.

Rule: reproduce the production per-rank load in the small test (samples per rank, length distribution, padding micro-batches implied by
the production rank count) or measure on the production layout directly; treat a small-scale result as a candidate, swap it in at a
checkpoint boundary with a rollback checkpoint kept, and compare the first production steps against the steps before the swap.

### Fixed per-step costs

Large models make every fixed cost per step visible. Break the step into phases on every rank (wall time, and per phase: allocator
retries, faults, CPU user time) and attack the largest first. Model A Megatron, two rounds on the production layout:

| Phase | Before | Cause | Fix | After |
|---|---|---|---|---|
| Weight sync | 118–120 s | Per-tensor work on 55,374 adapter tensors | [Per-tensor fixes](weight-update/03-sync-robustness.md#sync-time-is-per-tensor-not-per-byte) | 33–82 s |
| Frozen-weight restore | 10–53 s, once 395 s | Lazy file mapping, reclaimed page cache | [Populate, node-shared file](checkpoint-resume-memory/03-pitfalls.md#19-file-backed-offload-of-frozen-weights) | 2.5–3.4 s; 7–37 s once the node-shared file was added (trades restore time for page cache) |
| vLLM sleep / wake | 7–17 s / 2.5–3 s | A new pinned backup every sleep | [Retained exact-size backup](checkpoint-resume-memory/03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned) | ~2 s / ~2 s |
| forward+backward | 216–288 s | DP token imbalance; CPU waiting on the GPU inside every layer | Balance tokens across DP ranks per mini-batch (longest-first greedy, equal sample counts; the loss is normalized over the global mini-batch, so only summation order changes); remove the host syncs | 156–243 s |

The host syncs were found by py-spy on one production rank during forward+backward: the MoE all-to-all dispatcher's device-to-host
token count (36% of samples; inherent to that dispatcher), the NVFP4 QAT size-class index built from a Python list with boolean
indexing (31%), and the per-layer copy of a RoPE parameter to the CPU (16%). Building the indices on the host once and caching the
frequencies cut forward+backward by 11% on 2 nodes; a unit test runs the new code under `torch.cuda.set_sync_debug_mode("error")` and is
bitwise against the old code on the same inputs. None of this showed on the 2-node layout, which is why the profile came from production.

### Collect all attribution information in one profile

1. First verify the actual serving config, inputs and output statistics; for inference-only use vLLM serve/the existing client, and do not launch training for the profile. Distributed executors use the mp/Ray actually needed.
2. Add low-overhead, default-off NVTX at semantic module boundaries: attention, indexer/Top-K, index preparation, MLA, MHC, shared/routed MoE, norm, linear, sampling, communication. Dense models do not reuse DS4's module list.
3. Use nsys first to find the full timeline and hotspots; confirm CUDA Graph node tracing is on and records the actual replay kernels. A report with only graph launches and no node detail does not support per-kernel comparison.
4. Count calls, GPU time, CPU launch/sync, allocation and shape per path; aggregate by rank, phase, module.
5. Choose representative prefill, steady decode, mixed and tail windows. Then use ncu on a few specific kernels/shapes to study bandwidth, occupancy, memory access and compute bottlenecks; do not treat wall time under ncu perturbation as throughput.
6. Finally measure throughput on the same full workload without profiler/heavy dumps. Record whether profile and formal timing used the same code and config.

Kernel names are not module identity: copy/fill/cast, `index_select`, generic elementwise, GEMM, and different shapes of the same template are reused in many places across layers. Distinguishing "the indexer's gather" from other gathers needs NVTX/launch correlation plus shape. Parent/child NVTX inclusive times cannot be added; the sum of concurrent kernel GPU durations does not equal wall time; label multi-GPU sum, per-GPU mean and critical-path time separately.

When a wrapper/overlay also rewrites the kernel selector, use **three layers of evidence** to confirm "who actually launched this kernel": per-worker class, object count and import source at init; input shape, module name and dispatch at the real forward callsite; launch correlation and GPU kernel in the final profiler. Merely seeing that some linear objects were overridden, or merely seeing some Humming shape, does not prove what they launched at replay; setting only the env vars the overlay needs while the worker did not inherit that entry point also creates a misleading "candidate enabled". Graph can hide ordinary Python CPU op events, so a `cpu_op` count of 0 alone cannot assert the callsite did not run. When evidence conflicts, first do a minimal diagnostic probe that keeps the numerical path unchanged; do not fall back to an unverified default operator just to raise throughput.

The frontend/control plane is also in scope for decomposition: py-spy shows the API process main-loop share, asyncio loop-lag shows whether the frontend is saturated (in the R3 case most frontend main-loop time was in `output_handler`; numbers in the case).

For roofline bounds, first list the peak compute for the actual dtype and the bandwidth of the memory level under study; L2/shared-memory/non-Tensor-Core kernels cannot borrow HBM/Tensor Core bounds. Do not guess which hardware metric was used from chart coordinates.

### Single-operator replay and synthetic micro benchmarks

A synthetic shape matrix proves algorithmic edges; real rollout replay provides score distributions, ties, masks, non-contiguous layouts, cold/hot buffers and call frequencies. The two complement each other. Weight by really sampled shapes/frequencies, keep high percentiles/rare edges; report small-M decode and large-M prefill separately. Workspace allocation has two timing definitions, included and excluded; when the original allocates per call and the candidate preallocates, do not compare only kernel body. Micro A/B fixes inputs and resources, candidate bitwise against baseline; one change at a time, covering at least the actually used batch=1/4/32/128, different neighbor compositions, prefill/decode, Graph, mask/tie/edges; shapes not run are marked uncovered. When Triton is banned, a "temporary benchmark" cannot be used to bypass it.

### Ablations must detect interactions

With two optimizations A and B, measure `base / A / B / A+B`; when there are also network/Graph/cache switches, add the corresponding comparisons, without mechanically running all combinations. Existing A+B data is reused only under the same weights, inputs, software and timing conditions. The first round after resuming from checkpoint N is usually labeled step N+1; adjacent names do not prove the same policy; pin the checkpoint content digest and export fingerprint, and do not mix BI0/BI1 or R3/no-R3 branches. When looking only at rollout, exit with a clean stop mechanism after completion, without running optimizer/save that pollutes timing. Record that all requests really completed. Top-K and index post-processing can usually be ablated orthogonally, but shared buffers, layouts or Graph create interactions, and combinations must still be re-accepted.

To locate which segment of a long path carries the cost, use chained ablation that toggles segment by segment along the data flow; the arm definitions and the add-back check are in [ablation method 04](debugging-mismatch/04-ablation-method.md#segment-by-segment-isolation-add-the-total-difference-back-up).

### Optimization priority and correctness gates

1. Eliminate idle/duplicate work: launch grid mismatched with real workload, duplicate copies, invalid tokens.
2. Reduce host scheduling/sync and kernel launches; restore usable Graph without changing the output contract.
3. Fuse boundaries such as gather/dequant/index post-processing, preserving the original rounding, traversal order and padding/sentinels.
4. Only then consider changing kernel algorithms, tiling or collectives; these may change reduction order and need new equivalence evidence.

For example, a quantization kernel hard-launches 1024 blocks per expert while decode has very few valid rows. When shrinking the grid the stride must change accordingly; under Graph capture use a host-known capacity upper bound, and do not read the device token count to produce a dynamic launch; the grid is still measured on real shapes. If dequant/gather are independent per-element operations, re-tiling may stay bitwise equivalent; fusing into sparse attention must preserve scale rounding and KV traversal/reduction order.

Each candidate first runs same-input kernel comparison (real dtype/shape, non-contiguous or mask layouts, empty experts, tails, extremes, ties, reused dirty buffers, Graph replay), then fixed-token engine gates, and finally actual rollout/training. Passing formatting on test code does not replace compiling and running. New performance measurements are always rerun on the final code version; a JIT refactor that looks behavior-preserving statically can still fail to compile. When a candidate shows no gain or regresses, keep the conclusion and remove it from the default path.

### Delivery

Each item includes the actual kernel/path, correctness before and after, micro latency, rollout and full-step results, number of independent samples, Graph/topology scope, raw receipts, and unsupported configs. Report whether quality evaluation and long-term stability have been done; do not substitute kernel speedup for an end-to-end conclusion.

## 4. Lever conclusion table

Switches that were measured. Numbers are in the corresponding case; this only records direction and dead ends not to retry.

| Model / arm | Winner | Dead ends (tested, do not retry) | Structural floor |
|---|---|---|---|
| DS4-Flash GB200 DP8/EP8 BI=0 | FlashInfer trtllm MoE (`--moe-backend auto` + new FlashInfer); `cudagraph_mode=FULL_AND_PIECEWISE` | deepep_low_latency, `--api-server-count>1`, NCCL_ALGO/PROTO, max-num-batched-tokens 4k/8k, linear backend fitrt/cudnn (cutlass crashes), DBO (forces V1 runner, V1 does not accept PIECEWISE) | MoE decode is weight-bound, gemm1 is near the HBM roofline; DBO splitting ubatches rereads weights and always loses; communication overlap is capped at the few µs of shared experts |
| DS4-Flash BI=1 | Merging deterministic top-k + fusion PRs together (each nearly ineffective alone, must be tested together); DeepEP LL over MNNVL (`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1` + nvl8 overlay) is the biggest lever; after that, shared experts aux stream only helps under MNNVL | DBO, TP2/DP4, TP4/DP2, PR20 bi_gemm, async-scheduling (already on by default), splitting shared MLP to fill the combine window | MoE backend locked to DEEPGEMM (the only BI fp8 MoE) → a2a locked to deepep LL; per-layer dispatch+combine is serial; cross-layer overlap breaks BI; finally blocked by the TTFT floor |
| DS4 cudagraph | Pin FULL_AND_PIECEWISE under BI (faster than both FULL and PIECEWISE) | PIECEWISE: graph vs. eager fallback is decided by total token count vs `max_cudagraph_capture_size`, a function of neighbors, which breaks BI | FULL is slow on TTFT (mixed prefill always eager) |
| Nemotron-H BI=0 | PC off (when R3 forces PC on, upstream #58851 is best); R3 finish-only + per-step packed IPC (overlay, not in PR/image; equivalence in case) brings makespan back near the R3-off floor; then packing per-step logprobs into a list for the frontend (overlay) makes the frontend no longer saturated, and BI=0 is 16% faster than BI=1 | Raising the zero-copy threshold (no effect); `max_num_seqs` 256 (slower); PR 57968 keys-only as is | PC on (Mamba align mode) has a fixed cost: extra eager launches per decode step, 8K prefill split in two |
| Nemotron-H BI=1 vs BI=0 | — | — | Under fixed-length bench BI=1 has a measurable throughput loss, per-pair and median in case; BI=1 Mamba2 rejects PC; when both arms are flat e2e, first check whether the frontend is saturated (BI=1 engine-bound and BI=0 frontend-bound are different bottlenecks) |
| Nemotron-H 256K training update (both arms) | BI=1 per-GPU per-micro-batch token cap 64K → 128K: in production update is only about 5% faster per token (13.1 → 12.4 s / million tokens), actor reserved rises from 99 GiB to 168 GiB (184 GiB per GPU, cap taken as 170), small gain, thin headroom | Less recompute (only some layers, only Mamba layers, none): both arms OOM at 32K–64K per GPU. BI=0 changing per-GPU token cap (no difference across 32K–192K): with CP>1 the recipe disables dynamic batch, each micro-batch is fixed at one sequence, so this parameter has no effect | update is only 14–18% of the full step; rollout is about 80%, determined by sandbox execution and the slowest few trajectories. Compute the shares before deciding which segment to tune; before tuning a parameter, confirm in the resolved config that it takes effect |
| Model A FSDP2 trainer (NVFP4 frozen base + QAT, LoRA r32, FSDP×32, 2026-10) | Batched NVFP4 experts with one host sync per MoE forward (a per-expert Python loop projected 18.5 min per step at micro-batch 1) and a fused one-pass NVFP4 decode; local-shard AdamW and grad clip (DTensor `foreach` is quadratic in tensor count and stalled between 6k and 9k LoRA tensors; the model has 55k); model-side chunked logprobs; old logprob from the update forward; micro-batch 4 with `sleep_mode_offload_cudagraph` and `enable_nccl_comm_suspend` (policy update 252–299 → ~195 s) | cu_seqlens packing at FSDP×32 (slower, [§3](#measure-at-the-production-scale)) | Eager attention keeps FP32 scores of every query chunk for backward; step medians 439–508 s per segment, rollout ~120–130 s of it |
| Model A Megatron backend (TP1 PP1 EP32, same recipe, 2026-10) | Token balancing across DP ranks, removing per-layer host syncs, per-tensor sync fixes, populate-before-restore, retained sleep backup ([§3](#fixed-per-step-costs)); step median 422 s, later 357 s, against FSDP segment medians of 439–508 s | Dynamic-batching budget 6144 (OOM at EP32, 4096 holds); `MADV_WILLNEED` (same 3.5 GB/s as populate); `mlock` of frozen weights (Ray counts it, +228 GiB per node); deterministic algorithms (not a day-0 requirement) | The all-to-all dispatcher's per-layer device-to-host token count; rollout (~120 s) is the same vLLM on both backends, so the step cannot fall below it |
| Model A vLLM rollout with NVFP4 expert LoRA (GB200, decode; LoRA made decode 1.85× slower than no LoRA) | Split-K shrink / split-N expand MoE-LoRA kernel for ≤256 tokens: per-call 35–96 → 9–52 µs (w13), 64-request rollout 151 → 116 s, batch-1 decode 40.6 → 56.8 tok/s | Not a pure gain: one extra launch, an FP32 partial-sum round trip and a resident workspace; above 256 tokens the original kernel wins, and the path switch at 256/257 changes rounding for the same token. A deterministic split-K `lora_shrink` cost ~5% throughput and is not used (non-repeatability is not a defect at day-0) | Small-M decode is latency-bound: few programs, long serial K loops |
| Any BI=0 on fork | — | Letting the `.so` exist makes it silently run the unoptimized BI quantization kernel; mount the fixed `.so` or `VLLM_BATCH_INVARIANT_KERNEL_LIB=/nonexistent` | — |

Sources: the three Model A rows: the Model A case notes (not included here); the three DS4 rows [DS4 case](../../cases/deepseek-v4-flash/README.md#serving-performance-levers-gb200-dp8ep8); the two Nemotron rows [r3-return-path](../../cases/nemotron/r3-return-path.md), [bi0-prefix-caching-cost](../../cases/nemotron/reports/bi0-prefix-caching-cost.md), [prefix-cache-perf](../../cases/nemotron/reports/prefix-cache-perf.md), [delivered results](../../cases/nemotron/results/final/README.md); fork gating [DS4 case](../../cases/deepseek-v4-flash/README.md#serving-performance-levers-gb200-dp8ep8); BI kernel selection [BI kernel appendix](training-model-support/appendix-bi-kernels.md). 256K update row: single-node probes and production metrics from Nemotron SWE-bench Pro 256K training (2026-10).
