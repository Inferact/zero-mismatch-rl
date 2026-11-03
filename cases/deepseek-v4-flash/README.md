# DeepSeek V4

## Current progress

- Compiled on: 2026-09-20; this was document reorganization only, validation was not rerun.
- Owner/task: not filled in.
- Phase and conclusion: this pass only organized historical lifecycle evidence; subsequent batch invariance work is a separate scope.
- Next step: verify the final July combination and results; complete the PR/commit index.
- Live workspace: the original task workspace (not included here).
- Verification card: no old card found on this machine in this pass; no inferred status was created.

## Versions (serving sub-workflow)

Recorded in the 2026-09 ledger; serving status at the time was `evaluation`.

- checkpoint: `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`; draft: inline_mtp (same repository and revision); shared weights verified
- manifest hash: `f680d01644593e7dd17e1d62ca4168788659b964bc97b9438f108cdc95a9af6e`; size 159609485896 bytes
- Image: `vllm/vllm-openai:nightly`, digest `sha256:987393f42c48b8a649961a3484d95d400db184b64e4e1bb7f77cb91536d0f05e`; vLLM commit `d223c900d85224c02f2162ee2c757a769e99f519`
- Next step: on the current baseline, combine only the required fixes and rerun the eager and default CUDA Graph lifecycle.
- Note: the four final local fix branches and their historical experiment boundaries are in the task workspace's inference-fix closeout report (2026-08-30, not included here).

## Results and fixes

- [consistency-case.md](consistency-case.md): five classes of differences on the deterministic path, weights and scaled-down models, sparse attention and Top-K
- [cp-compressed-boundary.md](cp-compressed-boundary.md): CP compressed boundary misalignment (Megatron #190/#194)
- [history-audit.md](reports/history-audit.md)

Each subsequent fix record: problem → effective change → repository/branch/commit/PR and merge status → regression evidence on the originally failing path.
The current status of related PRs must be verified; open/closed in historical reports only reflects the original audit date.

## Reproduction and handoff

New conclusions must save the exact checkpoint, image/code version, execution configuration, commands and key metrics;
full logs are located via machine/path or an accessible link. Update this page per the repository's directory conventions;
complex issues additionally get reports, and general lessons go into the guides.

## Serving performance levers (GB200 DP8/EP8)

Source: DS4 work records (agent session notes from 2026-08/09, no report in the repository; numbers only reflect the fork image and workload at the time). Workload: DeepSeek-V4-Flash fp8, GB200 2×4, DP8/EP8/TP1, 256 concurrency × 150 in / 3000 out.

**BI=0 arm**

- Winners: FlashInfer 0.6.18 overlay + `--moe-backend auto` (= FLASHINFER_TRTLLM MoE) 12075 → 14484 tok/s (about +20%); adding `cudagraph_mode=FULL_AND_PIECEWISE` → 15.1–16.0k (median about 15.3k, +5–10%).
- Dead ends (all tested): deepep_low_latency −12%; `--api-server-count 4` −5% (also turns off stats logging); NCCL_ALGO/PROTO Simple/LL128 −4%, NVLS does not support int8 AllGather; max-num-batched-tokens 4k/8k neutral; linear backend fitrt/cudnn neutral, cutlass crashes in `deep_gemm_fp8_o_proj`; DBO forces the V1 runner, and V1 rejects PIECEWISE for DS4, so it cannot be combined with FULL_AND_PIECEWISE.
- Floor: steady state is compute-bound (15.3 of 15.7 ms/step); MoE trtllm bmm 4.9 ms ≈ 87% of HBM bandwidth, each rank reads 34.6 GB of expert weights per step (805 MB per layer × 43 layers); gemm1 79.6 µs ≈ 6.75 TB/s, 84% of the B200 peak of 8 TB/s. DBO splits into 2 ubatches so expert weights are read twice per layer, measured −50% (8355 tok/s). Within a layer dispatch→gemm1→gemm2→combine is serial; the only overlappable part is the shared expert (about 3 µs), measured overlap ceiling 5.1%.
- Fork gating pitfall: `is_batch_invariant_quant_kernel_enabled()` only checks whether the `.so` exists and does not read `VLLM_BATCH_INVARIANT`, so every BI=0 run going through deep_gemm MoE runs the unoptimized BI quant kernel (9.9% in nsys, grid y=1024, 35.7 µs); with the fixed `.so` mounted + `VLLM_BI_QUANT_TOKEN_BLOCKS=128` it is 5.3 µs, 16703 → 18313 tok/s; alternatively `VLLM_BATCH_INVARIANT_KERNEL_LIB=/nonexistent` falls back to the upstream kernel (1.4% slower).
- JIT pollution: the fork's MHC TileLang split-k factor varies with num_tokens; under BI=0 every new prefill size triggers a 10–14 s recompile, all DP ranks stall together, TTFT p99 20–60 s; BI=1 pins split_tokens=1 and is unaffected. The fix is the fixed split-k patch + checking `tl_after_bench=0` during the bench.
- KV zeroing: the fork's `KVBlockZeroer` picks 16 B program blocks by the gcd of each segment's page size, 1–12 ms per step (7.6% of wall clock in an old trace); fixed upstream in #52058, not carried by the fork; `VLLM_KV_ZERO_MODE=off` as an upper bound gives +13.9%, chunk mode unit test 60 µs per call.

**BI=1 arm**

- Starting point: BI=0 all-opt 16091.5 vs BI=1 all-opt 12097.6, a 33% gap. nsys overlap ratio BI=0 1.97 / BI=1 1.33: BI=1 does 14% less GPU work but takes 28% longer; the gap is overlap, not communication volume.
- Winners: PR18 `.cu` actually built into the image (12097 → about 12860; later verified the kernel body bytes are identical, the 2% is drift); community Megatron-LM fork PR16 + PR22 merged (deterministic top-k + fusion) +6.7% (13801/13593/13735 vs 12843), PR16 alone +2.3%, PR22 +0.1%, only the merged version achieves both; **DeepEP LL over MNNVL** (`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1` + nvl8 overlay) 13710 → 15324/14971, dispatch recv-wait 37 → 15 µs, comm-only 4.8 → 3.0 ms/step, gap 17.4% (after kernel PRs) → 4.5% (paired; 5.6% against the best BI=0 sample); additionally `VLLM_FORCE_SHARED_EXPERTS_AUX_STREAM=1` (effective only under MNNVL, zero under RoCE) gives BI=1 16661 vs BI=0 16288, final BI=1 +3.2% (16834 vs 16313).
- Dead ends: DBO + FULL_DECODE_ONLY 7377 vs 12004 (−39%, LL has fixed per-call latency and the call count doubles); TP2/DP4 10086 (−17%), TP4/DP2 8689 (−28%); PR20 bi_gemm `.so` no change; `--async-scheduling` is already on by default, +1.1% is noise; splitting the shared MLP to fill the combine window 16462 vs 16861 (deferring down_proj makes the wait block).
- Floor: MoE backend locked to DEEPGEMM (the only batch-invariant fp8 MoE; cutlass rejects block-fp8, triton resolves to BATCHED_TRITON without BI) → a2a locked to deepep LL; 43 layers × dispatch+combine = 86 serial round trips per step; cross-layer overlap would compute before inputs are ready and break BI; the +5% is blocked by the TTFT floor (deepep LL prefill dispatch 0.41 s).
- Same-fabric lesson: the arm environment script pinned BI=1's DeepEP to RoCE (`USE_MNNVL=0`, `NVL_PEERS=4`, `NVSHMEM_HCA_PREFIX=rocep`) while BI=0's NCCL already went over P2P/MNNVL; a whole round of optimization attributed the gap to fixed communication latency, costing about two days; outputs across transports are bitwise identical (4/4 probes), the transport does not touch BI. Check steps are in [Fault localization 05](../../skill/guides/debugging-mismatch/05-comm-fault-ladder.md#same-fabric-fairness).

**cudagraph modes (BI=1, same configuration)**: FULL_AND_PIECEWISE 16409 / FULL 16068 / PIECEWISE 10423 tok/s; FULL_AND_PIECEWISE is 2.1% faster than FULL and 1.57× faster than PIECEWISE; FULL is slow on TTFT (1792 vs 764 ms, mixed prefill is always eager). All three modes PASS bitwise under short-prompt probes, but that does not prove PIECEWISE is safe; the mechanism is in the next section.

## Batch invariance troubleshooting example

Source: DS4 work records (2026-08 agent session notes), corresponding to the four rules in [Ablation method 04](../../skill/guides/debugging-mismatch/04-ablation-method.md#four-rules).

1. For a leak with about a 40% hit rate, the probability of a single round being all green is 0.6.
2. `if cudagraph_mode == NONE` in `_split_budget` also changed the budget split, so the eager experiment changed two variables at once.
3. Eager is 3.5× slower than with graphs, so batch composition changes with it; before statistical comparison, pin batch composition with offline hand-built steps.
4. The long-prompt filler step exceeded `max_cudagraph_capture_size`, so graphs did not participate at all and all nine levels being identical was illusory; zero hits are labeled VACUOUS.

Mechanism by which piecewise cudagraph breaks batch invariance: the graph path and the eager path are each batch-invariant but not bitwise equal to each other; which one runs is decided by `total tokens <= max_cudagraph_capture_size`, independent of padding (dense capture 1..128 is bitwise identical to default capture sizes); raising the capture limit to 256 lets overflowing steps also enter the graph, and the leak disappears. On DS4, FULL lands as FULL_DECODE_ONLY, where the path is decided by the step type; if the capture limit is explicitly set smaller than `max_num_seqs × decode_query_len`, large decode batches fall back to eager and the issue reappears under FULL too.

## Historical failure reports

The three DS4 full32 checkpoint failures (Lustre EFAULT / `unexpected pos`, optimizer load OOM, stale FP8 scale after restore) are
summarized in the [checkpoint pitfalls guide](../../skill/guides/checkpoint-resume-memory/03-pitfalls.md); the corresponding fixes are
PRs #195/#196/#197 of a community Megatron-LM fork. The original failure reports, large raw logs, checkpoints and chat transcripts
are not included here; when reusing a conclusion, check the guide entry and the PR, and downgrade the evidence strength if neither confirms it.
