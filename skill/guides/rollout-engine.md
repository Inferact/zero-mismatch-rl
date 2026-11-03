# Rollout engine and sampling

The rollout engine (vLLM) in RL is not serving: it must repeatedly pause/sleep/wake/reload next to the training process, sampling randomness must be auditable, prefix cache and speculative decoding change the premise of "whether the same token sequence is bitwise identical", and RL-specific features such as R3 put hard constraints on engine configuration. The five sections of this doc cover these mechanisms; the last section is the debugging procedure "save one rollout and replay the old-logprob forward". Performance metrics and A/B method are in [Performance and profiling](performance-profiling.md); BI kernels and cudagraph are in the [BI kernel appendix](training-model-support/appendix-bi-kernels.md); weight-update completion criteria are in [Weight update](weight-update/README.md).

## 1. Sampling seed collision

**Mechanism.** Each vLLM DP engine resets the global RNG with the same `seed` at the end of warmup; requests without a seed sample from the global stream, every engine's counter starts at the same point, and the n-th request samples the same tokens on every engine. When GRPO distributes sibling samples of the same prompt across different DP engines, there is heavy duplication within the group: before the Nemotron fix only 64–70 of 256 responses were distinct and up to 43% of sibling pairs were identical; after the fix 256/256 ([seed PR report](../../cases/nemotron/reports/vllm-dp-sampler-seed-pr.md)). The fraction of zero-advantage groups barely changed before vs after the fix; zero advantage comes mainly from task difficulty and cannot be attributed to seed collision ([seedfix A/B](../../cases/nemotron/reports/seedfix-ab-and-aime-baseline.md)).

**Fix.** Reset the global RNG by `(seed, data_parallel_index)`; index 0 is bitwise identical to the original behavior (vllm-project/vllm#59788 merged). Use `data_parallel_index`, not `data_parallel_rank`: dense (non-MoE) models run independent engines under DP, and `reconfigure_for_independent_dp_rank()` resets `data_parallel_rank` to 0 (`vllm/v1/engine/core.py`), so 4 engines get exactly the same seed stream and n=4 is actually worse. Nemotron is MoE and uses `DPEngineCoreProc`, where the rank is correct, so the 1024/1024 validation on verl holds, but the gap for dense models can only be fixed by the index version.

**How to accept.** In the rollout dump, count per group the number of distinct samples and the fraction of identical sibling pairs. The baseline with different seeds is about 0.1% identical sibling pairs, all short answers of ≤8 tokens (the model gives `\boxed{}` directly), and the fraction sharing a prefix of ≥16 tokens is about 1–3%; above this level it is RNG correlation. Acceptance rules are in [validation](../validation.md#dp-seed-collision-identical-in-group-sibling-samples).

**Do not.** verl `rollout.full_determinism=True` gives the n responses in a group the same seed; do not turn it on.

**Comparison with other engines** (details and data tables in [sglang-dp-seed-collision](../../cases/nemotron/reports/sglang-dp-seed-collision.md)): SGLang's default mode has the same mechanism (all DP ranks share one `random_seed`, global torch CUDA RNG), but two siblings collide only if their row index and RNG offset are both the same when each is prefilled on its rank; this basically only happens in the first small batch of a freshly started server and is not measurable at P≥64; SGLang deterministic mode without a seed always uses `sampling_seed=42`, 100% identical, independent of DP; miles gives each sibling `rollout_seed+i`, and in non-deterministic mode each engine uses `random_seed=seed+rank`, avoiding it.

**Greedy validation basis.** When using greedy for A/B, n repeats are identical (Nemotron: 32 repeats identical), and the score can only be a multiple of 1/number-of-problems (for AIME's 30 problems, multiples of 1/30, 13/30 → 19–22/30). A/B before vs after the seed fix shows no difference under greedy; this does not mean the fix is ineffective, it means greedy does not sample at all. Rising entropy and the three config curves overlapping can only be seen in sampled runs.

## 2. Prefix cache semantics

**Premise.** "Enabling PC does not change numerics" requires the same request to be bitwise identical under hit, miss, and PC off. For pure attention models this is guaranteed by KV block reuse; for stateful layers (Mamba2 conv/ssm state) it depends on which states the engine copies.

**The Mamba2 pitfall.** vLLM's Mamba align mode only copies the two states conv/ssm; under BI=1, Mamba2 uses exact replay (ported from vllm#55905, storing three extra chunk buffers x/dt/B) and needs 5 states, so under BI=1 Mamba2 originally refused PC. The fix is to copy all states in each layer's declared state list, after which hit/miss/PC off are bitwise identical; early versions had a conv window alignment bug in state export (written tail-aligned, read from the start). Real-model correctness, gate3, and throughput are in [bi-mamba2-prefix-cache](../../cases/nemotron/reports/bi-mamba2-prefix-cache.md).

**Cost.** Enabling PC under BI=0 has a measurable cost: in Mamba align mode each decode step adds a batch of eager launches, and 8K prefill is split into two segments; upstream #58851's approach is best. Numbers in [bi0-prefix-caching-cost](../../cases/nemotron/reports/bi0-prefix-caching-cost.md), [prefix-cache-perf](../../cases/nemotron/reports/prefix-cache-perf.md).

**Confounder between the two arms.** Under BI=1 Mamba2 refuses PC, while under BI=0 R3's AuxOutput forces PC on, so PC on/off is one of the confounders in BI=0 vs BI=1 throughput comparisons; list confounders first, then compare — rules in [recipe evaluation 02](recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance), Nemotron's concrete items in the [recipe deviation table](../../cases/nemotron/recipe.md).

**Relation to cudagraph.** A PC hit changes the number of tokens in a step; piecewise cudagraph decides between the graph and eager fallback by total tokens vs `max_cudagraph_capture_size`; each path is batch-invariant on its own but they are not bitwise equal to each other; under BI pin `FULL_AND_PIECEWISE`, see the [BI kernel appendix](training-model-support/appendix-bi-kernels.md).

**Checks for stateful layers and caches.** Mamba/SSM also need to save chunk boundary state and partial input buffers; check aligned/unaligned split, cross-chunk, and recovery separately. Long-context KV/page/workspace indexing and capacity must cover the vicinity of boundaries.

When testing prefill, confirm whether the prefix cache was hit; if necessary, disable/isolate the cache during diagnosis so the target path executes. Separately verify production cache hit, reuse, and preemption/recompute paths. Do not make a reload comparison pass by clearing the cache and then claim the native lifecycle is fixed.

Performance fixes on the cache path are compared bitwise against "cache off" as the gold standard: put hit positions outside chunk boundaries, cover every DP rank, recompute hits after forced preemption, and chunks truncated by the scheduling budget, and confirm in the trace that the target branch actually executed; then add a hook that checks the device-side state index against the host prediction step by step, plus a per-problem quality eval. Two fixes that each pass individually may silently fail when stacked; the combination must pass the gate separately. Mamba example in [Nemotron prefix caching](../../cases/nemotron/consistency-case.md#mamba-prefix-caching-align-mode).

## 3. MTP / speculative decoding

- Nemotron checkpoint has `num_nextn_predict_layers: 1`, and the MTP block is unquantized (bf16). It does not take part in training; during weight sync either sync it separately or explicitly skip it in the mapping — do not let the draft silently stay on an old version ([Training-side model support 01](training-model-support/01-module-inventory.md)).
- Usable under BI=0. BI=1 + MTP requires the speculative verification logprobs to be bitwise identical to a whole-sequence forward; without that verification you cannot claim true on-policy.
- If used, both arms must use it, otherwise both throughput and logprob comparisons are invalid. The user decided not to use it for now.
- **Acceptance rate is a sensitive indicator that exposes weight-update errors.** A partial update that sends only some shards of a fused parameter (e.g. only the k/v shards of qkv) leaves the q rows as uninitialized GPU memory; the target model's text looks normal and only the `vllm:spec_decode_num_accepted_tokens` delta exposes it (draft acceptance length 3.99→1.00). The minimum granularity is all shards of a fused parameter; the safest is a whole layer; the first update after sleep level 2 must be full (wake only reallocates GPU memory, contents are uninitialized). Some models (Model C with its draft model, Model K) pack the same tensor into a cross-layer fused parameter, so there is no safe partial draft update, only full. See [Weight update](weight-update/README.md), [precision pitfalls table](precision-quant-qat/03-pitfalls.md).
- In the lifecycle matrix, L6/L9 use the dual criteria fixed-token + acceptance (from the day-0 serving validation procedure, not included here).

## 4. R3 (routed experts replay) functional constraints

- **What it returns.** Per token one row of `(num_moe_layers, top_k)` uint8 routed expert ids (Nemotron 52 layers × 6 = 312 B), returned to the trainer with the request's CompletionOutput for routing replay; 0 is a valid expert id, so add a distinctness check rather than a non-zero check.
- **PC must be on.** R3's AuxOutput is keyed by KV block hash; with PC off there is no key; this is why the BI=0 arm in §2 is forced to enable PC, and the keys-only form (PR 57968) does not lift this either.
- **Why per-step return is expensive.** 312 B exceeds the msgpack zero-copy threshold of 256 B, so one row per request per step becomes a separate zero-copy frame, and the frontend accumulates millions of small objects per request until the end before concatenating; the cost lands on the engine→frontend return path, not the capture hook. The fix is returning once at request finish (finish-only) + packing worker→engine per step into one ndarray; `output_processor` already exposes `routed_experts` only at finish, so semantics are unchanged. Trade-offs between the two API forms (inline ndarray, keys-only), the PR 57968 defect list, numbers, and file:line are in the [r3-return-path case](../../cases/nemotron/r3-return-path.md).
- **Sleep with R3 needs `pause(mode="keep")`.** The routed-experts AuxOutput connector only lets the prefix cache be reset from a `keep` pause, while verl's sleep paused with the default `abort`. Fix in verl: when `enable_rollout_routing_replay` is on, sleep with `mode="keep"` (servers are drained before sleep, so nothing is actually kept). Found on Model D.
- **There is another floor under R3.** msgpack unpacking of three small ndarrays per token for sampled logprobs is a fixed frontend cost (about 25–30 µs/token), larger than all other work once R3 is fixed; BI=1 does not see it when engine-bound, but BI=0's faster engine hits it, and the two arms being tied end-to-end is often for this reason, not kernels. Check the frontend asyncio busy fraction before comparing the two arms ([performance guide](performance-profiling.md), numbers in the [case](../../cases/nemotron/r3-return-path.md#second-floor-frontend-logprobs-unpacking-evening-of-2026-10-06)).
- **What does not change.** R3 introduces no GPU→CPU sync, async scheduling is preserved, capture writes into a statically preallocated buffer, and cudagraph is unaffected. flashinfer_cutedsl and Humming are both modular MoE (taking the router's computed top-k), so moving R3 into the kernel saves nothing.
- **Limits of the equivalence check.** BI=1 cannot do the equivalence check (Mamba2 BI refuses PC while AuxOutput forces PC, until the §2 fix lands). BI=0 compares sha with strictly serial single-request greedy, but BI=0 routing itself is not batch-invariant, and even an unpatched baseline repeatedly shows partial inequality against the same reference; so first measure drift with multiple baseline-vs-baseline runs, and a patch counts as equivalent only if it is fully equal to a clean reference; bitwise comparisons of concurrent greedy dumps cannot serve as evidence (see the equality section in the case).

## 5. Lifecycle and GPU memory ledger

**Three pause modes** (`EngineCore.pause_scheduler`): `abort` marks all running requests `FINISHED_ABORTED`, queues new requests, and optionally clears the cache; `wait` queues new requests but keeps stepping until running requests finish (not supported by the inproc engine); `keep` pauses everything, and the Future completes when the output queue is empty. In all three modes the Future returned by pause is the barrier: weight updates must wait for it to complete before starting, otherwise workers are still stepping. In `abort` mode the client receives aborted outputs and must resend them itself; requests arriving during pause are queued and released together at resume, using the **new** weights. In one RL step these are "orphan requests": rollout pauses before finishing its tail, or the agent loop keeps sending requests during pause, and these requests' tokens do not belong to any policy version. Before each pause confirm in-flight is 0, after resume confirm the queue is empty, and account per [policy versions](weight-update/02-policy-versions.md). The HTTP endpoints `/sleep`, `/wake_up`, `/is_sleeping`, pause/resume are dev endpoints and require `VLLM_SERVER_DEV_MODE=1`, otherwise 404.

**Three sleep levels** (`EngineCore.sleep`, `Worker.sleep`):

| level | What it does | GPU memory | Host memory |
| --- | --- | --- | --- |
| 0 | only pauses scheduling | unchanged | unchanged |
| 1 | copies weights to CPU, drops KV cache, clears prefix cache | frees weights and KV; the rest stays on the GPU | one weight backup per rank (DS4.1 78.61 GiB); pinned cache is not returned after wake, see [checkpoint 03 §11](checkpoint-resume-memory/03-pitfalls.md#11-sleep-backup-in-torch-pinned-memory-rounded-up-and-never-returned) |
| 2 | discards weights and KV, only saves `named_buffers` to CPU | frees weights and KV; the rest stays on the GPU | small |

- **Level 1 wake is directly usable**, contents bitwise identical to before sleep. **Level 2 `wake(weights)` only reallocates GPU memory, contents are uninitialized**; the first update must be full (§3), and only then wake KV. Whether level 2 is usable depends on the model's full reload path; verify it separately in G1.
- **Neither level frees everything.** sleep only manages two pools, `weights` and `kv_cache`; workspace, runtime state, CUDA graph pool, all-reduce workspace, NCCL buffers, and CUDA context remain; per measurements in [vllm#59055](https://github.com/vllm-project/vllm/issues/59055) on 4×GB300 it is 14–21 GiB per GPU (varies with model and config), and about 4–7 GiB after applying the PRs tracked by that issue and enabling `--sleep-mode-offload-cudagraph`, `--enable-nccl-comm-suspend`. Read the actual residual from the worker log `Sleep mode freed X GiB memory, Y GiB memory is still in use`.
- After wake, "reserved is high but no GPU memory available" is an allocator boundary problem: PyTorch's reserved-but-unallocated blocks cannot be used by another allocation path such as CuMem, see [checkpoint 04](checkpoint-resume-memory/04-gpu-oom-and-staging.md).

**Colocation is time-sharing.** The training process and vLLM take turns on the same GPUs; the GPU memory budget is computed as one ledger for the rollout phase and one for the training phase, not summed; `gpu_memory_utilization` depends on how much the training process leaves on the GPU during rollout, which is a quantity the training side can design. The level 1 vs level 2 trade-off, the relationship between TP and per-node host backups, and the questions to answer before writing a budget are all in [checkpoint 02 §5](checkpoint-resume-memory/02-host-memory.md#5-gpu-memory-trainingrollout-colocation-is-time-sharing-compute-per-phase).

**Metrics are cumulative.** `vllm:*_total` are cumulative since process start; when one engine runs multiple phases always use `after − before`; why the first phase happens to be correct and how to cross-check is in [ablation method 04](debugging-mismatch/04-ablation-method.md#metrics-are-cumulative).

**DP traffic lands on rank 0.** When the Rust frontend sends only one request at a time, all of them land on rank 0 (engine selection `min_by_key(routing_score)`, ties take the first), and a checker seeing "no events on rank 1" looks like broken routing but there is simply no traffic; use `X-data-parallel-rank` to pin requests to each rank in turn. The two frontends interpret this header differently: Python in hybrid mode validates against the local index `[0, dp_local_size)`, Rust uses the global rank; do not send it to a single-rank server. Python hybrid/external LB + a dense model fails to start (`DPAsyncMPClient` asserts `stats_update_address`, upstream #32252); test Python hybrid with a MoE model.

**Use external_launcher for multi-node.** The `vllm-openai` image has no Ray; use `distributed_executor_backend="external_launcher"`, with srun directly launching N ranks, `tensor_parallel_size × pipeline_parallel_size == WORLD_SIZE`, each rank running the same script and rank 0 writing to disk. Three pitfalls: when counting modules, TP replicas are double-counted (module objects exist on every TP rank; accumulate only the TP rank 0 copy; only the PP direction really splits layers); `cmd | grep | head` makes grep take SIGPIPE with exit code 13 and marks the job FAILED; only `SLURM_LOCALID==0` should install wheels/lay down overlays. Before a large model, first run the same multi-node config with a small model; gating/counting probe bugs are only caught this way. Crashed jobs release GPU memory slower than scheduling, so the launcher needs a GPU-clean barrier ([fault diagnosis 05](debugging-mismatch/05-comm-fault-ladder.md#dirty-gpu-a-crashed-job-releases-gpu-memory-slower-than-scheduling)).

---

## 6. Save one rollout, replay the old-logprob forward

### When to use

For a difference that appears only once across many RL steps, save the whole failing batch and the matching policy as early as possible. Rerunning a full rollout simultaneously changes tokens, request scheduling, lengths, cache, packing, and subsequent training data, making it hard to tell whether an intervention worked.

1. Save the framework batch and policy identity after the real rollout ends and before the old-logprob forward; attach trainer scores after comparison.
2. Verify the save hook executed, sequence/valid token counts match the logs, and the same data can be read back.
3. Use the framework's existing training-only/repeat entry to read the same batch; restore its corresponding weights.
4. Return after old-logprob; forbid optimizer step and regeneration; print/record the skipped phases and weight fingerprints before and after.
5. Run A/A first, then change one layout factor. If A/A itself is unstable, first check randomness, mutable buffers, async streams, and runtime state.

### Minimal matrix

| Variant | Change | Answers |
| --- | --- | --- |
| A | original packing | whether the frozen input reproduces |
| B | exact repeat of A | whether it is run-to-run nondeterministic |
| C | sequence permutation/packing | whether it depends on layout |
| D | micro-batch size or target position | whether it depends on call shape/neighbors |
| E | independent process or equivalent node | whether it depends on initialization/device state |

First run the variants that distinguish the current hypotheses; you need not run the full matrix every time. After reordering, restore by UID and response token position before comparing; a physical row no longer represents the same request. A failure in C does not prove the packing algorithm is wrong; it may also trigger CP, page, tile, or expert boundary errors.

### Keep the trigger condition when shrinking input

First keep the target sample and necessary neighbors, then reduce length/request count. After each reduction, run the criterion to confirm the failure is still there. A CP issue may depend on the remainder from the previous request's end to the rank's start; a Graph issue may depend on short fillers and the capture bucket. Keeping only the failing request often makes the problem disappear.

Freezing rollout and weights does not freeze all runtime state. Training-mode dropout, FP8 amax history, RNG, route replay, caches, and workspace must each be kept or reset according to the failure condition. Do not treat a fresh eval forward as an equivalent substitute for the original old-logprob path.

Keeping the same model resident reduces checkpoint loading cost, but no state may silently leak across variants. First verify A/A for a new process vs a reused process; when native libraries, autotune, or compile configuration are involved, restart the corresponding workers.

### Exiting the replay phase

Only after the original reproducer is turned off, return to normal generation, real backward/update/sync, and the next step's rollout. Training-only success only supports frozen-policy forward conclusions. If the error appears in backward, weight reload, or long-running state, save the corresponding state and use the minimal reproducer for that phase.

The rollout replay here is a debugging procedure; Nemotron's SSD exact replay is an algorithm in the model implementation that recomputes partial chunks to match prefill; do not conflate the two.
