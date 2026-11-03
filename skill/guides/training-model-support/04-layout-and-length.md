# Distributed layout, length boundaries and training-side acceptance

Record trainer and rollout topologies separately; do not write only the total GPU count. The same mathematical model has different data mappings and reduction paths under TP/CP/EP, padding, packing and collectives; long sequences trigger boundaries short sequences cannot reach; training-side correctness (gradients, optimizer) needs its own criteria and is not vouched for by forward 0 diff.

## Compute resources first, then request

Write the trainer and rollout process groups separately. With four GPUs per node, a single eight-GPU inference DP8/EP8 group usually spans two nodes; two DP4/EP4 replicas are not one EP8 group and do not cover cross-node EP communication. How the framework actually groups ranks is determined by the runtime.

If dense DP does not include CP, the usual dense world formula is `W = TP × PP × CP × DP`. Separately verify `W = ETP × EP × PP × EDP` and the framework's CP mapping on the expert side; **do not multiply EP into the dense world again**. With W=8 and TP1/PP2/CP2, ordinary DP=2; if some log calls DP×CP "DP", it shows 4. Clarify naming and group membership first; do not change GPU counts to match a config string.

Training-side GPU memory budget follows the layout: with PP1 each rank carries activations for all layers, and doubling sequence length doubles activations; if the per-route state saved for EP backward also stores slots that do not belong to this rank, it is amplified by the average number of destination ranks — only storing this rank's routes saves memory. Nemotron's measured numbers are in the [case](../../../cases/nemotron/consistency-case.md#numbers-for-training-side-gpu-memory-length-boundaries-and-optimizer-acceptance). The rollout-side `gpu_memory_utilization` must be set according to the GPU memory the training process still holds during rollout, written into the recipe per training mode; the number table is in [memory budget 02 §5](../checkpoint-resume-memory/02-host-memory.md#5-gpu-memory-trainingrollout-colocation-is-time-sharing-compute-per-phase).

## Packing and dynamic batching

Packing puts several samples into one row with segment boundaries; dynamic batching groups samples into micro-batches by a token
budget, so the number and composition of micro-batches changes per rank and per step. Both only re-lay-out the same per-sample
computation, so per-sample logprobs and the mini-batch gradient must equal the padded path up to rounding. They fail silently: the mean
logprob looks normal while one class of tokens, one boundary or one rank is wrong, so each failure class needs its own check, and each
check must be shown to catch a planted bug. The table is Model A's Megatron validation (a large MoE; 2026-10, THD packing + dynamic batching, R3,
sequence-mean loss, token TIS); it found no bug, and every check caught its plant.

| Class | What goes wrong | Check | Planted bug and what caught it |
|---|---|---|---|
| 1. Packed vs unpacked | Any of the below, seen in aggregate | Logprobs on loss-masked tokens against an unpacked reference, within 3× the noise floor (unpacked repeats, and a previously validated fixed packing against unpacked) | — |
| 2. Loss normalization | Loss or gradient scale depends on how many micro-batches a rank ran | Total loss and gradient cosine and norm ratio against the reference | Drop the micro-batch-count correction: norm ratio 0.52 |
| 3. Cross-sequence leakage | Attention, a sparse indexer's top-k, chunked attention or R3 replay rows cross a segment boundary | Perturb only the first sample of each pack; the others must be bitwise unchanged in a deterministic diagnostic run (13219 tokens, max diff 0) | Merge attention segments: mae 0.45 on the other samples; reverse R3 rows inside a sequence: mae 1.34 |
| 4. Position reset | Positions do not restart per segment | Reverse the sample order inside each pack (another legal packing); results must match | Positions not reset: 0.20, caught only by this reorder check in the deterministic run, because with relative RoPE the bug acted only through SSMax's log(position) and stayed inside the noise of check 1 |
| 5. Label shift at boundaries | First or last response token of a sample scored against the wrong target | First and last response token of every sample against the reference | Shift labels by one more: mae 2.70 |
| 6. Reorder alignment | After sorting by length, advantages, rollout logprobs, masks or R3 rows no longer belong to their sample | Returned logprob order; recompute the loss on CPU with the original-order inputs and compare to the reported loss | Reverse outputs, misalign advantages: recompute diff 0.97, norm ratio 52 |
| 7. Cross-rank | Unequal micro-batch counts hang FSDP or EP collectives; padding micro-batches add gradient | Micro-batch counts equalized across ranks; padding micro-batches contribute zero | Padding micro-batch duplicates the last real one: norm ratio 2.0 |
| 8. Kernel coverage | One-token dummy rows and padding micro-batches hit special kernels with shapes they never saw | Every arm runs through QAT, FP8-KV simulation and the indexer without error or NaN | — |

Rules:

- Run the checks through the production entry (the worker's `forward_backward` with the production config), not a re-implemented loop.
  Inputs span the model's thresholds (sparse top-k size, attention chunk size), include a fully masked and a partly masked sample, and
  give ranks unequal loads so some ranks get padding micro-batches.
- Bitwise checks (classes 3, 4) run in a deterministic diagnostic mode; everything else is judged against the noise floor of the
  production, non-deterministic numerics (the day-0 boundaries of the day-0 support workflow).
- Pass segment boundaries (`cu_seqlens`) into attention and the indexer and compute per segment, as Megatron's THD path does. Inferring
  boundaries from position resets and building a block-diagonal mask over the whole packed row keeps a cost quadratic in the packed
  length and is a new mechanism to validate; Model A's first FSDP attempt did this and was slower at small budgets.
- The token budget must be at least the longest sample (prompt + response cap): a 3072 budget against 4096-token responses cannot hold
  the samples it is meant to pack. Under expert parallelism a rank's expert compute receives rows from all ranks, so the memory of a
  budget is set by the global worst case: Model A on Megatron at EP32 ran step 1 with a 6144 budget and ran out of memory in step 2, 4096 held.
- After all classes pass, recheck θ0 on the full model, measure the speedup at the production rank count
  ([performance §3](../performance-profiling.md#measure-at-the-production-scale)), and swap it in at a checkpoint boundary.

### Boundaries under packing and CP

For the target UID, reconstruct the global request `[seq_start, seq_end)`, the current rank's `[rank_start, rank_end)`, local rows, cu_seqlens, positions, and all compression/reordering mappings.

The intersection is `[max(seq_start, rank_start), min(seq_end, rank_end))`. Only start < end contributes local rows; an overlap window must not let a completely disjoint request get this rank's output. Host cumulative boundaries must equal the number of rows the device compactor actually produced.

Prioritize coverage of before/after/exactly-equal boundaries, empty intersections, cross-rank cases, and remainders of compression ratio and tile. When the window constant is 8, do not test only multiples of 8; check 1…7. The minimal counterexample should be hand-computable, then parameterize/exhaust nearby legal configs.

Stateful modules (conv, SSM scan, PLE) must consume cu_seqlens; passing `cu_seqlens` does not mean the fallback consumed it, see [02 §4.4](02-reproduce-forward.md#44-packing-boundaries-for-mamba--stateful-modules).

## EP and collectives

Record expert mapping, tokens per rank, padding capacity, route order, shared/routed expert combine and quantization scales. Empty ranks still have to participate in collectives, in backward too: autograd only runs the backward of a collective whose output the loss depends on. In Model A's Megatron port, a rank that received zero routed tokens in some layer returned an empty expert output that did not depend on the dispatched tokens and routing probabilities, so autograd skipped the all-to-all backward on that rank and the other EP ranks hung in it (flight recorder: sequence numbers misaligned in the EP group). Tests with identical inputs on every rank never reach this path; give ranks different data. The fix connects the empty output to its inputs with a zero coefficient. Reused buffers that are not fully written can leave residual data. Initialize outputs with different sentinels to detect missed writes; do not zero-fill both sides and get a false green.

Fixing NCCL algorithm/channels may eliminate reduction differences on some stack, and may also significantly reduce throughput; it is a config choice that needs validation, not a universal env var list for all models. Validate each legal topology separately before relaxing configs or developing new deterministic collectives. Rules for the reduction order itself are in the next section.

MCore dev's grad-ready count assertion: the first batch records each parameter's grad hook count as golden, and afterwards a mismatched count raises an error when reduce is issued; micro-batches containing short sequences can make some parameter's hook count vary with data (Nemotron r10 hit it once at step 2, not reproduced in r12). When dp×cp=1 and expert_dp=1, turning off `overlap_grad_reduce` avoids it without changing numerics (with DP=1 the reduce has no actual communication); root cause not located. With EP4 + dist_opt and dp>1, gradient reduction over multiple micro-batches triggers early (reported).

## Bitwise consistency after changing the parallel layout

**Principle.** Floating-point addition is not associative. The parallel layout determines how reductions are grouped, in what order they add, and where they round: TP splits one matmul into per-rank partial sums, EP distributes a token's top-k expert outputs across ranks and then combines them. Different layouts compute different summation expressions; results differ in the last few bits, and after dozens of layers and sampling they become different tokens. So "bitwise consistent" only holds for the layout that was validated; if rollout or training changes any of TP/EP/DP, it must be re-proven.

**Rules.**

1. The reduction tree is part of the numerical contract: for each level, write down the dtype it adds in, the order, and where it rounds. Every legal rollout layout and the training side follow the same tree.
2. A degenerate layout (TP=1, EP=1) does not automatically equal a multi-rank layout. It must simulate the same tree: first group partial sums by rank, round at the same positions, then combine in the same order.
3. After changing layout, first compare raw bytes of per-token logprobs on fixed inputs, and only go to training once it passes. If `ppo_kl`/probs_diff is non-zero during training, localize and fix it; do not fall back to the old layout to work around it.
4. Localization: dump layer by layer, find the first layer whose output differs, confirm its inputs (for MoE also router logits) are bitwise identical; then the problem is in that layer's reduction, see [debugging 02](../debugging-mismatch/02-bisect.md).
5. Simulating reduction order has overhead; measure it, and report it separately from the gain of the layout change.

**Examples.**

| Model | Layout change | Where it differs | Fix and result |
|---|---|---|---|
| Nemotron-H (W4A16 MoE, BI=1) | Rollout changed from one DP4EP4 instance per node to one TP1 instance per GPU | EP4: each rank sums its share of expert outputs in FP32, rounds to BF16, and the per-rank partial sums are combined in FP32 as `((s0+s1)+(s2+s3))+(s4+s5)`. EP1: the six expert outputs are added sequentially in FP32 in one go, with no intermediate rounding. Max logprob diff 0.4 on the same prompt; greedy tokens diverge at positions 10–129 | Without EP, BI mode computes all experts in one launch and then reduces in EP4's order (vLLM `661cde6833`, not in the image). After the fix, both layouts' outputs are bitwise identical to the training side, `ppo_kl` is all 0 over 58 training steps; multi-turn replay 404 s vs 485 s, reduction overhead about 2.5% |
| DeepSeek V4.1 | Training TP=1, inference TP>1 | Reduction order of per-TP-rank partial sums | Both sides follow the same fixed reduction tree, including the TP=1 side (`VLLM_TP_INVARIANT_LEAVES`) |

## Async communication and lifecycle

Check producer/consumer streams, events, buffer owners and reuse timing. Passing after inserting a global synchronize only shows the issue is timing-related; do not adopt global sync as the final optimization. Localize the missing dependency and accept it on the same path.

## Long-sequence boundary points

Passing on short sequences does not mean passing on long ones. The points below must be tested specifically, and at real lengths:

| Boundary | Fact | How to test |
|---|---|---|
| KV page | Training-side attention replicates vLLM's KV page size; page boundaries are an independent numerical boundary | Sparse dY gradients across a page boundary |
| Fixed split attention | Split count and keys per split are fixed; the resulting `MAX_SEQ_LEN` is a length upper bound, existing for invariance, not a performance knob | Exceeding the bound must error, not silently truncate; below the bound take two real lengths, each with random/zero/sparse dY |
| Packing prepend | Prepending a short sequence shifts all row offsets of the target sequence | Two prepend groups compared bitwise against no prepend |
| Micro-batch fits the longest sequence | verl requires every micro-batch to fit the longest sequence (`ppo_max_token_len_per_gpu`, Nemotron 16384 vs DS4 4096; Nemotron's training side has no CP) | Run one OOM probe at the production length first; do not extrapolate from 8K |
| Embedding backward | PyTorch CUDA `embedding_dense_backward` accumulates at reduced precision with ≤3072 indices, triggered when a short sequence forms its own micro-batch | Tolerance tiered by length |
| CP compressed boundary | See [CP compressed boundary](../../../cases/deepseek-v4-flash/cp-compressed-boundary.md) | Before/after/exactly at the boundary |

Nemotron's concrete values (page size, split config, the two lengths, prepend length) are in the [case](../../../cases/nemotron/consistency-case.md#numbers-for-training-side-gpu-memory-length-boundaries-and-optimizer-acceptance).

<a id="the-d-term-in-attention-backward"></a>
## The D term in attention backward

After the FP8 attention forward rounds P to FP8, the visible O = fp8(P)·V does not equal softmax(QKᵀ)·V. If backward's `D = rowsum(dO⊙O)` uses the visible O while dS uses the recomputed P, then `rowsum(dS_i) ≠ 0`, and dQ gains an extra bias term that accumulates with the same sign per row (Nemotron about 3% on dQ/dK, v2 criterion FAIL at steps 150/250; after the fix 0.006–0.009, PASS). Mainstream FP8 attention (TE fp8_dpa, SageBwd) uses the visible O directly, so this is not the default practice and is decided by a human with full knowledge; the full survey is in [fp8-attn-bwd-survey](../../../cases/nemotron/reports/fp8-attn-bwd-survey.md).

Four things to check when wiring training backward:

1. **FA4 interface signature**: `_flash_attn_bwd` only uses the passed `out` to compute D in preprocess and has no parameter to pass D directly; the public interface errors out directly under `requires_grad` (forward-only), so the actor bypasses it and calls `_flash_attn_fwd/_bwd`. The fix is to recompute O and LSE in BF16 from dequantized Q/K/V at the start of backward and pass them to bwd (single-layer backward +4–15%, end-to-end <0.05%).
2. **varlen LSE layout**: whether LSE is ln or log2, arranged per head or per token, and whether the softmax scale is already folded in; if they don't match, recompute in backward with the training kernel.
3. **GQA 32/2**: 32 Q heads to 2 KV heads; dK/dV must be summed within the Q head group, and the summation order must be fixed to do A/A.
4. **When saved Q/K/V are released under recompute=full**: when the forward saves only dequantized Q/K/V and not O/LSE, recompute must read the same state; if released too early, backward recompute uses different values.

## Microbatch split and gradient differences

Splitting the same batch into different microbatches gives a relative gradient difference of about 2e-3, coming from loss normalization, accumulation order and reduced-precision embedding backward with ≤3072 indices. This is not a bug, but it means:

- The criterion's denominator should be `max(B16 reference floor, the implementation-under-test's own noise floor)`; do not use the FP32 ideal as the floor.
- Both arms of a comparison must use the same microbatch split; if the split changes, rerun A/A first.
- grad_norm cannot be compared directly between runs with different microbatch counts.

## Comparison matrix

With fixed policy/tokens, test in turn: original layout, reversed order, target placed first/last, changed micro-batch, CP-boundary-triggering samples; reconstruct each item by UID/global token coordinates. Real TP/EP changes may change legal arithmetic reduction; whether cross-topology bitwise equality is required is determined by the experiment contract.

Do not treat safe packing as having fixed a metadata bug: it may only lower the trigger probability. The original triggering layout and nearby boundaries should all pass after the fix. See [CP compressed boundary](../../../cases/deepseek-v4-flash/cp-compressed-boundary.md).

<a id="training-side-correctness-acceptance-gradient-criteria-v2-and-optauth"></a>
## Training-side correctness acceptance: gradient comparison criterion v2 and optauth

Forward 0 diff cannot prove backward and optimizer correctness. The training side has two independent acceptance gates; Nemotron's material is in [consistency-case step 7](../../../cases/nemotron/consistency-case.md#step-7-how-to-make-the-gradient-comparison-discriminating) and [skill-admission](../../../cases/nemotron/reports/skill-admission.md).

### Gradient comparison criterion v2

- The criterion (tolerance, denominator, threshold) is frozen before the run with its sha recorded; it is not changed after the run.
- First do an injected-error discrimination check: gradients ×1.25 and ×1.05 must both be judged FAIL; a criterion that cannot distinguish ×1.05 is useless.
- Accept multiple training snapshots (steps 0/50/150/250) separately; passing at step 0 does not mean later steps pass (the D-term issue only appeared at step 150).
- On failure, inject reference gradients block by block to bisect and find the first module out of bounds.
- v2 has been extended to FSDP2 (EP4 + FSDP2, all four snapshots PASS, worst-class median 1.08–1.39, floors all from B16).
- Tool bug: with `overlap_grad_reduce=False`, EP4 gradients were reduced one extra time before dump; fixed when porting the harness, which shows the gradient export path also needs A/A (‖g‖ bitwise equal, local shard and dump bitwise identical, replica diff 0/0).
- Cross-process bitwise reproducibility of backward is not required; but occasional non-bitwise results at the first step have a specific cause (Inductor's deterministic config is thread-local, and the autograd thread's first compilation chooses between 512/1024 blocking by timing); the fix is in [appendix determinism](appendix-bi-kernels.md#determinism-only-takes-effect-on-the-main-thread).

### Optimizer acceptance (optauth)

Verify that the optimizer consumes the clipped `main_grad`, the update lands on master, deployed bytes = requantize(master), and the bytes the consumer receives match the actor's. The criterion is frozen before the run; failed items must be classified as either "criterion modeling gap" (e.g. checks for replicated parameters not applicable when none exist, or Adam's equal-magnitude first-step update invalidating γ-type criteria while the per-coordinate error bound still holds) or implementation error — the former gets a written acceptance rationale, the latter gets fixed. Acceptance must cover every parameter group type: a probe that only looks at the first step and has no FP32 parameter with wd_mult=0 cannot see bugs like weight decay being applied to the wd_mult=0 group. Nemotron's results (pass/fail counts, A1/A6, the wd_mult bug) are in the [case](../../../cases/nemotron/consistency-case.md#numbers-for-training-side-gpu-memory-length-boundaries-and-optimizer-acceptance).

### Facts outside the two criteria

- When the base is frozen and only LoRA is trained, base modules must still propagate input gradients ([02 §4.2](02-reproduce-forward.md#42-a-frozen-base-still-needs-to-propagate-input-gradients)).
- The two-step RL loop (θ0→θ1→θ2 probe) accepts forward and sync, not gradient precision; the SFT three-arm comparison accepts training effectiveness. Report the three separately, see [validation](../../validation.md).
