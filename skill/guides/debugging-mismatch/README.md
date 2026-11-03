# Fault localization and mismatch bisect

When the training side and rollout disagree, or some stage of a multi-node run fails, this is how to find the first divergence point with the fewest observations and produce a root cause that can be reviewed — without dumping the whole model, and without changing precision first. Performance attribution is in [Performance](../performance-profiling.md); OOM and staging peaks are in [Checkpoint, resume and memory budget](../checkpoint-resume-memory/README.md).

## Symptom → first step

| Symptom | Do first |
|---|---|
| Not bitwise at θ0 already | [01 Isolate the side](01-isolate-the-side.md): four comparison groups; which two are equal rules out which side |
| Only some sequences are wrong | 01 same-prefix siblings: two samples of the same prompt must be bitwise identical on the shared prefix, otherwise it is the engine side; when the whole tail differs from the middle of the answer onward, first check whether the token before the divergence is a special id ([01 per-sequence shape](01-isolate-the-side.md#observe-the-shape-per-sequence)) |
| Wrong from a certain token position onward | [02 Bisect](02-bisect.md): single-request prefill, bisect along one axis of layer/module/position |
| Wrong only at step N | [Weight update](../weight-update/README.md) |
| Difference grows slowly during training (θ0 roughly right) | [07 Gap grows during training](07-drift-during-training.md); to pin it to one side see [01](01-isolate-the-side.md) |
| Is this difference noise or a defect; a reduced-layer slice shows nats of difference | [03 noise or defect](03-classify-and-fix.md#noise-or-defect-read-the-pattern-then-the-size) |
| Same operator inputs still differ | [Numerical contracts](../../numerical-contracts.md), [BI kernel appendix](../training-model-support/appendix-bi-kernels.md) |
| Multi-node fails to start / NCCL errors | [05 Communication fault ladder](05-comm-fault-ladder.md) |
| Not sure what to save to reproduce | [06 Save and compare protocol](06-artifacts.md) |

## Key rules

1. Find the earliest divergence, not the largest layer; later differences are all propagated from earlier ones.
2. Isolate the side before bisecting; classify before changing precision: input mismatch → weight mismatch → discrete path mismatch → kernel numerics → precision contract, with precision last.
3. Align probes by semantics (whether LSE is ln or log2, before or after norm, before or after the residual add); misaligned probes produce false divergences.
4. Stop bisecting when you reach a discrete decision (top-k, index); fix the routing before continuing. For MoE models compare with R3 before attributing a gap to precision. Classify a difference by its pattern before its size, and size small differences on the full model, not on a reduced-layer slice ([03 noise or defect](03-classify-and-fix.md#noise-or-defect-read-the-pattern-then-the-size)).
5. A single round does not count; hidden coupling (changing one switch also changes scheduling) cannot be attributed to the switch; the target path must actually be exercised, otherwise it is VACUOUS.
6. Two-node tests must prove that cross-node transfer happened; successful initialization does not mean reasonable bandwidth; a single `max_abs=0` does not prove every batch is consistent.
7. Things that look like precision but are something else: FlashInfer KV scale cache (update chain), determinism only taking effect on the main thread (switch scope).
8. When the gap grows during training, first split by token and look at the tail; overall KL and sequence-level statistics average the growth away ([07](07-drift-during-training.md)).

## Files

| File | Content |
|---|---|
| [01-isolate-the-side.md](01-isolate-the-side.md) | Per-sequence shape, what Miles already has, the cheapest comparisons (prefill / decode trace), metric definitions (logprob_mae / probability_mae / k3), four comparison groups, same-prefix siblings, validity of comparisons |
| [02-bisect.md](02-bisect.md) | Choosing a search axis, narrowing step by step with single-request prefill, per-module dumps to find the earliest divergence, semantically aligned probes, minimal but trustworthy observations, recommended logs, bisect limits, reviewable root cause, stopping conditions |
| [03-classify-and-fix.md](03-classify-and-fix.md) | Classification table (correctness / floating-point noise / correct quantization / not located) and handling order; two examples from this week: FlashInfer KV scale cache, determinism only on the main thread |
| [04-ablation-method.md](04-ablation-method.md) | A single round does not count, hidden coupling, speed switches change scheduling, the target path is actually exercised; per-segment isolation A0/A/E/B adding back to the total gap; py-spy down to the process; `vllm:*_total` is cumulative |
| [05-comm-fault-ladder.md](05-comm-fault-ladder.md) | Four things: topology / implementation / mode / build, six-level isolation ladder, historical symptoms, same-fabric fairness, dirty GPU, dual-dispatch order assertion; world size formula in [training side 04](../training-model-support/04-layout-and-length.md) |
| [06-artifacts.md](06-artifacts.md) | Archive per step, replayable raw data, per-token logprob comparison, per-layer trace, key points for writing replay comparisons; kit script index |
| [07-drift-during-training.md](07-drift-during-training.md) | θ0 right, grows during training: monitoring window and token splitting, growth in the tail, comparison of geometric-mean masking / token masking / token-level TIS, handling fixed gaps and growing gaps separately, trend-based stopping conditions |

GPU OOM and staging peaks have moved to [Checkpoint guide 04](../checkpoint-resume-memory/04-gpu-oom-and-staging.md).
