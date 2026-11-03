# Bisect and the first divergence

Goal: use minimal experiments to distinguish input/weight errors, operator implementation differences and quantization error, without building a model-specific debugging framework. Metric definitions in [01 Metric definitions](01-isolate-the-side.md#metric-definitions); classification in [03](03-classify-and-fix.md).

## Choose the search axis first

- Version regression: good/bad revisions on the same reproducer, a compatible environment and fixed inputs.
- Config regression: paired intervention on a single factor; use a 2×2 matrix when factors are coupled.
- Data trigger: request subset, length, neighbors, boundary remainders.
- Compute path: layer → attention/MoE/norm → internal operators → metadata/launch/instruction order.

Do not change image, weights, kernel and parallel layout at the same time. For dirty good/bad states save the full patch and run files; recording the same git SHA alone does not mean the same program. Build failures, incompatible dependencies and failure to reproduce are not a bad numerical revision; mark them skip/inconclusive.

## Single-request prefill: narrow step by step, do not dump the whole model directly

1. **Look at existing training logs first.** Check metric definitions, policy version and sample source; samples generated in different rounds can only hint at an anomaly and cannot prove that same-input mismatch grows with training.
2. **Score fixed inputs.** Save the token IDs and position/mask from one rendering; both sides use weights from the same source and the same adapter version, and each repeats scoring to measure its own noise. First rule out mismatched weights, inputs and score semantics ([01](01-isolate-the-side.md)).
3. **Minimal forward.** Single request, one prefill, one output token; keep the minimal prefix or layer set that covers the path relevant to the current divergence and its upstream dependencies, and keep both sides' configs identical. Do no training updates; confirm no cache hit skips the prefill, and distinguish extra calls from warmup/draft/decode. An eager probe does not stand in for graph/MTP acceptance.
4. **Find the earliest divergence.** First look at embedding, block outputs, and logits at the last valid prompt position. Align layout, valid tokens, shard, padding and deferred residual by semantics; do not guess correspondence by module name or tuple index. Expand only the earliest block that exceeds its own noise.
5. **Same-input operator replay.** Give both implementations the same logical inputs and weights to distinguish differences introduced by this operator from upstream propagation. If necessary replace only one operator's output and compare the final distribution; recurrent state must also satisfy the contract.
6. **Return to the original failing path.** After the real fix, re-verify with the real adapter, inputs and run config; passing with fewer layers/eager cannot replace verification of the full model, decode or CUDA Graph/MTP.

## Per-module dump: find the earliest divergence, not the largest layer

1. Build a semantic mapping table: layer number, function, input/output role, module paths on both sides, logical shape. FSDP wrapper names, fused QKV vs separate Q/K/V need not have the same names; combine before comparing when needed.
2. First capture block-boundary inputs/outputs and final logits, with only a few tokens/samples. Isolate the earliest block with significant divergence, then refine into norm, projection, router, expert, residual, etc.
3. Align valid tokens, position, mask, TP/FSDP shard, expert ID, dimension order and padding. Never compare a physical shard directly against the other side's full tensor.
4. Hooks must snapshot rather than keep tensor references that can be overwritten; hooks do not return replacement outputs and do not change computation. Under FSDP, sample at boundaries already materialized in this forward; do not all-gather the whole model or trigger GPU materialization of a huge frozen PLE just for observation; export bounded per layer/per necessary shard and free GPU staging promptly.
5. Record forward/recompute counts so checkpoint backward recomputation does not overwrite the normal forward. Graph replay does not re-execute Python hooks; an eager dump only describes eager and cannot pass for a graph measurement. For graph-specific problems use explicit in-graph observation or snapshots of existing buffers; do not clear the graph/re-capture to mask the problem.
6. If the operator's input already differs, do not directly judge it buggy; in an allowed local test give both sides the same logical inputs to determine whether the operator additionally amplifies the difference. When a fused kernel's internals are not visible, mark the observation boundary and do not guess the root cause.

Minimal artifacts: manifest (run conditions) + bounded tensor dump + comparison table. The table contains: module/role, shape/dtype, finite-value ratio, max abs, relative L2, cosine, first anomaly position. Only after the summary locates a block do you save that block's detailed intermediate tensors; do not by default persist the whole model, all tokens, in full. Artifact format in [06](06-artifacts.md).

## Align probes by semantics

Build a cross-stack boundary table: embedding, residual stream, norm input/output, Q/K/V, compressed rows/index, attention output, router logits/indices, expert inputs/outputs, combine, lm_head and scorer. Record tensor meaning, dtype, global token/shard coordinates and valid mask, rather than pairing only by module name. EP dispatch may reorder, duplicate or pad tokens; align expert tensors by semantic token / expert / shard keys, not by local offset; once expert GEMMs match, still separately compare the communication reduction, the combination of shared and routed experts, and its compiled rounding.

If the saved rollout's first response token already differs, fix that token and recompute once using only the original prompt, without continuing sampling. This probe rules out "the difference comes only from divergence in the subsequent generation trajectory", but cannot automatically attribute it to prompt prefill, attention or some kernel; you still need to compare same policy, position/mask/scorer and per-layer inputs. Nemotron DAPO's [prompt-only failure sample](../../../cases/nemotron/dapo-20260927.md) preserves this boundary.

First compare original-dtype summaries per layer or at a few checkpoints; after finding a difference, dump only the adjacent boundaries and target samples. Keep a clean sibling. Probes record execution counts, distinguishing initial forward, activation recomputation and graph replay. Snapshot async buffers before reuse; do not unexpectedly change the production stream order just to read values, which would hide the problem.

When the difference is unchanged by every knob you turn, first suspect the boundary is too coarse rather than turning more knobs. In an external TPU MiMo case, attention core and o_proj were captured as one, and the difference stayed at `7.8125e-03` across four configs: CP 8/2, segment alignment 128/2048, FP32 accumulation on/off; after splitting them, core was already 0 and the whole difference was in o_proj. By default give adjacent or fused GEMMs one boundary each.

## Minimal but trustworthy observations

- Detach and copy tensors before buffer reuse; save only the necessary semantic boundaries and tokens, and cap total bytes. Do not fully gather frozen weights for diagnosis. Model-specific mappings and experiment artifacts stay in the model's working directory.
- The manifest saves code/image version, checkpoint and adapter identity, input digest, position/mask, dtype/quantization, score semantics, graph/MTP, rank/layout and intervention options. Identical config does not mean identical weights.
- Each call gets its own directory; write the manifest after completion. Missing boundaries, duplicate calls and version/shape incompatibilities must refuse comparison, not silently take the intersection or truncate. Non-finite values fail explicitly; zero-norm metrics are marked undefined.
- Compare max abs, MAE, relative L2, cosine; for final full-vocab logits additionally look at log-softmax, probability total variation and argmax.
- Python hooks do not cover graph replay. State the observation scope explicitly; do not make reload pass by clearing the graph/re-capturing.

## Recommended logs: identity + metric definitions + evidence boundary

Reuse existing logs and JSON artifacts; do not add another logging framework. One manifest per run, one detailed report per comparison; the terminal prints only a summary, not full tensors. Mark unknown fields as unknown; do not later fill them in as verified.

```text
[mismatch] run=<id> step=<n> policy=<digest> input=<sha> phase=<prefill|decode> rank=<n> repeat=<n>
  scope=<fixed-input|independent-rollout> intervention=<none|name>
  metric=<name> value=<number> reference=<run/repeat/boundary>
  self_repeat=<noise> report=<relative-path> classification=<defect|numerical|quantization|unknown>
```

`metric` uses the names in [01 Metric definitions](01-isolate-the-side.md#metric-definitions) (`logprob_mae`, `probability_mae`, KL diagnostic statistic, tensor relative L2, vocab TV); do not mix them up.

Record phase durations separately for load, rollout, score, update, sync, so cold start and training time are not lumped together. The outer console log of a distributed job is only a forwarded copy. When it stops updating, first check the actual TaskRunner/worker raw logs, checkpoint times and completed step count; a single forward/backward stack does not prove a hang. Until training is confirmed to have stopped, do not restart a healthy job because log forwarding broke.

## Limits of bisection

Floating-point differences may vanish in later rounding, masking or routing, so "this layer's output differs" is not guaranteed to be monotonic in depth. First build a suspect interval with a coarse scan, then bisect/fine-scan; finally verify the covered boundaries before the target interval, and confirm responsibility with identical operator inputs. Do not claim O(log L) always finds the first difference in the whole model.

A cut point must carry complete state: residual, MHC mix, SSM/conv state, KV, positions, routing and necessary metadata. A fused upper-layer post and this layer's pre have no natural independent cut point. Prefix replacement after dropping hidden state manufactures new differences.

Only same-input operator A/B can distinguish upstream propagation from an error in this operator. Injecting reference activations, fixing routing or replacing kernels can confirm causality, but these are diagnostic interventions; a production fix can only be accepted after restoring the original path.

## Forming a reviewable root cause

At the end you should have: minimal input, last-identical/earliest-different boundary, same-input comparison, source mechanism, trigger condition, and the minimal change that makes the failure disappear. A config difference is only a lead: in Nemotron, trying two reduction blocks with the same kernel gave identical results, refuting the guess that attributed it solely to the block size difference.

If after three rounds you still only see the final score differ, add new observation points or recheck the sample mapping instead of rerunning in full. Keep failed samples and ruled-out hypotheses so the next round does not repeat the detour.

## Conclusions and stopping conditions

The report only answers: what was observed, what the comparisons proved, what is not yet proven, and one minimal next step. Correctness errors must be fixed; normal numerical differences are judged by noise and task impact; quantization error needs same-input evidence and cannot be attributed simply because inference is low-precision. When improvement is needed, evaluate in order: supported higher-precision rollout, QAT matching serving, real low-precision training; do not use precision changes to mask wrong weights or adapters.

**Replacing with a reference output is diagnosis, not a fix. A drop in local error does not mean a drop in training mismatch; ablation gains of individual operators cannot be added directly.** Once this round's evidence goal is reached, stop expanding, and do not change the training config or start new experiments on the side.
