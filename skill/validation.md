# Tiered regression and completion criteria

## Full-model admission: pass proxy train-inference forward first

When backends, compile paths or delivery images are replaced several times, first review against the [candidate evidence ledger](practices/evidence-and-experiments.md#6-candidate-evidence-ledger)
whether old results still apply; an old candidate's pass is kept only as history and is not entered into the current candidate's admission cells.

Before a full train-inference consistency task enters full-model correctness / training experiments, provide the following evidence item by item; missing or not-run items count as not passed, and repeatedly launching the full model does not substitute. In particular, list proxy inference-internal BI / PD and proxy training forward ↔ inference logprob comparison as two independent results: passing the former does not mean the latter passes.

| Gate | Minimum evidence |
| --- | --- |
| Proxy coverage | List of operator types, necessary adjacent-layer interactions, quantization geometry, state and parallel paths; explicitly state uncovered items |
| Inference BI / PD | Proxy batch, prefill / decode and mixed-scheduling comparisons with fixed weights and tokens; proves only inference-internal consistency |
| Training forward ↔ inference | With the same proxy, same policy weights and fixed tokens, verify separately native old-logprob recomputation and the grad-enabled actor forward actually used in training, each against the target production inference path; keep independent evidence per the dual-path contract below |
| Backward and quality | [Independent module gradient comparison](guides/training-model-support/06-backward-validation.md#layer-1-module-level-vjp-comparison), including reference implementation, tolerance and raw results; controlled SFT per task scope, and if not done, state explicitly that quality is not covered |
| Real RL updates | At least two effective updates, θ0 / θ1 same-policy comparisons and a final θ2 sync probe, verification of actual parameter changes and derived state, normal exit; timeline below |
| Performance feasibility | Launch / sync / memory overhead at real shapes and a proxy profile, ruling out bottlenecks such as per-token launches in batchable prefill / training; target Graph path usable |

This is an upgrade gate, not the final completion criterion. Two-step RL does not prove long-term convergence, and a proxy does not prove correctness at full depth, capacity, long context or across machines; after moving to the full model, extend coverage item by item rather than changing all variables at once. Inference-only tasks apply only the relevant gates, state explicitly that training is not accepted, and do not add training work on their own.

Before upgrading, per the [report template](templates/consistency-report.md#verification), record for each gate `pass` / `fail` / `inconclusive` / `not run`, the actual command, artifact paths, and the bound model geometry, weights, source / image, precision and parallel config; accept only evidence from the same candidate as current. For the training forward ↔ inference gate, a bitwise target requires 0 original-dtype mismatches on valid tokens; tasks with an agreed error budget use the predefined standard, and cannot use it to re-report a bitwise failure as a pass. Not executing the actual training forward, an empty mask, a single operator only, or only inference BI / PD passing must not be released to the full model. Keep the first failing sample on the proxy and localize layer by layer; after fixing, rerun the original failing case and the affected matrix.

### Dual-path contract for old-logprob and training forward

- **Native old-logprob recomputation:** use the framework's actual path and its `no_grad` / `forward_only` semantics; do not force autograd on for acceptance. Its result proves only that this scoring path is aligned with rollout.
- **Actual training forward:** with the same frozen θ, same token history and valid mask, run the grad-enabled actor branch actually used in training, capture logprobs before loss transforms, and compare independently against rollout. Keep the target packing, recomputation, operators and parallel config; do not substitute a diagnostic forward. Passing this item also does not replace the later independent backward verification.
- Record for each of the two: model mode, grad mode, output `requires_grad`, actual operator class / backend / source hash, original dtype, non-empty valid token count and mismatches. Diagnostics such as substitute attention / MoE or disabling Graph prove only the corresponding path; do not judge invalid solely from `forward_only`, and do not use a native old-logprob pass to prove the grad-enabled branch passes.
- The comparison happens before the corresponding policy update; it applies to both proxy and full-model θ0 admission. If the formal recipe has multiple minibatch updates within one round, do not require 0 diff between the updated actor and the old-θ rollout, and do not modify the original algorithm to manufacture equality; record the actual parameter version and compare only within the same version.

When adapters are dynamically injected / overlaid, the receipt must also record the actual object's class, source file and key config, and assert that the target callsite is hit; the script directory may be imported before the overlay on `PYTHONPATH`. One Nemotron trace silently loaded an old attention adapter this way; even though the same test output 0 diff, it cannot vouch for the new candidate.

If the user explicitly limits the minimal proxy to fixed-token logprob comparison, do not require it to run GSM8K, SFT or two-step RL; mark these training quality / update gates `not run`, and do not pretend they passed. Only after proxy train-inference forward passes may scope be expanded step by step as agreed, and before any multi-step full RL first complete a zero-update same-policy logprob comparison with **full-model θ0, real workload, real training mode and target production inference mode**, then complete the necessary independent gradient verification and effective update / sync checks. If the user explicitly requires the proxy to verify only forward, the order is likewise full zero-update comparison first, then fill in the training gates; do not launch multi-step RL directly. A fixed 8K / 1K proxy pass cannot replace a full variable-length real workload; when full θ0 fails, stop optimizer updates, keep the failing trajectory and shrink to the first divergence. This scope adjustment does not exempt the proxy forward gate.

### Limited exceptions allowing early full-model runs

- Establishing a BI=0 performance / quality baseline for the complete target model does not approve full-model BI debugging or training.
- When a failure depends on real depth, capacity, cross-layer state or cross-machine topology and cannot be reproduced on a semantics-preserving proxy, targeted full-model diagnosis is allowed; before the experiment, write down why the proxy is insufficient, the hypotheses to distinguish, the minimum scale and the stop condition.
- Expanding scale does not expand resource permissions. Do not use this to occupy extra GPUs, stop other jobs, or bypass the model / runner scope specified by the user.

### Faster iteration and failure fallback

For every change first judge its impact scope; start from the existing minimal counterexample, single operator or fixed replay, then run the relevant proxy gates; only after local evidence passes do a staged full run, rather than repeating all ten steps per commit. Changes to shared arithmetic or gradients must regress the affected forward / backward, weight-sync changes must regress post-update state, and final delivery still runs the full agreed acceptance.

When a full run fails, first save the inputs before the first divergence, weight version, metadata, cache / routing and necessary RNG state, and extract a module or few-layer reproducer. Iterate on it only after confirming the same failure still appears after shrinking; if the failure disappears, do not treat the shrunk test passing as a fix, and keep the necessary full reproduction.

Fixed rollout / module inputs can be cached and replayed in a reliably isolated resident process, avoiding repeated generation and loading of the large model; but the cache must be bound to fingerprints of weights, inputs, config and necessary state, new implementations must still rerun the comparison, and old pass results cannot be carried over. Each expansion of experiment scale should state the existing evidence and the remaining questions the new scale is to verify.

If a candidate fix changes the sampled trajectory or EOS stop point, first save the new token history actually generated by the candidate inference, then replay training old logprob with the same weight version; the old trajectory can still serve as a fixed-input causal diagnostic, but comparing new inference against old response lengths cannot count as an end-to-end pass of the same rollout. Diagnostic runtime overrides, formal source candidates, Graph/no-Graph, forward-only and real `train()` + autograd should be listed as separate evidence. Isolated packages may be used for localization, but final acceptance must return to the delivery code and the default target execution mode.

## Start from the cheapest failure criterion

1. Host metadata/shape arithmetic: minimal pure-function counterexample and boundary parameterization; exhaustive pre/post-fix equivalence when needed.
2. Single operator: same input, real dtype/layout, non-finite detection, repeated/cross-process and Graph.
3. Frozen batch replay: original failing A/C and A/A, full valid-token comparison; probe executed and UID mapping passes.
4. Engine: batch composition, prefill/decode, chunk boundaries, cache/Graph, actual parallel scope.
5. Real training (when in task scope): non-degenerate reward/advantage, finite and meaningful non-zero gradients, optimizer step, actual parameter change, sync, post-update frozen-token alignment.
6. Production acceptance: complete model, agreed multi-step, update/resume lifecycle, quality evaluation and paired performance.

Not every task requires all tiers. Report the tiers completed; when the user only asks for diagnosis, do not launch full training on your own. Tests should first extend existing suites; avoid assertions on source strings in place of behavioral tests.

## Preventing false greens

- Zero valid tokens, probe not executed, missing ranks/samples: invalid experiment.
- All-zero scores, all-false mask, the same sample repeated: check the producer and coverage; a passing comparison alone is not enough.
- Zero probability/KL/logPPL: still check the original logprob bytes.
- Zero gradient or all rewards equal: there may be no effective update; check advantage, update count and parameter differences.
- Parameters updated but rollout policy unchanged: check sync, alias, scales, derived weights and Graph.
- Small model/short output passes: confirm the complete model's mixed steps and long-context trigger conditions.
- The error magnitude of a shrunk proxy is not a full-model baseline: in the external MiMo case the 12-layer proxy gave `1.866e-12`, full depth with real weights gave `1.143e-02`. The proxy only verifies the execution path.
- In-training mismatch metric is 0: confirm that micro-batch actually executed the trainer forward. The framework may skip forward for zero-advantage batches and return 0; such a 0 is a trivial match, so report statistics only over non-zero-advantage batches.
- A success marker before the process is killed: distinguish a completed numerical experiment from normal exit/long-term stability.
- probs_diff is 0 but many in-group sibling samples are identical: a sampling diversity defect does not affect consistency metrics. Count distinct samples per rollout group; see DP seed collision below.
- Launcher shows `COMPLETED 0:0`: read the recipe's own exit code; when tests report passed, also look at the skipped count, GPU cases may be skipped wholesale by markers or environment gates.

### DP seed collision: identical in-group sibling samples

Criterion: in the rollout dump, count distinct samples per group and the fraction of fully identical sibling pairs; the baseline with different seeds is about 0.1%, all very short answers; clearly above that means a sampling diversity defect, unrelated to consistency metrics. verl `rollout.full_determinism=True` gives the n responses in a group the same seed; do not enable it. Mechanism, fix and numbers are in [rollout-engine §1](guides/rollout-engine.md#1-sampling-seed-collision).

## Unified metric and matrix for forward consistency

Fix the policy, token history, positions, mask and tensor semantics, and record item by item the original dtype / shape, valid token count (greater than 0), non-finite count, bitwise mismatch count, max absolute difference and first failing position. Different dtypes are first judged a contract inconsistency; do not convert and fake a bitwise pass. After aligning by semantics, compare contiguous tensor bytes that preserve bit patterns or same-width integer views; floating-point `==` does not distinguish positive and negative zero. NaN / Inf in a valid logprob is a failure even if the bytes are identical. Comparing sampled tokens cannot be extrapolated to the full vocabulary.

Keep cases and artifacts for each item of the target execution scope, and list unsupported / not-run items explicitly:

| Dimension | Comparison under fixed semantics |
| --- | --- |
| A/A | Repeated execution with the same weights, inputs and state; independent processes when needed |
| BI | Single request, different batch sizes, companion requests, permutation / position and packing |
| PD | Full prefill, chunked prefill and step-by-step decode using the same history, covering mixed scheduling |
| Length and branches | Real lengths, chunk / causal boundaries, and cases triggering sparse selection, MoE and special modules; not only random tokens |
| Execution state | Target Graph capture / replay, cache reuse / reset, weight reload and the actual parallel layout |

A bitwise target requires mismatch=0; error-budget tasks report against the predefined standard separately and cannot be recorded as a bitwise pass. Gradient tolerance is defined independently; see [module VJP](guides/training-model-support/06-backward-validation.md#layer-1-module-level-vjp-comparison) and [SFT comparison](#proxy-sft-three-way-comparison).

## Two-step RL and the final sync probe

Details of real multi-version execution, artifact sealing and failure recovery are in
[weight update 02 policy versions](guides/weight-update/02-policy-versions.md). The timeline below is the admission floor,
not a shortcut where "container started / training script exited 0" counts as a pass.

When the task includes the full train-inference loop, complete at least the following timeline; every actor recomputation uses the same policy and token history as the corresponding rollout:

```text
θ0 → rollout0 → actor(θ0) recompute and compare
   → backward0 / optimizer update → θ1 → sync/reload θ1
θ1 → rollout1 → actor(θ1) recompute and compare
   → backward1 / optimizer update → θ2 → sync/reload θ2
θ2 → fixed-probe rollout and actor(θ2) recompute, compare results after the final reload
```

The final probe fixes prompts / generation settings and saves its token history for actor replay; it does not need a third optimizer update. If outputs are also compared across policies, use the same frozen token history; do not rely on sampling happening to be identical.

At each stage save the policy content identifier, tokens / mask, the non-empty valid comparison count and the raw byte comparison result; for each update save finite loss / gradients, coverage of parameters that should have gradients, master update norm and optimizer step. For sync save the payload / version exported by the actor, the deployed weight / scale / layout / pack actually loaded by inference, the basis for cache refresh or reuse, and evidence that Graph references the current weights.

For bitwise tasks, θ0, θ1 and the final θ2 probe must all have mismatch=0, with two effective updates actually completed and a normal exit. Skipping updates, both sides reading a stale cache, an empty mask, or comparing only probability summaries is not allowed. If only two of the comparisons are completed, do not claim the last reload is covered.

A small master update may not cross a quantization bin, so not every deployed byte must change each step; nor can a version number alone prove sync. Add a controlled sync test that can trigger deployed-value changes, compare the actor-exported payload with the actually loaded state and verify the output. This test isolates the sync path and cannot replace evidence of two real optimizer updates.

## Proxy SFT three-way comparison

Definitions of the three paths, same-batch per-module gradient comparison, controlled variables for curve comparison and localization order are in [Backward validation layer 3](guides/training-model-support/06-backward-validation.md#layer-3-proxy-sft-three-arm-comparison). If module gradient comparison fails, fix the implementation first and do not start SFT; after all three layers pass, accept per the two-step RL and final sync probe above.

## Performance and model quality

When forward arithmetic, precision, scorer or inference path changes, re-review the affected quality evidence and run the same quality evaluation per the evaluation scope the human set at C1 ([ask the human first](README.md#ask-the-human-first)) and the target repository's submission requirements; if the human has not set the scope, ask, and do not pick the question set or pass line yourself. Record model/data, question count, prompt format, generation settings, statistical variation and incomplete parts; quality items not included in this round are marked unverified, and do not add SFT or long-horizon training on your own. Decreasing training loss or a two-step reward change is not proof of no quality regression.

Microbenchmarks, fixed-token parity, generation quality, normal exit and multi-step stability are different dimensions. Nemotron once had four non-zero updates and 139,260 bitwise-equal tokens but ended with a DataLoader error; this supports four-step alignment, not clean shutdown or 50-step stability.

The complete model covers, per the reviewed scope, real layer depth / weights, target context, concurrency, packing, parallelism / hardware, cache / Graph and update path, and reruns the affected correctness gates; do not extrapolate proxy curves into complete-model quality. Final performance is accepted on two items against the budget the human set at C1 ([ask the human first](README.md#ask-the-human-first)); metrics and pairing method are in [paired acceptance](guides/recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance): 8K / 1K Rust Bench at each concurrency, and the rollout-phase per-step generation time of the two-arm short runs of the same formal recipe. The two items are judged separately; Rust Bench meeting its target does not mean rollout meets its target: rollout length distribution, samples per question, weight sync, sleep / wake and prefix cache hits all differ from a fixed-length bench. Training forward / backward and sync times are handled according to whether C1 agreed an upper bound: if agreed, judge against it; if not, only record, and do not add a gate. Scope and stop conditions of the two-step short run are in [general two-step acceptance conventions](#general-two-step-acceptance-conventions).

## Failure classification and delivery

Use `pass`, `fail`, `inconclusive` and a specific reason. Keep source/patch/image/bundle hashes, commands, per-item counts, earliest difference, excluded hypotheses, and before/after results of the fix against the original reproducer. Later cleaned-up code must correspond to re-verification or explicit behavioral-equivalence evidence; old tests cannot vouch for new code.

Passed items are not erased because other items are incomplete, but the final conclusion must be scoped. Use the [report template](templates/consistency-report.md). Formal upstream contributions must also follow the target repository's duplicate checks, human line-by-line review, testing and AI assistance disclosure requirements; historical cases in this repository do not complete the review on the author's behalf.

If delivery includes an image, the gated object must be the final immutable artifact, not another development image plus source mounts. Verify the closure per [runtime-provenance.md](guides/environment-provenance/02-runtime-provenance.md); accept save per [01 §3.2](guides/checkpoint-resume-memory/01-checkpoint-and-resume.md#32-reproduce-and-discriminate), load per [§4](guides/checkpoint-resume-memory/01-checkpoint-and-resume.md#4-load-enough-capacity-does-not-mean-enough-for-the-transient-peak), and first resume per [§6](guides/checkpoint-resume-memory/01-checkpoint-and-resume.md#6-how-to-prove-a-resume-is-correct). Passing environment checks proves only the preconditions for running and does not replace GPU numerical and performance tests.

## Recipe review materials and two-step acceptance conventions

The integration points themselves are in [05-framework-adapter](guides/training-model-support/05-framework-adapter.md); only process conventions are here.

### Recipe and PR delivery contract

First locate the existing integration layer at the chosen version, then decide where to make changes. Megatron's
`experimental/lite/examples/verl/verl_mlite/` provides the verl engine adapter;
the external integration package can be installed normally into the unified image, and removing runtime overlays does not require
reimplementing the engine in the verl repo. Check the actual engine registration, model factory, batch conversion,
loss normalization, optimizer and online weight-sync paths, and list reusable interfaces and concrete gaps.

- **verl PR** by default contains the model's DAPO recipe and the unified image Dockerfile; public interface changes beyond these two
  are first submitted as a gap and proposal and implemented after human review; necessary regression tests may be kept.
- **Megatron PR** contains the model, operators and necessary support in the existing `verl_mlite` integration layer;
  do not move general code that does not belong in Megatron there just to keep the verl PR at two files.
- **vLLM PR** contains the necessary inference-side changes to model, kernels, state and weight loading.
- Per the [general delivery goal](practices/code-and-delivery.md#8-three-repository-delivery-from-experimental-implementation-to-three-prs-and-one-image), submit complete PRs to the three main repos,
  including the required model adaptation, not stacked on another unmerged feature PR; pin cross-repository dependencies
  and state the build order. Train-inference consistency uses mLite (model, operators and runtime separated, reusing the inference forward, independently integrating
  backward / STE, following the `verl_mlite` interface and the MCore distributed optimizer wrapper); the training baseline follows
  the default MCore path of the existing verl recipe; do not merge the two into one training scheme. mLite does not inherently guarantee
  0 diff or better performance; parallelism, recomputation, checkpoint and online sync must still be verified against the actual recipe.

### DAPO reference and review materials

[verl #7688](https://github.com/verl-project/verl/pull/7688), at review version
`640c5f0a317307e99636924804c40624242fa19b`, uses the integration package inside Megatron via `model_engine=mlite`
and `pkg://verl_mlite.config`, so the verl diff
only needs the recipe and Dockerfile. Follow its real `verl.trainer.main_ppo` execution chain;
do not assume its config keys, DeepSeek precision / topology, mounts or overlays suit the current model.

The reference case's model, training data and evaluation set are not defaults for new tasks. Each task chooses training data / reward
per the target model and the human agreement, distinguishes the training set from the quality evaluation set, and does not substitute on its own.
Complete config resolution and the necessary proxy smoke first, then submit for human review:

- Executable launch script, unified image digest, each repository's commit and the full resolved config.
- Parameter differences from the reference recipe and why: data / reward, prompt / response length,
  rollout count, batch / microbatch, parallelism, quantization / KV precision, optimizer, sync method.
- Proxy check results, known uncovered items, and the acceptance scope of full zero-update and two real updates.

The human confirms a specific recipe version; the formal full E2E starts after confirmation and proxy admission.
Changes affecting the above workload or numerical / update / sync semantics require re-review; path and logging changes that do not change actual inputs and execution
semantics only need recording. This gate does not block authorized builds and proxy development.

### General two-step acceptance conventions

- **Freeze versions first.** Record the heads of the three delivery PRs, the unified image digest and the full recipe
  config; changes affecting execution results require a version bump and the related verification; old artifacts cannot be used for acceptance.
- **Review the recipe first.** Submit the formal verl recipe adapted to the target model and the agreed dataset,
  its differences from the reference recipe and the resolved config; start formal full acceptance only after the human confirms the specific version.
  Confirming the process or scope does not approve a specific recipe that has not yet been submitted.
  The review checklist also marks which quality, resume and long-horizon checks are required / out of scope this round, and the evidence the target
  repository requires for submission; out of scope is not a pass, and existing required items cannot be hidden because the short run ended.
- **By default run two effective updates first.** Use that formal recipe, shortening only the acceptance run length to two steps;
  do not downgrade data, sequence length, rollout count, batch, optimizer, algorithm, parallelism or backends on your own.
  If the stop step also drives the learning-rate schedule, keep the learning rate / warmup semantics of the formal recipe's first two steps
  rather than compressing the whole schedule into two steps; the concrete implementation is reviewed together with the recipe.
- **Keep the existing correctness gates.** After the proxy real training forward passes, do the full θ0
  comparison, update / online sync to θ1, θ1 comparison, update / sync to θ2 and the final θ2 probe;
  compare same-policy original-dtype logprobs, prove actual parameter updates, and do not pad with fixed-weight replay.
- **Accept the two performance items agreed at C1.** On the frozen version use Rust Bench to measure 8192 input / 1024
  output, ignore EOS, c=1/8/32/64, and from the same short runs take the two arms' rollout-phase per-step generation time,
  accepting against the human-confirmed BI=0 baseline and performance budget (see [performance and model quality](#performance-and-model-quality)).
  Training forward, backward, sync and full RL step times are only recorded if no upper bound was agreed; do not add a gate.
- After the two updates and θ2 probe are complete and the two agreed performance items pass, stop that short run and report;
  overall acceptance is still judged item by item against the review checklist, and missing existing required items are stated as incomplete.
  Do not on your own extend to 100 steps, add an MCore long-horizon comparison or new quality evaluation gates, nor
  extrapolate two-step results into long-horizon training quality having passed.

### E2E comparison targets

Run through the real trainer entry: θ0 rollout → θ0 actor recomputation → effective update →
online sync θ1 → θ1 rollout / actor comparison → update / sync θ2 → θ2 probe;
execution details reuse the [multi-version RL gate](guides/weight-update/02-policy-versions.md#execution-threshold-of-at-least-two-updates).
0 diff means original-dtype logprobs are equal for the same weight version, same token history and valid mask;
it does not require θ0 and θ1 outputs to be unchanged. Each round verify the actual master update, policy version and
completion of online sync; deployed bytes may temporarily stay unchanged after quantization, so do not judge updates by step count or byte changes alone.
Disk replay is kept as a localization tool and does not replace online-sync acceptance of the resident rollout.

Finally re-verify with the image built from the delivery PRs and the reviewed recipe; standalone engine smoke tests, old
image numerical results and historical performance are recorded separately and cannot replace complete evidence from this execution entry.
