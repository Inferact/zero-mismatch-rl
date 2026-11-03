# Evidence and experiment rules

Shared by both workflows: what counts as evidence, how experiments are designed and recorded, how execution states are kept as separate evidence, and whether old evidence remains valid after switching candidates.
Mechanisms and localization methods live in the corresponding guides; only rules are here.

## 1. Evidence labels

PASS / FAIL / INVALID (target path not entered) / VACUOUS (condition had zero hits) / SUPERSEDED / HISTORICAL. These are evidence labels and do not add lifecycle states.

Mapping to the two other vocabularies: the tables in [validation.md](../validation.md) and [consistency-report](../templates/consistency-report.md) have only four columns, `pass / fail / inconclusive / not run`. PASS→`pass`, FAIL→`fail`; INVALID and VACUOUS are recorded as `inconclusive` with the reason in the notes (target path not entered / condition had zero hits); SUPERSEDED and HISTORICAL do not go into the current table and stay only in the [candidate evidence ledger](#6-candidate-evidence-ledger) and the case's historical reports. The day0 card's `verified / not_verified / stale` are lifecycle states describing whether a gate is "still covered by evidence now", not labels of individual pieces of evidence; one SUPERSEDED piece of evidence turns the corresponding gate into `stale`. Successful run, successful weight transfer, train-inference consistency and held-out score improvement are different conclusions and cannot substitute for each other.

## 2. Evidence that must not be confused

- Enabling `VLLM_BATCH_INVARIANT` does not mean bitwise alignment across frameworks; identical text does not mean identical logprobs.
- Zero difference in probability space, zero KL, or identical printed numbers cannot replace a valid logprob comparison in the original dtype.
- Replay with fixed inputs is a localization tool; statistics over old trajectories are not an unbiased KL under the current policy distribution.
- Identical function names, generated code or environment variables do not prove that the actual kernel, launch config and shared libraries are identical.
- Output replacement, forced routing, disabling Graph, bypassing bad packing may be used for causal experiments; they must be labeled as diagnostic interventions together with the execution conditions they change.
- Similar quality scores do not mean identical per-question results; same-question quality, bitwise logprobs and 8K/1K performance are accounted separately. When an acceptance script fails, check the producer receipt first; do not automatically resample or automatically declare a pass.
- Quantization rules are validated on the deployed bytes, not on dequantized weights ([precision pitfall P2](../guides/precision-quant-qat/03-pitfalls.md#p2-validating-rules-on-dequantized-weights-gives-false-positives)). QAT / low-precision training only narrows the numerical gap of the weights; 0 diff only comes from training calling the same kernels as inference; see [how much QAT can narrow the mismatch](../guides/precision-quant-qat/02-qat-and-ste.md#4-how-much-mismatch-qat-can-remove).
- Do not manufacture a pass by changing tokens, weights, model geometry, precision or tolerance. If the comparison baseline really must change, set up a new experiment and re-validate quality.
- Do not carry DS4's EP/MoE config over to dense Qwen; do not manufacture performance gains by lowering the BI=0 config, locally faking BI off, or removing existing correctness patches.
- A comparison between two sides cannot see an error both sides share. Training and rollout fed the same wrong token ids align
  perfectly ([tokenizer round trip](../guides/environment-provenance/01-environment-setup.md#6-do-a-tokenizer-round-trip-check-before-training));
  a grader that misreads the answer format moves both curves together ([grader checks](../guides/recipe-evaluation/02-evaluation-and-acceptance.md#three-grader-checks));
  a zero LoRA-B hides a wrong factor layout and PEFT folding expert LoRA into the weights ([LoRA §4](../guides/lora/README.md#4-minimal-implementation-principles-on-the-fsdp-side));
  a partial update of a fused parameter leaves uninitialized memory that only speculative-decoding acceptance exposes
  ([weight update 03](../guides/weight-update/03-sync-robustness.md#how-to-write-the-integrity-check)). Each of these needs its own
  check against an independent fact (decoded text, a known answer, a non-zero adapter, a completeness count).
- A check that never entered its path is INVALID, not PASS: prove the path ran before reading the result. Model A's first Megatron G5 pair
  "resumed" from a checkpoint path that the container never received (an environment whitelist dropped it), so P2 trained from scratch;
  only its log line `No checkpoint found` showed it. Runtime proofs ported from FSDP to Megatron counted hooks that backend never called
  ([runtime provenance](../guides/environment-provenance/02-runtime-provenance.md#prove-the-numerical-path-ran)). Tests follow the
  production order too: the Megatron port's tests loaded weights after wrapping the model in its BF16 module while production loaded
  them before, so an FP32 scale being cast to BF16 was found only by review; G2 checks gave identical inputs to every rank, so the
  EP rank that receives no tokens never occurred ([layout and length](../guides/training-model-support/04-layout-and-length.md#ep-and-collectives)).
- Source tests passing does not mean the final image passes. Development overlay results belong only to the corresponding experiment; the delivered image digest, installed source and build artifacts must match the acceptance object, and after repackaging the evidence chain must be re-established.

## 3. Inferences no longer used

Conclusions that were historically retracted or narrowed, and how to handle them from now on. Historical numbers and original criteria are kept in the corresponding case; new tasks do not inherit the case's
model, dataset, topology, performance statistics method or pass status.

| Old claim/behavior | Correct handling from now on |
| --- | --- |
| The assistant's summary said "done / 11.8%", so there is E2E data | Find the raw numerator and denominator for the same policy/load/config; if not found, retract the specific claim |
| BI1 is 2.45% faster, so the optimization succeeded | First audit the baseline backend, local BI bypasses, token counts and cache; do not automatically treat it as either fair or bogus |
| The Qwen experiment's 16.35% can directly vouch for the later PR132 image | The experiment stack and the release image closure have not been shown identical; the image gate is still unproven |
| Mounting the whole repository is easiest; if extensions are missing, just rebuild | Following the [overlay alignment rules](../guides/environment-provenance/02-runtime-provenance.md#boundary-between-overlay-and-native-wheel), align the image commit and worktree, check import/mount shadowing and necessary compilation; copying development files is not a delivery overlay |
| single-node DeepEP passed, so RDMA/HCA is fine | A two-node test with actual cross-node communication is required |
| Source pin / same Docker tag means native artifacts are identical | Verify branch, dirty patch, wheel, `.so`, dependencies and digest separately |
| Removing stable sort causes a mismatch, so the current best code has a bug | This is an artificial regression; per the user's request keep the patch and add real load before localizing |
| A fixed split count guarantees consistency across all batches and prefill/decode | Also needs evidence for in-interval reduction, epilogue, scheduling, mask and the full path |
| Checkpoint files are complete, so the first resume step must be consistent | Check invalidation of derived scale/packed weight/cache and the first sync |
| Any zero-diff/KL marker means the whole pipeline passed | Non-empty original-dtype token comparison, actual update, shutdown and long-term stability are accepted separately |
| Four layers once passed a fixed 8K/1K train-inference comparison, so later candidates can go straight to full training | The 3,072 tokens of the old artifact were indeed bitwise identical; but its layer adjacency, source, parallelism and load did not cover the later R19 candidate; proxy admission evidence must be rebuilt for the current candidate |
| Five optimizer steps completed, so train-inference consistency succeeded | Numerical acceptance is independent of execution completing; Nemotron DAPO θ0 already had 159,206 / 174,999 tokens unequal before any update |
| An FP8 single-op direct call matches the actor, so the production quantization module matches too | The compiled `forward_native` of Nemotron layer 16 once had 44 unequal FP8 bytes; compare the real module dispatch; an uncompiled direct call cannot substitute |
| Full backward gradients are finite and non-zero, so the gradient implementation is correct | This is only a coverage / broken-chain smoke test; module VJP with the same input, same dY and an independent reference, and effective optimizer update/sync, must each be accepted separately |
| GSM8K scores on both sides are close, so all outputs are identical and end-to-end performance is fine too | Per-question 2×2 pairing, current quality threshold and fixed-length throughput budget are validated separately; Nemotron differed by only 1 question this round, but first-round high-concurrency performance exceeded budget |
| The host wrapper exited 1, so the sealed model quality experiment must be rerun | First check which layer failed; Nemotron's BI=0 format parser misread the BI=1 `layers` field; use an independent acceptance checker with a recorded hash to re-check the original receipts; do not tamper with the producer or skip worker exit |
| The attention adapter file exists / training started fine, so the target class must have run | Record the actually instantiated class and its import source, and assert at worker startup; Nemotron r2 run stopped before any update because an old file was imported first |
| Rollout-ready acceptance failed, so the whole round must be resampled | First check the sealed worker receipts, RPC nesting structure and exit status; r3 was a control-plane parsing error; sealed numerical evidence must not be discarded |
| The user said clean up, so all local changes can be deleted | Exact targets, keep observability/unique evidence, confirm as requested; do not expand the authorization |

Each new case only adds "contract → minimal counterexample → discriminating experiment → root cause/minimal fix → original reproducer regression → scope/source". Disputed numbers come with the original small receipt and source; do not copy unverified assistant narrative. A patch's current merge status, recommended dependency versions and best-performing config are volatile facts; check the target repository when using them, and do not treat historical PR numbers as permanent guarantees.

## 4. Experiment contract

### First define the objects that must be equal

Write down the checkpoint/adapter/policy version, tokenizer and chat template, raw token IDs, position IDs, attention/response mask, padding/packing, and the logits processing on the training and inference sides.

For response token `y[t]` the comparison is `log p(y[t] | prompt, y[:t])`. Specify the prediction position of the first response token, whether EOS counts, and the exclusion rules for truncation and padding. Do not directly align logits rows from different positions.

Distinguish raw logits, raw logprobs, and logprobs after temperature/repetition penalty/top-k/top-p processing. Record which one the interface actually returns; the same seed does not guarantee the same request scheduling or sampling RNG consumption. The comparison should freeze tokens and not rely on resampling happening to produce the same text.

### Record these targets separately

| Target | Valid evidence | Cannot infer |
| --- | --- | --- |
| Single-side repeatability | A/A with same input and same state | Consistency across batches or across processes |
| batch/layout invariance | Semantically identical samples are identical under different positions, neighbors, packing | Training and inference are identical |
| prefill/decode parity | Two execution paths forced to the same tokens | backward or weight update is correct |
| Cross-stack forward parity | Real training forward against frozen rollout | Full-vocabulary logits are identical, if only sampled tokens are compared |
| Effective training loop | Non-degenerate gradients, parameter changes, sync, post-update parity | Convergence or long-term stability |

When the user requires bitwise, compare bytes in the original dtype and record the mismatching/valid token counts; do not turn it into a pass via float casts, probability-space rounding or loosening tolerance after the fact. If the target is an error budget to begin with, specify metrics and thresholds in advance and measure each side's own noise; there is no need to force general numerical alignment up to bitwise equality.

Finite logprobs are a necessary condition. Empty masks, all-zero placeholders, missed collection, NaN or wrong weight version cannot pass. Zero itself may be legitimate; whether it is a placeholder must be judged together with hook counts, score source and negative controls.

### Recipe and coverage record

When building the train-inference recipe, besides the comparison objects, also pin the following inputs; recover missing items from existing configs first:

| Item | What to record |
| --- | --- |
| Load | Training context length, rollout input/output length distribution, concurrency, packing, training token batch |
| Hardware | GPU architecture/count, GPU memory, interconnect and target TP/CP/EP/PP/DP; trainer and rollout recorded separately |
| Numerics | master/deployed weight, activation, KV, accumulation and gradient dtype; quantization granularity, scale rule and update method |
| Scope | Raw sampled logprob or full-vocabulary logits, applicable hardware/software, Graph/cache/reload paths |
| Budget | Throughput, peak GPU memory, quality and gradient acceptance criteria, number of experiments and stop conditions; when thresholds are unknown, report measurements and do not declare a pass on your own |

Record a measured baseline of the currently available config, then build a strong performance baseline following the main process; exhaustive peak tuning is not required first. The main process's Rust Bench 8192/1024, ignore EOS, c=1–64 is a fixed performance comparison and does not replace the target training context and natural-length load; when the user specifies another load, follow the user's request and clearly report the difference.

The proxy records "full-model feature → proxy location → test case → uncovered items", keeping the hidden/head/expert/quantization geometry and necessary layouts that affect kernel selection. Reuse existing modules and tests, but after changing model, dtype or parallelism re-check the contract; do not just change the config and carry over old acceptance conclusions.

### Environment fingerprint

Keep the code commit, dirty patch hash, untracked runtime files, image digest, GPU architecture, Torch/Triton/CUDA, actual import paths, native `.so` paths and hashes. Record trainer and rollout TP/PP/CP/EP/DP each, dtype/quantization/scale, kernel backend, Graph/compile/cache and autotune config.

Read the resolved config and the actual modules/libraries inside the worker; do not only save launcher settings. Pinning the same weights is not just recording a path: it needs the content version and the update status of necessary derived weights/scales. Before diagnosis, keep immutable known-good and known-bad artifacts.

### Conclusion scope

For every result write down model size, effective sample count, independent token count, lengths and boundaries, process count, GPU/parallel config, and number of updates. Repeating four samples to form batch32 still yields only four independent samples; mixed scheduling not triggered by a small model cannot cover the full model.

Write the machine budget, maximum number of experiments and stop conditions into the experiment record. GPU or system failures produce `inconclusive`; numerical comparison failures produce `fail`; only completing all declared acceptance items produces `pass` for the corresponding scope.

### Experiment discipline

Before comparing two arms, write the nine rules below into the experiment record; if any one is not met, the conclusion can only be marked "pending re-check". The mechanism and numbers for each are at the links; only the rules are listed here.

| Rule | How | Mechanism and example |
| --- | --- | --- |
| One recipe, one image | Both arms use the same image and same recipe, differing only by recipe switches; changes are injected via overlay with hashes recorded | [bi-switch-recipe](../guides/recipe-evaluation/bi-switch-recipe.md) |
| One seed per config | Default one seed per config; add more only when the user explicitly wants variance, and report separately | Multiple seeds spend budget on variance; most problems are systematic |
| Small-load iteration | During iteration use small batches, short outputs, a few steps; only final acceptance runs the recipe's default load | Running the default load during development allows only one or two iterations per day |
| Measure the floor, freeze criteria, plant errors | Measure each row's own floor first (repeat noise, self-perturbation, the trainer's own BF16-vs-FP32 distance); set the tolerance from it and fix it before the gate data; confirm with a planted error larger than that floor that the criterion can return FAIL; do not tune thresholds after a failure | Picking thresholds after the fact can always make results pass; a tolerance below the floor fails correct code: Model A's G0 end-to-end tolerance 0.02 was below the trainer's own BF16-vs-FP32 distance on the slice, and a planted error below BF16 rounding could not fail its row, so three plan revisions re-based the criteria on measured floors before the reruns; [paired acceptance](../guides/recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance) |
| No conclusion from a single e2e run | Measure noise first; on the same hardware and image report the median over multiple pairs, and report single pairs separately | [Performance and profiling](../guides/performance-profiling.md) |
| Same fabric | Before comparing communication bottlenecks confirm both arms use the same transport, and record relevant environment variables | [Communication fault ladder](../guides/debugging-mismatch/05-comm-fault-ladder.md) |
| Same-node comparison | The two paired arms run sequentially on the same node or the same set of nodes; node names go into the record | GPU memory fragmentation, resident processes and interconnect topology differ between nodes |
| JIT contamination check `tl_after_bench=0` | After running, check there were no new compilations after bench; results with recompilation are void | [Performance and profiling](../guides/performance-profiling.md) |
| Take metrics by key name | Take numbers from raw metrics (jsonl / W&B) by full key name, and write the key names in the report; do not take them by column position from log lines | Once the `actor/grad_norm` column was read as entropy, yielding a conclusion in the opposite direction, discovered only when re-checking the raw data |

List confounders before comparing; rules and examples are in [paired acceptance](../guides/recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance).

## 5. Execution state

Process rule: for every comparison write the actual execution state into the experiment record, and keep results from different execution states as separate evidence. Mechanisms and localization methods are in the corresponding guides and are not repeated here.

### Fields that must be recorded

- Resolved Graph mode, capture sizes, whether this step is prefill/decode/mixed, kernel call evidence; Graph capture success does not mean every step uses the same kind of graph, nor that numerics match eager.
- Whether prefix cache hit, whether it was disabled/isolated in the diagnostic phase; passing a comparison by disabling the cache does not mean the native lifecycle is fixed.
- Weight lifecycle stage: cold load, same-weight reload, real update, sync, sleep/wake, Graph replay, reload in a new process; one row each.
- Whether the chunk boundary state and partial input buffer of stateful layers such as Mamba/SSM are saved.

### Keep separate evidence

Eager, the compiled path with Graph disabled, and the production Graph each keep their own artifacts; hooks run during capture and do not re-run during replay, so a capture dump cannot represent replay. Diagnostic runtime overrides, formal source candidates, Graph/no-Graph, forward-only and real `train()` + autograd each get separate evidence. Passing in a new process does not prove the old process's cache/Graph is fixed; stacking two fixes that each passed must go through the gate on its own. After CUDA state corruption, terminate that experiment and do not keep using it to collect numerical evidence.

### Where mechanisms and methods live

| Question | Go to |
|---|---|
| Graph capture vs replay, fixed-composition Graph/eager 2×2 comparison, which bucket filler falls into | [BI kernel appendix](../guides/training-model-support/appendix-bi-kernels.md) (cudagraph row) |
| prefix cache semantics, Mamba chunk boundary state, bitwise comparison with "cache off as the gold standard" | [rollout-engine](../guides/rollout-engine.md) |
| Staged comparison of update and reload, storage address / alias / derived weight refresh | [weight-update 03](../guides/weight-update/03-sync-robustness.md) |
| Localizing asynchronous CUDA errors | [debugging-mismatch 01](../guides/debugging-mismatch/01-isolate-the-side.md) |

## 6. Candidate evidence ledger

Goal: prevent an "old pass" from being promoted into a "new pass".

When an experiment changes model size, quantization dispatch, attention, MoE, Graph or policy weights,
first create one ledger row for **each candidate execution closure**, then decide which results must be rerun. A historical case's
0 diff does not automatically become a guarantee for a new model or new candidate.

### What one evidence row binds

| Field | Record and verify |
| --- | --- |
| Object | Model's original layer numbers and real adjacency, parameter/quantization geometry, tokenizer, checkpoint content, policy version; the proxy's manually stitched boundaries listed separately |
| Execution on both sides | trainer/server image digest, source and overlay hashes, actually imported classes and files, resolved config, native/JIT artifacts |
| Path | `train()` / autograd / whether backward runs, actual dispatch of attention/norm/linear/MoE, compile/Graph/cache, TP/EP/PP and other layouts |
| Workload | prompt/response token content digest, effective mask, length distribution, packing, concurrency, reward/advantage and actually accepted sample count |
| Criterion | Original dtype, finite effective token count, bitwise mismatches, sample count, first difference, independent status of gradient/update/sync/clean exit |
| Artifacts | Inputs, scores, worker/rank receipts, command, time, parent version, hash and `pass` / `fail` / `inconclusive` / `not run` |

`policy_manifest_sha256`, exported `policy_content_id` and the rollout receipt hash
answer three different questions — **manifest version, weight content, run artifact** — and cannot be compared directly with each other. If the exported
content ID is defined as the hash of the export receipt, also follow the receipt to verify shard hashes and
the files actually loaded by the server; matching a single version string is not enough.

### Admission from cheap to expensive

1. **Module.** Pin the real triggering input and non-unit scales / real layout; first do A/A, then do visible forward of the candidate against
   an independent reference; when gradients are needed, check dX/dW/other differentiable quantities separately with the same dY.
   Passing covers only the tested shapes, arithmetic and VJP; it does not prove the whole layer or the training loop.
2. **Minimal real-adjacency proxy.** Cover the actual order of layer types, not just the set of operators. A fixed
   8192 + 1024 tokens can first validate inference BI/PD, then per the [dual-path contract](../validation.md#dual-path-contract-for-old-logprob-and-training-forward)
   separately validate native old-logprob and grad-enabled actor forward; when doing only fixed-token comparison, do not add GSM8K/RL
   to the proxy. Record source closure, actual class, Graph, non-empty effective tokens and original
   dtype mismatches. Whenever any of these changes, re-review whether this step can still be transferred.
3. **Full zero-update.** Use the full model on the **current** serving candidate to generate a real-load rollout, saving
   the new tokens and raw scores; the actor replays with the same policy and same history before the optimizer update.
   First check the single earliest failing sample, then expand to the agreed sample size; if unequal, stop at zero-update, narrow the boundary,
   and do not bypass it via five-step RL or quality evals.
4. **Per-version updates.** For each θk, first compare under the same policy, then check reward/advantage, backward,
   master parameters and optimizer step, export parent chain, then sync online to the resident rollout workers.
   Only after sync completes and load status is verified may θk+1 be sampled; after the final sync, run a separate fixed probe;
   "the last training step exited" does not mean the final sync passed.
5. **Independent acceptance.** Module gradients, full-model gradient coverage, real RL, quality, performance, clean exit and
   the final image each get their own result. Numbers from a previous candidate or another execution mode are kept as historical comparisons, but must not
   be filled into the current candidate's pass cells.

If a fixed-weight serving overlay rejects `reload_weights`, updates or sleep/wake,
then even if BI/PD and 8K/1K paired performance pass, it can only be listed as a serving screening candidate;
after restoring the lifecycle, rerun the affected proxy, zero-update and post-update online sync probes; do not
stitch historical throughput and another candidate's RL pass cells into one "completed" version.

If a stage fails midway, classify the failure as numerical plane, control plane, environment plane or resource plane; keep the sealed
rollout and the original producer hash. A control-plane acceptance bug may be checked against the original artifacts with a separately versioned recovery acceptance checker;
do not quietly modify old receipts or resample and still call it the same trajectory.

### When results must be invalidated and rerun

| Change | Minimum regression; old conclusions that cannot be carried over |
| --- | --- |
| Proxy original layer numbers/adjacency or hidden, head, expert geometry changes | Coverage table, BI/PD, training forward; the old proxy's 0 diff is no longer promotion evidence |
| Quantization algorithm, scale, actual dispatch, JIT/compiler changes | Real module input / quantized bytes, proxy, full zero-update; direct-call unit tests cannot replace the compiled module |
| attention/MoE operators or their training adapters change | Real class and source, same-input operator, related VJP, proxy and full zero-update; a pass on a substitute backend counts only as diagnosis |
| Graph/cache/packing/parallel layout changes | Corresponding BI/PD, replay and state lifecycle; eager passing does not automatically cover FULL Graph |
| optimizer, master→deployed export, reload changes | Gradients and one-step update, payload/shard hash, resident worker online sync and final probe; cold load listed separately; θ0 forward passing is not enough |
| Final image/overlay/native build changes | Verify the closure inside the worker and rerun affected numerics and performance; development mount evidence does not transfer |

If you can prove a change does not affect a given gate, write the dependency closure and reasoning into the ledger to avoid mechanical reruns;
do not judge equivalence based only on file names, PR descriptions, the same tag or "it looks like cleanup".

### Separate cells for performance and quality

For the baseline, first tune BI=0 under the established weight/KV precision and quality requirements, record the real dense GEMM,
MoE expert, all-to-all and attention backends, and freeze the denominator after human audit. BI=1 and BI=0
are paired with the same full load, same nodes and candidate closure; Rust Bench's 8192/1024, ignore
EOS, c=1/8/32/64 is just one set of fixed throughput loads and is not equal to natural DAPO lengths or GSM8K quality, so performance also needs paired rollout phases from short runs of both arms ([paired acceptance](../guides/recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance)).
First check effective tokens / error counts, then compute the performance loss per the [pre-reviewed statistical protocol](../guides/recipe-evaluation/02-evaluation-and-acceptance.md#pairing-and-computation); do not meet the budget by slowing down BI=0,
changing weight precision or altering prefix cache/Graph parameters.

Quality requires the same questions, prompt, chat template, seed, generation limit, grader and no-retry policy;
report per-question pairing and absolute score differences. A previous candidate's quality pass cannot migrate to a new candidate whose numerical path changed.
When the user changes acceptance thresholds, the current agreement governs, but keep the old thresholds and change date to explain historical reports.

## 7. Failure records

Failure records are kept as-is; new results are appended, never overwriting, deleting or editing. Each comparison gets its own W&B project; read back via the API to verify no steps are missing.
For frozen criteria, planted known errors and "no conclusion from a single e2e run", see [§4 experiment discipline](#experiment-discipline).
