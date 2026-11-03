# Train–inference consistency report

## Outcome and scope

State pass/fail/inconclusive for the requested scope. Record model/layers, independent samples, valid tokens, dtype, topology, Graph/cache paths, update count, and untested conditions.

Link the human-approved acceptance checklist: required versus out-of-scope checks, recipe revision, two-update stopping condition, quality/resume coverage and target-repository submission requirements. Report out-of-scope checks as unverified, not passed; finishing the short run does not complete missing required checks.

## Immutable experiment

Record source commits and dirty patch hashes, loaded libraries, image digest, GPU, checkpoint/policy identity, input bundle digest, score semantics, resolved config, commands, resource budget and artifacts.

Separate source, installed wheel, loaded native libraries and JIT specializations. State whether the measured runtime and final delivered image are identical; if not, identify the missing verification instead of inheriting the result.

For recipe work, record training context, rollout length distribution, concurrency, packing/token batch, hardware/interconnect and master/deployment/activation/KV/accumulation/gradient dtypes. Include quantization granularity, scale/update rules, and quality/throughput/memory budgets defined before testing. Unknown budgets remain unaccepted, even when measurements are available.

## Delivery manifest

For the three-PR / one-image RL delivery, fill this from the final clean build, not the development overlays. Use full SHAs and immutable image identity; mark missing evidence as not run rather than inheriting prototype results.

| Deliverable | Review / source identity | Build or acceptance evidence |
| --- | --- | --- |
| vLLM PR | URL; target owner/branch/base SHA; head SHA | Build command and wheel/native/JIT inputs |
| Megatron PR | URL; target owner/branch/base SHA; head SHA | Installed model / primitive / adapter identity |
| verl PR | URL; target owner/branch/base SHA; head SHA | Recipe and image build script paths |
| Unified image | Base and final platform digests; dependency pins | Build log; actual trainer and rollout worker imports/libraries; no development source mounts |
| Approved recipe | Script SHA, resolved config and human approval | Changes from reference recipe; dataset and topology |
| Final validation | Commands and artifact IDs tied to the above image | Proxy / full zero-update / θ0–θ2 / quality / performance status, each reported separately |

Confirm each PR is complete against the agreed base, not stacked on an unmerged feature branch. Explain every required runtime file outside those PRs as a pinned existing dependency; an uncommitted production patch is a delivery gap. Cross-repository dependency pins are expected, not stacked PRs. A changed base, build input or recipe requires reassessing the affected evidence.

## Proxy coverage and gradient contract (when training is in scope)

Map each full-model feature to its proxy layer/module and test artifact; identify omitted depth, geometry, topology and execution paths. For each module, record visible forward semantics, trainable variables, master versus dequantized surrogate values, STE/clipping/scale rules, fixed discrete choices and differentiable scores, VJP accumulation/collectives, and saved/recomputed state with its memory cost.

Link the independent reference and explain why it does not call the tested backward. Record shape/dtype/layout and random, zero, sparse and captured upstream-gradient coverage; per-input/parameter missing/nonfinite gradients, max absolute error, relative L2, norms and cosine where defined. Include predefined thresholds and rationale, original forward-byte comparison and one-step master-parameter deltas.

## Symptom and localization

Record per-request/token differences, A/A and A/C results, side-isolation evidence, last equal/first observed unequal semantic boundary, identical-input operator comparison, competing explanations and counterexamples.

## Cause and change

Explain the violated arithmetic/metadata/state contract and smallest triggering input. Identify actual repair versus diagnostic interventions. Link code changes and preserved regression. Use unknown if evidence does not establish a cause.

## Verification

Before escalating to full-model BI consistency or training, record the proxy upgrade decision. Do not merge inference-only and cross-stack parity into one result; a failed, missing, empty, or stale forward comparison blocks escalation. If the user scoped the proxy to fixed-token forward parity, mark SFT/RL as not run and state the later training gates rather than marking them passed.

| Proxy upgrade gate | Status (pass/fail/inconclusive/not run) | Bound command / artifact and valid-token count | Decision |
| --- | --- | --- | --- |
| Proxy feature/operator coverage | | | |
| Proxy inference BI and prefill/decode parity | | | |
| Proxy inference ↔ native old-logprob recomputation | | | |
| Proxy inference ↔ actual grad-enabled actor forward | | | |
| Backward / SFT / RL, if in proxy scope | | | |
| Upgrade to full-model BI work | | | |

For each cross-stack row, follow the [dual-path contract](../validation.md#dual-path-contract-for-old-logprob-and-training-forward): include model geometry, frozen policy/checkpoint identity, source/image digest, dtype, topology, model/grad modes, output requires_grad, actual backends, raw logprob mismatch count and the first mismatch. Preserve native no-grad scoring semantics; its pass does not prove grad-enabled forward parity. Capture comparable scores before updates and loss transforms.

| Gate | Command / artifact | Result and count | Scope limitation |
| --- | --- | --- | --- |
| Original failing replay | | | |
| Operator / metadata regression | | | |
| Engine and production layouts | | | |
| Independent module VJP and one optimizer update | | | |
| Controlled proxy SFT (A/B/C paths) | | | |
| Real backward/update/sync | | | |
| θ0 / θ1 parity and final θ2 synchronization probe | | | |
| Persistent-worker online synchronization and post-update parity | | | |
| Cold-load diagnostic, if in scope (not online-sync evidence) | | | |
| First post-checkpoint-resume export / parity | | | |
| Model quality | | | |
| Clean shutdown / sustained run | | | |
| Final immutable image, without development overlays | | | |

For two-update RL validation, keep a per-policy receipt for θ0, θ1 and θ2: input/mask identity, valid token count, original-byte mismatches, actual trainer-forward execution, update norm/count, exported versus loaded payload/scale/pack state and Graph/cache freshness. Record the controlled synchronization test that crosses quantization bins separately from real optimizer updates. Unchanged quantized bytes alone neither prove nor disprove synchronization.

Record both forward-path results separately at matching frozen policy versions, persistent rollout-worker identity and synchronization acknowledgments before new-policy sampling. Worker exit is checked at final shutdown, not required at each policy transition.

## SFT controls and quality (when training is in scope)

Separate SFT-A (production), SFT-B (same visible forward at the same snapshot plus an independent surrogate), and SFT-C (verified native training). These labels are independent of replay packing variants. Link same-checkpoint/same-batch A/B gradients and optimizer deltas; once trajectories diverge, do not assume later forward values remain identical. A/C differences include forward/quantization changes.

Record initial master and optimizer state, data/order, labels/masks, packing, effective-token normalization, batch/accumulation, learning-rate schedule, clipping, precision/loss scaling, RNG/dropout/recompute and intentional differences. Retain raw train, held-out and fixed-probe loss, gradient norm, update RMS, clipping/nonfinite/skipped-step counts, throughput and baseline/seed variation when available. Use a consistent optimizer-step or effective-token axis; report predefined windows/thresholds, missing controls and whether evidence is only a short smoke run.

## Performance

Record matched work, warm/cold timing, node pool and paired order, repetitions, latency/throughput, kernel/rollout/full-step results and numerical receipts. Separate measured improvements, rejected candidates and estimates.

For the 8K/1K acceptance, link the [approved aggregation protocol](../guides/recipe-evaluation/02-evaluation-and-acceptance.md#pairing-and-computation): formula, repetitions, per-concurrency thresholds and invalid-leg replacement rules. Include all raw pairs, exclusions with reasons and the approved aggregate per c; median paired loss and loss from median throughputs are different statistics. Report the rollout-phase per-step generation time of the two-arm short runs against the budget agreed with the human at C1 as the second acceptance item; other phase timings are diagnostics unless a budget was agreed.

Include training and synchronization time, peak memory, launch count versus token batch, CPU gaps and repeated quantization/allocation where relevant. Distinguish early feasibility profiles from final full-model optimization; normal autoregressive decoding is not evidence of an avoidable per-token prefill/training loop.

State whether this is a BI-only causal comparison or each side's best valid stack. List all differing flags/backends. Include prompt count versus expanded requests, actual input/output lengths, completed output tokens, wall-time denominator and whether the result is fixed-length synthetic or natural-EOS. Do not sum overlapping GPU ranges or additive percentages from separate ablations.

## Evidence corrections and retention

List superseded claims with the correcting artifact and reason. Record which original failure bundles, checkpoints and small receipts must be retained, their storage budget/location, and any missing eval steps. Never substitute a different policy lineage to fill a curve.

## Remaining work

List concrete gaps and the smallest next experiment. State whether production hooks/config were restored. Preserve raw evidence; do not upgrade historical or partial results into current completion.
