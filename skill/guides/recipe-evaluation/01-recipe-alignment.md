# Recipe alignment: align the math, not the names

The goal is to answer "how the reference implementation actually trains, which values are validated, and where we deviate". The items that can be copied directly (optimizer, clip, sample count; with LoRA, the module scope, expert factor layout, rank / alpha / dropout) are listed in [03 baseline hyperparameters](03-day0-baseline-hparams.md); when the model's own validated recipe uses other values, follow it. Beyond these there are no standard hyperparameters: learning rate and batch have only a no-reference starting point in 03 §2, loss reduction and updates per round do not even have a starting point, and all of them must be aligned separately for the model following this page. Reference recipe: for the day-0 route, Miles's recipe for that model; for the train-inference consistency route, [verl#7688](https://github.com/verl-project/verl/pull/7688). Write the alignment result into `cases/<model>/recipe.md`; Miles's cross-model facts are in a separate Miles recipe audit (not included here).

Check order: first freeze data/revision, actually rendered tokens, thinking/template, LoRA coverage and learning rate; then check loss reduction, batch/group, sampling budget and reward parser; finally compare held-out pre/post. Keep training reward and test scores separate, probability diff and logprob diff separate, initial values and post-resume remeasurement separate. Performance comparisons reuse a fixed rollout replay; measure old-logprob and actor update separately, and do not infer backward performance from the former; budget sync time separately for adapter-only and full merge.

## 1. Choose the reference

Pick **one** specific recipe; do not mix A's curves, B's default params and C's learning rate. For Miles: blog publication date, current PR head and in-image commit are three different things; evidence is graded B (blog report) / V (marked validated and has results) / D (launcher default) / W (smoke/SFT), and D and W cannot be promoted to an improvement recipe. Record full SHAs and the image digest; do not decide a fix is included based on image name or file existence.

## 2. Align the math, not the names

Check four things separately; one matching name does not mean the other three match:

1. **GRPO/GSPO**: how advantage and importance ratio are defined. GSPO's sequence-level ratio does not imply the loss must weight responses equally.
2. **Reduction S/T**: response i has T_i valid tokens, token loss is l_it:

   ```text
   response-equal S: (1/N) × Σ_i [(1/T_i) × Σ_t l_it]
   token-equal T:    Σ_i Σ_t l_it / Σ_i T_i
   ```

   For two responses of 1000 and 4000 tokens, under S they have equal total weight, under T the long one has 4×. This is a choice of objective, not an implementation bug. For ordinary non-empty responses in VERL, S corresponds to `seq-mean-token-mean` and T to `token-mean`; fully masked samples, masks after TIS rejection, dynamic batch, multi-LoRA and custom loss need their denominators checked separately. Seeing the reducer return `sum` is not enough; global batch, microbatch and DP/CP scaling come after it. Nor can you average per-rank local means and pass that off as the global token mean.
3. **In-group reward normalization**: whether it divides by group std, epsilon, how all-correct/all-wrong groups are handled. Reward scale (±1 vs 0/1) is essentially equivalent under binary judging + subtract mean and divide by std (except at the epsilon boundary); grading semantics (separator tokens, boxed extraction, tail truncation) are where the differences are; the specific differences between handlers are in §8.2 of the Miles recipe audit (not included here).
4. **Correction/replay**: which train-inference difference TIS, R3 and indexer replay each correct; R3 is mandatory for MoE models ([baseline hyperparameters §1](03-day0-baseline-hparams.md#1-fixed)), the rest follow the reference recipe; using the DAPO dataset does not mean DAPO's dynamic sampling and all its components are on; the KL diagnostic in logs is not a KL penalty.

## 3. Read the final CLI, take validated values

Trace launcher → argument defaults → loss reducer → distributed normalizer to the final resolved config; for unmerged models read the PR head, not main. Take actual arguments, not comments (an example comment says alpha=2×rank, the actual value is 32/32; our fixed alpha 64 has its own basis, see [03 §1.1](03-day0-baseline-hparams.md#11-when-training-with-lora)). Three reminders:

- **Default and validated values of the same entry point can differ by more than an order of magnitude** (example in §7.1 of the Miles recipe audit, not included here: a 40× gap).
- **Do not compare grad_norm across runs**: it varies with trainable targets, rank, parameter sharing, S/T, response mask, DP aggregation and statistic precision; a small value does not mean no learning, a large one is not good. Also look at actual LR/schedule, optimizer steps per round, whether delta-W changes, whether clipping keeps triggering, whether updates are swallowed by storage precision, and whether the same policy changes together in trainer and serving.
- **Frameworks silently change values you wrote; read what the engine actually received.** The resolved config is the framework's input, not the inference engine's input. When verl (main `5b550b0c`) assembles vLLM args it runs `args.update({"enable_expert_parallel": config.expert_parallel_size > 1})`, overriding `engine_kwargs.vllm.enable_expert_parallel=True`; `expert_parallel_size` defaults to 1, so EP is turned off without an error (a 30B-A3B example script shipped in the repo is written this way too). The symptom is every GPU holding all experts and their LoRA buffers, with vLLM reporting no GPU memory for KV cache, which looks like a memory config problem. To enable vLLM EP in verl set `actor_rollout_ref.rollout.expert_parallel_size=<TP×DP>`. After the job starts, check vLLM's actual launch args and logs (in this case `--enable_expert_parallel` and the per-GPU local experts count), not just the yaml.
- **One rollout ≠ one optimizer step**: P×G responses divided by global batch gives the expected number of updates; also check epochs, dynamic batch and drop rules; an x-axis of 30 rollouts cannot be compared with 30 steps without conversion (worked examples per recipe in §8.1 of the Miles recipe audit, not included here). Also report unique prompt count, generated token count, optimizer steps, wall time and GPU count.

## 4. Deviation table

Each model's `cases/<model>/recipe.md` lists: item, reference value, our value, why, which conclusion it affects; with the commit/PR number of the reference source. Do not change LR, rank, target layout, loss, length and TIS in the same round. Shared expert factors are an algorithmic choice (they change gradient aggregation), not just a parameter saving. List LoRA coverage targets by attention, GDN/KDA, shared, routed, head, and record intentionally frozen PLE/vision/indexer, not just the total trainable percentage; a successful transfer does not mean it took effect: check with zero adapter, non-zero adapter, replaced adapter and the same-policy trajectory after a real step, reusing the existing G4 (training weights sync correctly to vLLM) / G5 (two-step GRPO on the reduced-layer model) gates of the day-0 support workflow, without building a new publisher or partial transaction. Do not blindly copy resource policies such as offload, disabling cache or disabling graphs.

## 5. Two-arm rules

One recipe, one image, a switch selects the arm; deviations live only in the switch. See [bi-switch-recipe](bi-switch-recipe.md).

## 6. Minimal record card

Writing it into the experiment's existing manifest/report is enough:

```text
model + base revision + precision route:
image digest + framework/recipe SHA:
training mode + target allowlist + expert factor layout + rank/alpha:
loss reduction + advantage normalization + TIS/R3:
LR/schedule + optimizer/master/adapter dtype:
prompts × samples + global batch + optimizer steps/rollout:
template/thinking + reward handler + response/context limits:
eval dataset/version + sampling + pass metric + budget:
evidence: smoke / update / parity / held-out / resume:
```

Evidence tiers: model supported, smoke passed, update takes effect, numerically sane, held-out improvement and resumable training are different conclusions; none substitutes for another. Name different precision metrics separately (mean abs logprob diff, Schulman k3 KL and log-PPL diff are not cross-compared; the quantization noise floor differs per model, do not hard-code 0.002 for all).
