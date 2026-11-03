# Isolate the side first

Applies to: abnormal logprob differences between FSDP/LoRA training and vLLM rollout, or differences that grow with training. This guide is a method and source audit; no new experiments were run.

Denote the selected-token logprob saved by rollout as μ, and the real trainer old-logprob as π. `μ != π` does not by itself show that vLLM is wrong. The goal is not unconditional bit-exactness, but to rule out correctness errors, measure the numerical noise floor, and explain growth beyond the noise. For quantization mechanisms see the [precision and quantization guide](../precision-quant-qat/README.md); for adapter troubleshooting see the [LoRA guide](../lora/README.md).

## Observe the shape per sequence

Record the number of valid/differing tokens, the first divergence position, the longest run of differences, prompt/response lengths, UID/group, and the packing/rank mapping. Distinguish between: a few sequences differing entirely from the first response token, small drift in all sequences, differences only after crossing a chunk/page, and differences only after an update. The shape is for choosing experiments; it is not proof of a root cause.

**One sequence differs from some position mid-response through the end** (bitwise equal before it): look at the input token right before the first differing position, then check whether "sequences containing that token" and "mismatching sequences" are the same set. In text-only RL the policy can sample special ids such as image placeholders. A MoE checkpoint with a vision tower routes those ids as image tokens in the engine, by token id, with a separate router bias (e.g. `ffn.gate.bias_vl`); this holds with `language_model_only` too and for sampled ids, not just prompt ids. If the trainer uses only the text bias, or skips that bias at load, that position differs; it is written into KV and attention carries it to every later position, giving a contiguous tail. Rules:

- **Take the id range from the exact engine model file in use, never from a sibling model.** DeepSeek V4.1 sentinels are 129264–129268 (`IMAGE_SENTINEL_BASE_ID` in `vllm/models/deepseek_v41/common/mm_preprocess.py`); an earlier diagnosis read the DS4 file and started the range at 129257.
- **Match every per-token-id special path, not only the router bias.** On DS4.1 vLLM routes the range with `bias_vl` inside the top-k kernel (`csrc/moe/topk_softplus_sqrt_kernels.cu`), and Engram also treats 129264 as a dead token (breaks the n-grams, zeroes the gate). The trainer had all three gaps; a forced-input probe showed equal layer-0 router logits but different top-k ids/weights, and a second probe isolated the Engram gap. List every id-dependent branch in the engine model (router bias, hash/Engram lookups, embedding overrides) and mirror each.
- **Export and sync the frozen bias on every update.** A frozen param that is not exported is not in the synced set; a sleep level 2 reload then drops it, so rollout, trainer and cold-start prefill all differ. On DS4.1 the rollout `bias_vl` fingerprint matched the release only up to the first sleep. Fix: export `bias_vl` (FP32) and send it on every resync; check a per-layer checksum against the release after each sleep.
- **Do not mask the ids as a workaround.**

These ids are sparse. On DS4.1 SWE agentic RL with BI=1 (no full determinism, 8K-token responses, roughly one sentinel per 60K sampled tokens) it showed up as a late single-sample divergence and was first misattributed to preemption and spec decode; in math RL it was rare. With identical siblings (full determinism, same seed) few distinct sequences exist and it is almost never hit, which makes it look like "only distinct siblings fail". Verification: inject each sentinel id into prompts and require bitwise-equal logprobs. Evidence and fixes: [DS4.1 case](../../../cases/deepseek-v4.1-flash/README.md).

Confirm token shift, mask and EOS; comparing ±1/±2 offsets is fine for diagnosis, but a fix cannot be declared through a misaligned comparison. Check whether scores from other samples or versions are being referenced.

## Existing tools: what Miles has, and what cannot be claimed

> Source audit date: 2026-09-06; the version conclusions below do not represent re-verification against the latest upstream.

- A published day-0 support blog (for Model C) describes a canary that fixes a first-round trajectory and scores it on both sides with the same adapter version each step. **The complete public source for this cross-step automatic scheduling has not been located**, so no fictitious switch is provided.
- [prefill_logprobs.py](https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/rollout/generate_utils/prefill_logprobs.py) is a verified rescoring component: full `sample.tokens` as input, `max_new_tokens=0`, `return_logprob=True`, returns scores starting from `prompt_len-1`, with the current `lora_path` specified. It extracts the response tail, checks token IDs, rejects None, and writes back `rollout_log_probs`. The batch interface groups by prompt length and adapter and clears prefix/KV cache before scoring; it does not clear CUDA Graph.
- `--recompute-logprobs-via-prefill` rescores the current samples after rollout collection; it does not automatically freeze the first-round samples. Historical rollout logprobs saved by debug dump/replay are also not the same as current-version scores.

## The cheapest comparison: do not dump all layers yet

Save a small batch of real `prompt_ids + response_ids`, prompt lengths and the valid response mask; from then on only score, never regenerate answers, and never rewrite sequences through decode/re-tokenize. Record base revision, adapter digest/version, config, code SHA, hardware and kernel backend. Training and serving must be compared under the same policy version, with no weight update during scoring.

Repeat scoring the same version on each side to get each side's own noise floor, then compare early and late adapters. A rising mismatch with different samples every round is a troubleshooting signal; it does not by itself prove that the same-input error grows with training.

Under a deterministic implementation the difference between repeated scorings is 0; that is not the achievable floor. Measure two more floors; once a source of train-inference difference is below them, stop chasing it:

- **Inference side against itself:** the same service first decodes to get logprobs, then rescores prompt+response with prefill (once per request and once concurrently). On one model's θ0 its MAE was 0.041, the same order as trainer–rollout, while a budget frozen from another model was 0.03, which even vLLM against itself could not pass: the budget must be set from this model's floor.
- **Training side self-perturbation:** change one equivalent piece of arithmetic on the training side only (e.g. compute an activation in FP32 and round at the end instead of step-wise BF16), or multiply each layer output by `1 + 2^-9·randn`, and rescore the same batch of tokens.

Afterwards compare the three pairwise gaps (decode–prefill, decode–trainer, prefill–trainer): if they are the same order and grow proportionally with training, it is the model's sensitivity to numerical noise that is growing, not an implementation error on one side; only when one pair alone is notably larger should you look for differences on that side.

| Path | Method | What it answers |
|---|---|---|
| Prefill | vLLM full input token IDs + `prompt_logprobs`, slice out the response | Difference between training teacher-forcing and a serving full-sequence forward |
| Decode trace | Normal prompt, force fixed response IDs, keep real logits/logprob | Extra differences from KV/recurrent state, step-by-step generation, decode kernels |

The 0 in `prompt_logprobs=0` means no extra top-k is returned, not that the target token score is turned off. If the interface does not allow zero generation, generate 1 token and discard it, reading only the input scores. The first input token has no conditional score; the first response token is predicted by the logits at the last prompt position, so do not shift by one. Trace switches, runner support and MTP limitations must be checked against the actual vLLM version; do not quietly disable production MTP for a probe. A no-MTP diagnostic comparison may be done explicitly, but its conclusion cannot cover the MTP path. Prefill agreement does not imply decode or MTP agreement.

## Metric definitions

- Prefer comparing raw logprobs; if comparing the sampling distribution, unify temperature, penalty, top-k/top-p and other processing. temperature=0 in the request does not automatically prove which kind of logprob is returned; check the backend semantics.
- `logprob_mae` (token MAE): over valid tokens, `sum(mask * abs(lp_train-lp_serve)) / sum(mask)`; also report p95/max and positions.
- `probability_mae`: over valid tokens, `mean(abs(exp(lp_train)-exp(lp_serve)))`; verl's `training/rollout_probs_diff_mean` belongs to this one, not logprob MAE.
- To get per-token logprob differences in VERL: enable `actor_rollout_ref.rollout.calculate_log_probs=true`, and after the trainer computes `old_log_probs` call `verl.utils.debug.metrics.calculate_log_prob_diff(old_log_probs, rollout_log_probs, response_mask.bool())`. It returns a 1-D absolute difference over valid tokens, not a mean; compute the mean and quantiles yourself, and keep the mask indices for localization. The default debug metric applies `exp` first and then calls the same function, so what it yields is a probability difference. The above was checked against VERL main commit 8e4a572 (2026-10, `verl/utils/debug/metrics.py`); with another version or fork first check the return shape, the mask and whether `exp` is applied first, and record the VERL revision in the run record.
- Relative L2 and cosine are for tensors; do not conflate them with logprob MAE/KL; mark zero vectors separately. Full-vocab log-softmax, probability total variation and argmax are looked at separately and reported apart from the token metrics above; a global shift of raw logits does not mean a change in the probability distribution; there is no cross-model universal threshold.
- Schulman k3 form: let `d=lp_train-lp_rollout`, and compute `exp(d)-1-d`. Only when samples come from the current rollout distribution does it carry the corresponding KL-estimate semantics; on a fixed old trajectory it is only a diagnostic statistic and cannot be called the current policy KL. Check for overflow; do not silently clip and then claim the difference is small.

## Set up four comparison groups

1. Inference vs itself: repeated scoring on frozen tokens, batch1/mixed batch, decode/full-prefill.
2. Training vs itself: A/A and A/C packing replay with the same weights.
3. Cross-stack: μ/π for semantically identical tokens, plus full logits when needed.
4. Score layer: run each scorer on exactly the same logits, checking dtype, temperature, logsumexp and the vocabulary-shard reduction.

When training A/C differs while inference vs itself is stable, focus first on the training layout, but keep the inference verification scope. When both sides are stable on their own but differ cross-stack, first check weights, arithmetic semantics and the scoring path. When it differs only after an update, move to checking version, derived weights/scale, alias and Graph pointers ([weight update](../weight-update/README.md)).

## Same-prefix siblings

Same policy, same prompt, same response prefix, under the same score processing rules, should yield the same conditional probabilities. If siblings have the same μ but different π, that is evidence the training side is sensitive to layout/state. For processed logprobs, also ensure the penalty history and processing config are identical.

Same-prefix comparison is limited to the common prefix and cannot continue past the generation fork. Use clean siblings and repeated runs as negative controls; do not describe "one side looks more stable" as full-range proof of correctness.

## Locating CUDA asynchronous errors

The reported location of an asynchronous illegal access may be later than the faulting kernel. Save the first failure, per-rank logs, inputs/version and execution phase; locate it with scoped synchronization, CUDA core dump/cuda-gdb or sanitizer. Once a process's CUDA state is corrupted, terminate that experiment and do not keep using it to collect numerical evidence. Do not globally disable async or change the attention backend and then directly declare a long-term stability problem solved.

## Validity of comparisons

Disabling cache may cause prefills that never ran before to actually run; disabling Graph also changes speed and online scheduling. To attribute something to Graph itself, fix the composition of each step offline and compare. Negative controls must verify the execution count of the target branch; zero difference with the path never exercised counts as not covered. More experiment design rules in [04 Ablation method](04-ablation-method.md).

After isolating the side, bisect along one axis per [02](02-bisect.md); classification and handling order in [03](03-classify-and-fix.md).
