# Inference-side quantization inventory: every rounding point must be listed, measured, and handled

Most of the fixed gap between training and inference comes from quantization done somewhere on the inference side with no counterpart on the training side. [01 Conversion chain](01-conversion-chain.md) covers how weight bytes go from the master to what vLLM receives; this page covers **every place in the rollout forward that rounds values to low precision**, not only weights. Write the inventory in G0; do the single-switch measurements before training starts in G6 and on the first batch of checkpoints. Every item needs a conclusion before the formal long run.

## 1. Why this is a separate step

The checkpoint quantization config describes only the weights (and a few activation scales). KV cache, the indexer's K cache, Q/P quantization inside attention, and FP8 dispatch in MoE communication are all determined by **vLLM launch arguments or kernel selection**. They are not in `hf_quant_config.json`, and you will not see them when inventorying the checkpoint per [01 §2](01-conversion-chain.md#2-first-inventory-the-checkpoints-complete-quantization-rules).

Model D (2026-10) missed exactly this: the training side did QAT on weights and fake quant on activations (NVFP4 for experts, MXFP8 for non-experts), but the rollout KV cache was NVFP4 (prefill used a kernel with Q in FP8 and KV in FP4) and the indexer KV was also low precision, while the training side had no quantization on K/V at all. Debugging took several rounds because all earlier checks (weight bytes, adapter transfer, activation emulation) matched. In the end, on the step-80 checkpoint, the rollout was switched to `kv_cache_dtype: auto` and BF16 indexer KV with everything else unchanged (both switches turned off together, not measured separately):

| Rollout config | KL (`rollout_corr/kl`) | Per-token \|decode − trainer\| |
| --- | --- | --- |
| NVFP4 KV + low-precision indexer KV (formal config) | 0.087 | 0.167 |
| BF16 KV + BF16 indexer KV | 0.051 | 0.122 |

The remaining 0.122 equals the training-side self-perturbation floor on the same checkpoint ([how to measure](../debugging-mismatch/01-isolate-the-side.md#the-cheapest-comparison-do-not-dump-all-layers-yet)), meaning the remaining gap is at the noise-floor level and no systematic difference beyond these two items is visible. This comparison used a single job; had it been done before training started in G6, the earlier rounds of debugging could have been skipped.

## 2. Build the inventory: read from the running engine, do not infer from the model name

Each row is one rounding point. Sources in priority order: vLLM startup logs and parsed engine args → names of the kernels/backends actually selected → checkpoint quantization config → source code. Model cards and image names do not count.

| Category | Points to list | Where to read the actual value |
| --- | --- | --- |
| Weights | Format of each tensor family (routed/shared expert, attention, dense MLP, lm_head, embedding, MTP) | Checkpoint quantization config + loader logs, per [01 §2](01-conversion-chain.md#2-first-inventory-the-checkpoints-complete-quantization-rules) |
| Activations | Whether each GEMM input is quantized, static or dynamic, block direction | Quant method class name + kernel name; W8A8 / W4A4 and W4A16 are different things |
| KV cache | `kv_cache_dtype`, source of k/v scales (checkpoint calibration values or runtime) | Engine args, attention backend logs |
| Other caches | Storage dtype of the DSA/indexer K cache, MLA compressed cache, linear-attention/Mamba state | Model-specific engine args (e.g. indexer KV dtype), cache spec |
| Inside attention | Whether the prefill/decode kernel quantizes Q and the softmax probabilities P (e.g. Q8KV4, FP8 attention) | Backend and kernel names, checked separately for prefill and decode |
| MoE communication | Whether dispatch/combine is FP8 or lower | all2all backend config and logs |
| LoRA | Storage dtype of A/B on the serving side; whether the delta is added before or after base output rounding | See [LoRA: at which step the delta is rounded](../lora/README.md#where-the-delta-is-rounded) |
| Logits and sampling | lm_head output dtype, `logprobs_mode` (raw or after temperature/top-p processing) | Engine args |

Prefill and decode may use different kernels; list the two paths separately. Read the inventory on the final run config (including LoRA, TP, EP): enabling LoRA may replace the base kernel, changing the activation row accordingly (example and handling in [LoRA §1](../lora/README.md#part-1-1a-serving-lora)). Write the inventory into the G0 `run-plan.yaml`, alongside the precision route.

## 3. For each item, state the training-side counterpart

| Inference-side rounding point | Inference-side format and kernel | Training-side counterpart | How to turn off as a single switch | θ0 contribution | Post-training contribution | Handling |
| --- | --- | --- | --- | --- | --- | --- |
| Example: KV cache + indexer KV (measured together) | NVFP4, prefill Q8KV4; indexer low precision | None | `kv_cache_dtype: auto` + BF16 indexer KV | Not measured | KL 0.087 → 0.051 (step 80) | Unresolved, see §5 |

"Training-side counterpart" must be one of three: **same kernel** (BI=1 route), **emulation** (fake quant; state the function and grid, and whether they are the same as the inference side), or **none**. Items marked "none" must have a single-switch measurement number.

## 4. Single-switch measurement

1. Fix a batch of rollout samples (prompt + response tokens); the trainer side recomputes old logprob with the same weights.
2. On the inference side, turn off only one rounding point at a time (switch to BF16 or disable that quantization), keep all other config, image, and nodes unchanged, and score the same batch of tokens.
3. For each config report: KL, per-token mean \|Δlogprob\|, fraction of tokens with \|Δlogprob\| > 1 (the tail, see [Gap grows during training](../debugging-mismatch/07-drift-during-training.md)). Repeat each config once to give the repeat noise.
4. First measure both noise floors, inference side and training side ([how to measure](../debugging-mismatch/01-isolate-the-side.md#the-cheapest-comparison-do-not-dump-all-layers-yet)). If an item's contribution is below the noise floor, stop chasing it.
5. **Measure once at θ0 and once after some training steps.** With the same numerical noise, the model becomes more sensitive as training proceeds and the contribution grows ([Gap grows during training §2](../debugging-mismatch/07-drift-during-training.md#2-where-the-growth-is-all-in-the-tail)), so measuring only at θ0 may underestimate it (Model D did not measure this item at θ0; this point is an inference). Re-measure on the first batch of checkpoints (e.g. step 30), using a diagnostic job, not inside the training job.

Turning off inference-side quantization is only for measurement; it does not change the formal rollout config.

## 5. Handling after measurement

- For items whose contribution exceeds the floor, pick one: emulate on the training side; switch the inference side to higher precision (record the throughput cost in the performance table); or keep the gap and rely on token-level TIS to contain its effect on training (see [Gap grows during training](../debugging-mismatch/07-drift-during-training.md)), stating this in the report.
- **Emulation must be accepted on end-to-end numbers; matching a single layer does not count.** Run the §4 measurement before and after adding emulation; if it gets worse, do not adopt it. Records of three cases where single layers matched but end-to-end got worse (including Model D's NVFP4 KV emulation) are in [QAT §4.1](02-qat-and-ste.md#41-training-side-quantization-simulation-per-layer-aligned-end-to-end-worse).
- Write unresolved items into the report's known limitations, with the §4 numbers.

## 6. Checklist

- [ ] G0: the inventory covers every category in §2, with prefill and decode separate; every row has a source (log line or engine arg).
- [ ] G0: every row states the training-side counterpart (same kernel / emulation / none).
- [ ] Before training starts in G6: per-item single-switch measurement at θ0, including both floors and repeat noise.
- [ ] First batch of checkpoints: re-measure items whose training-side counterpart is "none".
- [ ] Every item has a handling conclusion; emulated items have end-to-end before/after comparisons.
