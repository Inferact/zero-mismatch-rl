# Precision, Quantization and QAT

Precision-related knowledge for train-inference consistency is maintained only here: which conversions weights go through from the optimizer master to the bytes vLLM actually uses, and what invariant each step must hold; what QAT and low-precision training are and how much mismatch they can remove; and the pitfalls already hit. Transport, sync-completion checks, generic post-load mechanics and CUDA Graph refresh are not covered here, see [Weight update](../weight-update/README.md); the LoRA route is in [LoRA](../lora/README.md); numerical contracts that do not involve quantization are in [numerical-contracts](../../numerical-contracts.md).

Reading order: this page → [04 Inference-side quantization inventory](04-inference-quant-inventory.md) (mandatory in G0, before any precision decision) → [01 Conversion chain](01-conversion-chain.md) (before export, sync, or adding a new format) → [02 QAT and STE](02-qat-and-ste.md) (before choosing training precision or changing backward) → [03 Pitfalls](03-pitfalls.md) (look up by symptom).

## Find by symptom

| Symptom / question | Go to |
| --- | --- |
| Weight bytes, adapter transfer and activation simulation all match, but a fixed train-inference gap remains | [04 Inference-side quantization inventory](04-inference-quant-inventory.md): KV cache, indexer K cache and attention-internal quantization are not in the checkpoint config; measure each with a single toggle |
| Added quantization simulation on the training side; per-layer error went down but end-to-end got worse | [QAT §4.1](02-qat-and-ste.md#41-training-side-quantization-simulation-per-layer-aligned-end-to-end-worse) |
| Garbled output or silent errors after refit or sync | [P13–P16](03-pitfalls.md#export-and-receive) |
| Weights unchanged, but probs_diff grows after export/sync | [P1](03-pitfalls.md#p1-scale-rule-differs-from-the-checkpoint-requantization-flips-codes), [P10](03-pitfalls.md#p10-both-sender-and-receiver-quantize), [P11](03-pitfalls.md#p11-receiver-picks-quantization-targets-by-name-suffix-and-misses-fused-moe) |
| NVFP4/FP8 block scale or packed bytes do not match the checkpoint | [Conversion chain §2](01-conversion-chain.md#2-first-inventory-the-checkpoints-complete-quantization-rules), [P1–P5](03-pitfalls.md#quantization-rules-and-scales) |
| Rollout is NVFP4; should training use QAT or a BF16 actor | [QAT §1](02-qat-and-ste.md#1-choose-the-precision-route-first), [§4](02-qat-and-ste.md#4-how-much-mismatch-qat-can-remove) |
| QAT is on but the mismatch did not change | [P7](03-pitfalls.md#p7-qat-switch-on-target-modules-not-replaced), [P8](03-pitfalls.md#p8-training-grid-differs-from-deployment-grid) |
| QAT / train-inference consistency training loss does not decrease; deployed weights are byte-identical | [QAT §6](02-qat-and-ste.md#6-why-qat-does-not-train), [P28](03-pitfalls.md#p28-master-starting-point-lies-on-the-quantization-grid) |
| Whether to add activation fake-quant | [QAT §4.1](02-qat-and-ste.md#41-training-side-quantization-simulation-per-layer-aligned-end-to-end-worse) |
| Same operator inputs, inference module output still differs | [P17–P19](03-pitfalls.md#training-side-numerics) |
| Fixed logprob bias under FSDP | [P20](03-pitfalls.md#p20-fsdp-bf16-policy-overrides-the-fp32-parameter-contract), [P24](03-pitfalls.md#p24-dividing-by-temperature-in-bf16) |
| Some LoRA/adapter gradients are always zero | [P21](03-pitfalls.md#p21-fake-quant-truncates-gradients) |
| Adding a new quantization format | [Conversion chain §3](01-conversion-chain.md#3-what-each-format-outputs), [§5](01-conversion-chain.md#5-minimal-adaptation-for-a-new-quantization-format) |
| Implementing or changing backward/STE | [QAT §5 STE contract](../training-model-support/06-backward-validation.md#first-write-down-the-gradient-contract); the reuse table and comparison protocol are in [Training side 03](../training-model-support/03-reuse-backward.md) |
| After reload, derived scale/alpha is stale; only the second update goes wrong | [03 cross-references](03-pitfalls.md#cross-reference-derived-value-and-cache-bugs) → [Weight update case-map](../weight-update/01-case-map.md) |

## Key rules

1. "NVFP4/FP8" is only a format name; if any of block scale selection, global scale convention, rounding, or activation static/dynamic differs, the bytes differ. First inventory the checkpoint's complete rules.
2. Use the rules of a mainstream implementation consistent with the checkpoint (TE, ModelOpt, NeMo-RL); do not invent your own. Validate starting from the true BF16 release, not from dequantized weights.
3. The quantization function in training QAT and the exporter are the same function on the same grid (range, axis, scale, rounding).
4. Compute scales over the complete logical tensor; fusion groups share a scale (VERL NVFP4: QKV, gate/up fused global scale); quantizing each shard separately and concatenating is forbidden.
5. Each tensor family has exactly one conversion owner and is quantized only once; post-load reordering does not count as quantization.
6. Activation scales (`input_scale`, `input_amax`, `k_scale`, `v_scale`) are calibration values, passed through unchanged, and cannot be derived from weights; runtime scales are read from the actual runner.
7. High-precision training + low-precision rollout is not QAT; neither fake-QAT nor true low-precision GEMM implies low-precision optimizer/gradients.
8. Weight QAT reduced Nemotron-H probs_diff mean from about 0.025 to about 0.010; only a training forward using the same kernels as inference reaches 0. Whether adding activation fake-quant helps depends on upstream mismatch; measure first.
9. STE backward gets its own acceptance: independent reference, same dY, tolerance agreed in advance; 0 forward diff does not prove the gradient is correct.
10. On every sync, check exported bytes against received bytes tensor by tensor, and check deployed bytes = `requantize(master)`.
11. Start the master from the true BF16 release, not the dequantized checkpoint: the latter sits entirely on grid points, and with a small lr the NVFP4 deployed values take thousands of steps to start changing. Consequently θ0 bytes do not equal the official checkpoint (Nemotron-H: NVFP4 99.776% identical, FP8 24/46 tensors differ); the cost is quantified with a proxy SFT comparison and confirmed by a human ([QAT §6.2](02-qat-and-ste.md#62-fix)).
12. G0 lists the inference-side quantization inventory: for every rounding point in the rollout forward, state whether the training side uses the same kernel, a simulation, or nothing; for items marked "nothing", do one single-toggle measurement before training starts and one on the first batch of checkpoints ([04](04-inference-quant-inventory.md)).

## Terms

- **NVFP4**: values are FP4 E2M1, one FP8 E4M3 block scale per 16 elements, plus one FP32 global scale.
- **block scale / global scale**: the former scales per block (`weight_scale`, `scale`), the latter per whole tensor or fusion group (`weight_scale_2`, `weight_global_scale`, `scale2`).
- **amax**: the maximum absolute value of a tensor (or block), used to compute the scale.
- **4over6 / MSE**: TE NVFP4 block scale selection settings; the Nemotron-H checkpoint uses it.
- **E4M3 / E2M1 / UE8M0**: FP8 (4 exponent bits, 3 mantissa bits) / FP4 (2 exponent bits, 1 mantissa bit) / unsigned 8-bit exponent (scale is a power of 2).
- **W4A16 / W4A4 / W8A8**: weight bit width / activation bit width.
- **QDQ**: quantize then immediately dequantize, giving a high-precision value on the grid.
- **fake-quant / fake-QAT**: the training forward uses QDQ'd values while the GEMM stays high precision; this is not low-precision training.
- **STE**: straight-through estimator, treats the derivative of quantization rounding as 1 (or with a clipping mask).
- **master**: the high-precision parameters managed by the optimizer.
- **refit**: writing new weights into the running rollout engine after a training update.
- **conversion owner**: the single party responsible for converting a tensor family to the inference format.
- **fusion group**: a group of tensors fused in serving (QKV, gate/up) that must share a scale.
- **R3**: MoE routing replay; the training side uses the expert choices recorded by rollout.
- **BI=1 / BI=0**: the `VLLM_BATCH_INVARIANT` switch; 1 is the true on-policy arm where training calls inference kernels, 0 is the regular baseline.
- **probs_diff**: verl's `training/rollout_probs_diff_*`, the token probability difference between rollout and training forward.
- **TIS / `rollout_is`**: rollout importance sampling correction.
- **θ0**: policy weights before the first update.
- **oracle**: the reference implementation or reference value that decides pass/fail.
- **VJP**: the backward function that, given upstream gradients, computes input and parameter gradients.

## Files

| File | Contents |
| --- | --- |
| [01-conversion-chain.md](01-conversion-chain.md) | Conversion chain master → export → transport → receive → deployed bytes; invariants and validation per step; conversion owner; minimal adaptation for new formats |
| [02-qat-and-ste.md](02-qat-and-ste.md) | Choosing a precision route; four kinds of paths; fake-QAT implementation; how much mismatch QAT removes; STE contract; why QAT does not train |
| [04-inference-quant-inventory.md](04-inference-quant-inventory.md) | Inference-side quantization inventory: which points to list, where to read actual values, training-side counterpart, single-toggle measurement, measure once at θ0 and once after training, what to do after measuring |
| [03-pitfalls.md](03-pitfalls.md) | P1–P28 pitfall table, ordered quantization rules → export and receive → training-side numerics → experiment methodology; derived-value bugs cross-referenced to weight update |

## Evidence index

- Nemotron-H: [NVFP4 stagnation research](../../../cases/nemotron/reports/research-nvfp4-rl-stagnation.md), [proxy SFT three-arm comparison](../../../cases/nemotron/reports/sft-three-arm.md), [2026-10 delivery](../../../cases/nemotron/consistency-case.md#2026-10-lightning-nvfp4-true-on-policy-delivery) (FP8 dispatch and Q scale see [DAPO case](../../../cases/nemotron/dapo-20260927.md))
- Model B: ModelOpt NVFP4 schema (case report not included)
- Model E: two precision differences with causal evidence (case report not included)
- DeepSeek V4.1: Miles precision contract comparison (raw history archive, not included)
- Model F FP8 LoRA: E2E retrospective (case report not included)
- Model D: NVFP4 KV and low-precision indexer KV are not simulated on the training side; single-toggle comparison in [04 §1](04-inference-quant-inventory.md#1-why-this-is-a-separate-step), end-to-end results of training-side KV simulation in [QAT §4.1](02-qat-and-ste.md#41-training-side-quantization-simulation-per-layer-aligned-end-to-end-worse)
- Model C: MXFP4 v2, draft buffer and large-model validation (case report not included)
