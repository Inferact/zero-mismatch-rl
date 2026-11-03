# Conversion chain: from master weights to the bytes vLLM receives

This page follows the data flow of one weight sync: what each step produces, what it must satisfy, and how to verify it. Transport, completion checks and CUDA Graph refresh are in [Weight update](../weight-update/README.md).

## 1. Four stages

```text
A. master weights     FP32/BF16 optimizer master (plus model-declared FP32 exceptions)
   ↓ forward materialization
B. weights the training forward actually sees
                      BF16; under QAT DQ(Q(W)), i.e. dequantized values on the deployment grid
   ↓ export (gather → HF mapping → quantize/pack)
C. inference-format bytes  packed NVFP4 + block scale + global scale / FP8 + scale /
                      INT4 packed + scale / layers still BF16 in mixed precision
   ↓ transport → loader → post-load
D. tensors vLLM receives and kernels use (incl. post-load reordering and derived scales)
```

| Stage | Invariant | How to verify |
| --- | --- | --- |
| A | Record dtype separately for parameters and buffers; an FP32 master does not mean forward uses FP32 | Read optimizer parameter identity and dtype; see [§6](#6-dtype-contract-six-boundaries) |
| B | Under QAT, `Q` must be the same function on the same grid as C's export function | Unit test: B's weights = dequantize(C's bytes), bit for bit |
| C | Scales computed over the complete logical tensor; fusion groups share the global scale; excluded layers keep their original dtype; recomputing from the same master is idempotent | Compare packed bytes, block scale and global scale per tensor; see [§2](#2-first-inventory-the-checkpoints-complete-quantization-rules) |
| D | Each tensor family quantized only once; post-load only reorders and does not recompute scales; calibration values arrive unchanged | Check actor-exported bytes against bytes the consumer actually received, tensor by tensor |

Record the real formats and scales of weights, activations and KV (packed UE8M0, FP32 scale, scope of MXFP4/FP8); do not assume activation and packing rounding are identical across SM90/SM100. First confirm the scale was computed from this round's weights and that parameters do not point at stale tensors, then measure quantization error.

## 2. First inventory the checkpoint's complete quantization rules

1. From `hf_quant_config.json` and `quantization_config` in `config.json`, list each layer's format, which layers carry `input_scale`, and whether the KV cache carries k/v scales. Read layer by layer; do not infer from the model's overall name ([P12](03-pitfalls.md#p12-inferring-a-layers-precision-from-the-models-overall-label)). Example, Nemotron-H Lightning 30B-A3B: `MIXED_PRECISION`; the 46 FP8 layers are `mixer.in_proj`/`out_proj` of the 23 Mamba layers; the 5935 `W4A16_NVFP4` layers are routed/shared experts and lm_head; KV cache FP8; all 46 `input_scale` are fp32 scalars, only on FP8 layers.
2. Encode the BF16 release with the checkpoint's own scales; packed bytes should be 100% identical, which confirms the BF16 release corresponds to the checkpoint.
3. Starting from the **true BF16 release**, requantize with the candidate rule and report per tensor the match ratio of packed bytes, block scale and global scale. Dequantized weights already lie on the grid, so recomputing with any rule leaves them unchanged, and validating with them passes falsely ([P2](03-pitfalls.md#p2-validating-rules-on-dequantized-weights-gives-false-positives)). Measured match rates for each rule are in [P1](03-pitfalls.md#p1-scale-rule-differs-from-the-checkpoint-requantization-flips-codes) and [P3](03-pitfalls.md#p3-fp8-per-tensor-scale-convention-is-not-amax448).

Decisions:

- By default call the rules of a mainstream implementation (TE, ModelOpt, NeMo-RL) directly; do not invent a scale strategy. Deviate and report only when the mainstream implementation cannot do it. Nemotron's self-invented "enlarge the scale only on overflow" was later replaced by TE's ready-made rule.
- θ0 deployed bytes = the true BF16 release requantized by the rule, not the official checkpoint bytes: Nemotron-H measured NVFP4 bytes 99.776% identical, FP8 24/46 tensors different, proxy SFT cost about 0.03 nat ([QAT §6.2](02-qat-and-ste.md#62-fix), [DS4 practice differences §E.1](../../../cases/nemotron/reports/ds4-practice-differences.md#e1-correction-2026-10-03-pending-human-confirmation)). This is a policy that requires human confirmation.
- The BI=1 training forward, BI=0 QAT and the exporter call the same `requantize`; θ0 bytes are identical across the BI=1/BI=0 arms. `input_scale`, `k_scale` and `v_scale` are calibration values passed through from the checkpoint unchanged.
- When a single FP8 checkpoint is the source of truth, the training base is its deterministic dequantized derivative, and requantizing is not guaranteed to reproduce the original checkpoint byte for byte. Measure separately: ① bytes identical before and after sync with zero update; ② precision difference from the original checkpoint.

## 3. What each format outputs

| Format | Output | Must check |
| --- | --- | --- |
| block FP8 | FP8 weight + `weight_scale_inv`, scale per block | Commonly 128×128; FP32/UE8M0 branch, backend, exclusion scope |
| per-tensor FP8 | FP8 weight + `weight_scale` | Scale derived from the amax of the complete weight, not of an arbitrary shard; convention may not be amax/448 |
| MXFP8 | E4M3 weight + UE8M0 `weight_scale_inv` | 1×32 block direction, scale encoding, padding and high-precision exceptions |
| NVFP4 (Miles exporter) | packed weight + `weight_scale` + `weight_scale_2` | 1×16, two-level scale, shared outer scale for gate/up, activation contract |
| NVFP4 W4A16 (compressed-tensors, existing `fp4` sender) | `.weight_packed` + `.weight_scale` + `.weight_global_scale` | Same as above |
| NVFP4 W4A4 (ModelOpt, Model B) | packed `uint8` keeping the original parameter name + FP8 `scale` + FP32 `scale2` + BF16 `input_amax` + `original_shape` | Naming is not interchangeable with compressed-tensors; needs a separate adapter (Model B's handoff report §4, not included) |
| INT4 compressed-tensors | `weight_packed` + `weight_scale` + `weight_shape` | group_size, symmetric quantization, pack layout, the actual list of quantized tensors |
| DeepSeek V4.1 attention FP8 | F8_E4M3 weight + F8_E8M0 `.scale`, 32×32 block | Not rowwise 32-element MXFP8; see V4.1 training history archive source-23 for the schema and block32 threshold (raw history archive, not included) |

`weight_scale_inv` is only a loader protocol name; how its values are interpreted follows the corresponding implementation.

## 4. Who converts: conversion owner

A tensor family can have only one conversion owner. Once the sender has packed, the receiver must not quantize again. dtype reinterpretation, layout staging and post-load repacking do not count as quantization.

| Path | Training | Export / transport | Rollout load | owner |
|---|---|---|---|---|
| LoRA `merge=false` | BF16/FP32 adapter | Full adapter | Replace adapter on the already-loaded base | No quantization |
| FP8 rollout-only | BF16 | Full BF16 | `quant_weights()` converts to block-FP8 | rollout receiver |
| VERL FSDP QAT NVFP4 | BF16 + fake quant | After gathering the full tensor, `QATQuantizer` outputs packed NVFP4/scale | compressed-tensors | FSDP sender |
| VERL Megatron QAT NVFP4 | BF16 + ModelOpt fake quant | `QATWeightExporter` | ModelOpt | Megatron sender |
| Megatron FP8 E2E | TE FP8 | Bridge-specific FP8 export planning | vLLM FP8 | Bridge / rollout, per the fixed recipe |
| Nemotron-H BI=0 (verl Megatron) | BF16 + ModelOpt weight STE | Each sync requantizes by checkpoint rule into ModelOpt `MIXED_PRECISION` | ModelOpt | sender (`mixed_precision_export=True`) |
| Pre-quantized checkpoint passthrough | No requantization | Original tensor/scale | Per checkpoint layout | checkpoint producer |

- adapter-only: transfer only the adapter; the low-precision base stays in vLLM. In Miles Bridge only `weight_type == "base"` enters the quantization processor; adapters only get name/shape/shard alignment. [Bridge][bridge-latest] "adapter-only every round" does not mean the base is never restored in any lifecycle: initialization, a new engine joining, and offload restore must be audited separately.
- full merge: obtain logically complete HF tensors one by one and convert via the single owner; it does not require all-gathering the whole model at once, nor writing to disk every step.
- QAT: the training forward already simulates quantization; export produces the actual packed weights and scales.

### 4.1 FSDP path (VERL)

- With `actor_rollout_ref.actor.fsdp_config.qat.enable=true`, quantizable `nn.Linear` are replaced by `QATLinear` after the PEFT wrapper and before FSDP wrap; forward does NVFP4 W4A16/W4A4 fake quant; `QATLinear.forward` ultimately calls `F.linear`; after the optimizer step `invalidate_all_scales` clears the weight scale cache. [Implementation][qat-linear]
- Export: `FSDPEngine.get_per_tensor_param()` calls `full_tensor()` on each `DTensor` first, then feeds it to `QATQuantizer.quantize_with_fusion()`:

```text
FSDP local shards → full_tensor() all-gather → complete HF weight
→ full-weight amax / global_scale → QKV or gate/up fused_global_scale
→ per-row, per-group block scale → packed NVFP4 weight + scales
```

  The cost is all-gather and GPU staging; the quantizer caches per decoder layer to CPU and moves back to GPU, so peak GPU memory is not necessarily just one tensor. [Export][qat-export]
- `get_per_tensor_param_shard()` only outputs the local BF16 shard, does not call the quantizer, and produces no scale. Shard-local block quantization is an unsupported new protocol; it must at least solve:
  1. aligning shards with block boundaries, or correctly handling blocks that span shards;
  2. a collective for the complete weight's amax/global scale;
  3. a consistent fused global scale for QKV and gate/up;
  4. W4A4 explicitly carrying the activation-derived `input_global_scale`, which cannot be reconstructed from a weight shard.

  Until these contracts and an oracle are done, quantizing each shard separately and concatenating is forbidden.
- Non-QAT: export BF16 in HF coordinates; BF16 rollout does not quantize, and with `rollout.quantization=fp8` the receiver quantizes. For the LoRA merge exception see [P6](03-pitfalls.md#p6-lora-mergetrue-bypasses-fsdp-qat-export).

### 4.2 Megatron path

- VERL: ModelOpt installs fake-quant modules; `MegatronEngine.get_per_tensor_param()` exports HF weights via Megatron-Bridge, and `QATWeightExporter` reads amax/block metadata from the module quantizers to produce NVFP4 and scales. [ModelOpt exporter][modelopt-source]
- VERL FP8 E2E: TE handles FP8 forward/backward during training; when Bridge declares `export_weight_dtype=fp8`, VERL does not cache generic conversion tasks. The final rollout format is determined by `rollout.quantization` and the serving checkpoint contract; do not infer that wire tensors are already in vLLM layout just because training enabled FP8.
- Miles's raw iterator materializes batch by batch, restores TP/ETP/EP dimensions, then `convert_to_hf()` → `quantize_params()`; it does not all-gather the whole model at once. Both before and after quantization the conversion buffer peak matters; counting only the final FP8 size underestimates GPU memory. When Bridge already outputs packed tensors they must be passed through; the compressed-tensors path determines quantization scope by the actual set of `weight_packed` in the checkpoint index. [raw iterator][direct], [Bridge][bridge]
- Megatron capability cannot be inferred as FSDP capability: the Miles FSDP checkpoint guard refuses to initialize directly from a native FP8 checkpoint and requires dequantizing to BF16 first; the dtype cast in its weight update does not mean the Megatron quantizer is wired up. [FSDP guard][fsdpguard], [FSDP sync][fsdpsync]
- Distinguish artifacts: the offline converter prepares the initial checkpoint and may write to disk; the online exporter streams conversion in memory on every sync; receiver post-load reorders for kernels, which is not recomputing scales; a training checkpoint and a directly servable quantized HF checkpoint are different artifacts.

### 4.3 Receiver quantization (rollout-only)

`rollout.quantization=fp8`: the trainer exports BF16; the vLLM worker's `load_quanted_weights()` calls `quant_weights()`, which uses `scaled_fp8_blockwise()` to produce FP8 weight and `weight_scale_inv`, then replays the quant method's post-load. Besides the generic path there are a DeepSeek V4-specific entry point and an Ascend MXFP8 branch with different scale naming rules. [receiver][receiver-source] When the checkpoint is already FP8/MXFP4, first confirm whether it is requantization or passthrough. vLLM cannot quantize BF16 online into ModelOpt `MIXED_PRECISION`; see [P13](03-pitfalls.md#p13-quantization-config-identified-by-class-name-mixed-falls-through-to-plain-load_weights).

## 5. Minimal adaptation for a new quantization format

1. From the checkpoint config and the actual loader, list weight/scale keys, shape, dtype, block size, packing, fusion groups, and the unquantized tensor families.
2. Trace export → transport → receiver → post-load and mark the single owner of each family.
3. If a converter is missing, add the minimal format conversion at the existing owner's entry point; validate on a single tensor / fusion group first, then small-model cold load and hot update. Do not just change the dtype and pass it off as conversion.
4. Test zero merge, a nonzero update and a second update; for quantize-after-full-gather, ensure correctness first, then consider shard optimization.

Only one evidence table is needed: tensor family → source → wire format → serving format → owner → unit/small-model validation.

### Minimal acceptance

- [ ] Record training dtype, wire dtype/layout, rollout dtype/layout and owner.
- [ ] Fixed-token logprob of the zero-adapter path vs the base is within the same-backend noise baseline.
- [ ] After a nonzero update, tokenwise logprob/KL between trainer and rollout stays within bounds.
- [ ] name/shape/dtype/count/bytes of packed weights, scales and ignored families reconcile.
- [ ] Deployed bytes = `requantize(master)`; on every sync, exported bytes checked against received bytes tensor by tensor (Nemotron-H: 16 syncs, 291,472 tensors, 0 diff).
- [ ] FSDP QAT proves scales are computed after `full_tensor()`; when using shard export, prove it is unquantized, or provide the four shard-local contracts from §4.1 and a cross-rank oracle.
- [ ] Repeat sync twice; sleep/wake, MTP and existing CUDA Graph replay pass.
- [ ] Each family is quantized exactly once.

## 6. dtype contract: six boundaries

For each model record `checkpoint → load/bridge → wrapper/sharding → forward → backward/optimizer → export/serving` separately.

| Boundary | Miles/Megatron | VERL/FSDP | Must check |
|---|---|---|---|
| Source weights and conversion | Bridge may reorder, split, cast; model may override default conversion | HF loader, dequantized derived checkpoint, initialization dtype | Same revision; logical values/scale/layout consistent, not just the target dtype |
| Parameter storage | `params_dtype` plus model exceptions; FP32 master/optimizer | HF keep-precision list, PEFT master, FSDP shard | An FP32 master does not mean forward uses FP32 |
| Forward parameters | Float16Module selective conversion | FSDP2 `MixedPrecisionPolicy.param_dtype` | Measure inside forward, not only before wrapping |
| Operator compute and output | TE/FLA/TileLang/fused each define product, accumulation and output precision | HF/PEFT kernels and autocast | FP32 output does not prove FP32 products; BF16 output does not prove BF16 accumulation |
| Gradients and optimizer | TP/EP/SP/CP collectives, DDP | FSDP reduce dtype, gradient accumulation | FP32 reduction does not recover values already rounded in forward |
| Export and serving | Undo parallel layout, convert adapter/quantization format | gather/export HF adapter | Compare values in the same logical layout |

Miles's FP32 protection comes in four kinds: mark-to-skip before conversion (`mark_keep_in_fp32`, DSv4 mHC, sink); reset after declaration (`mark_param_dtype`/`enforce_marked_param_dtypes`, which does not recover precision already lost; Qwen3.5 explicitly calls `mark_param_dtype`, but the audit did not confirm the generic main flow calls `enforce_marked_param_dtypes`); bridge bypassing the default BF16 downcast (DSv4); and operator-level choices (Model B router, MoE activation/combine in FP32; Qwen GDN makes different choices for `A_log` and norm). It is not "always upcast special parameters". Evidence: a public PR, a public PR, a public PR (2026-09-16 audit).

FSDP side: HF `_keep_in_fp32_modules_strict` is a loading contract and does not automatically become a per-parameter FSDP cast exception (Model E measurements in [P20](03-pitfalls.md#p20-fsdp-bf16-policy-overrides-the-fp32-parameter-contract)); prefer respecting the model's existing contract, and changing compute precision must be validated with a same-model reference and actual serving; `param_dtype=None`, `param_dtype=fp32` and FP32 autocast are not the same thing.

## 7. Code entry points

Start here when looking up how a rule is actually implemented in code: for which fake quant the training forward uses, read the three FSDP entry points; for how exported bytes are produced, read NVFP4 export and the ModelOpt exporter; for how the rollout side receives, read the receiver. PR leads are only for tracing when a behavior was introduced.

FSDP [engine][fsdp-source], [module selection][qat-core], [fake quant][qat-linear], [NVFP4 export][qat-export]; Megatron [ModelOpt exporter][modelopt-source]; rollout [receiver][receiver-source]. Historical PR leads (do not reflect current state): LoRA merge early return [#5115](https://github.com/verl-project/verl/pull/5115)/[#5411](https://github.com/verl-project/verl/pull/5411)/[#7014](https://github.com/verl-project/verl/pull/7014); fake quant and sync quantization [#7149](https://github.com/verl-project/verl/pull/7149)/[#7224](https://github.com/verl-project/verl/pull/7224)/[#7577](https://github.com/verl-project/verl/pull/7577); adapter-only and sharded export [#7483](https://github.com/verl-project/verl/pull/7483)/[#6661](https://github.com/verl-project/verl/pull/6661).

> Source-code conclusions come from a 2026-09-14 read-only audit (VERL `753aed3e`, Miles `7daba3a4`, others pinned at `d2fc97ce`), no GPU runs, and do not represent latest upstream.

[fsdp-source]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/workers/engine/fsdp/transformer_impl.py
[qat-core]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/core.py
[qat-linear]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/linear.py
[qat-export]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/quantizer.py
[modelopt-source]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/modelopt/qat_weight_exporter.py
[receiver-source]: https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/vllm/vllm_quant_utils.py
[bridge-latest]: https://github.com/radixark/miles/blob/7daba3a4f2ef94497c5af6a502bdcc3de1a2c9c5/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py
[direct]: https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py
[bridge]: https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py
[fsdpguard]: https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/fsdp_utils/adaptations/class_patches.py
[fsdpsync]: https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/fsdp_utils/update_weight_utils.py
