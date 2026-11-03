# Known pitfalls

Ordered along the conversion chain: quantization rules → export and receive → training-side numerics → experiment methodology; P28 is filed under training-side numerics by content. Each item is written once; where details live elsewhere, a link is given.

## Quantization rules and scales

### P1 Scale rule differs from the checkpoint; requantization flips codes

- Symptom: weights unchanged, yet rollout has already drifted after sync.
- Cause: recomputing scales on the dequantized master with ModelOpt's default global = amax/(6·448) differs from the checkpoint's TE 4over6 rule; about 40% of Nemotron-H FP4 codes flip. Recomputing with TE 4over6 is idempotent and byte-identical.
- Fix: use the mainstream rule that matches the checkpoint; training QAT and the exporter call the same `requantize`.
- Verify: requantizing the same master twice is byte-identical; requantize from the true BF16 release and compare per tensor against the checkpoint. Nemotron-H measurements:

  | Rule | Match with checkpoint |
  | --- | --- |
  | TE NVFP4 4over6 + MSE, E4M3 upper bound 256, global = amax/1536 | packed 99.776%, global 100%; no setting closes the remaining gap |
  | ModelOpt default global = amax/(6·448) | packed about 50% |
  | Standard FP8 per-tensor amax/448 | See P3 |
- Source: Nemotron-H 2026-10 measurement; procedure in [Conversion chain §2](01-conversion-chain.md#2-first-inventory-the-checkpoints-complete-quantization-rules).

### P2 Validating rules on dequantized weights gives false positives

- Symptom: several candidate rules all report 100% match.
- Cause: dequantized weights already sit on the grid, so recomputing with any rule leaves them unchanged.
- Fix: validate starting from the true BF16 release.
- Verify: at least one known-wrong rule fails this test.
- Source: same as P1.

### P3 FP8 per-tensor scale convention is not amax/448

- Symptom: FP8 layers differ from the checkpoint after requantization.
- Cause: the official Nemotron-H Mamba projection scale is about 1.5×amax/448; 24 of 46 FP8 tensors differ from amax/448.
- Fix: infer the convention backwards from the checkpoint, or pass the checkpoint's scale through directly.
- Verify: report the scale match rate per tensor.
- Source: same as P1.

### P4 Computing scales on each shard separately

- Symptom: after FSDP/TP sharded export, concatenated packed weights differ from quantizing the full tensor.
- Cause: block boundaries cross shards; global scale and fused global scale need the full tensor.
- Fix: `full_tensor()`/gather first, then quantize; shard-local quantization must first satisfy the four contract items in [Conversion chain §4.1](01-conversion-chain.md#41-fsdp-path-verl).
- Verify: evidence shows the scale is computed after gather.
- Source: VERL 2026-09-14 audit.

### P5 gate/up not quantized as a pair

- Symptom: numerics are wrong after serving fuses gate/up, or the exporter reports a missing pair.
- Cause: the two HF tensors are fused in serving and need a shared outer scale.
- Fix: quantize them as a pair; the Miles NVFP4 exporter errors out when a pair is incomplete, and the offline converter has a test for pairing across safetensors shards.
- Verify: `tests/fast-gpu/test_quantizer_ci.py`, `test_mxfp8_quantizer.py`, `test_nvfp4_quantizer.py` (the last covers pair completeness, cross-shard gate/up, TE bitwise reference); they need matching hardware, and their existence does not mean they have been run and passed.
- Source: [NVFP4 processor](https://github.com/radixark/miles/blob/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py).

### P6 LoRA merge=true bypasses FSDP QAT export

- Symptom: with QAT and LoRA merge both on, rollout does not receive the QAT-exported packed weights.
- Cause: for `merge=true`, VERL FSDP directly returns `_merged_lora_per_tensor_param()` and skips `QATQuantizer`; with `rollout.quantization=fp8` it may still be requantized at the receiver. The `merge=false` branch still checks `_qat_enabled` afterwards.
- Fix: the first-version merge fallback does, per full tensor, gather → merge → quantize → yield → release.
- Verify: check the actual module replacement, naming, and export scope; do not declare the combination works just because both switches are on.
- Source: [VERL engine](https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/workers/engine/fsdp/transformer_impl.py), 2026-09-14 audit.

### P7 QAT switch on, target modules not replaced

- Symptom: the QAT arm's probs_diff equals the BF16 arm's.
- Cause: VERL `QATLinear` hooks in at `nn.Linear` and does not guarantee 3D expert Parameters or arbitrary PEFT wrappers; the Miles INT4 hook covers only routed-expert `GroupedLinear`; the framework's built-in QAT only modifies the GPT layer spec, while Mamba hybrid models read the hybrid stack spec.
- Fix: attach the quantizer after building the model from the actual spec; replace the quantizer with the exporter's same `requantize`.
- Verify: each rank prints the replaced module count and list and fails closed (139 for Nemotron-H, identical on all four ranks).
- Source: [Conversion chain §4.1](01-conversion-chain.md#41-fsdp-path-verl), [BI=0 integration pitfalls](../recipe-evaluation/bi-switch-recipe.md#bi0-arm-integration-pitfalls).

### P8 Training grid differs from deployment grid

- Symptom: QAT is declared, but the mismatch remains.
- Cause: in Miles main as of 2026-09-14, `run_qwen3_30b_a3b.py --rollout-int4` calls the converter without passing group size, `convert_hf_to_int4_direct.py` defaults to 32, and the trainer env sets 128 (confirmed at the code level, not run; do not claim OOM or reward collapse; the Kimi-K2.5 route aligns on 32). An external TPU case was bitwise 0 on an FP8 per-channel grid, but the native weights are 128×128 blocks, and the author notes the conclusion may not hold on a different grid.
- Fix: align item by item: quantized module scope, block/group size, axis, scale dtype/rounding, global scale scope, gate/up fusion, activation scale, TP/ETP partitioning, and high-precision exceptions. FP4 has only 1 mantissa bit, so error comes mainly from block partitioning; block size and direction must match the native weights. Another model's public recipe applies QDQ to experts after every update, following the constraints of the rollout MXFP4 GEMM.
- Verify: training-side QDQ output = dequantize(deployed bytes), bitwise.
- Source: [launcher](https://github.com/radixark/miles/blob/7daba3a4f2ef94497c5af6a502bdcc3de1a2c9c5/scripts/run_qwen3_30b_a3b.py), [converter](https://github.com/radixark/miles/blob/7daba3a4f2ef94497c5af6a502bdcc3de1a2c9c5/tools/convert_hf_to_int4_direct.py); constraints of unified NVFP4 in the [tech blog](https://www.lmsys.org/blog/2026-07-29-mxfp8-nvfp4-rl/) (its bit-exact quantizer test is format validation, not whole-model bit-exactness).

### P9 Activation scale derived from weights

- Symptom: W4A4 export errors out, or the exported activation scale differs from serving.
- Cause: activation scale depends on actual inputs. VERL W4A4 export throws when no initialized `input_global_scale` exists; Model B's `input_amax` cannot be derived from BF16 weights.
- Fix: reuse as-is from QAT calibration state or the rollout checkpoint; pass `input_scale`, `k_scale`, `v_scale` through.
- Verify: activation scales in the manifest are byte-identical to the checkpoint.
- Source: [VERL quantizer](https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/qat/quantizer.py), Model B's handoff report §4 (not included).

## Export and receive

### P10 Both sender and receiver quantize

- Symptom: numerical deviation larger than a single quantization; or packed tensors are treated as BF16 and quantized again.
- Cause: one tensor family has two conversion owners; Bridge already outputs packed tensors but the generic packer does not recognize them.
- Fix: assign exactly one owner per family; pass packed tensors through.
- Verify: the evidence table lists the owner per family; count/bytes close.
- Source: [Conversion chain §4](01-conversion-chain.md#4-who-converts-conversion-owner).

### P11 Receiver picks quantization targets by name suffix and misses fused MoE

- Symptom: under `rollout.quantization=fp8`, some experts are still BF16.
- Cause: `gate_up_proj`/`down_proj` do not end with `.weight` and are skipped; fixed in VERL [#7443](https://github.com/verl-project/verl/pull/7443). The Miles FP8 processor also selects scope by parameter name and supports only E4M3 with dynamic activation; it is not a general FP8 converter.
- Fix: check coverage using both checkpoint names and runtime modules/dtypes; review the list for each new model.
- Verify: list the serving dtype of each family.
- Source: [receiver](https://github.com/verl-project/verl/blob/753aed3e1c286ba6825a74342b28669e72c083ea/verl/utils/vllm/vllm_quant_utils.py), [Miles processors](https://github.com/radixark/miles/tree/d2fc97ce581577e255e494801d7568747d5a10d7/miles/backends/megatron_utils/megatron_to_hf/processors).

### P12 Inferring a layer's precision from the model's overall label

- Symptom: localization heads in the wrong direction.
- Cause: in the Nemotron-H "NVFP4" checkpoint, the `quant_algo` of `backbone.layers.16.mixer.out_proj` is FP8.
- Fix: read `hf_quant_config.json` layer by layer.
- Verify: the localization report states the actual `quant_algo` of the first diverging layer.
- Source: [DAPO case: first divergence](../../../cases/nemotron/dapo-20260927.md#further-localization-the-first-divergence-is-the-fp8-input-quantization-of-the-mamba-output-projection).

### P13 Quantization config identified by class name; MIXED falls through to plain load_weights

- Symptom: output is silently wrong after sync, with no error.
- Cause: vLLM has no path to quantize BF16 online into ModelOpt `MIXED_PRECISION`/W4A16; a sync patch that identifies the config by class name lets MIXED fall through to plain `load_weights`, writing into parameters that have already gone through `process_weights_after_loading`.
- Fix: the sender sends in the checkpoint format.
- Verify: on every sync, compare exported bytes against received bytes per tensor.
- Source: Nemotron-H 2026-10 measurement.

### P14 Partially updating fused parameters

- Symptom: output is silently wrong, with no error, after sending only some shards of a fused parameter.
- Cause: in layerwise reload, for a fused parameter that received only some shards, the unreceived shards are uninitialized GPU memory.
- Fix: the minimum granularity is all shards of one fused parameter; whole layers are safest. The first update after sleep level 2 must be a full update.
- Verify: fixed-token logprob after sync equals cold load.
- Source: Nemotron-H 2026-10 measurement; completion criteria and RFC classification in [Weight update case-map](../weight-update/01-case-map.md).

### P15 FlashInfer attention lazily caches KV scales

- Symptom: garbled output after refit (prompt NLL about 11, normally about 1), while byte checksums of all parameters and buffers match a normal load.
- Cause: the first forward multiplies the q·k scale and v scale into `bmm1_scale`/`bmm2_scale` and caches them on the impl. verl starts with `load_format=dummy` by default (scales are 1.0); after refit writes the real k/v scales, the cache is not invalidated. Triggered only with quantized KV cache + FlashInfer; the MLA-family FlashInfer backends and output quantization scales have the same caching pattern. Nemotron-H's BI=1 path uses its own attention backend and does not go through this cache.
- Fix: work around with `load_format=auto`; the fix is to invalidate the cache at reload finalize or recompute from the layer each step, see the internal vLLM fork fix (independent of the fork's other reload fixes, not submitted upstream).
- Verify: after refit, compare per layer the impl's `bmm1_scale`/`bmm2_scale` against `layer._k_scale_float`/`_v_scale_float`. Diagnostic order: checksums identical but output bad → state is cached on the impl or kernel object → per-layer forward hooks to find the first divergence (here `qkv_proj` output identical, `attn` output std 0.03→9.4).
- Source: Nemotron-H 2026-10 measurement.

### P16 post-load replaced by a no-op

- Symptom: drift after NVFP4 identity reload (Model B, two jobs: 8 decode tokens and 39 logprobs unequal, max absolute diff 1.9646437168121338), lifecycle same as BF16, baseline repeats bitwise equal.
- Cause: `FusedMoEWithLoRA` replaces the quant method with `FusedMoEModularMethod`, which inherits the base class's no-op `process_weights_after_loading`; reload calls only the current quant method, so ModelOpt NVFP4's format conversion and derived scales are not refreshed. The same replacement also happens under DP/EP prepare-finalize without LoRA.
- Fix: simple delegation does not work: the LoRA-held `_moe_kernel`/context must be updated at the same time, and the Graph storage contract must be satisfied.
- Verify: cold load → capture → fixed-token oracle → identity update with the same weights → compare again.
- Source: Model B's support audit (P0-3, and a clean NVFP4 identity-reload drift repro job; not included).

### Cross-reference: derived-value and cache bugs

Details are in the [Weight update case-map](../weight-update/01-case-map.md); only one line each is kept here so they can be found by quantization format:

| bug | Precision-side symptom | Go to |
| --- | --- | --- |
| NVFP4 reload stale derived values [vllm#57954](https://github.com/vllm-project/vllm/pull/57954) | After reloading weights ×1.1/×0.9, `g1_scale_c` still uses the previous round's alpha, relative RMS error 9.58%/22.49%; clamp/beta is re-folded on every reload, amplified about 2500ⁿ times on the n-th reload | [case-map #57954](../weight-update/01-case-map.md#57954-nvfp4-reload-stale-derived-values) |
| FlashInfer KV scale lazy cache | See P15 | [case-map](../weight-update/01-case-map.md#internal-bug--rfc-class) |

## Training-side numerics

### P17 Scale taken from the wrong object

- Symptom: unit-scale unit tests pass, real attention output and old-logprob fail.
- Cause: the training adapter quantized Q with `k_scale`. On the real runner, `q_scale=1.0`, `k_scale=0.03173828125`, `v_scale=0.003326416015625`.
- Fix: read scales from the runner/module actually executed; do not infer defaults from missing checkpoint fields or other tensors.
- Verify: replay with real non-unit scales and frozen Q/K/V; after the Nemotron-H fix, 5,832,704/5,832,704 BF16 outputs are bitwise identical.
- Source: [DAPO case: Q scale](../../../cases/nemotron/dapo-20260927.md#follow-up-fix-on-the-five-layer-proxy-the-q-scale-must-take-the-real-runner-value).

### P18 Same-named quantization module takes a different dispatch

- Symptom: calling `ops.scaled_fp8_quant` + `flashinfer_scaled_fp8_mm` directly is bitwise identical to training, but the inference linear module output has 1,361 differing BF16 elements; scales are identical.
- Cause: the module actually selects the **compiled** `forward_native`, which has 44 more differing FP8 elements than the uncompiled native and the CUDA custom op; uncompiled native and CUDA give identical results. Which compile transformation is responsible is unconfirmed.
- Fix: under Nemotron-H BI=1, static per-tensor `QuantFP8` is switched to the device custom op (GEMM and format unchanged).
- Verify: with the same op inputs, weights, and scales, compare FP8 bytes, GEMM output, and module output in order, calling the native/CUDA/actual module paths separately. When direct GEMM matches training but the module differs, check the quantization dispatch inside the module first; do not change the GEMM.
- Source: [DAPO case: first divergence](../../../cases/nemotron/dapo-20260927.md#further-localization-the-first-divergence-is-the-fp8-input-quantization-of-the-mamba-output-projection).

### P19 Quantizer not in the `nn.Module` tree; patch misses layers

- Symptom: the patch "ran", but the target layer still has 1,361 differing elements.
- Cause: traversing `model.modules()` finds only 6 registered modules, missing quantizers attached to plain objects such as `linear.quant_method.kernel.quant_fp8`.
- Fix: traverse the registered tree, quant-method kernel objects, and that version's legacy fields together.
- Verify: record per worker the hit count (52 after the fix), target layer names, actual `_forward_method`, and quantized output bytes; fail immediately if expected targets are zero or key layers are missing. Private field paths serve only as versioned case examples.
- Source: [DAPO case: r59–r68](../../../cases/nemotron/dapo-20260927.md#candidate-fix-and-full-zero-update-admission-r59r68).

### P20 FSDP BF16 policy overrides the FP32 parameter contract

- Symptom: Model E four-layer fixed-token logprob MAE 0.022982/0.023052.
- Cause: HC and attention sink, kept in FP32 after HF loading, lose precision when entering the uniform BF16 FSDP policy; a later `.float()` inside the op cannot recover it. In addition, the router's FP32 conversion is overridden by the outer BF16 autocast, changing the expert set.
- Fix: explicitly opt in with `wrap_policy.preserve_original_param_dtype=True` (`param_dtype=None`, BF16 autocast kept); this is a whole-model policy, not a per-parameter allowlist, and its communication/memory impact must be measured; it does not upcast existing BF16 base to FP32. Disable autocast inside the router forward.
- Verify: with original dtypes preserved, singleton, two-rank, and unwrapped are fully identical; the router has a regression test.
- Source: Model E training lessons §2 (not included).

### P21 fake-quant truncates gradients

- Symptom: in DSv4.1 LoRA training, 31 of 160 B matrices are zero, all in `self_attn.kv_proj`.
- Cause: `_fake_quant_fp8_block` goes through an FP8 cast in the autograd path, and small gradients underflow to 0; `_fake_quant_fp4_block` has no STE at all (codes go through integer comparisons and a LUT gather, so gradients can flow only through the scale, and the ue8m0 branch breaks even the scale path).
- Fix: keep the quantized forward unchanged and use identity for the input gradient: `output.detach() + (x - x.detach())`.
- Verify: a 1e-6 upstream gradient is preserved exactly; after the fix 160/160 B matrices are nonzero. Loss still rises, so this is not proof of improved training.
- Source: DSv4.1 FP8 STE, FP4 audit (raw history archives, not included).

### P22 Training forward and export quantize at different positions

- Symptom: the model already has FP4/FP8 STE, but the mismatch remains.
- Cause: the existing DSv4.1 STE acts on indexer/compressed/window KV activations, not as weight fake-quant on `W+ΔW`; the training forward is unmerged base + LoRA, while export is an FP32 merge followed by conversion to block32 FP8/E8M0.
- Fix: first confirm whether training sees the post-export rounding, then decide whether weight QAT is needed.
- Verify: sample the actual dtypes inside the forward, not just the code path.
- Source: DSv4.1 Miles precision contract comparison (raw history archive, not included).

### P23 KV cache aligned only on the dtype label

- Symptom: attention results differ with the same weights.
- Cause: in the old vLLM image, KV is FP8/block64 for the non-RoPE part and BF16 for the RoPE part, and the training side handled the numerics differently.
- Fix: set `kv_cache_quantization="fp8_ds_mla"` explicitly on the training side, with the same rounding for forward with and without gradients, and STE to keep gradients. This is not op-by-op equivalent to Miles' implementation; re-verify after upstream KV layout changes.
- Verify: compare attention output with the same weights.
- Source: DSv4.1 LoRA-era mismatch lessons §1 (not included in this public copy).

### P24 Dividing by temperature in BF16

- Symptom: fixed bias in logprob.
- Cause: the old VERL FSDP path divides by temperature in BF16/FP16 first, then upcasts.
- Fix: `logits.float()` first, then divide by temperature; upcast even when temperature is 1.
- Verify: on the same logits, scorers on both sides agree.
- Source: DSv4.1 LoRA-era mismatch lessons §2 (not included in this public copy).

### P25 After resume, derived scales still belong to the old weights

- Symptom: the first rollout after resume is wrong.
- Cause: BF16 parameters were updated, but the FP8 source-scale registry still trusts the initial weights.
- Fix: invalidate derived caches in the weight-restore lifecycle.
- Verify: probs_diff on the first step after resume matches the pre-resume baseline (Nemotron-H's kill→resume acceptance is still 0 after resume).
- Source: [Checkpoint guide 03 §13](../checkpoint-resume-memory/03-pitfalls.md#13-first-step-after-restore-uses-stale-quantization-scales).

### P28 Master starting point lies on the quantization grid

- Symptom: QAT / train-inference consistency training loss barely decreases, NVFP4 deployed weights are byte-for-byte unchanged, while the master is genuinely moving.
- Cause: the master is initialized from the dequantized checkpoint, so everything sits on FP4 grid points, half a spacing from decision boundaries; at small lr, the cumulative movement over a few hundred steps is not enough to change codes, and recomputing block scales does not flip codes earlier either.
- Fix: initialize the master from the true BF16 release and requantize every step with the same rule; θ0 bytes then differ from the official checkpoint, and that cost must be quantified with a proxy SFT comparison and documented. If the original weights are unavailable, apply a fixed in-cell jitter to the master and verify that θ0 bytes are unchanged. Do not use stochastic rounding, a larger lr, or LoRA.
- Verify: bytes identical before and after a zero-update sync → fast-tier RL 0 diff with deployed bytes starting to change → proxy SFT gap to the native baseline within budget ([QAT §6.2](02-qat-and-ste.md#62-fix)).
- Source: [NVFP4 stagnation investigation](../../../cases/nemotron/reports/research-nvfp4-rl-stagnation.md), [proxy SFT three-arm comparison](../../../cases/nemotron/reports/sft-three-arm.md), [DS4 practice differences §E.1](../../../cases/nemotron/reports/ds4-practice-differences.md#e1-correction-2026-10-03-pending-human-confirmation).

## Experiment methodology

### P26 The quantization oracle itself is wrong

- Symptom: GPU block32 quantization gate fails (zero block expected 2.27373675e-13, actual 1.16415322e-10).
- Cause: the oracle assumed Triton's epsilon position; the CUDA UE8M0 path additionally clamps the scale to 1e-10 before taking the power of 2.
- Fix: fix the oracle, not the kernel. This does not imply CUDA and Triton are bitwise identical on tiny/zero blocks.
- Verify: confirm the oracle's epsilon/floor position against the kernel source before declaring failure.
- Source: "Pinned-image GPU block32 quantization gate" in source-23 of the DSv4.1 training history archive (not included).

### P27 Attributing all logprob differences to quantization

- Symptom: the difference does not go away after changing precision.
- Cause: routing choice, attention/kernel, tokenization/mask, and stale adapters all produce differences. R3 constrains routing; it does not make FP8 GEMM equal BF16. TIS is a statistical correction; it does not fix wrong weights.
- Fix: rule out protocol errors first, then record the quantization/numerical noise floor.
- Verify: after ruling out protocol items (routing, mask, adapter version) one by one, measure the quantization noise floor on the same inputs.
- Source: 2026-09-14 Miles audit.
