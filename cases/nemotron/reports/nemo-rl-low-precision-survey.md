# NeMo-RL low-precision rollout vs training survey

Source: a local NeMo-RL checkout (a4e21a8, 2026-07-26) + `gh` queries against upstream (PR #3566 was merged on 2026-09-19, later than the local checkout). Read-only survey; no GPU runs.

## 1. FP8 rollout (vLLM FP8) + BF16/FP8 training
- rollout: vLLM loads BF16 weights and then quantizes online to DeepSeek-style FP8, weights in 128x128 blocks, activations **dynamic** per-token-group (1x128); the code has `assert activation_scheme == "dynamic"` and does not read the checkpoint's input_scale. Optional pow2 scales and keeping the first/last N layers in BF16.
  - `docs/fp8.md`; `nemo_rl/models/generation/vllm/quantization/fp8.py` (`FP8_BLOCK_QUANT_KWARGS` L33, assert L596); there is also `mxfp8_utils.py`.
- Training: Megatron + TransformerEngine, `fp8_cfg.fp8_recipe: blockwise|tensorwise`; the sample config defaults to blockwise (granularity aligned with rollout). On Hopper the recommendation is FP8 training + FP8 inference; on Blackwell FP32-scale blockwise training is not supported → the recommendation is **FP8 inference + BF16 training**. The docs recommend also turning on `use_importance_sampling_correction: true`.
  - `examples/configs/grpo_math_8B_megatron_fp8.yaml`; Llama-8B results: step time -15~25%, val curves match over 1000 steps.

## 2. NVFP4 (PR #3566, merged 2026-09-19)
- "TE NVFP4 training with per-token vLLM rollout": on the training side TE does NVFP4 W4A4 only for the routed-expert MLP, with **per-token dynamic scale** activations (`NVTE_NVFP4_ROW_SCALED_ACTIVATION=1`, RHT/2D/stochastic rounding off, backward goes through dequantized); rollout uses the new `nvfp4_pertoken` vLLM quantization mode (vLLM 0.26 `make_nvfp4_moe_kernel(per_token_activation=True)`), **with no static activation calibration needed**. refit transfers BF16, and the vLLM worker quantizes at load time. attention/router/shared expert/first 2 layers/last 4 layers stay BF16; the validation config has router replay on.
  - `docs/design-docs/te-nvfp4-per-token-rollout.md`; `nemo_rl/models/generation/vllm/quantization/nvfp4_pertoken*.py`; `examples/te_precision/attn_bf16_mlp_nvfp4.yaml`; recipe `grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml`.
  - Only validated on Qwen3-30B-A3B (GB200): rollout 1.40x, step time -14%. Requires the vLLM `RoutedExperts` container; Mamba is out of scope. The PR also fixed a ModelOpt refit regression under vLLM 0.26 caused by Nemotron-H going through `AutoWeightsLoader`.

## 3. QAT (ModelOpt "QARL")
- The training side uses ModelOpt fake-quant (quantize in forward, STE in backward); rollout has two modes: fake-quant (default, transfers folded BF16 weights) or `real_quant: true` (transfers packed NVFP4 weights and scales, using vLLM's real ModelOpt NVFP4 kernels). The `quant_cfg` of policy and generation must match; mismatched modes raise directly.
  - `docs/guides/quantization-aware-rl.md`; `docs/design-docs/modelopt-real-quant-architecture.md`; PR #2983 (MoE+Mamba real-quant); `nemo_rl/modelopt/`.
  - Validated: W4A16 GRPO converges; generic W4A4 GRPO has known convergence issues; W4A4 on routed experts only (`nvfp4_experts.yaml`) ran 300 steps to completion on Qwen3-30B-A3B.

## 4. Activation quantization: do the training side and vLLM agree, and how is the static input_scale handled
- FP8 route: both sides use dynamic activation scales (vLLM 1x128 group, TE blockwise). No checkpoint-provided static input_scale is loaded; everything starts from BF16 and is re-quantized.
- NVFP4 per-token (#3566): both sides are dynamic per-token.
- ModelOpt QARL: activations use a block-16 dynamic E4M3 scale plus a **static global scale (amax)**, calibrated at initialization **on the training side** with `algorithm: max` (default 512 cnn_dailymail samples), saved as quantizer state, and sent to vLLM with the weights on every refit (for W4A4, one input scale per projection/per expert). The vLLM side only runs dummy data through initialization once, then overwrites with the amax sent by the actor (`vllm_quant_patch.py` L91). So both sides use the same static scale, sourced from the training side's own amax, not from an external checkpoint.
- No path was found that "directly loads a ModelOpt FP8 W8A8 / Nemotron-H NVFP4 mixed-precision checkpoint and keeps its static input_scale". The real-quant exporter explicitly rejects static weight quantizers (they cannot be reconstructed after TP/EP gather). The W4A8 recipe (`nvfp4_w4a8_fp8.yaml`: FP8 per-tensor static amax activations) can only go through fake-quant.

## 5. Nemotron-H / hybrid Mamba low precision
- There are QARL recipes for Nano3 (30B-A3B) and for another large MoE model: `grpo-nanov3-30ba3b-4n4g-megatron-qa-nvfp4-w4a16-real.yaml`, and 16-node w4a16 and w4a4 real-quant 300-step recipes for the other model (plus a BF16 control).
- In these real-quant recipes, `real_quant_ignore` keeps `*.mixer.in_proj*`, `*.mixer.out_proj*`, `*mixer.conv1d*`, attention, router, shared expert, lm_head and several whole layers in **BF16**, quantizing only the routed experts. **There is no scheme with FP8 Mamba in/out_proj.** The only one that quantizes the Mamba projections is `nano3_nvfp4_default.yaml` (NVFP4 W4A4, used only with fake-quant in QA-distillation).

## 6. Other train/inference mismatch handling
- IS correction: `use_importance_sampling_correction`, with `truncated_importance_sampling_type` selectable among TIS / `icepop` (tokens outside the interval zeroed) / `seq-mask-tis` (masking by per-sequence geometric mean) (`examples/configs/grpo_math_1B.yaml` L81-88); monitor `gen_kl_error`.
- Router replay (R3): `docs/guides/router-replay.md`, working for Megatron MoE + vLLM.
- Batch invariance / true on-policy: **not yet merged** on main. PR #3015 (Llama BF16 true-on-policy, closed), #3208 / #3531 / #3848 (Megatron inference bitwise, gen_kl=0, none merged, BF16 only), Issue #4039 (RFC: vLLM rollout on the upstream batch-invariant kernels). Everything so far is BF16 only; there is no low-precision + BI combination.

## Implications for the B1 baseline
NeMo-RL's approach is to have the training side use activation quantization of the same granularity as rollout (dynamic per-token/block, or a static amax calibrated by the training side itself and pushed to vLLM), and never to reuse an external checkpoint's static input_scale. In addition, Nemotron-H's Mamba in/out_proj stay BF16 in all their real-quant recipes. B1's W4A16 experts/lm_head only do weight fake-quant, consistent with their W4A16, and that part is fine by itself (they also keep lm_head in BF16). The gap is in the FP8 Mamba in/out_proj: during vLLM inference activations are quantized with the checkpoint's static input_scale, while the B1 training side does not quantize activations, so this part of the error is not reflected on the training side at all. NeMo-RL has no ready-made scheme to copy. Following their line of thinking there are two options: one is static FP8 per-tensor fake-quant of the in/out_proj inputs on the training side using the checkpoint's input_scale (corresponding to ModelOpt's W4A8 fake-quant approach); the other is to revert these two layers to BF16 or switch to dynamic activation scales (corresponding to their ignore list).
