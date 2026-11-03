# QAT quantizes only weights while the deployment layer also quantizes activations: survey

## Conclusion
1. **It is a common gap.** QAT in open-source RL is basically all weight-only (W4A16), because the rollout side is A16 anyway, so the problem does not exist there. As soon as the deployment layer also quantizes activations (W8A8 / W4A4) while the training side only fake-quantizes weights, the activations on the two sides do not match, and this part of the deviation goes straight into the train-rollout mismatch. Frameworks handle it in only two ways: either the training side also applies Q/DQ to activations, with scales consistent with rollout; or they switch to native FP8/FP4 training so that the training forward and rollout go through the same quantization. No one knowingly leaves the gap unaddressed.

2. Frameworks
- **verl (local)**: `verl/utils/modelopt/quantize.py` only supports `w4a16` and hard-codes `"*input_quantizer": {"enable": False}`, so activations are not quantized. `qat_weight_exporter.py` only exports input_scale when an input_quantizer exists, which it does not under normal configurations.
- **verl (upstream)**: PR #6992 (W4A8, draft) fake-quantizes activations on the training side to **dynamic FP8 E4M3 1x128 blocks**, and the vLLM side applies the same Q/DQ to the GEMM input, using a shared implementation. PR #7971 (native NVFP4 W4A4) does not do QAT; it uses TE NVFP4 training instead, and both training and vLLM use **dynamic per-token** activation scales. PR #7895 is MXFP4 experts QAT for DSv4. None of the three uses the static input_scale from the checkpoint.
- **miles**: INT4 QAT is W4A16 (`docs/advanced/int4-qat.md`), so there is no activation gap. FP8 goes through native FP8 training, `--fp8-recipe blockwise` or MXFP8, and the training forward and rollout have consistent quantization configurations (`docs/advanced/fp8-low-precision.md`). FP8 refit requires `activation_scheme == "dynamic"` (`quantizer_fp8.py:19`). DSv4's KV uses `fp8_simulate` for per-token FP8 STE (`miles_plugins/models/deepseek_v4/ops/qat.py`). NVFP4 export fills `input_scale` with 1 directly (`quantizer_nvfp4.py:138`), which shows the default is dynamic activations without a static scale.
- **slime**: same origin as miles. INT4 fake-QAT is W4A16; FP8 uses TE blockwise native training with online activation quantization; refit likewise asserts `activation_scheme == "dynamic"` (`slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py`, `docs/en/advanced/low-precision.md`).
- **ModelOpt**: the standard `FP8_DEFAULT_CFG` (`w8a8_fp8_fp8`) and `NVFP4_DEFAULT_CFG` (`w4a4_nvfp4_nvfp4`) **both quantize the input_quantizer**. FP8 is a per-tensor static amax obtained by `max` calibration. NVFP4 is a 1x16 dynamic block plus a static global scale. So in ModelOpt-native QAT the activation scale comes from calibration, and both weights and activations are in the loop.

3. Common fixes
- (a) Fake-quantize activations on the training side, **using the checkpoint's static input_scale directly as the scale**. Bitwise the same basis as the server, lowest cost.
- (b) Switch both the training side and the server to dynamic per-token FP8 activations (the route of verl #6992 / #7971, miles, slime). Requires changing the rollout quant config.
- (c) Turn on the input_quantizer with ModelOpt's default W8A8 config, recalibrate amax or let it update during training, then send the new input_scale down with refit.
- (d) Switch to native FP8 training (TE), so the training forward and rollout go through the same GEMM quantization.
- (e) Do nothing and rely on TIS/MIS to absorb the mismatch. This only mitigates; it cannot achieve true on-policy.

4. Recommendation for B1: use (a) for Mamba in/out_proj. On the training side, apply FP8 E4M3 per-tensor Q/DQ (STE backward) to the inputs of these two layers using the checkpoint's static `input_scale`, and keep fake-quantizing the weights. input_scale stays fixed and is passed back unchanged at refit, with no recalibration. The W4A16 experts/lm_head need no change.

## References
- Local verl (a verl fork worktree): `verl/utils/modelopt/{quantize.py,qat_weight_exporter.py,vllm_modelopt_patch.py}`
- verl PRs: https://github.com/verl-project/verl/pull/6992 , /pull/7971 , /pull/7895 , /pull/6455
- miles (local checkout): `docs/advanced/{int4-qat.md,fp8-low-precision.md}`, `examples/low_precision/README.md`, `miles/backends/megatron_utils/megatron_to_hf/processors/{quantizer_fp8.py,quantizer_nvfp4.py}`, `miles_plugins/models/deepseek_v4/ops/qat.py`
- slime (local checkout a74ae3a): `docs/en/advanced/low-precision.md`, `slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py`
- ModelOpt: `modelopt_recipes/configs/ptq/units/w8a8_fp8_fp8.yaml` (both weights and activations fp8), `presets/model/{fp8,nvfp4}.yaml`
- Did not check whether NeMo-RL has activation-side QAT; PR search results were only about FP8/MXFP8 refit.
