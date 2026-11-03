# Known issues with serving-side LoRA in vLLM

Issues hit on vLLM's LoRA path during Model D day-0 RL (2026-10, NVFP4, LoRA rank 32, verl + vLLM, 16×GB300). RL reloads the base or adapter every step; these paths are rarely exercised in inference serving, and upstream test coverage is thin. When onboarding a new model, check the image in use against this page item by item; whether a fix is in the image is decided by the image's source, not by PR status.

## Find by symptom

| Symptom | Entry |
|---|---|
| With LoRA on, reloading base weights reports a shape error, or no error but output degrades | [MoE reload with LoRA skips post-processing](#moe-reload-with-lora-skips-post-processing) |
| After reload expert weights are unchanged while other layers changed | [Reloading with unwrapped names skips experts](#reloading-with-unwrapped-names-skips-experts) |
| All adapter loads fail through the Rust frontend | [Rust frontend LoraRequest field misalignment](#rust-frontend-lorarequest-field-misalignment) |
| Adapter reread from disk every batch, rollout slows down | [load_inplace flag not cleared](#load_inplace-flag-not-cleared) |
| NVFP4 model with expert LoRA rejected at startup, or switched to Marlin | [NVFP4 expert LoRA supports only per-token activation scale](#nvfp4-expert-lora-supports-only-per-token-activation-scale) |
| Model D qkv / indexer cannot attach LoRA, or the quantization exclusion list breaks with LoRA on | [Model D's five-slice qkv](#model-ds-five-slice-qkv) |
| Output NaN under CUDA graph with LoRA on | [Breakable cudagraph NaN](#breakable-cudagraph-nan) |
| Wrong numerics after eager reload, NaN after GPT-OSS reload, expert LoRA has no effect under BI=1 | [Known unfixed](#known-unfixed) |

## MoE reload with LoRA skips post-processing

RL base sync goes `collective_rpc("reload_weights")` → `GPUModelRunner.reload_weights` → per-layer `_layerwise_process` → that layer's `quant_method.process_weights_after_loading`. With LoRA on, the MoE layer's quant method is replaced by the wrapper class `FusedMoEModularMethod`, whose `process_weights_after_loading` does nothing, so the wrapped quantization method (e.g. Marlin repacking, NVFP4 kernel-format conversion) no longer runs. When the kernel layout and checkpoint shape differ, a shape error is raised; when shapes match, checkpoint-format tensors stay in the kernel as-is, no error is raised, and output silently degrades.

Fix: the wrapper forwards the call to the wrapped method, the wrapped method keeps the LoRA context when rebuilding the kernel, and the LoRA layer keeps using the same kernel object ([vllm#63319](https://github.com/vllm-project/vllm/pull/63319); also on an internal branch). The test is Qwen3-MoE with LoRA on reloading its own checkpoint with perplexity unchanged; it fails on the old code.

## Reloading with unwrapped names skips experts

With LoRA on, wrapped parameter names gain a `base_layer` segment (e.g. `experts.N.down_proj.base_layer.weight`), while checkpoint-format reload (`reload_weights` calls `load_weights`) sends unwrapped names. Most non-expert layers still match; the routed expert name mapping `fused_moe_make_expert_params_mapping` also adds `base_layer.` on the checkpoint side, so no name matches and expert weights are silently skipped.

There is no generic fix upstream. Model D in a public PR makes `load_weights` find wrapped parameters by unwrapped names, and Model A does the same for its non-expert and expert mappings (internal fork PRs, with the LoRA-wrapped MoE scale and pre-MM norm reload in a follow-up fork PR); other models either do the same or have the sender send wrapped names directly. For acceptance, reload expert weights that differ from the current ones and confirm the expert output changed; reloading the same weights cannot detect skipping.

## Rust frontend LoraRequest field misalignment

Python's `LoRARequest` dropped `tensorizer_config_dict` in [vllm#60024](https://github.com/vllm-project/vllm/pull/60024), but the Rust frontend's `LoraRequest` still has it. Both sides encode msgpack arrays positionally; Rust puts `nil` where Python expects `load_inplace`, the engine fails to decode, and all adapter requests through the Rust frontend fail. The fix removes the field and adds a `LoRARequest` case to the existing Python msgpack compatibility test ([vllm#63295](https://github.com/vllm-project/vllm/pull/63295)).

Why it went unnoticed: the Rust frontend's CI filters out LoRA tests with `-k "not lora"`. Before running LoRA RL with the Rust frontend, first send one request with an adapter on the binary in use; the binary must be built from the same version as the Python side.

## load_inplace flag not cleared

When `POST /v1/load_lora_adapter` carries `load_inplace=true`, the server stores the `LoRARequest` used for this load, and all later generation requests naming that adapter reuse it. `load_inplace` stays `True` on it, and `WorkerLoRAManager.add_adapter` reloads whenever it sees this flag, so every batch that uses the adapter rereads it from disk. RL that pushes a new adapter every step hits exactly this. The fix is to clear the flag on the stored request after a successful load, in both the Python and Rust frontends; the upstream fix for the same issue, [vllm#41482](https://github.com/vllm-project/vllm/pull/41482), is not merged (Python only), and an internal branch has it fixed.

## NVFP4 expert LoRA supports only per-token activation scale

The production path is FlashInfer TRTLLM routed MoE: the W13 delta is fused into FC1 before the activation function via `gemm1_lora_delta`, and the W2 delta is computed outside the kernel from the FC1 activation the kernel returns ([LoRA guide §1](README.md#part-1-1a-serving-lora)). When the FC1 input uses per-token scale (`kNvfp4DynamicToken`, ModelOpt's `moe.activation='nvfp4_per_token'` or online `nvfp4_per_token` quantization), the returned activation carries its scale and the W2 delta can be computed; with static scale the kernel returns packed FP4 activations without block scales, the W2 delta cannot be computed, and this combination must be rejected at startup rather than computed wrongly.

Upstream [vllm#63024](https://github.com/vllm-project/vllm/pull/63024) adds `TrtLlmNvFp4LoRAExperts`, which requires explicitly selecting `moe_backend="flashinfer_trtllm"`; `auto` still selects Marlin with LoRA on (W4A16, a deviation from the production path, see LoRA guide §1). Model D's 200-step GRPO used the same class on the internal branch, TP4, per-token FC1 scale. When the checkpoint uses static scale, first check whether it can be swapped for a per-token version before deciding to accept the deviation.

## Model D's five-slice qkv

Model D's sparse attention layers fuse q, k, v, index_q, index_k into one projection, so LoRA needs five slices, each distributed to TP ranks the same way the base weight is split (index_q is split by KV head like k/v, index_k is replicated on each rank); the first few dense attention layers are still three-slice qkv and drop the two always-empty indexer entries under the same five-entry mapping.

The pitfall is `packed_modules_mapping`: the class-level mapping is also used by the quantization exclusion list, and changing the class attribute directly to five entries changes which modules are left unquantized. So the five-entry mapping is set per instance and the class attribute is left untouched (a public PR). When changing the mapping for another model's fused projection, first check who else reads this mapping.

## Breakable cudagraph NaN

In images older than [vllm#59941](https://github.com/vllm-project/vllm/pull/59941), the breakable cudagraph path produces NaN with LoRA on. When using an old image, first confirm it contains this fix; if not, switch images — do not work around it with eager (CUDA graph must be on).

## Known unfixed

No fix as of 2026-10-23; when hit, handle per the conclusions here and do not re-investigate:

- After LoRA wraps experts, reloading with checkpoint original names skips expert weights (above; except Model D and Model A).
- In eager mode reload uses stale scales: quantization scales are not updated together with the weights. The formal path with CUDA graph on is unaffected, but results are untrustworthy when eager is used as a comparison.
- GPT-OSS outputs NaN after reload.
- Under `VLLM_BATCH_INVARIANT=1`, expert LoRA may not take effect. When the BI arm enables expert LoRA, first confirm with a scaled-up nonzero adapter that expert output changes with the adapter ([minimal verification order](README.md#3-minimal-verification-order)).

## Common traits

These issues share three traits; use them when onboarding a new model:

1. **Paths only RL exercises.** Reload, in-place loading and per-step adapter swaps almost never happen in inference serving, and upstream tests only test a single load at startup. Every path you use must be exercised at least once on the actual image.
2. **Silent.** Skipped weights, stale scales and missing post-processing mostly raise no error. Verify with weights different from the current ones or a scaled-up nonzero adapter, and check that the output actually changes.
3. **The wrapper layer breaks the base layer's contract.** The LoRA wrapper changes parameter names, quant method and module mapping; all code that iterates over these (loading, post-processing, quantization exclusion) must be checked for whether it passes through the wrapper.
