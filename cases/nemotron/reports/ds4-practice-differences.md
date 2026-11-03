# Differences and compromises between Nemotron and the DS4 practice (verl #7688)

Updated: 2026-10-02. Compared against:
- verl #7688 head `0c849d86`, aligned mode (a local copy of the scripts was kept);
- the `ds4_vllm_align_preview` branch of a community Megatron-LM fork, i.e. DS4's mLite `vllm` implementation.

As the human requested: every place where we did not follow DS4 is listed here.

Commits in the "Commit" columns are on internal fork branches of verl and Megatron; their SHAs are omitted.

## A. Changed to match DS4

| Item | Before | Now | Commit |
| --- | --- | --- | --- |
| batch / length / optimizer | #7688's old head (32, 8192, betas 0.999) | 128×8, mini 32, 2048+14000, overlong 4096, lr 1e-6, betas (0.9, 0.95), wd 0.1 | verl (internal commit) |
| quantization scale | Fixed to the checkpoint value throughout | θ0 uses the checkpoint value, recomputed after the first update (DS4 `invalidate_bound_source_scales`) | Megatron (internal commit) |
| How mLite is selected | `actor@...=mlite_actor` | `model_engine=mlite` + a critic placeholder config generated at runtime (same as DS4) | verl (internal commit) |
| `grad_offload` | Kept (mLite does not recognize it, startup error) | `~...engine.grad_offload` (same as DS4) | verl (internal commit) |
| trainer | `use_v1=True` | `use_v1=False` (same as DS4; v1 needs `transfer_queue`, which is not in the image) | verl (internal commit) |
| `log_prob_use_dynamic_bsz` | Explicitly False | Not set, defaults to following the actor's setting (DS4 does not set it either) | verl (internal commit) |
| `engine.impl` | Registered name `lite`, actually the aligned implementation | `impl=vllm` (same as DS4 aligned mode) | Megatron and verl (internal commits) |
| vLLM config of the training process | `external_launcher` | `"mp"`. DS4 uses the default `VllmConfig` and builds no vLLM process groups; Nemotron must build the groups, see C3 | Megatron (internal commit) |
| EP4 expert combine | Runtime replacement of the Humming expert's `apply` and `humming_forward` | mLite explicitly calls each stage in order and only replaces the final weighted sum (DS4's approach is to call vLLM kernels explicitly in its own module) | Megatron (internal commit) |
| Refreshing the deployment after init | Replace the optimizer's `reload_model_params` | Use the runtime's `post_model_load_hook` (same as DS4's `_post_dist_opt_model_load`) | Megatron (internal commit) |
| actor parameter offload | `param_offload=True` | `param_offload=False`, `optimizer_offload=True` (same as DS4). With param offload on, moving parameters changes the data_ptr of the quantized master, so the pre-export check errors (r8) | verl (internal commit) |
| How changes get into the container | Rebuild the image on every Python change | Per-file sync with before/after SHA verification, following the infra's patch-sync procedure | `scripts/sync_python_patch.py`, `apply_python_patch.sh` |

## B. Different from DS4, but with a reason (needs human confirmation)

| Item | DS4 | Nemotron | Reason |
| --- | --- | --- | --- |
| `ppo_max_token_len_per_gpu` | 4096 | 16384 | verl requires each micro-batch to fit the longest sequence (`seqlen_balancing.py:384`); Nemotron's training side has no CP |
| `rollout.full_determinism` | True | False | When True, the 8 responses of the same group get the same seed (`vllm_async_server.py:632-633`, a cross-review finding) |
| Training topology | EP4 plus FSDP2 optimizer | PP4, EP1, dist_opt | The NVFP4 actor requires world size = PP = 4 |
| Deployment format | BF16 parameters converted to FP8 on the fly; experts changed from FP4 to FP8 | Keep the checkpoint's NVFP4 W4A16 + FP8 | The task requires deploying Lightning NVFP4 |
| master precision | BF16 parameters + FP32 optimizer master | Parameters of quantized layers are themselves the FP32 master; ordinary layers are BF16 | Quantized layers encode FP4 directly from FP32, skipping one BF16 rounding |
| Quantizer | `per_block_cast_to_fp8` (ue8m0) | vLLM native `scaled_fp4_quant` (global = amax/(6×448)) and per-tensor `scaled_fp8_quant` | Different formats; each uses vLLM's native quantizer for its format |
| Weight sync | On GB200, verl's native `vLLMColocateWorkerExtension`, relying on verl's built-in FP8 quantization handling (runtime patch) | `LayerwiseReloadWorkerExtension` in `verl_mlite`, with vLLM's `copy_immediately` | verl does not understand ModelOpt mixed precision; DS4 on h100 also uses verl_mlite's worker extension. Another agent is working on a vLLM-native eager scheme; re-evaluate when its conclusion is in |
| Package layout | `deepseek_v4/{lite,vllm}/` | Only the registered name changed, no `vllm/` subdirectory created | The human chose the minimal change (Nemotron has no lite implementation) |
| `max_num_batched_tokens` | 8192 | 16384 | Does not affect results under BI; existing chunk2048 cross-run evidence |

## C. Unresolved compromises (blocking or needing a decision)

1. **[Human decided, 2026-10-02: keep, consistent with DS4] The upstream verl_mlite engine patches at import time**: `mlite_engine.py:43` calls `_patch_bucketed_weight_sender()`, replacing verl's `BucketedWeightSender` (prefetch). DS4 uses the same engine, so the DS4 practice itself carries this patch. This conflicts with the handoff note's "no monkeypatch in the delivery", and needs a human decision: keep it (consistent with DS4), or change it to an explicit interface. Also, an earlier agent's sync gates did not load the engine, so that batch of evidence does not cover this patch.
2. **vLLM layerwise reload has a branch added specifically for Nemotron** (`_preserve_checkpoint_scale_dtype`), and a new `copy_immediately` option. This is part of our vLLM PR; DS4 does not need it.
3. **The training process needs vLLM process groups**: Nemotron directly reuses vLLM's layer modules (FusedMoE/Humming, ModelOpt linear), which need `get_tp/ep_group`. DS4's modules are self-written, so it does not need them.
4. **Humming GEMM type**: in the earlier gates, training used `auto` and inference used `indexed`, because the image's environment variable was set to `auto`. Now the recipe unifies both sides to `indexed`; the training side still needs verification.
5. **Gate scripts and the recipe have different environments**: the gate scripts turned off TF32 manually and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`; verl sets `:16:8`. Turning off TF32 is now the protocol's responsibility; later verification goes by recipe runs.
6. **Backward contract name** `moe-fixedscale-…`: verified that "fixed-scale" means the scale is treated as a constant in backward and not differentiated ("fixed-scale identity STE", `fp8_attention.py:45`), not that the scale stays unchanged throughout training. After scale recomputation the name is still accurate; not changed.
7. **The NGC image shinit problem**: every container needs `-e BASH_ENV= -e ENV=`, otherwise NCCL gets switched to a cloud-provider NET plugin. The delivery image should fix this at the root. DS4's Dockerfile does not handle it; its runtime environment does not use that plugin.
8. **[Resolved, Megatron internal commit] Very high memory peak of the native Mamba backward at 16k length**: the cause was that the reference VJP's inter-chunk recurrence did a dense broadcast (quadratic in chunk count, 32.5 GiB at 16k); changed to per-chunk computation, bitwise identical to HF, peak 6.7 GiB; gate4 long16k has finished ([`gate4-vjp.md`](gate4-vjp.md) §12). Original text: the gate4 test tool OOMed at 16k by requesting 31.5 GiB at once. The peak under formal PP4 training still needs measuring.
9. **MCore FP32 shard detach fix** (Megatron internal commit): the same as NVIDIA/Megatron-LM#7013, which is not merged. DS4 does not need it because all its parameters are BF16.
10. **mLite allows a 5-layer proxy to take the PP4 full training path** (Megatron internal commit): so that the formal recipe can be run with the proxy.
11. **Full-model memory**: after switching to `param_offload=False`, the FP32 quantized master stays resident on the GPU during rollout. Each PP stage has about 13 layers, the expert master is about 5 GB per layer, plus vLLM's 0.7 memory fraction; whether the full model fits still needs measuring. DS4 is unaffected because its parameters are BF16 and sharded by EP.
12. **The export path goes through C1's patch**: r8's call stack shows the export goes through `produce`/`next_bucket` in `verl_mlite/compat.py`, i.e. the replaced prefetching sender.

## D. Deviations added after 2026-10-02 (all decided by an agent within the review window; need human re-review)

1. **Scale refresh differs from DS4** (replaces row 2 of table A): DS4 recomputes the scale after the first update. Nemotron did this at first too (full-r1), and the result was that FP4 codes changed by about 40% (rel L2 6%) without any weight update at all, and after the first step the q99 of |Δlogprob| reached 0.79. The cause is that the master is dequantized from the ModelOpt checkpoint, and the scales recomputed by the vLLM quantizer differ from the original ones. Changed to `grow_scales`: only grow the scale on overflow, otherwise keep the original value (two Megatron internal commits); drift dropped to q99 0.44. DS4's BF16→FP8 ue8m0 recomputation is idempotent, so it does not have this problem.
2. **The MoE backward GEMM uses TF32 truncation, not FP32**: the contract is changed to v3 `moe-fixedscale-grouped-tf32rz-bf16edges-v3`. What changed is the documentation, to match the arithmetic actually executed; neither the kernel nor the thresholds changed (step7 review F1).
3. **gate4 has 110 entries judged FAIL under the pre-declared T1**, all kept in the record. The agent judged none of them to be implementation errors, and step 7 was cleared:
   - F2: the composed layer has a different addition order; passes after switching to a T0 bitwise reference;
   - F3: PyTorch's embedding backward at ≤3072 tokens;
   - F4: FTZ in TE FusedAdam.
4. **The effective weight decay is about 19% larger than nominal**: the decay factor 1−lr·wd is rounded in FP32, and lr 1e-6 × wd 0.1 = 1e-7, below FP32's resolution near 1. This is framework behavior; not changed.
5. **D5**: when dp×cp=1 and expert_dp=1, turn off `overlap_grad_reduce` to avoid the MCore golden grad-ready assertion (r10). Affects performance only.
6. **Rollout `gpu_memory_utilization` is 0.35**, DS4 uses 0.7: the FP32 quantized master stays resident on the GPU (C11), which limits KV capacity.
7. **The exactness observer and fail-closed exist only in the diagnostic branch**; the delivered recipe has no online bitwise check.

## E. Human decision (2026-10-03): scale refresh switches to mainstream scheme A

`grow_scales` (scheme B) is no longer used. Instead, consistent with NeMo-RL / TE: from θ0 on, requantize from the dequantized master with the standard NVFP4 / FP8 rules (global = amax/(6·448), block scale = block amax/(6·global), represented in E4M3; FP8 per-tensor = amax/448). Every later step uses the same rule.

Cost: θ0 is no longer the official checkpoint's codes, with about 6% rel L2 deviation relative to the official one.

Requirements:
- The θ0 rollout must also use the requantized weights;
- The training-side quantization result must be synced to vLLM before the first rollout.

To be implemented, and re-accepted on the delivery image.

### E.1 Correction (2026-10-03, pending human confirmation)

The checkpoint's quantization rule was found in the TE source: NVFP4 4over6 with an E4M3 upper bound of 256 (`TransformerEngine/transformer_engine/common/cast/nvfp4/quantize_4over6_kernel.cuh`). The TE docs say this rule targets RL and post-training scenarios.

Requantizing the dequantized master with it gives a result numerically identical to the official checkpoint (see the requant results notes, not included). So scheme A should adopt "the 4over6-256 rule, recomputed with the same rule every step", instead of the standard 448 single-candidate rule. That way θ0 equals the official checkpoint, with no 6% loss.

> **Further correction (late 2026-10-03, supersedes "numerically identical" and "θ0 equals the official checkpoint" in the previous paragraph)**: the previous paragraph only holds on **dequantized** weights (on grid points both MSE and MAE pick the original candidate back, so the test cannot distinguish rules). Starting from the real BF16 release:
> - The delivered TE `NVFP4Quantizer(4over6, e4m3_max=256, err_mode="MSE")` reproduces the checkpoint: packed bytes 99.776%, block scale 99.772%, global 100% (stored as `amax/1536`), dequantized values 99.929%; a hand-written MSE reference gets 99.967%; no TE setting closes the remaining gap (see the BF16-source results notes, not included).
> - FP8 (scheme A, amax/448): 24 of 46 FP8 tensors (with ModelOpt-calibrated scales) differ between θ0 and the checkpoint.
> - So the delivered θ0 (BF16 master scheme A) is **not equal** to the official checkpoint bytes; for the cost see the SFT comparison (A-bf16 3.4685 vs C-bf16 3.4373, about 0.03 nat). Scheme A was confirmed by the human (2026-10-03); the BF16-master θ0 strategy is still on the human confirmation list ([`skill-admission.md`](skill-admission.md)).
