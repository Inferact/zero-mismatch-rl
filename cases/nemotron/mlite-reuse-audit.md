# Nemotron-H reuse audit of the mLite shared scaffold (read-only, 2026-10)

The placement rules and summary table are in the [training-side module inventory](../../skill/guides/training-model-support/01-module-inventory.md); this document is the full per-file table.

Megatron-LM mLite branch HEAD as of the internal Megatron-LM fork PR (2026-10). Paths are relative to `experimental/lite/megatron/lite/` in the repository; `N = model/nemotron_h`, `P = primitive`.

In one sentence: Nemotron-H has 3824 lines in total, and the only real reuse of `primitive/` is the protocol-layer orchestration tools (parallel/thd/cp/recompute/optimizer/ckpt export/bundle), `P/config.load_hf_config_dict`, and `ops/logprob.vocab_parallel_entropy`. All modeling modules (norm, linear, router, experts, dispatcher, attention, HF loader) are written as a separate set under `N/vllm/primitive/`. DS4 is the opposite: `primitive/modules/*` and `primitive/ckpt/hf_weights` are assembled directly.

## 1. The 20 files

| file | lines | implements | shared counterpart | reuse status | verdict |
|---|---|---|---|---|---|
| `N/__init__.py` | 3 | Exports Config | Same as DS4 | — | Keep |
| `N/config.py` | 155 | HF config dataclass, validation | `deepseek_v4/config.py` | `P/config.load_hf_config_dict` | Keep (model dir) |
| `N/quantization.py` | 207 | NVFP4/FP8 requantize, encode/decode, master version, `load_quantized_weight` | `P/quantization/{mxfp4,qat,block_fp8}.py` | None. The E2M1 grid at :8 duplicates `mxfp4.py:31-36`; `fp8_encode` at :50 duplicates `qat.py:375`; :176-207 hand-reads index.json, duplicating `P/ckpt/hf_weights.py:172 SafeTensorReader` | Move the numerics to a new `primitive/quantization/nvfp4.py` (Nemotron's E2M1 is RNE while mxfp4 is ties-down, so it needs parameterization); IO uses SafeTensorReader |
| `N/checkpoint.py` | 486 | HF name mapping, FP8 KV scale, Export spec, BF16 master loading and reversibility check, theta0 gate, a self-written `load_hf_weights` | `P/ckpt/hf_weights.py:904 load_hf_weights`, `:115 HFWeights`, `:172 SafeTensorReader` | Export reuses `export_hf_weights`; load is entirely self-written (:410-486, :174-183, :47-75), with 4 places each reading weight_map | Ordinary tensor load goes through `load_hf_weights` + WeightSpec; keep the quantized-master check and theta0 gate. The top level importing back into `vllm/primitive/*` (:19, :95, :129, :191, :420) is a layering inversion |
| `N/vllm/__init__.py` | 1 | — | `deepseek_v4/lite/` | — | The impl name should be `lite` |
| `N/vllm/model.py` | 188 | Layer/Model assembly, PP stage | `deepseek_v4/lite/model.py` | None. norm, Embedding, lm_head self-built; PP split computed by hand in protocol.py:330-333 | Keep assembly; PP split uses `P/parallel/pp.py:162 build_pipeline_chunk_layout` |
| `N/vllm/protocol.py` | 409 | mlite protocol, contract checks, BI checks, forward_step, FSDP2/dist_opt, post-step requant | `deepseek_v4/lite/protocol.py` | Much reuse: ModelBundle, init_parallel, cp, thd, recompute, fsdp2/megatron_wrap optimizer, ckpt, export_hf_weights, nested_from_packed | Keep. :110/:125 hard-code the 52-layer pattern, :166 hard-codes (2688,128,1856,6): a single-checkpoint implementation |
| `N/vllm/primitive/__init__.py` | 1 | — | — | — | The directory name collides with `megatron.lite.primitive`; rename or split it up |
| `N/vllm/primitive/dense.py` | 647 | (1) generic bridge of visible forward + BF16-master VJP :20-89; (2) VJPs and modules for RMSNorm/residual/gated :91-252; (3) Humming/CuTe NVFP4 linear :290-391; (4) CheckpointProjectionFactory :394-478; (5) Nvfp4/Fp8TrainingLinear :481-632 | `P/transformer_engine.py:33 Linear`, `:45 RMSNorm`; `P/utils/moe.py:27 _te_general_gemm`; `P/quantization/qat.py:653 WeightFakeQuant` | None. `native_linear_vjp` :20-32 duplicates `_te_general_gemm`; RMSNorm does not use te.RMSNorm because the forward must align with vLLM `cuda_rms_norm`; :406-430 hand-reads weight_map again | (1)(2) are model-independent → `primitive/ops/aligned_vjp.py`; (3)(5) serving kernel adapters can stay; (4) IO uses SafeTensorReader |
| `N/vllm/primitive/logprob.py` | 36 | Chunked lm_head + vLLM `compute_token_logprobs` forward + FP32 log_softmax STE | `P/ops/linear_cross_entropy.py:45`, `P/ops/logprob.py:10` | `vocab_parallel_entropy` | Keep; the DS4 `aligned_selected_log_probs` cited in the docstring does not exist anywhere in the repo |
| `N/vllm/primitive/attention/backward.py` | 46 | `_Fp8AttentionVJP`: dequantize Q/K/V, recompute BF16 O/LSE, use FA4 deterministic bwd | No FP8-KV variant | None | Model-specific, keep |
| `N/vllm/primitive/attention/module.py` | 185 | `Fa4Fp8KVAttention` (replicates vLLM FlashAttnFixedSplit 6768-token page, FP8 Q/KV, split-KV); thin GQA wrapper | `P/modules/gqa.py:45 GQAttention` (does not support disabling rope/qk-norm, or FP8 KV) | None | Keep; reusable once GQAttention supports `rotary=None, qk_norm=False` |
| `N/vllm/primitive/mamba/backward.py` | 105 | `_PackedConvVJP` (causal_conv1d bwd + fixed-order FP32 dW), `_PackedScanVJP` (mamba_ssm bwd) | None (no mamba in P) | — | Model-specific, keep (core value) |
| `N/vllm/primitive/mamba/module.py` | 211 | `SSMMeta`, `packed_conv/packed_scan` (vLLM forward), `MambaMixer` | Structurally analogous to `P/modules/gated_delta_net.py:107`; `SSMMeta` duplicates `P/utils/packed_seq.py:14 PackedSeqParams` and `P/parallel/thd` (the comment at :155 admits it) | None | Keep the scan/conv bridge; a generic Mamba2 should go in `primitive/modules/mamba2.py`; derive SSMMeta from PackedSeqParams/ThdPackMeta |
| `N/vllm/primitive/moe/module.py` | 185 | `_FixedRouteVJP` + Router (vLLM grouped_topk forward), `_CombineVJP`, ReLU² SharedExperts, MoE assembly | `P/modules/router.py:166 SigmoidTopKRouter` (includes router replay); `P/utils/moe.py:213/236`; `P/modules/mlp.py:8 SwiGLUMLP` | None | Router forward keeps vLLM grouped_topk; the VJP at :333-353 has the same math as SigmoidTopKRouter and can be reused; does not support R3 router_replay; SharedExperts should push mlp.py to support relu2 |
| `N/vllm/primitive/moe/grouped.py` | 705 | Pinned FlashInfer private W4A16 signature/sha :18-63, `CuteDslRoutedExperts` :66-270, `Nvfp4ExpertWeights` :278-480, `_te_grouped_gemm` :482, ReLU² `routed_vjp` :500-565, `ep4_routed_experts` | `P/modules/experts.py:64 Experts` (te.GroupedLinear); `P/utils/moe.py:89/160 permute/unpermute`; `P/ckpt/hf_weights.py:1132 _load_expert_weight` | None. :482-497 rewrites TE grouped GEMM; :326-328 hand-reads weight_map | Keep the W4A16 forward + grouped VJP; `ep4_routed_experts` hard-codes 4 ranks |
| `N/vllm/primitive/moe/communication.py` | 251 | DeepEP normal-mode EP: `reduce_ep4_parts` (replicates the FlashInfer one-sided combine summation order), `_deepep_buffer`, dispatch/combine, `EPRoutedExpertsVJP` | `P/modules/dispatcher.py:29 _build_deepep_buffer`, `:60/:140`, `:189 TokenDispatcher` | None. :87-108 duplicates dispatcher.py:29-49; the DS4 `_get_deepep_buffer` cited in the docstring does not exist; :32-58 hard-codes 4 parts, 6 routes, `ids // 32` | buffer reuses `_build_deepep_buffer`; keep the serving-order reduction semantics but parameterize EP size/topk |

(The other 3 `__init__.py` files are docstrings.)

## 2. Violations of docs/porting.md and architecture.md

1. The impl name is not `lite` (registry.py:120 registers `"vllm"`, directory `nemotron_h/vllm/`; the other 5 models are all `<model>/lite/`).
2. Hybrid architecture package: porting.md explicitly says "Keep these out of the first PR: Hybrid model implementation packages".
3. The three-layer split is blurred: the model layer builds its own same-named `primitive/` subtree, mixing generic and model-specific pieces.
4. Dependency direction: top-level `checkpoint.py`/`quantization.py` import back into the impl subtree, and `dense.py`/`grouped.py` in turn import top-level quantization, forming a cycle. DS4 puts checkpoint inside `deepseek_v4/lite/`.
5. `tests/unit/runtime/test_layering_contracts.py:38-45` does not include `nemotron_h`, so the layering guard does not apply.
6. Nemotron is the only model that hard-depends on vllm, FlashInfer private APIs, mamba_ssm and causal_conv1d, and it calls `init_batch_invariance()` in build_model (protocol.py:316-318); the docs do not mention this.

## 3. How DS4 does it

- `model.py` (650) assembles primitives: te.RMSNorm, CSA, HCA/mHC, MTPLossAutoScaler, thd, fp8 recipe; loss/logprob use `P/ops/linear_cross_entropy`. Only assembly and three specific deviations (CSA, mHC, hash routing).
- `moe.py` (102) vs Nemotron `moe/` (637): SigmoidTopKRouter + Experts + TokenDispatcher + SwiGLUMLP; the only specific code is hash-route (wired into R3 router_replay).
- `checkpoint.py` (475): one WeightSpec, with load/export handed to `load_hf_weights`/`export_hf_weights`/`stream_export_to_shards`.
- `resync.py` (125): quantized write-back uses `P/quantization/{block_fp8,mxfp4}`.

## 4. Truly model-specific, should be kept

Mamba2 bridge (the two VJPs in `mamba/backward.py`, `mamba/module.py:164-233`); `Fa4Fp8KVAttention` + `_Fp8AttentionVJP`; W4A16 MoE (CuteDslRoutedExperts, Nvfp4ExpertWeights, ReLU² routed_vjp, sum_route_grads); serving-order reduction (needs parameterization); Humming/CuTe NVFP4 and FP8 serving linear; vLLM forward-value logprob; BF16-master reversibility check and theta0 gate; ModelOpt 4over6 requantize rule.

## 5. Should have been reused but was not

1. safetensors index reading written 5 times → `SafeTensorReader`.
2. Ordinary tensor HF load → `load_hf_weights` + WeightSpec.
3. DeepEP buffer → `_build_deepep_buffer`.
4. TE GEMM wrappers → `_te_general_gemm` / `te.GroupedLinear`.
5. Generic bridge of visible forward + reference VJP → `primitive/ops/`.
6. Router → `SigmoidTopKRouter` + `group_limited_topk`, with the forward swapped to vLLM grouped_topk and wired to router_replay.
7. Shared expert → add an activation parameter to `mlp.py`.
8. Duplicated E2M1 grid and FP8 encoding → new `primitive/quantization/nvfp4.py`.
9. PP split → `build_pipeline_chunk_layout`.
10. `SSMMeta` → derive from `PackedSeqParams`/`ThdPackMeta`.
11. Two docstrings cite DS4 functions that do not exist.
