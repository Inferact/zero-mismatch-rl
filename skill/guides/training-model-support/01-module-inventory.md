# Module inventory and placement rules

What to do on day one: list, per module, the inference kernel, whether the training framework has a same-math implementation, whether there is quantization rounding, whether there is discrete selection, and whether there is a backward; then decide between MCore / FSDP2 / mLite, and where each module goes in the code tree. The Nemotron lesson: only after three weeks did we find that FA4, Mamba scan and W4A16 MoE had no backward; making the inventory up front would have avoided that.

Training support means a training implementation that we maintain and that does correct backward. Config registration, same-family fallback and the vLLM inference implementation are not it; third-party implementations that appear in the image only after release (new Transformers modeling, NeMo AutoModel, etc.) are not it either, for the reasons in the [README](README.md#what-you-have-on-release-day). Write it yourself per [02](02-reproduce-forward.md), into real source files, then verify by loading it in a new process through the actual loader; an implementation that exists only in a REPL or a runtime monkeypatch does not count as passing evidence.

## 1. How to list the inventory

One row per module, fill in all columns below; write unknown for blanks, never "should exist".

| Column | Question to answer |
|---|---|
| Inference kernel | The function vLLM actually calls (not the class name in the model file), including quantized variants, attention backend, and branches under cudagraph |
| Training-framework same-math implementation | Model-agnostic libraries: mLite `primitive/modules`, `ops`; MCore / TE; mamba_ssm, causal_conv1d. Differing only in rounding counts as same math; a different activation clamp definition does not. Same-model implementations that appear only after release do not go here |
| Quantization rounding | Whether the forward has FP8/NVFP4 rounding; list weights and activations separately |
| Discrete selection | Top-k expert IDs, sparse indices; IDs are fixed and not differentiated, continuous scores need derivatives |
| backward | Whether the library has one; if so, whether its intermediates (LSE, rstd, token permutation) match what our forward saves |
| Conclusion | Reuse / glue / write new (decision rules in [03](03-reuse-backward.md)) |

Nemotron-H filled in:

| Module | Inference kernel | Same-math training impl | Rounding | Discrete | backward | Conclusion |
|---|---|---|---|---|---|---|
| Mamba SSD scan | `mamba_chunk_scan_combined_varlen` | mamba_ssm | None | None | mamba_ssm `_mamba_chunk_scan_combined_bwd` | Reuse |
| conv1d | `causal_conv1d_fn` | causal_conv1d | None | None | Yes; dW needs fixed-order FP32 | Reuse + small glue |
| Attention (FP8 KV) | FA4 fixed split + FP8 Q/K/V | FA4 BF16 | Yes | None | FA4 only has BF16 backward | Dequantize and recompute BF16 O/LSE, then reuse |
| Mamba in/out_proj (FP8), shared expert (NVFP4) | FlashInfer / NVFP4 GEMM | TE `general_gemm` | Yes | None | TE | Reuse after fixing STE |
| Routed experts (NVFP4 W4A16) | CuTe-DSL / Humming MoE | TE grouped GEMM | Yes | Yes | No whole-segment backward | Glue `routed_vjp` |
| RMSNorm / gated RMSNorm, routing scores | BI kernel / grouped top-k | PyTorch formula | None | None for scores | autograd | Reuse |
| MTP block, draft | Not part of training | — | — | — | — | See §4 |

## 2. Loadable ≠ trainable

The FSDP2 route saves rewriting the compute graph into Megatron, model-parallel layout and weight bridging; it does not save the model itself. forward, backward, PEFT injection, special parameter precision, initialization/residency, weight sync, checkpoint restore and input semantics all have to be written by us or verified item by item. Being able to construct a model object (whether from `AutoModel` or any library in the image) is not the same as having a training implementation.

### 2.1 The three kinds most easily missed by follow-up fixes

1. **Recompute breaks only gradients, not the first forward.** [Bridge #27](https://github.com/radixark/Megatron-Bridge/pull/27) fixes multi-LoRA recompute replay and the grouped-expert DDP flag; do not use an old-logprob test as backward/recompute acceptance.
2. **Being able to save native shards does not mean being able to export a serving adapter.** [Bridge #33](https://github.com/radixark/Megatron-Bridge/pull/33): shared-outer A/B with different expert dimensions causes `Incomplete adapter export`. Shared-outer and ordinary packed-expert must not be mixed.
3. **Introducing an integrity mechanism once does not mean it is still needed later.** Model B a public PR removes duplicate grouping, but the PR states CI did not enforce cross-bucket, and the correctness argument comes from the accumulator source. Verify the necessary contract, do not skip checks, and do not copy a whole framework upon seeing one historical fix.

**FSDP generality saves sharding adaptation; it does not guarantee every additional training feature is generic.** Miles's follow-up fixes likewise show that after "release-day forward runs" there are still many independent boundaries to accept.

Dependency upgrades also recreate cost: [Megatron #93](https://github.com/radixark/Megatron-LM/pull/93) records that Miles #2706 merged first while the dependency branch did not, so main CI did not recognize the new parameters; adding the CLI field alone was not enough, the fix also included a missed `tid2eid` sync, the shared-expert clamp, and MTP router not participating in target replay. "Dependency PR CI green" is not "main image runs".

### 2.2 Cost map: what can be saved and what cannot

Levels only indicate complexity once triggered, not that every model needs the development: low = config/wiring an existing mechanism; medium = local Python change and targeted verification; high = new autograd/kernel, multiple layout/collective/persistence contracts. Without trigger evidence, mark "to verify"; do not fix preemptively.

| Check surface | Signs that trigger development cost | FSDP + LoRA boundary | Minimal verifiable check | Magnitude |
|---|---|---|---|---|
| Dependencies and model registration | Registration/import path of our model did not take effect and another class was actually loaded; PR dependency not merged; PEFT/TE ABI change | No Megatron spec needed, but still need our model + PEFT + serving combination | Image digest, actual imported file/version, few-layer forward/backward | Low–medium |
| Model forward/backward | Only an inference implementation; inference-only kernel; gradient NaN; detach on a new branch | FSDP does not generate backward; LoRA also needs input gradients through frozen blocks | Few layers, nonzero loss, check intended adapter gradients are finite and the optimizer step actually changes parameters | Medium–high |
| Structure/special parameters | gated activation, latent norm, sink, router bias, HC, compressed index tables | Our model must cover parameters, buffers and runtime-derived state | Config comparison; source weight → post-load value; state inventory beyond named_parameters | Medium |
| Parameter and operator precision | FP32 parameters already rounded by a BF16 wrapper/loader; incomplete quantization exclusion list; FP32 operators downcast to BF16 under the outer autocast | Neither FSDP mixed precision nor the training forward's autocast defines the per-parameter, per-operator precision contract for the model | Under the real autocast check both dtype **and value**; you cannot go BF16→FP32 and claim precision restored | Medium |
| Freezing, initialization, residency | Full tensor moved to GPU before sharding; PLE CPU lookup table gathered; full optimizer created by mistake | FSDP fitting at steady state does not mean init/export/restore peaks fit | Per-phase CPU/GPU peaks; requires_grad/optimizer parameter set; no new full diagnostic gather | Medium–high |
| LoRA injection and semantics | fused QKV/gate, 3D experts, shared factor, ParamWrapper, not going through nn.Linear | PEFT cannot assume all-linear covers every target; both HF and serving adapter schemas must be supported | Resolved target list; zero/nonzero/replacement; A/B/scale/layout comparison | Medium–high |
| Parameter sharding and gradient reduction | TP/EP/SP/CP/PP-specific layouts, shared adapter sum/average, cross-stage state | Plain FSDP DP saves most Megatron cost; enabling model parallelism later needs a separate review | Small single-/multi-GPU gradient and one-step comparison on the current topology | Medium–high, as needed |
| Cross-sample/long-sequence state | packing, convolution boundaries, recurrent state, position/channel inconsistency | The FSDP2 route convention as of 2026-09 is no packing; mLite uses thd packing, and every stateful module must consume sequence boundaries | Score unequal-length A/B separately vs merged; see [04](04-layout-and-length.md) | Medium–high |
| Weight format and quantization | Wrong 3D expert slicing, wrong gate/up order, wrong scale format/axis, only low-precision checkpoint supported | The FSDP2 route saves the HF↔Megatron bridge, but still has FP8/FP4→training and training→serving contracts | Single checkpoint source of truth; per-family value/shape/scale comparison; native sync A–A | Medium–high |
| Sync and lifecycle | Sleeping storage unreadable; buffer not sent; adapter replacement leftovers; cross-bucket state leak | Cannot rely on FSDP to cover exporter/receiver automatically; no new publisher/partial protocol | Native update return, receiver identity, repeated sync, production sleep/wake/graph/MTP | Medium |
| Recompute and persistence | Activation checkpoint recompute uses the wrong adapter; save missing a factor; resume missing optimizer | FSDP/PEFT framework existing does not mean any combination is verified | Few-layer nonzero step→save→fresh reload→next step; compare against an uninterrupted reference | Medium |
| Input/loss semantics | thinking/template/EOS inconsistency, token roundtrip, wrong loss mask/normalizer | Unrelated to FSDP; the system runs but the training objective differs | Actual IDs, effective response mask, temperature and reduction; fixed-sample old-logprob | Low–medium |
| Extra features | MTP training, router/indexer replay, VL/audio, async multi-adapter | Each is an independent support surface; not included by default just because Miles has it | Expand only on user request; verify data alignment/state/gradients/version separately | Medium–high |

### 2.3 Seven questions for the lowest-cost route

| Order | Question to answer | Action on failure |
|---|---|---|
| 1 | Is our own training implementation written into source, loaded by the actual loader, and doing correct backward? | If not, write a minimal implementation per [02](02-reproduce-forward.md); do not substitute config registration or an implementation that appeared in the image only after release |
| 2 | Which side does the packing switch actually take effect on (FSDP2 route `use_remove_padding=False`; mLite thd)? | Stop at the config check; do not let the recipe, config overrides or backend defaults silently switch paths |
| 3 | Is the target LoRA family usable on both training and serving sides? | Mark unsupported per family; see the [LoRA guide](../lora/README.md) |
| 4 | Are the single checkpoint, precision, special parameters and actual training residency consistent? | Locate the specific cast/load/layout; do not first attribute it to "inherent quantization error" |
| 5 | Do few-layer nonzero training, native sync, and save/restore pass? | Localize forward/backward/export/resume separately; do not repeatedly cold-start the full large model |
| 6 | For the same policy/tokens, what is the full-model mismatch vs its own noise? | Check ordinary batch boundaries, weights and semantics first; then do bounded module dumps per [Debugging](../debugging-mismatch/README.md) |
| 7 | Do the full production lifecycle and held-out improvement pass? | Report system correctness and training effect separately; do not use one in place of the other |

For each risk record only: `evidence link / current implementation and version / supported, unsupported or unknown / cheapest verification / owner and magnitude if a change is triggered`. Reuse existing model-agnostic operators and libraries first; unknown does not count as supported. Dynamic batching (grouping by length/adjusting microbatches) is not the same feature as cross-sample concatenation. If the chosen implementation requires packing and the framework path does not support it, report the incompatibility first; do not enable it automatically to pass tests.

## 3. Placement rules

### 3.1 mLite (Megatron `experimental/lite/megatron/lite`)

Read `experimental/lite/docs/porting.md` and `architecture.md` first. Three layers: `runtime` (lifecycle and train-step orchestration), `model` (registration + model-specific build/load/export protocol), `primitive` (parallel state, TP layers, checkpoint conversion, MoE utilities, optimizer wrappers and other reusable pieces). runtime knows no model details: it fetches the protocol from the registry, builds a typed config, and hands model construction to the protocol.

| Rule | Content |
|---|---|
| Model directory holds only four things | `model/<name>/`: config, checkpoint mapping (WeightSpec), model assembly, protocol. The impl name is `lite` (`<name>/lite/`), nothing else |
| Modules go into primitive | Reusable norm / linear / router / experts / dispatcher / attention go into `primitive/modules`; generic VJP bridges into `primitive/ops`; quantization encoding into `primitive/quantization`; PP splitting, thd, CP into `primitive/parallel`; HF read/write uses `primitive/ckpt/hf_weights` (`SafeTensorReader`, `load_hf_weights`, `export_hf_weights`) |
| Model-specific math still goes in primitive and is registered | Pieces other models may also use, such as the Mamba2 bridge or FP8-KV attention, go in `primitive/modules/<x>.py`; do not create a same-named `primitive/` subtree under the model directory |
| No file for reused backward | Modules reusing TE / mamba_ssm backward do not get a separate `backward.py`; necessary glue is written in the module file |
| One-way dependency | `model/` imports `primitive/`, not the reverse; top-level checkpoint/quantization does not import the impl subtree |
| Register with the layering guard | Add the new model to `tests/unit/runtime/test_layering_contracts.py`, otherwise the guard does not apply |
| Do not use mixed-architecture packages | porting.md: "Keep these out of the first PR: Hybrid model implementation packages"; new modules of a hybrid model (Mamba + attention + MoE) are still split into primitive as above |
| Document third-party kernel dependencies | Models that hard-depend on vllm, flashinfer private APIs, mamba_ssm, causal_conv1d must say so in the docs, including calling `init_batch_invariance()` in build_model |

DS4 is the positive example: `model.py` (650 lines) assembles te.RMSNorm, CSA, HCA/mHC, thd, fp8 recipe, with only assembly and three model-specific deviations; `moe.py` at 102 lines = SigmoidTopKRouter + Experts + TokenDispatcher + SwiGLUMLP, with hash-route as the only model-specific code; `checkpoint.py` is one WeightSpec, with load/export delegated to common functions.

### 3.2 MCore / Megatron-Bridge

- Bridge `models/<name>/`: provider + HF↔MCore weight mapping; modules MCore lacks (Mamba/SSM variants, special norms, HC) go into the corresponding `megatron/core` subpackage, not into Bridge.
- Mapping checks values, not just shapes; see [02 §4.3](02-reproduce-forward.md#43-mapping-checks-cover-values-and-semantics-not-just-shape).
- MCore kernels and vLLM may be different math (MoE activation clamp definition); when the difference is in the formula, the BI=0 route will not align either; list it first.
- The Megatron fork is only compatible with Megatron-Bridge v0.6.2; the version is pinned in the image.

### 3.3 FSDP2 (VERL's HF interface path)

- Write the model ourselves: write a PyTorch model following vLLM semantics, wrapped in the Transformers `PreTrainedModel` / config interface, so VERL's loader, PEFT and HF-format save/load work directly. "HF" here means only the interface and weight format, not ready-made modeling in Transformers.
- Delivery form: the model implementation goes into verl mainline, registered in verl to the transformers Auto classes (`AutoConfig` / `AutoModelForCausalLM` etc.), with a verl PR alongside the recipe ([Code and delivery §8](../../practices/code-and-delivery.md#verl-a-thin-recipe-not-a-second-training-framework)). verl's upstream FSDP extension guide advises against copying full modeling into `verl/models`, but that targets models already implemented in transformers; Day-0 has no upstream implementation, so follow this rule and put it in mainline. Determine the registration point from the current checkout's FSDP engine load chain (`get_hf_auto_model_class` → `from_pretrained`).
- Development and delivery use the same verl source and the same registration entry; do not modify the checkpoint's `config.json` / `auto_map` (the low-precision checkpoint is the single source of truth), and do not enable `trust_remote_code` to load modeling shipped with the checkpoint. Load logs must prove that the class actually used is the one we registered, not another same-named implementation in the image.
- VERL's automodel engine (`automodel_actor`) uses NeMo AutoModel's model implementation and is not this route; it also shards with FSDP2, but that is no reason; routes are distinguished by the source of the forward.
- The precision contract lives in the module: the VERL training forward runs under bf16 autocast; computations that need FP32 (position encoding angles, router logits and bias, norm statistics, etc., per vLLM) disable autocast inside the module and specify dtype explicitly.
- Prefer autograd for backward; add operators only when backward is missing or unusably slow (the 160-line PyTorch reference for the Mamba scan), placed next to the model file.
- Full model initialization: VERL FSDP2 by default does `from_pretrained` of the full model on rank 0's CPU and then broadcasts, which does not fit at frontier scale (one frontier model's BF16 base is 1.6 TB). Change to meta → PEFT → LoRA initialized on CPU with the same seed → `apply_fsdp2` → each rank reads only its own Shard(0) rows from safetensors. Routed experts of a low-precision checkpoint are dequantized on read (bitwise identical to offline conversion), so no offline BF16 copy is needed; the text config used for training goes through `model.hf_config_path`. Implementation in verl-project/verl#8257.
- The ABI of `--system-site-packages`, the actual value of `use_remove_padding`, and whether PEFT targets go through nn.Linear are the three common pitfalls of this route.
- Under EP4 + FSDP2, DCP expert shards on each rank have the same key and same offset, so restore cross-reads other ranks' experts; keys must include the EP rank; see the [checkpoint guide](../checkpoint-resume-memory/README.md).

## 4. Parts not in training

MTP block, EPLB redundant experts and draft models are in the checkpoint or the inference process, but not in the training graph. Give them a separate row in the inventory, stating three things:

| Part | Mapping | Weight sync | Facts |
|---|---|---|---|
| MTP block | Not loaded on the training side; WeightSpec explicitly lists it as skip rather than relying on missing-key tolerance; things like `_ignore_unexpected_suffixes` are only for explicitly listed suffixes | Not sent; the receiver's expected list is generated from the manifest, not inferred from the number of tensors received | Nemotron checkpoint has `num_nextn_predict_layers: 1`, MTP block unquantized bf16; usable under BI=0; if used, both arms use it; BI=1 + MTP requires speculative verification logprobs to be bitwise identical to the full forward; decided not to use it this round |
| EPLB redundant experts | Training side has only logical experts; the physical-to-logical expert mapping is on the inference side | All physical replicas of the same logical expert must be refreshed; #57954 leftover: derived parameters become orphaned under EPLB | Integrity checks count logical experts, not physical tensors |
| draft / speculative model | Independent checkpoint, independent mapping | Published separately from target, revision verified separately; shared weights (embedding, lm_head) must state an owner | See the serving validation of the day-0 support workflow (§2.4.1, not included here) and the target/draft row of the [LoRA guide](../lora/README.md) |

General rule: skipped parts must appear as "explicitly skipped" in load logs and the manifest, not as "no error". Partial updates such as sending only the k/v shard of qkv leave the q rows as uninitialized GPU memory; see [02 §4.3](02-reproduce-forward.md#43-mapping-checks-cover-values-and-semantics-not-just-shape).

## 5. Example: Nemotron-H reuse audit against mLite

Full text in [cases/nemotron/mlite-reuse-audit.md](../../../cases/nemotron/mlite-reuse-audit.md) (audited at a fixed commit of the training branch). In one sentence: Nemotron-H is 3824 lines, and the only things that truly reuse `primitive/` are protocol-layer orchestration utilities, `load_hf_config_dict` and `vocab_parallel_entropy`; all modeling modules are rewritten separately under `model/nemotron_h/vllm/primitive/`.

| File (`N = model/nemotron_h`) | Lines | What it does | Verdict |
|---|---|---|---|
| `N/config.py` | 155 | HF config dataclass, validation | Keep (model dir) |
| `N/quantization.py` | 207 | NVFP4/FP8 requantize, E2M1 grid, FP8 encoding, hand-reading index.json | Move numerics to `primitive/quantization/nvfp4.py` (E2M1 RNE vs mxfp4 ties-down needs parameterization); IO uses `SafeTensorReader` |
| `N/checkpoint.py` | 486 | HF name mapping, Export spec, BF16 master reversibility check, θ0 gate, self-written `load_hf_weights` | Ordinary tensors go through `load_hf_weights` + WeightSpec; keep quantized master check and θ0 gate; top level importing back into the impl subtree is a layering inversion |
| `N/vllm/model.py` | 188 | Layer/Model assembly, PP stage | Keep assembly; PP splitting uses `build_pipeline_chunk_layout`, not hand computation |
| `N/vllm/protocol.py` | 409 | protocol, contract checks, BI check, FSDP2/dist_opt, post-step requant | Keep; the hardcoded 52-layer pattern and (2688,128,1856,6) show it is a single-checkpoint implementation |
| `N/vllm/primitive/dense.py` | 647 | visible forward + BF16-master VJP generic bridge; RMSNorm VJP; Humming/CuTe NVFP4 linear; projection factory; FP8/NVFP4 TrainingLinear | Generic bridge and RMSNorm VJP are model-agnostic → `primitive/ops/aligned_vjp.py`; `native_linear_vjp` duplicates `_te_general_gemm`; serving kernel adaptation can stay |
| `N/vllm/primitive/logprob.py` | 36 | Chunked lm_head + vLLM `compute_token_logprobs` + FP32 log_softmax STE | Keep; the DS4 function referenced in the docstring does not exist |
| `N/vllm/primitive/attention/*` | 231 | `Fa4Fp8KVAttention` (replicates FlashAttnFixedSplit 6768-token page, FP8 Q/KV), `_Fp8AttentionVJP` | Model-specific, keep; reusable once GQAttention supports `rotary=None, qk_norm=False` |
| `N/vllm/primitive/mamba/*` | 316 | `_PackedConvVJP`, `_PackedScanVJP`, `SSMMeta`, `MambaMixer` | The two VJPs are the core value, keep; generic Mamba2 should go in `primitive/modules/mamba2.py`; derive `SSMMeta` from `PackedSeqParams`/`ThdPackMeta` |
| `N/vllm/primitive/moe/module.py` | 185 | `_FixedRouteVJP` + Router, `_CombineVJP`, ReLU² SharedExperts | Router VJP math is the same as `SigmoidTopKRouter` and reusable, but router_replay is not wired; SharedExperts should push `mlp.py` to add an activation parameter |
| `N/vllm/primitive/moe/grouped.py` | 705 | FlashInfer W4A16 signature/sha pinning, `CuteDslRoutedExperts`, `Nvfp4ExpertWeights`, `_te_grouped_gemm` rewrite, ReLU² `routed_vjp` | Keep W4A16 forward + grouped VJP; TE grouped GEMM wrapper is duplicated; `ep4_routed_experts` hardcodes 4 ranks |
| `N/vllm/primitive/moe/communication.py` | 251 | DeepEP normal-mode EP, `reduce_ep4_parts` (replicates FlashInfer one-sided combine summation order), `EPRoutedExpertsVJP` | Buffer reuses `_build_deepep_buffer`; keep serving-order reduction semantics but parameterize EP size/topk |

Six violations of the placement rules: impl name `vllm` instead of `lite`; mixed-architecture package; a same-named `primitive/` built inside the model layer; circular imports between `checkpoint.py`/`quantization.py` and the impl subtree; layering guard not extended with `nemotron_h`; third-party kernel hard dependencies and `init_batch_invariance()` not documented.

Should have been reused but wasn't: safetensors index reading written 5 times; ordinary tensor HF load; DeepEP buffer; TE GEMM wrapper in two places; visible forward + reference VJP generic bridge; Router (`SigmoidTopKRouter` + `group_limited_topk`, forward swapped to vLLM grouped_topk, wired to router_replay); shared expert; E2M1 grid and FP8 encoding; PP splitting; `SSMMeta`; two docstrings referencing nonexistent DS4 functions.

Truly model-specific and should be kept: the two Mamba2 VJPs + packed conv/scan; `Fa4Fp8KVAttention` + FP8 attention VJP; W4A16 MoE forward + ReLU² grouped VJP; serving-order reduction (parameterized); Humming/CuTe NVFP4 and FP8 serving linear; vLLM forward-value logprob; BF16-master reversibility check and θ0 gate; ModelOpt 4over6 requantize rules.

<a id="part-1"></a>
<a id="part-2"></a>
## Appendix A: Miles cross-model comparison

> Two source audits dated 2026-09-08 (PR history) and 2026-09-16 (Megatron implementation); version conclusions do not mean the latest upstream was re-verified. Model E's four-layer experiment numbers are in its own case notes (not included here).

### A.1 Scan scope and evidence boundary

Scan window 2026-03-08 to 2026-09-08: model-related commit history on Miles main, model docs/plugins, and the related open/merged PRs of Miles, radixark/Megatron-LM and radixark/Megatron-Bridge. Miles main pinned to `3de96596f`. Checked along three lines: first support → follow-up fixes → dependency repos, not just release blogs.

Covers Qwen3.5/3.6, Model F, DeepSeek V3.2/V4, Model G (three releases), Kimi K2.5, Model C, Model B, Nemotron, Gemma4, Model D. A keyword search for Model E found no corresponding Miles PR; this cannot be taken to mean no adaptation is needed or that it is already supported. Publicly visible searches cannot cover private branches, deleted code or all unlinked fixes.

Status in the table below is from GitHub queries; PR validation results are author-reported, not reproduced by us. We read the main listed PR bodies/file lists, plus key source for Qwen GDN, Model B's conv/LoRA and the DSV4 bridge; not every diff was audited line by line. Do not convert PR total line counts or commit spans into person-days; stacked PRs mix in baseline updates, so porting cost cannot be judged from total diff size.

The second audit directly checked Miles `97122c1ed1b7b27a547662057d30ad981128e83e`'s dtype utilities, Qwen GDN, DSv4 bridge/sink and Model B's plugin, and Megatron `73b54618f7e58e0f25f619bcfecbe2640765475a`'s mHC/Float16Module. On the VERL side, distinguish upstream mechanisms from our runtime: FSDP2 parameter policy, HF model implementation, PEFT, training-side autocast and the export end can each change behavior independently; do not treat a local patch as an upstream default. **Source confirmed ≠ runtime path confirmed ≠ numerical experiment passed.** A function existing does not prove the current recipe calls it; `.float()` existing does not prove there was no earlier rounding; similar file names do not prove the same model variant. The per-boundary dtype contract is in the [precision guide](../precision-quant-qat/01-conversion-chain.md#6-dtype-contract-six-boundaries).

<a id="part-1-2-what-else-was-fixed-after-first-support"></a>
### A.2 What was actually fixed beyond first support

M = merged, O = not merged.

| Model line | First support/main entry | Follow-up support and actual problems | Most important reminder for us |
|---|---|---|---|
| Qwen3.5/3.6 | [Miles #740](https://github.com/radixark/miles/pull/740) M: GDN plugin, HF bidirectional mapping, fused experts; [#1022](https://github.com/radixark/miles/pull/1022) M: 3.6 | [#852](https://github.com/radixark/miles/pull/852) M: SP/CP gradients and tied embedding conversion; [#885](https://github.com/radixark/miles/pull/885) M: FLA and Megatron CP layouts disagree; [#975](https://github.com/radixark/miles/pull/975) M: `A_log` downcast to BF16 in several places; [#1597](https://github.com/radixark/miles/pull/1597) M: GDN LoRA target mapping | Not using CP avoids porting its layout, but not the mixed-dtype, shared-weight and LoRA coverage checks |
| DeepSeek V3.2/V4 | [#963](https://github.com/radixark/miles/pull/963), [#1181](https://github.com/radixark/miles/pull/1181) M: DSA/V4 plugin, compressor, indexer, sparse MLA backward, bridging; depends on [Megatron #28](https://github.com/radixark/Megatron-LM/pull/28) | [#1642](https://github.com/radixark/miles/pull/1642) M: forward fine, backward NaN due to TileLang shared-memory merge; [#1505](https://github.com/radixark/miles/pull/1505) M: specific projections should not be quantized; [#2717](https://github.com/radixark/miles/pull/2717) M: new checkpoint and MXFP4→FP8; [#2706](https://github.com/radixark/miles/pull/2706) M: restore and migration between native/plugin implementations | Inference kernel ≠ trainable kernel; checkpoint minor versions, quantization scope and shared-expert activation also change correctness |
| Model G (first releases) | a public PR M: second release; a public PR M: LoRA | a public PR M: sparse MLA backward NaN, optimizer/host peak, KV capacity; a public PR M: top-k `-1` out-of-bounds read/write, all-invalid rows 0/0; a public PR M: vendored kernel sync fix | Finite loss does not guarantee finite gradients; **sparse index padding is not sequence packing**; with packing disabled these boundaries still need checking |
| Kimi K2.5 | The five-piece set of [#1219–1223](https://github.com/radixark/miles/pull/1223) M: processor, VL-aware conversion, VL model, shared-outer LoRA, base CPU backup | [Bridge #5](https://github.com/radixark/Megatron-Bridge/pull/5) M: old runtime backport; [#2377](https://github.com/radixark/miles/pull/2377) M: checkpoint YaRN parameters | "Same family" cannot replace verifying config/position/adapter schema; multimodal and text-only acceptance are separate |
| Model B | a public PR M: native Megatron full-parameter; a public PR M: separate native LoRA, distributed adapter export | a public PR M: expert activation/combine FP32; a public PR M: removes redundant atomic groups, the converter already accumulates across buckets | LoRA is an independent support surface, not filling in a rank; integrity should be owned by one explicit layer, not stacked redundant protocols |
| Model C | a public PR O: linear attention, latent experts, LoRA, MXFP4 conversion/sync | a public PR O: special gated activation, latent norm, shared LoRA TP gradient sum; later rebase of #1825 switched to reading the sleeping trainer's host backup | Backend hooks in private/dedicated images are not necessarily upstream; a successful load can still miss real norm weights. TP>1 validation of the TP gradient hook is explicitly not done in that PR |
| Model F large MoE | a public PR O: NVFP4 rollout, BF16 trainer, adapter-only | Fixed fused 3D expert loading missing the EP offset, which made every rank load the same expert group; sleeping-side host backup, NCCL/graph version constraints | This PR is for the very large large MoE, **not the Flash-Next variant**; FSDP does not directly reuse the EP bug, but expert ID/layout checks are still needed |
| Model F Flash-Next variant | a public PR O: HC, CPU PLE, QSA, custom ops/mapping | a public PR O: injectable output contraction; a public PR O: follow-up vision RL; a public PR O: support for another native GDN packed path | Capabilities of different GDN implementations cannot vouch for each other; residency and restore of a giant CPU embedding are an independent cost |
| Model G Flash (later release) | a public PR O: linear attention/DSA-kpool/mHC, bridge, replay | a public PR O: LoRA/linear-attention target mapping; a public PR O: bridge/SP PEFT fix; a public PR O: VL | Architecture support, LoRA and VL are three different acceptance sheets; docs already having a recipe does not mean all code is merged |
| Nemotron | [#1033](https://github.com/radixark/miles/pull/1033), [#1284](https://github.com/radixark/miles/pull/1284) M; [Megatron #51](https://github.com/radixark/Megatron-LM/pull/51) M: Mamba hybrid/MoE/MTP | [#1150](https://github.com/radixark/miles/pull/1150) M: expert overlap/PP; [#1175](https://github.com/radixark/miles/pull/1175) M: bridge; [#2269](https://github.com/radixark/miles/pull/2269) M: restore necessary backup; [#3163](https://github.com/radixark/miles/pull/3163): old/new parameter name differences caused 46 tensors to be missed in load/sync | Hybrid layer order, PP and memory saver do not compose correctly automatically; successful model construction is not proof of completeness |
| Gemma4 | [#1232](https://github.com/radixark/miles/pull/1232) M: reuses the Bridge LLM provider, iterator wiring; [Bridge #7](https://github.com/radixark/Megatron-Bridge/pull/7), [#8](https://github.com/radixark/Megatron-Bridge/pull/8) M | backend/head dimension constraints; two support modes, imported provider and native plugin | When the structure is close to an existing architecture, reusing a generic provider skeleton saves a lot of code; but that comes after release, Day-0 still estimates cost as a self-written model |
| Model D | a public PR O: MSA block indexer, sparse GQA, three bridges, VL WIP | The PR states the vision merge does not satisfy SP, vision/projector is not really trained and is lost on save; indexer frozen; fused varlen kernel is a later performance item | Text SFT success does not prove multimodal RL/LoRA success; use a simple correct PyTorch reference first, do not proactively develop fused kernels for performance |

### A.3 Core implementation: which differences are by design and which need fixing

What is aligned is the same model's math, weights, inputs and update semantics; not rewriting FSDP into Megatron, and not making every parameter FP32.

| Topic | Miles/Megatron | VERL/FSDP | Alignment target and minimal check |
|---|---|---|---|
| Model compute graph | provider/spec, native/plugin, dedicated kernels, paired with bridge | Our self-written PyTorch model (HF interface), then PEFT injected | Same model math and parameter coverage; FSDP is not responsible for supplying a missing backward |
| Parameter and activation parallelism | TP/EP/PP/SP/CP can change layer layout, expert assignment and collectives | FSDP mainly shards parameters/gradients/optimizer; HSDP adds a replica dimension | No need to copy the topology; check gradient sum/average, shared parameters and logical weights on the current topology |
| Norm / residual / HC | Model-specific epsilon, gate, residual dtype, fusion order | Self-written implementation and the boundary of the training forward's outer autocast | Look at formula and precision separately. mHC/iHC of different models are not interchangeable |
| MoE | expert dispatch, grouped GEMM, shared expert, router config | Self-written expert implementation, PEFT 3D factor or fallback | Expert ID, gate/up order, scaling, activation and combine precision; a frozen base still needs input gradients |
| Sparse/linear attention | DSA/KDA/GDN, compressor/indexer with custom forward/backward | Depends on the chosen HF/FLA/kernel; FSDP does not understand its semantics | mask, position, invalid top-k indices, cache/state reset, forward/backward; testing only dense attention is not enough |
| Packing / CP | Every stateful operator must consume sequence boundaries and pass state across ranks correctly | The remove-padding/packed path also needs model adaptation | A correct attention mask does not mean convolution, recurrence or PLE are correct |
| LoRA | native/bridge target, shared factors, model-parallel gradients and exporter | PEFT target_modules/target_parameters, fused parameters, factor-aware path | List real target/A/B/alpha/r/layout; filling in a rank is not completing support, and merge is not needed by default |
| Quantization and update | bridge/quantizer can change exported weights; QAT is an independent mechanism | Whether the training side quantizes depends on the actual backend/export branch | Make the conversion owner and granularity explicit; single checkpoint source of truth |
| Initialization/residency | meta init, CPU backup, sleeping trainer, model-parallel allocation | meta/FSDP materialize, CPU special tables, offload/export | Measure init, steady state, sync and save peaks separately; do not rule out OOM because steady state fits |
| Recompute / graph | Dedicated autograd/recompute, buffer lifetime | activation checkpoint + HF/PEFT; serving graph is independent | Recompute uses the same adapter, gradients and addresses valid; do not clear graphs or re-capture to mask update errors |
| Checkpoint / resume | native shards ≠ HF/adapter format directly servable | FSDP shard, adapter, optimizer, RNG/step state | One nonzero step→save→new-process resume→next step; verify restore and export layer by layer |
| Input/loss | tokenizer/template, loss mask, normalizer, sampling/rescoring rules | VERL data, reward, advantage, old-logprob path | Fixed IDs and mask, unified temperature and logprob definition; zero gradient when rewards in a group are constant is not an FSDP bug |
| MTP / multimodal / replay | Each needs model, bridge, data and training coverage | Also extra support surfaces | Target success does not mean MTP; text success does not mean vision/audio; do not enable by default because another framework has it |

### A.4 Applying this table to each model family

| Model family | Key implementation and precision contract | Conclusion/boundary |
|---|---|---|
| Plain dense Transformer / Gemma4 etc. | norm, RoPE, shared embedding/head, LoRA target, loss normalization | Simple structure, low self-writing cost; no HC does not mean no precision audit needed |
| Qwen3.5/3.6, Qwen Next GDN | `A_log`, gated norm, convolution/recurrent state, CP/packing | Source confirms the Qwen3.5 plugin declares `A_log` FP32 and norm follows config dtype; cannot extrapolate that every Next branch is the same |
| Model F Flash-Next variant | HC, PLE/CPU tables, QSA/GDN, shift/sample boundaries | There is a known packing integer boundary error case (see A.5); keep separate from Model F's 2.4T branch |
| Model F large MoE | 3D experts, EP offset, quantization and adapter export | Verify routing/sharding mapping independently; not a synonym for Flash-Next |
| DeepSeek V3.2/V4 | sparse MLA, sink/indexer, V4 compressor/mHC, FP32 bridge | Directly confirmed DSv4 sink/mHC protection and bridge downcast exceptions; V3.2 cannot directly take all of V4's structure |
| Model G (all releases, incl. Flash) | sparse attention backward, invalid top-k, the Flash release's linear attention/mHC, LoRA mapping | dtype not verified per parameter for all latest branches; keep unknown |
| Kimi K2.5 / Model C | K2.5 multimodal/shared factors; Model C linear attention, latent experts/norm, quantization | Do not treat Model E's sink or DSv4 mHC as Model C's structure |
| Model E | iHC, sink, full/shared indexer, FP32 load→FSDP forward | Local four-layer experiments in its own case notes (not included here); no verified Miles same-model implementation |
| Model B | router, expert activation/combine, sconv/relative position, native LoRA, multimodal tower | Source confirms router/activation/combine FP32 config; forward/backward and boundaries need separate acceptance |
| Model D | MSA/block indexer, sparse GQA, bridge, scope of trainable modules | Same-model FSDP dtype not verified; do not fill in guesses |
| Nemotron/Mamba hybrid | Recurrent state, hybrid layer order, MoE/MTP | Do not infer recurrent state correctness from attention tests; our implementation is in §5 and [cases/nemotron](../../../cases/nemotron/README.md) |

Direct examples: a public PR, Model B's plugin config and ops (same Miles commit), a public PR.

### A.5 How to use Model F's packing evidence

Model F Flash-Next variant, same step100 adapter, duplicated samples of the same request: the packed second copy has probability MAE 0.13123 vs vLLM; in an ordinary batch both copies are 0.00419. The actual PLE shift function shows an integer difference where the second sample references the last token of the first. This proves the cross-sample boundary in the current model integration is wrong; it does not prove all error comes from PLE, nor that VERL/FSDP's generic sharding logic is wrong.

With packing off, the full model keeps CUDA Graph, MTP and native sync; two runs give `rollout_probs_diff_mean` 0.00600 / 0.00656, the same order as Miles forward/sync's 0.00737 / 0.00643. The two groups are not the same policy/generated samples, so per-token equivalence or framework superiority cannot be claimed. Raw evidence is in that model's own debug report, not part of this knowledge repo.

### A.6 Minimal per-model flow and review entry points

1. Fix identity: image digest, actual imported files/SHA, HF/PEFT/backend, checkpoint revision; also check target/draft shared weights.
2. List exceptions, not every tensor: special parameters/buffers, dtype and value source, cast location, quantization exclusion scope, LoRA target. Items without evidence are unknown.
3. Few-layer tests that preserve model semantics: cover the necessary layer types; fixed IDs, same adapter. First unwrapped vs FSDP, then vs serving; forward, backward and export separately.
4. Nonzero update closed loop: prove the intended LoRA gets gradients, changes after a step, and the receiver uses it; test save/resume. A full loop with zero advantage is not nonzero update verification.
5. Fixed samples across steps: score both ends with the same policy version, and repeatedly measure self-noise; save raw tokens/mask/logprobs.
6. Back to formal training: keep the original recipe; check production graph/MTP/lifecycle and held-out improvement. Diagnostic NLL, manual perturbation and few-layer results cannot substitute.

Borrow only mechanisms that trigger this model's problems; do not uniformly upgrade to FP32, do not add QAT by default, do not force-port TP/EP/packing, do not clear graphs, do not add a partial-update protocol.

Review entry points: a public PR → specific recipe → plugin/spec → bidirectional bridge → dependency PRs; key source: the GDN boundary in a public PR, SP/CP conv and autograd in Model B's plugin `ops.py`, adapter/reduction markers in its `lora.py`, FP32 and routing-table mapping in a public PR. Later scans reuse model path/keyword history from `git log --since=2026-03-08 origin/main`, then query all-state PRs in the three repos. Recipe/effect see the [recipe guide](../recipe-evaluation/01-recipe-alignment.md); conversion owner see the [quantization guide](../precision-quant-qat/01-conversion-chain.md#4-who-converts-conversion-owner); adapter contract see the [LoRA guide](../lora/README.md).
