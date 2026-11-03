# Gate 4 — independent backward / VJP validation (Nemotron-H Lightning NVFP4 5-layer proxy)

> Status: complete (r5). §0 was written before any gradient comparison run, 2026-10-01T16:02Z;
> at that time `g4_refs.py` sha256 `8f7a1712…`, `g4_vjp.py` sha256 `f8345393…`). Results are in the following sections.

## 0. Pre-declared contract & tolerances (frozen before running)

### 0.1 Reference implementation classes

| Code | Meaning | Calls the production backward? |
| --- | --- | --- |
| CIR | same-contract reference: rewritten following each module's declared surrogate/native function and dtype flow, using pure PyTorch ops + autograd (or the guide's explicit STE formulas `dX = G @ W_ref`, `dW = Gᵀ @ X_ref`); quantization ops (NVFP4 decode, FP8 quantize/dequantize) are all implemented independently, and W_ref is decoded independently from the raw safetensors bytes | No |
| IMR | FP64 independent math reference: same declared contract, all computation in FP64 (keeping only the rounding points declared in the native dtype flow, e.g. SSD's `dt+dt_bias`/softplus in BF16, MoE's BF16 edges); attention uses an explicit softmax analytic VJP | No |
| DIAG | diagnostic only, does not take part in verdicts | No |

Discrete state (MoE expert ids) is fixed to the result of the production visible forward; continuous score/combine weights keep their derivatives.

### 0.2 Tolerance classes (assigned per gradient, frozen before running)

Metrics: max_abs, relative L2 `||g-g_ref||₂/max(||g_ref||₂,1e-30)`, cosine, norm ratio, non-finite count, nonzero count, bitwise mismatch count (for the same dtype, compared as integer bit patterns, distinguishing ±0).

| Class | Verdict | Applies to | Rationale |
| --- | --- | --- | --- |
| T0 | bitwise mismatch = 0, and non-finite = 0 on both sides | reference and production execute the same declared arithmetic (same torch ops and dtype flow, or the same FP32 cuBLAS GEMM formula) | the same declared contract should be bitwise identical |
| T1 | rel L2 ≤ 1e-3, cosine ≥ 0.99999, norm ratio ∈ [0.999, 1.001], non-finite = 0 | same contract but not reproducible op by op: (a) SSD production native uses a custom `_ChunkProductSum` backward, the reference is an independent chunk-loop implementation (different FP32 reduction order); (b) routed experts' compact CuTe TF32 grouped GEMM (the reference uses FP32 cuBLAS on TF32-ized weight operands); (c) FP32 accumulation order of repeated tokens in embedding; (d) ≥3-way BF16 gradient fan-in (MoE input x: router/routed/shared; attention input: q/k/v), where the autograd engine's accumulation order is not part of the contract; plus all downstream gradients whose path passes through any of the above | FP32 reordering error ~1e-6 relative; occasional 1-ulp flips (2⁻⁸) on BF16 edges at a fraction ≲1e-3 give rel L2 ~1e-4; 10× margin |
| T2 | rel L2 ≤ 2e-2, cosine ≥ 0.9998, norm ratio ∈ [0.98, 1.02], non-finite = 0 | all IMR (FP64) comparisons | the FP64 reference does not reproduce the step-by-step rounding of production BF16 gradient tensors (RMS relative error of a single BF16 rounding ≈1.1e-3, accumulated over several steps along the chain) nor flash-attention's internal BF16 intermediates (historical similar 835-token comparison rel L2 0.0013–0.0031) |
| zero dY | all classes: count_nonzero of both production and reference gradients is 0 | dY all zero | — |

Hierarchy rule: at primitive level and module/block level, classes are annotated per gradient; at block/model level the gradient class is the strictest unreachable one along the path (passing through any T1 item makes it T1). Specifically:
- Head (norm_f → lm_head → logprob): all T0;
- Mamba block: `out_proj.weight`, `mixer.norm.weight` T0, the rest (downstream of scan) T1;
- MoE block: `shared_experts.{up,down}_proj.weight` T0, the rest T1;
- Attention block: `q/k/v/o_proj.weight` T0, input gradients and block norm T1;
- Model level (runtime real backward's `main_grad` vs the reference chain): `lm_head.weight`, `norm_f.weight` T0, the rest T1; the `main_grad` contract is modeled as "zero FP32 buffer += param.grad" (-0 → +0).

### 0.3 Pre-handling of the MoE routed contract conflict

`nvfp4_moe_vjp.SURROGATE_CONTRACT = moe-fixedscale-grouped-bf16edges-v2` is documented as "FP32 padded expert GEMMs with BF16 intermediate edges", and `validate_matmul_policy` forbids TF32; but the `compact-f32-tma-nosplit` backend chosen in production (`nvfp4_compact_gemm.py`) describes itself as "FP32 storage … TFloat32 MMA interpretation, FP32 accumulation". BF16 values (x, h, gv, gu) are exact in TF32; only the NVFP4 dequantized weights (with an arbitrary FP32 global scale) get truncated. Pre-decided:
- `CIR-tf32rz`: weight operands have their low 13 bits truncated (RZ) per PTX tf32 operand semantics, then FP32 GEMM — judges implementation correctness (T1);
- `CIR-fp32-literal`: FP32 GEMM strictly per the V2 text — judges "conformance to the declared V2 contract" (T1); if tf32rz passes and fp32-literal fails, it is recorded as a **contract discrepancy**, and the threshold is not relaxed;
- `DIAG-tf32rn` (RN truncation) diagnostic only; `IMR-fp64` judged by T2.

### 0.4 dY and inputs

- dY: fixed random (fixed seed, same dtype as the output), all zero, sparse (2 channels on each of tokens 0/127/128/T/2/T-1), real (block output gradients captured by hooks in the runtime real training backward; primitive-level real dY = the upstream of that primitive obtained by propagating the captured real block dY through the reference chain, with production and reference using the same copy). IMR only runs random/real.
- Inputs: runtime forward captures of the proxy (real layers 2–6: mamba, moe, mamba, attention, moe; θ0 weights) on real rollouts. Three cases: `dapo0` (single real DAPO sequence, 1224 tokens), `packed` (3 real DAPO sequences, 3579 tokens, multiple boundaries), `long` (synthetic_8192: 8192 synthetic prompt + 1024 real generated tokens = 9216 tokens, reaching the routed VJP 9216 limit).
- Determinism: in the container `CUBLAS_WORKSPACE_CONFIG=:4096:8` (vLLM BI initialization rewrites it in-process to `:16:8`; the actual value is recorded), `torch.use_deterministic_algorithms(True)`, TF32 off (`fp32_precision=ieee`); every comparison first does production A/A and reference A/A (must be bitwise identical) before attribution.
- Discriminative-power supplement: at θ0, master == dequantized (false-green risk), so at the end of the dapo0 case θ0+δ (master multiplied by 1+0.05·N(0,1), then refresh) is retested on shared up_proj (NVFP4), mamba in_proj (FP8), and layer1 routed experts, and the size of the difference from "using master instead of dequantized" is reported to prove that the comparison can tell them apart.

### 0.5 One optimizer step (pre-declared)

Same θ0 params and optimizer state (step 0, exp_avg=0), runtime `optimizer_step` once (Megatron DistributedOptimizer + TE FusedAdam, lr 1e-6, β=(0.9,0.999), eps 1e-8, wd 0, clip 1.0). Reference: from production `main_grad`, independently compute the global norm, clip coefficient and Adam first step in FP64, then round to FP32. Verdict (per trainable parameter): max FP32 ulp difference between the updated master and the reference ≤ 1, fraction of unequal elements ≤ 1e-3, delta cosine ≥ 0.99999; grad norm relative difference ≤ 1e-5. The update comparison driven by the reference gradients is diagnostic only, not judged.

### 0.6 Addendum (written before 2026-10-01T17:44Z; r5 started at node time 17:44 UTC; does not change any threshold in 0.1–0.5)

r2–r4 exposed four errors in **the reference implementation/metrics themselves**, handled as "fix the reference, do not relax the threshold"; evidence is kept in the r4 receipt:
1. Wrong linearization point in the router reference: the native VJP of the second visible_forward (routing weights) should be evaluated at the *visible* logits (BF16 `torch.mm(..., out_dtype=fp32)`); r2–r3 references wrongly used the recomputed FP32 logits (r2 router primitive T0 failed with rel ~1e-4). Corrected from r4; the router primitive passes T0 bitwise.
2. FP8 activation quantization formula: vLLM `static_scaled_fp8_quant` is `x * fp32(1/s)`, while reference r4 used `x / s`; on layer2 in_proj (s=0.11607) real inputs, 30156 elements land exactly on FP8 midpoints and the two formulas round in different directions (`debug_fp8.py` receipt: div 30156 mismatches, mul 0). From r5 the STE activation reference multiplies by the reciprocal; KV cache quantization is still division (consistent with `reshape_and_cache_flash`, verified bitwise).
3. Mislabeled upstream dY of the o_proj primitive in the attention chain (r2 shape error, produced no verdict).
4. The optimizer ulp metric counted ±0 as 2³¹ ulp: changed to a monotonic integer key (±0 coincide), thresholds unchanged; also added an "error/ulp(max(|p0|,|Δ|))" diagnostic column (not part of the verdict) to identify cancellation when p≈Δ.

Added **stricter** checks (only added, they do not replace the pre-declared verdicts; pre-declared T1 verdicts are reported as is):
- `DIAG-chain-engine-order` (attention, T0): models the 3-way BF16 fan-in in the autograd engine's actual execution order (the later-created v/k computed first, q added last);
- `DIAG-hybrid-prod-routed` (MoE, T0) / `DIAG-hybrid-prod-scan` (Mamba, T0): the only primitive in the reference chain that cannot be reproduced op by op (compact routed VJP / SSD) takes the production primitive's gradient (each has already been compared separately at primitive level under T1), and everything else is an independent reference, to prove the composition/wiring is bitwise correct;
- `runtime-consistency` (T0): the runtime real training backward's `main_grad` and each block's input gradient are bitwise identical to "the same module's production backward called in isolation + the captured real dY";
- embedding's `DIAG-fp32-indexput`/`DIAG-bf16-indexput` (T0, only diagnoses the accumulation precision of the PyTorch embedding backward).

---

## 1. Conclusion (r5, final receipt)

**Under the pre-declared criteria, gate 4 does not pass as a whole**; but all failing items have been traced to "contract text discrepancy / pre-declared tolerance estimated too tight / framework numerical behavior", and **no production backward implementation bug was found**. Evidence chain:

1. Each primitive (all 14 kinds, all instances) vs the independent reference: everything whose contract is reproducible op by op is **T0 bitwise identical** (3 cases × random/zero/sparse/real dY); the two that are not reproducible op by op, SSD and compact routed VJP, pass within T1; all FP64 IMR pass within T2.
2. Composition level (module/block): after replacing the only non-reproducible primitive with the production primitive (which has been compared separately), the rest of the independent reference chain is **bitwise identical** (`DIAG-hybrid-*` and `DIAG-chain-engine-order` all pass T0).
3. The 38 `main_grad`s and each block's input gradient from the runtime real training backward are **47/47 bitwise identical** to "the production module called in isolation + the captured real dY" (3 cases).
4. Production A/A and reference A/A are all bitwise identical (dapo0 704/1306, packed 684/1286, long 684/1286, 0 failures).

| Failing item (under pre-declared criteria) | Nature | Blocking? |
| --- | --- | --- |
| F1 routed experts vs the **literal V2 FP32 contract** (`CIR-fp32-literal`) rel L2 1.2e-3–4.8e-3 > T1 | **Contract discrepancy**: the compact backend uses TF32 MMA (weight operands truncated by 13 bits, RZ); passes T1 vs the TF32-RZ emulation reference (≤8.1e-4), does not match the RN emulation → the hardware truncates | Needs a human ruling: change the contract text to declare TF32 operands, or switch to an exact FP32 GEMM |
| F2 composition-level T1 failures: attention input gradient (3-way BF16 fan-in) rel ≤2.9e-3; MoE `gate.weight`/block norm rel ≤1.8e-3; model level 19/14/11 parameters rel ≤5.8e-3 | The pre-declared T1 (1e-3) underestimated how BF16 summation order and T1 primitive differences are amplified by BF16 rounding / accumulate layer by layer; engine-order and hybrid references are bitwise identical | Not an implementation error; a formal "pass" would require re-declaring the tolerance and rerunning on new data |
| F3 embedding (dapo0 only, 1224 tokens) rel 6.1e-3 > T1 | PyTorch CUDA `embedding_dense_backward` takes a path with BF16 intermediate accumulation when there are ≤3072 indices (synthetic repro: T=1224/3072 mismatch, T=3073 bitwise identical to the exact FP32 sum); mLite does not customize this op | Framework numerical behavior, not an mLite bug; recommended to document (embedding gradient precision is lower for short micro-batches) |
| F4 one optimizer step: 24/38 parameters' master differs from the FP64 reference by >1 ulp | When p0=0 (FP8/NVFP4 zero codes) or p0≈−Δ, the 1-ulp criterion measured by p1 is too strict; plus TE FusedAdam FTZ on subnormal gradients (layer0 in_proj row 9570, layer2 in_proj rows 5769/10204, |g_clip|<1.2e-38; production does not update, the reference updates by ~1e-36 to 1e-41) | Not an implementation error; grad norm relative difference 1.3e-7, clipping takes effect, delta cos ≥0.9999999999, 38/38 parameters updated |

## 2. Environment and identity

- Image: `<image>` (a stage-2 Nemotron training image), ID `sha256:545f17045844e1ce41b67c7df77e4fad5d157ca4e0d5eb40abf9c8a8ba97330b` (stage3 not used; stage3 only adds the verl_mlite rollout extension). The sha256 of the installed sources matches the internal mLite dev branch (after the image's base commit, only layerwise_reload changed), e.g. `nvfp4_moe_vjp.py d32b4833…`, `grouped_gemm_tma_f32.py b30d3192…` (= COMPACT_KERNEL_SHA), `fp8_training.py 914bc6d6…`; all listed in the receipt's `installed_sources`.
- One GB200 node, single process per GPU (torchrun standalone); dapo0/packed/long on GPU0/1/2 respectively, launches staggered by 30 s; container names `g4-*`.
- Model: `<model-dir>` (proxy with real layers 2–6, contiguous, θ0); rollouts `$WORKDIR/runs/g3-serve-stage2r1/rollouts.json` on that node's local disk (sha256 `adb3b8f2…`).
- Determinism: `torch.are_deterministic_algorithms_enabled()=True`, `fp32_precision=ieee`, container `CUBLAS_WORKSPACE_CONFIG=:4096:8`, actually `:16:8` in-process after vLLM BI initialization (recorded in the receipt). In the BI process `torch.bmm` is replaced by a Triton kernel (FP32 inputs mean TF32), so all references use only 2D `torch.mm`/elementwise ops, and FP64 never goes through bmm.
- Runtime sanity (each case): training-forward logprobs vs rollout raw bytes 0 mismatches; `d(log_probs) == −mask/count` holds bitwise; visible replay and runtime forward block outputs bitwise identical; saved FP8 Q/K/V operands bitwise identical to independent quantization; all W_ref bitwise identical to production `_dequantized`/surrogate snapshots.

## 3. Inputs

| case | sequences | tokens | effective loss tokens | boundaries | MoE routing (experts used in layer1/4, max count) |
| --- | --- | --- | --- | --- | --- |
| dapo0 | dapo_0 (real DAPO prompt+generation) | 1224 | 1024 | [0,1224] | 83/97, 978/885 |
| packed | dapo_1+dapo_2+dapo_0 | 3579 | 3072 | 3 sequences | 110/113 |
| long | synthetic_8192 (synthetic prompt 8192 + real generation 1024) | 9216 (= routed VJP limit) | 1024 | [0,9216] | 123/124, max 4815 |

Every case has empty experts (covering empty expert) and repeated routing; chunk boundaries 127/128 are covered by the sparse dY.

## 4. Per-module results (primitive level, worst over 3 cases × dY{random,zero,sparse,real})

| Module (instances) | Declared gradient contract | Reference implementation | Gradient / class | Worst metric | Verdict |
| --- | --- | --- | --- | --- | --- |
| RMSNorm (5 block norms + norm_f) | visible `rms_forward` (including FP32 residual add); VJP = native autograd (`x.float()+r.float()`→`r_out` BF16; `y·rsqrt(mean(y²)+eps)`→BF16·w), evaluated at the visible inputs; norm_f's unused residual output treated as a materialized zero gradient | CIR same-expression autograd; IMR FP64 analytic formula | dx,dresidual,dw: T0 / T2 | CIR 0 mismatches; IMR rel ≤3.7e-3 | pass |
| GatedRMSNorm (2) | native `x·silu(g)` grouped RMS (group 512)→BF16·w | CIR / IMR FP64 | dx,dgate,dw | 0 / ≤4.8e-3 | pass |
| Fp8TrainingLinear (in_proj×2, out_proj×2) | fixed-scale two-sided identity STE: dX=(G·W_deq)→BF16, dW=Gᵀ·X_deq (FP32); W_deq=ckpt FP8×weight_scale; X_q=fp8(x·fp32(1/s_in)) saturating RNE | CIR: W decoded independently from raw safetensors bytes; activation quantization implemented independently (byte-by-byte check against vLLM quantization, 0 mismatches) | dx,dW: T0 / T2 | 0 / ≤1.7e-3 | pass |
| Nvfp4TrainingLinear (shared up/down×2, lm_head) | fixed-scale identity weight STE (W4A16): dX=(G·W_deq)→BF16, dW=Gᵀ·X (FP32); W_deq=FP4 level×(e4m3 scale×global) | CIR: independent NVFP4 decoder; IMR FP64 | dx,dW: T0 / T2 | 0 / ≤2.7e-3 | pass |
| BF16 Linear (q/k/v/o) | visible `linear_batch_invariant`; VJP=native `F.linear` autograd | CIR / IMR | T0 / T2 | 0 / ≤1.7e-3 | pass |
| causal conv1d (2) | native = HF `causal_conv1d_fn` fallback (BF16 depthwise conv + SiLU), per sequence, no cross-sequence history | CIR / IMR explicit FP64 convolution | dx,dW(6144×1×4),db | 0 / ≤3.7e-3 | pass |
| SSD scan (2) | native = `ssd_reference.chunk_scan` (HF mamba2_chunk_scan, custom `_ChunkProductSum` backward), dt+dt_bias and softplus in BF16, the rest FP32; A=−exp(A_log.float()) | independent chunk loop + explicit state recurrence (FP32: T1; FP64: T2) | dx,ddt,dA_log,dB,dC,dD,ddt_bias | T1 ≤1.4e-4 (A_log, D bitwise); T2 ≤1.0e-4 | pass |
| FA4 FP8-KV attention (1) | visible FA4 (Q scale 1, K/V static FP8, scale rounded through BF16); surrogate: SDPA on dequantized Q/K/V, identity STE | CIR: independent FP8 quantize/dequantize + identity bridge + per-sequence SDPA autograd; IMR: FP64 explicit softmax analytic VJP | dq,dk,dv: T0 / T2 | 0 / ≤9.9e-3 | pass |
| Router (2) | two visible_forwards: logits (visible BF16 mm→FP32; native `F.linear(x.float(),W.float())`); routing weights (visible grouped_topk; native sigmoid→gather(ids)→normalize, **evaluated at the visible logits**); ids fixed | CIR / IMR | dx,dW: T0 / T2 | 0 / ≤1.7e-3 | pass |
| Routed experts (compact V2, 2) | V2: FP32 grouped GEMM (text) + BF16 edges u,h,v,gv,gh,gu; dr=Σ(dy·v) FP32; dx accumulated in FP32 over slots 5→0; weights = ckpt dequantized snapshot (fixed-scale STE), ids fixed | per-expert 2D mm: `CIR-tf32rz` (judges implementation), `CIR-fp32-literal` (judges contract conformance), `DIAG-tf32rn`, `IMR-fp64` | dx,dup,ddown,droutes | tf32rz T1 ≤8.1e-4 **pass**; fp32-literal ≤4.8e-3 **FAIL(F1)**; tf32rn FAIL (diagnostic); IMR ≤4.8e-3 pass | **contract discrepancy** |
| Shared experts (2) | NVFP4 up → ReLU² (BF16 autograd, at visible u) → NVFP4 down | CIR / IMR | dx,dWu,dWd: T0/T2 | 0 / ≤2.7e-3 | pass |
| MoE combine (2) | `shared + routed·2.5` native BF16 | CIR | T0 | 0 | pass |
| logprob head | visible `compute_token_logprobs`; native `log_softmax(logits.float()).gather` | CIR / IMR FP64 | dlogits | 0 / ≤1.9e-6 | pass |
| Embedding | native nn.Embedding backward | FP64 exact sum→BF16 (T1); DIAG FP32/BF16 index_put | dW | dapo0 rel 6.1e-3 **FAIL(F3)**; packed rel 1.6e-4, long 0 pass | framework behavior |

θ0+δ discriminative power (end of dapo0, master×(1+0.05N), after refresh): shared up_proj (NVFP4) and layer0 in_proj (FP8) dX/dW **T0 bitwise**, layer1 routed experts tf32rz T1 pass; at this point the fraction master≠deployment is 84%/99.9%/91%, and the dX difference of "wrongly using master instead of dequantized" is rel 5.0e-2/2.6e-2/5.5e-2 — proving the comparison can distinguish master from dequantized values. Independent decoding is bitwise identical to production `_dequantized`/snapshot; the independent encoder is byte-identical to production packing (informational).

## 5. Composition level and model level

| Scope | Reference | Result |
| --- | --- | --- |
| head (norm_f→lm_head→logprob) | CIR-chain, T0 | all bitwise (3 cases × 4 dY) |
| Mamba mixer / block (layer0, 2) | CIR-chain (T0: out_proj.weight, mixer.norm.weight; rest T1) | pass; T1 items rel ≤4.6e-4 |
| same | DIAG-hybrid-prod-scan (T0) | all bitwise |
| Attention mixer / block (layer3) | CIR-chain | q/k/v/o weight bitwise; input gradient rel ≤2.9e-3 **FAIL(F2)**, block norm.weight ≤4.9e-3 FAIL |
| same | DIAG-chain-engine-order (T0; v, k first, q added last) | all bitwise |
| MoE mixer / block (layer1, 4) | CIR-chain-tf32rz | shared weights bitwise; x, experts T1 pass; gate.weight ≤1.8e-3, block norm.weight ≤1.8e-3 **FAIL(F2)** |
| same | CIR-chain-fp32-literal | FAIL (F1 propagation) |
| same | DIAG-hybrid-prod-routed (T0) | all bitwise |
| Model (38 parameters, runtime `main_grad` vs fully independent reference chain, loss dY) | lm_head/norm_f T0, rest T1 | lm_head, norm_f bitwise; T1 failing parameter counts dapo0 19/38, packed 14/38, long 11/38, failing items rel ≤5.8e-3 **FAIL(F2, layer-wise accumulation)** |
| runtime-consistency (T0) | main_grad vs isolated production module + captured real dY; block input gradient vs the gradient captured at the previous block | **47/47 × 3 cases bitwise** |

Coverage: all 38/38 trainable parameters have primitive-level, block-level, model-level and runtime-consistency comparisons, `allow_unused=False`, nothing missing, no non-finite values:
`embeddings.weight`; layer0/2 (Mamba) `norm.weight, dt_bias, A_log, D, in_proj.weight(FP32 master), conv1d.weight, conv1d.bias, norm.weight, out_proj.weight(FP32 master)`; layer1/4 (MoE) `norm.weight, gate.weight, experts.weights.up_proj/down_proj(FP32 master), shared_experts.up_proj/down_proj.weight(FP32 master)`; layer3 (attention) `norm.weight, q/k/v/o_proj.weight`; `norm_f.weight`; `lm_head.weight(FP32 master)`. Non-trainable quantities (FP8/NVFP4 scales, KV scales, `e_score_correction_bias`, deployed packed weights) are fixed buffers per the contract.

## 6. One optimizer step (3 cases, Megatron DistributedOptimizer + TE FusedAdam, step 1)

- 38/38 parameters found their owner, `update_successful=True`; grad norm production 472.58590698 vs FP64 reference 472.58596937 (rel 1.3e-7, criterion ≤1e-5 pass), clip coeff 0.0021 (clipping takes effect); updated BF16 model params == master cast (38/38 bitwise).
- Pre-declared per-parameter criteria (≤1 ulp, unequal fraction ≤1e-3, delta cos ≥0.99999): dapo0 24/38, packed 23/38, long 22/38 **FAIL(F4)**; all parameters delta cos ≥ 0.9999999999. The diagnostic column (error/ulp(max(|p0|,|Δ|))) is ≤ 6 except for FP8 in_proj; FP8 in_proj's large errors all come from "dead rows" with p0=0 and |g|<1e-12 (`debug-optimizer-ftz*.txt`: layer0 row 9570, 2678 elements; layer2 rows 5769/10204, 2638 elements; |g_clip| is in the FP32 subnormal range, and production FTZ does not update them).

## 7. A/A

Every comparison runs production twice and the reference twice and compares bitwise: dapo0 production 704 / reference 1306, packed 684/1286, long 684/1286, **all 0 failures**.

## 8. Run history (failures not hidden)

| run | Result | Cause (all on the test-script/reference side) |
| --- | --- | --- |
| r1 | error | the loss gradient hook was attached to the `log_probs[None]` view and never fired |
| r2 | error | mislabeled upstream dY for the attention o_proj primitive; wrong linearization point in the router reference (router T0 rel ~1e-4) |
| r3 | error | FP64 router reference leaf dtype |
| r4 | OOM (optimizer phase) | the optimizer reference held all FP64 gradients at once; also exposed the FP8 activation quantization formula issue (division vs multiplying by the reciprocal) and the ±0 ulp metric issue |
| r5 | complete (status=failed, see §1) | final receipt |

There were also diagnostic runs: `g4-dbgfp8` (FP8 quantization formula), `g4-dbgopt`/`g4-dbgopt2` (optimizer FTZ), `g4-dbgemb`×4 (embedding 3072 threshold). No run had NCCL errors; after a reminder about IPC conflicts with the sync-gate containers, launches were staggered by 30 s and each launch first checked for running `sync-gate*` containers.

## 9. Commands and receipts

```bash
# local scripts -> node
<gate4-scripts>/sync.sh
# on the node: wait until sync-gate is idle, then launch the three cases staggered by 30 s (dapo0 with --perturb)
bash $WORKDIR/scripts/gate4/launch-when-free.sh r5 dapo0 packed long
# actual command for a single case (run-g4.sh):
sudo -n docker run --rm --name g4-<case>-r5 --gpus all --device /dev/nvidia-caps-imex-channels/channel0 \
  --cap-add IPC_LOCK --network host --ipc host --shm-size 64g --ulimit memlock=-1 \
  -v <model-dir>:/model:ro \
  -v $WORKDIR/scripts/gate4:/opt/g4:ro -v $WORKDIR/runs:/runs \
  -e VLLM_BATCH_INVARIANT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_HUMMING_MOE_GEMM_TYPE=auto \
  -e CUDA_VISIBLE_DEVICES=<gpu> -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONPATH= -w /tmp \
  <image> \
  /opt/ds4-venv/bin/torchrun --standalone --nproc_per_node 1 /opt/g4/g4_vjp.py --case <case> \
  --rollouts /runs/g3-serve-stage2r1/rollouts.json --output /runs/g4-<case>-r5 [--perturb]
```

- Scripts (gate-4 scripts directory): `g4_refs.py (bdaa8733…), g4_vjp.py (3610fc1a…), run-g4.sh, launch-when-free.sh, sync.sh, summarize.py, debug_fp8.py, debug_opt.py, debug_emb.py, explore.py`
- Node receipts: `$WORKDIR/runs/g4-{dapo0,packed,long}-r5/receipt.json` and `.log`; r1–r4 kept in the same directory.
- Local receipts (gate-4 results directory): `receipt-{dapo0,packed,long}-r5.json`, `summary-*-r5.md` (per-gradient tables), `aggregate-r5.md` (cross-case summary), `debug-fp8-quant.txt`, `debug-optimizer-ftz.txt`, `debug-optimizer-ftz-r2.txt`.

## 10. Minimal repros

- F1: `g4_refs.routed_ref_grads(x, up, down, routes, ids, dy, "fp32")` vs production `Nvfp4RoutedDeployment.forward` (compact backend) on dapo0 layer1 real inputs, random dY: dx rel L2 2.8e-3; `"tf32rz"` mode 7.6e-4. Only 68–69% of the weight elements are exactly representable in TF32 (receipt `state_checks.frac_*_tf32_exact`).
- F3: `debug_emb.py 1224` (pure PyTorch, BF16 weight, deterministic mode): `F.embedding` backward differs from the exact FP32/FP64 sum in 3229 elements; `debug_emb.py 3073`: 0.
- F4: `debug_opt.py`, see `debug-optimizer-ftz*.txt`.

## 11. Not covered (not run)

- Gradient reduction for TP/EP/CP/PP>1 and DP>1 (the proxy is TP1/EP1/PP1, DP world 1); the `diagnostic_full_training` (PP4, `recompute_surrogate=True`) path; the padded-v2 routed backend (not the production config).
- Multi-micro-batch gradient accumulation, activation recompute, backward under CUDA Graph/compile (the production training path is eager).
- logprob temperature ≠1, entropy term, nonzero weight decay, the formal recipe's lr/schedule; optimizer step 2 and later (m/v nonzero).
- Artificially constructed MoE top-k tie cases; the long case's prompt is synthetic tokens (the generated part is real).
- No re-verdict on new data with revised tolerances (a "pass" for F2/F4 requires a separate pre-declaration and rerun).

## 12. long16k (2026-10-02, candidate image + overlay bundle p7ssd, on another node)

- Previously 16k could not finish: in the reference VJP the Mamba inter-chunk recurrence was a dense broadcast (`new_states` in `ssd_reference.chunk_scan`), which allocated 32.5 GiB at once and OOMed. A Megatron fork commit changed it to per-chunk computation. Bitwise comparison against the HF dense implementation (`scripts/ssd_recurrence_check.py`, 4096/9216/16384 tokens, real geometry): forward, final state and all 7 input gradients equal; at 16384 tokens the peak drops from 134 GiB to 6.7 GiB. This function is also the recompute path of the production training backward.
- `run-g4.sh long16k-p7 … --case long16k`: 16048 tokens, rc=0, 3394 records, 110 fail. r5 (9216 tokens) also had 110. After aligning by (scope, module, ref, dy, grad), 11 went from pass to fail and another 11 from fail to pass, all belonging to the existing categories F1 (routed expert TF32), F2 (multi-term sums exceeding 1e-3) and F4 (optimizer ulp); **no new category appeared**.
- All forward preconditions hold: each layer's replay equals runtime, the operator inputs saved by attention equal the independent FP8 results, head logprobs equal runtime, and all 38/38 parameters were compared.
- Raw receipt: `receipt-long16k-p7.json` in the gate-4 results directory.
- Conclusion unchanged: under the preset tolerances it is still fail/inconclusive, F1–F4 await a human decision; no production backward implementation error was found at 16k.

## 13. Revisions per the step-7 agent review (2026-10-02, the step-7 agent review of gate-4 F1–F4)

Ruling: **conditional A**. The FAILs obtained under the pre-declared criteria are kept as is and not re-recorded as pass; step 7 is recorded as "implementation correct, with documented numerical exceptions".

- **F1 → contract v3** (Megatron fork commit): added contract `moe-fixedscale-grouped-tf32rz-bf16edges-v3`, which states "TF32 operands (dequantized weights truncated toward zero), FP32 accumulation". V2 was not modified in place. `validate_matmul_policy` still forbids torch global TF32; only the compact kernel explicitly routes its own operands through TF32. The recipe default was changed to v3 (verl recipe commit). Numerics completely unchanged. Note: the tf32rz reference's max rel is 8.1e-4, only about 1.2× margin from the 1e-3 threshold.
- **F2 correction**: in long16k, the `norm.weight` rel of the `layers.3` block is 8.29e-3, exceeding the 5.8e-3 upper bound written in §1/§5; corrected here. The model level (38 parameters) has no hybrid reference; its correctness rests on a composition argument. The basis: block-level engine-order and hybrid references are bitwise identical on all 4 cases, and runtime-consistency is 47/47 bitwise identical. Given that, model-level differences can only come from two sources: one is the accumulation order of autograd fan-in; the other is the deviation of the two T1 primitives (routed TF32 and SSD native) propagating along the chain. Both are already covered bitwise by hybrid references at block level.
- **F4 correction**: what gets FTZed is not the gradient itself but the intermediate 0.1·g in the Adam update, and only the 3 all-zero dead channel rows of the FP8 in_proj are affected. lm_head, experts and other parameters have 4–6 ulp deviations with 2%–7% unequal elements; this category was previously unexplained and needs to be verified with a reference of the actual optimizer path, see the next item.
- **Actual optimizer path**: the formal recipe offloads the entire optimizer to CPU (HybridDeviceOptimizer) and enables precision-aware and decoupled weight decay, which differs from the TE FusedAdam tested by gate4. The observer gained a first-step optimizer probe (`MLITE_EXACTNESS_OPT_PROBE`). On r14 (proxy formal recipe, bundle p12probe+vseed, on another node) the FP64 AdamW first-step formula was used as the reference; results saved as `opt-probe-r14.txt` in the gate-4 results directory:
  - The optimizer is ChainedOptimizer (precision-aware, CPU offload), lr=1e-6, betas=(0.9, 0.95), eps=1e-8; weight matrices have wd=0.1 (decoupled); grad_norm 0.041 is below clip 1.0, so no clipping.
  - 4 ranks, 13 FP32 master parameters (first 65536 elements each: Mamba in/out_proj, routed experts, shared experts, lm_head) compared against the FP64 reference rounded to FP32: **max 1–3 ulp**, 19%–32% of elements deviate, the rest bitwise equal.
  - The AdamW first step involves roughly 6–7 FP32 operations, each rounding by at most 0.5 ulp, so ≤3 ulp is as expected for an FP32 implementation. No sign of wrong wd grouping, lr or bias correction. The 4–6 ulp gate4 saw on TE FusedAdam is the same kind of phenomenon.
  - Conclusion: the actual optimizer path is implemented correctly. This item is after-the-fact evidence with no pre-declared ulp threshold; only observed values are recorded, and the earlier criteria are not rewritten.
  - **Correction (late 2026-10-03)**: the "implemented correctly" in the previous sentence was later overturned by pre-declared criteria: optimizer-auth v1 (its README in the run's results) found on the earlier base commit that wd_mult was ignored (the wd_mult=0 group actually had wd=0.1; fixed on the PP4 candidate head), plus A1/A4/A5 failures (diagnosed as gaps in the measurement model, in the in_proj main_grad diagnostic note). The r14 probe only looked at the first step and had no FP32 parameters with wd_mult=0, so it could not see the wd bug. The current conclusion is optimizer-auth v2 on p5 (in the run's final-p5 results).
- **PP4 path gradients** (new evidence, `scripts/run-grad-pp.sh`, `compare_grads.py`, on another node, bundle p11final+vseed): the packed case of the same rollout (3 sequences, 15536 tokens) was run with PP1 and with PP4 (`diagnostic_full_training`); **all 38/38 parameters' main_grad are bitwise equal**. On both sides old-logprob and training forward vs rollout are 0 mismatches. So the gradient evidence gate4 obtained on PP1 applies directly to PP4. Note: when the PP4 process wound down, rank 2 exited with exitcode 1 (all cases had completed and produced output); this is a harness problem. The single case's gradient file was overwritten because files are named by case, so it was not part of the comparison.
- Supplement (after the agent re-review): the PP4 evidence has been archived in the gate-4 results directory (`pp4-grads/`), including per-parameter comparison output and sha256 of the gradient files. The full optimizer probe is in `opt-probe-r14.txt`, covering all parameters on 4 ranks. BF16 parameters (norm, A_log, D, conv, qkvo, router, embedding) show no change in the BF16 view after the first-step update (lr=1e-6, a single-step update is far below BF16 resolution), so their wd grouping cannot be verified this way; that needs reading the optimizer's internal FP32 main params. After modeling "the decay factor 1−lr·wd computed in FP32" (`opt-probe-r14-fp32decay.txt`), the deviation fraction of all 13 FP32 masters decreases (e.g. 28%→24%), with max ulp unchanged. This indicates the implementation very likely computes the decay factor in FP32: the effective lr·wd corresponding to 0.9999999 is 1.19e-7, 19.2% larger than nominal. This is listed as a recipe risk and does not affect consistency.
