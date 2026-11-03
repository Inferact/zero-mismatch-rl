# Training forward and backward (`impl=vllm`)

How the mLite trainer computes the same token logprobs as the BI=1 vLLM rollout, and what was
measured about its gradients and training dynamics.

Provenance: the module contract below was developed in parallel tracks on top of an early #257 head
(`c2a185af0`) and validated on the earlier Inferact parity branch (#260, closed). The published
#257 by Yan Bai (NVIDIA) implements the same contract (the actor forward calls the rollout's ops;
owned backward; release-format export) and carries both attention cores; the end-to-end runs on the
#257 line are in [long runs](long-runs.md), [#257 gaps](pr257-gaps.md) and
[SWE-bench Pro](swe-agentic-rl.md). Numbers in this report come from proxies and the full model
on the parity branch unless stated otherwise.

## Starting point

At `c2a185af0`, #257 changed the model construction hook, added `deepseek_v41/vllm/*`, the
registry and tests; `examples/verl` was untouched. The `vllm` model overrode only the MLP (MegaMoE),
block (vLLM mHC) and head; attention, Engram and the forward were inherited from native mLite.

| Module | Training forward | Backward | Quantization |
|---|---|---|---|
| Embedding | Core `VocabParallelEmbedding` | autograd | none |
| Engram tables | mLite `ShardedFrozenTable` (host FP8 + E8M0/32, decoded to BF16 on GPU, all-reduce) | frozen | FP8 storage |
| Engram wkv / gate | `nn.Linear` BF16 + torch FP32 gating | autograd | none |
| mHC pre | vLLM `mhc_pre_delayed_overlap`; the fused `mhc_shifted_post_pre` not executed | owned (FP32 replay + autograd.grad) | none |
| mHC post | vLLM `mhc_post_tilelang` | owned analytic VJP | none |
| wq_a / q_norm / wq_b / wkv / kv_norm | Core TE Linear + wrapped torch norm | autograd | BF16 |
| RoPE / inverse RoPE | Core unfused YaRN; ratio-1 layers use the compress theta | autograd | — |
| Compressor (ratio 2) | mLite shared compressor, FP32 weight projection | autograd | none |
| Indexer projections | Core CSA indexer, frozen, input detached | frozen | BF16 |
| Indexer scores / top-k | Core `indexer_topk` (ratio 2); naive fused QK top-k + candidate pool + `torch.topk` (ratio 1) | discrete | BF16 |
| Sparse attention core + sink | Core unfused compressed sparse attention (torch FP32), or FlashMLA forward + cuDNN DSA backward | autograd / cuDNN | BF16 q/kv, KV not quantized |
| wo_a | Core einsum BF16 | autograd | BF16 |
| wo_b | Core TE row-parallel linear | autograd | BF16 |
| Router gate | vLLM `bf16_fixed_mm` | owned | BF16 |
| Router top-k | vLLM `dsv4_topk` | owned (ids not differentiable) | — |
| Shared expert | DeepGEMM `fp8_gemm_nt`, activation 1×32 UE8M0, weight re-quantized each step to 32×32 FP8 | STE | FP8 fake quant |
| Routed experts | non-fused MegaMoE: weights MXFP4 each step, FP8 activations, grouped FP8×FP4 GEMM; EP all-gather + all-to-all; combine in slot order in FP32 with the shared expert first | STE | MXFP4 + FP8 |
| Final collapse | Core torch FP32 aggregate (vLLM `hc_collapse` not wired) | autograd | — |
| Final norm | vLLM `rms_norm_cuda` | owned | — |
| Head | vLLM `bf16_fixed_mm`, FP32 logits, torch divide by temperature | owned | BF16 |
| Logprob | Core vocab-parallel cross entropy | autograd | — |
| MTP | not built | — | — |

Gaps to bitwise vLLM BI=1 logprobs at that point: the attention core (Core, not vLLM mega/FlashMLA);
attention projections on TE/einsum; norms not on vLLM kernels; RoPE; compressor; indexer precision
and top-k tie-break; KV quantization not simulated; Engram projection/gating/all-reduce order; mHC
fused path unused; final collapse; CE/logprob and temperature order; MoE verified only at EP1 small
shapes; no weight sync; no verl wiring (`impl_cfg` keys rejected, no `unpack_forward_output`, no
HF-name export or re-quantization); CP/PP numerics; no GPU end-to-end result.

## Principle

- Every module of the training forward **calls the same inference op, in the same order and
  layout**. No re-implementation of arithmetic. The backward is owned (module VJPs and STE).
- New kernels are written in CuTe DSL / CUDA / CUTLASS (not Triton or TileLang), each verified
  bitwise or against a pre-set tolerance on all target shapes.
- V2 model runner only; TP=1 (DEP) on the rollout.
- Proxy first (about 4 layers covering every layer type: cr=0, cr=2 owner/reader, cr=1, index
  owner), then the full model.

## Attention core interface

Shared by the module track and the two core tracks:

```text
attention_core(q, swa_kv, comp_latent, topk_local, positions, attn_sink, layer_cfg, rope_cache) -> o
  q:           wq_b output, bf16 [S_local, H=64, 512] (mega: fused permuted layout)
  swa_kv:      kv_norm output, bf16 [S, 512], no RoPE applied
  comp_latent: compressor latents of this rank's group-boundary tokens ((pos+1) % cr == 0),
               in token order, [N_local, 512], no RoPE; None on cr=0 layers
  topk_local:  indexer local compressed indices, int32 [S_local, top_k], ascending, -1 at the end
  positions:   int64 [S_local] (with CP: global in-document positions of this rank)
  attn_sink:   fp32 [H], trainable
  o:           (data, scale, grad)
```

- Output: `data` / `scale` are the FP8 `[T, 8, 4096]` values and scales of the inference
  `fused_inv_rope_fp8_quant` (group 32, TMA-aligned packed UE8M0), fed directly to the wo_a
  `fp8_einsum`. `grad` is a 0-stride BF16 `[T, 8, 4096]` tensor whose values are meaningless; it only
  carries the gradient. The wo_a Function takes it as an input and returns d(dequantized o) as its
  gradient: an identity STE through the FP8 output.
- `rope_cache` is the layer's vLLM `rotary_emb.cos_sin_cache` (FP32 `[1048576, 64]`), built on the
  GPU with `build_deepseek_v4_rope`. A CPU-built cache differs by 1–3 FP8 elements in the indexer q.
- Multi-document packing: pass the global `cu_seqlens`; `positions` are in-document positions.
  `cu_seqlens` can be a GPU tensor; the plan is cached by tensor object (including `_version`), so
  within a micro-batch only the first layer reads it to host and every later layer runs without a
  host sync, provided all layers share the same tensor object.
- CP: all inputs are this rank's contiguous rows; the core all-gathers KV and latents and
  reduce-scatters in FP32 in the backward. The `attn_sink` gradient is a per-rank partial sum reduced
  by the DP×CP gradient reduction. Forward is bitwise equal to CP=1.
- The training process must set `VLLM_BATCH_INVARIANT=1` (the core raises otherwise).
- Layout: with the mega core, `wq_b` rows are permuted with `q_fused_permutation` and `wo_a` columns
  with `o_fused_permutation` inside the forward; parameters stay in release order, so export needs no
  inverse permutation. With FlashMLA both stay in release order.

## What the deployment forward does

1. **Assembled forward, bitwise vs vLLM BI=1 prefill.** The whole model in vLLM's layer order over
   the native mLite parameter tree (construction, release loader, DistOpt, checkpoints stay native):
   wq_a+wkv / q_norm+kv_norm / wq_b (MXFP8 BI GEMMs, fused-layout row permutation), compressor
   (`fused_save_compress_norm`), frozen indexer (same ops recomputed, BI top-k, candidate blocks),
   wo_a (`fp8_einsum`) / wo_b, Engram (MXFP8 wkv + fused post-wkv gate), mHC (`pre_delayed` /
   `shifted_post_pre` / `post_tilelang` / `hc_collapse`), MoE via DeepGEMM `fp8_fp4_mega_moe` over the
   EP group (image sentinel ids routed with `ffn.gate.bias_vl` as vLLM does), head and
   `native_log_softmax`. Compressor `wkv`/`wgate` and Engram q/k are FP32 masters; the forward rounds
   them to BF16 with round-to-nearest-even, the same conversion the export uses.
2. **Owned backward on tensor cores.** Weight and input gradient GEMMs (linear VJPs, wo_a, head,
   MegaMoE experts and shared expert) take the decoded operands in BF16 (exact) and accumulate in
   FP32; weight gradients go straight into the FP32 `main_grad`. Before this, with TF32 off under
   BI, they ran as FP32 SIMT GEMMs, about 76% of the attention-path backward time.
3. **Mega attention core.** The forward replays the vLLM mega-attention op sequence; the backward is
   an owned sm_100a kernel (TMA gather4 K loads, tcgen05/UMMA GEMMs with TMEM accumulation,
   persistent CTAs, fused RoPE/layout, no host sync after the first layer of a micro-batch). dKV
   accumulates with FP32 atomics by default; a debug switch makes dKV reproducible run to run at
   1.3–1.5× backward time. At cr2 / 8320 tokens the earlier mma.sync backward kernel took 9.33 ms
   (forward + backward 11.66 ms); the UMMA kernel takes 4.04 ms (forward + backward 6.41 ms).
4. **FlashMLA sparse core.** Bitwise forward on both the gather path and the paged decode path
   (auto-selected by whether the installed vLLM has `sparse_paged_indices`); the backward reuses
   Core's cuDNN DSA sparse backward. vLLM's FlashMLA library must load before pip `flash_mla` (they
   share weak C++ symbols; with the wrong order the BI paged kernel reports missing TOPK_LENGTH /
   BATCH_INVARIANT features on the first layer).
5. **Context parallel.** Contiguous equal THD slices; compressor kv/score rows all-gathered within
   CP with a reduce-scatter backward; indexer K uses all latents, q only this rank's rows; the Engram
   hash uses the full input ids, then slices. CP2 and CP4 × EP4 forward bitwise equal to CP1 and to
   vLLM.
6. **Training lifecycle.**
   - Activation recompute `full` / `attn` / `moe` with reentrant checkpointing. Compressor latents
     are explicit segment inputs/outputs and the no-grad indexer state is snapshotted at segment
     entry, so the replay reads the same cross-layer state. Before this, the deployment forward
     ignored `impl_cfg.recompute` and kept every layer's activations, which did not fit the recipe.
   - Parameter readiness: the forward runs Core DDP's forward pre-hooks before reading raw
     parameters, so `overlap_param_gather` completes after each optimizer step even with no export in
     between. Without it, the forward after the second of two steps with no export read stale shards
     (11701/11701 tokens differed from a forward after a forced sync; 0 after the fix). Only recipes
     with mini-batch < train batch hit this. Separately, export must call
     `start_param_sync(force_sync=True)` first, or the export reads other ranks' stale shards and θ1
     mismatches.
   - Packing and dynamic batching: verl's global token-mean loss reaches every token with weight 1/N
     independent of micro-batch grouping, document order, CP split and DP layout; the native loss no
     longer depends on the CP split; MegaMoE backward handles empty EP chunks (each EP rank runs the
     same number of chunks, so a rank with fewer rows had an empty tail chunk that crashed in
     `dequantize_activation`'s `reshape(..., -1)`); `DSV41_MEGAMOE_MAX_TOKENS` bounds the trainer
     MegaMoE chunk.
   - Engram: q/k gradients reach the FP32 masters without a BF16 rounding of the leaf gradient; the
     frozen tables stay out of checkpoints (resume uses the release bytes).
7. **MegaMoE backward contract.** The backward uses the operands the fused kernel actually consumes:
   the GEMM1 accumulator rounded to BF16, and the FP32 activation re-quantized 1×32 UE8M0 E4M3 with no
   BF16 in between, one expert at a time. The earlier backward's intermediate differed from the
   kernel's by 1.3e-2. The embedding gradient accumulates into the FP32 `main_grad`.
8. **Vision sentinel ids.** vLLM routes ids 129264–129268 with `ffn.gate.bias_vl` and treats 129264
   as a dead Engram token. The trainer must do the same, and `bias_vl` must be exported on every sync
   (a level-2 sleep and layer-wise reload drops it otherwise). See
   [SWE-bench Pro](swe-agentic-rl.md#vision-sentinel-ids) and the
   [isolation guide](../../../skill/guides/debugging-mismatch/01-isolate-the-side.md).

## Forward parity checks

Proxies: `proxy4` (layers 0–3) and `proxy6` (layers 0, 1, 14, 15, 20, 24; compress ratios 0, 0, 2,
2, 1, 1; covering Engram, index owner/reader and candidate blocks).

| Check | Setup | Result |
|---|---|---|
| Layer chain vs a vLLM dump, mega core | proxy6, 3000 tokens, no-grad and grad, 2 packed documents | 110/110 taps bitwise each; prompt logprobs 2999/2999, 2999/2999, 2998/2998 |
| Layer chain vs a vLLM dump, FlashMLA core | proxy6, 3000 tokens, 2 packed documents | 110/110 taps bitwise; logprobs 2999/2999, 2998/2998 |
| Assembled model, EP4 | mega 3000 tokens, mega 2 docs, FlashMLA 3000 tokens | log_probs bitwise on all 4 ranks |
| CP2 × EP4 vs vLLM and vs CP1 | mega, 3000 tokens | forward bitwise on all ranks |
| Export round trip | full release | BF16/FP32 7009 tensors byte-identical; 6965 quantized pairs with identical decoded values (FP8 block weights/scales and MXFP4 code bytes can differ because a power of two can sit in the code or in the scale) |

## Packing, dynamic batching and loss normalization

proxy4, 2 nodes, EP8; thresholds fixed before the runs. Real verl
`MegatronLiteEngine.forward_backward_batch`, one frozen global batch of 64 sequences (72062 loss
tokens, responses of 16–6000 tokens). Variants: one sequence per micro-batch (a), dynamic batching at
16384 tokens (b) and at 7000 tokens with shuffled order (c), CP1 vs CP2.

| Comparison | logprob mismatch (old-logprob / grad path) | loss rel. diff | grad-norm rel. diff | global grad rel-L2 | dense per-param max |
|---|---|---|---|---|---|
| A/A rerun (CP1 / CP2) | 0 / 0 | 0 | 1.4e-5 / 5.6e-5 | 1.9e-4 | 1.5e-3 |
| b, c vs a (CP1 and CP2) | 0 / 0 | ≤ 5.2e-8 | ≤ 5.4e-5 | 6.5e-4 – 8.7e-4 | ≤ 2.0e-3 |
| CP2 vs CP1 (a / b) | 0 / 0 | 0 / 2.4e-8 | ≤ 5.1e-5 | 6.8e-4 / 9.3e-4 | ≤ 2.0e-3 |
| Planted bug: per-micro-batch denominator | 0 / 0 | 3.2 – 7.5 | 3.2 – 7.5 | 3.24 – 7.54 | — |

Above 1e-2 only mHC scalar sums (cancellation, max 7.8e-2) and a few low-token experts (≤ 1.6e-2),
reported, not gated. Full recompute vs none: every recomputed segment output bitwise equal to the
original (896 segments), logprobs 0/72062 on both paths, gradient rel-L2 2.0e-4 – 2.3e-4 (A/A floor
1.7e-4).

Recompute memory (proxy4 CP2, forward + backward peak minus static, GiB, at 2048 / 8192 rows per
GPU): none 15.5 / 26.5; `full` 7.7 / 19.6; `full` + MegaMoE chunk 2048 7.7 / 13.7; + tensor-core
backward 5.0 / 11.0. Extrapolated to the full model: about 30 GiB of activations at 5120 tokens per
GPU, worst case 48 GiB when verl's grouping adds one more sequence to a micro-batch, with about 61
GiB available; with the MegaMoE chunk at 8192 the worst case is 78 GiB and OOMs. The MegaMoE backward
all-gathers each chunk over the whole EP group (`EP × capacity × 4096` BF16 for `a_all` and `g_all`:
4 GiB each at EP64 and 8192 rows, 1 GiB each at 2048 rows), which is why the recipe uses 2048.

## Two-step RL through verl (C8)

proxy4, 2 nodes, EP8 actor CP2, rollout DEP4 × 2, BI=1 mega, DSpark, level-2 sleep, optimizer-state
offload, recompute full, packed micro-batches. Each run checks θ0/θ1/θ2: the old-logprob (no grad)
and train-forward (grad) paths token by token against the rollout.

| Run | Result |
|---|---|
| Distinct siblings (`full_determinism=False`), with sampled image-sentinel ids, after the `bias_vl` fix | 0 mismatch on both paths (107524 / 102526 / 104776 tokens); 16 sequences contain sampled sentinel ids, all 0 diff (before the fix each such sequence diverged after the sentinel); `rollout_probs_diff_max` 0 |
| Packed, repacked old-logprob batch, tensor-core backward | 0 mismatch (45326 / 63290 / 52376 tokens); three repackings 0/102112 |
| Packed, distinct siblings | 0 mismatch (54390 / 68518 / 71284 tokens); 64 distinct responses; sync payload identical across 8 ranks for syncs v1–v4 |

Checkpoint (proxy, 2 nodes): resume from step 2 gives step-3 old logprobs bitwise equal to the
uninterrupted run (110632 tokens); re-saving right after resume is byte-identical to the original
checkpoint (17509 items).

`full_determinism` on the rollout: with True, siblings of one prompt share a seed (only 16 of 64
responses distinct in this proxy), and it also sets NCCL/cuBLAS determinism in the vLLM workers. With
False and before the `bias_vl` fix, one sequence per run mismatched on a contiguous response tail;
the root cause was the sentinel routing, not seeds or NCCL.

## Independent gradient check (C7)

FP64 reference that does not call the tested backward.

- Isolated modules (proxy6, thresholds fixed in advance): attention parameters, dX, experts, shared
  expert / router and Engram q/k all pass before and after the tensor-core backward (attention
  parameters median rel 4.4e-3, max 5.7e-3; experts max 2.6e-3; Engram q/k max 9.96e-3 against 1e-2);
  2% planted errors are caught. Three isolated mHC scalar entries (gradient sums of about 6e-4 with
  cancellation) sit at the FP32 floor just above 1e-2.
- Whole model: the number of attention / expert entries above threshold varies run to run for the
  same code and image (attention 8–15, experts 3–13), so single counts are not compared between
  commits. All failing experts are low-token L1 experts where the FP64 reference amplifies the
  upstream activation-gradient drift (rel = 3.5–5× its upstream delta). Feeding the reference the
  trainer's own upstream gradient, every expert dW matches to ≤ 2.9e-3 (4098 experts × 4 runs).
  On the final parity commit the whole model had 35 entries above 1e-2 (attention 12, experts 13,
  mHC 1, others). Not waived.

## Attention: mega vs FlashMLA

Same node, image and code; proxy6 attention weights, layers with cr 0 / 2 / 1; single- and
multi-document packing; CP1 and CP2; mean of 10 after 3 warmups. Core only, forward / backward /
forward+backward in ms:

| CP | cr | tokens | mega | mega deterministic | FlashMLA | FlashMLA deterministic |
|---|---|---|---|---|---|---|
| 1 | 0 | 8192 | 0.45 / 1.48 / 1.93 | 0.43 / 1.96 / 2.39 | 1.73 / 2.46 / 4.20 | 1.73 / 4.18 / 5.91 |
| 1 | 2 | 8192 | 0.86 / 4.36 / 5.22 | 0.87 / 5.91 / 6.78 | 2.59 / 4.29 / 6.88 | 2.55 / 8.13 / 10.68 |
| 1 | 2 | 16384 | 1.57 / 8.60 / 10.17 | 1.58 / 11.31 / 12.89 | 4.65 / 8.41 / 13.06 | 4.59 / 16.34 / 20.92 |
| 2 | 2 | 8192 | 1.09 / 1.99 / 3.08 | 1.09 / 2.75 / 3.85 | 3.02 / 2.60 / 5.63 | 3.04 / 4.63 / 7.66 |

FlashMLA / mega forward+backward over all 36 configurations: 1.28 – 2.60. The backward kernels are on
par; the difference is FlashMLA's forward (paged decode kernel plus cache write and plan).

Full attention layer (projections, compressor, indexer, core, wo_a/wo_b; forward under grad,
backward to x and all weights): before the tensor-core backward the two cores were within 4% (CP1
cr2 8192: 83.9 vs 84.4 ms) because FP32 SIMT GEMMs dominated; with it, the mega path is 2.9–6.7×
faster (cr2 8192: 84.0 → 21.2 ms) and the core is 13–30% of the path. The mega backward kernel is
still about 7× its forward on ratio>0 layers (cr2, 8320 tokens: 0.55 ms forward vs 4.04 ms
backward); the rest is L2-bound dKV atomics and per-row boundaries.

The deterministic sparse-attention backward needs a 26.55 GiB workspace at 32K context with CP4,
which OOMed the SWE-bench Pro memory check; the recipe runs it non-deterministic (gradients are not
bitwise reproducible run to run; logprobs are unaffected).

## SFT: `impl=vllm` vs `impl=lite`

Question: does training through the quantized deployment forward learn as well as the native BF16
path? Same source checkpoint, data and order (GSM8K main/train, fixed shuffle), labels and masks,
packing, DistOpt Adam, clip 1.0, seed; global token-mean loss; grad norm from DistOpt after reduction
and before clipping; param sync forced after every step. The two arms differ only in `--impl`.

### Proxy (proxy6, 2 nodes, 300 steps)

Global batch 64 (DP8, 2 micro-batches), lr 1e-5 warmup 10 cosine to 1e-6, weight decay 0.1, TP1 EP8
DP8 CP1.

| Arm | mean loss, steps 1–10 / 11–50 / 51–100 / 101–200 / 201–300 |
|---|---|
| vllm | 16.16 / 7.70 / 2.68 / 1.67 / 1.37 |
| lite | 14.98 / 4.43 / 1.76 / 1.16 / 0.85 |
| vllm, deterministic dKV vs atomic (A/A) | loss within 0.009, grad norm within 2–5% |
| lite rerun | 300/300 steps bitwise |

- Step 1 agrees (loss 16.4088 vs 16.4169, per-category grad norm within 1%, embedding 7%).
- After that `impl=vllm` trains systematically slower: last-10-step loss 1.299 vs 0.745, grad norm
  25–55% higher.
- Freezing routed experts (lr 0) in both arms still leaves 0.43 nat, so most of the gap is on the
  dense side. In `impl=vllm`, 51% of routed-expert BF16 values changed by step 50 but only 0.53% of
  MXFP4 codes (MXFP8/FP8 matrices: 79% vs 12%).
- The proxy is an extreme regime: initial loss 16.4 (above the uniform 11.8), grad norm about 50
  against clip 1 every step.

### Full model (16 nodes, 150 steps, about 2.6 epochs)

EP64, CP1, DP64, global batch 128, micro-batch 1; lr 1e-5 warmup 10 cosine to 1e-6; recompute full;
chunked optimizer offload. GSM8K: 7473 samples, 58.4 steps per epoch, so steps 59 and 117–118 start
epochs 2 and 3. Median step time: vllm 14.0 s, lite 8.3 s; peak memory: vllm 142.7 GiB, lite 140.1 GiB.
A third arm (vllm with atomic instead of deterministic dKV) measures A/A noise.

θ0: step-1 loss 0.91642 (vllm) vs 0.91620 (lite); grad norm 8.420 vs 8.363 (+0.7%); per-group grad
norm within 3.4% (experts +1%, norm +1.3%, Engram +3.4%, compressor +1.8%, embedding +2.5%),
`attn_sink` +7.7% on an absolute value of 0.0047.

| Steps | lite loss | vllm loss | gap | lite grad norm | vllm grad norm |
|---|---|---|---|---|---|
| 2–5 | 0.840 | 0.906 | +0.066 | 7.28 | 8.35 |
| 6–10 | 0.479 | 0.793 | +0.314 | 1.92 | 6.30 |
| 11–20 | 0.395 | 0.459 | +0.064 | 0.80 | 1.61 |
| 21–40 | 0.371 | 0.388 | +0.017 | 0.60 | 0.84 |
| 41–58 | 0.365 | 0.370 | +0.005 | 0.51 | 0.53 |
| 59–80 (epoch 2) | 0.206 | 0.310 | +0.104 | 0.66 | 0.55 |
| 81–117 | 0.182 | 0.290 | +0.108 | 0.55 | 0.65 |
| 118–135 (epoch 3) | 0.054 | 0.219 | +0.166 | 0.51 | 0.64 |
| 136–150 | 0.045 | 0.216 | +0.171 | 0.57 | 0.67 |

- Epoch 1: vllm starts slower (largest gap 0.37 at step 8), then catches up to within 0.005 by
  steps 41–58.
- The gap steps up at each epoch boundary. Each time samples repeat, lite's loss drops sharply
  (0.36 → 0.21 → 0.05) and vllm's less (0.37 → 0.30 → 0.22): on unseen data both learn about the same;
  the difference is in fitting samples already seen. vllm's loss is above lite's on 98.7% of steps.
- A/A (deterministic vs atomic dKV): steps 1–2 loss bitwise equal; mean |loss diff| 0.0011, max
  0.0047. vllm vs lite mean |loss diff| 0.098, about 90× A/A. Single-step grad-norm spikes occur in
  every arm at different steps (e.g. 3.36 vs 0.62 at the same step between the two vllm arms) and
  are clipped.

Quantization visibility at step 150 (1×32 UE8M0 approximation; share of codes that changed / share
of BF16 values that changed):

| Group | vllm arm's own update | lite arm's update passed through the vLLM quantizer |
|---|---|---|
| Routed experts (MXFP4) | 0.39% / 47.9% | 0.23% / 45.0% |
| MXFP8 / FP8 matrices | 8.6% / 73.4% | 5.7% / 60.1% |
| Shared expert | 12.8% / 81.6% | 8.5% / 71.4% |

Reading: lite reaches a training loss of 0.045 mostly through changes smaller than one quantization
step, which the quantized forward cannot see. The gap comes from the forward's weight quantization
(on the proxy, BF16 dense weights recover most of it). No sign of a bug: θ0 loss and grad norm
match, A/A noise is two orders of magnitude below the gap, and epoch 1 converges to the same level.
For RL: with one pass over non-repeated samples both paths learn at a similar rate; multi-epoch
fitting of the same samples under-fits on the quantized path.
