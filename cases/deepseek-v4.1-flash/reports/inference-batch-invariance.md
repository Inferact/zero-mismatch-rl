# Inference batch invariance (vLLM #62365)

Goal: with `VLLM_BATCH_INVARIANT=1` (BI=1) and CUDA graphs on, token ids and FP32 logprobs are
bitwise identical regardless of batch composition and order, DP rank, `max_num_batched_tokens`,
prefix cache, decode vs re-feed and MTP on/off, at a throughput cost the RL recipe can afford.

Hardware and topology for every number here: GB200, one node × 4 GPUs, DEP4 (TP1 × DP4 + EP4),
DSpark with 5 draft tokens, FULL_AND_PIECEWISE graphs, `nvfp4_ds_mla` KV cache, model revision
`dba1be0a40`.

Versions: the gate matrix and the 12-pair throughput were measured on the stack before it was
rebased onto current vLLM main; after the rebase the full gate matrix was re-run on an image built
from the branch (all 0 diff, 6711 token/logprob records bitwise equal to the pre-rebase image). The
sparse FlashMLA path numbers were measured at `f1ff0f0341`. Later commits (RL integration fixes,
small-M router, DeepInfer calls) were tested as Python overlays on that image.

## Gates

Each gate uses 16 prompts of 7–12000 tokens, greedy, and compares token ids, logprobs and
`prompt_logprobs`. Numbers are "rows that differ / rows compared"; the max logprob difference is 0
in every case.

| Gate | mnbt 2048 | mnbt 8192 | mnbt 16384 |
|---|---|---|---|
| alone vs batch / reversed; batch vs reversed | 0/16 ×3 | 0/16 ×3 | 0/16 ×3 |
| DP rank 0 vs ranks 1–3 | — | 0/16 | — |
| alone pinned to rank 0 vs batch on unpinned ranks | — | 0/16 | — |
| prefix-cache miss vs hit | 0/16 | 0/16 | — |
| decode vs `prompt_logprobs` re-feed | 0/16 | 0/16 | 0/16 |
| full prefill vs truncated | 0/13 | 0/13 | 0/13 |
| shared-prefix hit vs miss | 0/3 | 0/3 | 0/3 |
| MTP (seeded alone / batch / reversed) | 0 (acceptance length 1.548 / 3.634 / 3.666) | 0 | 0 (seeded only) |
| mixed-batch combinations | — | 0 | — |

Also: across runs with mnbt 2048 / 8192 / 16384, alone and batch, 6 pairwise comparisons all 0/16;
at mnbt 8192, batch, all-rank, all-rank repeat, filler and unpinned vs alone, and all-rank vs its
repeat, all 0/16.

Sparse FlashMLA path (`FLASHMLA_SPARSE_DSV41`, no mega tier, at `f1ff0f0341`): full gate matrix 0
diff over 28 rows across mnbt 2048 / 8192 / 16384, MTP, batch combinations, cross-rank, graph vs
eager and short prompts of 1–31 tokens; `tests/v1/attention/test_deepseek_v4_swa_visible.py` 167
passed under BI=0 and BI=1.

## Paired throughput

Workload: 8192 input / 1024 output tokens, DSpark 5 tokens, synthetic acceptance length 3.51,
temperature 0; each pair runs one warmup round then three measured rounds per arm. Both arms use the
same image and recipe; only `VLLM_BATCH_INVARIANT` differs.

Mega tier (opt-in, `FLASHMLA_MEGA_ATTN_DSV41`), 12 pairs over 6 nodes, alternating AB / BA:

| | c=1 | c=8 | c=32 | c=64 |
|---|---|---|---|---|
| Median BI=1 / BI=0 | 0.911 | 0.922 | 0.919 | 0.907 |
| Lowest single pair | 0.905 | 0.915 | 0.909 | 0.876 |
| BI=0 median tok/s | 411.8 | 2263.1 | 5960.2 | 8896.2 |
| BI=1 median tok/s | 374.9 | 2092.7 | 5464.5 | 8109.4 |

- Pairs below 0.90 occur only at c=64: 4 of 12 (0.876, 0.889, 0.889, 0.893), on 4 different nodes.
  The c=64 median clears 0.90 by 0.007 and may fall below it on a re-run.
- Pair-to-pair noise is about ±2%. Earlier builds of the same stack measured 0.907 / 0.929 / 0.917 /
  0.911 (6 pairs) and 0.913 / 0.927 / 0.931 / 0.908 (8 pairs). The BI=0 arm is within 2.9% of the
  frozen BI=0 baseline of 410 / 2216 / 5795 / 8909 tok/s.

Sparse FlashMLA path (the RL recipe default), 4 pairs: 0.911 / 0.890 / 0.876 / 0.835.

RL rollout step (128 trajectories per DEP4 node, response up to 8192, real rejection sampling),
BI=1 / BI=0: sparse FlashMLA 1.072 fixed length, 1.023 natural EOS; mega attention 1.034 / 1.036.

## Deviations from vLLM main

"Forced" means BI=1 requires it. "Opt-in" means off by default; when on it is itself bitwise batch
invariant but rounds differently from the default BI path, so the trainer must call the same
kernel. "Both arms" means BI=0 also takes it.

1. **Attention.** Default under BI: sparse FlashMLA; every causal row (decode, spec verify,
   prefill, `prompt_logprobs` re-feed) runs in one call on the unsplit sparse decode kernel
   (`flash_mla_with_kvcache(enable_batch_invariant=True)`), reading the fp8 SWA and nvfp4 compressed
   caches directly; the BI CUDA op `sparse_paged_indices` derives each token's slots and lengths from
   (request, position) only. Opt-in: `VLLM_BI_MEGA_ATTN=1` selects `FLASHMLA_MEGA_ATTN_DSV41`
   (faster, gates 0 diff).
2. **KV dtype.** Under BI `auto` resolves to `nvfp4_ds_mla`, as BI=0 does (it was `fp8_ds_mla`),
   and is pinned at launch. Changes a default, so the numerics match BI=0.
3. **MoE.** MegaMoE as in main. The EP combine becomes a BF16 reduce-scatterv plus an FP32 sum in
   rank order. Forced: the summation order must not depend on rank arrival order.
4. **CUDA graphs.** The recipe's FULL_AND_PIECEWISE stays, as in BI=0; FULL_DECODE_ONLY is a
   fallback. The spec-verify batch keeps the FULL decode graph. FAP gates are 0 diff and faster.
5. **SWA bounded replay.** Exact under BI: decoder trimming keeps W+(n−1)(W−1) rows; a prefix hit
   recomputes W+L(W−1) tokens. Forced: BI=0 recomputes only the last W rows, which is an
   approximation, so a prefix hit or a re-feed would differ from a full forward.
6. **MXFP8 GEMM.** Default: K32 `block32_scaled_mm`. Opt-in `VLLM_BI_MXFP8_GEMM=tensor_core`: a
   fixed-tactic CuTe DSL GEMM with split-K tactics excluded. Higher tensor-core throughput.
7. **mHC.** Default: TileLang with a fixed split. Opt-in `VLLM_BI_MEGA_MHC=1` (fixed 16 splits,
   aux layers on Mega too) and `VLLM_BI_MHC_OVERLAP=1` (split overlap kernel at every boundary, the
   coefficients produced on a side stream inside the FULL graph). BI=0's mega_mhc/TileLang switches
   implementation by M.
8. **BF16 projections** (lm_head, router, compressor score). Default: K256 `bf16_fixed_mm`. Opt-in
   `VLLM_BI_BF16_GEMM=tensor_core` (DeepGEMM in deterministic mode). A plain GEMM picks tile and
   split by M.
9. **SWA block stride.** Aligned to `lcm(swa, compressed, 32)`. Both arms; the full model is already
   aligned, so BI=0 is unchanged. The CuTe DSL K gather needs a 32 B stride.
10. **DSpark.** Spec decoding is allowed under BI with an 8-row ring; non-causal draft rows use the
    original decode kernel. Forced: main rejects spec under BI.
11. **Sparse prefill combine tail.** `[combined_len, width)` is written as -1. Both arms (bug fix): a
    reused workspace kept stale indices that leaked into results.
12. **Other BI ops.** Deterministic top-k, stable expert permute, fused SiLU-mul quant,
    `hc_collapse`, NVFP4 gather, BI logits/logprob. Forced: main has no deterministic version.
13. **Mega Gate.** Opt-in `VLLM_BI_MEGA_GATE=1`: DeepGEMM `bf16_mega_gate` for every M in
    deterministic mode; the precompile key includes top_k and the routing config. BI=0 uses
    GateLinear for M≤16; GateLinear plus `fused_topk_bias` switches with M.
14. **Layer-staged exact replay.** Opt-in, performance only: `VLLM_BI_REPLAY_STAGE_LAYERS=2` replays
    two layers per stage, each stage computes only the tail rows it needs, and each stage's graph
    carries a padding mask. Saves about 43% of the exact-replay prefill cost. The padding mask also
    fixed an out-of-range read of Mega Gate across DP ranks.
15. **Mega decode/prefill SWA-first layout.** No change to the Triton combine kernel; under BI the
    mega prefill calls the CUDA op `combine_swa_first_indices`, which reads the same
    query_start_loc / seq_lens / gather_lens as the combine and writes the decode kernel's slot
    layout [SWA padded to the window and filled with -1; top-k; -1 up to round_up(top_k+window,128)]
    with length window + max(min((pos+1)//cr, k), 0). Forced when mega attention is on: decode and
    `prompt_logprobs` must be bitwise equal, and pos has one source. Verified bitwise against the
    earlier Triton SWA-first output on 168 cases; 57 unit cases pass in both modes; 18 decode ==
    prefill cases pass.
16. **Decode compressed length by position.** Under BI the length of a causal decode row is
    min((pos+1)//cr, k), clamped at ≥0, the same formula as prefill (it was the count of valid
    entries). Forced when mega attention is on: with -1 inside the top-k, or more valid entries than
    the positional length, both sides must mask the same way.
17. **`empty_cache` before MegaMoE allocates symmetric memory.** Both arms, no numeric change: after
    profiling the caching allocator holds memory and the symmetric allocation OOMs at mnbt 16384.
18. **lm_head falls back to cuBLAS with a one-time warning.** An fp16 lm_head or K%8≠0 goes back to
    cuBLAS under BI with a "BI not guaranteed" warning, so other models' configurations still start.
    DeepSeek-V4.1 is bf16 with K=5120 and is unaffected.
19. **MXFP8 prequant.** Opt-in `VLLM_BI_MXFP8_PREQUANT=1`: the fused Q/KV RMSNorm and the attention
    output produce MXFP8 directly for the fixed GEMM. Only with the tensor-core tier.

Later opt-in switches on the PR, all off by default and all requiring the same switch on the
trainer:

- `VLLM_BI_BF16_SPLITK=4`: BF16→FP32 projections with N ≤ 384 (router, indexer `weights_proj`,
  draft gate) run a CuTe DSL cluster split-K kernel with min(S, ceil(K/256)) K slices for every M;
  the slice count depends only on K. `VLLM_BI_TOPK_TRITON=1`: the sqrt-softplus top-6 runs the
  per-row Triton kernel under BI. Decode BI=1 / BI=0 with both on: c=1 1.092 → 1.037, c=8 1.051 →
  1.022. The Triton and CUDA top-6 give the same ids with weights differing by up to 9e-8, so the
  trainer must run the same switches.
- `VLLM_BI_WO_FP8_EPILOGUE=1` (mega attention only): `wo_a`'s `fp8_einsum` writes MXFP8 codes and
  UE8M0 scales from its epilogue and `wo_b` consumes them on DeepGEMM, bit-equal to the default path.
  A follow-up keeps `wo_b`'s DeepGEMM scales as a Parameter so a layer-wise reload refreshes them;
  without it the RL proxy lost bitwise agreement after the first weight update
  (`rollout_probs_diff_max` 0.171 / 0.239).
- `VLLM_BI_INDEXER_DEEPSELECT=1`: the indexer's index-ordered top-k over zero-based rows runs on the
  vendored DeepSelect (DeepInfer's indexer call). Ties at the boundary follow a fixed visit order that
  depends only on row length and dtype, so it is row invariant but breaks ties differently from the
  native kernel; the trainer passes `zero_starts=True` under the same switch (Megatron #271).
  Row invariance on GB300: 262 shapes (top-k 512 / 1024 / 2048 / 4096, row lengths up to 600K at
  segment and window edges, batch 1..2048 across the SM-count edge, 2 strides, 3 positions), 0
  batch-dependent mismatches. Speed vs the native BI top-k on GB300 (rows × length, µs): 8×2K
  17.4 → 6.6, 4096×8K 475 → 211, 384×32K 365 → 50, 8192×64K 10265 → 1843; end to end at 128K
  context 8.34 → 7.43 ms per token.

## RL integration fixes

Found while running two-step RL with verl and mLite against the branch (level-2 sleep with
`sleep_preserve_parameter_names`, layer-wise reload):

- The mega-attention load permutations (`q_perm` int64 64×512 shared by each layer's `wq_b` weight
  and scale, `o_perm` int64 4096 for `wo_a.weight`, `o_chunk_perm` int64 128 for
  `wo_a.weight_scale`; 40 target layers + 3 drafter layers) were allocated in the sleep weights pool
  and held only by the weight-loader closure. Level-2 sleep discarded them, so reload after wake
  permuted `wq_b`/`wo_a` with garbage. The loader now keeps a host copy. Test:
  `tests/kernels/test_dsv41_mega_attn_layouts.py::test_permute_on_load_owns_its_permutation`.
- The Engram peer-exchange symmetric-memory pointer tables (`buffer_ptrs_dev`,
  `signal_pad_ptrs_dev`, 4 int64 each, read by `barrier()`) were allocated from the weights pool by
  `symm_mem.rendezvous` during model init and zeroed by level-2 sleep: illegal address on the first
  forward after wake. `rendezvous` now runs in a private `MemPool`. Test:
  `tests/basic_correctness/memory/cumem/test_cumem.py::test_symm_mem_handle_outside_weights_pool_survives_level2`.
- (V2 runner, not model-specific.) An idle DP rank padded its single dummy request to the busiest
  rank's token count, which can exceed `max_model_len`; its block table is only `max_model_len`
  wide, and the BI nvfp4 gather read past it and hung while peers timed out in the MegaMoE barrier.
  `make_dummy` now splits the dummy batch into `cdiv(num_tokens, max_model_len)` requests. Test:
  `tests/v1/worker/test_gpu_input_batch_v2.py::test_make_dummy_dp_padding_respects_max_model_len`.

Validation: full model, level-2 sleep with only the preserve list, then layer-wise reload: 5080
target and 400 drafter tensors bitwise equal to a cold load; the crash repro (two ≥800-token
prefills on one DP rank) passed 27/27 rounds with and without DSpark.

How the two lost states were found: traversing loader closures found `wo_a`'s permutation; with
only that fixed, non-scratch state was 0 diff but the next forward still hit an illegal address;
an allocation-stack trace attributed all 0.65 GiB of ownerless bytes in the weights pool (runner
model state, sampler and structured-output buffers built in `load_model`, allocation slack of large
parameters, and the four Engram rendezvous tables); restoring only the four pointer tables after
wake gave 0 diff, so the runner state did not need preserving. With both fixes and no diagnostic
write-back: 5080 target + 400 drafter tensors 0 diff, and the node's used host memory while
sleeping was 319 GiB.

## Not covered

1. `prompt_logprobs` with very large row counts can still JIT-compile the lm_head GEMM while
   serving: the tensor-core tier precompiles only up to max_num_seqs × (1 + spec) rows.
2. −0 / +0: in the EP combine a non-owner +0 row can turn the owner's BF16 −0 into +0, so DEP>1 is
   not bitwise equal to an unsharded run. Consistency within a DEP group is unaffected.
3. The standalone (non-fused) mHC prenorm fallback picks its tile by M and is not rejected under BI;
   the shipped configuration (Mega or overlap mHC) never reaches it.
4. Validated on GB200 only for serving gates and throughput; kernels build for sm80/90/100/110/120
   but have no numeric tests on other architectures (DeepSelect and FlashMLA were later checked on
   GB300).
5. MTP acceptance lengths were only observed in gates; throughput used a synthetic acceptance length.
6. Not exercised end to end: the PIECEWISE segment of exact replay under FAP, KV connectors (P/D),
   1M `max_model_len`.
7. Prefix-hit recompute cost grows with the number of hits; agentic multi-turn traffic was about
   12–13% slower before the SWA replay changes described in
   [SWE-bench Pro](swe-agentic-rl.md#prefix-hit-replay).
8. Only the V2 model runner is supported under BI.
