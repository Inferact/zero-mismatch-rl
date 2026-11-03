# Why BI0 is slow with prefix caching on, and why BI1 must turn it off (2026-10-04, two GB200 nodes)

## Conclusions

- **The cause of BI0 being 11–16% slower is fully explained, and comes entirely from the Mamba "align" mode.** With prefix caching on, vLLM automatically sets the Mamba cache to align mode, which brings two overheads:
  1. **decode**: 28 extra eager kernel launches per step, on the host-side critical path, raising TPOT by 14–20%. This is the main cause at c≥8.
  2. **prefill**: an 8K prompt is cut at a block boundary (4176 + 4016) and runs in two steps. Each prefill step has about 100 ms of fixed overhead, so TTFT doubles at c=1. This item contributes about 5% of throughput at c=1, and is amortized as c grows.
- **It is not caused by block size or KV capacity.** With PC off, changing the block to 6768 or 8192 leaves throughput basically unchanged (median −5% to +5%, within noise). Max concurrency is ≥576x in every config, while only 16 requests per rank are actually needed; preemption is 0 in every config.
- **The reason BI1 must turn PC off has been localized to one line of code.** When align mode copies Mamba state across blocks, it only copies two states, conv and ssm. Exact replay appends 3 more partial-chunk input buffers (x/dt/B) after the states, and these 3 are not copied. After a prototype fix (copy funcs extended from 2 to 5), every hit and miss request is bitwise identical to the PC-off reference; in the control arm that only relaxes the check without adding the copies, every prompt longer than one block goes wrong.
- **BI1 with PC on costs 1–10% on random prompts.** Only workloads with long shared prefixes (multi-turn chat, agents) could possibly earn this back.

## 1. Environment and arms

- Image `<image>` (the p6 Nemotron package image, built from an internal vLLM commit), BI0 only.
- Protocol same as the formal p6 bench: 8K in / 1K out, c=1/8/32/64, a fresh server for every leg.
- 6 rounds in total: r1–r3 on one node, r4–r6 on another. Within a node the order is forward, reverse, forward; all arms in the same round use the same seed.

| Arm | Config | Effective attn block / mamba mode | KV tokens (max concurrency) |
|---|---|---|---|
| a1 | PC off (p6 frozen BI0) | 4176 / none | 14.88M (1615x) |
| a2 | PC off + `--block-size 6768` | 6768 / none | 10.71M (1162x) |
| a3 | PC on (p5 frozen bi0.args) | 4176 / align (mamba block 4176) | 9.47M (1027x) |
| a4 | PC on + `--block-size 8192` | 8192 / align | 5.31M (576x) |
| a5 | PC off + `--block-size 8192` | 8192 / none | 8.85M (960x) |

Correcting one premise in the task description: the frozen BI0 PC-on config does **not** have `--mamba-block-size 6768`; the effective block is 4176, the same as with PC off; 6768 is BI1's block size. Also, in align mode `--mamba-block-size` is unconditionally overridden to the attention block size by `interface.py:977`, so the "smallest legal mamba block" is 4176, i.e. a3. So the fourth arm was changed to a4 (block 8192): an 8192-token prompt fits exactly in one block and is not cut, which separates the prefill split from the decode overhead. This comes from reading the source; no separate server with `--mamba-block-size 6768` was started to verify it.

## 2. Results (median over 6 paired rounds)

Throughput differences relative to a1, with the min and max over 6 rounds in parentheses:

| c | a2 bs6768 | a5 bs8192 | **a3 PC on** | a4 PC on + bs8192 |
|---|---|---|---|---|
| 1 | +4.7% (−12…+15) | +1.5% (−4…+7) | **−14.9%** (−30…−3) | −8.9% (−26…+3) |
| 8 | −4.9% | −1.0% | **−13.5%** | −15.7% |
| 32 | −4.8% | −8.7% | **−16.2%** | −15.5% |
| 64 | −3.6% | −0.6% | **−13.4%** | −12.4% |

TTFT and TPOT separated, each the median over 6 paired rounds:

| Comparison | c=1 TTFT / TPOT | c=8 | c=32 | c=64 (TPOT) |
|---|---|---|---|---|
| a3 vs a1 (PC on vs off) | +80% / +9% | +39% / +13% | +26% / +18% | +17% |
| **a4 vs a5** (both block 8192, differing only in PC) | **+1% / +14%** | +3% / +16% | +8% / +20% | +14% |
| a3 vs a4 (differing only in whether prefill is cut) | **+85%** / +4% | +41% / −4% | +17% / −6% | −2% |

How to read these two tables:
- **Block size is not the cause**: the differences of a2 and a5 from a1 are within BI0's bimodal noise.
- **TPOT +14–20% comes from align mode itself, unrelated to whether prefill is cut**: a4 does not cut prefill, and its TPOT is still that much higher.
- **TTFT doubling comes from prefill being cut**: a3's TTFT is 85% higher than a4's, while a4 and a5 have the same TTFT.

## 3. Mechanism (torch profiler, one node, code at the same internal vLLM commit)

### decode (c=64 steady state, 4 ranks, about 180 steps each)

| | Step time | GPU idle/step | Out-of-graph eager launches/step | Real compute (excluding a2a wait) |
|---|---|---|---|---|
| PC off | 5318 us | 913 us | 10 | ≈3.8 ms |
| PC on | 5750 us | 1468 us | 38 | ≈3.7 ms |
| PC on + bs8192 | 6591 us | 1676 us | 38 | — |

- The extra GPU kernels are all tiny, about 80 us per step in total:
  - one `mamba_get_block_table_tensor` (`backends/utils.py:1163`) for each of the 4 mamba KV cache groups, each being floordiv, clamp, arange, add, to(int64), gather, copy, 7 torch ops, 28 launches in total;
  - the `preprocess_mamba_align_fused_kernel`, `precopy_mamba_align_fused_kernel` and `postprocess_mamba_fused_kernel` that V2 `mamba_hybrid.py` runs every step, plus one DtoD snapshot.
- GPU compute time does not change. What changes is GPU idle time, 0.55–0.75 ms more per step. This shows the PC-off decode loop is already partly host-bound (idle is already 0.9 ms), and the added eager ops and Python metadata work lengthen the step one for one.
- DP+EP's a2a requires the 4 ranks to synchronize every step, so the host overhead of the slowest rank sets the step time of all ranks.
- The code comment saying "launch is only 0.3% of TPOT" only counted 1 triton launch, not the 28 torch ops of the 4 groups.

### prefill (c=1, single request)

- PC off: 1 step `execute_context_1(8192)`, wall clock 121 ms, GPU kernels total 50 ms.
- PC on: 2 steps `(4176)` + `(4016)`, about 118 ms each, GPU kernels about 27 ms each.
- The split point comes from `_mamba_block_aligned_split` in `scheduler.py`: `last_cache_position = 8192 − 8192 % 4176 = 4176`, so the chunk must stop there.
- Each prefill step has about 70–90 ms of non-GPU fixed cost (prefill exceeds the cudagraph cap of 128 and takes the eager compile path, plus DP sync). Splitting into two steps pays it twice.

### Explained / unexplained

- Explained: TPOT +14–20% (eager metadata ops on the host critical path); c=1 TTFT ×1.8 (prefill cut in two); KV capacity and block size are irrelevant.
- Unexplained:
  - The actual CPU time of each eager op was not measured item by item (profiler had stacks off);
  - BI0's own bimodality (same config jumps between 305–354 at c=1 and 9055–11340 at c=64) is still unexplained.
- Validity:
  - In r3, a3 c=32 had 1 random prompt that happened to hit 4176 tokens (validated=false); no effect on the conclusions;
  - All other 119 points are valid, preemption 0, server_exit 0.

## 4. Why BI1 must turn PC off, and the prototype fix

### Source localization

- In BI mode, the Mamba2 state is 5 tensors: conv, ssm (fp32 chunk-boundary state), plus three partial-chunk buffers `x/dt/B`, each of shape `(chunk_size=128, …)` (`mamba2_state_shape` in `mamba_utils.py`). Exact replay replays every step from the nearest chunk boundary plus these buffers.
- Every time it crosses a block, align mode uses copy funcs to copy the previous column's state into the new column (`precopy_mamba_align_fused_kernel`, copying state by state per copy func). But `MambaStateCopyFuncCalculator.mamba2_state_copy_func` only returns two, `(conv, temporal)`. `validate_mamba_state_copy_funcs` allows the "copyable prefix" to be shorter than the number of states, so x/dt/B are **silently skipped**: the new column holds stale or uninitialized buffers.
- This is unrelated to whether chunks and blocks are aligned. In fact neither 4176 nor 6768 is a multiple of 128, but as long as the buffers are copied along, any position can be replayed exactly. So no constraint like "the block must be a multiple of chunk_size" is needed.
- Everything else is compatible at the source level:
  - attention: BI1 uses FLASH_ATTN with `num_splits=1`, and a prefix hit takes exactly the chunked prefill path (gate3 chunk2048 already verified BI); cascade is off by default.
  - exact-replay metadata is derived only from `seq_lens − query_lens`, independent of how the hit happened.

### Prototype

- On an uncommitted prototype branch in a dedicated worktree, based on the same internal vLLM commit. Two changes in total (a `fix.diff` in the run's results directory):
  - under BI, `mamba2_state_copy_func` returns `(conv, temporal×4)`;
  - `check_batch_invariant_config` allows `"align"`.
- Dev images: `<image>` naive and fix variants, equivalent to a docker commit of two files on top of p6. naive only relaxes the check. Not for delivery.

### Correctness test

- Method: BI1, PC on; all requests pinned to DP rank 0; temperature 0, 128 output tokens. The comparison is token ids plus the float64 bit pattern of each token's logprob.
- Reference: the outputs of the same requests on the original p6 image with BI1 and PC off; the reference's own replicas agree.

| Scenario (hit tokens) | naive | **fix** | fix + `--prefix-match-unit 16` |
|---|---|---|---|
| A 14100 cold (0) | Wrong (token 0) | **=** | = |
| A resent (13536 / 14096) | Wrong | **=** | = |
| B same prefix, different suffix (13536) | Wrong | **=** | = |
| Mixed batch of 6 requests (33840 / 35232) | A/B/C/D wrong, E/Q (<1 block) right | **all =** | all = |
| Q 1500 sibling samples ×1, ×1, ×4 (0 / 1488 once, 5952 one batch) | = | = | **= (fine-grained tail hit)** |
| C 9000 follow-up (6768 / 8992) | Wrong | = | = |

- naive also goes wrong on A cold start **with no hit at all**: the prompt is longer than one block, prefill is cut at the block boundary, and the buffers are lost on the cross-boundary copy. This confirms that the missing buffer copy is the only breaking point.
- 13536 % 128 = 96 and 1488 % 128 = 80 are both non-chunk-aligned hits, and they are also bitwise identical after the fix.
- Limitations:
  - Each scenario was run only once;
  - decode crossing a block boundary is not covered (needs more than 6768 output tokens);
  - PD disaggregation and multi-rank routing are not covered.
- Also, align mode only keeps each request's latest Mamba state, so shallower prefix hits are lost. For example D shares only the first 7000 tokens of A, and its hit is 0.

### Performance (fix image, BI1, 8K/1K random prompts, one node, 3 alternating rounds)

| c | BI1 PC off | BI1 PC on | Throughput difference (3 rounds) |
|---|---|---|---|
| 1 | 281 | 264 | −5.6, −5.9, −6.4% (TTFT +73%, TPOT +4%) |
| 8 | 1954 | 1843 | −4.4, −3.7, −8.9% |
| 32 | 6238 | 5451 | −25.2, −4.8, −10.2% |
| 64 | 8787 | 8508 | −3.5, −0.6, −1.4% (2 points had accidental hits, invalid) |

- BI1's align cost is smaller than BI0's: BI1's per-step GPU time is already longer, so part of the extra host overhead is hidden.

### Can PC win back BI1's performance

- **Not on the current bench.** Random prompts have no hits, so PC on only pays the 1–10% cost.
- **Small gain for RL rollout (GRPO, n samples sharing a short prompt).** The default hit granularity is 6768 tokens, while DAPO prompts are only a few hundred tokens, so hits are 0. With `--prefix-match-unit 16` each sibling sample saves the prompt's prefill; but rollout time is mostly spent on decoding long outputs, and decode TPOT pays another 4–15%. Net gain is expected to be negative or about 0. This was not measured.
- **Scenarios that might earn it back**: multi-turn chat or agent workloads where shared prefixes exceed several thousand tokens and are reused repeatedly. The prerequisite is first cutting align mode's per-step host overhead, e.g. merging the 4 groups' `mamba_get_block_table_tensor` into one kernel, or putting it into the cudagraph.

## Files

- Local run results directory for this ablation:
  - Scripts: leg.sh, ablation.sbatch, prof-inner.sh, summarize.py, analyze_trace.py
  - Configs: configs/
  - Summary: summary.txt
  - BI1 prototype: bi1-pc-proto/ (diff, client, compare.txt, per-leg result.json)
- Shared storage: server.log and metrics for all legs. The profiler traces are under `prof-r1-*/trace{A,B}`, and the BI1 performance legs under `b1-*`. The prototype runs are in a separate runs directory.
- Jobs: two ablation jobs, one profile job, one BI1 prototype job, one BI1 PC performance job. Only the two assigned nodes were used; all finished normally.
