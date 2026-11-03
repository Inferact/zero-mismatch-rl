# Nemotron-H NVFP4: prefix caching on vs off (2026-10-05/06, four GB200 nodes)

Commit labels used below (all on unpushed internal fix branches): **base** = the base commit; **C1** (dec), **C2** (#57329 merge), **C2t** (its unit test), **C3** (p29b), **C4** (p29c perf commit); **R1** and **R2** = the review-r8 fix and test-fix commits. The final p29c head is **R2**.

## Conclusions

- **On the final image, PC on (align mode) is slower than PC off at every measurement point.**
  - Throughput: BI0 4–20% slower, BI1 0–8% slower.
  - TPOT: BI0 up 13–21%, BI1 up 3–7%.
  - TTFT at c=1: roughly doubles for both.
  - With PC on, vLLM only uses align mode (`MambaCacheMode` has only none and align).
- **Final candidate p29c** (an unpushed internal fix branch): C4 plus the review-r8 fixes R1 and R2; the final head is **R2**. Performance numbers were all measured on C4; when spec decode is off the r8 fixes do not change any write, so these numbers apply to the final head (see item 5 of section 2). After 12 paired rounds:
  - **BI1 basically catches up**: medians for c=8/32/64, decode-heavy and RL-like are between −0.4% and +4.9%, TPOT within ±1%. Only c=1 still has −2%, with 9 of 12 rounds landing between −2% and 0.
  - **BI0 is within noise**: medians c=1 −2%, c=8 −2%, c=32 −4.5%, c=64 +3.7%, decode-heavy −2.9%. BI0 itself is bimodal, with ±10–25% between rounds of the same config, so these differences cannot be resolved; only decode-heavy TPOT +2% has a consistent direction across rounds.
- **Correctness: every candidate passed all gates; no silent errors found.**
  - GSM8K: BI1 PC off and PC on are both 426/500, every round; after the r8 fix BI1 PC on is still 426, and all 500 questions are identical item by item to pre-fix p29c and the PC-off final. BI0 is within noise of PC off.
  - BI1 bipc bitwise probe: 1064/1064 identical; one more run after the r8 fix (R1), total 1140/1140.
  - Budget-cut checkpoint probe: 368/368 identical.
  - PCPERF_CHECK: bad count is 0 on all workers, also 0 after the r8 fix.
  - gate3: PD 4/4, BI and cross-run all pass; the rollouts sha256 is still `06daedf0…`.

## 1. Results

8K/1K measurement points are c=1/8/32/64. Decode-heavy is 1K in / 4K out, c=64. RL-like is 64 DAPO prompts × 8 consecutive samples, output 2048, all sent at once. GSM8K is 40-shot, first 500 questions; every question hits 1 block (BI0 4176 tokens, BI1 6768 tokens).

Values are the throughput change of "PC on relative to PC off in the same round", taken as the median of the per-round paired ratios; TPOT change in parentheses.

| Arm | Rounds | BI | c=1 | c=8 | c=32 | c=64 | decode-heavy | RL-like | GSM8K (PC on) |
|---|---|---|---|---|---|---|---|---|---|
| PC on as-is (final / bimpc) | 7 / 6 | BI0 | −20% (+21%) | −8% (+18%) | −10% (+14%) | −4% (+13%) | −5% (+7%) | −6% (+3%) | 429–435 |
| | | BI1 | −5.5% (+3%) | −5% (+4%) | −7% (+3%) | −8% (+7%) | 0% (+5%) | −5% (+3%) | 426 ×6 |
| dec | 12 | BI0 | −3% (+1%) | −6% (+5%) | −3% (+1%) | +4% (−1%) | −2% (+1%) | −5% (−1%) | 428–436 |
| | | BI1 | −4% (+1%) | −4% (0%) | −4% (+1%) | −2% (+2%) | +1% (+1%) | 0% (0%) | 426 ×12 |
| p29 (dec + #57329 + BI export) | 12 | BI0 | −4% (+4%) | −2% (+2%) | −5% (+4%) | +2% (+6%) | −3% (+1%) | −1% (0%) | 426–439 |
| | | BI1 | −1.5% (+1%) | **−9% (+3%)** | **−8% (+8%)** | +1% (0%) | 0% (+2%) | −1% (−1%) | 426 ×12 |
| **p29c (final)** | 12 | BI0 | −2% (+1%) | −2% (+2%) | −4.5% (+3%) | **+3.7%** (−1%) | −2.9% (+2%) | −2.5% (+1%) | 421–441 |
| | | **BI1** | −2.0% (+1.3%) | **+0.1%** (+0.1%) | **+1.1%** (−0.1%) | **+4.9%** (+1%) | **+0.7%** | −0.4% | **426 ×12** |

- Absolute PC-off throughput (median of the 12 p29c rounds, tok/s):
  - BI0: 348 / 2252 / 6963 / 9825, decode-heavy 14709, RL 35000;
  - BI1: 286 / 1960 / 5949 / 9037, decode-heavy 12122, RL 27898.
- c=1 TTFT relative to PC off: as-is +95%, dec +80–89%, p29c BI0 +0.6%, BI1 +3%.
- RL-like sends 512 requests at once; when DP load is unbalanced there is an extra round, and the duration jumps between two levels, 24 s and 30 s (bimodal). So RL throughput is only a reference; TPOT is what is stable. RL prompts are only about 150 tokens, shorter than a block, so PC on never hits; this item measures pure overhead.
- Per-round values are in the per-arm summary files (base, cand, c2) in the run's results directory.

## 2. Which changes are needed (p29c = base + the 4 commits below + the r8 fix, head R2)

1. **C1: remove redundant launches on the decode side; Python only, no new kernel.**
   - (a) The Mamba metadata builder switches to the multi-group aligned index precomputation that MRV2 already has. This is a minimal port of #58851 that keeps the existing cross-group reuse in `update_block_table`, so the #58851 × #57329 silent error does not occur, and BI's per-group refresh is not affected.
   - (b) Without spec decode, skip the align postprocess: it has no effect in that case, and skipping it saves 1 copy and 1 kernel per step.
   - (c) When the host side determines that in this step no request crosses a block and there is no new prompt, skip the preprocess and precopy kernels.
   - Effect: eager launches per step drop from 37 to 9 (PC off is 9), Triton launches from 11 to 9 (PC off is 8). The TPOT penalty basically disappears.
2. **C2: merge #57329 (@dac3aa469a), fixing prefill being split.**
   - An 8192-token prompt is back to one step, and c=1 TTFT returns to parity with PC off.
   - **The BI path implements checkpoint export separately**: without changing the chunk grid, write the checkpoint slot from the state at the nearest chunk boundary plus the x/dt/B buffers. Without this step, #57329 makes BI1 silently wrong, because it only exports checkpoints on the non-BI path.
   - C2t is the corresponding unit test: bitwise comparison against the "split and run in one step" result; all 4 scenarios pass.
3. **C3: resolve the exact-replay write slots only once per KV cache group.**
   - Previously every layer re-gathered; checkpoint writes were also merged into the existing boundary/store writes.
   - The goal is to cut the extra 12–17 ms of host time in BI1 prefill steps.
4. **C4: a prefill chunk truncated by the token budget may also export that prompt's last checkpoint.** This is the key to fixing the BI1 c=8/32 regression.
   - **Root cause**: under DP+EP, while any rank is running prefill, the other ranks wait for it at the a2a. So the number of prefill steps directly becomes global decode stalls.
   - When the budget cannot fit a second prompt, the arms behave differently:
     - PC off: the second prompt runs 8191 tokens first, and the remaining 1 token runs as decode in the next step, costing almost nothing;
     - Original p29: the second prompt is cut at a block boundary, and the remaining 1424 tokens need a separate prefill step, about 100 ms per such step.
   - **Evidence**: at c=64 the number of prefill steps is 46–48 vs 38–42 for PC off (step trace).
   - **Change**: allow it only when "before the end of the truncated chunk, the last reusable boundary is the same as the last boundary of the whole prompt". This keeps the worker, scheduler and MambaManager in agreement on the checkpoint position; in all other cases it still cuts at block boundaries.
   - **Effect**: BI1 c=8 goes from −9% to +0.1%, c=32 from −8% to +1.1%; BI0 c=64 TPOT goes from +6% to −1%.
5. **R1 + R2: review-r8 fix and extra tests.**
   - **conv window alignment (R1)**: `store_prefill_checkpoint` and `store_exact_replay_checkpoint_conv` used to write conv history tail-aligned to `conv_state.size(-1)`, but the conv kernel reads from `[0, width-1)`. With spec decode on, the cache is `width-1+num_spec` wide, so after a prefix-hit restore the window read is shifted by num_spec tokens. Now `state_len = conv_kernel_size - 1` is passed and only `[0, state_len)` is written.
   - Without spec decode the cache is exactly `width-1` wide and the writes are exactly the same as before the fix, so the BI probes did not catch it earlier, and the bench and probe conclusions on C4 apply directly.
   - At the same time, the host-side skip check was factored out into `align_running_block_moves`, and an outdated comment about where the exact-replay slots are resolved was corrected.
   - New tests: exact-replay batched checkpoint (multiple rows, mixed n_pre, builder-resolved slots, export into a widened conv cache); store kernel on a widened conv cache (parameterized by `num_spec`); MRV2 host skip check against `preprocess_mamba_align_fused_kernel`; scheduler split of budget-cut chunks; MambaManager reserved position against the worker rule.
   - R2 only changes tests: R1's manager test failed under dense retention (`retention_interval=None`) (job A), because dense retention also reserves a checkpoint for the previous chunk; changed to assert per retention mode, then it passed (job B).

## 3. Correctness gates

| Check | Scope | Result |
|---|---|---|
| GSM8K (40-shot, 500 questions) | Run on every bench leg | BI1: PC off, dec, p29, p29b, p29c all 426 (bitwise deterministic under BI); BI0: every arm within PC off's 426–439 range |
| BI1 bipc bitwise probe | 11 scenarios × DP rank 0–3, hit positions 13536/6768/33840 (not on chunk boundaries); dec, p29, p29b, p29c 2 runs each, PC-off reference 6 runs in total | **1064/1064** bitwise identical to the PC-off golden |
| Budget-cut checkpoint probe | p29c; two 9000-token prompts arrive together, one of them runs 7384 tokens first; afterwards the first 7000 tokens of both each hit 6768 (K_reuse total 13536) | **368/368** bitwise identical; the trace confirms `in=7384 out=7384`, showing the truncation path was actually taken |
| PCPERF_CHECK hook | Per-step check of state_idx on the GPU against the host-side prediction | All bipc PC-on legs and check bench legs (BI0/BI1): pre_bad=0, post_bad=0 |
| gate3 | p29 (one node), p29c (another node) | PD 4/4, BI 6/6, cross-run 4/4, rollouts.json sha256 still `06daedf0…` |
| Unit tests (in image, C4) | | BI end-to-end PC on/off comparison 4/4 (V1/V2); exact replay 7/7 (including the 4 new checkpoint bitwise cases); nemotron_h_bi 7/7; attn_bi 10/10; ckpt_state 6/6; mamba_utils 43/43; precopy_align 75/75; scheduler mamba 2/2. The PR's own `test_mamba2_prefill_checkpoints` has 3 failing cases because it pulls a Llama config from HF and the cluster is offline |
| Unit tests (after the r8 fix, p29c image with changed files mounted, 0 mismatch) | job A (R1, one node); job B (R2, only the two changed files rerun) | ckpt_state 8/8; exact replay 9/9 (once with BI unset and once with BI=1); precopy_align 76/76; hybrid_state 6/6; chunk_split 50/50; partial_hits 119/119 (job B; in job A `test_budget_cut_chunk_reserves_the_worker_checkpoint[None]` failed, fixed by R2); eagle_resume 13/13; prefix_caching -k mamba 19/19; scheduler -k mamba 2/2; metadata_builder 1/1; mamba2_bi_cfg 10/10; mamba2 BI e2e 4/4. `test_mamba2_prefill_checkpoints` still has 3 cases failing due to HF being offline (Meta-Llama-3-8B) |
| Gates after the r8 fix | job C (one node, p29c image with R1's 5 changed files mounted) | One new bipc PC-on leg (g4-p29cr8-pcon1): 76/76, total **1140/1140** bitwise identical; PCPERF_CHECK pre_bad=0, post_bad=0 (bipc leg and GSM8K leg); GSM8K BI1 PC on **426**, all 500 questions identical item by item (output, tokens, finish reason) to pre-fix p29c and the PC-off final. gate3 was not rerun |

## 4. Remaining gap

- **BI1 c=1: −2%** (TPOT +1.3%, TTFT +3%); **BI0 decode-heavy: TPOT +2%**.
  - Speculation (not measured separately): it comes from the 1 extra out-of-graph Triton launch per decode step, i.e. the aligned index precomputation (PC off has 8, p29c has 9), plus some host-side scheduler overhead with PC on.
  - Removing it needs a CPU-side mirror of the block table, or fusing the index computation into the existing gather kernel. Under the "no new Triton kernel" constraint, this was not done this time.
- **BI0 medians** (c=1/8/32 −2/−2/−4.5%) fall within BI0's bimodal noise: ±10–25% between rounds of the same arm. 12 rounds cannot resolve a ±3% difference.
- **Not covered**:
  - True PD disaggregation: KV connectors are rejected under BI, so it cannot be measured;
  - Spec decode: D2/D3 only take effect without spec decode. The r8 conv window fix changes the checkpoint conv layout with spec decode on (a correctness fix, covered by unit tests), but no end-to-end run or bench with spec decode on was done;
  - V1 runner throughput: only the V1 end-to-end in the unit tests was run.
- **Known item (P2 from the p29c review, not fixed)**: the worker-side `compute_mamba_prefill_checkpoints` has no prefill_end gate and also computes a checkpoint column for intermediate chunks; under sparse retention (`retention_interval=0`) MambaManager does not reserve for that intermediate chunk, and the column is left as the null block. `test_budget_cut_chunk_reserves_the_worker_checkpoint` now encodes this inconsistency as an assertion (manager side `reserved is None` and the column `is_null`, while the worker still points at that column).
  - Why it is safe: the worker's write lands in the null block. The null block is not hashed and is never read back as a prefix hit; it is just a discard slot, so it cannot pollute other requests' state; the cost is one extra write/launch.
  - Risk: correctness depends on the convention that "the null block is a discard slot". If future code reads the null block or asserts on it, this will break. The thorough fix is to pass prefill_end into `compute_mamba_prefill_checkpoints` to align with the manager's decision.
- **For the RL-like scenario to benefit from PC**, short prompts must also hit, e.g. `--prefix-match-unit 16`, and sibling samples of the same group must land on the same DP rank. This was not tested.

## 5. Artifacts

- **Branches** (in a dedicated fix worktree, none pushed):
  - pc-perf-base: base
  - pc-perf-dec: C1
  - pc-perf-p29: C2t
  - pc-perf-p29b: C3
  - pc-perf-p29c: R2 (C4 + r8 fixes R1, R2)
- **Dev images**: `<image>` variants bimpc, dec, p29, p29b and p29c (Nemotron dev images for this study).
  - All are enroot commits on top of the final image, installing only the changed `.py` files.
  - Each passed a pre-image check, with `content_mismatch 0` after installation.
- **Run results directory (local)**:
  - Summaries: summary-{base,cand,c2}.txt;
  - Gates: bipc-compare.txt, cutprobe-compare.txt, pcperf-check.txt;
  - Other: prof-summary.txt, unit-p29-summary.txt, and scripts/ (including the test-only hook).
- **Shared storage**: server.log, result.json and traces for all legs.
- **Jobs** (all only on the four assigned nodes):
  - Baseline: 4 jobs;
  - Candidates: 6 jobs;
  - p29c: 4 jobs;
  - r8 fix: unit-test jobs A and B (one node), gate job C (another node, bipc + GSM8K);
  - Gates/dev: 17 jobs;
  - Only cancelled one of its own jobs (a build, done instead inside a dev job).
- A later message said three additional nodes could also be used, but that conflicted with the original exclusion list, so they were not used.
