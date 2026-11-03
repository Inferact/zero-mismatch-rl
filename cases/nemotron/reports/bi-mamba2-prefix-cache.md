# Mamba2 prefix caching under BI=1: formal fix and validation (2026-10-04 on node A; second round 2026-10-05 on nodes A/B/C)

Commit SHAs of the internal vLLM fork are omitted; container images are described rather than named.

## Second-round conclusions (2026-10-05, filling the gaps from round 1 §6; details in §9)

- **The patch applies cleanly on top of the current delivery head.** A scratch branch = delivery head + cherry-pick of the fix commit, no conflicts; the vllm/ and tests/ changes are line-for-line identical to the original commit. It lives in its own worktree. The delivery branch was not touched and nothing was pushed.
- **New dev image** `<image>` (dev image with the fix on the delivery head; sha256 `e3f50415…0f6aff`), based on the current candidate image (`253dc2d3…`). pre-image 2/2 consistent; after install, 2588 `vllm/**/*.py` files have 0 inconsistencies with the git export. All in-image unit tests pass; the negative control still fails.
- **Real model, 11 probes × 3 runs × DP ranks 0–3**: 684 requests (456 with PC on, 228 with PC off), all bitwise identical to PC off.
  - The 12 copies with PC off (3 runs × 4 ranks) are identical to each other and to the output of the round-1 image.
  - Every step checked that "the pinned engine's prompt-token increment == this step's prompt tokens"; 0 routing errors.
  - Hit counts are exactly the same across 3 runs × 4 ranks, and the same as round 1.
- **Forced preemption** (32 blocks and 16 seqs per engine; each rank sends 8 requests of 6000/12000 tokens concurrently, 2000 output tokens each):
  - PC on: 174 preemptions over 3 runs; PC off: 16 over 2 runs. All 360 requests are bitwise identical to the output of PC-off solo sends (no batching, no preemption).
  - Hits of preempted requests do not go into `/metrics`, so a test-only logging hook recorded the lookups during recompute: **49 recomputes hit 6768 tokens** (6768 mod 128 = 112, not on a chunk boundary), 66 missed (block already evicted).
  - The recomputes that hit include boundaries inside the prompt as well as boundaries crossed during decode (an S request was preempted at 7534/7536 tokens and hit 6768).
- **Real PD disaggregation cannot be tested**: under BI, `Mamba2AttentionBackend.check_batch_invariant_config` rejects `kv_transfer_config` outright ("does not support KV connectors"); the combination BI + PC + KV connector does not exist.
  - The delivery gate3 "PD" is a consistency check of decode logprobs vs teacher-forced prefill. Rerun on the new image with PC on: PD 4/4, BI 6/6, cross-run 4/4 all pass; the rollouts.json sha256 is still `06daedf0…`, the same as round 1 and PC off.
- **Measured copy cost** (torch profiler, c=64 shared-prefix load, 400 engine iterations × 4 ranks per leg; 5-state copy vs 2-state copy (negative-control hook), 3 runs each):
  - **The extra cost is mostly per step, not in the few steps that cross a block.** The fused pre-copy kernel's grid is `(num_reqs, num_states, 16)`; the number of states goes from 46 (23 layers × 2) to 115 (23 layers × 5), and steps that do not cross a block still launch these extra programs, which then exit early. One fast-exit goes from 7.8 µs to 17.1 µs, i.e. **about 9.4 µs more per step**, about 0.18% of the 5.3 ms TPOT.
  - Steps that actually copy (only 2–4 per rank per 400 steps at c=64) go from about 58 µs to about 105 µs.
  - Pre-copy's share of GPU kernel time goes from 0.10% to 0.20% (median over two nodes).
  - The round-1 report's statement "only happens when crossing a block, about 10 µs" was wrong and has been corrected in §3. The block-aligned scheme (block % chunk == 0) cannot save this cost either, because the grid does not change.

## First-round conclusions (2026-10-04)

- **The fix is 2 source changes**, on a fix branch based on the delivery branch, in its own worktree. Nothing was pushed and the delivery branch was not changed.
  1. `MambaStateCopyFuncCalculator.mamba2_state_copy_func`: under BI returns `(conv, temporal×4)`, covering all 5 states declared by BI. BI=0 still returns 2; behavior unchanged.
  2. `Mamba2AttentionBackend.check_batch_invariant_config`: remove the check that rejects prefix caching. `MambaCacheMode` has only two values, `none` and `align`, and upstream main is the same, so removing the check is equivalent to enabling only the already-validated `align`.
- **Correctness**: hit and miss are bitwise identical to PC off, covering a small model and the real model.
  - Small-model e2e (mamba2-130m, block 400, chunk 256): both the V1 runner and the V2 runner pass.
  - Negative control: the same test with only the copy reverted to 2 states; every prompt longer than 1 block is wrong, prompts shorter than 1 block are unaffected.
  - Lightning NVFP4 (BI1, delivery serve args): all 11 scenarios bitwise identical to PC off, and fine-grained hits with `--prefix-match-unit 16` are identical as well. None of the hit positions lie on chunk boundaries: 13536, 6768, 1488, 6688 mod 128 are 96, 112, 80, 32.
  - gate3 with PC on: PD 4/4, BI 6/6, cross-run 4/4 all pass; rollouts.json sha256 is the same as p6/cb4 (`06daedf0…`).
- **Performance** (BI1, PC on vs PC off, 3 pairs):
  - Random 8K/1K: PC on costs 6–18%; medians at c=1/8/32/64 are −9.2%, −17.6%, −10.9%, −9.7%. The cost comes from align mode itself (at c=1 TTFT roughly doubles, TPOT increases 2–20%), not from the newly added copy: the copy only happens when crossing a block, once every 6768 tokens, copying 29 MiB more each time.
  - Shared prefix (90% of requests share a 7168-token prefix, each hit is 6768 tokens): c=1/8/32/64 give −9.7%, −5.5%, **+7.8%, +9.8%**; at c≥32 TTFT drops 30–50%. Only high-concurrency, long-prefix-reuse scenarios recoup the cost. See §5.3.
- **Not recommended to change to "block must be a multiple of chunk"**. Three reasons:
  - It does not eliminate copying. `--prefix-match-unit` tail hits (1488, 6688) and mid-decode states naturally land in the middle of a chunk.
  - The block size would have to change from 6768 to 6784, which requires new configuration logic.
  - The cost of the copy buffer itself is negligible.

## 1. Change and design

```
vllm/model_executor/layers/mamba/mamba_utils.py   +4   copy 5 states under BI
vllm/v1/attention/backends/mamba2_attn.py         -5   remove PC rejection
tests/v1/attention/test_mamba2_batch_invariant.py      number of copy funcs == number of BI states; align accepted
tests/v1/determinism/test_mamba2_batch_invariance.py   e2e: hit/miss vs PC off, V1/V2
.buildkite/test_areas/misc.yaml                   +1   H100 BI job runs this file (required by the test-tethering hook)
```

**Why the copy count is not derived automatically from "the length of the state shape list"**

- The convention of `validate_mamba_state_copy_funcs` is "the copy functions are a copyable prefix of the states", i.e. there may be fewer copy functions than states. This is intentional: ReplaySSM's ring buffer (`append_replayssm_ring`) is also a state declared by Mamba2, but align mode forces a flush at block boundaries and restarts from an empty ring in the new block (`effective_base` in `mamba_attn.py`), so the ring should not be copied.
- Changing to "copy every declared state" would change non-BI ReplaySSM behavior and also require changing the generic validator.
- The current code puts the BI branch under the same `envs.VLLM_BATCH_INVARIANT` condition as the shape (`mamba2_state_shape`) and dtype (`mamba2_state_dtype`). A unit test was also added asserting `len(copy_funcs) == len(mamba2_state_shape(...))` (BI) and `== 2` (BI=0). If BI states are added or removed later, this test fails directly.
- Every model that uses `MambaMixer2` (nemotron_h, mamba2, zamba2, falcon_h1, granitemoehybrid) gets its copy functions through this calculator, so one change covers them all. Other layer types (GDN, KDA, short conv, mamba1) are unaffected.

**The patch applies cleanly on top of upstream #55905**: `git apply --check` passes on #55905 head `7b7dcafd05`. Only the buildkite yaml is offset by 80 lines and mamba_utils by 1 line.

## 2. Places where align mode touches state (checked one by one)

| Path | What does the copy | Conclusion |
|---|---|---|
| V2 `MambaHybridModelState.preprocess_state` → `preprocess_mamba_align_fused_kernel` + `run_fused_precopy` | `MambaSpecDecodeGPUContext`: each layer registers states by the number of copy functions; `temporal` copies the whole slot | Automatically becomes 5; this is what the delivery path (V2) uses |
| V2 `postprocess_state` → `run_fused_postprocess_align` | Same as above | Only active under spec decode; spec decode is rejected under BI |
| V1 `mamba_utils.preprocess_mamba/postprocess_mamba` (`MambaCopyBuffers`, `zip(kv_caches, copy_funcs)`) | The same set of copy functions | Automatically becomes 5; covered by the e2e V1 parameter |
| `MambaPrefillCheckpointBuilder` (checkpoints in the middle of prefill) | Only GDN/KDA (`num_prefill_checkpoint_blocks>0`); Mamba2 is 0 | Not involved |
| `--prefix-match-unit` tail hit (`_cache_partial_tail_block`) | After a hit, copying into its own block goes through the same precopy | Covered; real-model hits at 1488 and 6688 are bitwise identical |
| State writes | Exact replay writes the tail input shorter than one chunk into the same slot's buffer at every step; the state index is the same as conv/ssm | No change needed |
| Preemption/recompute | After freeing blocks, re-prefill from 0 or from the hit position; `add_request` resets state_idx by num_computed | Still goes through the hit path; round 2 deliberately induced preemption: 49 recomputes hit 6768, all bitwise identical (§9.2) |
| KV connector / offload of boundary states | BI already rejects `kv_transfer_config` | Not involved |

## 3. Comparison with the block-aligned-to-chunk scheme

| | Copy buffer (this fix) | Require block % chunk == 0 |
|---|---|---|
| Code | 2 places | Must change the hybrid block size derivation and validate the user's `--block-size`; 6768 → 6784 |
| `--prefix-match-unit` tail hits | Correct (1488, 6688 verified) | The tail still lands in the middle of a chunk, **the copy buffer is still needed** |
| decode crossing a block | Copy | Buffer is empty at the boundary, copy can be skipped |
| Cost | Per block crossing, each request copies an extra 23 layers × (x 1 MiB + dt 16 KiB + B 256 KiB) ≈ 29 MiB (previously 47 MiB). **Round-2 measurement (§9.4)**: copy steps go from about 58 µs to about 105 µs; in addition every step's pre-copy launch costs about 9.4 µs more (the grid expands by number of states, and fast-exit pays it too), together about 0.2% of TPOT | Saves these 29 MiB, but attention pages get larger and mamba page padding increases; the per-step launch grid does not change, so the 9.4 µs cannot be saved |

Conclusion: the aligned scheme is neither complete nor cheaper, so it is not adopted.

## 4. Tests (run inside the image)

Image: `<image>` (dev image with the fix on the previous delivery head), sha256 `014fad99…eecb5e7`.

- Build method: in the previous candidate image (`a1fca9e6…`), install these 2 `.py` files with enroot, equivalent to docker commit. No overlay and no source mount.
- Two checks at build time:
  - pre-image: these 2 files were identical to the delivery-branch version before the change.
  - after install: all 2588 `vllm/**/*.py` files are byte-identical to the git-exported source, 0 inconsistencies.
- The test code is taken from the fix commit, whose vllm/ is identical to the image's source commit; the two commits differ only in test files.
- The small model is loaded read-only from the HF cache on Lustre, with `HF_HUB_OFFLINE=1`.

| Test | Result |
|---|---|
| `tests/v1/attention/test_mamba2_batch_invariant.py` (including 3 new cases) | 10 pass |
| `tests/v1/determinism/test_mamba2_batch_invariance.py` (2 existing + new V1/V2) | **4 pass** (316 s) |
| Negative control: the same new test, with an import hook reverting the copy to 2 states in all processes (`unit/negctl/sitecustomize.py`) | **2 fail, as expected**: prompt indices 1,2,3,4,6,7,… inconsistent (400/700/1000/350+decode/…); the two 300-token ones agree |
| Regression: nemotron_h_bi 7, mamba2_exact_replay 3, mamba2_ssd_bi 7, test_mamba_utils 43, test_precopy_mamba_align 75, test_memcpy_u64_tiled 120 | all pass |

About the e2e test:

- The first version asserted two more hits, neither of which happened:
  - "a 400-token prompt plus 100 tokens", expecting a hit of 400;
  - "the second turn after decode crosses a boundary", expecting a hit of 400.
- The reason is align mode's retention policy (`MambaManager.reachable_block_mask`): a request only retains the last boundary state at ≤ prompt_len−1; boundary states produced by decode and boundary states that fall exactly at the end of the prompt are not retained. This policy is unrelated to BI; there is no BI branch anywhere in `vllm/v1/core`. Upstream #50551 is changing this.
- The final version asserts only the three stable hits: 700→400, 650→400, 1000→800. Other scenarios only compare outputs, as miss coverage, including decode crossing a block.

## 5. Real model (Lightning NVFP4, BI1, DP4/EP, delivery serve args, max-model-len 16048, attention block 6768)

### 5.1 Correctness probes (`bipc/`, adding two-turn conversations M1/M2 on top of the prototype client)

All requests are pinned to DP rank 0, temperature 0, 128 output tokens. Compare token ids and the float64 bit pattern of each token's logprob. The reference arm is PC off on the same image.

| Scenario (hit tokens) | PC on | PC on + `--prefix-match-unit 16` |
|---|---|---|
| A 14100 cold (0) | = | = |
| A resend (13536 / 14096) | = | = |
| B same prefix, different suffix (13536) | = | = |
| D shared 7000 (0, retention policy) | = | = |
| Mixed batch of 6 requests (33840 / 35232) | all = | all = |
| Q 1500 sibling samples ×1, ×1, ×4 (0 / 1488, 1488, 5952) | = | = |
| C 9000 follow-up (6768 / 8992) | = | = |
| M1 6700 + 128 output (decode crosses 6768) (0) | = | = |
| M2 = M1 + first 100 output tokens + 300 new tokens (0 / 6688) | = | = |

The reference arm is self-consistent (multiple copies of each prompt are identical). See `bipc/compare.txt`.

### 5.2 gate3 (`gate3/run-g3.sbatch`, stage and proxy scripts the same as ds4form-inf)

| leg | Result |
|---|---|
| serve (PC off) | pass; PD 4/4, BI 6/6; cross-image identical with cb4/p6 |
| prefixcache (**PC on**, align, block 6768) | **pass**; PD 4/4 (14000/14000/1024/512 tokens, mismatch 0), BI 6/6, cross-run 4/4 vs serve; rollouts.json sha256 `06daedf0…` same as p6 |

The gate3 server did not print hit statistics, so the number of hits was not confirmed for this item; the hit path is proven by 5.1.

### 5.3 Throughput (`bench/`, vllm-rs binary the same as the delivery bench, 3 pairs, order AB BA AB, fresh server per leg)

**Random 8K/1K** (same load as the delivery bench):

| c | PC off → on (p1 / p2 / p3) | Median Δ | TTFT median (off→on) | TPOT median (off→on) |
|---|---|---|---|---|
| 1 | 282→256 / 283→256 / 283→264 | **−9.2%** | 115–126 → 243–266 ms | 3.42 → 3.48–3.66 ms |
| 8 | 1850→1747 / 2098→1697 / 1981→1633 | **−17.6%** | 163–287 → 296–385 ms | 3.62–3.87 → 4.12–4.48 ms |
| 32 | 6190→5439 / 6219→5540 / 5361→5077 | **−10.9%** | 485–582 → 608–679 ms | 4.5 → 5.0–5.3 ms |
| 64 | 9437→8519 / 8397→8218 / 9048→7919* | **−9.7%** | 558–840 → 710–936 ms | 5.5–6.1 → 5.95–6.75 ms |

\* In p3 c=64, PC on had one accidental hit (6768 / 2.1M tokens), and validation marked it invalid. Dropping this pair, the median is −5.9%.

- All 6 legs had 0 preemptions and 0 hits (except that p3 c64 one).
- This is the same order of magnitude as the prototype report's "1–10%"; c=8 is larger, with large same-arm variance (PC off alone ranges 1850–2098 at c=8).

**Shared prefix 8K/1K** (one job):

- The load uses the bimodal mode of the vllm-rs random dataset: `--random-cache-hit-fraction 0.9 --random-cache-ratio 0.875`, token-id input.
- 90% of requests start with the same 7168-token prefix followed by 1024 unique tokens; the other 10% are fully random. Each point uses its own seed, so the prefixes also differ.
- One hit is at most 6768 tokens (1 block); the theoretical hit-rate ceiling is about 74%.

| c | PC off → on (p1 / p2 / p3) | Median Δ | Hit rate (tokens) | TTFT median (off→on) |
|---|---|---|---|---|
| 1 | 279→272 / 281→253 / 279→252 | **−9.7%** | 21–41% | 117–132 → 173–258 ms |
| 8 | 1984→2044 / 1967→1859 / 2003→1752 | **−5.5%** | 65–70% | 172–259 → 143–269 ms |
| 32 | 6203→6922 / 6085→6560 / 6616→6287 | **+7.8%** | 72–77% | 528–608 → 297–346 ms |
| 64 | 10029→10685 / 10085→11138 / 9788→10746 | **+9.8%** | 72–74% | 649–774 → 464–516 ms |

- All 18 points valid, 0 preemptions.
- Conclusion: PC only pays off when the prefix is long (≥ 1 block, 6768 tokens) and heavily reused.
  - At c≥32, the prefill saved by hits outweighs the align overhead: throughput improves 8–10% and TTFT drops 30–50%.
  - At c≤8, the align overhead still dominates and the result is negative or close to 0.
- By inference, GRPO-style prompts of a few hundred tokens do not reach 1 block and will not hit at the default granularity, so enabling PC for RL rollout is not recommended. This agrees with the prototype report's judgment; rollout was not measured this time.

The first version of the shared-prefix load used the `prefix_repetition` dataset and was discarded: it is a text dataset and gets re-tokenized, so prompts occasionally became 8193 tokens and exceeded max-model-len; at c=64 tokenization also failed to converge. It was rerun with the bimodal mode of the random dataset, using token ids with exact lengths. The discarded points are kept in `pc-p*/sp-*` and not used in the summary.

## 6. Limitations

The first three items listed in round 1 have been filled in round 2 (§9): probes 3 runs × ranks 0–3, forced preemption with recovery from hits, profiler measurement. Real PD disaggregation is rejected under BI and cannot be tested. Remaining limitations:

- Throughput has only 3 pairs, with large variance at c=8/64. Round 2 did not rerun throughput on the new image: the two changed files are byte-identical to the round-1 image, and the new changes on the delivery head are in MoE/quantization checks, not on the align path.
- The extra 9.4 µs per step could still be reduced, e.g. by having the kernel first check `src_col` per request so that requests not crossing a block are not expanded into the state dimension. That requires changing the upstream align kernel and is outside this fix.
- In preemption scenarios, "recompute hits" are only visible through the test-only logging hook, because vLLM does not export hit counts of preempted requests (`PrefixCacheStats.preempted_*` is not in Prometheus).

## 7. Duplicate-work check (run once at about 15:45 and once at 18:12 UTC on 2026-10-04, per vLLM's AGENTS.md)

The additional results in the second run (53948, 50717, 56082, 59857, 57180, 55688, 59759, etc.) were each checked by title; none is related to this fix. #55905 is still OPEN, head `7b7dcafd05`.

```
gh issue view 55524 --repo vllm-project/vllm --comments     # RFC: 0 comments; the design update lists "prefix caching for Mamba2 layers" as a startup-rejection item
gh pr list --repo vllm-project/vllm --state open --search "55524 in:body"   # 57539 55627 56244 55905 54993
gh pr list --repo vllm-project/vllm --state open --search "55905 in:body"   # 55895 56244 55627 57539 57123
gh pr list --repo vllm-project/vllm --state open --search "mamba prefix caching batch invariant"
gh pr list --repo vllm-project/vllm --state open --search "VLLM_BATCH_INVARIANT mamba prefix"
gh pr list --repo vllm-project/vllm --state open --search "exact replay prefix caching"
gh pr list --repo vllm-project/vllm --state open --search "mamba align copy batch invariant"
```

- #55627, #55905 and #56244 all keep the `mamba_cache_mode != "none"` rejection (confirmed with `gh pr diff`).
- #57539 is kernel-only.
- #54993 is a different approach (fixed chunking + recovery), and also requires `mamba_cache_mode='none'`.
- #55868 only does GDN.
- The remaining results are ordinary align/prefix-cache fixes and do not duplicate this fix.

## Files

- Local results directory for this fix (`bi-mamba2-pc/`):
  - `build/`: scripts, logs, manifest, sha256.
  - `unit/`: sbatch, test list, negative-control hook, `run1` (initial tests), `run2` (final tests + negative control).
  - `bipc/`: client, compare, per-leg result.json, `compare.txt`.
  - `gate3/`: sbatch, receipt, cross-image json.
  - `bench/`: scripts, configs, `pc-summary.txt`, `sp2-summary.txt`.
  - the driver and unit-rerun Slurm logs (`driver-<job-id>.out`, `unit2-<job-id>.out`).
- Lustre `$WORKDIR/bi-mamba2-pc/`: all run directories, including server.log and metrics.
- Source exports: `$WORKDIR/src/bi-mamba2-pc-{<image-source-commit>,<fix-commit>}`.
- Three jobs: one for build, unit, probes, gate3 and the random bench; one unit rerun; one shared-prefix bench. All on node A, all finished normally.

## 9. Second round (2026-10-05)

Working directory: Lustre `$WORKDIR/bi-mamba2-pc/r2/` (including server.log and traces), with a local mirror in the results directory under `r2/` (scripts, logs, summaries; no traces or result.json). Jobs: one on node A (build, unit tests, probes, gate3), three on node B (preemption, profiler), one on node C (profiler). Two of the node-B jobs were cancelled by me; reasons below.

### 9.1 Image and unit tests

- The build script is the same as round 1, except the base sha in the manifest became a parameter. Build log: pre ok 2/2, `content_mismatch 0`, `git_only []`.
- Unit tests (inside the image, test code taken from the scratch-branch commit):

| Test | Result |
|---|---|
| test_mamba2_batch_invariant | 10 pass |
| test_mamba2_batch_invariance (including V1/V2 PC e2e) | 4 pass |
| Negative control (2-state copy) | 2 fail, as expected |
| nemotron_h_bi 7, exact_replay 3, ssd_bi 7, ssd_emit 15, mamba_utils 43, precopy_align 75, memcpy_u64 120 | all pass |

`ssd_emit` is a test newly added on the delivery head.

### 9.2 Real-model correctness (delivery serve args, BI1, DP4/EP, V2 runner)

**Probes** (`r2/bipc/`, `bipc_client.py` scenarios and prompts the same as round 1):

- Each of the 4 ranks gets its own thread, running the full 11-step plan concurrently; every request carries `X-data-parallel-rank`. The prefix cache is per engine, so every rank starts cold.
- 3 runs per arm, fresh server each time, order ref pcon pmu pmu pcon ref ref pcon pmu.
- Results in `bipc-compare.txt`: 684/684 bitwise identical, 0 routing errors. Per-step hits exactly match round-1 table 5.1, e.g. A_warm 13536/14096, Q_batch4 0/5952, M2 0/6688.

**Forced preemption** (`r2/preempt/`):

- serve args: delivery args plus `--num-gpu-blocks-override 32 --max-num-seqs 16`. The first attempt used 40 blocks and 64 seqs, and the server refused to start: "max_num_seqs (64) exceeds available Mamba cache blocks (40)"; that job was cancelled for this reason, and its legs are kept in `failed-kv40-seqs64/`.
- Load: 8 prompts with no shared prefixes (6 of 12000 tokens that cross 13536 during decode; 2 of 6000 tokens that cross 6768 during decode), 2000 output tokens each. Each rank sends all 8 concurrently (burst1), sends them again (burst2), and finally they are sent one by one on rank 0 (solo).
- The gold standard is the PC-off solo output.

| Leg | Preemptions (burst1 + burst2, total over 4 engines) | Recompute lookups of preempted requests (hook) | Inconsistent |
|---|---|---|---|
| pre-off1 | 4 + 4 | no hook | 0/72 |
| pre-off2 | 4 + 4 | 8, all 0 hits (PC off) | 0/72 |
| pre-on1 | 29 + 30 | no hook | 0/72 |
| pre-on2 | 28 + 24 | 52: 22 hit 6768, 30 at 0 | 0/72 |
| pre-on3 | 33 + 30 | 63: 27 hit 6768, 36 at 0 | 0/72 |

- burst1's prompts share no prefixes, so any hit in burst1 can only come from a preempted request hitting its own previously computed prefix.
- The hook is `preempt/hook/sitecustomize.py`: enabled with `PREEMPT_HIT_LOG=1`, it wraps `KVCacheManager.record_prefix_cache_stats`, only prints logs, and does not change behavior.
- With PC on there are many more preemptions, because cached blocks occupy the pool; this is expected.

**gate3** (`r2/gate3/`, stage and proxy scripts the same as round 1):

- serve (PC off) and prefixcache (PC on) both pass: PD 4/4, BI 6/6, PC on vs PC off cross-run 4/4.
- The rollouts.json sha256 of both legs is `06daedf0…`, the same as round 1.

**Real PD disaggregation**: under BI, having a KV connector is rejected at startup (`check_batch_invariant_config` in `mamba2_attn.py`), so only the PD consistency check above can be tested.

### 9.3 First profiler attempt discarded

- nsys (`--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown`, `--profiler-config '{"profiler":"cuda"}'`) captured data, but nsys hung while shutting down the app, was killed after 300 s, and wrote no report.
- That job was cancelled for this reason; scripts and logs are kept in `prof/failed-nsys/`.
- Switched to vLLM's built-in torch profiler: CUDA activity only, no stacks, `max_iterations=400`.

### 9.4 Copy cost (`r2/prof/`, `prof-summary.txt`)

- Same load as round 1's shared-prefix bench: vllm-rs random dataset bimodal, 8K/1K, c=64. First run a 64-request warmup point, then `/start_profile`, run 64 requests, capture the first 400 engine iterations.
- fix is the image as is; neg is the same image plus the round-1 negative-control hook (2-state copy; 4 NEGCTL lines in server.log confirm it took effect).
- 3 runs per arm: fix1 neg1 neg2 fix2 on node B, neg3 fix3 on node C.

| | fix (5 states) | neg (2 states) | Difference |
|---|---|---|---|
| fast-exit launch mean | 17.1 µs | 7.8 µs | **+9.3 µs / step** |
| copy launch mean (> 3× median) | 105 µs (2–3 per rank / 400 steps) | 58 µs (2–4) | +47 µs / each |
| pre-copy total over 400 steps (per rank) | 6.6–7.5 ms | 3.1–3.4 ms | +3.8 ms, i.e. 9.4 µs / step |
| Share of GPU kernel time | 0.20% (median) | 0.10% | +0.1 percentage point |

- Copies across blocks concentrate in a few steps: at c=64 the 64 requests enter almost simultaneously, and the copies from hits and from prefill crossing boundaries are merged into the same launch.
- So the real cost is dominated by the grid expansion paid on every step: `(num_reqs, num_states, 16)` programs, with states going from 46 to 115.
- fix3 (node C) has a higher total GPU kernel time (4.3–7.2 s, other legs about 3 s), so its share is lower; the absolute time of pre-copy itself matches node B.
- Throughput drops during profiling (TPOT about 56 ms), so only kernel durations are used, not end-to-end throughput.

### 9.5 Duplicate-work check (2026-10-05 00:21 UTC, per vLLM's AGENTS.md; raw output in `r2/dupcheck.txt`)

```
gh issue view 55524 --repo vllm-project/vllm --comments     # empty output; --json shows OPEN, 0 comments, last updated 2026-09-10
gh pr list --repo vllm-project/vllm --state open --search "55524 in:body"   # 57539 55627 56244 55905 54993
gh pr list --repo vllm-project/vllm --state open --search "55905 in:body"   # 55895 56244 55627 57539 57123
gh pr list --repo vllm-project/vllm --state open --search "mamba prefix caching batch invariant"
gh pr list --repo vllm-project/vllm --state open --search "VLLM_BATCH_INVARIANT mamba prefix"
gh pr list --repo vllm-project/vllm --state open --search "exact replay prefix caching"
gh pr list --repo vllm-project/vllm --state open --search "mamba align copy batch invariant"
gh pr list --repo vllm-project/vllm --state open --search "mamba2 batch invariance prefix"   # new this round
```

- #55905 is still OPEN, head still `7b7dcafd05`. #55627, #55905 and #56244 still keep the `mamba_cache_mode != "none"` rejection (confirmed with `gh pr diff`). #54993 still requires `mamba_cache_mode='none'`.
- Newly appearing related results compared with round 1:
  - #59826 (hybrid Mamba prefix cache boundary and tail saving), #59726 (fused kernel for align-mode state index), #58676 (TP2 Mamba2 hit regression test).
  - Each one's file list or diff was checked: none changes `mamba2_state_copy_func`, and none touches the BI rejection logic.
- No open PR enables prefix caching for Mamba2 under BI.

### 9.6 Issue found this round: credential leak (handled)

- In the container, `env | sort > container-env.txt` pulls in the host environment, which included a host-side messaging credential token; the 3 `container-env.txt` files from round 1 had it too.
- All Lustre and local copies from r1 and r2 have been changed to `REDACTED`; a recheck found 0 remaining occurrences. None of these files are in git.
