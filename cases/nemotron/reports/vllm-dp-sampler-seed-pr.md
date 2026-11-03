# vllm-dp-sampler-seed-pr

> Status update (2026-10-06): vllm-project/vllm#59788 was merged on 2026-10-05 (the `(seed, data_parallel_index)` version). The text below is the pre-merge draft and handling recommendation, kept as originally written; it no longer represents a to-do item.

Draft; no PR opened and nothing pushed. Recommended as the replacement body for vllm-project/vllm#59788; for how to replace it, see "How to handle #59788" at the end. Branch: a local minimal-fix branch based on upstream 6e517b15c1.

Title: `[Bugfix] Give each data-parallel engine its own global RNG streams`

---

## Purpose

Every data-parallel engine re-seeds the global RNGs with the same engine `seed` at the end of warm-up (`Worker.compile_or_warm_up_model`). Requests without an explicit `seed` sample from those global streams:

- Model Runner V2 draws each unseeded request's per-request seed from the global NumPy RNG (`SamplingStates.add_request`).
- The fused FlashInfer top-k/top-p path (V2), and V1's sampler and rejection sampler, use the global CUDA generator.

So the n-th unseeded request on every DP engine samples the same tokens. GRPO-style rollouts are hit hardest: they send the n siblings of a prompt across DP engines, and many groups come back with identical siblings.

This PR mixes the DP index into that final re-seed. `set_random_seed(seed, data_parallel_index)` seeds with `SeedSequence((seed, data_parallel_index))` when the index is non-zero.

- **Index 0 keeps the plain seed.** DP=1, and DP rank 0, are bitwise unchanged.
- The seed and the index are combined as a pair, not as `seed + index`. RL frameworks that give replicas `seed + replica_rank` therefore do not collide.
- **Placement.** The re-seed happens after weights are loaded and after profiling and CUDA graph capture. Anything seeded at init stays identical across engines, including `load_format=dummy` weights; the seed in `init_device` is untouched.
- **`data_parallel_index`, not `data_parallel_rank`.** Dense models reset the rank to 0 on each independent DP engine.
- **Unchanged paths.** Requests with an explicit `seed` use their own generators (V1) or per-request seeds (V2), so they are not affected.

Global RNG consumers on the serving path, none of which needs identical streams across DP engines:

- V2 `SamplingStates` default seeds.
- `flashinfer_sample` / `random_sample` / V1 `rejection_sampler` noise for unseeded rows.
- The MoE routing simulator (`torch.rand`, debug only).
- Python `random`, used for KV-cache metric sampling and the tmp-file suffixes of tiering fs I/O.

Within one engine, every TP/PP worker gets the same `(seed, index)`, so the ranks stay consistent with each other.

## Duplicate check

- `gh pr list --repo vllm-project/vllm --state open --search` with each of:
  - `data parallel seed`
  - `default seed data parallel`
  - `sampler generator data parallel`
  - `identical samples data parallel`
  - `data_parallel_rank seed`
  - `seed dp rank sampler`
  - `model runner v2 seed`
  - `duplicate outputs data parallel`
  - `same seed data parallel engines`
  - `set_random_seed data parallel`
  - `global RNG data parallel`
  - `data_parallel_index seed`
- The only match is #59788. It is my own earlier, larger version: per-rank generators inside the V2 sampler, ~40 lines, V2 only, and keyed on `data_parallel_rank`, so it does not fix dense models. This PR replaces it.
- #57018 (bounding request seeds to int64) touches request seeds, not the default streams.
- Issue searches found no existing report.

## Test Plan

- Unit test `tests/utils_/test_torch_utils.py::test_set_random_seed_differs_per_data_parallel_index`. It checks that index 0 equals the plain seed, that each index is reproducible, that 4 indices give 4 distinct streams, and that the engine seed still matters.
- E2E on GB200 ×4 with Nemotron-H Lightning 30B-A3B NVFP4: DP4 + EP, `VLLM_BATCH_INVARIANT=1`, `--seed 42`, V2 runner unless noted. The workload is GRPO-like: 32 prompts × 8 samples, temperature 1.0, top_p 1.0, max_tokens 512, 6 consecutive rounds on one server. It is submitted either as one `n=8` request per prompt or as 8 × `n=1`.
- Determinism checks with every request pinned to a rank via `X-data-parallel-rank`:
  - explicit-seed outputs before vs after;
  - a seeded request alone vs inside a 64-request mixed batch under batch invariance;
  - run-to-run reproducibility of unseeded requests over two server launches.
- Train/inference parity: the verl GRPO recipe's quick alignment gate (rollout vs actor `probs_diff`), with rollout dumps to count distinct samples per group.

## Test Result

Unit tests: `pytest tests/utils_/test_torch_utils.py tests/v1/worker/test_gpu_worker.py` gives 57 passed. Without the fix, the new test fails.

Collisions per round, shown as unique completions of 256, groups with a duplicate of 32, and identical sibling pairs of 896:

| | round 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| before, V2, n=8 | 64 · 32 · 42.9% | 68 · 32 · 41.3% | 70 · 32 · 40.6% | 133 · 32 · 20.4% | 132 · 32 · 20.5% | 134 · 32 · 20.1% |
| before, V2, 8 × n=1 | 151 · 32 · 13.7% | 152 · 31 · 15.8% | 180 · 32 · 8.9% | 188 · 32 · 8.1% | 189 · 28 · 8.5% | 183 · 31 · 9.2% |
| before, V1, n=8 | 142 · 32 · 18.3% | 148 · 32 · 17.0% | 192 · 32 · 7.4% | 192 · 32 · 7.1% | 192 · 32 · 7.1% | 192 · 32 · 7.1% |
| after, V2 n=8 / V2 8 × n=1 / V1 n=8 | 256 · 0 · 0% | 256 · 0 · 0% | 256 · 0 · 0% | 256 · 0 · 0% | 256 · 0 · 0% | 256 · 0 · 0% |

Determinism, in every case with identical tokens and logprobs:

| Check | Before | After |
|---|---|---|
| Explicit seeds, 4 ranks, before vs after | – | identical |
| Seeded request alone vs in a 64-request mixed batch (BI), ranks 0 and 3 | identical | identical |
| Unseeded, same dispatch, two server launches | identical | identical |
| Unseeded, distinct outputs across 4 ranks | 1 of 4 | 4 of 4 |
| Unseeded, rank 0, before vs after | – | identical |

GRPO quick gate (verl recipe, BI on, 16 prompts × 4, 3 steps): rollout vs actor `probs_diff` max/mean = 0 on every step both before and after. Distinct samples per step: before 30/48/45 of 64 (step 2 had zero advantage in every group); after 64/64/64.

## AI assistance

This change was developed with AI assistance (Claude Code). The submitting human must review every changed line and re-run the tests above before this is marked ready.

- [ ] Human submitter has reviewed every changed line and run the tests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## How to handle #59788 (for the human)

Recommendation: force-update #59788 to this minimal version and change its title to the title above (dropping `[Model Runner V2]`). Reasons:

- The current #59788 uses `data_parallel_rank`, which has no effect for dense models under DP, and its body is also written in terms of rank.
- It only fixes V2; V1 shows the same problem in measurements (see the table above), and the minimal version fixes both.
- The diff shrinks from about 40 lines plus two test files to 6+3 lines of source plus one 15-line unit test.
- DP=1 and rank 0 are bitwise unchanged, which makes it easy for reviewers to accept.

Needs human confirmation before changing: the force-push discards the PR's original commit history. The PR is currently a draft with no review comments, so the impact is small.

Two more points:
- #59788 contains only the first two commits of the local branch (same patch-id), without the index line.
- The third commit on an earlier local branch was signed off with a different email address than the first two; the minimal version uses a single sign-off address throughout.
