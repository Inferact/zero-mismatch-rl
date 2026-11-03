# R3 return path: why the BI=0 arm's rollout is slower than BI=1 (2026-10-04 to 06)

In the 100-step BI=0 vs BI=1 comparison the rollout throughput is inverted (BI=1 25,931 vs BI=0 22,611 tok/s; source: [100-step comparison report](reports/bi0-vs-bi1-100step.md)). The cause is not a kernel; it is the path by which R3 (routed-experts replay data) is returned step by step from EngineCore to the frontend. Methods are in [Performance and profiling](../../skill/guides/performance-profiling.md) and [Ablation method](../../skill/guides/debugging-mismatch/04-ablation-method.md). Image vLLM: the final-delivery vLLM head (an internal commit on a vLLM fork).

## Workload

Nemotron-H 30B-A3B NVFP4, DP4×EP4, `flashinfer_cutedsl` MoE, replay bench: 1024 requests, with each request's output length fixed to the length obtained by letting the bi1_recipe arm generate naturally once with the RL sampling parameters (744 hit EOS, 280 reached the 14000 cap; mean 7312, max 14000), ignore EOS for fixed length, metric makespan. Repeat noise of the same configuration is about ±5% (nor3, two runs on the same node: 176.8 / 185.1 s). Rerun difference of the e2e step-1 gen time is <1% (bi1 256.5/255.0 s, bi0 284/285 s). One seed per configuration; repeat only when noise has to be resolved. Order-of-magnitude reference (Nemotron-H, GB200, 4 GPUs): in RL rollout under high-concurrency decode, a single request runs at about 40 tok/s (source: order-of-magnitude record from the former performance topic, no committed report), engine aggregate 23–27k tok/s ([100-step comparison](reports/bi0-vs-bi1-100step.md)); per-request values from the low-concurrency bench are in [delivery results](results/final/README.md#1-bench-bench).

## Makespan table (s)

Each cell lists all runs of that arm (two batches of experiments, nodes not fixed), with the mean in parentheses.

| arm | meaning | makespan, all runs |
|---|---|---|
| bi0_nor3 | R3 off (floor) | 174.2 / 186.5 / 176.0 / 175.8; another batch 176.8 / 185.1 (mean 179.1) |
| bi0_recipe | current path: SHM store, R3 bytes returned with the response every step | 230.5 / 227.8 / 242.8 / 241.4 / 244.5 / 226.9; another batch 227.0 / 225.8 / 232.2 (mean 233.2) |
| bi1_recipe | BI=1 arm (no R3) | 202.7 / 203.7 / 204.6 (mean 203.7) |
| ablation A0 | capture hook only (no snapshot/commit/return) | 183.5 / 180.2 / 184.1 (mean 182.6) |
| ablation A | capture + worker data plane + packing, scheduler does not return | 181.4 / 180.9 (mean 181.1) |
| ablation E | worker→engine IPC in place, scheduler discards (not put into EngineCoreOutput) | 186.7 / 188.5 / 193.6 (mean 189.6) |
| ablation B | capture off, return path on (fake data) | 229.6 / 241.3 / 221.7 (mean 230.9) |
| bi0_thr | zero-copy threshold raised to 1e6 | 228.5 / 252.1 / 231.4 (mean 237.3, not a lever) |
| finish-only (scheduler side only) | R3 returned only when the request finishes, worker→engine IPC unchanged | 193.7 / 213.7 / 208.1 (mean 205.2) |
| fixpack | finish-only + per-step packed IPC | 189.2 / 192.4 (mean 190.8) |
| fixpack2 | same as above, second version | 189.5 |
| fixpack2 + threshold 1e8 | — | 185.0 |
| fixpack + client does not request R3 | final state not serialized either | 180.4 |

Conclusion: about 80% of the R3 overhead is in the per-step engine→frontend return path (E vs fully on differs by about 45 s); the capture hook itself costs about 0 (B ≈ fully on; A0/A ≈ nor3). After fixpack, the remaining ~10 s (189–192 vs 180.4 when R3 is not requested) is in serializing the whole final-state payload.

## Root cause (file:line, final-delivery vLLM head)

- One row per token of (52 layers, 6) uint8 = 312 B, larger than `VLLM_MSGPACK_ZERO_COPY_THRESHOLD` (256). `vllm/v1/serial_utils.py:236-254 _encode_ndarray` turns the single row per request per step into a zero-copy frame: 128 EngineCoreOutputs per engine per step → 129 ZMQ frames (`vllm/v1/engine/core.py:1916-1919 encode_into + _send_msg_tracking_payload`).
- The frontend `vllm/v1/engine/core_client.py:1154-1156` receives frames with copy=False, `serial_utils.py:388-396 _decode_ndarray` builds an ndarray per row with frombuffer, `output_processor.py:703-705` appends 1 row per step to `routed_experts_chunks`, and only calls `np.concatenate` at finish (`output_processor.py:459-461`). The frontend keeps about 7312 zmq Frames + memoryviews + ndarrays per request (4 engines × 128 req), millions of live objects. py-spy: 86% of the recipe frontend is in `output_handler`, while the nor3 frontend is nearly idle.
- Login-node micro-benchmark (128 req/step, 1 engine): at thr=256, encode+zmq 0.50 ms, decode 0.12 ms; at thr=1e6, 0.15/0.14 ms; without R3, 0.015/0.03 ms. worker→engine pickle: 128 AuxRequestOutputs 0.21+0.13 ms vs a single packed array 0.012+0.015 ms.
- R3 does not introduce a GPU→CPU sync: the snapshot `.to(uint8)` is on the main stream and the D2H is non-blocking on the copy stream (`vllm/v1/worker/gpu/async_utils.py:163-166`); async scheduling is still True; capture writes into a statically preallocated buffer and does not break cudagraph (the cgm control arm shows the same delta as non-cgm).

## Fix: fixpack (overlay of 3 files, no monkeypatch, no new kernel)

1. `vllm/distributed/aux_output_connector/connector.py`: `AuxOutputSchedulerConnector.take_output` returns only once, when the request finishes (`_pending_rows` accumulates per request and concatenates at finish). The semantics are equivalent to the baseline: output_processor already only concatenated and exposed routed_experts at finish.
2. New `AuxStepOutput`: each step the worker concatenates all newly captured rows into one ndarray + per-request (token_start, lo, hi), so the IPC pickles only one buffer (`worker.py _commit_output`).
3. `vllm/v1/executor/ray_utils.py`: adapted to the new type (this experiment does not use Ray).

Moving R3 into the MoE kernel is not a way out: flashinfer_cutedsl and Humming are both modular MoE (they receive the top-k computed by the router), and a monolithic one (#44214 style) saves nothing.

## Equivalence

BI=1 cannot do the equivalence check: Mamba2 BI refuses prefix caching, while AuxOutput forces PC. Instead use BI=0 + strictly serial single-request greedy (8 prompts × 8 samples, max_tokens 512, sample i pinned to DP rank i%4), so the batch composition of each engine is fixed, and compare the sha of tokens and routed_experts. The reference run is baseline eq r2 (the one judged clean in the PR 57968 evaluation batch).

| comparison (A vs B) | sha equal |
|---|---|
| baseline eq r2 vs PR57968+SHM eq r1 | 64/64 |
| PR57968+SHM eq r1 vs PR57968+Mooncake eq r1 / r2 (cross-node, cross-backend) | 64/64 / 64/64 |
| PR57968+Mooncake eq r1 vs r2 | 64/64 |
| baseline eq r2 vs **fixpack** eq r2 | **64/64** |
| baseline eq r2 vs **fixpack2** eq r3 | **32/64** |
| baseline eq r2 vs baseline eq r1 (first batch, same node) | 48/64 |
| baseline eq r2 vs baseline eq r1 (second batch, another node) | 48/64 |
| baseline eq r2 vs baseline eq r4 (second batch, a third node) | **16/64** |
| baseline eq r1 (first batch) vs baseline eq r1 (second batch) | 32/64 |
| baseline eq r2 vs baseline eq r5 / r7 (third batch) | 32/64 / 64/64 |
| pairwise among the four baselines from the second batch on (r1/r4/r5/r7, 6 pairs) | 32, 48, 48, 48, 16, 32 |

How to read it:

- **Drift of the baseline itself recurs; it is not a one-off event.** The unpatched baseline is unequal to the reference run four times, at 48/64, 48/64, 16/64, 32/64 (another run, r7, happened to be 64/64), on three different nodes; the two baseline r1 runs are only 32/64 against each other. What was seen in the first batch — "all 16 unequal on DP rank 1, from prompt row 23/24 on, only MoE layers, not reproduced on a same-node rerun" — is a detail of one of these occurrences and cannot be generalized into "the only exception". This is routing not being batch-invariant under BI=0 (serial single requests still do not pin other state inside the DP engine), unrelated to the patch; but it means "64/64" can only be obtained by comparing against a clean reference.
- **fixpack's equivalence holds only for the pair "fixpack eq r2 vs baseline eq r2" (64/64)**; it is also 64/64 against PR57968+SHM.
- **fixpack2's equivalence is not established**: 32/64 against the reference, 32/64 against fixpack, 16/64 against baseline r4. A patch difference cannot be excluded, nor can another baseline-style drift; until fixpack2 reruns to 64/64, the 189.5 s row in the table only counts as a performance number.
- Bitwise dump comparison (`bitwise.py`: 128 concurrent greedy requests of varying lengths, pinned ranks): baseline dump vs fixpack dump have 51/128 tokens and 48/128 routed_experts equal; but between two **baseline** dumps only 1/128 tokens and 0/128 routed are equal, and the finish-only dump is only 31/128 against the baseline. This shows that 128-request concurrent greedy is itself not reproducible under BI=0; these numbers only prove that both sides' outputs have consistent shape/dtype and can be parsed, and can neither prove nor refute fixpack's equivalence; the equivalence conclusion comes only from the serial eq mode above.

## Evaluation of PR 57968 (Mooncake keys-only + ShmBlockObjectStore)

The current path is already an SHM store (`store.py BlockObjectStore`; the PR only renames it `ShmBlockObjectStore`). Same-node control makespan: current 232 / PR+SHM 247 (+6%, within the historical fluctuation of 226–244, leaning toward noise but higher both times) / PR+Mooncake rdma 214–219, tcp 205 (−7%). The gain is entirely in prefill: TTFT mean 52–56 → about 40 s (40.5, ≈ nor3's 40.7), TPOT flat; batch retrieval of 1024 records at the end takes 1.5–1.6 s (the streaming fetch p50 of 22 s is an in-process bench artifact; in verl the fetch happens in the trainer process).

Defects:

- The 2356 keys returned are all `r3-tail:<uuid>`; not a single one is a content-addressed block key. `MooncakeOutputPublisher.take_output` only references a block key when one step's output fully covers a hash block (4176 tokens), which decode's 1 row per step never satisfies; at block boundaries a synchronous `get_concatenated` reads back the whole 1.3 MB block only to slice 1 row; otherwise each step's `rows.tobytes()` accumulates into the scheduler (about 7800 312 B bytes objects per request), and at the end the scheduler thread synchronously `put_batch`es 2.28 GB again. Content addressing/dedup exists in name only, and storage is 2×. Per-request object churn moves from the frontend to the scheduler; it does not disappear.
- A race that always crashes: at a block boundary the scheduler reads back at the instant the worker has just finished the put, the Mooncake master returns `replica_not_ready` → `BlockObjectStoreError` → EngineCore fatal error; 2 out of 3 replays crashed. To get numbers, a bounded retry of ≤5 s was added to `get_concatenated` in the overlay (not PR code; triggered 6 times with rdma, 11 times with tcp).
- The PC requirement is not lifted: `vllm/config/vllm.py` still has `raise ValueError("AuxOutput Connector requires prefix caching.")`; the key is the KV block hash, a structural dependency.
- New sync points: the worker calls `store.flush()` each step and waits for the Mooncake background write queue to drain; synchronous reads at block boundaries.
- New dependencies: a Mooncake master + one 4 GB segment per EngineCore + a trainer-side reader. Consumption in verl needs `aux_output_keys` passed through (`CompletionOutput.aux_output_keys`; under the mooncake backend `routed_experts=None`), and the consumer builds a `MooncakeDistributedStore` reader (segment=0, same `VLLM_AUX_OUTPUT_MOONCAKE_CONFIG_PATH`), concatenates via `batch_get_buffer(keys)` then `np.frombuffer(uint8).reshape(-1,52,6)`; retention is controlled by Mooncake lease/eviction.

Conclusion: **recommend the finish-only route; do not adopt keys-only as this PR currently stands**. Both eliminate the churn of 1-row ndarray chunks per step on the frontend; finish-only sends the whole R3 in one go from EngineCore to the frontend, needs no Mooncake, does not change the scheduler's per-step path, introduces no race, and needs no trainer changes. If keys-only is wanted in the future, the correct approach is for the scheduler not to read back blocks but to reference the block keys the worker has already published directly, uploading only the tail that is less than one block.

## Second floor: frontend logprobs unpacking (evening of 2026-10-06)

After fixpack2 the BI=0 bench makespan falls at 178–184 s, the same level as nor3 without R3 (178.3, n=8), but the frontend asyncio thread is still 100% busy, while the BI=1 recipe frontend is only 0.98 busy with a loop lag of 2.3 s. On an e2e single-step rollout the two arms are nearly the same (BI=1 256 s, BI=0+fixpack2 258–274 s); fixpack2 only gains −4…−10% in e2e. py-spy (e2e, fixpack2): verl AgentLoop's 4 workers are <1% busy; the engine step rate of 12.9 ms/step matches the bench; the vLLM frontend (the AsyncLLM output path inside the verl process) in steady state spends 72% in `serial_utils.decode` (msgpack ext unpacking of 3 small logprobs ndarrays per token: `ext_hook` 33% + `_decode_ndarray` 33%) and 20% in `output_processor`. Frontend unit cost: bench 24.6 µs/token, e2e about 30 µs/token.

Conclusion: on the bench BI=1 is engine-bound and BI=0 is frontend-bound; in e2e both arms are pressed onto the same frontend floor (about 250 s), so BI=0's step-rate advantage does not show. Verification arms:

| arm | makespan (s) | frontend busy fraction | note |
|---|---|---|---|
| bi1_recipe | 203.2 (n=5) | 0.98 | engine-bound |
| bi1 without sampled logprobs | 199.1 (n=2) | 0.13 | removing frontend work barely changes it → engine-bound confirmed |
| bi0 fixpack2 | 178.7 (n=4) | 1.00 | frontend floor |
| bi0 fixpack2 without sampled logprobs | 163.5 (n=3: 157.9/165.2/167.5) | 0.17 | BI=0 engine floor |
| bi0 fixpack2 + `max_num_seqs` 256 | 197.2 (n=3) | 0.6 | slower, not adopted |
| **bi0 fixpack2 + frontend overlay** | **169.8 (n=4: 164.4/165.7/167.5/181.4)** | 0.82–0.88 | 16% faster than BI=1 |
| bi1 + frontend overlay | 204.1 (n=2) | 0.79 | unchanged, again confirming engine-bound |

Frontend overlay (`patches/frontend`, 3 more files changed on top of fixpack2, still an experimental overlay, not delivered): on the scheduler side, the step's three batched logprobs arrays are converted with a single `.tolist()`, sliced per request into a pure-list `LogprobsWire` struct and sent with `EngineCoreOutput`; the frontend `LogprobsProcessor` consumes the lists directly, with a fast path when `num_logprobs==0` and there is no tokenizer. Login-node micro-benchmark (128 requests × 1 token × 3 arrays): per token 9.4 → 2.1 µs (floor without logprobs 0.35). The zero-copy threshold has no effect here: these arrays are already <256 B and inlined.

Equivalence: unit-level round trip (real MsgpackEncoder/Decoder, old path vs new path) 1024 comparisons, 0 unequal; same-node back-to-back eq mode (64 requests, serial greedy): for the BI=1 recipe arm, final vs frontend tokens, logprobs and cum_logprob are 64/64 all equal; for the BI=0 arm, 48/64 requests have identical tokens and the logprobs of these 48 are all equal (the other 16 are all on one DP rank, exactly the same set as the R3 sha mismatches; this is BI=0's own cross-run drift, unrelated to the overlay, see the previous section).

Upstreamable: the `LogprobsWire` change is unrelated to R3; it addresses a generic per-token overhead in the vLLM frontend. The candidate PR content and required tests are in the workspace's frontend report (not included here).

## Implications for the 100-step comparison

100-step comparison (source: [100-step comparison report](reports/bi0-vs-bi1-100step.md)): time per step BI=1 530 s vs BI=0 601 s, rollout throughput 25,931 vs 22,611 tok/s, old-logprob 62 vs 106 s. The rollout inversion is caused by the return path in this document (per-step R3 return); after removing it, both arms are pressed together onto the frontend logprobs-unpacking floor (previous section); the old-logprob gap comes from the different training backends. The two arms' other confounders (PC on/off, gpu_mem, training engine, R3) are listed first and compared afterwards; rules in [recipe evaluation 02](../../skill/guides/recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance), specific items in the [recipe deviation table](recipe.md). The cost of PC itself is in [bi0-prefix-caching-cost](reports/bi0-prefix-caching-cost.md) and [prefix-cache-perf](reports/prefix-cache-perf.md).

## 100-step retrain (2026-10-07)

Same image, same recipe, seed 42, five arms of 100 steps each: BI=1, BI=1 + LogprobsWire, BI=0 baseline, BI=0 + finish-only, BI=0 + finish-only + LogprobsWire ("finish-only" = the routed-experts finish-only return fix, "LogprobsWire" = the frontend LogprobsWire fix; fixes mounted as overlays, not in the delivery image). Numbers are means over the first 100 steps; gen/token = mean gen / mean response length.

| arm | gen (s) | step (s) | old-logprob (s) | mean train-inference prob diff | score | mean response length | gen/token (ms) | vs BI=1 gen/token | vs BI=1 step |
|---|---|---|---|---|---|---|---|---|---|
| BI=1 | 177.4 | 541.9 | 64.0 | 0.0000 | 0.390 | 4589 | 38.66 | 0 | 0 |
| BI=1 + LogprobsWire | 175.5 | 539.5 | 63.7 | 0.0000 | 0.403 | 4629 | 37.91 | −1.9% | −0.4% |
| BI=0 baseline | 183.9 | 562.4 | 101.8 | 0.0102 | 0.523 | 4200 | 43.78 | +13.3% | +3.8% |
| BI=0 + finish-only | 176.5 | 561.8 | 103.2 | 0.0102 | 0.519 | 4377 | 40.33 | +4.3% | +3.7% |
| BI=0 + finish-only + LogprobsWire | 171.3 | 577.8 | 112.1 | 0.0105 | 0.504 | 4459 | 38.42 | −0.6% | +6.6% |

- rollout: the two layers of fixes bring BI=0's per-token generation time from +13.3% to on par with BI=1 (−0.6%; +1.3% against BI=1 that also has LogprobsWire). On the bench BI=0 is 16% faster, but this did not materialize in e2e: both e2e arms are still pressed onto the same frontend floor, and "BI=0 10% faster than BI=1" was not achieved.
- Whole step: BI=0 is still 4–7% slower; the gap is in old-logprob (Megatron-Bridge + QAT forward 102–112 s vs mLite 64 s), not in rollout. For BI=0 to catch up overall, the next step is the training-side forward.
- The fixes do not affect train-inference consistency: the mean train-inference prob diff of all three BI=0 arms is around 0.010.
- The three BI=0 arms score 0.10–0.13 higher with 3–8% shorter responses: one seed, recorded only, no conclusion drawn.

Run records: a W&B project, grouped `lr2-BI1`/`lr2-BI0`, five runs named `lr2-*` (backfilled step by step from jsonl); logs in the workspace `long-run/`, comparison script `compare-lr2.py 100`.

## Reproduction

The descriptions of the two draft PRs on a personal vLLM fork (based on a branch at the final-delivery vLLM head; not linked here) contain a full Reproduction section (full script text, length data, per-arm parameters, expected results, pitfalls, slimming guidance), using placeholders instead of internal paths:

- routed experts finish-only return (branch `draft/r3-finish-only`, = `bi0opt/patches/final` = fixpack2, the three files byte-identical)
- frontend LogprobsWire (branch `draft/logprobs-wire`, only the 3 frontend files; measured stacked on top of the finish-only PR, = `bi0opt/patches/frontend`)

Placeholder → meaning:

| placeholder | meaning |
|---|---|
| `$WORKDIR` | the experiment work directory on the shared filesystem (mounted as `/bi0opt` in the container; scripts in `scripts/`, results in `results/`, logs in `logs/`, overlays in `patches/{final,frontend}`) |
| `$RESULTS` | `$WORKDIR/results` |
| `$IMAGE` | `<image>`: the final2 delivery image (final-delivery vLLM head + pinned Megatron and verl fork commits); the e2e recipe script is also inside it |
| `$MODEL_DIR` | `<model-dir>`: the NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 checkpoint |
| `$BF16_DIR` (e2e only) | `<model-dir>`: the NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 checkpoint |
| `$DATA_DIR` | the DAPO-Math-17k parquet `data` directory |
| `$VAL_DIR` (e2e only) | the AIME-2024 parquet `data` directory |
| `$RUNS` (compare-lr2) | the shared runs directory of the long runs |
| replay length file | `$WORKDIR/results/bi1_recipe_natural_r1_<job-id>.lengths.json` (inlined in the PR as a list of 1024 integers; sha256 after reconstruction `637407a1…afb353`) |
| `$PARTITION` / `$ACCOUNT` | `<partition>` / `<account>` |
| `NODE_PREFIX` | `<node-prefix>` |
| `$NODE` | replay: seven nodes; eq: four nodes; e2e single step: four nodes; lr2 100 steps: one node per arm (bi1, bi0-base, bi0-fix, bi1-fe, bi0-fe) |

Raw scripts (not desensitized): `$WORKDIR/scripts/` (`node_job3.sh`, `in_container3.sh`, `arms.sh`, `bench.py`, `bench_r3.py`, `summarize.py`, `eq_check.py`, `lp_unit_eq.py`, `spy_agg.py`; the `subset_check.py` newly added for the PR and the micro-benchmark `microbench.py` have also been put there). e2e launchers: `rollout1.sbatch`, `rollout1_spy.sbatch` in the e2e work directory; 100 steps: `run-long5.sbatch` + `lr2-*.env` + `submit-chain.sh` + `compare-lr2.py` in the long-run work directory.

Differences between the PR descriptions and the internal report: the replay lengths were not taken from the verl rollout log but generated naturally once by `bench.py --natural` on the bi1_recipe arm with the RL sampling parameters (744 EOS, 280 reaching 14000); the PR's Benchmark section saying "taken from a real RL rollout" is inaccurate, and the Reproduction section states it correctly. The measured R3 eq comparisons between baselines are 16/32/48 of 64 (6 pairs: 32, 48, 48, 48, 16, 32); in the table in §4 of the workspace's R3 return-path report, "baseline r1 (node A) vs baseline r4 (node B) 16/64" recomputes to 32/64 (the 16/64 in `bi0opt/results` is r4 vs r7, both on node B); the 16/64 in this document's "Equivalence" section is baseline eq r2 vs r4 from the PR-57968 evaluation batch, a different pair, and is unaffected.
