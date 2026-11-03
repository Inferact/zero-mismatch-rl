# sglang-dp-seed-collision

## Conclusion
- **SGLang's default mode has the same problem by mechanism, but it rarely triggers in practice.** Every DP rank has the same `random_seed` (when unspecified, one is drawn at random at startup and broadcast to all ranks), and sampling uses the global torch CUDA RNG. Two sibling samples produce the same long output only when, during prefill on their respective ranks, **both the row index and the RNG offset are the same**; this basically only happens in the first small batch after a fresh server start. At P≥64 it is indistinguishable from the baseline.
- **SGLang deterministic mode without a seed: 100% identical.** This has nothing to do with DP; a single rank behaves the same: requests without a seed all use `sampling_seed=42` and then sample by hashing (seed, position).
- **miles avoids this problem.** In deterministic mode miles sends `rollout_seed+i` for each sibling sample, and measurements match the baseline (0.1% identical, all short answers of ≤8 tokens). In non-deterministic mode miles does not send a seed, but each engine uses `random_seed=args.seed+rank`; dp_size>1 also requires dp-attention, and the usual deployment is multiple engines behind a router, so the "multiple DP ranks sharing one seed" situation does not arise.
- **The vLLM problem is much more severe, and the existing fix has no effect on dense models.** The fix branch distinguishes ranks with `parallel_config.data_parallel_rank`, but for dense (non-MoE) models under DP this value is reset to 0 by `reconfigure_for_independent_dp_rank()` (`vllm/v1/engine/core.py:1388-1398`), so the 4 engines get exactly the same seed stream, and the n=4 case gets even worse. In a **draft copy**, switching it to `data_parallel_index` gives 0 collisions in every combination. Nemotron is MoE and goes through `DPEngineCoreProc`, so its rank is correct, and the validation on verl (1024/1024) holds. This gap needs to be fixed on the delivery branch (the delivery branch was not changed this time).

## Setup
- One node (GB200×4). Model Qwen2.5-1.5B-Instruct (local). Data DAPO-Math-17k, sliced in order, batches 0/1/2 non-overlapping. Parameters temperature 1.0, top_p 1, top_k -1, max_new_tokens 512; other parameters follow the model's generation_config (the same on both sides, repetition_penalty 1.05). All requests are sent concurrently in one go. Two submission modes: `n4` one n=4 request per prompt; `sep` 4 n=1 requests, submitted in prompt-major order.
- SGLang: `lmsysorg/sglang:v0.5.21` (commit e00930c5, 2026-09-29) → converted to `<image>` (an arm64 sqsh of that tag). Launch arguments `--dp-size 4`, DP dispatch is the default `load_balance_method=round_robin`, attention uses trtllm_mha. In deterministic mode the workspace overflows during prefill graph capture under flashinfer, so `--attention-backend triton` is used instead.
- **No separate build of sglang-miles**: compared with e00930c5, sglang-miles (258bc1a8) has identical `sampling_batch_info.py` (seed=42 default), `set_random_seed`, the `tp_worker` seed broadcast, and `data_parallel_controller.py`. `sampler.py` only changes where the temperature division for logprobs happens under RL on-policy, which does not affect the random source. So results measured on main represent it.
- vLLM: candidate image `<image>` (the delivery-candidate vLLM/verl image; V2 Model Runner, seed=0), `--data-parallel-size 4`. The fixed version bind-mounts the 4 files of the fix commit (identical to the candidate's vLLM baseline otherwise); `fixidx` is a draft copy with only one line changed: `data_parallel_rank`→`data_parallel_index`.

## Results (each cell is the range over 3 batches; at P=16 the group count is 3×16, at P=256 it is 3×256)

| Arm | Submission | P | Distinct outputs/total | Groups with a duplicate | Identical sibling pairs | Non-identical pairs sharing a prefix ≥16 |
|---|---|---|---|---|---|---|
| SGLang default | n4 | 16 | 187/192 | 0–19% | 0–7.3% | 2–19% |
| SGLang default | n4 | 256 | 3068/3072 | 0–0.8% | 0–0.1% | 3% |
| SGLang default | sep | 16 / 256 | 192/192 · 3070/3072 | ≤0.4% | ≤0.1% | 1–3% |
| SGLang default, first batch after a fresh start (3 times) | n4 | 16 | 49–58/64 | 31–88% | 6–16% | 19–35% |
| SGLang default, first batch after a fresh start (3 times) | sep | 16 | 56–64/64 | 0–44% | 0–8% | 0–26% |
| SGLang deterministic, no seed | n4/sep | 16/256 | exactly 1/4 | 100% | 100% | – |
| SGLang deterministic + miles-style seed (what miles does) | sep | 16 / 256 | 191/192 · 3067/3072 | 0–6% / ≤0.8% | ≤1% / 0.1% | 0–3% |
| vLLM unfixed | n4 | 16 / 256 | 105/192 · 2786/3072 | 100% / 24–34% | 40–50% / 7–8% | 63–100% / 72–90% |
| vLLM unfixed | sep | 16 / 256 | 151/192 · 2946/3072 | 63–81% / 9–16% | 10–22% / 2–4% | 12–49% / 33–50% |
| vLLM fixed (data_parallel_rank) | n4 | 16 / 256 | 60/192 · 2413/3072 | 100% / 42–53% | 84–88% / 20–26% | 50–100% / 74–95% |
| vLLM fixed (data_parallel_rank) | sep | 16 / 256 | 152/192 · 3004/3072 | 50–88% / 3–15% | 8–21% / 1–3% | 6–25% / 14–26% |
| vLLM fixed + explicit seed | sep | 16 / 256 | all distinct | 0 | 0 | 1–5% |
| vLLM fixed, DP=1 | n4/sep | 16 | all distinct | 0 | 0 | 1–5% |
| vLLM fix switched to data_parallel_index (draft) | n4/sep | 16 / 256 | all distinct | 0 | 0 | 1–3% |

Baseline: with different seeds, about 0.1% of sibling pairs are identical, all short answers of ≤8 tokens (the model directly gives `\boxed{}`), and about 1–3% share a prefix ≥16. Anything above this level is caused by RNG correlation.

## Mechanism (SGLang, from `meta_info.dp_rank` and a scheduler hook)
In a draft `sitecustomize` (env-gated, not delivered), every `run_batch` records the rid, the row index, and the offset of the CUDA default generator. Aggregated over the 3 P=16 runs after a fresh start:
- Among cross-rank sibling pairs, when **both the row index and the generator offset are the same** at prefill, 27/128 pairs produce the same long output (>16 tokens); the other pairs share a prefix and then diverge, because the composition of later decode batches differs and the row indices shift.
- Pairs with a different row index or offset: only 1/920 pairs have the same long output.
- round-robin sends the 4 siblings of one prompt to the 4 ranks in turn, and the arrival order on each rank is exactly the same, so as long as the batch splits on each rank happen to match, the row indices align naturally. After a fresh start every rank's offset is 0, and a small batch is usually prefilled in one go, so alignment is most likely. Under heavier load, each rank's prefill split differs with ZMQ arrival times (for example at P=256 rank0's first batches are 48/16/4/50 and rank1's are 40/24/115…), the offsets quickly diverge, and the collisions disappear.

The vLLM mechanism is known: unfixed, every engine uses the global numpy RNG to assign a seed to the n-th request in the same sequence; for why the fix fails for dense models, see "Conclusion".

## Practical impact
- Under SGLang's default DP deployment, GRPO is only affected with small rollout batches on a freshly started (or freshly restarted) server, and only the first batch, to a much lesser degree than vLLM. At P≥64 it cannot be measured.
- Deterministic mode without a seed is a 100% degeneration, but it only hits you if you write your own client, turn on deterministic mode and do not pass a seed; miles already handles it.
- vLLM: with dense models + DP the fix is still ineffective; it needs to switch to `data_parallel_index` (or use both) before delivery.

## Files
- Raw data: `results/sglang-seed/raw/*.jsonl` in the run's results directory (one line per completion: prompt_idx/sib/submit_pos/rid/dp_rank/seed/output_ids).
- Scheduler logs: `dbg/<arm>/sched_gpu*.jsonl`. Summary tables: `summary.json`, `summary.tsv`.
- Scripts: `_bench/{client.py,run_sglang.sh,run_vllm.sh,analyze.py,sgl_dbg/sitecustomize.py}`. Server logs: `logs/`.
