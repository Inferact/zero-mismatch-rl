# SWE-bench Pro agentic RL

Multi-turn agentic RL (ReAct agent with shell and editor tools in remote sandboxes) on the full model
with BI=1, to check that zero mismatch holds with prefix caching, many turns, long contexts and
independent sampling, and what it costs in rollout speed against BI=0.

Code: a Megatron integration branch = #257 at `3bc04c83a` plus the changes later opened as #264–#270
and the trainer side of #271 (DeepSelect zero-based rows; added before the 20-step runs). vLLM: the
#62365 build with the level-2 sleep and idle-DP fixes, plus as overlays the opt-in small-M router
(`VLLM_BI_BF16_SPLITK=4`, `VLLM_BI_TOPK_TRITON=1`) and, for the 20-step and 128K runs, the opt-in
DeepInfer calls (WO epilogue, inert on the FlashMLA core, and DeepSelect) with the `wo_b` scale-refresh fix. The
agent loop is in #270.

## Integration

- The verified verl legacy trainer (`use_v1=False`) with a new verl agent loop. Per sample it runs the
  uni-agent `swe_bench_pro` task (sandbox, ReAct agent, grading, all uni-agent code unchanged) and
  replaces only the agent's HTTP chat client with an in-process token-in / token-out session.
- Not the uni-agent `AgentFrameworkRolloutAdapter`: it needs the verl v1 trainer with TransferQueue
  (not in the image), it was written against a later verl API (renamed continuous-token methods), and
  verl's bundled `deepseek_v4` encoder and tool parser use V4's DSML format, which differs from V4.1's.
- V4.1 codec: rendering with vLLM's `deepseek_v41_encoding.encode_messages`, incremental turns
  rendered with `context`; tool calls parsed with vLLM's `deepseek_v41` parser. Unit tests: per-turn
  incremental tokens equal the whole-conversation rendering (chat and thinking modes); tool-call round
  trip.
- Per-session DP pin: `data_parallel_rank` is fixed by a hash of a session key, so every turn of an
  episode hits the same replica and DP rank (prefix cache reuse). One request id per session.
- Rollout-only environment: a compat patch applies `VERL_MLITE_ROLLOUT_ENV=K=V` only to the vLLM
  rollout server actor and its engine processes; the server prints its own `VLLM_BATCH_INVARIANT` at
  the first generate, which is the evidence for which arm ran. (The trainer runs the BI contract in
  both arms, so generic batch-invariant logs cannot tell the arms apart.)
- Trajectory handling: one chain, append only; the loop raises if the agent rewrites history, instead
  of silently misaligning tokens.

Host-side checks before GPU runs: gold patches on 5 repositories (Go/Python/JS) all reward 1 and a
no-patch control reward 0; the 64 instances used for the paired runs all reward 1 with gold patches
(32 concurrent, p50 13.5 s per instance, max 178 s); a scripted policy (apply gold + submit) gets
reward 1, submit-only gets 0; every turn's context equals the full V4.1 rendering token for token;
`response_mask` and logprobs aligned. 256 concurrent sandbox creations all succeeded (create p50 4.4 s,
p95 6.9 s, max 8.7 s).

## Recipe

Both arms share the image, recipe and data; they differ only in the rollout's `BATCH_INVARIANT`.

| Item | Value |
|---|---|
| Topology | 16 nodes × 4 GB200, colocated; actor EP64 / CP4 (32K) or CP16 (128K) / TP1 / PP1; one DEP4 rollout replica per node (16 replicas) |
| Data | SWE-bench Pro instances with prebuilt 8-CPU / 16 GB sandbox templates and verified gold patches (375). For the 2-step pairs: 64 fixed instances, rows 0–31 at step 1 and 32–63 at step 2, identical in both arms. For 20-step runs: 64 held out (per repository, seed 42), 311 for training, shuffled with seed 42, 9 steps per epoch |
| Batch | 32 prompts × n=8 = 256 episodes per step; mini-batch = train batch (one update per step, on-policy) |
| Context | prompt 8192 + response 24576 (32K), later 8192 + 122880 (128K). Each turn's `max_tokens` is the remaining response budget; over budget the episode ends and is graded on the repository state at that point |
| Agent | uni-agent ReAct, tools `str_replace_editor` / `shell` (command timeout 120 s) / `submit`, max 150 steps |
| Sandbox | hosted E2B, lease 600 s, runtime timeout 7200 s |
| Grading | 1 if all FAIL_TO_PASS and PASS_TO_PASS pass, else 0; eval timeout min(900 s, per-instance calibration); no length penalty |
| Session | wall-clock cap 3600 s per episode; on timeout or sandbox failure the sample's `response_mask` is all 0 (no loss) and the failure is counted |
| Sampling | temperature 1.0, top_p 1.0, top_k −1; V4.1 chat mode (no reasoning) |
| Rollout engine | V2 runner; prefix cache on with session pinning; DSpark k=5 (drafter frozen); KV `nvfp4_ds_mla`; `deep_gemm_mega_moe`; `allgather_reducescatter`; FULL_AND_PIECEWISE graphs; max_num_seqs 64, mnbt 8192; gmu 0.65; sleep level 2; Engram CPU offload; `full_determinism=False` |
| Algorithm | GRPO, `norm_adv_by_std=False`, KL 0, clip 0.2 / 0.28 / c 10, token-mean, lr 1e-6 constant, weight decay 0.1, grad clip 1.0; old logprobs recomputed by the actor (bypass off) so `rollout_probs_diff` is defined |
| Actor | `impl=vllm`, `attention_core=flashmla`; actor determinism off (the deterministic sparse-attention backward needs a 26.55 GiB workspace at 32K with CP4 and OOMed) |

| | BI=1 | BI=0 |
|---|---|---|
| Rollout | `VLLM_BATCH_INVARIANT=1` + all `VLLM_BI_*`; attention pinned to `FLASHMLA_SPARSE_DSV41`; `head_dtype=float32`; `no_swa_bounded_replay=True`, `prefix_cache_retention_interval=128` (see [prefix-hit replay](#prefix-hit-replay)) | `VLLM_BATCH_INVARIANT=0` on the rollout server only (every `VLLM_BI_*` is gated by it); vLLM default kernels, attention backend and head dtype |
| Actor | identical | identical |
| Expected `probs_diff` | 0 | non-zero, recorded only |

BI=0 does not silently use BI kernels: the gate `is_batch_invariant_quant_kernel_enabled()` reads
`VLLM_BATCH_INVARIANT` before checking for the kernel library.

### Deviations from the uni-agent training recipe

Reference: uni-agent's quickstart Qwen3 MoE training script and ReAct task config; the Nemotron SWE
recipe in this repository is the second reference.

| Item | uni-agent | Nemotron | This recipe | Reason |
|---|---|---|---|---|
| Trainer | v1 `colocate_async` + TransferQueue + adapter | v1 sync + TransferQueue + adapter | legacy sync trainer + verl agent loop running uni-agent task/agent/grader | No TransferQueue in the image, verl API mismatch; with synchronous on-policy, BI=1 diff = 0 has a clear definition |
| Async | warmup 1 batch, staleness allowed | 0 staleness | none | same |
| Model / parser | Qwen3-Coder-30B, `qwen3_coder` | Nemotron, `qwen3_coder` | DS4.1-Flash, V4.1 encoding + vLLM `deepseek_v41` parser | model |
| Data | SWE-reBench train, Verified eval | SWE-bench Pro mixed (375) | SWE-bench Pro subset | both arms same samples, same order |
| Batch | 64 × 8, mini 16 (4 updates per step) | 32 × 8, mini 32 | 32 × 8, mini 32 | one update per step, strictly on-policy |
| Context | 8K + 128K | 32K, later 256K | 8K + 24K, later 8K + 120K | memory checked first at 32K |
| Training parallelism | TP4 PP2 CP2 EP8 (8 × 8 GPUs) | PP1 EP4 CP4 (8 nodes) | TP1 PP1 CP4/CP16 EP64 (16 × 4) | TP=1 required by BI |
| Rollout parallelism | TP4 | DP4 EP4 → TP1 | DEP4 per node | TP1 required by BI |
| max_steps / shell timeout | 200 / 120 | 150 / 120 | 150 / 120 | |
| Session cap | none | 3600 s | 3600 s | cuts the long tail |
| Trajectory selection | all | longest | single chain, append only | |
| Unfinished episodes | not masked | not masked | not masked (graded on final repo) | |
| Sandbox failure | `max_failed_prompt_fraction` | 0.25 + fail on agent error | mask the sample, count failures | the legacy trainer cannot drop prompts |
| Group over-sampling | — | K=4 | none | both arms must see the same samples |
| Old logprob | recomputed | from rollout (bypass) | recomputed | `rollout_probs_diff` needed |
| MTP | none | on and trained | DSpark on, drafter frozen | DS4.1 mLite does not train MTP |
| Prefix cache | on | on + DP pin | on + replica and DP pin | |
| gmu / mnbt | 0.7 / prompt + response | 0.35 / 16K | 0.65 / 8192 | DS4.1 recipe values |
| Thinking | Qwen non-thinking | — | V4.1 chat | the recipe has no V4.1 setting; chat matches non-thinking |

## Speed metrics

- Gate metric: `timing_s/gen`, the rollout phase wall clock (first sandbox creation to last episode
  graded), per step and summed. Budget: sum(BI=1) / sum(BI=0) ≤ 1.15.
- LLM-only metrics from per-turn traces (start/end of each generate, context length, tokens
  generated): `llm_sum` (sum of all LLM call durations, session-seconds), `llm_tok_ms` (per generated
  token), per-turn LLM time, `llm_crit_s` (the largest per-episode LLM time), `tool_sum` (episode wall
  minus LLM time: sandbox creation, tools, grading). Both arms use the same task set and sandbox
  service, so `tool_sum` should be close; if not, trajectory shapes differ.
- Also from vLLM: prefix-cache hit rate, generation throughput, spec acceptance.

With only a few steps of 256 episodes the noise is dominated by trajectory differences, so the
per-token and per-turn LLM times matter more than `gen`.

## Prefix-hit replay

The first full 2-step pair was much slower under BI=1 (`llm_sum` ratio 2.11). Cause: exact SWA
replay on prefix hits.

- After every prefix-cache hit, BI=1 recomputed `exact_replay_rows(128, 40)` = 5208 tokens exactly
  (a hit of up to 5208 tokens recomputes everything); BI=0 recomputes 128. Simulated from traces, BI=1
  prefill computed about 79M rows of which 64M were replay; BI=0 about 11M. vLLM's prompt/iteration
  counters exclude replay, so the reported 15.1M vs 9.1M only showed the short-hit full recompute.
- The two row counts are consistent: 2541 = 128 + 19 × 127 is the decoder trimming (after layer 20,
  19 layers compute only tail rows); 5208 = 2541 + 21 × 127 extends that through layers 0–20.
  Tightening by the turn's new token count S (mean about 610, p50 295) would save only about 10%.
- Single-node reproduction on the same load: BI=1 / BI=0 1.63–2.12.

Options:

| Arm | What | Cost |
|---|---|---|
| bi1 (before) | exact replay of 5208 rows + decoder trimming | about 5k extra tokens per hit |
| bi1nr | `--no-swa-bounded-replay`: SWA KV goes into the prefix cache and a hit reads it directly; decoder trimming turns off with it | with `prefix_cache_retention_interval=0` SWA keeps only the "end of prompt" boundary, so the previous turn's output is recomputed |
| bi1nrd | bi1nr + `--prefix-cache-retention-interval 128`: decoded SWA blocks are cached too, so a hit extends past the previous turn's output | SWA blocks stay in the LRU (about 21 KB per token, more than ten times the compressed KV) |

DeepInfer instead snapshots an SWA ring buffer per sequence (window + rollback + 128) for producer
layers only (layers 0–20); consumer layers recompute via prefill trimming. bi1nrd is the equivalent
available in vLLM, but it stores all 40 layers and keeps every SWA block in the LRU. At 16 sessions
per node the hit volume matched BI=0 (cached 6.765M vs 6.780M) with no preemption and an unchanged KV
pool (2.02M tokens per rank).

Correctness (BI=1):

| Check | Result |
|---|---|
| 40-layer, 8-expert proxy, multi-turn hits (4 sessions × 7 turns, seeded, DSpark, prefix cache on), rollout vs the `impl=vllm` trainer token by token | 4480 generated tokens per arm, 0 differ, `probs_diff_max` 0; the three arms' tokens and logprobs bitwise equal to each other and on re-run; every turn hit, and bi1nrd hits include the previous turn's output |
| Real verl proxy smoke with the two new arguments | 2 steps, 3 turns per episode, `rollout_probs_diff_max` 0 |
| Full model, SWE-shaped load, bi1 / bi1nr / bi1nrd across configs, launches and passes | 462 turns × 11 BI=1 passes, generated ids and logprobs all bitwise equal |

Speed, same-node ABBA (16 sessions from real episode shapes, tool gaps capped at 5 s, 462 turns and
85K generated tokens per pass, about one full step per node), `llm_sum` in session-seconds, two
launches each:

| Arm | Pass 1 (cold) | Pass 2 (steady) | Pass 2 / bi0 | ITL ms (pass 2) | Iteration tokens |
|---|---|---|---|---|---|
| bi0 | 420 / 422 | 338 / 324 | 1.00 | 12.9 / 12.7 | 403K |
| bi1 | 642 / 650 | 586 / 594 | 1.78 | 21.9 / 22.2 | 554K |
| bi1nr | 480 / 488 | 403 / 401 | 1.21 | 15.8 / 15.7 | 530K |
| **bi1nrd** | 512 / 499 | **369 / 353** | **1.09** | 14.7 / 14.0 | 418K |

The remaining about 9% is decode: ITL 1.11–1.13×, the BI kernels themselves.

Two traps on the way to the full run:
- verl drops engine kwargs whose value is boolean False when building the command line, so
  `swa_bounded_replay=False` never reached vLLM. Pass `no_swa_bounded_replay=True` instead (vLLM
  receives `--no_swa_bounded_replay`).
- verl sets `VLLM_LOGGING_LEVEL=WARN` in the Ray runtime env, hiding vLLM INFO lines such as "Exact
  SWA replay"; pass `VLLM_LOGGING_LEVEL=INFO` to check.

Full-model 2-step results against the same BI=0 run (`llm_sum` ratio step 1 / step 2 / total; ITL
ratio at step 2): arguments not applied 2.26 / 1.97 / 2.14; both arguments 1.60 / 1.22 / 1.45 (1.14);
plus the small-M router switches 1.48 / 1.16 / 1.35 (1.09). After the first 40 s of each step the
steady per-token ratio was 1.09 and 1.05; the excess was the first-turn burst at each step start.

## Step-1 cold start

- Source: FlashInfer CuTe DSL MXFP8 GEMM kernels (BI=1 via `bi_mxfp8_gemm`, BI=0 via `mxfp8_gemm`).
  Compiled kernels are cached only in process memory (keyed by tactic, not M); the
  nvidia-cutlass-dsl 4.8.0 in the image has no file cache for compiled kernels, so a cache directory
  stays empty regardless of environment variables.
- Trigger: when vLLM's FlashInfer autotune result file already exists on the node, startup reads it
  and skips tuning, and so skips compilation; the selected tactics are then compiled during the first
  inference steps (about 10 per rank, several seconds each, and one rank compiling stalls every rank
  through EP synchronization). A startup that tunes from scratch compiles every tactic up front.
- Single node, first 20 s of a cold pass, ms per token: with an autotune cache hit BI=1 32.2 / BI=0
  28.4 (about 40 runtime compilations); with a fresh autotune directory per job (including a
  sleep-2 → reload → wake cycle as in RL step 1) BI=1 6.2 / 6.1 vs BI=0 5.4 / 5.3, no runtime
  compilation; the cold pass-1 `llm_sum` ratio was 1.04 and 1.13.
- Fix without code changes: give every job a new, empty `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` (passed
  through the Ray runtime env). Cost: about 170 s more startup (413 vs 241 s), before training, not
  counted in rollout time. BI uses only non-split-K tactics, and every tactic gives the same bits per
  row, so this should not affect `probs_diff`; a proxy run with re-tuning had 0 at steps 1–3. A code
  fix would precompile the selected tactics on cache hit, or export CuTe DSL AOT.

## Runs and results

| Run | Context / CP | Steps | Result |
|---|---|---|---|
| Paired 2-step A/B (before DeepSelect) | 32K / CP4 | 2 per arm | `llm_sum` B/A 1.117 / 1.093, total 1.108; `gen` 1.128 / 1.020, total 1.082; ITL 1.07; BI=1 `probs_diff` 0 / 0; 0 infra failures |
| G6 short job | 32K / CP4 | 2 + save + resume 1 | `probs_diff` 0 / 0 / 0; save 94 / 97 s on 16 nodes; host peak 889 GB on the head node; `update_weights` 88 s (local-expert sync not enabled) |
| BI=1, 20 steps | 32K / CP4 | 20 | `probs_diff_max` 0 at all 20 steps; checkpoints at 10 and 20 complete; held-out resolved rate (64 × 4) 0.742 at step 0 → 0.707 at step 20; `update_weights` about 88 s |
| G6 short job | 128K / CP16 | 2 + save + resume 1 | `probs_diff` 0 / 0 / 0; save 95 / 98 s; `update_weights` 8.7 s; host peak 897 GB; step 1770–2193 s; mean response 33K–42K tokens; 0–1 of 256 truncated; 0 timeouts; reward 0.90–0.93 |
| BI=1 | 128K / CP16 | 10 (stopped) | `probs_diff_max` 0 at all 10 steps; held-out resolved rate at step 0 0.891 |
| BI=0 | 128K / CP16 | 20 | rc=0, no restart; reward 0.84–0.94; mean response 29.5K–45.0K; response clip ≤ 0.016; truncated 0–5 of 256; 0 timeouts; 0 infra failures; `probs_diff_max` 0.80–1.00 (expected non-zero); step 924–2256 s; held-out mean@4 0.875 → 0.922, best@4 0.985 → 0.987 |
| Fair BI=1 pair for the BI=0 run | 128K / CP16 | 2 | `probs_diff_max` 0 / 0 |

## 128K context

Ledgers before choosing CP (measured at 32K with CP4):

1. Training phase per GPU: actor peak 139.3 GB allocated (every trajectory at 32K, 8K tokens per CP
   rank), 148.8 GB reserved; vLLM keeps about 12 GiB while asleep; `nvidia-smi` peak 178–182 GB of
   184.3 GiB, 3–7 GiB headroom. At 128K, tokens per CP rank are 131072 / CP: CP4 32K (not feasible),
   CP8 16K (to measure), CP16 8K (same as measured).
2. Rollout phase per GPU: vLLM at gmu 0.65 × 184.3 = 120 GiB, of which KV 29.5 GiB, about 2.01M tokens
   per DP rank; the trainer keeps about 34–45 GiB resident. The KV pool does not change with context:
   4 sessions × 128K = 524K tokens per DP rank, 26% of the pool.
3. Host per node: peak 893 GB on the head node (minimum available 135 GB), others ≤ 846 GB. Longer
   context adds only about 3 GB for the driver-side batch.
4. Sleep level 2 and DEP4 per node do not change with context.

Memory check (16 longest prompts × n=4 with `ignore_eos`, every trajectory filled to 122880 tokens,
one trajectory per training micro-batch, session cap raised to 10800 s, BI=1):

- CP8: OOM in the actor update; the indexer (compressed attention) needed 6.88 GiB with 155 GiB
  allocated and 6.3 GiB free. Rollout and old-logprob passed.
- CP16: passed. Actor peak 129.0 GB allocated, 148.6 GB reserved; `nvidia-smi` peak 167.5 GiB (about
  17 GiB headroom); host peak 815 GB; `probs_diff_max` 0; `update_weights` 8.9 s; old-logprob 75 s,
  update 251 s; generation 4690 s (120K tokens with `ignore_eos`).
- 6 of 64 episodes lost the sandbox shell during the 120K generations and were masked.

Decision: CP16 at 128K (8K tokens per CP rank, the same as 32K with CP4); dense DP drops from 16 to 4.

## Rollout speed

128K, BI=1 10 steps vs BI=0: `gen` ratio 1.27 over 10 steps but `llm_sum` 0.961 and `llm_tok_ms`
0.976. The gap was sandbox creation time in the BI=1 run's time window (time from episode start to
first generation, p50 55–353 s at steps 1–8 vs 9–11 s in the BI=0 runs), an external load unrelated
to BI.

Fair pair at 128K (BI=1 and BI=0, 2 steps each, no pre-training evaluation, same prompts):

| | Step 1 | Step 2 | Total |
|---|---|---|---|
| `gen` (s) | 831 / 1250 = 0.66 | 1092 / 894 = 1.22 | 0.897 |
| `llm_sum` | 1.066 | 1.020 | 1.043 |
| `llm_tok_ms` | 2.749 / 2.649 = 1.038 | 2.745 / 2.666 = 1.030 | 1.034 |
| Per-turn LLM | 1.080 | 1.029 | |
| ITL ms | 10.89 / 10.51 | 10.84 / 10.61 | |
| `tool_sum` | 1.001 | 0.996 | |
| `llm_crit_s` | 162 / 145 | 149 / 119 | |

Prefix hit 0.981 / 0.982, 0 preemptions, spec acceptance 0.70–0.72, `update_weights` 8.4–9.0 s in both
arms. Under the same conditions BI=1's LLM side is about 3–4% slower per token; `gen` is set by the
slowest episode's sandbox and tool tail and swings between 0.66 and 1.22 per step.

Single-node concurrency sweep at 128K (same rollout configuration, no sleep mode; load built from
3068 real BI=1 trajectories, tool gaps capped at 5 s; spec acceptance about 0.58 because tokens are
synthetic; bi0, bi1, bi1, bi0 launches, median of two):

| Sessions per node | Running (bi0 / bi1) | Tokens per step (bi0 / bi1) | LLM ms per token bi1/bi0 | ITL bi1/bi0 | Prefill time bi1/bi0 | Decode time bi1/bi0 | Wall bi1/bi0 | Preemptions |
|---|---|---|---|---|---|---|---|---|
| 16 | 1.8 / 1.9 | 16.1 / 16.7 | 1.045 | 1.024 | 1.21 | 1.02 | 1.005 | 0 |
| 64 | 12.7 / 12.8 | 21.9 / 22.1 | 0.980 | 0.941 | 1.25 | 0.95 | 0.996 | 0 |
| 256 | 69.3 / 73.7 | 53.0 / 54.6 | 1.084 | 1.063 | 1.26 | 1.07 | 1.016 | 0 |

Launch-to-launch drift: at 256 sessions BI=0 measured 15.93 and 14.59 ms per token in two launches;
adjacent-pair ratios 1.06 and 1.11. The model-side gap stays ≤ 8.4% at every concurrency (worst pair
1.11). BI=1 prefill is 21–26% slower but is about 5% of model time.

## Vision sentinel ids

Symptom on a 1-node proxy with BI=1, DSpark and prefix cache: one sample per run matched the trainer
up to a mid-response position and mismatched from there to the end (e.g. 2790 tokens from position
5402 of an 8192-token truncated response, max |Δlogp| 0.54; `probs_diff_max` 0.004–0.016 for the
step). Prefix-cache resets were all successful and there were 0 preemptions, and the first occurrence
was at step 1 before any weight sync.

Cause: every mismatching tail started right after a sampled id in 129264–129268 (image and
multimodal placeholder sentinels; the V4.1 base is 129264, not V4's 129257). Three id-dependent gaps
between the trainer and vLLM:

1. Routing: vLLM routes these ids with `ffn.gate.bias_vl` (the release has a vision tower, also under
   `language_model_only`, also for sampled ids); the trainer skipped `bias_vl` at load and did not
   pass token ids to routing. A forced-input probe: layer-0 router logits equal, top-k ids and weights
   different.
2. Engram: vLLM treats 129264 as dead (breaks the n-grams and zeroes the gate); the trainer hashed and
   gated it as text.
3. Resync: level-2 sleep + layer-wise reload does not keep `bias_vl` unless exported; fingerprints
   matched the release at the first sleep and differed at every later one.

The exposure grows with independent sibling sampling and long responses: about one sentinel per 60K
sampled tokens on the proxy. Fix (#268): route with `bias_vl`, mark the Engram dead token, export
`bias_vl` in FP32 on every sync. Verification: 129264, 129266, 129267, 129268 and a control id
injected into a 512-token document, all logprobs bitwise equal; the routing GPU unit test passes for
BI=0 and BI=1; SWE BI=1 smoke with 129264 and 129267 injected into all 32 prompts, `probs_diff_max` 0
on both steps, rollout `bias_vl` checksum equal to the release on 40 layers × 4 DP ranks × 3 sleeps.
The rules are in the [isolation guide](../../../skill/guides/debugging-mismatch/01-isolate-the-side.md).
