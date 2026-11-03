# #257 gaps and follow-ups

When the earlier parity branch (#260) was closed, all training-side work moved onto #257. This
report records the read-only audit of #257 at `8df84b5d3` against the validated parity branch, the
three follow-up PRs that closed the gaps needed for the 16-node recipe (#264, #265, #266), and the
first full-model and proxy runs on that combination.

Path shorthand: `L/` = `experimental/lite/megatron/lite/`, `V/` = `L/model/deepseek_v41/vllm/`,
`S` = `experimental/lite/examples/verl/scripts/run_deepseek_v41_dapo.sh`. Line numbers refer to
`8df84b5d3`.

- None of the parity branch's commits are ancestors of `8df84b5d3`, so every conclusion below
  compares code, not history.
- At `8df84b5d3`, #257 did not modify `megatron/core/`.

## Item-by-item audit

| # | Item | Needed on #257? | Evidence |
|---|---|---|---|
| 1 | Reference-cycle leak of the per-forward state under full recompute (found on the parity branch, see [long runs](long-runs.md#trainer-memory-leak-on-the-earlier-branch)) | No (avoided by structure) | #257 keeps cross-layer latent/indexer state in `KVSourceState.values` (`L/primitive/parallel/source_state.py`); recompute uses `KVSourceState.checkpoint`, whose closure does hold `self` (same shape as the leak), but the whole forward runs inside `with self.source_state.microbatch()`, whose `finally` clears `values`, so the cycle is broken when the forward returns. Confirmed in the G6 short job below: no per-step growth |
| 2 | `bias_vl` vision routing | Deferred at the time | Later opened as #268 |
| 3a | DistOpt `dp_reshardable` save | Present | `L/primitive/ckpt/distckpt.py` identical to the parity branch |
| 3b | Frozen Engram tables out of the checkpoint | Present | Generic path in `distckpt.py`; DS4.1 path moved to `L/primitive/ckpt/row_state.py` |
| 3c | Streamed D2H save + fsync/DONTNEED + Lustre EFAULT retry | **Needed** | Core `filesystem_async.py` is unmodified and `S` forces `MCORE_DIST_CKPT_STREAM_D2H=0`. The old path adds 84–94 GiB per node at save; full-model runs on the new path already peaked at 873.8–877.0 GB against a limit of about 895, so the old path would exceed it. EFAULT: 9 on the first full-model save, each fixed by a chunk retry |
| 3d | `empty_cache` after save + gmu 0.65 | Present | `mlite_engine.py` `finally` calls `aggressive_empty_cache(force_sync=True)`; `S` defaults gmu to 0.65 |
| 4a | Engram q/k FP32 gradients | Present | `V/engram.py` promotes saved q/k with `.float()` before making them leaves |
| 4b | Parameter readiness | Present | `V/model.py` calls Core's `ensure_params_ready`; export forces `start_param_sync(force_sync=True)` |
| 5a | Tensor-core backward GEMMs | Present (#257's own version) | `V/linear.py` (BF16 operands, FP32 output, `main_grad` accumulation), MoE via TE grouped GEMM, head via `linear(fp8=False)` |
| 5b | FlashMLA load order | Present (sitecustomize part) | `verl_mlite/compat.py`; the parity branch's readable error on wrong order only affects diagnostics |
| 6 | Mega attention | Not at `8df84b5d3`; added later on #257 (`3bc04c83a`, `attention_core=mega`) | At `8df84b5d3` the protocol accepted only `flashmla` and `S` set the rollout to `FLASHMLA_SPARSE_DSV41` with `VLLM_BI_MEGA_ATTN=0`; so the mega-arm performance numbers of the parity branch do not carry over |
| 7a | Chunked optimizer-state offload | Wiring present; **exact-size pinned buffers missing** | Core `chunked_optimizer_state_offload.py` still uses `pin_memory=True`; the caching host allocator rounds to a power of two (+41% measured), about 170 GB more per node, which exceeds the host budget |
| 7b | Gradient release (`grad_offload`) | Present, off by default | `S` sets `GRAD_OFFLOAD=False` ("opt-in until its separate RL gate passes"); the recipe needs True to free 64.4 GiB per GPU in the rollout phase |
| 7c | Level-2 sleep preserve list | Present | `S` passes `'*.bias_vl'`, `'draft.*'`; compat checks the preserve list |
| 7d | `DSV41_MEGAMOE_MAX_TOKENS` | **Needed** | `S` forwards the variable but nothing reads it: `V/moe.py` hard-codes `capacity = 8192`. #257 keeps the EP all-gather backward: each chunk's `a_all` / `g_all` is `EP × capacity × 4096` BF16, 4 GiB each at EP64 and 8192 rows, 1 GiB each at 2048 rows |
| 7e | Mega-attention deterministic dKV switch | Not needed | Only affects the mega backward |
| 7f | Validation output cap | Present | compat |
| 7g | Per-micro-batch two-path logprob probe | Optional, absent | Zero-diff evidence comes from verl's own `rollout_probs_diff` |
| 7h | Validation dump directory | Missing, script level | Pass `trainer.validation_data_dir=...` as an extra override |
| 8a | Native loss independent of the CP split | Not needed on the verl path | verl computes its own loss |
| 8b | MegaMoE empty chunks | Should not be needed | `V/moe.py` pads to the EP group's max row count before the all-gather, so a 0-row rank participates; the 0-row GPU path at EP64 was not run |
| 8c | DistOpt gather before export | Present | `start_param_sync(force_sync=True)` |
| 8d | FP32 DistOpt shards as leaf views (Core) | Not needed for the recipe | Only `HybridDeviceOptimizer` (`offload_fraction>0`) needs it; chunked offload uses 0 |
| 8e | Engram q/k rounded BF16 RNE into the gate | Present | `V/engram.py` |
| 8f | Layer-wise reload, bucketed sync, rollout sleep level | Present | #257's compat patch set is a superset |

### #257 defaults that differ from the recipe

| Switch | #257 default | Recipe | Note |
|---|---|---|---|
| `ACTOR_CP` | 1 | 2 | #257 supports CP |
| Attention / rollout backend | flashmla → `FLASHMLA_SPARSE_DSV41`, `VLLM_BI_MEGA_ATTN=0` | mega on the parity branch | Only flashmla existed at `8df84b5d3` |
| `VLLM_BI_MEGA_GATE` / `VLLM_BI_MEGA_MHC` | `${…:-0}` | 1 / 1 on the parity branch | **Trap**: an outer script that exports 1 overrides #257's 0 and forwards it into Ray, while #257's trainer uses the non-mega gate (`dsv4_topk` / `topk_hash_softplus_sqrt`). Rollout and trainer then route differently. Export 0 or nothing |
| `GRAD_OFFLOAD` | False | True | Otherwise gmu 0.65 does not fit |
| `MCORE_DIST_CKPT_STREAM_D2H` | forced 0 | 1 + `MCORE_DIST_CKPT_DROP_CACHE_BYTES=1GiB` | Needs #266 |
| `ROLLOUT_FULL_DETERMINISM` | True | False | False needs the sentinel routing fix (#268) to be safe |
| Head dtype | float32 on both sides (`hf_overrides={head_dtype: float32}`) | BF16 on the parity branch | Consistent on both sides, but numerically a different recipe |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 | unset | |
| CUDA graphs | `compilation_config.mode:0`, capture up to 16 | FULL_AND_PIECEWISE up to 8190 | Batches above 16 tokens run eager; #269 later switched to vLLM's default capture sizes |
| Indexer | `indexer_kv_dtype: mxfp4`, `indexer_sparse_logits: false` | unset | |
| Engram config | adds `embedding_across_dp: true`, `dp_shared_memory: false` | `cpu_offload`, `use_thp` | Host ledger must be recomputed |
| `logprobs_mode` | `raw_logprobs` | unset | |
| lr / weight decay | 1e-5 / 0 | 1e-6 / 0.1 | Human-chosen recipe items |

## Follow-up PRs

Each does one thing; they were split from one local branch and their trees matched the commits
tested below.

| PR | Change | Touches Core? |
|---|---|---|
| [#264](https://github.com/ISEEKYAN/Megatron-LM/pull/264) MegaMoE training chunk size | `V/moe.py`: `capacity = int(os.environ.get('DSV41_MEGAMOE_MAX_TOKENS', '8192'))`; the symmetric-buffer key already includes `max_tokens`. Test: 300 rows through chunk sizes 8192 and 128, output and dx bitwise equal, both buffer keys created | No |
| [#265](https://github.com/ISEEKYAN/Megatron-LM/pull/265) Exact-size pinned buffers for chunked offload | `L/primitive/optimizers/megatron_wrap.py`: with chunked offload on, replace Core's `ChunkedOptimizerStateOffloader._new_cpu_buffer` with a numpy-owned buffer registered with `cudaHostRegister` at exact size (unregistered when the storage is freed; falls back to `pin_memory=True` if registration fails). Tests: buffer pinned, storage bytes equal tensor bytes, async copy round trip | No (replaced from mLite) |
| [#266](https://github.com/ISEEKYAN/Megatron-LM/pull/266) Bounded save-time host memory + EFAULT retry | Core `filesystem_async.py` (+106/−2): `MCORE_DIST_CKPT_STREAM_D2H=1` streams each item through a pinned buffer; `MCORE_DIST_CKPT_DROP_CACHE_BYTES` does fsync + `POSIX_FADV_DONTNEED`; local files written unbuffered with chunk rewrite on EFAULT. Launcher default changes from forced 0 to 1, drop-cache default 1 GiB, forwarded through the Ray runtime env | **Yes** |

#266 is a trimmed version of the parity branch's change (+221 in Core): it drops the stage-bytes cap
(the stager buffer is sized by the largest item) and the item-level bounce retry, because on the
first full-model save all 9 EFAULTs succeeded on the first retry of a 64 MiB chunk and the item-level
retry never fired. Bypassing Core would have meant copying a `FileSystemWriterAsync` subclass and a
save strategy into mLite (about 250 lines), which is more fragile.

Unit tests (one GB200 node): #264 block-size bitwise test and the existing native-parity MoE test
pass; #265 4 passed; #266 (`torchrun`, 2 GPUs) stream on/off both round-trip bitwise and drop the page
cache, EFAULT rewrites the chunk from its start, EIO is not retried: 4 passed. Two problems in the
base were fixed along the way: the DS4.1 test `conftest.py` rejects any imported vLLM module although
the MoE GPU test imports vLLM (run with `--noconftest`), and the test's router weights stayed on host
(added `.cuda()`). Exact-size pinning cannot be measured in a unit test on GB200 (pinned memory does
not show in RSS; `host_memory_stats` is empty in this Torch), so it was verified through the G6 host
peak.

## G6 short job (full model)

Code: #257 at `8df84b5d3` + #264, #265, #266. Image with vLLM #62365 (including the level-2 sleep and
idle-DP fixes), no overlays. 16 nodes. Overrides as in
[recipe and memory](../recipe.md#launcher-overrides-used-by-the-validated-runs), with
`ROLLOUT_FULL_DETERMINISM=False`, no validation, save at step 2.

- Leg A: steps 1–2, save at step 2 to node-local disk, rc=0.
- Fan-out: 69 files that existed on one node only were copied through Lustre to all 16 nodes (rank
  0's `.metadata`, `metadata.json`, `__0_0.distcp` of 99 GB, `lr_scheduler.pt`; `data.pt`; 64
  `rng_state_rank_*`).
- Leg B: fresh Ray cluster, `resume_path=global_step_2`, steps 3–6, rc=0. Job COMPLETED 0:0 in
  1:39:24 with no restart. A traceback at exit after step 6 (DataLoader worker SIGKILLed) did not
  affect results.

| Step | Params | probs_diff max / mean | Grad norm | Step s | Gen s |
|---|---|---|---|---|---|
| 1 | θ0 | 0 / 0 | 0.0449 | 612 | 291 |
| 2 | θ1 (save 97 s) | 0 / 0 | 0.0395 | 525 | 218 |
| 3 | θ2 (after resume) | 0 / 0 | 0.0388 | 500 | 213 |
| 4 | θ3 | 0 / 0 | 0.0442 | 470 | 257 |
| 5 | θ4 | 0 / 0 | 0.0396 | 405 | 198 |
| 6 | θ5 | 0 / 0 | 0.0419 | 418 | 206 |

Rank-0 trainer `memory_allocated` at each weight export (before the rollout wake), GiB:

| Leg | First sync (untrained) | Later exports |
|---|---|---|
| A | 32.007 | 38.129 → 38.129 |
| B | 32.007 | 37.590 → 37.590 → 37.672 → 37.672 |

The rise after the first training step is optimizer and gradient residency; after that it is flat
(one +0.08 GiB in 6 steps), with no +0.73 GiB per step as in the leak on the earlier branch.

Host memory at save (`free` used, GB = 1e9), per node over the 97 s save window: 874, 829, 829, 823,
835, 825, 834, 841, 826, 825, 831, 829, 832, 823, 822, 817. Highest 873.5 GB = 813.5 GiB on the driver
node, under the limit; by Ray's measure (cgroup current − file) 782.6 GB against a threshold of
910.7 GB. Saving raised each node by +9 to +22 GB (old path: +84 to +94 GiB). Written to node-local
disk, so no EFAULT; the Lustre EFAULT path was covered only by unit tests here.

Timing: leg A about 41 min (startup 19 min, 2 steps 19 min, save 97 s), fan-out about 5 min (99 GB
through Lustre and back), leg B about 51 min (startup 21 min, 4 steps 30 min). Generation (200–290 s
per step) is not comparable with the parity branch's mega-arm numbers: different attention backend,
FP32 head, CUDA graph capture up to 16 only.

This job did not fingerprint the synced weights, so "θ2 after resume equals uninterrupted θ2" was
not checked here; the proxy study below checks it.

## Proxy resume study

Code: same as the G6 short job. Proxy: 40 layers + 3 MTP layers, experts 0..15 (gate rows 0..15),
byte copies of the release otherwise. 2 nodes × 4 GB200. Recipe conventions of the G6 job (BI=1,
sparse FlashMLA, FP32 head, `VLLM_BI_MEGA_*=0`, CP2 / EP8, gradient release, chunked offload, lr
1e-6, weight decay 0.1, MegaMoE chunk 2048, sleep level 2, DSpark), scaled to batch 16 × 8 and
response 512.

Deviations: `ROLLOUT_FULL_DETERMINISM=True` (with False, vLLM's per-engine sampler is unseeded and
request ids are `uuid4` routed by load, so rollouts are not reproducible even without a resume; verl
at `b9c513c44c06` injects a seed and deterministic request ids only when True); a content-hash reward
(the proxy scores 0 on DAPO).

Runs: U (one node pair, steps 1–3 uninterrupted), R (another node pair: leg A steps 1–2 with a
node-local save at 2, leg B fresh Ray cluster resuming for step 3), S (same pair as U: leg A steps
1–3 with a save at 2, leg B fresh Ray cluster resuming from that save for step 3).

Primary comparison U vs S leg B (same node pair):

- **θ2 state: bitwise.** Optimizer masters and `exp_avg` / `exp_avg_sq` (6702 entries over 8 ranks),
  param groups, LR scheduler, RNG; model parameters and buffers at the step-3 old-logprob forward
  (10328 entries). Also R: leg A θ2 before save vs leg B θ2 after load bitwise (6702 entries).
- **Step-3 batch: bitwise.** Responses, input ids, rollout logprobs (vLLM), old logprobs (trainer),
  rewards, advantages (128 sequences).
- **Step-3 metrics:** all non-timing metrics equal except `actor/grad_norm` 0.2039460986852646 (U)
  vs 0.20392650365829468 (resumed). θ3 then differs (1086 of 1100 `exp_avg` parameters). Equal step-3
  gradients: head, final norm, layer 39 FFN (experts, router, shared experts, `ffn_norm`), layer 39
  `ffn_mixes.alpha_{pre,post,res}` and `attn_mixes.alpha_pre`. First differing gradients in backward
  order: layer 39 `ffn_mixes.{bias, mapping_proj.weight}` and all layer 39 attention parameters.
- Control U vs S leg A (both uninterrupted, same pair): bitwise θ0..θ3 and identical grad norm on all
  3 steps.
- Control U vs R leg A (different node pairs, no resume): step-1 batch bitwise, but step-1 grad norm
  0.2963424324989319 vs 0.2964489161968231 with the same layer-39 signature, so the training backward
  is not bitwise across node pairs either.
- `probs_diff_max` 0.0 at every step of U, R and S.

Reading: resume restores state, rollout and forward exactly. The backward of the first layer reached
(layer 39 attention and its mHC mixing) is not reproducible across processes or node pairs. The
kernel responsible was not isolated.
