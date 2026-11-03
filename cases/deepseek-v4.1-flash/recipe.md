# DeepSeek-V4.1-Flash recipe and memory budget

The DAPO recipe for full-parameter DeepSeek-V4.1-Flash on 16 nodes × 4 GB200, how it fits in GPU and
host memory with training and rollout colocated, and the checkpoint fixes it needed.

The recipe was first assembled and validated on the earlier parity branch (#260, mega attention, BF16
head, "recipe v1"); the #257 runs reuse it with the switches listed in
[launcher overrides](#launcher-overrides-used-by-the-validated-runs). Units: host memory in GiB from
`free -g` unless marked GB (1e9); GPU in GiB (184.3 GiB usable per GB200).

## Topology and data

- 16 nodes × 4 GB200 (64 GPUs). Actor TP1 / CP2 / EP64 / PP1; rollout one DEP4 vLLM engine per node
  (16 engines), TP1.
- DSpark on (5 tokens, probabilistic draft sampling, block rejection); drafter frozen at the release.
- Prompt mode `chat`: single-turn rendering of the release `encode_messages`, ending with
  `<｜Assistant｜></think>` (no reasoning segment).
- Rollout sleep level 2.
- Data: DAPO-Math-17K; batch 128 prompts × n=8, mini-batch 128 (one optimizer step per step);
  response 8192, prompt 2048; validation AIME 2024, n=8, temperature 1.0 / top_p 1.0, 4096 output
  tokens (verl at `b9c513c44c06` uses `response_length` for validation, so a compat patch sets
  `max_tokens` only on validation requests).

## Differences from the DeepSeek-V4-Flash recipe

| Item | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Reason |
|---|---|---|---|
| Nodes / EP / rollout DP | 8 / 32 / 8 | 16 / 64 / 4 (DEP4 per node) | On 8 nodes the per-rank optimizer state doubles and host memory does not fit (see [host ledger](#host-memory-per-node)) |
| Actor CP | 2 | 2 | CP2 and CP4 assembled forward bitwise and gradients verified |
| `param_offload` | True | False | Parameters stay on GPU (32.2 GiB per GPU fits in the rollout phase); verl's `param_offload` also drags the optimizer to host |
| Optimizer | FSDP2 + precision-aware + `offload_fraction` 1 | DistOpt Adam + chunked optimizer-state offload (2048 MB window) | CPU Adam needs about 18 B/param of host memory and does not fit on 16 nodes; chunked offload computes on GPU and keeps master and both moments on host (12 B/param); decoupled weight decay (AdamW) by default |
| `grad_offload` | False | True (release the gradient buffer, no host copy) | Frees 64.4 GiB per GPU in the rollout phase |
| Weight format | FP8 `hf_overrides`, block-FP8 resync | Release native format (block FP8 32×32 UE8M0 / MXFP4), no override | The release is already quantized; export and cold load agree tensor by tensor |
| MTP training, CE fusion, fused kernels | on | off | Not implemented in the DS4.1 mLite path |
| Attention / MoE / all-to-all | default, deep_gemm, deepep_low_latency | sparse FlashMLA or mega attention, `deep_gemm_mega_moe`, `allgather_reducescatter` | DS4.1 BI path |
| KV cache | fp8 | `nvfp4_ds_mla` | DS4.1 rollout config |
| Engram | — | `cpu_offload=True, use_thp=True` | Tables of 183 GiB are frozen and live in host pinned memory (read via UVA) |
| Sleep preserve | — | `sleep_preserve_parameter_names=['*.bias_vl','draft.*']` | See [level-2 sleep](#level-2-sleep) |
| gmu / mnbt / max_num_seqs | 0.6 / 2048 / 128 | 0.65 / 8192 / 64 | gmu 0.70 gives about 27 GiB of KV per GPU, far more than needed; 0.65 was the pre-authorized fallback after a wake OOM; 8192 matches the MegaMoE inference buffer |
| `load_format` | dummy | auto | The server reads the release directly, so θ0 can be checked before the first sync |
| `val_before_train` | False | True | Pre-training evaluation |
| Actor recompute | — | `full` | Does not fit without it |
| ppo / log-prob max tokens per GPU | 10240 / 10240 | 5120 / 10240 | With CP2 the micro-batch cap is 10240 / 20480 tokens, enough for the longest sequence |

## Level-2 sleep

Two pieces of state are lost by level-2 sleep, are not rewritten by sync, and are read later. Both
lived in the CuMem weights pool without a name (not a parameter, buffer or tensor attribute):

| State | Lost effect | Fix |
|---|---|---|
| Mega-attention load permutations for `wq_b` / `wo_a` (40 target + 3 drafter layers) | After reload, all 40 layers' `wq_b` / `wo_a` weights and scales are wrong | The loader keeps a host copy |
| Engram peer-exchange symmetric-memory pointer tables (4 int64 each) | All zero after wake; illegal address on the first forward | `rendezvous` runs in a private MemPool |

Both fixes are on vLLM #62365 ([details](reports/inference-batch-invariance.md#rl-integration-fixes)). With
them, the preserve list `['*.bias_vl','draft.*']` is enough; a diagnostic compat mode that retained
ownerless bytes was retired. Engram tables are CPU pinned tensors (the GPU holds only a UVA view) and
are not in the CuMem pool, so level 2 does not touch them; Engram's non-table parameters are in the
sync payload. On a proxy, 4 rounds of level-2 sleep left the drafter's 2424 private entries and the
Engram tables' 48 entries unchanged; the drafter's `embed_tokens` / `lm_head` are shared with the
target and change with each sync, as expected.

Why not level 1: its weight backup measured 592 GiB per node (4 × 80 GiB rounded up by
`pin_memory`) and was not returned after wake; vLLM alone reached 891 GiB on a node.

## GPU memory per GPU

Training and rollout time-share the GPU, so the two phases have separate ledgers (rollout at gmu
0.70 when this was written; the recipe now uses 0.65).

| Item | Rollout phase | Training phase (vLLM asleep) | Basis |
|---|---|---|---|
| vLLM | 133.7–134.0 (gmu 0.70, incl. context) | 18.0–18.7 (level-2 residue) | Full-model `nvidia-smi` |
| Actor BF16 parameters (17.28B) | 32.2 | 32.2 | Resident |
| FP32 gradient buffer | 0 (released) | 64.4 | Gradient release |
| Optimizer master | 0 | 32.1 at the step + 2 × 2 chunk window | Chunked offload; on host during forward/backward |
| Context / NCCL / MegaMoE symmetric buffers | about 5–8 | about 5–8 | Estimate |
| Activations and logits (CP2, 10240 tokens per GPU, full recompute) | 0 | headroom about 60 in forward/backward | Estimate, then measured in G6 |
| Total | about 171–174 | 120 + activations in forward/backward; about 156 at the step | Rollout phase is the tightest |

Measured: a full-model short job at gmu 0.70 reached 184.2 GB per GPU at the step 1 → 2 wake, and
after a checkpoint save the trainer kept about 1.1 GB more cache, so the next KV wake OOMed. Fix: an
unconditional `aggressive_empty_cache(force_sync=True)` at the end of `save_checkpoint`, and gmu
0.65. Later full runs peaked around 179.6k–180.5k MiB per GPU (`nvidia-smi`) with actor peak
allocated 135.5 GiB.

KV capacity and throughput at gmu 0.65 vs 0.70: KV per rank 27.05 GiB / about 1.088M tokens vs 36.27
GiB / about 1.458M tokens (about 106 vs 142 concurrent 10240-token sequences). Production needs at
most 64 × 10240 ≈ 655k tokens per rank, so 0.65 does not limit rollout. Throughput 0.65 / 0.70 at
c = 1 / 8 / 32: 0.991–1.004; at c=64 0.951 and 0.996 in the two pairs.

## Host memory per node

Node 956 GiB. Ray's memory monitor (version 2.59, checked in the container) kills at 95% of the step
cgroup's `memory.max` (958.6e9 B = 892.77 GiB), measured as `memory.current − active_file −
inactive_file`, i.e. 910.7e9 B. The memory sampler's `*_gb` columns are GB (1e9); comparing them to
GiB budgets once understated the margin.

| Item | Rollout phase | Training phase (vLLM asleep) | Basis |
|---|---|---|---|
| Node baseline (outside the job, OS, Lustre) | 81 | 81 | Measured 78–83 before serving |
| vLLM (Engram tables with THP 186–191 + worker processes) | 221 | 239 | Measured; sleeping adds preserved-page copies (drafter 2.36 × 4) |
| Trainer, 4 ranks: master + 2 moments (12 B × 8.63B) | 4 × 96.4 = 386 | 386 | Chunked offload keeps them on host; with exact-size pinning the allocator adds nothing |
| Trainer, 4 ranks: rest of process RSS | 4 × 26 = 104 | 104 | Per-rank RSS 102–108 GiB minus 78.9 GiB of state on a proxy |
| Total | about 792 | about 802 | |

Per-rank sharded parameters: 8.63B = 8.493B experts + 0.137B dense DistOpt shard. The Engram tables
(186–191 GiB) and the optimizer state (386 GiB) are on the host under any offload choice. Proxy node
peaks by optimizer choice: CPU Adam above 854 (killed); chunked + `pin_memory` 799; chunked + exact
pin 684. `pin_memory()` / `torch.empty(pin_memory=True)` round every block up to a power of two
(+41% measured on optimizer state); pinning with `cudaHostRegister` at exact size keeps
`is_pinned()` and async copies (about 200 GB/s). Eight nodes would need about 4 × 191 = 764 GiB of
optimizer state per node, over 1000 with vLLM and the baseline.

## Checkpoint

- **Save host OOM.** The DistOpt `fully_reshardable` format all-gathers the whole DP group's param /
  exp_avg / exp_avg_sq into new FP32 CPU buffers on every rank (actor RSS 47 → 170 GiB per rank, node
  889 GiB, killed by Ray). Fix: save `dp_reshardable` directly from the already offloaded pinned
  state; load follows the format recorded in the checkpoint. Resume needs the same TP/PP/EP; DP can
  change.
- **Frozen Engram tables stay out of the checkpoint**; resume uses the bytes loaded from the release.
- **Proxy round trip.** Resume from step 2: step-3 old logprobs bitwise equal to the uninterrupted
  run (32/32 micro-batches, 110632 tokens); rollout, score, entropy and pg_loss equal, grad norm
  2.57507 vs 2.57483 (non-deterministic dKV). Re-saving right after resume is byte-identical item by
  item to the original: 17525 items / 892 GB with Engram tables, 17509 items / 791 GB without.
- **Size.** Proxy without Engram 791.1e9 B = 56.5e9 trainable parameters × 14 B (BF16 2 + FP32
  master/m/v 12). Full model (40 layers, about 552e9 trainable) without Engram about 7.73e12 B
  (7.0 TiB): about 483 GB per node and 121 GB per rank on 16 nodes.
- **Save-time host memory.** The unmodified synchronous save copies every GPU tensor of the rank to
  host first (`preload_tensors`, non-blocking, into the caching host allocator, not returned), and
  written pages stay in the page cache. The fix (#266) streams D2H one item at a time through one
  exact-size pinned buffer (`MCORE_DIST_CKPT_STREAM_D2H=1`) and every 1 GiB does fsync +
  `POSIX_FADV_DONTNEED` (`MCORE_DIST_CKPT_DROP_CACHE_BYTES=1073741824`), with an unbuffered write
  that rewrites a chunk on EFAULT. Proxy A/B (2 nodes, writing to Lustre, model only, about 53 GB per
  node):

  | | Node used growth | cgroup file growth | Actor RSS |
  |---|---|---|---|
  | Old path | +84 / +94 GiB | +99 / +109 GiB | 109 → 128 GiB, not returned |
  | New path | ≤ +17 / +19 GiB | +4 / +10 GiB | 109.04 → 109.07 GiB |

- **EFAULT on Lustre.** The first full-model save (7.726 TB to Lustre, about 9 minutes) hit 9 EFAULT
  (`Bad address`) errors; each succeeded on the first chunk retry. See the
  [checkpoint pitfalls guide](../../skill/guides/checkpoint-resume-memory/03-pitfalls.md).
- **Node-local checkpoints.** With `CKPT_DIR` on node-local disk only the node of global rank 0
  holds `metadata.json`, `.metadata`, `__0_0.distcp` (the main copy of replicated parameters, 99–137
  GB), `lr_scheduler.pt` and the RNG states, and only the driver node holds `data.pt`. Resuming in a
  fresh Ray cluster needs those files copied to all 16 nodes first (fails with "is not a distributed
  checkpoint" otherwise). A shared filesystem does not need this step.
- `mlite_engine.save_checkpoint` ignores `max_ckpt_to_keep`, so retention must be handled outside the
  trainer.

## Formal runs on the earlier parity branch

Code: the closed #260 branch (mega attention, BF16 head, deterministic dKV off, recompute full, CP2,
gmu 0.65, `full_determinism=False`). Not published code; kept because it validates the recipe and the
full-model numbers above.

| Run | Result |
|---|---|
| G6 short job, first try (gmu 0.70) | Pre-training AIME mean@8 0.751 (with `full_determinism=True`, siblings identical, not comparable); step 1 0, 382 s, actor peak 131.7 GiB; step-2 save 7.726 TB to Lustre in about 9 minutes with 9 EFAULT retries; node used +19–21 GB during save (peak 873.8 GB); then CUDA OOM at the KV wake after the save |
| G6 short job, after the fix | Resume from step 2, steps 3–4 rc=0, both 0; wake after the step-3 save (node-local, 99 s) normal; host peak 874.6 GB; per-GPU `nvidia-smi` peak 179.9k MiB |
| Formal 3-step run | θ0/θ1/θ2 old-logprob and train-forward paths 0 mismatch, max_abs 0 (3,553,540 / 4,072,110 / 3,789,788 tokens), 1024 distinct responses per step; probs_diff 0 on all three steps; every optimizer step changed parameters bitwise on all 64 ranks (476–500 per rank); grad norm 0.046 / 0.046 / 0.043; sync v1–v4 payload identical across 64 ranks (93,414 tensors, about 46.3k changing between syncs); AIME θ2 mean@8 0.738 (pre-training 0.737), best@8 0.899, maj@8 0.809, truncation about 10.4%; θ3 0.734; step time 342 / 333 / 319 s; actor peak 135.5 GiB allocated / 149.1 reserved; host peak 877.0 GB |
| Deliverable re-run (same config, image built with the vLLM fixes, no overlays) | θ0/θ1/θ2 0 mismatch, max_abs 0 (3,619,162 / 4,021,462 / 3,782,656 tokens), 1024 distinct responses per step; probs_diff 0 ×3; sync v1–v4 identical (93,414 tensors); AIME θ2 0.745, θ3 0.747; step time 451 / 395 / 363 s |
| Deliverable resume (fresh Ray cluster) | From step 2: steps 3–4 rc=0; the first sync after resume sent a θ2 payload bitwise equal to the uninterrupted run's θ2 (5,978,496 tensor fingerprints, 0 differ); steps 3/4 0 mismatch (3,872,254 / 3,817,840 tokens); saves 100 / 87 s. After resume 270 vLLM state tensors differed from the uninterrupted run, all Engram peer-exchange runtime buffers (metadata, ids, rows, pointer tables), not weights |

AIME is not expected to improve in two steps; the check is that it does not regress.

Rollout performance on this configuration (mega attention arm, same-node pairs): rollout step B/A
fixed length 1.019 / 1.034 / 1.034 / 1.029 (median 1.031), natural 1.026 / 1.020 / 1.031 / 1.022
(median 1.024), against a budget of 1.10; serving bench (8192 in / 1024 out, greedy, 5 pairs) median
B/A 0.864 / 0.887 / 0.896 / 0.871 at c = 1 / 8 / 32 / 64, against a budget of ≥ 0.80.

## Launcher overrides used by the validated runs

Published `run_deepseek_v41_dapo.sh` defaults vs what the #257-line DAPO runs passed:

| Switch | Published default | Validated runs | Note |
|---|---|---|---|
| `ACTOR_CP` | 1 | 2 | |
| `GRAD_OFFLOAD` | False | True | Without it each GPU keeps about 64.4 GiB more in the rollout phase; gmu 0.65 does not fit |
| `LEARNING_RATE` / weight decay | 1e-5 / 0 | 1e-6 / 0.1 | Weight decay via an appended Hydra override |
| `DSV41_MEGAMOE_MAX_TOKENS` | not read (8192 fixed) | 2048 | Needs #264 |
| `MCORE_DIST_CKPT_STREAM_D2H` / `MCORE_DIST_CKPT_DROP_CACHE_BYTES` | forced 0 / unset | 1 / 1 GiB | Needs #266 |
| `ROLLOUT_FULL_DETERMINISM` | True under BI=1 | False | Distinct siblings; requires #268 to be safe against sampled sentinel ids |
| `VLLM_BI_MEGA_GATE` / `VLLM_BI_MEGA_MHC` / `VLLM_BI_MEGA_ATTN` | 0 / 0 / follows the core | 0 / 0 / 0 | The trainer uses the non-mega gate; exporting 1 for the rollout alone makes rollout and trainer route differently |
| `DSV41_ATTENTION_CORE` / `DSV41_HEAD_DTYPE` | flashmla / float32 | flashmla / float32 | FP32 head on both sides (`hf_overrides={head_dtype: float32}` for vLLM) |
| `VLLM_BI_INDEXER_DEEPSELECT` | 1 (after #271) | not set (run predates #271) | Must match on both sides |
| CUDA graphs | vLLM default capture sizes (after #269) | FULL_AND_PIECEWISE, capture sizes 1–16 | |

Unchanged defaults the runs relied on: gmu 0.65, response 8192, batch 128 × 8, mini-batch 128,
DSpark on, sleep level 2, chunked offload with 2048 MB chunks, `PARAM_OFFLOAD=False`, indexer KV
`mxfp4` with `indexer_sparse_logits=false`, `logprobs_mode=raw_logprobs`, V2 model runner, Engram
`cpu_offload`, `use_thp`, `embedding_across_dp`. If you run with a pre-started Ray cluster, the
switches above must reach the actors through `ray_kwargs.ray_init.runtime_env.env_vars`; the launcher
forwards the paired kernel switches, but any switch you export yourself must be in that list.
