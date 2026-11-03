# Pitfalls

Each entry is written as Symptom / Cause / Fix / How to verify / Source. Facts specific to a model, framework or version are marked. Nemotron means Nemotron-H 30B-A3B single node 4×GB200 (2026-10); SHA256s of the source reports are in the [nemotron README](../../../cases/nemotron/README.md#historical-report-sha256).

## Save

### 1. unexpected pos is a masked EFAULT

- **Symptom:** Save reports `unexpected pos 704 vs 598`, often wrapped by Ray as `ActorDiedError`. DS4 full32 and Nemotron both hit this same pair of numbers.
- **Cause:** The torch ZIP writer reports this error in `write_end_of_file`; the real error is an earlier payload `write()` returning `OSError(14, EFAULT)`. 598 is the position of the data record header, 704 is the position after the header is written; the difference of 106 is the header itself, not one write coming up 106 bytes short.
- **Fix:** Wrap item writes so the exception carries the file, offset, item name and original OSError (Megatron dist_ckpt: `_write_item_checked`).
- **How to verify:** With a stream that raises OSError, a unit test reproduces these two numbers exactly.
- **Source:** DS4 `ds4-v9-baseline-dcp-lustre-short-write`; Nemotron [`baseline-ckpt-crash.md`](../../../cases/nemotron/reports/baseline-ckpt-crash.md); internal Megatron-LM fork.

### 2. EFAULT root cause unknown

- **Symptom:** Three projects hit EFAULT (errno 14, `Bad address`) when writing checkpoints or exporting in real jobs:
  - DS4 full32: writing a ZIP checkpoint on Lustre.
  - DS4.1: when publishing weights after restore, the diagnostic `safetensors.save_file` reported `I/O error: Bad address`; rerunning with the same input hit it again on the second publish.
  - Nemotron: Megatron dist_ckpt writing expert weight slices (10–320 MB). Separately, exporting HF safetensors under memory pressure also reported EFAULT (tensors on offload/pinned memory).
- **Relation to host cache release (recorded only here):** Nemotron runs with "gc + host cache release after save" hit EFAULT 4/4 times on save 2–4; switching to pageable staging (host cache was empty anyway, only gc took effect) still hit it; 19 saves without the release all passed. So one can only say "gc + host cache release after save is correlated with EFAULT", not attribute it to releasing the pinned cache itself. Without the release it also occurred twice, both on the first save after resume.
- **Concentrated on a process's first save:** In the BI=0 arm of Nemotron SWE 256K run (Megatron dist_ckpt writing directly to Lustre, 8 nodes 32 ranks, no post-save release), four processes did a "first save after launch or resume", and three of them failed to save: once a fatal EFAULT (see Fix below); twice Ray reported a node heartbeat loss 3–9 s after the save started and the whole job exited (one of those also first hit 3 EFAULTs, all succeeding on item-level retry; that node then went down in Slurm, the other node did not). The single success caught 5 EFAULTs with write-level retry, and the same process's 7 later saves had no EFAULT at all. Add the 2 "first save after resume" cases above: EFAULT concentrates on each process's first save point. The two heartbeat losses only show timing adjacency to the first save; heartbeats have a timeout, so the node may have stopped responding before the save started; the cause is not located, and whether it shares a source with EFAULT is unknown.
- **Cause:** Kernel-side root cause **not located**. Nemotron has ruled out Lustre being full, memory pressure, rotation-script races, and races between the writer and the release; isolated small write tests do not reproduce it (DS4.1's 240 single-GPU write probes also do not reproduce). The elimination process is in the sources.
- **Which layer the safety net goes in:** EFAULT can occur on any `write()`, including the one issued from the buffer at `flush`/`close` for a buffered file, even when data comes from ordinary heap memory. A retry wrapping only one layer of calls (one item, one `torch.save`) misses the other `write()`s. Put the retry on the lowest-level write: open the file unbuffered, write in chunks, and when a chunk hits EFAULT, seek back to the start of that chunk and rewrite.
- **Fixes (all safety nets, not root-cause fixes), by save path:**
  - Megatron dist_ckpt (Nemotron BI=0): two layers. Item level: on EFAULT on a seekable stream, rewrite that item once through a pageable bounce buffer and log it; if both attempts fail, both diagnostics go into the error message and the exception chains to the retry's exception; `MCORE_DIST_CKPT_EFAULT_RETRY=0` disables it. This layer alone is not enough: in one SWE 256K save, retries for two items succeeded but a third rank still failed with a bare `OSError(14)` (it did not go through the item-level error wrapper; inferred to come from flushing the buffer at `close`), the whole save reported `CheckpointException`, and left a half-written directory without a completion marker. Write level: open files with `buffering=0`, wrapped in a stream that writes in 64 MiB chunks and on EFAULT seeks back to the chunk start and retries up to 5 times (overlay on `filesystem_async.py`, not in the image or upstream). This path removed the post-save host cache release.
  - mLite DCP (Nemotron BI=1, EP4 + FSDP2): sources record no EFAULT on this path; it writes to node-local disk first via `MLITE_DCP_LOCAL_STAGE_DIR`.
  - torch ZIP checkpoint written to Lustre (DS4): write to node-local disk first, then publish ([01 §3.3](01-checkpoint-and-resume.md#33-reliable-publishing), PR #196 of a community Megatron-LM fork).
  - safetensors export: DS4.1 first serializes tensors into owned bytes and then writes, with extra memory bounded to one tensor; Nemotron only saves Megatron dist ckpt and does not export HF safetensors under memory pressure.
- **How to verify:** Per the acceptance criterion in [01 §3.2](01-checkpoint-and-resume.md#32-reproduce-and-discriminate). In the Nemotron resume repro, the first save after resume hit EFAULT, the bounce retry succeeded, and things were normal afterwards; with the new image, consecutive saves and the kill→resume chain all passed without triggering a retry. Write-level retry: a unit test injects 3 EFAULTs into the stream (including one raised after a partial write) and the read-back is bitwise identical; the first SWE 256K save triggered 5, all succeeding on the first retry, the 7 later saves all passed, and resuming from one of them loaded successfully; the first save after that resume was interrupted by a node heartbeat loss, so the kill→resume→save-again item has **not yet passed**. DS4.1's bytes path passed in one save/restore sequence. A successful retry shows the data is readable; it does not justify claiming the root cause is fixed; any post-save host cache release must pass this acceptance before being kept.
- **Source:** DS4 `ds4-v9-baseline-dcp-lustre-short-write`; Nemotron [`baseline-ckpt-crash.md`](../../../cases/nemotron/reports/baseline-ckpt-crash.md); DS4.1 restart gate (raw history archive, not included); internal Megatron-LM fork.

### 3. Killed by Ray after writing a checkpoint

- **Symptom:** The checkpoint is fully written, and on the next step Ray kills a worker.
- **Cause:** Freshly written files stay in the page cache and are charged to the job cgroup; Ray looks at cgroup usage.
- **Fix:** `fsync` + `posix_fadvise(DONTNEED)` on the new files of completed directories; apply only to completed directories within the tracker.
- **How to verify:** Look at `memory.current` and Ray's actual criterion, not `free`.
- **Source:** Nemotron.

### 4. Pinned buffers rounded up to powers of 2

- **Symptom:** Nemotron chunked optimizer offload: 88.2 GiB of state per rank occupies 146.9 GiB pinned (about 67% more); before saving the cgroup was already at about 840 GiB (limit 892.8), and Ray killed it during save.
- **Cause:** The torch caching host allocator rounds each pinned block up to a power of 2.
- **Fix:** In `ChunkedOptimizerStateOffloader`, large buffers switch to exact-size private anonymous `mmap` + `cudaHostRegister`; below 1 MiB still uses the cache pool (same approach as vLLM `pin_tensor` on the DS4 side). The unregister finalizer is attached to the object the storage holds (attached to the tensor wrapper, it would unregister early while aliases are alive); the finalizer does not run at process exit (`atexit=False`), since exit itself unmaps, which also avoids a synchronous hang when the device is stuck. glibc exact allocation costs more and cannot replace mmap (see table below). Do not tune pinned rounding with a global environment variable: after Model E lowered the rounding threshold, serving CuMem reported `cudaMemcpy invalid argument` on the second sleep.
- **How to verify:** Split cgroup `memory.stat` into anon/shmem. Quick mode, save every step:

  | Version | anon | shmem | Non-reclaimable | Peak (GiB) |
  | --- | --- | --- | --- | --- |
  | Original | 87 | 692 | 779 | 865 |
  | glibc exact size | 662 | 105 | 767 | 850 |
  | mmap exact size | 437 | 105 | 542 | 685 |

- **Source:** Nemotron [`baseline-ckpt-crash.md`](../../../cases/nemotron/reports/baseline-ckpt-crash.md); internal Megatron-LM fork (`chunked_optimizer_state_offload.py`); Model E capacity notes.

### 5. Save staging stays in the pinned cache pool

- **Symptom:** After the first save, host memory grows by a chunk and does not drop; it stays stable across repeated saves. Nemotron mLite DCP (EP4 + FSDP2) about 196 GiB (stable at about 198 GiB over 6 consecutive saves); Megatron dist_ckpt about 24.6 GiB per rank.
- **Cause:** DCP's pinned write staging stays in the CUDA host cache pool; it is cache not returned, not a leak.
- **Fix:** First reduce resident usage via §4. Calling `torch.accelerator.empty_host_cache()` after save is only optional: on mLite DCP it saves about 196 GiB (host usage between two saves 620 → 420 GiB); on Megatron dist_ckpt see [§2](#2-efault-root-cause-unknown), it has been removed.
- **How to verify:** Save several times in a row and check that shmem stays flat.
- **Source:** Nemotron.

### 6. No checkpoint rotation fills the disk

- **Symptom:** Default 100 steps, saving every 5 steps, accumulates 7–8.6 TB; once the disk is full it crashes during save.
- **Cause:** The mLite engine ignores `max_actor_ckpt_to_keep` (same for DS4). Like DS4, the Nemotron delivered recipe does not rotate; listed as a known limitation.
- **Fix:** Rotate yourself in the launcher: delete only directories within the tracker that are complete and not being written, and persist a deletion record.
- **Source:** Nemotron, DS4.

### 7. Container tmp used for staging

- **Symptom:** Using `/tmp` inside the container for checkpoint staging makes host memory spike.
- **Cause:** `/tmp` inside the container may be tmpfs.
- **Fix:** Point staging to a real node-local disk (Nemotron mLite: `MLITE_DCP_LOCAL_STAGE_DIR`).
- **Source:** Nemotron.

### 8. Repeatedly creating DeviceMesh accumulates threads

- **Symptom:** During DS4 save investigation, thread count grew with each save.
- **Fix:** Cache the mesh. EFAULT persisted after caching; verify the two bugs separately.
- **Source:** DS4.

## Load

### 9. Optimizer load duplicates anonymous memory

- **Symptom:** Node cgroup OOM during DS4 restore.
- **Cause:** Each rank eagerly loads a 107,036,544,596-byte optimizer shard and then copies it into the already allocated FP32 AdamW buffer; four ranks doing this at once, the duplicated anonymous memory blows the cgroup.
- **Fix:** For compatible ZIP checkpoints use `torch.load(..., mmap=True)`, so file tensors are read as file-backed mappings without first copying an anonymous optimizer. It is not zero memory: touched pages, page cache and the target state still use memory. Check compatibility with the checkpoint format, file lifecycle, network filesystem and loader API; do not blindly add the parameter to every `torch.load`.
- **How to verify:** Load model + optimizer + RNG/extra at real scale (DS4 full32 about 264 s, no new cgroup OOM; this does not mean any concurrency is safe).
- **Source:** DS4 `ds4-v9-checkpoint-resume-optimizer-oom`; PR #195 of a community Megatron-LM fork.

### 10. FSDP CPU offload duplicates pinned memory on load

- **Symptom:** DS4.1 full-model initialization hit host OOM inside `load_state_dict` (about 185 million KiB RSS per rank, mostly anonymous memory; CUDA used only 45.78 GiB).
- **Cause:** `CPUOffloadPolicy` defaults to `pin_memory=True`; FSDP2's post-load hook `reset_sharded_param()` runs `.cpu().pin_memory()` on unpinned local shards, creating a second storage, while the state dict still references the original storage. A plain `to_empty(device="cpu")` is already pinned and does not reproduce it; it only triggers when Automodel's custom materializer produces unpinned storage.
- **Fix:** `CPUOffloadPolicy(pin_memory=False)`, everything else unchanged.
- **How to verify:** A two-rank small test compares the number of live storages (True: two, False: one); full-model initialization gets past the original OOM point, and RSS stays roughly flat during `load_state_dict`. This only proves the load peak, not the absence of other peaks.
- **Source:** DS4.1 training history source-29 (raw history archive, not included).

## sleep / wake

### 11. Sleep backup in torch pinned memory: rounded up and never returned

- **Symptom:** host memory jumps at the first level-1 sleep and does not come back after wake; later sleeps or large host transfers add
  more. Another model (TP8): each vLLM worker's RSS was 182–198 GB after the first sleep and 219 GB after the second, and Ray killed it. DS4.1: a
  large level-2 update after an L1 sleep/wake hit host OOM (MaxRSS 892.66 GiB). Model A (TP8, 85.19 GiB of weights per GPU): +578 GiB per
  node held after wake, against 341 GiB of weights.
- **Cause:** `CuMemAllocator.sleep` backs each block with `torch.empty(..., pin_memory=True)`. PyTorch's caching host allocator rounds
  each request up to a power of two (§4) and, after wake drops the tensors, keeps the blocks cached instead of returning them; CUDA
  `empty_cache` does not touch this cache. Blocks of a different size in the next cycle allocate anew.
- **Fix:** back each block with an exact-size page-locked buffer (a private mmap registered with `cudaHostRegister`), unregistered and
  unmapped on wake by default, or kept and reused across cycles when steady-state sleep/wake time matters. Upstream:
  [vllm#64630](https://github.com/vllm-project/vllm/pull/64630) (`--sleep-mode-retain-cpu-backup` for the retained mode, a config option
  next to `sleep_mode_offload_cudagraph`). Model A production ran the retained mode: sleep 7.5 → 1.9 s, wake 2.6 → 1.9 s, no growth.
  Older workarounds: `pin_memory=False` (the TP8 model above; large pageable copies have hung on GB300, §18) and `torch._C._host_emptyCache()` after wake
  (DS4.1; private API, verified only on that image's Torch 2.13).
- **How to verify:** host memory while asleep equals the backup size; it is returned after wake (release mode) or stays flat over three
  cycles (retained mode); fixed-token outputs after wake are bitwise equal to before sleep.
- **Look for the same pattern elsewhere:** every large pinned allocation made through torch is rounded and cached the same way. When
  fixing it, search for siblings: vLLM's XPU sleep allocator had the identical code and was fixed in the same PR; LoRA CPU caches and CPU
  weight offload allocate pinned tensors per adapter or parameter and are candidates.
- **Source:** DS4.1 l1-pinned-cache-host-oom (raw history archive, not included); TP8 model 2026-10-10 probe; Model A 2026-10.

### 12. Actor offload stacks with the serving sleep backup

- **Symptom:** Model E full BF16, four nodes FSDP16/TP16, was killed by Ray at the first `sleep_replicas`, before step 1, with host usage about 864–867 GiB (limit 882.87 GiB).
- **Cause:** Actor CPU offload and the serving sleep CPU backup are resident at the same time; it is a capacity problem.
- **Fix:** Switch to a topology that can hold it (eight nodes FSDP32/TP32 ran all 15 steps), not reducing the actor microbatch.
- **Source:** Model E capacity notes.

## Resume

### 13. First step after restore uses stale quantization scales

- **Symptom:** After DS4 restored from a non-zero step, the first rollout had a train-inference mismatch, and the next step recovered consistency by itself.
- **Cause and fix:** See [01 §5](01-checkpoint-and-resume.md#5-derived-state-after-restore).
- **How to verify:** On the same checkpoint, compare the first export's scale fingerprint and old-logprob before and after the fix.
- **Source:** DS4 `ds4-v9-resume-stale-fp8-source-scales`; PR #197 of a community Megatron-LM fork.

### 14. resume_mode=auto starts from step 0

- **Symptom:** `resume_mode=auto` is set, but training restarts from step 0, with the log saying `latest_checkpointed_iteration.txt` was not found (one Model B job); or it resumes from the correct step but data starts from the beginning: trainer.log has only one line `No dataloader state found ... will start from scratch`, after which prompts from the previous segment repeat, invisible in the metrics (Model D, 2026-10).
- **Cause:** The checkpoint is on node-local disk; the files needed to resume are spread across nodes, and newly allocated nodes or nodes that changed role do not have them:
  - Model and optimizer shards are on the node of each rank (Model B and Model E both use multi-node NVMe);
  - verl's latest-step pointer `latest_checkpointed_iteration.txt` and dataloader state `global_step_N/data.pt` are written only by the driver (TaskRunner) on its own node, which is not necessarily the submit node, nor necessarily rank 0's node;
  - When verl rotates by `max_actor_ckpt_to_keep`, it deletes shards only on rank 0's node; other nodes only add and never delete: a rotated-out step looks complete on other nodes while rank 0's shard is actually gone.
  - Megatron dist checkpoints have the same shape: each node holds only its own ranks' `__<rank>_0.distcp`, and `.metadata` is on rank 0's node, so a multi-node `resume_mode=latest` from node-local disk fails. Model A merged the 32 rank shards into one directory on the shared filesystem (93 GB per step: 32 `.distcp`, `.metadata`, `metadata.json`, 32 adapter shards) and resumed from it with `resume_mode=from_path`.
- **Fix and verification:** Resume with `resume_mode=resume_path` and an explicit step, instead of relying on auto to find the pointer; read the latest step on every node and take the maximum. Before launch check per item 1 of [01 §6](01-checkpoint-and-resume.md#6-how-to-prove-a-resume-is-correct): on every node `ls global_step_N/actor | grep -c rank_` equals that node's rank count, and `data.pt` is copied to the same path on every node. After resume grep for `No dataloader state found`; stop if it appears. For long runs set a large retention count. For how auto-resume scripts read the step, see [06 §4](06-unattended-long-run.md#4-the-daemon-scripts-own-pitfalls).
- **Source:** Model B long training and recovery notes; Model E full-model results; Model D 2026-10 segmented long run.

### 15. Resume overwrites the metrics jsonl

- **Symptom:** After resume, metrics from the previous segment are gone.
- **Cause:** The verl file logger opens the metrics jsonl with `wb`.
- **Fix:** Back it up before each segment starts; when merging, deduplicate by step and keep the later segment.
- **Source:** Nemotron.

## Long-run growth

### 16. mimalloc per-thread heaps

- **Symptom:** Nemotron grows about +0.5–0.6 GiB per step, mainly in 1 GiB-aligned anonymous mappings of 8 `ray::AgentLoopWorker` processes (231 threads each).
- **Cause:** That NV torch's c10 CPU allocator statically links mimalloc. When multiple threads each allocate CPU tensors, each thread's heap keeps its own pages, neither returning them to the system nor letting other threads use them; RSS grows with "number of threads that have allocated tensors × each one's historical peak", reclaimed only when the thread exits. `MIMALLOC_*` environment variables and `malloc_trim` have no effect; libc10 does not export `mi_collect`. CPU-only repro: 200 pool threads saturate at about 12 GiB after about 15 steps; rebuilding the thread pool every step stays flat at 2.2 GiB; single-thread allocation or cross-thread free does not grow. The allocation site in production: Ray serializes return values on the async actor's task-execution thread (not the event-loop thread), where `DataProto.__getstate__` does `contiguous().consolidate()`, allocating a batch-sized CPU tensor on that thread every step.
- **Fix:** a patch in the internal verl fork: `generate_sequences` consolidates the batch and calls `lock_()` on the event-loop thread (tensordict < 0.10's `consolidate()` does not lock); `__getstate__` skips re-consolidation only when the batch is consolidated **and** locked (checking only the consolidated flag would drop modifications after deserialization, found in review); numpy serialization mode is unchanged. The driver `ray::TaskRunner` has the same kind of arena growth; that PR gives no data on it.
- **How to verify:** Confirm the allocator: `grep -c mimalloc /proc/<pid>/maps` or look for `mi_` symbols in `libc10.so`. Unit tests pass under tensordict 0.10.0 and 0.8.3. In an 18-step validation, total RSS of the 8 workers at steps 1/9/18 was 18.9/21.7/20.8 GiB (before fix 16.6/24.7/29.8); `probs_diff` was 0 on all 18 steps, step time 750 s (before fix 755 s).
- **Source:** Nemotron [`host-mem-growth.md`](../../../cases/nemotron/reports/host-mem-growth.md) §2, §3, §7; internal verl fork.

### 17. dynamo hits the recompile limit

- **Symptom:** Trainer grows about 40–60 MiB per step (about 10% of total growth), with many recompile warnings.
- **Cause:** In mLite Nemotron-H `dense.py`, the RMSNorm VJP uses `torch.compile(fullgraph=True, dynamic=False)`, and dynamic batch size gives each micro-batch a different shape. After the first 8 shapes are compiled, every call enters dynamo, fails the guard, raises `FailOnRecompileLimitHit` and falls back to eager (about 4700 times per rank per step); `guard_failures` is unbounded, each leaving about 2.2 KiB.
- **Fix:** a patch in the internal Megatron-LM fork: remember compiled shapes per function + shape/dtype, and after the limit send uncompiled shapes straight to eager; before skipping, check the number of dynamo cache entries on the code object, so new shapes recompile after `torch._dynamo.reset`. Compiled shapes still use the compiled version, bitwise unchanged numerically. Do not switch to `dynamic=True` (it changes the reduction configuration and requires re-verifying determinism), and do not periodically call `torch._dynamo.reset()` (it clears all compile caches). That PR is reviewed separately and is not in the delivered image.
- **How to verify:** compiled + fallback grows linearly by 2.23 KiB per call in the repro, eager by 0; the official `test_nemotron_h_training_vjp.py` harness overall=PASS.
- **Source:** Nemotron `host-mem-growth.md` §4; final delivery README.

## Host–device copies

### 18. Large pageable copies hang in libcuda

- **Symptom:** On the first adapter sync, 2/16 ranks sat in `cudaMemcpyDtoHAsync` for over 30 minutes, GPU utilization about 3%, py-spy stack unchanged, no Xid/ECC; the other ranks waited at the next collective. A 26 GB H2D also hung after the first 53 layers succeeded (another model, 2026-10, GB300, driver 610.43).
- **Cause:** Not located. Only occurs when copying between large pageable host memory and GPU; pinned copies (including sleep backups of over 580 GB each) have never hung.
- **Fix:** Route large copies through pinned buffers, or keep them on GPU and stream; treat newly added pageable paths (e.g. `.cpu()` before `torch.save`) as a risk.
- **How to verify:** After switching the same path to pinned, multiple full-scale runs no longer hang.
- **Source:** another model's 2026-10 full-model LoRA training.

### 19. File-backed offload of frozen weights

Frozen base weights that leave the GPU between phases (SkyRL Megatron offloads them to a file and maps it back) trade GPU memory for
host page cache and disk reads. Each of the three properties of a file mapping turned into a cost in Model A (Megatron, EP32, 8×4 GB200,
2026-10):

| Property | What happened | Fix |
|---|---|---|
| Private mapping (`torch.from_file(shared=False)`) | Copy-on-write: restoring to the GPU touched every page and turned it into anonymous memory (+16 GiB per 16 GiB restored), so the offload saved no host memory and the first smoke run hit host OOM | Shared mapping: restored pages stay reclaimable page cache |
| Lazy mapping | The restore page-faulted during the H2D copy at about 2.1 GB/s: 10–53 s per step | `MADV_POPULATE_READ` before the copy: about 1.5 s |
| Reclaimable page cache | Under host memory pressure the kernel dropped part of it; that step's restore re-read from disk and took 395 s instead of 3 s | Pinning (`mlock`) is counted by Ray's memory metric (about +228 GiB per node) and the GPU had no room, so the remedy was to shrink the cache: data-parallel replicas of the same dense frozen weights share one file per node (about 135 GiB less page cache per node); experts stay private |

The offload directory also needs disk: about 205–220 GB per node for Model A. Check every node's free space before using it; on the GB300
nodes the container's `/data` sits on a host `/tmp` with about 150 GB, so the directory has to be mounted per node.

- **How to verify:** anonymous memory does not grow by the restored size; restore time per rank is logged every step (a spike means a
  re-read); restored tensors are bitwise equal to the originals; with sharing, every rank logs the shared size and no "differs from this
  rank" fallback.
- **Source:** Model A Megatron port 2026-10 (SkyRL commits for shared mapping, populate-before-restore and node-shared offload files).
