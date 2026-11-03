# BI=0 baseline crashes after checkpoint save (GB200 cluster, 2026-10-05)

Commit and image labels below stand for internal commits and images; SHAs are omitted. Megatron: **P1** (exact-size pinned buffer) and **P2** (checked write), on top of the previous delivery head. verl: **V-base** (the clean one-recipe branch head), **V-mp1**/**V-mp2** (the withdrawn monkeypatch commits), **V-ref** (the reference commit for DRY_RUN comparisons). Scratch investigation commits: **S-glibc** (glibc exact buffer) and **S-mmap** (mmap exact buffer).

## Conclusions

- **Both failures share one precondition: host memory is already right at the limit before saving.**
  - Before each BI=0 save, the cgroup already holds about 840 GiB, against a limit of 892.8 GiB.
  - The bulk is the pinned buffers of chunked optimizer offload. torch's caching host allocator rounds each parameter's pinned block up to a power of 2: each rank's actual state is 88.2 GiB, but it occupies 146.9 GiB pinned.
  - On top of that, the D2H staging during save is cached in the pinned pool and not returned, about 24.6 GiB per rank.
- **Failure 1: Ray OOM kill.** All 11 failures in the morning were this one.
  - The failing legs did not set `RAY_memory_usage_threshold`; the log says verbatim `exceeded threshold of 910669774848B (95.0% ...)`.
  - Setting it to 0.985 is not enough either: one job was still killed at 98.5% while saving at step 25.
- **Failure 2: "unexpected pos 704 vs 598".**
  - This is torch's zip writer masking the real error at `write_end_of_file`. The real error is the payload's `write()` returning **OSError(14, EFAULT)**.
  - All samples occur on the model dist_ckpt, an expert weight slice of 10–320 MB, unrelated to memory pressure: it still occurs in reproductions where the cgroup holds only 600–830 GiB.
  - It is **not** caused by a race with the evict/rotate script: fadvise only applies to completed directories before `latest_checkpointed_iteration`; the reproduction run is in quick mode, where the rotate script does not take effect at all, and it still errs.
  - It is **also not** Lustre ENOSPC: OST usage is 86–93%, not full.
  - The kernel-side root cause of EFAULT has not been located.
- **Fix (moved from verl into the Megatron-LM source, no more monkeypatch):**
  - Two new commits on the delivery branch of the internal Megatron-LM fork, fast-forward pushed to P2:
    - P1: `ChunkedOptimizerStateOffloader._new_cpu_buffer` switches to exact-size pinned buffers (private anonymous mmap + `cudaHostRegister`, unregistered when the buffer is freed).
    - P2: when `FileSystemWriterAsync` fails to write an item, it raises an error carrying the file, offset, item and the original OSError, instead of only "unexpected pos"; on EFAULT it rewrites once via a pageable bounce buffer, as a logged fallback (can be disabled with `MCORE_DIST_CKPT_EFAULT_RETRY=0`).
  - The verl one-recipe branch dropped V-mp1 and V-mp2, back to V-base, no longer containing any patch.
  - The new image `<image>` (one-recipe image with Megatron P2 and verl V-base) has passed verification.
  - **Switching the two lr-bi0 chains was blocked by the permission classifier and not executed; a human needs to do it (see the end).**
## Evidence

### Memory composition (cgroup memory.stat; pinned memory under cudaHostAlloc is counted as shmem)

quick mode, one save per step:

| run | anon | shmem (pinned) | non-reclaimable total | cgroup peak |
|---|---|---|---|---|
| Control: frozen image | 87 | 692 | 779 | 865 |
| glibc exact buffer (S-glibc) | 662 | 105 | 767 | 850 |
| **mmap exact buffer (S-mmap)** | **437** | **105** | **542** | **685** |

- The glibc version saved almost no memory, because the glibc allocator itself has about 1.55x overhead. In the unit test, 9.29 GiB of 19 MiB buffers actually occupied 14.4 GiB anon; with mmap + register it is exactly 9.31 GiB.
- The control's shmem 692 = 4 × 146.9 of optimizer pinned + 98 of save staging.
- On disk the optimizer file is 94.7 GB = 88.2 GiB per rank, consistent with `offloaded optimizer state 88.2 GiB` in the new log.

### Forensics of EFAULT

- A wrapper was put around mcore's `_write_item` to extract the masked exception. For example:
  `file=.../global_step_2/actor/model/dist_ckpt/__3_0.distcp item_start=13654982515 item=decoder.layers.45.mlp.experts.experts.linear_fc2.weight[113] masked=OSError(14, 'Bad address')`
- With a stream that raises OSError, `unexpected pos 704 vs 598` can be reproduced exactly in a unit test: 598 is the position of the data record header, and 704 is the position after the record header is written.
- EFAULT occurrence per config:

| Config | Result |
|---|---|
| With post-save host cache release (gc + `_host_emptyCache`) | 4/4 EFAULT at the 2nd–4th save |
| Same, but with pageable staging (cache already empty, only gc takes effect) | Still EFAULT |
| Without release | Control 4/4 pass, N 5/5, S-glibc run 5/5, S-mmap run 5/5 |
| Formal leg (old image, no release) | 2 EFAULTs, both at the first save after resume |
| Resume reproduction (S-glibc, job A → job B) | EFAULT at the first save after resume, **bounce retry succeeded**, later saves normal, exit 0 |

- None of the standalone tests reproduced EFAULT:
  - Lustre 2.17: `hybrid_io=1`, `hybrid_io_write_threshold_bytes=2097152`, `unaligned_dio=1`;
  - cudaHostAlloc, pageable and registered memory, with both O_DIRECT and buffered writes, all normal;
  - A 4-rank × 6-round staging + torch.save simulation (items 320 MB–1.8 GB): 0 failures;
  - A small 48G test filling the cgroup with pinned memory: 0 failures.
- So EFAULT only appears in the real job environment; which exact condition triggers it is still undetermined.
- The save path is synchronous: `async_save=False`, going through `execute_sync`, with 1 thread, the main thread writing directly; release happens after the barrier. So there is no UAF race between the writer and release.

### Fix verification

| Verification | Result |
|---|---|
| S-mmap quick, 5 consecutive saves (one node) | All succeeded, exit 0, peak 685 GiB |
| S-mmap resume chain (one job saves 3 times → the next job resumes from global_step_3, then saves 4, 5, 6; another node) | All succeeded, exit 0, no EFAULT |
| New image BI=0 quick, SAVE_FREQ=1 | 3 saves, exit 0; probs_diff_mean about 0.01, entropy about 0.45, consistent with the old image's smoke2 |
| New image BI=1 quick | 3 steps `rollout_probs_diff_max = 0` |
| BI=1 DRY_RUN | Byte-identical to V-ref (aligned and quick) |
| New image mcore | The two patched files are identical to the verification image |

## Changes

### Current delivery (supersedes the "first version" below)

- **Megatron-LM** (the Megatron ds4form workspace, pushed to the fork's delivery branch, previous head..P2):
  - P1 `megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py`:
    - Adds `_exact_size_pinned_empty`; buffers smaller than 1 MiB still use the caching allocator.
    - Same approach as `simple_kv_offload/cuda_mem_ops.pin_tensor` in vLLM on the DS4 side (`cudaHostRegister` to bypass power-of-2 rounding).
    - The upstream NVIDIA dev branch's offloader is still `pin_memory=True`; there is no existing approach to reuse directly.
  - P2 `megatron/core/dist_checkpointing/strategies/filesystem_async.py`: adds `_write_item_checked`.
- **Unit tests**: all added to existing test files, run in the new image against the installed mcore.

| Test | Result |
|---|---|
| New `test_canonical_cpu_buffers_are_pinned_at_their_exact_size` in `tests/unit_tests/test_optimizer_state_offloading.py`; the whole file run with 4 ranks | 53 passed |
| 3 new `TestWriteItemChecked` cases in `tests/unit_tests/dist_checkpointing/test_async_save.py`: reports the original error, reads back consistently after an EFAULT rewrite, can be disabled; together with the Nvrx cases | 9 passed |
| The existing `test_async_is_equivalent_to_sync` in the same file | Already fails on 4 GPUs (requires world_size divisible by 8); fails on the baseline too |

- **verl**: the one-recipe branch has been reset back to V-base (the two commits were never pushed).
- **Image** `<image>` (one-recipe image with Megatron P2 and verl V-base):
  - sha256 ca7c040f…; install+commit on one node, in one job.
  - Build scripts `one-recipe-image/build-meg.sbatch` + `inside-meg.sh`: on top of the original scripts, reinstall megatron-core from the Megatron-LM fork at P2 (`--no-deps`).
  - The manifest has `megatron_revision` = P2; the package set is the same as the previous image.
  - The previous build log is kept alongside.

### Verification of the new image (same method as the previous round)

| Verification | Result |
|---|---|
| BI=0 quick, SAVE_FREQ=1, 5 consecutive saves (`ckmeg-bi0-5save`, one node) | All succeeded, exit 0, peak 682 GiB; probs_diff_mean about 0.0095, consistent with the old image |
| kill→resume (`ckmeg-bi0-resume`, another node) | The first job was scancelled after global_step_3 was on disk; the second job loaded the optimizer from global_step_3 and resumed, saved 4, 5, 6, exit 0, peak 782 GiB |
| BI=1 quick (`ckmeg-bi1-quick`, a third node) | 3 steps `rollout_probs_diff_max = 0`, exit 0 |
| BI=1 DRY_RUN (verl V-base vs V-ref, aligned and quick) | Byte-identical |

- None of these 4 runs triggered the EFAULT retry.

### First version (withdrawn)

- verl V-mp1, V-mp2, and an image built from them (Megatron previous head, verl V-mp2): same content, but implemented as a monkeypatch.
- At that time c3–c16 of lr-bi0-s42 and c3–c16 of lr-bi0-bf16-s42 were switched to this image (14 queued legs each). The first leg of the bf16 chain was already running.
- The scratch investigation branch (S-glibc…S-mmap) and `baseline-r3/run-baseline-ckfix.sbatch` are kept as evidence.

### Not done yet: switch the two lr-bi0 chains to the P2 image

- My switch commands (cancel the pending legs, then resubmit with the new image) were blocked by the permission classifier, **not executed, and the queue was not changed at all**.
- A human needs to do it. Do it once each for lr-bi0-s42 and lr-bi0-bf16-s42 (each pinned to its own node):
  1. Note the leg currently running (at the time, c2 and c3 respectively).
  2. `scancel` all PENDING legs of that chain.
  3. From "running leg number + 1" up to c16, submit one by one:
     `sbatch -J <tag>-c<N> -w <node> --dependency=afterany:<previous> -o long-run/<tag>-c<N>-%j.slurm.out long-run/run-long4.sbatch <new-image> <tag> long-run/<tag>.env --hardware gb200 --mode aligned`
  - lr-bi1-s42 is left alone.

## Leftovers

- The trigger condition of EFAULT has not been located; for now there is only the retry fallback.
  - Observed pattern: the first save after resume, and saves after gc + host emptyCache, are the most likely to hit it.
  - The successful retry shows the data itself is readable; the problem lies in how the kernel write path handles this user buffer.
  - If the retry fails, the log will contain `checkpoint write failed ... masked=...`.
- The save peak under the formal aligned workload has not been measured yet; check the new chain's first save point (a multiple of step 5). The non-reclaimable part is expected to be about 550 GiB.
- Data left by my jobs:
  - `runs/br3-ckfix*`, `runs/ckmeg-*`, `runs/smoke3-*`: checkpoints deleted, only logs and samples remain.
  - `baseline-r3/ckptcrash/`: only the unit test scripts remain, `out/` deleted.
