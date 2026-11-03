# Cold start and caches: don't pay the full price on every startup

Cold-start cost comes in five kinds: image import, weight loading, wheel/uv resolution, compile caches (Triton/Inductor/vLLM compile, FlashInfer autotune), and third-party kernel source builds. Each has a form that "looks like it's running but is actually waiting for nothing"; first identify which kind it is before acting.

## 1. Principles

- Node, container and network checks follow your cluster's cold-start preflight procedure directly, including NVMe pre-sharding for repeated MoE/EP startups; do not maintain a second preflight here.
- Put weights, wheel/uv cache, virtual environments and Triton/Inductor/vLLM compile caches on node-local disk (NVMe) and mount them explicitly into the container; reuse them across repeated experiments in the same environment. Verify mounts and writability on each node separately.
- Isolate caches by image/code/hardware compatibility boundaries. A cache directory existing does not mean a cache hit: a new image, new source commit or new GPU architecture each needs a new directory; otherwise hitting stale artifacts is harder to debug than recompiling.
- When a dependency conflict appears, first track down what introduced it. If the FSDP route hits Megatron/mbridge installation requirements, first check the extra/dependency declarations; do not treat an unrelated training backend as required adaptation work for FSDP.
- Cold-start artifacts (prefetched wheels, compiled .so files, weights copied to local disk) are all part of the run identity and go into the [02 runtime manifest](02-runtime-provenance.md#use-one-runtime-manifest-to-answer-what-exactly-is-running-now).

## 2. Compile caches: where they go decides whether they can load

enroot containers bind the host's `/dev/shm` in as `noexec` (`/etc/enroot/mounts.d/10-system.fstab`). `cuda_utils*.so` compiled by Triton/Inductor cannot be loaded from there, reporting

```text
ImportError: .../cuda_utils.cpython-312-aarch64-linux-gnu.so: failed to map segment from shared object
```

The worker dies right at the `torch.compile` stage, which looks like a broken compiler. Model weights in `/dev/shm` are fine (read-only mmap); only caches that need exec fail.

Approach: bind a host local-disk directory into the container for caches, and point all four variables at it:

```bash
-m /local/cache:/cache \
-e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
-e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e FLASHINFER_WORKSPACE_BASE=/cache/fi
```

The container's `/tmp` is not noexec and works, but it is a per-container private tmpfs (see [checkpoint guide 05](../checkpoint-resume-memory/05-cluster-notes.md#1-container-tmp-is-a-per-container-private-tmpfs)), so every container start recompiles, and it consumes job memory; other jobs cannot read it, so "warm the cache with one job first" doesn't work.

**For multiple ranks on the same node, one cache per rank.** When several processes share one cache, they either overwrite each other with concurrent writes or wait on each other. When adding a new kind of cache, first answer: who writes, who reads, does the reader wait for the writer, and what if the writer crashes. Two instances:

- Shared `TRITON_CACHE_DIR`: concurrent cache writes trample each other. One directory per rank.
- The FlashInfer autotune cache is saved only by rank 0: when multi-GPU vLLM on the same node restarts (with an existing cache), ranks wait on each other in the kernel warmup stage, hanging with no progress and no error. Upstream vllm-project/vllm#57635 changed it so each rank saves its own; when the vLLM in the image predates it, mount that PR's `kernel_warmup.py` as an overlay (both arms of Nemotron SWE 256K run ran this way, and every restart afterwards was normal).

## 3. Weight loading: two layers of mmap pitfalls on Lustre

vLLM loads safetensors via mmap by default. On Lustre:

| Model size | Symptom | Cause |
| --- | --- | --- |
| ≲50 GB | Explicit `--safetensors-load-strategy prefetch` works, but the first cold-cache load may still sit at 0% for 20 minutes (4 workers × 8 threads concurrently prefetching the same files fight each other); the second startup hits the cache and is normal | mmap page-fault reads on Lustre are small random reads with no readahead |
| >54 GB | prefetch is ineffective; a 231 GB model stuck at 0/50 shards for 70 minutes | Lustre client page cache cap is about 54 GB (`llite.max_cached_mb`), so warming is wasted |

Approach: copy large models to `/dev/shm` first (tmpfs, aggregate copy about 15 GB/s, 231 GB in about 16 s), load from tmpfs, and `rm -rf` to free it after the run. Plain single-stream `dd` at 2.7 GB/s shows the network is fine; do not go tuning Lustre parameters. `/dev/shm` is bound from the host and shared across containers, but other jobs on the same node use it too, so you must delete after use.

HF download on compute nodes with `snapshot_download(max_workers=16)` is about 1.5 GB/s; `du` lags badly on Lustre and early readings underestimate, so do not change plans based on it. Run downloads as Slurm jobs; `nohup` does not survive session restarts.

## 4. Building third-party kernel packages from source

Some kernel packages have no prebuilt artifacts at all (instance: Tencent hpc-ops, which vLLM `VLLM_ENABLE_HPC_OPS=1` depends on; GitHub releases empty, PyPI 404, not baked into the image either, must `make wheel`). Building in images like `vllm-openai:nightly` requires fixing three things in a row, and each error point is far from the root cause:

| Sticking point | Symptom | Fix |
| --- | --- | --- |
| PyPI throttled by the cluster to about 12 kB/s (GitHub at full speed) | `pip install cmake` hangs for ten minutes at 0% CPU | Prefetch wheels and the official cmake tarball on the login node; in the job, `pip install --no-index --find-links` |
| Container has no `git`; `setup.py` only uses it once to get a short hash as the version | `FileNotFoundError: 'git'` | Write a shim that only answers `rev-parse --short=7 HEAD` and exits 127 for other calls; don't install git |
| Image CUDA libraries come from `nvidia-*-cu13` pip wheels; headers are in `site-packages/nvidia/cu13/include/`, not `/usr/local/cuda/include` | `fatal error: cusparse.h` (every entry.cc fails) | `export CPATH=<site-packages>/nvidia/cu13/include` |

After the fixes, `SM_ARCH=100 make wheel` takes about 50 s wall clock. Once built, two rules:

- **Hardware gates are mutually exclusive**: sm90-only and sm100-only entry points in the same package are never both reached on one machine (hpc-ops: `HPC_ATTN`/`hpc_moe` are sm90 only; iHC/gated MLA are sm100/103 only). GB200 = sm100, so only the latter can be verified.
- **If the gate isn't reached, don't test anything**: check the log for the enabled line (e.g. `HPC iHC enabled ...`); a harder approach is to `collective_rpc` into the worker and count actual instances. Gates may also check conditions such as `hidden_size`, `hc_mult`, `VLLM_BATCH_INVARIANT`. Same-class lesson: at BI=0 the gate only checked whether the .so existed, so the BI kernel ran for nothing; see the [BI kernel appendix](../training-model-support/appendix-bi-kernels.md).

## 5. Image import and startup

enroot importing large images, converted images losing the NGC environment, entrypoints swallowing commands, NCCL switches in GB200 containers — these one-line facts are collected in [checkpoint guide 05 cluster pitfalls](../checkpoint-resume-memory/05-cluster-notes.md#7-other-cluster-pitfalls-one-line-each); details belong to cluster operations notes. Only the two directly related to cold start are recorded here:

- `ENROOT_TEMP_PATH` cannot be on Lustre: the last step `enroot-mksquashovlfs` overlay-mounts all layer directories and then compresses, and overlayfs does not accept a network filesystem as lowerdir; the symptom is running for over an hour without producing a `.sqsh`, then starting `rm --one-file-system -rf .../enroot.XXXX`. `ENROOT_CACHE_PATH` on Lustre is fine. To judge whether the import is progressing, look at `enroot-mksquashovlfs`, not `mksquashfs`.
- `enroot import docker://` reuses the single bearer token obtained at the start for the whole run, so large images (56 GB measured at 20m12s) always get 401 before finishing downloading; retrying doesn't help — take the local tar fallback path.
