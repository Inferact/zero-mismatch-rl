# Source of the roughly +0.5 GiB per step host memory growth in long runs (2026-10-05)

Data sources:
- Run 1 (lr-fsdp2-a, two job segments, one node): hostmem/cgroupmem;
- lr-bi1-s42 (one job on another node, a 100-step run by another agent): per-minute procmem (smaps_rollup PSS) and memstat, steps 1–12;
- Read-only captures of `/proc/<pid>/smaps` and `task/*/comm` for that job's TaskRunner / AgentLoopWorker (`srun --overlap`);
- Two single-process reproductions (one standalone job on a third node; a small CPU-only process overlapped onto the lr-bi1-s42 job, ≤24G).

## Conclusion

| Question | Answer |
|---|---|
| Which metric grows | MemTotal−MemAvailable and cgroup anon. cgroup `memory.current` is masked by the page cache. shmem is flat (3.3→3.6 GiB after a save); file is unrelated to the growth |
| Which processes | **8 `ray::AgentLoopWorker`**: each +0.04–0.06 GiB/step, together about +0.35–0.4 GiB/step, monotonically rising; **`ray::TaskRunner`** (driver): floor 10.3→14.5 GiB (steps 2–9, about 0.6 GiB/step), but before validation at step 10 it dropped back to 1.1 GiB once, so it is not grow-only; gcs_server +0.015 GiB/step |
| What does not grow | WorkerDict (4 trainer ranks, per-step peak 71.7→70.0 GiB, slightly down), vLLM Worker/EngineCore/HttpServer, RewardLoopWorker, raylet |
| Which category | anon. Specifically the **mimalloc arenas statically linked into libc10** (1 GiB aligned anonymous mappings), i.e. torch CPU tensor memory. The Python heap (`[heap]`, small anonymous blocks) is essentially unchanged |
| Mechanism (confirmed by reproduction; the production call site is not confirmed) | The c10 CPU allocator in this NV torch build is mimalloc. When several threads each allocate CPU tensors, each thread's heap keeps its own pages, does not return them to the system, and does not let other threads reuse them. RSS grows with "number of threads that ever allocated tensors × each thread's historical peak". None of the MIMALLOC_* environment variables help; memory can be reclaimed once the thread exits |
| Another small source | The trainer's RMSNorm VJP uses `torch.compile(dynamic=False)` and hits recompile_limit about 4700 times per rank per step; the dynamo global tables keep 2.2 KiB each time, about 40–60 MiB/step in total, roughly 10%. It is offset by a decrease in another part of WorkerDict, so it is not visible in procmem |

## 1. Old logs (run 1)

- hostmem.txt split by step (second job segment, excluding save steps). Each step has two plateaus:
  - The high plateau, about 3 minutes, is gen + old_log_prob. The optimizer is offloaded to CPU at this point, WorkerDict about 72 GiB per rank;
  - The low plateau is update_actor, with the optimizer back on GPU.
- Median of the high plateau: step 31 432 GiB → step 61 450 → step 99 474 GiB, about 0.6 GiB/step. The low plateau rises as well.
- The per-step peak of cgroup `memory.current` has no trend: it includes the page cache and reaches 893 GiB at save steps. The old logs have no anon/file/shmem split and no per-process data, so the category or the process cannot be determined from them.
- After resume the high plateau went from about 455 back to about 433 GiB, which shows the growth accumulates over the process lifetime.

## 2. lr-bi1-s42 split by process

Minimum within each step (a floor not affected by the phase), in GiB:

| Step | TaskRunner | 8×AgentLoopWorker | vLLM Worker×4 | EngineCore×4 | Other (gcs/raylet/…) | Node anon p90 | shmem |
|---|---|---|---|---|---|---|---|
| 3 | 10.70 | 14.20 | 14.91 | 4.02 | 5.01 | 347.9 | 8.7 |
| 5 (save) | 12.19 | 14.30 | 14.91 | 4.02 | 5.03 | 350.5 | 3.3 |
| 7 | 13.27 | 14.55 | 14.91 | 4.01 | 5.06 | 351.4 | 3.4 |
| 9 | 14.53 | 16.60 | 14.91 | 4.02 | 5.09 | 355.5 | 3.5 |
| 11 | 10.74 | 16.90 | 14.91 | 4.02 | 5.32 | 352.8 | 3.5 |

- Note: the TaskRunner column includes 8 idle Ray workers with the same name, 0.5 GiB each. Values of the main process itself: 10.4–11.3 at step 9, dropped back to 1.12 before validation at step 10, 6.6–7.9 at step 11.
- Per-step peak of the 4 WorkerDicts: 71.7 → 70.0 GiB/rank, slightly down.
- smaps breakdown:
  - TaskRunner: the growth is in 15 mimalloc arenas aligned to 1 GiB, 9.7–11.7 GiB in total; newly opened arenas expanded from 2 GiB to 4 GiB. `[heap]` stays at 0.53 GiB, plasma 1.21 GiB.
  - One AgentLoopWorker: 17:49→18:20 (about 2.3 steps) mimalloc arenas +108 MiB, plasma mappings +108 MiB (the PSS share of shared pages; node shmem unchanged), Python small blocks +3 MiB.
- An AgentLoopWorker has 231 threads, 179 of them named `ray::IDLE`; the count does not change with steps. TaskRunner has 70 threads.

## 3. Reproduction: thread-level retention in mimalloc

`memdiag/mi_threads.py`, same image, CPU only. Each "step" allocates 1024 int64 tensors of random length (1000–14000 tokens × 8), `cat`s them into a batch and then frees everything; the live set does not grow:

| Allocation pattern | RSS after 15 steps |
|---|---|
| Allocated on the main thread | 1.2 GiB, flat |
| Allocated on 1 pool thread, freed on the main thread | 1.2 GiB, flat (cross-thread free itself is not the problem) |
| 16 pool threads | 2.1 GiB at 10 steps |
| **200 pool threads** | **5.8 → 8.8 → 12.4 GiB, saturating at about 12 GiB after about 15 steps** |
| 200 threads, new thread pool created and shut down every step | **2.2–2.3 GiB, flat** |
| 200 threads + `MIMALLOC_PURGE_DELAY=0` / `ABANDONED_PAGE_PURGE=1` / `TARGET_SEGMENTS_PER_THREAD=2` / `GENERIC_COLLECT=100` / eager commit off | 6.9–8.6 GiB (10 steps), no effect |

- libc10 does not export `mi_collect` or similar symbols, so memory cannot be reclaimed actively from Python.
- This is consistent with the earlier report that "mimalloc does not return memory to the system after free". A more accurate statement: the memory stays in the heap of the thread that allocated it and can only be reclaimed after that thread exits.

## 4. dynamo hitting recompile_limit (about 10%)

- Location: `megatron/lite/model/nemotron_h/vllm/primitive/dense.py:134-137`, `_VJP_COMPILE = dict(fullgraph=True, dynamic=False, ...)`.
  - `use_dynamic_bsz` makes every micro-batch a different shape. The first 8 shapes are compiled once each; after that every call enters dynamo, fails the guard, raises `FailOnRecompileLimitHit` and falls back to eager.
- Count: in the run log the counter maxima per rank are about 255k for frame2, about 100k for frame3 and about 5k for frame4, i.e. about 4700 per rank per step.
- Reproduction (the standalone job): compiled plus fallback is 2.23 KiB per call, growing linearly; eager is 0. Accumulation points reported by tracemalloc:
  - `torch/_dynamo/guards.py:5329` `guard_failures[code].append(reason_str)` (unbounded);
  - `CapturedTraceback` in `metrics_context.py:177`;
  - the timing list in `utils.py:838`.

## 5. Proposed fixes (delivery code not changed)

1. **AgentLoopWorker / TaskRunner (the main part)**:
   - First confirm which threads allocate CPU tensors. For the next run use `memdiag/py/memdiag.py` (mounted via .pth, active only when `MEMDIAG_DIR` is set) plus py-spy sampling to see the thread stacks.
   - Once confirmed, there are two options:
     - Construct CPU tensors only on a fixed small set of threads. For example, if AgentLoopWorker's `_agent_loop_postprocess` / `_postprocess` or anything inside `run_in_executor` constructs tensors, construct them on the event-loop thread instead;
     - Or recycle these thread pools at the end of every step (`shutdown` and recreate); in the reproduction this dropped from 12 GiB to 2.2 GiB and stayed flat.
   - Environment variables do not help. Replacing the c10 allocator requires a different torch build and is not recommended.
   - Risk assessment:
     - In the reproduction 200 threads saturate after about 15 steps; the production slope shows no sign of saturation within 100 steps (the old run was still rising from step 26 to 100);
     - At the old run's slope, 100 steps is about +45 GiB, still far from the cgroup 893 GiB / Ray 848 GiB limits;
     - Run 1's save peak was 682 GiB, about 165 GiB of headroom. At 0.6 GiB/step, a save might reach Ray's kill line after about 270 steps.
2. **dense.py recompile**:
   - In `compiled_vjp_or_eager`, remember the compiled shapes keyed by "function + tensor shape/dtype". After the first `FailOnRecompileLimitHit`, shapes that were never compiled go straight to eager and no longer enter dynamo.
   - Numerics stay bitwise identical to now (the 8 compiled shapes still use the compiled version), while removing about 19k useless dynamo entries per step and the flood of warnings.
   - Changing to `dynamic=True` is not recommended (it changes the reduction configuration and determinism would need to be re-validated), and neither is a periodic `torch._dynamo.reset()` (it clears all compile caches).

## 6. Process notes

- I submitted one 20-step diagnostic run (one job, `memdiag/run-memdiag.sbatch` + `memdiag.env`, no other job's files were changed). After the coordinator switched to the lr-bi1-s42 samples, I cancelled it during initialization; it produced no step data.
- On the lr-bi1-s42 job I only ran overlap steps that read `/proc`, plus 3 small CPU-only reproductions (each ≤24G memory, 8 cores, about 5 minutes). No attach, no signals.
- All files are in `$WORKDIR/memdiag/`:
  - `recompile_leak.py`, `recleak-*.txt`: recompile reproduction;
  - `mi_threads.py`: mimalloc thread reproduction;
  - `tr/smaps-*.txt`, `tr/alw-smaps-*.txt`: smaps snapshots of TaskRunner and AgentLoopWorker;
  - `py/memdiag.py`, `zz_memdiag.pth`, `hostsample.sh`: reusable samplers.

## 7. Fix results (2026-10-06)

- **AgentLoopWorker**: a fix PR on a verl fork (based on upstream main 8718ca30). The batch is consolidated on the event-loop thread first and explicitly locked after consolidation (tensordict 0.8's consolidate does not lock), and serialization no longer re-consolidates a batch that is "already consolidated and locked". All 51 tests pass under tensordict 0.10.0, and the 3 new tests also pass under 0.8.3 (two test jobs).
  - 18-step validation (one job, run `verlfix-bi1`): total RSS of the 8 workers at steps 1/9/18 was 18.9/21.7/20.8 GiB; before the fix the same configuration gave 16.6/24.7/29.8 GiB. Probability diff 0 for all 18 steps, 750 s per step (755 s before the fix).
  - Unit tests (one job): 51 pass with the fix; without the fix the two targeted tests fail. Review found that "modifying after deserialization and then re-serializing drops keys"; this was changed to also check the lock state, and a test was added (it fails on the old version; one job).
- **dense.py recompile**: a fix PR on a community Megatron-LM fork.
