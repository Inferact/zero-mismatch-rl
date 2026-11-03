# EP4 + CuTe-DSL routed experts (GB200 cluster, 2026-10-04)

Conclusion: EP4 (DeepEP normal mode) + CuTe-DSL training PASSes both the quick gate and the formal-length round: the training and inference forwards are bitwise identical, and the optimizer receipt passes. At formal length, FSDP2's update_actor takes about half the time of PP4 CuTe-DSL, and is about 6% faster than EP4 Humming on another cluster. Data is in `results/ep4-cutedsl/`.

## Code (committed locally only, not pushed)

- A Megatron fix branch: one commit on top of the predecessor's commit.
  - The predecessor's commit only did a cherry-pick: CuTe-DSL only took the EP1 emulation path, i.e. one launch covering all 128 experts. The EP path in `ep.py` only recognized the Humming API (`routes`, `rank_partial`, `expert_map`), and `CuteDslRoutedExperts` required holding all experts, so it could not run under EP4.
  - `CuteDslRoutedExperts` can now hold only the `num_local` experts starting at `offset`. `moe_sort` uses `local_expert_offset`, the GEMM uses `num_local_experts=32`, and the tactic is the fixed BI tactic, exactly the same as one EP rank of the serving-side `CuteDslFusedMoEW4A16Runner`.
  - Added `rank_partial(save=)`: the rank's partial sum is this launch's own `moe_unpermute`. The forward only saves fc1, visible and activated for this rank's own slots.
  - `ep.py` branches by backend. Under CuTe-DSL, the backward expands activated and hands it to `routed_vjp`. Partial sums are sent back to the source rank via DeepEP, where the source rank reduces them with `reduce_ep4_parts` (FlashInfer one-sided order); this step is unchanged.
  - Humming can still be selected. The determinism-related commits (the optional deterministic-backward branches) are not included.
- A verl recipe branch:
  - Based on an earlier recipe commit, with four commits cherry-picked. The last two had conflicts, which were resolved; the first two are dependencies of the last two, so they were brought along.
  - Added one commit: when `ACTOR_OPTIMIZER=fsdp2`, the rollout `gpu_memory_utilization` default changes to 0.65; see the pitfalls below for why.
- Diagnostic branches, not delivered:
  - Megatron: the fix branch plus the 12 exactness-observer commits from the ep4-diag work.
  - verl: the recipe branch plus one commit adding `VERL_REQUIRE_BITWISE_LOGPROBS`.
  - The observers used for measurement are an exception and do not count as monkeypatches.

## Images

- DeepEP: `<image>` (the p6 package image with DeepEP added).
  - The base is the p6 package image, plus `nvidia-nvshmem-cu13==3.4.5` and a community DeepEP fork at a pinned commit. Build parameters `DEEPEP_NUM_MAX_NVL_PEERS=4`, `TORCH_CUDA_ARCH_LIST=10.0a;10.3a`, consistent with the DeepEP image Dockerfile used on the other cluster.
  - Built with uv `--no-build-isolation`, script `$WORKDIR/scripts/make-deepep-sqsh.sh`, about 15 minutes on one node.
  - Source cloned on the login node (on Lustre, `$WORKDIR/src/DeepEP-<commit>`). nvshmem downloads normally from PyPI on compute nodes.
  - The version shows as `1.2.1` without a commit suffix because `.git` was removed at build time; the source SHA has been verified.
- Transport: normal mode runs on a single node with 4 GPUs and `num_rdma_bytes=0`, without initializing NVSHMEM. Inside the pyxis container, the 4-GPU DeepEP unit tests and full-model training both work with no extra settings.
- dev image: `<image>` (dev image with the diagnostic Megatron and verl branches), produced from the DeepEP image plus make-dev-sqsh.

## Unit tests (in the image, one node, `-k "cutedsl or ep4"`): 16 passed

- New: the partial sum computed by a rank deployment holding 32 experts is bitwise identical to the serving runner; the saved fc1/visible/activated are bitwise identical to the EP1 deployment on this rank's slots. M takes 1/7/64/65/513/4096.
- The 4-GPU DeepEP EP4 test was run for both the Humming and CuTe-DSL backends, with results:
  - the forward is bitwise identical to the single-rank EP4 serving reduction;
  - dx and d_routes are bitwise identical to EP1;
  - repeated backward runs give identical results;
  - θ0 and two requantized snapshots were all checked.
- Log: `unit-<commit>.log`.

## Quick gate (dev profile, 3 RL steps, FSDP2, one node, one job `ep4c-fast2`): PASS

- θ0/θ2/θ4 old-logprob vs rollout: 0 mismatches. Micro-batch counts 16/20/16, token counts 113.9k/124.0k/114.3k; the raw bytes gate is equal at every step.
- Grad-enabled training forward: 0 mismatches against rollout, 0 mismatches against old-logprob, micro-batch counts 8/12/8.
- optimizer: 6 steps, the grad norm sequences of the 4 ranks are bitwise identical, and all master values changed.
- `fast2-receipt.json`.

dist_opt had headroom, so it was also run (one job `ep4c-fastd1`): PASS, θ0/θ2/θ4 all 0 mismatches, training forward 0/0, grad norms identical across the 4 ranks (`fastd1-distopt-receipt.json`).

| dev profile (step 1/2/3) | update_actor (s) | GPU peak alloc / reserved | Node host peak |
|---|---|---|---|
| EP4 CuTe-DSL FSDP2 | 88.8 / 24.6 / 19.7 | 144.0 / 150.8 GiB | 444 GiB |
| EP4 CuTe-DSL dist_opt | 128.7 / 51.3 / 56.3 | 80.5 / 100.3 GiB | 757 GiB |
| PP4 CuTe-DSL dist_opt (GB200 cluster run `p6-fast1`, original p6 image) | 127.7 / 31.7 / 26.6 | 76.8 / 87.6 GiB | about 740 GiB (cpu_memory_used) |

## Formal length (recipe default workload, `ACCEPTANCE_STEPS=3`, `SAVE_FREQ=-1` no checkpoint, `VERL_REQUIRE_BITWISE_LOGPROBS=1`, FSDP2, one node, one job `ep4c-formal2`): PASS, exit 0

- θ0/θ4/θ8 old-logprob vs rollout: 0 mismatches. Micro-batch counts 544/532/588, token counts 7.64M/7.57M/8.27M; the raw bytes gate is equal at every step.
- Training forward: 0/0 (44/48/56 micro-batches).
- optimizer: 12 steps, grad norms of the 4 ranks bitwise identical, all master values changed.
- `formal2-receipt.json`.

| Formal length, step 1/2/3 | update_actor (s) | Per million response tokens (s) | One RL step (s) |
|---|---|---|---|
| EP4 CuTe-DSL FSDP2 (GB200 cluster, this round) | 518 / 436 / 477 | 67.9 / 57.7 / 57.6 | 906 / 803 / 888 |
| EP4 Humming FSDP2 (other cluster, ep4f2) | 571 / 462 / 515 | 74.3 / 61.1 / 61.1 | 1029 / 879 / 984 |
| PP4 CuTe-DSL dist_opt (GB200 cluster run `p6-runA`, diagnostic image, 2 steps) | 1033 / 907 | 134.8 / 120.1 | 1847 / 1671 |

- Other metrics: GPU peak allocated 152.9 GiB, reserved 160.7 GiB. Node host peak 439 GiB (`hostmem.txt` is whole-node usage).
- Comparison basis:
  - Compared with EP4 Humming on the other cluster, the clusters differ. Normalized by response tokens it is about 6–9% faster, which includes differences from both the GB200 nodes and CuTe-DSL that cannot be separated.
  - The PP4 row comes from a diagnostic run in another session. It carries observers such as optauth and final-probe and may be slower; but on the same cluster under the dev profile PP4 is also 26–35% slower than EP4 FSDP2, so the direction is consistent.

## Pitfalls

- FSDP2 plus rollout `gpu_memory_utilization=0.7` OOMs during vLLM's FlashInfer autotune warmup. The cause is that the FA4 split buffer needs 8 GiB while the actor process resides at about 45 GiB/GPU (`fast1-vllm-oom.txt`, two jobs).
  - It works after changing to 0.65; the recipe change has been committed and only takes effect under fsdp2.
  - The recipe in the dev image used for this round is still 0.7; this time it was set to 0.65 via an env override.
- The exactness observer's `memory_report` reports "invalid python storage" under FSDP2. Only the diagnostic feature is broken; results are unaffected.
- The training log shows many dynamo recompiles in `_gated_rms_norm_vjp`. PP4 has them too; it is a pre-existing phenomenon and was not addressed.

## Not done

- Runs with checkpoint save: the host OOM fix is being done elsewhere.
- dist_opt at formal length.
