# final: final delivery (2026-10-06) + step-10 bench, step-7 EP4 + FSDP2, training-side acceptance

Commit labels: the vLLM, Megatron-LM and verl changes live on internal forks, so their commit SHAs are replaced by labels in this document. vLLM: V0 (the wheel/candidate head), V1 (the final-delivery head). Megatron fork: M0 (cand3 head), M1 (candidate head), M1d (M1 + 2 diag commits), M2 (first final-delivery head), M2d (M2 + diag commits), M3 (second final-delivery head), M4 and M5 (later checkpoint-fix heads; M5 is in the final image), M6 (Megatron PR head). verl fork: R1 (candidate head), R1d (R1 + 1 diag commit), R2, R2d (R2 + diag commit), R3, R4 (the pin commit in the final image), R5 (verl PR head).

## 0. Final delivery (2026-10-06, one recipe)

This section supersedes the "final-delivery head" in the sections below. Sections 1–5 below are the acceptance done on 10-04/05 on earlier heads; the BI=1 conclusions carry over (reasons in 0.4).

### 0.1 Three PRs

| PR | branch | head | content |
| --- | --- | --- | --- |
| vLLM PR on an internal fork | `pr/nemotron-h-bi-cutedsl-p6` | V1 | vLLM: Nemotron-H BI + CuTe-DSL (depends on vllm-project/vllm#55905) |
| Megatron-LM PR on an internal fork | `pr/nemotron-h-ep4-cutedsl` | M6 | mLite actor (EP4 + FSDP2), including checkpoint item-write error reporting/EFAULT retry and an exact-size pinned offload buffer. The image was built with M5; the later M6 only makes DCP save call `empty_host_cache` when torch has it (the image's torch has it, behavior unchanged; tests passed in one job) |
| verl PR on an internal fork | `pr/nemotron-h-ep4-cutedsl` | R5 | one recipe (BI=1 true on-policy / BI=0 conventional baseline) + Dockerfile + shared verl changes needed by BI=0. R3..R5 = 12 commits from a local one-recipe branch (reviewed in r7–r9) + 1 pin commit R4 (vLLM V1, version `0.30.1rc1.dev633+g<sha>`; Megatron M5; a pinned Megatron-Bridge commit unchanged) + the r10 test commit R5 (only changes `tests/utils/test_modelopt_quantize_on_cpu.py`; the verl package installed in the image is still R4, unaffected), all pushed fast-forward |

Separate PRs / branches (not part of this delivery):

| item | location | description |
| --- | --- | --- |
| Megatron dense recompile | a separate PR/branch on the internal Megatron fork | Skip dynamo once the RMSNorm VJP shape exceeds the recompile limit (host memory leak). r10: before skipping, check the number of dynamo cache entries of the code object, and recompile new shapes after `torch._dynamo.reset`; two globals merged into `_VJP_STATE`; tests leave no global entries and cover recorded shapes with different strides and cache reset. The official harness `test_nemotron_h_training_vjp.py` run mounted on the final image (one job, one node) overall=PASS. Reviewed/merged separately, not in the Megatron PR or the image |
| vLLM KV-scale reload | a separate PR on the internal vLLM fork, branch `fix/flashinfer-kv-scale-reload` | Recompute the FlashInfer BMM scale after a weight reload changes the KV scale. Not in the vLLM PR or the image |
| p29c prefix-cache performance | a local branch (not pushed) | Performance fix for the align mode of Mamba2 BI prefix caching, see [prefix-cache-perf](../../reports/prefix-cache-perf.md). Scratch, not in the image |

### 0.2 Final image

| item | value |
| --- | --- |
| path | `<image>` (the final2 delivery image: vLLM V1, Megatron M5, verl R4) |
| sha256 | `a3817a770d88ab140f08d37fa221d4903f0245dd56b432fc79831db3c06ce0e8` (`.sha256` file; recomputed on the login node, matches) |
| build | enroot commit (one job, one node), on top of an earlier one-recipe base image (sha256 `e299fb92…3d7b`). vLLM: the in-package changes between V0 and V1 are only 2 .py files (`ssd_emit.py`, `models/config.py`), overwritten per file with the dist-info RECORD hashes updated; the compiled artifacts are still from the V0 wheel, so `vllm.__version__` still shows `0.30.1rc1.dev631+g<V0>`; megatron-core reinstalled from M5 (`0.19.0+<M5>`); verl reinstalled from R4; recipe installed to `/opt/nemotron/recipe/` |
| in-image manifest | `/opt/nemotron/image-manifest.json` and `/opt/nemotron/{vllm,megatron,verl}-revision`: vllm V1 (`vllm_wheel_revision` V0, `vllm_python_overlay` lists 2 files), megatron M5, verl R4, megatron-bridge the pinned commit |
| out-of-image manifest | `<image>.manifest.json` |
| scripts and logs | `final2/` (`scripts/` holds the build/inside/verify/validate scripts; `verify-before-export.txt`, `validate-*.log`, `quick/`) |

### 0.3 How to run (one recipe)

```bash
# Inside the image, a single GB200 node with 4 GPUs
export MODEL_PATH=<NVFP4 checkpoint> BF16_MASTER_PATH=<BF16 release> TRAIN_FILES=<DAPO parquet> VAL_FILES=<small val parquet>
export MLITE_DCP_LOCAL_STAGE_DIR=<node-local-disk>          # BI=1 checkpoint staging
R=/opt/nemotron/recipe/run_nemotron_h_lightning_nvfp4_true_on_policy_megatron.sh
bash $R --hardware gb200 --mode quick_alignment_test                          # BI=1 (default) true on-policy, 3 steps
bash $R --hardware gb200 --mode aligned                                       # BI=1 formal recipe
VLLM_BATCH_INVARIANT=0 bash $R --hardware gb200 --mode aligned                # BI=0 baseline: Megatron-Bridge/mcore + R3 + ModelOpt weight QAT
VLLM_BATCH_INVARIANT=0 ACTOR_PRECISION=bf16 bash $R --hardware gb200 --mode aligned   # BI=0 bf16 actor
# DRY_RUN=1 prints the command; COMPOSE_ONLY=1 prints the Hydra config
```

On the cluster, use `$WORKDIR/long-run/run-long4.sbatch <image> <tag> <env file> --hardware gb200 --mode ...` (env files `lr-bi1-s42.env` / `lr-bi0-s42.env`).

### 0.4 Verification on the final image

| item | job | result |
| --- | --- | --- |
| Installed files vs git head, byte by byte (per each dist's RECORD) | the build job (before export: vllm/megatron-core/verl), the post-export verification job (on the exported sqsh via pyxis, all 5 dists) | **0 mismatch**: vllm 3225 files (against `git archive` of V1), megatron-core 622, megatron-lite 153, verl-mlite 17 (against M5), verl 505 (against R4). Files present only in the image are build artifacts/third-party copies (`.so`, `_version.py`, `third_party/deep_gemm`, etc.). The installed recipe is byte-identical to the head |
| BI=1 DRY_RUN vs R3 | the DRY_RUN job | all 8 combinations (2 modes × fsdp2/dist_opt × `NCCL_CUMEM_ENABLE` unset/0) **IDENTICAL**, same with `VLLM_BATCH_INVARIANT` unset and `=1` (compared after normalizing mktemp directory names) |
| BI=0 DRY_RUN, in-build COMPOSE_ONLY | the post-export verification job, the build job | 2 modes × qat/bf16 all rc=0; COMPOSE_ONLY BI=1/0 × 2 modes all rc=0 |
| BI=1 quick (3 steps) | one job, `runs/final2-bi1-quick` | exit 0; **probs_diff max = mean = 0** on all three steps; grad norm 0.17/0.10/0.44; host peak 442 GiB |
| BI=0 quick (3 steps, QAT) | one job, `runs/final2-bi0-quick` | exit 0; probs_diff mean **0.0097 / 0.0092 / 0.0100** (single-token max 0.65–0.98); grad norm 0.41/0.16/0.36; host peak 415 GiB |
| CPU unit test `test_modelopt_quantize_on_cpu.py` | the post-export verification job (R4 version); another job (mounting the R5 version, `review-fixes/r10verl/`) | both 5 passed (modelopt 0.44.0, TE 2.17.0). The R5 version constructs placeholder arguments from the `_GroupedLinear.forward` signature, so it also holds for TE ≥ 2.18 |
| CPU config unit tests (`tests/workers/config` and 4 more locations) | the DRY_RUN job | head: 55 passed / 3 failed / 10 skipped; R3: 54 / 4 / 10. The 3 failures occur on both (image-environment related, not introduced by this PR); the extra one on R3, `test_megatron_actor_config_from_yaml`, is the hydra 1.4 rejection of `../`, which this PR fixes |
| mcore util / fused forward unit tests | the post-export verification job | head 26 passed / 1 failed, same on R3 (`test_megatron_bridge_wrapper_chain_reaches_native_hook`: stub object lacks `moe_n_hash_layers`, a pre-existing issue) |

Note: in the final image `vllm.__version__` is `0.30.1rc1.dev631+g<V0>` (V0 wheel + 2-file overwrite); a Dockerfile build would produce `dev633+g<V1>` (the value its smoke test asserts, never run on this cluster); the file contents are the same (RECORD byte-by-byte 0 mismatch).

Long runs (one recipe, seed 42, aligned, still running; the image is an earlier one-recipe image; legs 1–2 of the two BI=0 chains are on an earlier verl commit, leg 3 onward on a later verl commit — not on the PR branch, containing verl-side offload/EFAULT monkeypatches (later superseded by Megatron M4/M5) and the old ModelOpt class-attribute QAT patch (later superseded by registry replacement), numerically equivalent but not PR code): `lr-bi1-s42` 62 steps, probs_diff 0 on every step; `lr-bi0-s42` 48 steps, probs_diff mean 0.0095–0.0109, score rising to 0.50–0.66; `lr-bi0-bf16-s42` 62 steps, starts collapsing around step 41 (no rollout correction, known limitation).

The old BI=1 conclusions (bench, step 7, optauth, closed loop, 100-step long run; sections 1–5 below) carry over: the BI=1 command is byte-identical to R3; vLLM V0..V1 only adds a config-time TP>1 rejection + removes a redundant guard + tests; Megatron M3..M5 only changes checkpoint write error reporting, the offload buffer and tests.

The verl PR body is kept in the workspace's PR-body notes (already applied with `gh pr edit`; not included here).

---

Dates: 2026-10-04/05. Jobs: one batch on the candidate image; a second batch on the final-delivery heads (see section 5). Job lists are in `sacct.txt` and `final-heads/sacct.txt`.

| item | value |
| --- | --- |
| candidate image | `<image>` (candidate image: vLLM V0, Megatron M1, verl R1). sha256 `8ededb8962e128e1bb997911ea10a3e1b8b2bb46f0891a17d2775017ce1864af`, recomputed on the login node before running, matches `.sha256` |
| diagnostic image | `<image>` (diagnostic image: vLLM V0, Megatron M1d, verl R1d). sha256 `108204d38dcb…1286`, recomputed, matches |
| heads | vLLM V0, Megatron M1, verl R1 (diagnostic: Megatron M1d = M1 + 2 diag commits; verl R1d = R1 + 1 diag commit) |
| final-delivery heads | Megatron M2, verl R2, vLLM unchanged. Relative to the heads above, only checkpoint and pin changes, see section 5 |

## Conclusions

| item | result | cand3 (pre-final) |
| --- | --- | --- |
| bench, 5-pair median loss c=1/8/32/64 | **8.5% / 8.6% / 11.9% / 1.9%, all 4 c values ≤20%, pass** | 12.7 / 10.6 / 14.4 / 11.6% |
| step 7 v2 EP4 + FSDP2, snapshots 0/50/150/250 | **4/4 PASS, all self-checks pass** | 4/4 PASS |
| optauth (frozen criteria 4864cbb4…), 2 runs + 1 run on the final-delivery heads | **all three 19 PASS / 2 FAIL (A1, A6)**, as expected | r4: 19/2 |
| closed loop + θ2 probe (diagnostic image, formal workload, 3 steps) | **PASS** (on M1 in one job; also PASS on the final-delivery heads in a later job) | r4: PASS |

One point needs attention: optauth2's A6 has 93 failures, 3 of which are not on the step of the first non-zero gradient (`layers.48/51.norm.weight` on versions 1 and 2, γ 0.985–0.990). The same happened on r0 (rv-optauth2) and r1 (rv1-optauth2), so it is not a new phenomenon this time. But the statement "A6 only fails on the first step of the norms" does not strictly hold; see section 3.

## 1. bench (`bench/`)

- The protocol, scripts, configs and vllm-rs (sha `7c0378b0…`) are all the same as cand3. Only the image, the Lustre work directory (`$WORKDIR/final/bench`), the node-local directory (`<node-local-disk>`, a fresh model copy and JIT cache) and the label prefix (`fnm`) changed. The denominator is BI0npc on this image.
- One pair per node: p1–p5 on five different nodes, one job each, run in parallel. Leg order is AB BA AB BA AB (A = BI0).
- Per-leg check (`runs/fnm-backend-check.txt`): all 10 legs are `FLASHINFER_CUTEDSL`, FULL_AND_PIECEWISE, `prefix_caching=False`, server_exit=0, leg_rc=0. All 40 points valid. KV cache: BI0 14,879,890 tokens, BI1 10,759,680 tokens, same as cand3. Cold start 745–901 s.

| c | BI0/BI1 tok/s: p1 / p2 / p3 / p4 / p5 | per-pair loss % | **median** | cand3 median |
| --- | --- | --- | --- | --- |
| 1 | 379/285, 339/283, 312/286, 302/280, 288/284 | 25.0, 16.4, 8.5, 7.4, 1.5 | **8.5%** | 12.7% |
| 8 | 2512/1903, 2137/1919, 2181/1994, 1829/1949, 1884/1963 | 24.2, 10.2, 8.6, −6.5, −4.1 | **8.6%** | 10.6% |
| 32 | 7776/6599, 6234/5682, 6778/5660, 6988/6230, 7036/6196 | 15.1, 8.9, 16.5, 10.8, 11.9 | **11.9%** | 14.4% |
| 64 | 9015/8763, 8657/6312, 10023/9837, 8689/8926, 8005/8387 | 2.8, 27.1, 1.9, −2.7, −4.8 | **1.9%** | 11.6% |

Median throughput per arm (tok/s):

| | c=1 | c=8 | c=32 | c=64 |
| --- | --- | --- | --- | --- |
| BI0 | 312 | 2137 | 6988 | 8689 |
| BI1 | 284 | 1949 | 6196 | 8763 |
| cand3 BI0 / BI1 | 326 / 286 | 2203 / 2006 | 7044 / 6377 | 9876 / 8602 |

- The median loss is lower than cand3 mainly because BI0 is low this time (c=64 median 8689 vs cand3's 9876), not because BI1 got faster. BI1 is essentially the same as cand3.
- 3 single pairs exceed 20%: c=1 p1 25.0%, c=8 p1 24.2%, c=64 p2 27.1%. As before, these are caused by BI0 landing in the fast mode or BI1 landing in the low mode (p2 BI1 at c=64 is 6312).
- `runs/` does not contain metrics snapshots; they are on Lustre at `$WORKDIR/final/bench/runs/`.

## 2. Step 7 v2, EP4 + FSDP2 (`step7/`, one node, one job)

**Pre-run change check** (`step7/meg-M0-M1-code.diff`, 12 files under model/primitive):
- The only numerical change on the gradient path is `dense.py`: the `torch.compile` of 3 norm VJPs gained `options={"deterministic": True}`. This is exactly why the rerun was required.
- The rest are removals of construction-time and forward self-checks (stage quantization contract, FP8 KV scale coverage, NaN scale, geometry, batch-invariant mode assertions, etc.), comments, and checkpoint-related changes.
- The symbols used by the harness (`Fp8TrainingLinear`, `Nvfp4TrainingLinear`, etc.) are all still there.

**Harness and criteria**: the stage-delivery harness (an internal commit), with per-file sha256 identical to cand3's `harness.sha256` (`step7/formal-ep4-fsdp2/harness.sha256`). Criteria `criteria-step7-v2-ep4.json` `1a7caaf4…`, EP4 appendix `e50668bf…`, FSDP2 appendix `fd4bfb10…`, all unchanged. Parameters same as cand3: 250 steps, snapshots 0/50/150/250, A/A with rev and rot, `ACTOR_OPTIMIZER=fsdp2`.

**Self-checks**: all pass.

| check | result |
| --- | --- |
| runtime ‖g‖ bitwise equal to the dump pass | yes (246.87528992) |
| relative difference of the FP64 norm vs runtime | 1.1e-8 |
| local shards of the 4 ranks at step 0 bitwise identical to the dump | yes |
| b_ref dense parameter/gradient replica_max_abs_diff over the 4 snapshots | all 0 / 0 |
| step-0 loss | A 16.059851 (same as cand3) |

| snapshot | verdict | worst class median | max single-tensor ratio (relL2) | G3 exceedances | ‖g‖ A/B16/B32 |
|---|---|---|---|---|---|
| 0 | PASS | router 1.04 | L2 in_proj 1.40 (0.029) | 0 | 246.88/246.57/247.69 |
| 50 | PASS | attention_linear 1.02 | L2 mixer.norm 1.54 (0.016) | 0 | 82.32/82.35/82.35 |
| 150 | PASS | shared_experts 1.08 | L2 norm 1.49 (0.015) | 0 | 19.90/19.88/19.85 |
| 250 | PASS | shared_experts 1.04 | L0 D 1.31 (0.049) | 0 | 14.29/14.28/14.30 |

- G0/G1/G4 pass on all 4 snapshots. The floors of all classes come from B16 (floor_from_A = 0). No non-finite values and no skipped steps during training. Step-250 val NLL 3.6635 (cand3 3.6617).
- Step-0 ‖g‖ changed from cand3's 246.87497 to 246.87529; the step-0 class medians and single-tensor maxima are the same as cand3. This is consistent with the change in how the norm VJPs are compiled.
- cand3's item closest to the threshold (step-50 routed_experts 1.39) is 0.83 this time. The worst class median over the 4 snapshots is ≤1.08 (threshold 1.5).
- Snapshots were placed in node-local /tmp, 323 GB in total, deleted at the end of the job (`rc.txt`).

## 3. optauth (`optauth/`, diagnostic image, quick_alignment_test, two jobs on two nodes)

- Scripts: `scripts/run.sbatch` + `scripts/optauth.env`, same as the ds4form/training README (the env file is byte-identical to rv4-optauth).
- Criteria: `results/optimizer-auth/criteria-v2-fsdp2.json` sha256 `4864cbb4…`, recomputed, unchanged. Verdict script `optauth_reduce_v2_fsdp2.py`.
- Both runs exit 0, oom=false, host memory peaks 495.1 / 463.6 GiB.

| run | verdict | A6 failures (γ range) | grad norm per RL step |
| --- | --- | --- | --- |
| final-optauth | 19 PASS / 2 FAIL (A1, A6) | 88 (0.942–1.023), all on version 4, all `layers.N.norm.weight` | 0, 0, 0.26 |
| final-optauth2 | 19 PASS / 2 FAIL (A1, A6) | 93 (0.959–1.015): version 0 has 87 `norm.weight` and 3 `norm_f.weight`; versions 1/2 have 3 (`layers.51.norm.weight` γ 0.990, `layers.48.norm.weight` γ 0.985/0.988) | 0.20, 0.15, 0.13 |

- **A1 FAIL**: same reason as before — only A1f's requirement "check at least one replicated parameter"; under this configuration `replicated_keys_checked=0`. `model_mismatch=0`, no step with multiple clip coefficients.
- **A6 FAIL**:
  - final-optauth: the advantages of the first two RL steps are all 0, the first non-zero gradient is on version 4, and all 88 failures are on version 4.
  - final-optauth2: 90/93 are on the step of the first non-zero gradient (version 0). The other 3 are on versions 1/2, all norm.weight, with |γ−1| of 0.010–0.015 against a threshold of 0.01.
  - Rechecking earlier runs with the same script: rv-optauth2 (r0) has 1 on version 2, rv1-optauth2 (r1) has 45 on version 2, all norm.weight. rv4-optauth/rv4-optauth2 have none. So norm failures beyond the first step existed before and were not introduced this time, but the claim "only on the first step" is not strict.
- The other items are the same as r4: A4 normal-domain max ratio 0.124; in A5, β ∈ [−0.216, −0.151] with wd_mult=1, |β| ≤ 0.049 with wd_mult=0; Q1 18,780 tensors / 65.7 GB, 0 byte differences; K1, K2 (16 updates, 291,472 tensors) both 0 mismatches; B1 RNE 3,824 times + 2,224 masters, 0 mismatches; B2 0; K3 is an empty check.

## 4. Closed loop + θ2 probe (`closed-loop/final-loop/`, diagnostic image, one node, one job)

- `--hardware gb200 --mode aligned`, `closed-loop.env` (TOTAL_TRAINING_STEPS=3, SAVE_FREQ=-1, TEST_FREQ=-1), worker_extension_cls swapped for the observer. The env file is byte-identical to r4-loop2. No files under `checkpoints/`.
- Receipts generated with `exactness_reduce.py --ranks 4 --steps 8 --versions 3 --logit-ranks 0 1 2 3`, `optauth_reduce_v2_fsdp2.py` (reference only) and `final_receipt.py --probe`. `closed-loop-receipt.json` has `pass=true`.

| item | final-loop | r4-loop2 |
|---|---|---|
| θ0 old-logprob vs rollout: tokens / mismatches | 7,902,028 / 0 | 7,942,569 / 0 |
| θ0 grad-enabled forward vs rollout / vs old-logprob | 754,577 / 0 / 0 | 717,089 / 0 / 0 |
| θ1 old-logprob | 8,353,054 / 0 | 8,337,426 / 0 |
| θ1 grad-enabled forward | 774,248 / 0 / 0 | 789,697 / 0 / 0 |
| θ2 probe (no update) old-logprob | 7,787,479 / 0 | 7,732,654 / 0 |
| verl byte gate steps 1/2/3 (FP32); step 3 `final_probe_no_update=1` | 1/1/1 | same |
| 8 optimizer steps × 1512 FP32 master shards all changed, step counter +1 | ✓ (grad norm 0.054–0.123) | ✓ |
| sync K2: 16 updates, 291,472 tensors, 0 mismatched / 0 missing / 0 extra | ✓ | ✓ |
| K1 / Q1 / B1 | 0 / 0 / 0 | 0 / 0 / 0 |
| exit / host memory peak | 0 / 501.2 GiB | 0 / 494 GiB |
| time (s, step / gen / old_logprob / update_actor) | 1375/251/120/828; 1348/280/115/780; probe 524 | — |

- `exactness-receipt.json` has `pass=false`, solely because the probe version by design does not run the training forward, same as before.
- The Traceback at exit comes from torch's `dump_compile_times` atexit callback and a weakref finalizer, appearing after the 3 steps completed; exit is still 0.

## 5. Final-delivery heads (Megatron M2, verl R2, vLLM V0)

The results of sections 1–4 are bound to Megatron M1 / verl R1. Afterwards the delivery heads moved to:

- Megatron M1..M2 has two commits: one checks the resume sidecar on all ranks; M2 clears the step directory before overwriting. The `git diff --stat` is in `final-heads/meg-M1-M2.stat`, only 4 files: `verl_mlite/engine/mlite_engine.py` (only changes how load_checkpoint reads the LR scheduler state), `primitive/ckpt/dcp.py`, and two checkpoint tests. The model, VJP, optimizer step and weight-sync code are unchanged.
- verl R1..R2 only changes the Dockerfile's `MEGATRON_REVISION` pin, 1 line.
- So the forward, backward and sync paths are byte-identical to M1, and **the bench and step 7 do not need to be rerun**. Sections 3 and 4 under SAVE_FREQ=-1 also do not go through the changed code.

Afterwards the delivery heads moved again to Megatron M3, verl R3:
- M2..M3 has only one commit, which only changes `tests/unit/runtime/checkpoint/test_training_checkpoint.py` (4 insertions, 4 deletions, see `final-heads/meg-M2-M3.stat`).
- R2..R3 only changes the Dockerfile's `MEGATRON_REVISION` to M3.
- The package code installed in the image is the same as M2, so the r5 candidate image, the r5 diagnostic image and the results above are equally valid for M3 / R3.

Verification on the final-delivery heads (`final-heads/`). The closed loop and optauth were submitted by the coordinator after user approval (two jobs), with the same commands as sections 3 and 4, only the diagnostic image swapped:

| item | result |
| --- | --- |
| r5 candidate image `<image>` (vLLM V0, Megatron M2, verl R2) | sha256 `253dc2d3…24a8`, recomputed, matches. The source exports (Megatron M2, verl R2) in the work directory are identical to `git archive` (only build/egg-info extra). `verify-image.sh` (one job): `vllm_bad=0`, megatron/lite, verl_mlite, recipe all SAME; the remaining differences are line-for-line identical to r4's `verify-cand2.out` |
| r5 diagnostic image `<image>` (vLLM V0, Megatron M2d, verl R2d) | New local diag branches: Megatron = M2 + cherry-picks of the 2 diag commits (no conflicts); verl = R2 + cherry-pick of the diag commit. Both diag patches have the same content as the cr4 patches (Megatron differs only in context line numbers). Built with `make-ds4form-sqsh.sh` (one job), sha256 `f4f1713de9a4…bdcbf`; `verify-image.sh` (another job) is line-for-line identical to r4's `verify-diag2.out`. Patches in `final-heads/*.patch`. Branches are local only, not pushed |
| quick gate dist_opt (r5 candidate image, `ACTOR_OPTIMIZER=dist_opt`, one job) | exit 0; probs_diff max/mean both 0 over 3 steps, pearson 1.0, grad norm 0.005/0.179/0.440. Host memory peak 772.8 GiB (r4's dist_opt gate was 765.3 GiB) |
| closed loop + θ2 probe (r5 diagnostic image, one job, `final-heads/closed-loop/`) | **PASS** (`closed-loop-receipt.json`). old-logprob vs rollout: 7,743,240 / 8,251,297 / 7,598,975 tokens, 0 mismatches; grad-enabled forward: 741,339 / 783,448 tokens, 0 mismatches against both rollout and old-logprob; byte gate steps 1/2/3 all 1, step-3 probe flag 1; 8 steps × 1512 shards all changed, grad norm 0.077–0.169; K2 16 updates, 291,472 tensors, 0 mismatched / 0 missing / 0 extra; K1 / Q1 / B1 0; exit 0, oom=false, host peak 479.0 GiB; time 1362/1343/probe 514 s; 0 files in `checkpoints/`. env byte-identical to section 4 |
| optauth (r5 diagnostic image, one job, `final-heads/optauth/`) | **19 PASS / 2 FAIL (A1, A6)**, criteria 4864cbb4 unchanged. A1 likewise only because `replicated_keys_checked=0` (model_mismatch 0). A6 100 failures (γ 0.957–1.015): 96 on version 0 (the step of the first non-zero gradient; 93 `norm.weight` + 3 `norm_f.weight`), 4 `norm.weight` on version 2 (L38, L48, γ 0.987–0.990), the same phenomenon as optauth2 in section 3. A4 0.124; A5 wd_mult=1 β ∈ [−0.213, −0.150], wd_mult=0 \|β\| ≤ 0.039; Q1 18,780 tensors, 0 byte differences; K1, K2 0 mismatches; B1 3,824 + 2,224, 0 mismatches; B2 0. exit 0, host peak 467.0 GiB; grad norm 0.19/0.10/0.37 |

## Files

- `bench/`: scripts, configs, `vllm-rs.sha256`, `runs/` (including `fnm-pair-summary.txt` and `fnm-backend-check.txt`).
- `step7/`: the M0→M1 code diff, `formal-ep4-fsdp2/` (gcheck, b-step*.json, metrics, meta, a-checks.log, b-replica.log, run-info, rc, harness.sha256).
- `optauth/final-optauth{,2}/`: `receipt-v2-fsdp2.json`, raw exactness records, env, trainer jsonl, log tails.
- `closed-loop/final-loop/`: `closed-loop-receipt.json`, `exactness-receipt.json`, `optauth-v2-fsdp2-info-receipt.json`, env, trainer jsonl, log tails.
- `scripts/`: run.sbatch, optauth.env, closed-loop.env (copied from ds4form/training/scripts).
- `final-heads/`: image verification output, diff stats, diag patches, dist_opt gate.
- Lustre: `$WORKDIR/final/bench/`, `$WORKDIR/step7-v2/runs-delivery/final-formal-ep4-fsdp2/`, `$WORKDIR/runs/{final-optauth,final-optauth2,final-loop,fh-gate-distopt}/`.
