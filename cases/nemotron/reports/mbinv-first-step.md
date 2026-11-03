# mbinv first step not bitwise: root cause and fix

## Conclusions

- **Root cause**: in PyTorch 2.13 (nv26.07), Inductor config overrides are thread-local (`torch/utils/_config_module.py`, ContextVar). `torch.use_deterministic_algorithms(True)` sets `torch._inductor.config.deterministic` to True on the main thread, but on autograd's CUDA device thread it is still the default False. The torch.compile'd norm VJP in `dense.py` is first compiled on that thread, during backward (`_residual_rms_norm_vjp` of `norm_f`, graph `2/0`). That compilation sees `deterministic=False`, so the generated looped-reduction kernel turns on `dynamic_scale_rblock`: besides `R0_BLOCK=1024` it also generates an `R0_BLOCK=512`, and picks one **by timing** at the first launch. The two have different in-row reduction orders, so the results differ bitwise. At the end of a dynamo compilation it calls `torch.use_deterministic_algorithms(prior)` (`convert_frame.py:374`), and only from then on is that thread deterministic=True. Later compilations (`2/1` etc.) all have a single fixed 1024 config. From step 1 on, `norm_f` hits the newer `2/1` (this is an inference: dynamo checks the newest cache entry first, and results from step 1 on are stable within a step and across steps, and differ from `2/0`), so the later steps are identical to each other; step 0 uses `2/0`, and differs from later steps whenever that timing picked 512.
- **Forward is unaffected**: in the bad runs, all forward Function outputs (including recompute), selected log-probs and loss are bitwise identical between step 0 and later steps with the same k. The first differing tensors are `grad_x`/`grad_residual` of the `norm_f` VJP. That VJP's `grad_weight` and all of its inputs are identical.
- **RL closed loop**: the rollout-training contract is bitwise-identical forward logprobs, which this problem does not touch, so the closed-loop bitwise PASS is consistent with this. What it affects is the norm VJP gradient of the shape that each actor process first compiles on the autograd thread: which RBLOCK gets picked is decided by timing. As a result, training trajectories are not bitwise reproducible across runs. This is backward bitwise reproducibility, which the human no longer requires, but it does exist, and there is a one-line fix.
- **Fix** (on a scratch fix branch, an internal commit): the three VJPs are changed to `torch.compile(..., options={"deterministic": True})`, no longer depending on which thread the compilation happens on. Verification: on the node where it reproduces easily, the baseline showed the symptom in 5 of 8 runs; 12 runs with the fix showed it zero times. On another node, the original sequence, fsdp2 and dist_opt were run 4 times each, all bitwise identical.

## Reproduction and localization

Harness: `mbfs.py`. It is the same as `mbinv/mb_invariance.py`, plus an observer `probe.py` that only measures and does not change numerics: it records a bitwise digest of the fwd/recompute/bwd inputs and outputs of every Nemotron autograd Function, the selected log-probs, every parameter gradient, and the Triton autotune timings; VJP calls are additionally run once compiled and once eager for comparison. The image is the delivery candidate `<image>` (the final ds4form candidate image from the r2 cross-review fixes), dist_opt, sequence `1,1,1`.

Node X below is the node where the problem reproduces easily; node Y is a second node.

| Runs (node) | Result | Bad ranks | Elements differing between step0 and step1 / relL2 |
|---|---|---|---|
| p1 r1–r6 (node X) | Bad 4/6 | r2:{3} r3:{0,2,3} r5:{2,3} r6:{0,2,3} | 124,456,929 / 8.4e-4; 245,372,548 / 1.06e-3; 225,340,157 / 9.8e-4; 245,372,548 / 1.06e-3 |
| p5 r1–r2 (node X, probe records Inductor's choice) | Bad 1/2 | r2:{0,2,3} | 245,360,566 |
| p2 r1–r2, p3 r1–r4, p4 r1 (node Y, p2 uses 1,2,3,1,4,1) | Bad 0/7 | – | 0 |

- The same set of bad ranks produces exactly the same number of differing elements (r3 and r6 are both 245,372,548). So this is a binary choice on each rank, not random noise, consistent with "1024 or 512".
- Rank 1 never went bad. It has only 481 tokens (size hint 512), and `_iter_rblock_scale_candidates` decides there is no need to raise occupancy, so no 512 candidate is generated. The other three ranks have 701/881/891 tokens (hint 1024) and do generate it. A single-GPU mini repro reproduces this (`mini_repro.py`, see below).
- The first per-op difference on a bad rank: op 4 = `_ResidualRMSNormVJP` bwd (`norm_f`), all input digests identical, outputs 0/1 differ. Calling the same compiled function a second time in the same step gives the same result as the first; the two calls in step 1 are identical to each other but differ from step 0; eager differs from both. This shows step 0 and step 1 use two different compiled artifacts.
- Downstream propagation: all upstream gradients on the bad rank; on other ranks only routed-expert weight gradients differ (they receive the bad rank's dy); after the dist_opt reduction all shards are affected. This matches the historical "120M–260M elements, relL2 about 1e-3".
- **Decisive evidence** (p5 r2, bad): on ranks 0/2/3, graph `2/0`'s kernel `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sub_0` timed `R0_BLOCK=1024: 0.0268 ms`, `512: 0.0225 ms`, `Best config … R0_BLOCK: 512`. Exactly these three ranks have step 0 different from later steps, and rank 1 is identical. In good runs (p5 r1 etc.), 1024 timed 0.0127 ms and was selected, and step 0 was identical to later steps.
- Direct evidence from the compile log (the p3 r1 log, `TORCH_LOGS=output_code,recompiles_verbose`): in `_residual_rms_norm_vjp`'s graph `2/0`, each kernel's `inductor_meta` has `'deterministic': False, 'are_deterministic_algorithms_enabled': True`; graph `2/1` (same shape, recompiled due to a guard failure) has `'deterministic': True`. After the probe turned on `triton_heuristics` DEBUG logging (p3 r4, p4, p5), ranks 0/2/3 show `Dynamically scale down R0_BLOCK 1024 -> 512` and `Best config …`.
- Mini repro (`mini_repro.py`, one job, a fresh process and fresh cache each time): main thread `inductor.deterministic=True`, backward thread False the first time and True the second time. Rows 701/891 show the scale-down and timing; row 481 does not. With `TORCHINDUCTOR_DETERMINISTIC=1` or `options={"deterministic": True}`, the backward thread is True the first time as well, and there is no scale-down. On a quiet single GPU 1024 is faster every time (0.0128 vs 0.0174 ms), so numerics in the mini repro do not change. In real training there is communication and kernels from other streams on the GPU, so the timing gets perturbed and 512 sometimes wins. The different incidence across nodes (node X 5/8, node Y 0/7) may also come from this.

## Ruled-out candidates (evidence)

- cuBLAS/cuBLASLt, TE GEMM, CuTe-DSL/FlashInfer MoE, FA4 bwd, DeepEP: on bad ranks they all come after the first difference point; on good ranks all of these ops are bitwise identical between step 0 and later steps.
- Uninitialized memory, RNG, FSDP2/dist_opt first-step state: the difference starts at a pure function call with exactly identical input digests, and it occurs with both fsdp2 and dist_opt.
- mamba_ssm's Triton autotune: it really does autotune by timing (9 configs, 99 benches at step 0). The reason is vLLM's `env_override.py:176` `setdefault("TRITON_CACHE_AUTOTUNING","1")`, so `mamba_ssm.utils.determinism.autotune_configs` keeps all configs. But once selected it is fixed within the process, the `init_to_zero` pre_hook zeroes the accumulators, and in good runs step 0's `_PackedScanVJP` is bitwise identical to later steps. So it is not the cause of the first-step symptom. It is one source of backward not being bitwise across processes or jobs: within one job, the first process's canonical grad norm differs from later processes', e.g. p1 r1 is 246.87589 and r2–r6 are 246.87506; later processes reuse Triton's on-disk autotune cache.
- Recompute: in bad runs, the recomputed forward Function outputs are bitwise identical between the two steps.

## Fix and verification

`experimental/lite/megatron/lite/model/nemotron_h/vllm/primitive/dense.py`:

```python
_VJP_COMPILE = dict(fullgraph=True, dynamic=False, options={"deterministic": True})
_compiled_rms_norm_vjp = torch.compile(_rms_norm_vjp, **_VJP_COMPILE)
_compiled_residual_rms_norm_vjp = torch.compile(_residual_rms_norm_vjp, **_VJP_COMPILE)
_compiled_gated_rms_norm_vjp = torch.compile(_gated_rms_norm_vjp, **_VJP_COMPILE)
```

Verification method: image unchanged, only the modified `dense.py` bind-mounted in as a single file (sha256 `7b675072…`). This is a scratch experiment, not part of the delivery. The harness runs without the probe, equivalent to the original harness.

| Job | Node | Sequence / backend | Runs | Same-k not bitwise | Compile log |
|---|---|---|---|---|---|
| fix1 | node X | 1,1,1 / dist_opt | 12 | 0 | `'deterministic'` is True for every kernel (672 kernels, False appears 0 times) |
| fix2 | node Y | 1,2,3,1,4,1 / dist_opt+fsdp2 | 4+4 | 0 | – |
| env1 (no code change, only `TORCHINDUCTOR_DETERMINISTIC=1`) | node Y | 1,1,1 / dist_opt | 3 | 0 | `'deterministic'` True for all kernels (168) |
| Control: p1+p5 baseline | node X | 1,1,1 / dist_opt | 8 | 5 | bad ranks' `2/0` picked R0_BLOCK=512 |

Using node X's baseline incidence of 5/8, the probability of 12 consecutive clean runs after the fix is about (3/8)^12 ≈ 1e-5. The fix also has a side effect: in fix1, the step 0 grad norm of runs 2–12 is 246.87503 in every case, now bitwise identical across processes too. Run 1 is 246.87593, which is mamba's first Triton autotune timing; see above.

A more defensive approach: add `TORCHINDUCTOR_DETERMINISTIC=1` to the recipe environment. It takes effect as an env default at config import, visible to every thread and every compilation, including compilations on the autograd thread of vLLM forward functions in recompute (grouped_topk, MoE combine). Both the mini repro and env1 verified that it works. Recommend doing both: the code fix does not depend on the environment; the env covers any "first compilation" on the autograd thread. Right now the autograd thread's first compilation happens to be `norm_f`'s VJP, so recompute's forward is not affected; but that is just how the execution order happens to be, with no guarantee at all.

## Other observations (outside this fix)

- The norm VJP is `dynamic=False`. Every new token count triggers a recompile, and past `recompile_limit=8` it falls back to eager (the harness log has `hit config.recompile_limit (8)`). eager and compiled numerics differ (probe: `first_eq_eager=False`), so backward numerics depend on which shapes the process has seen and in what order. The human has decided backward need not be bitwise; recorded here only.
- The comment at `protocol.py:349` says verl's full_determinism does not set `use_deterministic_algorithms`. The verl in the image (`verl/workers/engine/utils.py:59`) actually sets it with `warn_only=True`.

## Files

- Harness, observer and analysis scripts on shared storage (`mbfs.py`, `probe.py`, `analyze.py`, `analyze_vjp.py`, `mini_repro.py`, `mbfs.sbatch`, `mini.sbatch`, `fix/dense.py`)
- Run results on shared storage (`mbfs-{p1,p2,p3,p4,p5,fix1,fix2,env1}/`, `mbfs-mini.out`)
- Fix branch: a scratch Megatron fix branch in its own worktree (one internal commit on top of the delivery base), not pushed; the delivery branch has no changes
