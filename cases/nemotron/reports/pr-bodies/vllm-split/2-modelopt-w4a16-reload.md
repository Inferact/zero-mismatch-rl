## Split of the combined fork PR

Preview in a fork (6 parts; base `split/base` = upstream `6e517b15c1`).

How the split relates to the combined fork PR (head: the delivery vLLM tree):

- Splitting alone reproduced the combined PR exactly: the stage-A union of the six branches has the same tree as the combined PR's head.
- The posted heads then add behavior-neutral simplification commits, so their union is **not** byte-identical to the combined PR. It equals the stage-B union; the diff between the combined PR's head and the stage-B union is 8 files, +168/−160: an unreachable `is_act_and_mul` raise removed in `fused_moe/oracle/nvfp4.py` (PR 3); `fixed_split_unsupported_reason` folded into `FlashAttnFixedSplitImpl.__init__` (PR 5, same checks and messages); a 3-line comment removed in `models/nemotron_h.py` (PR 6); the GateLinear / FP8 bmm / Humming / router tests moved from the Nemotron-H test files into `tests/v1/determinism/test_batch_invariant_gemm_dispatch.py` (PR 4, +1 line in `.buildkite/test_areas/misc.yaml`).
- The stage-B union was re-run on gate3 (GB200, DP4/EP4, BI=1): same result as the combined PR, same `rollouts.json` sha256 (`06daedf0…5990`). Unit-test results below are per branch.

1. Mamba2 BI exact replay (port of https://github.com/vllm-project/vllm/pull/55905, co-authored with its author; review context only)
2. ModelOpt W4A16 NVFP4 layerwise reload fix
3. CuTe-DSL W4A16 NVFP4 MoE experts
4. Batch-invariant GEMM dispatch fixes
5. FA4 fixed split-KV backend
6. Nemotron-H batch-invariant wiring (stacked on 1 and 5, base `split/6-base`; needs 4 to load W4A16 checkpoints under BI)

---
## Purpose

Fix layerwise reload (`initialize_layerwise_reload` → `load_weights` →
`finalize_layerwise_processing`) for ModelOpt checkpoints with W4A16 NVFP4
layers (`W4A16_NVFP4`, or `MIXED_PRECISION` with W4A16 layers).

Weight-only NVFP4 checkpoints carry no `input_scale`, or a deprecated one.
Two problems:

1. The `input_scale` placeholders (routed experts `w13/w2_input_scale`,
   linear `_DropInputScale`) count toward the layer's reload load total, so
   every W4A16 layer stays deferred until finalize and keeps all its staged
   tensors alive. Measured on Nemotron-H Lightning NVFP4 (128 experts): 16.5 GiB
   extra on one GPU during a weight update. Fix: `weight_loader_numel = 0` on
   those placeholders, so the layer is processed as soon as its weights arrive.
   Only the reload trigger reads this attribute; cold loading and kernels are
   unchanged.
2. With (1), a deprecated `input_scale` in a later bucket arrives after the
   layer was processed and `input_scale` dropped, which raised
   (`AutoWeightsLoader` ValueError / `RoutedExperts` AttributeError). Fix:
   ModelOpt configs with W4A16 layers add `.input_scale` to
   `_ignore_unexpected_suffixes` (consulted only when the parameter is absent,
   so FP8 input scales still load and are still waited for); `RoutedExperts`
   skips a missing `input_scale` for NVFP4 W4A16 MoE.

Files: `quantization/modelopt.py`, `quantization/base_config.py` (type
annotation so the subclass override type-checks), `fused_moe/routed_experts.py`,
`tests/quantization/test_modelopt.py`. Independent of the other PRs; base
`6e517b15c1`.

## Duplicate check (2026-10-05)

```bash
gh pr list --repo vllm-project/vllm --state open --search "W4A16 NVFP4 layerwise reload"
gh pr list --repo vllm-project/vllm --state open --search "modelopt W4A16 input_scale"
gh pr list --repo vllm-project/vllm --state open --search "weight_loader_numel input_scale"
```

No PR addresses reload of W4A16 layers.

## Test plan

```bash
pytest tests/quantization/test_modelopt.py
```

`test_modelopt_w4a16_reload_ignores_legacy_input_scale[late=False|True]`:
a toy model with one W4A16 and one FP8 `ReplicatedLinear` under a
mixed-precision config; after a layerwise reload the W4A16 layer is processed
before its legacy `input_scale` arrives (or ignores it when it arrives first),
the FP8 layer still waits for its `input_scale`, and the reloaded state dict
equals the cold-loaded one bit for bit.

## Test results

GB200 (one node), image `<image>` (the delivery-candidate image, sha256 `f9cae7eb…`) with this branch's `vllm/` files overlaid on the installed delivery tree (checked: pre-overlay files equal the delivery tree, overlay byte-equal to the branch, every changed module imports); tests copied from the branch. Script: `unit.sbatch` in the run's results directory.

| Test | Result |
|---|---|
| `tests/quantization/test_modelopt.py` | 38 passed, 3 skipped, 1 failed: `test_modelopt_w4a16_respects_linear_backend[auto-MarlinNvFp4LinearKernel]`, pre-existing on the base (being fixed by #59598); both new cases pass |
| `pre-commit run --from-ref 6e517b15c1 --to-ref HEAD`; `pre-commit run mypy-3.12 --hook-stage manual` (same range) | pass |

Not output-affecting (cold-load path and kernels unchanged), so no model eval;
end-to-end it is exercised by every weight update of the Nemotron-H RL runs
(sync check 0 mismatches over 291,472 tensors, 16 updates).

## AI assistance

Written with AI assistance (Claude). The human submitter must review every
changed line and rerun the tests before posting.

- [ ] Human reviewed every changed line and ran the tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)
