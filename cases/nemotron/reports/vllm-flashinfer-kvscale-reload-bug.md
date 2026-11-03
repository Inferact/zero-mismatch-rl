# vLLM bug: FlashInfer attention caches the KV scale and does not refresh it after weight reload

Date: 2026-10-05. Found in the baseline-r3 smoke (B1/B2 rollouts were all garbage). Present in both upstream main (origin/main 1388100560) and the delivery tree. Draft, not submitted.

## Symptom

vLLM starts with `load_format=dummy`, the real checkpoint is written in via layerwise reload, and afterwards the model outputs garbage. Trigger conditions:
- the checkpoint carries static FP8 KV scales (Nemotron-H Lightning NVFP4: 6 attention layers, k_scale 0.032–0.34);
- `kv_cache_dtype=fp8_e4m3`, attention backend FlashInfer.

verl's rollout by default does dummy initialization followed by a first refit, so RL hits this as soon as it starts.

## Root cause

`vllm/v1/attention/backends/flashinfer.py` (origin/main lines 2058–2065) lazily computes and caches two values **on the first forward**:

```python
if self.bmm1_scale is None:
    self.bmm1_scale = self.scale
    if is_quantized_kv_cache(self.kv_cache_dtype):
        self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
if self.bmm2_scale is None:
    self.bmm2_scale = 1.0
    if is_quantized_kv_cache(self.kv_cache_dtype):
        self.bmm2_scale *= layer._v_scale_float
```

- Under dummy initialization `_k_scale_float` and `_v_scale_float` are both 1.0, and after profile or warmup runs the cached values are fixed at bmm1 = 1/√128 = 0.0884, bmm2 = 1.0.
- Reload goes through process_weights_after_loading in `kv_cache.py`, which updates `layer._k_scale_float` and `_v_scale_float` to the checkpoint values.
- But `impl.bmm1_scale` and `impl.bmm2_scale` are never invalidated, so K/V are dequantized with the wrong scale.

## Evidence (one node, TP1, eager, BI=0, delivery image)

| Scenario | Prompt NLL of the two texts | Greedy output |
|---|---|---|
| Normal checkpoint load | 1.018 / 0.840 | ' 156' (12×13) |
| Normal load, then refit with checkpoint bytes | 1.003 / 0.753 | ' 156' |
| Dummy init, then refit with checkpoint bytes | **11.40 / 11.72** | 'amp going safety come safety' |
| Same, but `impl.bmm1_scale` and `impl.bmm2_scale` set to None after refit | **0.987 / 0.875** | ' 156' |

Checks before and after refit:
- After refit, for both initialization modes, the byte checksums of all 714 parameters and buffers are identical, and all float attributes on modules and kernels are also identical (the cached ones are attributes on impl, not on the module).
- Per-layer forward hooks show that the first divergence is at the first attention layer (layer 5): the `qkv_proj` output is identical, and the `attn` output std changes from 0.031 to 9.36.
- All 6 layers cache bmm1=0.0884, bmm2=1.0, while `_k_scale_float` is 0.0317, 0.0494, 0.0645, 0.0770, 0.337 and 0.107 respectively.

Related files are in `$WORKDIR/baseline-r3/checks/`:
- scripts: `refit_check.py`, `refit_fn.py`, `refit.sbatch`;
- logs: `refit-ckpt-e0-s0-dummy-<job-id>.log`, `refit-ckpt-e1-s0-dummy-<job-id>.log` (RESET_BMM=1), `refit-ckpt-e1-s0-auto-*.log`;
- per-layer statistics: `hooks-auto.json`, `hooks-dummy.json`.

## Minimal reproduction (outline)

1. `LLM(<checkpoint with FP8 k/v scales>, load_format="dummy", kv_cache_dtype="fp8_e4m3", attention_backend="FLASHINFER")`;
2. Use `collective_rpc` to run `initialize_layerwise_reload(model)`, then call `model.load_weights(...)` on all checkpoint tensors, and finally `finalize_layerwise_processing(model, model_config)`;
3. Compare prompt NLL: after refit it is close to random; after clearing `impl.bmm1_scale` and `impl.bmm2_scale` it returns to normal.

Two more points:
- The native `reload_weights()` raises `NotImplementedError: Model reloading with dummy format` directly under the dummy format, so the reproduction has to go through the layerwise API. This is also the path that RL frameworks such as verl and NeMo-RL actually take.
- Not limited to dummy initialization: any reload that changes the KV scale hits it. For example, a QAT checkpoint with recalibrated KV scales, where the KV scale changes during a hot update.

`repro_flashinfer_kvscale_reload.py` is a version written with the native `reload_weights`; under the dummy format it is rejected by the NotImplementedError above and needs to be switched to the layerwise API (i.e. `refit_fn.py`).

## Proposed fix

Pick one:
- Compute from `layer._k_scale_float` and `_v_scale_float` on every forward; the overhead is only a Python float multiplication;
- Or set `impl.bmm1_scale` and `impl.bmm2_scale` to None in the layerwise reload finalize (attention layers, the attention-scale handling path in `layerwise.py`).

`flashinfer_mla.py`, `tokenspeed_mla.py` and `flashinfer_mla_sparse.py` have the same lazy caching pattern and need to be checked as well.

## Impact on the baseline and handling

- The baseline recipe was changed to `actor_rollout_ref.rollout.load_format=auto` (on a scratch verl branch): vLLM first loads the real checkpoint, so the KV scales are already correct at the first forward, and later refits do not change the KV scales (the exporter passes the checkpoint's static k/v scales through).
- The true-on-policy BI=1 path does not trigger it: it uses `NemotronHFixedFA4Backend`, which does not go through this FlashInferImpl cache.
