# Weight sync

Goal: bring per-step `timing_s/update_weights` on the full model (EP64 trainer, one DEP4 rollout per
node) from about 106 s to under 10 s while keeping BI=1 train-inference agreement bitwise. The
changes are Megatron #267 and two vLLM loader changes.

## How the sync works

verl's naive colocated backend: each trainer rank's `get_per_tensor_param` → mLite
`export_hf_weights` → a prefetch thread fills 2 GiB buckets → ZMQ + CUDA IPC to the vLLM worker on
the same GPU → compat `load_quanted_weights` (clone + `model.load_weights`, vLLM layer-wise reload).
No NCCL transfer between trainer and rollout; the only cross-node traffic is the EP all-gather inside
the export.

The export quantizes with the same quantizers the trainer forward applies (`to_release`), so the
rollout holds the trainer's per-step fake-quant weights byte for byte.

## Where 106 s went

On the DAPO long run, `update_weights` took 105.9–109.7 s per step; `rollout.update_weights` (export
generator + CUDA IPC + vLLM load, timed on replica 0 rank 0) took 102.1–103.1 s; the remaining
about 3.7 s were wake (weights / KV cache), two `aggressive_empty_cache` calls and waiting between
ranks.

The export all-gathered **all 384 routed experts** over EP64 in BF16 to **every** trainer rank; each
rank quantized all of them to MXFP4 (per tensor: quantize → dequantize → quantize, two
`torch.equal`) and sent the whole set to its vLLM rank, which kept only its quarter (96 experts)
and discarded the rest inside `load_weights`.

Profile on a 1-node proxy (40 layers + 3 MTP layers, experts 0..7, trainer EP4/CP2, rollout DEP4,
recipe settings, batch 16 × 8, response 512, 3 steps), per rank, steady state (3rd and 4th sync):

| Component | Count | Time | Data |
|---|---|---|---|
| `timing_s/update_weights` (flashmla / mega) | — | 8.0–8.9 / 7.7 s | — |
| Trainer-side session | 1 | 4.7–5.7 s | 15.7 GB sent |
| MXFP4 quantization (11.8M elements per tensor) | 960 | 2.0 s (2.1 ms per tensor) | 22.6 GB BF16 in |
| FP8 block quantization (10 dense kinds) | 330 | 0.46 s | 13.7 GB |
| EP4 all-gather (NVLink within the node) | 40 | 0.15–0.2 s | 22.6 GB |
| vLLM `load_bucket` (clone + `load_weights`) | 85 buckets / 3134 tensors | 1.7 s (0.54 ms per tensor) | 15.7 GB |
| vLLM layer-wise processing | 519 layers | about 0.4 s | — |
| vLLM `wake_weights` (level-2 remap) | 1 | 0.3 s (one rank 1.3 s) | — |
| `wake_kv_cache` | 1 | 0.11 s | — |

On the vLLM side the per-tensor cost is the layer-wise reload wrapper (about 0.6 ms per tensor), of
which about 0.25 ms is a dry run of the loader on meta tensors under a TorchDispatchMode to count
loaded elements, plus a whole-model `named_parameters` per bucket (4 ms). MegaMoE expert layers were
put on meta, every expert tensor was cloned into a cache, and the layer was materialized, loaded and
copied back once complete.

Extrapolation to the full model, per rank: 40 layers × 384 experts × 3 matrices = 46,080 MXFP4
tensors at 2.1 ms ≈ 97 s; BF16 experts 384 × 40 × 35.4M × 2 B = 1.09 TB, of which each rank receives
1.07 TB over InfiniBand (`NCCL_MNNVL_ENABLE=0`) at 40–50 GB/s per GPU ≈ 21–27 s, serialized with the
quantization on one thread; vLLM load about 93.4k tensors × 0.5 ms ≈ 47 s (three quarters are
non-local experts that pay the full wrapper cost before being dropped), overlapped with the producer.
Quantization + gather match the measured 102 s.

## Changes

Megatron (#267):

| Change | Effect |
|---|---|
| Send each expert only to the rollout rank that holds it: the expert's owner quantizes its local experts (no EP gather); per (layer, w1/w2/w3) one `all_to_all` sends only the MXFP4 codes and scales to the trainer rank that shares a GPU with the vLLM rank owning that expert (vLLM linear placement, rollout EP rank = rank % rollout EP). vLLM-side compat asserts it receives only local experts and exactly a full layer. Switch: launcher `SYNC_LOCAL_EXPERTS=True` → `VERL_MLITE_SYNC_ROLLOUT_EP` (off by default) | Quantization 46,080 → 720 per rank; BF16 all-gather of 1.07 TB → MXFP4 `all_to_all` of 72 GB; vLLM receives about 24.3k instead of 93.4k tensors |
| Encode all local experts of a projection in one `to_release` call (MXFP4 blocks do not cross experts; already canonical slices are byte-stable over extra rounds) | Quantization calls 720 → 120 per rank |
| `gc.collect(); gc.freeze()` at the end of verl engine `initialize()` (`VERL_MLITE_GC_FREEZE=0` disables) | verl runs 3 `gc.collect` per sync inside two `aggressive_empty_cache` calls, about 0.95 s each |

vLLM:

| Change | Effect |
|---|---|
| Layer-wise reload gains a per-layer `reload_in_place` switch; `DeepGemmMegaMoEExperts` turns it on (its loader writes the final kernel layout directly, with no post-load processing) | Experts are no longer put on meta, clone-cached, dry-run counted and materialized and copied back |
| The DS4.1 `load_weights` indexes the expert mapping by `experts.<id>.w<k>.` instead of substring-matching 1152 entries | 40 µs per tensor at 384 experts, about 1 s per rank on the full model |

verl unchanged.

## Proxy before / after (40 layers, 8 experts, 1 node, mega, steady state, per rank)

| Component | Baseline | Owner-side experts | + batched encode + vLLM changes | + `gc.freeze` |
|---|---|---|---|---|
| `timing_s/update_weights` | 7.7–7.8 s | 6.0–6.1 s | 5.8 s | **2.1–2.6 s** |
| Trainer session (export + IPC + wait for vLLM) | 4.7–5.7 s | 2.9–3.0 s | 2.6–2.7 s | 1.9 s |
| MXFP4 quantization | 960 calls, 2.0 s | 240, 0.69 s | 120, 0.47 s | same |
| FP8 quantization | 330 calls, 0.46 s | 0.46 s | 0.46 s | 0.46 s |
| vLLM receive | 3.3–3.4 s, 3134 tensors | 1.6 s, 1694 tensors | 1.4 s (expert layers bypass layer-wise) | 1.55 s mean |
| vLLM cost per expert tensor | about 0.6 ms | same | 0.115 ms (loader writes directly) | same |
| `aggressive_empty_cache` × 2 (3 × `gc.collect`) | about 2.9 s | about 2.9 s | about 2.9 s | 0.05 s |
| `wake_weights` / `wake_kv_cache` | 0.31–0.35 / 0.11 s | same | same | 0.33 / 0.11 s |

Sparse FlashMLA arm: baseline 7.7 s → all changes except `gc.freeze` 5.8–6.0 s.

Full model measured afterwards: `update_weights` 8.4–9.5 s per step on 16-node DAPO and SWE-bench Pro
runs with `VERL_MLITE_SYNC_ROLLOUT_EP=4`. Runs that forgot to set it (the SWE launcher was forked
before the switch existed and did not forward it) stayed at 87–89 s with the same code.

## Agreement

- Every arm, θ0/θ1/θ2: `rollout_probs_diff_max` = 0.
- vLLM full-state fingerprints at every level-2 sleep (1265 target + 100 drafter entries, scratch
  excluded): baseline vs each change, 4 ranks × 4 sleeps, all identical; between sleeps 2 and 3, 288
  non-expert weights changed (the fingerprint is sensitive).
- A non-trivial cross-rank exchange (actor EP2, lr 1e-4, so a rank needs another rank's experts):
  baseline vs all changes, 32 fingerprint files, 0 differences; synced experts differ from the cold
  load in 160 entries, the same in both arms. Within two steps at lr 1e-4 the MXFP4 expert codes do
  not flip (weight changes are far below the FP4 step), so θ1/θ2 expert bytes are covered by a CPU
  unit test: gloo, 4 ranks, EP4, rollout EP 2 and 4, different random values per rank; the sharded
  export equals the filtered full export byte for byte.
- vLLM unit tests: the new in-place reload and mapping-index tests pass; `test_reload.py` 89 passed, 2
  failed (a small dense model's engine start and a Llama FP8-KV reload perplexity), neither of which
  uses MegaMoE or `reload_in_place`.
- No forward code changes, so BI=0 and BI=1 take the same path.
