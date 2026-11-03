# VERL / Megatron integration

This file describes integration approaches already used in practice; it does not promise that any upstream version has the same config keys. First find, in the current checkout, the actual call paths for rollout dump, old-logprob and skip/repeat, and record the commit. Do not treat options from a local patch as upstream API.

Process content (recipe and PR delivery contract, DAPO reference and review materials, general two-step acceptance convention, E2E comparison targets) has moved to [validation.md recipe review materials and two-step acceptance convention](../../validation.md#recipe-review-materials-and-two-step-acceptance-conventions); ownership of the three-repo PRs is in [Code and delivery §8](../../practices/code-and-delivery.md#8-three-repository-delivery-from-experimental-implementation-to-three-prs-and-one-image). This file covers only the integration points.

## Where the integration layer is

First locate the existing integration layer in the chosen version, then decide where to change things. Megatron's `experimental/lite/examples/verl/verl_mlite/` provides a verl engine adapter (`model_engine=mlite`, `pkg://verl_mlite.config`); the external integration package can be installed normally into the unified image, and removing the runtime overlay does not require reimplementing the engine in the verl repo. Check the actual engine registration, model factory, batch conversion, loss normalization, optimizer and online weight sync paths, and list reusable interfaces and concrete gaps. Train-inference consistency uses mLite; the training baseline follows the default MCore path of the existing verl recipe; do not merge the two into one training scheme.

## Capture point

After the rollout batch returns to the trainer and before the first old-logprob, save the DataProto or the framework's equivalent batch; append trainer scores after comparison. Record policy version, UID, input/response IDs, mask, position IDs, reward and route replay data (if needed). If the framework reorders the batch further, also save the mapping from original UID to packed rows.

Recording only the rollout score is not enough: the weights of the failing step may disappear after the next optimizer update. Keep a pre-update checkpoint or a recoverable weight snapshot, and bind its content fingerprint to the batch. After strict loading, compare model structure and weight keys, and reject silent missing/unexpected keys.

## Historical DS4 entry

The local historical launcher used:

```text
skip.rollout.enable=true
skip.rollout.action=repeat
skip.rollout.dump_dir=<captured-cache>
skip.rollout.steps=<selected-steps>
DS4_SKIP_UPDATE=1
```

These came from the patched VERL/bootstrap of that time; do not assume the current version supports them as is. You must be able to find where `DS4_SKIP_UPDATE` is read and returned in both the code and the execution log; the env var being present does not mean it takes effect. The raw cache contains `gen_batch.dp` and metadata; it is framework input, not standard score JSON.

## Adapter minimal contract

The synchronous entry takes the raw batch, the matching checkpoint, variant and output directory; it exits after doing the following:

1. Load and record the actual policy, effective config, libraries and parallel layout.
2. Read the same rollout; adjust packing per the given variant, keeping the original logical UID.
3. Run the real old-logprob forward; restore UID order and export per-token logprobs (see [Saving and comparing](../debugging-mismatch/06-artifacts.md#compare-per-token-logprobs)).
4. Record no generation, no optimizer/update, and the initialization strategy for necessary mutable buffers.
5. Wait for this run's workers/collectives to finish, close resources it created, and return an exit code.

This is a no-update localization adapter that keeps the native old-logprob `no_grad`/`forward_only`
semantics; it does not replace the [independent comparison against the real grad-enabled actor forward](../../validation.md#dual-path-contract-for-old-logprob-and-training-forward),
nor acceptance of the real trainer's update/online sync.

A/B/C can run in one resident worker, provided weight and state isolation has been verified; the runner itself only calls the synchronous entry and does not take over Ray/Slurm. A remote scheduler must be blocked on by the project launcher, which handles timeout and reclamation, so the local process does not exit while the GPU job keeps running.

When injecting/overlaying the adapter dynamically, the receipt must also record the actual object's class, source file and key config, and assert that the target callsite is hit; the script directory may be imported before the overlay on `PYTHONPATH` (one Nemotron trace silently loaded an old attention adapter because of this).

## Entropy when old-logprob is not recomputed (bypass)

Bypass mode uses the logprobs returned by rollout directly as old-logprob, saving one forward. Two consequences: `actor/ppo_kl` measures exactly the difference between rollout and the training forward (should be constantly 0 under BI=1); entropy no longer comes from the no-grad old-logprob forward but from the update forward with gradients.

- **Put quantities used only as metrics explicitly under `no_grad`.** With `entropy_coeff=0` the loss still contains `0 * entropy_loss`; entropy stays in the graph, backward still runs through it, and a copy of the logits is saved for it.
- **The chunked entropy in the verl Megatron engine cannot do backward (upstream bug).** `entropy_from_logits_with_chunking=True` splits the logits into multiple views of the same tensor and computes entropy on each; each view's backward rewrites the saved logits in place and then restores them. The views share one version counter, so after the first chunk's backward, the logits saved by the remaining chunks are seen as modified, raising `one of the variables needed for gradient computation has been modified by an inplace operation`. verl by default does not chunk and does not compute entropy, so it is not hit; it always reproduces when long context requires chunking and entropy is also wanted, and the same with `entropy_coeff != 0` plus chunking. The fix is [verl-project/verl#8195](https://github.com/verl-project/verl/pull/8195) (chunks use a non-in-place backward). Before it merges: when `entropy_coeff=0`, the trainer passes a flag and the engine computes entropy under `no_grad` with the same chunking.
- **After changing how the forward is computed (chunking, no recompute), validate gradients once separately.** Gradients of Nemotron chunked logprob: in BF16, bitwise identical to unchunked; in FP32, max difference about 1e-8, with max gradient about 4.

## Training extension

Real backward validation should be independent of the localization loop. When sharing the inference forward with a native VJP, check activation checkpointing/recompute, CP metadata lifetime, PP payload and parameter registration. Do not bypass the standard training entry for validation and then claim the native trainer is done.

After a real update, go through the existing weight sync, verify sender/receiver policy and necessary derived buffers, then export a new rollout and new trainer scores. An old bundle can only be replayed with the old policy; output differences between old and new policy are expected change, not a train-inference consistency failure.

A training process that integrates vLLM-layer modules (FusedMoE, ModelOpt linear) must create vLLM process groups; `param_offload=True` moves parameters, changing the quantized master's `data_ptr` and making the pre-export check fail, so enable only optimizer offload (same as DS4); see [Necessary differences between Nemotron and DS4](../../../cases/nemotron/consistency-case.md#necessary-differences-from-the-ds4-approach).
