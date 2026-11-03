# Multi-version RL: making every step's rollout, replay, update and sync auditable

Applies when the task requires real policy updates, not just a forward comparison at fixed weights. First satisfy
[the proxy and full-model zero-update threshold](../../validation.md#full-model-admission-pass-proxy-train-inference-forward-first);
do not localize a θ0 difference that has not yet reached zero by launching multi-step training. If the user restricts the proxy to fixed-token
logprob only, the proxy need not run RL, but the zero-update comparison before full real training still cannot be skipped.

## This mistake and the delivery boundary

In this Nemotron experiment, training and inference were placed in two different images, and validation was done by saving
rollouts to disk, training replay and update, exporting weights and then cold-starting inference, yet the results
were described as acceptance of a complete RL loop. That evidence only proves numerical alignment of the tested policy versions and the offline
update chain; it does not prove that the target RL framework's resident rollout, online weight sync and runtime environment
integration are correct. This was an execution and reporting mistake, not a documentation requirement to use two images.

This workflow turns that experience into a general threshold: training and inference use the same immutable delivery image digest,
the same Python/Torch and runtime dependencies, may be deployed on separate nodes/processes, and complete at least two real updates through the target RL
framework's actual execution entry point; each round validates original-dtype logprob 0 diff before the update, the actual parameter update,
and new-policy logprob 0 diff after online sync. Cold start after a disk export
cannot replace online-sync acceptance; merely installing the two components in the same image does not mean the loop passes either.

Saved rollouts, offline replay, separate images and cold start can be used to localize problems, but must be explicitly labeled
as diagnostic evidence. Same-image is this workflow's delivery convention, not a technical limitation of all RL systems;
when the process does not cover some delivery requirement, add the validation; do not assume the diagnostic path
is equivalent to the production path. Cold-start loading and resident-worker reload have different state lifecycles and must
be accounted for separately; the former passing does not prove the latter passes.

## Each version is an immutable transaction

For each θk, produce in order four cross-checkable artifacts: the policy exported by the actor, the policy actually
loaded by the server and its rollout, the old logprob from the actor replaying that rollout **before the update**,
and the optimizer update and next-version export. It is recommended that the ready marker reference the artifacts' content hashes and the parent version
identifier; ready is published only after verifying that online sync completed, the weights actually loaded, score integrity, and artifact sealing.
Version switching does not require workers to exit: θ0 is loaded at initial startup, subsequent θ use the resident rollout
workers' native sync interface, and sampling may start only after completion receipts from all required ranks; normal exit
is accepted separately at the end of the whole run. Exit records from cold-start diagnostics cannot replace this online evidence chain.

Do not mix the training manifest SHA, the `policy_content_id` defined by the export receipt, and the rollout
receipt SHA into one ID. Separately verify that the manifest binds the running code, the export receipt
binds the shards, and the server actually loaded the shards, and only then hand the same-version scores to the actor. If the training
receipt stores both the manifest hash and the content ID, the report must state what each field means; do not misreport version misalignment just because two
hex strings differ.

| Stage | Questions that must be answerable | False pass |
| --- | --- | --- |
| Export θk | Which master/deployment parameters, scales, packs and shards were exported? Does the content ID differ from θk−1, and is the parent chain correct? | Only the version number increments while the server still reads old files |
| Initial load / online sync | What are the worker's actual source, image and backend? Do the sync receipts and loaded payload correspond to the current θ, and are derived weights and cache/Graph updated correctly? | Only the version string is updated, or a freshly created process cold-loads instead of resident-worker sync |
| Rollout sealing | Which tokens, original-dtype scores, masks and sampling config did the same policy generate? Are the requests and all expected worker receipts complete? | Only partial replies, or wrongly requiring workers to exit at the end of each version |
| Pre-update dual-path comparison | Are native old-logprob and the actual grad-enabled actor forward each compared against the rollout per the [dual-path contract](../../validation.md#dual-path-contract-for-old-logprob-and-training-forward)? | Empty mask, zero advantage skipping forward, diagnostic attention, or only the no-grad branch validated |
| Optimizer and sync | Are gradients finite and meaningful, did actual master parameters change, was the new export produced by this update? | Step counter increments but weights/deployed values not checked |

Do not directly call the θk rollout vs. the θk+1 actor score a train-inference comparison. To compare policies before and after an update,
freeze **the same token history** and do a separate cross-policy diagnosis; each version's formal consistency comparison must be
same-version policy, same token history, same valid mask. Low-precision deployed weights may have unchanged bytes at
some step because of quantization buckets, so master change, the new payload's content ID and server-side load proof must be
reported separately; "all deployed bytes unchanged" alone also cannot imply the optimizer did not work.

## Execution threshold of at least two updates

1. θ0 first does a zero-update real-training-mode replay; on mismatch, stop the optimizer, seal the first failing
   sample and narrow the operator boundary; do not generate a full five-step trajectory first. Record total valid tokens, original dtype,
   mismatch count, actual attention/MoE/quantization classes, and the state of every PP rank.
2. Only when the comparison passes and reward/advantage can produce a non-degenerate update, run the first backward,
   optimizer step and θ1 export. Check actual parameter/master shard changes, finite gradients,
   the version parent chain, and that all ranks participated normally; loss decrease does not replace this evidence.
3. Load θ1 into the rollout workers via the target RL framework's actual sync path, verify the weights the server
   loaded, save a new rollout, then replay on the θ1 actor **before the update**; only after that also passes,
   update to θ2. Online-sync tasks must keep resident workers and the necessary Graph/cache lifecycles;
   cold start may be done separately as a load diagnosis but cannot replace this threshold.
4. At minimum add a θ2 final sync probe: load θ2 via the same target sync path, save fixed tokens,
   replay on the θ2 actor. If the task requires five steps, keep repeating version by version; do not extrapolate the last three steps from the first two.

To save iteration time, on failure you may replay only the saved old-logprob forward, reusing the rollout's
tokens, scores, weights and state; do not restart a full rollout and training for every kernel change.
But after changing the inference generation policy, a new end-to-end acceptance must save **new trajectories generated by the new policy**.

## Accept the control plane and the arithmetic plane separately

Do four independent checks: source closure, runtime object identity, numerical comparison, and job lifecycle.
A dynamic overlay/script directory may import an old adapter first: write the actual class,
`__module__`/source file, key scales and hit counts of the target callsite into the receipt, and fail closed immediately
on any mismatch. Identical source hashes do not show that the target class was instantiated; the existence of a new overlay file even less
shows that the worker used it.

Control-plane parsing must match the actual RPC structure. For example, the top level returned by an RPC may be **one array,
and only inside the array are multiple worker receipts**; do not assume each top-level item is a worker. Validate the schema first,
then check worker count, request count, score count and sync completion status; exit status is checked when the whole run is closed or
in a separate cold-start diagnosis, not as a condition for advancing each θ. A coordinator receiving an expected SIGTERM after worker
results are sealed is not the same fault as a worker non-zero exit or unsealed files;
if the original acceptance script misjudges, keep the original receipts and the fixed producer hash, and recover with a separately versioned, auditable
acceptance correction; do not rewrite historical records or forge ready.

Each stage should ideally publish only one atomic ready marker; consumers consume after verifying the manifest hash,
version parent chain, policy content ID, file hashes and non-empty samples. Before retrying, first determine whether the stage
already completed: if the rollout is sealed, replay/resume the actor, do not re-sample; explicitly mark unconfirmed parts
`inconclusive`. Write time budgets and resource leases into the manifest; after a timeout, an old ready from an
incomplete stage must not be used to advance to the next θ.

The final state of parallel PP training must be checked on **every rank**; looking only at the last
rank, which saves logprobs, may miss optimizer, gradient or export failures on other ranks. For normal service exit, also
distinguish successful execution of workers/EngineCore from the coordinator's expected shutdown signal.

## Reporting conventions

Per version, list rollout candidate count, accepted count, valid token count, original-dtype mismatch,
reward/advantage validity, per-rank optimizer steps, parameter/master shard differences,
export and load content IDs, normal exit. Clearly distinguish completed from in progress; θ0/θ1 passing only proves
same-policy replay for the first two versions and two updates, and does not automatically prove θ2 reload, five-step stability, GSM8K
quality, BI=0/BI=1 performance, or the final image.
