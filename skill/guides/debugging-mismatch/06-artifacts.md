# Save and compare protocol

Large tensors and checkpoints stay in the model's task workspace (`results/<run-id>/`, with a provenance sidecar per artifact); source code and small summaries go into version control. No general scripts are provided for comparing logprobs, per-layer traces and replay comparisons, because each framework dumps differently; write them in the project following the points below. Fixed-token comparison before/after weight updates, provenance sidecars and source snapshots were implemented in a companion toolkit (`<toolkit>/`, not included in this repo); the table lists what each script does so you can write the equivalent:

| Need | Script |
|---|---|
| logprob/argmax comparison of the same fixed tokens before and after a weight update (fixed-token oracle) | `<toolkit>/scripts/vllm_weight_update_client/validation/fixed_token_oracle.py`, `compare_fixed_token_oracle_sequence.py` |
| GSM8K record comparison before/after update | `<toolkit>/scripts/vllm_weight_update_client/validation/run_gsm8k_chat_eval.py`, `compare_gsm8k_records.py` |
| `.meta.json` provenance sidecar for the results directory | `<toolkit>/scripts/vllm_weight_update_client/provenance/write_run_sidecars.py` |
| Before an experiment, pin uncommitted changes as a snapshot commit + clean worktree, verify the container's actual import paths and file hashes | `<toolkit>/scripts/rl_source_snapshot.py {prepare,verify,render-check,image-digest,cleanup}` |
| Trainer-less NCCL weight update client and pause/reset/sleep/wake wrappers | `<toolkit>/scripts/vllm_weight_update_client/run_vllm_weight_update.py`, `run_vllm_lifecycle.sh` (see [weight update](../weight-update/README.md)) |
The manifest records artifact path, size, hash, generating command, code/image/policy version, number of valid samples and hook execution count. Confirm the write to disk succeeded before cleaning up compute processes. Do not automatically delete existing artifacts.

## Archive experiment materials by step

Create a step index before the experiment starts, and fill in status and evidence promptly after each run ends; do not wait until the task ends to organize. Reuse the project's existing workspace; below is an example of logical categories — it does not require moving historical files or copying large artifacts; existing materials can be linked by explicit paths:

```text
<task-workspace>/
  steps/
    INDEX.md                     # current step, evidence entry per step, gates not passed
    01-baseline/
      INDEX.md                   # tuning comparison table, candidates and review records
      <run-id>/
        config/                  # resolved config, env vars, version manifest
        scripts/                 # actual server / bench commands or pinned-version entry
        results/                 # raw results / logs, or persisted artifact manifest
        summary.md               # conclusion, limitations, failure reasons and next experiment
    05-trainer-forward/           # categorized by C0–C10 gate number, create as needed
    10-final-acceptance/          # references baseline ID and this candidate version
  results/<run-id>/               # large tensors / checkpoints, not in Git
```

- **Every run is traceable.** Save the actual launch command, full effective config, code/image/checkpoint identity, nodes and runtime, log/probe evidence of the actual backend, exit status and raw result paths; a backend specified on the CLI does not mean that backend actually executed. A branch name is not provenance: record branch, snapshot commit, base SHA, patch, tree hash and image digest (kit `rl_source_snapshot.py`), and generate `.meta.json` for the results directory with kit `write_run_sidecars.py`. Do not write secrets into the archive; replace sensitive values with references. Source / small summaries go into Git; large files go to persistent artifact storage with size and hash recorded — do not leave only a path that disappears when the node is released.
- **Baseline tuning: one row per run.** The comparison table lists at least run ID, candidate configs such as MoE/communication, actual backend, load and concurrency, repeat rounds, valid input/output tokens, cache hits, throughput/latency, exit status and evidence link; mark inapplicable fields N/A. Keep failed, incomplete, diagnostic, and protocol-mismatch-excluded runs too, with reasons; the summary states explicitly which run IDs were included — do not cherry-pick the best round or keep only the final mean.
- **A candidate is not a reviewed baseline.** The step index separately marks the currently recommended candidate and the human-confirmed baseline ID; the latter links the exact config/image/code, the result set used and the review record; if not confirmed, write "pending review". Later performance comparisons reference that ID; do not replace the denominator with new contents of a same-named script or old results from another backend.
- **Reruns keep history; changes have boundaries.** New runs use new IDs and do not overwrite old configs, logs or raw results; mark invalidated conclusions with the reason and the replacing run. When continuing the same experiment, keep separate attempt/segment and exit records; when the same evidence is reused across steps, reference the original — do not copy it into another proof of passing.
- **Handoff starts by reading the index.** Each step's record is updated by the person responsible for that step; reporting, recovery and acceptance first check raw evidence via the index; missing files, mismatched identity or chat-only conclusions are marked as evidence gaps and cannot be used to declare a pass. For historical materials, add to the index first; do not delete or bulk-move them on your own.

## Replayable raw data

Keep the real framework batch: prompt/response IDs, attention/response masks, positions, lengths, UID/group/sibling, rollout logprobs, necessary reward and routing records; additionally save the packing mapping, parallel config, and how to restore the weights corresponding to that step's rollout. The framework's raw batch is the replay input; the per-token logprobs exported below are only for comparison and cannot replace the full training input.

The rollout at step N usually comes from the policy before update N. Saving only the checkpoint after the step-N update may not reproduce it. Under partial rollout or async staleness, different segments of one sequence may be generated by different policies (after interruption, re-prefill and continue under the new policy); record the policy version per segment — replaying the whole sequence with a single checkpoint would treat expected policy differences as inconsistency. Freeze weights, mutable buffers, RNG/dropout state, and state whether the scorer uses eval or training mode.

## Compare per-token logprobs

Each side (rollout vs training old-logprob, or two replays) exports a `{request_id: (logprobs, response_mask)}`, saved with `torch.save` as original-dtype tensors; do not convert to JSON or reconstruct from printed logs. Key points:

- **Align by logical request, not by physical row.** First remove padding and restore the UID order before packing; the request sets must be identical on both sides — missing ones must raise an error, do not compare only the intersection.
- **Compare only valid response tokens.** Positions outside the mask do not participate; each entry needs at least one valid token; an empty batch counts as an invalid experiment.
- **Bitwise comparison looks at bytes, not values.** Numeric comparison treats `+0.0` and `-0.0` as equal and two NaNs as unequal; compare integer views: `a.view(torch.int32)` (float32) or `a.view(torch.int16)` (bf16). Do not cast first and then claim bitwise consistency in the original dtype.
- **Same contract on both sides.** Same policy version, same raw/processed definition (for processed, record temperature, truncation, penalty), same input; compare hashes of the input tokens before comparing scores — differences from different inputs are not numerical differences.
- **Report** valid token count, number of byte mismatches, the first mismatching (request, position), max/mean absolute error. Consistent sampled-token logprobs do not mean consistent full logits.

```python
def compare_logprobs(a, b):
    assert a.keys() == b.keys(), "request sets differ"
    n = bad = 0; first = None; max_err = 0.0
    for uid in sorted(a):
        (x, mx), (y, my) = a[uid], b[uid]
        assert torch.equal(mx, my), f"{uid}: masks differ"
        x, y = x[mx], y[my]
        ix = x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
        iy = y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)
        diff = (ix != iy).nonzero()
        if len(diff) and first is None:
            first = (uid, int(diff[0]))
        n += len(x); bad += len(diff)
        max_err = max(max_err, (x.double() - y.double()).abs().max().item())
    return {"tokens": n, "mismatched": bad, "first": first, "max_abs": max_err}
```

## Per-layer trace: find the first diverging boundary

After a logprob mismatch, record tensors with forward hooks at the same set of semantic boundaries on both sides (e.g. each layer's attention output, MoE routing, residual), find the first differing boundary in execution order, then do a bounded dump there. Key points:

- **Keys are semantic, not rank-based.** Each record's key is `(request_id, boundary name, logical slice)`, e.g. `("step7-p2-s3", "layer2.attn.out", "tokens=0:128")`; first reassemble TP/CP/EP shards into global coordinates and remove padding, then record. Naming and order must match on both sides; a missing probe invalidates the whole experiment and cannot be skipped.
- **Store hashes first, tensors on demand.** Take SHA256 of the contiguous tensor's original-dtype bytes (`hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()`), also recording shape, dtype and semantic execution index. A hash only tells you where it differs, not by how much; after finding the first differing boundary, store raw tensors only there to look at the error.
- **Clone before buffer reuse.** Inference-side workspaces and CUDA Graph output buffers get overwritten by the next call; in the hook, `clone()` before hashing or saving.
- **First difference is not the root cause.** To establish an operator as the source, you also need to prove its inputs are identical on both sides (record its input hashes too).

## Replay comparison

Fix the rollout and the weights, change only one variable (packing order, batch composition, parallel layout, kernel switch), replay old-logprob, and use the comparison above to judge whether that variable affects the result. Key points:

- **One process per variant**, using a synchronous entry point (blocking until replay completes); do not use background submission like `sbatch` — on timeout or failure you cannot reclaim the job. Variants involving compilation, library or initialization environment changes must start a new process.
- **Each variant writes to a new directory**, not overwriting old results; record the actual command, environment, exit code and elapsed time. Preview the command and budget before executing; on failure, stop at that variant.
- **Prove the replay is really frozen**: record the weight fingerprint, confirm no generation or optimizer updates, confirm the packing actually in effect; do not rely only on command-line arguments.
- Record experiments with [consistency-experiment.yaml](../../templates/consistency-experiment.yaml).
