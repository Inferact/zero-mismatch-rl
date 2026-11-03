# Operations and handoff

Rules for long tasks, resources, review, job hygiene and handoff. Protocols for the agent workspace (an append-only event log, handoff notes, a provenance sidecar next to each artifact) are
assumed to come from your own agent tooling and are not repeated here.

## General conventions

A day-0 task is short and its scarce resources are the owner's attention and the allocated nodes. Most process rules below
follow from protecting those two.

- **Keep the critical path moving.** The critical path is the next gate, or during G6 the full-model run itself. Side
  investigations (a non-repeatability, a bisect, a benchmark, an optimization) run on separate nodes or in parallel and never
  hold it, unless they are a correctness gate the workflow names. Typical split on an eight-node day-0: all eight on the full
  run, one extra node for bisect and benchmarks.
- **When a step is done, start the next one.** Do not wait for a hold job to expire, a meeting, or a reply about something already
  decided. Model A once waited 15 minutes on an allocation that ran nothing but its placeholder before starting the G6 preflight.
- **An owner order is executed, not re-tested.** When the owner fixes a configuration, run it; checks that are not gates run
  beside it, and a problem they find is fixed at the next checkpoint boundary. Model A lost an hour re-testing an owner-ordered
  micro-batch and sleep-offload change when the owner had asked to go straight to the full eight-node run.
- **Nodes never sit idle.** When checking progress and seeing idle nodes, schedule parallel work on them.
- **Delegated agents get the boundaries up front.** Every brief states the day-0 boundaries
  (no determinism requirement, CUDA graphs on, explicit dtypes, no self-written engines). A requirement one agent invents is not
  forwarded to others as a constraint: in the Model A Megatron port one agent made deterministic algorithms the default and the
  orchestrator passed it to two more lanes before the owner reverted it, since nobody had asked for determinism.
- **When hitting a permission block or cost cap, stop and hand over to the human; do not bypass it.**
- **Report concisely,** one point at a time, with little jargon; list items that need human judgment separately. When an earlier
  claim turns out wrong, say so and give the corrected claim and its evidence; do not let the old claim stand by omission.
- One recipe one image, one seed per config, small load during development: rules in [experiment discipline](evidence-and-experiments.md#experiment-discipline),
  execution details in [job hygiene](#job-hygiene). Multi-party review and large PR splitting are covered below and in
  [code and delivery §9](code-and-delivery.md#9-prs-and-cleanup).

## Restoring context: verify task identity first

Read the project handoff, the [per-step artifact index](../guides/debugging-mismatch/06-artifacts.md#archive-experiment-materials-by-step), durable events and the latest actual logs; check the current candidate against the human-reviewed baseline ID; do not guess the config from scattered launchers or old summaries; then pin it on a small card:

```text
model / repo / target branch / image digest
training or inference-only; correctness or performance
policy checkpoint; requests / n / actual lengths
trainer groups; rollout groups; required GPU/node count
BI0 baseline; BI1 candidate; Graph/cache requirements
authorized jobs/nodes; stop condition; next decisive test
```

Historically, a small unrelated model task was once mistaken for a DS4 eight-node experiment, and a verified ARM image was once taken to be missing DeepEP. The fix is not to keep adding topology mnemonics, but to **verify the goal, object, preconditions and authorization with evidence before acting**: does this command change the variable currently under study? Do its inputs/versions really exist? Which explanation can its result rule out?

When information sources conflict, execution logs / loaded artifacts and raw results take precedence over old summaries, PR descriptions or what the assistant itself said. Reports can go stale; when you find this, keep the correction and its reason; do not infer that something should pass now because "I said earlier it passed".

## Resources and stop semantics

- Use the scheduler to query job ID, owner, nodes and status; use the necessary read-only checks to distinguish other people's jobs, your own old allocations, leftover containers and workers. A process on a shared machine does not mean you may kill it.
- Arrange per-group real resources only when the user authorizes running multiple groups in parallel; if two nodes can do DP8/EP8, there is no need to request eight nodes. Insufficient resources do not turn the task into a different model.
- Do not bypass the scheduler to sneak onto nodes. Directly using unmanaged machines requires user/project policy permission; record ownership and exclusivity checks.
- Release the specified job/container; do not use broad `pkill` / account-wide cleanup. After stopping, verify allocation, GPU process and container state; if anything is left, resolve its ownership first.
- "Stop at the first mismatch" means first persist minimal failure evidence, then stop that run; "stand still, do not stop the instance" means do not release it on your own. stop/resume is the user's intent, not something a timeout script guesses.
- During long waits, update real progress: completed stages, first error, next discriminating experiment. Repeatedly launching the same failing config is not progress; with no new evidence, change the probe instead of repeating promises.

## Dumps have a budget; the workspace is not a tensor store

Prefer saving input tokens, policy/version, and summaries and statistics aligned by semantic key; only after identifying the failing UID/layer/boundary save original-dtype tensors on a small scope. Record hook counts, effective token count and bytes written; set total size, sample/layer/rank counts, trigger count and expiry policy.

Rough capacity: `elements × dtype bytes × module boundaries × layers × ranks × capture count`. Full dumps of long 8K outputs, many layers and many replicas quickly reach tens of GB; the ~94GB and 12GB dumps that appeared in historical working directories are the result of having no budget in advance.

Large artifacts go in authorized scratch/artifact paths, not in the source root and not committed to Git. Material / small reports are archived in the agreed directories; observability code is kept as a small patch or dedicated source; removing the default heavy dump does not mean deleting diagnostic capability. Do not copy large weights into benchmark/recipe directories.

Before cleanup, list exact targets, type/size, whether it is the only recovery point, how it will be kept, and confirm as the user requires. A user's old "delete these" cannot be generalized into deleting the whole dirty worktree; first save modifications/patches that still need to be kept, and do not commit or reset them unless asked.

## Resume, eval and W&B

For checkpoint content, commit protocol, resume correctness validation and host memory, see [checkpoint, resume and host memory](../guides/checkpoint-resume-memory/README.md).

Resuming the same experiment keeps the real global step, run ID (if the platform's resume semantics allow), policy chain and config record. Merging historical curves keeps the source run/segment, deduplication rule and breakpoints; do not write back fabricated samples or stitch different policies into one continuous experiment.

When comparing reward, total token count or speed, use **the same step range** and the same statistical convention; keep training and validation separate. Check response length, actual token count, scheduling and profiler/initialization phases before explaining "it got faster after step 15"; do not guess performance optimizations from curves alone.

A ratio mean close to 1 does not prove every per-token ratio = 1; mask, sample weighting, clip, the new post-training policy and aggregation scope all affect the plot. Check the raw definition, effective tokens and maximum deviation; do not treat a mean or display precision as bitwise.

Eval binds an exact checkpoint. For step0, explicitly disable resume and look at the resolved config; for step10, confirm that 10 was really restored. Checkpoint retention may already have deleted 20/30; do not fill in with checkpoints from R3/baseline or another BI chain. Check historical validation records first; if neither exists, report the gap.

Check the effective resolution of the validation set: the DAPO-format AIME-2024 parquet is 30 questions each repeated in 32 rows; with greedy sampling the 32 copies give identical results, the score can only change in units of 1/30, and differences across 10 steps are drowned out. Switch to `do_sample=True` for avg@32, or remove the duplicate rows; do not point `VAL_FILES` at the training set (`val_batch_size` defaults to the whole file).

A confidential model uploads nothing to W&B; metrics stay in the file logger. Otherwise, W&B can stay out of the training job: the job keeps `WANDB_MODE=disabled`, and the login node incrementally replays from the file-logger jsonl, with a fixed run id + `resume="allow"`, appending only rows with step greater than already uploaded; multi-segment resumes are stitched by "if the later segment starts at step s, drop the earlier segment's records ≥ s", and the resume point and job are recorded in tags/notes. W&B steps can only increase; a recomputed old step will not overwrite. Read the key from a file only into the upload subprocess's environment, strip the env dict before upload, and after upload grep the artifact directory with the key file to confirm no leak.

Launching val_only does not mean the eval is done; require the result marker, actual question count / completed count, metrics written to disk and upload confirmation. Saving/evaluating every 10 steps is a separate config; do not infer from the total training step count that a ckpt exists.

## Multi-agent review and test harness

### Stage audit: no stage builds on unreviewed work

**Why.** A day-0 task moves fast through many small decisions, and one author, human or agent, repeats the same kinds of cheap
mistake: a conclusion that the evidence does not support, a check that would also pass on broken code, a pass rule stricter or
looser than the workflow, a change that reinvents an existing implementation or breaks a mode a later stage needs. A second
reader who starts from the evidence alone catches these in minutes. In Model A (2026-10) independent reviewers found, among others: the
recipe's `kv_cache_dtype: auto` had silently resolved to FP8 while the trainer did not simulate an FP8 cache; a commit credited
with fixing a batched-prefill error had not changed it at all (worst error 5.104 nats before and after); and the orchestrator had
waived three owner-only preflight rows after seeing the data. The same task also showed what the author alone produces: a split-K
non-repeatability reported as a correctness bug and turned into a new default, and a hand-written quantizer whose operation order
differed from the serving kernel.

**Who.** In G0 the owner names the reviewers of record and confirms the granularity; there is no default. At least two distinct
reviewers, none of them the author. Independent means a separate agent that starts from the audit package alone: a fresh agent on a
different model or an external CLI counts; a fork of the author's session, or a sub-agent briefed with the author's conclusion, does
not. Fix the reviewers and the model's confidentiality together before the first package: in one task the first rounds went to external
reviewers on redacted copies before the model was restricted to local reviewers, and those rounds no longer counted.
Record the owner's wording in the decision record (Day-0: `run-plan.yaml`) and the author and reviewers in the card's `review` block.

**What.** Audited before it is used:

1. each stage verdict, before the card or progress records it as passed;
2. each batch of code changes, before a gate or long run uses it, including launcher and swap tooling;
3. each plan revision or pass-criteria change, before the data that depends on it is produced;
4. any root-cause claim that changes a default or the plan.

**Package.** One directory per audited item; for gates `artifacts/<stage>/<run-id>/audit/`, for code batches and plan revisions an
`audit/<subject>/` directory next to the gate that first uses them, linked from that gate's verdict. Contents: `MANIFEST.txt` (sha256 of
every reviewed file) or `CHANGE.diff`; `PROMPT.md` with what to attack; `R<n>-<reviewer>.md`, written by the reviewer itself (the
validator checks that every round has one per reviewer but cannot prove who wrote it, so an author-written report is a falsified
record); the author's `RESPONSE-r<n>.md`; `verdict.yaml` from [the template](../templates/audit-verdict.yaml). Large raw material
stays in the task workspace and is cited by path and hash; a web source is saved into the case, since only local files can be hashed.
`PROMPT.md` asks at least: does the evidence support the conclusion or only its wording; could the checks pass on broken code; which
required paths are untested; are the criteria stricter or looser than the workflow; does the change reuse the reference or serving
implementation or reinvent it; does it break another mode or a later stage.

**Rounds and closure.** Round 1 is independent. The author answers every finding in `RESPONSE-r<n>.md` with ACCEPT plus the fix, or
REJECT plus new evidence. In the next round each reviewer reads all reports and responses and, per finding of its own, withdraws or
maintains it. A finding is closed when fixed, or rejected and withdrawn by the reviewer who raised it. After three rounds, or earlier if
the author asks, remaining P0/P1 go to the owner, whose decision is recorded verbatim in `verdict.yaml`. P0 stops the dependent work at
once; P1 must be closed or owner-accepted before the verdict; P2 is recorded. `verdict.yaml` names the exact subject (run id, commit
range or plan revision, and the sha256 of what reviewers saw) and is `closed` only with zero open P0/P1; any later change to the subject
voids it. The per-file review below is how a PR's code is reviewed; its findings close under this rule.

**Owner-only findings are batched.** Many findings can only be closed by the owner: a pass line, a waiver, an accepted risk, a skipped
rerun. Collect them per gate into one decision list with the evidence for each option and put it to the owner in one message; record
the answers verbatim. Model A ran two to four rounds on each of twelve packages, and eleven still ended open, many on findings waiting for an owner
decision nobody had asked for, so no gate check became `verified` although the work behind them was done.

**Audits run asynchronously; they never block the work.** Start reviewers in the background as soon as a package exists and keep
working. A P0 interrupts the work it affects at once; P1/P2 are fixed alongside the next steps; work that does not depend on the audited
item continues, and nodes never wait for a review. Inside one allocation (Day-0 G1→G5) the next stage may run on top of an unclosed
audit; record it as provisional in progress and leave the card where it is: a check becomes `verified`, and the lifecycle advances,
only when its audit closes. If an audit fails, the provisional stages built on it are rerun. What waits is only the verdict.

**Reopening.** Only a confirmed P0/P1 that invalidates evidence, scope or pass criteria reopens a stage, through the existing `stale`
mechanism: set `stale`, name the finding and its audit path in `stale_reason`, and return to the earliest affected stage. Rejected
findings and P2 never reopen anything.

**Adoption.** A task already in flight adopts the rule at its next stage: move the card to schema 2, add `review`, and audit every
check that is already `verified` before the lifecycle advances. The validator's NEXT for a schema-1 card is this migration, and
schema-1 cards cannot reach `complete`.

**Confidential material.** A confidential model's packages go only to reviewers that run locally (agents run by this team, e.g. a fresh
agent on another model), never to an external service, however well redacted. For other private material, scan the whole package (no
truncation) for private model names, paths, checkpoint and job IDs, and credentials before sending it to an external service; replace
names with the case codename and send only at zero hits. Record the scan command and the hit count in the audit directory, never the
matched text. If a package cannot be made readable without private content, it goes only to local reviewers; if that leaves fewer
than two, the owner names a local substitute for that package.

Passing an audit does not transfer responsibility; see [minimal changes, contribution and delivery](#minimal-changes-contribution-and-delivery).

### Code review and test harness

- **Per-file review + adversarial verification.** Split review units by PR, two perspectives per unit (correctness; simplification, comparison with the reference implementation, PR text); each P0/P1 is handed to 2 independent refuters before the author acts on it; closure follows the [stage audit](#stage-audit-no-stage-builds-on-unreviewed-work) rule. Review runs in parallel with implementation, results are relayed as they arrive, and only P0 interrupts implementation. Re-review the changed scope after each round of fixes. Typical findings (Nemotron instance): PR body not matching the head (old SHA, nonexistent files, unfounded "byte-identical" claims), inaccurate recipe header comments, tests silently skipped, CI not running new tests, both arms sharing an output path.
- A review agent's "consistent with the reference implementation" is only a lead: before delivery, open the diff file by file yourself, compare against the reference implementation (e.g. DS4's vLLM / Megatron / verl delivery), and write down the reason each file is kept, deleted or rewritten; for every added check, guard or config file, first ask whether the reference implementation has it.
- **False green in the test harness.** GPU cases missing the `gpus` marker are skipped as CPU cases; when the conftest environment gate is unset the whole batch of GPU cases skips (see [preventing false green](../validation.md#preventing-false-greens)); a module-level fixture leaving a default NCCL group makes later checkpoint tests silently skip; a `gpus(4)` case that internally `mp.spawn`s 4 ranks runs 4 times under torchrun; in multi-rank fixtures a single-rank failure becomes an NCCL timeout hang, so exchange results with `all_gather_object`. The CI config must really include the newly added BI test files.
- Run tests in the target image, not a local venv (version drift produces false failures); distinguish HF download failures caused by offline clusters and cases that already fail on the target branch from real failures via baseline comparison, and state them.

## Job hygiene

- During development use a small load (e.g. 16 prompts × 4, output ≤2048, 3 steps) with unchanged criteria; only final acceptance runs the recipe's default load. One seed per config.
- During iteration, install the changed files on the existing image (copy them directly into the container; do not rebuild the image for each experiment) and then commit it as a dev image, with a manifest alongside (base image id, each repository's SHA, before/after hashes of changed files, full-tree content mismatch of 0), clearly marked as non-deliverable; the formal build from clean source is done only once, at delivery.
- Job scripts do not dump the full environment (tokens passed through from the host would land in logs); record only a whitelist such as `VLLM_*` / `NCCL_*` / `CUDA_*`.
- Start processes with `setsid` and stop only by process group; on shared nodes, same-named `ray::WorkerDict` and vLLM workers are by default someone else's. Observer subshells must not inherit `set -e`, or sampling exits entirely when a process disappears. Do not modify a bash script that is currently executing.
- Project conventions (adjust per user request): reports use the language the human uses; record human decisions verbatim by date in the decision record file; do not re-ask items already decided or delegated; items that can be closed out per the reference implementation's default are written as "per default, can be overridden" and work continues; items the workflow lists as human-only (train-inference consistency's [ask the human first](../README.md#ask-the-human-first), Day-0's G0 owner confirmation items) have no default; if missing, ask.

## Minimal changes, contribution and delivery

Change config through the entry script the user specified, avoiding drift between bootstrap5/7 and multiple ad-hoc launchers. In the end, consolidate one actually runnable launch entry, with defaults, necessary overrides and image/code pins clearly visible; do not treat exports that exist only in chat as deliverables.

Fixes go in the owner of the state/feature: vLLM backend/selector problems are fixed in vLLM first, not by wrapping a global monkeypatch in a model-specific training stack; a numerically matched forward must keep the real backward. If a reusable function already exists, call it; avoid building your own synonymous logprob metrics and compatibility switches.

For PR descriptions, conflict handling and cross-repository submission order, see [code and delivery §9](code-and-delivery.md#9-prs-and-cleanup).

You may request an authorized independent audit, but cannot hand responsibility over to the auditor. Before every declaration of completion, check: are all user goals covered? Do the results belong to the final code/image? Are untested items hidden by wording? If evidence is insufficient, write down the gap; do not dress it up with percentages or success markers.
