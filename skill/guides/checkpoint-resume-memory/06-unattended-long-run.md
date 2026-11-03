# Unattended long runs: hangs, time limits, node loss and code changes must all be handled on the cluster

Formal long runs last tens of hours, job time limits are usually one day, nodes fail, and nobody watches in between. An alert sent to
someone who cannot reach the cluster right now is as good as no alert. This page covers what a daemon on the cluster does by itself
("detect → preserve the scene → stop → resume from the latest checkpoint"), how code is changed during a run without losing the
lineage, how the allocation is held, and what node failures look like. For correctness of saving and resuming themselves see
[01](01-checkpoint-and-resume.md); for resume pitfalls with node-local checkpoints see [03 §14](03-pitfalls.md#14-resume_modeauto-starts-from-step-0);
for the shared disk see [05 §6](05-cluster-notes.md#6-shared-nfs-it-fills-up-and-it-can-be-overwhelmed).

## 1. What went wrong

Model D (2026-10, 4-node GB300, TIS training from step 0):

1. The weight sync after step 74 (verl `update_weights`) returned only 18 hours later (log `update_weights done, time cost: 65381.04s`), with no output at all in between, and then the job hit its time limit and was killed. verl's weight sync has no timeout.
2. In the same window the login node's sshd refused new connections (`Exceeded MaxStartups` during handshake); compute nodes can only be reached via the login node, so all local monitoring and polling disconnected. The watchdog on the login node did report "no new step for 70 minutes", but nobody received it.
3. After the time limit, the auto-resume script misparsed the step count from the checkpoint pointer (§4), and the submitted job failed in 12 seconds; only after fixing it did the run resume from step 70.

Another LoRA training run on the same cluster (also verl + vLLM level-1 sleep) hit a "silent hang during weight sync" twice, possibly the same mechanism: one worker stuck in the `cumem` `cudaMemcpy` of vLLM level-1 sleep, GPU utilization 0, no error, job not exiting, and the stuck node's `/proc/pressure/memory` full avg about 22% (0 on other nodes). Root cause not found; it reproduced after changing nodes, so it is not a bad node.

Model A (2026-10, 8×4 GB200, two full-model training runs of about two days in total, FSDP then Megatron): the runs went through 20 segments. Node
failures ended six of them (§7); the rest were planned code swaps (§5), owner-ordered configuration changes, startup checks that
refused a mismatched configuration, and launches refused because a node was not usable. Both runs kept one checkpoint lineage throughout,
because every one of these events went through the same stop-and-resume path.

## 2. What the daemon does

The daemon runs on the cluster (login node) and does not depend on a local ssh session. Loop:

```text
pick a usable hold job → per-node prep (GPU-clean barrier, checkpoint staging, disk) → launch one segment
  (resume from the latest COMPLETE checkpoint in the shared mirror) → check the resolved resume step
→ read the global step from the trainer log every 2 minutes
   ├─ job ended (time limit, node failure, requeue) → back to "pick a usable hold job"
   ├─ no new step beyond threshold                  → preserve the scene (§3) → stop → back to "pick"
   ├─ STOP file present                             → stop at the next safe point, exit
   └─ target step reached                           → exit
restart cap reached, or no progress over N restarts → stop and wait for a human
```

- **Thresholds.** Alert and hang criteria for single events (sync, save) follow the checkpoint and weight sync time table of the day-0 performance-tuning workflow (not finished after 5x the baseline counts as hung). The daemon additionally sets a per-step threshold with margin for validation and save steps: Model D normally takes about 17–20 minutes per step, threshold 100 minutes; the first step after resume includes loading and pre-training validation, so it is set to 180 minutes. Model A used 40 minutes without step progress. Startup deadlines come from measured startup on the same configuration: a Model A slice engine took about 504 s to come up with the FlashInfer MoE autotuner and missed a 600 s default health deadline; the full model took about 15 minutes.
- **One frozen run identity.** At start, copy every code input (trainer, framework, model directory, serving overlay) into the run
  directory, hash it into a tuple file, and run only from those copies (before runs had their own copies, Model A's shared overlay files were
  replaced under running containers twice); every rank checks the files it actually imported against the
  tuple ([runtime provenance](../environment-provenance/02-runtime-provenance.md#prove-the-numerical-path-ran)). Startup checks compare
  the resolved engine arguments with the tuple's baseline and refuse to train on a mismatch: Model A twice refused an owner-ordered change
  until its new baseline was written into a new tuple. A baseline changes only through a new tuple with an `events.log` line saying
  why; an in-place edit of the baseline lets the operator move the guard that checks them (an audit finding on Model A's hand reset).
- **GPU-clean barrier before every launch.** A crashed job frees GPU memory more slowly than Slurm reschedules, and a dead peer can leave
  memory on the survivors: after Model A lost a node, the remaining nodes held 1.5–2.9 GiB per GPU with no process, and later one node held
  4.4 GiB. Check every node's per-GPU used memory before launch; above the threshold, wait or exclude the node, never launch on it.
- **Checkpoints go to a shared mirror.** Ranks write node-local checkpoints; the daemon mirrors each complete one to the shared
  filesystem and resumes only from a mirror copy that has its completion marker, so a lost node does not take the latest checkpoint
  with it. Megatron multi-node checkpoints must be merged there before they can be read at all ([03 §14](03-pitfalls.md#14-resume_modeauto-starts-from-step-0)).
- **Read the mirrored logs.** Runtime logs are on node-local disk ([05 §6](05-cluster-notes.md#6-shared-nfs-it-fills-up-and-it-can-be-overwhelmed)); the daemon reads the copy mirrored to the shared disk; if the mirror stops updating that also counts as "no new step".
- **Only touch your own jobs.** Cancel only by the job id recorded at your own submission; do not `pkill` by name ([05 §7](05-cluster-notes.md#7-other-cluster-pitfalls-one-line-each)). In Model D, `pkill -f '<pattern>'` matched the ssh command running it and killed its own session.
- **One line per event** in an append-only `events.log` under the shared results directory: launch (job, nodes, resume step, tuple),
  stop and its reason, swap, restart count, human actions with their reason. When a human comes back they read this first.
- **Monitors key on (segment, step).** A segment that resumes from an earlier checkpoint replays step numbers; Model A's tripwire keyed on
  the step alone and skipped four replayed steps.

## 3. On a hang, preserve the scene before cancelling

After cancelling you can see nothing. Before cancelling, use `srun --jobid <job> --overlap` to collect on every node:

- `nvidia-smi` utilization and GPU memory (a hang usually shows utilization 0);
- `free -g`, `/proc/pressure/memory` (the hang of the LoRA run in §1 coincided with host memory pressure);
- processes sorted by CPU;
- grab stacks if possible: when the node has `ptrace_scope=1`, attaching directly from the host does not work; use py-spy inside the container, called via the process root: `/proc/<pid>/root/<venv in container>/bin/py-spy dump --pid <host pid>` (put the script on the shared disk and run it with the `srun --overlap -w <node>` above). Look at which call it is stuck in, in the order TaskRunner → EngineCore → each worker.

Write the scene together with the last few dozen lines of the trainer log into the results directory, then cancel.

## 4. The daemon script's own pitfalls

Each of these made an auto-resume fail or give false alarms, and none of them report an error:

| Pitfall | Symptom | Fix |
| --- | --- | --- |
| Background process is not a login shell | `nohup bash x.sh &` has no Slurm PATH, `squeue` is not found, returns empty, and is judged "job disappeared" | `nohup setsid bash -lc 'bash x.sh' > log 2>&1 < /dev/null &` |
| One task of a multi-node srun probe exits non-zero | Slurm terminates the whole step, the node that had output is killed too, the function returns empty | Append `; true` to the probe command |
| Extracting a number from command output | `head -1 \| tr -dc 0-9` picked up the digits from the error message `cat: <checkpoint dir>/...: No such file`, yielding `0313120`, which was submitted as the resume step | Give the output a fixed prefix, anchor the whole line with `sed -n 's/^LATEST=\([0-9]*\)$/\1/p'`, then check it is not below the expected lower bound |
| Trusting only your own computed step | The launcher may resolve a different `resume_from_path` | After submission read the resolved config; cancel if `resume_from_path` is not the expected `global_step_<N>` |
| Watching the job over one long-lived ssh connection | A disconnect means exit 255, and monitoring disappears with it | Run the daemon on the cluster; locally only poll its output files over short connections |
| A requeued job is treated as gone | Slurm requeues the hold job onto other nodes after a node failure, but the daemon waits only for the next job in its list (Model A, twice) | A requeued hold job stays a candidate; check its new node list and run the per-node prep again |
| Environment passed by whitelist | A variable the launcher sets on the host never reaches the container (`enroot -e` passes only listed names); Model A's resume-pair check ran with a default checkpoint path and trained from scratch | Put run parameters on the command inside the container, and check the resolved value in the log |

## 5. Changing code during the run

A long run will want new code: a speedup, a fix, an owner-ordered setting. The run's evidence holds only if every step can be tied to
the code that produced it and every change can be undone.

1. **Stop at a checkpoint.** Write the STOP file atomically; the daemon stops after the next COMPLETE checkpoint and records it.
2. **Anchor before risky changes.** Hard-link the checkpoint into a read-only anchor directory outside rotation (Model A kept 40, 45, 50
   and the final 105 of the FSDP run; step 40 was kept permanently at the owner's request in case packing caused problems). Rotation never reaches an anchor.
3. **Stage, then apply.** Stage the new code copies, model directory, overrides and serving-overlay files next to the old ones, write a
   new tuple, and apply only while no step runs. Apply all or nothing: a half-copied snapshot is rolled back, and a failed swap is
   recorded in `events.log` and never leaves the run stopped silently. Keep the old tuple and copies for rollback.
4. **Gate the first steps after the swap**, against the steps before it: the first weight sync is byte-identical to the last sync of the
   previous segment (so keep a sender manifest at every checkpoint step; with a manifest only every 50 syncs Model A could not do this
   check after one swap); the per-rank runtime proofs pass; the train-inference diff stays in its previous range; host and GPU peaks are
   not higher; the change's own signal is visible (Model A: per-rank token counts after balancing, the shared-offload log line on every rank).
5. **Roll back by tuple** when a gate fails or the change is slower in production: Model A's packing swap at step 45 was slower at FSDP×32;
   when a node failure ended that segment at step 49, the run resumed from checkpoint 45 on the old tuple and steps 46–49 were discarded.

The change itself still needs owner confirmation and a plan revision when it touches a validated path
(see the change classes of the day-0 performance-tuning workflow).

## 6. Holding the allocation

The hold job is the run's machines; losing it costs more than any step.

- **Never `scancel` a hold job to move it.** Freed nodes go to other users' pending jobs at once. Model A cancelled a requeued hold whose nodes could
  not start steps and lost three of its nodes to other users' jobs; the run ran on four nodes for a while until the owner released their own
  debug node and more nodes came free.
- **Shrinking can be blocked.** Releasing some nodes of a hold job waits for every step on them; in Model A a lingering extern step kept
  one node and blocked the shrink. Cancel a hold only after its replacement is submitted.
- **Chain a follow-on hold before the limit.** With a one-day partition limit, submit the next hold with
  `--dependency=afterany:<current job>` and the same excludes as soon as the current one starts, and add it to the daemon's job list.
- **Exclude nodes that cannot run steps.** A requeued job can land on a node whose job namespace never formed (`srun` fails in
  `container_p_join` because the job's `.ns` does not exist); every launch fails in prep. Submit the replacement with `-x <node>` and
  record the node.
- **Respect standing allocations.** The owner's debug node stays held unless the owner releases it.

## 7. Node failures

A node that dies in a run is either a hardware fault, which strikes at random times, or something the run does to it, which strikes
at a particular phase. Only a phase timeline tells the two apart, so before blaming a node, the GPU or memory, line every death up
against what each node was doing in the minute before it.

**What Model A saw (a GB200 cluster, 2026-10).** Six nodes died in two days, each on a different node,
each silently: sampling stopped, IMEX on the peers lost the node 19–31 s later, Slurm marked it `DOWN Not responding` 4–8 minutes after
that, and it stayed down until an administrator rebooted it. All six died 18–62 s after a local checkpoint write had finished, in the
window where every node copied the new checkpoint to Lustre (4 threads, 256 MiB writes, `fdatasync` each) while vLLM woke and the
weights were synced. That window held 46 of the 243 wakes and all six deaths; the other ~1,576 node-wakes had none (p ≈ 3.5e-5). Per
checkpoint the chance of losing a node was about 13%, so a 5-step checkpoint interval cost two to three nodes per 100 steps.

**The likely mechanism (about 60%).** The Lustre 2.17 client turns writes and reads above its hybrid-IO thresholds (2 MiB write, 8 MiB
read) into direct IO, which pins user pages and builds large bulk RPCs. That path failed with EFAULT 23 times on 11 nodes during the
copies, and earlier silent deaths on the same cluster followed the same pattern with no RL in them: a 1.3 TB checkpoint written straight
to Lustre (two nodes, 2026-10-22) and a 1.5 TB Hugging Face download to Lustre without any GPU use (two nodes within a minute,
2026-09-14). The nodes run with `libcfs_panic_on_lbug=1`, `kernel.panic=0`, `nmi_watchdog=0` and no kdump, so a Lustre LBUG or a
kernel lockup stops the node without a log. GPU and C2C counters were clean (a GPU-side cause about 20%), and memory does not fit: the
FSDP deaths happened at 63% of the cgroup limit, while Megatron nodes at the limit survived (about 10–15%). Our first hypothesis, the
wake's host-to-device copies plus host memory pressure, was wrong; ordinary wakes were identical in time and never failed.

Rules:

- **Large writes to the shared filesystem are a node risk, not only a disk cost.** Write in pieces below the client's direct-IO
  threshold (≤1 MiB writes, ≤4 MiB verification reads, keeping `fdatasync` and `fadvise`), or have administrators disable hybrid IO.
- **Keep shared-filesystem traffic out of the GPU transition window and stagger it.** Start the mirror after the next generation phase
  has begun, one node at a time with an offset, instead of all nodes during sleep, wake and sync.
- **Checkpoint frequency is a risk knob.** Each save is an exposure; choose the interval with the measured per-save loss rate next to
  the steps a loss costs.
- **Make the next failure attributable.** Ask administrators for kdump, `kernel.panic=10` (reboot and leave a console record),
  `nmi_watchdog=1` with lockup panics, and the kernel journal, console history and Lustre server evictions for each death window.
  On our side: log mirror start and end on every node, write a heartbeat line before calling `nvidia-smi` and give it a timeout (a hung
  driver and a dead node then look different), and sample at 1 s from the checkpoint write to the end of the sync, including dirty and
  writeback pages, IO pressure and the Lustre client's dirty bytes.
- **Not a bad-node list.** Excluding nodes helps only when deaths repeat on the same node; here six deaths hit six nodes, and the
  cluster had 30 such events on 14 nodes since August across many workloads. Report the pattern to the administrators with the windows.
- The recovery path still matters most: the watchdog resumes from the shared mirror (§2), the GPU-clean barrier keeps the next launch off
  memory the dead peer left behind, and follow-on holds keep the allocation (§6).

## 8. When the login node is unreachable

Compute nodes can usually only be reached via the login node; if the login node's sshd is full (`Exceeded MaxStartups`) or down, you see nothing locally. During that time:

- training and the daemon on the cluster are unaffected, so the daemon must be self-sufficient in the sense of §2;
- do not reconnect aggressively from local (every reconnect also takes an sshd handshake slot); try every few minutes;
- after reconnecting, first read the daemon's record lines, then the training log, to confirm whether there were automatic restarts in between and which step it resumed from.
