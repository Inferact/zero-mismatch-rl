# Ablation method: make A/B really differ in only one variable

Experiment-design rules we repeatedly tripped over when chasing intermittent numerical leaks (the batch-invariance kind) and doing performance attribution. Each rule corresponds to one wrong attribution and a whole wasted GPU window; the instances of the four rules come from the DS4 batch-invariance investigation, numbers in the [DS4 case](../../../cases/deepseek-v4-flash/README.md#batch-invariance-troubleshooting-example). For workload definitions, fair A/B and lever conclusions on the performance side, see [Performance and profiling](../performance-profiling.md); for process rules, see [experiment discipline](../../practices/evidence-and-experiments.md#experiment-discipline).

## Four rules

1. **A single 0-diff run does not count.** For a leak that hits less than half the time, the probability of a single run being all green is over half. Once you have a statistical conclusion, recheck it with a deterministic reproducer; "became clean" especially needs rechecking — 0-diff is necessary but not sufficient.
2. **"Turning off X fixes it" — first check whether X's switch also changes other branches.** Example: with cudagraph mode NONE, the budget-splitting function incidentally took a different branch, so the eager experiment changed two variables at once. Hidden coupling cannot be attributed to the switch itself.
3. **A switch that changes speed also changes scheduling composition.** Eager is several times slower than graphs, and batch composition changes with it; before a statistical comparison, pin each step's batch composition with offline hand-built steps, otherwise "clean when off" may just mean it never collided. Disabling cache is the same: prefills that previously were not executed will actually execute.
4. **The target path must actually be exercised.** Using long prompts as filler looks "heavier", but that step exceeds `max_cudagraph_capture_size` and the graph is not involved at all, so all settings being identical is an illusion. Negative controls record how many times the target branch executed; zero hits are marked as vacuous validation (VACUOUS), which does not count as validation.

Before locating, draw a 2×2 (or higher-dimensional) matrix to separate the variables, then decide which cells to run.

## Segment-by-segment isolation: add the total difference back up

How much overhead a long path contributes is not guessed: cut it into several arms at boundaries, each arm enabled only up to a certain segment, and see how much each segment adds back:

| Arm | Enabled up to | What it answers |
|---|---|---|
| All off | No capture | Baseline |
| A0 | Capture only (hook on, no data plane) | Cost of the capture kernel/sync points themselves |
| A | Capture + worker data plane + packing, not sent to scheduler | Worker-side processing cost |
| E | worker→engine IPC on, scheduler discards | Inter-process transfer cost |
| B | No capture, only the return path (fake data) | Return-path cost, independent of capture |
| All on | Current path | Total difference |

The differences between arms should roughly add up to the difference between all-on and all-off; if they don't, there is interaction or an arm was not cut cleanly. This is how the Nemotron-H R3 return path was located: A0 and A were nearly level with baseline, E added a little, B was close to all-on, so about 80% of the overhead landed on the per-step engine→frontend return path, not the capture hook; numbers and file:line root cause in the [R3 case](../../../cases/nemotron/r3-return-path.md).

Measure same-config repeat noise first (numbers in the case); draw no conclusion from arm differences smaller than noise; take the median over multiple pairs, report single pairs separately.

## Pin "where it's slow" to processes and call stacks

First break the makespan difference down by process: py-spy the engine, worker and frontend once each and see which function the main loop time falls in. In the R3 case, the bulk of the frontend main loop was in `output_handler` (numbers in the case), while in the no-R3 control arm the frontend was nearly idle; this step moved suspicion from the GPU side to the number of frontend objects. Only parts with a large share are worth optimizing; GPU utilization is not evidence of completion.

## Metrics are cumulative

When running multiple phases on the same engine (concurrency sweep, parameter sweep), `vllm:iteration_tokens_total_*`, `vllm:spec_decode_num_*` and `vllm:num_preemptions_total` in `/metrics` are cumulative since process start. Always write each phase's cost as `after - before`, and note in the script that "only the first phase happens not to need subtraction": for the first phase the engine just started and before is all zeros, so it is accidentally correct, matches other sources, and creates the illusion that the method is fine. For cross-checking, recompute the prefill volume with an independent client-side measure (prompt_tokens/cached_tokens in request records); if the two don't match, the counter definition is wrong.

This kind of "the verification method itself is broken" (a regex that never matches, a hard-coded version number, a cumulative value not subtracted) gives reasonable numbers and self-consistent conclusions, but the quantity is not the one you think; the first time each new script produces numbers, reconcile it once against an independent measure.

## Report speed switches and numerical switches separately

The same switch (PC, cudagraph, BI) changes both the numerical path and throughput. When reporting "probs_diff with BI on/off" and "throughput with BI on/off", first list the confounders that change at the same time, rule them out or fix them one by one, then compare; list confounders first, then compare — rules in [recipe evaluation 02](../recipe-evaluation/02-evaluation-and-acceptance.md#paired-acceptance), Nemotron's specific items in the [recipe deviation table](../../../cases/nemotron/recipe.md). On the communication side there is also same-fabric fairness, see [05](05-comm-fault-ladder.md#same-fabric-fairness).
