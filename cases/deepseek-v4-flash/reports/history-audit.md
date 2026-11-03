# Historical evidence audit (2026-08-28)

This page was split out of the original cross-model audit and keeps the conclusions of that time; it has not been re-verified and cannot serve as a current pass status.

## 3. DPSK V4 historical main line

### 3.1 2026-07-28: fix the experiment contract first, then locate the runtime

Many early failures happened before any real update: the container did not load the target source, `BASE_URL` was not passed,
the dev API was not enabled, the container had no `python` alias, the publisher shared GPUs with the server. Their common lesson
is: the launcher preflight must prove that the target code, commands, API and resources have all taken effect; otherwise the run can only
be marked invalid and does not count as a model failure.

The effective technical main line was then:

- update-only 61/61 buckets and four fixed-token oracles passed;
- eager level-2's `attn_sink` writing directly to meta storage was fixed by an explicit loader;
- in CUDA Graph level-2, parameter contents matched but derived runtime buffer addresses changed, requiring in-place refresh at the real
  loader/post-load lifecycle boundary; the graph must not be reset or recaptured;
- only after target-only eager and default CUDA Graph passed in turn did native MTP start.

### 3.2 2026-07-28 to 07-30: MTP and load-plan

- The draft's shared embedding/head belongs to the target; the load plan needs root ownership and cannot require the
  draft transaction to resend shared target weights.
- In EP mode the training side may send the complete expert set, but this rank's loader rejects non-local experts.
  The startup plan may only record an application after the loader returns success; otherwise expected is an EP multiple of
  observed, the RoutedExperts staging buffer cannot be released and eventually OOMs.
- `finish_weight_update=200`, identical token IDs or a single close GSM8K score cannot close the issue; the final
  default CUDA Graph + native MTP + level-2 target/draft update must pass independent
  pre/pre-repeat/post/post-repeat oracles.
- Several stale-scale, kernel-storage and graph-closure candidates were directly refuted by A/B. The final
  integration failure came from omitting the already-verified `attn_sink` prerequisite from the stack, not from needing a new graph fix.
  This shows every experiment must record the complete patch stack, not only the increment currently under investigation.

Evidence entry points (bug reports in the original task workspace, not included here):

- `dpskv4-native-mtp-shared-weight-load-plan`
- `ep-local-receipt-plan`
- `dpskv4-native-mtp-cudagraph-logprob-drift`
- `dpskv4-mhc-broadcast-finalize-order`

### 3.3 2026-08: batch invariance is a separate post-Day-0 project

August's sparse MLA, mHC, MoE, DeepEP, CUDA Graph and batch invariance work produced
many reusable methods, but it is not a continuation of July's lifecycle admission, and its throughput or soak
results cannot be used in reverse to substitute for weight-update correctness.

The correct reading boundaries at the time were:

- July lifecycle: `results/_legacy/`, bug reports and July events;
- August batch invariance: the then-current `INDEX.md`, `ASSUMPTIONS.md`, `CATALOG.md`,
  `README-TRIAD.md` and `BRANCHES.md`.


For general diagnosis methods see [fault localization](../../../skill/guides/debugging-mismatch/README.md); for code and experiment version management see [code and delivery](../../../skill/practices/code-and-delivery.md).
