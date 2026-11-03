# DeepSeek V4: CP compressed boundary misalignment

## Source and scope

The simplified implementation finally merged is PR #194 of a community Megatron-LM fork, whose description says it is a redo of #190. #190 keeps the more complete root-cause and validation write-up. The final #194 uses an inline empty-intersection guard; do not describe #190's helper/test structure as the final file contents of #194.

Internal source: the model workspace's CP boundary debug retrospective (in its batch invariance reports, not included here). What follows are the conclusions of that historical experiment, not results re-executed when this skill was installed.

## Symptom → locating the side

In real RL on 43-layer Flash-Base, 32 GB200, training CP2, most steps had zero difference, while a few steps had a maximum difference in probability space of about 0.15–0.63. A whole-batch dump found that in a typical failing batch only 3/256 sequences differed, but from response token 0 onward, all 8192 tokens differed; the other samples were clean. This is closer to context/index misalignment than to "occasional quantization error on a few tokens".

On the common response prefix of same-prompt siblings, rollout μ was the same while trainer π differed. Independent inference-only repeats, batch variants and decode→prefill comparisons were consistent over about 14.74 million tokens. This focused the subsequent investigation on the training old-logprob, rather than guessing Top-K or DeepEP from aggregate diffs alone.

## Fixed rollout → A/B/C

Save the real batch, skip generation/update, and replay only the old-logprob. A is the original packing, B repeats A, C changes packing. B equals A, and C consistently changed some entire sequences; a comparison on different GPUs did not change the result. The intermittent E2E failure was reduced to a deterministic reproducer triggered by layout.

Per-layer/module probes further narrowed it to the local layout of attention's compressed K/index. Comparing against request boundaries found that a request that had already ended to the left of the current CP rank was wrongly counted into the local compressed groups when its end was 1–7 tokens from the rank's start.

## Minimal example

```text
requests: [0, 8296), [8296, 16592)
rank 1:   [8296, 16592)
ratio=4, d_comp=8
```

The first request has no intersection with rank 1. The old host code first computed:

```text
owned_start = 8296
owned_end = 8296
first_group = (8296 - 8 + 4 - 1) // 4 = 2072
last_group = 8296 // 4 = 2074
count = 2
```

The device compactor correctly produced 0 rows, but the host cumulative boundary counted 2 extra groups, shifting the indices of all subsequent requests. The left look-back window belongs to the same request; a completely disjoint request must not contribute rows to the current rank.

The final fix handles the empty intersection before computing groups:

```python
if owned_start >= owned_end:
    local_boundaries.append(local_boundaries[-1])
    continue
```

## Validation and lessons

The historical validation of #190 included 17 CPU tests, equivalence with the verified patch over 2,964,000 boundary configurations, and a full-model training-only replay. Before the fix the A/C sequence equality ratio was 0.90234375, i.e. 25/256 sequences changed; after the fix μ/A, A/C and μ/C were all 1.0. There is also a record of all-zero aggregate metrics over a normal ten-step training run; aggregate evidence by itself does not replace raw byte comparison.

Safe packing was tried, but it cannot guarantee every batch is safe and does not fix the contract. The transferable method is: save one failure's data → locate the side → fixed forward replay → find the first difference → tabulate boundary remainders → compare host/device output counts → minimal fix → tiered regression on the original reproducer and normal training.
