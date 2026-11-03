# p6 (CuTe-DSL) inference-side acceptance (2026-10-04)

Conclusions:
- Formal bench: the 5-pair median loss at all 4 c values is ≤20%: c=1 10.4%, c=8 8.9%, c=32 7.8%, c=64 15.1%.
- gate3 full run: pass.
- GSM8K: the difference between BI0 and BI1 is not significant; the difference between p6 BI1 and p5 BI1 is not significant either.

## Identity

| item | value |
| --- | --- |
| image | `<image>` (the p6 package image), sha256 `d2f76230d214…6331233f`. vLLM `0.30.1rc1.dev618+g<sha>` (an internal commit), RECORD sha256 `b545c3f4…`, recorded in every leg's `runtime-identity.txt` |
| model | Lightning NVFP4 bee7596, copied from Lustre to node-local `<node-local-disk>/model`, config sha256 `f1d98b53…`, same as p5 |
| launch | pyxis: `srun --container-image --no-container-mount-home --container-remap-root`, passing only the variables the protocol needs via `--export`. In-container scripts are identical to p5 (`inner-docker.sh`, `proxy_serve_gate.py`, gsm8k `inner.sh`). No overlay or monkeypatch |
| bench client | vllm-rs rebuilt from the same source as the p5 client: the `rust/` of an internal vLLM commit, version string `0.26.1rc1.dev0+<local>.g<sha>`, release + `native-tls-vendored`, cargo target in `/tmp` on the build node. The new binary's sha256 is `7c0378b0…`, different from p5's `f21af157…` (built on a different machine). See `build/` |
| nodes | bench: five nodes, one pair each. gate3: one node. GSM8K: BI1 and BI0 each on its own node |

Note: the `NCCL_*` variables in the container (`NCCL_MNNVL_ENABLE=1`, `NCCL_IB_HCA`, etc.) come from the cluster's enroot hook and were not passed in by us. They are identical across legs and recorded in `container-env.txt` and in the receipt's `env`.

## 1. 8K in / 1K out paired bench (formal matched-flags protocol)

The protocol is the same as `final-p5/bench-matched`:
- 5 pairs, order AB BA AB BA AB (A = BI0), c = 1/8/32/64, a fresh server per leg, the two legs of a pair share a seed label.
- The two legs of a pair run on the same node; different pairs run in parallel on different nodes.

Configuration (`bench/configs/`):
- BI0 = `bi0npc.args`, byte-identical to p5's file.
- BI1 = the same arguments plus env `VLLM_BATCH_INVARIANT=1`. Both p6 arms use `flashinfer_cutedsl`, so the two arms differ only in the BI environment variable.
- Both arms disable prefix caching and do not pass `--block-size`. The values chosen automatically by vLLM are BI0 4176 and BI1 6768, same as p5.

Validity: all 40 points `valid=true`. Prefix cache hits 0, preemptions 0, all server_exit=0. The first leg on each node is a cold JIT start (743–904 s), later ones 150–300 s; cold start only affects startup time. Backend check in `bench/runs/p6m-backend-check.txt`: every leg is `FLASHINFER_CUTEDSL`, `prefix_caching=False`, FULL_AND_PIECEWISE.

| c | BI0/BI1 tok/s: p1 / p2 / p3 / p4 / p5 | per-pair loss | **median** | ≤20% |
| --- | --- | --- | --- | --- |
| 1 | 301/290, 378/287, 316/284, 302/285, 339/284 | 3.7, 24.0, 10.4, 5.7, 16.0% | **10.4%** | pass |
| 8 | 2288/1993, 2536/2009, 2064/1978, 1816/1961, 2185/1990 | 12.9, 20.8, 4.2, −8.0, 8.9% | **8.9%** | pass |
| 32 | 6683/6655, 7873/6316, 6428/5926, 6031/6406, 7350/6457 | 0.4, 19.8, 7.8, −6.2, 12.1% | **7.8%** | pass |
| 64 | 8507/7935, 10734/8725, 10599/9001, 10309/9519, 10358/8703 | 6.7, 18.7, 15.1, 7.7, 16.0% | **15.1%** | pass |

Median TPOT (ms):
- BI0: c=1 2.55–3.18, c=64 4.93–5.45.
- BI1: c=1 3.34–3.40, c=64 5.45–6.08.

Notes:
- BI1 is very stable: c=1 is between 284 and 290. The fluctuation comes from BI0: c=1 between 301 and 378, the same BI0 bimodality seen on p5, cause unexplained.
- The only single pair above 20% is p2: c=1 24.0%, c=8 20.8%. p2's BI0 landed in the fast mode.
- p4's BI0 landed in the slow mode, giving negative losses at c=8 and c=32.
- Compared with p5 (same protocol, on another cluster): median losses 26.6/18.2/21.8/30.4%. p6 is 10.4/8.9/7.8/15.1%. The two are not on the same cluster, so only the direction can be read.
- Raw data is in `bench/runs/p6m-p<N>-<arm>/`; each leg has serve.args, extra.env, runtime-identity, container-env, server.log, and per-point result.json/validated.json. Summary in `p6m-pair-summary.txt`. Metrics snapshots were not copied into the repo; the originals are under `$WORKDIR/p6-inf/bench/runs/` on Lustre.

## 2. gate3 full run (inference BI / PD / chunk)

Script: `gate3/run-g3-full.sbatch`, the pyxis version of `final-p5/gate3-full/run-g3-full.sh`.
- serve-args differ from p5 only in `humming` → `flashinfer_cutedsl`; everything else is literally identical. max-model-len 16048.
- env is `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_V2_MODEL_RUNNER=1`, with the Humming-specific `VLLM_HUMMING_MOE_GEMM_TYPE` removed.
- Run time 2026-10-04 05:02–05:28Z, one node.

| leg | result |
| --- | --- |
| serve (mnbt 16384) | **pass**. PD 4/4 mismatch 0 (14000 + 14000 + 1024 + 512 = 29536 tokens). BI 6/6 tokens and scores bitwise identical: 14000-token target request, batch 2/4/8, placed at the first/last position |
| chunk2048 (mnbt 2048) | **pass**. Cross-run comparison against the serve leg's rollouts, 4/4 tokens and scores bitwise identical |
| prefixcache | **not applicable (rejected as expected)**: `ValueError: VLLM_BATCH_INVARIANT=1 with Mamba2 layers does not support prefix caching yet` (`gate3/runs/prefixcache-refusal.txt`) |

Receipts are in `gate3/runs/g3full-r1/{serve,chunk2048,prefixcache}/receipt.json`, with each leg's server log inside.

## 3. GSM8K

The protocol is the same as `final-p5/gsm8k`:
- Data is openai/gsm8k revision `740312add88f…`, parquet sha256 matches p5. 8-shot CoT, the full test set of 1319 questions.
- Requests go to `/v1/completions`, temperature 0, seed 42, max_tokens 7680, concurrency 256. prompt sha256 `a071ff43…190213`, same as p5.

Configuration:
- BI0 = `bi0.args`: CuTe-DSL, prefix caching on, same as p5 GSM8K's BI0.
- BI1 = the BI1 configuration of the p6 bench.

| group | correct / 1319 | accuracy | Wilson 95% CI | no answer | hit the cap |
| --- | --- | --- | --- | --- | --- |
| p6 BI0 official | 1110 | 84.15% | 82.09–86.03 | 24 | 9 |
| p6 BI1 official | 1105 | 83.78% | 81.69–85.67 | 18 | 8 |
| p5 BI1 official (quoted from `final-p5/gsm8k`) | 1108 | 84.00% | 81.93–85.88 | 19 | 5 |

| pair | difference | 95% CI | only the former correct / only the latter correct | McNemar exact p | answers differ |
| --- | --- | --- | --- | --- | --- |
| p6 BI1 − p6 BI0 | −0.38 pp | −1.94 … +1.19 | 58 / 53 | 0.70 | 164 |
| p6 BI1 vs p5 BI1 | p5 − p6 = +0.23 pp | −1.35 … +1.81 | 55 / 58 (p6 / p5) | 0.85 | 162 |

(`summary.json` also has p5 BI1 − p6 BI0: −0.15 pp, p = 0.92.)

Notes:
- Nodes: BI1 ran on one node. BI0's first run was on another node (one job) and was cancelled by the coordinator because that node had been assigned to other work. The incomplete output was moved to `$WORKDIR/p6-inf/gsm8k/discarded/` on Lustre and not used. It was then rerun on a different node (a second job).
- Both servers exited normally (server_exit=0), and all 1319 questions have results.

## Files

- `bench/`: leg.sh, pair.sbatch (pyxis), inner-docker.sh (same as p5), validate_point.py, pair_summary.py, configs, runs.
- `gate3/`: sbatch, proxy_serve_gate.py (same as p5, sha256 `f8661ae3…`), three serve-args files, runs.
- `gsm8k/`: scripts (eval/inner/analyze same as p5, plus leg.sbatch), configs, runs (`p5-bi1-official/` was copied from p5 for McNemar), summary.json.
- `build/`: vllm-rs build script, log and sha256.
- Lustre work directory: `$WORKDIR/p6-inf/`. The model copy and JIT cache remain in `<node-local-disk>` on the four bench/gate3 nodes.
