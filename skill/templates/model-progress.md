# Model stage record

Model: TBD; owner/task: TBD; last updated: TBD.
Single card: `model-verification-card.yaml`; current blocker and next step: TBD.

| Stage | Latest attempt (date/run ID) | Output links and actual conclusion | Gap/next step |
|---|---|---|---|
| G0 config | Not started | Unverified | |
| G1 inference update | Not started | Unverified | |
| G2 training load | Not started | Unverified | |
| G3 save/restore | Not started | Unverified | |
| G4 weight sync | Not started | Unverified | |
| G5 few-layer two-step | Not started | Unverified | |
| G6 full training/eval | Not started | Unverified | |
| G7 delivery | Not started | Unverified | |

For each attempt save: input version, command, expected/actual metrics, conclusion, evidence and next step.
Small outputs go in `artifacts/Gn/<run-id>/`; for large files note machine/path and hash, and excerpt the key evidence.
The fix list records the issue, repo/branch/commit, PR/merge status and regression results.
This table indexes existing outputs; do not copy logs, do not overwrite failure history, do not create another state machine.
