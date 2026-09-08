# Evidence curriculum R2

## 2026-09-08 06:43 UTC

Run `h176-summary-evidence-expository-step16-r2`, code `1177c4eaa`, continuing
from evidence R1 checkpoint `b8cd9656a6b61b23386bd50a24aac47a5175d565bbbc9b8ed80ba91d9dfc7202`.
Launch dispatched through `q35-spade-open:Launcher`; verify trainer initialization
and live metrics before describing it as progressing.

- Forty authored TRAIN phase samples across twenty source cases, including four
  longer expository chapters; previously acquired sources and contrasts retained.
- Dataset: `/home/ubuntu/rlm/data/q35-2b-document-summary-evidence-sft-v2-r1`;
  parquet SHA-256 `7b4de4b7e8ee3b4ecf2707eb417abe6db96c051bd7a19879b3705a9b09579256`.
- Actual model-visible next-file feedback, not the generic gate wrapper, is used
  in the teacher episodes. Teacher notes never enter evaluation.
- Exact-host mask audit: 201,459 rendered tokens, 9,815 supervised tokens,
  longest sample 5,455 tokens; no truncation at 16,384, current-phase assistants
  only, CUDA uninitialized during audit. Initial audit-script comparison differed
  only in tool-dictionary key order; matching the actual trainer serialization
  made the independent token sequence identical. Dataset/renderer code unchanged.
- Sixteen full-weight BF16 AdamW updates, LR 1e-6, batch 8, two GPUs; numerical
  dtypes unchanged, fresh optimizer. Save weights only at step 16; 30-minute run
  timeout. Packed rows mean optimizer steps are not equivalent to corpus epochs.
- Native SFT dry-run passed. Forty-six focused tests passed locally; no tests
  were run beside live GPU work. Host had 7.3 GiB free before the new checkpoint.

Hypothesis: longer and better-matched exposure improves source-linked note
creation and concise realization on ordinary prose. Inspect training health,
then compare R1 and R2 under the same native Prime Agent scaffold. Do not infer
semantic utility from loss or artifact presence. No checkpoint promotion;
broader fresh-document evaluation and task-owner integration remain unfinished.

## 2026-09-08 06:44 UTC

The launch stopped at argument validation before any training process or update:
an additional CLI check still imposed the old eight-update maximum, although the
config builder and native dry-run accepted sixteen. That duplicate check is now
corrected. Preserve the preflight-only `...-r2` directory; execute under fresh
run name `h176-summary-evidence-expository-step16-r2a` with the same dataset,
source model, sixteen updates and numerical settings.

## 2026-09-08 06:50 UTC

Run `h176-summary-evidence-expository-step16-r2a` is progressing under code
`5c9b740ba`. Both ranks initialized at 06:47 UTC; update 5 completed at about
23.9 seconds per step. Both GPUs showed 94% utilization and roughly 11.6 GiB
device memory in the read-only check. No evaluation or repository test is being
run alongside the trainer. Inspect final loss/stability and stable export before
starting the matched postflight.

## 2026-09-08 06:55 UTC

All sixteen updates and the stable export completed. Final loss 0.4023674,
NaNs zero, final gradient norm 22.5, final step 23.4 seconds; peak allocation
11.2 GiB per device. New checkpoint:
`/home/ubuntu/rlm/outputs/q35-2b-document-summary-evidence-sft-v2/h176-summary-evidence-expository-step16-r2a/weights/step_16`,
SHA-256 `da1ec708bf191313797c97f5e729baeef57deffae294b90901eb6a8fafe7691d`.
The complete training receipt matches source and dataset hashes. Its generic
renderer-audit field is null; the actual separate audit is preserved in
`evidence-sft-r2-renderer-audit.json`. No semantic or promotion claim follows.

Next matched postflight: R1 versus R2, exceptions plus now-development-exposed
city-shade, fixed Verifiers `53efed80`, same runtime/sampling, one model per GPU.
Only artifact export and the empty-routing-log receipt fix changed in the
launcher. Host free space is 3.2 GiB; this permits the no-update comparison but
must be addressed under retention rules before another weight checkpoint.

## 2026-09-08 07:13 UTC

Matched postflight `evidence-sft-r2-paired-r1-vs-r2` completed with four native
traces, no captured errors, and unchanged R1/R2 hashes. Readable artifact exports
match trace text exactly. Results are in `evidence-sft-r2-results.json`.

R1 saved four short exceptions bullets but omitted approval conjunctions and
other operational qualifiers. On city-shade it saved broad source-linked notes,
then a prose paragraph rather than bullets. Its wording changed an unestablished
immediate cooling effect into a claim that planting did not immediately cool
neighborhoods. Neither output is a semantic pass.

R2 reached max turns on both chapters. Both first failures were string-literal
`.write_text` calls after it had authored notes (69 and 285 words). The longer
draft retained substantial source relationships, but retries wandered into
syntax errors and a loop overwrote all but the last paragraph. No final summary
was saved. This is a live execution regression, not a demonstrated loss of all
underlying drafting ability and not a model-capacity ceiling.

Next no-update comparison uses Verifiers `3321579b`: first-error feedback supplies
a corrected file-write cell with the model's literal text unchanged. It only
recognizes an explicit task-output destination, never executes code, and records
the original draft hash. The native model must still choose and execute the
suggestion. This is syntax-scaffold assistance, not learned execution or semantic
repair. Both checkpoints receive the same scaffold; runtime, tasks and sampling
stay fixed. Forty-two local tests and Ruff pass; an offline replay recognizes
both observed first failures. Native evaluation config dry-run passes from the
isolated environment. The top workspace's existing dependency conflict remains;
no dependency pins were changed to work around it.

No further optimizer run or checkpoint promotion is justified by the lower
training loss alone. There is no artificial GPU-hour ceiling on this allocation.
Broader chapter utility and existing task-owner integration remain active work.
