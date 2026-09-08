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
