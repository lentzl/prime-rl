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

## 2026-09-08 07:15 UTC

Launched `evidence-sft-r2-paired-literal-write-repair-r1` through the visible
Launcher under Prime-RL `a504c467d`, Verifiers `3321579b`. Both frozen inference
engines loaded their checkpoints and returned healthy responses; approximately
8.6 GiB is allocated per GPU. The paired evaluator is starting the two chapter
arms. Results are pending; startup is not a utility result. Existing R1/R2
weights and sampling are unchanged. No additional infrastructure was acquired.

## 2026-09-08 07:24 UTC

The syntax-repair comparison completed, four episodes and unchanged checkpoints,
with zero captured provider/task errors. Exact text exports are verified against
the native traces. See `literal-write-repair-r1-results.json`.

- R1 exceptions: followed two literal corrections and saved four bullets. Still
  lost the additional approval relation, identifiers/timestamps and merge-safety
  qualifier; incorrectly described importing as restoring the ticket system.
- R2 exceptions: received the exact correction but treated the suggestion as
  already executed, attempted nonexistent parent sends, and hit max turns with
  no notes. Do not describe suggested code as successful tool repair.
- R1 city-shade: saved only paragraph one, hit max turns, no summary.
- R2 city-shade: completed natively with no literal repair firing. Saved extensive
  notes and five bullets (96 words), retaining the main finding, local-comparison
  qualification and dependence on access/maintenance/location. Lost independent
  mechanism measurement limits, the recommendation for route-based measurement,
  temporary-structure/mature-tree distinction, diversity risk and repeat-study
  proposal. Notes retained most of these, locating the main loss in realization.

Trajectories varied before any interception, despite fixed sampling settings;
the completed longer output cannot be attributed to a repair that never fired.
This is partial applied progress, not dependable summarization or promotion.

Next screen uses the same R1/R2 models, native workflow and sampling on two
complete public chapters: Bennett Chapter I (907 words) and Carroll Chapter I
(2,141 words). Both source texts, exact retrieval/normalization provenance and
pre-run semantic review points are in `chapter-probes/`. They are excluded from
the training exporter; possible pretraining familiarity remains explicit.
Use `DOCUMENT_SUMMARY_PROBES='alice-ch1 bennett-ch1'` with the existing paired
driver. No new harness or optimizer update. Assess end-to-end usefulness and
whether the all-paragraph-ID notes contract creates avoidable bookkeeping on
long prose. Existing task-owner integration remains unfinished.

## 2026-09-08 07:27 UTC

Public-chapter run `evidence-r2-public-chapters-r1` is launched through the visible
Launcher under Prime-RL `773b9ccbe`, Verifiers `3321579b`. Both inference engines
are healthy, using about 8.6 GiB per GPU. Driver selects `alice-ch1 bennett-ch1`,
one frozen model per GPU, four episodes total. Both native config dry-runs,
forty-six local focused tests, Ruff and shell syntax checks passed. No host tests
were run beside live work. Results are pending; no success claim from startup.

## 2026-09-08 07:40 UTC

Public-chapter screen finished with four error-free native episodes and unchanged
R1/R2 hashes. `public-chapters-r1-results.json` records full outputs, native trace
hashes and verified exports. Both Alice runs and R1's Bennett run copied the
entire source into notes. R1 Alice produced prose and invented escape into the
garden, reordered events and contradictory size changes. R2 Alice produced
eleven unmarked lines with an invented garden entry and incorrect growth/shrink
sequence. R1 Bennett produced five bullets but changed equal daily time into
“No one receives more or less than they give.” R2 Bennett attempted procedural
actor extraction with reversed enumeration variables, repeated failed repairs,
and hit max turns without notes. No dependable chapter output or promotion.

This identifies realization errors even when source information is fully present,
plus a note-selection/programming problem. Before another tiny authored-data
update, compare a pinned published upstream Qwen reference under the same native
workflow. The reference is diagnostic, not a reset of the acquired lineage; exact
ancestor identity is not established. Upstream repository revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, published weight SHA-256
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
Preparation preserves original configuration and weight bytes, verifies the hash,
and records zero optimizer updates. Local metadata/free-space dry-run and Ruff
passed. No reference download or evaluation is claimed by this entry.

## 2026-09-08 07:48 UTC

The pinned upstream reference downloaded and passed its published weight hash
check. Its original configuration is unchanged; `REFERENCE.json` records every
staged file hash and zero optimizer updates. A verified archive of superseded
commit-revision v5 freed 4.1 GiB first; only that remote weight file was removed,
and full local file checksums matched. Current models and all remote receipts/logs
remain. Host now has approximately 3 GiB free after staging the reference.

Run `upstream-vs-r2-public-chapters-r1` was dispatched through the visible Launcher
after successful reference preparation. It uses the existing paired driver on
both public chapters: upstream versus R2, one model per GPU. Remote evaluation
code is `c162799ca`, Verifiers `3321579b`. The local publication of `5ed1062e`
was still pending, so the already-committed preparation script was transferred
as `/home/ubuntu/rlm/artifacts/prepare-q35-upstream-reference-5ed1062e.py` and its
SHA-256 verified on both machines:
`6890a46705398dfd7f7d99b9d3a2f99e66af9dfc941d026a498d68296817f2cf`.
No live checkout was modified to bypass publication. Both inference engines
loaded. The chat-template file hashes match; tokenizer file serialization hashes
differ and are being checked for semantic rather than cosmetic differences.
Results are pending; no diagnostic or promotion verdict follows from startup.

## 2026-09-08 07:50 UTC

Both engines are generating concurrently at approximately 27% utilization and
8.6 GiB per GPU. The tokenizer files differ in serialized fields, but actual
token IDs match on both complete chapters, a special-token/tool-protocol sample,
and twenty native Alice trace message contents (23 samples total). Bennett is
1,168 tokens and Alice 2,661 under either tokenizer. This rules out input-token
differences on the checked text, not every possible tokenizer behavior.
Publication of `5ed1062e` has now completed; the running checkout remains pinned
to `c162799ca` and must not be updated during evaluation.
