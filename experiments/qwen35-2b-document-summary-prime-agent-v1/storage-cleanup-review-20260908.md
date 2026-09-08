# Training-host storage review — 2026-09-08

Status: Owner approved permanent deletion of superseded intermediate weights
without verified backups ("Yes you may"). The 18-file first pass is inventoried
in `storage-cleanup-20260908-approved/manifest.json`, SHA-256
`05b2cc8022053b6c9b4b415be6e1d0ac54815f440f42c67c1a28d07d52e8a24b`.
Execution completed through Launcher at 2026-09-08 22:10:37 UTC. All 18 exact
weight files were removed, totaling 79,678,059,552 bytes (74.206 GiB). Free
space rose from 6,546,321,408 to 86,224,359,424 bytes (80.303 GiB). Protected
weight hashes were reverified, and the retained non-weight evidence manifest
was exactly unchanged. These removed historical weights have no verified
recovery copy. No enclosing run tree or Docker container was removed.
The receipt is `storage-cleanup-20260908-approved/receipt.json`, SHA-256
`5836abe4894644f4e1db4a7778646638cc62b1f2a491f0ab2a69b7be2232b4c4`;
the exact-file deletion journal is `storage-cleanup-20260908-approved/events.jsonl`,
SHA-256 `485373dc4e308c74d9fd8592f1b5019cb15c14289efc37e98c8f46309a0f1dbb`.
The R10 launch script was dispatched only after successful cleanup verification.
The earlier package-cache prune is recorded separately in
`package-cache-recovery-20260908.json`.

## Measurements

Host: `ubuntu@216.81.200.39`, root filesystem `/dev/sda2`.
Fresh follow-up: 539,810,795,520 total bytes, 510,994,317,312 used bytes,
6,546,509,824 available bytes. GPU compute-process query returned no processes.
Output trees account for about 424 GiB; the older dual-dense GRPO and recursive
return trees account for about 80 and 79 GiB respectively. Docker is not the
main consumer: 12 active, preexisting containers occupy about 6.1 GB and remain
outside cleanup scope.

## First review set: recursive return weights

Root: `/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1`.
The live inventory contains 19 regular `model.safetensors` files, each
4,426,558,864 bytes with hardlink count one. Below, each listed step denotes
`<root>/<run>/weights/step_<step>/model.safetensors`, not its enclosing run.

Keep `c160-child-minimal-balanced-consolidation-v9`, step 2. Its promotion and
recovery role are explicitly documented in the recursive strand's
`c160-child-minimal-balanced-consolidation-v9-receipt.json` and
`remote-output-cleanup-v1-receipt.json`.

The other 18 files total 79,678,059,552 logical bytes (74.206 GiB):

| Run | Steps for review |
| --- | --- |
| c160-child-runtime-compute-v4 | 8 |
| c158-c-return-sft-forced28-root8-v2 | 4, 6 |
| c160-child-compute-mix-v3 | 10 |
| c158-c-return-sft-forced24-root8-v1 | 4 |
| c158-c-return-sft-compute28-root8-v2 | 1, 2, 3 |
| c158-c-return-sft-nearmiss2x2-v1 | 2 |
| c160-child-tight-reporting-consolidation-v7 | 6 |
| c160-child-return-sft-forced40-v1 | 4 |
| c160-child-balanced-live-compute-v6 | 8 |
| c160-child-live-context-compute-v5 | 8 |
| c160-child-gentle-balanced-consolidation-v8 | 8 |
| c160-child-natural-compute-replay-v2 | 8 |
| c158-c-return-sft-compute56-root16-v3 | 6, 8, 10 |

This is potential capacity, not verified reclaimable capacity or a claim that
all 18 files are disposable. The nearmiss screen explicitly rejected that
candidate; the v7 receipt explicitly records non-promotion. The other files
still require per-checkpoint retention and reference review before selection.

## Preservation and remaining checks

Keep current summary owner R4 and worker R9, their meaningful recovery points,
e33/H176 baselines, protected C158/C160 and promoted v9, and the protected typed
topology milestones. This list is a minimum, not an exhaustive inventory of
protected lineages. Preserve datasets, configs, receipts, metrics, traces,
controller state, and all preexisting runtime containers.

Before any destructive action: obtain the pending user retention decision;
resolve current controller initial/current/promoted references and scheduled
consumers; verify each selected run is terminal; inventory and hash exact
selected weight files; preserve compact evidence; dispatch only exact targets
through the Launcher pane; measure actual free bytes afterward. No broad tree
deletion or age-only selection. If backup is the chosen retention route,
verify currently readable recovery bytes first; historical iCloud hashes do
not establish current recoverability.

No new Strategist comment was returned after channel comment 5591817566 during
this review. The automatic goal continuation is not permission to delete
unbacked checkpoints or publish new Hugging Face backups.

## Follow-up reference check

A fresh host search of JSON/TOML/JSONL files under `/home/ubuntu/rlm/state`
found no references to the recursive-return output tree. In the old
`grpo-autonomous-v3` directory, only `events.jsonl` and `hf-publications.jsonl`
reference that tree; extracting all matching model paths from both files
returned only the retained v9 step-2 checkpoint, none of the 18 review files.
The v3 launch receipt independently identifies C158 and v9 as its initial
frontier. The last saved v3 frontier contains GRPO child 176 and coordinator
177, both outside the recursive-return shortlist. The terminal event is
sequence 91, `train_failed`, dated 2026-08-30 18:33:52 UTC. A fresh host process
query found no matching training, inference, or autonomous-controller process
other than the read-only inspection shell itself; the STOP file alone was not
used as proof of inactivity.

This narrows the saved-controller reference risk. It does not prove a complete
search of every historical worktree, scheduler or external consumer, establish
backup recoverability, or authorize deletion. The existing protected C158,
C160, v9, GRPO-176/177 and summary-lineage checkpoints remain excluded.
