# Source-linked notes experiment

## Scope and resources

The next capability experiment uses the existing trained worker and Prime Agent
IPython session: source -> worker-authored notes -> final English bullets. It is
not a new harness or a replacement lineage. The first exceptions-chapter episode
is development diagnosis, not fresh confirmation or a promotion test. No teacher
notes are injected. The source remains available during realization; trace
inspection must distinguish using stored notes from rereading the original.

On 2026-09-08 the owner clarified: "Why do we have the ceiling? Just use the two gpu
machine as effectively as you can". This authorizes efficient continued use of the
existing two-GPU host, superseding reliance on the earlier proposed four-hour
pilot ceiling. It does not authorize another rental or paid service. Prior
inference logs show approximately 6.114 aggregate GPU-hours between first and last
timestamp across 154 server logs; this excludes training and startup/teardown
outside those timestamps and is not a complete spend ledger.

## Diagnostic contract

- The worker reads `source.md` and writes its own `notes.md`, with source paragraph
  IDs and explicit actors, conditions, linked actions, deadlines and qualifiers.
- On completion, the workflow saves those bytes as `notes-extracted.md`, then
  requests 3-5 English bullets in `summary.md`, at most 68 words for exceptions.
- Original source, scratch notes, captured notes and summary are saved in the
  trace. Notes have no final-prose word limit.
- The gate checks presence and phase order only. Completion reward means both
  artifacts exist, **not** that the summary is faithful or useful. Keyword-group
  scores and source-ID presence remain diagnostic proxies. Inspect relationships
  directly against the source at both transitions.
- This is a terminal-worker isolation episode. Returning accepted notes to the
  task owner through native delegation remains subsequent integration work.

## Verification before model execution

The 41 focused environment tests pass with the checkout's Verifiers import path;
Ruff passes; the native evaluation dry-run accepts
`smoke-evidence-exceptions.toml`. The new workflow regression covers missing notes,
premature summary rejection, snapshot persistence across scratch edits, and the
deliberate separation between artifact completion and semantic correctness.

No model result is recorded yet.
