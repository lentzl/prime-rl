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

## First live result: workflow completes, content does not yet pass

Run: `worker-h176-summary-evidence-notes-v8-exceptions-r1`, Prime-RL `44bb6cbe9`,
Verifiers `e79e1e7b2a52487e2b0a13c440b249c19301992a`. The v8 checkpoint remained
unchanged before/after: SHA-256
`2a8575a01814af2cd4914f9984d0001d909f89c1414534e90220b476df64b28c`.
No optimizer update occurred. The servers exited and both GPUs returned to idle.

The episode finished normally in eight model calls, with no captured error,
empty-tool loop or length finish. Agent time was 28.65 seconds, including 24.14
seconds in model calls. The worker read the source, authored and wrote notes,
stopped, then explicitly read `notes-extracted.md` before writing its summary.
It did not reread the original source in phase two. Scratch and captured notes
match exactly. This demonstrates the working state path on one development
chapter, not general summarization utility.

Source-level inspection identifies two distinct losses:

1. **Source -> notes:** The first three records retain the outage condition,
   identifiers/timestamps, recovery order, no-overwrite rule, both duplicate
   records until review, and unresolved differences until cause documentation.
   The fourth drops **also**: operations-manager approval is no longer explicitly
   additional to support-lead approval for deletion/deadline changes. The notes'
   keyword-group proxy is nevertheless 1.0, illustrating its limitation.
2. **Notes -> summary:** The final first line drops ticket identifiers and
   timestamps that were explicitly present in the notes. It retains the outage
   condition that the earlier direct-revision result lost. The additional-approval
   ambiguity persists. All four lines lack Markdown bullet markers.

The raw final text is 55 whitespace-delimited words, below the 68-word limit;
there was room to retain the missing fields. Historical bullet-based metrics
remain exactly as emitted: bullet count/concision/non-copy/proxy = 0 and
`summary_words=0`, because the parser found no bullets. This does **not** mean the
file was empty or contained no relevant facts. Artifact completion reward 1.0
means both files exist, not that the summary passed. A structured excerpt with
the actual nodes, artifacts, metrics and full-trace hash is in
`evidence-notes-r1.json`.

Next: teach qualification-preserving notes and faithful realization from saved
notes on varied source chapters, including contrastive additional-versus-exclusive
approval rules. Reuse earlier training sources where sound, correcting omissions
in their older targets; do not rerun the same three-answer fitting loop. Teach
the actual read/write/stop transitions through Prime Agent and retain a fresh
document split for subsequent confirmation. No new expert population is needed.
