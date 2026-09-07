# Prime Agent document summarization v1

This is an applied capability check, not a training campaign. It asks the existing
terminal-worker lineage to summarize English chapters into concise, grounded bullet
points through the native Prime Agent harness.

The first run uses `smoke-worker.toml`: one chapter, one depth-zero worker, one typed
JSON artifact. It measures whether the current H176 worker can preserve important
facts, cite source paragraph IDs, and compress the source without copying it. No
weights are updated.

The initial unscaffolded probe is retained as a diagnostic: H176 drafted a partial
summary but repeatedly passed a path directly to `json.dump`, ignored the nested
bullet schema, and never produced an artifact. The current contract adds only a
generic correct JSON-file pattern and an explicit schema self-check. It does not
expose hidden facts or reference wording. This second probe separates recoverable
tool protocol from a real summarization weakness before we train anything.

Only after that isolation passes do we use `smoke.toml` to test the end-to-end owner
workflow: the owner spawns three named Prime Agent children, receives their explicit
reports, and assembles the chapter summaries. Each job embeds the full contract so
the owner cannot accidentally weaken it while delegating.

The scorer checks artifact structure, paragraph grounding, concision, source-copying,
and hidden decision-relevant fact coverage. Hidden facts are evaluator-only and are
not written into the runtime.

The existing baked Prime Agent runtime image is reused because summarization needs no
new runtime dependency. Training is justified only by the first demonstrated weak
role: worker summarization if the direct probe fails; delegation or fan-in if the
worker passes but the owner workflow fails.
