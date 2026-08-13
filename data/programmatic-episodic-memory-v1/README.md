# Programmatic Episodic Memory V1

Synthetic bootstrap data for teaching Prime Agent models to treat long-running interaction
history as an **external computational object** rather than relying on attention or vague
recall.

Inspiration: PRO-LONG — https://github.com/alexisfox7/PRO-LONG.git

This dataset does **not** copy ARC-AGI tasks or PRO-LONG logs. It generalizes the memory
mechanism to ordinary long-horizon agent work: recover distant state, resolve superseded
requirements, identify successful prior attempts, compute over corrected events, recover
after context reset, prefer append-only history over stale notes, and reuse compact derived
indexes across requests.

## Capability target

The learned policy should be:

1. Recognize when the current request depends on past events.
2. Retrieve instead of guessing.
3. Query `/workspace/history.log` programmatically with Python.
4. Return only a compact relevant slice to active context.
5. Prefer the latest valid/corrected event rather than the first matching event.
6. Treat `history.log` as lossless source of truth and workspace notes/indexes as disposable
   derived state.
7. Build/reuse compact indexes when repeated retrieval has cognitive value.
8. Recover correctly after context reset/compaction.
9. Avoid history access on self-contained controls.

This is the temporal analogue of the project principle:

> Externalize when externalization has positive cognitive value.

## Materialized splits

- `train.jsonl`: 72 rows, 12 policy families.
- `familiar_heldout.jsonl`: 36 fresh rows from the same policy families with longer horizons.
- `semantic_ood.jsonl`: 18 rows using unseen JSONL history format and unseen temporal
  operations.

Train/familiar histories contain 32–224 append-only events. OOD histories contain 128–384
events. Relevant information is deliberately separated by large distractor spans.

The training split cycles three instruction conditions where appropriate:

- explicit history wording;
- natural wording that requires the model to infer historical dependence;
- explicit context-reset/compaction wording.

The OOD split uses natural requests.

## Training families

- `latest_state`: recover the latest active value.
- `accepted_requirement`: distinguish accepted policy from later rejected proposals.
- `successful_attempt`: find the attempt that actually succeeded among failures.
- `correction_aggregate`: retrieve distant base/delta/correction events and compute.
- `provenance_conflict`: recover the latest evidence verdict and source provenance.
- `checkpoint_resume`: find the latest stable checkpoint before corruption.
- `stale_note_override`: verify stale derived workspace notes against append-only history.
- `repeated_lookup_index`: build an index once, then reuse persistent IPython state on a
  second request without rereading history.
- `multi_key_join`: join distant events to reconstruct current state.
- `context_reset_resume`: recover the current plan after active context loss.
- `constraint_update`: obey the latest user constraint after supersession.
- `direct_control`: self-contained request where the correct policy is **not** to retrieve.

## OOD families

- JSONL latest-revision retrieval.
- Temporal-window computation over the last three relevant measurements.
- Supersession-chain reconstruction.

## Row format

Rows are Arrow-safe and compatible with the existing Prime-RL SFT representation:

- `messages_json`: JSON-encoded OpenAI wire messages.
- `tools`: JSON-encoded `ipython` tool schema.
- `workspace_files_json`: synthetic files that an executable environment can materialize,
  principally `/workspace/history.log` and occasionally `/workspace/notes.txt`.
- `metadata_json`: split/family/horizon/retrieval-policy/expected-answer metadata.

No `reasoning_content` is fabricated.

The SFT trajectory itself includes a compact IPython retrieval call and its expected tool
output. The complete synthetic history is stored only in `workspace_files_json`, not stuffed
into model context.

## Regeneration

```bash
uv run python scripts/generate_programmatic_episodic_memory_v1.py \
  --output-dir /ephemeral/subagent-rung/data/programmatic-episodic-memory-v1
```

When `datasets` is installed, the generator also emits `train.parquet`,
`familiar_heldout.parquet`, and `semantic_ood.parquet` directly.

Default deterministic seed: `20260813`.

## Recommended use

This is a **bootstrap seed**, not the final long-horizon curriculum.

Near-term:
1. Mix a modest amount into the broad 27B Harness Mastery SFT/bootstrap stream.
2. Keep familiar-heldout and semantic-OOD entirely outside training.
3. Add a small fast memory screen to the Harness Mastery battery.
4. Build an executable environment that materializes `workspace_files_json` and verifies
   the actual file-access/retrieval behavior.
5. Later collect successful native 27B memory trajectories and shift toward on-policy
   refinement/distillation.

Do not let this family dominate the broad Harness Mastery curriculum. The goal is to teach
a transferable retrieval decision, not to create a `history.log` ritual.

## Important future extension

The strongest follow-up should deliberately exceed active model context and include explicit
session/context resets. The model should continue from a small current prompt plus external
history. For smaller 9B/4B/2B students, this capability is especially valuable because it
lets them compensate for limited internal working memory with exact external retrieval.
