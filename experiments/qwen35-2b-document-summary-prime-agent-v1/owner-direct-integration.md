# Native owner / Markdown worker integration

## 2026-09-08 — first native trial reviewed

The e33/R6 run completed at its 32-call boundary without a chapter spawn, receipt
or summary. The owner repeatedly misread the index schema and retried an
unhashable `dict_keys` inspection. The two requests routed to the worker policy
were automatic `/refine` calls, not chapter workers. This run does not establish
R6's summarization quality. Full evidence is in
[owner-e33-r6-direct-public-r1-results.json](owner-e33-r6-direct-public-r1-results.json).

Local recovery now documents the actual ordered index fields and preserves
owner/child responsibilities, using the runtime prompt's explicit recursive
depth. An owner must not substitute its own draft for missing child summaries;
a child must save and report its own file, not follow owner delegation guidance.
The initial owner prompt asks it to display the index before constructing code.
All 47 existing task checks and scoped Ruff pass, including owner, child and
unscoped empty/repeated-call feedback. This is local preparation, not a measured
behavioral gain. R7's live host remains frozen at the original trial's commits.

Next owner teaching should practice native index-to-spawn actions and recovery,
then receipt-gated fan-in, using TRAIN-only examples and foundational rehearsal.
The original trial and its unsupported reflection outputs remain diagnostic
evidence, not positive targets. Worker learning continues independently from R6.

## 2026-09-08 — local preparation verified; live run pending

The taskset's `owner_direct` mode connects the existing document owner to the
direct-summary worker interface. The owner reads an answer-free index, retains
named native RLM children and yields for messages. Children read their own full
chapters, author and save Markdown, then send a receipt naming the chapter and
assigned summary file. The owner assembles those saved texts unchanged in index
order. No required paragraph notes or typed bullet reports are added.

The completion gate checks saved-file assembly only. The reward also requires
matching native child receipts, but neither proves semantic fidelity, retained
handles, exactly-once sends or source-access discipline. Those require actual
trace and source/output inspection. Missing outputs are never filled in.

The smoke launcher now selects the depth-zero model from the configured task
mode. Owner modes use the owner checkpoint; terminal probes use the worker.
The existing proxy sends depth-positive requests to the worker. Previously the
summary smoke launcher always passed the worker to both role engines; a supplied
owner path alone therefore could not establish owner routing.

Local verification:

- 47 taskset cases passed, including existing modes, default three-chapter jobs,
  ordered two-file input, source-only setup, missing file capture, matching and
  mismatched child receipts, and changed/reordered assembly rejection.
- The generated gate runs locally; deliberately unchecked prose can pass an
  unchanged assembly check. This tests the documented non-semantic boundary.
- Ruff and shell syntax checks passed. The native evaluator resolves the new
  config without model calls; actual full-chapter task materialization is checked
  separately from config parsing.
- A temporary check executed the real artifact-export section and real launcher
  mode-selection section: Unicode and CRLF text were byte-exact, missing output
  remained absent, empty captured output remained empty, receipts matched, and
  owner_direct/owner/direct_probe selected the intended depth-zero checkpoint.
- Pinned Prime Agent source `97b994c` shares the parent's working directory with
  RLM children and disables autonomous continuation for depth-positive sessions
  in `resolveRuntimeSessionOptions`. The owner gate is not a child-summary gate.
  Actual child execution and stopping remain to be observed in the live probe.

`smoke-owner-direct.toml` selects complete Alice and Bennett first chapters, in
that order. Both are development-exposed and TRAIN-excluded, not fresh
confirmation. The native harness/version and sampling match the direct probes;
depth is one and the existing owner workflow's 20-minute episode allowance is
retained. No teacher summary is supplied.

The owner remains the acquired adaptive-core-v1-e33 checkpoint:

`/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2`

Remote STABLE and weight SHA256 were rechecked during preparation:
`e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47`.
The intended worker is the R6 descendant after its final checkpoint is verified.
No restart, replacement harness, owner weight update or live integration success
is claimed. R6's frozen GPU checkout is unchanged. Synchronization and the native
probe wait for verified training completion and process drain.
