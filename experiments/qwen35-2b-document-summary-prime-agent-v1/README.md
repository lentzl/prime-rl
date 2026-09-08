# English chapter summarization with Prime Agent

The current learning path is source → worker-authored, source-linked notes →
English key bullets, within one persistent Prime Agent/IPython session. It reuses
the existing worker lineage and runtime. The broader goal includes integrating
the useful worker into the existing task-owner/delegation path; that integration
is not established by a terminal-worker test.

## Ordinary chapter input

`smoke-evidence-city-shade.toml` demonstrates a UTF-8 chapter file via
`env.taskset.chapter_path`. Paragraphs separated by blank lines receive source IDs
for the episode. The worker authors notes, the workflow captures them, and the
worker reads those notes to write 3–5 English bullets. No teacher notes or
reference summaries are supplied at evaluation.

With a model endpoint already running, use the native evaluation entry point
from the repository root and override the source path as needed:

```bash
uv run eval @ experiments/qwen35-2b-document-summary-prime-agent-v1/smoke-evidence-city-shade.toml \
  --env.taskset.chapter-path /absolute/path/to/chapter.md \
  --model MODEL_NAME --client.base-url http://127.0.0.1:8102/v1
```

The existing summary evaluation driver also extracts exact captured source,
notes, captured notes and summary text into `document/artifacts/*.md` when each
is present. Missing output remains missing; the exporter does not fill it in.
These are model-produced diagnostic artifacts, not certified summaries. Full
calls, errors and lifecycle information remain in the native trace.

## Training and interpretation

Evidence curriculum R2 retains the sixteen earlier source cases and adds four
longer synthetic explanatory chapters. Forty extraction/realization phase
samples teach faithful selection and file persistence using the current
model-visible stage feedback. Only current-phase assistant messages receive
loss. Teacher content is authored TRAIN supervision, not native policy replay.
City-shade, Northstar and Cedar evaluation source texts are excluded.

File presence, paragraph IDs, bullet format and length are structural checks.
The fixed-fixture keyword metric is only a proxy; unrestricted chapter files
have no fabricated semantic score. Review actual source → notes and notes →
summary relationships before claiming useful fidelity. Current checkpoints are
experimental; falling training loss does not imply promotion.

Use the existing two-GPU allocation efficiently, including matched models on
separate GPUs. Explicit run bounds protect against runaway experiments; cumulative
GPU-hour totals are not permission gates. No new rental, paid service, storage
purchase or rental extension is implied.

Current evidence: `paired-r3-results.json`, `paired-r3-status.md`, and
`evidence-sft-r2-status.md`. Earlier experiment records remain preserved.
