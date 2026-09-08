# English chapter summarization with Prime Agent

The current learning path is a complete chapter → 3–5 English key bullets,
within one persistent Prime Agent/IPython session. It reuses
the existing worker lineage and runtime. The broader goal includes integrating
the useful worker into the existing task-owner/delegation path; that integration
is not established by a terminal-worker test.

## Ordinary chapter input

`smoke-evidence-direct-city-shade.toml` demonstrates a UTF-8 chapter file via
`env.taskset.chapter_path`. Paragraphs separated by blank lines receive source IDs
for the episode. The worker reads the source and directly writes key bullets;
there is no required intermediate notes file. No teacher notes or
reference summaries are supplied at evaluation.

With a model endpoint already running, use the native evaluation entry point
from the repository root and override the source path as needed:

```bash
PYTHONPATH="$PWD/deps/verifiers/environments/document_summary_v1:$PWD/deps/verifiers" \
uv run eval @ experiments/qwen35-2b-document-summary-prime-agent-v1/smoke-evidence-direct-city-shade.toml \
  --env.taskset.chapter-path /absolute/path/to/chapter.md \
  --model MODEL_NAME --client.base-url http://127.0.0.1:8102/v1
```

The existing summary evaluation driver also extracts exact captured source,
notes, captured notes and summary text into `document/artifacts/*.md` when each
is present. Missing output remains missing; the exporter does not fill it in.
These are model-produced diagnostic artifacts, not certified summaries. Full
calls, errors and lifecycle information remain in the native trace.

The `smoke-evidence-direct-alice-ch1.toml` and
`smoke-evidence-direct-bennett-ch1.toml` configs select `direct_probe`: read the
same complete chapter and directly write 3–5 key English bullets in Prime Agent.
No notes stage or paragraph-ID checklist is required. The gate checks bullet
structure and word limits only; it supplies no reference answer. Existing
file-write recovery remains available. This changes the prompt, stages and
structural feedback together, so comparisons do not isolate notes alone.
Use `DOCUMENT_SUMMARY_PROBES='direct-alice-ch1 direct-bennett-ch1'` with the paired
driver. The staged path and its results remain intact.

`upstream-staged-public-chapters-r1-results.json` records the published upstream
reference versus R2 before this simplification. Neither model produced usable
key-bullet summaries in that four-episode screen; the upstream Bennett output
was a paragraph-by-paragraph obligation report. This cautions against attributing
the failures to training alone. The reference is not a proven exact ancestor.

`direct-public-chapters-r1-results.json` records the first direct comparison:
all four episodes save summary text, but factual inventions/reversals and format
failures remain. Native ACP stopping reasons must be retained: visible text is
not evidence that the autonomous correction allowance remained available.
The direct configs allow 32 turns and 262,144 cumulative tokens while retaining
the existing 10-minute episode timeout and output ceiling. This allows a bounded
correction opportunity on longer chapters; it is not a campaign resource quota.

## Training and interpretation

R3 completed 32 full-dense updates from R2 on 40 direct read/write/stop episodes:
20 retained TRAIN cases and 20 complete public chapters. R4 preparation retains
all 40 cases exactly and adds four reviewed chapters for 44 episodes; its planned
64-update run continues from R3 without requiring a held-out pass threshold.
See `direct-sft-r3-run.json`, `direct-sft-r4-run.json`, and
`public-chapter-training-v1.md`. Experimental progress and reliable-model
promotion remain separate; the Owner accepts behavioral-collapse risk.

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

The R2/R3 direct comparison is `r2-vs-r3-direct-four-chapters-r1`; its results
must be reviewed before claiming improvement. Earlier staged results and
experiment records remain preserved.
