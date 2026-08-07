# IPython Foundations

This experiment trains persistent notebook semantics before file, skill, or recursive
agent curricula. It starts from the complete adaptive-skills smoke snapshot and uses
the `ipython-foundations-v1` environment pinned through `deps/verifiers`.

## Setup

```bash
export HF_TOKEN="$HF_KEY"
uv sync --all-extras
uv pip install -e deps/verifiers/environments/ipython_foundations_v1
uv run hf download \
  lentzl/rlm-prime-agent-qwen35-adaptive-skills-smoke-r1-20260806 \
  --revision f453c92bc67453c03c82b6e40481abc71e1c3772 \
  --local-dir /ephemeral/models/qwen35-adaptive-skills-smoke-r1
```

Hydrate the whole repository at the pinned revision rather than selecting only model
files. This preserves the model and its companion snapshot artifacts together.

## Three Rungs

Start the colocated inference server and wait for both ports:

```bash
uv run inference @ configs/debug/ipython-foundations/inference.toml
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8100/health
```

Run each held-out baseline before its matching training recipe. Rung 1 uses SDPO. The
later rungs first seed the structural behavior with SFT because sparse process rewards
did not teach the required call boundaries reliably.

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_completion_eval.toml \
  --model /ephemeral/models/qwen35-adaptive-skills-smoke-r1 \
  --output-dir /ephemeral/ipython-rungs/evals/01-completion-base
uv run rl @ configs/debug/ipython-foundations/01-completion-rl.toml

uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_assignment_eval.toml \
  --model /ephemeral/ipython-rungs/outputs/01-completion/weights/step_4 \
  --output-dir /ephemeral/ipython-rungs/evals/02-assignment-base
uv run python scripts/export_ipython_assignment_sft.py \
  /ephemeral/ipython-rungs/data/02-assignment-sft/train.json \
  --harness-trace /ephemeral/ipython-rungs/evals/02-assignment-base/traces.jsonl
uv run sft @ configs/debug/ipython-foundations/02-assignment-sft.toml

uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_state_eval.toml \
  --model /ephemeral/ipython-rungs/outputs/02-assignment-sft/weights/step_24 \
  --output-dir /ephemeral/ipython-rungs/evals/03-state-base
uv run python scripts/export_ipython_state_sft.py \
  /ephemeral/ipython-rungs/data/03-state-sft-replay/train.json \
  --harness-trace /ephemeral/ipython-rungs/evals/03-state-base/traces.jsonl \
  --assignment-replay /ephemeral/ipython-rungs/data/02-assignment-sft/train.json
uv run sft @ configs/debug/ipython-foundations/03-state-sft.toml
```

Restart inference from each merged checkpoint before evaluating or starting the next
rung. Rung 2 writes `step_24`; rung 3 writes `step_20`. Every gate uses four samples
for each held-out task to reduce single-sample noise. Completion must improve
`process_aligned` and accuracy while reducing calls; assignment must improve
`silent_assignment_recovered`; state must improve `cross_turn_state_reused`. Re-run
all earlier gates after each cumulative rung and do not advance on answer accuracy
alone.

LoRA weight checkpoints must contain actual deltas from their starting checkpoint.
Before publishing, sample adapted tensors to confirm nonzero differences and verify
that every numeric EOS field resolves to the tokenizer's `<|im_end|>` token. The
state-only SFT ablation passed its own gate but erased rung-2 behavior; assignment
replay in the cumulative rung is therefore required.

After the cumulative SFT checkpoint passes all three gates, optionally refine state
behavior with SDPO:

```bash
uv run rl @ configs/debug/ipython-foundations/03-state-rl.toml
```

Only after all three rungs pass should `rl.toml` reintroduce recovery and subprocess
families. For recovery, require `recovery_round_coverage` to improve. For subprocesses,
require complete process-result observation and error-directed operation revision
without increasing raw PDF-byte fallbacks.

## Rung 4: Error-Directed Recovery

The locally verified cumulative checkpoint is the start of this rung:

```bash
uv run hf download \
  lentzl/rlm-prime-agent-qwen35-ipython-foundations-r1-20260807 \
  --revision 4bb1d809b79c34705f23ad2f069dea5ec09db943 \
  --local-dir /ephemeral/models/qwen35-ipython-foundations-r1
uv run inference @ \
  configs/debug/ipython-foundations/04-recovery-inference.toml
```

Run the recovery and foundation-regression baselines independently before training:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_recovery_eval.toml
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_foundation_regression_eval.toml
uv run rl @ configs/debug/ipython-foundations/04-recovery-rl.toml
```

This rung uses GRPO because the learner must execute the stale operation and receive
the live kernel traceback before choosing its correction. Prompt-inserted or fixed
golden traceback text is not used. Guided training states the recovery invariant but
does not reveal executable repair code; held-out evaluation removes the family-specific
hint while retaining the shared notebook policy.

Evaluate every four-step checkpoint on both profiles. Select a checkpoint only when
`recovery_round_coverage`, `document_text_extracted`, and
`summary_reused_extraction` improve while `document_extra_errors`,
`repeated_error_signatures`, `file_acquisition_calls`, and raw-byte fallbacks do not.
The earlier completion, silent-assignment, state-continuity, and answer-accuracy gates
must remain within baseline variance. Do not publish the final step merely because it
is last; publish the best checkpoint that passes both capability and regression gates.

## Rung 5: Typed File Processing

Start from the locally verified recovery checkpoint rather than the original Qwen
snapshot:

```bash
uv run hf download \
  lentzl/rlm-prime-agent-qwen35-ipython-recovery-r2-20260807 \
  --revision cb6acc9b6187b34645c83b9e5b876c2ea226bb9c \
  --local-dir /ephemeral/models/qwen35-ipython-recovery-r2
uv run inference @ \
  configs/debug/ipython-foundations/05-file-processing-inference.toml
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_file_processing_eval.toml
```

Build a compact replay set and the file-processing demonstrations. One assignment
instance contributes 12 examples, one state instance contributes four examples, and
the new matrix contributes 20, for 36 total examples:

```bash
uv run python scripts/export_ipython_assignment_sft.py \
  /ephemeral/ipython-rungs/data/05-assignment-replay/train.json \
  --instances 1
uv run python scripts/export_ipython_state_sft.py \
  /ephemeral/ipython-rungs/data/05-state-replay/train.json \
  --instances 1 \
  --assignment-replay \
    /ephemeral/ipython-rungs/data/05-assignment-replay/train.json
uv run python scripts/export_ipython_file_processing_sft.py \
  /ephemeral/ipython-rungs/data/05-file-processing-sft/train.json \
  --instances 5 \
  --replay /ephemeral/ipython-rungs/data/05-state-replay/train.json
uv run sft @ configs/debug/ipython-foundations/05-file-processing-sft.toml
```

The 18 SFT steps are exactly two epochs at batch size four. Restart inference from
`/ephemeral/ipython-rungs/outputs/05-file-processing-sft/weights/step_18` and run the
file-processing and foundation-regression gates independently. The supervised seed
teaches the short call structure; GRPO remains responsible for executing controlled
failures in the live kernel and adapting to their actual output:

```bash
uv run inference @ \
  configs/debug/ipython-foundations/05-file-processing-inference.toml \
  --model.name \
    /ephemeral/ipython-rungs/outputs/05-file-processing-sft/weights/step_18
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_file_processing_eval.toml \
  --model /ephemeral/ipython-rungs/outputs/05-file-processing-sft/weights/step_18
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_foundation_regression_eval.toml \
  --model /ephemeral/ipython-rungs/outputs/05-file-processing-sft/weights/step_18
uv run rl @ configs/debug/ipython-foundations/05-file-processing-rl.toml
```

Evaluate SFT before spending RL compute. Advance only if structured-result inspection,
path reuse, extraction, and `grounded_file_answer` improve without regressing silent
assignment or state continuity. During GRPO, compare steps 4, 8, and 12; reject a
checkpoint that raises repeated-cell or extra-error counts even if answer accuracy
improves. Malformed, scanned, encrypted, and unknown inputs may terminate with an
evidenced limitation, but an empty parser result without diagnosis is not success.

### Rung 5 run record

The first run started from
`lentzl/rlm-prime-agent-qwen35-ipython-recovery-r2-20260807` at revision
`cb6acc9b6187b34645c83b9e5b876c2ea226bb9c`. The 36-example replay mix completed
18 SFT steps, followed by a bounded four-step GRPO run. Held-out file-processing
evaluation used the same 18 tasks and sampling settings for every candidate:

| Candidate | Process score | Grounded answer | Path reuse | Outcome observed | Extra errors | Repeated calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recovery base | 0.103 | 0.167 | 0.056 | 0.167 | 4.444 | 1.611 |
| SFT step 18 | 0.354 | 0.222 | 0.333 | 0.389 | 1.833 | 0.056 |
| GRPO step 3 | **0.492** | **0.389** | **0.444** | **0.667** | **1.556** | 0.278 |
| GRPO step 4 | 0.338 | 0.389 | 0.444 | 0.556 | 1.944 | 0.722 |

Step 3 was selected instead of the final checkpoint. It retained perfect completion
and cross-turn state-continuity scores over four held-out samples per family. Silent
assignment recovery remained 1.0 and its process score improved from 0.300 after SFT
to 0.771, but exact final-answer accuracy was only 0.25; assignment reliability is
therefore the next regression target. The selected policy also repeated more cells
than the SFT seed, so the no-repeat metric remains a hard gate for the next rung.

Local Omnigent validation of the published merge confirmed clean PDF extraction, CSV
aggregation, silent assignment, session reload, and generation termination; its full
suite reported 258 passed and one skipped. The remaining failures define the next
rung: a full-document request repeated `len(reader)` seven times instead of repairing
it to `len(reader.pages)`, a recovery probe changed its operation without proving a
successful repair, one CSV reply ignored the JSON contract, and a PDF answer reversed
an extracted negation.

## Rung 6: Document Control

Use the verified rung-5 merge as the starting point:

```bash
uv run hf download \
  lentzl/rlm-prime-agent-qwen35-file-processing-r1-20260807 \
  --revision bf46092e9792359edfd514a2cd57108827e6c171 \
  --local-dir /ephemeral/models/qwen35-file-processing-r1
uv run inference @ \
  configs/debug/ipython-foundations/06-document-control-inference.toml
```

The new family pairs affirmative and negated source claims and executes four real PDF
failures, including `len(reader)`. A changed post-traceback cell receives only partial
credit; aligned repair requires successful all-page extraction and displayed source
evidence. Final answers must parse as the exact requested JSON object.

Generate 24 new examples and replay all 36 rung-5 examples. At batch size four, 30 SFT
steps are exactly two epochs over the resulting 60-example dataset:

```bash
uv run python scripts/export_ipython_document_control_sft.py \
  /ephemeral/ipython-rungs/data/06-document-control-sft/train.json \
  --instances 6 \
  --replay /ephemeral/ipython-rungs/data/05-file-processing-sft/train.json
uv run sft @ configs/debug/ipython-foundations/06-document-control-sft.toml
```

The first two-epoch run made the document-control gate perfect but regressed the held-out
assignment gate: process score fell from `0.771` to `0.591`, final correctness from
`0.250` to `0.125`, and identical retries rose from `0.250` to `0.750`. Preserve that
step-30 checkpoint as an ablation and test the otherwise identical one-epoch recipe:

```bash
uv run sft @ \
  configs/debug/ipython-foundations/06-document-control-sft-1epoch.toml
```

Do not select either SFT duration from training loss. Compare the document-control,
file-processing, and foundation gates below; only then point the GRPO config at the
selected checkpoint.

Evaluate the published base and SFT result on three independent gates before GRPO:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_document_control_eval.toml
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_file_processing_eval.toml
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_foundation_regression_eval.toml
uv run rl @ configs/debug/ipython-foundations/06-document-control-rl.toml \
  --max-steps 4
```

Advance only when `repair_outcome_observed`, `full_document_text_extracted`,
`json_contract_followed`, and `source_grounded_claim` improve without increasing
identical calls or repeated error signatures. File-processing process score and the
completion, assignment, and state gates must remain within baseline variance. Compare
each available adapter rather than assuming the final GRPO step is best.
