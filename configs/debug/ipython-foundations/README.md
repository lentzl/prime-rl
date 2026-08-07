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
