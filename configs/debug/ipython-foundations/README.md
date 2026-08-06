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

Run each held-out baseline before its matching training recipe. Rung 2 starts from the
merged step-8 weights written by rung 1, and rung 3 starts from rung 2:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_completion_eval.toml
uv run rl @ configs/debug/ipython-foundations/01-completion-rl.toml

uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_assignment_eval.toml
uv run rl @ configs/debug/ipython-foundations/02-assignment-rl.toml

uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_state_eval.toml
uv run rl @ configs/debug/ipython-foundations/03-state-rl.toml
```

Restart inference from each merged step-8 weight directory before evaluating or
starting the next rung. Every gate uses four samples for each held-out task to reduce
single-sample noise. Completion must improve `process_aligned` and accuracy while
reducing calls; assignment must improve `silent_assignment_recovered`; state must
improve `cross_turn_state_reused`. Do not advance a checkpoint on answer accuracy alone.

Only after all three rungs pass should `rl.toml` reintroduce recovery and subprocess
families. For recovery, require `recovery_round_coverage` to improve. For subprocesses,
require complete process-result observation and error-directed operation revision
without increasing raw PDF-byte fallbacks.
