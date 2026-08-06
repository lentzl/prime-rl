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

## Smoke Gate

Start the colocated inference server and wait for both ports:

```bash
uv run inference @ configs/debug/ipython-foundations/inference.toml
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8100/health
```

Run the untrained ten-stream held-out baseline, then the four-step smoke:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_foundations_eval.toml
uv run rl @ configs/debug/ipython-foundations/rl.toml \
  --max-steps 4 \
  --output-dir /ephemeral/outputs/prime-agent-qwen35-ipython-foundations-smoke-r1
```

After the final weight update is loaded by inference, repeat the same eval command.
Proceed to the unmodified 48-step `rl.toml` only if all optimizer steps are finite,
rollouts complete without infrastructure errors, and held-out stream accuracy or the
family diagnostics improve without an increase in identical consecutive IPython calls.
The subprocess family must also improve complete result observation and error-directed
operation revision without increasing raw PDF-byte fallbacks.
For recovery streams, require `recovery_round_coverage` to improve rather than relying
on aggregate answer accuracy: each round must expose real kernel feedback and repair it
with retained state.
