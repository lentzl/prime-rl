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

Run the continuity baseline, then the four-step continuity smoke:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_continuity_eval.toml
uv run rl @ configs/debug/ipython-foundations/continuity-rl.toml \
  --max-steps 4 \
  --output-dir /ephemeral/outputs/prime-agent-qwen35-ipython-continuity-smoke-r1
```

After the final weight update is loaded by inference, repeat the same eval command.
The continuity eval runs eight held-out task definitions twice to reduce single-sample
noise. Promote the four-step adapter to the 16-step `continuity-rl.toml` only if
`cross_turn_state_reused`, `silent_assignment_recovered`, and `process_score` improve
without more identical calls or IPython calls. Do not promote on answer accuracy alone.

Only after continuity passes should `rl.toml` reintroduce recovery and subprocess
families. For recovery, require `recovery_round_coverage` to improve. For subprocesses,
require complete process-result observation and error-directed operation revision
without increasing raw PDF-byte fallbacks.
