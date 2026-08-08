# Subagent Communication

This rung starts from the selected Qwen 3.5 2B file-processing checkpoint and trains
the native Prime Agent depth-one protocol. It is Stage 1 of the
[recursive-specialization target](../../../docs/recursive-specialization-target.md):
the goal here is reliable delegation mechanics, not yet autonomous decomposition or
recursive specialization.

Hydrate the pinned starting point:

```bash
export HF_TOKEN="$HF_KEY"
uv run hf download \
  lentzl/rlm-prime-agent-qwen35-file-processing-r1-20260807 \
  --revision bf46092e9792359edfd514a2cd57108827e6c171 \
  --local-dir /ephemeral/models/qwen35-file-processing-r1
```

Start inference, establish the held-out baseline, then run GRPO:

```bash
uv run inference @ configs/debug/subagent-communication/inference.toml
uv run eval @ deps/verifiers/configs/prime_agent_qwen35_subagent_communication_eval.toml \
  --output-dir /ephemeral/subagent-rung/evals/base
uv run eval @ deps/verifiers/configs/prime_agent_qwen35_subagent_communication_train_probe.toml \
  --output-dir /ephemeral/subagent-rung/evals/train-probe
uv run rl @ configs/debug/subagent-communication/rl.toml
```

The train probe mirrors one GRPO group for each delegated family. Start RL only when
at least one family shows nonzero within-group protocol variance; otherwise seed the
native protocol with supervised traces before returning to on-policy optimization.

Restart inference from each four-step checkpoint and rerun the held-out eval. Delegated
answer credit is gated on complete protocol alignment, while `answer_accuracy` remains
available as a diagnostic metric. Select a checkpoint only when `answer_accuracy` and
`protocol_aligned` improve together. The
`direct` family must retain zero spawns, `single` must retain one named handle and one
child reply, `parallel` must retain two named handles and two replies, and `followup`
must withhold the multiplier at spawn and show both message directions. Any increase
in `duplicate_cells` is a regression. Re-run the IPython foundation and
file-processing gates before publishing.
