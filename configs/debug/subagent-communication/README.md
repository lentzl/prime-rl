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
at least one family shows nonzero within-group protocol variance. The initial probe
showed partial spawn variance but no delegated payloads or genuine child replies, so
seed the single-child protocol before returning to on-policy optimization:

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/01-single-sft-r4/train.json \
  --instances 8 \
  --harness-trace /ephemeral/subagent-rung/evals/step12-heldout-direct-single/traces.jsonl
uv run sft @ configs/debug/subagent-communication/01-single-sft.toml
```

The 96 compact examples are balanced between direct coordinator restraint,
single-child parent behavior, and child reply behavior. Delegated parent traces spawn
first with the same silent assignment produced by the live harness, then preserve local
state across two useful calls while the child runs. Child traces read, compute, and send
their reply in one compact tool call. Parents then use bounded native `agent_observe`
polling until the child is no longer streaming, avoiding a race with premature parent
finalization without guessing or reading the delegated shard.
They use a separate RNG seed
and instance IDs starting at 100. RL and its calibration probe use a third seed and
instance IDs starting at 20; held-out eval remains on v4/v5 at the default IDs. At
batch size four, step 12 is half an epoch and step 24 is one epoch; both checkpoints
are retained. Evaluate both on uncontaminated guided and standard held-out splits
before choosing a seed for GRPO refinement or advancing to the parallel-child stage.

The rejected r2 seed computed coordinator-local work before spawning. Its step-12
policy reached the same partial protocol score in all four disjoint rollouts but sent
zero child replies: the parent finalized while the child's concurrent generation was
still in flight. Step 24 then exhausted the turn budget in both confirmation rollouts
despite near-zero imitation loss. Treat spawn-first scheduling and live protocol eval,
not supervised loss, as admission requirements for later communication rungs.
Protocol progress and protocol-gated answer correctness have equal reward weight so
that repairing one observable protocol step provides a useful signal at 2B scale.

Restart inference from each 12-step checkpoint and rerun the held-out eval. Delegated
answer credit is gated on complete protocol alignment, while `answer_accuracy` remains
available as a diagnostic metric. Select a checkpoint only when `answer_accuracy` and
`protocol_aligned` improve together. The
`direct` family must retain zero spawns, `single` must retain one named handle and one
child reply, `parallel` must retain two named handles and two replies, and `followup`
must withhold the multiplier at spawn and show both message directions. Any increase
in `duplicate_cells` is a regression. Re-run the IPython foundation and
file-processing gates before publishing.

After a checkpoint passes both the guided single-child probe and the standard held-out
direct/single gate, refine only that admitted family first:

```bash
uv run rl @ configs/debug/subagent-communication/02-single-grpo.toml
```

This short run starts from the one-epoch r4 SFT checkpoint and keeps the disjoint RL
seed and instance offset. Do not substitute the mixed-family `rl.toml` until native
single-child communication remains reliable after GRPO and the parallel/follow-up
families have passed their own on-policy probes.

The admitted step-24 checkpoint scored 4/4 on held-out direct tasks and 4/4 on
held-out standard single-child tasks. Its first two complete GRPO groups were also
4/4 exact and protocol-aligned, so the zero-advantage filter correctly rejected all
eight rollouts and no optimizer update occurred. Stop rather than resampling an
already-saturated rung. Advance to parallel fan-out/fan-in while retaining direct and
single examples:

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/03-parallel-sft-r1/train.json \
  --instances 8 \
  --families direct single parallel \
  --harness-trace /ephemeral/subagent-rung/evals/r4-step24-heldout-direct-single/traces.jsonl
uv run sft @ configs/debug/subagent-communication/03-parallel-sft.toml
```

This produces 192 examples: 32 direct parents, 32 single parents, 32 single children,
32 parallel parents, and 64 parallel children. Parallel parents spawn both named
children before coordinator-local work, retain both handles, use bounded native
observation, and require both explicit replies. Reply order alternates across template
variants so fan-in does not depend on alpha finishing first.
