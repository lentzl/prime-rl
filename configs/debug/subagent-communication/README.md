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

The full-epoch parallel checkpoint retained 4/4 exact held-out direct and single-child
behavior. On held-out parallel tasks, all four rollouts spawned two children and
received both replies; three were exact and three were fully protocol-aligned. Since
the guided parallel probe still had useful reward variance, refine this rung with the
low-memory inference server left running on the same GPU:

```bash
uv run inference @ configs/debug/subagent-communication/inference.toml \
  --vllm.model /ephemeral/subagent-rung/outputs/03-parallel-sft-r1/weights/step_48
uv run rl @ configs/debug/subagent-communication/04-parallel-grpo.toml
```

`num_infer_gpus = 0` means externally managed inference, not no inference. Keep the
17%-utilization server healthy on ports 8000 and 8100 while the trainer uses the
remaining GPU memory.

Evaluate GRPO checkpoints 2 and 4 against both the guided parallel probe and the
held-out direct/single/parallel gate. Advance from the strongest admitted checkpoint,
which may remain the parallel SFT checkpoint if either policy update regresses an
earlier family. Update the model path in `05-followup-sft.toml` accordingly, then
generate the bidirectional follow-up corpus and train its half- and full-epoch
checkpoints:

Both GRPO checkpoints were rejected by the guided gate. The admitted SFT step 48
averaged 1.292 reward and 0.75 answer accuracy with four clean completions. GRPO step
2 averaged 1.188 and 0.5625, hit the turn limit once, and increased repeated cells;
step 4 averaged 1.167 and 0.4375 and also hit the turn limit once. Both preserved the
basic two-child shape, but neither improved it without behavioral regression. The
follow-up rung therefore starts from SFT step 48.

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/05-followup-sft-r1/train.json \
  --instances 8 \
  --families direct single parallel followup \
  --harness-trace /ephemeral/subagent-rung/evals/parallel-selected-heldout/traces.jsonl
uv run sft @ configs/debug/subagent-communication/05-followup-sft.toml
```

The 256 examples preserve all earlier families and add 32 follow-up parents plus 32
follow-up children. The child computes and retains a subtotal, asks the parent for a
withheld multiplier, receives it in a later turn, and sends the completed result back.
The exporter rejects any example containing an identical repeated tool call. This
keeps the supervised protocol aligned with the runtime `duplicate_cells` admission
metric rather than teaching polling loops that our evaluator later penalizes.

Select between steps 32 and 64 with
`prime_agent_qwen35_subagent_followup_train_probe.toml`, then run
`prime_agent_qwen35_subagent_followup_rung_eval.toml`. Admission requires all four
families to preserve answer correctness and their exact protocol shape; the follow-up
family must show both message directions while withholding the multiplier from the
initial child prompt.

The first follow-up seed did not pass this gate. Step 32 had no aligned rollouts and
three of four samples exhausted 32 turns. Step 64 terminated cleanly and reduced
repeated cells, but still had zero aligned rollouts: children confused parent/child
message direction, parents failed to retain handles, and some children attempted
interactive `input()` instead of consuming the ordinary parent message. Do not refine
either checkpoint with RL.

Repair role conditioning from the admitted parallel checkpoint instead:

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/06-followup-role-sft-r2/train.json \
  --instances 8 \
  --families direct single parallel followup \
  --followup-copies 3 \
  --harness-trace /ephemeral/subagent-rung/evals/parallel-step48-heldout/traces.jsonl
uv run sft @ configs/debug/subagent-communication/06-followup-role-sft.toml
```

This 384-example repair mix keeps the 192 earlier-family examples and intentionally
raises follow-up parent/child traces to half of the corpus. The spawn contract names
the child role, forbids self-delegation and child-directed messages from that role,
and makes ending then resuming after a parent message explicit. Steps 48 and 96 remain
behavioral selection points; lower training loss alone is not admission evidence.

The role-conditioned repair also failed admission. Step 48 completed all four guided
rollouts but had zero answer accuracy and zero aligned protocols. Step 96 became
blocked inside its first harness episode for more than six minutes with no model
requests after the initial exchange, so stop it rather than waiting through four
rollout timeouts. Exact transcript repetition improved imitation loss but did not
teach the causal pause/resume boundary.

Do not add another hand-authored SFT variant. Collect successful follow-up episodes
from a stronger instruct teacher in the real Prime Agent harness, preserve the actual
root and child branches, and export only observed successful protocols for 2B
distillation. Use `Qwen/Qwen3.5-9B` as the temporary teacher: it fits this 48 GiB GPU
for inference, shares the learner's tokenizer and chat family, and does not change the
2B deployment target. Teacher traces must pass the same scorer before entering a new
SFT corpus; failed or partially aligned traces remain eval evidence, not training
demonstrations.
