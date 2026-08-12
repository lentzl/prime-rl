# Subagent Communication

This rung starts from the selected Qwen 3.5 2B file-processing checkpoint and trains
the native Prime Agent depth-one protocol. It is Stage 0 of the
[recursive-specialization target](../../../docs/recursive-specialization-target.md):
the goal here is reliable delegation mechanics, not yet autonomous decomposition or
recursive specialization.

The completed 2B evidence and the controlled capacity transition are summarized in
[QWEN35_2B_CAPABILITY_REPORT.md](QWEN35_2B_CAPABILITY_REPORT.md) and
[QWEN35_4B_TRANSITION_PLAN.md](QWEN35_4B_TRANSITION_PLAN.md). Rung 37 remains the
retained 2B checkpoint even while the coordinator experiment advances to 4B.
The three disjoint standard gates and bidirectional screen are immutable under the
[Frozen Capacity Battery V1](FROZEN_CAPACITY_BATTERY_V1.md); new training curricula
must use separate tasks and may not evolve this historical test surface.

Do not run `pytest` on a host with live Prime-RL training or inference processes.
The repository-wide module fixture in `tests/conftest.py` deliberately runs
`pkill -f torchrun` and `pkill -f VLLM` to clean CI zombies, so even a focused unit
test terminates an unrelated live experiment. Validate before launch or on a separate
host; during a run, use read-only log, trace, and health checks only.

The 4B parent-admission process-reward probes are configured in runs 102 through 105.
Run 102 is the rejected mixed-reward control, run 103 verifies that strict sparse
reward correctly produces no update, and run 104 is the rejected one-step dense probe.
Run 105 completed a bounded four-step dense dose from the unregressed run-100 step-8
initialization. Its step 3 improved held-out admission but remained `0/8`
protocol-complete on the full single-child development gate, so no checkpoint was
promoted or published. The selection rule and exact evidence are recorded in
[`qwen35-4b-admission-control-results-v1.json`](qwen35-4b-admission-control-results-v1.json).

The next probe targets the complete causal prefix rather than another admission-only
dose. `106-qwen35-4b-causal-chain-preflight.toml` scores five ordered stages: spawn
first, bind the exact safe contract, perform coordinator-local work, receive the child
reply, and finish with a protocol-aligned correct answer. A later stage receives no
credit after an earlier stage fails. Run the eight-rollout preflight from run-100 step
8 before defining any optimizer update; continue only if it has nonzero within-group
variance beyond the first stage.

The run-100 step-8 preflight produced causal rewards
`[0, 0, 0, 0.2, 0, 0, 0, 0.8]`. The strongest trajectory completed the first four
ordered stages and failed only final answer completion, while otherwise successful
late-spawn trajectories correctly received zero. This admits one `1e-7` GRPO update
in `107-qwen35-4b-causal-chain-grpo.toml` on a fresh task. The update must still pass
fresh direct, causal-chain, and full normal single-child gates before promotion.

Run 107 was numerically stable and preserved the fresh direct screen at `4/4`, but its
training group contained only one spawn-first trajectory (`0.2`) and no deeper causal
prefix. On the disjoint run-108 gate, only one of eight trajectories again reached
spawn-first and none bound the contract or progressed further. The checkpoint is
rejected without running Frozen Capacity Battery V1. The causal scorer remains useful;
the missing ingredient is successful behavior to learn from, not weaker admission.
The next intervention will collect real-harness Qwen3.5-9B teacher trajectories and
retain only complete five-stage chains before distilling into the run-100 step-8 4B
learner. Exact evidence is recorded in
[`qwen35-4b-causal-chain-results-v1.json`](qwen35-4b-causal-chain-results-v1.json).

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
mkdir -p /ephemeral/subagent-rung/xdg-config
export XDG_CONFIG_HOME=/ephemeral/subagent-rung/xdg-config
export VLLM_NO_USAGE_STATS=1
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

Prime Agent 0.7.0 source identifies the representation bug. Agent messages use
`deliveryMode = "steer"`: a busy target queues the prompt without blocking the sender,
and an idle target accepts it as a new prompt. The first two curricula incorrectly
kept the parent in one tool-calling turn and simulated progress with polling. For a
causal exchange, the parent must end its turn after spawn and again after sending the
multiplier, allowing each queued child message to resume it.

The first dense follow-up GRPO run exposed a scorer bug before producing an admissible
checkpoint. Its count-based gate credited two child-to-parent messages and one
parent-to-child message even when the child guessed and sent its result before the
parent follow-up. The corrected scorer correlates incoming message IDs with their
originating successful sends and requires request send, parent follow-up send, and
result send in that order. Those three phases contribute separate protocol checks so
GRPO retains within-group signal before any rollout masters the full exchange, while
`protocol_aligned` remains all-or-nothing. Re-scoring step 4 reduced aligned rollouts
from two of four to one of four. Step 2 retained two fully aligned rollouts and three
causally ordered rollouts, so it is the repair starting point; neither checkpoint had
an exact answer.

Run `09-followup-causal-grpo.toml` on fresh generated instances. Select only with a
fresh guided probe using different seed and offset values, then run the full held-out
family gate. Online protocol reward or message counts alone are not admission
evidence.

The phase-dense run was stopped after step 2. It improved request and parent-follow-up
frequency, but produced zero post-follow-up child results; the second update retained
only one looping group after zero-advantage filtering. Do not continue optimizing that
checkpoint. Use `10-handshake-grpo.toml` from the selected pre-repair step 2 instead.
Its opt-in `handshake` family keeps the real daemon, child admission, turn boundaries,
and both message directions, but replaces shard parsing and arithmetic with a withheld
nonce that the resumed child must echo. Advance back to full follow-up only after a
fresh handshake probe shows exact answers and causally aligned exchanges.

The first handshake collection exposed a second evaluator issue: tool call IDs are
session-local, so root and child branches both emit values such as `call_0`. Pairing
tool outputs by call ID alone attributed successful sends to unrelated cells. Pair by
the trace graph edge `(assistant node, call ID)` instead. With that correction, the
untrained handshake batch contains four fully causal exact echoes out of eight, plus
two partial exchanges. Discard the update made under the incorrect attribution and
rerun the same clean-output config from the selected full-follow-up step 2.

Do not gate admission on assigning the `rlm()` return value. Prime Agent can address a
named child directly through `receiver_name` and `agent_observe`; a fresh handshake
trace completed the exact causal exchange this way. Keep `retained_handles` as a
diagnostic metric, while protocol admission requires the native spawn, explicit
messages, correct causal order, and no repeated cells rather than one Python style.

The first fresh step-2 handshake probe produced one exact causally aligned exchange
out of four, three exact final answers, and no turn-limit failures. This is a foothold,
not admission to the full rung. Run the two-step `11-handshake-grpo-r2.toml` refinement
on new instances with the behavior-based gate, then require a stronger fresh probe
before restoring the full shard-follow-up task.

The R2 refinement is rejected. Although its online batches still contained causal
handshakes, a fresh four-instance probe of step 2 produced one exact answer, one causal
exchange, zero aligned exchanges, and two turn-limit failures. This regressed from the
first step-2 probe's three exact answers, one exact aligned exchange, and zero turn-limit
failures. Retain `10-handshake-grpo-r1/weights/step_2` as the handshake candidate and
test whether its causal behavior transfers to fresh full follow-up instances; do not
promote or continue `11-handshake-grpo-r2/weights/step_2`.

The transfer comparison also rejects the handshake candidate as the next full-follow-up
checkpoint. On the same fresh seed and offset, the selected full-follow-up parent
`08-followup-grpo-r1/weights/step_2` completed the causal exchange in three of four
rollouts and passed the full protocol gate in two. The handshake candidate reached the
request and parent-follow-up phases in three rollouts but produced no post-follow-up
child result, leaving causal and aligned completion at zero. It terminated more cleanly
and had fewer duplicate cells, but that process-control gain does not compensate for
erasing the required final message. Keep the handshake experiment as evidence about
the isolated mechanism; resume full-follow-up work from `08-followup-grpo-r1/weights/step_2`.

The paired parent traces identify a supervised-data bug rather than an arithmetic
capacity limit. A child retained the correct subtotal and received the correct
multiplier, but multiplied by an invented constant. The exporter had shown the real
`[from parent]` message and then assigned the gold multiplier directly in the next code
cell, bypassing the observation-to-state operation we expected the learner to infer.
The corrected examples first bind the exact message body to `parent_message_body`, parse
it with `int(...)`, and only then combine it with the retained subtotal. Export-time and
unit-test validation reject direct numeric multiplier assignments.

Generate a mixed-family repair set with follow-up examples at 40% of the corpus, then
train half- and full-epoch checkpoints from the stronger full-follow-up parent:

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/12-followup-message-grounding-sft-r1/train.json \
  --instances 8 \
  --families direct single parallel followup \
  --followup-copies 2 \
  --seed 20260907 \
  --instance-offset 120 \
  --harness-trace /ephemeral/subagent-rung/evals/followup-parent-step2-compare-fresh/traces.jsonl
uv run sft @ configs/debug/subagent-communication/12-followup-message-grounding-sft.toml
```

The 320 examples yield selection points at steps 40 and 80. Promote only if a fresh
guided follow-up probe shows the child deriving its multiplier from the incoming body
and sending a correct post-follow-up result, followed by retention of the earlier
standard direct, single, and parallel gates.

Native Prime Agent children steer their parent asynchronously after the spawn turn.
Run delegated-family selection through Prime Agent's `--autonomous` mode with the
shape-only completion gate written by the task setup; a conventional one-response
Verifiers episode tears down the daemon before a later child steer can resume the root.
Keep this isolated probe below Prime Agent's default turn-25 continual-refinement
checkpoint so delegation and refinement are measured separately. Gate feedback must
preserve the existing child and tell an idle parent to yield without tools; a generic
"fix the result" message caused the 2B coordinator to reopen delegated shards and spawn
replacement children.

Both `12` checkpoints are rejected under this correct lifecycle. Step 40 reached a
child request in one fresh trajectory but malformed the parent follow-up, then polled
until the turn cap. Another fresh trajectory repeatedly generated invalid
`await variable = await ...` spawn syntax. Step 80 regressed further and failed to
repair the same syntax after real tracebacks. The pre-SFT
`08-followup-grpo-r1/weights/step_2` remains the stronger repair base: it spawned
legally and its child recovered a parser error and requested the multiplier, although
the parent still failed to bind the multiplier from the original task.

The audit also found two supervision contradictions. The environment system prompt
still instructed parents to poll `agent_observe`, contrary to Prime Agent's native
"end the turn and accept explicit messages" contract. In addition, trace-seeded SFT
exports copied that obsolete environment paragraph verbatim. Parent prompts and SFT
demonstrations are now event-driven, and the exporter replaces only the environment
tail while retaining the authentic Prime Agent base prompt.

Repair the native messaging atoms at low rate from the stronger parent:

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/13-native-message-control-sft-r1/train.json \
  --instances 8 \
  --families direct single parallel followup \
  --protocol-atoms \
  --seed 20260915 \
  --instance-offset 220 \
  --harness-trace /ephemeral/subagent-rung/evals/parent08-followup-eventdriven-r1/traces.jsonl
uv run sft @ configs/debug/subagent-communication/13-native-message-control-sft.toml
```

The 416-example set contains 160 focused atoms for legal spawn assignment, each
message direction, traceback-informed correction, depth-one restraint, and yielding
without polling, alongside direct/single/parallel replay. The run uses LoRA BF16
AdamW at `8e-6` for only 32 steps and stores steps 16 and 32. Select by fresh autonomous
causal traces; training loss is not admission evidence.

Train the corrected turn-boundary trace once from the admitted parallel checkpoint;
do not carry forward the rejected follow-up weights or exact-copy oversampling:

```bash
uv run python scripts/export_subagent_communication_sft.py \
  /ephemeral/subagent-rung/data/07-followup-turn-boundary-sft-r3/train.json \
  --instances 8 \
  --families direct single parallel followup \
  --harness-trace /ephemeral/subagent-rung/evals/parallel-step48-heldout/traces.jsonl
uv run sft @ configs/debug/subagent-communication/07-followup-turn-boundary-sft.toml
```

The exporter now fails if either incoming child message is not preceded by an
assistant turn boundary. Follow-up parents use no polling cells: spawn and yield;
resume on the request; reply and yield; resume on the result; return bare JSON.

This native-idle representation is correct for a persistent Prime Agent daemon but
not for the current single-response Verifiers episode: a plain root assistant message
ends the episode before a later steer can resume it. Three guided step-32 samples
stopped after only 2--4 turns with rewards between 0.625 and 0.75. Preserve this as a
harness-integration finding; do not train further on the turn-boundary checkpoints.

Use the r1 full-epoch seed as the highest-variance live policy and optimize its dense
protocol components directly:

```bash
uv run inference @ configs/debug/subagent-communication/inference.toml \
  --vllm.model /ephemeral/subagent-rung/outputs/05-followup-sft-r1/weights/step_64
uv run rl @ configs/debug/subagent-communication/08-followup-grpo.toml
```

The four-step run is exploratory and retains steps 2 and 4. Admit neither by training
reward alone: both must beat r1 on a disjoint guided probe and preserve the earlier
direct, single, and parallel held-out families.

## Autonomous retained-state refinement

Rung 14 repaired the supervised parent-state boundary: follow-up parents establish
`multiplier` and assign the result of `await rlm(...)` to `child` in the same cell,
then reuse both variables after the child request. Its step 16 checkpoint progressed
through the request and parent follow-up on a fresh autonomous probe, but the child did
not return the correct final result. Treat it as the supervised parent for online
refinement, not as an admitted orchestrator.

Rung 15 ran two GRPO updates through the complete autonomous Prime Agent lifecycle.
The second online batch reached the causal request, follow-up, and result phases in
half of its samples, but retained no child handles. On the matched held-out follow-up
task, step 2 completed one causal exchange while the paired rollout repeated 11 cells;
neither retained the `rlm` result or returned the exact answer. Keep step 2 as the
pre-retention comparison, not as an admitted checkpoint.

Rung 17 added dense stateful-control credit and ran three synchronous updates with
`max_train_batch_lead = 0`. Online handle retention rose from 12.5% to 25%, but final
causal completion fell to zero. On the matched held-out pair, one rollout retained the
handle but never sent the follow-up, while the other sent the follow-up without
retaining the handle; both exhausted 20 turns. This proves that retention and causal
messaging can each occur, but the policy still trades them off.

Rung 18 gated all stateful-control credit on a retained child and made one lower-rate
GRPO update. Its effective group included the sole retained-handle trajectory, so the
signal was not removed by zero-advantage filtering. Nevertheless, both matched
held-out rollouts discarded the handle. They did both complete the full causal message
ordering, and one stopped autonomously in 14 turns without repeated cells, making rung
18 the strongest causal-communication candidate. It is still rejected as the first
orchestrator because retained state, exact answer synthesis, and reliable stopping are
all missing.

## OPSD omitted-path repair

The evidence-bound rung 30 preserved direct and parallel behavior but failed the
held-out single-child gate: all three parent prompts compressed the delegated request
to `Read the file` and omitted the task-specific path. This was a delegation-fidelity
failure, not a document-parser failure: the coordinator discarded the known absolute
path before the child ever saw the task. Broad replay SFT, concentrated parent-control
SFT, and a 64-example exact-spawn micro-rung did not change that first action. Do not
continue literal SFT for this failure.

Prime-native OPSD supplies a more local learning signal. The taskset stores a
task-specific successful demonstration, while the ordinary student prompt remains
unchanged. Before training, use the teacher-conditioned evals to establish that the
same live policy can express the missing behavior when given the demonstration:

```bash
uv run eval @ configs/debug/subagent-communication/35-opsd-teacher-preflight.toml
uv run eval @ configs/debug/subagent-communication/35-opsd-teacher-variance.toml
```

The deterministic trajectory and all three sampled trajectories preserved the exact
path, named the child, and retained the handle. This `4/4` teacher admission is the
necessary precondition for OPSD; do not train if it fails. OPSD uses one on-policy
rollout per example (`group_size = 1`) and the paper's demonstration template. With
LoRA, set `trainer.sdpo_loss.teacher_regularization = "live-policy"`; the EMA teacher
does not support LoRA. Sampling temperature must be exactly `1.0` for every distilled
token. `RLConfig` now validates this during dry-run rather than after rollout
collection.

Run one conservative update, inspect its held-out initial action, then extend only
when the path probability moves:

```bash
uv run rl @ configs/debug/subagent-communication/36-single-path-opsd.toml
uv run eval @ configs/debug/subagent-communication/36-initial-spawn-eval.toml
uv run rl @ configs/debug/subagent-communication/37-single-path-opsd-dose.toml
```

Rung 36 moved the held-out exact-path rate from `0/3` to `1/3`. In the dose run,
step 2 was selected over step 3: its on-policy batch carried the path in `4/4`
trajectories, it had the lowest SDPO loss (`0.233`) and mismatch KL, and its gradient
norm was `8.19` rather than step 3's `42.25`. The selected checkpoint is:

```text
/ephemeral/subagent-rung/outputs/37-single-path-opsd-dose-r1/weights/step_2
```

Fresh held-out evaluation of this checkpoint produced:

- initial-action gate: `4/5` exact task-specific paths;
- full single-child gate: `3/3` exact, protocol-aligned solves;
- direct no-delegation regression: `3/3` exact solves with zero spawns;
- parallel regression: `3/3` exact answers and exact payloads, with `2/3` fully
  protocol-aligned fan-ins; one trajectory finalized after only one observable child
  reply.

This is the first robust causal evidence that OPSD repaired the omitted-path behavior
without erasing direct work or parallel payload binding. Treat rung 37 step 2 as the
current Stage-0 candidate, not the final orchestrator. Its next gates are standard
instruction levels on unseen variants, reliable two-reply parallel fan-in,
bidirectional follow-up, traceback recovery, output contracts, and clean stopping.

The selected adapter was merged into the complete dense rung-18 parent before remote
preservation. The export audit found 617 tensors in both artifacts. All 96 LoRA target
matrices exactly matched `base + 2.0 * B @ A` after BF16 conversion, and all 521
non-target tensors remained bit-identical. Greedy adapter and dense-merge evaluations
then produced the same exact path-bearing spawn action. A separate sampled dense
full-episode smoke exposed the candidate's remaining variance: it shortened the child
prompt to `Read the file`, received only a progress message, and entered a repeated
invalid async-repair loop. This is not a merge mismatch, but it is direct evidence
that rung 37 is not yet the frozen master.

## Standard-prompt and parallel-provenance gates

The first broad standard-prompt gate used 12 unseen tasks (four each of direct,
single-child, and parallel) at temperature `0.8`. The authoritative trace metrics for
rung 37 were `4/4` direct, `2/4` single, and `3/4` parallel joint answer-and-protocol
solves: `9/12` overall with mean reward `1.6944`. Single failures included redundant
messages or recovery loops. The fourth parallel trace received both correct child
replies but ignored their message bodies, tried to parse delegated shard files as
message envelopes, and exhausted 24 turns. Path-bearing delegation therefore
transferred more strongly than asynchronous reply binding.

The first parallel OPSD extension reused one task-level coordinator demonstration for
all three trainable branches. The on-policy batch looked healthy, but fresh paired
evaluation rejected the resulting rung 41 checkpoint: parallel accuracy fell from
`3/4` to `2/4`, and mean reward fell from `1.694` to `1.625`. Trace inspection exposed
the semantic error. Child branches were being scored against a coordinator
demonstration; children sometimes computed without sending, while the coordinator
invented polling APIs instead of consuming already-delivered messages.

OPSD now accepts either one demonstration string or an exact initial-question keyed
mapping. The subagent environment supplies separate coordinator and child
demonstrations, so the algorithm remains harness-agnostic while each branch receives
supervision for its actual role. Multi-branch sampled-node deduplication also ensures
that a shared sampled prefix contributes loss only once. The corrected rung 43 update
trained cleanly (`loss 0.0321`, mismatch KL `0.0004`, gradient norm `2.61`) and improved
parallel answer accuracy to `4/4`. It nevertheless reduced parallel protocol alignment
to `2/4`, introduced extra sends and duplicate cells, and reduced single-child answer
accuracy to `3/4`; mean reward was `1.611`. Rung 43 is therefore rejected too.

Coordinator-only OPSD then established two additional controls. Rung 45 used null
demonstrations for child branches but still distilled every sampled coordinator
response. It fell to `6/12` (`4/4` direct, `1/4` single, `1/4` parallel; mean reward
`1.4375`) because the broad coordinator loss disturbed spawning and waiting as well
as final synthesis. Rung 47 filtered the loss to the immediate response after each
child message. Its first gate improved to `10/12`, but a disjoint replication exposed
a capability transfer rather than a promotion: rung 37 scored `4/4` direct, `2/4`
single, and `3/4` parallel, while rung 47 scored `4/4`, `4/4`, and `1/4`. Both were
`9/12`; rung 47 strengthened single-child completion at the expense of parallel
fan-in because it also trained the response after only the first child reply.

The environment-specific `keep_complete_fan_in_response` filter corrects that causal
boundary. It selects exactly the first coordinator response after every expected
child has replied, selects it only once, and leaves child branches and all earlier
coordinator actions at zero SDFT weight. The one-update rung 50 smoke trained cleanly
on four nearly perfect trajectories (`1.9375` reward, loss `0.0760`, mismatch KL
`5.5e-6`, gradient norm `3.75`). Its held-out gate was stopped after 11 episodes once
promotion became mathematically impossible: `4/4` direct, `2/4` single, and `1/3`
parallel were joint solves, for `7/11` and mean reward `1.5909`; even a final success
could only reach `8/12`, below rung 37's `9/12`.

Rungs 53 and 54 tested whether the smoke simply lacked corrective diversity. Four
cumulative complete-fan-in updates used 16 fresh on-policy trajectories at half the
learning rate. The first three updates were numerically stable; the fourth batch
exposed a paraphrased child question that was absent from the exact demonstration
map. Coordinator-only mappings now include a `"*": null` fallback, so any canonical
or paraphrased child branch is excluded without environment-specific role logic in
Prime-RL. The repaired final update completed with reward `1.9375`, loss `0.0381`,
effectively zero mismatch KL, and gradient norm `2.69`.

A new paired gate still rejected the scaled candidate. On the identical 12 tasks,
rung 37 scored `4/4` direct, `3/4` single, and `1/4` parallel (`8/12`, mean reward
`1.7292`). Rung 54 scored `4/4`, `2/4`, and `2/4` (`8/12`, mean reward `1.6250`). The
narrow SDFT signal converted one parallel failure but introduced one single-child
failure and did not improve the total. This is useful causal evidence: selective SDFT
can move the intended behavior in Qwen3.5-2B, but repeated task-level demonstrations
still trade adjacent harness capabilities rather than producing robust mastery.

Keep rung 37 step 2 as the selected Stage-0 candidate. Preserve null branch mappings
and generic OPSD token filters as the correct abstractions, but do not add another
dose of this same fan-in demonstration. The next intervention should collect and
score the concrete process-control failures themselves: consume visible child message
bodies, prohibit invented polling after explicit replies, repair from actual
tracebacks, and stop after a valid final object. The broader remaining curriculum
still includes bidirectional follow-up, output contracts, and clean stopping.

## Post-fan-in process-control GRPO

The next experiment measured the coordinator phase after all expected child messages
were visible. The branch-aware metric ignores child-side cells and permits ordinary
local aggregation, but counts coordinator failures, repeated cells, and calls that
cannot advance a completed single or parallel fan-in (`rlm`, observation, roster,
receive, or message calls). Each count contributes `1 / (1 + count)` to a dense
control score; a missing complete fan-in scores zero. The reward is opt-in through
`reward_post_fan_in_control`, so historical and unrelated tasksets are unchanged.

Before training, the rung 56 probe sampled four rollouts for each of two identical
fresh parallel prompts. Both groups had substantial within-prompt variance. Revised
total rewards ranged from `1.103` to `2.667` and from `1.464` to `3.000`; the second
group contained two completely clean trajectories. This admitted the signal for a
single conservative GRPO update rather than proving the checkpoint would improve.

Rung 57 started from rung 37 step 2, used eight parallel trajectories, learning rate
`5e-7`, and one update. Training was stable: reward `2.4643`, exact answer accuracy
`0.8125`, complete fan-in `7/8`, clean post-fan-in control `6/8`, loss `0.0445`,
mismatch KL `0.00057`, gradient norm `0.457`, and peak memory `10.2 GiB`.

The fresh paired gate rejected the update. Rung 37 scored `4/4` direct, `2/4` single,
and `3/4` parallel joint solves (`9/12`, mean historical reward `1.7986`). Rung 57
scored `4/4`, `2/4`, and `1/4` (`7/12`, mean `1.6667`). Parallel clean control fell
from `4/4` to `3/4`; single clean control fell from `3/4` to `2/4`, with 14 failed
post-fan-in cells and nine repeated cells in the candidate's single-child traces.

This result falsifies the assumption that adding an accurate local reward to the
same arithmetic fan-in distribution is sufficient. Six of eight training trajectories
already saturated the new control signal, so the one-step update mostly learned from
adjacent answer and protocol variation and harmed retention. Keep rung 37 selected.
The next process-control batch must include the actual unsolved failure regimes across
single and parallel tasks, plus explicit retention groups; do not run another
parallel-only update merely because its on-policy batch reward is high.

## Cross-family retention and dense training

Rung 59 sampled four trajectories from each train-split single and parallel prompt to
find tasks with genuine policy variance rather than repeatedly optimizing saturated
examples. Rung 37 jointly solved `25/32` trajectories (`12/16` single and `13/16`
parallel). Rung 60 then made one low-rate GRPO update on one mixed-success prompt from
each family. On the exact paired gate it reached `10/12`, but only by exchanging the
retained model's split: rung 37 scored `4/4` direct, `4/4` single, and `2/4` parallel;
rung 60 scored `4/4`, `2/4`, and `4/4`. This is capability transfer, not mastery.

A separate OPD update against frozen rung 37 restored the original `4/4`, `4/4`,
`2/4` split without improving it. Combining hard-parallel GRPO and single-child OPD
inside one optimizer step did not solve the interference: rung 64 scored `8/12`, and
a second weighting control scored `9/12`. Independent component normalization and
per-source loss weights are now available for principled mixed objectives, but loss
weighting alone cannot replace broader causal examples.

Before testing full-parameter SDPO, the selected rung-37 adapter was exported through
Prime-RL's checkpoint path as a standalone dense model. Both artifacts contained 617
tensors. All 96 adapted matrices matched `base + 2.0 * B @ A` at every sampled
coordinate, all 521 untouched tensors were bit-identical, and all numeric EOS fields
remained `248046`. This dense export is the exact training base for subsequent
full-weight experiments, not a behaviorally chosen replacement for rung 37.

Multi-step full-weight runs must use
`configs/debug/subagent-communication/inference-dense.toml`, not the shared LoRA
inference profile. With `enable_lora = true`, vLLM wraps Qwen3.5 linear-attention
parameters under `base_layer`; an in-place dense broadcast then fails because the
checkpoint correctly contains the unwrapped `conv1d.weight` key. A frozen OPD teacher
must also run in a separate worker, as configured by
`inference-dense-teacher.toml` on ports `8200/8300`. LoRA training can share one
worker because the base weights remain resident while adapters change. A dense policy
broadcast replaces the resident base itself, so sharing that worker would silently
turn the supposed frozen teacher into the updated policy after step 1.

## Branch-matched SDPO and SDFT preservation

Prime-RL's SDPO path now pairs successful and failed multi-agent branches by their
initial user question. A successful coordinator branch supervises only the matching
coordinator branch, and each child branch is matched independently; reasoning, tool
calls, and assistant text are preserved in the successful replay. Prime Agent stores
user content as OpenAI-style text-part lists, so branch keys are normalized through
the renderer's content helpers before exact matching. Ambiguous duplicate questions
remain a hard error rather than receiving a guessed teacher.

Rung 67 applied one full-weight, EMA-teacher SDPO update to eight trajectories of one
mixed-success parallel task. Seven trajectories were trainable. Training was stable
(`loss 0.0103`, mismatch KL `0.0006`, gradient norm `19.5` clipped to `1.0`, peak
memory `23.2 GiB`), but the paired gate scored `9/12`: `4/4` direct, `3/4` single,
and `2/4` parallel. Branch-correct hindsight is therefore operational end to end, but
this isolated dose did not improve the selected policy.

Rung 69 combined the same parallel SDPO group with eight fresh one-rollout OPSD/SDFT
preservation examples. The SDFT source used coordinator-only demonstrations at
`loss_weight = 0.25`; dynamic child prompts were excluded with the explicit wildcard
null mapping instead of unsafe fuzzy matching. All 16 trajectories were trainable,
and the full-weight update was stable (`loss 0.0059`, mismatch KL `0.0005`, gradient
norm `16.625` clipped to `1.0`, peak memory `23.2 GiB`). Its exact paired result was
`10/12`: `4/4` direct, `3/4` single, and `3/4` parallel. Relative to rung 37, it
converted one parallel failure but introduced one single-child failure.

Rung 69 is rejected and rung 37 remains selected. The experiment supports a narrow
conclusion: SDFT is effective for a known deterministic harness procedure, as already
shown by the original exact-path repair, but generic coordinator rehearsal does not
guarantee retention of adjacent multi-agent behavior. The next curriculum must train
the concrete failures themselves, include direct/single/parallel retention in every
batch, and earn promotion on repeated held-out gates rather than a single aggregate
score.

## Bidirectional capacity boundary

Rungs 70 through 77 isolated the missing bidirectional primitive. Standard held-out
follow-up and handshake traces did not complete one causally ordered child request,
parent response, and child result. Full-weight OPSD updates were numerically stable,
but guided request training did not transfer to standard language: all four rung-76
screens and the rung-77 response-phase screen had zero aligned protocols. Rung 77
itself produced zero child requests across eight held-out tasks.

Direct child-role probes explain why. Given an explicit API instruction, the 2B child
sent `await agent_message.send('need multiplier', receiver_role='parent')` in `24/24`
post-compute samples. Under the canonical standard prompt it instead emitted plain
assistant text in `24/24` samples. The frozen 2B teacher also failed this transition:
the original standard demonstration produced `0/16` tool calls, and a response-aligned
request-only demonstration produced only `1/24`. SDFT cannot provide a dependable
target when the admitted teacher distribution does not express the action.

This closes the current 2B coordinator search without discarding the model. The exact
dense rung-37 snapshot is retained privately on HF at revision
`b469454738dfc911f43233f172ca4ff920ea695d` and remains the preferred 2B expert-child
base. The next coordinator experiment starts from untouched `Qwen/Qwen3.5-4B`, reuses
the same task families and gates, and recalibrates teacher admission and optimization
rather than replaying 2B update doses blindly.

## Thinking 27B native ownership admission

The teacher-first program now starts from untouched thinking-mode
`Qwen/Qwen3.5-27B`, not from a no-thinking adapter or a larger dose of the earlier
synthetic ownership bootstrap. Runs 249 and 250 sampled 64 answer-free guided first
decisions and admitted only responses that passed all seven structural ownership
checks. Seven native decisions passed, covering every train prompt variant. The
exporter recomputes admission from the trace, requires native reasoning and one
IPython action, removes the temporary guidance, caps each task at one example, and
records immutable source and dataset hashes.

Run 251 made one rank-16 SFT step at `5e-8` over that corpus. The update was stable but
run 252 rejected it on an unguided paired gate. Follow-up dense ownership improved
from `0.250` to `0.330`, but both arms retained the multiplier `0/16` times and
completed `0/16` strict transitions. The candidate also regressed the unseen handshake
control from `0.848` to `0.795` dense ownership. Do not publish, continue, or distill
from run 251. Exact artifacts and hashes are recorded in
`qwen35-27b-native-ownership-results-v1.json`. The next experiment must focus the
teacher signal on the state-binding decision itself and first repair the prior OPSD
trainer/inference discrepancy; another ordinary full-response SFT epoch is not the
next step.

## Qwen3.5-27B trainer/inference alignment

The first-response discrepancy was an inference-state precision error, not evidence
against OPSD. A zero-update replay with vLLM's default bfloat16 recurrent SSM cache
reproduced the failure: 268 selected tokens had mean mismatch KL `13.1937`, while a
fresh vLLM prefill reproduced the rollout log probabilities closely. Qwen3.5 uses
chunked GDN for prefill and recurrent GDN only during decode, so forcing recurrent
GDN across the trainer's full sequence is not a faithful repair and was discarded.

Setting vLLM `mamba_ssm_cache_dtype = "float32"` while retaining the trainer's
standard chunked forward restored agreement. Across two independent first responses
and 482 selected tokens, mean mismatch KL was `0.000220`, median was zero, p95 was
`0.000839`, and mean absolute trainer/inference log-probability delta was `0.00713`.
Both FSDP ranks completed a zero-learning-rate forward/backward step with gradient
norm `0.4785`. The audit also established that a two-rank run needs at least two
sequences; a one-sequence diagnostic can put ranks on different collective paths.

Exact controls, hashes, discarded diagnostics, and the implementation decision are
recorded in `qwen35-27b-first-response-alignment-results-v1.json`. The next allowed
intervention is one low-rate first-response OPSD step from untouched thinking-mode
27B, followed by a disjoint paired natural gate. Do not broaden the curriculum or
promote the adapter merely because its training step is numerically healthy.

Run 254 completed that intervention with healthy numerics over 2,219 selected tokens:
loss `0.00871`, mismatch KL `0.000255`, gradient norm `0.3438`, and no rollout errors.
The disjoint run-255 gate nevertheless rejected it. Follow-up answer and causal
completion remained `4/4`, but protocol alignment fell from `1/4` to `0/4`, mean
coordinator access to the child-owned path rose from `0.5` to `1.5`, and
bidirectional-control score collapsed from `0.477` to zero. All four candidate
follow-up coordinators read the delegated file; the untouched base did so in one.

This separates the two problems cleanly. FP32 recurrent state fixed the numerical
error, but whole-response demonstration-conditioned OPSD remains the wrong teaching
surface for this primitive. It guided 218-371 tokens per response and did not isolate
the state assignment, retained handle, and ownership-transfer actions. Keep untouched
thinking-mode 27B and do not retry the dose at another learning rate. The next
candidate mechanism is action-local successful-sibling SDPO on grouped native first
decisions, admitted before any update. Exact artifacts and metrics are recorded in
`qwen35-27b-aligned-first-response-opsd-results-v1.json`.

The next zero-update audit narrowed the existing OPSD target without changing its
teacher context. Prime-RL now retains every sampled token in the teacher completion
and carries separate target offsets for selected tokens; this is required for any
non-contiguous SDPO or OPSD filter. The Qwen-specific environment filter selected
only the first coordinator's serialized `<tool_call>...</tool_call>` span. Across
two fresh trajectories it selected 216 tokens in exactly two contiguous spans, with
zero reasoning, child, or later-coordinator tokens. Mean mismatch KL remained
`0.000129` and the maximum was `0.00520`.

This audit validates the mechanics, not the policy. The selected spans are the
student's on-policy actions; the demonstration conditions the teacher distribution
evaluated on those states. The next falsifiable intervention therefore keeps the
same demonstration and `1e-7` learning rate as run 254 but applies gradients only to
the first executable action. If its fresh paired gate still increases coordinator
access to child-owned files, reject this reprompt family rather than changing the
learning rate or broadening the dose. Exact audit artifacts and hashes are recorded
in `qwen35-27b-action-local-token-audit-results-v1.json`.

Run 257 completed that intervention over 679 selected action tokens. Its loss
(`0.00661`), mismatch KL (`0.000192`), and gradient norm (`0.668`) were healthy, and
every sample shipped to the two FSDP ranks carried a nonzero routed loss. The latter
required fixing `TrainSink`: a first attempt exposed that a trainable parent rollout
could ship its all-zero child branch and put the ranks on different teacher-replay
collective paths.

The fresh run-258 gate rejects the adapter. On follow-up tasks, answers remained
`4/4` and natural causal completion rose from `3/4` to `4/4`, but protocol alignment
fell from `2/4` to `0/4`, coordinator access to child-owned paths rose from `0.0` to
`1.75` per trace, and bidirectional control fell from `0.500` to `0.159`. Handshake
behavior was exactly flat at `4/4` protocol alignment and `0.886` control. Action
locality therefore did not fix the target: both whole-response and action-only
demonstration-reprompt OPSD reinforce the wrong ownership behavior. Do not search
another learning rate or mask width in this family. Exact evidence is recorded in
`qwen35-27b-action-local-opsd-results-v1.json`.

Run 259 implements the next bridge's mandatory zero-update admission screen rather
than another teacher demonstration. `ownership-invariant-v1` materializes eight
resource families in the runtime and scores only the first child-owned coordinator
decision. Each of eight tasks received eight untouched thinking-mode 27B rollouts.
The frozen strict gate produced `0/64` successes and zero mixed groups, so native
sibling SDPO was not admitted and no weights changed. Correcting the verifier to
recognize coordinator dictionaries as persistent state did not alter that outcome.

Ten trajectories missed only the literal post-spawn-statement atom. Their trailing
statements merely displayed or stored the returned handle; none polled, discovered,
read the delegated path, or continued local work. They remain recorded as failures
until the definition is resolved prospectively. If passive handle inspection remains
forbidden, revise the ordinary executable curriculum or sampling and repeat Phase A.
If it is deemed non-substantive, revise the verifier first and rerun all 64 rollouts;
do not rescore this batch into admission after observing it. Exact counts and hashes
are in `qwen35-27b-native-sibling-admission-results-v1.json`.

The prospective semantic-yield screen and its predeclared sampling extension now
admit the native sibling source. Run 260 alone produced `5/64` strict successes but
only three mixed groups. Run 261 added eight independent rollouts to the same tasks;
the combined group size of 16 produced `11/128` successes across six mixed resource
families and both phrasings. This exceeds the frozen four-group, three-family,
two-phrasing gate without changing any weights. Exact counts and hashes are recorded
in `qwen35-27b-native-sibling-admission-results-v2.json`.

Run 262 is the required zero-LR replay audit. It samples one empirically mixed JSON
group and one empirically no-success CSV group at group size 16, disables environment
feedback, preserves authentic native successful reasoning, and filters loss to the
first coordinator tool call. The audit must prove sibling provenance, no-success
zero-target routing, mask locality, and trainer/inference agreement before the first
native-sibling SDPO update.

Run 262 uses live-policy teacher regularization because Prime-RL rejects EMA with
LoRA: the current EMA implementation creates a second full model. At zero learning
rate, and at teacher-evaluation time in a single-step dose, live policy is exactly the
newly initialized EMA teacher. This equivalence does not extend to multiple optimizer
steps; that later lane needs adapter-only EMA support or full-weight capacity.

The first Run 262 sample was inconclusive: both the nominally mixed JSON group and the
CSV no-success control produced `0/16` strict successes. All 32 samples therefore had
zero SDPO signal and were correctly removed before the trainer; there was no token
export or weight change. Run 263 keeps every audit semantic fixed and broadens only
the sampling cohort to four previously mixed families plus the CSV control. Its
`80` rollouts are a reliability measure for obtaining an authentic replay, not a
learning-budget increase because the optimizer remains at zero learning rate.

Audit the completed run from its serialized native traces and trainer token exports:

```bash
uv run python scripts/audit_native_sibling_sdpo_run.py \
  --config configs/debug/subagent-communication/264-qwen35-27b-native-sibling-sdpo-token-audit-served-fix.toml \
  --traces /ephemeral/subagent-rung/outputs/264-qwen35-27b-native-sibling-sdpo-token-audit-served-fix-r1/run_default/rollouts/step_1/train/all/traces.jsonl \
  --token-exports /ephemeral/subagent-rung/outputs/264-qwen35-27b-native-sibling-sdpo-token-audit-served-fix-r1/run_default/token_exports/step_1 \
  --output /tmp/run264-native-sibling-audit.json
```

The report reconstructs the exact first successful sibling selected in each
same-task group, retains its full native demonstration and content hash, maps every
trainer-exported sequence back to its trace branch, and rejects any discrepancy
between the configured first-tool-call filter and nonzero SDPO token weights.

Run 263 is an invalid plumbing diagnostic, not a zero-success model result. The
separate Verifiers server reconstructed tasks from the serialized `taskset.task`
subtree, whose yield policy was still literal even though the taskset-level curriculum
knob was semantic. It was stopped at 51/80 rollouts with zero trainer steps and no
token exports. Verifiers commit `be7e0576` propagates that policy across the served
boundary and adds a reconstruction regression test. Run 264 is the fresh repeat, with
the semantic task subtree also explicit in its TOML.

Run 264 passes Phase B. Four of five 16-rollout groups each produced two strict
native successes; the JSON group remained the no-success control. The trainer
exported all 64 replay branches from the four mixed groups and none from the control.
The auditor matched every export to its exact same-group sibling provenance and
configured first-tool-call mask: 6,152 tokens were selected, all eight unique teacher
demonstrations retained native reasoning, and mean sequence mismatch KL was
`0.000173`. The zero-LR step reported loss `0.0038` and gradient norm `0.6758` but
could not update weights. Exact teacher text, IDs, hashes, trace/export hashes, and
numerical evidence are in
`qwen35-27b-native-sibling-sdpo-token-audit-served-fix-results-v1.json`.

Run 265 is the single authorized learning dose. It is identical to Run 264 except for
experiment identity, output path, and `trainer.optim.lr=1e-7`. It must be followed by
the frozen ownership, direct-control, natural follow-up, handshake, and independent
repeat screens before another optimizer step is considered.

The valid Run 265 rerun completed one optimizer step. All five groups produced native
successes, yielding 78 replay branches and 8,526 selected first-tool-call tokens. The
nonzero-dose audit matched every export to its exact trace branch, mask, and same-group
native teacher; mean sequence mismatch KL was `0.000287`. The trainer reported loss
`0.00314`, gradient norm `0.7656`, LR `1e-7`, and a successful update. The stable adapter
and exact native demonstrations are recorded in
`qwen35-27b-native-sibling-sdpo-dose-results-v1.json`.

Run 266 rejected that candidate. On 32 matched ownership/direct tasks it produced two
strict gains and one loss, below the frozen six-gain minimum; child-owned path leakage
and direct overdelegation both remained zero. The natural screen found a sharp split:
handshake protocol improved from `2/4` to `4/4` and mean bidirectional control from
`0.341` to `0.886`, but follow-up causal completion fell from `4/4` to `3/4` and four
coordinator accesses to child-owned paths remained. Both primary gates therefore failed,
Run 267 was not launched, and no second optimizer step is permitted. Exact paired
metrics and trace hashes are in `qwen35-27b-native-sibling-selection-results-v1.json`.

Run 268 returns to untouched 27B and performs no learning. It samples 16 native
rollouts for each of the eight coordinator-owned admission families using the same
resource seed and instance as the child-owned Run 265 cohort. The prospective admission
gate requires strict native successes in at least six families, with at least two
families producing multiple successes. Passing would justify a paired native-sibling
SDPO design that teaches both ownership arms from authentic on-policy reasoning;
failure would show that coordinator-native teacher supply must be solved before such
a cohort is viable.

For a learning dose, reuse the structural audit while explicitly declaring the expected
learning rate. The no-success fallback is not required again when it was already proven
by the immediately preceding zero-LR Phase-B audit:

```bash
uv run python scripts/audit_native_sibling_sdpo_run.py \
  --config configs/debug/subagent-communication/265-qwen35-27b-native-sibling-sdpo-dose.toml \
  --traces /ephemeral/subagent-rung/outputs/265-qwen35-27b-native-sibling-sdpo-dose-r1/run_default/rollouts/step_1/train/all/traces.jsonl \
  --token-exports /ephemeral/subagent-rung/outputs/265-qwen35-27b-native-sibling-sdpo-dose-r1/run_default/token_exports/step_1 \
  --expected-learning-rate 1e-7 \
  --no-require-no-success-group \
  --output /tmp/run265-native-sibling-audit.json
```

The first Run 265 launch is invalid operator interference, not model evidence. A
concurrent remote pytest validation activated Prime-RL's autouse zombie-cleanup
fixture, whose broad `pkill -f torchrun` and `pkill -f VLLM` commands terminated the
live trainer and inference workers after 23 saved traces and before any optimizer
step. The partial output was archived and hashed in
`qwen35-27b-native-sibling-sdpo-dose-test-interference-v1.json`. Repeat the exact
committed config with no concurrent Prime-RL tests; do not use the partial traces.

Run 266 prospectively freezes that primary selection. Its matched ownership screen
contains 16 child-owned and 16 coordinator-owned tasks over two unseen instances of
all eight training resources, using only held-out ownership phrasings. Candidate
promotion requires at least six paired strict gains, at most one paired strict loss,
zero coordinator access to child-owned paths, and zero child spawns on coordinator-
owned direct controls. The natural gate reuses the already frozen Run 258 follow-up
and handshake tasks and requires every family to preserve answer accuracy, natural
causal completion, protocol alignment, and mean bidirectional control, with zero
coordinator access to delegated paths.

Run 267 is a disjoint repeat over two further ownership instances and the historical
Run 239 natural-control repeat. It is required only if both primary Run 266 reports
pass. Neither selection run may update weights. The exact comparators are
`summarize_ownership_candidate_selection.py` and
`summarize_natural_control_selection.py`; their criteria were committed before the
Run 265 candidate existed.
