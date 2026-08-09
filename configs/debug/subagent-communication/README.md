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
to `Read the file` and omitted the task-specific path. Broad replay SFT, concentrated
parent-control SFT, and a 64-example exact-spawn micro-rung did not change that first
action. Do not continue literal SFT for this failure.

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
single-child, and parallel) at temperature `0.8`. Rung 37 solved all four direct
tasks, all four single-child answers, and three of four parallel answers. Only two
single traces were fully protocol-aligned because they sent a redundant second child
message. The fourth parallel trace received both correct child replies but ignored
their message bodies, tried to parse the delegated shard files as message envelopes,
and exhausted 24 turns. Overall, 10/12 episodes jointly satisfied answer and protocol
requirements. This confirms that path-bearing delegation transferred more strongly
than asynchronous reply binding.

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

Keep rung 37 step 2 as the selected Stage-0 candidate. Preserve branch-specific OPSD
as the correct multi-agent abstraction, but do not apply another all-branch dose to
this model. A future SDFT retry for fan-in should explicitly train only coordinator
branches and leave already-functional child behavior untouched, then face the same
paired direct/single/parallel gate. The broader remaining curriculum still includes
bidirectional follow-up, traceback recovery, output contracts, and clean stopping.
