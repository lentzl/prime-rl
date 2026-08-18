# Qwen3.5 27B Procedural Harness Master V1

This experiment makes the generated end-to-end Prime Agent coordinator policy the
optimization target. The untouched 27B thinking checkpoint is first measured on
frozen VALID-GEN and OOD-GEN indices. Training will then consume fresh,
non-overlapping TRAIN-GEN windows and promotion will depend on transfer back to
the frozen splits.

Generator `2026-08-16.v2` preserves V1's task assignments while repairing the
verification contract: completion gates validate each final JSON field's type,
including booleans, and verification prompts state that the digest is evidence
rather than the final computed result. Admission and promotion evidence from the
older generator is not reused.

Verifier revision `4f8293ac` also closes a bootstrap-reward loophole found in
the first full-weight launch. Calls to `agent_message.list_messages`,
`agent_message.list_agents`, `agent_message.recv`, and `rlm.list_subagents`
were already treated as polling positions for yield/order detection, but were
marked only as child discovery rather than the contract's forbidden `poll`
atom. Two live trajectories therefore received shaped reward for explicitly
polling after spawn. The launch was stopped after one optimizer step and is
invalid for promotion or further training. Its raw checkpoint is retained only
as a diagnostic artifact. All admission and training selection must be
rescored or rerun on `4f8293ac` or later.

Verifier revision `528e064d` repairs two additional natural-trajectory
classifications found before the corrected launch produced an optimizer step.
An awaited `agent_message.send(...)` remains successful when a later print in
the same cell hides the send result, provided the cell has no traceback. A
child request such as "Please provide the multiplier" is also equivalent to
the generator's canonical "need multiplier" message. Replaying the observed
trajectory on this revision changes it from partial shaping to a full hard-gate
success. Runs used for optimization must use `528e064d` or later.

Verifier revision `207c0d5b` closes the same polling loophole for fixed waits.
The first replacement-sampling launch admitted a partially successful sibling
that sent the requested follow-up value but then waited for the child with
`asyncio.sleep(...)` calls. Because sleep was not recorded as polling, the
trajectory incorrectly retained the no-forbidden-actions gate and received
bootstrap reward. The run was stopped at batch 4/16 before any optimizer step
or checkpoint and is not evidence. Calls named `sleep` or ending in `.sleep`
now emit the forbidden `poll` atom and prevent both initial and post-follow-up
waits from being classified as harness-native yield. Runs used for optimization
must use `207c0d5b` or later.

Verifier revision `073c224b` resolves Python aliases before classifying calls.
The sleep-fixed launch exposed this through
`from agent_message import list_agents` followed by `await list_agents()` in a
later persistent IPython cell. The unqualified call escaped the literal
`agent_message.list_agents` check and incorrectly received bootstrap reward.
The launch was stopped at batch 0/16 with no optimizer step or checkpoint and
is not evidence. The verifier now carries import, `from`-import, `as`, and
simple assignment aliases across coordinator cells. Replaying the exact live
trace changes its no-forbidden-actions gate from one to zero and its bootstrap
reward from `0.017857` to zero. Runs used for optimization must use `073c224b`
or later.

The initial baseline uses 24 episodes per split. That covers each VALID family
four times and each OOD family three times while keeping time-to-first signal
short. Later promotion evaluations can increase `count` without changing split
indices or the generator seed.

The primary reward is `harness_score`, a hard conjunction of exact answer,
required atoms, forbidden-atom absence, ordering, and cardinality. Dense metrics
exist only to diagnose why the hard gate failed.

Delegated episodes run Prime Agent in its native autonomous mode with a
task-specific completion gate. The gate keeps the ACP session alive until the
generated child-message counts and final JSON key/type shape are present; it
does not contain expected values or score policy invariants. Exact correctness
remains exclusively verifier-side in `harness_score`.

`train-admission.toml` samples eight independent rollouts for one fresh
TRAIN-GEN task from each V1 family. It is not an evaluation split. Its purpose
is to measure whether the untouched policy supplies at least one hard-gate
success and one failure in a GRPO comparison group. Families with homogeneous
groups carry no hard-reward policy-gradient signal and must not silently enter
the first optimization batch.

`bootstrap-grpo.toml` is the first benchmark-directed weight update. It uses
four synchronous full-weight BF16 AdamW steps from the untouched pinned 27B,
with eight on-policy attempts per fresh TRAIN-GEN task. The only reward is the
conjunctive executable `harness_score`; homogeneous groups are rejected before
training. The launcher refuses to start unless the admission screen contains
at least one informative non-direct comparison group.

`bootstrap-shaped-grpo.toml` is the fail-closed fallback when every delegated
hard-reward group is homogeneous. It preserves the same full-weight run but adds
at most `0.1 * bootstrap_progress`, where progress is multiplicative across exact
answer, zero forbidden actions, required atoms, ordering, and cardinality. Any
forbidden control action receives no shaping. This config may bootstrap policy
variance, but only the hard VALID/OOD score can promote its checkpoints.
The shared launcher requires a complete, rescored, error-free admission and
automatically selects hard reward when any delegated group has hard-gate
variance. It selects constrained shaping only when the hard groups are
homogeneous and at least one delegated group has measured bootstrap-progress
variance. For the first update it narrows the generated training stream to the
families that demonstrated within-group signal; promotion still evaluates all
families on the frozen broad splits.

The first launch on the repaired natural-message verifier was stopped after two
homogeneous-zero groups because `oversampling_factor=0.5` was mistakenly read as
a finite allowance of 24 raw rollouts. In Prime-RL it only derives the eight
in-flight episode permits. When the enforced zero-advantage filter rejects a
group, `TrainSink` reports its admission deficit and `RolloutDispatcher` returns
those slots to the synchronous policy budget, prioritizing a fresh replacement
group from the same source. Sampling can therefore continue until two varied
eight-sibling groups fill the 16-rollout optimizer batch. The stopped run
produced no optimizer step and is not evidence. The bootstrap configs retain
eight active episodes to respect the two-GPU recursive inference envelope.

Inference reserves 80% of each of its two L40S GPUs. This leaves room for the
largest full-weight NCCL staging bucket while retaining enough KV cache for the
eight concurrent, short bootstrap rollouts.

The launcher disables FlashInfer sampling JIT because the retained Prime host
ships CUDA runtime libraries but no `nvcc`; vLLM then uses its prebuilt sampling
path for integrated training inference.
It also hydrates Prime-RL's lock-pinned `flash-attn` extra with `--inexact` and
restores the pinned Prime router wheel when absent. It imports the trainer before
launch, so a fresh runtime-only CUDA host does not fail after admission merely
because optional prebuilt wheels were omitted.

## Reward-connectivity ramp

The alias-fixed full-composition hard-GRPO admission was stopped at exactly 32
fresh trajectories before any optimizer step. It produced zero hard passes and
no checkpoint: final answers were often correct (`0.8125`), but all 32 traces
performed forbidden control behavior, required-atom coverage averaged `0.65625`,
and ordering passed only `0.0625`. This is a disconnected hard-reward basin, not
training evidence, so repeating full-composition GRPO is rejected.

The main training program now keeps the full benchmark unchanged while crossing
the reward-connectivity threshold through strict harness-action rungs:

1. `atomic_state`: persistent coordinator action across two IPython calls.
2. `atomic_send`: one child, retained handle, passive yield, explicit delivery.
3. `atomic_child_request`: encode a request protocol in the initial child
   prompt, retain its handle, yield, and accept the explicit request.
4. `atomic_followup`: retained state and two causal resume/message cycles.
5. `atomic_parallel`: two children spawned before yield and explicit fan-in.

Each untouched/current-policy admission is eight fresh rollouts of one rung. An
all-pass rung is already mastered and advances without optimization; a mixed
group is eligible for hard GRPO; an all-fail rung is reward-disconnected and
must bootstrap only its missing causal transition before admission is repeated.
Promotion is cumulative, and the frozen broad VALID/OOD batteries remain the
external target throughout the ramp.
Candidate gates set `HARNESS_ACTION_ADMISSION_START_INDEX` to a fresh generated
index disjoint from both the untouched admission and the active training stream.

Curriculum contract `2026-08-18.harness-actions-v3` retains the frozen V2
assignments and adds the `atomic_child_request` prefix rung. V2 clarified that
`atomic_state` returns the original retained value as `marker` and the computed
sum as `result`. The hidden oracle already required that distinction, but the
V1 public prompt did not state it. Episode seeds and executable contracts are
unchanged. On the same R5 checkpoint and task seed, the ambiguous prompt scored
`7/8` despite perfect state/action metrics; the clarified prompt scored `8/8`
with every hard component at `1.0`.

The valid one-step `atomic_send` lineage currently reads: untouched `4/8`; R2
`3/8`; R3 `5/8`; R4b `6/8`; R5 `6/8`. Their accepted training batches scored
`0.625`, `0.500`, `0.5625`, and `0.6875`, respectively, with no rollout errors,
truncation, or off-policy samples. The first R4 attempt stalled at `15/16`
rollouts before any optimizer step because the environment-level deadline was
missing; it has no checkpoint and is not evidence. R5 preserves mastered
`atomic_state`, but `atomic_send` remains a mixed hard-reward group and has not
earned promotion.

R2 through R5 restarted from the config's same `1000000..1000511` TRAIN-GEN
window. Prime-RL deterministically reshuffles a finite taskset from epoch one,
so each restart began from the same prompt sequence even though completions were
fresh. Later restarts must set `HARNESS_ACTION_TRAIN_START_INDEX` to a disjoint
window. This keeps hard GRPO on-policy while preventing repeated one-step runs
from collapsing semantic diversity around the first accepted task groups.

Fresh-window R6 trained on `1001000..1001511` and scored `13/16` in its
accepted batch, but the disjoint send gate remained `6/8`. R7 used
`1002000..1002511`, scored `10/16` in training, and improved the next send gate
to `7/8` while retaining state at `8/8`. R8 then used
`1003000..1003511`, scored `6/16`, and regressed its fresh send gate to `5/8`.
This `atomic-send-grpo-r8` checkpoint is rejected and descendants must branch
from R7. Near this boundary, a
single two-group update at `5e-7` is too noisy for reliable promotion. The
launcher therefore supports validated `HARNESS_ACTION_BATCH_SIZE` and
`HARNESS_ACTION_TRAIN_LR` overrides so stabilization runs can use more
independent groups and a smaller update without changing the hard reward.
`HARNESS_ACTION_MAX_STEPS` can cap an intervention at one checkpoint before a
fresh promotion gate rather than accumulating several updates near a boundary.
The launcher derives `oversampling_factor = 8 / batch_size`, preserving the
two-GPU limit of eight concurrent recursive episodes even when more groups are
accumulated for one optimizer batch.

Live trace review also found that Prime Agent's runtime notice
`RLM child completed without sending a reply` was being classified as an
explicit child result. The verifier now accepts only real `agent_message`
deliveries, benchmark-injected failures, and visible child failure transitions;
completion notices cannot satisfy message or cardinality atoms.

The action-ramp configs also set a 900-second environment-level episode
deadline. The agent-level rollout budget only advances while a Prime Agent
segment is active; it does not bound environment-side waiting after the
coordinator yields. Without the outer deadline, one missing child delivery can
leave a GRPO comparison group in flight indefinitely even though the model and
container are idle.

Do not run Prime-RL's pytest suite on a host with active training. Its global
test fixture intentionally executes `pkill -f torchrun` before each test module
to remove CI zombies, including unrelated live trainers on the same machine.
An `atomic_child_request` optimizer attempt was invalidated this way after
rollout collection but before checkpoint export; it produced no weights and is
not training evidence. Run tests before launch or on another host. During a
live run, limit checks to commands that do not load `tests/conftest.py`, such as
shell syntax checks, Ruff, or direct standalone scripts.

Episode containers use the experiment's pinned Prime Agent runtime image rather
than installing Node and Prime Agent from the network during every setup. Build
it with `scripts/build_prime_agent_runtime_image_v1.sh`; the builder verifies
the exact Node and Prime Agent versions in a fresh container. This keeps a
transient npm or package-host outage from entering the rollout cohort as a model
failure. The failure-local follow-up SDPO bootstrap is intrinsically one step:
it collects four admitted response transitions, performs one full-weight
update, and then returns control to the frozen cumulative gates.

Failure-local SDPO R2 is a mechanically valid but non-promoted descendant of
R7. It selected one answer-free `reply_to_child_request` response from each of
four accepted traces (447 tokens total) and applied one full-weight BF16 AdamW
step at `1.25e-7`. The update validator independently matched all 447 selected
tokens to the trainer's SDPO stream, found zero RL, CE, or reference-KL token
mass, and measured a positive `10.8125` gradient norm. On a frozen generated
draw, R2 scored state `8/8`, send `6/8`, and follow-up `0/8`. R7 scored `8/8`,
`2/8`, and `0/8` on those exact task indices. The paired send improvement is
evidence that the narrow update moved harness behavior, but it did not cross
the natural follow-up boundary. R7 therefore remains canonical while R2 is
retained only as an experimental branch pending replicated paired screens. An
earlier apparent R2 state score of `0/8` is invalid and excluded: the gate was
started without an inference server and all episodes failed with HTTP 503.
Checkpoint gates now bootstrap their own server when no endpoint is supplied,
and direct local admission refuses to run against an unhealthy endpoint.

A second frozen draw replicated the direction of the send effect but rejected
R2 as a follow-up base. On exact task indices, R7 versus R2 scored send `4/8`
versus `5/8` and follow-up `0/8` versus `0/8`, both with zero rollout errors.
Across the two paired draws, send is therefore R7 `6/16` versus R2 `11/16`:
both produced exact final answers in all 16 episodes, while all required atoms
and ordering rose from `7/16` to `11/16`. Follow-up remained disconnected at
`0/16` for both. R2 had no all-required or ordered follow-up trajectories and
did not improve the aggregate required-atom or final-answer diagnostics. The
failure-local update learned a transferable part of explicit sending, but not
the complete request/reply/yield/resume transition it was intended to unlock.
Repeating the same reply-only target is therefore rejected; R7 remains the
canonical branch point for a more finely connected follow-up curriculum.

Two intermediate R2 replication attempts are excluded. The first completed
only seven send episodes before one Prime Agent process ignored the episode
deadline; the second completed seven follow-up episodes before the same idle
orphan recurred. Updating to upstream ACP `0.12.1` fixed notification ordering
but did not bound this process failure. The Verifiers fork now exposes an
opt-in `process_timeout_ms` that wraps Prime Agent with GNU `timeout`, and all
procedural configs reserve 60 seconds between that hard process deadline and
the episode deadline. The final R2 follow-up replicate completed all eight
episodes without errors under the watchdog. Gate batteries may also select a
comma-separated subset through `HARNESS_ACTION_GATE_RUNGS` while preserving
the canonical per-rung task-index offsets.

An exact eight-episode, 840-second watchdog reproduction closed normally after
the detached-descendant repair. Evaluator RSS stayed between 165 and 167 MiB,
the external 512 MiB guard never fired, all eight traces serialized as
`ok=true` without trace errors, and no containers or GPU processes remained.
Five episodes ended at `max_turns`, three at `user_closed`, and all eight hard
scores were zero; this diagnostic task window is not promotion evidence. Some
timed-out streaming requests logged `Cannot call write() after write_eof()` in
the interception server even though their enclosing traces finalized cleanly.
That separate stream-close race remains an infrastructure diagnostic. A prior
attempt in which the evaluator reached roughly 426 GiB RSS is excluded and was
not reproduced under either a one-episode probe, a short eight-episode probe,
or this exact-duration run.

The later `atomic-child-request-grpo-r8-retry1` checkpoint is a distinct,
rejected R7 descendant despite the reused short `R8` label. It applied one
full-weight BF16 AdamW hard-GRPO step at `2.5e-7` to 32 trainable
`atomic_child_request` trajectories, with training reward `25/32`, loss
`0.0002`, entropy `0.3242`, mismatch KL `0.0003`, and gradient norm `0.6406`.
Its stable 12-shard export passed EOS validation. On the exact same frozen draw,
R7 versus this candidate scored state `8/8` versus `8/8`, send `5/8` versus
`3/8`, and child request `6/8` versus `4/8`, all without runtime errors. The
candidate therefore regressed both its trained target and a prerequisite and
is rejected without another optimizer step. R7 remains canonical. Future
interventions use globally unique rung labels; full checkpoint slugs remain the
authoritative identity for these historical runs.

The protected R7 checkpoint is preserved in the private Hugging Face repository
[`R7 Hugging Face repository`](https://huggingface.co/lentzl/rlm-prime-agent-qwen35-27b-harness-r7-20260818).
It is a complete full-weight export, not an adapter: 12 safetensors shards,
tokenizer and processor metadata, resolved run configs, a stable marker, and
validated `<|im_end|>` EOS metadata are present. The model card records the
exact producing Prime-RL and Verifiers revisions. R7 is a protected research
branch point, not yet a mastered Prime Agent teacher.

`harness-success-sft.toml` defines the next globally unique R9 intervention.
It samples fresh `atomic_child_request` trajectories from protected R7, rejects
every rollout whose executable hard reward is below `1.0`, and applies one
full-weight CE step at `1e-7` to 16 naturally sampled successes. It introduces
no handcrafted policy trace or answer target. The managed two-GPU endpoint
serves both the trainable initialization and the nominally frozen SFT source;
`max_steps=1` with `max_train_batch_lead=0` ensures the complete teacher batch
is collected before the only update can change that endpoint. This arrangement
must not be extended to multiple steps. R9 is promoted only if an exact paired
fresh bank preserves state and send while improving child request; follow-up is
measured as a downstream control. Otherwise R9 is rejected and R7 remains
canonical.

The valid `atomic-child-request-success-sft-r9-retry1` run collected 16
distinct hard successes from 25 R7 attempts. Every admitted trace had the
exact final answer, all required atoms, no forbidden atoms, correct ordering,
and exact cardinality; the batch covered explicit, `natural_a`, and
`natural_b` instruction styles. The full-weight BF16 AdamW step completed with
loss `0.0049`, entropy `0.2941`, gradient norm `2.1406`, and learning rate
`1e-7`. Its stable 12-shard export passed the ChatML EOS validation. An earlier
launch is excluded because the default `0.9` vLLM reservation exhausted memory
during the startup broadcast before any rollout or optimizer step. The template
now reserves `0.8` and caps inference at 16 sequences.

On the exact paired generated bank beginning at index `2600000`, R7 versus R9
scored state `8/8` versus `8/8`, send `3/8` versus `8/8`, child request `5/8`
versus `4/8`, and follow-up `0/8` versus `0/8`, with zero rollout errors. The
send gain was complete across every diagnostic, but the trained target lost one
hard pass: child-request final-answer exactness fell from `1.0` to `0.75`,
required-atom coverage from `0.9792` to `0.8750`, and ordering from `0.8750` to
`0.7500`. R9 therefore fails the predeclared target-improvement rule and is not
replicated or promoted; R7 remains canonical. The result is still causal
evidence that rejection-conditioned CE can consolidate a native communication
action. Because each child-request trace contains both coordinator and child
tokens, the large send gain may reflect an easier child-side signal while the
full coordinator request/resume transition remains underweighted. A follow-up
should test that attribution directly rather than repeat the same CE step.

R10 tested that attribution from the protected R7 source. Prime Agent's
existing session-affinity header provides an opaque, stable identifier for each
agent session. A successful runtime lineage probe recorded one identifier on
all five coordinator model calls and a different identifier on both child
calls; both matched the actual parent/child runtime metadata. Verifiers now
records that client-supplied identifier without forwarding client routing
headers to the model provider. Prime-RL's opt-in
`sampled_session_scope = "root"` SFT mode uses it to retain only the primary
trace-root session and fails closed when call lineage or branch/sample
alignment is missing or ambiguous. Ordinary SFT retains its previous `all`
default.

The valid `atomic-child-request-success-sft-r10-root-session-r1` run collected
16 distinct hard successes from 21 fresh R7 attempts. Every admitted trajectory
had hard reward `1.0`; no handcrafted or golden target was introduced. Exact
re-rendering showed that mixed-role routing would train 56 samples and 12,293
action tokens. Root-session routing instead trained 40 samples and 9,789
coordinator tokens while removing all 16 child-only samples and all 2,504 child
tokens. The one full-weight BF16 AdamW CE step at `1e-7` completed with loss
`0.0077`, entropy `0.3216`, gradient norm `3.2500`, and peak allocated memory
of 32.5 GiB. Its stable 12-shard export passed the ChatML EOS validation.

On the same frozen generated bank beginning at index `2600000`, R7 versus R10
scored state `8/8` versus `8/8`, send `3/8` versus `3/8`, child request `5/8`
versus `4/8`, and follow-up `0/8` versus `0/8`. All 32 R10 episodes completed
without rollout errors. Child-request all-required coverage fell from `0.875`
to `0.750`, ordering from `0.875` to `0.750`, final-answer exactness from
`1.0` to `0.875`, and no-forbidden coverage from `0.750` to `0.625`. The
candidate therefore fails the predeclared target-improvement rule and is not
promoted or uploaded; R7 remains canonical. The eight-episode draw is not
evidence of broad harm, but it is sufficient to reject the candidate under the
frozen rule.

Together, R9 and R10 support a narrower causal conclusion. Mixed-role CE can
strongly consolidate the child-side send action, but that gain disappears when
child-session tokens are removed, while full-response coordinator CE still
does not improve the longer event-control transition. Root-only routing still
trained substantial free-form reasoning and visible-text spans around the
sparse contract actions. A bounded action-local coordinator probe can test
that remaining token-dilution confound; repeating broad successful-trace CE or
the rejected hard-GRPO dose cannot.

The first action-local attempt,
`atomic-child-request-success-sft-r11-action-local-r1`, is invalidated before
behavioral gating. It collected 16 hard-success R7 trajectories from 25 fresh
attempts and mechanically completed one full-weight BF16 AdamW step at `1e-7`
with loss `0.0071`, entropy `0.2522`, gradient norm `3.9375`, and a stable
12-shard export. The authoritative token export contained 32 records and 3,327
active CE tokens, with no RL, reference-KL, or SDPO signal. Decoding every
active span revealed that the selector retained every coordinator tool call
before the child request. Consequently it reinforced failed duplicate spawn
signatures, redundant status cells, polling/listening calls, and early result
construction alongside the intended action. R11 therefore does not test the
predeclared causal intervention and is neither evaluated nor promoted. Its
checkpoint is disposable; R7 remains canonical.

The repaired selector is fail-closed. An eligible native trace must contain a
single minimal coordinator cell that directly assigns the required state and a
successful named `rlm(...)` handle with the inline child-to-parent request
protocol, followed by a sampled no-tool yield response and the final visible
response after the child request. Failed attempts can remain context but get no
CE; intervening tool calls, status computation, polling, reasoning, child
tokens, and completion-gate retries are excluded. Replaying the repaired mask
over the exact R11 batch admits only two clean trajectories and 273 tokens, one
`explicit` and one `natural_b`, which demonstrates that replacement collection
is necessary rather than silently training the contaminated batch. The next
valid probe starts again from R7, collects 16 eligible native trajectories, and
uses the strategist's lower `5e-8` dose before the frozen paired gate.

After training, the checkpoint-battery launcher refuses partial exports and
evaluates the untouched pinned checkpoint plus every stable training step on the
same frozen 24-task VALID-GEN and 24-task OOD-GEN screens. A checkpoint passes
this first screen only when combined hard passes improve and neither split
regresses. That screen nominates a checkpoint for a larger replicated promotion
evaluation; it does not promote a model from one stochastic draw.

The repaired low-dose action-local probe,
`atomic-child-request-success-sft-r12-causal-lowdose-retry1`, is mechanically
valid. Starting again from protected R7, it admitted 16 executable native hard
successes with 16 distinct trace and episode IDs, 15 unique prompts, all three
instruction styles, five child names, and varied task values. Exact replay and
the stable trainer export agreed on 32 root samples and 2,239 active CE tokens.
Every active token belonged to one of three whitelisted native spans: the
successful state/spawn/prompt/handle action, its no-tool passive-yield response,
or the final post-request response. All included their sampled turn-ending
token. The export contained zero child, reasoning, polling, failed-spawn,
status, unrelated-tool, RL, reference-KL, or SDPO tokens.

R12 applied one full-weight BF16 AdamW step at `5e-8`. It completed with loss
`0.0066`, entropy `0.2426`, gradient norm `6.6563`, and peak allocated memory
of 34.5 GiB. Its stable 12-shard checkpoint passed index and ChatML EOS
validation. Thus R12 is the requested causal test of token dilution, unlike
the invalid R11 attempt.

On the exact same frozen generated bank beginning at index `2600000`, R7 versus
R12 scored state `8/8` versus `8/8`, send `3/8` versus `1/8`, child request
`5/8` versus `4/8`, and follow-up `0/8` versus `0/8`. Child-request final-answer
exactness stayed at `1.0`, but all-required and ordering each fell from `0.875`
to `0.500`, required-atom coverage fell from `0.9792` to `0.8333`, and
bootstrap progress fell from `0.6667` to `0.5167`. Send all-required and
ordering each fell from `0.375` to `0.250`. All 32 candidate episodes completed
without evaluation errors. R12 therefore fails both the target-improvement and
prerequisite-retention rules; it is not replicated, promoted, or uploaded. R7
remains canonical.

R12 closes the surgical atomic self-imitation family under its predeclared
stopping rule. Hard GRPO regressed the target, mixed-role successful-trace CE
learned the easier child-side send action, root-only full-response CE did not
improve coordinator control, and the clean action-local root update still
regressed both child request and send. Further LR, role, or mask micro-slicing
would no longer test a specific unresolved confound. The next intervention
must change the curriculum distribution: a broader semantically varied,
natural executable ramp should repeatedly exercise complete event-driven
parent/child policies while R7 and this frozen action battery remain protected
controls.
