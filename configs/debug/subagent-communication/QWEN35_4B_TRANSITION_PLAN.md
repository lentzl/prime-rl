# Qwen3.5 4B Coordinator Transition

## Thinking-mode contract

Thinking mode is required for the coordinator teacher and for every later 9B, 4B,
and 2B distillation rung. Historical runs with `enable_thinking = false` remain frozen
diagnostics but cannot qualify a teacher or student. Teacher collection must retain
authentic model-generated reasoning together with the executable tool/message
trajectory. Do not manufacture reasoning text for synthetic SFT examples, and do not
compare a thinking teacher with a non-thinking student as if model size were the only
changed variable.

The 4B experiment asks a narrow question: does doubling coordinator capacity resolve
the standard-language protocol binding and cross-family interference observed at 2B?
It is not a change to the recursive-specialization architecture, and it does not
discard the retained 2B model.

## What transfers unchanged

Reuse the behavioral curriculum rather than the 2B optimizer history:

- the Prime Agent harness and persistent IPython execution model;
- direct, single-child, parallel, handshake, and follow-up task families;
- standard, guided, and held-out instruction levels;
- fixed train/eval instance separation, message provenance, and causal-order scoring;
- structured-result inspection, real traceback repair, no-repeat, output-contract,
  and clean-stop gates;
- branch-aware SDPO matching, selective OPSD filters, and explicit retention sources;
- the rule that online reward admits an update but never promotes a checkpoint;
- repeated paired evaluation against the frozen predecessor on identical tasks.

This gives us a controlled capacity comparison. A task should not become easier merely
because the model changed.

The historical battery and the development loop have separate jobs. New, disjoint
development tasks may guide checkpoint and curriculum selection. The three standard
12-task gates and bidirectional screen may not. Their untouched-model baselines are
already recorded; V1 now remains sealed until a 4B candidate has been selected without
its feedback. Run it once for the final comparison and do not tune or replace the
candidate afterward. This prevents the 4B search from gradually overfitting the exact
failure boundary that stopped the 2B search.

The final report must distinguish a matched capacity ablation from a best-system
comparison. Only a replay that holds training examples, order, dose, selection rule,
harness, and evaluation fixed can support the claim that capacity class was the intended
independent variable. Comparing the retained best 2B with an independently optimized
best 4B remains operationally important, but it measures the resulting systems rather
than capacity alone.

## What must be recalibrated

Do not copy model-specific training choices blindly:

- Start from untouched `Qwen/Qwen3.5-4B`, not from a reshaped or partially copied 2B
  checkpoint. The architectures have no valid weight-transfer path.
- Run the complete Stage-0 baseline before training. The 4B instruct model may already
  pass rungs that required repair at 2B.
- Re-run teacher admission for every OPSD/SDFT target. A demonstration that failed as
  a 2B teacher target may be reliable at 4B, while a saturated target should be skipped.
- Recalibrate LoRA rank, learning rate, batch size, and number of updates from short
  probes. The 2B doses are evidence about behavior, not safe 4B hyperparameters.
- Keep BF16 training and inference for the first comparison. Do not confound capacity
  with FP8 or quantization.
- Prefer LoRA for baseline repairs and curriculum screening. Escalate to higher rank or
  a short full-weight run only when a clean LoRA plateau is demonstrated.
- Use a separately frozen teacher worker whenever dense policy broadcasts would mutate
  a shared inference worker.

## Execution order

### 1. Untouched 4B baseline

Evaluate `Qwen/Qwen3.5-4B` on:

1. IPython foundations and continuity;
2. structured file/tool result inspection and traceback repair;
3. direct restraint;
4. standard and guided single-child delegation;
5. standard and guided parallel fan-out/fan-in;
6. handshake and complete bidirectional follow-up;
7. output contracts and clean stopping.

Use the existing held-out task IDs, seeds, renderer, temperature, and token budgets so
the comparison to rung 37 remains paired.

These historical tests are frozen in
[FROZEN_CAPACITY_BATTERY_V1.md](FROZEN_CAPACITY_BATTERY_V1.md). Validate their content
hashes and model-only config equivalence before every comparison; never change V1 in
response to a 4B result or use it as an iterative training-development set.

The first two paired gates are ready as:

```bash
uv run inference @ configs/debug/subagent-communication/80-qwen35-4b-baseline-inference.toml
uv run eval @ configs/debug/subagent-communication/81-qwen35-4b-standard-baseline.toml
uv run eval @ configs/debug/subagent-communication/82-qwen35-4b-bidirectional-baseline.toml
```

Run the existing IPython foundation, continuity, recovery, and file-processing eval
configs against the same server before selecting a training rung. These are regression
gates; document extraction itself remains an expert capability rather than a reason to
train more file formats into the coordinator.

### 2. Minimal missing-skill bootstrap

Train only the earliest failed prerequisite. If direct work and single-child
delegation already pass, do not replay their historical SFT ladder as active training;
retain them as evaluation and preservation sources. Prefer natural standard prompts,
using guided prompts only to diagnose whether the primitive exists.

The untouched 4B model scored `4/12` on all three disjoint standard gates:
`4/4` direct, `0/4` single, and `0/4` parallel. Its earliest failed prerequisite is
therefore native single-child communication, not bidirectional messaging. Replay the
original disjoint 96-example bootstrap once, but recalibrate the optimizer on fresh
development tasks rather than assuming the 2B dose transfers:

```bash
uv run sft @ configs/debug/subagent-communication/85-qwen35-4b-single-sft.toml
```

This is a matched curriculum intervention, not test-set adaptation. The initial
`5e-5` run collapsed from loss `0.6317` at step 1 to `0.0031` at step 12 and `0.0001`
at step 24. Both checkpoints then scored `0/4` on the fresh direct prerequisite
screen, so that optimizer schedule was rejected before any frozen test was used.

A controlled `1e-5` replay held every other setting fixed, stopped at 12 steps, and
saved adapter-only checkpoints at steps 4, 8, and 12:

```bash
uv run sft @ configs/debug/subagent-communication/90-qwen35-4b-single-sft-low-rate.toml
uv run inference @ configs/debug/subagent-communication/91-qwen35-4b-low-rate-lora-inference.toml
uv run eval @ configs/debug/subagent-communication/92-qwen35-4b-low-rate-direct-screen.toml
uv run eval @ configs/debug/subagent-communication/93-qwen35-4b-low-rate-dev-eval.toml
```

All three checkpoints passed the direct screen `4/4` and the repeated development
direct family `8/8`. They also produced the same single-child headline result:
answer `2/8`, protocol `0/8`, joint `0/8`. Step 4 was the safest checkpoint, with one
duplicate cell and no rollout errors; step 8 produced six duplicates and one rollout
error, while step 12 produced one duplicate and two rollout errors. Step 4 is retained
as the next-rung initialization, not as a promoted coordinator.

Trace inspection localizes the failure. The parent usually attempts delegation but
often reads or computes the child-assigned shard itself, spawns too late, or replaces
the explicit reply path with `agent_observe` and repeated calls. The 96 demonstrations
are not missing the intended semantics: they contain 32 direct-parent, 32
single-parent, and 32 single-child examples, including an explicit child
`agent_message.send`, a waiting parent turn, and a resumed parent finalization. The
next rung should therefore isolate the admission -> child reply -> parent resume
transition while retaining direct examples, rather than adding more epochs to the
same mixed replay.

Exact hashes and metrics are recorded in
[`qwen35-4b-single-sft-dose-results-v1.json`](qwen35-4b-single-sft-dose-results-v1.json).
No low-rate checkpoint passed development promotion, so Frozen Capacity Battery V1
was deliberately not run.

The isolated child-reply continuation also failed promotion. It added a role-correct
child `agent_message.send` atom and separate parent spawn, local-compute, wait, and
finalize atoms while retaining full direct and single trajectories. On a fresh matched
gate, step 4 increased child-to-parent messages from `2/8` to `5/8`, but both the
original adapter and continuation remained `0/8` protocol-complete. The continuation
also reduced answer credit from `6/8` to `2.33/8` and introduced two rollout errors.
Step 8 failed the direct prerequisite at `3/4`. Step 12 restored direct behavior to
`8/8`, but remained single-child protocol `0/8`, with four max-turn traces and one
error. No checkpoint was selected and the frozen battery remained untouched.

The traces show why the local event did not compose. In a representative solved
episode, the parent read the delegated file itself, then called `rlm('sub-task',
name='shard-worker')`. The child received neither the path nor the reply contract;
`delegated_payloads` stayed zero even when an explicit reply later occurred. The next
admission probe must score the complete first-turn invariant: spawn before reading the
remote shard, include its exact path and reply instruction, retain the returned handle,
and perform only the coordinator-local computation before yielding. Do not add another
isolated child-send dose until this parent-side transition is admitted.

Exact hashes and matched metrics are recorded in
[`qwen35-4b-single-reply-control-results-v1.json`](qwen35-4b-single-reply-control-results-v1.json).

The first on-policy admission probes then separated reward correctness from optimizer
behavior. A mixed-reward update was rejected because inherited task rewards dominated
the new signal. Strict admission-only reward produced eight zeros and therefore no
update. Dense admission-only process reward produced a stable GRPO update and one
complete on-policy trajectory, while inherited rewards remained exactly zero. Its
fresh direct screen stayed `4/4`, but its unchanged admission gate regressed from the
run-100 initialization's `2/8` exact and `1/8` complete admissions to `1/8` and `0/8`.
The checkpoint is rejected and will not be published or used as an initialization.

The bounded dose test started again from run-100 step 8, not from the rejected policy,
and made four stable low-rate updates over distinct new training instances. All four
checkpoints preserved the compact `4/4` direct screen. Step 1 merely matched the
initialization's `2/8` exact and `1/8` complete admissions; step 2 reached `2/8` and
`0/8`; step 4 reached `1/8` and `0/8`. Step 3 was the only checkpoint to exceed the
admission bar, at `3/8` exact and `2/8` complete, and preserved direct behavior `8/8`
on the broader gate. It nevertheless remained `0/8` protocol-complete on normal
single-child tasks, with only `3/8` answer credit, two child replies, 11 duplicate
cells, and five max-turn traces.

No checkpoint is promoted or published. The result is narrower but real: dense process
reward can move the 4B parent's first-turn admission primitive, yet optimizing that
primitive alone does not compose child reply and parent resumption. Training reward
rose monotonically from `0.1250` to `0.3958` while held-out admission peaked at step 3
and then regressed, so online reward cannot select this rung. Frozen Capacity Battery
V1 remains untouched. Exact probe metrics and the predeclared rule are recorded in
[`qwen35-4b-admission-control-results-v1.json`](qwen35-4b-admission-control-results-v1.json).

A stricter follow-up replaced admission-only reward with a five-stage prefix-gated
causal objective: spawn first, bind the retained contract, do coordinator-local work,
receive the child reply, and finish correctly. The run-100 step-8 preflight contained
one trajectory through the fourth stage, so it admitted one conservative `1e-7` GRPO
update. Run 107 stayed numerically stable and preserved direct behavior `4/4`, but the
training group and fresh run-108 gate each contained only one spawn-first trajectory
and no transfer into contract binding or later stages. Run 107 is rejected, no model
is published, and V1 remains sealed.

Do not weaken the causal scorer or continue blind updates from run 107. The next
development intervention is to sample Qwen3.5-9B in the real Prime Agent harness,
retain only trajectories that complete all five stages, and distill those successful
chains into the run-100 step-8 4B learner before retrying causal RL. The exact hashes,
metrics, and decision are recorded in
[`qwen35-4b-causal-chain-results-v1.json`](qwen35-4b-causal-chain-results-v1.json).

That teacher-distillation intervention is now complete. Seven clean 9B trajectories
yielded fourteen balanced coordinator/child branches. Four low-rate LoRA steps were
stable, and both saved checkpoints preserved the paired direct prerequisite at `4/4`.
Neither checkpoint crossed the causal boundary: the initialization, step 2, and step 4
all remained `0/8` complete chains. Step 4 did move the earliest behavior from `0/8`
to `2/8` spawn-first traces and improved answer credit from `6/8` to `7/8`, but contract
binding and every later ordered stage remained `0/8`.

Both distilled checkpoints are rejected and the run-100 step-8 adapter remains the
accepted 4B checkpoint. The result rules out a tiny seven-trace distillation dose as a
sufficient repair, not teacher distillation itself. A next attempt must add successful
trajectory diversity and balance the complete sequence of coordinator and child
decisions; it must not tune against Frozen Capacity Battery V1, which remains sealed.
Exact hashes and paired metrics are recorded in
[`qwen35-4b-admitted-teacher-distillation-results-v1.json`](qwen35-4b-admitted-teacher-distillation-results-v1.json).

A subsequent clean-causal curriculum made the first genuine full-chain observation.
Its broad 48-step replay preserved direct behavior, and step 36 completed `2/8` clean
causal trajectories on the selection set with no polling, observations, failed cells,
or duplicate cells in those successful episodes. The same checkpoint scored `0/8` on
a disjoint confirmation, so it remained an experimental initialization rather than a
promotion.

A narrower 16-step causal-mastery continuation then preserved direct behavior `4/4`
at every checkpoint. Step 12 produced one full causal completion on the original
disjoint confirmation set, but it was not clean: the eight episodes accumulated 14
roster polls, six observation calls, 11 failed cells, and seven duplicate cells. Step
16 erased the ordered chain. A fresh unseen confirmation of step 12 returned `0/8`
clean and `0/8` complete, with 19 roster polls and 15 duplicate cells.

This is evidence of capacity, not reliability. The 4B learner can express the complete
spawn -> contract -> local work -> explicit child reply -> finalization sequence, but
the behavior is seed-fragile and still dominated by polling and repair failures. No
focused checkpoint is promoted or published, run-100 step 8 remains the accepted 4B
checkpoint, and Frozen Capacity Battery V1 remains sealed. Exact hashes and metrics
are recorded in
[`qwen35-4b-causal-mastery-results-v1.json`](qwen35-4b-causal-mastery-results-v1.json).

The next intervention moved the teacher-data path from synthetic demonstrations to
real harness trajectories. Under the initial metric-level clean scorer, natural 9B sampling completed four
of eight causal chains but no clean chains. An explicit message-resumption prompt then
produced clean child-reply trajectories whose only remaining defect was wrapping the
correct JSON object in prose. The exporter now admits those traces only when the real
trajectory has zero polling, observations, failures, and duplicates and the embedded
JSON exactly matches the verifier answer; it replaces only the final coordinator text
with bare JSON. A 32-rollout collection yielded five untouched strict traces and nine
additional canonicalizable traces, which combined with three preflight traces into 17
coordinator/child trajectory pairs across ten tasks.

One conservative epoch over those 34 real branches started from run-100 step 8. Every
checkpoint preserved direct behavior `4/4`. Step 2 produced one fully clean causal
completion on the disjoint offset-2300 gate; steps 4, 6, and 8 produced none. The step-2
event did not repeat on the fresh offset-2500 confirmation (`0/8`). No checkpoint is
promoted or published, and Frozen Capacity Battery V1 remains sealed. This improves
the evidence from synthetic capability glimpses to one clean transfer from corrected
real trajectories, but it still does not establish reliable 4B coordination. Exact
hashes and metrics are recorded in
[`qwen35-4b-clean-message-distillation-results-v1.json`](qwen35-4b-clean-message-distillation-results-v1.json).

A later branch-level audit found that this scorer was not sufficient for teacher-data
admission. It counted failed IPython cells but did not inspect sampled non-IPython tool
calls. Of the 34 branches used above, 20 contained calls to undeclared tools such as
`agent_message`, `agent_observe`, or `rlm.list_subagents`, and 19 contained an explicit
`Tool ... not found` result. Only three source traces, producing six branches, survived
an executable-branch re-export. A child could also keep making tool calls after a
successful `agent_message.send` without violating the old aggregate metrics.

The exporter now requires every sampled call to name a tool declared by the trace,
rejects tool failure results, and requires child branches to stop tool use after their
first parent-message send. An eight-step 9B LoRA run over the contaminated dataset was
therefore invalidated despite stable optimization and clean direct-task retention; its
remaining checkpoint screens were stopped, and no checkpoint was selected or used for
4B compression. Exact audit counts and partial metrics are recorded in
[`qwen35-9b-clean-message-mastery-results-v1.json`](qwen35-9b-clean-message-mastery-results-v1.json).

A focused 9B OPSD teacher intervention then tested whether the missing coordinator
ordering could be induced without another broad SFT replay. The first attempt was a
silent no-op: `subagent-admission-v1` appended its self-contained delegation contract
after the base taskset had keyed the OPSD demonstration maps, so the final parent
question fell through to the null wildcard. The admission adapter now re-keys every
applicable demonstration map when it changes the prompt, and the Prime RL train sink
no longer ships rollouts with no nonzero loss component. The focused environment tests
and Prime RL unit tests pass with these guards.

After that repair, one BF16 rank-16 OPSD step over four on-policy 9B trajectories was a
real update: loss `0.0164`, gradient norm `0.3066`, mismatch KL `0.0003`, and all 128
LoRA-B tensors became nonzero. A same-task disjoint comparison nevertheless found no
behavioral improvement. Both the adapter and untouched 9B produced exactly one of eight
correct spawn-and-contract prefixes, zero local-work continuations, and zero completed
causal chains. The adapter SHA-256 is
`73b0a56ff1b75d037dda11a6bbfbe6f0071e916b321f09e135b610a72adfa86a`;
adapter and base screen trace SHA-256 values are
`ab796144dc327d184c658f932394ffc2d9d46d8235aa8c0cdabcb679345b5c57` and
`34da37022005a7a2c488d32aca7559dbfaf2a71ac462e24b5f5a8e747f95ac3a`.

Do not transfer or promote this 9B adapter. It validates the repaired one-GPU OPSD
path, not a useful teacher. Run-100 step 8 remains the accepted 4B development
checkpoint, and Frozen Capacity Battery V1 remains sealed.

A final contract-only control tested whether the stale run-100 delegation wording was
the remaining blocker. The corrected dataset used fresh standard tasks and made each
single-child question self-contained: both the user question and the demonstrated
spawn supplied the exact weighted-checksum formula and the exact string-valued
`agent_message.send` call. It contained 32 direct-parent restraint examples and 32
spawn -> local-work -> yield coordinator prefixes. Eight BF16 rank-16 LoRA steps at
`1e-6` started from run-100 step 8 and saved steps 2, 4, 6, and 8.

The paired disjoint screen rejects every checkpoint. Run-100 step 8 and corrected step
4 each reached only the first causal stage in `1/8` traces; corrected steps 2, 6, and 8
reached it in `0/8`. No model bound the full admission contract, reached ordered local
work, or completed a chain. The baseline retained answer accuracy in `8/8`, while the
four continuations scored `6/8`, `7/8`, `7/8`, and `6/8`. They also accumulated 43-66
duplicate cells and up to two max-turn episodes per screen.

Trace inspection explains why more contract-only epochs are not justified. Parents
still compute local work before spawning, wrap `rlm` in a nested event loop, emit empty
IPython cells, or paraphrase away the executable reply API. Child branches can compute
the right checksum but call an undeclared `agent_message` tool or pass an integer to
`agent_message.send`, then fail to make a constrained correction from the resulting
traceback. The next rung therefore starts again from run-100 step 8 and balances full
coordinator and child trajectories with isolated executable recovery states. It is not
a continuation of any rejected step-199 adapter. Exact hashes and metrics are recorded
in [`qwen35-4b-corrected-contract-results-v1.json`](qwen35-4b-corrected-contract-results-v1.json).

The next executable-causal replay then tested the complete control boundary rather
than contract text alone. It started again from run-100 step 8 and trained one full
epoch over 160 fresh examples: direct restraint; full coordinator and child branches;
spawn/local/yield prefixes; isolated child send, parent wait, and fan-in states; and
three traceback-conditioned repairs for integer message arguments, undeclared direct
tool calls, and nested event-loop spawning. BF16 rank-16 LoRA used `5e-7` for 40 stable
steps with no NaN losses, saving steps 8, 16, 24, 32, and 40.

The paired selection screen showed a meaningful but non-repeatable progression. The
accepted baseline reached only spawn-first in `3/8`. Step 16 reached contract binding
in `1/8`. Step 24 produced one clean child reply after a fully aligned admission and
ordered local work, reaching the fourth causal stage in `1/8`; the final answer was
correct, but eight duplicate inert cells kept the protocol from completing. Steps 32
and 40 regressed to `0/8` at every causal stage. Step 24 preserved direct restraint and
accuracy at `8/8`, exactly matching the baseline on a paired standard direct screen.

The required fresh causal confirmation rejected step 24. On eight new episodes, it
reached spawn-first only once and no later stage, for cumulative causal reward `0.2`.
The matched run-100 baseline independently produced another fourth-stage trajectory
and cumulative reward `0.8`. This proves that the 4B class can express spawn -> contract
-> local work -> explicit child reply, but the event remains inside baseline variance
and is not evidence that the continuation improved reliability. Step 24 is retained
only as an experimental trace source; it is not promoted, published, or eligible for
Frozen Capacity Battery V1. Run-100 step 8 remains the accepted 4B checkpoint.

The next intervention should target event-loop process control rather than add another
broad epoch: when no child message has arrived, repair empty/comment-only IPython calls
into a plain waiting response; after a child message, return the grounded JSON without
another tool call; and preserve the exact top-level `await rlm(...)` plus `local`
binding. Selection must require repeated disjoint complete chains with zero duplicate
cells, not another isolated fourth-stage trace. Exact hashes and metrics are recorded
in [`qwen35-4b-executable-causal-results-v1.json`](qwen35-4b-executable-causal-results-v1.json).

The isolated resumption-control continuation did not solve that boundary. Twelve
stable BF16 LoRA steps trained four explicit continuation states: replace an inert
parent wait cell with plain assistant text, finalize directly on an incoming child
message, repair an inert child wait into one `agent_message.send`, and stop after a
successful delivery receipt. Steps 4, 8, and 12 all failed promotion. A repeat screen
also found zero complete chains for both run-100 step 8 and executable step 24, although
step 24 improved answer accuracy from `5/8` to `7/8`.

A subsequent trace-graph audit invalidated the one apparent clean completion from the
first screen. The legacy clean scorer inspected only calls parsed inside IPython cells;
it did not reject sampled calls to undeclared tools such as `agent_observe`, inert
comment/pass/string cells, child tool calls after a successful parent send, coordinator
cells after fan-in, or a later coordinator read of the delegated shard. The verifier
now enforces all of those constraints and exposes a six-stage clean-causal prefix:
spawn first, bind the complete contract, perform local work, preserve tool discipline,
receive an explicit child reply, and finish directly with the correct answer. Under the
corrected scorer every run-206 and run-207 model has `0/8` tool-disciplined child replies
and `0/8` clean completions. Run-100 step 8 therefore remains the accepted checkpoint;
step 24 and every run-205 continuation remain rejected, and Frozen Capacity Battery V1
remains sealed. Exact hashes and both legacy and corrected metrics are recorded in
[`qwen35-4b-resumption-control-results-v1.json`](qwen35-4b-resumption-control-results-v1.json).

The final 4B intervention applied first-coordinator-response OPSD directly to the
corrected six-stage clean-causal objective. Four BF16 rank-16 updates were numerically
healthy, but a fresh paired eight-rollout screen found zero clean-causal progress for
the accepted run-100 step 8 checkpoint and for every OPSD checkpoint. All five models
scored `0/8` for spawn-first, contract binding, local work, explicit child reply, and
clean completion under the strict scorer. Answer accuracy was `7/8` for the accepted
checkpoint, `8/8` at OPSD step 1, `7/8` at step 2, and `6/8` at steps 3 and 4. The
intervention therefore changed the policy without inducing the target control sequence
and began to erode task accuracy at the larger doses. No OPSD checkpoint is promoted
or published. This closes further 4B coordinator optimization on the present
curriculum; the next capacity experiment qualifies an untouched 27B model as a
trajectory teacher before distilling into smaller Qwen classes.

That 27B capacity experiment separated expressibility from transfer. Untouched
`Qwen/Qwen3.5-27B` completed `15/16` strict guided clean-causal rollouts, but its fresh
standard screen solved only `3/6` answers and produced no clean delegated trace. A
24-step BF16 rank-16 SFT run over 224 mixed harness examples was numerically healthy.
Step 4 improved fresh standard answers from `3/6` to `5/6` and protocol alignment from
`4/6` to `5/6`, including both parallel answers, but delegated clean execution remained
`0/4`. Step 12 reached `4/4` on the guided clean-causal screen while regressing standard
answers to `3/6`; step 24 regressed guided completion to `1/4`.

The required fresh bidirectional comparison rejected step 4 as a teacher. Both the
untouched model and adapter produced `0/8` causal follow-up exchanges and `0/8`
protocol-aligned traces. Step 4 also reduced fractional answer credit from `4.6667/8`
to `1.3333/8`. The demonstrations contain the complete parent and child message
trajectory, so this is not a missing-target defect. The broad low-dose SFT taught
guided imitation non-monotonically but did not transfer resumption and message
consumption into standard prompts. No 27B adapter is promoted or published. The next
teacher intervention should isolate real standard-prompt transition states: child
request consumption, one direct parent follow-up from retained state, child result
consumption, and immediate grounded finalization. Exact settings, hashes, and metrics
are recorded in
[`qwen35-27b-harness-mastery-results-v1.json`](qwen35-27b-harness-mastery-results-v1.json).

A focused continuation from that step-4 adapter then made both bidirectional families
exportable and trained 12 more low-rate steps over 176 examples, 128 of which were
complete follow-up or handshake parent/child trajectories. The original compact gate
found zero literal requests for the source and all three continuation checkpoints, but
that criterion was later identified as an artificial wire-phrase proxy: the natural
task asks the child to request a missing concept, not to emit fixed text. Rescoring
provenance and order while accepting explicit natural requests gives untouched
thinking 27B `2/4` causal exchanges, both in the handshake family, and the historical
step-12 adapter the same `2/4`. Both remain `0/4` clean because of polling, failed
cells, repeated actions, answer errors, and post-result tool use. Exact settings and
hashes are recorded in
[`qwen35-27b-bidirectional-results-v1.json`](qwen35-27b-bidirectional-results-v1.json).

### 3. Bidirectional admission (superseded proxy)

The earlier plan required a fixed child request phrase. Do not use that criterion for
future teacher admission. Require an explicit request naming the missing concept, a
direct parent follow-up after that request, and a distinct child result afterward,
all linked by real message provenance. Keep literal-phrase compliance only as a
historical diagnostic. Train the remaining natural transition and bounded process
control rather than a magic string.

### 4. Paired promotion gate

A 4B checkpoint becomes the coordinator candidate only when repeated disjoint gates
show all of the following:

- direct `4/4` with no unnecessary delegation;
- single-child `4/4` answer and protocol alignment;
- parallel `4/4` answer and protocol alignment;
- handshake and follow-up causal chains aligned, including both message directions;
- no regression in IPython continuity, traceback repair, output contracts, or stopping;
- lower variance across seeds than the retained 2B model.

Report active parameters, latency, generated tokens, and GPU time beside behavioral
quality. The 4B root is justified only if the reliability gain exceeds its deployment
cost.

## Model-role decision

The expected heterogeneous system remains:

```text
4B coordinator candidate
  -> 2B rung-37-derived expert child
  -> additional specialists only when measured utility justifies them
```

This is a hypothesis, not a forced outcome. If 4B does not improve the complete
protocol after the controlled curriculum, retain the 2B system and test a different
architecture or stronger coordinator. If 4B succeeds, freeze it separately while
preserving the 2B HF snapshot as the expert substrate and comparison baseline.
