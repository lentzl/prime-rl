# Recursive Specialization Research Target

This document defines the long-term target for training in this project and the
sequence of experiments required to reach it. It is a research thesis, not an
assumption that a hierarchy will work. Every structural addition must beat a simpler
alternative under a comparable budget.

## North Star

Build a sparsely activated population of small RLM-capable models that can deepen its
competence through recursive specialization while preserving a general reasoning and
tool-use substrate.

The system should learn to place durable knowledge and repeatable procedures in its
environment, skills, and specialist descendants. Model weights should concentrate on
reasoning, exploration, decomposition, routing, verification, synthesis, and learning
how to use those external capabilities. A deployed unit is therefore always a
versioned combination of model weights, harness, tools, skills, and interfaces.

The central transition to demonstrate is:

```text
domain generalist
    -> creates or acquires useful specialist children
    -> learns to coordinate and supervise those children
    -> can shed redundant solver capacity
    -> becomes a smaller or more efficient domain coordinator
```

The immediate bootstrap target is deliberately narrower than a general-purpose
assistant. First train the smallest viable Qwen policy to master Prime Agent itself:
its persistent IPython control plane, RLM delegation, asynchronous messaging,
recovery, routing, and synthesis. That policy becomes the first orchestrator. Do not
spend its limited capacity learning document parsing in depth. Document handling is
the first expert-child role, giving us the earliest concrete test of whether a small
coordinator plus a specialist can grow more effectively than one increasingly dense
generalist.

A successful hierarchy may then repeat the transition recursively:

```text
general coordinator
    -> math coordinator
        -> probability specialist
            -> stochastic-process specialist
        -> combinatorics specialist
    -> coding coordinator
        -> Python specialist
        -> systems specialist
```

The taxonomy above is illustrative. Specialization boundaries must emerge from
coherent, recurring failures and measured utility, not from a taxonomy chosen in
advance.

## Research Thesis

The project tests five linked claims:

1. A small instruct model can acquire a robust, portable RLM substrate: persistent
   computation, environment use, feedback-driven repair, skill use, external memory,
   and bounded self-delegation.
2. Narrow specialization can produce more depth per trained and activated parameter
   than continuing to train a single general model.
3. A parent can learn a different role because a child exists: identify the child's
   domain, formulate useful subproblems, preserve shared state, judge the result, and
   synthesize it into a final answer.
4. Once children carry enough depth, the parent can be compressed toward coordination
   without reducing subtree performance.
5. The same differentiate, coordinate, consolidate, and prune cycle can repeat below
   a specialist, producing useful recursive depth rather than a flat expert zoo.

The interesting research object is not a static mixture of agents. It is the process
by which capacity differentiates into specialists, learns interfaces between roles,
reconsolidates useful abstractions, and removes capacity that no longer earns its
cost.

## System Invariants

These constraints should remain true across curriculum rungs and architectures:

- Every node retains generic RLM behavior. A specialist is not a one-shot classifier
  or a memorized answer bank.
- Parents retain breadth, decomposition, routing, interface management, synthesis,
  judgment, uncertainty, and the ability to challenge child outputs.
- Children carry depth. Upward learning must not simply duplicate all child knowledge
  in the parent.
- Only relevant branches should be activated for a task. Quality, latency, token use,
  GPU cost, and coordination cost are all part of system performance.
- Real environment outcomes and natural textual feedback are preferred whenever they
  are available. Synthetic demonstrations are useful for bootstrapping structure, but
  must not become a permanent substitute for executable feedback.
- Training is cumulative only when earlier held-out gates still pass. A new capability
  does not excuse regressions in notebook semantics, repair, grounding, or protocol
  control.
- Thinking mode is part of the policy contract, not an evaluation-time option. Train,
  collect, distill, and evaluate the coordinator and every smaller Qwen student with
  thinking enabled. Preserve authentic model-generated reasoning in admitted teacher
  trajectories; do not fabricate reasoning traces for synthetic SFT targets or
  promote a checkpoint only because it succeeds with thinking disabled.
- Structural changes are reversible and versioned. A child can be rejected, merged,
  replaced, absorbed, or deleted.
- A frontier model may design curricula and audit the system, especially early on,
  but it must not hide whether the trained population itself has learned the target
  behavior.

## Learning Operators

The architecture is broader than any one optimization method. We should use each
existing PrimeRL primitive only for the signal it actually supplies. The current
implementation contracts are documented in the [PrimeRL algorithm
abstraction](algorithms.md).

### GRPO and executable feedback

Use GRPO when the learner must act, observe a real outcome, and improve behavior under
an objective verifier. This remains the default for tool use, recovery, routing, and
whole-system task success. Hierarchical GRPO is relevant later when proposers and
solvers have distinct role-aware credit, not as the initial system architecture.

### SDPO and hindsight

[SDPO](https://arxiv.org/abs/2601.20802) uses the current policy conditioned on rich
feedback as a self-teacher and distills those feedback-informed predictions into the
unconditioned policy. In a parent-child interaction, a paper-faithful use is:

```text
parent attempt
    -> child or environment returns informative feedback
    -> the same parent, conditioned on that feedback, predicts a better response
    -> SDPO distills the hindsight improvement into the parent
```

Directly treating child logits as the parent's SDPO target is not SDPO. We should use
SDPO to improve recovery, routing, decomposition, evaluation, and synthesis only when
the teacher remains the feedback-conditioned version of the same policy.

SDPO is also not assumed to prevent forgetting. Recent continual post-training
[evidence](https://arxiv.org/abs/2607.01763) reports that dense self-distillation can
accelerate in-domain specialization while increasing drift or collapse when teacher
signals are unstable. Every SDPO rung therefore keeps independent capability-retention
gates, and the current multi-turn replay extension remains experimental until tested
separately from the paper-faithful complete-response path.

### OPD, MOPD, and capability integration

[MOPD](https://arxiv.org/abs/2606.30406) trains independent domain teachers and then
distills their token distributions on trajectories generated by a common student.
This is a strong candidate for integrating same-role capabilities without the
exposure bias of imitating only teacher-generated trajectories.

Raw MOPD is not automatically the correct parent-child operator. A mature parent and
child solve different roles: the child supplies depth while the parent coordinates.
Use OPD/MOPD directly when teacher and student are meant to perform the same behavior.
For parent coordination, distill interface knowledge selectively or train against
whole-system outcomes and specialist critiques. The optimization target is to make
the parent better because specialists exist, not to make it independent of them.

[CaMOPD](https://arxiv.org/abs/2605.27115) is relevant if specialization damages
general competence. Its alternating recovery and preservation updates, together with
teacher-student gap selection, provide a concrete hypothesis for reducing conflicting
gradients. We should adopt it only after measuring such a conflict in our setting.

## Step-by-Step Program

Each stage has an explicit gate. Failure at a gate changes or contracts the design;
it does not justify automatically proceeding to a more complex stage.

### Stage 0: Master the harness, then map its compressibility

First teach Qwen3.5-27B Instruct the clean Prime Agent policy while capacity is
unlikely to be the limiting factor. Freeze the smallest 27B checkpoint that masters
the harness naturally, then distill that same canonical teacher directly into 9B,
4B, and 2B thinking-mode students. The root is a control policy, not the first domain
expert. Persistent IPython is its working memory and coordination substrate; RLM
children and explicit messages are its expandable cognition.

The 27B checkpoint is initially a policy-discovery instrument and research oracle,
not an assumption about the eventual production coordinator. This reverses the order
of discovery without discarding the bottom-up result: first remove likely capacity as
a confound and learn the clean natural-language policy; then ask how much of exactly
that policy each smaller capacity class can absorb. The independently trained rung-37
2B remains the critical bottom-up control for separating an optimization boundary
from a representational one.

Required behavior includes:

- accepting silent assignments and reusing variables across turns;
- inspecting structured tool results and real tracebacks before the next action;
- preserving successful state while repairing only the failing operation;
- assigning and retaining RLM child handles in persistent state;
- distinguishing admission handles, messages, observations, and final results;
- spawning one or several named children without duplicating delegated work;
- yielding after asynchronous sends instead of polling or inventing APIs;
- completing causal child request, parent follow-up, and child-result exchanges;
- routing, synthesizing, checking, and returning the requested output contract;
- stopping cleanly without protocol leakage or uncontrolled retry loops.

The teacher admission suite must cover direct IPython work, one-child delegation, parallel
fan-out/fan-in, bidirectional follow-up, malformed child output, delayed replies, and
recoverable tool or messaging failures. Every delegated trajectory is scored by
message provenance and order, not merely by a correct final value.

Admission must also distinguish useful externalization from unnecessary delegation.
Use paired tasks whose surface form and answer difficulty are similar but whose hidden
state, scale, persistence, or parallelism makes external computation or delegation
valuable in only one member. A 27B model that earns task reward by solving everything
internally has not mastered the policy we want to distill.

Gate: freeze a 27B teacher only when it repeatedly passes all harness families on
unseen seeds under realistic Prime Agent autonomous continuation and chooses sensible
externalization boundaries. Then distill directly from that frozen teacher into each
student size. Do not default to a `27B -> 9B -> 4B -> 2B` cascade, because teacher
degradation would confound the capacity comparison. Progressive distillation remains
a later ablation if the direct teacher-student gap is itself a measured blocker.

Use the same frozen behavioral battery, canonical teacher, tokenizer contract,
thinking-mode contract, and promotion criteria for 9B, 4B, and 2B. Prefer on-policy
OPD for the main transfer so each student receives dense teacher guidance on states it
actually visits; use admitted executable teacher trajectories for bootstrap SFT and
diagnostics rather than treating pristine-trace imitation as sufficient. The retained
rung-37 2B is the bottom-up control. Comparing it with a directly distilled 2B
separates a likely capacity boundary from a curriculum or optimization failure.

Current status: Qwen3.5-2B has mastered useful IPython foundations. A Prime-native
OPSD repair produced a new Stage-0 candidate at
`37-single-path-opsd-dose-r1/weights/step_2`: it preserved task-specific paths in
`4/5` held-out initial actions, solved full single-child and direct gates `3/3`, and
returned exact answers on all three parallel gates. Parallel message provenance was
fully aligned in only `2/3`, and the previously isolated bidirectional follow-up chain
is still not reliable. The model is therefore materially closer but not admitted as
the first orchestrator yet.

On a broader 12-task standard-prompt gate, its authoritative trace metrics were `4/4`
direct, `2/4` single-child, and `3/4` parallel joint answer-and-protocol solves:
`9/12` overall. Subsequent OPSD experiments isolated the causal surface progressively:
role-specific branch demonstrations, null child branches, immediate post-child
responses, and finally only the first response after complete fan-in. Null branch
mappings now include a wildcard so paraphrased child questions are safely excluded,
and Prime-RL supports generic per-token OPSD filters without importing environment
logic.

The narrowest four-update dose improved parallel performance on a fresh paired gate
from `1/4` to `2/4`, but single-child performance fell from `3/4` to `2/4`; both rung
37 and the new candidate scored `8/12`, while mean reward fell from `1.7292` to
`1.6250`. Earlier immediate-post-message distillation showed the same capability
trade on a disjoint replication (`4/4` single and `1/4` parallel versus rung 37's
`2/4` and `3/4`). Selective SDFT can therefore move the 2B policy, but this task-level
demonstration has not delivered robust fan-in without adjacent interference. No later
checkpoint supersedes rung 37.

An opt-in, branch-aware process-control reward then targeted the observed behavior
directly. It activates only after every expected child message is visible and scores
the absence of coordinator-side failures, repeated cells, polling, new delegation,
and unnecessary messaging. Two four-sample probes showed strong within-task reward
variance, but a one-step parallel-only GRPO update still failed retention. On a fresh
paired gate, rung 37 scored `4/4` direct, `2/4` single, and `3/4` parallel (`9/12`),
while the updated checkpoint scored `4/4`, `2/4`, and `1/4` (`7/12`). Clean
post-fan-in control also fell in both delegated families. The metric is diagnostically
useful, but its first training distribution was already 75% saturated and did not
isolate the rare failures. Process-control training must include actual failure
regimes and explicit cross-family retention rather than another parallel-only batch.

The selected adapter has also been merged into a standalone dense candidate. All 96
updated matrices and 521 unchanged tensors passed exact export checks, and
adapter/dense greedy spawn behavior matched. A sampled dense smoke still omitted the
path and looped during repair, confirming that the remaining reliability problem
belongs to the policy rather than the export pipeline.

A later cross-family probe made the interference explicit. One hard-example GRPO
update moved the paired gate from rung 37's `4/4` direct, `4/4` single, and `2/4`
parallel split to `4/4`, `2/4`, and `4/4`: the total remained `10/12`, while the
capability moved between delegated families. Frozen-policy OPD restored the original
split, and two joint GRPO-plus-OPD weightings reached only `8/12` and `9/12`.

The full-weight path is now exercised from an exactly validated dense export of rung
37. Branch-matched SDPO can replay successful coordinator and child branches against
their corresponding failed branches in real Prime Agent traces. Its first isolated
dose scored `9/12`. A mixed update then combined eight hard-parallel SDPO trajectories
with eight fresh coordinator-only OPSD/SDFT preservation trajectories at one-quarter
loss weight. It trained stably and scored `4/4` direct, `3/4` single, and `3/4`
parallel (`10/12`). That balanced the two delegated families but still exchanged one
single success for one parallel success relative to rung 37. Neither full-weight
candidate is promoted.

The final bidirectional isolation established the 2B capacity boundary more directly.
The child produced the correct parent-message tool call in `24/24` guided post-compute
probes, but in `0/24` canonical standard probes did it translate "send to your parent"
into a tool call. The frozen 2B teacher also failed teacher admission: even a
response-aligned request-only demonstration yielded the required tool call only
`1/24` times. Four guided full-weight doses and a response-phase repair produced no
aligned held-out bidirectional exchange. These results preserve the exact dense
rung-37 2B model as a future expert-child base and frozen bottom-up baseline.
Subsequent capacity probes motivate the current 27B teacher-first program rather than
further independent bottom-up rediscovery at every model size.

### Stage 1: Train the first expert child for documents

After the Stage 0 teacher and student capacity map are frozen, train document handling
as a separate expert role. This child owns the depth that should not burden the small
orchestrator:

- selecting parsers from extension, MIME type, magic bytes, and observed structure;
- extracting grounded content from text, Markdown, CSV, JSON, PDF, and DOCX;
- repairing missing parsers, malformed inputs, encoding failures, scanned PDFs, and
  password protection without repeating unchanged calls;
- preserving source locations, page references, uncertainty, and negation;
- returning a compact, typed result that the orchestrator can inspect and synthesize.

The expert may start from a harness-capable student checkpoint, including the retained
bottom-up 2B, but its specialization gate is independent of the root's gate. It can be
larger than the coordinator if document depth requires it; sparse activation, not
uniform node size, is the efficiency target.

Gate: retain the document expert only when it robustly beats the Stage 0 coordinator
on held-out document tasks and exposes a stable interface rather than prose that the
root must reinterpret heuristically.

### Stage 2: Learn the orchestrator-document interface

Pair the frozen Stage 0 orchestrator with the validated document expert and collect
real Prime Agent trajectories. Train the root to recognize document work, formulate a
self-contained request, retain the child handle, pass only necessary state, wait for
the explicit result, inspect its structure, request correction when evidence is
missing, and synthesize a final answer grounded in the child's output.

Train the expert to accept that contract, report recoverable failures explicitly, and
return evidence rather than silently guessing. Optimize whole-system success while
keeping role-local diagnostics for routing, extraction, grounding, communication,
and synthesis.

Gate: root plus document expert must beat the root alone, expert alone, fixed routing,
and an equivalently costed dense baseline after latency, generated tokens, and active
parameters are counted. The root must retain every Stage 0 harness gate.

### Stage 3: Grow from measured failure clusters

Use failures of the orchestrator-document pair to decide what grows next. A coherent
gap may justify deeper document specialization, a verifier child, a retrieval expert,
or another domain expert. The next node is earned by recurring evidence rather than a
predeclared taxonomy.

For every candidate, compare continued training of an existing node, a larger dense
model, and the expanded sparse system under matched or explicitly reported cost.

Gate: keep a new child only when its marginal whole-system contribution exceeds its
training, inference, coordination, and maintenance cost.

### Stage 4: Generalize orchestration across experts

Once at least two specialists earn their place, train expert selection, parallel
delegation, cross-checking, conflict resolution, and budget-aware stopping. Use
whole-system reward for end-to-end success. Use feedback-conditioned SDPO for
same-policy hindsight where child or environment feedback improves a later decision;
use OPD/MOPD only for same-role transfer or carefully selected interface knowledge.

Gate: learned orchestration must beat fixed routing, always-delegate, and
never-delegate baselines while retaining the document interface and all Stage 0
harness behavior.

### Stage 5: Transition and compress the parent

Once the child is dependable, test whether the parent's optimum has changed from
solver toward coordinator. Train a smaller parent or compress the existing one using
successful parent-child interactions, failure cases, and generic RLM replay.

Prefer role compression before arbitrary layer pruning: explicitly train the smaller
model for decomposition, routing, supervision, and synthesis rather than expecting a
compressed solver to become a coordinator automatically.

Gate: replace the parent only if the whole subtree preserves or improves quality and
robustness while moving the measured cost frontier. The compressed parent must still
recognize child errors and handle out-of-domain tasks safely.

### Stage 6: Demonstrate recursive specialization

Apply the same failure-clustering and marginal-utility rule to the child. Let it propose
or receive one candidate grandchild only after its own recurring gap is established.

Compare:

```text
A. Continue training the child
B. Enlarge the child
C. Child plus grandchild under the existing parent
```

Gate: the grandchild survives only if the complete root-to-leaf subtree beats those
alternatives. This experiment, not the existence of a flat expert bank, is the first
direct test of recursive depth through specialization.

### Stage 7: Add controlled continual evolution

Only after a useful population exists, introduce proposer-solver dynamics as a
curriculum mechanism. The proposer should generate situations around observed
weaknesses, including routing, cooperation, repair, memory, and boundary recognition,
not merely more domain questions.

Use population disagreement, such as `4p(1-p)`, as one learnability signal: tasks that
everyone solves or nobody can engage with provide less immediate training value than
tasks near the current competence frontier. Disagreement is not sufficient by itself;
tasks must also be relevant to real failures and resistant to reward hacking.

Classify accumulated failures before training:

```text
training gap | specialist gap | coordinator gap | harness gap | invalid task
```

Operate on three timescales:

- Fast: agents solve tasks and adapt through RLM state.
- Medium: proposer-solver curricula improve existing agents and interfaces.
- Slow: supervised architectural changes create, merge, compress, or retire nodes.

Gate: the self-generated curriculum must improve held-out real-task performance more
efficiently than frontier-teacher-only curriculum generation. Slow structural changes
remain externally audited until that loop itself has earned trust.

### Stage 8: Consolidate and prune continuously

Periodically run subtree ablations and overlap tests. Distill useful abstractions
upward, merge redundant children, remove experts with no marginal value, and retrain
interfaces after every structural change.

The intended rhythm is:

```text
differentiate -> learn coordination -> consolidate -> prune -> differentiate again
```

This is successful only if system utility improves over time without unbounded model,
latency, or maintenance growth.

## Evaluation Contract

Every experiment reports model-level and system-level results. At minimum, track:

- final task success and verifier reward;
- process reliability, repeated actions, recovery success, and groundedness;
- generic RLM capability retention;
- routing precision, recall, calibration, and abstention;
- child-call utility and unnecessary delegation rate;
- decomposition and synthesis success;
- parent detection of incorrect or incomplete child outputs;
- active parameters, generated tokens, wall-clock latency, peak memory, and GPU time;
- training and maintenance cost for each retained node;
- performance after removing, replacing, or bypassing each child;
- overlap between specialists and change in subtree performance after consolidation.

A useful working objective is:

```text
system utility = task quality
               - lambda_latency * latency
               - lambda_tokens * generated tokens
               - lambda_compute * GPU cost
               - lambda_structure * coordination and maintenance cost
```

Report the raw terms as well as any scalarization. The coefficients express deployment
preferences and must not be tuned after seeing which architecture wins.

## Stop Conditions

The recursive hierarchy should be reduced or abandoned if repeated controlled tests
show any of the following:

- a matched dense model consistently dominates the tree;
- routing and synthesis overhead erase specialist gains;
- small parents cannot supervise children reliably;
- specialists lose the common RLM substrate despite replay and preservation training;
- specialization boundaries are unstable across task samples;
- recursive children do not outperform additional training or capacity in their
  parent;
- training and lifecycle cost grows faster than deployed utility;
- self-generated curricula optimize artificial disagreements rather than real needs.

These outcomes would still provide useful evidence. The project should contract to
the simplest architecture supported by the experiments: one RLM, a flat specialist
bank, a larger coordinator, or a distilled dense model.

## Harness Mastery Pivot (2026-08-12)

The primary 27B objective is now to produce the strongest practical Prime Agent and
RLM-harness master, not to conserve untouched Qwen behavior after every small update.
Untouched thinking-mode Qwen3.5-27B remains a frozen baseline. The eventual canonical
teacher will be the trained 27B checkpoint that demonstrates the broadest repeatable
harness mastery while retaining enough general reasoning to interpret tasks, decompose
work, judge child outputs, recover, and synthesize. Modest unrelated regression is
acceptable when it buys material harness competence. Teacher-first means teacher
mastery first, not teacher conservation first.

Work now proceeds in two explicitly separate lanes:

1. The surgical/scientific lane freezes Runs 257-270, their low-rank one-step updates,
   causal probes, token/provenance audits, and predeclared gates as diagnostic evidence.
   It remains the microscope for supervision quality, masks, shortcuts, and causal
   semantics. Its ultra-conservative promotion rules do not govern mastery training.
2. The Harness Mastery lane may move the 27B materially through higher-rank LoRA,
   longer schedules, diverse verified executable bootstrap data, and then on-policy
   RL or native-success SDPO. Partial or full-weight training remains available if
   adapter capacity becomes an empirical bottleneck.

The mastery curriculum covers the coupled policy rather than one boundary at a time:

- direct solving and correct non-delegation;
- useful REPL externalization and persistent coordinator state;
- single-child delegation and parallel fan-out with retained stable handles;
- child-to-parent results, parent-to-child follow-up, causal yield, and resumption;
- fan-in, grounded synthesis, ownership transfer, verification, and reclaim;
- bounded child failure, timeout, challenge, and recovery;
- natural mixed tasks where the policy must choose internal reasoning, the REPL, one
  child, or multiple children from cognitive value rather than an explicit recipe.

The ownership invariant is phase-aware. While a child actively owns delegated work,
the coordinator must not bypass that contract and independently perform the same work.
After a result, failure, escalation, timeout, or explicit ownership transfer, parent
inspection can be valid for synthesis, verification, or recovery. Historical
first-decision probes retain their stricter no-path-access rule for comparability; new
environments should measure unauthorized active-phase bypass instead of permanent
resource exclusion.

Freeze a Prime Agent Harness Mastery Battery before choosing checkpoints. It must have
train-calibration, familiar-heldout, and semantic-OOD splits and report each capability
separately. Track checkpoint trajectories rather than terminating on a one-sample
fluctuation. True catastrophic guards remain: systemic always/never delegation,
ordinary direct-task collapse, loss of persistent state, uncontrolled spawning or
nontermination, systematic active-ownership bypass, and reasoning collapse severe
enough to make the policy unusable as a teacher.

Battery v1 builds directly on Prime's current Verifiers work. The packaged
`prime-agent-capabilities-v1` environment promotes Prime's native ACP evidence for
real IPython execution, persistent kernel state, subagent lifecycle and quiescence,
continual `/refine` state, and loud cancellation into reusable evaluation tasks.
`oolong-synth-v1` supplies natural long-context file externalization. Our
`subagent-communication-v1` and `ownership-invariant-v1` extensions cover the
higher-level routing, fan-out/fan-in, bidirectional messaging, and active ownership
semantics Prime does not yet package as a training environment. This composition is
intentional: extend Prime's substrate instead of rebuilding the harness in a parallel
stack.

The executable battery is pinned to Verifiers `053fd44aa`, which composes current
upstream main `afbf16523`, Prime's staged Prime Agent harness tip `7e3bfc4a1`, and
the complete staged Oolong/ACP-metadata lineage through `141d48abe`. Do not assemble
the Oolong adapter by cherry-picking only its taskset commits: lifecycle scorers need
the ancestry that records inbound ACP `_meta` histories. An initial ten-task 27B
foundations attempt made this failure mode concrete: IPython and persistence scored,
but subagent lifecycle, harness state, and cancellation raised `MissingAcpMeta` after
successful agent turns. Those zeros are invalid infrastructure observations, not a
27B capability result. Re-run the unchanged battery after the composed pin passes a
live metadata-dependent smoke.

The first capacity comparison uses the same thinking-enabled, structurally validated
bootstrap stream and schedule at LoRA ranks 64 and 128. It saves dense checkpoints and
compares learning rate, memory, optimization stability, and the frozen battery across
training time. Rank is an engineering variable, not a claim that adapters are
intrinsically sufficient. The first run may bootstrap complete executable trajectories
without fabricating private chain-of-thought; later RL and native-success SDPO must
broaden and de-template the learned policy on student-visited states.

Promotion now asks whether a trained 27B repeatedly masters unseen executable Prime
Agent tasks well enough to generate trajectories and policy guidance for direct OPD
into 9B, 4B, and 2B students. Pareto dominance over untouched Qwen at every checkpoint
is not required. Direct distillation from the canonical 27B remains the default; a
27B-to-9B-to-4B-to-2B cascade is only a later measured ablation.

## Historical Surgical Record

The next training work remains Stage 0, but its scope is now strictly harness mastery.
Preserve rung 37 step 2 as the 2B expert seed and baseline. The 4B coordinator search
is closed on the present curriculum, and the first 27B teacher qualification is also
complete under the historical no-thinking contract. Untouched 27B can express the
strict guided causal procedure (`15/16`), but
a 24-step mixed SFT run did not transfer it reliably to ordinary prompts: its best
early checkpoint improved a fresh standard screen from `3/6` to `5/6` answers while
still producing zero clean delegated traces, then scored `0/8` causal exchanges on a
fresh bidirectional screen. Later checkpoints overfit the guided contract
non-monotonically. Do not distill from or publish these adapters. These runs remain
useful diagnostics, but no no-thinking result can qualify the teacher or a student.

The first thinking-mode bidirectional screen also exposed an invalid proxy. Its
original scorer required the literal child messages `need multiplier` and `need
nonce`, although the natural task only requires an explicit request for the missing
concept. The literal metrics remain recorded, but promotion now uses provenance and
causal order with a concept-bearing natural request. On the same four untouched-27B
traces, this changes causal exchange from `0/4` literal to `2/4` natural, both on
handshake tasks; clean completion remains `0/4`. The unresolved capability is
file-backed follow-up plus bounded repair, grounding, and stopping, not rote request
wording. A fresh disjoint natural-contract screen must confirm this boundary before
the next optimizer update.

That qualification screen uncovered two further environment confounds before an
optimizer update. First, the cleanliness metric treated the child's required final
`agent_message.send` after a parent follow-up as forbidden post-request work; it now
permits exactly the provenance-linked result send while rejecting additional child
tools. The generic one-way fan-in boundary was also replaced for bidirectional tasks:
only coordinator cells after the final causally linked child result are now forbidden.
Second, the completion gate used the same "result not ready" feedback for
missing child evidence and for a completed exchange followed by prose-wrapped JSON.
It now reports completed evidence explicitly and asks for one bare JSON response with
no further tools. The original follow-up prompt also left the arithmetic relation
between subtotal, multiplier, and result implicit while advertising an unrelated
weighted-checksum formula. The new `explicit_bidirectional_v2` prompt contract states
that relation directly; `historical_v1` remains the default so the frozen 2B battery
does not drift. These are measurement repairs, not learned protocol hints.

The corrected untouched-27B natural admission then returned all eight answers and
completed `7/8` natural causal exchanges. Under the no-leakage contract, however,
only `1/4` file-backed follow-ups and `3/4` handshakes were protocol-aligned, with
`0/8` promotion-clean traces. Three follow-up coordinators reopened work assigned to
their child; the remaining failures were invented messaging APIs, polling, observation,
extra post-request work, and one missing parent follow-up. Mean bidirectional-control
scores were `0.6818` for follow-up and `0.7727` for handshake. This establishes that
27B capacity is sufficient to express the mechanism while externalization discipline
and asynchronous process control remain the narrow training target. The next action
is a four-rollout-per-prompt variance probe followed, only if contrast exists, by one
low-rate GRPO update and independent promotion screens.

The next 27B intervention must be transition-focused rather than another broad epoch.
Train and independently score the standard-prompt states that currently fail: consume
a child request, send one direct follow-up from retained state, consume the child's
result, and finalize immediately without polling or observation. Include matched
single, parallel, and direct retention examples, but select first on repeated fresh
bidirectional causal exchanges and then on the frozen standard families. Only after a
thinking-enabled 27B checkpoint passes both gates should its real, executable,
thinking-enabled trajectories become the teacher corpus for matched 9B, 4B, and 2B
thinking-mode distillation. Preserve the accepted 2B model as an expert seed even if
a larger coordinator wins.

The first hard-gated natural-control GRPO dose was a valid optimization run but failed
that first selection gate. One rank-16 LoRA step at `1e-7` trained on all eight
rollouts with finite loss, KL, and gradient norm. On eight disjoint natural tasks,
both the untouched base and candidate answered `8/8`, and the candidate slightly
reduced mean coordinator access to delegated paths from `0.875` to `0.750`. It also
regressed natural causal completion from `8/8` to `6/8`, reduced mean protocol score
from `0.9375` to `0.8750`, and produced no clean trace. The apparent increase in mean
bidirectional control initially came from a scoring flaw: a tidy but non-causal
follow-up still received partial control credit. Control is now gated on completion
of the natural request-response-result chain and has focused regression coverage.
Under the corrected metric, control regressed from `0.4432` to `0.4091`. Reject this
adapter, skip its frozen-family screens, and retain untouched thinking-mode 27B as the
reference. The next intervention must preserve causal completion while targeting
coordinator reopening of delegated file work; training success or lower average
leakage alone is insufficient.

A second one-step GRPO dose narrowed scalar group credit to the first responses at
the five causal state transitions selected by
`keep_bidirectional_state_transitions`. The authoritative serialized-trace audit
shows that the filter retained `7,095/19,361` sampled action tokens (`36.6%`), and the
optimizer remained stable. Its first disjoint screen
appeared to reduce delegated-path leakage from `0.875` to `0.375`, but causal exchange
already fell from `8/8` to `7/8`. On an independent repeat, leakage instead increased
from `0.500` to `0.875` and causal exchange fell from `8/8` to `5/8`. Combined across
both screens, the candidate answered `16/16` but completed only `12/16` causal chains
versus the base's `16/16`; both produced `7/16` protocol-aligned and `0/16` clean
traces. Reject the adapter without running the frozen family gates. Retain the generic
action-filter mechanism, whose selective and default semantics are tested, but stop
broadcasting one scalar rollout advantage as if it specified the correct action at
each retained state.

The next intervention therefore returns to untouched thinking-mode 27B and uses
per-transition demonstration-conditioned OPSD. The environment already exposes a
role- and state-aligned `turn_demonstrations` sequence, and the same transition filter
selects the exact coordinator and child responses to train. This supplies a distinct
dense target at each student-visited state while preserving the on-policy trajectory;
it is a causal-policy repair, not broad harness imitation. Promotion still requires
repeat natural admission with no causal regression, followed by untouched frozen
family and externalization-choice gates.

That first transition-wide OPSD dose was optimizer-stable but is rejected as a teacher
candidate. Across two disjoint paired screens, it improved natural causal exchange
from `11/16` to `13/16` and protocol alignment from `5/16` to `7/16`, but the gain was
confined to the simpler handshake family. On the target file-backed follow-ups, causal
completion stayed `6/8`, protocol alignment fell from `1/8` to `0/8`, delegated-path
access only moved from `1.375` to `1.250` per episode, and failed cells increased from
`1.625` to `2.125`. Neither arm produced a clean trace, and the candidate introduced
one answer failure. In the independent repeat, three of four candidate follow-up
coordinators reopened the child-owned file and all four used unnecessary polling,
observation, roster, or API-discovery work before completion-gate repair. The adapter
therefore learned part of the message handshake without learning the ownership
boundary that motivates externalization.

The next falsifiable intervention isolates that boundary. Return to untouched
thinking-mode 27B, select only the first coordinator response, and condition its OPSD
teacher on the first matching `turn_demonstrations` step: preserve the hidden value,
spawn the named child with only its assigned path, retain the handle, and end the turn.
Use a larger disjoint follow-up batch for variance reduction, but still exactly one
low-rate optimizer step. Evaluate on fresh paired follow-up and handshake tasks. Do not
train later parent or child transitions until this first ownership decision improves
without answer or causal regression; broadening a failed boundary would make the
result less interpretable.

The isolated first-response OPSD intervention also failed and is rejected without an
independent repeat. Its serialized audit was exact: all `1,875` selected tokens belonged
to one first coordinator response in each of eight traces, with zero child or later
coordinator weight. Nevertheless, on the first disjoint paired gate, file-backed
follow-up leakage increased from `0.75` to `1.25` path accesses per episode, causal
completion stayed `2/4`, alignment fell from `1/4` to `0/4`, and neither arm produced a
clean trace. The training diagnostics are additionally unsafe: mean trainer/inference
mismatch KL was `74.1952`, gradient norm was `1256`, and the clipped loss collapsed to
`3.35e-8`, compared with roughly `0.0002` mismatch KL in the previous valid OPSD run.
Do not reuse this adapter or apply another dose until that discrepancy is understood.

This result changes the teacher-building method, not the top-down program. The 27B
teacher does not itself need to discover a deterministic harness primitive through
OPSD. A first attempt to collect native thinking trajectories under an executable
privileged demonstration produced `8/8` correct final answers but `0/8` measured
request-response-result chains and no admissible traces. The coordinator still read
the child-owned file, polled or inspected child sessions, and continued using tools
after visible messages. The demonstration supplied the answer and protocol description,
not a clean executable policy, so none of these trajectories may become teacher data.

Bootstrap only the first ownership decision with 32 short standard-prompt examples:
retain coordinator-owned state, spawn and retain the child handle without reading its
path, then yield without polling. Keep thinking enabled and supervise concise rationales
for both ownership and event-driven waiting. This synthetic atom is an initialization
aid, not the target policy. Immediately rerun fresh natural trajectory collection and
admit only complete native-thinking traces that preserve delegated ownership and execute
the causal message exchange cleanly. Once the natural 27B policy qualifies, freeze it
and use on-policy teacher guidance on the smaller students' own states as planned.

The one-epoch thinking bootstrap was stable but is rejected. On eight fresh paired
tasks, both untouched base and candidate completed `8/8` natural causal exchanges.
The candidate increased aggregate protocol alignment from `4/8` to `5/8`, but on the
four target file-backed follow-ups it increased coordinator path access from `0.75`
to `1.00` per trace, failed cells from `1.0` to `1.5`, and observation calls from
`0` to `1.25`; neither arm produced a clean trace. The result also clarifies why a
larger dose is unsafe: both the good spawn action and the bad coordinator action contain
the delegated path and a read instruction, so positive token imitation can increase
path salience without learning the structural ownership boundary.

Return again to untouched thinking-mode 27B. The next intervention must contrast the
two actions on policy: reward path use inside the child prompt but reject coordinator
path access outside `rlm(...)`, then apply that signal only to the first coordinator
response. This is narrower than the earlier multi-transition control GRPO dose and
avoids asking token imitation to infer an AST-level distinction from shared words.
Do not run another epoch of the same retained-spawn SFT data.

The follow-up contrastive experiment made the structural target explicit in the
Verifiers environment. Its ownership-only profile disables all broad task rewards and
scores six facts about the first coordinator cell: one successful spawn, retained
handle, expected child name, delegated path inside the child prompt, secret withholding,
and no delegated-path use outside `rlm(...)`. All 68 environment tests pass with this
mode opt-in. A strict eight-rollout group scored all zero and correctly produced no
optimizer update. A first-turn dense preflight then exposed useful natural contrast:
five direct file reads scored `1/6`, two correct-path spawns with discarded handles
scored `5/6`, and one payload-free discarded spawn scored `4/6`.

One dense GRPO update over that boundary was numerically healthy (`8/8` trainable,
reward `0.375`, loss `-0.00119`, gradient norm `0.0781`, mismatch KL `0.000306`). It
nevertheless moved sharply backward on a disjoint natural gate. Follow-up ownership
score fell from `0.708` to `0.333`, coordinator path access rose from `0.25` to `1.50`
per trace, natural causal completion fell from `4/4` to `3/4`, and alignment fell from
`1/4` to `0/4`; handshake retention also regressed. Reject and do not repeat the
adapter. Keep the reward and action filter as diagnostics, but pause one-step updates:
the remaining problem is reliable transfer of an AST-level control boundary, not lack
of a scalar definition or lack of 27B capacity.

The next teacher-first experiment replaced another optimizer guess with native
rejection sampling. Two answer-free guided collections sampled `64` untouched,
thinking-mode 27B first decisions. The corrected seven-part ownership gate admitted
only `7/64` (`10.9%`), but those seven successes covered all four prompt variants and
seven distinct tasks. This is important positive evidence: 27B can naturally reason
through and execute the exact action, including persistent coordinator state, retained
child handle, delegated payload, secret withholding, and path isolation. It does not
yet select that policy reliably. Temporary guidance was stripped from every exported
system prompt, failed and near-miss responses were excluded, and no answer was inserted.

One conservative rank-16 LoRA SFT step at `5e-8` consumed five packed native examples.
It was numerically clean (`loss 0.3187`, gradient norm `0.4414`, no NaNs), but failed
an unguided paired `32 + 32` first-decision gate. On file-backed follow-ups, mean dense
ownership rose from `0.250` to `0.330`, retained handles from `0/16` to `3/16`, and
path isolation from `3/16` to `6/16`. The decisive behavior remained unchanged:
both base and candidate persisted the multiplier in `0/16` traces and completed
`0/16` strict transitions. On the unseen handshake control, dense ownership regressed
from `0.848` to `0.795`, with nonce persistence still `0/16`. Reject the adapter.

This narrows the next intervention. Full-response imitation spreads loss over long
rationales, repeated task descriptions, child prompts, diagnostics, and the few tokens
that actually bind coordinator state. A larger dose of the same objective is not
justified. Preserve untouched thinking-mode 27B and the seven admitted traces, but
next make the teaching signal transition- and token-specific on ordinary student
states. The preferred route is to repair and validate the existing Prime OPSD/SDPO
teacher-replay path against these native admitted decisions, with exact first-response
mask audits and safe trainer/inference KL before any update. If that cannot isolate
the state-binding decision cleanly, expand the native admission corpus before changing
weights rather than canonicalizing failed model behavior by hand.

That prerequisite is now satisfied. The catastrophic first-response mismatch came
from vLLM storing Qwen3.5 recurrent SSM state in bfloat16 between chunked prefill and
decode while the trainer retained float32 chunked state. A zero-update audit with an
FP32 vLLM SSM cache reduced mean mismatch KL from `13.1937` to `0.000220` across 482
selected tokens; median KL was zero and p95 was `0.000839`. A proposed full-sequence
recurrent trainer path was explicitly rejected because vLLM uses chunked GDN for
prefill and recurrent GDN only for decode. The top-down program can therefore resume
without changing the paper-level objective or accepting an unfaithful model forward.

The next experiment is deliberately narrow: one rank-16, `1e-7`, first-response-only
OPSD update from untouched thinking-mode 27B over eight short on-policy follow-up
states, followed immediately by a disjoint paired natural follow-up/handshake gate.
Only an improvement at the trained ownership boundary without answer, causal, or
retention regression permits a broader teacher-mastery curriculum.

The aligned intervention was numerically valid but behaviorally rejected. Its 2,219
selected tokens had mean mismatch KL `0.000255`, loss `0.00871`, and gradient norm
`0.3438`; the former instability is resolved. Yet on the disjoint natural follow-up
gate, protocol alignment fell from `1/4` to `0/4`, delegated-path access rose from
`0.5` to `1.5` per trace, and bidirectional control fell from `0.477` to zero. All
four candidate coordinators read the child-owned file, versus one untouched-base
trace. Handshake retention was approximately flat and all answers remained correct,
so the failure is specifically the ownership boundary the update targeted.

Do not search another learning rate for the same whole-response objective. The
demonstration-conditioned teacher scored 218-371 tokens from each student's sampled
response and did not isolate the executable state-binding action. Native successful
sibling trajectories remain the preferred hindsight source, and Prime-RL now exposes
the required generic SDPO action filter. They are not yet an immediately usable
training source: the existing unguided ownership screen produced `0/32` strict
transitions, so a new ordinary group cannot satisfy sibling admission without a
curriculum change.

A zero-update action-local OPSD audit instead preserved the full first response as
teacher context while selecting only the serialized first coordinator tool call.
Across two fresh trajectories it selected 216 tokens in exactly two contiguous
`<tool_call>...</tool_call>` spans, selected zero rationale, child, or later-turn
tokens, and retained mean trainer/inference mismatch KL of `0.000129`. This establishes
the mechanics but not behavioral value. One `1e-7` action-local OPSD update is the
next controlled bridge because it changes only the loss surface relative to the
rejected whole-response run. A fresh paired gate must reject it if coordinator access
to child-owned paths rises again. Native sibling SDPO resumes only after the natural
policy itself supplies enough strictly admitted successes.

That bridge has now been falsified. Run 257 applied one numerically healthy update to
679 first-action tokens, but the disjoint follow-up gate moved the intended boundary
in the wrong direction: protocol alignment fell from `2/4` to `0/4`, child-owned path
access rose from `0.0` to `1.75` per trace, and bidirectional control fell from `0.500`
to `0.159`. Answers remained `4/4`, causal completion improved from `3/4` to `4/4`,
and the held-out handshake family was unchanged, so this is not a general capability
collapse. It is specific evidence that the demonstration-reprompt teacher assigns
the wrong local policy even when gradients are restricted to the executable action.

The teacher-first program therefore keeps untouched thinking-mode 27B and closes the
whole-response/action-local OPSD branch. Do not tune its learning rate, widen its
mask, or accumulate more steps. The next intervention must change where successful
policy information comes from. Native successful-sibling SDPO remains the preferred
hindsight mechanism, but its admission problem must be solved without converting a
hand-authored reprompt into the teacher policy. Select the next bridge by the smallest
experiment that can produce and admit genuine natural successes while retaining the
frozen follow-up and handshake boundary.

The first native-sibling admission screen now supplies that measurement. Eight
executable child-owned resource families were each sampled eight times from untouched
thinking-mode 27B. Under the predeclared whole-atom verifier, the run produced `0/64`
strict successes and zero mixed groups, so Phase A failed and no SDPO replay or weight
update was allowed. A corrected rescore accepted semantically equivalent coordinator
dictionary state in addition to top-level variables; it did not change the strict
outcome. The dominant failures were stopping after one decision, assigning persistent
state before the spawn, and retaining the returned handle.

There is one narrow unresolved scoring boundary. Ten trajectories satisfied every
other ownership atom and differed from the synthetic target only by passively
displaying the retained handle, evaluating it as the final cell expression, or storing
it in the coordinator-state dictionary after admission. None polled, discovered
agents, opened the delegated resource, or performed local task work. They remain
failures in the recorded result because the no-post-spawn rule was predeclared. Before
rerunning Phase A, decide prospectively whether that rule forbids every AST statement
after `await rlm(...)` or only substantive environment/task actions. Do not launch the
zero-LR SDPO audit until a fresh screen passes the unchanged mixed-group, resource, and
phrasing thresholds under the chosen definition.

The prospective semantic-yield rerun resolved that boundary without changing the
frozen literal result. Semantic yield permits only passive display or storage derived
from the returned handle in the same first cell; it still rejects second cells,
polling, discovery, messaging, delegated-resource access, computation, and all other
post-spawn work. A fresh eight-rollout screen produced five strict successes but only
three mixed groups, one below admission. A predeclared sampling extension then added
eight independent rollouts on each identical task. The combined group size of 16
produced `11/128` strict successes across six mixed resource families and both
ownership phrasings, passing every Phase-A threshold. No weights changed.

The next step is therefore the mandatory zero-LR native-sibling SDPO replay audit,
not a learning update. It must use group size 16 to match the admitted signal and
establish same-group strict-sibling provenance, zero target for groups without a
success, first-coordinator-tool-call-only loss, authentic successful reasoning in the
teacher context, and small trainer/inference mismatch. Only if all five checks pass
may one rank-16 `1e-7` update be considered.

Prime-RL currently rejects EMA self-distillation together with LoRA because its EMA
runtime instantiates a second full model. Run 262 therefore uses live-policy teacher
regularization. This does not change the zero-LR audit or a single-step bridge: at the
only teacher evaluation, live policy and a newly initialized EMA teacher have
identical parameters; they diverge only after the optimizer step, when the proposed
bridge ends. Multi-step native SDPO will require adapter-only EMA support or a larger
full-weight training deployment rather than silently extending this equivalence.

Run 262 did not obtain a replay signal. Its fresh 16-sample JSON group and the
predeclared 16-sample CSV no-success control both produced zero strict successes.
SDPO consequently created no teacher targets, `TrainSink` removed all 32 zero-loss
samples, and no trainer step, token export, or weight update occurred. This is a
sampling failure rather than an algorithm failure: Phase A admits the population-level
native signal, but one nominally mixed group is too brittle for the replay audit.

Run 263 repeats the identical zero-LR semantics over four Phase-A mixed families and
the same CSV no-success control, all at group size 16. The model, verifier, seed,
temperature, teacher construction, reasoning policy, token filter, and optimizer are
unchanged. Phase B still requires an observed same-group native sibling replay and all
five provenance/mask/numerical checks; the larger batch cannot itself count as a pass.

Run 263 exposed a served-task configuration bug and is not experimental evidence.
Prime-RL materializes task data client-side but a separate Verifiers server rebuilds
task behavior from `env.taskset.task`. The ownership taskset applied semantic yield to
the client-side `Task` while leaving that serialized task subtree at its literal
default. This produced the diagnostic contradiction `passive_handle_tail=1` together
with `post_spawn_action=1`. The run was stopped after 51 rollouts, before any trainer
step or token export. Verifiers commit `be7e0576` propagates the policy through the
served config and tests the exact server reconstruction path. Run 264 repeats the
otherwise frozen audit from scratch and also declares the task policy explicitly.

Run 264 passes the complete zero-LR Phase-B gate. The four replay groups each
contained two strict native successes, while the JSON control contained none. All 64
eligible replay branches mapped exactly to same-group native sibling demonstrations;
the no-success group exported zero targets. Nonzero SDPO weights selected only the
configured first coordinator tool call, covering 6,152 tokens. Every one of the eight
unique native teachers retained reasoning, mean sequence mismatch KL was `0.000173`,
and LR remained exactly zero. The committed result ledger preserves exact teacher
text and content hashes as well as hashes for the traces, audit, and rank exports.

Run 265 therefore applies exactly one rank-16 native-sibling SDPO update at `1e-7`.
No second dose is allowed until the candidate is compared with untouched 27B on the
frozen ownership and direct-control screens, natural follow-up and handshake gates,
and an independent repeat. This keeps the next inference causal: Phase B established
that the implementation trains on the intended source and tokens; Phase C asks
whether one minimal update improves that behavior without damaging coordinator
controls.

The first Run 265 launch was invalidated by operator interference before an optimizer
step. Running the Prime-RL test suite concurrently activated its autouse
`cleanup_zombies` fixture, which uses broad process-name kills for `torchrun` and
`VLLM`; this terminated the live trainer and inference server after 23 partial traces.
Those artifacts are archived and hashed but excluded from all model conclusions. The
identical predeclared dose must be rerun without concurrent tests.

The selection boundary is now executable rather than interpretive. The primary
Run 266 battery pairs 16 held-out-phrasing child-owned tasks with 16 matched
coordinator-owned direct controls. It requires at least six strict task gains and no
more than one loss, while treating any candidate coordinator access to a child-owned
path or any candidate delegation on a direct control as an immediate rejection. The
existing disjoint Run 258 follow-up/handshake gate must also preserve each family's
answers, natural causal completion, protocol alignment, and mean bidirectional
control, with no delegated-path access by the coordinator. If both pass, Run 267
repeats the ownership battery on unseen instances and the natural gate on the already
frozen Run 239 offset. These rules were fixed before observing Run 265 and cannot be
relaxed to admit its outcome.

Do not apply another 2B dose of the same complete-fan-in demonstration: even
coordinator-only, response-only SDFT traded
single-child and parallel reliability. The first parallel-only process-control GRPO
update also regressed from `9/12` to `7/12`, despite a strong on-policy batch. The next
curriculum must reproduce actual unsolved failures across single and parallel tasks,
score visible-message consumption and bounded traceback repair, and include direct and
delegated retention groups in every update. The later full-weight SDPO plus
coordinator-SDFT preservation control reached a balanced `10/12` but did not dominate
the retained baseline, so another reweighting of the same arithmetic examples is not
the next experiment. Independently, make the complete
bidirectional causal chain reliable:

```text
child request -> parent response from retained state -> child final result
```

Then rerun fresh held-out direct-IPython, one-child, parallel fan-out/fan-in,
follow-up, traceback-repair, output-contract, and clean-stop gates at standard as well
as guided instruction levels. Do not add more document formats to this root
curriculum. Once one checkpoint passes the complete suite repeatedly, freeze and
publish it as the first orchestrator.

The next major milestone is the document expert trained against its own extraction,
repair, and grounding suite. Only after both roles pass independently do we train and
evaluate their interface. This sequence turns the long-term growth thesis into three
immediate falsifiable questions: can the smallest model coordinate, can a specialist
add depth, and does their combined system beat either alone at comparable cost?
