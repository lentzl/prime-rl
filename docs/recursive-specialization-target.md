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

### Stage 0: Bootstrap the smallest harness-native orchestrator

Start with Qwen3.5-2B Instruct and train only the capabilities needed to operate Prime
Agent reliably. The root is a control policy, not the first domain expert. Persistent
IPython is its working memory and coordination substrate; RLM children and explicit
messages are its expandable cognition.

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

The admission suite must cover direct IPython work, one-child delegation, parallel
fan-out/fan-in, bidirectional follow-up, malformed child output, delayed replies, and
recoverable tool or messaging failures. Every delegated trajectory is scored by
message provenance and order, not merely by a correct final value.

Gate: promote the smallest checkpoint that repeatedly passes all harness families on
unseen seeds and under realistic Prime Agent autonomous continuation. Compare LoRA
against a higher-rank or short full-parameter run only if clean online optimization
plateaus. Move from 2B to 4B only if 2B still cannot sustain the complete protocol
after that controlled comparison.

Current status: Qwen3.5-2B has mastered useful IPython foundations. A Prime-native
OPSD repair produced a new Stage-0 candidate at
`37-single-path-opsd-dose-r1/weights/step_2`: it preserved task-specific paths in
`4/5` held-out initial actions, solved full single-child and direct gates `3/3`, and
returned exact answers on all three parallel gates. Parallel message provenance was
fully aligned in only `2/3`, and the previously isolated bidirectional follow-up chain
is still not reliable. The model is therefore materially closer but not admitted as
the first orchestrator yet.

On a broader 12-task standard-prompt gate, this candidate retained all direct and
single-child answers but solved only three of four parallel answers; 10/12 episodes
jointly met answer and protocol requirements. Two follow-up OPSD experiments were
rejected. A shared coordinator demonstration reduced parallel accuracy to `2/4` by
teaching the wrong role to child branches. Role-conditioned demonstrations restored
parallel answer accuracy to `4/4`, but protocol alignment remained `2/4`, duplicate
actions increased, and single-child accuracy fell to `3/4`. The infrastructure now
supports exact initial-question keyed demonstrations per branch, but no later
checkpoint supersedes rung 37.

The selected adapter has also been merged into a standalone dense candidate. All 96
updated matrices and 521 unchanged tensors passed exact export checks, and
adapter/dense greedy spawn behavior matched. A sampled dense smoke still omitted the
path and looped during repair, confirming that the remaining reliability problem
belongs to the policy rather than the export pipeline.

### Stage 1: Train the first expert child for documents

After the Stage 0 checkpoint is frozen, train document handling as a separate expert
role. This child owns the depth that should not burden the small orchestrator:

- selecting parsers from extension, MIME type, magic bytes, and observed structure;
- extracting grounded content from text, Markdown, CSV, JSON, PDF, and DOCX;
- repairing missing parsers, malformed inputs, encoding failures, scanned PDFs, and
  password protection without repeating unchanged calls;
- preserving source locations, page references, uncertainty, and negation;
- returning a compact, typed result that the orchestrator can inspect and synthesize.

The expert may start from the harness-capable checkpoint, but its specialization gate
is independent of the root's gate. It can be larger than the coordinator if document
depth requires it; sparse activation, not uniform node size, is the efficiency target.

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

## Immediate Work

The next training work remains Stage 0, but its scope is now strictly harness mastery.
Use rung 37 step 2 as the retention base. If SDFT is used again for asynchronous
fan-in, restrict its loss to the coordinator branch; distilling child branches moved
answer binding and protocol control in opposite directions. Independently, make the
complete bidirectional causal chain reliable:

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
