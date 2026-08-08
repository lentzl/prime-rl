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

### Stage 0: Stabilize the RLM substrate

Train one Qwen3.5-2B instruct policy in Prime Agent to reliably use persistent IPython,
tools, files, external state, and portable skills. The current IPython foundation and
file-processing ladder belongs here.

Required behavior includes:

- accepting silent assignments and reusing variables across turns;
- inspecting structured tool results before acting on them;
- reading tracebacks and changing the failed operation rather than repeating it;
- preserving successful state while repairing only the failing step;
- selecting processing methods from observed type, MIME, extension, and content;
- grounding final answers in observed evidence, including negation;
- following output contracts and stopping without protocol leakage;
- using, creating, and revising skills without overfitting to one vendor's package
  layout.

Gate: a checkpoint must improve the target behavior across repeated held-out samples
while all earlier foundation gates remain within variance. No hierarchy is introduced
until these behaviors are stable enough that coordination failures can be separated
from basic harness failures.

Current status: the published file-processing rung is the stable starting point. It
has demonstrated persistent state, file acquisition, parser selection, and useful
repair behavior, but repeated-call control, structured-result inspection, strict
output formatting, and source-grounded negation are not yet robust.

### Stage 1: Establish the generic RLM baseline

Extend the stable substrate from notebook basics to invariant agent skills:

- decompose work into testable subproblems;
- store durable procedures and facts outside the immediate context;
- retrieve and apply portable skills;
- use recursive model calls only when they add measurable value;
- compare hypotheses through execution rather than monologue;
- recognize uncertainty and terminate with an evidenced limitation.

Evaluate Qwen3.5-2B against its untrained base and, if necessary, a 4B baseline. The
comparison asks whether 2B can learn the substrate well enough, not whether it wins a
single benchmark by chance.

Gate: choose the smallest model that meets robust process and task-success thresholds.
If 2B cannot sustain metacognitive control, use 4B for parent roles while keeping 2B
available for narrower specialists.

### Stage 2: Train one domain parent

Choose one broad domain with diverse tasks, objective or auditable outcomes, and enough
failure volume to identify structure. Start with a single model, not a predeclared
expert tree.

Train it to solve tasks, classify its uncertainty, decompose problems, preserve RLM
behavior, and produce diagnostic trajectories. Record failures with environment state,
feedback, retries, resource use, and eventual outcome.

Gate: the parent must beat the generic RLM in-domain without unacceptable regression
on the generic RLM suite. Its failure distribution must contain at least one stable,
coherent cluster rather than only random mistakes.

### Stage 3: Create the first child specialist

Select one recurring failure cluster that is narrow enough to train and broad enough
to recur. A frontier teacher can initially design the curriculum and audit labels, but
the child must ultimately learn through executable tasks and real feedback.

Initialize the child from the shared RLM-capable base or parent, then train narrow
depth while replaying generic RLM behavior. Do not assume that a narrower prompt alone
constitutes specialization.

Run three matched comparisons:

```text
A. Continue training the parent
B. Use a larger or equivalently more expensive dense parent
C. Parent plus the candidate specialist
```

Match or explicitly account for active parameters, generated tokens, latency, GPU
time, and training cost. Evaluate both the specialist's narrow domain and the complete
system.

Gate: retain the child only if its marginal whole-system contribution exceeds its
training, inference, coordination, and maintenance cost. Otherwise discard it and
continue with the simpler parent.

### Stage 4: Learn the parent-child interface

Give the parent access to the validated child and collect real interaction trajectories.
Train the parent to:

- recognize when the child's depth is relevant;
- formulate a self-contained, useful subproblem;
- pass the necessary state without flooding the child;
- detect missing assumptions or suspicious child answers;
- request clarification or verification when needed;
- integrate the result into a grounded final response;
- avoid delegation when direct solving is cheaper or safer.

Use whole-system reward for end-to-end success. Use feedback-conditioned SDPO for
same-parent hindsight where child or environment feedback improves the parent's next
behavior. Use OPD/MOPD only for same-role transfer or carefully selected interface
knowledge.

Gate: the learned router must beat fixed routing, always-delegate, and never-delegate
baselines. Parent plus child must still beat the best matched dense baseline after
coordination overhead is counted.

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

The next training work remains Stage 0. Improve reward semantics for no-repeat control,
structured-result inspection, successful repair, strict output contracts, and
source-grounded claims while preserving the verified file-processing checkpoint.

After that gate is stable, the next major milestone is not a recursive tree. It is a
single generic RLM baseline with portable skill use and measured recursive-call value.
Only then should we select one domain parent and run the first specialist-versus-dense
comparison.

That sequence keeps the long-term idea ambitious while making every near-term run
answer one falsifiable question.
