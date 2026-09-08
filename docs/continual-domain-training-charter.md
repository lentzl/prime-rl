# Continual domain training: standing charter

**Version 1 — 8 September 2026**  
**Current application:** English document summarization within RLM + the existing Prime Agent harness and persistent REPL.  
**Purpose:** consolidate the Owner's agreed training philosophy, not launch a new experiment or replace Trainer's current plan.

> Develop a substantially more capable small-model system through sustained, cumulative weight training on an evolving, experience-informed curriculum. Continue from the latest valid experimental state by default; use evaluation to steer learning, checkpoints to make exploration recoverable, and separate evidence to decide when a model is reliable enough to promote.

## 1. What the pretraining analogy means

The analogy is **continuity of learning**, not pretraining from scratch or a requirement to use an unsupervised objective. We keep developing the learned system rather than treating each small update as an isolated candidate that must prove itself before training may continue.

The foundation is the acquired e33/H176 role lineages and their current descendants, RLM delegation, Prime Agent execution, and persistent REPL state. New domain competence is added to those foundations. Do not reset a role to e33/H176 at every training block, restart from vanilla Qwen, or build a replacement orchestration harness. Vanilla remains an eventual comparison, not a development dependency, unless the Owner explicitly changes that instruction.

The current learning path can remain supervised: source-grounded examples, synthetic tasks, corrected student experience, and executable tool/action targets. Different update objectives may become useful later, but implementing a new optimizer framework is not required to adopt this charter. Continuous learning means sustained cumulative development; collection, training and evaluation may still happen in practical blocks.

The ambition is useful **system-level capability** under scarce deployment inference. Small gains are footholds for further learning, not the final objective. We are willing to change candidate weights substantially and accept recoverable behavioural-collapse risk to explore that ambition.

## 2. Training frontier and reliable references are different

The **experimental frontier** is the latest valid checkpoint of the role being trained. Continue from it by default even when some tasks fail, scores are mixed, or another capability has regressed. Those observations guide the next data mixture, exposure and correction targets; they do not impose a minimum capability threshold for permission to learn.

**Reliable references and recovery checkpoints** serve different purposes. Protected references remain intact; recoverable predecessor snapshots permit comparison and branching. Promotion to a reliable reference or deployable version requires appropriate evidence, but lack of promotion does not disqualify a checkpoint as the next training parent.

A failed complete episode can still contain useful reads, sound reasoning products and valid partial actions. Retain and teach that useful behaviour together with source-checked corrections. Never label a hallucinated continuation as a correct target merely because it came from the latest model.

“Continue by default” is not a command to follow a ruined lineage indefinitely. Trainer may repair forward after a mild regression or rewind after a severe, persistent collapse. Neither choice requires another strategic approval when it remains inside the agreed application and resources.

## 3. The evolving learning loop

```text
Current trained frontier
    -> experience and behavioural inspection
    -> source-grounded targets, corrections and new cases
    -> refreshed training mixture
    -> meaningful weight-update block
    -> checkpoint and proportionate evaluation
    -> continue from the new frontier

Severe collapse or technical corruption
    -> inspect the cause, recover an earlier state when appropriate
    -> change the recipe or data
    -> continue learning
```

Trainer owns this loop end to end: source selection, curriculum, target authorship, implementation, training duration, diagnosis and routine follow-on decisions. Prepare follow-on data during running jobs where practical. Data need not be regenerated every optimizer step, and an evaluation need not interrupt every block.

Use an evolving mixture of relevant source material, newly generated cases, recent failures with better targets, useful successful behaviour, and rehearsal of established competence. Preserve the abilities the wider organization still needs: bounded assignment, child return, grounded retrieval, REPL use, persistence and completion. Rehearsal is a way to continue learning without losing foundations, not a requirement for every sentinel to pass before training proceeds.

The corpus should be **curated and replenished, not append-only by rule**. Preserve historical datasets and provenance, but allow removal, correction or downweighting of poor targets and redundant examples in subsequent training versions. Repeating three answers many times is exposure, not diversity. Add new documents and genuinely different relationships, not only renamed entities or larger row counts.

Synthetic supervision should address the actual learning boundary. Examples in this campaign include observation versus inference, attempted versus completed events, temporal versus causal relationships, additional versus replacement requirements, quantities with their units, and main conclusions versus illustrative anecdotes. These are illustrative curriculum directions, not a prescribed new batch or checklist.

## 4. Train boldly without reinforcing an error loop

The student supplies useful **states and experience**, not unquestionable truth. Ground targets in the original source and actual environment outcomes. Teacher-authored and student-generated material remain distinguishable; checked synthetic examples need not wait for a fully successful autonomous rollout.

Train the behaviour in the state where it will be needed. Preserve the real tool interface, role sequence, observable information and completion boundary. When a failed draft remains in context, mask its target loss and supervise the correction. If an action changes, replay or regenerate subsequent observations rather than attach an incompatible old tool result. Do not put unseen answers or privileged evaluation annotations into the student's context.

Validation should target real risks with existing tooling: source fidelity, executable observations, correct loss masks and usable context length. Reuse established checks for unchanged paths; do not turn every example into a proof campaign. Correct extraction or formatting alone does not establish semantic quality.

Grow training exposure when there is useful data and a learning opportunity. There is no standing one-, eight-, or other fixed-update ceiling, and no requirement that each block show a monotonic score improvement. Equally, larger update counts or automatic doubling are not substitutes for useful supervision. Trainer chooses block size, learning rate and mixture from observed behaviour, optimization health and available host time.

## 5. Recovery and optimization continuity

A mild or localized regression normally calls for targeted corrections, rehearsal or a mixture adjustment while continuing from the frontier. A severe, persistent loss of basic behaviour can justify returning to a recoverable predecessor and changing the data, exposure or learning rate. Numerical corruption, unusable training targets or broken execution infrastructure call for repair of the affected work; a provider/harness failure must not be counted as evidence that the model lacks capability.

Keep enough state for meaningful recovery using the existing retention system. At suitable resume points, retain optimizer/scheduler state, step counters and data/RNG state where supported and affordable, in addition to weights and their provenance. Weights-only restoration is a new training continuation, not an exact resume.

For an otherwise continuous run, preserve optimizer and schedule continuity where practical rather than restarting them solely because a reporting block ended. A deliberate optimizer reset can still be reasonable when the training setup changes. Record it. This is a prospective engineering preference, **not a requirement to redesign the trainer, retain every full optimizer snapshot, or interrupt the active run**. Work within the machine's actual disk capacity and preserve protected or irreplaceable artifacts.

## 6. Evaluate to steer, not to ration learning

Inspect actual outputs and complete episodes at useful intervals. Separate semantic fidelity, relevant coverage, action execution, artifact completion, stopping and resource use; neither a loss curve nor a formatting reward is the whole task. Use modest behavioural sentinels and occasional broader comparisons, not a mandatory full matrix after every update. Choose the cadence for how much it informs the next decision relative to its cost.

Development tasks may be revisited to diagnose and correct failures. Reserve genuinely unexposed documents for occasional confirmation; once a result influences the curriculum, it is development evidence thereafter. Keep related source variants in the same split and report pretraining familiarity where it cannot be excluded. Preserve historical measurements rather than silently reinterpreting a changed evaluator as the old test.

Evaluation should usually lead to “here is what we teach next,” not “the checkpoint has not earned more updates.” Strong performance claims and reliable-model promotion remain separate decisions. The newest training parent is allowed to be imperfect.

For this application, a chapter worker may read and directly write its summary in the REPL. Source-linked notes remain available when useful, not mandatory for every synopsis. Worker improvement must feed back into actual task-owner delegation and artifact handling; do not make perfect performance on a short rewrite exercise an indefinite prerequisite for using the organization.

## 7. Resource use and responsibilities

Optimize for useful learning and capability per **rented machine-hour**. The Owner pays for the existing two-GPU host while it is idle as well as busy. There is no campaign-wide GPU-hour quota. Per-run timeouts, checkpoints and review points prevent runaway work; they do not require leaving the host idle or requesting a fresh allowance after each block.

One hands-on Trainer and one coherent campaign can use both GPUs, overlap CPU preparation with training, and schedule useful inference/evaluation work when memory and process isolation allow it. High utilization alone is not the goal: do not run known-useless jobs merely to occupy hardware. An actual blocker should be named while unaffected useful work continues where possible.

The existing host, the Owner's rental-duration instructions and any real spending/security constraints remain the boundary. No additional machine, paid service, storage purchase or rental extension is authorized by this charter.

Trainer owns routine experimental decisions. The Strategist is a second pair of eyes and active support for difficult problems, not a required teacher-data supplier or per-run approver. Updates should link useful existing evidence and state the interpretation and next action. The conversation still needs activation to respond; lack of a reply is not a stop instruction. Scope changes and reliable-model promotion remain subject to the established Owner decisions.

## 8. Scope and immediate application

This charter formalizes **within-domain** continual training. It does not launch coding or robotics campaigns, a weight-merging programme, lateral subsystem messaging, or new parallel training strands. Those are interesting future extensions, not assignments created by this document. Recurrence and latent communication remain candidate ways to strengthen the system, not prerequisites for continuing current weight training.

At drafting, Trainer's latest posted status was R5 learning from R4, with 128 planned updates and 88 episodes, after R4 continued from R3 for 64 updates. That is consistent with the charter. This is a reported status, not a live machine inspection or a result from R5. Leave the active run and frozen evaluation conditions unchanged. Apply the policy through the normal learning loop; no acknowledgement gate, rewritten roadmap or new reporting schema is required.

This document consolidates the Owner's earlier continuity and resource instructions. Where older Strategist advice would impose an artificial GPU-hour allowance or require semantic admission before the next training block, this charter supersedes that advice. It does not weaken data integrity, checkpoint protection or actual Owner constraints.

## Existing project records

- [Owner's successive-update direction](https://github.com/lentzl/rlm/issues/1#issuecomment-5582796792) and [committed implementation note](https://github.com/lentzl/prime-rl/commit/d3117e47fc3ad7f2633aaac49d7d2340ea3bdcc9).
- [Withdrawal of artificial GPU-hour ceilings](https://github.com/lentzl/rlm/issues/1#issuecomment-5580223241).
- [Prime Agent/REPL and trained-lineage continuity](https://github.com/lentzl/rlm/issues/1#issuecomment-5571491636).
- [Trainer ownership and Strategist support](https://github.com/lentzl/rlm/issues/1#issuecomment-5573696684).
- [R5 launch and cumulative curriculum update](https://github.com/lentzl/rlm/issues/1#issuecomment-5584151773).

This is an operating agreement derived from project decisions, not a claim that a particular training algorithm or biological analogy has been experimentally validated.
