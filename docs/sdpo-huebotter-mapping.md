# Huebotter SDPO and the Algorithm Abstraction

This note maps Huebotter-style SDPO onto the algorithm abstraction introduced in
`feat/algorithm-abstraction-v1`. It is intentionally a design note, not an API
proposal yet.

## Current Prime Pieces

The abstraction already has the important generic pieces for distillation-style
training:

- `opd` samples from the live policy, scores the sampled tokens under a teacher
  model, and routes action tokens into the `ref_kl` loss component.
- `opsd` uses the same `ref_kl` loss component, but rebuilds the reference
  scoring prompt with an expert demonstration. This is the SDFT-shaped path.
- The trainer is algorithm-blind. It consumes per-token component streams and
  applies `rl`, `ce`, and `ref_kl` losses independently.

That means Huebotter-style SDPO should not start as a separate trainer mode. The
first question is whether it is an `opd`/`opsd`-family reference-scoring
algorithm with a different way to build the teacher context.

## Loss-Level Mapping

The older sampled-token SDPO primitive we used as a reference had the per-token
form:

```text
(log p_student(token) - log p_teacher(token)).detach() * log p_student(token)
```

The `ref_kl` component uses the teacher gap as the policy-gradient signal:

```text
ref_kl = log p_teacher(token) - log p_student(token)
loss ~= -ref_kl.detach() * importance_ratio
```

At the fully on-policy point where trainer and inference logprobs match, both
give the same gradient direction with respect to the sampled token logprob. The
Prime implementation also adds the machinery needed by the asynchronous training
stack: importance-ratio correction, mismatch masking, and separate component
normalization.

So the likely upstream contribution is not a new loss function unless a direct
reference test finds a real mathematical mismatch. A better first contribution
would be a small test/documentation patch that makes this relationship explicit.

## Algorithm-Level Gap

Huebotter-style SDPO is different from plain `opd` and from `opsd` mainly in how
the teacher context is constructed:

- `opd` scores the sampled completion under the original rollout context and a
  separate teacher model.
- `opsd` scores the sampled completion under a demo-conditioned prompt.
- SDPO needs a hindsight/feedback-conditioned self-teacher context: the policy
  first acts, the environment returns feedback, and the teacher distribution is
  obtained from the policy family conditioned on that feedback.

In Prime terms, this suggests an `opd`/`opsd`-family algorithm whose
`score_batch` hook rebuilds the reference prefix from rollout metadata such as:

- original prompt
- sampled rollout or failed attempt
- environment feedback
- optional successful previous rollout
- optional sibling/group context

The trainer-side `ref_kl` component can probably remain unchanged.

`opsd` is already close to the needed mechanism: it reads a configurable
`demo_key`, applies a configurable template to the last user message, renders
that reference prefix through the policy renderer, scores the sampled completion
under the reference context, and scatters those completion logprobs back onto
the trainable positions of the original sample.

That suggests a useful first experiment before adding a new algorithm type:
represent SDPO hindsight feedback as the `opsd` demonstration field and use a
feedback-oriented template. If that is enough, the missing Prime contribution is
documentation and examples. If it is not enough, the gap should be stated in
terms of what `opsd` cannot express, for example multi-turn feedback selection,
sibling-aware context construction, or a structured feedback object instead of a
single demonstration string.

The current `opsd` implementation is explicitly single-step. That is a good fit
for SDFT-style prompt/response tasks, but it is not enough for multi-turn agent
rollouts such as RLM harness episodes. In a multi-turn trace, sampled assistant
tokens are interleaved with environment observations, user feedback, tool
outputs, and later sampled assistant turns. A feedback-conditioned SDPO scorer
must decide which context each sampled turn is scored under and preserve that
stepwise alignment. Simply taking all trainable tokens and scoring them after
one rewritten prompt would lose the observations between turns.

This narrows the likely SDPO gap to one of:

- a single-turn `opsd` configuration pattern for environments that can emit
  hindsight feedback as a demonstration string
- a multi-turn `opsd` extension that scores each sampled segment under a
  rewritten prefix while preserving intervening context
- a new `sdpo`/feedback-distillation algorithm if the feedback object or sibling
  context is too structured for the existing `opsd` template model

## Contribution Shape

A review-friendly sequence would be:

1. Add reference tests or documentation showing the equivalence between the
   sampled-token SDPO primitive and Prime's `ref_kl` gradient at the on-policy
   point.
2. Identify the smallest generic metadata needed for feedback-conditioned
   reference scoring. Avoid RLM-specific names in Prime core.
3. Prototype a new reference-scoring algorithm only if `opd`/`opsd` cannot be
   configured cleanly for hindsight-conditioned prompts.
4. Keep environment-specific replay construction in `verifiers` or task
   packages unless Prime already has a matching generic abstraction.

## Working Hypothesis

The core SDPO contribution should live at the reference-scoring/replay boundary,
not in the trainer hot path.

In other words: preserve Prime's algorithm abstraction, reuse `ref_kl`, and make
the missing piece the construction of high-quality hindsight-conditioned teacher
contexts.
