# SDPO Reference Tests

The trainer-side SDPO tests use fixed outputs generated from
`lasgroup/SDPO@c52586b`, specifically the author-lineage
`verl.trainer.ppo.core_algos.compute_self_distillation_loss` implementation.
The reference repository is not imported, vendored, or installed by Prime.

## Why fixed constants

An implementation-against-implementation test can let both sides change at
once, makes the test environment depend on a large training stack, and hides
the exact contract under test. Fixed inputs and outputs make a reviewed commit
the immutable oracle and keep Prime's unit suite small and deterministic.

The constants are checked into
`tests/unit/train/rl/sdpo_reference_cases.py`. Their generator lives outside
Prime's runtime and records the source commit in its output. Regenerating them
is an explicit provenance update, not part of running the tests.

## Coverage

The cases cover the independent mathematical branches of the reference loss:

- sampled-token self-distillation and response masking
- sequence-level self-distillation masking
- one-sided importance-ratio clipping
- full-vocabulary forward KL, reverse KL, and Jensen-Shannon divergence
- student-selected top-K distillation with the aggregate tail bucket
- token- and sequence-level truncated rollout importance sampling
- gradients through the student distribution only

The integration tests separately cover the details that fixed scalar outputs
cannot establish: causal token alignment, selecting top-K IDs from the student,
scoring those same IDs in the feedback-conditioned teacher context, packing
teacher replay spans, and composing SDPO with Prime's other normalized loss
components.

## Why these probabilities

The fixtures are deliberately small, non-uniform distributions. They include
different student and teacher rankings, meaningful mass outside top-K, masked
positions, and ratios both below and above clipping thresholds. Symmetric or
near-one-hot fixtures would make several wrong implementations produce the same
answer and are therefore avoided.

This is a branch-complete unit oracle, not statistical validation. CUDA/FSDP
acceptance remains a separate requirement because unit constants cannot prove
distributed model lifecycle, memory use, or end-to-end Verifiers transport.

## End-to-end fidelity contract

The integrated `sdpo` path follows the paper's training algorithm, not merely
the standalone loss:

| Paper requirement | Prime implementation |
|---|---|
| On-policy rollout groups | `sdpo` only accepts policy sampling; each group supplies attempts for one task. |
| Hindsight context | A successful sibling is used as the correct solution, otherwise natural environment feedback is used when available. |
| Reprompt shape | Earlier prompt messages are preserved and the final user turn is replaced with the paper template. |
| Original-response replay | The teacher re-evaluates the exact sampled completion; it never generates a replacement response. |
| Per-token estimator | Student and teacher distributions are aligned at every sampled response position and aggregated by token mean. |
| Stop-gradient teacher | Teacher scoring runs without gradients. |
| Regularized teacher | The default teacher is a separately checkpointed EMA of the student, updated after each optimizer step. |
| Failed updates | Non-finite gradient norms skip the optimizer, scheduler, and EMA updates together. |
| Approximate logit distillation | Top-K IDs are selected from the student and those same IDs are scored by the teacher, with one aggregate tail bucket. |
| Off-policy correction | The sampled-token ratio and truncated rollout importance weights both use current-policy versus rollout logprobs. |
| Packed replay isolation | Every teacher replay resets position IDs, which Prime's packed-attention path converts into independent sequence boundaries. |

The default algorithm behavior also matches the released experiment configs:
successful attempts are not reprompted with themselves, thinking traces are
removed from sibling demonstrations, environment feedback is suppressed when
a successful solution is available, and teacher prompts are right-truncated.
Setting `dont_reprompt_on_self_success = false` restores the paper template's
self-success behavior and follows the reference implementation's first-success
selection order.

The two paper training regimes remain configuration choices rather than two
different implementations:

| Setting | Top-K | Divergence (`alpha`) | EMA rate | Learning rate |
|---|---:|---:|---:|---:|
| Without rich feedback | 100 | generalized Jensen-Shannon (`0.5`) | `0.05` | `1e-5` |
| With rich feedback | 20 | reverse KL (`1.0`) | `0.01` | `1e-6` |

The debug config selects the rich-feedback setting. Both regimes use sampling
temperature `1.0`, token-level rollout importance clipping at `2`, AdamW, and
gradient-norm clipping at `1.0`.

The two importance factors in the loss are deliberate, not duplicate plumbing.
The released SDPO config sets `self_distillation.is_clip = 2`, while the
experiment launchers separately set `rollout_correction.rollout_is = "token"`
with threshold `2`. The reference implementation multiplies the sampled-token
ratio and the rollout-correction weight, so Prime preserves that composition.

## Scope boundaries

The current integrated runtime intentionally covers the paper's Section 3 and
Section 4 single-response training setup. It does not yet claim:

- the Appendix trust-region teacher alternative to EMA
- Section 5 test-time training or its advantage clipping
- multimodal or context-parallel teacher replay
- a full-vocabulary runtime path; the tested primitive supports it, while the
  integrated trainer requires the paper's memory-efficient Top-K path

These are explicit extensions, not silent approximations of the supported
path. The trainer rejects unsupported combinations before they can produce a
plausible but semantically different run.

## Multi-turn replay extension

`orchestrator.algo.multi_turn_replay = true` extends teacher replay to a
linear agent trajectory. It constructs one teacher span for each sampled
assistant turn and attributes only the non-sampled environment messages before
the next sampled turn as that turn's feedback. Rollout-level `info.feedback`
is reserved for the final turn. A turn with neither local feedback nor a
successful sibling is not assigned an SDPO target.

Teacher spans are packed in trajectory order up to `model.seq_len`. If their
aggregate length is larger, the trainer evaluates multiple teacher batches
and writes each result back to the span's original student positions. This
keeps the configured per-forward context bound without dropping turns. A
single teacher span larger than `model.seq_len` remains an explicit error.

This option is outside the paper's evaluated scope. In particular, the
"multi-turn" baseline in Section 5 is repeated complete-answer sampling with
an evolving context and fixed weights, while Section 5 SDPO repeatedly updates
weights between complete-answer batches. Neither is an intra-trajectory,
per-action SDPO objective. The extension preserves the paper's core replay
invariant — the teacher evaluates the student's exact sampled tokens — but its
per-turn feedback attribution is a Prime-specific experimental design.

## Section 5 control-flow smoke

`configs/debug/algo/sdpo_ttt_smoke.toml` exercises the published
test-time-training schedule without presenting the reverse-text debug task as
a reproduction of the paper's Qwen3-8B/LCBv6 result. It selects one fixed
finite task, samples 16 attempts, applies one update, broadcasts the new policy,
and repeats the same task for a second attempt group. Setting
`max_train_batch_lead = 0` disables Prime's normal one-batch pipeline lead so
the dispatcher issues at most one rollout batch per policy version and the
second group cannot be sampled before the updated policy is live. The
config also fixes the
published no-thinking, temperature, Top-K, reverse-KL, EMA, optimizer,
weight-decay, gradient-clipping, and rollout-importance settings.

Table 12 additionally lists `Clip advantages = 5.0`. The public
`lasgroup/SDPO` repository contains neither the TTT launcher nor an implemented
advantage-clipping path, so its exact semantics cannot be established from the
released reference. Prime does not guess whether this denotes one-sided value
clipping, symmetric value clipping, or another transformation. Full Section 5
reproduction remains open until that behavior and the fixed hard-question data
are available; the smoke test validates only the independently specified
control flow.
