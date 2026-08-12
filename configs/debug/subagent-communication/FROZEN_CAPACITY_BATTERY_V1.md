# Frozen Capacity Battery V1

This battery preserves the behavior boundary that ended the Qwen3.5 2B coordinator
experiment. It is a historical test set, not a source of training examples and not a
moving benchmark. Future curricula may add new train and development tasks, but they
must not alter these prompts, seeds, scorers, harness settings, or sampling parameters.

## Contents

The battery contains three disjoint 12-task standard-prompt gates with four direct,
four single-child, and four parallel tasks each, followed by the held-out
bidirectional follow-up and handshake screen.

| Gate | Offset | Seed | Retained 2B result |
| --- | ---: | ---: | --- |
| Standard generalization | 1600 | 20261109 | `10/12`: direct `4/4`, single `3/4`, parallel `3/4` |
| Fresh paired | 2300 | 20261213 | `10/12`: direct `4/4`, single `4/4`, parallel `2/4` |
| Cross-family | 2900 | 20270111 | `11/12`: direct `4/4`, single `4/4`, parallel `3/4` |
| Bidirectional | 3400 | 20270207 | `0/15` valid traces; one of 16 attempts missing |

The standard score is a joint answer-and-protocol solve, not answer accuracy alone.
The bidirectional score requires the causally ordered child request, parent response,
and child result. The retained 2B policy produced no aligned exchange.

The untouched Qwen3.5 4B baseline produced the same result on every standard gate:
direct `4/4`, single `0/4`, and parallel `0/4`, for `4/12`. Its bidirectional screen
was `0/16`, with zero child requests, parent follow-ups, child results, or causal
exchanges. This is the pretraining capacity baseline; it should not be confused with
the matched-curriculum 4B comparison that follows.

## Freeze contract

The machine-readable manifest records SHA-256 digests for all four historical configs,
their model-only 4B counterparts, and the complete Verifiers environment package that
generates and scores the tasks. It also records Prime-RL and Verifiers revisions,
Prime Agent `0.7.0`, historical summaries, and raw-trace digests.

The raw 2B traces are preserved with the retained model in private HF repository
`lentzl/rlm-prime-agent-qwen35-orchestrator-candidate-r1-20260809`, beginning at
repository revision `24792f80e462529eb410c46f7b669092f4000879`. The original dense
model revision remains independently pinned.

Only `model` and `output_dir` may differ between a historical config and its capacity
comparison counterpart. Validate the freeze before every comparison:

```bash
uv run python scripts/validate_frozen_capacity_battery.py
```

If a new scorer or task correction becomes necessary, create battery V2 and report it
separately. Never silently repair V1 or select training checkpoints using V1 outcomes.
After the recorded untouched-model baselines, keep V1 sealed throughout iterative 4B
development. Select the final 4B candidate using only disjoint train and development
tasks, then run V1 once as the historical capacity comparison. Do not use that result
to tune, replace, or repair the candidate.

## Interpretation

V1 freezes the behavior surface that ended the 2B coordinator experiment. It therefore
supports three deliberately different comparisons:

1. Untouched 2B versus untouched 4B measures the capacity-class difference before our
   intervention, provided model revision, harness, prompts, and sampling remain fixed.
2. A 2B and 4B trained from their untouched instruct checkpoints with the same examples,
   ordering, update budget, checkpoint rule, and promotion policy is the strongest
   training-time capacity ablation. Any necessary model-specific optimizer change must
   be declared as a remaining confound.
3. The retained best 2B versus the eventual best 4B measures the practical system
   improvement. Because their optimization histories may differ, this comparison must
   not be described as capacity alone causing the result.

All three use the same frozen behavioral boundary. The battery prevents benchmark and
scorer drift; it does not by itself make independently developed policies a causal
single-variable experiment. Report results by family and protocol event, not only as
mean reward. The retained 2B snapshot remains independently useful as a worker or
expert base even if 4B is the smallest reliable coordinator.
