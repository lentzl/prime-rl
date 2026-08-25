# Qwen3.5-2B coordinator-only dense SPADE lineage

This lineage isolates coordinator learning while holding the protected K4 child
checkpoint fixed. Prime Agents evaluations still run both models and require a
routing audit proving that both served the trajectory.

## Fixed starting state

- Coordinator C4 SHA-256:
  `d3c852479335151840ed901f61a2997cc7d5235569ab0e14dff0332201657507`
- Child K4 SHA-256:
  `55cebb265ddc513b173dab2c5e4f4366f7d59ff3daa24f0bbf4d2bdfc00b89a1`
- Trainable roles: coordinator only
- Full optimizer steps per cycle: coordinator `1`, child `0`
- Learning rate: `1e-6`
- LoRA updates: `0`
- Admission floor: four complete qualifying trajectories and four distinct task
  keys; never relaxed

## Curriculum

1. Evaluate strict `e0d3_uncapped_yield_exact_child` on six fresh keys without
   an update.
2. If it fails, descend to `e0d2_capped_yield_exact_child` and train only from
   a complete 4/4-admitted bank.
3. Apply one full-parameter coordinator update with K4 unchanged.
4. Retest strict `e0d3` and the exact capped rung that admitted the parent.
5. Reject only the coordinator candidate if parent retention fails. A partial
   bank may close a failed gate only when even perfect missing episodes cannot
   reach four; it can never authorize training.

The runner uses append-only hash-chained controller and command journals,
immutable update receipts, a 32 GiB evaluator address-space cap, and fresh bank
indices beginning at `4600000`.

## Bounded result (2026-08-23)

The run completed two coordinator-only full updates and stopped at its configured
update budget:

- C4 strict baseline: `1/4` complete qualifiers before the gate was
  mathematically closed; capped source: `5/6`.
- C5 SHA-256:
  `1f6944f4c9d896411387d5ba63e3f81fc58f7240fdb3edf4de1aeecb5371abf2`.
  Training loss `0.1684`, gradient norm `29.125`. Strict: `0/5` before
  mathematical closure; exact capped retention: `5/6`.
- C6 SHA-256:
  `61c31f8d658096b9b000d073519119f165df59bf356227a426ccdaa026a1d081`.
  Training loss `0.1398`, gradient norm `23.75`. Strict: `1/6`; exact capped
  retention: `4/6`.
- K4 remained byte-identical at
  `55cebb265ddc513b173dab2c5e4f4366f7d59ff3daa24f0bbf4d2bdfc00b89a1`
  and received zero optimizer steps in both receipts.

The coordinator did not establish an accepted strict-yield learning ramp. The
unchanged four-trajectory/four-key gate therefore keeps C4/K4 as the protected
canonical pair. C6 is retained only as an experimental candidate that passed
its exact parent-retention rung.
