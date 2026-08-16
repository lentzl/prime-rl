# Qwen3.5 27B Procedural Harness Master V1

This experiment makes the generated end-to-end Prime Agent coordinator policy the
optimization target. The untouched 27B thinking checkpoint is first measured on
frozen VALID-GEN and OOD-GEN indices. Training will then consume fresh,
non-overlapping TRAIN-GEN windows and promotion will depend on transfer back to
the frozen splits.

The initial baseline uses 24 episodes per split. That covers each VALID family
four times and each OOD family three times while keeping time-to-first signal
short. Later promotion evaluations can increase `count` without changing split
indices or the generator seed.

The primary reward is `harness_score`, a hard conjunction of exact answer,
required atoms, forbidden-atom absence, ordering, and cardinality. Dense metrics
exist only to diagnose why the hard gate failed.
