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

Delegated episodes run Prime Agent in its native autonomous mode with a
task-specific completion gate. The gate keeps the ACP session alive until the
generated child-message counts and final JSON key/type shape are present; it
does not contain expected values or score policy invariants. Exact correctness
remains exclusively verifier-side in `harness_score`.

`train-admission.toml` samples eight independent rollouts for one fresh
TRAIN-GEN task from each V1 family. It is not an evaluation split. Its purpose
is to measure whether the untouched policy supplies at least one hard-gate
success and one failure in a GRPO comparison group. Families with homogeneous
groups carry no hard-reward policy-gradient signal and must not silently enter
the first optimization batch.

`bootstrap-grpo.toml` is the first benchmark-directed weight update. It uses
four synchronous full-weight BF16 AdamW steps from the untouched pinned 27B,
with eight on-policy attempts per fresh TRAIN-GEN task. The only reward is the
conjunctive executable `harness_score`; homogeneous groups are rejected before
training. The launcher refuses to start unless the admission screen contains
at least one informative non-direct comparison group.
