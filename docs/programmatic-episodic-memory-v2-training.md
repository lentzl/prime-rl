# Programmatic Episodic Memory v2 training comparison

This lane tests whether demonstration-conditioned on-policy self-distillation
(Prime-RL `algo.type = "opsd"`) teaches natural retrieval policy more robustly
than cloning the same verified demonstrations with plain SFT. It is a bootstrap
experiment inside the broader Prime Agent mastery program, not a replacement
for GRPO or for the frozen Harness Mastery battery.

## Frozen inputs

- Common start: Run 331 Step 4, selected by the 21-task checkpoint screen.
- Train split: 1,200 examples across 25 familiar families.
- Familiar heldout: 300 examples; admission samples one frozen instance from
  every family twice (50 traces per arm).
- Semantic OOD: 96 examples across eight unseen families; admission samples one
  frozen instance from every family twice (16 traces per arm).
- Neither heldout split is used for weight updates.
- Demonstrations contain observable assistant tool calls and answers, never
  fabricated `reasoning_content`.

## Teacher admission

The gate was preregistered while only unconditioned traces existed. Both
conditioned arms must meet every core behavior minimum (0.90 familiar, 0.80
OOD), may not regress any core component by more than 0.05, and must improve
aggregate strict success by at least 0.08 absolute or by at least 0.04 absolute
with at least 50% relative error reduction. Every planned trace must complete
cleanly. Core components include answer and grounding, retrieve/no-retrieve,
tool validity, bounded retrieval, no repeated cells, stale-note resolution,
context-reset recovery, persistent-index reuse, and current-turn override.

Run the assessment with:

```bash
.venv/bin/python scripts/assess_programmatic_memory_teacher_admission.py \
  /ephemeral/subagent-rung/evals/336-339-qwen35-27b-memory-v2-teacher-admission-r2 \
  --output /ephemeral/subagent-rung/evals/336-339-qwen35-27b-memory-v2-teacher-admission-r2/admission.json \
  --require-pass
```

The SDFT launcher repeats this check and refuses to train on a failed or
incomplete admission. Plain SFT remains a valid baseline even if SDFT is not
admitted.

## Matched exposure

The renderer audit measured one training pass as 869,372 rendered sequence
tokens and 113,338 assistant/action tokens. Four passes produce 3,477,488
rendered tokens. With 4,096-token packs and global batch six, the plain-SFT lane
uses 154 updates at about 91.9% packing utilization.

The SDFT lane uses one on-policy rollout per example, group size one, global
batch 24, and 200 updates: exactly 4,800 dispatched episodes or four nominal
passes. Its sampled-token budget is inherently policy-dependent and must be
reported from admitted training samples. Equal examples and epochs are exact;
equal token exposure is not claimed. Checkpoint comparisons must include actual
trainable-token counts, not only optimizer steps.

Both lanes use full-weight BF16 AdamW on six trainer GPUs, learning rate 5e-7,
the same Step-4 start, and epoch-scale checkpoints. SFT trains only observable
assistant/tool actions. SDFT samples the unconditioned live policy and applies
dense demo-conditioned reverse-KL on those student-visited tokens.

## Execution

After all evaluation GPU processes have stopped:

```bash
scripts/run_programmatic_memory_v2_training.sh sft
scripts/run_programmatic_memory_v2_training.sh sdft
```

Run one lane at a time. Evaluate their epoch checkpoints on the unchanged
familiar/OOD environment and the frozen 74-task Harness Mastery battery. Prefer
SDFT only if it improves natural memory behavior at comparable exposure without
retrieval shortcuts or broader harness regressions.
