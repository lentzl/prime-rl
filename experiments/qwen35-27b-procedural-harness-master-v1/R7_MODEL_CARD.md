---
library_name: transformers
base_model: Qwen/Qwen3.5-27B
tags:
  - prime-agent
  - prime-rl
  - reinforcement-learning
  - rlm
  - research
---

# Qwen3.5-27B Prime Agent Harness R7

This private research checkpoint is the protected R7 branch point of the
Procedural Harness Master action ramp. It is a full-weight BF16 checkpoint, not
a LoRA adapter. R7 is useful as the current starting point for further Prime
Agent training, but it is **not** a mastered teacher checkpoint and should not
be presented as one.

## Provenance

- Base: `Qwen/Qwen3.5-27B` at
  `fc05daec18b0a78c049392ed2e771dde82bdf654`.
- Prime-RL experiment code:
  [`69675ed082f567d7d586e6255b19d0a548a06f9e`](https://github.com/lentzl/prime-rl/commit/69675ed082f567d7d586e6255b19d0a548a06f9e).
- Verifiers environment:
  [`a3325cad087f2fe7dc944a4cc3f27713eb83bd50`](https://github.com/lentzl/verifiers/commit/a3325cad087f2fe7dc944a4cc3f27713eb83bd50).
- Prime Agent: `0.7.2-beta.495.1.97b994c`.
- Source checkpoint: action-ramp R6.
- Training data: Procedural Harness Master `atomic_send`, TRAIN-GEN window
  `1002000..1002511`, master seed `20260816`.
- Optimizer: full-weight AdamW, BF16 optimization/reduction, learning rate
  `5e-7`, batch size `16`, group size `8`, one GRPO update.
- Training hard reward: `10/16`; no rollout errors, truncation, or off-policy
  samples were reported for the accepted batch.

The uploaded `run-config/` directory contains the resolved trainer,
orchestrator, and inference configurations from the producing run.

## Measured Boundary

R7's historical fresh cumulative gate was `atomic_state=8/8` and
`atomic_send=7/8`. On the later frozen paired task draw beginning at index
`2300000`, R7 scored:

- `atomic_state`: `8/8`
- `atomic_send`: `5/8`
- `atomic_child_request`: `6/8`
- `atomic_followup`: `0/8`

The difference between the historical and paired send scores is why checkpoint
selection now uses same-draw comparisons and replicated fixed banks instead of
unpaired single gates.

## Integrity

- 12 non-empty safetensors shards.
- Stable export marker present.
- Generation and nested text configuration use `<|im_end|>` as EOS, token ID
  `248046`.
- Tokenizer, chat template, and processor metadata are included.

See the
[`qwen35-27b-procedural-harness-master-v1` experiment ledger](https://github.com/lentzl/prime-rl/tree/exp/prime-native-harness-official-first/experiments/qwen35-27b-procedural-harness-master-v1)
for the complete lineage, rejected descendants, verifier contracts, and frozen
promotion policy.
