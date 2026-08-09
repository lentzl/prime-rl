---
base_model: lentzl/rlm-prime-agent-qwen35-file-processing-r1-20260807
library_name: transformers
pipeline_tag: text-generation
tags:
  - prime-agent
  - rlm
  - subagent-communication
  - opsd
  - sdft
  - tool-use
---

# Qwen3.5 2B Prime Agent Orchestrator Candidate R1

This is an interim dense Stage-0 checkpoint for the smallest-model Prime Agent
orchestrator program. It merges the selected LoRA adapter from
`37-single-path-opsd-dose-r1/weights/step_2` into its complete dense rung-18 base, so
a fresh machine can hydrate one model snapshot without reconstructing the preceding
adapter chain.

The selected adapter was trained with Prime-native on-policy self-distillation
(OPSD/SDFT). The live Qwen3.5 2B policy sampled one trajectory per synthetic example,
while the same policy conditioned on a task-specific successful demonstration supplied
the per-token distribution target. Training used BF16 LoRA rank 16, alpha 32,
AdamW at `1e-6`, and three cumulative OPSD updates. Step 2 was selected by held-out
behavior and lower distillation loss rather than by recency.

On fresh guided tasks in the matching Prime Agent 0.7.0 harness, the selected policy
produced:

- exact task-specific child paths in 4/5 initial-action samples;
- 3/3 exact, fully protocol-aligned single-child solves;
- 3/3 exact direct solves with zero unnecessary child spawns;
- 3/3 exact parallel answers and path-bearing child prompts, with 2/3 fully
  protocol-aligned two-reply fan-ins.

The OPSD repair specifically addressed a stable failure where the parent compressed a
concrete request to `Read the file` and dropped the path. The matched pre-repair policy
retained the path in 0/3 samples; one OPSD update reached 1/3, and the selected
cumulative checkpoint reached 4/5 plus 3/3 complete single-child solves.

This is not yet the frozen RLM Master. Remaining gates include standard-prompt and
unseen-variant coverage, reliable parallel reply provenance, bidirectional follow-up,
traceback recovery, output contracts, and clean stopping. Document processing remains
a future expert-child specialization rather than a reason to enlarge this root model.

The `training/` directory contains the matching curriculum configs, Verifiers
environment, source-state manifest, and local source diff used for this result. The
model uses `<|im_end|>` token ID 248046 as EOS in generation and nested text config
metadata.

## Dense export validation

The dense export contains 617 model tensors. Each of the 96 LoRA targets is exactly
equal to `base + 2.0 * B @ A` after conversion back to BF16, while every non-target
tensor is bit-identical to the dense parent. Greedy evaluations of the adapter and
merged export produced the same exact path-bearing child-spawn action.

One additional temperature-0.8 dense smoke did not complete: it omitted the path from
the child prompt and repeated invalid async-repair cells after receiving a progress
message. This is retained as evidence of policy variance, not hidden as an export
failure. Use this checkpoint as the next training and evaluation base, not as a
production-ready autonomous orchestrator.
